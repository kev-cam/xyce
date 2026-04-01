// PyMS .HDL handler — compile Verilog-A and register device at parse time.
//
// When the netlist parser encounters .HDL "file.va", it calls
// pyms_register_hdl() which:
//   1. Finds xyce_device_gen.py (via PYMS_DIR env or relative to Xyce source)
//   2. Runs it to parse the VA and generate C++ device wrapper
//   3. Compiles the C++ to a .so (linking against libxyce.so)
//   4. dlopen's the .so — the __attribute__((constructor)) auto-registers
//      the device type with Xyce's device factory
//
// After this, Xyce can parse device instance lines (node count, params known).
// The per-instance VAE .so compilation happens later in processParams().

#include <Xyce_config.h>
#include <N_DEV_PyMS.h>
#include <N_DEV_RegisterDevices.h>
#include <N_ERH_ErrorMgr.h>

#include <cstdlib>
#include <cstdio>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif

namespace Xyce {
namespace Device {

// Registry of .hdl VA sources
static std::map<std::string, std::string> va_registry_;

void pyms_register_va(const std::string &module_name, const std::string &va_path) {
    va_registry_[module_name] = va_path;
}

const std::string &pyms_get_va(const std::string &module_name) {
    static const std::string empty;
    auto it = va_registry_.find(module_name);
    return it != va_registry_.end() ? it->second : empty;
}

bool pyms_has_va(const std::string &module_name) {
    return va_registry_.count(module_name) > 0;
}

// Find xyce_device_gen.py
static std::string find_device_gen() {
    // Check PYMS_DIR environment variable
    const char *pyms_dir = getenv("PYMS_DIR");
    if (pyms_dir) {
        std::string path = std::string(pyms_dir) + "/vae/xyce_device_gen.py";
        struct stat st;
        if (stat(path.c_str(), &st) == 0)
            return path;
    }

    // Check relative to Xyce source (common development layout)
    const char *candidates[] = {
        "/usr/local/src/Xyce-8/xyce/utils/PyMS/vae/xyce_device_gen.py",
        "./utils/PyMS/vae/xyce_device_gen.py",
        nullptr
    };
    for (int i = 0; candidates[i]; i++) {
        struct stat st;
        if (stat(candidates[i], &st) == 0)
            return candidates[i];
    }
    return "";
}

// Find Xyce include directories for compiling plugins
static std::string find_xyce_includes() {
    // Check XYCE_SRC environment variable
    const char *xyce_src = getenv("XYCE_SRC");
    if (!xyce_src)
        xyce_src = "/usr/local/src/Xyce-8/xyce/src";

    const char *xyce_build = getenv("XYCE_BUILD");
    if (!xyce_build)
        xyce_build = "/usr/local/src/Xyce-8/xyce-build";

    // Collect all subdirectories of src/ as include paths
    std::ostringstream incs;
    incs << "-I" << xyce_build << "/src ";

    // Core directories needed by device plugins
    const char *subdirs[] = {
        "DeviceModelPKG/Core", "DeviceModelPKG/OpenModels",
        "LinearAlgebraServicesPKG", "UtilityPKG",
        "ErrorHandlingPKG", "IOInterfacePKG",
        "ParallelDistPKG", "TopoManagerPKG",
        "AnalysisPKG", "NonlinearSolverPKG",
        "TimeIntegrationPKG", "LoaderServicesPKG",
        "CircuitPKG", "DakotaLinkPKG",
        "MultiTimePDEPKG", "OutputPKG",
        nullptr
    };
    for (int i = 0; subdirs[i]; i++) {
        incs << "-I" << xyce_src << "/" << subdirs[i] << " ";
    }
    return incs.str();
}

bool pyms_register_hdl(const std::string &va_path) {
#ifdef HAVE_DLFCN_H
    // Step 1: Find xyce_device_gen.py
    std::string gen_script = find_device_gen();
    if (gen_script.empty()) {
        Report::UserError0() << ".HDL: cannot find xyce_device_gen.py. "
            "Set PYMS_DIR to the PyMS directory.";
        return false;
    }

    // Step 2: Create a cache directory for the compiled plugin
    const char *cache_base = getenv("PYMS_CACHE");
    std::string cache_dir;
    if (cache_base) {
        cache_dir = cache_base;
    } else {
        cache_dir = "/tmp/pyms_hdl_cache";
    }
    mkdir(cache_dir.c_str(), 0755);

    // Step 3: Run xyce_device_gen.py to parse VA and generate C++
    std::string cmd = "python3 " + gen_script + " " + va_path +
                      " --output " + cache_dir + " 2>&1";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) {
        Report::UserError0() << ".HDL: failed to run PyMS: " << cmd;
        return false;
    }

    // Capture output to get the module name
    char buf[4096];
    std::string output;
    while (fgets(buf, sizeof(buf), fp))
        output += buf;
    int rc = pclose(fp);

    if (rc != 0) {
        Report::UserError0() << ".HDL: PyMS compilation failed for " << va_path
            << "\n" << output;
        return false;
    }

    // Parse module name from output: "Module: <name>, Ports: ..."
    std::string module_name;
    auto pos = output.find("Module: ");
    if (pos != std::string::npos) {
        auto end = output.find(",", pos + 8);
        if (end != std::string::npos)
            module_name = output.substr(pos + 8, end - pos - 8);
    }

    if (module_name.empty()) {
        Report::UserError0() << ".HDL: could not extract module name from "
            << va_path << "\nPyMS output: " << output;
        return false;
    }

    // Register the VA source for later VAE compilation
    pyms_register_va(module_name, va_path);

    // Step 4: Compile the generated C++ to a .so
    std::string NAME = module_name;
    for (auto &c : NAME) c = toupper(c);

    std::string cpp_file = cache_dir + "/N_DEV_PYMS_" + NAME + ".C";
    std::string so_file = cache_dir + "/pyms_" + module_name + ".so";

    // Check if .so already exists and is newer than the .va
    struct stat va_stat, so_stat;
    if (stat(va_path.c_str(), &va_stat) == 0 &&
        stat(so_file.c_str(), &so_stat) == 0 &&
        so_stat.st_mtime > va_stat.st_mtime) {
        // Cached .so is up to date — just load it
    } else {
        // Compile
        std::string includes = find_xyce_includes();
        const char *xyce_build = getenv("XYCE_BUILD");
        if (!xyce_build) xyce_build = "/usr/local/src/Xyce-8/xyce-build";

        std::string compile_cmd =
            "g++ -shared -fPIC -O2 -std=c++17 "
            "-I" + cache_dir + " " +
            includes +
            "-L" + std::string(xyce_build) + "/src -lxyce "
            "-o " + so_file + " " + cpp_file + " 2>&1";

        fp = popen(compile_cmd.c_str(), "r");
        if (!fp) {
            Report::UserError0() << ".HDL: failed to compile plugin for "
                << module_name;
            return false;
        }
        output.clear();
        while (fgets(buf, sizeof(buf), fp))
            output += buf;
        rc = pclose(fp);

        if (rc != 0) {
            Report::UserError0() << ".HDL: g++ compilation failed for "
                << module_name << "\n" << output;
            return false;
        }
    }

    // Step 5: dlopen the .so — constructor auto-registers the device
    void *dl = dlopen(so_file.c_str(), RTLD_NOW);
    if (!dl) {
        Report::UserError0() << ".HDL: failed to load compiled plugin "
            << so_file << ": " << dlerror();
        return false;
    }

    // Step 6: Set VAE_SO_DIR so processParams can find the VAE math .so
    // The VAE .so will be compiled lazily by processParams when first needed.
    // Point VAE_SO_DIR to our cache directory so it looks there.
    setenv("VAE_SO_DIR", cache_dir.c_str(), 0);  // don't override if already set

    Report::UserWarning0() << ".HDL: compiled and registered " << module_name
        << " from " << va_path;
    return true;

#else
    Report::UserError0() << ".HDL: dynamic loading not supported (no dlfcn.h)";
    return false;
#endif
}

} // namespace Device
} // namespace Xyce
