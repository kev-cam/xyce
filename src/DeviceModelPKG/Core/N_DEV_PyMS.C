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
#include <cstring>
#include <cctype>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <dirent.h>
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

// Find xyce_device_gen.py. Lookup order:
//   1. PYMS_DIR env var (development override)
//   2. $XYCE_PREFIX/share/xyce/PyMS/vae/xyce_device_gen.py (install
//      tree; PREFIX = $XYCE_PREFIX or compiled-in /usr/local)
//   3. Development source-tree fallbacks
static std::string find_device_gen() {
    auto try_path = [](const std::string &p) -> std::string {
        struct stat st;
        return (stat(p.c_str(), &st) == 0) ? p : std::string();
    };

    const char *pyms_dir = getenv("PYMS_DIR");
    if (pyms_dir) {
        std::string p = try_path(std::string(pyms_dir) + "/vae/xyce_device_gen.py");
        if (!p.empty()) return p;
    }

    const char *xyce_prefix = getenv("XYCE_PREFIX");
    std::string prefix = xyce_prefix ? xyce_prefix : "/usr/local";
    std::string p = try_path(prefix + "/share/xyce/PyMS/vae/xyce_device_gen.py");
    if (!p.empty()) return p;

    const char *candidates[] = {
        "/usr/local/src/xyce/utils/PyMS/vae/xyce_device_gen.py",
        "/usr/local/src/Xyce-8/xyce/utils/PyMS/vae/xyce_device_gen.py",
        "./utils/PyMS/vae/xyce_device_gen.py",
        nullptr
    };
    for (int i = 0; candidates[i]; i++) {
        std::string r = try_path(candidates[i]);
        if (!r.empty()) return r;
    }
    return "";
}

// Find Xyce include directories for compiling plugins
static std::string find_xyce_includes() {
    auto exists = [](const std::string &p) {
        struct stat st; return stat(p.c_str(), &st) == 0;
    };

    // Source tree (headers). XYCE_SRC wins; otherwise probe the known
    // layouts via a sentinel header so we use the LIVE source tree
    // rather than whatever stale subset got `make install`ed into the
    // default system include path. /usr/local/include is missing some
    // group headers (e.g. N_DEV_Inductor.h), which silently broke the
    // level=N auto-loader for those device families.
    std::string xyce_src;
    if (const char *e = getenv("XYCE_SRC")) {
        xyce_src = e;
    } else {
        const char *cands[] = { "/usr/local/src/xyce/src",
                                "/usr/local/src/Xyce-8/xyce/src", nullptr };
        const char *sentinel = "/DeviceModelPKG/OpenModels/N_DEV_Inductor.h";
        for (int i = 0; cands[i]; i++)
            if (exists(std::string(cands[i]) + sentinel)) { xyce_src = cands[i]; break; }
        if (xyce_src.empty()) xyce_src = "/usr/local/src/Xyce-8/xyce/src";
    }

    // Build tree (configured headers + libXyceLib.so). Same probe order
    // as the compile step below.
    std::string xyce_build;
    if (const char *e = getenv("XYCE_BUILD")) {
        xyce_build = e;
    } else if (exists("/usr/local/src/xyce-build/src/libXyceLib.so")) {
        xyce_build = "/usr/local/src/xyce-build";
    } else {
        xyce_build = "/usr/local/src/Xyce-8/xyce-build";
    }

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

// ---------------------------------------------------------------------------
// Auto-registry: scan installed .va sources once at startup so that
// ``.model foo nmos level=110`` without a preceding ``.HDL`` directive
// still auto-loads the corresponding compact model.
//
// Roots scanned:
//   - $XYCE_PREFIX/share/xyce/verilog-a       (compile-time /usr/local)
//   - colon-separated entries from $XYCE_VA_PATH
// Each root is recursed up to a small fixed depth — the install layout
// alternates between
//     verilog-a/<MODEL>/code/<file>.va        (BSIM-CMG, PSP103, ...)
//     verilog-a/<MODEL>/<file>.va             (toys/, fbh_hbt-2.1/, ...)
//     verilog-a/<FAMILY>/<MODEL>/<file>.va    (BSIM-SOI_4/bsimsoi4.7.0/)
//
// For each .va, the module decl carries xyceModelGroup and
// xyceLevelNumber on either ``(* ... *)`` (Accellera form) or via an
// ADMS-convention macro: ``\`attr(...)`` / ``\`ATTR(...)`` /
// ``\`P(...)``. All those forms expand to attribute lists at run time;
// here we treat the bare macro-call form as a textual variant since the
// auto-registry only needs the strings, not a full preprocess.
//
// The collected (name, level) → va_path table is consulted from the
// findConfiguration / getModelType miss-paths in N_DEV_Configuration.C.
// ---------------------------------------------------------------------------
namespace {

static std::map<std::pair<std::string, int>, std::string> &auto_registry()
{
    static std::map<std::pair<std::string, int>, std::string> reg;
    return reg;
}

// Return ``<dir>/<name>.va`` when invoked on ``<dir>/<name>_main.va``
// and the wrapper actually ``\`include``s us — ADMS-style compact
// models split the entry point in two (bsimcmg.va defines
// ``\`define attr(txt) (*txt*)`` then ``\`include "bsimcmg_main.va"``,
// and the module decl with the attr macro lives in the body file).
// Without this, scan_va_for_attrs returns the body's path and codegen
// later loses the wrapper's macro defines.
static std::string prefer_wrapper(const std::string &path)
{
    static const std::string suffix = "_main.va";
    if (path.size() <= suffix.size()) return path;
    if (path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0)
        return path;
    std::string candidate = path.substr(0, path.size() - suffix.size()) + ".va";
    struct stat st;
    if (stat(candidate.c_str(), &st) != 0) return path;
    std::ifstream f(candidate);
    if (!f.is_open()) return path;
    std::string head(std::istreambuf_iterator<char>(f), {});
    // Only reroute if the wrapper actually ``\`include``s us, so we
    // don't accidentally pick an unrelated sibling that happens to
    // share a stem.
    auto slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos)
                       ? path : path.substr(slash + 1);
    if (head.find(base) == std::string::npos) return path;
    return candidate;
}

// Pull "key=value" pairs from the body of an attribute block, writing
// xyceModelGroup / xyceLevelNumber into the out-params. The body is
// the inner text between (* ... *) or `MACRO(...). Only quoted values
// are recognised — bare numeric or identifier values are out of scope.
static void parse_attr_body(const std::string &body,
                            std::string &group_out, int &level_out,
                            bool &got_group, bool &got_level)
{
    size_t i = 0, n = body.size();
    while (i < n) {
        while (i < n && !std::isalpha((unsigned char)body[i]) && body[i] != '_')
            ++i;
        size_t k_start = i;
        while (i < n && (std::isalnum((unsigned char)body[i]) || body[i] == '_'))
            ++i;
        if (k_start == i) break;
        std::string key = body.substr(k_start, i - k_start);
        while (i < n && std::isspace((unsigned char)body[i])) ++i;
        if (i >= n || body[i] != '=') continue;
        ++i;
        while (i < n && std::isspace((unsigned char)body[i])) ++i;
        if (i >= n || body[i] != '"') continue;
        ++i;
        size_t v_start = i;
        while (i < n && body[i] != '"') ++i;
        if (i >= n) break;
        std::string val = body.substr(v_start, i - v_start);
        ++i;
        if (key == "xyceModelGroup") {
            group_out = val;
            got_group = true;
        } else if (key == "xyceLevelNumber") {
            try { level_out = std::stoi(val); got_level = true; }
            catch (...) {}
        }
    }
}

// Find the byte-range of the next ``(* ... *)`` block starting at or
// after ``from``. Returns ``{npos, npos}`` if none.
static std::pair<size_t, size_t>
find_attr_paren(const std::string &s, size_t from)
{
    size_t open_ = s.find("(*", from);
    if (open_ == std::string::npos) return {std::string::npos, std::string::npos};
    size_t close_ = s.find("*)", open_ + 2);
    if (close_ == std::string::npos) return {std::string::npos, std::string::npos};
    return {open_, close_};
}

// Find a ``\`MACRO(...)`` invocation at or after ``from``. The macro
// name is any identifier following a backtick; the body is whatever
// sits inside its first parenthesised arg-list. Used for HICUM `ATTR,
// Diode CMC `P, MVS / BSIM-CMG `attr — all of which expand to
// ``(* body *)`` at preprocess time.
static std::pair<size_t, size_t>
find_attr_macro(const std::string &s, size_t from, size_t &body_start_out,
                size_t &body_end_out)
{
    size_t i = from;
    while (i < s.size()) {
        i = s.find('`', i);
        if (i == std::string::npos) break;
        size_t j = i + 1;
        if (j >= s.size() || !(std::isalpha((unsigned char)s[j]) || s[j] == '_')) {
            ++i;
            continue;
        }
        while (j < s.size() && (std::isalnum((unsigned char)s[j]) || s[j] == '_'))
            ++j;
        size_t k = j;
        while (k < s.size() && std::isspace((unsigned char)s[k])) ++k;
        if (k < s.size() && s[k] == '(') {
            // Find matching ')'
            int depth = 1;
            size_t b = k + 1;
            while (b < s.size() && depth > 0) {
                if (s[b] == '(') ++depth;
                else if (s[b] == ')') --depth;
                if (depth == 0) break;
                ++b;
            }
            if (depth == 0) {
                body_start_out = k + 1;
                body_end_out = b;
                return {i, b + 1};
            }
        }
        i = j;
    }
    return {std::string::npos, std::string::npos};
}

// Scan a single .va for an attribute block carrying xyceModelGroup /
// xyceLevelNumber. Three shapes:
//   1. ``module name(ports) (* k=v ... *) ;``       (Accellera form)
//   2. ``(* k=v ... *) module name(ports) ;``       (pre-decl, EKV, toys/)
//   3. ``module name(ports) `MACRO(k=v ...) ;``     (`attr / `P / `ATTR)
//
// Walks the file as plain text. Avoiding std::regex here — its
// libstdc++ implementation blows the stack on long inputs with lazy
// quantifiers (a 220KB BSIM-SOI source recursed >130000 frames in
// initial testing).
static bool scan_va_for_attrs(const std::string &path,
                              std::string &group_out, int &level_out,
                              std::string &module_path_out)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    bool got_group = false, got_level = false;

    // Walk every ``module ... ( ... )`` we find — modules with no
    // attrs are skipped; we keep going until one yields both fields
    // or the file is exhausted.
    size_t scan = 0;
    while (scan < content.size()) {
        // Locate next ``module`` keyword as a standalone token.
        size_t mp = content.find("module", scan);
        if (mp == std::string::npos) break;
        // Must be word-bounded (prev char and next char not part of
        // an identifier).
        bool left_ok = (mp == 0) ||
            !(std::isalnum((unsigned char)content[mp - 1]) || content[mp - 1] == '_');
        bool right_ok = (mp + 6 < content.size()) &&
            !(std::isalnum((unsigned char)content[mp + 6]) || content[mp + 6] == '_');
        if (!left_ok || !right_ok) { scan = mp + 6; continue; }

        // Skip module name + whitespace, then expect ``(``.
        size_t p = mp + 6;
        while (p < content.size() && std::isspace((unsigned char)content[p])) ++p;
        while (p < content.size() &&
               (std::isalnum((unsigned char)content[p]) || content[p] == '_'))
            ++p;
        while (p < content.size() && std::isspace((unsigned char)content[p])) ++p;
        if (p >= content.size() || content[p] != '(') { scan = mp + 6; continue; }
        // Skip port list (balanced ``()``).
        int depth = 1; ++p;
        while (p < content.size() && depth > 0) {
            if (content[p] == '(') ++depth;
            else if (content[p] == ')') --depth;
            ++p;
        }
        if (depth != 0) break;
        // Skip whitespace + an optional ``;``.
        while (p < content.size() && std::isspace((unsigned char)content[p])) ++p;

        // ---------- shape 1: ``module ... (* k=v ... *)`` ----------
        if (p + 1 < content.size() && content[p] == '(' && content[p + 1] == '*') {
            size_t bend = content.find("*)", p + 2);
            if (bend != std::string::npos) {
                std::string body = content.substr(p + 2, bend - (p + 2));
                parse_attr_body(body, group_out, level_out, got_group, got_level);
                if (got_group && got_level) {
                    module_path_out = prefer_wrapper(path);
                    return true;
                }
            }
        }
        // ---------- shape 3: ``module ... `MACRO(k=v ...)`` --------
        if (p < content.size() && content[p] == '`') {
            size_t body_s, body_e;
            auto mr = find_attr_macro(content, p, body_s, body_e);
            // Only treat as attr-block when the macro starts at p
            // (immediately after the module decl, before ``;``).
            if (mr.first == p) {
                std::string body = content.substr(body_s, body_e - body_s);
                parse_attr_body(body, group_out, level_out, got_group, got_level);
                if (got_group && got_level) {
                    module_path_out = prefer_wrapper(path);
                    return true;
                }
            }
        }
        // ---------- shape 2: ``(* k=v ... *) module ...`` ----------
        // Look backwards from ``module`` for an immediately-preceding
        // ``*)`` (allowing whitespace + comments).
        if (mp > 4) {
            // Scan backwards skipping spaces/comments (just whitespace
            // is enough — real files don't put C comments between the
            // attr block and ``module``).
            ssize_t q = (ssize_t)mp - 1;
            while (q >= 0 && std::isspace((unsigned char)content[q])) --q;
            if (q >= 1 && content[q] == ')' && content[q - 1] == '*') {
                // Find matching ``(*`` before this ``*)``.
                size_t close_ = (size_t)q - 1;
                size_t open_ = content.rfind("(*", close_);
                if (open_ != std::string::npos) {
                    std::string body = content.substr(open_ + 2, close_ - (open_ + 2));
                    parse_attr_body(body, group_out, level_out, got_group, got_level);
                    if (got_group && got_level) {
                        module_path_out = prefer_wrapper(path);
                        return true;
                    }
                }
            }
        }

        scan = p + 1;
    }
    return false;
}

// Recursive directory walk, bounded so we never run away on a
// pathological install tree. Picks up the three layouts noted above.
static void scan_va_dir(const std::string &dir, int depth_remaining)
{
    DIR *d = opendir(dir.c_str());
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] == '.') continue;
        std::string entry = dir + "/" + e->d_name;
        struct stat st;
        if (stat(entry.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            if (depth_remaining > 0) scan_va_dir(entry, depth_remaining - 1);
            continue;
        }
        std::string fn = e->d_name;
        if (fn.size() < 4) continue;
        if (fn.compare(fn.size() - 3, 3, ".va") != 0) continue;
        // Skip ``.vams`` (discipline/nature) files — they never carry
        // an xyceModelGroup attribute and parsing them wastes I/O.
        if (fn.size() >= 5 &&
            fn.compare(fn.size() - 5, 5, ".vams") == 0) continue;
        std::string group, module_path;
        int level = 0;
        if (!scan_va_for_attrs(entry, group, level, module_path)) continue;
        const char *names_mosfet[] = {"m", "nmos", "pmos", nullptr};
        const char *names_diode[]  = {"d",                 nullptr};
        const char *names_bjt[]    = {"q", "npn", "pnp",   nullptr};
        const char *names_res[]    = {"r",                 nullptr};
        const char *names_cap[]    = {"c",                 nullptr};
        const char *names_ind[]    = {"l",                 nullptr};
        const char **names = nullptr;
        if      (group == "MOSFET")    names = names_mosfet;
        else if (group == "Diode")     names = names_diode;
        else if (group == "BJT")       names = names_bjt;
        else if (group == "Resistor")  names = names_res;
        else if (group == "Capacitor") names = names_cap;
        else if (group == "Inductor")  names = names_ind;
        if (!names) continue;
        for (int i = 0; names[i]; i++) {
            auto key = std::make_pair(std::string(names[i]), level);
            // First-wins: if two .va files declare the same (name,
            // level) (e.g. binning variants), keep whichever readdir
            // returned first. The .HDL directive remains the way to
            // pick a specific variant.
            if (auto_registry().count(key)) continue;
            auto_registry()[key] = module_path;
        }
    }
    closedir(d);
}

static void populate_auto_registry()
{
    static bool done = false;
    if (done) return;
    done = true;

    std::vector<std::string> roots;
    const char *xyce_prefix = getenv("XYCE_PREFIX");
    std::string prefix = xyce_prefix ? xyce_prefix : "/usr/local";
    roots.push_back(prefix + "/share/xyce/verilog-a");

    const char *vp = getenv("XYCE_VA_PATH");
    if (vp) {
        std::string s = vp;
        size_t start = 0;
        while (start < s.size()) {
            size_t colon = s.find(':', start);
            std::string entry = (colon == std::string::npos)
                ? s.substr(start) : s.substr(start, colon - start);
            if (!entry.empty()) roots.push_back(entry);
            if (colon == std::string::npos) break;
            start = colon + 1;
        }
    }
    // Depth=3 covers the deepest known layout (BSIM-SOI_4/<ver>/<file>.va)
    // with one level of headroom; readdir-skipping non-.va files keeps
    // the cost flat.
    for (const auto &r : roots) scan_va_dir(r, 3);

    if (getenv("XYCE_PYMS_AUTO_DEBUG")) {
        std::cerr << "[XYCE_PYMS_AUTO] registry size: "
                  << auto_registry().size() << "\n";
        for (const auto &kv : auto_registry())
            std::cerr << "  (" << kv.first.first << ", " << kv.first.second
                      << ") -> " << kv.second << "\n";
    }
}

}  // anonymous namespace

bool pyms_try_auto_register(const std::string &name, int level)
{
    populate_auto_registry();
    // The model-type name reaches us in both lower-case (device-letter
    // dispatch) and the case the netlist used (model-card lookup);
    // try verbatim first, then a lowercased variant.
    auto it = auto_registry().find(std::make_pair(name, level));
    if (it == auto_registry().end()) {
        std::string lower = name;
        for (auto &c : lower)
            c = (char)std::tolower((unsigned char)c);
        it = auto_registry().find(std::make_pair(lower, level));
    }
    if (it == auto_registry().end()) {
        if (getenv("XYCE_PYMS_AUTO_DEBUG"))
            std::cerr << "[XYCE_PYMS_AUTO] no entry for ("
                      << name << ", " << level << ")\n";
        return false;
    }
    if (getenv("XYCE_PYMS_AUTO_DEBUG"))
        std::cerr << "[XYCE_PYMS_AUTO] auto-loading " << it->second
                  << " for (" << name << ", " << level << ")\n";
    return pyms_register_hdl(it->second);
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
        if (!xyce_build) {
            // Try both common build layouts before giving up
            struct stat st;
            if (stat("/usr/local/src/xyce-build/src/libXyceLib.so", &st) == 0)
                xyce_build = "/usr/local/src/xyce-build";
            else
                xyce_build = "/usr/local/src/Xyce-8/xyce-build";
        }

        // libXyceLib.so is the linker name (the install-tree symbol
        // resolution comes through it). The PyMS plugin only needs
        // the type info from the headers — at runtime it dlopen's
        // into a process that already has libXyceLib loaded, so
        // --allow-shlib-undefined lets us link without resolving the
        // full Trilinos dependency chain.
        std::string compile_cmd =
            "g++ -shared -fPIC -O2 -std=c++17 "
            "-Wl,--allow-shlib-undefined "
            "-I" + cache_dir + " " +
            includes +
            "-L" + std::string(xyce_build) + "/src -lXyceLib "
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
