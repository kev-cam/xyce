// PyMS dynamic model loader for Xyce
//
// Compiles Verilog-A models on-demand via PyMS and loads the resulting
// .so for device evaluation.  Used by any device that needs to evaluate
// a Verilog-A model at runtime.
//
// Usage:
//   #include <N_DEV_PyMS.h>
//
//   // At setup (once per unique parameter set):
//   PyMSHandle h;
//   pyms_compile("psp103.va", params, &h);
//
//   // At eval (every Newton iteration):
//   pyms_eval(&h, voltages, F, Q);
//   pyms_jacobian(&h, voltages, dFdV, dQdV);

#ifndef N_DEV_PYMS_H
#define N_DEV_PYMS_H

#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif

#include <string>
#include <map>
#include <cstring>

namespace Xyce {
namespace Device {

struct VaeState {
    double V[16];
    double Vt;
};

typedef void (*VaeEvalFn)(VaeState*, double*, double*);

struct PyMSHandle {
    void *dl;
    VaeEvalFn eval_fn;
    VaeEvalFn jac_fn;
    int n_nodes;
    int n_branches;

    PyMSHandle() : dl(nullptr), eval_fn(nullptr), jac_fn(nullptr),
                   n_nodes(0), n_branches(0) {}

    bool valid() const { return eval_fn && jac_fn; }

    void close() {
#ifdef HAVE_DLFCN_H
        if (dl) { dlclose(dl); dl = nullptr; }
#endif
        eval_fn = jac_fn = nullptr;
    }
};

// Global registry: .hdl "file.va" → path
// Populated by netlist parser when .HDL is encountered
void pyms_register_va(const std::string &module_name, const std::string &va_path);
const std::string &pyms_get_va(const std::string &module_name);
bool pyms_has_va(const std::string &module_name);

// Load a pre-compiled .so
inline bool pyms_load_so(const char *so_path, PyMSHandle *h) {
#ifdef HAVE_DLFCN_H
    h->close();
    h->dl = dlopen(so_path, RTLD_NOW);
    if (!h->dl) return false;
    h->eval_fn = (VaeEvalFn)dlsym(h->dl, "vae_eval");
    h->jac_fn = (VaeEvalFn)dlsym(h->dl, "vae_jacobian");
    return h->valid();
#else
    return false;
#endif
}

// Eval: pack voltages, call .so
inline void pyms_eval(PyMSHandle *h, const double *node_voltages,
                      int n_nodes, double vt,
                      double *F, double *Q) {
    VaeState state;
    memset(&state, 0, sizeof(state));
    for (int i = 0; i < n_nodes && i < 16; i++)
        state.V[i] = node_voltages[i];
    state.Vt = vt;
    h->eval_fn(&state, F, Q);
}

inline void pyms_jacobian(PyMSHandle *h, const double *node_voltages,
                          int n_nodes, double vt,
                          double *dFdV, double *dQdV) {
    VaeState state;
    memset(&state, 0, sizeof(state));
    for (int i = 0; i < n_nodes && i < 16; i++)
        state.V[i] = node_voltages[i];
    state.Vt = vt;
    h->jac_fn(&state, dFdV, dQdV);
}

} // namespace Device
} // namespace Xyce

#endif // N_DEV_PYMS_H
