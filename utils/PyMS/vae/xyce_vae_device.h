// VAE Generic Analog Device for Xyce
//
// Loads a VAE-compiled .so at runtime and stamps F/Q/dFdV/dQdV
// into Xyce's DAE system.  Netlist syntax:
//
//   YVA <name> <nodes...> <modelname> [instance params]
//   .model <modelname> VA so=<path.so> [nports=4]
//
// The .so must export:
//   void vae_eval(VaeState*, double* F, double* Q)
//   void vae_jacobian(VaeState*, double* dFdV, double* dQdV)
// where VaeState = { double V[n_nodes]; double Vt; }

#ifndef Xyce_VAE_DEVICE_H
#define Xyce_VAE_DEVICE_H

#include <dlfcn.h>
#include <string>
#include <vector>

struct VaeState {
    double V[16];
    double Vt;
};

typedef void (*VaeEvalFn)(VaeState*, double*, double*);

struct VaeHandle {
    void* dl;
    VaeEvalFn eval_fn;
    VaeEvalFn jacobian_fn;
    int n_nodes;
    int n_branches;
    int prev_regime_key;

    VaeHandle() : dl(nullptr), eval_fn(nullptr), jacobian_fn(nullptr),
                  n_nodes(0), n_branches(0), prev_regime_key(-1) {}

    bool load(const char* so_path) {
        dl = dlopen(so_path, RTLD_NOW);
        if (!dl) return false;
        eval_fn = (VaeEvalFn)dlsym(dl, "vae_eval");
        jacobian_fn = (VaeEvalFn)dlsym(dl, "vae_jacobian");
        // Optional metadata
        auto nn = (int(*)())dlsym(dl, "vae_n_nodes");
        auto nb = (int(*)())dlsym(dl, "vae_n_branches");
        if (nn) n_nodes = nn();
        if (nb) n_branches = nb();
        return eval_fn && jacobian_fn;
    }

    void close() {
        if (dl) { dlclose(dl); dl = nullptr; }
    }

    ~VaeHandle() { close(); }
};

#endif // Xyce_VAE_DEVICE_H
