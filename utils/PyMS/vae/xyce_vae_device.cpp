// VAE VectorComputeInterface implementation for Xyce
//
// Bridges a VAE-compiled .so into Xyce's DAE framework via the
// GeneralExternal (YGENEXT) device.  At each Newton iteration,
// Xyce calls computeXyceVectors() which:
//   1. Packs node voltages into VaeState
//   2. Calls vae_eval → F[], Q[]
//   3. Calls vae_jacobian → dFdV[], dQdV[]
//   4. Stamps results into Xyce's F, Q, dFdx, dQdx vectors/matrices
//
// Usage from C++ (test harness or Xyce integration):
//   VaeVectorCompute vc;
//   vc.load("/path/to/regime.so", n_nodes, n_branches);
//   // Attach to YGENEXT instance via setVectorLoader(&vc)
//
// The eval function can also return a regime key for timestep control.
// When the key changes between calls, the simulator should limit the
// timestep and binary-search for the regime crossing point.

#include "xyce_vae_device.h"
#include <cstring>
#include <cstdio>
#include <cmath>

class VaeVectorCompute {
public:
    VaeHandle handle;
    int n_ext_nodes;   // external nodes visible to Xyce
    int n_branches;    // number of branch contributions

    VaeVectorCompute() : n_ext_nodes(0), n_branches(0) {}

    bool load(const char* so_path, int ext_nodes, int branches) {
        if (!handle.load(so_path)) {
            fprintf(stderr, "VaeVectorCompute: failed to load %s: %s\n",
                    so_path, dlerror());
            return false;
        }
        n_ext_nodes = ext_nodes;
        n_branches = branches;
        return true;
    }

    // Called at every Newton iteration by Xyce
    // sV[i] = node voltages (solution variables)
    // F[i], Q[i] = current/charge contributions per node (KCL)
    // dFdx[i][j], dQdx[i][j] = Jacobian entries
    bool computeXyceVectors(
        std::vector<double>& sV,
        double time,
        std::vector<double>& F,
        std::vector<double>& Q,
        std::vector<double>& B,
        std::vector<std::vector<double>>& dFdx,
        std::vector<std::vector<double>>& dQdx)
    {
        if (!handle.eval_fn || !handle.jacobian_fn)
            return false;

        int n = n_ext_nodes;

        // Pack voltages into VaeState
        VaeState state;
        memset(&state, 0, sizeof(state));
        for (int i = 0; i < n && i < 16; i++)
            state.V[i] = sV[i];
        state.Vt = 8.617087e-5 * 300.15;  // kT/q at nominal temp

        // Call VAE eval
        double F_vae[32], Q_vae[32];
        memset(F_vae, 0, sizeof(F_vae));
        memset(Q_vae, 0, sizeof(Q_vae));
        handle.eval_fn(&state, F_vae, Q_vae);

        // Call VAE jacobian
        double dFdV[32*16], dQdV[32*16];
        memset(dFdV, 0, sizeof(dFdV));
        memset(dQdV, 0, sizeof(dQdV));
        handle.jacobian_fn(&state, dFdV, dQdV);

        // Map branch contributions to node KCL equations
        // For now: direct 1:1 mapping (branch i → node i)
        // A proper implementation would use the branch-to-node
        // incidence matrix from the model metadata.
        for (int i = 0; i < n && i < n_branches; i++) {
            F[i] += F_vae[i];
            Q[i] += Q_vae[i];
        }

        // Stamp Jacobian
        for (int i = 0; i < n && i < n_branches; i++) {
            for (int j = 0; j < n; j++) {
                dFdx[i][j] += dFdV[i * n + j];
                dQdx[i][j] += dQdV[i * n + j];
            }
        }

        return true;
    }
};

// Standalone test harness — validates the .so without Xyce
#ifdef VAE_STANDALONE_TEST
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <regime.so> [n_nodes] [n_branches]\n", argv[0]);
        return 1;
    }
    int n_nodes = argc > 2 ? atoi(argv[2]) : 5;
    int n_branches = argc > 3 ? atoi(argv[3]) : 15;

    VaeVectorCompute vc;
    if (!vc.load(argv[1], n_nodes, n_branches)) {
        fprintf(stderr, "Failed to load .so\n");
        return 1;
    }

    // Test with sample voltages
    std::vector<double> sV(n_nodes, 0.0);
    sV[0] = 0.6;  // VD
    sV[1] = 0.6;  // VG

    std::vector<double> F(n_nodes, 0.0);
    std::vector<double> Q(n_nodes, 0.0);
    std::vector<double> B(n_nodes, 0.0);
    std::vector<std::vector<double>> dFdx(n_nodes, std::vector<double>(n_nodes, 0.0));
    std::vector<std::vector<double>> dQdx(n_nodes, std::vector<double>(n_nodes, 0.0));

    bool ok = vc.computeXyceVectors(sV, 0.0, F, Q, B, dFdx, dQdx);
    printf("computeXyceVectors: %s\n", ok ? "OK" : "FAILED");
    printf("F: ");
    for (int i = 0; i < n_nodes; i++) printf("%.4e ", F[i]);
    printf("\nQ: ");
    for (int i = 0; i < n_nodes; i++) printf("%.4e ", Q[i]);
    printf("\ndFdx diagonal: ");
    for (int i = 0; i < n_nodes; i++) printf("%.4e ", dFdx[i][i]);
    printf("\n");

    return 0;
}
#endif
