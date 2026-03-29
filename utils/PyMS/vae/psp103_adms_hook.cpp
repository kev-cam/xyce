// PSP103VA VAE→ADMS hook
//
// Drop-in replacement for the computation in updateIntermediateVars().
// Loads a VAE-compiled regime .so and maps its output to ADMS arrays.
//
// Build: g++ -O2 -shared -fPIC -o psp103_adms_hook.so psp103_adms_hook.cpp -ldl
// Test:  g++ -O2 -DHOOK_STANDALONE_TEST -o test_hook psp103_adms_hook.cpp -ldl -lm

#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <cmath>

struct VaeState {
    double V[16];
    double Vt;
};

typedef void (*VaeEvalFn)(VaeState*, double*, double*);

// ADMS layout for PSP103VA (from N_DEV_ADMSPSP103VA.h)
enum {
    NODE_D=0, NODE_G=1, NODE_S=2, NODE_B=3,
    NODE_NOI=4, NODE_NOI2=5, NODE_GP=6, NODE_SI=7,
    NODE_DI=8, NODE_BP=9, NODE_BI=10, NODE_BS=11, NODE_BD=12,
    N_NODES=13
};
enum {
    PROBE_V_DI_GND=0, PROBE_V_BS_GND=1, PROBE_V_BD_GND=2,
    PROBE_V_SI_GND=3, PROBE_V_BP_GND=4, PROBE_V_GP_GND=5,
    PROBE_V_NOI_GND=6, PROBE_V_NOI2_GND=7,
    PROBE_V_B_BI=8, PROBE_V_BD_BI=9, PROBE_V_BS_BI=10, PROBE_V_BP_BI=11,
    PROBE_V_D_DI=12, PROBE_V_S_SI=13, PROBE_V_G_GP=14,
    PROBE_V_DI_BD=15, PROBE_V_SI_BS=16, PROBE_V_SI_BP=17,
    PROBE_V_DI_SI=18, PROBE_V_GP_SI=19,
    N_PROBES=20
};

// VAE active nodes (after collapsing): D, G, S, B, NOI
// The VAE .so was compiled with these 5 nodes.
enum { VAE_D=0, VAE_G=1, VAE_S=2, VAE_B=3, VAE_NOI=4, VAE_N_NODES=5 };

// Branch mapping: VAE F[branch] → ADMS (node_plus, node_minus)
struct BranchMap {
    int vae_idx;
    int node_plus;   // ADMS node that gets +=
    int node_minus;  // ADMS node that gets -=
    bool is_static;  // true=F (static), false=Q (dynamic)
};

static const BranchMap branch_map[] = {
    // VAE branches from the compiled regime
    { 0, NODE_DI,  NODE_BP, true },   // I(DI,BP) — impact ionization
    { 1, NODE_DI,  NODE_SI, true },   // I(DI,SI) — drain current
    { 2, NODE_GP,  NODE_SI, true },   // I(GP,SI) — gate current
    { 3, NODE_GP,  NODE_DI, true },   // I(GP,DI) — gate current
    { 4, NODE_GP,  NODE_BP, true },   // I(GP,BP) — gate-bulk current
    { 5, NODE_SI,  NODE_BP, true },   // I(SI,BP) — GISL
    { 6, NODE_BS,  NODE_SI, true },   // I(BS,SI) — junction
    { 7, NODE_BD,  NODE_DI, true },   // I(BD,DI) — junction
    // F[8] = V(B,BI) — voltage constraint, handled separately
    { 9, NODE_BP,  NODE_SI, true },   // I(BP,SI)
};
static const int N_BRANCH_MAP = sizeof(branch_map) / sizeof(branch_map[0]);

// Map VAE node voltage index → ADMS GND probe index
// VAE V[0]=V_D → not a GND probe (external node)
// VAE V[1]=V_G → not a GND probe
// VAE V[2]=V_S → not a GND probe
// VAE V[3]=V_B → not a GND probe
// VAE V[4]=V_NOI → PROBE_V_NOI_GND
// For collapsed nodes: V_DI=V_D, V_SI=V_S, V_GP=V_G, V_BP=V_B etc.
// The VAE .so only has 5 active nodes; internal nodes are folded.
static const int vae_node_to_gnd_probe[] = {
    -1,                 // VAE_D → external D (no GND probe, use li_D)
    -1,                 // VAE_G → external G
    -1,                 // VAE_S → external S
    -1,                 // VAE_B → external B
    PROBE_V_NOI_GND,   // VAE_NOI
};

// Hook state
static void* vae_dl = nullptr;
static VaeEvalFn vae_eval = nullptr;
static VaeEvalFn vae_jacobian = nullptr;

extern "C" bool vae_hook_load(const char* so_path) {
    if (vae_dl) dlclose(vae_dl);
    vae_dl = dlopen(so_path, RTLD_NOW);
    if (!vae_dl) { fprintf(stderr, "vae_hook: %s\n", dlerror()); return false; }
    vae_eval = (VaeEvalFn)dlsym(vae_dl, "vae_eval");
    vae_jacobian = (VaeEvalFn)dlsym(vae_dl, "vae_jacobian");
    return vae_eval && vae_jacobian;
}

// Main hook: called from updateIntermediateVars
// probeVars[20]: voltage probes (already computed by ADMS framework)
// staticContributions[13]: F output
// dynamicContributions[13]: Q output
// d_staticContributions[13][20]: dF/dV output
// d_dynamicContributions[13][20]: dQ/dV output
extern "C" bool vae_hook_eval(
    const double* probeVars,
    double* staticContributions,
    double* dynamicContributions,
    double d_staticContributions[][N_PROBES],
    double d_dynamicContributions[][N_PROBES])
{
    if (!vae_eval || !vae_jacobian) return false;

    // Pack probe voltages into VaeState
    // The VAE .so was compiled with collapsed nodes:
    //   V[0]=V_D, V[1]=V_G, V[2]=V_S, V[3]=V_B, V[4]=V_NOI
    // External nodes (D,G,S,B) aren't in probeVars — they come from
    // the solution vector via li_D etc. For the hook, we reconstruct:
    //   V_D = V_DI + probeVars[V_D_DI]  (but V_D_DI = V_D - V_DI, so V_D = V_DI + V_D_DI)
    //   V_DI = probeVars[V_DI_GND]
    VaeState state;
    memset(&state, 0, sizeof(state));
    state.V[VAE_D]   = probeVars[PROBE_V_DI_GND] + probeVars[PROBE_V_D_DI];
    state.V[VAE_G]   = probeVars[PROBE_V_GP_GND] + probeVars[PROBE_V_G_GP];
    state.V[VAE_S]   = probeVars[PROBE_V_SI_GND] + probeVars[PROBE_V_S_SI];
    state.V[VAE_B]   = probeVars[PROBE_V_BP_GND] + probeVars[PROBE_V_BP_BI]; // B = BP + V_B_BI... actually B = BI + V_B_BI
    state.V[VAE_NOI] = probeVars[PROBE_V_NOI_GND];
    state.Vt = 8.617087e-5 * 300.15;

    // Call VAE eval
    double F_vae[16], Q_vae[16];
    memset(F_vae, 0, sizeof(F_vae));
    memset(Q_vae, 0, sizeof(Q_vae));
    vae_eval(&state, F_vae, Q_vae);

    // Stamp branch contributions into node arrays
    for (int b = 0; b < N_BRANCH_MAP; b++) {
        const BranchMap& bm = branch_map[b];
        double val = F_vae[bm.vae_idx];
        if (bm.node_plus >= 0)
            staticContributions[bm.node_plus] += val;
        if (bm.node_minus >= 0)
            staticContributions[bm.node_minus] -= val;
    }

    // Handle V(B,BI) voltage constraint
    // F[8] in VAE = -(V_B) = voltage source equation
    staticContributions[NODE_B] += F_vae[8];
    staticContributions[NODE_BI] -= F_vae[8];

    // Stamp Q contributions similarly
    for (int b = 0; b < N_BRANCH_MAP; b++) {
        const BranchMap& bm = branch_map[b];
        double val = Q_vae[bm.vae_idx];
        if (val == 0.0) continue;
        if (bm.node_plus >= 0)
            dynamicContributions[bm.node_plus] += val;
        if (bm.node_minus >= 0)
            dynamicContributions[bm.node_minus] -= val;
    }

    // Call VAE jacobian
    double dFdV[16 * 8], dQdV[16 * 8];
    memset(dFdV, 0, sizeof(dFdV));
    memset(dQdV, 0, sizeof(dQdV));
    vae_jacobian(&state, dFdV, dQdV);

    // Map VAE node-voltage Jacobian to ADMS probe-indexed Jacobian
    // VAE: dFdV[branch * VAE_N_NODES + vae_node]
    // ADMS: d_staticContributions[adms_node][probeID]
    //
    // For GND probes (V_DI_GND etc.), the derivative w.r.t. probe =
    // derivative w.r.t. node voltage directly.
    // For external nodes (D,G,S,B) without GND probes, we use the
    // series resistance probes: V_D_DI, V_G_GP, V_S_SI
    //
    // Since the VAE .so was compiled with collapsed nodes (DI=D, SI=S, GP=G),
    // dF/dV_D in VAE corresponds to dF/d(V_DI_GND) + dF/d(V_D_DI) in ADMS.
    // With collapse, V_D_DI = 0 always, so dF/dV_D ≈ dF/d(V_DI_GND).

    for (int b = 0; b < N_BRANCH_MAP; b++) {
        const BranchMap& bm = branch_map[b];
        // For each VAE node voltage, find the corresponding ADMS probe
        // VAE node 0 (V_D) → ADMS probe V_DI_GND (collapsed)
        // VAE node 1 (V_G) → ADMS probe V_GP_GND (collapsed)
        // VAE node 2 (V_S) → ADMS probe V_SI_GND (collapsed)
        // VAE node 3 (V_B) → ADMS probe V_BP_GND (collapsed, B≈BP≈BI)
        // VAE node 4 (V_NOI) → ADMS probe V_NOI_GND
        static const int vae_to_probe[] = {
            PROBE_V_DI_GND,  // VAE_D → V_DI (collapsed)
            PROBE_V_GP_GND,  // VAE_G → V_GP (collapsed)
            PROBE_V_SI_GND,  // VAE_S → V_SI (collapsed)
            PROBE_V_BP_GND,  // VAE_B → V_BP (collapsed)
            PROBE_V_NOI_GND, // VAE_NOI
        };

        for (int v = 0; v < VAE_N_NODES; v++) {
            double dval = dFdV[bm.vae_idx * VAE_N_NODES + v];
            if (dval == 0.0) continue;
            int probe = vae_to_probe[v];
            if (bm.node_plus >= 0)
                d_staticContributions[bm.node_plus][probe] += dval;
            if (bm.node_minus >= 0)
                d_staticContributions[bm.node_minus][probe] -= dval;
        }
    }

    // V(B,BI) jacobian
    for (int v = 0; v < VAE_N_NODES; v++) {
        double dval = dFdV[8 * VAE_N_NODES + v];
        if (dval == 0.0) continue;
        static const int vae_to_probe[] = {
            PROBE_V_DI_GND, PROBE_V_GP_GND, PROBE_V_SI_GND,
            PROBE_V_BP_GND, PROBE_V_NOI_GND
        };
        int probe = vae_to_probe[v];
        d_staticContributions[NODE_B][probe] += dval;
        d_staticContributions[NODE_BI][probe] -= dval;
    }

    return true;
}

#ifdef HOOK_STANDALONE_TEST
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <regime.so>\n", argv[0]);
        return 1;
    }

    if (!vae_hook_load(argv[1])) {
        fprintf(stderr, "Failed to load .so\n");
        return 1;
    }

    // Simulate probeVars at VGS=0.6V VDS=0.6V
    double probeVars[N_PROBES] = {};
    probeVars[PROBE_V_DI_GND] = 0.6;  // V_DI = VD (collapsed)
    probeVars[PROBE_V_GP_GND] = 0.6;  // V_GP = VG (collapsed)
    probeVars[PROBE_V_SI_GND] = 0.0;  // V_SI = VS
    probeVars[PROBE_V_BP_GND] = 0.0;  // V_BP = VB
    probeVars[PROBE_V_NOI_GND] = 0.0;
    // Derived probes
    probeVars[PROBE_V_DI_SI] = 0.6;   // V_DI - V_SI
    probeVars[PROBE_V_GP_SI] = 0.6;   // V_GP - V_SI

    double sC[N_NODES] = {};
    double dC[N_NODES] = {};
    double dsC[N_NODES][N_PROBES] = {};
    double ddC[N_NODES][N_PROBES] = {};

    bool ok = vae_hook_eval(probeVars, sC, dC, dsC, ddC);
    printf("vae_hook_eval: %s\n", ok ? "OK" : "FAILED");

    printf("\nstaticContributions (non-zero):\n");
    const char* names[] = {"D","G","S","B","NOI","NOI2","GP","SI","DI","BP","BI","BS","BD"};
    for (int i = 0; i < N_NODES; i++)
        if (sC[i] != 0.0) printf("  [%2d] %-4s = %+.6e\n", i, names[i], sC[i]);

    printf("\nd_staticContributions (non-zero):\n");
    for (int i = 0; i < N_NODES; i++)
        for (int j = 0; j < N_PROBES; j++)
            if (dsC[i][j] != 0.0)
                printf("  [%s][probe %d] = %+.6e\n", names[i], j, dsC[i][j]);

    // Benchmark
    int N = 1000000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        memset(sC, 0, sizeof(sC));
        memset(dC, 0, sizeof(dC));
        memset(dsC, 0, sizeof(dsC));
        memset(ddC, 0, sizeof(ddC));
        vae_hook_eval(probeVars, sC, dC, dsC, ddC);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);
    printf("\nBenchmark: %.1f ns/call (%d iterations, %.3fs)\n",
           dt*1e9/N, N, dt);

    return 0;
}
#endif
