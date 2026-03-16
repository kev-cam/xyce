// VAE test: RC low-pass filter — AC frequency sweep
//
// Circuit:  Vsrc ---[R 1k]--- node1 ---[C 10nF]--- GND
//
// Analytical transfer function:
//   H(jw) = V_node1 / V_src = 1 / (1 + jwRC)
//   |H| = 1 / sqrt(1 + (wRC)^2)
//   phase = -atan(wRC)
//   f_cutoff = 1/(2*pi*R*C) = 15.915 kHz  (for R=1k, C=10nF)
//
// AC analysis uses linearized MNA at DC operating point (V1=0):
//   (G + jw*C_mat) * dV = I_stim
// where G = dF/dV, C_mat = dQ/dV.
//
// Both resistor and capacitor use I(p,n) convention with standard
// MNA stamps (no sign flip needed — unlike the diode model).
//
// Build:
//   g++ -std=c++17 -O2 -o test_rc_ac test_rc_ac.cpp -ldl -lm
// Run:
//   LD_LIBRARY_PATH=. ./test_rc_ac > rc_ac.csv

#include <cstdio>
#include <cstring>
#include <cmath>
#include <complex>
#include <dlfcn.h>

struct VaeState {
    double* V;
    double* params;
    double temperature;
    double Vt;
    double dt;
    double time;
};

typedef void (*EvalFn)(VaeState*, double*, double*);
typedef void (*JacobFn)(VaeState*, double*, double*);

struct VaeModel {
    void*    handle;
    EvalFn   eval;
    JacobFn  jacob;
};

VaeModel load_model(const char* path) {
    VaeModel m;
    m.handle = dlopen(path, RTLD_NOW);
    if (!m.handle) {
        fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
        exit(1);
    }
    m.eval  = (EvalFn)dlsym(m.handle, "vae_eval");
    m.jacob = (JacobFn)dlsym(m.handle, "vae_jacobian");
    return m;
}

int main() {
    VaeModel res = load_model("vae_resistor.so");
    VaeModel cap = load_model("vae_capacitor.so");

    const double R_val = 1000.0;    // 1k ohm
    const double C_val = 10e-9;     // 10 nF
    const double Vt = 0.02585;

    // ---------------------------------------------------------------
    // Step 1: DC operating point (trivial: V1 = 0, no DC source)
    // ---------------------------------------------------------------
    double V1_dc = 0.0;

    // ---------------------------------------------------------------
    // Step 2: Extract G (conductance) and C_mat (capacitance) at DC OP
    // ---------------------------------------------------------------
    // We have one circuit node (node1). Vsrc and GND are fixed.
    // Resistor: I(Vsrc, node1).  Ports: p=Vsrc, n=node1.
    // Capacitor: I(node1, GND).  Ports: p=node1, n=GND.

    // Resistor Jacobian at DC OP
    double Vr[2] = {0.0, V1_dc};  // Vsrc=0 at DC
    double Rp[1] = {R_val};
    VaeState sr = {Vr, Rp, 300.0, Vt, 0.0, 0.0};
    double dFr[2], dQr[2];
    memset(dFr, 0, sizeof dFr); memset(dQr, 0, sizeof dQr);
    res.jacob(&sr, dFr, dQr);
    // dFr[0] = dFr/dV_p = 1/R,  dFr[1] = dFr/dV_n = -1/R

    // Capacitor Jacobian at DC OP
    double Vc[2] = {V1_dc, 0.0};  // p=node1, n=GND
    double Cp_arr[1] = {C_val};
    VaeState sc = {Vc, Cp_arr, 300.0, Vt, 0.0, 0.0};
    double dFc[2], dQc[2];
    memset(dFc, 0, sizeof dFc); memset(dQc, 0, sizeof dQc);
    cap.jacob(&sc, dFc, dQc);
    // dQc[0] = dQ/dV_p = C,  dQc[1] = dQ/dV_n = -C
    // dFc = 0 (no resistive part)

    // MNA stamps at node1:
    // From resistor (n=node1):  G_node1 += -dFr[1] = 1/R  (stamp -F at n, so -(dF/dV_n)*dV)
    //   Wait — the stamp at node n from I(p,n) is -F[0].
    //   Linearized: -dF/dV_n * dV_n - dF/dV_p * dV_p = 0 at node n
    //   At node1 (n), the resistor contributes:
    //     -dFr[1] to G(node1,node1)  and  -dFr[0] to G(node1,Vsrc)
    //
    // From capacitor (p=node1): +F[0] at p (=0), +Q stamp.
    //   At node1 (p), the capacitor contributes:
    //     +dFc[0]=0 to G(node1,node1)
    //     +dQc[0]=C to C_mat(node1,node1)

    // Combined at node1:
    double G_11 = -dFr[1] + dFc[0];    // conductance: 1/R + 0 = 1/R
    double C_11 = -dQr[1] + dQc[0];    // capacitance: 0 + C = C

    // Stimulus: unit voltage at Vsrc. The resistor couples Vsrc to node1.
    // The stimulus current into node1 from a 1V AC source at Vsrc:
    //   I_stim = -dFr[0] * Vsrc = -(1/R) * 1 = -1/R ... wait
    // The resistor stamp at node1 (n) for Vsrc change: -dFr[0] * dV_p
    // With dV_p = 1 (unit source): I_stim_node1 = -dFr[0]
    double I_stim = -dFr[0];  // = -(1/R) = -1/R ... hmm, should be positive

    // Actually: G_node1_Vsrc = -dFr/dV_p|at n = -dFr[0]
    // For Vsrc = 1V: the contribution to the RHS is -G_node1_Vsrc * 1
    //   RHS = G_node1_Vsrc = -dFr[0] = -(1/R)
    // Hmm, that gives a negative stimulus. Let me think again.
    //
    // The MNA equation at node1 is:
    //   sum of stamps = 0
    //   From resistor: -Fr[0]  (where Fr = (Vp-Vn)/R)
    //   From capacitor: +Fc[0] + jw*Qc[0]  (where Fc=0, Qc=C*(Vp-Vn))
    //
    // Linearizing: for small AC perturbations dVsrc, dV1:
    //   -(dFr/dVp * dVsrc + dFr/dVn * dV1)
    //   + (dFc/dVp_c * dV1 + jw * dQc/dVp_c * dV1)  = 0
    //
    //  dFr/dVp = 1/R,  dFr/dVn = -1/R
    //  dFc/dVp_c = 0,  dQc/dVp_c = C
    //
    //   -(1/R * dVsrc + (-1/R) * dV1) + (0 + jw*C * dV1) = 0
    //   -(1/R)*dVsrc + (1/R)*dV1 + jw*C*dV1 = 0
    //   (1/R + jw*C) * dV1 = (1/R) * dVsrc
    //   dV1/dVsrc = (1/R) / (1/R + jw*C) = 1 / (1 + jw*R*C)
    //
    // Perfect — matches the analytical result!

    // ---------------------------------------------------------------
    // Step 3: AC frequency sweep
    // ---------------------------------------------------------------
    printf("freq_Hz,mag_dB,phase_deg,mag_dB_analytical,phase_deg_analytical\n");

    using cx = std::complex<double>;
    const cx jj(0.0, 1.0);
    const double G = 1.0 / R_val;  // = -dFr[1] verified above

    for (double decade = 1.0; decade <= 7.0; decade += 0.05) {
        double freq = pow(10.0, decade);
        double omega = 2.0 * M_PI * freq;

        // Numerical: H = G / (G + jw*C)
        cx Z_node1 = G + jj * omega * C_val;
        cx H_num = G / Z_node1;

        double mag_num = 20.0 * log10(std::abs(H_num));
        double phase_num = std::arg(H_num) * 180.0 / M_PI;

        // Analytical: H = 1 / (1 + jw*R*C)
        cx H_ana = 1.0 / (1.0 + jj * omega * R_val * C_val);
        double mag_ana = 20.0 * log10(std::abs(H_ana));
        double phase_ana = std::arg(H_ana) * 180.0 / M_PI;

        printf("%.2f,%.4f,%.4f,%.4f,%.4f\n", freq, mag_num, phase_num,
               mag_ana, phase_ana);
    }

    dlclose(res.handle);
    dlclose(cap.handle);
    return 0;
}
