// VAE test: full-wave bridge rectifier — transient simulation
//
// Circuit:
//              D1(a→cathode=out_p, c→anode=n1)
//   n1 ------>------ out_p ------<------ n2
//              D3(a→cathode=out_p, c→anode=n2)
//                      |
//                   [R_load]
//                      |
//   n1 ------<------ out_n ------>------ n2
//              D2(a→cathode=n1, c→anode=out_n)
//              D4(a→cathode=n2, c→anode=out_n)
//
// VA model convention: port "a" = cathode, "c" = anode.
// Forward bias when V(ci,a) = V_anode - V_cathode > 0.
// I(a,ci) = Id > 0 flows cathode→anode inside device.
// External (circuit) current = Id flowing into the anode node.
//
// When n1 > n2 (positive half cycle):
//   D1 conducts: current enters out_p from D1 (I_D1 = d1.Id)
//   D4 conducts: current leaves out_n through D4 (I_D4 = d4.Id)
//
// KCL at out_p: I_D1 + I_D3 - I_R = 0
// KCL at out_n: I_R - I_D2 - I_D4 = 0
//
// Build:
//   g++ -std=c++17 -O2 -o test_bridge test_bridge.cpp -ldl -lm
// Run:
//   LD_LIBRARY_PATH=. ./test_bridge > bridge.csv

#include <cstdio>
#include <cstring>
#include <cmath>
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
    if (!m.handle) { fprintf(stderr, "dlopen(%s): %s\n", path, dlerror()); exit(1); }
    m.eval  = (EvalFn)dlsym(m.handle, "vae_eval");
    m.jacob = (JacobFn)dlsym(m.handle, "vae_jacobian");
    return m;
}

// Evaluate one diode instance (Rs=0, ci shorted to c).
// cathode_V maps to model port a(0), anode_V maps to c(1)=ci(2).
struct DiodeResult {
    double Id;          // branch current (positive in forward bias)
    double dId_dVcath;  // dId/d(cathode voltage) = dF/dV_a
    double dId_dVano;   // dId/d(anode voltage) = dF/dV_c + dF/dV_ci
};

DiodeResult eval_diode(VaeModel& d, double cathode_V, double anode_V,
                       double* params, double Vt) {
    double V[3] = {cathode_V, anode_V, anode_V};
    VaeState s = {V, params, 300.0, Vt, 0.0, 0.0};
    double F[2], Q[2], dF[6], dQ[6];
    memset(F, 0, sizeof F); memset(Q, 0, sizeof Q);
    memset(dF, 0, sizeof dF); memset(dQ, 0, sizeof dQ);
    d.eval(&s, F, Q);
    d.jacob(&s, dF, dQ);
    DiodeResult r;
    r.Id = F[0];
    r.dId_dVcath = dF[0*3 + 0];
    r.dId_dVano  = dF[0*3 + 1] + dF[0*3 + 2];
    return r;
}

int main() {
    VaeModel diod = load_model("vae_diode.so");
    VaeModel res  = load_model("vae_resistor.so");

    const double Vt = 0.02585;
    const double R_load = 100.0;
    double dp[4] = {1e-14, 0.0, 0.0, 300.0};  // Is, Cp, Rs=0, Temp

    const double Vpk = 5.0;
    const double freq = 60.0;
    const double omega = 2.0 * M_PI * freq;
    const double t_end = 3.0 / freq;
    const double dt = 1e-5;

    double Vop = 0.0, Von = 0.0;

    printf("time_ms,V_n1,V_n2,V_out_p,V_out_n,V_load,I_load_mA\n");

    for (double t = 0; t <= t_end; t += dt) {
        double Vn1 =  Vpk * sin(omega * t);
        double Vn2 = -Vpk * sin(omega * t);

        for (int iter = 0; iter < 200; iter++) {
            // D1: cathode=out_p, anode=n1.  Conducts when n1 > out_p.
            DiodeResult d1 = eval_diode(diod, Vop, Vn1, dp, Vt);
            // D2: cathode=n1, anode=out_n.  Conducts when out_n > n1.
            DiodeResult d2 = eval_diode(diod, Vn1, Von, dp, Vt);
            // D3: cathode=out_p, anode=n2.  Conducts when n2 > out_p.
            DiodeResult d3 = eval_diode(diod, Vop, Vn2, dp, Vt);
            // D4: cathode=n2, anode=out_n.  Conducts when out_n > n2.
            DiodeResult d4 = eval_diode(diod, Vn2, Von, dp, Vt);

            // Resistor: I(out_p, out_n) = (Vop - Von) / R
            double Vrl[2] = {Vop, Von};
            double Rp[1] = {R_load};
            VaeState sr = {Vrl, Rp, 300.0, Vt, 0.0, 0.0};
            double Fr[1], Qr[1], dFr[2], dQr[2];
            memset(Fr, 0, sizeof Fr); memset(Qr, 0, sizeof Qr);
            memset(dFr, 0, sizeof dFr); memset(dQr, 0, sizeof dQr);
            res.eval(&sr, Fr, Qr);
            res.jacob(&sr, dFr, dQr);

            // KCL at out_p: D1 + D3 deliver current in, R takes it out.
            //   f0 = d1.Id + d3.Id - Fr[0]
            double f0 = d1.Id + d3.Id - Fr[0];

            // KCL at out_n: R delivers current in, D2 + D4 take it out.
            //   f1 = Fr[0] - d2.Id - d4.Id
            double f1 = Fr[0] - d2.Id - d4.Id;

            // Jacobian 2x2: [df0/dVop, df0/dVon; df1/dVop, df1/dVon]
            // D1 cathode=out_p: dId/dVop = d1.dId_dVcath
            // D3 cathode=out_p: dId/dVop = d3.dId_dVcath
            double J00 = d1.dId_dVcath + d3.dId_dVcath - dFr[0];

            // D2 anode=out_n: dId/dVon = d2.dId_dVano
            // D4 anode=out_n: dId/dVon = d4.dId_dVano
            // f0 doesn't depend on Von through diodes (D1,D3 anodes are n1,n2 = fixed)
            double J01 = -dFr[1];

            double J10 = dFr[0];  // Fr depends on Vop

            // f1 = Fr - d2.Id - d4.Id
            // D2 anode=out_n: d(d2.Id)/dVon = d2.dId_dVano
            // D4 anode=out_n: d(d4.Id)/dVon = d4.dId_dVano
            double J11 = dFr[1] - d2.dId_dVano - d4.dId_dVano;

            double det = J00 * J11 - J01 * J10;
            if (fabs(det) < 1e-30) break;

            double dVop = -(J11 * f0 - J01 * f1) / det;
            double dVon = -(J00 * f1 - J10 * f0) / det;

            if (fabs(dVop) > 2*Vt) dVop = copysign(2*Vt, dVop);
            if (fabs(dVon) > 2*Vt) dVon = copysign(2*Vt, dVon);

            Vop += dVop;
            Von += dVon;

            if (fabs(dVop) < 1e-12 && fabs(dVon) < 1e-12 &&
                fabs(f0) < 1e-14 && fabs(f1) < 1e-14)
                break;
        }

        double V_load = Vop - Von;
        double I_load_mA = V_load / R_load * 1000.0;

        printf("%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f\n",
               t * 1000.0, Vn1, Vn2, Vop, Von, V_load, I_load_mA);
    }

    dlclose(diod.handle);
    dlclose(res.handle);
    return 0;
}
