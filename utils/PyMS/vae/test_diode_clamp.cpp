// VAE test: diode clamp circuit
//
// Circuit:  Vsup ---[R 1k]--- node1 ---[D]--- GND
//
// Sign convention note:
//   The diode_simple.va model defines I(a,ci) <+ Is*(limexp(V(ci,a)/Vt)-1).
//   Port "a" is the cathode, "c" is the anode (despite the naming).
//   In forward bias (V_ci > V_a), Id > 0, and the branch I(a,ci) carries
//   current from cathode→anode INSIDE the device.  This is opposite to the
//   SPICE convention where I_D > 0 flows anode→cathode.
//
//   For circuit MNA:  the standard stamp for I(a,ci)=Id puts -Id at ci(anode).
//   But SPICE puts +I_D at the anode.  Since Id = I_D numerically, the diode
//   stamp must be NEGATED in the KCL assembly.
//
//   Connection: a=GND(cathode), c=ci=node1(anode), Rs=0.
//   KCL at node1:  +Fd[0] - Fr[0] = 0   (negated diode + normal resistor)
//
// Build:
//   g++ -std=c++17 -O2 -o test_diode_clamp test_diode_clamp.cpp -ldl -lm
// Run:
//   LD_LIBRARY_PATH=. ./test_diode_clamp > diode_clamp.csv

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
    if (!m.handle) {
        fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
        exit(1);
    }
    m.eval  = (EvalFn)dlsym(m.handle, "vae_eval");
    m.jacob = (JacobFn)dlsym(m.handle, "vae_jacobian");
    return m;
}

int main() {
    VaeModel res  = load_model("vae_resistor.so");
    VaeModel diod = load_model("vae_diode.so");

    const double R_val = 1000.0;
    const double Vt = 0.02585;   // kT/q at 300K

    // Diode params: Is=1e-14, Cp=0, Rs=0 (short ci to c), Temp=300
    double diode_params[4] = {1e-14, 0.0, 0.0, 300.0};

    printf("Vsup,V_node1,I_mA\n");

    double V1 = 0.0;  // initial guess, carried across sweep

    for (double Vsup = 0.0; Vsup <= 2.001; Vsup += 0.01) {

        for (int iter = 0; iter < 200; iter++) {
            // --- Resistor: I(Vsup, node1) ---
            // ports: p=Vsup, n=node1
            double Vr[2] = {Vsup, V1};
            double Rp[1] = {R_val};
            VaeState sr = {Vr, Rp, 300.0, Vt, 0.0, 0.0};
            double Fr[1], Qr[1], dFr[2], dQr[2];
            memset(Fr, 0, sizeof Fr); memset(Qr, 0, sizeof Qr);
            memset(dFr, 0, sizeof dFr); memset(dQr, 0, sizeof dQr);
            res.eval(&sr, Fr, Qr);
            res.jacob(&sr, dFr, dQr);
            // Fr[0] = (Vsup-V1)/R,  dFr[1] = dFr/dV_n = -1/R

            // --- Diode: I(a=GND, ci=node1) ---
            // a(0)=GND, c(1)=node1, ci(2)=node1  (Rs=0 shorts ci to c)
            double Vd[3] = {0.0, V1, V1};
            VaeState sd = {Vd, diode_params, 300.0, Vt, 0.0, 0.0};
            double Fd[2], Qd[2], dFd[6], dQd[6];
            memset(Fd, 0, sizeof Fd); memset(Qd, 0, sizeof Qd);
            memset(dFd, 0, sizeof dFd); memset(dQd, 0, sizeof dQd);
            diod.eval(&sd, Fd, Qd);
            diod.jacob(&sd, dFd, dQd);
            // Fd[0] = Id = Is*(exp(V1/Vt)-1)

            // KCL at node1 (negate diode stamp, see header comment):
            //   f = +Fd[0] - Fr[0] = 0
            double f = Fd[0] - Fr[0];

            // Jacobian: df/dV1 = dFd[0]/dV_c + dFd[0]/dV_ci - dFr[0]/dV_n
            double J = dFd[0*3+1] + dFd[0*3+2] - dFr[0*2+1];

            if (fabs(J) < 1e-30) break;
            double dV = -f / J;

            // Limit voltage step to 2*Vt for diode stability
            if (fabs(dV) > 2.0 * Vt)
                dV = copysign(2.0 * Vt, dV);

            V1 += dV;

            if (fabs(dV) < 1e-12 && fabs(f) < 1e-14)
                break;
        }

        double I_mA = (Vsup - V1) / R_val * 1000.0;
        printf("%.4f,%.6f,%.6f\n", Vsup, V1, I_mA);
    }

    dlclose(res.handle);
    dlclose(diod.handle);
    return 0;
}
