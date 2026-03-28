#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <time.h>
#include <string.h>

typedef struct { double V[12]; double Vt; } VaeState;
typedef void (*EvalFn)(VaeState*, double*, double*);

int main(int argc, char** argv) {
    const char* so_path = argv[1];
    void* lib = dlopen(so_path, RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    EvalFn eval_fn = (EvalFn)dlsym(lib, "vae_eval");
    EvalFn jac_fn = (EvalFn)dlsym(lib, "vae_jacobian");
    if (!eval_fn || !jac_fn) { fprintf(stderr, "dlsym failed\n"); return 1; }

    VaeState s;
    memset(&s, 0, sizeof(s));
    s.V[0] = 0.6; s.V[1] = 0.6; /* VD, VG */
    s.V[5] = 0.6; s.V[7] = 0.6; /* GP=VG, DI=VD */
    s.Vt = 0.02586;

    double F[16], Q[16], dFdV[256], dQdV[256];
    int N = 1000000;

    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        eval_fn(&s, F, Q);
        jac_fn(&s, dFdV, dQdV);
    }

    /* Benchmark eval */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        eval_fn(&s, F, Q);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_eval = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    /* Benchmark jacobian */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        jac_fn(&s, dFdV, dQdV);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_jac = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    /* Benchmark both */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        eval_fn(&s, F, Q);
        jac_fn(&s, dFdV, dQdV);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_both = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    printf("VAE PSP103.6 bare C benchmark (%d iterations)\n", N);
    printf("  Eval:     %.2f ns/call  (%.0f calls/s)\n", dt_eval*1e9/N, N/dt_eval);
    printf("  Jacobian: %.2f ns/call  (%.0f calls/s)\n", dt_jac*1e9/N, N/dt_jac);
    printf("  Both:     %.2f ns/call  (%.0f calls/s)\n", dt_both*1e9/N, N/dt_both);
    printf("  Ids = %.6e A\n", F[1]);

    dlclose(lib);
    return 0;
}
