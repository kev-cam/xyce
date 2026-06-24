// Microbenchmark: parallel device-model evaluation with processor affinity.
// Models Xyce's hot path: N independent "devices", each evaluated per Newton
// iteration (transcendental-heavy, like diode/MOSFET), partitioned across P
// core-pinned threads, synchronised by a sense-reversing spinlock barrier
// (the ShmComm primitive). Measures throughput vs threads, pinned vs not,
// and across working-set sizes to expose the aggregate-L2 ("32 caches") effect.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { double v[8]; double stamp[8]; } Device;   // 128 B / device

static Device*  g_dev;
static long     g_n;
static int      g_iters;
static int      g_P;
static int      g_pin;
static atomic_int g_count;
static atomic_int g_sense;
static double   g_t0, g_t1;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                         return t.tv_sec + t.tv_nsec*1e-9; }

// sense-reversing spinlock barrier (no futex/kernel)
static __thread int t_sense = 0;
static void barrier(void){
  int s = (t_sense ^= 1);
  if (atomic_fetch_add(&g_count,1)+1 == g_P){
    atomic_store(&g_count,0);
    atomic_store(&g_sense,s);
  } else {
    while (atomic_load_explicit(&g_sense,memory_order_acquire)!=s)
      __builtin_ia32_pause();
  }
}

static inline void eval(Device* d){
  double g=0;
  for(int i=0;i<8;i++){ double x=d->v[i]; g += exp(x*0.0625) + x*x; }
  for(int i=0;i<8;i++) d->stamp[i] = g*d->v[i] - d->stamp[i];
}

typedef struct { int tid; } Arg;

static void* worker(void* p){
  int tid = ((Arg*)p)->tid;
  if (g_pin){
    // fill physical cores first (even cpus 0,2,..), then SMT siblings (odd)
    int cpu = (tid < 16) ? tid*2 : (tid-16)*2+1;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu,&set);
    sched_setaffinity(0,sizeof(set),&set);
  }
  long lo = g_n *  tid    / g_P;          // contiguous (cache-local) partition
  long hi = g_n * (tid+1) / g_P;
  barrier();
  if (tid==0) g_t0 = now();
  for (int it=0; it<g_iters; ++it){
    for (long i=lo;i<hi;++i) eval(&g_dev[i]);
    barrier();                            // per-Newton-iteration sync
  }
  if (tid==0) g_t1 = now();
  return 0;
}

int main(int argc,char**argv){
  g_n     = (argc>1)? atol(argv[1]) : 1L<<18;   // devices
  g_P     = (argc>2)? atoi(argv[2]) : 1;
  g_iters = (argc>3)? atoi(argv[3]) : 200;
  g_pin   = (argc>4)? atoi(argv[4]) : 0;
  g_dev = aligned_alloc(64, g_n*sizeof(Device));
  for(long i=0;i<g_n;i++){ for(int k=0;k<8;k++){g_dev[i].v[k]=0.5+0.001*k; g_dev[i].stamp[k]=0;} }
  pthread_t th[64]; Arg arg[64];
  atomic_store(&g_count,0); atomic_store(&g_sense,0);
  for(int t=0;t<g_P;t++){ arg[t].tid=t; pthread_create(&th[t],0,worker,&arg[t]); }
  for(int t=0;t<g_P;t++) pthread_join(th[t],0);
  double wall=g_t1-g_t0;
  double evps = (double)g_n*g_iters/wall;
  printf("N=%-9ld WS=%5.1fMB P=%-2d pin=%d  %.3fs  %.1f Meval/s\n",
         g_n, g_n*sizeof(Device)/1048576.0, g_P, g_pin, wall, evps/1e6);
  return 0;
}
