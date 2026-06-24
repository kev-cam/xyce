// Prototype: threaded device LOAD, faithful to Xyce's pattern.
//   Circuit = inverter chain: M nodes, one "device" per node connecting (i,i+1)
//   (like N_DEV_Resistor/MOSFET Instance::loadDAEFVector). Each device:
//     i0 = eval(sol[a],sol[b]);  fVec[a] += i0;  fVec[b] -= i0;   (shared fVec)
//   The eval is the expensive part (transcendental, ~MOSFET); the stamp is cheap
//   but RACES on shared nodes. Three loaders, ITERS Newton-iters each:
//     serial   : the reference
//     atomic   : threads stamp shared fVec via atomic add (simplest correct)
//     tlbuf    : threads stamp into per-thread fVec, merged by node-range (no
//                contention; the general/production approach)
//   Threads are core-pinned (sched_setaffinity) + sense-reversing spinlock barrier.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

static inline void atomic_add_d(double* p, double v){
  uint64_t* up=(uint64_t*)p, old=__atomic_load_n(up,__ATOMIC_RELAXED), nw; double d;
  do { __builtin_memcpy(&d,&old,8); d+=v; __builtin_memcpy(&nw,&d,8); }
  while(!__atomic_compare_exchange_n(up,&old,nw,1,__ATOMIC_RELAXED,__ATOMIC_RELAXED));
}

static long    M;          // nodes (~= circuit size)
static int     P, ITERS, PIN, MODE;   // MODE: 0 serial,1 atomic,2 tlbuf
static double *sol, *fVec;
static double **tlf;       // tlf[t][M] per-thread buffers (tlbuf mode)
static atomic_int bc, bs;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                         return t.tv_sec + t.tv_nsec*1e-9; }
static __thread int t_sense=0;
static void barrier(void){
  int s=(t_sense^=1);
  if(atomic_fetch_add(&bc,1)+1==P){ atomic_store(&bc,0); atomic_store(&bs,s); }
  else while(atomic_load_explicit(&bs,memory_order_acquire)!=s) __builtin_ia32_pause();
}
// MOSFET-ish I-V: a couple of exp() — the expensive per-device math.
static inline double eval(double va,double vb){
  double vds=va-vb, vgs=va*0.5+0.7;
  double id = (vgs-0.4)>0 ? 120e-6*(vgs-0.4)*(vgs-0.4)*(1.0+0.02*vds) : 0.0;
  id += 1e-14*(exp((vds)*38.6)-1.0);          // body-diode-ish
  return id;
}
static double g_t0,g_t1;

static void* worker(void* p){
  int tid=(int)(long)p;
  if(PIN){ int cpu=(tid<16)?tid*2:(tid-16)*2+1;
           cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s); sched_setaffinity(0,sizeof(s),&s); }
  long lo=M*tid/P, hi=M*(tid+1)/P;
  barrier(); if(tid==0) g_t0=now();
  for(int it=0; it<ITERS; ++it){
    if(MODE==1){                              // atomic stamp into shared fVec
      for(long i=lo;i<hi;++i) fVec[i]=0.0;     // zero own range
      if(tid==P-1) fVec[M]=0.0;
      barrier();
      for(long i=lo;i<hi;++i){ double d=eval(sol[i],sol[i+1]);
        atomic_add_d(&fVec[i],   d);
        atomic_add_d(&fVec[i+1],-d); }
      barrier();
    } else {                                  // tlbuf: stamp local, smart merge
      double* f=tlf[tid];
      for(long i=lo;i<=hi;++i) f[i]=0.0;       // zero own range + right boundary node
      for(long i=lo;i<hi;++i){ double d=eval(sol[i],sol[i+1]); f[i]+=d; f[i+1]-=d; }
      barrier();
      for(long n=lo;n<hi;++n) fVec[n]=f[n];    // own contributions, O(M) total
      if(tid>0) fVec[lo]+=tlf[tid-1][lo];      // + left neighbour's boundary device
      barrier();
    }
  }
  if(tid==0) g_t1=now();
  return 0;
}

static double run_serial(void){
  double t0=now();
  for(int it=0;it<ITERS;++it){
    memset(fVec,0,M*sizeof(double));
    for(long i=0;i<M;++i){ double d=eval(sol[i],sol[i+1]); fVec[i]+=d; fVec[i+1]-=d; }
  }
  return now()-t0;
}

int main(int argc,char**argv){
  M=(argc>1)?atol(argv[1]):200000;
  ITERS=(argc>2)?atoi(argv[2]):500;
  PIN=(argc>3)?atoi(argv[3]):1;
  sol=aligned_alloc(64,(M+1)*sizeof(double));
  fVec=aligned_alloc(64,(M+1)*sizeof(double));
  for(long i=0;i<=M;++i) sol[i]=0.3+0.4*((i*2654435761u>>16)&255)/255.0;
  double ser=run_serial();
  double serchk=fVec[M/2];
  printf("M=%ld (~%.1fMB load WS)  ITERS=%d  pin=%d\n", M, (double)M*16/1048576.0, ITERS, PIN);
  printf("  serial            %.3fs   1.00x\n", ser);
  for(int mode=1; mode<=2; ++mode){
    const char* nm = mode==1?"atomic":"tlbuf ";
    for(int p=2;p<=32;p*=2){
      P=p; ITERS=ITERS; MODE=mode; atomic_store(&bc,0); atomic_store(&bs,0);
      if(mode==2){ tlf=malloc(P*sizeof(double*)); for(int t=0;t<P;++t) tlf[t]=aligned_alloc(64,(M+1)*sizeof(double)); }
      pthread_t th[64];
      double t0=now();
      for(int t=0;t<P;++t) pthread_create(&th[t],0,worker,(void*)(long)t);
      for(int t=0;t<P;++t) pthread_join(th[t],0);
      double w=g_t1-g_t0;
      double chk=fVec[M/2];
      printf("  %s  P=%-2d  %.3fs  %4.1fx  %s\n", nm, p, w, ser/w,
             (fabs(chk-serchk)<1e-9*fabs(serchk)+1e-15)?"ok":"MISMATCH");
      if(mode==2){ for(int t=0;t<P;++t) free(tlf[t]); free(tlf); }
    }
  }
  return 0;
}
