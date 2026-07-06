//-------------------------------------------------------------------------
// N_DEV_ThreadPool.h
//
// A persistent, core-pinned, PURE-SPIN fork/join thread pool for the device
// per-instance load passes (e.g. Master::updateState). The device load is
// ~76% of a large-circuit step and embarrassingly parallel (each instance's
// updateIntermediateVars writes only its own members + its own per-instance
// state-vector indices -- no shared-node contention), so we just partition the
// instance loop across pinned threads.
//
// Design (per dkc / the nvc model): workers SPIN, never sleep. The per-Newton-
// iteration dispatch happens ~100k times per run; thread suspend/restart is
// ~us and would dwarf a dispatch, so we trade a hot core for minimum response
// time. The dispatch generation `gen_` is a single line workers only READ
// (shared, no coherence storm); completion is signalled via PER-WORKER flags,
// each padded onto its own cache line, so the barrier has NO contended shared
// counter (a single fetch_add counter ping-pongs P-ways/dispatch).
//
// MPI is the wrong primitive on one node (message-multiplexed, sync/imbalance
// bound). Enabled only when XYCE_DEVICE_THREADS > 1 (env); default off =>
// serial, behaviour identical to before. Pin map fills physical cores first.
//-------------------------------------------------------------------------

#ifndef Xyce_N_DEV_ThreadPool_h
#define Xyce_N_DEV_ThreadPool_h

#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#endif

namespace Xyce {
namespace Device {

class ThreadPool
{
public:
  // Process-wide singleton. P from XYCE_DEVICE_THREADS (clamped to hw threads).
  static ThreadPool & instance()
  {
    static ThreadPool pool;
    return pool;
  }

  int size() const { return P_; }

  // Run fn(lo,hi) over a P-way contiguous partition of [0,n). The calling
  // thread runs partition 0 (stays hot/pinned), workers run 1..P-1. Barrier on
  // return. Falls back to a single serial call when P<=1 or n is tiny.
  void parallel_for( std::size_t n, const std::function<void(std::size_t,std::size_t)> & fn )
  {
    if ( P_ <= 1 || n < (std::size_t) P_ * 64 ) { fn( 0, n ); return; }
    n_  = n;
    fn_ = &fn;
    std::uint64_t g = gen_.fetch_add( 1, std::memory_order_release ) + 1;  // dispatch
    fn( 0, part_hi( 0 ) );                               // main does partition 0
    // Pure low-latency spin on per-worker flags (each its own cache line) --
    // no contended shared counter, no sleep.
    for ( int t = 1; t < P_; ++t )
      while ( arrived_[t].v.load( std::memory_order_acquire ) < g )
        cpu_relax();
  }

private:
  ThreadPool()
  {
    P_ = 1;
    if ( const char * e = std::getenv( "XYCE_DEVICE_THREADS" ) )
    {
      int v = std::atoi( e );
      int hw = (int) std::thread::hardware_concurrency();
      if ( hw <= 0 ) hw = 1;
      if ( v < 1 ) v = 1; if ( v > hw ) v = hw;
      if ( v > 64 ) v = 64;
      P_ = v;
    }
    build_cpu_map_();
    pin_self( 0 );
    for ( int t = 1; t < P_; ++t )
      workers_.emplace_back( [this,t]{ worker( t ); } );
  }

  // CCX-aware participant->cpu map. XYCE_CCX_SETS gives one cpu list per
  // cache domain, colon separated (e.g. "0-7,16-23:8-15,24-31" for a 2-CCD
  // Threadripper). Two placement modes:
  //   - one rank per domain (local MPI size >= #sets): rank r's whole pool
  //     binds inside set[r % #sets] -- ranks get disjoint L3 domains;
  //   - single rank spanning domains: consecutive participants bind in
  //     contiguous blocks per set, so parallel_for's contiguous instance
  //     ranges align device blocks with cache domains (the devices a CCD's
  //     workers touch stay resident in that CCD's L3).
  // Unset => the legacy physical-cores-first heuristic, unchanged.
  void build_cpu_map_()
  {
    const char * spec = std::getenv( "XYCE_CCX_SETS" );
    if ( !spec ) return;
    std::vector< std::vector<int> > sets( 1 );
    for ( const char * p = spec; ; ++p )
    {
      if ( *p >= '0' && *p <= '9' )
      {
        int a = (int) std::strtol( p, (char **) &p, 10 ), b = a;
        if ( *p == '-' ) b = (int) std::strtol( p + 1, (char **) &p, 10 );
        for ( int c = a; c <= b; ++c ) sets.back().push_back( c );
        --p;
      }
      else if ( *p == ':' ) sets.push_back( std::vector<int>() );
      else if ( !*p ) break;
    }
    while ( !sets.empty() && sets.back().empty() ) sets.pop_back();
    if ( sets.empty() ) return;
    int nsets = (int) sets.size(), lrank = 0, lsize = 1;
    if ( const char * e = std::getenv( "OMPI_COMM_WORLD_LOCAL_RANK" ) ) lrank = std::atoi( e );
    if ( const char * e = std::getenv( "OMPI_COMM_WORLD_LOCAL_SIZE" ) ) lsize = std::atoi( e );

    cpumap_.assign( P_, -1 );
    if ( lsize >= nsets )
    {
      const std::vector<int> & s = sets[ lrank % nsets ];
      for ( int t = 0; t < P_; ++t ) cpumap_[t] = s[ t % s.size() ];
    }
    else
    {
      for ( int t = 0; t < P_; ++t )
      {
        int st = t * nsets / P_;
        int lo = ( st * P_ + nsets - 1 ) / nsets; /* first participant of set */
        const std::vector<int> & s = sets[st];
        cpumap_[t] = s[ ( t - lo ) % s.size() ];
      }
    }
    if ( lrank == 0 && std::getenv( "XYCE_DEVICE_THREADS" ) )
    {
      std::fprintf( stderr, "[threadpool] CCX map (P=%d, %d sets, lrank %d/%d):",
                    P_, nsets, lrank, lsize );
      for ( int t = 0; t < P_; ++t ) std::fprintf( stderr, " %d", cpumap_[t] );
      std::fprintf( stderr, "\n" );
    }
  }
  ~ThreadPool()
  {
    stop_.store( true, std::memory_order_release );
    gen_.fetch_add( 1, std::memory_order_release );
    for ( auto & w : workers_ ) if ( w.joinable() ) w.join();
  }
  ThreadPool( const ThreadPool & ) = delete;
  ThreadPool & operator=( const ThreadPool & ) = delete;

  static inline void cpu_relax()
  {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  }

  void pin_self( int participant ) const
  {
#if defined(__linux__)
    int cpu;
    if ( (int) cpumap_.size() == P_ && cpumap_[participant] >= 0 )
      cpu = cpumap_[participant];                       // CCX-aware map
    else
      // Legacy: fill physical cores first (even logical cpus on SMT-2).
      cpu = ( participant < 16 ) ? participant * 2 : ( participant - 16 ) * 2 + 1;
    cpu_set_t set; CPU_ZERO( &set ); CPU_SET( cpu, &set );
    pthread_setaffinity_np( pthread_self(), sizeof(set), &set );
#endif
  }

  std::size_t part_hi( int t ) const { return n_ * ( t + 1 ) / P_; }
  std::size_t part_lo( int t ) const { return n_ *   t       / P_; }

  void worker( int t )
  {
    pin_self( t );
    std::uint64_t my = 0;
    for ( ;; )
    {
      while ( gen_.load( std::memory_order_acquire ) == my )       // spin: read-only
      {
        if ( stop_.load( std::memory_order_acquire ) ) return;
        cpu_relax();
      }
      my = gen_.load( std::memory_order_acquire );
      if ( stop_.load( std::memory_order_acquire ) ) return;
      (*fn_)( part_lo( t ), part_hi( t ) );
      arrived_[t].v.store( my, std::memory_order_release );        // own cache line
    }
  }

  // each worker's completion flag on its own cache line (no false sharing)
  struct alignas(64) PadFlag { std::atomic<std::uint64_t> v{0}; };

  int P_;
  std::vector<int> cpumap_;
  std::vector<std::thread> workers_;
  std::atomic<std::uint64_t> gen_{0};
  PadFlag arrived_[64];
  std::atomic<bool> stop_{false};
  std::size_t n_{0};
  const std::function<void(std::size_t,std::size_t)> * fn_{nullptr};
};

} // namespace Device
} // namespace Xyce

#endif // Xyce_N_DEV_ThreadPool_h
