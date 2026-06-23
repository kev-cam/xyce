//-------------------------------------------------------------------------
// N_PDS_ShmComm.h
//
// Purpose : Shared-memory short-circuit of Epetra_MpiComm for on-node ranks.
//
//   ShmComm overrides the Krylov hot-path collective, SumAll(double*,...)
//   (Epetra dot-products / 2-norms reduce through Epetra_MultiVector::
//   Comm_->SumAll every iteration), and performs the reduction through a POSIX
//   shared-memory segment + a sense-reversing barrier instead of MPI_Allreduce,
//   bypassing Open MPI's PML/BTL machinery. Everything not overridden falls
//   through to Epetra_MpiComm (real MPI), and large reductions also fall back,
//   so it is always correct.
//
//   Crucially, Epetra_BlockMapData clones the comm (Comm.Clone()) into every
//   Map, and the base Epetra_MpiComm::Clone() returns a base Epetra_MpiComm --
//   which would drop the override. So Clone() is overridden to return a
//   ShmComm. The shm segment is a process-global singleton set up once (and
//   collectively, at the first -- N_PDS -- comm construction); all clones share
//   it. Because Xyce is SPMD (every rank executes the same SumAll on the same
//   logical comm), one global barrier is safe across all comm instances.
//
//   Prototype "(A)" short-circuit: still N processes, but the collective talks
//   straight to shared memory. Step (B) -- threads-as-ranks in one process --
//   removes even the shared-segment copy.
//-------------------------------------------------------------------------

#ifndef Xyce_N_PDS_ShmComm_h
#define Xyce_N_PDS_ShmComm_h

#ifdef Xyce_PARALLEL_MPI

#include <mpi.h>
#include <Epetra_MpiComm.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>

namespace Xyce {
namespace Parallel {

namespace ShmDetail {

enum { SHM_MAXCOUNT = 256 };   // reductions wider than this fall back to MPI

struct ShmBar { std::atomic<int> count; std::atomic<int> sense; };

struct ShmState {
  void *   seg;
  ShmBar * bar;
  double * data;
  int      rank;
  int      nproc;
  int      sense;     // process-local barrier sense (SPMD-consistent across ranks)
  long     calls;     // SumAll short-circuited (diagnostic)
  bool     ok;
  bool     tried;
  bool     off;       // XYCE_SHMCOMM_OFF=1 -> fall back to MPI (A/B timing)
  char     name[48];
};

inline ShmState & state()
{
  static ShmState s = { 0, 0, 0, 0, 0, 0, 0, false, false, false, {0} };
  return s;
}

inline void atexit_report()
{
  ShmState & s = state();
  if ( s.ok && getenv( "XYCE_SHMCOMM_DEBUG" ) )
    std::fprintf( stderr, "[ShmComm] rank %d: %ld SumAll short-circuited\n",
                  s.rank, s.calls );
  if ( s.seg ) { munmap( s.seg, sizeof(ShmBar) + (long) s.nproc * SHM_MAXCOUNT * sizeof(double) ); s.seg = 0; }
  if ( s.ok && s.rank == 0 && s.name[0] ) shm_unlink( s.name );
}

// One-time, collective setup of the shared segment (called from the first --
// N_PDS -- comm construction, which is collective across all ranks).
inline void setup( MPI_Comm comm )
{
  ShmState & s = state();
  if ( s.tried ) return;
  s.tried = true;
  MPI_Comm_rank( comm, &s.rank );
  MPI_Comm_size( comm, &s.nproc );
  if ( s.nproc <= 1 ) return;

  const long bytes = sizeof(ShmBar) + (long) s.nproc * SHM_MAXCOUNT * sizeof(double);
  char name[48];
  if ( s.rank == 0 )
    std::snprintf( name, sizeof name, "/xyce_shmcomm_%d", (int) getpid() );
  MPI_Bcast( name, sizeof name, MPI_CHAR, 0, comm );
  std::strncpy( s.name, name, sizeof s.name );

  int fd = -1;
  if ( s.rank == 0 ) {
    fd = shm_open( name, O_CREAT | O_RDWR | O_TRUNC, 0600 );
    if ( fd >= 0 && ftruncate( fd, bytes ) != 0 ) fd = -1;
  }
  MPI_Barrier( comm );
  if ( s.rank != 0 ) fd = shm_open( name, O_RDWR, 0600 );
  if ( fd < 0 ) return;

  s.seg = mmap( 0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
  close( fd );
  if ( s.seg == MAP_FAILED ) { s.seg = 0; return; }
  s.bar  = reinterpret_cast<ShmBar*>( s.seg );                 // shm is zero-filled
  s.data = reinterpret_cast<double*>( (char*) s.seg + sizeof(ShmBar) );
  MPI_Barrier( comm );                                         // bar visible to all
  s.ok = true;
  s.off = ( getenv( "XYCE_SHMCOMM_OFF" ) != 0 );               // A/B toggle
  std::atexit( atexit_report );
}

inline void barrier()
{
  ShmState & s = state();
  int sense = ( s.sense ^= 1 );
  if ( s.bar->count.fetch_add( 1, std::memory_order_acq_rel ) + 1 == s.nproc ) {
    s.bar->count.store( 0, std::memory_order_relaxed );
    s.bar->sense.store( sense, std::memory_order_release );
  } else {
    while ( s.bar->sense.load( std::memory_order_acquire ) != sense )
      __builtin_ia32_pause();
  }
}

} // namespace ShmDetail


class ShmComm : public Epetra_MpiComm
{
public:
  ShmComm( MPI_Comm comm ) : Epetra_MpiComm( comm )
  {
    ShmDetail::setup( comm );
    if ( getenv( "XYCE_SHMCOMM_DEBUG" ) )
      std::fprintf( stderr, "[ShmComm] ctor rank %d/%d ok=%d\n",
                    ShmDetail::state().rank, ShmDetail::state().nproc,
                    (int) ShmDetail::state().ok );
  }

  // Copy ctor: the singleton is already up; just chain Epetra_MpiComm's copy.
  ShmComm( const ShmComm & other ) : Epetra_MpiComm( other ) {}

  virtual ~ShmComm() {}

  // Epetra_BlockMapData clones the comm into every Map; keep the override alive
  // on the clone so the MultiVector reductions stay short-circuited.
  Epetra_Comm * Clone() const { return new ShmComm( *this ); }

  // The hot path: shared-memory all-reduce in place of MPI_Allreduce.
  int SumAll( double * partial, double * global, int count ) const
  {
    ShmDetail::ShmState & s = ShmDetail::state();
    if ( !s.ok || s.off || count > ShmDetail::SHM_MAXCOUNT )
      return Epetra_MpiComm::SumAll( partial, global, count );

    ++s.calls;
    double * slot = s.data + (long) s.rank * ShmDetail::SHM_MAXCOUNT;
    std::memcpy( slot, partial, count * sizeof(double) );
    ShmDetail::barrier();                         // all partials written
    for ( int i = 0; i < count; ++i ) {
      double sum = 0.0;
      for ( int r = 0; r < s.nproc; ++r )
        sum += s.data[ (long) r * ShmDetail::SHM_MAXCOUNT + i ];
      global[i] = sum;
    }
    ShmDetail::barrier();                          // all read before reuse
    return 0;
  }
};

} // namespace Parallel
} // namespace Xyce

#endif // Xyce_PARALLEL_MPI
#endif // Xyce_N_PDS_ShmComm_h
