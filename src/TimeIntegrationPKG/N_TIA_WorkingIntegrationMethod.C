//-------------------------------------------------------------------------
//   Copyright 2002-2025 National Technology & Engineering Solutions of
//   Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
//   NTESS, the U.S. Government retains certain rights in this software.
//
//   This file is part of the Xyce(TM) Parallel Electrical Simulator.
//
//   Xyce(TM) is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   Xyce(TM) is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with Xyce(TM).
//   If not, see <http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// Purpose       : This file contains the functions which define the
//		             time integration methods classes.
//
// Special Notes :
//
// Creator       : Buddy Watts
//
// Creation Date : 6/1/00
//
//
//
//
//-----------------------------------------------------------------------------

#include <Xyce_config.h>

#include <N_ERH_ErrorMgr.h>
#include <N_TIA_DataStore.h>
#include <N_TIA_StepErrorControl.h>
#include <N_TIA_TIAParams.h>
#include <N_TIA_TimeIntegrationMethods.h>
#include <N_TIA_WorkingIntegrationMethod.h>
#include <N_UTL_FeatureTest.h>
#include <N_LAS_Vector.h>
#include <N_LAS_EpetraHelpers.h>
#include <Epetra_MultiVector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace Xyce {
namespace TimeIntg {

namespace {

typedef std::map<int, std::pair<const char *, Factory> > Registry;

Registry &
getRegistry() 
{
  static Registry s_registry;

  return s_registry;
}

} // namespace <unnamed>

void
registerFactory(int type, const char *name, Factory factory)
{
  std::pair<Registry::iterator, bool> result = getRegistry().insert(Registry::value_type(type, std::pair<const char *, Factory>(name, factory)));
  if (!result.second && name != (*result.first).second.first)
    Report::DevelFatal0() << "Time integration factory " << type << " named " << name << " already registered with name " << (*result.first).second.first;
}

TimeIntegrationMethod *
createTimeIntegrationMethod(
  int                   type,
  const TIAParams &     tia_params,
  StepErrorControl &    step_error_control,
  DataStore &           data_store)
{
  Registry::const_iterator it = getRegistry().find(type);
  if (it == getRegistry().end())
    return 0;

  return (*(*it).second.second)(tia_params, step_error_control, data_store);
}

const char *
getTimeIntegrationName(int type) 
{
  Registry::const_iterator it = getRegistry().find(type);
  if (it == getRegistry().end())
    return "<none>";

  return (*it).second.first;
}

//-----------------------------------------------------------------------------
// Function      : WorkingIntegrationMethod::WorkingIntegrationMethod
// Purpose       : constructor
// Special Notes :
// Scope         : public
// Creator       : Buddy Watts, SNL
// Creation Date : 6/01/00
//-----------------------------------------------------------------------------
WorkingIntegrationMethod::WorkingIntegrationMethod(Stats::Stat parent_stat )
  : timeIntegrationMethod_(0),
    jacLimitFlag(false),
    jacLimit(1.0e+17),
    timeIntegratorStat_("Time integrator", parent_stat),
    predictorStat_("Predictor", timeIntegratorStat_),
    completeStepStat_("Successful Step", timeIntegratorStat_),
    rejectStepStat_("Failed Step", timeIntegratorStat_),
    updateCoefStat_("Update Coef", timeIntegratorStat_),
    residualStat_("Load Residual", timeIntegratorStat_),
    jacobianStat_("Load Jacobian", timeIntegratorStat_),
    initializeStat_("Initialize",  timeIntegratorStat_),
    updateLeadStat_("Lead Currents",  timeIntegratorStat_)
{}

//-----------------------------------------------------------------------------
// Function      : WorkingIntegrationMethod::~WorkingIntegrationMethod()
// Purpose       : destructor
// Special Notes :
// Scope         : public
// Creator       : Buddy Watts, SNL
// Creation Date : 6/01/00
//-----------------------------------------------------------------------------
WorkingIntegrationMethod::~WorkingIntegrationMethod()
{
  delete timeIntegrationMethod_;
}

//-----------------------------------------------------------------------------
// Function      : WorkingIntegrationMethod::createTimeIntegMethod
// Purpose       : Creates the time integration method class --- assigning a
//                 pointer and the Leading Coefficient value of the method.
// Special Notes :
// Scope         : public
// Creator       : Buddy Watts, SNL
// Creation Date : 6/01/00
//-----------------------------------------------------------------------------
void WorkingIntegrationMethod::createTimeIntegMethod(
  int                   type,
  const TIAParams &     tia_params,
  StepErrorControl &    step_error_control,
  DataStore &           data_store)
{
  jacLimitFlag = tia_params.jacLimitFlag;
  jacLimit = tia_params.jacLimit;

  oracleSec_ = &step_error_control;
  oracleDs_ = &data_store;

  delete timeIntegrationMethod_;
  timeIntegrationMethod_ = createTimeIntegrationMethod(type, tia_params, step_error_control, data_store);

  if (!timeIntegrationMethod_)
    Report::DevelFatal0().in("WorkingIntegrationMethod::createTimeIntegMethod") << "Invalid integration method " << type << " specified";

  if (VERBOSE_TIME)
    Xyce::lout() << "  Integration method = " << timeIntegrationMethod_->getName() << std::endl;
}

bool WorkingIntegrationMethod::isTimeIntegrationMethodCreated()
{
  return timeIntegrationMethod_ != 0;
}

double WorkingIntegrationMethod::partialTimeDeriv() const
{
  double pdt = timeIntegrationMethod_->partialTimeDeriv();
  if (jacLimitFlag && pdt > jacLimit)
    pdt = jacLimit;

  return pdt;
}

//-----------------------------------------------------------------------------
// Behavioral-oracle headroom hooks (dkc's observational-model direction).
//
// XYCE_ORACLE_RECORD=<file>  appends (time, accepted solution) after every
//                            successful step — the "rotate to trainer" feed.
// XYCE_ORACLE_REPLAY=<file>  interpolates a recorded trajectory at each new
//                            step time and overlays it as Newton's initial
//                            guess.  Only ds.nextSolutionPtr is touched:
//                            ds.xn0Ptr keeps the polynomial predictor, so
//                            LTE error estimation and step control are
//                            untouched, and a wrong oracle costs iterations,
//                            never correctness.
// Replaying a run against its own recording measures the perfect-oracle
// ceiling on Newton iterations — the number every real (cycle-behind,
// observation-trained) model is judged against.
//
// XYCE_ORACLE_SHM=<path>     memory-resident ring (mmap a tmpfs path, e.g.
//                            /dev/shm/xyce_live) shared with a live trainer.
//                            Producer cost per accepted step: one in-cache
//                            row copy + a release-store of the sequence
//                            counter — no syscalls, no stdio, no disk. The
//                            trainer samples rows in place at its own pace;
//                            ring overrun just means it trains on a sampled
//                            window (advisory contract: never blocks, never
//                            is waited for). XYCE_ORACLE_SHM_ROWS sizes the
//                            ring (default 2048 rows).
//-----------------------------------------------------------------------------
namespace {

struct Oracle
{
  int mode = 0;                       // 0 off, 1 record, 2 replay
  long n = 0;
  FILE *f = 0;
  unsigned long recrows = 0;          // rows written; row 0 (DC op) is flushed
  double *shmbase = 0;                // XYCE_ORACLE_SHM ring (header page +
  unsigned long shmrows = 0;          //   R rows of (t, solution)), tmpfs
  double *stobase = 0;                // XYCE_ORACLE_SHM_STORE ring: (t, sto[m])
  unsigned long storows = 0;          //   regime keys etc.; created lazily at
                                      //   the first accepted step (m known then)
  FILE *fsto = 0;                     // XYCE_ORACLE_RECORD_STORE: store-vector
  long m = -1;                        // channel (regime keys etc.); header
                                      // written on first accepted step
  std::vector<double> times;
  std::vector<double> vals;           // times.size() rows x n, row-major
  std::vector<double> ders;           // matching xdot rows (Hermite interp)
  bool inited = false;
  unsigned long overlays = 0;
  double clamp = 0;                   // XYCE_ORACLE_CLAMP: per-node gate
  // audit (XYCE_ORACLE_AUDIT=1): distance of each guess to the accepted
  // solution — answers whether the oracle start is actually closer
  bool audit = false;
  std::vector<double> polyGuess, orcGuess;
  double polyL2 = 0, orcL2 = 0, polyLinf = 0, orcLinf = 0;
  unsigned long audits = 0;
};
Oracle s_oracle;

void oracle_report()
{
  if (s_oracle.mode == 2)
    std::fprintf(stderr, "[oracle] replay overlays applied: %lu\n",
                 s_oracle.overlays);
  if (s_oracle.audit && s_oracle.audits)
  {
    double n = (double) s_oracle.audits;
    std::fprintf(stderr, "[oracle] audit over %lu steps: mean L2 poly %.3e "
                 "oracle %.3e | worst Linf poly %.3e oracle %.3e\n",
                 s_oracle.audits, s_oracle.polyL2 / n, s_oracle.orcL2 / n,
                 s_oracle.polyLinf, s_oracle.orcLinf);
  }
  if (s_oracle.f) std::fclose(s_oracle.f);
  if (s_oracle.fsto) std::fclose(s_oracle.fsto);
  s_oracle.f = 0;
  s_oracle.fsto = 0;
}

double * oracle_vec(Xyce::Linear::Vector & v, int & len)
{
  Epetra_MultiVector & m =
    dynamic_cast<Xyce::Linear::EpetraVectorAccess &>(v).epetraObj();
  len = m.MyLength();
  return m.Pointers()[0];
}

#if !defined(_WIN32)
double * oracle_make_ring(const char * path, long n, long rows)
{
  size_t bytes = 4096 + (size_t) rows * (1 + n) * sizeof(double);
  int fd = ::open(path, O_CREAT | O_RDWR | O_TRUNC, 0644);
  if (fd < 0) return 0;
  void * p = MAP_FAILED;
  if (::ftruncate(fd, (off_t) bytes) == 0)
    p = ::mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) return 0;
  long * h = (long *) p;
  h[1] = n;
  h[2] = rows;
  h[3] = 0;                                     // seq
  // publish the magic last so a waiting trainer never sees a
  // half-initialized header
  __atomic_store_n(&h[0], 0x584C495645L, __ATOMIC_RELEASE);
  return (double *) p;
}

unsigned long oracle_ring_rows()
{
  long rows = 2048;
  if (const char * r = std::getenv("XYCE_ORACLE_SHM_ROWS"))
    rows = std::atol(r) > 16 ? std::atol(r) : 2048;
  return (unsigned long) rows;
}

void oracle_ring_push(double * base, unsigned long rows, double t,
                      const double * x, int len)
{
  long * h = (long *) base;
  unsigned long seq = (unsigned long) h[3];
  double * slot = base + 512 + (seq % rows) * (size_t) (1 + len);
  slot[0] = t;
  std::memcpy(slot + 1, x, (size_t) len * sizeof(double));
  __atomic_store_n(&h[3], (long) (seq + 1), __ATOMIC_RELEASE);
}
#endif

void oracle_init(int n)
{
  s_oracle.inited = true;
  const char * rec = std::getenv("XYCE_ORACLE_RECORD");
  const char * rep = std::getenv("XYCE_ORACLE_REPLAY");
#if !defined(_WIN32)
  if (const char * sp = std::getenv("XYCE_ORACLE_SHM"))
  {
    unsigned long rows = oracle_ring_rows();
    s_oracle.shmbase = oracle_make_ring(sp, n, (long) rows);
    if (s_oracle.shmbase)
      s_oracle.shmrows = rows;
  }
#endif
  if (const char * rs = std::getenv("XYCE_ORACLE_RECORD_STORE"))
  {
    s_oracle.fsto = std::fopen(rs, "wb");
    if (s_oracle.fsto) std::atexit(oracle_report);
  }
  if (rec)
  {
    s_oracle.f = std::fopen(rec, "wb");
    if (s_oracle.f)
    {
      long nn = n;
      std::fwrite(&nn, sizeof nn, 1, s_oracle.f);
      s_oracle.mode = 1;
      s_oracle.n = n;
      std::atexit(oracle_report);
    }
  }
  else if (rep)
  {
    FILE * f = std::fopen(rep, "rb");
    if (!f) return;
    long nn = 0;
    if (std::fread(&nn, sizeof nn, 1, f) != 1 || nn != n)
    {
      std::fprintf(stderr, "[oracle] replay size mismatch (%ld vs %d)\n", nn, n);
      std::fclose(f);
      return;
    }
    double t;
    std::vector<double> row(n);
    while (std::fread(&t, sizeof t, 1, f) == 1 &&
           std::fread(row.data(), sizeof(double), n, f) == (size_t) n)
    {
      s_oracle.times.push_back(t);
      s_oracle.vals.insert(s_oracle.vals.end(), row.begin(), row.end());
    }
    // derivatives by non-uniform centered differences over the recorded
    // grid (the integrator keeps history, not xdot; the adaptive grid is
    // densest exactly where slopes are steep, so FD is accurate there)
    {
      size_t np = s_oracle.times.size();
      s_oracle.ders.assign(np * (size_t) n, 0.0);
      const std::vector<double> & T = s_oracle.times;
      for (size_t i = 0; i < np; ++i)
      {
        size_t im = i ? i - 1 : i, ip = (i + 1 < np) ? i + 1 : i;
        const double *xm = &s_oracle.vals[im * n], *xp = &s_oracle.vals[ip * n];
        const double *xc = &s_oracle.vals[i * n];
        double h0 = T[i] - T[im], h1 = T[ip] - T[i];
        double *d = &s_oracle.ders[i * n];
        for (long k = 0; k < n; ++k)
        {
          if (h0 > 0 && h1 > 0)
            d[k] = (h0 * (xp[k] - xc[k]) / h1 + h1 * (xc[k] - xm[k]) / h0) /
                   (h0 + h1);
          else if (h1 > 0) d[k] = (xp[k] - xc[k]) / h1;
          else if (h0 > 0) d[k] = (xc[k] - xm[k]) / h0;
        }
      }
    }
    std::fclose(f);
    s_oracle.n = n;
    s_oracle.mode = 2;
    s_oracle.audit = std::getenv("XYCE_ORACLE_AUDIT") != 0;
    if (const char * c = std::getenv("XYCE_ORACLE_CLAMP"))
      s_oracle.clamp = std::atof(c);
    std::atexit(oracle_report);
    std::fprintf(stderr, "[oracle] replay loaded: %zu points x %ld\n",
                 s_oracle.times.size(), s_oracle.n);
  }
}

} // namespace

void WorkingIntegrationMethod::obtainPredictor()
{
//  Stats::TimeBlock x(predictorStat_);

  timeIntegrationMethod_->obtainPredictor();

  if (oracleDs_ && oracleSec_ && oracleDs_->nextSolutionPtr)
  {
    if (!s_oracle.inited)
    {
      int len = 0;
      oracle_vec(*oracleDs_->nextSolutionPtr, len);
      oracle_init(len);
    }
    if (s_oracle.mode == 2 && !s_oracle.times.empty())
    {
      const std::vector<double> & T = s_oracle.times;
      double t = oracleSec_->nextTime;
      if (t >= T.front() && t <= T.back())
      {
        if (s_oracle.audit)
        {
          int len = 0;
          double * p = oracle_vec(*oracleDs_->nextSolutionPtr, len);
          s_oracle.polyGuess.assign(p, p + len);
        }
        size_t hi = std::lower_bound(T.begin(), T.end(), t) - T.begin();
        if (hi == 0) hi = 1;
        if (hi >= T.size()) hi = T.size() - 1;
        size_t lo = hi - 1;
        double dt = T[hi] - T[lo];
        double w = (dt > 0) ? (t - T[lo]) / dt : 0.0;
        int len = 0;
        double * x = oracle_vec(*oracleDs_->nextSolutionPtr, len);
        const double * a  = &s_oracle.vals[lo * s_oracle.n];
        const double * b  = &s_oracle.vals[hi * s_oracle.n];
        const double * da = &s_oracle.ders[lo * s_oracle.n];
        const double * db = &s_oracle.ders[hi * s_oracle.n];
        // cubic Hermite: linear interp of samples is cruder than the
        // integrator's own 2nd-order representation at switching edges and
        // hands Newton nonphysical mid-swing states; with the recorded
        // derivatives the interpolant's error is far below LTE
        double h00 = (1 + 2 * w) * (1 - w) * (1 - w);
        double h10 = w * (1 - w) * (1 - w);
        double h01 = w * w * (3 - 2 * w);
        double h11 = w * w * (w - 1);
        long m = (len < s_oracle.n) ? len : s_oracle.n;
        // Per-node safety gate: the audit showed the oracle is 2.1x closer
        // in mean L2 but its rare mid-swing errors (nonphysical crowbar
        // states on drifted edges) cost more than its accuracy earns —
        // Newton and the limiters answer to the worst node, not the mean.
        // Where oracle and polynomial disagree beyond the clamp, keep the
        // physically-benign polynomial value.
        if (s_oracle.clamp > 0)
        {
          for (long i = 0; i < m; ++i)
          {
            double o = h00 * a[i] + h10 * dt * da[i] + h01 * b[i] +
                       h11 * dt * db[i];
            double d = o - x[i];
            if (d > -s_oracle.clamp && d < s_oracle.clamp)
              x[i] = o;
          }
        }
        else
        {
          for (long i = 0; i < m; ++i)
            x[i] = h00 * a[i] + h10 * dt * da[i] + h01 * b[i] + h11 * dt * db[i];
        }
        ++s_oracle.overlays;
        if (s_oracle.audit)
          s_oracle.orcGuess.assign(x, x + len);
      }
    }
  }
}

void WorkingIntegrationMethod::obtainPredictorDeriv()
{
//  Stats::TimeBlock x(predictorStat_);
  timeIntegrationMethod_->obtainPredictorDeriv();
}

void WorkingIntegrationMethod::obtainCorrectorDeriv()
{
  timeIntegrationMethod_->obtainCorrectorDeriv();
}

int WorkingIntegrationMethod::getOrder() const
{
  return timeIntegrationMethod_->getOrder();
}

int WorkingIntegrationMethod::getUsedOrder() const
{
  return timeIntegrationMethod_->getUsedOrder();
}

int WorkingIntegrationMethod::getMethod() const
{
  return timeIntegrationMethod_->getMethod();
}

int WorkingIntegrationMethod::getNumberOfSteps() const
{
  return timeIntegrationMethod_->getNumberOfSteps();
}

int WorkingIntegrationMethod::getNscsco() const
{
  return timeIntegrationMethod_->getNscsco();
}

void WorkingIntegrationMethod::getInitialQnorm(TwoLevelError & tle) const
{
  return timeIntegrationMethod_->getInitialQnorm (tle);
}

void WorkingIntegrationMethod::getTwoLevelError(TwoLevelError & tle) const
{
  return timeIntegrationMethod_->getTwoLevelError(tle);
}

void WorkingIntegrationMethod::setTwoLevelTimeInfo()
{
  return timeIntegrationMethod_->setTwoLevelTimeInfo();
}

void WorkingIntegrationMethod::updateCoeffs()
{
 
//  Stats::TimeBlock x(updateCoefStat_);
  return timeIntegrationMethod_->updateCoeffs();
}

void WorkingIntegrationMethod::updateAdjointCoeffs()
{
  return timeIntegrationMethod_->updateAdjointCoeffs();
}

void WorkingIntegrationMethod::rejectStepForHabanero ()
{
  return timeIntegrationMethod_->rejectStepForHabanero();
}

void WorkingIntegrationMethod::initialize(const TIAParams &tia_params)
{
//  Stats::TimeBlock x( initializeStat_ );
  return timeIntegrationMethod_->initialize(tia_params);
}


void WorkingIntegrationMethod::initializeAdjoint (int index)
{
  return timeIntegrationMethod_->initializeAdjoint(index);
}

void WorkingIntegrationMethod::completeStep(const TIAParams &tia_params)
{

//  Stats::TimeBlock x(completeStepStat_);

  timeIntegrationMethod_->completeStep(tia_params);

  if (s_oracle.mode == 1 && oracleDs_ && oracleSec_ && oracleDs_->currSolutionPtr)
  {
    int len = 0;
    double * x = oracle_vec(*oracleDs_->currSolutionPtr, len);
    double t = oracleSec_->currentTime;
    std::fwrite(&t, sizeof t, 1, s_oracle.f);
    std::fwrite(x, sizeof(double), len, s_oracle.f);
    // Keep the tail visible to a concurrent trainer: row 0 is the DC op
    // (ensemble watcher ingests it mid-run), and every 64th row bounds a
    // live tuner's lag at 64 steps for ~µs of flush cost.
    if ((s_oracle.recrows++ & 63) == 0)
      std::fflush(s_oracle.f);
  }

#if !defined(_WIN32)
  if (s_oracle.shmbase && oracleDs_ && oracleSec_ && oracleDs_->currSolutionPtr)
  {
    int len = 0;
    double * x = oracle_vec(*oracleDs_->currSolutionPtr, len);
    oracle_ring_push(s_oracle.shmbase, s_oracle.shmrows,
                     oracleSec_->currentTime, x, len);
  }

  // Regime/store stream for the per-regime trainer. Created lazily on the
  // first accepted step because the store width is unknown at init.
  if (oracleDs_ && oracleSec_ && oracleDs_->currStorePtr)
  {
    if (!s_oracle.stobase)
    {
      if (const char * sp = std::getenv("XYCE_ORACLE_SHM_STORE"))
      {
        static bool tried = false;
        if (!tried)
        {
          tried = true;
          int m = 0;
          oracle_vec(*oracleDs_->currStorePtr, m);
          if (m > 0)
          {
            unsigned long rows = oracle_ring_rows();
            s_oracle.stobase = oracle_make_ring(sp, m, (long) rows);
            if (s_oracle.stobase)
              s_oracle.storows = rows;
          }
        }
      }
    }
    if (s_oracle.stobase)
    {
      int m = 0;
      double * sto = oracle_vec(*oracleDs_->currStorePtr, m);
      oracle_ring_push(s_oracle.stobase, s_oracle.storows,
                       oracleSec_->currentTime, sto, m);
    }
  }
#endif

  if (s_oracle.fsto && oracleDs_ && oracleSec_ && oracleDs_->currStorePtr)
  {
    int m = 0;
    double * sto = oracle_vec(*oracleDs_->currStorePtr, m);
    if (s_oracle.m < 0)
    {
      s_oracle.m = m;
      long mm = m;
      std::fwrite(&mm, sizeof mm, 1, s_oracle.fsto);
    }
    double t = oracleSec_->currentTime;
    std::fwrite(&t, sizeof t, 1, s_oracle.fsto);
    std::fwrite(sto, sizeof(double), m, s_oracle.fsto);
  }

  if (s_oracle.audit && oracleDs_ && oracleDs_->currSolutionPtr &&
      !s_oracle.polyGuess.empty() && !s_oracle.orcGuess.empty())
  {
    int len = 0;
    double * x = oracle_vec(*oracleDs_->currSolutionPtr, len);
    double pl2 = 0, ol2 = 0;
    size_t m = s_oracle.polyGuess.size() < (size_t) len
                 ? s_oracle.polyGuess.size() : (size_t) len;
    for (size_t i = 0; i < m; ++i)
    {
      double dp = s_oracle.polyGuess[i] - x[i];
      double dor = s_oracle.orcGuess[i] - x[i];
      pl2 += dp * dp;
      ol2 += dor * dor;
      double ap = dp < 0 ? -dp : dp, ao = dor < 0 ? -dor : dor;
      if (ap > s_oracle.polyLinf) s_oracle.polyLinf = ap;
      if (ao > s_oracle.orcLinf) s_oracle.orcLinf = ao;
    }
    s_oracle.polyL2 += std::sqrt(pl2);
    s_oracle.orcL2 += std::sqrt(ol2);
    ++s_oracle.audits;
    s_oracle.polyGuess.clear();
    s_oracle.orcGuess.clear();
  }
}


void WorkingIntegrationMethod::completeAdjointStep(const TIAParams &tia_params)
{

//  Stats::TimeBlock x(completeStepStat_);

  return timeIntegrationMethod_->completeAdjointStep(tia_params);
}


void WorkingIntegrationMethod::rejectStep(const TIAParams &tia_params)
{
//  Stats::TimeBlock x(rejectStepStat_);
  return timeIntegrationMethod_->rejectStep(tia_params);
}

double WorkingIntegrationMethod::computeErrorEstimate() const
{

//  Stats::Stat ErrorStat_("error estimation", timeIntegratorStat_);
//  Stats::TimeBlock x(ErrorStat_);
  return timeIntegrationMethod_->computeErrorEstimate();
}

void WorkingIntegrationMethod::updateStateDeriv ()
{
  return timeIntegrationMethod_->updateStateDeriv ();
}

void WorkingIntegrationMethod::updateLeadCurrent ()
{

//  Stats::TimeBlock x( updateLeadStat_);
  return timeIntegrationMethod_->updateLeadCurrentVec ();
}

void WorkingIntegrationMethod::obtainResidual()
{
//  Stats::TimeBlock x(residualStat_);
  return timeIntegrationMethod_->obtainResidual();
}

void WorkingIntegrationMethod::obtainSensitivityResiduals()
{
  return timeIntegrationMethod_->obtainSensitivityResiduals();
}

void WorkingIntegrationMethod::updateSensitivityHistoryAdjoint()
{
  return timeIntegrationMethod_->updateSensitivityHistoryAdjoint();
}

void WorkingIntegrationMethod::updateSensitivityHistoryAdjoint2()
{
  return timeIntegrationMethod_->updateSensitivityHistoryAdjoint2();
}

void WorkingIntegrationMethod::obtainFunctionDerivativesForTranAdjoint()
{
  return timeIntegrationMethod_->obtainFunctionDerivativesForTranAdjoint();
}

void WorkingIntegrationMethod::obtainSparseFunctionDerivativesForTranAdjoint()
{
  return timeIntegrationMethod_->obtainSparseFunctionDerivativesForTranAdjoint();
}

void WorkingIntegrationMethod::obtainAdjointSensitivityResidual()
{
  return timeIntegrationMethod_->obtainAdjointSensitivityResidual();
}


void WorkingIntegrationMethod::obtainJacobian()
{

//  Stats::TimeBlock x(jacobianStat_);
  return timeIntegrationMethod_->obtainJacobian();
}

void WorkingIntegrationMethod::applyJacobian(const Linear::Vector& input, Linear::Vector& result)
{
//  Stats::TimeBlock x(jacobianStat_);
  return timeIntegrationMethod_->applyJacobian(input, result);
}

bool WorkingIntegrationMethod::printMPDEOutputSolution(
  Analysis::OutputMgrAdapter &  outputManagerAdapter,
  const double                  time,
  Linear::Vector *              solnVecPtr,
  const std::vector<double> &   fastTimes )
{
  return timeIntegrationMethod_->printMPDEOutputSolution(
    outputManagerAdapter, time, solnVecPtr, fastTimes );
}

bool WorkingIntegrationMethod::printWaMPDEOutputSolution(
  Analysis::OutputMgrAdapter &  outputManagerAdapter,
  const double                  time,
  Linear::Vector *              solnVecPtr,
  const std::vector<double> &   fastTimes,
  const int                     phiGID )
{
  return timeIntegrationMethod_->printWaMPDEOutputSolution(
    outputManagerAdapter, time, solnVecPtr, fastTimes, phiGID );
}

bool WorkingIntegrationMethod::printOutputSolution(
  Analysis::OutputMgrAdapter &  outputManagerAdapter,
  const TIAParams &             tia_params,
  const double                  time,
  Linear::Vector *              solnVecPtr,
  const bool                    doNotInterpolate,
  const std::vector<double> &   outputInterpolationTimes,
  bool                          skipPrintLineOutput )
{
//  Stats::TimeBlock x( othersStat_);
  return timeIntegrationMethod_->printOutputSolution(
    outputManagerAdapter, tia_params, time, solnVecPtr, doNotInterpolate, outputInterpolationTimes, skipPrintLineOutput) ;
}

bool WorkingIntegrationMethod::saveOutputSolution(
  Parallel::Machine                     comm,
  IO::InitialConditionsManager &        initial_conditions_manager,
  const NodeNameMap &                   node_name_map,
  const TIAParams &                     tia_params,
  Linear::Vector *                      solnVecPtr,
  const double                          saveTime,
  const bool                            doNotInterpolate)
{
  return timeIntegrationMethod_->saveOutputSolution(comm, initial_conditions_manager, node_name_map, tia_params, solnVecPtr, saveTime, doNotInterpolate);
}

void WorkingIntegrationMethod::stepLinearCombo()
{
  if (timeIntegrationMethod_)
  {
    timeIntegrationMethod_->stepLinearCombo();
  }
}

bool WorkingIntegrationMethod::getSolnVarData( const int & gid, std::vector<double> & varData ) 
{ 
  if (timeIntegrationMethod_)
  {
    return timeIntegrationMethod_->getSolnVarData( gid, varData );
  }
  return false;
}

bool WorkingIntegrationMethod::setSolnVarData( const int & gid, const std::vector<double> & varData ) 
{ 
  if (timeIntegrationMethod_)
  {
    return timeIntegrationMethod_->setSolnVarData( gid, varData );
  }
  return false;
}

bool WorkingIntegrationMethod::getStateVarData( const int & gid, std::vector<double> & varData ) 
{ 
  if (timeIntegrationMethod_)
  {
    return timeIntegrationMethod_->getStateVarData( gid, varData );
  }
  return false;
}

bool WorkingIntegrationMethod::setStateVarData( const int & gid, const std::vector<double> & varData ) 
{ 
  if (timeIntegrationMethod_)
  {
    return timeIntegrationMethod_->setStateVarData( gid, varData );
  }
  return false;
}

bool WorkingIntegrationMethod::getStoreVarData( const int & gid, std::vector<double> & varData ) 
{ 
  if (timeIntegrationMethod_)
  {
    return timeIntegrationMethod_->getStoreVarData( gid, varData );
  }
  return false;
}

bool WorkingIntegrationMethod::setStoreVarData( const int & gid, const std::vector<double> & varData ) 
{ 
  if (timeIntegrationMethod_)
  {
    return timeIntegrationMethod_->setStoreVarData( gid, varData );
  }
  return false;
}

} // namespace TimeIntg
} // namespace Xyce
