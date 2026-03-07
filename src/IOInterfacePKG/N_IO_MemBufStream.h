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
// Purpose        : In-memory streambuf that replaces file I/O for .PRINT
//                  output.  When the buffer fills, fork() a child to dump
//                  the slice to disk (COW handles the snapshot) while the
//                  parent resets the write pointer and continues.
//
//                  A read interface lets the same process consume data
//                  directly from the live buffer (zero-copy pointer),
//                  falling back to slice files if the reader falls behind.
//
// Special Notes  : Activated by FILE=mem://<basepath> in .PRINT lines.
//                  Slice files written as <basepath>/slice_NNNN.bin
//
// Creator        : ltz project
//
// Creation Date  : 2026/03/07
//
//-----------------------------------------------------------------------------

#ifndef Xyce_N_IO_MemBufStream_h
#define Xyce_N_IO_MemBufStream_h

#include <streambuf>
#include <ostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace Xyce {
namespace IO {

//-----------------------------------------------------------------------------
// Class         : MemBufStreambuf
// Purpose       : std::streambuf backed by a malloc'd block with fork+COW
//                 slice dumping and a zero-copy read interface.
//
// Write path (std::ostream):
//   Xyce outputters write via *os_ as usual.  Data goes into buf_.
//   When pptr() crosses threshold_, fork().  Child has COW snapshot,
//   dumps to slice file, exits.  Parent resets setp() and continues.
//
// Read path (getReadPtr / advanceRead):
//   Reader gets a direct pointer into buf_ for unread data.
//   Between synchronous simulateUntil() steps the reader is always
//   caught up, so it reads from memory with no file I/O.
//   If the reader falls behind a flush, readFromSlice() recovers
//   data from the slice files the child wrote.
//-----------------------------------------------------------------------------
class MemBufStreambuf : public std::streambuf
{
public:
  MemBufStreambuf(const std::string &base_path,
                  size_t capacity = 16 * 1024 * 1024,
                  size_t threshold = 12 * 1024 * 1024)
    : base_path_(base_path),
      capacity_(capacity),
      threshold_(threshold),
      slice_num_(0),
      total_flushed_(0),
      read_offset_(0),
      read_slice_(0),
      finalized_(false)
  {
    buf_ = static_cast<char *>(std::malloc(capacity_));
    setp(buf_, buf_ + capacity_);
    mkdir(base_path_.c_str(), 0755);
  }

  virtual ~MemBufStreambuf()
  {
    if (!finalized_)
      finalize();
    while (waitpid(-1, nullptr, WNOHANG) > 0)
      ;
    std::free(buf_);
  }

  /// Write remaining buffer data to a slice file for archival, but
  /// keep buf_ readable via getReadPtr().  Does NOT advance slice_num_
  /// so the reader still sees the data as "live buffer", not as a
  /// slice file it needs to open.
  /// Safe to call multiple times.
  void finalize()
  {
    if (!finalized_)
    {
      size_t len = pptr() - buf_;
      if (len > 0)
      {
        // Archival write — slice file for replay/FUSE
        write_slice_file(buf_, len, slice_num_);
        // Don't increment slice_num_ or total_flushed_ —
        // the reader should consume this from buf_ directly.
      }
      write_index_with_pending(len);
      while (waitpid(-1, nullptr, WNOHANG) > 0)
        ;
      finalized_ = true;
    }
  }

  // ----- Read interface (called from C API, same process as writer) -----

  /// Get pointer to unread data in the live buffer.
  /// Returns number of bytes available.  *data points directly into buf_.
  /// Zero-copy: no allocation, no file I/O.
  ///
  /// If the reader fell behind a flush (read_slice_ < slice_num_),
  /// returns 0 and the caller should use readFromSlice() first.
  size_t getReadPtr(const char **data)
  {
    if (read_slice_ < slice_num_)
    {
      // Reader fell behind — data is in slice files
      *data = nullptr;
      return 0;
    }

    size_t written = static_cast<size_t>(pptr() - buf_);
    if (read_offset_ >= written)
    {
      *data = nullptr;
      return 0;
    }

    *data = buf_ + read_offset_;
    return written - read_offset_;
  }

  /// Advance the read pointer by n bytes.
  void advanceRead(size_t n)
  {
    read_offset_ += n;
  }

  /// Read data from slice files that the reader missed.
  /// Copies into dest, up to max_len bytes.  Returns bytes copied.
  /// Advances reader state through slice files until caught up with
  /// the live buffer.
  size_t readFromSlices(char *dest, size_t max_len)
  {
    size_t total_read = 0;

    while (read_slice_ < slice_num_ && total_read < max_len)
    {
      // Wait for the child to finish writing this slice
      // (brief spin — child is just doing a write+exit)
      std::string path = slicePath(read_slice_);
      int tries = 0;
      while (!fileExists(path) && tries < 100)
      {
        usleep(1000);  // 1ms
        ++tries;
      }

      FILE *fp = std::fopen(path.c_str(), "rb");
      if (!fp)
      {
        // Slice file missing — skip
        read_slice_++;
        read_offset_ = 0;
        continue;
      }

      // Seek past already-read portion of this slice
      if (read_offset_ > 0)
        std::fseek(fp, read_offset_, SEEK_SET);

      size_t chunk = max_len - total_read;
      size_t got = std::fread(dest + total_read, 1, chunk, fp);
      total_read += got;

      if (std::feof(fp) || got == 0)
      {
        // Done with this slice
        std::fclose(fp);
        read_slice_++;
        read_offset_ = 0;
      }
      else
      {
        read_offset_ += got;
        std::fclose(fp);
      }
    }

    return total_read;
  }

  /// Check if the reader has fallen behind (data is in slice files).
  bool readerBehind() const
  {
    return read_slice_ < slice_num_;
  }

  /// Reset reader to beginning of current live buffer.
  void resetReader()
  {
    read_slice_ = slice_num_;
    read_offset_ = 0;
  }

  /// Get the base path for slice files.
  const std::string &basePath() const { return base_path_; }

  /// Get current slice count.
  int sliceCount() const { return slice_num_; }

  /// Get total bytes flushed to slices.
  size_t totalFlushed() const { return total_flushed_; }

  /// Get bytes currently in the live buffer.
  size_t liveBytes() const { return static_cast<size_t>(pptr() - buf_); }

protected:
  virtual int overflow(int c) override
  {
    if (c == traits_type::eof())
      return traits_type::not_eof(c);

    if (pptr() >= buf_ + threshold_)
      flush_slice_fork();

    if (pptr() >= epptr())
      return traits_type::eof();

    *pptr() = static_cast<char>(c);
    pbump(1);
    return c;
  }

  virtual std::streamsize xsputn(const char *s, std::streamsize n) override
  {
    std::streamsize written = 0;

    while (n > 0)
    {
      if (pptr() >= buf_ + threshold_)
        flush_slice_fork();

      std::streamsize avail = epptr() - pptr();
      if (avail <= 0)
        break;

      std::streamsize chunk = (n < avail) ? n : avail;
      std::memcpy(pptr(), s, chunk);
      pbump(static_cast<int>(chunk));
      s += chunk;
      n -= chunk;
      written += chunk;
    }

    return written;
  }

  virtual int sync() override
  {
    return 0;
  }

private:
  /// Fork a child to dump the buffer.  Parent resets and continues.
  /// COW gives the child a frozen snapshot of the pages — no memcpy.
  void flush_slice_fork()
  {
    size_t len = pptr() - buf_;
    if (len == 0)
      return;

    while (waitpid(-1, nullptr, WNOHANG) > 0)
      ;

    int slice = slice_num_;
    size_t flush_len = len;

    pid_t pid = fork();
    if (pid == 0)
    {
      // Child: COW pages — buf_ is a frozen snapshot
      write_slice_file(buf_, flush_len, slice);
      _exit(0);
    }
    else if (pid > 0)
    {
      // Parent: advance bookkeeping, reset buffer.
      // Any reader that hasn't consumed this data will find it
      // in slice file `slice` once the child writes it.
      total_flushed_ += flush_len;
      slice_num_++;

      // If reader was pointing into this buffer, it now needs
      // to read from the slice file instead.
      // read_offset_ stays as-is; read_slice_ < slice_num_ triggers
      // the fallback path in getReadPtr().

      setp(buf_, buf_ + capacity_);

      // If reader was caught up, advance it to the new buffer.
      if (read_slice_ == slice && read_offset_ >= flush_len)
      {
        read_slice_ = slice_num_;
        read_offset_ = 0;
      }
    }
    else
    {
      flush_slice_sync();
    }
  }

  void flush_slice_sync()
  {
    size_t len = pptr() - buf_;
    if (len == 0)
      return;

    write_slice_file(buf_, len, slice_num_);
    total_flushed_ += len;
    slice_num_++;

    // Advance reader past the sync'd data
    if (read_slice_ < slice_num_)
    {
      read_slice_ = slice_num_;
      read_offset_ = 0;
    }

    setp(buf_, buf_ + capacity_);
  }

  void write_slice_file(const char *data, size_t len, int num)
  {
    std::string path = slicePath(num);
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (fp)
    {
      std::fwrite(data, 1, len, fp);
      std::fclose(fp);
    }
  }

  std::string slicePath(int num) const
  {
    char filename[512];
    std::snprintf(filename, sizeof(filename), "%s/slice_%04d.bin",
                  base_path_.c_str(), num);
    return std::string(filename);
  }

  static bool fileExists(const std::string &path)
  {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
  }

  void write_index_with_pending(size_t pending_len)
  {
    int total_slices = slice_num_ + (pending_len > 0 ? 1 : 0);
    size_t total = total_flushed_ + pending_len;

    std::string idx_path = base_path_ + "/index.txt";
    std::ofstream idx(idx_path.c_str(), std::ios_base::out);
    if (idx.good())
    {
      idx << "slices " << total_slices << "\n";
      idx << "total_bytes " << total << "\n";
      for (int i = 0; i < total_slices; ++i)
      {
        char fn[64];
        std::snprintf(fn, sizeof(fn), "slice_%04d.bin", i);
        idx << i << " " << fn << "\n";
      }
    }
  }

  char *buf_;
  size_t capacity_;
  size_t threshold_;
  int slice_num_;
  size_t total_flushed_;
  std::string base_path_;

  // Reader state
  size_t read_offset_;   // byte offset within current segment
  int read_slice_;       // which slice the reader is in (slice_num_ = live)
  bool finalized_;       // true after finalize() called
};

//-----------------------------------------------------------------------------
// Class         : MemBufStream
// Purpose       : std::ostream that owns a MemBufStreambuf.
//                 Drop-in replacement for std::ofstream in openFile().
//                 Also exposes the read interface for the C API.
//-----------------------------------------------------------------------------
class MemBufStream : public std::ostream
{
public:
  MemBufStream(const std::string &base_path,
               size_t capacity = 16 * 1024 * 1024,
               size_t threshold = 12 * 1024 * 1024)
    : std::ostream(&sbuf_),
      sbuf_(base_path, capacity, threshold)
  {}

  MemBufStreambuf &membuf() { return sbuf_; }

private:
  MemBufStreambuf sbuf_;
};

} // namespace IO
} // namespace Xyce

#endif // Xyce_N_IO_MemBufStream_h
