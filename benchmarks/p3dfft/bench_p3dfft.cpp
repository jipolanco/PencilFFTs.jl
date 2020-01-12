#include <p3dfft.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// File based on p3dfft.3/sample/C++/test3D_r2c.C and
// p3dfft/sample/C/driver_rand.c.

constexpr char USAGE[] = "./bench_p3dfft N [repetitions] [output_file]";

using Complex = std::complex<double>;

template <int N> using Dims = std::array<int, N>;

constexpr int DEFAULT_REPETITIONS = 100;

// Description of P3DFFT timers (see also P3DFFT_timers.md).
const std::array<std::string, 12> TIMERS_TEXT = {
    "MPI_Alltoallv (X -> Y)",                   // 0
    "MPI_Alltoallv (Y -> Z)",                   // 1
    "MPI_Alltoallv (Y <- Z)",                   // 2
    "MPI_Alltoallv (X <- Y)",                   // 3
    "FFT r2c X",                                // 4
    "pack + unpack data (X -> Y)",              // 5
    "FFT c2c Y",                                // 6
    "pack + unpack data (Y -> Z) + FFT c2c Z",  // 7
    "pack + unpack data (Y <- Z) + iFFT c2c Z", // 8
    "iFFT c2c Y",                               // 9
    "pack + unpack data (X <- Y)",              // 10
    "iFFT c2r X",                               // 11
};

void print_timers(const std::array<double, 12> &timers);

struct PencilSetup {
  const int conf;
  Dims<3> dims_global;
  Dims<3> start, end, size;
  PencilSetup(Dims<3> dims_global, int conf)
      : conf(conf), dims_global(dims_global) {
    Cp3dfft_get_dims(start.data(), end.data(), size.data(), conf);
  }
};

template <class T, size_t N>
auto print(T &io, const Dims<N> &dims) -> decltype(io) {
  if (N == 0) return io << "()";
  io << "(" << dims[0];
  for (size_t n = 1; n < N; ++n) io << ", " << dims[n];
  io << ")";
  return io;
}

// This is adapted from test3D_r2c.C
std::vector<double> &init_wave(std::vector<double> &u, const PencilSetup &pen) {
  Dims<3> local_dims; // local permuted dimensions (i.e. in storage order)
  Dims<3> glob_start; // starting position in global array

  for (int i = 0; i < 3; ++i) {
    local_dims[i] = pen.size[i];
    glob_start[i] = pen.start[i];
  }

  auto &gdims = pen.dims_global;  // non-permuted global dimensions

  std::vector<double> sinx(gdims[0]), siny(gdims[1]), sinz(gdims[2]);
  double twopi = 8 * std::atan(1.0);

  for (int i = 0; i < gdims[0]; ++i)
    sinx[i] = std::sin((i + glob_start[0]) * twopi / gdims[0]);
  for (int i = 0; i < gdims[1]; ++i)
    siny[i] = std::sin((i + glob_start[1]) * twopi / gdims[1]);
  for (int i = 0; i < gdims[2]; ++i)
    sinz[i] = std::sin((i + glob_start[2]) * twopi / gdims[2]);

  assert(u.size() == size_t(local_dims[0]) * local_dims[1] * local_dims[2]);
  auto *p = u.data();
  for (int k = 0; k < local_dims[2]; ++k)
  for (int j = 0; j < local_dims[1]; ++j) {
    auto sinyz = siny[j] * sinz[k];
    for (int i = 0; i < local_dims[0]; ++i)
      *p++ = sinx[i] * sinyz;
  }

  return u;
}

// We follow the p3dfft example: the normalisation is performed on the
// transformed (complex) array, and not in the real data after backward
// transform.
template <typename T>
void normalise(std::vector<T> &x, Dims<3> gdims) {
  double f = 1.0 / (double(gdims[0]) * double(gdims[1]) * double(gdims[2]));
  for (auto &v : x) v *= f;
}

double compare(const std::vector<double> &x, const std::vector<double> &y) {
  auto N = x.size();
  assert(N == y.size());
  double diff2 = 0;
  for (size_t n = 0; n < N; ++n) {
    auto diff = x[n] - y[n];
    diff2 += diff * diff;
  }
  return diff2;
}

double transform(const PencilSetup &pencil_x, const PencilSetup &pencil_z,
                 std::vector<double> &ui, std::vector<double> &ui_final,
                 std::vector<Complex> &uo, int repetitions, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  init_wave(ui, pencil_x);

  unsigned char op_f[] = "fft";
  unsigned char op_b[] = "tff";

  // p3dfft transform functions take pointers to real values.
  double *uo_ptr = reinterpret_cast<double *>(uo.data());

  // Warm-up
  Cp3dfft_ftran_r2c(ui.data(), uo_ptr, op_f);
  normalise(uo, pencil_x.dims_global);
  Cp3dfft_btran_c2r(uo_ptr, ui_final.data(), op_b);

  // Check result
  {
    double diff2_local = compare(ui, ui_final);
    double diff2;
    MPI_Reduce(&diff2_local, &diff2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0)
      std::cout << "L2 error: " << diff2 << std::endl;
  }

  // Initialise timers
  Cset_timers();

  double t = -MPI_Wtime();
  for (int n = 0; n < repetitions; ++n) {
    Cp3dfft_ftran_r2c(ui.data(), uo_ptr, op_f);
    normalise(uo, pencil_x.dims_global);
    Cp3dfft_btran_c2r(uo_ptr, ui_final.data(), op_b);
  }
  t += MPI_Wtime();
  t *= 1000 / repetitions;  // time in ms

  // Gather timing statistics.
  std::array<double, 12> timers;
  Cget_timers(timers.data());
  for (auto &t : timers) t *= 1e3 / repetitions;  // milliseconds

  if (rank == 0) {
    std::cout << "Average time over " << repetitions << " iterations: " << t
              << " ms" << std::endl;
    std::cout << "\nP3DFFT timers (in milliseconds)"
                 "\n===============================\n";
    print_timers(timers);
  }

  return t;
}

void print_timers(const std::array<double, 12> &timers) {
    for (int i = 0; i < timers.size(); ++i) {
      std::cout << " (" << std::setw(2) << std::right << (i + 1) << ")  "
                << std::setprecision(5) << std::setw(12) << std::left
                << timers[i] << TIMERS_TEXT[i] << std::endl;
      if ((i + 1) % 4 == 0)
        std::cout << std::endl;
    }

    // Average some timings for easier comparison with PencilFFTs.
    auto fw_alltoallv = (timers[0] + timers[1]) / 2;
    auto bw_alltoallv = (timers[2] + timers[3]) / 2;

    auto fw_r2c = timers[4];  // FFT X
    auto fw_c2c = timers[6];  // FFT Y
    auto bw_c2c = timers[9];  // iFFT Y
    auto bw_c2r = timers[11]; // iFFT X

    // Assume that FFTs in Z cost the same as FFTs in Y.
    auto fw_fft = (fw_r2c + 2 * fw_c2c) / 3;
    auto bw_fft = (bw_c2r + 2 * bw_c2c) / 3;

    // Pack + unpack time. We subtract the estimated time of FFTs in Z.
    auto fw_pack_unpack = timers[5] + timers[7] - fw_c2c;
    auto bw_pack_unpack = timers[8] + timers[10] - bw_c2c;

    std::cout << "Forward transforms\n"
              << "  Average Alltoallv = " << fw_alltoallv << "\n"
              << "  Average FFT       = " << fw_fft << "\n"
              << "  Average (un)pack  = " << fw_pack_unpack << "\n";

    std::cout << "\nBackward transforms\n"
              << "  Average Alltoallv = " << bw_alltoallv << "\n"
              << "  Average FFT       = " << bw_fft << "\n"
              << "  Average (un)pack  = " << bw_pack_unpack << "\n";

    // Verify times.
    auto fw_total_time = 2 * (fw_alltoallv + fw_pack_unpack) + 3 * fw_fft;
    auto bw_total_time = 2 * (bw_alltoallv + bw_pack_unpack) + 3 * bw_fft;
    auto total_time = fw_total_time + bw_total_time;
    std::cout << "\nTotal from timers: " << total_time << " ms/iteration\n";
}

struct BenchOptions {
  Dims<3> dims;
  int repetitions;
  std::unique_ptr<std::string> output;

  BenchOptions(int argc, char *const argv[]) {
    if (argc < 2) {
      throw std::runtime_error("Error: wrong number of arguments.");
    }

    int N = std::stoi(argv[1]);
    dims = {N, N, N};

    repetitions = argc >= 3 ? std::stoi(argv[2]) : DEFAULT_REPETITIONS;

    if (argc >= 4)
      output = std::unique_ptr<std::string>(new std::string(argv[3]));
  }
};

struct BenchResults {
  const BenchOptions &opt;
  double time_avg;
  Dims<2> proc_dims;

  void write_to_file() const {
    if (!opt.output) return;
    auto &filename = *opt.output;
    bool file_existed = std::ifstream(filename.c_str()).good();

    std::ofstream io(filename, std::ios::app);

    if (!file_existed)  // write header
      io << "# (1) Nx  (2) Ny  (3) Nz  (4) num_procs  "
         << "(5) P1  (6) P2  (7) repetitions  (8) time_ms\n";

    auto sep = "  ";
    for (auto d : opt.dims) io << sep << d;

    auto num_procs = proc_dims[0] * proc_dims[1];
    io << sep << num_procs;
    for (auto n : proc_dims) io << sep << n;

    io << sep << opt.repetitions << sep << time_avg << "\n";
  }
};

void run_benchmark(const BenchOptions &opt, MPI_Comm comm) {
  int myrank, Nproc;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &Nproc);

  if (myrank == 0) {
    auto dims = opt.dims;
    std::cout << "Number of processes:   " << Nproc << std::endl;
    std::cout << "Dimensions:            " << dims[0] << "×" << dims[1] << "×"
              << dims[2] << std::endl;
    std::cout << "Number of repetitions: " << opt.repetitions << std::endl;
    if (opt.output)
      std::cout << "Output file:           " << *(opt.output) << std::endl;
    std::cout << std::endl;
  }

  // Data dimensions.
  auto dims = opt.dims;

  // Create 2D Cartesian topology.
  Dims<2> pdims = {0, 0};
  MPI_Dims_create(Nproc, pdims.size(), pdims.data());

  if (pdims[0] > pdims[1]) {
    pdims[0] = pdims[1];
    pdims[1] = Nproc / pdims[0];
  }

  if (myrank == 0) {
    print(std::cout << "Dimensions: ", dims) << "\n";
    print(std::cout << "Processes:  ", pdims) << "\n";
  }

  // Initialise P3DFFT
  Dims<3> memsize;  // not used...
  Cp3dfft_setup(pdims.data(), dims[0], dims[1], dims[2], MPI_Comm_c2f(comm),
                dims[0], dims[1], dims[2], 0, memsize.data());

  // Get dimensions for input array - real numbers, X-pencil shape. Note that we
  // are following the Fortran ordering, i.e. the dimension with stride-1 is X.
  PencilSetup pencil_x(dims, 1);

  // Get dimensions for output array - complex numbers, Z-pencil shape.
  // Stride-1 dimension could be X or Z, depending on how the library was
  // compiled (stride1 option)
  PencilSetup pencil_z(dims, 2);

  // Allocate and initialise.
  std::vector<double> ui(pencil_x.size[0] * pencil_x.size[1] *
                         pencil_x.size[2]);
  auto ui_final = ui;
  std::vector<Complex> uo(pencil_z.size[0] * pencil_z.size[1] *
                          pencil_z.size[2]);

  double t_ms =
      transform(pencil_x, pencil_z, ui, ui_final, uo, opt.repetitions, comm);

  Cp3dfft_clean();

  // Write results
  if (myrank == 0)
    BenchResults{opt, t_ms, pdims}.write_to_file();
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  auto comm = MPI_COMM_WORLD;
  int myrank;
  MPI_Comm_rank(comm, &myrank);

  try {
    auto opt = BenchOptions(argc, argv);
    run_benchmark(opt, comm);
  } catch (std::exception &e) {
    if (myrank == 0) {
      std::cerr << e.what() << "\n";
      std::cerr << "Usage: " << USAGE << "\n";
    }
  }

  MPI_Finalize();
  return 0;
}
