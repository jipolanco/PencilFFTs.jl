#include <p3dfft.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// File based on p3dfft.3/sample/C++/test3D_r2c.C

constexpr char USAGE[] = "./bench_p3dfft N [repetitions] [output_file]";

template <int N> using Dims = std::array<int, N>;

constexpr int DEFAULT_REPETITIONS = 100;

template <class T, size_t N>
auto print(T &io, const Dims<N> &dims) -> decltype(io) {
  if (N == 0) return io << "()";
  io << "(" << dims[0];
  for (size_t n = 1; n < N; ++n) io << ", " << dims[n];
  io << ")";
  return io;
}

template <class T>
std::vector<T> allocate_data(const p3dfft::grid &grid) {
  auto &dims = grid.ldims;
  int size = dims[0] * dims[1] * dims[2];
  return std::vector<T>(size);
}

// This is adapted from test3D_r2c.C
std::vector<double> &init_wave(std::vector<double> &u,
                               const p3dfft::grid &grid) {
  Dims<3> local_dims; // local permuted dimensions (i.e. in storage order)
  Dims<3> glob_start; // starting position in global array

  for (int i = 0; i < 3; ++i) {
    int m = grid.mem_order[i];  // permutation
    local_dims[m] = grid.ldims[i];
    glob_start[m] = grid.glob_start[i];
  }

  auto &gdims = grid.gdims;  // non-permuted global dimensions

  // Note: the following assumes that there's no permutation of input data!!
  // (This seems to be assumed by the original code...)
  assert(grid.mem_order[0] == 0 && grid.mem_order[1] == 1 &&
         grid.mem_order[2] == 2);

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
void normalise(std::vector<T> &x, int gdims[3]) {
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

double transform(p3dfft::grid &grid_i, p3dfft::grid &grid_o, int repetitions) {
  using Real = double;
  using Complex = p3dfft::complex_double;
  auto comm = grid_i.mpi_comm_glob;
  int rank;
  MPI_Comm_rank(comm, &rank);

  // For some weird reason, creating these copies fixes a "free(): invalid
  // pointer" error.
  //
  // I'm not sure where the error comes from, but:
  //
  // - The error dissapeared when the inverse transforms were commented out
  //   (trans_b.exec(...)).
  //
  // - Apparently the error occurs when the destructor of the `uo` vector (a
  //   std::vector<Complex>) is called. Maybe the pointer is also deleted by
  //   p3dfft??.
  //
  // P3DFFT has a few problems with memory management...
  //
  auto grid_i_copy = p3dfft::grid(grid_i);
  auto grid_o_copy = p3dfft::grid(grid_o);

  // Transform types
  std::array<int, 3> transform_ids_fw = {
      p3dfft::R2CFFT_D, p3dfft::CFFT_FORWARD_D, p3dfft::CFFT_FORWARD_D};
  std::array<int, 3> transform_ids_bw = {
      p3dfft::C2RFFT_D, p3dfft::CFFT_BACKWARD_D, p3dfft::CFFT_BACKWARD_D};
  p3dfft::trans_type3D transforms_fw(transform_ids_fw.data());
  p3dfft::trans_type3D transforms_bw(transform_ids_bw.data());
  p3dfft::transform3D<Real, Complex> trans_f(grid_i, grid_o, &transforms_fw);
  p3dfft::transform3D<Complex, Real> trans_b(grid_o, grid_i, &transforms_bw);

  auto ui = allocate_data<double>(grid_i);
  auto ui_final = ui;
  auto uo = allocate_data<Complex>(grid_o);

  init_wave(ui, grid_i);

  // Warm-up
  trans_f.exec(ui.data(), uo.data(), false);
  normalise(uo, grid_i.gdims);
  trans_b.exec(uo.data(), ui_final.data(), true);

  // Check result
  {
    double diff2_local = compare(ui, ui_final);
    double diff2;
    MPI_Reduce(&diff2_local, &diff2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0)
      std::cout << "L2 error: " << diff2 << std::endl;
  }

#ifdef TIMERS
  p3dfft::timers.init();
#endif

  double t = -MPI_Wtime();
  for (int n = 0; n < repetitions; ++n) {
    trans_f.exec(ui.data(), uo.data(), false);
    normalise(uo, grid_i.gdims);
    trans_b.exec(uo.data(), ui_final.data(), true);
  }
  t += MPI_Wtime();
  t *= 1000 / repetitions;  // time in ms

  if (rank == 0)
    std::cout << "Average time over " << repetitions << " iterations: " << t
              << " ms" << std::endl;

#ifdef TIMERS
  p3dfft::timers.print(grid_i.mpi_comm_glob);
#endif

  return t;
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

  if (Nproc == 1) {
    throw std::runtime_error(
        "Error: P3DFFT will fail without any warning if run with 1 process!");
  }

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

  // Input grid
  Dims<3> gdims_i = dims;
  int dim_conj_sym = -1;
  Dims<3> proc_order_i = {0, 1, 2};
  Dims<3> mem_order_i = {0, 1, 2};
  Dims<3> pgrid_i = {1, pdims[0], pdims[1]};
  p3dfft::grid grid_i(gdims_i.data(), dim_conj_sym, pgrid_i.data(),
                      proc_order_i.data(), mem_order_i.data(), comm);

  // Output grid
  Dims<3> gdims_o = gdims_i;
  dim_conj_sym = 0;
  gdims_o[0] = gdims_o[0] / 2 + 1;
  Dims<3> pgrid_o = {pdims[0], pdims[1], 1};
  Dims<3> mem_order_o = {2, 1, 0};
  p3dfft::grid grid_o(gdims_o.data(), dim_conj_sym, pgrid_o.data(),
                      proc_order_i.data(), mem_order_o.data(), comm);

  double t_ms = transform(grid_i, grid_o, opt.repetitions);

  // Write results
  if (myrank == 0)
    BenchResults{opt, t_ms, pdims}.write_to_file();
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  p3dfft::setup();

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

  p3dfft::cleanup();
  MPI_Finalize();
  return 0;
}
