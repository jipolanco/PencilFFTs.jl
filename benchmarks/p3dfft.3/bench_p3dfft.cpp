#include <p3dfft.h>

#include <array>
#include <cassert>
#include <iostream>
#include <vector>

// File based on p3dfft.3/sample/C++/test3D_r2c.C

template <int N> using Dims = std::array<int, N>;

constexpr Dims<3> GLOBAL_DIMS = {128, 128, 128};
constexpr int NUM_REPETITIONS = 10;

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

void transform(p3dfft::grid &grid_i, p3dfft::grid &grid_o) {
  using Real = double;
  using Complex = p3dfft::complex_double;
  auto comm = grid_i.mpi_comm_glob;
  int rank;
  MPI_Comm_rank(comm, &rank);

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
  for (int n = 0; n < NUM_REPETITIONS; ++n) {
    trans_f.exec(ui.data(), uo.data(), false);
    normalise(uo, grid_i.gdims);
    trans_b.exec(uo.data(), ui_final.data(), true);
  }
  t += MPI_Wtime();

  if (rank == 0)
    std::cout << "Average time over " << NUM_REPETITIONS
              << " iterations: " << (t / NUM_REPETITIONS) << " s" << std::endl;

#ifdef TIMERS
  p3dfft::timers.print(grid_i.mpi_comm_glob);
#endif
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  p3dfft::setup();

  auto comm = MPI_COMM_WORLD;

  int myrank, Nproc;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &Nproc);

  if (Nproc == 1) {
    std::cerr << "Error: P3DFFT will fail without any warning if run with 1 "
                 "process!\n";
    MPI_Finalize();
    return 1;
  }

  // Data dimensions.
  auto dims = GLOBAL_DIMS;

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

  transform(grid_i, grid_o);

  p3dfft::cleanup();

  MPI_Finalize();
  return 0;
}
