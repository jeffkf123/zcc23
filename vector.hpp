#ifndef vector_hpp
#define vector_hpp

#include <omp.h>
#include <memory>
#include <utility>

#ifndef HAVE_MPI
using MPI_Comm           = int;
const int MPI_COMM_SELF  = 0;
const int MPI_COMM_WORLD = 1;
#else
#  include <mpi.h>
#endif

enum class MemorySpace
{
  Host,
  CUDA
};

template <typename Number>
class CellCSigmaMatrix
{
public:
    static const int C = 32;  // Block size

    CellCSigmaMatrix(std::size_t size)
    : size(size)
    {
        values.resize(size);
        column_indices.resize(size);
        block_lengths.resize(size / C);

        // Allocate device memory
        cudaMalloc(&d_values, size * sizeof(Number));
        cudaMalloc(&d_column_indices, size * sizeof(int));
        cudaMalloc(&d_block_lengths, (size / C) * sizeof(int));
    }

    ~CellCSigmaMatrix()
    {
        // Free device memory
        cudaFree(d_values);
        cudaFree(d_column_indices);
        cudaFree(d_block_lengths);
    }
      Vector<Number> multiply(const Vector<Number>& vec) const
    {
        Vector<Number> result = vec.copy_to_device(); // Ensure the result vector is on the device

        // Call the CUDA kernel
        int num_blocks = (size + C - 1) / C;
        multiplyKernel<<<num_blocks, C>>>(d_values, d_column_indices, d_block_lengths, vec.begin(), result.begin(), size);

        return result.copy_to_host(); // Copy the result back to the host
    }

    void computeOnHost(const std::vector<Number>& crs_values,
                       const std::vector<int>& crs_column_indices,
                       const std::vector<int>& crs_row_pointers)
    {
        // Convert CRS format to CELL-C-Sigma format
        int block_idx = 0;
        for (std::size_t row = 0; row < size; row += C)
        {
            for (std::size_t col = 0; col < size; col += C)
            {
                for (int i = 0; i < C; ++i)
                {
                    int crs_idx = crs_row_pointers[row + i];
                    while (crs_idx < crs_row_pointers[row + i + 1] && crs_column_indices[crs_idx] < col + C)
                    {
                        values[block_idx] = crs_values[crs_idx];
                        column_indices[block_idx] = crs_column_indices[crs_idx] % C;
                        ++block_idx;
                        ++crs_idx;
                    }
                }
            }
            block_lengths[row / C] = block_idx - (row / C) * C * C;
        }
          // Transfer data to device
        cudaMemcpy(d_values, values.data(), size * sizeof(Number), cudaMemcpyHostToDevice);
        cudaMemcpy(d_column_indices, column_indices.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_block_lengths, block_lengths.data(), (size / C) * sizeof(int), cudaMemcpyHostToDevice);
    

  }

private:
    std::vector<Number> values;
    std::vector<int> column_indices;
    std::vector<int> block_lengths;

    Number* d_values;
    int* d_column_indices;
    int* d_block_lengths;

    std::size_t size;

    void multiplyKernel(const Number* d_values, const int* d_column_indices, const int* d_block_lengths, const Number* d_vec, Number* d_result, std::size_t size)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= size)
            return;

        extern __shared__ Number shared_vec[];
        int block_start = blockIdx.x * C;
        int block_end = block_start + C;

        // Load vector elements into shared memory
        for (int i = threadIdx.x; i < C && block_start + i < size; i += blockDim.x)
        {
            shared_vec[i] = d_vec[block_start + i];
        }
        __syncthreads();

        Number sum = 0;
        int block_idx = row / C * C * C + threadIdx.x * C;
        for (int j = 0; j < C; ++j)
        {
            int col = d_column_indices[block_idx + j];
            sum += d_values[block_idx + j] * shared_vec[col];
        }

        atomicAdd(&d_result[row], sum);
    }
    
};

unsigned int get_n_mpi_ranks(MPI_Comm communicator)
{
  int n_ranks = 1;
#ifdef HAVE_MPI
  MPI_Comm_size(communicator, &n_ranks);
#endif
  return n_ranks;
}

unsigned int get_my_mpi_rank(MPI_Comm communicator)
{
  int my_rank = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(communicator, &my_rank);
#endif
  return my_rank;
}

template <typename Number>
Number mpi_sum(const Number local_sum, MPI_Comm communicator)
{
#ifdef HAVE_MPI
  if (std::is_same<Number, double>::value)
    {
      double global_sum = 0;
      MPI_Allreduce(
        &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, communicator);
      return global_sum;
    }
  else if (std::is_same<Number, float>::value)
    {
      float global_sum = 0;
      MPI_Allreduce(
        &local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, communicator);
      return global_sum;
    }
  else if (std::is_same<Number, std::size_t>::value)
    {
      std::size_t global_sum = 0;
      MPI_Allreduce(
        &local_sum, &global_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, communicator);
      return global_sum;
    }
  else
    {
      std::cout << "Unknown number type" << std::endl;
      std::abort();
    }
#else
  return local_sum;
#endif
}

#ifdef DISABLE_CUDA
#define AssertCuda(error_code)
#else
#define AssertCuda(error_code)                                          \
  if (error_code != cudaSuccess)                                        \
    {                                                                   \
      std::cout << "The cuda call in " << __FILE__ << " on line "       \
                << __LINE__ << " resulted in the error '"               \
                << cudaGetErrorString(error_code) << "'" << std::endl;  \
      std::abort();                                                     \
    }

template <typename Number>
__global__ void set_entries(const std::size_t N,
                            Number scalar,
                            Number *destination)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N)
    destination[idx] = scalar;
}

template <typename Number>
__global__ void vector_update(const std::size_t N,
                              Number my_scalar,
                              Number other_scalar,
                              const Number *source,
                              Number *destination)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N)
    destination[idx] = my_scalar * destination[idx] + other_scalar * source[idx];
}

#endif


template <int block_size, typename Number>
__global__ void do_dot(const std::size_t N,
                       const Number *    vec1,
                       const Number *    vec2,
                       Number *          result_device)
{
    __shared__ Number cache[block_size];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    Number temp = 0;
    while (tid < N)
    {
        temp += vec1[tid] * vec2[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(result_device, cache[0]);
}

/////////////////// implementation of actual vector class ///////////////////

template <typename Number>
class Vector
{
public:
  static const int block_size = 256;

  // Create a serial vector of the given size
  Vector(const std::size_t global_size, const MemorySpace memory_space)
    : communicator(MPI_COMM_SELF),
      data(nullptr),
      global_size(global_size),
      locally_owned_range_start(0),
      memory_space(memory_space)
  {
    resize(global_size);
  }

  // Create a parallel vector of given global size with the half-open local
  // range owned by the current processor
  Vector(const std::size_t                         global_size,
         const std::pair<std::size_t, std::size_t> locally_owned_range,
         const MemorySpace                         memory_space,
         const MPI_Comm                            communicator)
    : communicator(communicator),
      data(nullptr),
      global_size(global_size),
      locally_owned_range_start(locally_owned_range.first),
      memory_space(memory_space)
  {
    resize(locally_owned_range.second - locally_owned_range.first);
  }

  Vector(const Vector &other)
    : communicator(other.communicator),
      data(nullptr),
      global_size(other.global_size),
      locally_owned_range_start(other.locally_owned_range_start),
      memory_space(other.memory_space)
  {
    resize_fast(other.local_size);
    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaMemcpy(data,
                              other.data,
                              local_size * sizeof(Number),
                              cudaMemcpyDeviceToDevice));
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = other.data[i];
      }
  }

  ~Vector()
  {
    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaFree(data));
      }
    else
      delete[] data;
  }

  Vector &operator=(const Vector &other)
  {
    global_size               = other.global_size;
    locally_owned_range_start = other.locally_owned_range_start;
    if (memory_space != other.memory_space)
      {
        std::cout << "Cannot assign from one memory space to another"
                  << std::endl;
        exit(EXIT_FAILURE);
      }

    resize_fast(other.local_size);

    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaMemcpy(data,
                              other.data,
                              local_size * sizeof(Number),
                              cudaMemcpyDeviceToDevice));
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = other.data[i];
      }
    return *this;
  }

  const Number &operator()(const std::size_t index) const
  {
    return data[index];
  }

  Number &operator()(const std::size_t index)
  {
    return data[index];
  }

  void operator=(const Number scalar)
  {
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        const unsigned int n_blocks =
          (local_size + block_size - 1) / block_size;
        set_entries<<<n_blocks, block_size>>>(local_size, scalar, data);
#endif
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = scalar;
      }
  }

  // computes this += other_scalar * other
  void add(const Number other_scalar, const Vector &other)
  {
    sadd(1., other_scalar, other);
  }

  // computes this = my_scalar * this + other_scalar * other
  void
  sadd(const Number my_scalar, const Number other_scalar, const Vector &other)
  {
    assert_size(other);

    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        // TODO implement for GPU
#endif
      }
    else
      {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < local_size; ++i)
          data[i] = my_scalar * data[i] + other_scalar * other.data[i];
      }
  }

  Number l2_norm() const
  {
    const Number norm_sqr = norm_square();
    if (std::isfinite(norm_sqr))
      return std::sqrt(norm_sqr);
    else
      {
        std::cout << "Norm not finite, aborting";
        std::abort();
        return 0;
      }
  }

  Number norm_square() const
  {
    return dot(*this);
  }

  Number dot(const Vector &other) const
  {
    assert_size(other);

    Number local_sum = 0;
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        Number *result_device;
        AssertCuda(cudaMalloc(&result_device, sizeof(Number)));
        AssertCuda(cudaMemset(result_device, 0, sizeof(Number)));
        const unsigned int n_blocks =
          (local_size + block_size - 1) / block_size;
        do_dot<block_size, Number><<<n_blocks, block_size>>>(local_size,
                                                             data,
                                                             other.data,
                                                             result_device);
        cudaMemcpy(&local_sum, result_device, sizeof(Number),
                   cudaMemcpyDeviceToHost);
#endif
      }
    else
      {
#pragma omp parallel for reduction(+ : local_sum)
        for (std::size_t i = 0; i < local_size; ++i)
          local_sum += data[i] * other.data[i];
      }

    return mpi_sum(local_sum, communicator);
  }

  Number* begin()
  {
    return data;
  }

  const Number* begin() const
  {
    return data;
  }

   Vector copy_to_device()
  {
    if (memory_space == MemorySpace::CUDA)
    {
      return *this;
    }
    else
    {
      Vector<Number> other(global_size,
                           std::make_pair(locally_owned_range_start,
                                          locally_owned_range_start +
                                              local_size),
                           MemorySpace::CUDA,
                           communicator);
      AssertCuda(cudaMemcpy(other.data, data, local_size * sizeof(Number), cudaMemcpyHostToDevice));
      return other;
    }
  }

  Vector copy_to_host()
  {
    if (memory_space == MemorySpace::CUDA)
    {
      Vector<Number> other(global_size,
                           std::make_pair(locally_owned_range_start,
                                          locally_owned_range_start +
                                              local_size),
                           MemorySpace::Host,
                           communicator);
      AssertCuda(cudaMemcpy(other.data, data, local_size * sizeof(Number), cudaMemcpyDeviceToHost));
      return other;
    }
    else
    {
      return *this;
    }
  }

  std::size_t size() const
  {
    return global_size;
  }

  std::size_t size_on_this_rank() const
  {
    return local_size;
  }

private:
  MPI_Comm    communicator;
  Number *    data;
  std::size_t local_size;
  std::size_t global_size;
  std::size_t locally_owned_range_start;
  MemorySpace memory_space;

  void assert_size(const Vector &other) const
  {
    if (local_size != other.local_size)
      {
        std::cout << "The local sizes of the two vectors " << local_size
                  << " vs " << other.local_size << " do not match" << std::endl;
        std::abort();
      }
  }

  void resize(const std::size_t local_size)
  {
    resize_fast(local_size);
    this->operator=(0.);
  }

  void resize_fast(const std::size_t local_size)
  {
    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaFree(data));
        AssertCuda(cudaMalloc(&data, local_size * sizeof(Number)));
      }
    else
      {
        delete[] data;
        data             = new Number[local_size];
      }
    this->local_size = local_size;
  }
};

#endif 

//