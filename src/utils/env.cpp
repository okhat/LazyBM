#include <sys/time.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <chrono>
#include "utils/env.h"


int Env::rank  ;  // my rank

int Env::nranks;  // num of ranks

bool Env::is_master;  // rank == 0?

MPI_Comm Env::MPI_WORLD;


void Env::init(RankOrder order)
{
  int mpi_threading;
  MPI_Init_thread(0, nullptr, MPI_THREAD_MULTIPLE, &mpi_threading);

  /* Set Environment Variables. */
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  is_master = rank == 0;

  MPI_WORLD = MPI_COMM_WORLD;
  if (order != RankOrder::KEEP_ORIGINAL)
    shuffle_ranks(order);
}

void Env::finalize()
{ MPI_Finalize(); }

void Env::exit(int code)
{
  finalize();
  std::exit(code);
}

void Env::barrier()
{ MPI_Barrier(MPI_WORLD); }


void Env::shuffle_ranks(RankOrder order)
{
  std::vector<int> ranks(nranks);

  if (is_master)
  {
    int seed = order == RankOrder::FIXED_SHUFFLE ? 0 : now();
    srand(seed);
    std::iota(ranks.begin(), ranks.end(), 0);  // ranks = range(len(ranks))
    std::random_shuffle(ranks.begin(), ranks.end());

    // Keep master in place
    std::swap(*ranks.begin(), *std::find(ranks.begin(), ranks.end(), 0));
    assert(ranks[0] == 0);
  }

  MPI_Bcast(ranks.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Group world_group, reordered_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group_incl(world_group, nranks, ranks.data(), &reordered_group);
  MPI_Comm_create(MPI_COMM_WORLD, reordered_group, &MPI_WORLD);

  MPI_Comm_rank(MPI_WORLD, &rank);
}


double Env::now()
{
  // struct timeval tv;
  // auto retval = gettimeofday(&tv, nullptr);
  // assert(retval == 0);
  // return (double) tv.tv_sec + (double) tv.tv_usec / 1e6;

  const static auto reference_time = std::chrono::high_resolution_clock::now();

  auto current_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(current_time - reference_time).count() / 1000.0 / 1000.0;
}
