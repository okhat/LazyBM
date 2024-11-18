#ifndef COMM_H
#define COMM_H

#include <vector>
#include <mpi.h>
#include "utils/log.h"

class Comm
{
  public:
    template <class T>
    static MPI_Request isend(const std::vector<T> &v, int dst, int tag)
    {
        MPI_Request request;
        MPI_Isend(v.data(), v.size() * sizeof(T), MPI_BYTE, dst, tag, Env::MPI_WORLD, &request);
        return request;
    }

    template <class T>
    static MPI_Request irecv(const std::vector<T> &v, int src, int tag)
    {
        MPI_Request request;
        MPI_Irecv(v.data(), v.size() * sizeof(T), MPI_BYTE, src, tag, Env::MPI_WORLD, &request);
        return request;
    }

    template <class T>
    static MPI_Request irecv(T *v, size_t size, int src, int tag)
    {
        MPI_Request request;
        MPI_Irecv(v, size * sizeof(T), MPI_BYTE, src, tag, Env::MPI_WORLD, &request);
        return request;
    }

    static void wait_all(std::vector<MPI_Request> &reqs)
    {
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    static float all_reduce(float f)
    {
        float output;
        MPI_Allreduce(&f, &output, 1, MPI_FLOAT, MPI_MIN, Env::MPI_WORLD);
        return output;
    }

    static std::vector<float> all_reduce(std::vector<float> &f)
    {
        std::vector<float> output(f.size());
        MPI_Allreduce(f.data(), output.data(), f.size(), MPI_FLOAT, MPI_MIN, Env::MPI_WORLD);
        return output;
    }

    static MPI_Request iall_reduce(std::vector<float> &f, std::vector<float> &output)
    {
        MPI_Request request;
        MPI_Iallreduce(f.data(), output.data(), f.size(), MPI_FLOAT, MPI_MIN, Env::MPI_WORLD, &request);
        return request;
    }

    template <class T>
    static std::vector<T> bcast_with_size(std::vector<T> &v, int root)
    {
        uint64_t size;

        if(Env::rank == root)
        {
            size = v.size();
            MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, Env::MPI_WORLD);
            MPI_Bcast(v.data(), size * sizeof(T), MPI_BYTE, root, Env::MPI_WORLD);
            return v;
        }
        
        else
        {
            MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, Env::MPI_WORLD);
            std::vector<T> out(size);
            MPI_Bcast(out.data(), size * sizeof(T), MPI_BYTE, root, Env::MPI_WORLD);
            return out;
        }
    }
};

#endif