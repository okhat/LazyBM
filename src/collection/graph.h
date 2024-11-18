#pragma once

#include "rectangular_matrix.h"
#include "utils/common.h"
#include "utils/hashers.h"

struct EdgeListArgs
{
    std::string           path;
    uint32_t              ndocs, nterms, rank_ndocs = 0;
    std::vector<uint32_t> doc_shard_offsets;

    EdgeListArgs(std::string path, uint32_t ndocs, uint32_t nterms) : path(path), ndocs(ndocs), nterms(nterms)
    {
    }
    EdgeListArgs(std::string path, uint32_t ndocs, uint32_t nterms, uint32_t rank_ndocs,
                 std::vector<uint32_t> doc_shard_offsets)
        : path(path), ndocs(ndocs), nterms(nterms), rank_ndocs(rank_ndocs), doc_shard_offsets(doc_shard_offsets)
    {
    }
};

template <class Weight>
class Graph
{
   public:
    using WTriple = Triple<Weight>;

    const EdgeListArgs        args;
    const uint32_t            ndocs, rank_ndocs, nterms;
    std::vector<FILE *>       files;
    uint64_t                  nbytes = 0, nedges = 0;
    std::vector<uint32_t>     doc_shard_offsets;
    const std::vector<bool> & needed_terms;
    RectangularMatrix<Weight> A;

    Graph(EdgeListArgs args, const std::vector<bool> &needed_terms);

    uint64_t open_edge_lists();
    void     read_edge_lists();
};

/* Implementation */
#include "graph.hpp"