#pragma once

struct CollectionStats
{
    uint32_t ndocs;
    uint32_t nterms;
    uint64_t ntokens    = 0;
    float    avg_doclen = 0.0;
};

struct StaticDocStats
{
    uint32_t degree = 0;
    uint32_t len    = 0;
    float    xx;
    float    yy;
    // NOTE: Saved to disk.... so loaded from disk, so cannot be changed...
};

// struct DocStats
// {
//     // uint32_t degree = 0;
//     // uint32_t len    = 0;
//     float normalized_len;
//     // float lmdir;
//     // float spl;
//     // float pl2;
// };

/*
 * NOTE: Must be a struct of UNSIGNED ints, due to MPI_Allreduce call.
 */
struct TermStats
{
    uint32_t df = 0;
    uint32_t cf = 0;  // FIXME: > 32-bit??!
};

using DocID = uint32_t;
using TermID = uint32_t;