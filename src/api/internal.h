#ifndef _TERM_STATS_H_
#define _TERM_STATS_H_

#include <cstdint>
#include <cmath>

#include "collection/stats.h"

struct Query;


struct Term2Doc
{
    const CollectionStats &C;

    Term2Doc(const CollectionStats &C) : C(C)
    {
        // TODO: Crash if program_idx hasn't been set properly..
    }

    // float send_to_self(const DocStats &D)
    // {
    //     custom_send_to_self = false;
    //     return 0.0;
    // }

    // float apply(const CollectionStats &C, const Query &Q, const DocStats &D, float score)
    // {
    //     custom_apply = false;
    //     return score;
    // }

    // float threshold()
    // {
    //     custom_threshold = false;
    //     return 0.0;
    // }

    float ubound(float a, float b)
    {
        return std::max(a, b);
    }

    bool custom_send_to_self = true;
    bool custom_apply = true;
    bool custom_threshold = true;
};

struct Doc2Term
{
};

#endif
