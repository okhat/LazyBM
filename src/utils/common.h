#pragma once

#include <cassert>
#include "folly/EliasFano.h"
#include "utils/comm.h"
#include "utils/env.h"
#include "utils/log.h"

#define irg_likely(x) __builtin_expect((x), 1)
#define irg_unlikely(x) __builtin_expect((x), 0)

// #define SHARD_RADIX 9
#define SHARD_NDOCS (1 << SHARD_RADIX)

#define FULL_EVAL_SHARD_RADIX 13
#define FULL_EVAL_SHARD_NDOCS (1 << FULL_EVAL_SHARD_RADIX)

/*
 * Gemini-inspired Edge/Triple Type, via an explicit specialization for the
 * unweighted pair case.
 */
template <class Weight>
struct Triple
{
    uint32_t term, doc;
    Weight   weight;

    Triple &operator=(const Triple<Weight> &other)
    {
        doc    = other.doc;
        term   = other.term;
        weight = other.weight;
        return *this;
    }

    bool operator==(const Triple<Weight> &other) const
    {
        return doc == other.doc and term == other.term;
    }
} __attribute__((packed));

struct Empty
{
};

template <>
struct Triple<Empty>
{
    uint32_t term, doc;

    Triple &operator=(const Triple<Empty> &other)
    {
        doc  = other.doc;
        term = other.term;
        return *this;
    }

    bool operator==(const Triple<Empty> &other) const
    {
        return doc == other.doc and term == other.term;
    }
};

using Pair = Triple<Empty>;
