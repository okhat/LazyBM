// // NOTE/FIXME: LM-Dir is non-stationary in ES!
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "api/internal.h"
#include "api/query_term.h"
#include "fmath.h"

struct LMDir : Term2Doc
{
    // constexpr static float mu = 2000.0;
    // constexpr static float mu = 1000.0;
    constexpr static float mu = 2500.0;

    using Term2Doc::C;
    using Term2Doc::Term2Doc;

    using SelfMessage = float;
    using Message     = float;
    using Score       = float;

    float __int_as_float(int32_t a)
    {
        float r;
        memcpy(&r, &a, sizeof(r));
        return r;
    }
    int32_t __float_as_int(float a)
    {
        int32_t r;
        memcpy(&r, &a, sizeof(r));
        return r;
    }

    float my_faster_logf(float a)
    {
        float   m, r, s, t, i, f;
        int32_t e;

        e = (__float_as_int(a) - 0x3f2aaaab) & 0xff800000;
        m = __int_as_float(__float_as_int(a) - e);
        i = (float)e * 1.19209290e-7f;  // 0x1.0p-23
        /* m in [2/3, 4/3] */
        f = m - 1.0f;
        s = f * f;
        /* Compute log1p(f) for f in [-1/3, 1/3] */
        r = fmaf(0.230836749f, f, -0.279208571f);  // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        t = fmaf(0.331826031f, f, -0.498910338f);  // 0x1.53ca34p-2, -0x1.fee25ap-2
        r = fmaf(r, s, t);
        r = fmaf(r, s, f);
        r = fmaf(i, 0.693147182f, r);  // 0x1.62e430p-1 // log(2)
        return r;
    }

    float logf_fast(float a)
    {
        union {
            float f;
            int   x;
        } u = {a};
        return (u.x - 1064866805) * 8.262958405176314e-8f; /* 1 / 12102203.0; */
    }

    inline float fast_log2(float val)
    {
        int *const exp_ptr = reinterpret_cast<int *>(&val);
        int        x       = *exp_ptr;
        const int  log_2   = ((x + 23) & 255) - 128;
        x &= ~(255 << 23);
        x += 127 << 23;
        *exp_ptr = x;

        val = ((-1.0f / 3) * val + 2) * val - 2.0f / 3;  // (1)

        return (val + log_2) * 0.69314718f;
    }

    struct TermStats
    {
        float p;

        TermStats(const CollectionStats &C, const ::TermStats &T)
        {
            p = float(T.cf) / float(C.ntokens);
            p = mu * p;
        }
    };

    struct DocStats
    {
        float len;

        DocStats(const CollectionStats &C, const ::StaticDocStats &D)
        {
            len = D.len;
        }
    };

    float send(const TermStats &T, float tf, const DocStats &D)
    {
        return std::log(tf / T.p + 1.0);
    }

    Score receive(const std::vector<QueryTerm> &Q, const QueryTerm &t, const TermStats &T, Message msg)
    {
        return msg;
    }

    SelfMessage self_send(const DocStats &D, char *, float)
    {
        return std::log(mu / (D.len + mu));
    }

    Score self_receive(const std::vector<QueryTerm> &Q, float msg)
    {
        return Q.size() * msg;
    }

    Score threshold(uint32_t nterms)
    {
        return 0.0f;
    }  // TODO: Give this the query!

    struct FinalScore
    {
        Score    score;
        uint32_t id         = 1000000;
        uint32_t local_rank = 100000;
        char     ClueWebID[26 + 1];

        FinalScore() : score()
        {
        }

        FinalScore(Score s, uint32_t id, uint32_t r, const char *str) : score(s), id(id), local_rank(r)
        {
            strncpy(ClueWebID, str, 26 + 1);
        }

        bool operator<=(const FinalScore &other) const
        {
            return score <= other.score;
        }
    };
};

// using ChosenTerm2Doc = LMDir;
