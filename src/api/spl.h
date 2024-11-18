#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "api/internal.h"
#include "api/query_term.h"
#include "fmath.h"

struct SPL : Term2Doc
{
    // static constexpr float c = 7.5;

    using Term2Doc::C;
    using Term2Doc::Term2Doc;

    using SelfMessage = float;
    using Message     = float;
    using Score       = float;

    struct TermStats
    {
        float lambda;

        TermStats(const CollectionStats &C, const ::TermStats &T)
        {
            lambda = float(T.cf) / float(C.ndocs);
        }
    };

    struct DocStats
    {
        float spl;

        DocStats(const CollectionStats &C, const ::StaticDocStats &D)
        {
            // spl = std::log(1 + 7.5 / (float(D.len) / C.avg_doclen));
            spl = std::log(1 + 9.0 / (float(D.len) / C.avg_doclen));
        }
    };

    Message send(const TermStats &T, float tf, const DocStats &D)
    {
        float lambda      = T.lambda;
        float ndt         = tf * D.spl;
        float numerator   = std::pow(lambda, float(ndt) / float(ndt + 1.0)) - lambda;
        // float numerator   = fmath::exp((float(ndt) / float(ndt + 1.0)) * fmath::log(lambda)) - lambda;
        float denominator = 1 - lambda;

        return -1.0 * std::log(numerator / denominator);
        // return -1.0 * fmath::log(numerator / denominator);
    }

    Score receive(const std::vector<QueryTerm> &Q, const QueryTerm &t, const TermStats &T, Message msg)
    {
        return msg;
    }

    SelfMessage self_send(const DocStats &D, char *, float)
    {
        return 0.0;
    }

    Score self_receive(const std::vector<QueryTerm> &Q, SelfMessage msg)
    {
        return 0.0;
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

// using ChosenTerm2Doc = SPL;
