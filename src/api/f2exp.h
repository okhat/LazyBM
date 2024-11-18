#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "api/internal.h"
#include "api/query_term.h"

struct F2EXP : Term2Doc
{
    // static constexpr float f2exp_s = 0.5;
    // static constexpr float f2exp_s = 0.6;
    static constexpr float f2exp_s = 0.7;

    using Term2Doc::C;
    using Term2Doc::Term2Doc;

    using SelfMessage = float;
    using Message     = float;
    using Score       = float;

    struct TermStats
    {
        float idf;

        TermStats(const CollectionStats &C, const ::TermStats &T)
        {
            idf = float(C.ndocs + 1) / T.df;
            idf = std::pow(idf, 0.35f);  /* Value of k=0.35 from Lucene. */
            // idf = std::log(idf);
        }
    };

    struct DocStats
    {
        float normalized_len;

        DocStats(const CollectionStats &C, const ::StaticDocStats &D)
        {
            normalized_len = float(D.len) / C.avg_doclen;
        }
    };

    static float log2(float a)
    {
        return std::log(a) / 0.6931471806f;
    }

    Message send(const TermStats &T, float tf, const DocStats &D)
    {
        return tf / (tf + f2exp_s + f2exp_s * D.normalized_len);
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
        return msg;
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

// using ChosenTerm2Doc = F2EXP;
