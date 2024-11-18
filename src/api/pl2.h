#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "api/internal.h"
#include "api/query_term.h"

struct PL2 : Term2Doc
{
    // static constexpr float c = 19.5;

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
        float pl2;

        DocStats(const CollectionStats &C, const ::StaticDocStats &D)
        {
            // pl2 = std::log(1.0 + 19.5 / (float(D.len) / C.avg_doclen)) / 0.6931471806f;
            // pl2 = std::log(1.0 + 8.0 / (float(D.len) / C.avg_doclen)) / 0.6931471806f;
            pl2 = std::log(1.0 + 12.0 / (float(D.len) / C.avg_doclen)) / 0.6931471806f;
        }
    };

    static float log2(float a)
    {
        return std::log(a) / 0.6931471806f;
    }

    Message send(const TermStats &T, float tf, const DocStats &D)
    {
        if (T.lambda >= 1) return 0.0; /* As proposed by Fang et al. (TOIS'10) */
        float lambda = T.lambda;

        float tfn = tf * D.pl2;

        float a = tfn * log2(tfn / lambda);
        float b = (lambda - tfn) * 1.4426950;
        float d = 0.5 * log2(2 * 3.14159 * tfn);

        return (a + b + d) / (tfn + 1.0);
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

// using ChosenTerm2Doc = PL2;
