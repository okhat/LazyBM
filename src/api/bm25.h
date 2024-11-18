#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "api/internal.h"
#include "api/query_term.h"

struct BM25 : Term2Doc
{
    // static constexpr float k1 = 1.2;
    // static constexpr float b  = 0.75;

    static constexpr float k1 = 0.9;
    static constexpr float b  = 0.2;

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
            idf = log10((C.ndocs - T.df + 0.5) / (T.df + 0.5));
            idf = std::max(1.0E-6f, idf);
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

    Message send(const TermStats &T, float tf, const DocStats &D)
    {
        // if(T.idf < 1.5 and tf <= 1) return 0.0;

        float numerator   = tf * (k1 + 1);
        float denominator = tf + k1 * (1 - b + b * D.normalized_len);

        return T.idf * numerator / denominator;
    }

    Score receive(const std::vector<QueryTerm> &Q, const QueryTerm &t, const TermStats &T, Message msg)
    {
        // return t.term_weight * msg;
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
};

#include "api/f2exp.h"
#include "api/lmdir.h"
#include "api/pl2.h"
#include "api/spl.h"

#include "api/numterms.h"

// using ChosenTerm2Doc = BM25;