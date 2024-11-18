#ifndef _LMDIR_H_
#define _LMDIR_H_

#include <cstdint>
#include <cmath>
#include "api/internal.h"

// template <typename T>
struct NumericScore
{
    float score = 0;

    enum
    {
        LessOrEqual,
        Larger,
        Uncomparable
    };

    NumericScore() {}
    NumericScore(const DocStats &D) {}

    void operator+=(const NumericScore &other)
    {
        score += other.score;
    }

    void compare(const NumericScore &other)
    {
        return (score > other.score) ? Larger : LessOrEqual;
    }
};

struct BM25 : NumericScore
{
    static constexpr float k1 = 1.2;
    static constexpr float b = 0.75;

    using NumericScore::NumericScore;

    void recieve(const DocStats &D, const TermStats &T, float tf)
    {
        float idf = log10((C.ndocs - T.df + 0.5) / (T.df + 0.5));
        float num = tf * (k1 + 1);
        float denom = tf + k1 * (1 - b + b * (D.doclen / C.avg_doclen));
        score = idf * num / denom;
    }
};

struct LMDir : Term2Doc
{
    constexpr static float mu = 2000.0;

    using Term2Doc::C;
    using Term2Doc::Term2Doc;

    struct Term : Msg
    {
        float msg;

        Msg(const TermStats &T)
        {
            msg = float(T.cf) / float(C.ntokens);
        }

        void initialize()
        {
        }
    };

    struct DocScore : Score
    {
        float score = 0.0;

        DocScore() {} // TODO: Is this needed?

        DocScore(const DocStats &D)
        {
            return log10(mu / (D.doclen + mu));
        }

        void initialize(Query..)
        {
        }

        void recieve(const DocStats &D, float msg, float w)
        {
            return 0.0;
        }

        DocScore combine(DocScore other)
        {
            score += other.score;
            return *this;
        }

        Score::Order compare(DocScore other)
        {
            return Score::True; // Or False, Or Uncomparable
        }

        DocScore finalize(Query..)
        {
            return *this;
        }
    };
};

// using ChosenTerm2Doc = LMDir;

#endif

// NOTE/FIXME: LM-Dir is non-stationary in ES!