#ifndef TFIDF_H
#define TFIDF_H

#include <cstdint>
#include <cmath>
#include "api/internal.h"


struct TFIDF : Term2Doc
{
    using Term2Doc::C;
    using Term2Doc::Term2Doc;

    using Message = float;
    using Score = float;

    float send(const TermStats &T, float tf, const DocStats &D)
    {
        float idf = log10(float(C.ndocs) / float(T.df));
        float tfidf = idf * log10(1.0 + tf);

        return tfidf;

        float idf = 1 + log10(float(C.ndocs) / (1.0 + float(T.df)));
        // return sqrtf(tf) * (idf * idf) / D.norm;
    }

    float receive(uint32_t query_nterms, bool from_self, float msg)
    {
        return msg;
    }
};

// using ChosenTerm2Doc = TFIDF;

#endif
