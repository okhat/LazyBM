#pragma once

#include <string>

struct QueryTerm
{
    bool negated  = false;
    bool use_bm25 = true;

    float k1    = 0.9;
    float b     = 0.75;
    float pl2_c = 1.0;

    float bm25_weight = 1.0, lmdir_weight = 1.0, pr_weight = 0.0, spl_weight = 1.0, pl2_weight = 1.0;
    float f2exp_s     = 1.0;
    float term_weight = 1.0;

    float lmdir_mu = 2000.0;

    QueryTerm()
    {
    }

    QueryTerm(std::string &term)
    {
        if (term[0] == '-')
        {
            negated = true;
            term.erase(term.begin());
            // term = "";
        }
    }

    QueryTerm(std::string &term, float a, float b, float c)
    {
        if (term[0] == '-')
        {
            negated = true;
            term.erase(term.begin());
            // term = "";
        }

        bm25_weight  = a / (a + b + c);
        lmdir_weight = b / (a + b + c);
        pr_weight    = c / (a + b + c);
    }
};
