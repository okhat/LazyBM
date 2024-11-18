#pragma once

#include "api/query_term.h"
#include "collection/collection.h"

struct Query
{
    constexpr static uint32_t MaxQueryLength = 512;
    const Collection &        C;
    std::vector<uint32_t>     terms;
    std::vector<QueryTerm>    user_terms;

    Query(const Collection &C) : C(C)
    {
    }
    void from_text(const std::string &text)
    {
        terms.clear();
        user_terms.clear();

        std::istringstream iss(text);
        std::string        term;

        float a = rand(), b = rand(), c = rand();

        while (iss >> term)
        {
            user_terms.emplace_back(term, a, b, c);  // could modify term
            uint32_t idx = C.term_label2idx(term);

            if (idx == C.UNK)
            {
                // LOG.info<true, false>(" %s [UNK]  ", term.c_str());
            }
            else
            {
                // LOG.info<true, false>(" %s [%uk]  ", term.c_str(), C.G.A.user_term_stats[idx].df / 1000);
            }

            terms.push_back(idx);
        }

        // LOG.info<true, false>("\n");
    }

    static void broadcast_text(std::string &text)
    {
        char buf[MaxQueryLength];

        if (Env::is_master)
        {
            uint32_t query_nbytes = text.length() + 1;

            if (query_nbytes >= MaxQueryLength) Env::exit(1);

            std::copy(text.data(), text.data() + query_nbytes, buf);
            MPI_Bcast(buf, MaxQueryLength, MPI_BYTE, 0, Env::MPI_WORLD);
        }

        else
        {
            MPI_Bcast(&buf, MaxQueryLength, MPI_BYTE, 0, Env::MPI_WORLD);
            text = buf;
        }
    }
};

// TODO: If buf too short, handle by marking the last bit in buf and sending a variable-sized buffer.