#include "qprogram/query.h"

uint32_t QueryProgram::query2terms(const Query &query)
{
    /* Cleanup from previous query. */
    while (not pq.empty()) pq.pop();

    maxscore_terms.clear();
    essentials.clear();
    nonessentials.clear();
    ub_pfxsum.clear();

    /* Remove UNKs. */
    qterms.resize(query.terms.size());
    std::iota(qterms.begin(), qterms.end(), 0);

    auto is_unk = [&](auto idx) { return query.terms[idx] == Collection::UNK; };
    qterms.erase(std::remove_if(qterms.begin(), qterms.end(), is_unk), qterms.end());

    /*  */
    uint32_t next_doc = UINT32_MAX;

    std::sort(qterms.begin(), qterms.end(), [&](const auto &a, const auto &b) {
        return A.terms[query.terms[a]].local_df > A.terms[query.terms[b]].local_df;
    });

    DocIDsReader doc_ids_reader{A.doc_ids};

    for (auto i : qterms)
    {
        uint32_t idx = query.terms[i];

        assert(A.terms[idx].local_df >= 1);

        auto term_stats = UserTermStats(A.collection_stats, A.user_term_stats[idx]);

        // maxscore_terms.push_back(
        //     {idx, i, A.coll[idx], term_stats, A.maxscores[idx], uint32_t(-1), Score(), A.terms[idx].local_df});

        maxscore_terms.push_back({idx, i, EFWrapper(doc_ids_reader, idx, A.terms[idx].local_df, A.terms[idx].local_cf),
                                  term_stats, A.maxscores[idx], uint32_t(-1), Score(), A.terms[idx].local_df});

        next_doc = std::min(next_doc, uint32_t(maxscore_terms.back().reader.docid()));
    }

    return next_doc;
}

void QueryProgram::terms2docs(const Query &query, bool reset_threshold)
{
    const uint32_t max_doc = A.rank_ndocs;

    if (reset_threshold) threshold = model.threshold(query.terms.size());
    uint32_t next_doc              = query2terms(query);

    /* Compute Prefix Sum */
    auto doc_maxscore       = model.self_receive(query.user_terms, A.global_doc_bound);

    ub_pfxsum.push_back(Score());
    for (auto &t : maxscore_terms) ub_pfxsum.push_back(ub_pfxsum.back() + t.maxscore);

    uint32_t optionals = 0;

    while (next_doc < max_doc)
    {
        // pivot_selections++;
        // doc_evals++;

        uint32_t doc = next_doc;
        next_doc     = UINT32_MAX;

        auto score = model.self_receive(query.user_terms, A.doc_messages[doc]);

        for (uint32_t i = optionals; i < maxscore_terms.size(); i++)
        {
            auto &t      = maxscore_terms[i];
            auto &reader = t.reader;

            if (reader.docid() == doc)
            {
                // send_calls++;
                // next_calls++;
                score += model.send(t.idf, reader.freq(), A.doc_stats[doc]);
                reader.next();
            }

            next_doc = std::min(next_doc, uint32_t(reader.docid()));
        }

        for (int i = int(optionals) - 1; i >= 0; i--)
        {
            if (score + ub_pfxsum[i + 1] <= threshold) break;

            auto &t      = maxscore_terms[i];
            auto &reader = t.reader;

            if (reader.docid() < doc)
            {
                // next_calls++;
                reader.next_geq(doc);
            }
            
            if (reader.docid() == doc)
            {
                // send_calls++;
                score += model.send(t.idf, reader.freq(), A.doc_stats[doc]);
            }
        }

        if (topk_insert(doc, score))
        {
            while (optionals < maxscore_terms.size())
            {
                if (doc_maxscore + ub_pfxsum[optionals + 1] <= threshold)
                    optionals++;

                else
                    break;
            }
        }
    }
}
