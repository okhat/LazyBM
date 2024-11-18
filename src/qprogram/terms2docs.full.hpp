#include "qprogram/query.h"

// TODO: Is this relying on undefined behavior? Passing by value what might be destructed?
uint32_t QueryProgram::query2terms(const Query &query)
{
    /* Cleanup from previous query. */
    while (not pq.empty()) pq.pop();

    daat_terms.clear();
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

    DocIDsReader doc_ids_reader{A.doc_ids};

    for (auto i : qterms)
    {
        uint32_t idx = query.terms[i];

        assert(A.terms[idx].local_df >= 1);
        auto reader = doc_ids_reader.get_docs(idx, A.terms[idx].local_df);
        reader.next();

        auto fiterator = doc_ids_reader.get_freqs(idx, A.terms[idx].local_df, A.terms[idx].local_cf);

        auto term_stats = UserTermStats(A.collection_stats, A.user_term_stats[idx]);

        daat_terms.push_back({idx, i, reader, fiterator, term_stats});
        next_doc = std::min(next_doc, reader.value());
    }

    return next_doc;
}

void QueryProgram::terms2docs(const Query &query, bool reset_threshold)
{
    const uint32_t max_doc = A.rank_ndocs;

    if (reset_threshold)
        threshold     = model.threshold(query.terms.size());  // NOTE: This best called before removing the UNKs?
    uint32_t next_doc = query2terms(query);

    while (next_doc < max_doc)
    {
        // doc_evals++;

        uint32_t doc = next_doc;
        next_doc     = UINT32_MAX;

        auto score = model.self_receive(query.user_terms, A.doc_messages[doc]);

        for (auto &t : daat_terms)
        {
            auto &fiterator = t.fiterator;
            auto &reader    = t.reader;

            if (reader.value() == doc)
            {
                // send_calls++;

#if PRECOMPUTE_SEND
                auto msg = A.cache[t.idx][reader.position()];
                score    = score + msg;
#else
                auto partial = model.send(t.idf, fiterator.advance_and_read(), A.doc_stats[doc]);
                // auto partial = model.receive(query.user_terms, query.user_terms[t.query_idx], t.idf, msg);
                score = score + partial;
#endif

                reader.next();
            }

            next_doc = std::min(next_doc, reader.value());
        }

        // TODO: Don't call self_send! Rely on precomp.
        // auto msg     = model.self_send(A.doc_stats[doc], C.doc_idx2label(doc), C.doc_idx2pr(doc));
        // auto partial = model.self_receive(query.user_terms, msg);
        // score        = score + partial;

        if (irg_unlikely(not(score <= threshold)))
        {
            topk_insertions++;

            if (irg_likely(pq.size() == TopK)) pq.pop();

            pq.push({doc, score});

            threshold = (pq.size() == TopK) ? pq.top().score : threshold;
        }
    }
}
