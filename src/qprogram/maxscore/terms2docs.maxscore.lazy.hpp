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

        maxscore_terms.push_back({idx, i, EFWrapper(doc_ids_reader, idx, A.terms[idx].local_df, A.terms[idx].local_cf),
                                  term_stats, A.maxscores[idx], uint32_t(-1), Score(), A.terms[idx].local_df,
                                  A.maxscores[idx] / 255.0f});

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
    global_doc_maxscore = model.self_receive(query.user_terms, A.global_doc_bound);
    ub_pfxsum.push_back(Score());

    DocIDsReader doc_ids_reader{A.doc_ids};

    for (auto &t : maxscore_terms)
    {
        ub_pfxsum.push_back(ub_pfxsum.back() + t.maxscore);

        if (true)
        {
            if (t.local_df < (1 << 15))
            {
                t.maxscore = Message();
                auto reader = EFWrapper(doc_ids_reader, t.idx, A.terms[t.idx].local_df, A.terms[t.idx].local_cf);
                // auto reader = A.coll[t.idx];
                A.bmw_maxscores[t.idx].clear();
                A.bmw_maxscores[t.idx].resize(2 + (max_doc >> SHARD_RADIX));

                while (reader.docid() < A.rank_ndocs)
                {
                    uint32_t shard = reader.docid() >> SHARD_RADIX;

                    auto msg                      = model.send(t.idf, reader.freq(), A.doc_stats[reader.docid()]);
                    A.bmw_maxscores[t.idx][shard] = std::max(A.bmw_maxscores[t.idx][shard], msg);
                    t.maxscore                    = std::max(t.maxscore, msg);
                    reader.next();

                    // next_calls++;
                    // send_calls++;
                }
            }
        }
    }

    uint32_t optionals = 0;
    uint32_t max_range = 1 + (max_doc >> SHARD_RADIX);

    for (uint32_t range = 0; range < max_range; range++)
    {
        auto ubsum = model.self_receive(query.user_terms, A.doc_bounds[range]);

        for (uint32_t i = optionals; i < maxscore_terms.size(); i++)
        {
            auto &t = maxscore_terms[i];
            ubsum += get_block_bound(A, t.idx, t.z, t.local_df, range);
        }

        if (ubsum + ub_pfxsum[optionals] <= threshold) continue;

        for (uint32_t i = 0; i < optionals; i++)
        {
            auto &t = maxscore_terms[i];
            ubsum += get_block_bound(A, t.idx, t.z, t.local_df, range);
        }

        if (ubsum > threshold)
        {
            uint32_t offset = range << SHARD_RADIX;
            uint32_t endpos = std::min(max_doc, offset + SHARD_NDOCS);

            process_range(query, range, offset, endpos);

            while (optionals < maxscore_terms.size())
            {
                if (not(global_doc_maxscore + ub_pfxsum[optionals + 1] <= threshold)) break;
                optionals++;
            }
        }
    }
}

__attribute__((__always_inline__)) inline void QueryProgram::process_range(const Query &query, uint32_t range,
                                                                           uint32_t offset, uint32_t endpos)
{
    doc_maxscore = model.self_receive(query.user_terms, A.doc_bounds[range]);

    essentials.clear();
    nonessentials.clear();
    ub_pfxsum1.clear();

    uint32_t next_doc = UINT32_MAX;

    auto ubsum = Score();
    ub_pfxsum1.push_back(ubsum);
    for (uint32_t i = 0; i < maxscore_terms.size(); i++)
    {
        auto &t = maxscore_terms[i];

        t.block_maxscore = get_block_bound(A, t.idx, t.z, t.local_df, range);

        auto ubsum_ = ubsum + t.block_maxscore;

        if (t.block_maxscore < 0.0000001) continue;

        if (doc_maxscore + ubsum_ <= threshold)
        {
            ubsum = ubsum_;
            ub_pfxsum1.push_back(ubsum);
            nonessentials.push_back(i);
        }
        else
        {
            essentials.push_back(i);

            if (t.reader.docid() < offset)
            {
                // next_calls++;
                t.reader.next_geq(offset);
            }

            next_doc = std::min(next_doc, uint32_t(t.reader.docid()));
        }
    }

    while (next_doc < endpos) next_doc = process_doc(query, range, next_doc);
}

#define BE_LAZY true

__attribute__((__always_inline__)) inline uint32_t QueryProgram::process_doc(const Query &query, uint32_t range,
                                                                             uint32_t doc)
{
    auto score = model.self_receive(query.user_terms, A.doc_messages[doc]);
    // pivot_selections++;

    uint32_t next_doc     = UINT32_MAX;
    int      nonessential = nonessentials.size();

#if BE_LAZY
    if (nonessential)
    {
        /* First-level Scoring. */
        auto ub = score;

        for (auto i : essentials)
        {
            auto &t = maxscore_terms[i];
            ub += (t.reader.docid() == doc) ? t.block_maxscore : 0.0;
        }

        for (int i = nonessential - 1; i >= 0; i--)
        {
            if (ub > threshold) break;
            if (ub + ub_pfxsum1[i + 1] <= threshold) break;

            auto &t = maxscore_terms[nonessentials[i]];

            if (t.reader.docid() < doc)
            {
                // next_calls++;
                t.reader.next_geq(doc);
            }

            ub += (t.reader.docid() == doc) ? t.block_maxscore : 0;
        }

        if (ub <= threshold)
        {
            for (auto i : essentials)
            {
                auto &t = maxscore_terms[i];
                if (t.reader.docid() == doc)
                {
                    // next_calls++;
                    t.reader.next();
                }

                next_doc = std::min(next_doc, uint32_t(t.reader.docid()));
            }

            return next_doc;
        }
    }
#endif

    /* Second-level Scoring. */
    // doc_evals++;

    for (auto i : essentials)
    {
        auto &t      = maxscore_terms[i];
        auto &reader = t.reader;

        if (reader.docid() == doc)
        {
            // send_calls++;
            // next_calls++;

            auto partial = model.send(t.idf, reader.freq(), A.doc_stats[doc]);
            score += partial;
            reader.next();
        }

        next_doc = std::min(next_doc, uint32_t(reader.docid()));
    }

    for (int i = nonessential - 1; i >= 0; i--)
    {
        if (score + ub_pfxsum1[i + 1] <= threshold) return next_doc;

        auto &t      = maxscore_terms[nonessentials[i]];
        auto &reader = t.reader;

        if (reader.docid() < doc)
        {
            // next_calls++;
            reader.next_geq(doc);
        }

        if (reader.docid() == doc)
        {
            // send_calls++;
            auto partial = model.send(t.idf, reader.freq(), A.doc_stats[doc]);
            score += partial;
        }
    }

    topk_insert(doc, score);

    return next_doc;
}
