#pragma once
// {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0}
// BM25 in {32: 5, 64: 9}
// PL2 in {6.0, 12.0, 22.0, 43.0, 85.0}
// SPL in {8: 3.0, 16: 7.0, 32: 15.0, 29.0, 52.0, 77.0}
// LMDir in {3.0, 7.0, 15.0, 32.0, 56.0, 85.0}

#include <set>
#include <atomic>
#include "api/bm25.h"
#include "collection/collection.h"
#include "collection/vbmw.hpp"

void optimize_model(Collection& C, RectangularMatrix<EdgeWeight>& A, const std::string& queries_path, uint32_t nqueries)
{
    using UserTermStats = ChosenTerm2Doc::TermStats;
    using Message       = ChosenTerm2Doc::Message;
    DocIDsReader doc_ids_reader{A.doc_ids};

    /* Finalize the local doc stats. */
    for (auto& d1 : A.static_doc_stats) A.doc_stats.emplace_back(A.collection_stats, d1);

    auto model = ChosenTerm2Doc(A.collection_stats);

    LOG.info("\n\nProcessing VBMW_COST = %.1f\n", VBMW_COST);

    for (uint32_t i = 0; i < A.terms.size(); i++)
    {
        A.bmw_wdoc.emplace_back();
        A.bmw_maxscores.emplace_back();
        A.bmw_wdoc.back().resize(2 + (A.terms[i].local_df / SHARD_NDOCS));
        A.bmw_maxscores.back().resize(2 + (A.terms[i].local_df / SHARD_NDOCS));
        std::fill(A.bmw_wdoc.back().begin(), A.bmw_wdoc.back().end(), A.rank_ndocs);
        A.maxscores.emplace_back();

        uint32_t s = std::max(A.rank_ndocs >> SHARD_RADIX, A.terms[i].local_df / SHARD_NDOCS);
        A.doc_maxscores.emplace_back();
        A.doc_maxscores.back().resize(2 + s);
    }

    std::atomic<size_t> total_postings(0);
    std::atomic<size_t> total_blocks(0);

    std::vector<DocCompression::Reader> readers;
    std::vector<FreqIterator>           fiterators;

    for (uint32_t i = 0; i < A.terms.size(); i++)
    {
        auto& term = A.terms[i];
        readers.emplace_back(doc_ids_reader.get_docs(i, A.terms[i].local_df));
        fiterators.emplace_back(doc_ids_reader.get_freqs(i, A.terms[i].local_df, A.terms[i].local_cf));
    }

#pragma omp parallel for schedule(dynamic, 8) num_threads(8)
    for (uint32_t i = 1; i < A.terms.size(); i++)
    {
        if (i % (A.terms.size() / 100) == 0)
            LOG.info("Processing term %u (df = %u, cf = %u)\n", i, A.terms[i].local_df, A.terms[i].local_cf);

        auto& term      = A.terms[i];
        auto& reader    = readers[i];
        auto& fiterator = fiterators[i];

        auto term_stats = UserTermStats(A.collection_stats, A.user_term_stats[i]);
        auto term_bound = Message();

        std::vector<std::tuple<uint64_t, float>> scores;

        for (reader.next(); reader.value() < A.rank_ndocs; reader.next())
        {
            uint32_t doc            = reader.value();
            uint32_t freq           = fiterator.advance_and_read();
            auto     current_impact = model.send(term_stats, freq, A.doc_stats[doc]);
            scores.emplace_back(doc, current_impact);
            term_bound = model.ubound(term_bound, current_impact);

            uint32_t current_shard = reader.position() / SHARD_NDOCS;
            auto     doc_impact    = model.self_send(A.doc_stats[doc], C.doc_idx2label(doc), C.doc_idx2pr(doc));
            A.doc_maxscores[i][current_shard] = std::max(A.doc_maxscores[i][current_shard], doc_impact);
        }

        auto p = ds2i::score_opt_partition(scores.begin(), scores.size(), VBMW_COST);

        total_postings += term.local_df;
        total_blocks += p.docids.size();

        LOG.info("[%u] total_postings / total_blocks = %f\n", i, double(total_postings) / double(total_blocks));

        A.bmw_maxscores[i] = p.max_values;
        A.bmw_wdoc[i]      = p.docids;
        A.maxscores[i]     = term_bound;

        A.bmw_wdoc[i].push_back(A.rank_ndocs);
        A.bmw_maxscores[i].emplace_back();
    }

    A.doc_bounds.clear();

    for (uint32_t doc = 0; doc < A.rank_ndocs; doc++)
    {
        if ((doc & (SHARD_NDOCS - 1)) == 0) A.doc_bounds.emplace_back();

        auto current_impact = model.self_send(A.doc_stats[doc], C.doc_idx2label(doc), C.doc_idx2pr(doc));
        A.doc_bounds.back() = model.ubound(A.doc_bounds.back(), current_impact);
        A.global_doc_bound  = model.ubound(A.global_doc_bound, current_impact);
        A.doc_messages.push_back(current_impact);
    }

    A.doc_bounds.emplace_back();
}
