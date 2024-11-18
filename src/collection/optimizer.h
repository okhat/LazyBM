#pragma once

#include <set>
#include "api/bm25.h"
#include "collection/collection.h"

float get_block_bound(RectangularMatrix<EdgeWeight>& A, uint32_t idx, float z, uint32_t local_df, uint32_t range)
{
    if (local_df < (1 << 15)) return A.bmw_maxscores[idx][range];
    if (local_df < (1 << 17)) range >>= 1;
    if (local_df < (1 << 18)) range >>= 1;

    return A.qmaxscores[idx][range] ? (z * A.qmaxscores[idx][range] + 0.0001) : 0.0f;

    // return maxscores[range];
    // if (local_df < (52075585 / 100)) range >>= 2;
}

void optimize_model(Collection& C, RectangularMatrix<EdgeWeight>& A, const std::string& queries_path, uint32_t nqueries)
{
    using UserTermStats = ChosenTerm2Doc::TermStats;
    using Message       = RectangularMatrix<EdgeWeight>::Message;

    auto         model = ChosenTerm2Doc(A.collection_stats);
    DocIDsReader doc_ids_reader{A.doc_ids};

    /* Finalize the local doc stats. */
    for (auto& d1 : A.static_doc_stats) A.doc_stats.emplace_back(A.collection_stats, d1);

    for (uint32_t i = 0; i < A.terms.size(); i++)
    {
        A.qmaxscores.emplace_back();
        A.bmw_maxscores.emplace_back();
        A.doc_maxscores.emplace_back();
        A.maxscores.emplace_back();
        A.cache.emplace_back();

        A.bmw_wdoc.emplace_back();
        A.bmw_wdoc.back().resize(2 + (A.terms[i].local_df / SHARD_NDOCS));
        std::fill(A.bmw_wdoc.back().begin(), A.bmw_wdoc.back().end(), A.rank_ndocs);

        A.bmw_sdoc.emplace_back();
        A.bmw_sdoc.back().resize(2 + (A.terms[i].local_df / SHARD_NDOCS));
        std::fill(A.bmw_sdoc.back().begin(), A.bmw_sdoc.back().end(), A.rank_ndocs);

        uint32_t s = std::max(A.rank_ndocs >> SHARD_RADIX, A.terms[i].local_df / SHARD_NDOCS);
        A.qmaxscores.back().resize(2 + s);
        A.bmw_maxscores.back().resize(2 + s);
        A.doc_maxscores.back().resize(2 + s);
    }

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

        for (reader.next(); reader.value() < A.rank_ndocs; reader.next())
        {
            uint32_t doc = reader.value();

            auto freq = fiterator.advance_and_read();
            if (freq > 9999999)
            {
                LOG.info("i = %u (position = %u)\n", i, fiterator.reader.position());
                // break;
            }

            auto impact = model.send(term_stats, freq, A.doc_stats[doc]);
            term_bound  = std::max(term_bound, impact);

#if BMW or BMM or IBMM or LBMM or VBMW or LBMW
            uint32_t current_shard            = reader.position() / SHARD_NDOCS;
            A.bmw_maxscores[i][current_shard] = std::max(A.bmw_maxscores[i][current_shard], impact);
            A.bmw_wdoc[i][current_shard]      = doc;
            A.bmw_sdoc[i][current_shard]      = std::min(A.bmw_sdoc[i][current_shard], doc);

            auto doc_impact = model.self_send(A.doc_stats[doc], C.doc_idx2label(doc), C.doc_idx2pr(doc));
            A.doc_maxscores[i][current_shard] = std::max(A.doc_maxscores[i][current_shard], doc_impact);
#else
            uint32_t current_shard = doc >> SHARD_RADIX;
            if (term.local_df < (1 << 18)) current_shard >>= 1;
            if (term.local_df < (1 << 17)) current_shard >>= 1;

            // if (term.local_df < (52075585 / 100)) current_shard >>= 2;
            A.bmw_maxscores[i][current_shard] = std::max(A.bmw_maxscores[i][current_shard], impact);
#endif

            if (A.bmw_maxscores[i][current_shard] < -0.0001)
                LOG.info("A.bmw_maxscores[i][current_shard] = %f\n\n\n", A.bmw_maxscores[i][current_shard]);
        }

        A.maxscores[i] = term_bound;

        Message z = term_bound / 255.0f;

        for (uint32_t j = 0; j < A.bmw_maxscores[i].size(); j++)
        {
            if (A.bmw_maxscores[i][j] > 0.000000001)
                A.qmaxscores[i][j] = std::min(255, std::max(0, int(std::ceil(A.bmw_maxscores[i][j] / z))));
            // std::max(std::min(int(std::ceil((A.bmw_maxscores[i][j]) / z)), 255), 0);

            if (z * A.qmaxscores[i][j] + 0.0001 < A.bmw_maxscores[i][j])
            {
                LOG.info("%u, %u, %f, %f, %f\n", A.qmaxscores[i][j], uint32_t(1 + A.bmw_maxscores[i][j] / z),
                         z * A.qmaxscores[i][j], A.bmw_maxscores[i][j], term_bound);
                Env::exit(0);
            }
        }
    }

#if BMW or BMM or IBMM or LBMM or VBMW or LBMW
    /* Histogram for score distributions. */
    std::vector<uint32_t> score_fraction_counter(101);

    for (uint32_t i = 0; i < A.terms.size(); i++)
    {
        uint32_t this_size = int(std::ceil(A.terms[i].local_df / float(SHARD_NDOCS)));
        for (uint32_t j = 0; j < this_size; j++)
        {
            float fraction = std::lround(100.0 * A.bmw_maxscores[i][j] / A.maxscores[i]);
            score_fraction_counter[fraction] += 1;
        }
    }

    uint32_t denom    = accumulate(score_fraction_counter.begin(), score_fraction_counter.end(), 0);
    float    added_up = 0.0;

    for (uint32_t p = 0; p <= 100;)
    {
        uint32_t q         = 0;
        uint32_t numerator = 0;

        for (; q < 2; q++)
        {
            if (p + q > 100) break;
            numerator += score_fraction_counter[p + q];
        }

        printf("%u--%u%%,  %.1f%%\n", p, p + q - 1, 100.0 * float(numerator) / denom);
        added_up += 100.0 * float(numerator) / denom;
        p += q;
    }

    LOG.info("[togrep] added_up = %.3f\n", added_up);
#endif

    A.doc_bounds.clear();

    for (uint32_t doc = 0; doc < A.rank_ndocs; doc++)
    {
        if ((doc & (SHARD_NDOCS - 1)) == 0) A.doc_bounds.emplace_back();

        auto impact         = model.self_send(A.doc_stats[doc], C.doc_idx2label(doc), C.doc_idx2pr(doc));
        A.doc_bounds.back() = model.ubound(A.doc_bounds.back(), impact);
        A.global_doc_bound  = model.ubound(A.global_doc_bound, impact);
        A.doc_messages.push_back(impact);
    }

    A.doc_bounds.emplace_back();
}
