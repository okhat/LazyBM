#include <omp.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// NOTE: Use those normally...
#define CachePrefix "/datasets2/IRg/CatB/1.pisa_CRAWL_filtered_fixed/"
#define COMPRESS_UPPER_BOUNDS true
#define FILTER_TERMS true
#define CACHE_REUSE true

// NOTE: Effectiveness...
// #define CachePrefix "/datasets2/IRg/CatB/1.pisa_URL_effectiveness_filtered/"
// #define COMPRESS_UPPER_BOUNDS true
// #define FILTER_TERMS true
// #define CACHE_REUSE true

// NOTE: But, May 24th 10:00 am, let me try this.
// #define CachePrefix "/datasets2/IRg/CatB/1.pisa_CRAWL_filtered_fixed-replicate2/"
// #define COMPRESS_UPPER_BOUNDS true
// #define FILTER_TERMS true
// #define CACHE_REUSE false

#define GlobalPrefix "/datasets2/ClueWeb12_graph/CatB/1x/"
#define MaxDiskShard 2048
using EdgeWeight = uint32_t;

std::vector<uint32_t> all_shards(MaxDiskShard);

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "collection/collection.h"

#if VBMW
#include "collection/optimizer.vbmw.h"
#else
#include "collection/optimizer.h"
#endif

#include "qprogram/qprogram.h"
#include "qprogram/query.h"

int main(int argc, char **argv)
{
    Env::init();

#ifdef BMW
    LOG.info("BMW, SHARD_RADIX=%u\n", SHARD_RADIX);
#endif

#ifdef LBMW
    LOG.info("LBMW, SHARD_RADIX=%u\n", SHARD_RADIX);
#endif

#ifdef VBMW
    LOG.info("VBMW, SHARD_RADIX=%u, VBMW_COST=%f\n", SHARD_RADIX, VBMW_COST);
#endif

#ifdef DBMW
    LOG.info("DBMW, SHARD_RADIX=%u\n", SHARD_RADIX);
#endif

#ifdef DBMM
    LOG.info("DBMM, SHARD_RADIX=%u\n", SHARD_RADIX);
#endif

#ifdef MaxScore
    LOG.info("MaxScore, SHARD_RADIX=%u\n", SHARD_RADIX);
#endif

#ifdef WAND
    LOG.info("WAND, SHARD_RADIX=%u\n", SHARD_RADIX);
#endif

#ifdef LazyMaxScore
    LOG.info("LazyMaxScore, SHARD_RADIX=%u\n", SHARD_RADIX);
#endif

    std::iota(all_shards.begin(), all_shards.end(), 1);

    srand(0);

    if (Env::nranks > 1) std::random_shuffle(all_shards.begin(), all_shards.end());

    /* Command line arguments. */
    if (argc != 4)
    {
        LOG.info("Usage: %s <index path> <ndocs> <nterms>\n", argv[0]);
        Env::exit(0);
    }

    char *   path   = argv[1];
    uint32_t ndocs  = std::atoi(argv[2]);
    uint32_t nterms = std::atoi(argv[3]);

    std::string Queries_5K = "/datasets/okhattab/queries/mqt_new.txt";
    // std::string Queries_5K = "/datasets/okhattab/queries/PISA_trec_13_14.txt";
    uint32_t    nqueries   = 1000;
    Collection  C({path, ndocs, nterms}, Queries_5K, nqueries);

    optimize_model(C, C.G.A, Queries_5K, nqueries);

    /* Set up the Query Program. */
    QueryProgram qp(C);
    Query        query(C);

#ifdef DAAT
    for (uint32_t xx = 0; xx < 1; xx++)
#else
    for (uint32_t xx = 0; xx < 2; xx++)
#endif
    { /* Queries. */
        std::ifstream queries_file(Queries_5K);

        std::vector<double> query_num_by_length;
        std::vector<double> query_total_time_by_length;
        query_num_by_length.resize(9);
        query_total_time_by_length.resize(9);

        std::vector<double> query_times;
        query_times.reserve(nqueries);
        std::string text;

        float checksum = 0;
        qp.next_calls = qp.topk_insertions = qp.pivot_selections = qp.send_calls = qp.doc_evals = 0;

        for (auto q = 0u; q < nqueries; q++)
        {
            Env::barrier();
            // LOG.info<true, false>("#> Query %04u\n", q);

            if (Env::is_master) std::getline(queries_file, text);

            // Query::broadcast_text(text);
            query.from_text(text);

            uint32_t bucket = std::min(query.terms.size(), 8ul);
            query_num_by_length[bucket]++;

            if (xx == 0)
                qp.local_TopK = qp.TopK = 10;
            else
                qp.local_TopK = qp.TopK = 1000;

            double t = Env::now();
            qp.terms2docs(query, true);
            t = (Env::now() - t) * 1000.0;

            checksum += roundf(qp.threshold * 1000) / 1000;
            // LOG.info("[togrep] %.3f", qp.threshold);

            bool good_case = qp.distributed_topk(false);

            query_times.push_back(t);
            query_total_time_by_length[bucket] += t;

            // for (uint32_t t = 0; t < 1000; t++)
            // {
            //     const auto &doc = qp.topk[t];
            //     // LOG.info("[togrep] %u   Q0   %u   %u   %f   IRg\n", 1 + q, doc.id, 1 + t, doc.score);
            //     LOG.info("[togrep] %u   Q0   %s   %u   %f   IRg\n", 201 + q, doc.ClueWebID, 1 + t, doc.score);
            // }
        }

        LOG.info<true, false>("\n\n\n");

        for (int r = 0; r <= 0; r++)
        {
            if (r == Env::rank)
            {
                LOG.info<false>("Top-(%u, %u) Avg Time: %.3f ms\n\n", qp.local_TopK, qp.TopK,
                                std::accumulate(query_times.begin(), query_times.end(), 0.0) / double(nqueries));

                for (uint32_t i = 1; i < query_num_by_length.size(); i++)
                {
                    if (query_num_by_length[i])
                    {
                        LOG.info<false, false>("[Length %u: %.3f] ", i,
                                               double(query_total_time_by_length[i]) / double(query_num_by_length[i]));
                    }
                }

                std::sort(query_times.begin(), query_times.end());

                LOG.info("\n 25-percentile:  %.3f ms\n", query_times[query_times.size() * 25 / 100]);
                LOG.info("50-percentile:  %.3f ms\n", query_times[query_times.size() * 50 / 100]);
                LOG.info("90-percentile:  %.3f ms\n", query_times[query_times.size() * 90 / 100]);
                LOG.info("95-percentile:  %.3f ms\n", query_times[query_times.size() * 95 / 100]);
                LOG.info("99-percentile:  %.3f ms\n", query_times[query_times.size() * 99 / 100]);

                LOG.info("Avg pivot_selections = %lu, doc_evals = %lu, send_calls = %lu, next_calls = %lu\n",
                         qp.pivot_selections / nqueries, qp.doc_evals / nqueries, qp.send_calls / nqueries,
                         qp.next_calls / nqueries);

                LOG.info<false, false>("\n");
            }
        }

        LOG.info("\n Checksum = %.4f\n", checksum);
        LOG.info("Queries_5K = %s\n", Queries_5K.c_str());

        LOG.info<true, false>("\n\n\n");
    }

    return 0;
}
