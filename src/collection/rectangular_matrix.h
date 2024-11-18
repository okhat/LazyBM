#pragma once

#include <math.h>
#include <sys/stat.h>
#include <cassert>
#include <vector>
#include "api/bm25.h"
#include "collection/doc_ids.h"
#include "collection/stats.h"
#include "structures/fixed_vector.h"
#include "tsl/sparse_map.h"
#include "utils/common.h"
#include "utils/dist_timer.h"

template <class Weight>
struct RectangularMatrix
{
    using WTriple = Triple<Weight>;

    struct Term
    {
        uint64_t shard_exists_offset;
        uint64_t bounds_offset;
        uint32_t local_df;
        uint32_t local_cf;
        uint32_t term_idx;
    };


    /* Collection Cardinalities. */
    const uint32_t nterms;
    const uint32_t rank_ndocs;

    /* Term Mappings. */
    std::vector<bool> rank_terms;
    tsl::sparse_map<uint32_t, uint32_t> term_sparse2dense;

    /* Statistics (for collection, *local* terms, and *local* docs). */
    CollectionStats                       collection_stats;
    std::vector<TermStats>                user_term_stats;
    std::vector<StaticDocStats>           static_doc_stats;
    std::vector<ChosenTerm2Doc::DocStats> doc_stats;

    /* Compressed Edges. */
    std::vector<size_t> cbounds_offsets;
    std::vector<Term>   terms;
    DocIDs              doc_ids;

/* Weights. */
#if PRECOMPUTE_SEND
    using Message     = uint32_t;
    using SelfMessage = uint32_t;
#else
    using Message     = ChosenTerm2Doc::Message;
    using SelfMessage = ChosenTerm2Doc::SelfMessage;
#endif

    std::vector<uint8_t> cbounds_idxs;
    std::vector<Message> cbounds;

    float                             index_ub;
    std::vector<std::vector<Message>> cache;
    std::vector<Message>              bounds;
    std::vector<SelfMessage>          doc_bounds;
    std::vector<SelfMessage>          doc_messages;

    std::vector<bool> shard_exists;

    /* MaxScore! FIXME: ! */
    std::vector<Message>               maxscores;
    
    std::vector<std::vector<uint8_t>>  qmaxscores;
    std::vector<std::vector<Message>>  bmw_maxscores;
    std::vector<std::vector<Message>>  doc_maxscores;
    std::vector<std::vector<uint32_t>> bmw_wdoc;
    std::vector<std::vector<uint32_t>> bmw_sdoc;
    
    SelfMessage                        global_doc_bound;
    std::vector<std::vector<uint32_t>> fwd;
    std::vector<std::vector<uint32_t>> fwd_freq;

    RectangularMatrix(uint32_t global_ndocs, uint32_t rank_ndocs_, uint32_t nterms, uint64_t ub_rank_ntriples)
        : nterms(nterms), rank_ndocs(rank_ndocs_), doc_ids(rank_ndocs_, ub_rank_ntriples)
    {
        assert(rank_ndocs > 0);
        assert(global_ndocs >= rank_ndocs);

        collection_stats.ndocs  = global_ndocs;
        collection_stats.nterms = nterms;

        static_doc_stats.resize(rank_ndocs);
        // doc_stats.resize(rank_ndocs);
    }

    void load()
    {
        std::string prefix{CachePrefix};

        /* Load static_doc_stats. */
        FILE *f = fopen((prefix + std::to_string(Env::rank) + ".doc_stats.bin").c_str(), "r");
        assert(f != NULL);
        fread(static_doc_stats.data(), static_doc_stats.size(), sizeof(StaticDocStats), f);
        fclose(f);

        std::string s;
        size_t      r;
        struct stat st;

        /* TODO: Load terms mapping! */
        s = (prefix + std::to_string(Env::rank) + ".terms.bin");
        f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        terms.resize(st.st_size / sizeof(Term));
        assert(st.st_size % sizeof(Term) == 0);
        fread(terms.data(), terms.size() * sizeof(Term), 1, f);
        fclose(f);

        // std::vector<uint32_t> buckets(32);
        // for (auto &t : terms)
        // {
        //     uint32_t b = std::log(t.local_df) / std::log(2);
        //     buckets[b]++;
        // }

        // size_t sum = 0;
        // for (int b = 31; b >= 0; b--)
        // {
        //     sum += buckets[b];
        //     LOG.info("Bucket >= %u:    %u\n", 1 << b, sum);
        // }

        // Env::exit(0);

        /* Load collection_stats. */
        s = (prefix + std::to_string(Env::rank) + ".collection_stats.bin");
        f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        r = fread(&collection_stats, sizeof(collection_stats), 1, f);
        assert(r == 1);
        fclose(f);

        doc_ids.load();

        // return true;
    }

    void save()
    {
        std::string prefix{CachePrefix};
        mkdir(prefix.c_str(), 777);

        /* Save static_doc_stats. */
        FILE *f = fopen((prefix + std::to_string(Env::rank) + ".doc_stats.bin").c_str(), "w");
        assert(f != NULL);
        size_t r = fwrite(static_doc_stats.data(), static_doc_stats.size(), sizeof(StaticDocStats), f);
        // assert(r == 1);
        fclose(f);

        /* TODO: SAVE terms mapping! */
        f = fopen((prefix + std::to_string(Env::rank) + ".terms.bin").c_str(), "w");
        assert(f != NULL);
        r = fwrite(terms.data(), terms.size() * sizeof(Term), 1, f);
        assert(r == 1);
        fclose(f);

        /* Save collection_stats. */
        f = fopen((prefix + std::to_string(Env::rank) + ".collection_stats.bin").c_str(), "w");
        assert(f != NULL);
        r = fwrite(&collection_stats, sizeof(collection_stats), 1, f);
        assert(r == 1);
        fclose(f);

        doc_ids.save();
    }

    uint32_t rank_nshards()
    {
        return ((rank_ndocs - 1) >> SHARD_RADIX) + 1;
    }

    void add_triple(WTriple &triple)
    {
        collection_stats.ntokens += triple.weight;
        static_doc_stats[triple.doc].degree++;
        static_doc_stats[triple.doc].len += triple.weight;
    }

    void add_term(uint32_t term, const std::vector<WTriple> &triples)
    {
        assert(term < nterms);

        uint64_t local_cf = 0;
        for (auto &t : triples) local_cf += t.weight;
        assert(local_cf < (1lu << 32));

        terms.push_back({0, 0, uint32_t(triples.size()), uint32_t(local_cf), term});

        doc_ids.add_term(triples);
    }

    void finalize(bool recreate)
    {
        if (recreate)
        {
            doc_ids.finalize();
            save();
        }
        else
        {
            load();
        }

        rank_terms.resize(nterms);
        term_sparse2dense.reserve(terms.size() + 100);
        for (uint32_t i = 0; i < terms.size(); i++)
        {
            uint32_t term    = terms[i].term_idx;
            rank_terms[term] = true;
            assert(term_sparse2dense.find(term) == term_sparse2dense.end());
            term_sparse2dense[term] = i;
        }

        LOG.info("terms.size() = %u\n", terms.size());

        /* Finalize the local term statistics. */
        {
            size_t                quantum = 10 * 1000 * 1000;
            std::vector<uint32_t> dfs(quantum);
            std::vector<uint32_t> cfs(quantum);

            for (uint32_t offset = 0; offset < nterms; offset += quantum)
            {
                uint32_t endpos = std::min(offset + quantum, size_t(nterms));

                uint32_t total_across_nodes = 0;
                for (uint32_t i = offset; i < endpos; i++)
                {
                    dfs[i - offset] = 0;
                    cfs[i - offset] = 0;

                    if (rank_terms[i])
                    {
                        uint32_t idx    = term_sparse2dense[i];
                        dfs[i - offset] = terms[idx].local_df;
                        cfs[i - offset] = terms[idx].local_cf;

                        total_across_nodes += 1;
                    }
                }

                MPI_Allreduce(MPI_IN_PLACE, &total_across_nodes, 1, MPI_UNSIGNED, MPI_SUM, Env::MPI_WORLD);

                if (total_across_nodes > 0)
                {
                    MPI_Allreduce(MPI_IN_PLACE, dfs.data(), dfs.size(), MPI_UNSIGNED, MPI_SUM, Env::MPI_WORLD);
                    MPI_Allreduce(MPI_IN_PLACE, cfs.data(), cfs.size(), MPI_UNSIGNED, MPI_SUM, Env::MPI_WORLD);
                }

                for (uint32_t i = offset; i < endpos; i++)
                {
                    if (rank_terms[i])
                    {
                        user_term_stats.push_back({});
                        user_term_stats.back().df = dfs[i - offset];
                        user_term_stats.back().cf = cfs[i - offset];
                    }
                }
            }
        }

        /* Finalize the collection statistics. */
        {
            MPI_Allreduce(MPI_IN_PLACE, &collection_stats.ntokens, 1, MPI_UNSIGNED_LONG, MPI_SUM, Env::MPI_WORLD);
            collection_stats.avg_doclen = float(collection_stats.ntokens) / float(collection_stats.ndocs);
        }

        // /* Finalize the local doc stats. */
        // for (uint32_t i = 0; i < static_doc_stats.size(); i++)
        // {
        //     auto &d1  = static_doc_stats[i];
        //     auto &d2  = doc_stats[i];
        //     d2.degree = d1.degree;
        //     d2.len    = d1.len;

        //     d2.normalized_len = float(d2.len) / collection_stats.avg_doclen;
        //     d2.lmdir          = std::log(2000.0 / (d2.len + 2000.0));
        //     d2.spl            = std::log(1 + 7.5 / d2.normalized_len);
        //     d2.pl2            = std::log(1.0 + 19.5 / d2.normalized_len) / std::log(2.0);
        // }

        LOG.info("Statistics finalized for %lu terms. Computing Impacts ...\n", terms.size());
    }
};
