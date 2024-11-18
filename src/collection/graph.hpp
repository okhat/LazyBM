#include <sys/stat.h>
#include <sys/types.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "graph.h"
#include "utils/common.h"
#include "utils/dist_timer.h"

template <class Weight>
Graph<Weight>::Graph(EdgeListArgs args, const std::vector<bool> &needed_terms)
    : args(args),
      ndocs(args.ndocs),
      rank_ndocs(args.rank_ndocs),
      nterms(args.nterms + 1), /* for one-based numbering, having a dummy term 0 */
      doc_shard_offsets(args.doc_shard_offsets),
      needed_terms(needed_terms),
      A(ndocs, rank_ndocs, nterms, open_edge_lists())
{
    if (not CACHE_REUSE) read_edge_lists();

    for (auto f : files) fclose(f);

    A.finalize(not CACHE_REUSE);
}

template <class Weight>
uint64_t Graph<Weight>::open_edge_lists()
{
    std::string prefix{GlobalPrefix};
    std::string suffix{".edges.bin"};

    for (uint32_t i = Env::rank; i < MaxDiskShard; i += Env::nranks)
    {
        uint32_t    s = all_shards[i];
        std::string path{prefix + std::to_string(s) + suffix};

        files.push_back(fopen(path.c_str(), "r"));

        if (not files.back())
        {
            LOG.info("Unable to open input file\n");
            Env::exit(1);
        }

        struct stat st;
        if (stat(path.c_str(), &st) != 0)
        {
            LOG.info("stat() failure\n");
            Env::exit(1);
        }

        nbytes += st.st_size;
        nedges += st.st_size / sizeof(WTriple);
        assert(nedges * sizeof(WTriple) == nbytes);
    }

    size_t ub_rank_ntriples = std::min(nedges, needed_terms.size() * rank_ndocs);

    LOG.info("Files appear to have %lu edges (%u-byte weights).\n", nedges, sizeof(WTriple) - sizeof(Triple<Empty>));
    LOG.info("Relevant terms can have at most %lu edges.\n", ub_rank_ntriples);

    return ub_rank_ntriples;
}

template <class Weight>
void Graph<Weight>::read_edge_lists()
{
    LOG.info("Reading input files ...\n");

    std::vector<WTriple> buffer;

    uint64_t             offset      = 0;
    uint64_t             last_offset = 0;
    std::vector<WTriple> term_triples;

    uint32_t docID, freqValue;

    // FILE *index_docs  = fopen("/datasets/okhattab/indexes/CatB/URL_index.docs", "r");
    // FILE *index_freqs = fopen("/datasets/okhattab/indexes/CatB/URL_index.freqs", "r");

    FILE *index_docs  = fopen("/datasets/okhattab/indexes/CatB/index.docs", "r");
    FILE *index_freqs = fopen("/datasets/okhattab/indexes/CatB/index.freqs", "r");

    fread(&docID, sizeof(uint32_t), 1, index_docs);
    assert(docID == 1);
    fread(&docID, sizeof(uint32_t), 1, index_docs);
    LOG.info("%u, %u\n", docID, ndocs);
    assert(docID == ndocs);

    uint32_t above15 = 0, above17 = 0, above18 = 0;
    uint64_t num_blocks_with_64_postings = 0;

    for (uint32_t termID = 0; termID < nterms; termID++)
    {
        uint32_t termDF, termDF2;
        size_t   df1_read = fread(&termDF, sizeof(uint32_t), 1, index_docs);
        size_t   df2_read = fread(&termDF2, sizeof(uint32_t), 1, index_freqs);
        assert(df1_read == 1);
        assert(df2_read == 1);
        assert(termDF == termDF2);

        uint64_t termCF = 0;

        above15 += termDF >= (1 << 15);
        above17 += termDF >= (1 << 17);
        above18 += termDF >= (1 << 18);

        num_blocks_with_64_postings += std::ceil(float(termDF) / 64.0);

        // while (fread(&docID, sizeof(uint32_t), 1, index_docs) and fread(&freqValue, sizeof(uint32_t), 1,
        // index_freqs))
        for (uint32_t i = 0; i < termDF; i++)
        {
            size_t docid_read = fread(&docID, sizeof(uint32_t), 1, index_docs);
            size_t freq_read  = fread(&freqValue, sizeof(uint32_t), 1, index_freqs);
            assert(docid_read == 1);
            assert(freq_read == 1);

            termCF += freqValue;
            buffer.push_back({termID, docID, freqValue});
            assert(freqValue > 0);
            assert(termCF < (1u << 31));
            assert(docID < ndocs);
        }

        if (offset - last_offset > 10 * 1000 * 1000)
        {
            LOG.info("| %8lu Terms (%f%%) -- buffer.capacity() = %.3f GiB \n", offset, termID / double(nterms) * 100.0,
                     buffer.capacity() * sizeof(WTriple) / 1024.0 / 1024.0 / 1024.0);
            last_offset = offset;

            LOG.info("###> [incomplete] num_blocks_with_64_postings = %u\n", num_blocks_with_64_postings);
        }

        auto cmp = [](const auto &t1, const auto &t2) {
            return t1.term < t2.term or (t1.term == t2.term and t1.doc < t2.doc);
        };
        __gnu_parallel::sort(buffer.begin(), buffer.end(), cmp);

        for (uint32_t i = 0; i < buffer.size(); i++)
        {
#if FILTER_TERMS
            bool need_this_term = needed_terms[buffer[i].term];
#else
            bool need_this_term = true;
#endif

            offset++;
            A.add_triple(buffer[i]);
            if (need_this_term) term_triples.push_back(buffer[i]);

            if ((i == buffer.size() - 1) or (buffer[i].term != buffer[i + 1].term))
            {
                assert(i == buffer.size() - 1);
                if (need_this_term) A.add_term(term_triples.back().term, term_triples);
                term_triples.clear();
            }
        }

        buffer.clear();
    }

    LOG.info<false, false>("[%d]", Env::rank);
    Env::barrier();
    LOG.info<true, false>("\n");

    LOG.info("###> above15 = %u, above17 = %u, above18 = %u\n", above15, above17, above18);
    LOG.info("###> num_blocks_with_64_postings = %u\n", num_blocks_with_64_postings);
}
