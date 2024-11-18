#pragma once

#include <math.h>
#include <cassert>
#include <vector>
#include "api/bm25.h"
#include "collection/stats.h"
#include "folly/EliasFano.h"
#include "structures/fixed_vector.h"
#include "tsl/sparse_map.h"
#include "utils/common.h"
#include "utils/dist_timer.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

// ds2i::optpfor_block; ds2i::varint_G8IU_block; ds2i::interpolative_block;
// using BlockCodec = ds2i::varint_G8IU_block;
// typedef ds2i::block_freq_index<BlockCodec> collection_type;

template <typename Value, size_t kSkipQuantum = 0, size_t kForwardQuantum = 0>
struct Compression
{
    using Encoder = folly::compression::EliasFanoEncoderV2<Value, Value, kSkipQuantum, kForwardQuantum>;
    using List    = typename Encoder::CompressedList;
    using Layout  = typename Encoder::Layout;
    using Reader  = folly::compression::EliasFanoReader<Encoder, folly::compression::instructions::Default, true>;

    static size_t encode(const std::vector<Value>& v, uint8_t* buf)
    {
        return Encoder::encode_into_buffer(v.begin(), v.end(), v.back(), buf);
    }

    static size_t nbytes(size_t bound, size_t size)
    {
        return Layout::fromUpperBoundAndSize(bound, size).bytes();
    }

    static List list(size_t bound, size_t size, uint8_t* buf)
    {
        return Layout::fromUpperBoundAndSize(bound, size).fromList(buf);
    }

    static Reader reader(size_t bound, size_t size, uint8_t* buf)
    {
        return Reader(list(bound, size, buf));
    }
};

using DocCompression    = Compression<uint32_t>;
using FreqCompression   = Compression<uint32_t, 0, 256>;
using OffsetCompression = Compression<uint64_t, 0, 4096>;

struct FreqIterator
{
    FreqCompression::Reader reader;

    FreqIterator(const FreqCompression::Encoder::CompressedList& x) : reader(x)
    {
        reader.next();
    }

    void skipToPosition(uint32_t pos)
    {
        uint32_t curr = reader.position();
        if (irg_likely(curr < pos)) reader.skip(pos - curr);
    }

    uint32_t advance_and_read()
    {
        uint32_t prev = reader.value();
        reader.next();

        return 1u + reader.value() - prev;
    }
};

class DocIDs
{
    using WTriple = Triple<EdgeWeight>;

   public:
    const DocID rank_ndocs;

    uint8_t* docs          = nullptr;
    uint8_t* freqs         = nullptr;
    uint8_t* docs_offsets  = nullptr;
    uint8_t* freqs_offsets = nullptr;

    DocIDs(const DocID rank_ndocs, uint64_t ub_rank_ntriples) : rank_ndocs(rank_ndocs)
    {
        tmp_doc_offsets.push_back(0);
        tmp_freq_offsets.push_back(0);

        docs = static_cast<uint8_t*>(malloc(uint64_t(5) * ub_rank_ntriples / 4));  // FIXME: !!
        if (not docs) Env::exit(1);

        freqs = static_cast<uint8_t*>(malloc(uint64_t(3) * ub_rank_ntriples / 4));
        if (not freqs) Env::exit(1);
    }

    void add_term(const std::vector<WTriple>& triples)
    {
        size_t nbytes;

        /* Compress Doc IDs. */
        tmp_docs.clear();
        for (auto& triple : triples) tmp_docs.push_back(triple.doc);
        tmp_docs.push_back(rank_ndocs);

        nbytes = DocCompression::encode(tmp_docs, docs + tmp_doc_offsets.back());
        tmp_doc_offsets.push_back(tmp_doc_offsets.back() + nbytes);

        /* Compress Doc Freqs. */
        tmp_docs.clear();
        tmp_docs.push_back(0);
        for (auto& triple : triples) tmp_docs.push_back(tmp_docs.back() + triple.weight - 1);

        nbytes = FreqCompression::encode(tmp_docs, freqs + tmp_freq_offsets.back());
        tmp_freq_offsets.push_back(tmp_freq_offsets.back() + nbytes);
    }

    void finalize()
    {
        nterms       = tmp_doc_offsets.size();
        docs_nbytes  = tmp_doc_offsets.back();
        freqs_nbytes = tmp_freq_offsets.back();

        /* Finalize Doc IDs. */
        docs_offsets = static_cast<uint8_t*>(malloc(uint64_t(4) * tmp_doc_offsets.size()));
        if (not docs_offsets) Env::exit(1);

        docs_offsets_nbytes = OffsetCompression::encode(tmp_doc_offsets, docs_offsets);
        tmp_doc_offsets.clear();

        /* Finalize Doc Freqs. */
        freqs_offsets = static_cast<uint8_t*>(malloc(uint64_t(4) * tmp_freq_offsets.size()));
        if (not freqs_offsets) Env::exit(1);

        freqs_offsets_nbytes = OffsetCompression::encode(tmp_freq_offsets, freqs_offsets);
        tmp_freq_offsets.clear();

        // LOG.info("Document IDs use up %.2f GiBs\n", docs_nbytes / 1024.0 / 1024.0 / 1024.0);
        // LOG.info("Term Postings Offsets use up %.2f GiBs\n", offsets_nbytes / 1024.0 / 1024.0 / 1024.0);
    }

    void load()
    {
        struct stat st;
        std::string s;
        FILE*       f;
        size_t      r;
        std::string prefix{CachePrefix};

        s = prefix + std::to_string(Env::rank) + ".docs.bin";
        f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        docs_nbytes = st.st_size;

        LOG.info("NEW docs_nbyts = %lu\n", docs_nbytes);

        r = fread(docs, docs_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        s = (prefix + std::to_string(Env::rank) + ".freqs.bin");
        f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        freqs_nbytes = st.st_size;
        r            = fread(freqs, freqs_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        s = (prefix + std::to_string(Env::rank) + ".docs_offsets.bin");
        f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        docs_offsets_nbytes = st.st_size;
        docs_offsets        = static_cast<uint8_t*>(malloc(docs_offsets_nbytes + 7));
        r                   = fread(docs_offsets, docs_offsets_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        s = (prefix + std::to_string(Env::rank) + ".freqs_offsets.bin");
        f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        freqs_offsets_nbytes = st.st_size;

        LOG.info("freqs_offsets_nbytes = %lu\n", freqs_offsets_nbytes);

        freqs_offsets = static_cast<uint8_t*>(malloc(freqs_offsets_nbytes + 7));
        r             = fread(freqs_offsets, freqs_offsets_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        s = (prefix + std::to_string(Env::rank) + ".docs_nterms.bin");
        f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        r = fread(&nterms, sizeof(nterms), 1, f);
        assert(r == 1);
        fclose(f);
    }

    /* NOTE: Must be preceded by call to finalize() to be useful! */
    void save()
    {
        std::string prefix{CachePrefix};

        LOG.info("OLD docs_nbyts = %lu\n", docs_nbytes);

        FILE* f = fopen((prefix + std::to_string(Env::rank) + ".docs.bin").c_str(), "w");
        assert(f != NULL);
        size_t r = fwrite(docs, docs_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        f = fopen((prefix + std::to_string(Env::rank) + ".freqs.bin").c_str(), "w");
        assert(f != NULL);
        r = fwrite(freqs, freqs_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        f = fopen((prefix + std::to_string(Env::rank) + ".docs_offsets.bin").c_str(), "w");
        assert(f != NULL);
        r = fwrite(docs_offsets, docs_offsets_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        f = fopen((prefix + std::to_string(Env::rank) + ".freqs_offsets.bin").c_str(), "w");
        assert(f != NULL);
        r = fwrite(freqs_offsets, freqs_offsets_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        f = fopen((prefix + std::to_string(Env::rank) + ".docs_nterms.bin").c_str(), "w");
        assert(f != NULL);
        r = fwrite(&nterms, sizeof(nterms), 1, f);
        assert(r == 1);
        fclose(f);
    }

   public:
    OffsetCompression::Reader get_docs_offsets_reader() const
    {
        return OffsetCompression::reader(docs_nbytes, nterms, docs_offsets);
    }

    OffsetCompression::Reader get_freqs_offsets_reader() const
    {
        return OffsetCompression::reader(freqs_nbytes, nterms, freqs_offsets);
    }

   private:
    size_t docs_offsets_nbytes  = 0;
    size_t freqs_offsets_nbytes = 0;

    size_t nterms       = 0;
    size_t docs_nbytes  = 0;
    size_t freqs_nbytes = 0;

    std::vector<uint32_t> tmp_docs;
    std::vector<size_t>   tmp_doc_offsets;
    std::vector<size_t>   tmp_freq_offsets;
};

class DocIDsReader
{
   public:
    DocIDsReader(const DocIDs& D)
        : docs(D.docs),
          freqs(D.freqs),
          rank_ndocs(D.rank_ndocs),
          docs_offsets_reader(D.get_docs_offsets_reader()),
          freqs_offsets_reader(D.get_freqs_offsets_reader())
    {
    }

    DocCompression::Reader get_docs(uint32_t compressed_term_idx, uint32_t df)
    {
        docs_offsets_reader.reset();
        docs_offsets_reader.next();

        assert(compressed_term_idx < docs_offsets_reader.size());

        if (compressed_term_idx > docs_offsets_reader.position())
            docs_offsets_reader.skip(compressed_term_idx - docs_offsets_reader.position());

        assert(docs_offsets_reader.valid());
        assert(docs_offsets_reader.position() == compressed_term_idx);

        return DocCompression::reader(rank_ndocs, df + 1, docs + docs_offsets_reader.value());
    }

    FreqIterator get_freqs(uint32_t compressed_term_idx, uint32_t df, uint64_t cf)
    {
        freqs_offsets_reader.reset();
        freqs_offsets_reader.next();

        assert(compressed_term_idx < freqs_offsets_reader.size());

        if (compressed_term_idx > freqs_offsets_reader.position())
            freqs_offsets_reader.skip(compressed_term_idx - freqs_offsets_reader.position());

        assert(freqs_offsets_reader.valid());
        assert(freqs_offsets_reader.position() == compressed_term_idx);

        return FreqIterator(FreqCompression::list(cf - df, df + 1, freqs + freqs_offsets_reader.value()));
    }

   private:
    uint8_t*    docs  = nullptr;
    uint8_t*    freqs = nullptr;
    const DocID rank_ndocs;

    OffsetCompression::Reader docs_offsets_reader;
    OffsetCompression::Reader freqs_offsets_reader;
};

class EFWrapper
{
   public:
    DocCompression::Reader reader;
    FreqIterator           fiterator;

    EFWrapper(DocIDsReader& doc_ids, uint32_t idx, uint32_t local_df, uint32_t local_cf)
        : reader(doc_ids.get_docs(idx, local_df)), fiterator(doc_ids.get_freqs(idx, local_df, local_cf))
    {
        reader.next();
    }

    uint32_t docid()
    {
        return reader.value();
    }

    uint32_t freq()
    {
        uint32_t pos  = reader.position();
        uint32_t curr = fiterator.reader.position();
        if (curr < pos) fiterator.reader.skip(pos - curr);

        uint32_t prev = fiterator.reader.value();
        fiterator.reader.next();

        assert(fiterator.reader.value() >= prev);

        return 1u + fiterator.reader.value() - prev;
    }

    void next()
    {
        reader.next();
    }

    void next_geq(uint32_t doc)
    {
        reader.skipTo(doc);
    }
};
