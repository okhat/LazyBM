#pragma once

#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include "graph.h"
#include "tsl/sparse_set.h"

class Collection
{
   public:
    std::vector<std::string> term2str;
    uint32_t                 global_nterms;

    /* String labels. */
    std::vector<uint32_t> doc_shard_offsets;
    std::vector<char>     doc_labels;
    std::vector<uint32_t> doc_labels_offsets;

   private:
    std::vector<float> doc_pageranks;

   public:
    std::hash<std::string>             str_hash;
    std::vector<std::vector<char>>     term_labels;
    std::vector<std::vector<uint32_t>> term_idxs;

    const uint32_t            nbuckets = 6000 * 1000;
    uint32_t                  rank_ndocs;
    std::vector<bool>         needed_terms;
    Graph<EdgeWeight>         G;
    static constexpr uint32_t UNK = UINT_MAX;

   public:
    Collection(EdgeListArgs args, const std::string &queries_path, uint32_t nqueries)
        : global_nterms(args.nterms),
          rank_ndocs(construct_idx2doc(args.ndocs)),
          needed_terms(setup_term2idx(queries_path, nqueries)),
          G({args.path, args.ndocs, args.nterms, rank_ndocs, doc_shard_offsets}, needed_terms)
    {
        LOG.info("(7)\n");

        construct_term2idx();
        LOG.info("(8)\n");

        LOG.info("Waiting for all nodes at barrier ...\n");
        Env::barrier();
    }

    char *doc_idx2label(uint32_t doc /* Local doc ID */)
    {
        if (doc >= rank_ndocs) return nullptr;
        return doc_labels.data() + doc_labels_offsets[doc];
    }

    float doc_idx2pr(uint32_t doc /* Local doc ID */)
    {
        if (doc >= doc_pageranks.size()) return 0.0;
        return doc_pageranks[doc];
    }

    uint32_t term_label2idx(const std::string &term) const
    {
        size_t      h      = term_bucket(term);
        const auto &idxs   = term_idxs[h];
        const auto &labels = term_labels[h].data();
        const auto &str    = term.c_str();

        size_t offset = 0;
        for (size_t i = 0; i < idxs.size(); i++)
        {
            if (strcmp(str, labels + offset) == 0) return idxs[i];
            offset += strlen(labels + offset) + 1;
        }

        return UNK;
    }

   private:
    size_t term_bucket(const std::string &term) const
    {
        size_t h = str_hash(term) % (nbuckets >> 1);
        return h + (term.size() > 15 ? (nbuckets >> 1) : 0);
    }

    uint32_t construct_idx2doc(uint32_t arg_ndocs)
    {
        /* NOTE: Just to handle 1-indexing! */
        // doc_labels_offsets.push_back(0);

        doc_shard_offsets.push_back(0);

        if (true)
        {
            std::ifstream documentf("/datasets/okhattab/indexes/CatB/cw12b.URL_ordered.documents");  // for effectiveness
            // std::ifstream documentf("/datasets/okhattab/indexes/CatB/cw12b.documents");

            std::string line;
            while (std::getline(documentf, line))
            {
                std::istringstream iss(line);
                std::string        label;
                iss >> label;

                doc_labels_offsets.push_back(doc_labels.size());
                doc_labels.insert(doc_labels.end(), label.data(), label.data() + label.length() + 1);
            }
        }
        else
        {
            for (uint32_t d = 0; d < arg_ndocs; d++)
            {
                std::string label = "clueweb12-0805wb-68-23447";
                doc_labels_offsets.push_back(doc_labels.size());
                doc_labels.insert(doc_labels.end(), label.data(), label.data() + label.length() + 1);
            }
        }

        doc_labels.shrink_to_fit();
        LOG.info("Doc_labels for %u docs use %.2f GiBs\n", doc_labels_offsets.size(),
                 doc_labels.size() / 1024.0 / 1024.0 / 1024.0);

        return arg_ndocs;
    }

    std::vector<bool> setup_term2idx(const std::string &queries_path, uint32_t nqueries)
    {
        std::vector<bool> output(global_nterms);

        // output[0] = true;

        tsl::sparse_set<std::string> inserted_strings;
        size_t                       num_needed_terms = 0;

#if CACHE_REUSE
        struct stat st;
        std::string prefix{CachePrefix};
        std::string s = (prefix + std::to_string(Env::rank) + ".term_idxs.bin");
        FILE *      f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        size_t                idxs_nbytes = st.st_size;
        std::vector<uint32_t> idxs(idxs_nbytes / sizeof(uint32_t));
        size_t                r = fread(idxs.data(), idxs_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        std::ifstream g(prefix + std::to_string(Env::rank) + ".term_strings.txt");
        // term2str.resize(38546221);

        size_t pos = 0;
        for (std::string line; std::getline(g, line);)
        {
            std::string term;

            std::istringstream iss(line);
            iss >> term;
            uint32_t idx = idxs[pos++];

            output[idx] = true;
            // term2str[idx] = term;
        }

#else
        std::ifstream                   queries_file(queries_path);
        std::unordered_set<std::string> queries_terms;

        std::string text, term;
        for (auto q = 0u; q < nqueries; q++)
        {
            std::getline(queries_file, text);

            std::istringstream iss(text);
            while (iss >> term)
            {
                if (term[0] == '-') term.erase(term.begin());

                queries_terms.insert(term);
            }
        }

        std::vector<char>     output_buffer;
        std::vector<uint32_t> output_idxs;

        std::ifstream termf("/datasets/okhattab/indexes/CatB/cw12b.terms");

        uint32_t idx = 0;
        for (std::string line; std::getline(termf, line);)
        {
            std::string term;

            if (line[0] != ' ')
            {
                std::istringstream iss(line);
                iss >> term;

#if FILTER_TERMS
                if (idx == 0 or (queries_terms.find(term) != queries_terms.end()))
#endif
                {
                    // if (not output[idx])
                    if (inserted_strings.find(term) == inserted_strings.end())
                    {
                        inserted_strings.insert(term);

                        num_needed_terms++;
                        output[idx] = true;
                        output_buffer.insert(output_buffer.end(), term.data(), term.data() + term.length() + 1);
                        output_buffer.back() = '\n';
                        output_idxs.push_back(idx);
                    }
                }
            }

            idx++;
        }

        FILE * f;
        size_t r;

        std::string prefix2{CachePrefix};
        f = fopen((prefix2 + std::to_string(Env::rank) + ".term_strings.txt").c_str(), "w");
        assert(f != NULL);
        r = fwrite(output_buffer.data(), output_buffer.size(), 1, f);
        assert(r == 1);
        fclose(f);

        f = fopen((prefix2 + std::to_string(Env::rank) + ".term_idxs.bin").c_str(), "w");
        assert(f != NULL);
        r = fwrite(output_idxs.data(), output_idxs.size() * sizeof(uint32_t), 1, f);
        assert(r == 1);
        fclose(f);

#endif

        LOG.info("Prepared list of %lu needed terms!\n", num_needed_terms);
        return output;
    }

    void construct_term2idx()
    {
        term_labels.resize(nbuckets);
        term_idxs.resize(nbuckets);

        struct stat st;
        std::string prefix{CachePrefix};
        std::string s = (prefix + std::to_string(Env::rank) + ".term_idxs.bin");
        FILE *      f = fopen(s.c_str(), "r");
        assert(f != NULL);
        stat(s.c_str(), &st);
        size_t                idxs_nbytes = st.st_size;
        std::vector<uint32_t> idxs(idxs_nbytes / sizeof(uint32_t));
        size_t                r = fread(idxs.data(), idxs_nbytes, 1, f);
        assert(r == 1);
        fclose(f);

        std::ifstream g(prefix + std::to_string(Env::rank) + ".term_strings.txt");

        size_t pos = 0;
        for (std::string line; std::getline(g, line);)
        {
            std::string term;

            std::istringstream iss(line);
            iss >> term;
            uint32_t idx = idxs[pos++];

            if (G.A.rank_terms[idx])
            {
                size_t h = term_bucket(term);
                term_labels[h].insert(term_labels[h].end(), term.data(), term.data() + term.length() + 1);
                term_idxs[h].push_back(G.A.term_sparse2dense[idx]);
            }
        }

        G.A.term_sparse2dense.clear();

        /* Info. */
        std::vector<size_t> lengths, sizes;
        for (uint32_t i = 0; i < term_labels.size() / 2; i++) lengths.push_back(term_labels[i].size());
        for (uint32_t i = 0; i < term_idxs.size() / 2; i++) sizes.push_back(term_idxs[i].size());
        std::sort(lengths.begin(), lengths.end());
        std::sort(sizes.begin(), sizes.end());

        LOG.info("shortest = %lu, median = %lu, longest = %lu\n", lengths.front(), lengths[lengths.size() / 2],
                 lengths.back());
        LOG.info("shortest = %lu, median = %lu, longest = %lu\n", sizes.front(), sizes[sizes.size() / 2], sizes.back());

        size_t nbytes = 0, nterms = 0;
        for (uint32_t i = 0; i < term_labels.size(); i++)
        {
            nterms += term_idxs[i].size();
            nbytes += term_labels[i].size() + sizeof(uint32_t) * term_idxs[i].size();
        }

        LOG.info("Term labels for %u terms use %.2f GiBs\n", nterms, nbytes / 1024.0 / 1024.0 / 1024.0);
    }
};
