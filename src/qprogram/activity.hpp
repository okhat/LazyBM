#pragma once

struct QueryProgram::DAAT_Term
{
    uint32_t               idx;
    uint32_t               query_idx;
    DocCompression::Reader reader;
    FreqIterator           fiterator;
    UserTermStats          idf;
};


struct QueryProgram::MaxScore_Term
{
    uint32_t idx;
    uint32_t query_idx;
    EFWrapper     reader;
    UserTermStats idf;
    Message       maxscore;
    uint32_t      maxscore_cache_range;
    Score         block_maxscore;
    uint32_t      local_df;
    float         z;
};

struct QueryProgram::BMW_Term
{
    uint32_t idx;
    uint32_t query_idx;
    EFWrapper     reader;
    UserTermStats idf;
    Message       maxscore;
    Message *     w;
    uint32_t *    wdoc;
    uint32_t      local_df;
    uint32_t      widx;
    uint32_t      maxscore_cache_range;
    Score         maxscore_cache;
    Score         block_maxscore;
    float         z;
};


struct QueryProgram::LocalBMW_Term
{
    uint32_t idx;
    uint32_t query_idx;
    EFWrapper reader;
    UserTermStats idf;
    Message       maxscore;
    Message *     w;
    uint32_t *    wdoc;
    uint32_t      local_df;
    uint32_t      widx;
    uint32_t      ub_range;
    Score         maxscore_cache;
    Score         block_maxscore;
    Message       ub;
};

struct QueryProgram::DocScore
{
    uint32_t doc;
    Score    score;

    friend bool operator>(const DocScore &a, const DocScore &other);
};

bool operator>(const QueryProgram::DocScore &a, const QueryProgram::DocScore &other)
{
    return not(a.score <= other.score);
}

bool QueryProgram::distributed_topk(bool restore)
{
    bool good_case = true;
    topk.clear();

    size_t pq_original_size = 0;
    while (not pq.empty())
    {
        // LOG.info("pq.top().doc = %u\n", pq.top().doc);
        pq_original_size++;
        topk.push_back(FinalScore(pq.top().score, pq.top().doc, pq.size(), C.doc_idx2label(pq.top().doc)));
        pq.pop();
    }

    std::reverse(topk.begin(), topk.end());
    topk.resize(local_TopK);

    return true; // NOTE: !! [May 23, 2020]

    all_topk.resize(Env::nranks * local_TopK);

    MPI_Gather(topk.data(), local_TopK * sizeof(FinalScore), MPI_BYTE, all_topk.data(), local_TopK * sizeof(FinalScore),
               MPI_BYTE, 0, Env::MPI_WORLD);

    if (Env::is_master)
    {
        std::partial_sort(all_topk.begin(), all_topk.begin() + TopK, all_topk.end(),
                          [](auto a, auto b) { return not(a <= b); });  // NOTE: Sorts by Score

        uint32_t max_depth = 0;
        for (uint32_t i = 0; i < TopK; i++)
        {
            max_depth = std::max(max_depth, all_topk[i].local_rank);
        }

        assert(all_topk[TopK - 1].ClueWebID);
        // LOG.info(" [[[%d]]] threshold = %f (%s) [as opposed to %f, max depth is %u]\n", restore,
        //          float(all_topk[TopK - 1].score), all_topk[TopK - 1].ClueWebID, float(threshold), max_depth);
        threshold = all_topk[TopK - 1].score;  // NOTE: Changes the threshold!

        if (max_depth == local_TopK and local_TopK < TopK)
        {
            // LOG.info("BAD CASE!\n");
            good_case = false;
        }
    }

    MPI_Bcast(&good_case, sizeof(bool), MPI_BYTE, 0, Env::MPI_WORLD);
    if (not good_case) MPI_Bcast(&threshold, sizeof(Score), MPI_BYTE, 0, Env::MPI_WORLD);

    if (not good_case)
    {
        for (uint32_t i = 0; i < pq_original_size; i++)
        {
            pq.push({topk[i].id, topk[i].score});
        }
    }

    return good_case;
}

bool QueryProgram::topk_insert(uint32_t doc, Score score)
{
    if (irg_unlikely(not(score <= threshold)))
    {
        if (irg_likely(pq.size() == local_TopK)) pq.pop();

        pq.push({doc, score});

        threshold = (pq.size() == local_TopK) ? pq.top().score : threshold;
        return true;
    }

    return false;
}