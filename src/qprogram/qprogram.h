#pragma once

#include <queue>
#include <string>
#include <unordered_map>

#include "structures/bitvector.h"
#include "utils/common.h"
#include "utils/dist_timer.h"

class QueryProgram
{
   public:
    constexpr static uint32_t MaxQueryNumTerms = 256;
    uint32_t                  local_TopK       = 10;
    uint32_t                  TopK             = 10;

    using UserTermStats = ChosenTerm2Doc::TermStats;

#if PRECOMPUTE_SEND
    using Message = uint32_t;
    using Score   = uint32_t;
#else
    using Message = ChosenTerm2Doc::Message;
    using Score   = ChosenTerm2Doc::Score;
#endif

    struct FinalScore
    {
        Score    score;
        uint32_t id         = 1000000;
        uint32_t local_rank = 100000;
        char     ClueWebID[26 + 1];

        FinalScore() : score()
        {
        }

        FinalScore(Score s, uint32_t id, uint32_t r, const char *str) : score(s), id(id), local_rank(r)
        {
            strncpy(ClueWebID, str, 26 + 1);
        }

        bool operator<=(const FinalScore &other) const
        {
            return score <= other.score;
        }
    };

    Collection &                   C;
    Graph<EdgeWeight> &            G;
    RectangularMatrix<EdgeWeight> &A;
    ChosenTerm2Doc                 model;

    struct DAAT_Term;
    struct MaxScore_Term;
    struct BMW_Term;
    struct LocalBMW_Term;
    struct DocScore;

    QueryProgram(Collection &C) : C(C), G(C.G), A(G.A), model(A.collection_stats)
    {
        daat_terms.reserve(MaxQueryNumTerms);
        qterms.reserve(MaxQueryNumTerms);

        essentials.reserve(MaxQueryNumTerms);
        nonessentials.reserve(MaxQueryNumTerms);

        ub_pfxsum.reserve(MaxQueryNumTerms);
        ub_pfxsum1.reserve(MaxQueryNumTerms);
        ub_pfxsum2.reserve(MaxQueryNumTerms);

        topk.reserve(TopK);
        all_topk.reserve(TopK * Env::nranks);
    }

    void terms2docs(const Query &, bool);
    bool distributed_topk(bool);

    //    private:
    uint32_t query2terms(const Query &);

    bool topk_insert(uint32_t doc, Score score);
    void process_range(const Query &query, uint32_t range, uint32_t offset, uint32_t endpos);
    uint32_t process_doc(const Query &query, uint32_t range, uint32_t doc);
    uint32_t skip_to_next_live_block(const Query &query, uint32_t check_up_to_list, uint32_t optionals,
                                     uint32_t endpos);

    Score    global_doc_maxscore;
    Score threshold;

    std::vector<DAAT_Term>        daat_terms;

    std::vector<MaxScore_Term> maxscore_terms;
    std::vector<BMW_Term>      bmw_terms;
    std::vector<LocalBMW_Term> lbmw_terms;
    std::vector<uint32_t>      qterms;
    std::vector<uint8_t>       essentials;
    std::vector<uint8_t>       nonessentials;
    std::vector<Score>         ub_pfxsum, ub_pfxsum1, ub_pfxsum2;

    std::vector<BMW_Term *>      ordered_enums;
    std::vector<LocalBMW_Term *> lbmw_enums;
    std::vector<MaxScore_Term *> wand_enums;
    std::vector<BMW_Term *>      bmw_enums;

    Score doc_maxscore;
    std::priority_queue<DocScore, std::vector<DocScore>, std::greater<DocScore>> pq;

    std::vector<FinalScore> topk, all_topk;

    /* Profiling */
    size_t doc_evals        = 0;
    size_t send_calls       = 0;
    size_t pivot_selections = 0;
    size_t topk_insertions  = 0;
    size_t next_calls       = 0;

    uint32_t optionals = 0;

    std::vector<std::pair<uint32_t, uint32_t>> final_intervals;
    std::vector<float>              bounds;
    std::vector<std::pair<uint32_t, uint32_t>> intervals;
    std::vector<uint32_t> positions;
};

/* Implementation. */
#include "qprogram/activity.hpp"

#ifdef DAAT
#include "qprogram/terms2docs.full.hpp"
#endif

#ifdef BMW
#include "qprogram/wand/terms2docs.bmw.hpp"
#endif

#ifdef LBMW
#include "qprogram/wand/terms2docs.lbmw.hpp"
#endif

#ifdef VBMW
#include "qprogram/wand/terms2docs.bmw.hpp"
#endif

#ifdef DBMW
#include "qprogram/wand/terms2docs.dbmw.hpp"
#endif

#ifdef BMM
#include "qprogram/maxscore/terms2docs.bmm.hpp"
#endif

#ifdef LBMM
#include "qprogram/maxscore/terms2docs.lbmm.hpp"
#endif

#ifdef IBMM
#include "qprogram/maxscore/terms2docs.ibmm.hpp"
#endif

#ifdef DBMM
#include "qprogram/maxscore/terms2docs.dbmm.hpp"
#endif

#ifdef MaxScore
#include "qprogram/maxscore/terms2docs.maxscore.hpp"
#endif

#ifdef WAND
#include "qprogram/wand/terms2docs.wand.hpp"
#endif

#ifdef LazyMaxScore
#include "qprogram/maxscore/terms2docs.maxscore.lazy.hpp"
#endif
