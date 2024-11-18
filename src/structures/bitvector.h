#ifndef BIT_VECTOR_H
#define BIT_VECTOR_H

#include <cassert>
#include <cstdint>
#include <cstring>

class BitVector
{
   protected:
    constexpr static uint32_t bitwidth = 32;

    constexpr static uint32_t lg_bitwidth = 5;

    constexpr static uint32_t bitwidth_mask = 0x1F;

   protected:
    uint32_t n;

    uint32_t* words;

    uint32_t* nnzs;

    uint32_t pos = 0;

    uint32_t cache = 0;

    bool owns_words = true;

   public:          /* Constructor(s) and Destructor(s). */
    BitVector() {}  // for FixedVector allocation

    BitVector(uint32_t n) : n(n), words(new uint32_t[buffer_nwords()]()), nnzs(words)
    {
        // LOG.info("BitVector(n)\n");
        *nnzs = 0;
        words++;
        rewind();
    }

    // Copy constructor: deep copy by default.
    BitVector(const BitVector& bv, bool deep = true)
        : n(bv.n), words(deep ? (new uint32_t[buffer_nwords()]) : bv.buffer()), nnzs(words)
    {
        // LOG.info("BitVector(bv)\n");
        owns_words = deep;
        words++;
        if (deep)
        {
            rewind();
            memcpy(buffer(), bv.buffer(), buffer_nbytes());
        }
    }

    virtual ~BitVector()
    {
        // LOG.info("~BitVector()\n");
        if (owns_words) delete[] buffer();
    }

    /* Non-assignable. */
    BitVector& operator=(BitVector const&) = delete;

   public: /* General Interface. */
    uint32_t size() const { return n; };

    uint32_t count() const { return *nnzs; };

    /*
     * Note that this completely delegates knowing and handling the original size to the user.
     * As intended, all the workings of the bitvector assume the new size until it is changed again.
     * For instance, using the copy constructor will produce a new bitvector of size n_ not old n.
     */
    void temporarily_resize(uint32_t n_)  // Requires that n_ is at most the _initial_ size.
    {
        assert(count() == 0);  // Requires a clear bit vector.
        rewind();
        untouch(n);  // Zero out the (old) n'th, sentinel bit.
        n     = n_;
        *nnzs = 0;
        rewind();  // Set the (new) n'th, sentinel bit.
    }

    void touch(uint32_t idx)
    {
        uint32_t x    = idx >> lg_bitwidth;
        uint32_t orig = words[x];
        words[x] |= 1 << (idx & bitwidth_mask);
    }

    void untouch(uint32_t idx)
    {
        uint32_t x    = idx >> lg_bitwidth;
        uint32_t orig = words[x];
        words[x] &= ~(1 << (idx & bitwidth_mask));
    }

    uint32_t check(uint32_t idx) const { return words[idx >> lg_bitwidth] & (1 << (idx & bitwidth_mask)); }

    void clear()
    {
        memset(buffer(), 0, buffer_nbytes());
        rewind();
    }

    void fill()
    {
        memset(buffer(), 0xFFFFFFFF, buffer_nbytes());
        *nnzs = n;
        rewind();
    }

    void fill(uint32_t from_word, uint32_t nwords_to_fill)
    {
        nwords_to_fill = std::min(nwords_to_fill, vector_nwords() - from_word);
        memset(words + from_word, 0xFFFFFFFF, nwords_to_fill * sizeof(uint32_t));
        *nnzs = nwords_to_fill * bitwidth;
        rewind();
    }

   public: /* Streaming Interface. */
    void rewind()
    {
        pos   = 0;
        cache = 0;

        /*
         * The last (extra) bit is always ON; used as a loop sentinel.
         * Below, we (re-)set the this bit, as it can be erased while pop()'ing.
         */
        touch(n);
    }

    void push(uint32_t idx) { touch(idx); }

    bool pop(uint32_t& idx)
    {
        while (words[pos] == 0) pos++;

        uint32_t lsb = __builtin_ctz(words[pos]);
        words[pos] ^= 1 << lsb;
        idx = (pos << lg_bitwidth) + lsb;

        return (idx < n);
    }

    bool next(uint32_t& idx)
    {
        while (words[pos] == 0) pos++;
        cache = cache ? cache : words[pos];

        uint32_t lsb = __builtin_ctz(cache);
        cache ^= 1 << lsb;
        idx = (pos << lg_bitwidth) + lsb;

        pos += (cache == 0);
        return idx < n;
    }

    template <bool destructive>
    bool advance(uint32_t& idx)
    {
        if (destructive)
            return pop(idx);
        else
            return next(idx);
    }


   protected: /* Buffer contains the non-zeros count, followed by a vector of n+1 bits. */
    uint32_t* buffer() const { return words - 1; }

    uint32_t vector_nwords() const
    {
        uint32_t n_ = n + 1;  // One more bit at the end, used as a loop sentinel.
        return n_ / bitwidth + (n_ % bitwidth > 0);
    }

    uint32_t buffer_nwords() const
    {
        return vector_nwords() + 1;  // One more integer, for the nnzs count.
    }

    uint32_t buffer_nbytes() const { return buffer_nwords() * sizeof(uint32_t); }

   public:
    static uint32_t get_bitwidth() { return bitwidth; }

    uint32_t get_nwords() const { return vector_nwords(); }
};

#endif
