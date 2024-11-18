/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Philip Pronin (philipp@fb.com)
 *
 * Based on the paper by Sebastiano Vigna,
 * "Quasi-succinct indices" (arxiv:1206.4300).
 */

#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "Likely.h"
#include "Portability.h"
#include "Range.h"
#include "CodingDetail.h"
#include "Instructions.h"
#include "Select64.h"
#include "Bits.h"

#if !FOLLY_X64
#error EliasFanoCoding.h requires x86_64
#endif

namespace folly
{
namespace compression
{
template <class Pointer>
struct EliasFanoCompressedListBase
{
  EliasFanoCompressedListBase() = default;

  template <class T = Pointer>
  auto free() -> decltype(::free(T(nullptr)))
  {
    return ::free(data.data());
  }

  size_t upperSize() const
  {
    return size_t(data.end() - upper);
  }

  size_t size = 0;
  uint8_t numLowerBits = 0;

  // WARNING: EliasFanoCompressedList has no ownership of data. The 7
  // bytes following the last byte should be readable.
  folly::Range<Pointer> data;

  Pointer skipPointers = nullptr;
  Pointer lower = nullptr;
  Pointer upper = nullptr;
};

typedef EliasFanoCompressedListBase<const uint8_t *> EliasFanoCompressedList;
typedef EliasFanoCompressedListBase<uint8_t *> MutableEliasFanoCompressedList;

template <class Value, class SkipValue = size_t, size_t kSkipQuantum = 0>
struct EliasFanoEncoderV2
{
  static_assert(std::is_integral<Value>::value && std::is_unsigned<Value>::value, "Value should be unsigned integral");

  typedef EliasFanoCompressedList CompressedList;
  typedef MutableEliasFanoCompressedList MutableCompressedList;

  typedef Value ValueType;
  typedef SkipValue SkipValueType;
  struct Layout;

  static constexpr size_t skipQuantum = kSkipQuantum;

  static uint8_t defaultNumLowerBits(size_t upperBound, size_t size)
  {
    if (UNLIKELY(size == 0 || upperBound < size))
      return 0;

    auto candidate = folly::findLastSet(upperBound) - folly::findLastSet(size);

    return (size > (upperBound >> candidate)) ? candidate - 1 : candidate;
  }

  // Requires: input range (begin, end) is sorted (encoding crashes if it's not).
  // WARNING: encode() mallocates EliasFanoCompressedList::data. As
  // EliasFanoCompressedList has no ownership of it, you need to call
  // free() explicitly.
  template <class RandomAccessIterator>
  static MutableCompressedList encode(RandomAccessIterator begin, RandomAccessIterator end)
  {
    if (begin == end)
      return MutableCompressedList();

    EliasFanoEncoderV2 encoder(size_t(end - begin), *(end - 1));
    for (; begin != end; ++begin)
      encoder.add(*begin);

    return encoder.finish();
  }

  explicit EliasFanoEncoderV2(const MutableCompressedList &result)
      : lower_(result.lower),
        upper_(result.upper),
        skipPointers_(reinterpret_cast<SkipValueType *>(result.skipPointers)),
        result_(result)
  {
    std::fill(result.data.begin(), result.data.end(), '\0');
  }

  EliasFanoEncoderV2(size_t size, ValueType upperBound)
      : EliasFanoEncoderV2(Layout::fromUpperBoundAndSize(upperBound, size).allocList()) {}

  void add(ValueType value)
  {
    const auto numLowerBits = result_.numLowerBits;
    const ValueType upperBits = value >> numLowerBits;

    // Upper sequence consists of upperBits 0-bits and (size_ + 1) 1-bits.
    const size_t pos = upperBits + size_;
    upper_[pos / 8] |= 1U << (pos % 8);

    // Append numLowerBits bits to lower sequence.
    if (numLowerBits != 0)
    {
      const ValueType lowerBits = value & ((ValueType(1) << numLowerBits) - 1);
      writeBits56(lower_, size_ * numLowerBits, numLowerBits, lowerBits);
    }

    if (skipQuantum != 0)
    {
      while ((skipPointersSize_ + 1) * skipQuantum <= upperBits)
      {
        // Store the number of preceding 1-bits.
        skipPointers_[skipPointersSize_++] = SkipValue(size_);
      }
    }

    lastValue_ = value;
    ++size_;
  }

  const MutableCompressedList &finish() const
  {
    return result_;
  }

private:
  // Writes value (with len up to 56 bits) to data starting at pos-th bit.
  static void writeBits56(unsigned char *data, size_t pos, uint8_t len, uint64_t value)
  {
    unsigned char *const ptr = data + (pos / 8);
    uint64_t ptrv = folly::loadUnaligned<uint64_t>(ptr);
    ptrv |= value << (pos % 8);
    folly::storeUnaligned<uint64_t>(ptr, ptrv);
  }

  unsigned char *lower_ = nullptr;
  unsigned char *upper_ = nullptr;
  SkipValueType *skipPointers_ = nullptr;

  ValueType lastValue_ = 0;
  size_t size_ = 0;
  size_t skipPointersSize_ = 0;

  MutableCompressedList result_;
};

template <class Value, class SkipValue, size_t kSkipQuantum>
struct EliasFanoEncoderV2<Value, SkipValue, kSkipQuantum>::Layout
{
  static Layout fromUpperBoundAndSize(size_t upperBound, size_t size)
  {
    // numLowerBits can be at most 56 because of detail::writeBits56.
    const uint8_t numLowerBits = std::min(defaultNumLowerBits(upperBound, size), uint8_t(56));

    // *** Upper bits.
    // Upper bits are stored using unary delta encoding.
    // For example, (3 5 5 9) will be encoded as 1000011001000_2.
    const size_t upperSizeBits =
        (upperBound >> numLowerBits) + // Number of 0-bits to be stored.
        size;                          // 1-bits.
    const size_t upper = (upperSizeBits + 7) / 8;

    // *** Validity checks.
    // Shift by numLowerBits must be valid.

    return fromInternalSizes(numLowerBits, upper, size);
  }

  static Layout fromInternalSizes(uint8_t numLowerBits, size_t upper, size_t size)
  {
    Layout layout;
    layout.size = size;
    layout.numLowerBits = numLowerBits;

    layout.lower = (numLowerBits * size + 7) / 8;
    layout.upper = upper;

    // *** Skip pointers.
    // Store (1-indexed) position of every skipQuantum-th 0-bit in upper bits sequence.
    if (skipQuantum != 0)
    {
      size_t numSkipPointers = (8 * upper - size) / skipQuantum;
      layout.skipPointers = numSkipPointers * sizeof(SkipValueType);
    }

    return layout;
  }

  size_t bytes() const
  {
    return lower + upper + skipPointers;
  }

  template <class Range>
  EliasFanoCompressedListBase<typename Range::iterator> openList(Range &buf) const
  {
    EliasFanoCompressedListBase<typename Range::iterator> result;
    result.size = size;
    result.numLowerBits = numLowerBits;
    result.data = buf.subpiece(0, bytes());

    auto advance = [&](size_t n) {
      auto begin = buf.data();
      buf.advance(n);
      return begin;
    };

    result.skipPointers = advance(skipPointers);
    result.lower = advance(lower);
    result.upper = advance(upper);

    return result;
  }

  MutableCompressedList allocList() const
  {
    uint8_t *buf = nullptr;
    // WARNING: Current read/write logic assumes that the 7 bytes
    // following the last byte of lower and upper sequences are
    // readable (stored value doesn't matter and won't be changed), so
    // we allocate additional 7 bytes, but do not include them in size
    // of returned value.
    if (size > 0)
      buf = static_cast<uint8_t *>(malloc(bytes() + 7));

    printf("this uses %lu bytes.\n", bytes());

    folly::MutableByteRange bufRange(buf, bytes());
    return openList(bufRange);
  }

  MutableCompressedList fromList(uint8_t *buf)
  {
    folly::MutableByteRange bufRange(buf, bytes());
    return openList(bufRange);
  }

  size_t size = 0;
  uint8_t numLowerBits = 0;

  // Sizes in bytes.
  size_t lower = 0;
  size_t upper = 0;
  size_t skipPointers = 0;
};

namespace detail
{

template <class Encoder, class Instructions, class SizeType>
class UpperBitsReader : SkipPointers<Encoder::skipQuantum>
{
  typedef typename Encoder::SkipValueType SkipValueType;

public:
  typedef typename Encoder::ValueType ValueType;

  explicit UpperBitsReader(const typename Encoder::CompressedList &list)
      : SkipPointers<Encoder::skipQuantum>(list.skipPointers),
        start_(list.upper)
  {
    reset();
  }

  void reset()
  {
    block_ = start_ != nullptr ? folly::loadUnaligned<block_t>(start_) : 0;
    position_ = std::numeric_limits<SizeType>::max();
    outer_ = 0;
    value_ = 0;
  }

  SizeType position() const
  {
    return position_;
  }
  ValueType value() const
  {
    return value_;
  }

  ValueType previous()
  {
    size_t inner;
    block_t block;
    getPreviousInfo(block, inner, outer_);
    block_ = folly::loadUnaligned<block_t>(start_ + outer_);
    block_ ^= block;
    --position_;
    return setValue(inner);
  }

  ValueType next()
  {
    // Skip to the first non-zero block.
    while (block_ == 0)
    {
      outer_ += sizeof(block_t);
      block_ = folly::loadUnaligned<block_t>(start_ + outer_);
    }

    ++position_;
    size_t inner = Instructions::ctz(block_);
    block_ = Instructions::blsr(block_);

    return setValue(inner);
  }

  ValueType skip(SizeType n)
  {
    position_ += n; // n 1-bits will be read.

    size_t cnt;
    // Find necessary block.
    while ((cnt = Instructions::popcount(block_)) < n)
    {
      n -= cnt;
      outer_ += sizeof(block_t);
      block_ = folly::loadUnaligned<block_t>(start_ + outer_);
    }

    // Skip to the n-th one in the block.
    size_t inner = select64<Instructions>(block_, n - 1);
    block_ &= (block_t(-1) << inner) << 1;

    return setValue(inner);
  }

  // Skip to the first element that is >= v and located *after* the current
  // one (so even if current value equals v, position will be increased by 1).
  ValueType skipToNext(ValueType v)
  {
    // Use skip pointer.
    if (Encoder::skipQuantum > 0 && v >= value_ + Encoder::skipQuantum)
    {
      const size_t steps = v / Encoder::skipQuantum;
      const size_t dest = folly::loadUnaligned<SkipValueType>(
          this->skipPointers_ + (steps - 1) * sizeof(SkipValueType));

      reposition(dest + Encoder::skipQuantum * steps);
      position_ = dest - 1;
    }

    // Skip by blocks.
    size_t cnt;
    size_t skip = v - (8 * outer_ - position_ - 1);

    constexpr size_t kBitsPerBlock = 8 * sizeof(block_t);
    while ((cnt = Instructions::popcount(~block_)) < skip)
    {
      skip -= cnt;
      position_ += kBitsPerBlock - cnt;
      outer_ += sizeof(block_t);
      block_ = folly::loadUnaligned<block_t>(start_ + outer_);
    }

    if (LIKELY(skip))
    {
      auto inner = select64<Instructions>(~block_, skip - 1);
      position_ += inner - skip + 1;
      block_ &= block_t(-1) << inner;
    }

    next();
    return value_;
  }

  ValueType jump(size_t n)
  {
    // Avoid reading the head, skip() will reposition.
    position_ = std::numeric_limits<SizeType>::max();

    return skip(n);
  }

  ValueType jumpToNext(ValueType v)
  {
    if (Encoder::skipQuantum == 0 || v < Encoder::skipQuantum)
    {
      reset();
    }
    else
    {
      value_ = 0; // Avoid reading the head, skipToNext() will reposition.
    }
    return skipToNext(v);
  }

  ValueType previousValue() const
  {
    block_t block;
    size_t inner;
    OuterType outer;
    getPreviousInfo(block, inner, outer);
    return static_cast<ValueType>(8 * outer + inner - (position_ - 1));
  }

  void setDone(SizeType endPos)
  {
    position_ = endPos;
  }

private:
  ValueType setValue(size_t inner)
  {
    value_ = static_cast<ValueType>(8 * outer_ + inner - position_);
    return value_;
  }

  void reposition(SizeType dest)
  {
    outer_ = dest / 8;
    block_ = folly::loadUnaligned<block_t>(start_ + outer_);
    block_ &= ~((block_t(1) << (dest % 8)) - 1);
  }

  using block_t = uint64_t;
  // The size in bytes of the upper bits is limited by n + universe / 8,
  // so a type that can hold either sizes or values is sufficient.
  using OuterType = typename std::common_type<ValueType, SizeType>::type;

  void getPreviousInfo(block_t &block, size_t &inner, OuterType &outer) const
  {
    outer = outer_;
    block = folly::loadUnaligned<block_t>(start_ + outer);
    inner = size_t(value_) - 8 * outer_ + position_;
    block &= (block_t(1) << inner) - 1;
    while (UNLIKELY(block == 0))
    {
      outer -= std::min<OuterType>(sizeof(block_t), outer);
      block = folly::loadUnaligned<block_t>(start_ + outer);
    }
    inner = 8 * sizeof(block_t) - 1 - Instructions::clz(block);
  }

  const unsigned char *const start_;
  block_t block_;
  SizeType position_; // Index of current value (= #reads - 1).
  OuterType outer_;   // Outer offset: number of consumed bytes in upper.
  ValueType value_;
};

} // namespace detail

// If kUnchecked = true the caller must guarantee that all the
// operations return valid elements, i.e., they would never return
// false if checked.
template <
    class Encoder,
    class Instructions = instructions::Default,
    bool kUnchecked = false,
    class SizeType = size_t>
class EliasFanoReader
{
public:
  typedef Encoder EncoderType;
  typedef typename Encoder::ValueType ValueType;

  explicit EliasFanoReader(const typename Encoder::CompressedList &list)
      : upper_(list),
        lower_(list.lower),
        size_(list.size),
        numLowerBits_(list.numLowerBits)
  {
    // To avoid extra branching during skipTo() while reading
    // upper sequence we need to know the last element.
    // If kUnchecked == true, we do not check that skipTo() is called
    // within the bounds, so we can avoid initializing lastValue_.
    if (kUnchecked || UNLIKELY(list.size == 0))
    {
      lastValue_ = 0;
      return;
    }
    ValueType lastUpperValue = ValueType(8 * list.upperSize() - size_);
    auto it = list.upper + list.upperSize() - 1;
    lastUpperValue -= 8 - folly::findLastSet(*it);
    lastValue_ = readLowerPart(size_ - 1) | (lastUpperValue << numLowerBits_);
  }

  void reset()
  {
    upper_.reset();
    value_ = kInvalidValue;
  }

  bool previous()
  {
    if (!kUnchecked && UNLIKELY(position() == 0))
    {
      reset();
      return false;
    }
    upper_.previous();
    value_ =
        readLowerPart(upper_.position()) | (upper_.value() << numLowerBits_);
    return true;
  }

  bool next()
  {
    if (!kUnchecked && UNLIKELY(position() + 1 >= size_))
    {
      return setDone();
    }
    upper_.next();
    value_ =
        readLowerPart(upper_.position()) | (upper_.value() << numLowerBits_);
    return true;
  }

  bool skip(SizeType n)
  {
    if (kUnchecked || LIKELY(position() + n < size_))
    {
      if (LIKELY(n < kLinearScanThreshold))
      {
        for (SizeType i = 0; i < n; ++i)
        {
          upper_.next();
        }
      }
      else
      {
        upper_.skip(n);
      }
      value_ =
          readLowerPart(upper_.position()) | (upper_.value() << numLowerBits_);
      return true;
    }

    return setDone();
  }

  bool skipTo(ValueType value)
  {
    if (!kUnchecked && value > lastValue_)
      return setDone();
    else if (value == value_)
      return true;

    ValueType upperValue = (value >> numLowerBits_);
    ValueType upperSkip = upperValue - upper_.value();
    // The average density of ones in upper bits is 1/2.
    // LIKELY here seems to make things worse, even for small skips.
    if (upperSkip < 2 * kLinearScanThreshold)
    {
      do
      {
        upper_.next();
      } while (UNLIKELY(upper_.value() < upperValue));
    }
    else
    {
      upper_.skipToNext(upperValue);
    }

    iterateTo(value);
    return true;
  }

  bool jump(SizeType n)
  {
    if (LIKELY(n < size_))
    { // Also checks that n != -1.
      value_ = readLowerPart(n) | (upper_.jump(n + 1) << numLowerBits_);
      return true;
    }

    return setDone();
  }

  bool jumpTo(ValueType value)
  {
    if (!kUnchecked && value > lastValue_)
      return setDone();

    upper_.jumpToNext(value >> numLowerBits_);
    iterateTo(value);

    return true;
  }

  ValueType lastValue() const
  {
    return lastValue_;
  }

  ValueType previousValue() const
  {
    return readLowerPart(upper_.position() - 1) |
           (upper_.previousValue() << numLowerBits_);
  }

  SizeType size() const
  {
    return size_;
  }

  bool valid() const
  {
    return position() < size(); // Also checks that position() != -1.
  }

  SizeType position() const
  {
    return upper_.position();
  }

  ValueType value() const
  {
    return value_;
  }

private:
  // Must hold kInvalidValue + 1 == 0.
  constexpr static ValueType kInvalidValue =
      std::numeric_limits<ValueType>::max();

  bool setDone()
  {
    value_ = kInvalidValue;
    upper_.setDone(size_);
    return false;
  }

  ValueType readLowerPart(SizeType i) const
  {
    const size_t pos = i * numLowerBits_;
    const unsigned char *ptr = lower_ + (pos / 8);
    const uint64_t ptrv = folly::loadUnaligned<uint64_t>(ptr);

    // This removes the branch in the fallback implementation of
    // bzhi. The condition is verified at encoding time.
    if (numLowerBits_ >= sizeof(ValueType) * 8)
      __builtin_unreachable();

    return Instructions::bzhi(ptrv >> (pos % 8), numLowerBits_);
  }

  void iterateTo(ValueType value)
  {
    while (true)
    {
      value_ = readLowerPart(upper_.position()) | (upper_.value() << numLowerBits_);

      if (LIKELY(value_ >= value))
        break;

      upper_.next();
    }
  }

  constexpr static size_t kLinearScanThreshold = 8;

  detail::UpperBitsReader<Encoder, Instructions, SizeType> upper_;
  const uint8_t *lower_;
  SizeType size_;
  ValueType value_ = kInvalidValue;
  ValueType lastValue_;
  uint8_t numLowerBits_;
};

} // namespace compression
} // namespace folly
