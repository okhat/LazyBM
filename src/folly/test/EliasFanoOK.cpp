/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "EliasFano.h"

using Encoder = folly::compression::EliasFanoEncoderV2<uint32_t, uint32_t, 0, 0>;
using Reader = folly::compression::EliasFanoReader<Encoder, folly::compression::instructions::Default, true>;

int main(int argc, char **argv)
{
  std::vector<uint32_t> data({932, 2438, 16307, 17803, 18523});

  const auto &list = Encoder::encode(data.begin(), data.end());
  Reader reader(list);

  // reader.reset();
  reader.next();
  printf("%lu\n", reader.position());

  reader.skip(3);
  printf("%lu\n", reader.position());

  return 0;
}
