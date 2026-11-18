# index-v3 fixtures

Indexes written by the pre-refactor code (commit `54db65c`, crate 1.7.1,
manifest `format_version` 3), used by `tests/old_index_fixture.rs` to check
that old on-disk indexes keep loading and give identical top-10 results.

All data is synthetic: 2000 documents, 500 terms, 5-29 terms per document
with a skewed term distribution, impacts uniform in [0.05, 5) (integers
1..15 for the `*_bitpackedint` variants, built from a separate integer forward
index that is not stored). RNG: `StdRng::seed_from_u64(20240601)`.
10 fixed queries (2-6 terms, weights in [0.1, 2)) are stored in
`expected.json` with, for each variant, the top-10 `[docid, score]` from WAND
and MaxScore (`MaxScoreOptions::default()`).

Variants (all compressed ones use `max_block_size` 128):

- `forward`: raw forward index
- `ef_identity`: Elias-Fano docids, `Identity` impacts
- `ef_quantizer8`: Elias-Fano, `Quantizer::new(8, 0., 5.)`
- `pfor_quantbitpacked8`: PFor docids, `QuantizedBitPackedFactory { nbits: 8 }`
- `bitpacking_quantbitpacked8`: bit-packing docids, same impacts codec
- `pfor_bitpackedint`, `ef_bitpackedint`: `BitPackedIntCompressor` (integer impacts)
- `split`: `SplitIndexTransform` (quantiles 50/64, 63/64) over Elias-Fano + `Identity`

Generator: a throw-away integration test in a checkout of `54db65c`
(not committed) that builds the forward index with `Indexer`, applies each
`CompressionTransform` / `SplitIndexTransform` via `IndexTransform::process`,
reloads each with `load_index`, runs `search_wand` / `search_maxscore`
(top-k 10) and dumps the results. Do not regenerate these files with newer
code: the point is that they were written by the old one.
