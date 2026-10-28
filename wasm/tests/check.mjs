// Checks the WASM build against native search results.
//
//   cargo run --release --example wasm_fixture -- <fixture-dir> [docs] [queries]
//   cargo build -p impact-index-wasm --target wasm32-unknown-unknown --release
//   wasm-bindgen --target nodejs --out-dir <pkg-dir> \
//       target/wasm32-unknown-unknown/release/impact_index_wasm.wasm
//   node wasm/tests/check.mjs <pkg-dir> <fixture-dir>

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { createRequire } from "node:module";

const [pkgDir, fixtureDir] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const { Builder, Index, DocSet, SegmentedIndex, memoryBytes } = require(join(pkgDir, "impact_index_wasm.js"));

/** Map from paths relative to `dir` to their contents */
function readIndex(dir) {
  const files = new Map();
  const walk = (d) => {
    for (const name of readdirSync(d)) {
      const path = join(d, name);
      if (statSync(path).isDirectory()) walk(path);
      else files.set(relative(dir, path).replaceAll("\\", "/"), new Uint8Array(readFileSync(path)));
    }
  };
  walk(dir);
  return files;
}

const fixture = JSON.parse(readFileSync(join(fixtureDir, "fixture.json"), "utf8"));
const k = fixture.top_k;
let failures = 0;

function compare(label, got, expected) {
  // Same scores (up to f32 rounding), and same documents above the k-th score
  // (documents tied at the k-th score may differ)
  const ok =
    got.length === expected.length &&
    got.every((g, i) => Math.abs(g.at(-1) - expected[i].at(-1)) <= 1e-5 * Math.max(1, Math.abs(g.at(-1))));
  const kth = expected.length ? expected.at(-1).at(-1) : 0;
  const above = (r) =>
    r
      .filter((x) => x.at(-1) > kth + 1e-5)
      .map((x) => x.slice(0, -1).join(":"))
      .sort()
      .join(",");
  if (!ok || above(got) !== above(expected)) {
    failures++;
    if (failures < 5) console.error(`${label}: got`, got.slice(0, 5), "expected", expected.slice(0, 5));
  }
}

function timed(label, queries, run) {
  const start = performance.now();
  const results = queries.map(run);
  const ms = (performance.now() - start) / queries.length;
  console.log(`${label}: ${ms.toFixed(2)} ms/query`);
  return results;
}

const queries = fixture.queries.map((q) => [new Uint32Array(q.terms), new Float32Array(q.weights)]);

let start = performance.now();
const compressed = Index.load(readIndex(join(fixtureDir, "compressed")));
console.log(`load compressed: ${(performance.now() - start).toFixed(0)} ms, ${compressed.numTerms} terms`);
const q = compressed.quantizer();
if (!q || q.nbits !== fixture.quantizer.nbits || q.min !== fixture.quantizer.min || q.max !== fixture.quantizer.max) {
  failures++;
  console.error("quantizer differs", q && { nbits: q.nbits, min: q.min, max: q.max }, fixture.quantizer);
}
// Warm-up
queries.forEach(([t, w]) => compressed.search(t, w, k));
timed("search compressed (EF docids)", queries, ([t, w]) => compressed.search(t, w, k)).forEach((r, i) =>
  compare(`compressed #${i}`, Array.from(r.docids, (d, j) => [d, r.scores[j]]), fixture.expected.compressed[i]),
);

const split = Index.load(readIndex(join(fixtureDir, "split")));
timed("search split", queries, ([t, w]) => split.search(t, w, k)).forEach((r, i) =>
  compare(`split #${i}`, Array.from(r.docids, (d, j) => [d, r.scores[j]]), fixture.expected.split[i]),
);

const segmented = new SegmentedIndex();
for (const [i, name] of ["seg0", "seg1"].entries()) {
  segmented.push(Index.load(readIndex(join(fixtureDir, name))), DocSet.fromDocids(new Uint32Array(fixture.deleted[i])));
}
timed("search 2 segments + tombstones", queries, ([t, w]) => segmented.search(t, w, k)).forEach((r, i) =>
  compare(
    `segmented #${i}`,
    Array.from(r.docids, (d, j) => [r.segments[j], d, r.scores[j]]),
    fixture.expected.segmented[i],
  ),
);

// Indexes built in WASM are the same as native ones (same options as
// `compressed/` in examples/wasm_fixture.rs)
if (fixture.docs) {
  start = performance.now();
  const builder = new Builder({ docids: "ef", nbits: 8, range: { min: 0, max: 3 }, blockSize: 128 });
  fixture.docs.forEach(([terms, values], docid) => builder.add(docid, new Uint32Array(terms), new Float32Array(values)));
  const built = builder.finish();
  console.log(`build ${fixture.docs.length} docs in WASM: ${(performance.now() - start).toFixed(0)} ms, clamped ${builder.clampedImpacts}`);
  const native = readIndex(join(fixtureDir, "compressed"));
  const names = (m) => [...m.keys()].sort().join(",");
  if (names(built) !== names(native)) {
    failures++;
    console.error("built files differ:", names(built), "vs", names(native));
  }
  for (const [name, bytes] of native) {
    if (name === "manifest.json") continue; // creation time
    const other = built.get(name);
    if (!other || other.length !== bytes.length || other.some((b, i) => b !== bytes[i])) {
      failures++;
      console.error(`${name} differs between the WASM and native builds`);
    }
  }
  if (!JSON.parse(new TextDecoder().decode(built.get("manifest.json"))).created.startsWith("20")) {
    failures++;
    console.error("manifest without creation date");
  }
  const rebuilt = Index.load(built);
  queries.forEach(([t, w], i) => {
    const r = rebuilt.search(t, w, k);
    compare(`rebuilt #${i}`, Array.from(r.docids, (d, j) => [d, r.scores[j]]), fixture.expected.compressed[i]);
  });

  // A segment quantised like the first one
  const segment = new Builder({ range: compressed.quantizer() });
  segment.add(0, new Uint32Array([1, 2]), new Float32Array([0.5, 3.5]));
  const q2 = Index.load(segment.finish()).quantizer();
  if (q2.min !== q.min || q2.max !== q.max || q2.step !== q.step || segment.clampedImpacts !== 1) {
    failures++;
    console.error("reference quantizer not reused", q2.min, q2.max, segment.clampedImpacts);
  }
  try {
    builder.add(0, new Uint32Array([1]), new Float32Array([1]));
    failures++;
  } catch (e) {}
}

// Errors surface as exceptions, and the instance keeps working
try {
  Index.load(new Map([["index.cbor", new Uint8Array([0xff, 0, 1])]]));
  failures++;
  console.error("corrupt index loaded");
} catch (e) {
  console.log(`corrupt index: ${e.message}`);
}
try {
  compressed.search(new Uint32Array([1, 2]), new Float32Array([1]), k);
  failures++;
} catch (e) {}
compare("after errors", Array.from(compressed.search(...queries[0], k).docids, (d, j) => [d, compressed.search(...queries[0], k).scores[j]]), fixture.expected.compressed[0]);

const memory = process.memoryUsage();
console.log(`WASM memory: ${(memoryBytes() / 2 ** 20).toFixed(0)} MB`);
console.log(`node memory: rss ${(memory.rss / 2 ** 20).toFixed(0)} MB, array buffers ${(memory.arrayBuffers / 2 ** 20).toFixed(0)} MB`);
const total = fixture.queries.length * 3;
console.log(failures ? `FAILED: ${failures} mismatches` : `OK: ${total} queries match native`);
process.exit(failures ? 1 : 0);
