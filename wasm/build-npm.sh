#!/usr/bin/env bash
# Builds the npm package `impact-index` (wasm-pack, ES module for browsers,
# workers and privileged Firefox/Zotero code) into OUT_DIR (default:
# wasm/pkg). Used by .github/workflows/release-npm.yml.
#
#   wasm/build-npm.sh [OUT_DIR]
#   (cd wasm/pkg && npm publish --dry-run)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "${1:-$here/pkg}" && cd "${1:-$here/pkg}" && pwd)

wasm-pack build --release --target web --out-dir "$out" "$here"

cp "$here/NOTICE" "$out/"
cd "$out"
# wasm-pack names the package after the crate (impact-index-wasm)
npm pkg set name=impact-index
npm pkg set 'files[]=NOTICE' 'files[]=LICENSE-MIT' 'files[]=LICENSE-APACHE'
npm pkg set description='Sparse retrieval (learned sparse / impact indexes) in WebAssembly: compressed indexes, exact block-max WAND / MaxScore'
npm pkg set homepage=https://impact-index.readthedocs.io/en/latest/wasm.html
npm pkg set 'keywords[]=information-retrieval' 'keywords[]=sparse-retrieval' \
    'keywords[]=search' 'keywords[]=webassembly' 'keywords[]=splade'
npm pkg get name version
