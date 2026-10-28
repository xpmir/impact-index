#!/bin/sh
# Builds synthetic indexes natively and checks that the WASM build loads
# them, gives the same search results, and rebuilds them byte for byte.
# Needs node and wasm-bindgen (same version as in Cargo.lock).
#
#   wasm/tests/check.sh [num-docs] [num-queries]
set -e
cd "$(dirname "$0")/../.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cargo run --quiet --release --example wasm_fixture -- "$tmp/fixture" "${1:-20000}" "${2:-50}"
cargo build --quiet --release -p impact-index-wasm --target wasm32-unknown-unknown
wasm-bindgen --target nodejs --out-dir "$tmp/pkg" \
    target/wasm32-unknown-unknown/release/impact_index_wasm.wasm
node wasm/tests/check.mjs "$tmp/pkg" "$tmp/fixture"
