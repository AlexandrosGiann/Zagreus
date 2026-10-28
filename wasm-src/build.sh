#!/bin/sh
# Builds the two WebAssembly variants into ../wasm/.
set -e
cd "$(dirname "$0")"
mkdir -p ../wasm
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown --target-dir target/simd
cp target/simd/wasm32-unknown-unknown/release/zagreus_kernels.wasm ../wasm/zagreus-simd.wasm
RUSTFLAGS="" cargo build --release --target wasm32-unknown-unknown --target-dir target/base
cp target/base/wasm32-unknown-unknown/release/zagreus_kernels.wasm ../wasm/zagreus-base.wasm
ls -l ../wasm
