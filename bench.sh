#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_BIN="${ROOT}/src/bench"
OUT_DIR="${ROOT}/docs/bench"

usage() {
    cat <<'EOF'
Usage: ./bench.sh model_path [-i "prompt"] [bench options...]

Run the inference benchmark for a GGUF model and store the result in docs/bench/.

Examples:
  ./bench.sh model.gguf -i "Who is Isaac Newton?"
  ./bench.sh model.gguf -i "Explain quantum computing." -r 10 -n 256 -T 8

Options are forwarded to src/bench, so any bench option works:
  -c  context size     -n  tokens to generate    -T  threads
  -t  temperature      -tp top-p                 -tk top-k    -P  min-p
  -r  repeat count     -i  input prompt          -o  output text file
EOF
}

[ $# -ge 1 ] || { usage; exit 2; }

MODEL="$1"
shift

if [ "${MODEL}" = "-h" ] || [ "${MODEL}" = "--help" ]; then
    usage
    exit 0
fi

if [ ! -f "${MODEL}" ]; then
    echo "error: model file not found: ${MODEL}" >&2
    exit 2
fi

if [ ! -f "${BENCH_BIN}" ]; then
    echo "error: bench binary not found: ${BENCH_BIN} (run 'make' first)" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="$(basename "${MODEL}" .gguf)"
RESULT="${OUT_DIR}/${BASE}_${TS}.txt"
GEN="${OUT_DIR}/${BASE}_${TS}.gen.txt"

# Add -o only if the user did not supply their own output file.
has_output=false
for arg in "$@"; do
    if [ "${arg}" = "-o" ] || [ "${arg}" = "--output" ]; then
        has_output=true
        break
    fi
done
if [ "${has_output}" = false ]; then
    set -- "$@" -o "${GEN}"
fi

echo "Model:  ${MODEL}"
echo "Args:   $*"
echo "Result: ${RESULT}"
echo

"${BENCH_BIN}" -m "${MODEL}" "$@" >"${RESULT}" 2>&1 || {
    echo "error: benchmark failed (exit $?)" >&2
    exit 1
}

echo
echo "Benchmark result saved to ${RESULT}"
