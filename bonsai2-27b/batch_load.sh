#!/bin/bash
# N concurrent chat completions against the batched server; wall time and aggregate tokens / s
PORT=$1; N=$2; MAXT=${3:-48}; OUT=/tmp/bload; rm -rf $OUT; mkdir -p $OUT
P=("Give me two uses of copper in one sentence." "Write a Python function that returns the n-th Fibonacci number." "Explain in two sentences why the sky is blue." "List three uses of copper." "What is the boiling point of water in Kelvin?" "Name the largest planet of the solar system." "Translate 'good morning' into Spanish." "Who wrote the Odyssey?")
t0=$(date +%s.%N)
for i in $(seq 0 $((N-1))); do
  curl -s -m 600 localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"x\", \"messages\": [{\"role\": \"user\", \"content\": \"${P[$((i % 8))]}\"}], \"max_tokens\": $MAXT}" > $OUT/r$i.json &
done
wait; t1=$(date +%s.%N)
python3 - "$OUT" "$t0" "$t1" <<PY
import json, glob, sys
tot = 0
for f in sorted(glob.glob(sys.argv[1] + "/r*.json")):
  r = json.load(open(f)); c = r["choices"][0]["message"]["content"]; u = r["usage"]; tot += u["completion_tokens"]; print(f"  {f[-7:]}: {u['completion_tokens']:3d} tokens | {c[:90]!r}")
dt = float(sys.argv[3]) - float(sys.argv[2]); print(f"  {len(glob.glob(sys.argv[1] + '/r*.json'))} concurrent requests: {tot} completion tokens in {dt:.1f} s wall -> {tot / dt:.1f} tok/s aggregate (prefill included)")
PY
