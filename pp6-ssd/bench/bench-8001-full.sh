#!/bin/bash
# 第 4 轮完整基准（无eager实例复测）；prefill 放最后防崩溃丢数据
cd $HOME/bench || exit 1
export VLLM_API_KEY=placeholder
NODE=$HOME/nodejs/bin/node
OUT=$HOME/bench/results8001-full
mkdir -p $OUT
B=http://127.0.0.1:8001
M=DeepSeek-V4.1-Flash
echo "== START $(date) =="
for mode in counting code prose; do
  for r in 1 2 3; do
    echo "-- 500 $mode r$r"
    $NODE scripts/benchmark-vllm.mjs --mode $mode --concurrency 1 --max-tokens 500 \
      --base-url $B --model $M --reasoning-effort none --output "$OUT/500-$mode-r$r.jsonl" || echo "FAIL 500 $mode r$r"
  done
done
echo "-- sweep 500"
$NODE scripts/benchmark-vllm.mjs --mode counting --concurrency 1,2,4,8,16,32 --max-tokens 500 \
  --base-url $B --model $M --reasoning-effort none --output "$OUT/sweep500.jsonl" || echo "FAIL sweep500"
for mode in counting code prose; do
  for r in 1 2 3; do
    echo "-- 2000 $mode r$r"
    $NODE scripts/benchmark-vllm.mjs --mode $mode --concurrency 1 --max-tokens 2000 \
      --base-url $B --model $M --reasoning-effort none --output "$OUT/2000-$mode-r$r.jsonl" || echo "FAIL 2000 $mode r$r"
  done
done
echo "-- sweep 2000"
$NODE scripts/benchmark-vllm.mjs --mode counting --concurrency 1,2,4,8 --max-tokens 2000 \
  --base-url $B --model $M --reasoning-effort none --output "$OUT/sweep2000.jsonl" || echo "FAIL sweep2000"
echo "-- prefill 65536 (LAST, may crash engine)"
$NODE scripts/benchmark-vllm.mjs --mode prefill --input-tokens 65536 --concurrency 1 \
  --base-url $B --model $M --output "$OUT/prefill-65536.jsonl" || echo "FAIL prefill"
echo "== DONE $(date) =="
