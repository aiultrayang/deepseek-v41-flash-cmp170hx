#!/usr/bin/env python3
"""DSV4.1-Flash PP6+EngramSSD 验收：正确性(乱码判据)+吞吐。

用法: python3 verify_ssd.py --base http://127.0.0.1:8001/v1 --model DeepSeek-V4.1-Flash [--key xxx] [--repeat 2] [--mode eager|graph]
判据（沿用四卡机 buliaoyin 标准）:
  - reasoning_content + content 合并检查：替换符/私有区码点/连续重复退化/语言混杂乱码 → 任一中招记 GARBLED
  - 24 样本 × repeat 以上，GARBLED 率 0% 为通过
"""
import argparse
import collections
import json
import re
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:8000/v1"
KEY = ""
MODEL = "Qwen3.5-122B-A10B-AWQ"

PROMPTS_CN = [
    "用三句话解释什么是惯性。",
    "写一首关于秋天雨夜的七言绝句，并给出押韵说明。",
    "把下面的句子翻译成英文：不登高山，不知天之高也。",
    "列出五种适合室内养的耐阴植物，并各给一句养护要点。",
    "解释牛顿第三定律，并举一个生活中的例子。",
    "为一家社区书店写三条开业宣传语，风格活泼。",
    "简述中国四大名著及其作者。",
    "计算 37*89 并说明竖式过程。",
    "写一段 Python 代码，用冒泡排序对列表排序，带注释。",
    "什么是复利？存 1 万元年利率 3%，五年后多少钱？",
    "给用户注册功能写三条验收标准。",
    "请用比喻解释 Transformer 的注意力机制。",
]
PROMPTS_EN = [
    "Explain the difference between a stack and a queue in two sentences.",
    "Write a haiku about the sea.",
    "List three prime numbers between 40 and 60.",
    "Summarize why the sky is blue in one paragraph.",
    "What is 15% of 240? Show your work.",
    "Translate: 'Knowledge is power' into French and explain the origin.",
]
PROMPTS_CODE = [
    "Write a TypeScript function that debounces calls, with types.",
    "Given a binary tree, write Python code for level-order traversal.",
    "Explain this SQL: SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*)>1;",
    "Fix the bug: for i in range(10): if i%2==0: total+=i — total starts undefined.",
]
LONG_TAIL = (
    "下面是一段会议记录，请总结要点并按决议/待办/风险分类：\n"
    + "第一季度营收增长但毛利率下滑；供应链交期从 45 天延长到 60 天；"
    "新客户集中于华东；建议 Q2 锁定二供；法务提醒数据出境合规窗口临近；"
    "研发人力缺口 3 人，HR 建议外包过渡两个月。\n" * 6
    + "请给出分类总结。"
)

ALL = PROMPTS_CN + PROMPTS_EN + PROMPTS_CODE + [LONG_TAIL, LONG_TAIL[:800] + "再精简一些。"]

BAD = re.compile(r"[\ufffd\uE000-\uF8FF]|(.)\1{9,}")


def garbled(t: str) -> str | None:
    if BAD.search(t):
        return "bad-chars/repeat"
    if len(t) < 12:
        return "too-short"
    # 中英夹杂乱码粗检：连续 6+ 个 ASCII 小写片段与 CJK 随机交错且无空格
    if len(re.findall(r"[a-z]{6,}", t)) >= 3 and re.search(r"[\u4e00-\u9fff][a-z]{6,}[\u4e00-\u9fff]", t):
        return "mixed-garble?"
    # 重复句子退化
    sents = [s.strip() for s in re.split(r"[。.!?\n]", t) if s.strip()]
    if len(sents) >= 5:
        c = collections.Counter(sents)
        if max(c.values(), default=0) / len(sents) > 0.6:
            return "repetition-collapse"
    return None


def p90(xs):
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(0.9 * (len(s) - 1)))))]


def ask(prompt: str, max_tokens: int, temperature: float = 0.7):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    txt = (msg.get("reasoning") or msg.get("reasoning_content") or "") + (msg.get("content") or "")
    usage = d.get("usage", {})
    ot = usage.get("completion_tokens", 0) or 1
    return txt, dt, ot / max(dt, 1e-6), usage


def main():
    global BASE, KEY, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--key", default=KEY)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--mode", default="?")
    args = ap.parse_args()
    BASE, KEY, MODEL = args.base, args.key, args.model
    total = bad = 0
    tps_all = []
    ttft_like = []
    for rnd in range(args.repeat):
        for i, p in enumerate(ALL):
            txt, dt, tps, usage = ask(p, max_tokens=768 if rnd == 0 else 256)
            g = garbled(txt)
            total += 1
            if g:
                bad += 1
            tps_all.append(tps)
            ttft_like.append(dt)
            flag = f"GARBLED[{g}]" if g else "ok"
            print(f"[{rnd:02d}/{i:02d}] {flag:>28} {dt:6.1f}s {tps:6.1f} tok/s :: {txt[:42]!r}", flush=True)
    print(f"\n== MODE={args.mode} 样本={total} 乱码={bad} ({bad/total*100:.1f}%)")
    print(f"   decode 中位 {statistics.median(tps_all):.1f} tok/s  p90 {p90(tps_all):.1f}  min {min(tps_all):.1f}")
    print(f"   端到端耗时 中位 {statistics.median(ttft_like):.1f}s  max {max(ttft_like):.1f}s")


if __name__ == "__main__":
    main()
