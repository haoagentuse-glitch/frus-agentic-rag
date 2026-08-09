"""(claim, chunk) score distribution.

Ran once to decide whether a support threshold on the attribution score was
possible. It is not: the score measures which passage a claim was written from,
so it cannot separate a supported claim from one that over-reads its source.
Kept because re-measuring is the cheapest way to check that conclusion still
holds after a reranker or prompt change.
"""
import asyncio, json
from frus_agentic_rag.agent.run import answer
from frus_agentic_rag.retrieval import rerank as rr

CASES = "eval/gold_cases_core.jsonl"

async def main():
    cases = [json.loads(x) for x in open(CASES) if x.strip()]
    cases = [c for c in cases if c.get("answerable", True)][:6]
    best_gold, best_other, all_scores = [], [], []
    for n, c in enumerate(cases, 1):
        gold = set(c.get("gold_documents") or [])
        a = await answer(c["question_en"], language="en", system="B3")
        ev = a.evidence
        if not a.claims or not ev:
            print(f"[{n}] {c['case_id']}: no claims/evidence", flush=True); continue
        pairs = [(cl.text, e.text[:1800]) for cl in a.claims for e in ev]
        sc = rr.score_pairs(pairs)
        m = len(ev)
        for i, cl in enumerate(a.claims):
            row = sc[i*m:(i+1)*m]
            j = max(range(m), key=lambda k: row[k])
            b = row[j]
            all_scores.append(b)
            is_gold = f"{ev[j].volume_id}:{ev[j].document_id}" in gold
            (best_gold if is_gold else best_other).append(b)
        print(f"[{n}/{len(cases)}] {c['case_id']} claims={len(a.claims)} ev={m}", flush=True)
    def q(xs, p):
        xs = sorted(xs); return xs[min(len(xs)-1, int(p*len(xs)))]
    print()
    print(f"每條主張的最佳分數，共 {len(all_scores)} 條")
    for p in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"   p{int(p*100):3d} = {q(all_scores, p):7.3f}")
    print(f"   最小 = {min(all_scores):.3f}   最大 = {max(all_scores):.3f}")
    print()
    print(f"最佳段落屬於 gold 文件 : n={len(best_gold):3d} 中位 {q(best_gold,0.5):7.3f}" if best_gold else "no gold")
    print(f"最佳段落非 gold        : n={len(best_other):3d} 中位 {q(best_other,0.5):7.3f}" if best_other else "no other")
    print()
    for t in (-6.0, -4.0, -2.0, 0.0, 2.0):
        kept = sum(1 for s in all_scores if s >= t)
        print(f"   min_score={t:5.1f} -> 保留 {kept:3d}/{len(all_scores)} ({kept/len(all_scores):.0%}) 的主張")
asyncio.run(main())
