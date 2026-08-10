"""P10: is the entailment verifier any good?

The verifier now decides which claims keep their citation, and its own accuracy
had never been measured. Running it over generated answers cannot measure it
either: a rejected claim there is either a false rejection or the verifier
correctly catching a hallucination, and telling those apart needs exactly the
ground truth this project does not have.

So the pairs are constructed instead, and both directions are unambiguous:

  positive   a sentence taken verbatim from the passage. A passage entails its
             own sentences, so a "not supported" here is a false rejection.
  negative   a sentence from an unrelated document, paired with this passage.
             A "supported" here is a false acceptance.

The caveat is worth stating: verbatim sentences are the easiest possible
positive, so the true-positive rate measured here is an upper bound on what the
verifier does with real paraphrased claims. A poor score is conclusive; a good
one is necessary, not sufficient.

    FRUS_GPU=1 ./scripts/dev.sh python scripts/probe_verifier.py
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

N_PAIRS = 20
SEED = 20260810


def _sentences(text: str) -> list[str]:
    from frus_agentic_rag.retrieval.focus import split_sentences

    return [s for s in split_sentences(text) if 60 < len(s) < 320]


async def main() -> None:
    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.agent.schemas import ClaimSupport
    from frus_agentic_rag.agent.verify import VERIFY_SYSTEM, verify_user
    from frus_agentic_rag.config import get_settings
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect

    rng = random.Random(SEED)
    tbl = connect().open_table(CHUNKS_TABLE)
    rows = (
        tbl.search()
        .where("subtype = 'historical-document' AND ordinal = 0")
        .select(["chunk_id", "document_id", "text"])
        .limit(4000)
        .to_list()
    )
    usable = [r for r in rows if len(_sentences(r["text"])) >= 2]
    rng.shuffle(usable)
    if len(usable) < N_PAIRS * 2:
        raise SystemExit("not enough passages with usable sentences")

    positives, negatives = [], []
    for i in range(N_PAIRS):
        host = usable[i]
        other = usable[N_PAIRS + i]
        positives.append((rng.choice(_sentences(host["text"])), host["text"]))
        negatives.append((rng.choice(_sentences(other["text"])), host["text"]))

    settings = get_settings()
    client = get_client()

    async def ask(pairs: list[tuple[str, str]]) -> list[bool | None]:
        """One call per batch of ten: the schema caps the list at twenty."""
        out: list[bool | None] = []
        for i in range(0, len(pairs), 10):
            batch = pairs[i : i + 10]
            try:
                res = await client.structured(
                    VERIFY_SYSTEM, verify_user(batch, settings.entailment_max_chars), ClaimSupport
                )
                by_i = {v.index: v.supported for v in res.claims}
                out.extend(by_i.get(j) for j in range(len(batch)))
            except Exception as exc:
                print(f"  batch failed: {type(exc).__name__}: {exc}", flush=True)
                out.extend([None] * len(batch))
        return out

    print(f"positives: {len(positives)} verbatim sentences from their own passage", flush=True)
    pos = await ask(positives)
    print(f"negatives: {len(negatives)} sentences from an unrelated document", flush=True)
    neg = await ask(negatives)

    pos_ok = [p for p in pos if p is not None]
    neg_ok = [n for n in neg if n is not None]
    tpr = sum(pos_ok) / len(pos_ok) if pos_ok else None
    fpr = sum(neg_ok) / len(neg_ok) if neg_ok else None

    result = {
        "criterion": "true-positive >= 0.90 and false-positive <= 0.20",
        "n_positive": len(pos_ok),
        "n_negative": len(neg_ok),
        "true_positive_rate": round(tpr, 3) if tpr is not None else None,
        "false_rejection_rate": round(1 - tpr, 3) if tpr is not None else None,
        "false_positive_rate": round(fpr, 3) if fpr is not None else None,
        "unanswered": (len(pos) - len(pos_ok)) + (len(neg) - len(neg_ok)),
        "caveat": (
            "verbatim positives are the easiest case, so the true-positive rate "
            "is an upper bound on performance against paraphrased claims"
        ),
        "pass": bool(tpr is not None and fpr is not None and tpr >= 0.90 and fpr <= 0.20),
    }
    Path("reports/probe_p10_verifier.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


Path("reports").mkdir(exist_ok=True)
asyncio.run(main())
