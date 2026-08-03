"""`frus` command line. Every long job here is resumable."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="FRUS bounded agentic RAG")
console = Console()


def _echo(payload: dict) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False, default=str))


@app.command()
def manifest(
    raw_dir: Path = typer.Option(None, help="Directory of FRUS volume XML"),
) -> None:
    """Scan the pinned snapshot: 694 XML -> per-volume inventory with SHA-256."""
    from frus_agentic_rag.ingest.manifest import build_manifest

    table = build_manifest(raw_dir=raw_dir)
    rows = table.to_pylist()
    published = [r for r in rows if r["status"] == "published"]
    _echo(
        {
            "xml_files": len(rows),
            "published_volumes": len(published),
            "planned_volumes": len(rows) - len(published),
            "type_document_divs": sum(r["n_document_divs"] for r in rows),
            "historical_document_divs": sum(r["n_historical_documents"] for r in rows),
        }
    )


@app.command()
def ingest(
    limit_volumes: int = typer.Option(None, help="Parse only the first N volumes"),
    volume: list[str] = typer.Option(None, help="Specific volume ids"),
    resume: bool = typer.Option(True, help="Skip volumes already parsed"),
    load: bool = typer.Option(True, help="Load parquet into LanceDB afterwards"),
    bm25: bool = typer.Option(True, help="Build the BM25 index afterwards"),
) -> None:
    """Parse TEI to chunk parquet, then load LanceDB and build BM25."""
    from frus_agentic_rag.index.build import (
        build_bm25_index,
        corpus_stats,
        load_into_lancedb,
        parse_all,
    )

    stats = parse_all(
        limit_volumes=limit_volumes, volume_ids=list(volume) if volume else None, resume=resume
    )
    result = {"parse": {k: v for k, v in stats.items() if k != "per_volume"}}
    if load:
        result["lancedb"] = load_into_lancedb(rebuild=not resume)
    if bm25:
        result["bm25"] = build_bm25_index()
    result["corpus"] = corpus_stats()
    _echo(result)


@app.command()
def index(
    bm25: bool = typer.Option(True, help="(Re)build the BM25 full-text index"),
    ann: bool = typer.Option(False, help="Build the vector ANN index (after embedding)"),
    reload_parquet: bool = typer.Option(False, "--reload", help="Reload all parquet into LanceDB"),
) -> None:
    """Rebuild indexes over already-parsed chunks."""
    from frus_agentic_rag.index.build import build_ann_index, build_bm25_index, load_into_lancedb

    out: dict = {}
    if reload_parquet:
        out["lancedb"] = load_into_lancedb(rebuild=True)
    if bm25:
        out["bm25"] = build_bm25_index()
    if ann:
        out["ann"] = build_ann_index()
    _echo(out)


@app.command()
def embed(
    all_: bool = typer.Option(False, "--all", help="Embed every parsed volume"),
    resume: bool = typer.Option(True, help="Skip volumes recorded as done"),
    device: str = typer.Option(None, help="cpu or cuda"),
    batch_size: int = typer.Option(None, help="Start batch size; halves on OOM"),
    limit_volumes: int = typer.Option(None, help="Embed only the first N pending volumes"),
) -> None:
    """Run BGE-M3 over parsed chunks, checkpointing per volume."""
    from frus_agentic_rag.index.embed import embed_all

    # --all means "no cap"; it wins over a stale --limit-volumes on the line.
    _echo(
        embed_all(
            resume=resume,
            device=device,
            batch_size=batch_size,
            limit_volumes=None if all_ else limit_volumes,
        )
    )


@app.command()
def benchmark(
    chunks: int = typer.Option(10000, help="Sample size"),
    device: str = typer.Option(None, help="cpu or cuda"),
    batch_size: int = typer.Option(None),
) -> None:
    """Measure BGE-M3 throughput and project the full-corpus ETA."""
    from frus_agentic_rag.index.embed import benchmark as run

    _echo(run(n_chunks=chunks, device=device, batch_size=batch_size))


@app.command()
def stats() -> None:
    """Write reports/corpus_stats.json and print it."""
    from frus_agentic_rag.index.build import corpus_stats

    _echo(corpus_stats())


@app.command()
def search(
    query: str,
    top_k: int = typer.Option(10),
    volume: list[str] = typer.Option(None),
) -> None:
    """Raw hybrid retrieval, no generation. Useful for eyeballing recall."""
    from frus_agentic_rag.retrieval.hybrid import hybrid_search_sync
    from frus_agentic_rag.retrieval.models import SearchFilters

    hits = hybrid_search_sync(
        query, SearchFilters(volume_ids=list(volume) if volume else []), top_k
    )
    t = Table("rank", "evidence_id", "date", "head", "bm25", "dense", "score")
    for i, h in enumerate(hits, 1):
        t.add_row(
            str(i),
            h.evidence_id,
            h.date_from,
            h.head[:60],
            str(h.rank_bm25 or "-"),
            str(h.rank_dense or "-"),
            f"{h.score:.4f}",
        )
    console.print(t)


@app.command("graph-smoke")
def graph_smoke() -> None:
    """Run the graph on fake tools across all four paths; assert budget bounds."""
    from frus_agentic_rag.agent.smoke import run_smoke

    _echo(run_smoke())


@app.command()
def ask(
    question: str,
    system: str = typer.Option("B3", help="B0, B1, B2 or B3"),
    language: str = typer.Option("zh-TW"),
) -> None:
    """Answer one question end to end."""
    import asyncio

    from frus_agentic_rag.agent.run import answer

    result = asyncio.run(answer(question, language=language, system=system))
    console.print(result.answer_text or f"[abstain] {result.abstain_reason}")
    for c in result.citations:
        console.print(f"  {c}")
    console.print(
        f"[dim]system={result.system} llm_calls={result.llm_calls} "
        f"retrieval={result.retrieval_calls} {result.latency_s:.1f}s[/dim]"
    )


@app.command("gold-build")
def gold_build(
    n_lookup: int = typer.Option(10),
    n_multihop: int = typer.Option(10),
    n_correction: int = typer.Option(5),
    n_unanswerable: int = typer.Option(5),
    out: Path = typer.Option(Path("eval/gold_cases.jsonl")),
) -> None:
    """Draft gold cases from the corpus. Output is marked as needing human review."""
    from frus_agentic_rag.eval_gold import build_gold_cases

    _echo(build_gold_cases(n_lookup, n_multihop, n_correction, n_unanswerable, out))


@app.command("eval")
def eval_cmd(
    systems: str = typer.Option("B0,B1,B2,B3"),
    cases: Path = typer.Option(Path("eval/gold_cases.jsonl")),
    languages: str = typer.Option("zh-TW,en"),
    limit: int = typer.Option(None),
    judge: bool = typer.Option(True, help="Use the external judge for answer correctness"),
    resume: bool = typer.Option(True, help="Reuse runs already in reports/ablation_runs.jsonl"),
) -> None:
    """Run the bilingual ablation and write reports/agent_ablation.json."""
    import asyncio

    from frus_agentic_rag.eval_run import run_ablation

    _echo(
        asyncio.run(
            run_ablation(
                systems=systems.split(","),
                cases_path=cases,
                languages=languages.split(","),
                limit=limit,
                use_judge=judge,
                resume=resume,
            )
        )
    )


if __name__ == "__main__":
    app()
