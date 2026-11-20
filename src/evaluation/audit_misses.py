from pathlib import Path

from src.evaluation.eval_retrieval import is_relevant, load_items
from src.retrieval.rerank import search_rerank
from src.retrieval.search import close

OUT = Path("data/audit_misses.md")

lines = []
for item in load_items():
    results = search_rerank(item["question"], k=3)
    if any(is_relevant(r, item) for r in results):
        continue
    lines.append(f"## [{item['id']}] {item['question']}\n")
    lines.append(f"Labeled answer contains: \"{item['keyphrase']}\"\n")
    for rank, r in enumerate(results, start=1):
        lines.append(f"**#{rank} {r['title']} (p.{r['page']})**\n")
        lines.append(f"{r['text'][:1500]}\n")
    lines.append("Verdict: \n")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
close()