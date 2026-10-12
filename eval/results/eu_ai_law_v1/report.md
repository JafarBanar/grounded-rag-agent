Model `gemma4:e4b`, embeddings `embeddinggemma`, thinking False, run 2026-10-12.

| Metric | agent | workflow |
|---|---|---|
| In-scope questions fully correct (all gold provisions cited) | 0.5 | 0.45 |
| Gold provisions cited (mean share) | 0.55 | 0.5 |
| Gold provisions retrieved (mean share) | 0.725 | 0.675 |
| In-scope questions answered (not refused) | 0.75 | 0.65 |
| Out-of-scope questions correctly refused | 1 | 1 |
| First draft passed the citation check | 0.625 | None |
| Seconds per question (mean) | 10.746 | 6.51 |
| Seconds per question (95th percentile) | 26.31 | 12.73 |
| Model calls per question | 3.208 | 1 |
| Tool calls per question | 1.625 | 0 |
| Tokens per question | 4775.5 | 1399.625 |

| Question type | agent: fully correct | workflow: fully correct |
|---|---|---|
| single | 0.7 (n=10) | 0.7 (n=10) |
| multi | 0.5 (n=6) | 0.333 (n=6) |
| cross | 0 (n=4) | 0 (n=4) |
