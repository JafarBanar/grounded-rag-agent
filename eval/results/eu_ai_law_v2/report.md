Model `gemma4:e4b`, embeddings `embeddinggemma`, thinking False, run 2026-10-12.

| Metric | agent | workflow |
|---|---|---|
| In-scope questions fully correct (all gold provisions cited) | 0.8 | 0.6 |
| Gold provisions cited (mean share) | 0.85 | 0.7 |
| Gold provisions retrieved (mean share) | 0.95 | 0.775 |
| In-scope questions answered (not refused) | 0.9 | 0.9 |
| Out-of-scope questions correctly refused | 1 | 1 |
| First draft passed the citation check | 0.75 | None |
| Seconds per question (mean) | 8.799 | 6.634 |
| Seconds per question (95th percentile) | 13.21 | 9.48 |
| Model calls per question | 3 | 1 |
| Tool calls per question | 1.583 | 0 |
| Tokens per question | 3756.75 | 1421.333 |

| Question type | agent: fully correct | workflow: fully correct |
|---|---|---|
| single | 0.9 (n=10) | 0.8 (n=10) |
| multi | 0.667 (n=6) | 0.667 (n=6) |
| cross | 0.75 (n=4) | 0 (n=4) |
