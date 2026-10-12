Model `gemma4:e4b`, embeddings `embeddinggemma`, thinking False, run 2026-10-12.

| Metric | agent | workflow |
|---|---|---|
| In-scope questions fully correct (all gold provisions cited) | 0.95 | 0.65 |
| Gold provisions cited (mean share) | 0.95 | 0.75 |
| Gold provisions retrieved (mean share) | 1.0 | 0.775 |
| In-scope questions answered (not refused) | 0.95 | 0.95 |
| Out-of-scope questions correctly refused | 1 | 1 |
| First draft passed the citation check | 0.75 | None |
| Seconds per question (mean) | 14.891 | 4.465 |
| Seconds per question (95th percentile) | 21.56 | 7.37 |
| Model calls per question | 5.583 | 1 |
| Tool calls per question | 3.167 | 0 |
| Tokens per question | 8102.583 | 1374.417 |

| Question type | agent: fully correct | workflow: fully correct |
|---|---|---|
| single | 1 (n=10) | 0.9 (n=10) |
| multi | 1 (n=6) | 0.667 (n=6) |
| cross | 0.75 (n=4) | 0 (n=4) |
