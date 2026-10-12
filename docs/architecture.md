# Architecture

The engine has one job: answer a question from a fixed set of documents, with a citation on
every factual sentence, and refuse when it cannot. Everything runs on local models through
Ollama, so private documents never leave the machine.

## Components

```mermaid
flowchart LR
    subgraph Ingest["Ingest (once per corpus)"]
        S1[EU acts, official XHTML] --> P1[eurlex parser]
        S2[Your files: md, docx, pdf] --> P2[file loader]
        P1 --> U[Citable units<br/>AI Act Art. 6(2), GDPR Art. 22(1),<br/>master-cv.md §Experience]
        P2 --> U
        U --> R[Cross-reference resolver]
        R --> I[(Hybrid index<br/>BM25 + embeddinggemma vectors)]
    end
    subgraph Agent["Agent (per question)"]
        Q[Question] --> L{{Gemma 4 tool loop}}
        L -- search --> I
        L -- read --> I
        L -- follow_refs --> I
        L --> D[Draft answer]
        D --> V[Citation check<br/>groundcite + evidence ledger]
        V -- pass --> A[Answer with citations]
        V -- fail --> F[One repair round]
        F --> L
        V -- still failing --> X[Refusal, listing what was checked]
    end
    A --> O[CLI / Web UI / MCP server / traces]
    X --> O
```

## Units: the citable pieces

Every document is cut into units whose id is also the citation the model must write.

| Source | Unit id examples | How it is cut |
|---|---|---|
| EU act article | `AI Act Art. 6`, `AI Act Art. 6(2)`, `AI Act Art. 5(1)(f)` | paragraphs from the official ids (`006.002`); lists split into lettered points |
| EU act definitions | `AI Act Art. 3(1)`, `GDPR Art. 4(11)` | numbered definitions, each with the article's opening sentence |
| EU act annex | `AI Act Annex III`, `AI Act Annex III(4)(a)` | top-level numbered points, then lettered points |
| EU act recital | `AI Act Recital 53` | one unit per recital |
| Markdown / Word | `master-cv.md §Experience`, `Project_Book.docx §LLM platform #2` | one unit per heading section, long sections split |
| PDF | `paper.pdf p.3` | one unit per page, long pages split |

Whole articles and annexes are kept as units for `read`, but search only returns the smaller
units, so results stay specific.

## Retrieval

* **BM25** with a tokenizer that keeps legal codes (`2016/679`) and hyphenated terms
  (`high-risk`, also indexed as `high` and `risk`). Strong for exact terms.
* **Dense vectors** from `embeddinggemma` (768 dimensions), with the model's recommended
  query and document prompts. Strong for paraphrases ("rank job applicants" finds
  "recruitment or selection of natural persons").
* **Reciprocal-rank fusion** of the two rankings. A paragraph and its own lettered points are
  never both shown, so the result list is not filled with near-duplicates.

## The agent loop

1. The system prompt states the rules: search first, read what you rely on, follow
   references, cite every sentence, answer `NOT FOUND:` when the documents do not answer.
2. Gemma 4 calls tools through Ollama's native tool calling. Each tool returns short text
   with unit ids, and records which units the model has seen (the **evidence ledger**).
3. If the model answers before using any tool, it is told to search first (a nudge; this
   does not use up a repair round).
4. The draft is checked:
   * every citation must exist in the corpus (no invented ids),
   * every citation must be in the evidence ledger (or be a parent or child of a unit in it),
   * at least `min_coverage` of the factual sentences must carry a citation (counted by
     `groundcite`).
5. On failure the problems go back to the model for one repair round. If it still fails, the
   user gets a refusal that lists the provisions that were checked, never the unverified draft.
6. Budgets: at most 12 tool calls and 8 model steps per question.

## Why a citation check in code

A model asked to cite sources can still cite something it never read, or add a sentence from
memory. Those answers look correct. The check makes that failure visible and turns it into a
refusal. During development the check also caught a bug in the sentence splitter of
`groundcite`: it cut `[GDPR Art. 22(1)]` at "Art." and counted the sentence as uncited. The
engine protects bracketed citations before checking.

## Baseline for comparison

`Workflow` is classic retrieve-once RAG with the same index, model and citation check: top-8
hybrid results, one model call. The evaluation runs both on the same questions, so the
agent's extra cost can be weighed against what it adds.

## Interfaces

* `gra ask <corpus> "<question>"`: prints each tool call and the checked answer.
* `gra serve`: FastAPI app; the browser shows each agent step live (server-sent events) and
  opens the source text of any citation.
* `gra mcp <corpus>`: the same tools (`search`, `read`, `follow_refs`, `ask`) as an MCP server
  over stdio, for Claude Desktop, IDEs or other agents.
* Traces: every run is appended to `data/traces/<corpus>/<date>.jsonl` with steps, timings
  and token counts.
