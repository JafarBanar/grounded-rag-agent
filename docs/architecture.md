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
        Q[Question] --> P[Plan: 1-4 search queries<br/>structured JSON]
        P --> L{{Gemma 4 tool loop}}
        L -- search --> I
        L -- read --> I
        L -- follow_refs --> I
        L --> D[Draft answer]
        D --> V[Citation check<br/>groundcite + evidence ledger]
        V -- pass --> K[Claim check, optional<br/>each sentence vs its source]
        K -- pass --> A[Answer with citations]
        K -- fail --> F
        V -- fail --> F[Repair round, max 2]
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

1. **Plan.** One call with a JSON schema as the output format splits the question into 1 to 4
   search queries. This fixed the main failure of earlier versions: answering the first half of
   a two-part question and stopping.
2. **Tool loop.** The system prompt states the rules: search first, read what you rely on,
   follow references, cite every sentence, answer `NOT FOUND:` when the documents do not
   answer, and end a partial answer with one `Not shown in the documents:` sentence. Gemma 4
   calls tools through Ollama's native tool calling. Each tool returns short text with unit
   ids and records which units the model has seen (the **evidence ledger**).
3. If the model answers before using any tool, it is told to search first (a nudge; this
   does not use up a repair round).
4. **Citation check** on the draft:
   * the answer must not be empty and must cite at least one unit,
   * every citation must exist in the corpus (no invented ids),
   * every citation must be in the evidence ledger (or be a parent or child of a unit in it),
   * at least `min_coverage` of the factual sentences must carry a citation (counted by
     `groundcite`; the gap sentence is not counted).
5. **Claim check** (`verify: true` in the corpus config): one structured call compares each
   cited sentence with the text of the units it cites and lists the unsupported ones.
6. On failure the problems go back to the model, for up to `repair_attempts` rounds (2 by
   default in the shipped configs). If it still fails, the user gets a refusal that lists the
   provisions that were checked, never the unverified draft.
7. Budgets: at most 12 tool calls and 8 model steps per question.

## Why a citation check in code

A model asked to cite sources can still cite something it never read, or add a sentence from
memory. Those answers look correct. The check makes that failure visible and turns it into a
refusal. During development the check also caught a bug in the sentence splitter of
`groundcite`: it cut `[GDPR Art. 22(1)]` at "Art." and counted the sentence as uncited. The
engine protects bracketed citations before checking.

The citation check has a known limit: it proves a cited unit was retrieved, not that the
sentence says what the unit says. On private documents this let one wrong answer through (a
colleague's work credited to the user, from a section that ingestion had filled with the wrong
table). The claim check in step 5 is the second line of defence for that class of error.
The evaluation also found a hole in the check itself: an empty reply has no uncited
sentences, so it passed. Empty and citation-free answers now fail.

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
