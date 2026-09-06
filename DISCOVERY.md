# UdaPlay — Discovery, Plan & Execution

**Project:** UdaPlay AI Research Agent — an intelligent gaming research agent that answers
from an internal vector database and falls back to web search when internal knowledge is
absent or low-confidence.

**Source of requirements:** [UdaPlay_Project_Requirements.txt](UdaPlay_Project_Requirements.txt)

**Status:** complete and verified end to end against the Vocareum endpoint. 89 offline tests
green; all three evaluation scenarios route correctly; Streamlit frontend running; DeepEval
scorecard wired.

---

## 1. Problem statement

Executives, analysts and gamers ask questions that fall into three shapes:

| Shape | Example | Where the answer lives |
|---|---|---|
| Settled historical fact | "When was Pokémon Red launched?" | Internal database |
| Present/future state | "What is Rockstar working on right now?" | Only the live web |
| Plausible near-miss | "When was God of War Ragnarök released?" | Web — but the DB holds a *similar* record (God of War, 2018) |

The third shape is the interesting one and the reason a naive RAG pipeline is not enough.
A vector search for "God of War Ragnarök" happily returns God of War (2018) with a decent
similarity score. Answering from it produces a confident, wrong, internally-cited answer.

**So the central design requirement is an explicit rejection step**, not just retrieval. The
system must be able to look at relevant-seeming context and conclude "this is about a
different game" or "this is historical and the question is present-tense".

Measured on the real endpoint, the rejection step works: the Ragnarök query retrieves God
of War (2018) at L2 distance 0.46 — a close match — and `evaluate_retrieval` still scores
it `confidence 0.00, is_sufficient=false`. Similarity alone would have answered "2018".

## 2. Architecture

```
   Streamlit UI (app.py)          CLI (python -m udaplay)
            │                              │
            └──────────────┬───────────────┘
                           ▼
                 ┌───────────────────┐
                 │  validate_query   │  length, control chars, NFKC
                 └─────────┬─────────┘  (rejects before any API call)
                           ▼
                    ┌──────────────┐
   session history ▶│  UdaPlay     │
                    │  agent       │
                    └──────┬───────┘
                           │ 1
                           ▼
                 ┌───────────────────┐
                 │  retrieve_game    │──▶ FAISS index (15 games)
                 │  (+ metadata      │    text-embedding-3-small (1536-d)
                 │   filters)        │
                 └─────────┬─────────┘
                           │ 2  internal context
                           ▼
                 ┌───────────────────┐
                 │ evaluate_retrieval│──▶ gpt-4o-mini, structured output
                 │                   │    {confidence_score,
                 └─────────┬─────────┘     is_sufficient, reasoning}
                           │
          is_sufficient ───┴─── is_sufficient
              == true            == false
                 │                  │ 3
                 │                  ▼
                 │        ┌───────────────────┐   ← turn_state gate:
                 │        │  game_web_search  │     blocked unless step 2 ran
                 │        │                   │──▶ Wikipedia (MediaWiki API)
                 │        └─────────┬─────────┘──▶ DuckDuckGo (ddgs)
                 │                  │
                 └────────┬─────────┘
                          │ 4
                          ▼
                ┌───────────────────┐
                │  validate_answer  │  Source line present? consistent
                └─────────┬─────────┘  with the tools that actually ran?
                          ▼
               Structured final answer
               + Key Details (Platform, Publisher, Release Date)
               + Source: Internal Game Database | Web Search
                          │
                          ▼
               logs/runs.jsonl  (one record per run: tools, confidence,
                                 latency, warnings) ──▶ DeepEval + UI stats
```

The four numbered steps are stated verbatim in the agent's system prompt
([udaplay/agent.py](udaplay/agent.py)), and each tool's docstring restates its position in
the sequence — the model reads those docstrings when choosing what to call, so the ordering
constraint lives where the decision is actually made. The prompt alone proved insufficient
(§5), so the order is additionally enforced in code by
[udaplay/turn_state.py](udaplay/turn_state.py).

## 3. Key decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Python package + CLI**, not notebooks | Testable, diffable, runs the same locally and on a host. `python -m udaplay demo` reproduces the full evaluation run. |
| 2 | **Wikipedia + DuckDuckGo** instead of Tavily | Both keyless and open source. They are complementary: Wikipedia is authoritative for settled facts, DuckDuckGo covers current news. See §5 for the rubric caveat. |
| 3 | **Direct MediaWiki API** instead of the `wikipedia` PyPI package | The package is unmaintained since 2014 and sends no `User-Agent`; Wikipedia now rejects it, surfacing as a bare `JSONDecodeError`. Verified failing before the swap. |
| 4 | **LangChain 0.3.30** pin | Last series where `create_tool_calling_agent` + `AgentExecutor` work as the requirements describe. LangChain 1.x moved them to `langchain-classic`. |
| 5 | `.env` **and** `config.env` both honoured | Requirements name `config.env`; local convention is `.env`. `.env` loads first and therefore wins. |
| 6 | Base URL **auto-detected from the key prefix** | A `voc-` key implies the Vocareum proxy. Same code runs locally, on the host, and against a stock `sk-` key with no edits. |
| 7 | Both `OPENAI_BASE_URL` **and** `OPENAI_API_BASE` are set | The `openai` SDK reads the first, `langchain-openai` reads the second. Setting only one lets a client silently fall through to `api.openai.com` and fail with an opaque 401. |
| 8 | Corpus **deliberately excludes** Ragnarök and anything after 2022 | Makes scenarios 2 and 3 genuine fallback tests. A test asserts the absence so a future dataset edit cannot quietly neuter them. |
| 9 | Source citation derived from the **trace**, not the model's claim | `to_structured_answer` reads which tools actually ran. A model asserting "Source: Internal Database" after a web call cannot produce a wrong citation. |
| 10 | Search backends **never raise** | DuckDuckGo rate-limits with HTTP 429. A demo that dies mid-run is worse than one reporting the outage inline. |
| 11 | Tool order enforced **in code**, not only in the prompt | The prompt was demonstrably ignored — see §5.1. [udaplay/turn_state.py](udaplay/turn_state.py) blocks `game_web_search` until `evaluate_retrieval` has run. |
| 12 | `turn_state` uses a **plain guarded global**, not a `ContextVar` | LangChain runs each tool inside `copy_context()`, so a `ContextVar` write inside a tool is discarded on return and cannot carry state between tool calls. Learned by building it that way first (§5.1). |
| 13 | Validation split **input vs output** | Input checks are cheap and deterministic and run before any paid API call. Output checks are advisory warnings, because a malformed answer is still more useful than an exception. |
| 14 | Two log sinks: rotating text + **JSONL run log** | They answer different questions — "what happened, in order?" versus "how did routing behave across the last N queries?". The JSONL log makes routing measurable after the fact and is what DeepEval and the UI stats read. |
| 15 | DeepEval judge built from the **same resolved config** as the agent | DeepEval defaults to `api.openai.com`, which a `voc-` key cannot reach. `OpenAIModel` accepts `base_url`, so no `DeepEvalBaseLLM` subclass is needed. |
| 16 | Evaluation grades **routing**, not just prose | `ToolCorrectnessMetric` plus deterministic trace assertions. An answer that is right *by luck* after skipping the evaluation step has not met the spec — and this is precisely the bug that §5.1 caught. |

## 4. Execution log

| Phase | Work | Status |
|---|---|---|
| 1 | Discovery, requirement → deliverable mapping | Done |
| 2 | `uv` environment, Python 3.11.15, `requirements.txt` (69 pinned runtime deps) | Done |
| 3 | `.env` handling with `voc-` detection and base-URL normalisation | Done |
| 4 | Pydantic v2 schemas | Done |
| 5 | Seed corpus: 15 games under `data/games/` | Done |
| 6 | Loader, FAISS lifecycle, similarity search with metadata filters | Done |
| 7 | Search backends — both verified live | Done |
| 8 | Three `@tool` functions | Done |
| 9 | `AgentExecutor` + per-session history | Done |
| 10 | CLI — `check`/`index`/`search`/`ask`/`chat`/`ui`/`demo`/`eval`/`logs` | Done |
| 11 | Input & output validation ([udaplay/validation.py](udaplay/validation.py)) | Done |
| 12 | Logging & run records ([udaplay/logging_setup.py](udaplay/logging_setup.py)) | Done |
| 13 | Streamlit frontend ([app.py](app.py)) | Done |
| 14 | Workflow gate ([udaplay/turn_state.py](udaplay/turn_state.py)) — added in response to §5.1 | Done |
| 15 | DeepEval suite ([evals/](evals/)) | Done |
| 16 | Test suite — 89 offline, 10 live | Done |
| 17 | Live end-to-end run against the Vocareum endpoint | Done |

### Verified during build

Each of these was checked by running it, not assumed:

- LangChain 0.3.30 classic agent imports resolve; `deepeval` 4.2.1 pulls in no LangChain
  dependency, so it cannot disturb the pin.
- `DuckDuckGoSearchResults` works against `ddgs` 9.16.0 and returned current GTA VI results.
- `WikipediaAPIWrapper` **fails** (`JSONDecodeError`); the direct MediaWiki call returns
  "released worldwide on November 9, 2022" for Ragnarök — the exact fact scenario 3 needs.
- The Vocareum proxy **does** serve `text-embedding-3-small`, returning 1536 dimensions.
  This was the highest-risk unknown; no fallback embedding model was needed.
- Retrieval ranks correctly: "racing game on PlayStation" → Gran Turismo at L2 0.78,
  Super Mario Kart at 1.21.
- All three scenarios route as specified, with `evaluate_retrieval` running in every case
  and zero output-format warnings.
- Multi-turn state works: "Who published it?" with no game named resolved correctly against
  the previous turn.
- The Streamlit app renders and its script executes without error, verified with
  Streamlit's own `AppTest` harness — curling the port only proves the shell HTML was
  served, since the script runs over a websocket.

## 5. Risks

### 5.1 The one that materialised: a silently skipped workflow step

Worth recording in full, because the failure was invisible to the obvious check.

Asked *"What is Rockstar Games working on right now?"*, the agent called `game_web_search`
**directly** — skipping both `retrieve_game` and `evaluate_retrieval`. The answer was
correct and well-cited, and the routing check passed, because that check compared
"did web search run?" against "should web search run?" and both were `true`. The mandated
two-tier workflow had simply not executed. Only the tool sequence in `logs/runs.jsonl`
revealed it:

```
'What is Rockstar Games working on right now?'   tools=game_web_search   conf=None
```

The model had reasoned, not unreasonably, that a present-tense question obviously cannot be
answered from a static database — so it went straight to the web. The system prompt said
"Never skip step 2". That was not enough.

**Fix:** enforce ordering in code. `game_web_search` now consults
[udaplay/turn_state.py](udaplay/turn_state.py) and, if `evaluate_retrieval` has not run this
turn, returns instructions instead of results. The agent then makes the missing calls and
retries — self-correcting rather than failing.

**And a second bug inside the fix.** The gate was first built on a `ContextVar`. It blocked
*every* call, and the agent looped to `max_iterations`. Cause: LangChain invokes each tool
inside `contextvars.copy_context()`, so a `ContextVar.set()` performed inside a tool is
discarded the moment that tool returns — it cannot carry state *between* tool calls, which
was the entire requirement. Replaced with a lock-guarded module global.

The first version of the gate test passed anyway, because it called `turn_state.record()`
directly rather than from inside a tool invocation — so it never exercised the boundary that
was broken. The test now drives both real tools through `.invoke()` and asserts state
survives from outside.

Two lessons folded into the design: **grade the trace, not the answer**, and treat prompt
instructions as advisory when a hard constraint is required.

### 5.2 Remaining risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Tavily is named in the rubric**; this build uses Wikipedia + DuckDuckGo | A strict grader could dock the "Agent Tools" row | Behaviour (keyless web fallback with citations) is equivalent. `game_web_search` is one function over swappable backends — adding Tavily is a small, localised change. |
| Host may block outbound traffic to non-proxy domains | Web fallback fails on the host though it works locally | `python -m udaplay check` isolates this; backends report the outage rather than crashing |
| DuckDuckGo HTTP 429 under repeated runs | Intermittent missing web results | Retry with backoff; Wikipedia covers settled facts independently; `game_web_search` reports the outage inline instead of raising |
| Host ships LangChain 1.x | Agent imports break | Only [udaplay/agent.py](udaplay/agent.py) is affected; `requirements.txt` pins 0.3.30 |
| **DeepEval judge and agent share `gpt-4o-mini`** | LLM-judged scores are more forgiving than an independent judge would be | Unavoidable — the proxy is the only endpoint. Documented in [evals/judge.py](evals/judge.py); thresholds set at 0.7 rather than 0.9 to avoid a flapping suite. Routing metrics are LLM-free and strict (threshold 1.0), so the hard requirement is graded deterministically. |
| `turn_state` is a process-wide global | Two concurrent turns in one process could see each other's tools | Degrades only toward permissiveness — it can miss an enforcement, never wrongly block a call, so worst case is the prompt-only behaviour. Single-user UI in practice. |
| Conversation history is in-process | Restarting the server loses history | Correct scope for this deliverable; a deployment would back it with Redis or a database |
| Streamlit reruns the script on every interaction | Duplicate log handlers, rebuilt agent per keystroke | `configure_logging` is idempotent (asserted by a test) and the agent sits behind `@st.cache_resource` |
| ~~Vocareum proxy may not serve `text-embedding-3-small`~~ | — | **Resolved:** verified serving, 1536 dimensions |
| ~~LLM ignores the mandated tool order~~ | — | **Materialised and fixed** — see §5.1 |

## 6. Requirement traceability

| Requirement | Implementation | Rubric row |
|---|---|---|
| Load & validate game JSON | [udaplay/data_loader.py](udaplay/data_loader.py), [udaplay/schemas.py](udaplay/schemas.py) | RAG Pipeline |
| Embeddings + FAISS index with persistence | [udaplay/vector_store.py](udaplay/vector_store.py) — `save_local`/`load_local` | RAG Pipeline |
| Similarity search utility + verification | `search()`, `format_results()`; `python -m udaplay search` | RAG Pipeline |
| `retrieve_game` | [udaplay/tools.py](udaplay/tools.py) | Agent Tools |
| `evaluate_retrieval` → confidence/sufficiency/reasoning | [udaplay/tools.py](udaplay/tools.py) + `RetrievalEvaluation` | Agent Tools |
| `game_web_search` | [udaplay/tools.py](udaplay/tools.py) + [udaplay/search_providers.py](udaplay/search_providers.py) | Agent Tools |
| Short-term conversational state | `RunnableWithMessageHistory`, per-`session_id` | Stateful Workflow |
| Two-tier fallback decision logic | System prompt + tool docstrings + [udaplay/turn_state.py](udaplay/turn_state.py) gate | Stateful Workflow |
| Citations distinguishing internal vs web | Prompt rules + trace-derived `SourceOrigin` + `validate_answer` | Stateful Workflow |
| 3 test queries with reasoning traces | [udaplay/demo.py](udaplay/demo.py) — `python -m udaplay demo` | Performance Demonstration |
| *Bonus:* long-term memory | `add_web_facts()` | Bonus |
| *Bonus:* structured outputs | `UdaPlayAnswer`, `to_structured_answer()` | Bonus |
| *Bonus:* metadata filters | `search(platform=…, publisher=…)` | Bonus |
| *Added:* chat frontend | [app.py](app.py) — `python -m udaplay ui` | — |
| *Added:* agent evaluation | [evals/](evals/) — `python -m udaplay eval` | — |
| *Added:* validation | [udaplay/validation.py](udaplay/validation.py) | — |
| *Added:* logging & observability | [udaplay/logging_setup.py](udaplay/logging_setup.py) — `python -m udaplay logs` | — |

## 7. Frontend

[app.py](app.py) is a Streamlit chat app: `python -m udaplay ui` (or `streamlit run app.py`).

- Headline, subheading and a one-paragraph explanation of what the agent does.
- Sample-question buttons on a fresh session — including the two that force the fallback.
- `st.chat_message` / `st.chat_input` conversation, with history preserved across turns via
  the same `session_id` mechanism the CLI uses.
- **A reasoning-trace expander under every answer**, captioned with the routing decision
  ("Web fallback" vs "Internal database only"), the confidence score and the latency, then
  each tool call with its input and output. This is what makes the two-tier decision
  visible rather than a claim.
- Sidebar: masked key and resolved endpoint, corpus size and year range, and running
  session stats (queries logged, web-fallback rate, mean confidence, mean latency) read
  from the JSONL run log.
- Input rejected by `validate_query` is reported in the UI and never reaches the API.

Both expensive resources are cached: the agent and index behind `@st.cache_resource`, the
corpus stats behind `@st.cache_data`. Without that, Streamlit's rerun-on-every-interaction
model would rebuild the agent on each keystroke.

## 8. Evaluation

`python -m udaplay eval` scores the agent; `pytest evals -m live` runs the same scenarios as
assertions for CI. Four scenarios, and for each one two layers:

**Deterministic checks** (free, unambiguous, and the ones that actually gate):
- Routing matches expectation (web fallback fired iff it should have).
- `evaluate_retrieval` was not skipped — the check §5.1 would have needed.
- Output format conforms (`validate_answer` returns no warnings).
- Required literal facts present, e.g. "1996" and "Game Boy" for Pokémon Red.

**DeepEval metrics**, each covering a failure the others miss:

| Metric | Judge | Catches |
|---|---|---|
| `ToolCorrectnessMetric` | none — deterministic | The mandated tool set did not run |
| `FaithfulnessMetric` | LLM | Answer not grounded in retrieved context — the near-miss failure of answering "Ragnarök" from the 2018 record |
| `AnswerRelevancyMetric` | LLM | Answer does not address the question |
| `CitationCorrectness` (custom `GEval`) | LLM | This project's own output contract: `Source:` line present and consistent with the tools that ran, Key Details block present. No off-the-shelf metric covers it. |

The judge is built from the same resolved config as the agent
([evals/judge.py](evals/judge.py)) so it reaches the Vocareum proxy. Judge and agent
therefore share a model — see the risk table; treat LLM-judged scores as a regression
signal, not an absolute quality measure. Routing is the hard requirement and is graded
without an LLM.

## 9. Validation & logging

**Validation** ([udaplay/validation.py](udaplay/validation.py)) guards two boundaries:

- *Input*, before the LLM: NFKC normalisation, whitespace collapsing, control-character
  stripping, and length bounds (3–500 chars). Deterministic and free, so junk never becomes
  a paid API call. Raises `InvalidQuery` with a message written for whoever typed it.
- *Output*, after the answer: is the mandated `Source:` line present, is the `Key Details`
  block there, does the citation match the tools that actually ran, and does a web-sourced
  answer list a URL? Returned as warnings rather than raised — a technically malformed
  answer is still more useful than an exception — and surfaced in the CLI trace, the UI
  expander, and the run log.

The citation cross-check is the valuable one: `used_web` comes from the execution trace, so
an answer claiming "Source: Internal Game Database" after a web call is caught.

**Logging** ([udaplay/logging_setup.py](udaplay/logging_setup.py)) writes two sinks:

- `logs/udaplay.log` — rotating (1 MB × 3), human-readable. Every tool call logs its
  arguments, result size, timing and, for retrieval, the best distance. A degraded search
  backend logs at WARNING.
- `logs/runs.jsonl` — one `RunRecord` per run: query, answer, full tool sequence with
  previews, confidence, `used_web_search`, latency, format warnings, error. This is what
  makes routing measurable after the fact, and it is what caught §5.1.

`python -m udaplay logs` renders recent runs plus aggregates. `configure_logging` is
idempotent, and `log_run` never raises — losing an observability record must not fail the
request it was observing.

## 10. Reproducing

```bash
uv sync --extra dev --extra eval     # or: pip install -r requirements.txt
cp .env.example .env                 # add the voc-... key

uv run pytest                        # 89 offline tests, no key, no network, ~2s
uv run python -m udaplay check       # environment + live endpoint
uv run python -m udaplay index       # build the FAISS index
uv run python -m udaplay demo        # the three scenarios, with traces
uv run python -m udaplay ui          # Streamlit frontend on :8501
uv run python -m udaplay eval        # DeepEval scorecard
uv run python -m udaplay logs        # recent runs + aggregates
uv run pytest -m live                # the 10 deferred live tests
```

Both `demo` and `eval` exit non-zero if routing differs from expectation, so either doubles
as an end-to-end regression check.

### Observed on the verification run

```
[PASS] Scenario 1: internal only (confidence 1.00) - When was Pokemon Red launched…
[PASS] Scenario 2: web fallback  (confidence 0.00) - What is Rockstar Games working on…
[PASS] Scenario 3: web fallback  (confidence 0.00) - When was God of War Ragnarok released?
3/3 scenarios routed as expected.

5 runs, 0 errors, web fallback 40%, mean latency 12,637 ms, 0 format warnings
```

`evaluate_retrieval` ran on all five runs, including both fallback cases — the property
§5.1 was about.
