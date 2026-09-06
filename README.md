# UdaPlay — AI Gaming Research Agent

A research agent that answers video-game questions from an internal FAISS vector database,
and falls back to keyless web search (Wikipedia + DuckDuckGo) when internal knowledge is
missing, off-target or too old to trust.

See [DISCOVERY.md](DISCOVERY.md) for the design rationale and requirement traceability.

**New here? Jump to the [Demo walkthrough](#demo-walkthrough) — 7 steps, about 10 minutes.**

## Layout

```
├── Dockerfile               Cloud Run image (Streamlit frontend)
├── .dockerignore            keeps .env, tests and evals out of the build context
├── app.py                   Streamlit chat frontend
├── udaplay/
│   ├── config.py            env loading, voc- key detection, LLM/embedding clients
│   ├── schemas.py           Pydantic v2 models (validation boundary)
│   ├── data_loader.py       game JSON -> validated records -> Documents
│   ├── vector_store.py      FAISS build / persist / reload / search
│   ├── search_providers.py  Wikipedia + DuckDuckGo backends
│   ├── tools.py             the three @tool functions the agent routes between
│   ├── turn_state.py        enforces the retrieve -> evaluate -> web order
│   ├── validation.py        input checks + output format checks
│   ├── logging_setup.py     text log + JSONL run records
│   ├── agent.py             AgentExecutor + per-session conversation history
│   ├── demo.py              the three evaluation scenarios
│   └── cli.py               python -m udaplay <command>
├── evals/                   DeepEval suite (judge, cases, metrics, scorecard)
├── data/games/              15 seed games, one JSON per game
├── tests/                   82 offline tests + 10 live tests (opt-in)
├── logs/                    udaplay.log (rotating) + runs.jsonl (per-run records)
├── requirements.txt         pinned runtime deps, for pip on a host
└── pyproject.toml           uv project definition
```

## Setup

### With uv (local)

```bash
uv venv --python 3.11
uv sync --extra dev
cp .env.example .env      # then add your key
```

### With pip (hosted JupyterLab or any other host)

Copy the project folder across, then from the project root:

```bash
pip install -r requirements.txt
cp .env.example .env      # then add your key
```

There is no build or install step — commands run as `python -m udaplay` from the project
root.

## Configuring the key

Put the key in `.env` (preferred) or `config.env` — both are read, `.env` wins:

```
OPENAI_API_KEY="voc-298..."
OPENAI_BASE_URL="https://openai.vocareum.com/v1"
```

A Vocareum `voc-…` key is served by Vocareum's own OpenAI-compatible proxy rather than by
`api.openai.com`, so the base URL matters. `OPENAI_BASE_URL` is optional: a `voc-` prefixed
key selects the Vocareum URL automatically. A standard `sk-…` key needs no base URL at all.
The same code therefore runs unchanged locally, on the host, and against a personal OpenAI
key.

No web-search key is needed — Wikipedia and DuckDuckGo are both keyless.

Verify before doing anything else:

```bash
python -m udaplay check              # config + corpus + one live call to each model
python -m udaplay check --offline    # config + corpus only, no API call
```

`check` prints the resolved base URL and a masked key, which is safe to paste into a bug
report.

## Demo walkthrough

Step by step, from a fresh clone to a working agent. Roughly 10 minutes, most of it waiting
on installs. Every command runs from the project root.

### Step 0 — Prerequisites

Python 3.11 or 3.12, and an OpenAI-compatible API key. If you have
[uv](https://docs.astral.sh/uv/) use the uv path; otherwise plain `pip` works everywhere.

```bash
cd /path/to/vibe-coding-setup
```

### Step 1 — Install

```bash
# Option A: uv (recommended)
uv venv --python 3.11
uv sync --extra dev --extra eval

# Option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

With uv, prefix later commands with `uv run`. With pip, activate the venv once and drop the
prefix. The rest of this walkthrough shows the uv form.

**Expected:** installs finish without errors. `--extra eval` adds DeepEval and is only
needed for Step 7.

### Step 2 — Add your key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY="voc-298...your-actual-key"
OPENAI_BASE_URL="https://openai.vocareum.com/v1"
```

Leave `OPENAI_BASE_URL` as-is for a Vocareum `voc-…` key. For a standard `sk-…` key,
comment that line out. No web-search key is needed.

> The placeholder is detected and rejected, so forgetting this step fails with a clear
> message rather than an opaque 401.

### Step 3 — Verify the setup

```bash
uv run python -m udaplay check
```

**Expected:**

```
UdaPlay environment
-------------------------------------------------------
api_key          voc-29...wxyz
key_flavour      vocareum
base_url         https://openai.vocareum.com/v1
chat_model       gpt-4o-mini
embedding_model  text-embedding-3-small
...
Corpus: 15 valid records, 0 problem(s)

Live endpoint check:
[ok]   embeddings 'text-embedding-3-small' -> dim 1536
[ok]   chat 'gpt-4o-mini' -> 'ready'
```

Two `[ok]` lines mean the key, the endpoint and both models work. **Do not continue past a
`[FAIL]` here** — everything downstream depends on it. Common causes:

| Symptom | Cause | Fix |
|---|---|---|
| `401` / `invalid api key` | `voc-` key hitting `api.openai.com` | Make sure `OPENAI_BASE_URL` is set in `.env` |
| `model not found` for embeddings | Proxy doesn't serve that model | Set `OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"` in `.env` |
| `still the placeholder` | Step 2 not completed | Put the real key in `.env` |

Offline check, if you just want to validate config and corpus without spending tokens:

```bash
uv run python -m udaplay check --offline
```

### Step 4 — Build the vector index

```bash
uv run python -m udaplay index
```

**Expected:**

```
Loaded and validated 15 game records.
Indexed 15 documents -> .../faiss_index_udaplay
```

This embeds the 15 games and writes `faiss_index_udaplay/` (`index.faiss` + `index.pkl`).
It is a one-off — later runs reload from disk. Use `--force` to rebuild after editing
`data/games/`.

Sanity-check retrieval on its own, with no LLM involved:

```bash
uv run python -m udaplay search "racing game on PlayStation"
```

**Expected:** Gran Turismo first (L2 distance ≈ 0.78), Super Mario Kart second (≈ 1.21).
Lower distance means a closer match.

### Step 5 — Watch the agent route

This is the interesting part. Run the three scenarios and compare their traces.

```bash
uv run python -m udaplay demo --skip-bonus
```

**Expected** — three scenarios, each printing its tool calls, then:

```
  [PASS] Scenario 1: internal only (confidence 1.00) - When was Pokemon Red launched...
  [PASS] Scenario 2: web fallback  (confidence 0.00) - What is Rockstar Games working on...
  [PASS] Scenario 3: web fallback  (confidence 0.00) - When was God of War Ragnarok released?

3/3 scenarios routed as expected.
```

What to look for in each trace:

- **Scenario 1** calls `retrieve_game` → `evaluate_retrieval`, gets `confidence 1.00`, and
  **stops** — no web call. Cites `Source: Internal Game Database`.
- **Scenario 2 and 3** call `retrieve_game` → `evaluate_retrieval` → `game_web_search`.
  Confidence is `0.00` and `is_sufficient` is `false`, which is what triggers the fallback.
  They cite `Source: Web Search (Wikipedia/DuckDuckGo)` with URLs.
- **Scenario 3 is the one worth reading closely.** Retrieval returns God of War (2018) at a
  close distance — a genuine near-miss — and the evaluator still rejects it because the
  question is about Ragnarök. Plain similarity search would have answered "2018".
- After the scenarios, a **two-turn exchange** shows state: the second question ("Who
  published it, and what platform was it on?") names no game and is answered from history.

Try single questions yourself:

```bash
uv run python -m udaplay ask "When was Gran Turismo released?"
uv run python -m udaplay ask "When was God of War Ragnarok released?" --structured
```

`--structured` also prints the Pydantic `UdaPlayAnswer` as JSON.

`demo` exits non-zero on a misroute, so it doubles as a regression check.

### Step 6 — The chat frontend

```bash
uv run python -m udaplay ui
```

Open <http://localhost:8501>. (On a remote host add `--headless`; change the port with
`--port 8502`.)

**Expected:** the UdaPlay headline, a short explanation, and four sample-question buttons.

Try this sequence to see the whole system in one place:

1. Click **"When was Pokemon Red launched, and on what platform?"** — answers from the
   internal database.
2. Expand **Reasoning trace** beneath the answer. The caption reads
   `Internal database only · confidence 1.00`, and lists each tool call with its output.
3. Ask **"What is Rockstar Games working on right now?"** — the caption now reads
   `Web fallback · confidence 0.00` and a third tool call appears.
4. Type **"What platform was it on?"** — no game named, answered from conversation history.
5. Check the **sidebar**: masked key, resolved endpoint, corpus size, and live session stats
   (queries logged, web-fallback rate, mean confidence, mean latency).

Stop the server with `Ctrl-C`.

### Step 7 — Tests, logs and evaluation

```bash
# 82 offline tests: no key, no network, ~2 seconds
uv run pytest

# Every run so far, with its tool sequence and timings
uv run python -m udaplay logs

# Live tests: needs the key, the index and network
uv run pytest -m live
```

`logs` is the quickest way to see routing behaviour across runs:

```
ok  2026-09-05T18:31:02+00:00  int  conf=1.00    9484ms  'When was Pokemon Red launched...'
ok  2026-09-05T18:31:20+00:00  web  conf=0.00   17447ms  'What is Rockstar Games working on...'

5 run(s), 0 error(s), web fallback 40%, mean latency 12637 ms, 0 format warning(s)
```

Finally, the DeepEval scorecard — four scenarios × four metrics, judged by an LLM:

```bash
uv run python -m udaplay eval                          # all scenarios
uv run python -m udaplay eval --scenario internal_match  # just one
```

**This costs the most tokens of anything here** (it runs the agent, then three LLM-judged
metrics per scenario) and takes several minutes. Read the `routing` column first: it is
deterministic and is the actual requirement. The LLM-judged columns are a regression signal
— judge and agent share `gpt-4o-mini`, so treat them as indicative, not authoritative. See
[Known limitations](#known-limitations).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `No FAISS index at …` | Run Step 4 |
| `[DuckDuckGo] Unavailable … Likely rate limited` | Expected under repeated runs; Wikipedia still answers. Wait a minute and retry |
| Both search backends unavailable | Outbound network is blocked; the RAG path still works |
| `ModuleNotFoundError: udaplay` | Run from the project root, not a subdirectory |
| Streamlit port already in use | `python -m udaplay ui --port 8502` |
| `could not import the eval suite` | `uv sync --extra eval` |

## Deploying to Cloud Run

[Dockerfile](Dockerfile) at the repo root builds the Streamlit frontend. This is what Cloud
Run's "Build Type: Dockerfile / `/Dockerfile`" option expects, so no further configuration
is needed on that screen.

### Deploy

```bash
gcloud run deploy udaplay \
  --source . \
  --region europe-west1 \
  --memory 2Gi \
  --allow-unauthenticated \
  --session-affinity \
  --set-env-vars OPENAI_BASE_URL=https://openai.vocareum.com/v1 \
  --set-secrets OPENAI_API_KEY=udaplay-openai-key:latest
```

Create the secret first:

```bash
printf 'voc-your-key-here' | gcloud secrets create udaplay-openai-key --data-file=-
gcloud secrets add-iam-policy-binding udaplay-openai-key \
  --member="serviceAccount:$(gcloud projects describe "$(gcloud config get-value project)" \
      --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

If you deploy from the console instead, set `OPENAI_API_KEY` and `OPENAI_BASE_URL` under
**Variables & Secrets**, and raise memory under **Container**.

### Flags that matter, and why

| Flag | Why |
|---|---|
| `--memory 2Gi` | The default 512 MiB **will OOM**. `faiss` + `langchain` + `streamlit` is a heavy import graph, and Cloud Run counts the in-memory filesystem against the same limit. 1 GiB usually works; 2 GiB has headroom |
| `--session-affinity` | Streamlit is a stateful websocket app. Without affinity a reconnect can land on a different instance and the session resets mid-conversation |
| `--set-secrets` for the key | Keeps the key out of the image and out of `gcloud` shell history. `--set-env-vars` would work but is worse |
| `--min-instances 1` (optional) | Each cold start rebuilds the FAISS index (15 embedding calls, a few seconds). Keeping one warm avoids that, at the cost of always-on billing |

`OPENAI_BASE_URL` is technically optional — a `voc-` prefixed key selects the Vocareum
proxy automatically — but setting it explicitly makes the deployment self-documenting.

### How the container differs from local

- **The index is built at startup, into `/tmp`.** It is not baked into the image, because
  that would require the API key at build time. First use costs 15 embedding calls.
- **Writes go to `/tmp`** via `UDAPLAY_INDEX_DIR` and `UDAPLAY_LOG_DIR`. These are read at
  import time, so they must be real environment variables — putting them in `.env` will not
  work.
- **Logs are ephemeral.** `/tmp` dies with the instance, so `runs.jsonl` does not persist
  across restarts and the sidebar's session stats reset. For durable logs, write to Cloud
  Logging or mount a GCS volume at `UDAPLAY_LOG_DIR`.
- **`tests/` and `evals/` are not in the image** (see [.dockerignore](.dockerignore)) — the
  frontend does not need them. Run the eval suite locally.

### Verified locally before deploying

The image was built and exercised with Docker, not just written:

```bash
docker build -t udaplay:test .
docker run -d -e PORT=9090 -p 9091:9090 udaplay:test
curl localhost:9091/_stcore/health        # -> ok
```

Checked: `import faiss` succeeds (needs `libgomp1`, which `python:slim` omits — without it
the image builds and then dies at import); Streamlit binds `0.0.0.0:$PORT` rather than
localhost; `$PORT` is honoured, not hardcoded; `streamlit` runs as PID 1 so Cloud Run's
SIGTERM reaches it (container stops in ~1s); the key is read from the environment with no
`.env` present; and the index builds into `/tmp` as non-root uid 1000.

## Usage reference

```bash
# Build the FAISS index from data/games (writes faiss_index_udaplay/)
python -m udaplay index
python -m udaplay index --force              # rebuild in place

# Raw similarity search -- no LLM, no agent
python -m udaplay search "racing game on PlayStation"
python -m udaplay search "open world" --publisher Rockstar -k 3

# One question through the full agent, with the reasoning trace
python -m udaplay ask "When was Pokemon Red launched, and on what platform?"
python -m udaplay ask "When was God of War Ragnarok released?" --structured
python -m udaplay ask "..." --quiet          # answer only

# Interactive multi-turn session (history persists across turns)
python -m udaplay chat

# Streamlit chat frontend
python -m udaplay ui
python -m udaplay ui --port 8502 --headless

# The three evaluation scenarios, end to end
python -m udaplay demo
python -m udaplay demo --skip-bonus          # omit the bonus demonstrations

# Observability and evaluation
python -m udaplay logs -n 50                 # recent runs + aggregates
python -m udaplay eval                       # DeepEval scorecard
```

## The three scenarios

`python -m udaplay demo` runs these and checks the routing:

| # | Query | Expected path |
|---|---|---|
| 1 | "When was Pokémon Red launched, and on what platform?" | Internal DB hit → evaluation passes → **no** web call → internal citation |
| 2 | "What is Rockstar Games working on right now?" | DB has only historical Rockstar titles → evaluation fails → web fallback |
| 3 | "When was God of War Ragnarök released?" | DB has God of War (2018) but not Ragnarök → near-miss rejected → web fallback |

The corpus deliberately stops at 2022 and omits Ragnarök so scenarios 2 and 3 exercise the
fallback for real. `demo` asserts on which tools actually ran, not on the wording of the
answer, and exits non-zero on a misroute.

## Tests

```bash
uv run pytest              # 82 offline tests: no key, no network, ~2s
uv run pytest -m live      # 10 live tests: needs a key, network, and a built index
uv run pytest evals -m live   # DeepEval scenarios as assertions (costs tokens)
```

Live tests are deselected by default so the suite stays runnable in any environment.

## Known limitations

- **The DeepEval judge shares a model with the agent** (`gpt-4o-mini`), because the
  Vocareum proxy is the only endpoint available. A model grading its own output is more
  forgiving than an independent judge, and in practice it is also noisy: on a verification
  run it scored a correct, correctly-cited answer 0.32 on citation and 0.60 on relevancy,
  with reasons that contradicted the actual trace. Read the deterministic `routing` and
  `format` columns as the real result; treat the LLM-judged scores as a regression signal.
- **The DeepEval scorecard has not been re-verified end to end** after the most recent
  metric changes (supplying citation ground truth from the trace, and scoring relevancy and
  faithfulness on the prose rather than the mandated structured blocks). Those two changes
  took `internal_match` from 2/4 to 4/4 and fixed `metadata_query` routing when run on
  those two scenarios, but the full four-scenario re-run was stopped before it finished.
  Run `python -m udaplay eval` to get a current scorecard.
- **Tavily is not used**, though the project requirements name it. Wikipedia + DuckDuckGo
  is behaviourally equivalent and keyless, but a strict reading of the rubric may expect
  Tavily specifically. `game_web_search` is a single function over swappable backends.
- **Corpus-inventory questions are inherently approximate.** "Which racing games do you
  have?" is answered from top-k retrieval, which cannot prove it found every match. The
  evaluator is instructed not to route such questions to the web, since the web cannot
  report the contents of this database.
- **`turn_state` is process-wide**, not per-session. Concurrent turns in one process could
  observe each other's tool calls. That only ever makes the ordering gate more permissive,
  never wrongly blocking a call.

## Notes

- **FAISS scores are L2 distances — lower is closer.** `format_results` labels them to
  avoid the obvious misreading.
- **Metadata filters are substring and case-insensitive**, so `--publisher Sony` matches
  "Sony Interactive Entertainment". They are applied after retrieval, so the search
  over-fetches candidates to still return `k` matches.
- **`load_index` uses `allow_dangerous_deserialization=True`**, required because FAISS
  stores its docstore as a pickle. Safe only for an index you built yourself.
- **Conversation history is in-process**, keyed by `session_id` — the right scope for a CLI
  demo. A deployment would back it with Redis or a database.
- **DuckDuckGo rate-limits.** Repeated runs may return HTTP 429; the backend retries with
  backoff and then reports the outage inline rather than raising. Wikipedia covers settled
  facts independently.
- **Wikipedia is queried through the MediaWiki API directly**, not the `wikipedia` PyPI
  package. That package is unmaintained since 2014 and sends no `User-Agent`, which
  Wikipedia now rejects — it surfaces as a bare `JSONDecodeError`.
- **Tool order is enforced in code**, not just in the prompt. `game_web_search` refuses to
  run until `evaluate_retrieval` has, returning instructions so the agent self-corrects.
  Without this the model skipped the evaluation step on present-tense questions; see
  §5.1 of [DISCOVERY.md](DISCOVERY.md).
- **Logs are written to `logs/`** — `udaplay.log` (rotating, human-readable) and
  `runs.jsonl` (one record per run). Both are gitignored.
