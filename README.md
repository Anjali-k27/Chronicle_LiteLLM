# Chronicle

Chronicle is a personal-data analyst built on a 5-agent LangGraph swarm
(Ingestion → Pattern → Timeline → Brutality → Synthesis) that reads across
Spotify, finance, fitness, GitHub, and journal data to produce an honest,
non-sugar-coated brief about what your own data says about you.

**This README lives at the repo root. The actual application — every
file and command mentioned below — is one level down, in the `chronicle/`
subdirectory.** That split matters most in §2 (cloning) — read it
carefully, it's the #1 place people get lost.

On top of the swarm sits a production infrastructure stack built up across
this course:

| Layer | File | What it does |
|---|---|---|
| Gateway | `api.py` | FastAPI app — sync/async/streaming analysis endpoints |
| Agent swarm | `agent.py` | LangGraph graph, 5 agent nodes, MCP tool calls mid-reasoning |
| MCP servers | `mcp_servers/` | 5 standalone data-source servers (Spotify, finance, fitness, GitHub, journal) |
| Tracing | `otel_setup.py` | OpenTelemetry → Arize Phoenix |
| Judge | `judge_pipeline.py` | LLM-as-judge grading (tool correctness, honesty, PII leaks) |
| Monitoring | `monitoring_daemon.py` | SRE-style SLOs + tripwires over the trace stream |
| Semantic cache | `semantic_cache.py` | Embedding-based response cache in front of the swarm |
| Model router | `model_router.py` | Dynamic per-agent model tiering + virtual-key budget tracking |

---

## 0. Read this first (the 5 mistakes everyone makes)

If you only read one section, read this one.

1. **There are two `cd`s, not one.** Cloning the repo puts you in the
   repo root — but every command below needs to run one level deeper,
   inside `chronicle/`. So it's always: `cd <repo-you-cloned>` then
   `cd chronicle`, not just one or the other. Every command below
   assumes your terminal's current directory *is* `chronicle/` — the one
   with `agent.py`, `api.py`, `Dockerfile` in it, **not** the one with
   just this `README.md` and a `chronicle/` folder in it. If a command
   says "file not found" or "No such file or directory", this is almost
   always why. Run `pwd && ls` to check — if `ls` doesn't show `agent.py`,
   you're one `cd chronicle` short.
2. **`.env.example` is a template, not your config.** Copying it is not
   enough — you must open `.env` afterward and paste in a *real* Gemini
   key. If you skip this, everything fails with
   `GEMINI_API_KEY environment variable is not set` or a 401 from Google.
3. **Don't wrap the key in quotes, and don't leave a trailing space.**
   Correct: `GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
   Wrong: `GEMINI_API_KEY="AIzaSy..."` or `GEMINI_API_KEY = AIzaSy...`
   A real Gemini key starts with `AIza`. If yours doesn't, you copied the
   wrong thing.
4. **Docker Desktop (or the Docker daemon) must actually be running**
   before `docker compose` commands — not just installed. If every
   `docker` command errors with something like `Cannot connect to the
   Docker daemon`, open Docker Desktop and wait for it to say "running."
5. **Pick ONE of §4a or §4b, not both at once.** They both want ports
   6006 and 8000. If you've run one and want to try the other, stop the
   first one first (`Ctrl+C` for §4a, `docker compose down` for §4b) —
   see §8 if a port still won't free up.

## 1. Prerequisites

Check each of these actually returns something before moving on —
don't assume:

```bash
python3 --version   # must print 3.11 or higher (3.12 confirmed working)
git --version        # any recent version
docker --version && docker compose version   # only needed for §4b
```

- A free Gemini API key — go to [aistudio.google.com](https://aistudio.google.com),
  click "Get API key" → "Create API key". It's free, no credit card.

## 2. Clone the repo

Copy-paste this whole block, don't split it up — it clones, then moves
straight into the actual app directory in one go:

```bash
git clone <your-fork-or-repo-url> chronicle-app
cd chronicle-app/chronicle
```

That's **two directories deep** from wherever you ran `git clone`:
`chronicle-app/` (the repo you just cloned, holding this `README.md`)
→ `chronicle/` (the actual app — every file and command in the rest of
this guide lives here).

**Check it worked — run this exact command:**

```bash
pwd && ls agent.py api.py Dockerfile requirements.txt
```

You should see all four filenames echoed back with no `No such file or
directory` errors, and `pwd` should end in `.../chronicle-app/chronicle`
(not `.../chronicle-app`). If `ls` errors on any of those four files,
you stopped one directory too early — run `cd chronicle` one more time
and re-check.

**Common mistake:** running `cd chronicle-app` and stopping there,
then trying to run `./run_dev.sh` or `python agent.py` — those files
aren't in `chronicle-app/`, they're in `chronicle-app/chronicle/`. If
you see "No such file or directory" for a file you're 100% sure exists
in this repo, this is why — you're one level too shallow.

(If you received this as a plain directory rather than a git remote —
someone handed you a folder instead of a URL — skip `git clone`. Find
where that folder's `README.md` (this file) sits, then `cd` from there
into its `chronicle/` subdirectory the same way, and run the same check
above.)

## 3. Configure your environment

```bash
cp .env.example .env
```

Now **open `.env` in an editor** (not just run the command and move on —
you have to actually edit the file) and replace the placeholder line:

```
GEMINI_API_KEY=your_actual_key_here
```

with your real key, no quotes, no extra spaces:

```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Save the file. **Check it worked:**

```bash
grep GEMINI_API_KEY .env
```

should print your real key, not `your_actual_key_here`, and it should
start with `AIza`. If it still says `your_actual_key_here`, you saved to
the wrong file or forgot to save.

`.env` is git-ignored — it will never accidentally get committed as long
as you don't rename it.

> **Note on model choice:** `model_router.py`'s frontier tier points at
> `gemini-pro-latest`, not a pinned `gemini-2.5-pro` — Google deprecated
> `gemini-2.5-pro` for new API keys (`404: no longer available to new
> users`). If your key predates that cutoff and you'd rather pin a
> specific model, change `PRIMARY_MODEL` in `model_router.py`, but verify
> it against `GET https://generativelanguage.googleapis.com/v1beta/models?key=...`
> first — a stale pin fails silently (empty `honest_analysis`/`final_brief`,
> not an exception) because the router's HTTP shim swallows non-2xx bodies.

## 4. Run it — pick ONE of these two ways

### 4a. Local, one-shot (`run_dev.sh`) — recommended for development

Brings up everything in the right order: venv, the 5 MCP servers
(ports 3001–3005), Phoenix (port 6006), then the API + UI (port 8000),
in the foreground.

```bash
./run_dev.sh
```

If you get `Permission denied`, the script isn't marked executable —
run `chmod +x run_dev.sh` once, then try again.

First run creates `.venv` and installs `requirements.txt` for you. If
`.env` doesn't exist yet it copies `.env.example` and **stops** — this is
expected, it's waiting for you to do §3, then re-run. `Ctrl+C` stops
everything this script started. Re-running is safe; it skips any port
already listening.

The very first run can take up to ~90s waiting on Phoenix specifically —
a cold `.venv` has no bytecode cache yet, and Phoenix's first import
(pandas/sqlalchemy/pyarrow/scikit-learn) is noticeably slower than every
run after. Every subsequent run is warm and comes up in a few seconds.
Don't Ctrl+C it during this wait thinking it's hung — watch for either
`✓ MCP servers + Phoenix are up.` or an explicit `✗` error line.

**Check it worked:** you should see this line and the terminal should
stay open (it's running in the foreground):

```
→ Starting Chronicle API + UI on :8000 (Ctrl+C to stop everything)...
```

Now open **http://localhost:8000** in your actual browser for the UI,
and **http://localhost:6006** for the Phoenix trace viewer. Then go do
§5 below.

### 4b. Docker Compose

```bash
docker compose build
docker compose up -d
```

This starts three containers:

| Service | Image | Port | Notes |
|---|---|---|---|
| `phoenix` | `arizephoenix/phoenix:latest` | 6006 (UI/OTLP-http), 4317 (OTLP-grpc) | `api` waits on its healthcheck before starting |
| `api` | built from `Dockerfile` | 8000 | The FastAPI gateway + LangGraph swarm |
| `ui` | `nginx:alpine` | 8080 | Serves `index.html` standalone (same page `api` also serves at `/`) |

The `api` container reads `.env` via `env_file` and overrides
`PHOENIX_COLLECTOR_ENDPOINT` to `http://phoenix:6006/v1/traces` (service
name, not `localhost`, since it's on the compose network).

**This does not run the 5 MCP servers as containers** — same as every
prior session in this repo. `MCPClientPool` inside the `api` container
will silently fall back to the built-in calibration dataset instead of
hitting live MCP data. `./run_dev.sh` (§4a) is the only path confirmed
working end-to-end with live MCP data. If you need both Docker and live
MCP, run the 5 servers separately (`bash mcp_servers/start_all.sh`) and
point `MCP_SOURCE_CONFIG` in `agent.py` at `host.docker.internal:<port>`
instead of `localhost:<port>`.

**Check it worked:**

```bash
docker compose ps
```

All three services (`phoenix`, `api`, `ui`) should say `Up` (phoenix
should say `Up ... (healthy)`). Then:

```bash
curl http://localhost:8000/health
```

should return JSON starting with `{"status":"ok","session":"14.2",...}` —
not a connection-refused error. If `api` isn't `Up`, run
`docker compose logs api` and read the last 20 lines before doing
anything else.

Now open **http://localhost:8000** (or **:8080**) in your browser, and
**http://localhost:6006** for Phoenix. Then go do §5 below.

Tear down when you're done:

```bash
docker compose down
```

## 5. UI verification checklist

Do this after §4, with the app open in an actual browser tab. Every item
below maps to a real element on the page — if something doesn't match,
that's your bug report.

**A. Page loads at all**
- [ ] http://localhost:8000 loads without a blank page or browser error
- [ ] Top-left shows the `Chronicle.` logo; top-right shows a session
      pill reading **Session 14.2** and "LiteLLM Dynamic Model Router"
- [ ] The session-progress strip near the top shows `S14.2 Model Router`
      as the active (🟢) step, with everything before it checked (✅)
- [ ] Left side: an "Analyst Chat" panel with a greeting message from
      "Chronicle" already in it

**B. Dashboard cards auto-populate (give it ~2 seconds after load)**

None of these need a click — they fetch on page load. If any of them
say `API offline — start python api.py`, the page can't reach the
backend: confirm the API is actually running (§4) and that you're not
blocked by a browser extension / ad-blocker on `localhost` requests.

- [ ] **Agent Status** — lists all 5 agents
- [ ] **VRAM Budget** — click FP16 / INT8 / INT4 / FP32 and confirm the
      numbers actually change between them
- [ ] **Tiered VRAM Budget** — rows filled with GB numbers
- [ ] **Monthly GPU Cost** — rows filled with `$` amounts
- [ ] **vLLM Deployment Config** — rows filled
- [ ] **OOM Safety Check** — rows filled, should read as safe
- [ ] **MCP Data Connectors** — 5 rows (spotify/finance/fitness/github/journal)
- [ ] **Gateway** — Session, Version, Uptime (ticking up), Graph = `compiled ✓`
- [ ] **Semantic Cache** — Hits/Misses/Hit Rate/USD saved show `0` (not `—`,
      that's fine on a fresh start — `—` stuck forever means the fetch failed)

**C. A real end-to-end analysis (the test that actually matters)**
- [ ] Click one of the "Try:" example chips (e.g. "🎵 Spotify · mood") —
      the input box fills with a real question
- [ ] Click **Send** — this streams via SSE under the hood
- [ ] **Live Event Timeline** fills in order: `stream_start` →
      5× `agent_handoff` (one per agent, in order: Ingestion, Pattern,
      Timeline, Brutality, Synthesis) → occasional `tool_call`/`tool_result` →
      `final_answer`
- [ ] **Agent Status** highlights each agent as it becomes active, in order
- [ ] **Inference Metrics** populates: Wall time, TTFT, Min/Max TTFT, Success
- [ ] A new message appears in the chat with actual brief text — not
      empty, not a raw error/stack trace
- [ ] This takes roughly 60–90 seconds for a real 5-agent run against
      live Gemini — that's normal, not a hang. If it's stuck past 3
      minutes with nothing new in the Live Event Timeline, something's
      actually wrong — check the terminal running the API for errors.

**D. The async path**
- [ ] Type a different question, click **Send Async** instead of Send
- [ ] An **Async Job Queue** card appears (it's hidden until first use)
      showing a Job ID, and status moving `queued` → `processing` →
      `completed`, with the active node updating and poll count rising
- [ ] Once `completed`, the result text appears in that card

**E. The semantic cache is actually caching**
- [ ] Re-send the *exact same* question you already sent in step C (or a
      close paraphrase, e.g. same question with one word changed)
- [ ] It should return in well under a second, instead of the ~60–90s a
      fresh question takes
- [ ] The **Semantic Cache** card's Hits counter increments and USD saved
      goes up from `0`

**F. Tracing is wired up**
- [ ] Open http://localhost:6006 (Phoenix) in a new tab
- [ ] A project (default name: `default`) exists with spans from the
      requests you just made — look for span names like
      `chronicle.ingestion`, `chronicle.pattern`, `chronicle.brutality`,
      `chronicle.synthesis`, `llm.call`

**G. Direct API sanity check (optional — proves the UI isn't hiding a broken backend)**
```bash
curl http://localhost:8000/health        # "status":"ok", "session":"14.2"
curl http://localhost:8000/health/live   # {"status":"alive"}
curl http://localhost:8000/health/ready  # {"status":"ready","graph":true,"mcp":true}
```

**If every box above is checked, the entire stack — swarm, router,
cache, tracing, UI — is confirmed working end-to-end.** If something
fails partway through, note exactly which lettered step it was and check
§8 (Troubleshooting) — the section headers there line up with the most
common failure points.

## 6. Automated verification tests

Every infrastructure file also carries its own standalone verification
suite — no test framework needed, just run the file. Each prints a
pass/fail table and exits non-zero on failure. Run these from inside
`chronicle/`, with your venv active:

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)   # loads GEMINI_API_KEY into the shell

python semantic_cache.py       # 6/6 — cache freshness, policy versioning, cost model
python judge_pipeline.py       # LLM-as-judge against a seeded trace (live Gemini call)
python model_router.py         # 8/8 — tier routing, virtual keys, FinOps ledger, live routing demo
python monitoring_daemon.py    # 6/6 — SLO tripwires + a synthetic degradation demo
python agent.py                 # 18/18 — full LangGraph swarm, OTel, async job queue, judge (live, ~3–4 min)
```

Every line above should end with something like `✓ Session X.Y
VERIFIED` or a table showing all checks passed. If a run ends with a ✗
and `Fix failing checks before proceeding`, read the specific check that
failed — the note under it usually says exactly what mismatched.

`agent.py`'s suite makes real Gemini calls through the full 5-agent graph
twice (once via `chronicle_stream_events()`, once via the async job
lifecycle) plus a judge call — it's the slowest at a few minutes.
Everything else finishes in seconds except its own live-call sections.

If Phoenix isn't running when you run these standalone, you'll see
`Transient error ... Connection refused` lines from the OTel exporter —
harmless (spans just don't get exported) and does not fail any check.
Start Phoenix first (`python -m phoenix.server.main serve`, or use
`./run_dev.sh` / Docker Compose) to avoid the noise.

## 7. API reference

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Serves the UI (`index.html`) |
| GET | `/health` | Full status: agents, VRAM/cost model, cache stats, capabilities |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (graph compiled, MCP pool built) |
| POST | `/analyze` | Synchronous — blocks until the full swarm finishes (~60–90s) |
| POST | `/analyze/stream` | SSE stream of agent handoffs, tool calls, and the final brief |
| POST | `/analyze/async` | 202 Accepted — returns immediately, poll for the result |
| GET | `/analyze/jobs/{job_id}` | Poll an async job's status/result |
| GET | `/vram-budget`, `/vram-budget/tiered`, `/cost-model`, `/survivability`, `/calibration-stats`, `/deployment-config`, `/oom-check`, `/concurrency-table` | Infrastructure-planning endpoints from earlier sessions (VRAM sizing, GPU cost projections, quantization survivability) |

Example:

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does my data say about my work-life balance?", "data_sources": ["spotify","github","journal"]}'
```

## 8. Project structure

```
chronicle-app/                    # ← repo root: where `git clone` puts you
├── README.md                     # ← this file
└── chronicle/                    # ← where you `cd` to for every command above
    ├── agent.py                  # LangGraph swarm: 5 agent nodes + graph builder
    ├── api.py                    # FastAPI gateway
    ├── model_router.py           # Dynamic model tiering + virtual-key budgets (Session 14.2)
    ├── semantic_cache.py         # Embedding cache in front of the swarm (Session 14.1)
    ├── monitoring_daemon.py      # SLOs + tripwires over the trace stream
    ├── judge_pipeline.py         # LLM-as-judge grading
    ├── otel_setup.py             # OTel tracer + Phoenix exporter config
    ├── job_store.py              # In-memory async job state machine
    ├── stream_schemas.py         # Typed SSE event schemas
    ├── mcp_servers/              # 5 standalone MCP data-source servers
    ├── index.html                # Chronicle UI
    ├── run_dev.sh                # One-shot local dev launcher
    ├── Dockerfile                # api image
    ├── docker-compose.yml        # phoenix + api + ui stack
    ├── requirements.txt
    ├── .env.example
    └── .gitignore
```

## 9. Troubleshooting

Match the exact error text below, top to bottom — most issues are one
of these.

- **`GEMINI_API_KEY environment variable is not set`** — `.env` exists
  but wasn't loaded into your current shell. Either use `run_dev.sh`
  (loads it for you), or run `export $(grep -v '^#' .env | xargs)`
  yourself before the command that failed. Opening a *new* terminal tab
  resets this — you have to re-export there too.
- **`your_actual_key_here` shows up anywhere in an error** — you copied
  `.env.example` to `.env` but never actually edited `.env` (or edited
  the wrong file). Go back to §3.
- **401 / "API key not valid" from Google** — the key was pasted with
  quotes, extra whitespace, or is simply wrong/revoked. Re-copy it fresh
  from [aistudio.google.com](https://aistudio.google.com), no quotes.
- **Empty `honest_analysis`/`final_brief` with no exception raised** — the
  concrete Gemini model behind a logical tier in `model_router.py`
  returned a non-2xx response; the shim swallows the error body into an
  empty string. Check the model name is current (§3 note above).
- **`CERTIFICATE_VERIFY_FAILED` on macOS** — already handled: every
  outbound Gemini call in this repo builds its `aiohttp` session with an
  explicit `ssl.create_default_context(cafile=certifi.where())` context
  rather than relying on the system default. If you still hit this,
  you're likely running code outside this repo's pattern.
- **`Permission denied` running `./run_dev.sh`** — `chmod +x run_dev.sh`
  once, then re-run.
- **`Cannot connect to the Docker daemon`** — Docker Desktop isn't
  actually running, just installed. Open it and wait for the whale icon
  to show "running" before retrying `docker compose` commands.
- **Port already in use** — `run_dev.sh` skips any of 3001–3005/6006/8000
  it finds already bound (safe to re-run); Docker Compose will fail
  outright if 6006, 8000, or 8080 are taken on the host. Find and stop
  whatever's holding the port (`lsof -i:8000` on macOS/Linux), or don't
  run §4a and §4b at the same time (see §0, mistake #5) — they both want
  the same ports.
- **`✗ Port 6006 never came up`** on the very first `run_dev.sh` run —
  cold-start timing, not a real failure; see the note at the end of §4a.
  Just re-run `./run_dev.sh` — the second run is warm and comes up fast.
- **UI shows "API offline — start python api.py" on every dashboard card** —
  the page loaded but can't reach the backend. Confirm the API is
  actually running and listening on port 8000 (`curl http://localhost:8000/health`
  from a terminal), and that you're viewing the page at `http://` not
  `https://`.
- **Docker `api` container can't reach live MCP data** — expected; see
  the note in §4b. Fallback to the calibration dataset is silent and by
  design (`sources_live` in the `/analyze` response tells you which).
