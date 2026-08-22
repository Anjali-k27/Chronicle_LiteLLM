"""
model_router.py — Chronicle Dynamic Model Router
Session 14.2. Step 11 of the production infrastructure build.

Sits between semantic_cache.py (14.1) and the LangGraph swarm (agent.py).
Cache hits: zero cost — router never called.
Cache misses: router assigns each Chronicle agent the cheapest model that
              clears its quality bar.

Consumes:
  - LOGICAL_MODEL_MAP to resolve tier names to concrete Gemini models
  - VirtualKeyRegistry for per-key budget tracking
  - LiteLLMShim as the in-process proxy simulator
  - ChronicleState from agent.py (routing_decision attribute from S13.1)

Produces:
  - routing_model OTel span attribute on every LLM call (consumed by S13.3)
  - per-agent spend ledger (the CFO number, per agent)

Session 14.3 extension:
  - model_router.py gains a FallbackShim subclass for cascade routing
  - Nothing else in this file changes in 14.3.
"""

import asyncio
import hashlib
import json
import os
import ssl
import time
from typing import Any, Dict, List, Optional

import aiohttp
import certifi
import google.generativeai as genai
from pydantic import BaseModel

# ── API config ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# ── SECTION 1: Logical Model Map ───────────────────────────────────────────
# Left side: logical names the Chronicle agents use.
# Right side: concrete Gemini models that back them in this environment.
# In production config.yaml: right side is provider/model-string.
# Swapping providers = editing the right side only. App code unchanged.

PRIMARY_MODEL = "gemini-pro-latest"  # frontier tier — "gemini-2.5-pro" now 404s
                                      # ("no longer available to new users") on
                                      # the Gemini API; the -latest alias is the
                                      # stable pointer Google recommends instead.
UTILITY_MODEL = "gemini-2.5-flash"  # utility tier

LOGICAL_MODEL_MAP: Dict[str, str] = {
    "frontier-model":       PRIMARY_MODEL,
    "mid-model":            PRIMARY_MODEL,   # collapse mid → frontier for simplicity
    "utility-model":        UTILITY_MODEL,
    "local-utility":        UTILITY_MODEL,   # local vLLM from Week 11 (simulated here)
    "semantic-classifier":  UTILITY_MODEL,
}

# ── SECTION 2: Chronicle Agent → Tier Assignment ───────────────────────────
# Each Chronicle agent from agent.py is assigned a logical tier.
# This is the right-sizing decision — one line per agent.
# Change a tier: edit here. No agent code changes.

AGENT_TIER_MAP: Dict[str, str] = {
    "ingestion":  "utility-model",   # MCP data fetch — no LLM reasoning needed
    "pattern":    "utility-model",   # correlation classification — regex-adjacent
    "timeline":   "utility-model",   # sequencing — statistical, not complex
    "brutality":  "frontier-model",  # multi-source honest analysis — needs frontier
    "synthesis":  "frontier-model",  # final brief composition — needs frontier
}

# ── SECTION 3: Intelligence Tax Calculation ────────────────────────────────

class ModelRate(BaseModel):
    """Per-million-token rates for one logical tier."""
    tier:               str
    input_per_mtok_usd: float
    output_per_mtok_usd: float


TIER_RATES: Dict[str, ModelRate] = {
    "frontier": ModelRate(tier="frontier", input_per_mtok_usd=15.00,  output_per_mtok_usd=60.00),
    "mid":      ModelRate(tier="mid",      input_per_mtok_usd=3.00,   output_per_mtok_usd=15.00),
    "utility":  ModelRate(tier="utility",  input_per_mtok_usd=0.15,   output_per_mtok_usd=0.60),
}


def cost_per_ticket_usd(logical_model: str, avg_in_tok: int = 500, avg_out_tok: int = 200) -> float:
    """
    What it does:   Compute USD cost for one Chronicle ticket on a given logical tier.
    When called:    By print_intelligence_tax_table() and by the verification test.
    Introduced:     Session 14.2. Permanent.
    """
    is_frontier = LOGICAL_MODEL_MAP.get(logical_model, "") == PRIMARY_MODEL
    rate        = TIER_RATES["frontier"] if is_frontier else TIER_RATES["utility"]
    return (avg_in_tok / 1_000_000) * rate.input_per_mtok_usd + \
           (avg_out_tok / 1_000_000) * rate.output_per_mtok_usd


def print_intelligence_tax_table() -> None:
    """
    What it does:   Print per-agent tier assignment and cost per ticket.
                    This is the table you present at architecture review.
    Introduced:     Session 14.2. Permanent.
    """
    print("\nINTELLIGENCE TAX TABLE — Chronicle Agent Tier Assignment")
    print("=" * 72)
    print(f"  {'Agent':<14} {'Logical Tier':<22} {'Concrete Model':<28} {'$/ticket'}")
    print("-" * 72)

    all_frontier_cost   = 0.0
    right_sized_cost    = 0.0
    avg_in, avg_out     = 500, 200

    for agent, logical in AGENT_TIER_MAP.items():
        concrete      = LOGICAL_MODEL_MAP[logical]
        cost          = cost_per_ticket_usd(logical, avg_in, avg_out)
        frontier_cost = cost_per_ticket_usd("frontier-model", avg_in, avg_out)
        all_frontier_cost += frontier_cost
        right_sized_cost  += cost
        print(f"  {agent:<14} {logical:<22} {concrete:<28} ${cost:.6f}")

    tax      = all_frontier_cost - right_sized_cost
    tax_pct  = tax / all_frontier_cost * 100 if all_frontier_cost else 0
    print("-" * 72)
    print(f"  {'All-frontier cost/ticket':<44}  ${all_frontier_cost:.6f}")
    print(f"  {'Right-sized cost/ticket':<44}  ${right_sized_cost:.6f}")
    print(f"  {'Intelligence Tax saved/ticket':<44}  ${tax:.6f}  ({tax_pct:.1f}%)")

    daily_tickets = 100_000
    monthly_days  = 30
    print()
    print(f"  At {daily_tickets:,} tickets/day for {monthly_days} days:")
    print(f"    All-frontier:  ${all_frontier_cost * daily_tickets * monthly_days:>10,.2f}/month")
    print(f"    Right-sized:   ${right_sized_cost  * daily_tickets * monthly_days:>10,.2f}/month")
    print(f"    Monthly saving: ${tax * daily_tickets * monthly_days:>9,.2f}")
    print()


# ── SECTION 4: ShimResponse and LiteLLMShim ───────────────────────────────

class ShimResponse(BaseModel):
    """
    OpenAI-compatible response shape returned by the LiteLLMShim.
    Every Chronicle agent sees this — never the raw Gemini response dict.
    Introduced: Session 14.2. Permanent.
    """
    model:         str    # concrete model used
    logical_model: str    # logical name the agent requested
    content:       str    # text response
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    latency_ms:    float
    virtual_key:   Optional[str] = None


class LiteLLMShim:
    """
    In-process simulator for the real LiteLLM proxy.

    The real proxy runs as:
        litellm --config config.yaml --port 4000
    and Chronicle nodes point an OpenAI client at http://localhost:4000.

    This shim keeps the same .completion() surface so Chronicle agent code
    (agent.py) is byte-identical between Colab/dev and production. The only
    difference in production: replace this shim with an httpx call to
    http://localhost:4000/v1/chat/completions.

    Introduced: Session 14.2. Permanent.
    """

    def __init__(self, logical_map: Dict[str, str]):
        """Store the logical→concrete model map and initialise the spend ledger."""
        self.logical_map   = logical_map
        self.spend_ledger: List[Dict[str, Any]] = []

    def resolve_model(self, logical_name: str) -> str:
        """
        Map a logical model name ('utility-model') to a concrete variant.
        Raises KeyError if the logical name is not registered.
        In production: this resolution lives in config.yaml, not here.
        Introduced: Session 14.2. Permanent.
        """
        if logical_name not in self.logical_map:
            raise KeyError(
                f"Unknown logical model: {logical_name!r}. "
                f"Registered: {list(self.logical_map.keys())}"
            )
        return self.logical_map[logical_name]

    async def completion(
        self,
        model:       str,
        messages:    List[Dict[str, str]],
        virtual_key: Optional[str]        = None,
        metadata:    Optional[Dict[str, str]] = None,
    ) -> ShimResponse:
        """
        OpenAI-compatible completion entry point.

        Args:
            model:       logical model name registered in self.logical_map.
            messages:    OpenAI-style list of {role, content} dicts.
            virtual_key: sk-vk-... token from VirtualKeyRegistry.
            metadata:    tags (agent, analysis_id) that land in the FinOps ledger.

        Returns:
            ShimResponse with cost and latency attribution.
            The OTel layer stamps routing_model from ShimResponse.model.

        In production this method becomes:
            return await openai_client.chat.completions.create(
                model=model, messages=messages, ...)
        pointed at http://localhost:4000.

        Introduced: Session 14.2. Permanent.
        """
        concrete = self.resolve_model(model)
        prompt   = "\n".join(m["content"] for m in messages)
        t0       = time.time()

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{concrete}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_ctx)
        ) as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()

        latency_ms = (time.time() - t0) * 1000

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            text = ""

        # Token estimate — production LiteLLM reads usage from provider response
        in_tok  = max(1, len(prompt) // 4)
        out_tok = max(1, len(text) // 4)
        cost    = self._estimate_cost(concrete, in_tok, out_tok)

        entry = {
            "logical_model":  model,
            "concrete_model": concrete,
            "input_tokens":   in_tok,
            "output_tokens":  out_tok,
            "cost_usd":       cost,
            "virtual_key":    virtual_key,
            "metadata":       metadata or {},
            "timestamp":      time.time(),
        }
        self.spend_ledger.append(entry)

        return ShimResponse(
            model         = concrete,
            logical_model = model,
            content       = text,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            cost_usd      = cost,
            latency_ms    = latency_ms,
            virtual_key   = virtual_key,
        )

    @staticmethod
    def _estimate_cost(concrete: str, in_tok: int, out_tok: int) -> float:
        """
        Estimate USD cost of one call using public list rates.
        Production LiteLLM reads the actual token count from provider usage metadata.
        The shim approximates with len(text) // 4 — sufficient for FinOps tracking.
        Introduced: Session 14.2. Permanent.
        """
        if concrete == PRIMARY_MODEL:
            return (in_tok / 1_000_000) * 15.00 + (out_tok / 1_000_000) * 60.00
        return (in_tok / 1_000_000) * 0.15 + (out_tok / 1_000_000) * 0.60


# ── SECTION 5: VirtualKeyRegistry ─────────────────────────────────────────

class VirtualKey(BaseModel):
    """One virtual key record."""
    vk:         str
    owner:      str
    scope:      List[str]
    budget_usd: float
    spent_usd:  float   = 0.0
    revoked:    bool    = False


class VirtualKeyRegistry:
    """
    Proxy-side registry that issues, tracks, and revokes virtual keys.

    Real provider keys (GEMINI_API_KEY, OPENAI_API_KEY, ...) live only in
    the proxy process. Chronicle services hold sk-vk-... tokens that can be
    scoped, budgeted, and revoked without touching any provider credential.

    The operations raw provider keys cannot do — scope, budget, expiry,
    instant revocation — are all first-class primitives here.

    Introduced: Session 14.2. Permanent.
    """

    def __init__(self):
        """Start with empty registry and audit log."""
        self.keys:  Dict[str, VirtualKey]   = {}
        self.audit: List[Dict[str, Any]]    = []

    def issue(self, owner: str, scope: List[str], budget_usd: float) -> str:
        """
        Mint a virtual key bound to (owner, scope, budget).
        Returns: sk-vk-... token.
        Introduced: Session 14.2. Permanent.
        """
        raw    = f"{owner}:{scope}:{budget_usd}:{time.time()}"
        suffix = hashlib.sha256(raw.encode()).hexdigest()[:16]
        vk     = f"sk-vk-{suffix}"
        self.keys[vk] = VirtualKey(vk=vk, owner=owner, scope=scope, budget_usd=budget_usd)
        self.audit.append({"event": "issue", "vk": vk, "owner": owner, "ts": time.time()})
        return vk

    def revoke(self, vk: str) -> None:
        """Revoke a virtual key. Subsequent authorize() calls return False."""
        if vk not in self.keys:
            raise KeyError(f"Unknown vk: {vk}")
        self.keys[vk].revoked = True
        self.audit.append({"event": "revoke", "vk": vk, "ts": time.time()})

    def authorize(self, vk: str, model_name: str, incremental_usd: float) -> bool:
        """
        Return True iff vk is valid, not revoked, in-scope, and under budget.
        Called by route_chronicle_agent() before each LLM call.
        A False here produces a 429-Budget-Exceeded — real provider never consulted.
        Introduced: Session 14.2. Permanent.
        """
        if vk not in self.keys:
            return False
        rec = self.keys[vk]
        if rec.revoked:
            return False
        if "all" not in rec.scope and model_name not in rec.scope:
            return False
        if rec.spent_usd + incremental_usd > rec.budget_usd:
            return False
        return True

    def charge(self, vk: str, usd: float) -> None:
        """Increment spent_usd after a successful call."""
        if vk in self.keys:
            self.keys[vk].spent_usd += usd


# ── SECTION 6: Semantic Classifier ─────────────────────────────────────────

class SemanticLevel(BaseModel):
    """Classifier verdict. Level 1/2/3 → utility/mid/frontier."""
    level:         int
    confidence:    float
    justification: str


CLASSIFIER_PROMPT = """You are a request-complexity classifier for an AI analysis system.
Read the user question and output a JSON object:
{"level":1|2|3,"confidence":0.0-1.0,"justification":"short string"}

Level 1 = simple single-source lookup or routine pattern question.
Level 2 = multi-source correlation, moderate reasoning required.
Level 3 = deep cross-source analysis, multi-step causal reasoning, honesty judgement.

Output JSON only — no other text."""

CONFIDENCE_FLOOR  = 0.75   # below this, fail safe and escalate to frontier
LEVEL_TO_LOGICAL  = {1: "utility-model", 2: "mid-model", 3: "frontier-model"}


async def semantic_classify(
    question:    str,
    shim:        LiteLLMShim,
    vk:          str,
    vk_registry: VirtualKeyRegistry,
) -> SemanticLevel:
    """
    What it does:   Classify one Chronicle analysis question into a complexity tier.
                    Runs on utility-model — never pay frontier prices to decide
                    whether a request needs the frontier tier.
    Returns:        SemanticLevel with level (1/2/3), confidence, and justification.
    Fail-safe:      If confidence < CONFIDENCE_FLOOR, escalate to frontier.
    When called:    By route_chronicle_agent() when use_semantic=True.
    Introduced:     Session 14.2. Permanent.
    Updated:        Session 14.2 — classifier call now goes through the same
                    vk_registry.authorize()/charge() gate as every other
                    LLM call. Previously it called shim.completion() directly,
                    so its spend landed in the ledger but never against the
                    virtual key's budget — a classifier-cost leak.
    """
    classifier_cost = cost_per_ticket_usd("semantic-classifier")
    if not vk_registry.authorize(vk, "semantic-classifier", classifier_cost):
        raise PermissionError(
            f"Virtual key {vk!r} failed authorization for 'semantic-classifier'. "
            f"Check scope, budget, or revocation status."
        )

    resp  = await shim.completion(
        model    = "semantic-classifier",
        messages = [
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user",   "content": question[:800]},
        ],
        virtual_key = vk,
        metadata = {"agent": "semantic_router"},
    )
    vk_registry.charge(vk, resp.cost_usd)

    data  = _parse_json_safe(resp.content)
    level = int(data.get("level", 3))
    conf  = float(data.get("confidence", 0.5))
    just  = data.get("justification", "")

    if conf < CONFIDENCE_FLOOR:
        level = 3
        just  = f"[CONFIDENCE_FLOOR={conf:.2f} breached] {just}"

    return SemanticLevel(level=level, confidence=conf, justification=just)


def _parse_json_safe(text: str) -> dict:
    """Strip markdown fences and parse JSON. Returns {} on failure."""
    clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(clean)
    except Exception:
        return {}


# ── SECTION 7: Chronicle Agent Router ─────────────────────────────────────

class RoutedResult(BaseModel):
    """
    Result of one Chronicle agent LLM call through the router.
    Carries the information needed to stamp OTel span attributes.
    Introduced: Session 14.2. Permanent.
    """
    agent:          str
    logical_model:  str    # what the agent requested
    concrete_model: str    # what actually ran
    content:        str
    input_tokens:   int
    output_tokens:  int
    cost_usd:       float
    latency_ms:     float


async def route_chronicle_agent(
    agent:        str,
    prompt:       str,
    shim:         LiteLLMShim,
    vk:           str,
    vk_registry:  VirtualKeyRegistry,
    question:     str             = "",
    use_semantic: bool            = False,
) -> RoutedResult:
    """
    What it does:   Route one Chronicle agent LLM call to the correct model tier.
                    Enforces virtual-key authorization before the call.
                    Optionally runs the semantic classifier for the brutality agent.
    When called:    By each Chronicle agent node (ingestion, pattern, timeline,
                    brutality, synthesis) in place of call_gemini_traced().
    Returns:        RoutedResult — the OTel layer reads routing_model from this.
    Introduced:     Session 14.2. Permanent.

    OTel integration:
        Inside each Chronicle agent node, after this call:
            span.set_attribute('routing_model',     result.logical_model)
            span.set_attribute('routing_model_actual', result.concrete_model)
            span.set_attribute('llm.token_count.input',  result.input_tokens)
            span.set_attribute('llm.token_count.output', result.output_tokens)
    """
    # Determine logical model
    logical = AGENT_TIER_MAP.get(agent, "utility-model")

    # Semantic pre-classification — only for brutality agent where complexity varies
    if use_semantic and agent == "brutality" and question:
        verdict = await semantic_classify(question, shim, vk, vk_registry)
        # Override tier if classifier escalates
        logical = LEVEL_TO_LOGICAL.get(verdict.level, logical)

    # Virtual-key authorization
    estimated_cost = cost_per_ticket_usd(logical)
    if not vk_registry.authorize(vk, logical, estimated_cost):
        raise PermissionError(
            f"Virtual key {vk!r} failed authorization for {logical!r}. "
            f"Check scope, budget, or revocation status."
        )

    # Route through the shim
    resp = await shim.completion(
        model    = logical,
        messages = [{"role": "user", "content": prompt}],
        virtual_key = vk,
        metadata = {"agent": agent},
    )

    vk_registry.charge(vk, resp.cost_usd)

    return RoutedResult(
        agent          = agent,
        logical_model  = logical,
        concrete_model = resp.model,
        content        = resp.content,
        input_tokens   = resp.input_tokens,
        output_tokens  = resp.output_tokens,
        cost_usd       = resp.cost_usd,
        latency_ms     = resp.latency_ms,
    )


# ── SECTION 8: FinOps Aggregation ─────────────────────────────────────────

def aggregate_spend_by(
    ledger:    List[Dict[str, Any]],
    dimension: str,
) -> Dict[str, float]:
    """
    What it does:   Group a spend ledger by a metadata dimension (agent, analysis_id).
    When called:    By print_finops_report() and the verification test.
    Returns:        Dict of dimension_value → total_usd.
    Introduced:     Session 14.2. Permanent.
    """
    out: Dict[str, float] = {}
    for entry in ledger:
        key = entry.get("metadata", {}).get(dimension, "unknown")
        out[key] = out.get(key, 0.0) + entry["cost_usd"]
    return out


def print_finops_report(shim: LiteLLMShim, vk_registry: VirtualKeyRegistry, vk: str) -> None:
    """Print the per-agent spend report. This is the FinOps dashboard output."""
    per_agent    = aggregate_spend_by(shim.spend_ledger, "agent")
    total        = sum(per_agent.values())
    vk_rec       = vk_registry.keys.get(vk)

    print("\nPER-AGENT SPEND (this session)")
    print("=" * 52)
    for agent, usd in sorted(per_agent.items()):
        print(f"  {agent:<20} ${usd*100:.6f} cents")
    print("-" * 52)
    print(f"  {'TOTAL':<20} ${total*100:.6f} cents")
    if vk_rec:
        print(f"  Virtual key budget:   ${vk_rec.budget_usd:.2f}")
        print(f"  Virtual key spent:    ${vk_rec.spent_usd:.6f}")
    print()


# ── SECTION 9: Chronicle End-to-End Demo ──────────────────────────────────

EASY_QUESTION = "What does my Spotify listening say about my mood this week?"
HARD_QUESTION = (
    "My commit rate dropped 40% in the last 3 weeks and my Spotify "
    "shifted from ambient to aggressive music at the same time. My "
    "finance tracker shows three large late-night food deliveries on "
    "the same days as the low-commit days. What does this actually "
    "mean about my current mental and professional state? Be honest."
)


async def run_chronicle_routing_demo(shim: LiteLLMShim, vk_registry: VirtualKeyRegistry, vk: str) -> None:
    """
    Run two Chronicle analyses — one easy (all utility tier), one hard (brutality → frontier).
    Demonstrates static routing for the first, semantic escalation for the second.
    Introduced: Session 14.2. Permanent.
    """
    print("=" * 72)
    print("STATIC ROUTING PATH — easy question")
    print("=" * 72)

    for agent in ["ingestion", "pattern", "timeline", "brutality", "synthesis"]:
        prompt = f"Chronicle {agent} agent. Question: {EASY_QUESTION}"
        result = await route_chronicle_agent(
            agent=agent, prompt=prompt,
            shim=shim, vk=vk, vk_registry=vk_registry,
            question=EASY_QUESTION, use_semantic=False,
        )
        print(f"  {result.agent:<12} "
              f"logical={result.logical_model:<22} "
              f"concrete={result.concrete_model}")

    print()
    print("=" * 72)
    print("SEMANTIC ROUTING PATH — hard question (brutality may escalate)")
    print("=" * 72)

    for agent in ["ingestion", "pattern", "timeline", "brutality", "synthesis"]:
        prompt = f"Chronicle {agent} agent. Question: {HARD_QUESTION}"
        result = await route_chronicle_agent(
            agent=agent, prompt=prompt,
            shim=shim, vk=vk, vk_registry=vk_registry,
            question=HARD_QUESTION, use_semantic=(agent == "brutality"),
        )
        tag = " ← SEMANTIC ESCALATION" if (
            agent == "brutality" and result.logical_model == "frontier-model"
        ) else ""
        print(f"  {result.agent:<12} "
              f"logical={result.logical_model:<22} "
              f"concrete={result.concrete_model}{tag}")

    print()
    print_finops_report(shim, vk_registry, vk)

    print("RESOLVED MODEL STRINGS")
    print("=" * 52)
    for logical, concrete in shim.logical_map.items():
        print(f"  {logical:<26} → {concrete}")
    print()


# ── SECTION 10: config.yaml reference ─────────────────────────────────────

LITELLM_CONFIG_YAML = """
# config.yaml — drop in next to your FastAPI service
# Start the proxy with: litellm --config config.yaml --port 4000
#
# The logical model names on the LEFT match LOGICAL_MODEL_MAP in model_router.py.
# Chronicle agents are ignorant of which provider fulfills a request.

model_list:
  - model_name: utility-model          # Ingestion, Pattern, Timeline
    litellm_params:
      model:   gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY

  - model_name: frontier-model         # Brutality, Synthesis
    litellm_params:
      model:   anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: mid-model              # optional middle tier
    litellm_params:
      model:   openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: semantic-classifier    # separate lane, same utility tier
    litellm_params:
      model:   gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY

  - model_name: local-utility          # Week 11 vLLM
    litellm_params:
      model:    openai/Meta-Llama-3.1-70B-Instruct-AWQ
      api_base: http://vllm.production.svc.cluster.local:8000/v1
      api_key:  os.environ/VLLM_INTERNAL_KEY

general_settings:
  master_key:   os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/LITELLM_DB_URL

litellm_settings:
  set_verbose: false

router_settings:
  routing_strategy: simple-shuffle
  num_retries:      2
  timeout:          30
"""


# ── SECTION 11: Verification ───────────────────────────────────────────────

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 14.2 — VERIFICATION TEST                           │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │    - AGENT_TIER_MAP assigns correct tiers per agent         │
    │    - cost_per_ticket_usd() returns correct values           │
    │    - LiteLLMShim.resolve_model() maps logical → concrete    │
    │    - LiteLLMShim.resolve_model() raises on unknown name     │
    │    - VirtualKeyRegistry: issue, authorize, charge, revoke   │
    │    - aggregate_spend_by() groups ledger by dimension        │
    ├─────────────────────────────────────────────────────────────┤
    │  PASS CRITERIA:                                             │
    │    ✓ brutality → frontier-model                             │
    │    ✓ pattern → utility-model                                │
    │    ✓ frontier cost >> utility cost (ratio ≥ 50×)             │
    │    ✓ resolve_model('utility-model') = UTILITY_MODEL          │
    │    ✓ resolve_model('unknown') raises KeyError                │
    │    ✓ authorize() False on revoked key                        │
    │    ✓ authorize() False on budget exceeded                    │
    │    ✓ aggregate_spend_by() groups by metadata key             │
    └─────────────────────────────────────────────────────────────┘
    """
    import time as _time
    checks = []
    start  = _time.monotonic()

    # CHECK 1: brutality → frontier
    checks.append({
        "label":  "AGENT_TIER_MAP: brutality → frontier-model",
        "passed": AGENT_TIER_MAP.get("brutality") == "frontier-model",
        "note":   f"Got: {AGENT_TIER_MAP.get('brutality')}",
    })

    # CHECK 2: pattern → utility
    checks.append({
        "label":  "AGENT_TIER_MAP: pattern → utility-model",
        "passed": AGENT_TIER_MAP.get("pattern") == "utility-model",
        "note":   f"Got: {AGENT_TIER_MAP.get('pattern')}",
    })

    # CHECK 3: frontier costs >> utility costs
    frontier_cost = cost_per_ticket_usd("frontier-model")
    utility_cost  = cost_per_ticket_usd("utility-model")
    ratio_ok      = frontier_cost >= utility_cost * 50
    checks.append({
        "label":  "Frontier cost ≥ 50× utility cost",
        "passed": ratio_ok,
        "note":   f"frontier=${frontier_cost:.6f} utility=${utility_cost:.6f} ratio={frontier_cost/utility_cost:.0f}×",
    })

    # CHECK 4: resolve_model returns correct concrete
    shim     = LiteLLMShim(LOGICAL_MODEL_MAP)
    resolved = shim.resolve_model("utility-model")
    checks.append({
        "label":  "resolve_model('utility-model') = UTILITY_MODEL",
        "passed": resolved == UTILITY_MODEL,
        "note":   f"Got: {resolved}",
    })

    # CHECK 5: resolve_model raises on unknown name
    try:
        shim.resolve_model("nonexistent-model")
        key_err_ok = False
    except KeyError:
        key_err_ok = True
    checks.append({
        "label":  "resolve_model() raises KeyError for unknown logical name",
        "passed": key_err_ok,
        "note":   "KeyError correctly raised",
    })

    # CHECK 6: VirtualKeyRegistry — revoked key
    reg = VirtualKeyRegistry()
    vk  = reg.issue("test-service", scope=["utility-model"], budget_usd=10.0)
    ok_before = reg.authorize(vk, "utility-model", 0.01)
    reg.revoke(vk)
    ok_after  = reg.authorize(vk, "utility-model", 0.01)
    checks.append({
        "label":  "authorize() returns False after revoke()",
        "passed": ok_before and not ok_after,
        "note":   f"Before revoke: {ok_before} · After revoke: {ok_after}",
    })

    # CHECK 7: VirtualKeyRegistry — budget exceeded
    vk2    = reg.issue("budget-test", scope=["utility-model"], budget_usd=0.001)
    reg.charge(vk2, 0.0015)   # spend over budget
    over_ok = not reg.authorize(vk2, "utility-model", 0.0001)
    checks.append({
        "label":  "authorize() returns False when budget exceeded",
        "passed": over_ok,
        "note":   f"spent=0.0015 budget=0.001 → authorize={not over_ok}",
    })

    # CHECK 8: aggregate_spend_by
    fake_ledger = [
        {"cost_usd": 0.001, "metadata": {"agent": "brutality"}},
        {"cost_usd": 0.0001, "metadata": {"agent": "pattern"}},
        {"cost_usd": 0.002, "metadata": {"agent": "brutality"}},
    ]
    agg = aggregate_spend_by(fake_ledger, "agent")
    agg_ok = (
        abs(agg.get("brutality", 0) - 0.003) < 1e-9
        and abs(agg.get("pattern", 0) - 0.0001) < 1e-9
    )
    checks.append({
        "label":  "aggregate_spend_by() groups ledger by metadata dimension",
        "passed": agg_ok,
        "note":   f"brutality=${agg.get('brutality', 0):.4f} pattern=${agg.get('pattern', 0):.4f}",
    })

    duration_ms = round((_time.monotonic() - start) * 1000)
    passed      = sum(1 for c in checks if c["passed"])
    total       = len(checks)
    return {
        "passed":      passed == total,
        "checks":      checks,
        "summary":     f"{passed}/{total} checks passed in {duration_ms}ms",
        "duration_ms": duration_ms,
    }


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Chronicle — Session 14.2 Verification               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    result = run_session_verification()
    print(f"  Verification: {result['summary']}\n")
    for check in result["checks"]:
        icon = "✓" if check["passed"] else "✗"
        print(f"  {icon} {check['label']}")
        print(f"      {check['note']}")
    print()

    if not result["passed"]:
        print("  ✗ Fix failing checks before running the demo.")
        sys.exit(1)

    print("  ✓ Session 14.2 VERIFIED.\n")

    # Intelligence Tax table
    print_intelligence_tax_table()

    # Print config.yaml reference
    print("LITELLM config.yaml REFERENCE")
    print("=" * 72)
    print(LITELLM_CONFIG_YAML)
    print("  Save as config.yaml · run: litellm --config config.yaml --port 4000")
    print()

    # End-to-end routing demo (requires GEMINI_API_KEY)
    if GEMINI_API_KEY:
        shim     = LiteLLMShim(LOGICAL_MODEL_MAP)
        registry = VirtualKeyRegistry()
        vk       = registry.issue(
            owner      = "chronicle-fastapi-prod",
            scope      = list(LOGICAL_MODEL_MAP.keys()) + ["all"],
            budget_usd = 50.0,
        )
        print(f"  Virtual key issued: {vk}\n")
        asyncio.run(run_chronicle_routing_demo(shim, registry, vk))
    else:
        print("  Set GEMINI_API_KEY to run the end-to-end routing demo.")

    print("  ✓ Session 14.2 COMPLETE.")
    print()
    print("  Next steps:")
    print("  1. Wire route_chronicle_agent() into agent.py")
    print("     (replace call_gemini_traced() calls with route_chronicle_agent())")
    print("  2. Stamp routing_model and routing_model_actual on OTel spans")
    print("  3. Add frontier_tier_fraction SLO to monitoring_daemon.py")
    print("  4. Deploy LiteLLM proxy for production: litellm --config config.yaml --port 4000")
    print()


# ══════════════════════════════════════════════════════════════════
# SESSION 14.3 HANDOFF — Fallbacks + Circuit Breaker
# ══════════════════════════════════════════════════════════════════
#
# What gets ADDED in Session 14.3 (extend, never remove):
#
#   FallbackShim subclasses LiteLLMShim.
#   .completion() wraps each provider call in asyncio.wait_for(5s).
#   On 429 / 401 / TimeoutError: transparently rolls to the next
#   concrete provider in the fallback chain for that logical model.
#
#   CircuitBreaker attaches to each provider.
#   Three consecutive failures trip OPEN (300s cooldown).
#   HALF_OPEN allows one probe request before CLOSED.
#
#   routing_model_actual OTel attribute:
#   May differ from routing_model when a fallback fired.
#   The S13.3 daemon adds fallback_rate_spike tripwire that reads it.
#
# What stays UNCHANGED from Session 14.2:
#   LOGICAL_MODEL_MAP
#   AGENT_TIER_MAP
#   VirtualKeyRegistry (issue / revoke / authorize / charge)
#   aggregate_spend_by()
#   cost_per_ticket_usd()
#   route_chronicle_agent() external interface
#   All OTel span attribute names
# ══════════════════════════════════════════════════════════════════
