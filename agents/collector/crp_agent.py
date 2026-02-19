"""
Cognitive Resolution Pipeline (CRP) Collector Agent — Variable-Centric Resolution
with Iterative Critical Investigator (ReAct pattern).

Architecture:

1. ARCHITECT — Decompose the question into *variables* and a *logic condition*.
2. CRITICAL INVESTIGATOR — For each variable, run a multi-turn Gemini
   grounded-search loop (up to 3 turns).  The investigator can Think,
   Search, Critique its own findings, and Re-Search.  This handles:
   - Entity disambiguation (ETF vs Strategy tracker)
   - Unit normalization (seconds → minutes, % → $)
   - Missing-data skepticism ("Closed" may not mean "no data")
3. ADJUDICATOR — Plug the resolved variables into the logic condition and
   evaluate YES / NO / INVALID.

This replaces the old linear Researcher → Analyst pipeline with a single
agent that self-corrects over multiple turns, mimicking how a human would
iteratively refine their research.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from agents.base import AgentCapability, AgentResult, BaseAgent
from core.schemas import (
    CheckResult,
    EvidenceBundle,
    EvidenceItem,
    Provenance,
    PromptSpec,
    ToolCallRecord,
    ToolExecutionLog,
    ToolPlan,
    VerificationResult,
)

if TYPE_CHECKING:
    from agents.context import AgentContext
    from core.schemas import DataRequirement


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

CRP_ARCHITECT_PROMPT = """\
You are the Architect of a Prediction Market Resolution Engine.

Convert the market question into a solvable "Variable Logic" plan.
Identify the UNKNOWN VARIABLES that must be fetched from the real world
to decide the outcome.

### RULES
1. Constants (thresholds, deadlines) embedded in the question are NOT
   variables — extract them directly into the logic_condition string.
2. Each variable must have a clear name, description, and expected type.
3. The logic_condition must be a single evaluatable expression using the
   variable names (e.g., "sell_out_time < 3 minutes").
4. List every explicit invalidity/cancellation condition you find.

### EXAMPLES

Input: "Will REKT sell out within 3 minutes?"
Output: {
  "variables": [
    {"name": "sell_out_duration", "description": "Time taken for REKT to sell out", "type": "Duration"}
  ],
  "constants": {"limit": "3 minutes"},
  "logic_condition": "sell_out_duration < 3 minutes",
  "invalidity_conditions": ["Sale cancelled", "Product not released"]
}

Input: "Jim Cramer wrong in January? (Target $105.25M)"
Output: {
  "variables": [
    {"name": "strategy_value", "description": "Inverse Cramer Strategy performance value on Jan 31", "type": "Currency (USD)"}
  ],
  "constants": {"target": "105.25 Million USD"},
  "logic_condition": "strategy_value > 105.25 Million",
  "invalidity_conditions": ["Strategy tracker discontinued", "Material modification to rebalancing rules"]
}

Input: "Did Trump play golf Saturday AND Sunday?"
Output: {
  "variables": [
    {"name": "played_saturday", "description": "Whether Trump played golf on Saturday", "type": "Boolean"},
    {"name": "played_sunday", "description": "Whether Trump played golf on Sunday", "type": "Boolean"}
  ],
  "constants": {},
  "logic_condition": "played_saturday AND played_sunday",
  "invalidity_conditions": []
}

### YOUR TASK
Return a single valid JSON object (no markdown fences).
"""


CRP_INVESTIGATOR_PROMPT = """\
You are a Critical Investigator for a High-Stakes Prediction Market.
Your goal is to find the EXACT value of a specific variable.

### THE VARIABLE
Name: {var_name}
Type: {var_type}
Description: {var_description}
Context (Market Question): {context}
Resolution Rules: {rules}
Logic Condition: {logic_condition}

### YOUR CORE PRINCIPLES
1. **Source Hierarchy:** The "Resolution Source" (if specified in Context) is
   authoritative. If it contradicts other news, trust the Resolution Source.
   If the source is down, check for archives or official shutdown notices.

2. **Entity Disambiguation:** Distinguish between a "Product" (e.g., an ETF
   stock ticker) and a "Data Point" (e.g., an index, strategy tracker, or
   dashboard). An ETF might delist, but the underlying index might still
   publish data. Check specifically for this.

3. **Be Skeptical of "Missing" Data:** If a search result says "Closed",
   "Discontinued", or "No Data," verify if it truly applies to the EXACT
   thing we are tracking. Try at least one alternative search query before
   declaring invalid.

4. **Normalization:** If you find the data in a different unit than the logic
   condition requires (e.g., % vs $, seconds vs minutes), convert it if you
   have enough information. If you need a base value for conversion, search
   for that too.

5. **Temporal Awareness:** Current date is {current_date}. Prefer the most
   recent sources. Ignore outdated speculation if newer confirming evidence
   exists.

### OUTPUT FORMAT
Reflect on your search results. If you have the definitive answer, output:

FINAL_ANSWER
{{"value": <the_normalized_value>, "unit": "<unit>", "status": "RESOLVED" | "INVALID_CONDITION_TRIGGERED", "confidence": <0.0-1.0>, "reasoning": "<step-by-step reasoning>", "evidence_sources": [{{"url": "...", "title": "...", "key_fact": "..."}}]}}

If you need to search more, output your thought process explaining what you
found, what's wrong with it, and what you plan to search next. The next
search will happen automatically.
"""


CRP_ADJUDICATOR_PROMPT = """\
You are the Adjudicator of a Prediction Market Resolution Engine.

You receive:
- A logic condition (the rule)
- Resolved variable values (the evidence)
- Invalidity conditions to check

### ADJUDICATION PROTOCOL
1. Plug each variable value into the logic condition.
2. Evaluate the condition strictly.
3. Check each invalidity condition. If ANY is triggered, verdict is "INVALID".
4. If any required variable is null/unresolved, verdict is "INVALID".
5. Calibrate confidence based on source quality and directness of evidence.

### OUTPUT FORMAT
Return a single valid JSON object (no markdown fences):
{
    "verdict": "YES | NO | INVALID",
    "reasoning_trace": "1) ... 2) ... 3) ... Final conclusion: ...",
    "variable_evaluation": {
        "<var_name>": {"value": ..., "unit": "..."},
        ...
    },
    "logic_result": true | false | null,
    "invalidity_check": {
        "conditions_triggered": [],
        "override": false
    },
    "confidence": <float 0.0 to 1.0>
}
"""

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
_GEMINI_TIMEOUT = 80  # seconds — stay below Cloudflare 524 threshold
_MAX_INVESTIGATOR_TURNS = 3  # ReAct turns per variable
_MAX_VARIABLE_ITERATIONS = 8  # total variable queue iterations


class CollectorCRP(BaseAgent):
    """
    Cognitive Resolution Pipeline (CRP) — Variable-Centric Resolution
    with iterative Critical Investigator (ReAct pattern).

    1) Architect — decompose into variables + logic
    2) Critical Investigator (per variable) — multi-turn Gemini grounded
       search with self-correction.  Can Think → Search → Critique → Re-Search.
    3) Adjudicator — evaluate logic with resolved variables

    Features:
    - Question-agnostic: handles financial, boolean, time, multi-variable
    - Self-correcting: investigator critiques its own findings over 3 turns
    - Entity disambiguation, unit normalization, missing-data skepticism
    - Gemini Google Search grounding for autonomous web research
    - Full audit trail via ToolCallRecords with per-turn reasoning logs
    """

    _name = "CollectorCRP"
    _version = "v4"
    _capabilities = {AgentCapability.LLM, AgentCapability.NETWORK}
    MAX_RETRIES = 2
    DEFAULT_MODEL = "gemini-2.5-flash"

    def run(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> AgentResult:
        ctx.info(f"CollectorCRP executing plan {tool_plan.plan_id}")

        if ctx.llm is None:
            return AgentResult.failure(error="LLM client not available")

        bundle = EvidenceBundle(
            bundle_id=f"crp_{tool_plan.plan_id}",
            market_id=prompt_spec.market_id,
            plan_id=tool_plan.plan_id,
            collector_name=self._name,
        )

        execution_log = ToolExecutionLog(
            plan_id=tool_plan.plan_id,
            started_at=ctx.now().isoformat(),
        )

        for req_id in tool_plan.requirements:
            requirement = prompt_spec.get_requirement_by_id(req_id)
            if not requirement:
                continue
            evidence = self._run_pipeline(ctx, prompt_spec, requirement, execution_log)
            bundle.add_item(evidence)

        bundle.collected_at = ctx.now()
        bundle.requirements_fulfilled = list(set(
            item.requirement_id for item in bundle.items if item.success
        ))
        bundle.requirements_unfulfilled = [
            r for r in tool_plan.requirements
            if r not in bundle.requirements_fulfilled
        ]
        execution_log.ended_at = ctx.now().isoformat()

        verification = self._validate_output(bundle, tool_plan)

        ctx.info(
            f"CRP collection complete: {bundle.total_sources_succeeded}/"
            f"{bundle.total_sources_attempted} succeeded, "
            f"{len(bundle.requirements_fulfilled)}/{len(tool_plan.requirements)} "
            f"requirements fulfilled"
        )

        return AgentResult(
            output=(bundle, execution_log),
            verification=verification,
            receipts=ctx.get_receipt_refs(),
            success=bundle.has_evidence,
            error=None if bundle.has_evidence else "No evidence collected",
            metadata={
                "collector": "crp",
                "bundle_id": bundle.bundle_id,
                "total_sources_attempted": bundle.total_sources_attempted,
                "total_sources_succeeded": bundle.total_sources_succeeded,
                "requirements_fulfilled": bundle.requirements_fulfilled,
                "requirements_unfulfilled": bundle.requirements_unfulfilled,
            },
        )

    # ------------------------------------------------------------------
    # Pipeline orchestrator
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        requirement: "DataRequirement",
        execution_log: ToolExecutionLog,
    ) -> EvidenceItem:
        evidence_id = self._generate_evidence_id(requirement.requirement_id)

        print("\n" + "=" * 70)
        print("CRP PIPELINE START (Variable-Centric + ReAct Investigator)")
        print(f"  Market: {prompt_spec.market.question}")
        print(f"  Requirement: {requirement.requirement_id} — {requirement.description}")
        print("=" * 70)

        # ---- STEP 1: ARCHITECT ----
        print("\n" + "-" * 70)
        print("STEP 1: ARCHITECT (Decompose into Variables + Logic)")
        print("-" * 70)

        plan, arch_record = self._step_architect(ctx, prompt_spec)
        execution_log.add_call(arch_record)

        if plan is None:
            print("[ARCHITECT] FAILED — could not decompose question")
            print("=" * 70 + "\n")
            return EvidenceItem(
                evidence_id=evidence_id,
                requirement_id=requirement.requirement_id,
                provenance=Provenance(
                    source_id="crp", source_uri="crp:architect",
                    tier=2, fetched_at=ctx.now(),
                ),
                success=False,
                error=f"Architect failed: {arch_record.error}",
            )

        variables = plan.get("variables", [])
        logic_condition = plan.get("logic_condition", "")
        invalidity_conditions = plan.get("invalidity_conditions", [])

        print(f"[ARCHITECT] Variables to resolve:")
        for v in variables:
            print(f"  - {v['name']} ({v.get('type', '?')}): {v.get('description', '')}")
        print(f"[ARCHITECT] Logic: {logic_condition}")
        print(f"[ARCHITECT] Constants: {plan.get('constants', {})}")
        print(f"[ARCHITECT] Invalidity conditions: {invalidity_conditions}")

        # ---- STEP 2: INVESTIGATE EACH VARIABLE ----
        print("\n" + "-" * 70)
        print("STEP 2: CRITICAL INVESTIGATOR (ReAct Loop per Variable)")
        print("-" * 70)

        resolved_vars: dict[str, Any] = {}
        variable_traces: list[dict[str, Any]] = []
        all_grounding_sources: list[dict[str, str]] = []

        # Build rules text for investigator context
        market = prompt_spec.market
        rules = market.resolution_rules
        rules_text = "\n".join(
            f"- [{r.rule_id}] {r.description}"
            for r in rules.get_sorted_rules()
        ) if rules and rules.rules else "No explicit rules."
        assumptions_list = prompt_spec.extra.get("assumptions", [])
        if assumptions_list:
            rules_text += "\nAssumptions:\n" + "\n".join(
                f"- {a}" for a in assumptions_list
            )

        for var_idx, variable in enumerate(variables):
            var_name = variable["name"]
            print(f"\n  {'=' * 50}")
            print(f"  VARIABLE [{var_idx + 1}/{len(variables)}]: {var_name}")
            print(f"  {'=' * 50}")

            result, grounding_sources, inv_record = self._step_investigate(
                ctx, variable, prompt_spec, logic_condition, rules_text,
            )
            execution_log.add_call(inv_record)
            all_grounding_sources.extend(grounding_sources)

            trace_entry: dict[str, Any] = {
                "name": var_name,
                "grounding_sources": grounding_sources,
                "investigation_result": result,
            }

            if result is not None and result.get("status") == "RESOLVED":
                val = result.get("value")
                resolved_vars[var_name] = val
                trace_entry["status"] = "RESOLVED"
                print(f"  [RESULT] RESOLVED: {var_name} = {val} "
                      f"({result.get('unit', '?')})")
            elif result is not None and result.get("status") == "INVALID_CONDITION_TRIGGERED":
                resolved_vars[var_name] = None
                trace_entry["status"] = "INVALID_CONDITION_TRIGGERED"
                print(f"  [RESULT] INVALID CONDITION: {result.get('reasoning', '')[:200]}")
            else:
                trace_entry["status"] = "UNRESOLVED"
                print(f"  [RESULT] UNRESOLVED for {var_name}")

            variable_traces.append(trace_entry)

        print(f"\n  [SUMMARY] Resolved variables: {resolved_vars}")

        # ---- STEP 3: ADJUDICATOR ----
        print("\n" + "-" * 70)
        print("STEP 3: ADJUDICATOR (Evaluate Logic)")
        print("-" * 70)

        adjudication, adj_record = self._step_adjudicate(
            ctx, logic_condition, resolved_vars, invalidity_conditions,
        )
        execution_log.add_call(adj_record)

        if adjudication:
            print(f"[ADJUDICATOR] Verdict:    {adjudication.get('verdict')}")
            print(f"[ADJUDICATOR] Confidence: {adjudication.get('confidence')}")
            print(f"[ADJUDICATOR] Logic result: {adjudication.get('logic_result')}")
            ic = adjudication.get("invalidity_check", {})
            triggered = ic.get("conditions_triggered", [])
            if triggered:
                print(f"[ADJUDICATOR] Invalidity triggered: {triggered}")
            trace = adjudication.get("reasoning_trace", "")
            if trace:
                print(f"[ADJUDICATOR] Reasoning:\n  {trace[:500]}")
        else:
            print("[ADJUDICATOR] FAILED — no adjudication produced")

        print("\n" + "=" * 70)
        verdict = adjudication.get("verdict") if adjudication else "N/A"
        confidence = adjudication.get("confidence") if adjudication else "N/A"
        print(f"CRP PIPELINE COMPLETE — Verdict: {verdict} (confidence: {confidence})")
        print("=" * 70 + "\n")

        # ---- Build EvidenceItem ----
        success = adjudication is not None and bool(resolved_vars)
        parsed_value = adjudication.get("verdict") if adjudication else None

        # Collect evidence_sources from all variable traces
        all_evidence_sources: list[dict[str, Any]] = []
        for t in variable_traces:
            inv = t.get("investigation_result") or {}
            for s in inv.get("evidence_sources", []):
                all_evidence_sources.append(s)
        evidence_sources = _normalize_evidence_sources(all_evidence_sources)

        extracted_fields = {
            "plan": plan,
            "resolved_variables": resolved_vars,
            "variable_traces": variable_traces,
            "grounding_sources": all_grounding_sources,
            "evidence_sources": evidence_sources,
            "adjudication": adjudication,
        }

        source_uri = "crp:pipeline"
        if all_grounding_sources:
            source_uri = all_grounding_sources[0].get("url", source_uri)

        raw_content = ""
        if adjudication:
            raw_content = str(adjudication.get("reasoning_trace", ""))[:500]

        return EvidenceItem(
            evidence_id=evidence_id,
            requirement_id=requirement.requirement_id,
            provenance=Provenance(
                source_id="crp",
                source_uri=source_uri,
                tier=2,
                fetched_at=ctx.now(),
                content_hash=self._hash_content(
                    json.dumps(extracted_fields, default=str)
                ),
            ),
            raw_content=raw_content,
            parsed_value=parsed_value,
            extracted_fields=extracted_fields,
            success=success,
            error=None if success else "CRP pipeline did not produce a verdict",
        )

    # ------------------------------------------------------------------
    # Step 1 — Architect
    # ------------------------------------------------------------------

    def _step_architect(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
    ) -> tuple[dict[str, Any] | None, ToolCallRecord]:
        market = prompt_spec.market
        semantics = prompt_spec.prediction_semantics
        rules = market.resolution_rules
        rules_text = "\n".join(
            f"- [{r.rule_id}] {r.description}"
            for r in rules.get_sorted_rules()
        ) if rules and rules.rules else "No explicit rules."

        assumptions_list = prompt_spec.extra.get("assumptions", [])
        assumptions_str = "\n".join(
            f"- {a}" for a in assumptions_list
        ) if assumptions_list else "None"

        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        user_prompt = (
            f"### MARKET QUESTION\n{market.question}\n\n"
            f"### EVENT DEFINITION\n{market.event_definition}\n\n"
            f"### PREDICTION SEMANTICS\n"
            f"Entity: {semantics.target_entity}\n"
            f"Predicate: {semantics.predicate}\n"
            f"Threshold: {semantics.threshold or 'N/A'}\n"
            f"Timeframe: {semantics.timeframe or 'N/A'}\n\n"
            f"### RESOLUTION RULES\n{rules_text}\n\n"
            f"### ASSUMPTIONS\n{assumptions_str}\n\n"
            f"### CURRENT DATE\n{current_time_str}\n\n"
            f"Decompose this into variables and a logic condition."
        )

        print(f"[ARCHITECT] User prompt:\n{user_prompt}")

        record = ToolCallRecord(
            tool="crp:architect",
            input={"market_id": prompt_spec.market_id},
            started_at=ctx.now().isoformat(),
        )

        messages = [
            {"role": "system", "content": CRP_ARCHITECT_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        plan = self._llm_call_with_retry(ctx, messages)
        print(f"[ARCHITECT] LLM response: {json.dumps(plan, indent=2, default=str)}")
        record.ended_at = ctx.now().isoformat()

        if plan is None:
            record.error = "Failed to decompose question"
            record.output = {"success": False}
        else:
            record.output = {
                "success": True,
                "variable_count": len(plan.get("variables", [])),
                "logic_condition": plan.get("logic_condition", ""),
            }

        return plan, record

    # ------------------------------------------------------------------
    # Step 2 — Critical Investigator (ReAct multi-turn per variable)
    # ------------------------------------------------------------------

    def _step_investigate(
        self,
        ctx: "AgentContext",
        variable: dict[str, Any],
        prompt_spec: PromptSpec,
        logic_condition: str,
        rules_text: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, str]], ToolCallRecord]:
        """
        Multi-turn ReAct investigation for a single variable.

        The investigator can Think → Search → Critique → Re-Search for up
        to _MAX_INVESTIGATOR_TURNS turns.  It signals completion by including
        "FINAL_ANSWER" in its response.
        """
        var_name = variable["name"]

        record = ToolCallRecord(
            tool=f"crp:investigator:{var_name}",
            input={"variable": var_name},
            started_at=ctx.now().isoformat(),
        )

        api_key = self._resolve_google_api_key(ctx)
        if not api_key:
            record.ended_at = ctx.now().isoformat()
            record.error = "Google API key not available (set GOOGLE_API_KEY)"
            record.output = {"success": False}
            return None, [], record

        current_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Build the investigator system prompt with variable-specific context
        system_prompt = CRP_INVESTIGATOR_PROMPT.format(
            var_name=variable["name"],
            var_type=variable.get("type", "Unknown"),
            var_description=variable.get("description", ""),
            context=prompt_spec.market.question,
            rules=rules_text,
            logic_condition=logic_condition,
            current_date=current_time_str,
        )

        # Initial user message kicks off the investigation
        initial_message = (
            f"Find the value of '{var_name}'. "
            f"Event: {prompt_spec.market.event_definition}\n"
            f"Search the web and report what you find."
        )

        print(f"  [INVESTIGATOR] System prompt (abbreviated):\n"
              f"    Variable: {var_name} ({variable.get('type', '?')})\n"
              f"    Description: {variable.get('description', '')}\n"
              f"    Logic: {logic_condition}")

        all_grounding_sources: list[dict[str, str]] = []
        turn_logs: list[dict[str, Any]] = []

        try:
            client = self._get_gemini_client(api_key)

            # Build multi-turn contents list
            # First turn: system context as part of user message + initial query
            contents = [
                {"role": "user", "parts": [{"text": system_prompt + "\n\n" + initial_message}]},
            ]

            for turn in range(_MAX_INVESTIGATOR_TURNS):
                print(f"\n  [INVESTIGATOR] Turn {turn + 1}/{_MAX_INVESTIGATOR_TURNS}")

                response = self._call_gemini_grounded_multiturn(client, contents)
                text = self._extract_gemini_text(response)
                grounding = self._extract_gemini_grounding(response)

                # Collect grounding sources
                turn_sources = [
                    {"url": s.get("uri", ""), "title": s.get("title", "")}
                    for s in grounding.get("sources", [])
                ]
                all_grounding_sources.extend(turn_sources)

                turn_log = {
                    "turn": turn + 1,
                    "response_text": text[:500],
                    "search_queries": grounding.get("search_queries", []),
                    "grounding_source_count": len(turn_sources),
                }
                turn_logs.append(turn_log)

                print(f"  [INVESTIGATOR] Search queries: {grounding.get('search_queries', [])}")
                print(f"  [INVESTIGATOR] Sources found: {len(turn_sources)}")
                for gs in turn_sources[:3]:
                    print(f"    - {gs.get('title', '?')}: {gs.get('url', '?')}")

                # Check if investigator produced a final answer
                if "FINAL_ANSWER" in text:
                    print(f"  [INVESTIGATOR] FINAL_ANSWER found on turn {turn + 1}")
                    result = self._extract_final_answer(text)
                    if result:
                        # Merge grounding sources into evidence
                        existing_urls = {
                            s.get("url", "")
                            for s in result.get("evidence_sources", [])
                        }
                        for gs in all_grounding_sources:
                            if gs["url"] and gs["url"] not in existing_urls:
                                result.setdefault("evidence_sources", []).append({
                                    "url": gs["url"],
                                    "title": gs.get("title", ""),
                                    "key_fact": gs.get("title", ""),
                                    "date_published": None,
                                })

                        print(f"  [INVESTIGATOR] Result: value={result.get('value')}, "
                              f"status={result.get('status')}, "
                              f"confidence={result.get('confidence')}")
                        reasoning = result.get("reasoning", "")
                        if reasoning:
                            print(f"  [INVESTIGATOR] Reasoning: {reasoning[:300]}")

                        record.ended_at = ctx.now().isoformat()
                        record.output = {
                            "success": True,
                            "turns_used": turn + 1,
                            "value": result.get("value"),
                            "status": result.get("status"),
                            "turn_logs": turn_logs,
                        }
                        return result, all_grounding_sources, record

                # No final answer yet — log the thinking and continue
                thought_preview = text[:200].replace("\n", " ")
                print(f"  [INVESTIGATOR] Thinking: {thought_preview}...")

                # Add model response + follow-up to conversation for next turn
                contents.append({
                    "role": "model",
                    "parts": [{"text": text}],
                })
                contents.append({
                    "role": "user",
                    "parts": [{"text": (
                        "Continue your investigation. Apply the Core Principles "
                        "(entity disambiguation, missing-data skepticism, "
                        "normalization). If you have enough evidence, output "
                        "FINAL_ANSWER with the JSON. Otherwise, explain what "
                        "you plan to search next."
                    )}],
                })

            # Exhausted all turns without FINAL_ANSWER
            print(f"  [INVESTIGATOR] Max turns reached without FINAL_ANSWER")
            record.ended_at = ctx.now().isoformat()
            record.error = f"Max turns ({_MAX_INVESTIGATOR_TURNS}) without FINAL_ANSWER"
            record.output = {
                "success": False,
                "turns_used": _MAX_INVESTIGATOR_TURNS,
                "turn_logs": turn_logs,
            }
            return None, all_grounding_sources, record

        except Exception as e:
            print(f"  [INVESTIGATOR] ERROR: {e}")
            record.ended_at = ctx.now().isoformat()
            record.error = f"Investigation failed: {e}"
            record.output = {"success": False, "turn_logs": turn_logs}
            return None, all_grounding_sources, record

    def _extract_final_answer(self, text: str) -> dict[str, Any] | None:
        """Extract the JSON object after FINAL_ANSWER marker."""
        # Find everything after FINAL_ANSWER
        marker_idx = text.find("FINAL_ANSWER")
        if marker_idx < 0:
            return None

        remainder = text[marker_idx + len("FINAL_ANSWER"):]

        # Try markdown code block first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", remainder)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try raw {…} extraction
        start = remainder.find("{")
        end = remainder.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(remainder[start:end])
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # Step 3 — Adjudicator
    # ------------------------------------------------------------------

    def _step_adjudicate(
        self,
        ctx: "AgentContext",
        logic_condition: str,
        resolved_vars: dict[str, Any],
        invalidity_conditions: list[str],
    ) -> tuple[dict[str, Any] | None, ToolCallRecord]:
        record = ToolCallRecord(
            tool="crp:adjudicator",
            input={"logic_condition": logic_condition},
            started_at=ctx.now().isoformat(),
        )

        user_prompt = (
            f"### LOGIC CONDITION\n{logic_condition}\n\n"
            f"### RESOLVED VARIABLES\n"
            f"{json.dumps(resolved_vars, indent=2, default=str)}\n\n"
            f"### INVALIDITY CONDITIONS\n"
            f"{json.dumps(invalidity_conditions, default=str)}\n\n"
            f"Evaluate the logic condition with the resolved variables "
            f"and produce a verdict."
        )

        print(f"[ADJUDICATOR] Prompt:\n{user_prompt}")

        messages = [
            {"role": "system", "content": CRP_ADJUDICATOR_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        adjudication = self._llm_call_with_retry(ctx, messages)
        print(f"[ADJUDICATOR] Response: {json.dumps(adjudication, indent=2, default=str)}")
        record.ended_at = ctx.now().isoformat()

        if adjudication is None:
            record.error = "Adjudication failed"
            record.output = {"success": False}
        else:
            record.output = {
                "success": True,
                "verdict": adjudication.get("verdict"),
                "confidence": adjudication.get("confidence"),
            }

        return adjudication, record

    # ------------------------------------------------------------------
    # Gemini SDK helpers
    # ------------------------------------------------------------------

    def _resolve_google_api_key(self, ctx: "AgentContext") -> str | None:
        if ctx.config and ctx.config.llm.provider == "google" and ctx.config.llm.api_key:
            return ctx.config.llm.api_key
        return os.getenv("GOOGLE_API_KEY")

    def _get_gemini_client(self, api_key: str) -> Any:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            )
        return genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=80_000),
        )

    def _call_gemini_grounded_multiturn(
        self, client: Any, contents: list[dict[str, Any]],
    ) -> Any:
        """Call Gemini with Google Search grounding, supporting multi-turn."""
        from google.genai import types

        def _do_call() -> Any:
            return client.models.generate_content(
                model=self.DEFAULT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_call)
            return future.result(timeout=_GEMINI_TIMEOUT)

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []):
                text = getattr(part, "text", None)
                if text:
                    return text
        return ""

    @staticmethod
    def _extract_gemini_grounding(response: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"sources": [], "search_queries": []}
        for candidate in getattr(response, "candidates", []):
            gm = getattr(candidate, "grounding_metadata", None)
            if gm is None:
                continue
            queries = getattr(gm, "web_search_queries", None)
            if queries:
                result["search_queries"] = list(queries)
            chunks = getattr(gm, "grounding_chunks", None)
            if chunks:
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        result["sources"].append({
                            "uri": getattr(web, "uri", ""),
                            "title": getattr(web, "title", ""),
                        })
        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _llm_call_with_retry(
        self,
        ctx: "AgentContext",
        messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        response = ctx.llm.chat(messages)
        raw_output = response.content

        last_error: str | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self._extract_json(raw_output)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = str(e)
                ctx.warning(f"CRP JSON parse attempt {attempt + 1} failed: {last_error}")
                if attempt < self.MAX_RETRIES:
                    repair_prompt = (
                        f"The JSON was invalid: {last_error}. "
                        "Please fix and return ONLY valid JSON."
                    )
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": repair_prompt})
                    response = ctx.llm.chat(messages)
                    raw_output = response.content

        ctx.error(f"CRP LLM call failed after {self.MAX_RETRIES + 1} attempts: {last_error}")
        return None

    def _validate_output(
        self,
        bundle: EvidenceBundle,
        tool_plan: ToolPlan,
    ) -> VerificationResult:
        checks: list[CheckResult] = []

        if bundle.has_evidence:
            checks.append(CheckResult.passed(
                check_id="has_evidence",
                message=f"Collected {len(bundle.items)} evidence items",
            ))
        else:
            checks.append(CheckResult.failed(
                check_id="has_evidence",
                message="No evidence was collected",
            ))

        unfulfilled = bundle.requirements_unfulfilled
        if not unfulfilled:
            checks.append(CheckResult.passed(
                check_id="requirements_fulfilled",
                message="All requirements fulfilled",
            ))
        else:
            checks.append(CheckResult.warning(
                check_id="requirements_fulfilled",
                message=f"Unfulfilled requirements: {unfulfilled}",
            ))

        valid_items = bundle.get_valid_evidence()
        for item in valid_items:
            adj = item.extracted_fields.get("adjudication")
            if adj and adj.get("verdict") in ("YES", "NO", "INVALID"):
                checks.append(CheckResult.passed(
                    check_id="crp_pipeline_complete",
                    message=f"CRP pipeline completed with verdict: {adj['verdict']}",
                ))
                break
        else:
            if valid_items:
                checks.append(CheckResult.warning(
                    check_id="crp_pipeline_complete",
                    message="CRP pipeline did not produce a clear verdict",
                ))

        ok = all(c.ok for c in checks)
        return VerificationResult(ok=ok, checks=checks)

    @staticmethod
    def _generate_evidence_id(requirement_id: str) -> str:
        key = f"crp:{requirement_id}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _normalize_evidence_sources(
    raw_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _TIER_MAP = {"tier 1": 1, "tier 2": 2, "tier 3": 3}
    normalized: list[dict[str, Any]] = []

    for src in raw_sources:
        if not isinstance(src, dict):
            continue

        raw_tier = src.get("credibility_tier", 3)
        if isinstance(raw_tier, str):
            tier = _TIER_MAP.get(raw_tier.lower().strip(), 3)
        else:
            tier = int(raw_tier) if raw_tier in (1, 2, 3) else 3

        raw_supports = src.get("supports", "N/A")
        if isinstance(raw_supports, bool):
            supports = "YES" if raw_supports else "NO"
        else:
            supports = str(raw_supports).upper() if raw_supports else "N/A"
            if supports not in ("YES", "NO", "N/A"):
                supports = "N/A"

        key_fact = src.get("key_fact", "")

        normalized.append({
            "url": src.get("url", ""),
            "source_id": src.get("source_id"),
            "credibility_tier": tier,
            "key_fact": str(key_fact)[:300] if key_fact else "",
            "supports": supports,
            "date_published": src.get("date_published"),
        })

    return normalized
