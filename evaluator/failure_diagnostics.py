"""Offline failure diagnostics for the public-set evaluation sessions.

WHAT THIS SCRIPT IS FOR
------------------------
This is a *development analysis tool only*. It re-runs the same simulated
shopper conversations the local evaluator runs, but afterwards -- only
afterwards, only outside the agent -- it looks up the true target product
for each miss and tries to explain why the agent didn't find it.

GROUND-TRUTH SAFETY
--------------------
`starter/agent.py` never sees the target `parent_asin`, and this script does
not change that. We only read `ground_truth` from `public_set.jsonl` *after*
`Agent.respond()` has already produced its Top-10 for every turn, purely for
scoring/labeling here in the diagnostics script. Nothing computed in this
file is fed back into the Agent, and nothing here is used to alter runtime
behavior, weights, aliases, or clarification logic.

WHAT IT PRODUCES
-----------------
- debug_logs/failure_diagnostics.csv   (one row per session that missed Top-10)
- debug_logs/failure_diagnostics_summary.json (aggregate counts)
- a printed summary to stdout
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

# Make both the project root (for `starter`) and this file's directory
# (for `local_evaluator`) importable regardless of where this script is run
# from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from starter.agent import Agent, _structured_product_fields, _shares_word, _terms  # noqa: E402
from . import local_evaluator as ev


DEEP_POOL_SIZE = 500  # diagnostic-only retrieval depth; normal runtime stays top_k=10


def rank_bucket(rank: int | None) -> str:
    """Classify a target's rank into one of the buckets requested in the brief."""
    if rank is None:
        return "not_top500"
    if rank <= 10:
        return "top10"  # shouldn't occur for a recorded miss, kept defensively
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    return "101-500"


def deep_rank(agent: Agent, terms: list[str], target_asin: str) -> int | None:
    """Search deeper than normal runtime (top_k=10) purely for diagnostics.

    Reuses the exact same BM25 expression/weights the agent already built
    for its last turn -- we're not changing ranking, just looking further
    down the same ordered list to see where the true target actually sits.
    """
    expression = " OR ".join(f'"{term}"' for term in terms)
    if not expression:
        return None
    rows = agent.connection.execute(
        f"SELECT parent_asin FROM products WHERE products MATCH ? "
        f"ORDER BY {agent._bm25_sql} LIMIT ?",
        (expression, DEEP_POOL_SIZE),
    ).fetchall()
    ids = [str(row[0]) for row in rows]
    if target_asin in ids:
        return ids.index(target_asin) + 1
    return None


def run_session(agent: Agent, sample: dict, products: dict[str, dict], categories: dict[str, list[str]]) -> dict:
    """Re-run one conversation exactly like local_evaluator.evaluate(), but
    also collect the extra bookkeeping (clarification attributes asked,
    final query terms, final state) that the diagnostics need.

    This deliberately mirrors local_evaluator.evaluate()'s turn loop so the
    hit/miss/rank result matches the real scored run.
    """
    session_id = f"diag_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])  # diagnostics-only read of ground truth
    intent_card, behavior = ev.materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = ev.initial_message(effective_sample, ev.coarse_category(categories.get(target, [])), disclosed)

    asked_attributes: list[str] = []
    hit_turn: int | None = None
    best_rank: int | None = None
    last_message_sent = user_message
    last_top10: list[str] = []

    for turn in range(1, ev.MAX_TURNS + 1):
        last_message_sent = user_message
        try:
            response = agent.respond(session_id, user_message, turn, ev.TOP_K)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        ask_attribute = response.get("ask_attribute")
        if isinstance(ask_attribute, str):
            asked_attributes.append(ask_attribute)

        ranked = ev.normalize_recommendations(response.get("recommendations"), set(products.keys()))
        last_top10 = ranked
        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            break
        if turn == ev.MAX_TURNS:
            break
        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = ev.customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )

    final_state = agent._sessions[session_id]
    # Recompute the terms used on the final turn, purely for diagnostics --
    # read-only, matches exactly what respond() built internally that turn.
    final_query_terms = agent._build_query_terms(final_state, last_message_sent)

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target": target,
        "hit": hit_turn is not None,
        "hit_turn": hit_turn,
        "best_rank": best_rank,
        "turns_used": hit_turn if hit_turn is not None else ev.MAX_TURNS,
        "asked_attributes": asked_attributes,
        "override_applied": override_applied,
        "final_state": {
            "category": final_state.get("category"),
            "brand": final_state.get("brand"),
            "department": final_state.get("department"),
            "material": final_state.get("material"),
            "color": final_state.get("color"),
            "max_price": final_state.get("max_price"),
            "use_case": final_state.get("use_case"),
            "style": list(final_state.get("style") or []),
        },
        "final_query_terms": final_query_terms,
        "final_top10": last_top10,
        "behavior": effective_sample.get("behavior", {}),
    }


def diagnose_miss(session: dict, agent: Agent, products: dict[str, dict], categories: dict[str, list[str]]) -> dict:
    """Everything below only runs for sessions that missed Top-10. This is
    where we read the true target's catalog metadata -- strictly for
    labeling this CSV row, never fed back to the Agent.
    """
    target = session["target"]
    product = products.get(target, {})
    structured = _structured_product_fields(product)
    target_categories = categories.get(target, [])

    rank = deep_rank(agent, session["final_query_terms"], target)
    bucket = rank_bucket(rank)

    causes: list[str] = []
    if bucket == "11-20":
        causes.append("NEAR_MISS_RANKING")
    elif bucket in ("21-50", "51-100"):
        causes.append("RETRIEVED_BUT_LOW")
    elif bucket in ("101-500", "not_top500"):
        causes.append("RETRIEVAL_FAILURE")

    state = session["final_state"]

    state_category = state.get("category")
    if state_category and target_categories:
        if not _shares_word(state_category, " ".join(target_categories)):
            causes.append("CATEGORY_MISMATCH")

    state_brand = state.get("brand")
    if state_brand and structured.get("brand"):
        a, b = state_brand.strip().lower(), structured["brand"].strip().lower()
        if a and b and a != b and a not in b and b not in a:
            causes.append("BRAND_MISMATCH")

    state_color = state.get("color")
    if state_color and structured.get("color"):
        if state_color.strip().lower() not in structured["color"].strip().lower():
            causes.append("COLOR_MISMATCH")

    state_material = state.get("material")
    if state_material and structured.get("material"):
        if state_material.strip().lower() not in structured["material"].strip().lower():
            causes.append("MATERIAL_MISMATCH")

    if session["scenario_type"] == "intent_override" and session["override_applied"]:
        override = session["behavior"].get("override") or {}
        new_value = str(override.get("new_value", ""))
        if new_value:
            new_value_terms = set(_terms(new_value))
            state_terms: set[str] = set()
            for key in ("category", "brand", "department", "material", "color", "use_case"):
                value = state.get(key)
                if value:
                    state_terms |= set(_terms(str(value)))
            for style_value in state.get("style") or []:
                state_terms |= set(_terms(str(style_value)))
            if new_value_terms and not (new_value_terms & state_terms):
                causes.append("STALE_STATE_SUSPECTED")

    query_terms = session["final_query_terms"]
    target_text = ev.searchable_text(product).lower()
    matching_terms = sum(1 for term in query_terms if term in target_text)

    price = product.get("price")

    return {
        "sample_id": session["sample_id"],
        "scenario_type": session["scenario_type"],
        "target_parent_asin": target,
        "target_title": product.get("title", ""),
        "target_categories": " | ".join(target_categories),
        "target_brand": structured.get("brand") or "",
        "target_department": structured.get("department") or "",
        "target_material": structured.get("material") or "",
        "target_color": structured.get("color") or "",
        "target_price": price if isinstance(price, (int, float)) else "",
        "final_rank": rank if rank is not None else "",
        "rank_bucket": bucket,
        "agent_top10": " | ".join(session["final_top10"]),
        "final_query_terms": " ".join(query_terms),
        "matching_query_terms": matching_terms,
        "state_category": state.get("category") or "",
        "state_brand": state.get("brand") or "",
        "state_department": state.get("department") or "",
        "state_material": state.get("material") or "",
        "state_color": state.get("color") or "",
        "state_max_price": state.get("max_price") if state.get("max_price") is not None else "",
        "state_use_case": state.get("use_case") or "",
        "state_style": " | ".join(state.get("style") or []),
        "clarification_attributes_asked": " | ".join(session["asked_attributes"]),
        "turns_used": session["turns_used"],
        "override_detected": session["scenario_type"] == "intent_override" and session["override_applied"],
        "failure_causes": " | ".join(causes),
        "_has_material_meta": bool(structured.get("material")),
        "_has_color_meta": bool(structured.get("color")),
        "_has_department_meta": bool(structured.get("department")),
        "_has_brand_meta": bool(structured.get("brand")),
        "_has_category_meta": bool(target_categories),
        "_has_price_meta": isinstance(price, (int, float)),
    }


def main() -> None:
    catalog_path = ROOT / "data" / "catalog.jsonl"
    dataset_path = ROOT / "data" / "public_set.jsonl"
    out_csv = ROOT / "debug_logs" / "failure_diagnostics.csv"
    out_summary = ROOT / "debug_logs" / "failure_diagnostics_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    samples = ev.load_jsonl(dataset_path)
    catalog_ids, categories, products = ev.catalog_index(catalog_path)

    agent = Agent(str(catalog_path))

    all_rows: list[dict] = []
    scenario_totals: Counter[str] = Counter()
    hit_count = 0
    miss_rows: list[dict] = []

    for sample in samples:
        session = run_session(agent, sample, products, categories)
        scenario_totals[session["scenario_type"]] += 1
        if session["hit"]:
            hit_count += 1
            continue
        row = diagnose_miss(session, agent, products, categories)
        miss_rows.append(row)

    # ---- write CSV -----------------------------------------------------
    fieldnames = [
        "sample_id", "scenario_type", "target_parent_asin", "target_title",
        "target_categories", "target_brand", "target_department", "target_material",
        "target_color", "target_price", "final_rank", "rank_bucket", "agent_top10",
        "final_query_terms", "matching_query_terms",
        "state_category", "state_brand", "state_department", "state_material",
        "state_color", "state_max_price", "state_use_case", "state_style",
        "clarification_attributes_asked", "turns_used", "override_detected",
        "failure_causes",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in miss_rows:
            writer.writerow(row)

    # ---- aggregate: bucket counts overall + by scenario -----------------
    bucket_order = ["11-20", "21-50", "51-100", "101-500", "not_top500"]
    overall_buckets: Counter[str] = Counter(row["rank_bucket"] for row in miss_rows)
    scenario_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in miss_rows:
        scenario_buckets[row["scenario_type"]][row["rank_bucket"]] += 1

    # ---- aggregate: failure cause counts ---------------------------------
    cause_counts: Counter[str] = Counter()
    for row in miss_rows:
        for cause in [c for c in row["failure_causes"].split(" | ") if c]:
            cause_counts[cause] += 1

    # ---- average rank where retrieved (i.e. rank is not blank) ----------
    retrieved_ranks = [row["final_rank"] for row in miss_rows if row["final_rank"] != ""]
    avg_retrieved_rank = statistics.fmean(retrieved_ranks) if retrieved_ranks else None

    # ---- attribute availability on misses --------------------------------
    n_miss = len(miss_rows)

    def pct(flag_key: str) -> float:
        if n_miss == 0:
            return 0.0
        return round(100.0 * sum(1 for r in miss_rows if r.get(flag_key)) / n_miss, 1)

    # re-derive the boolean flags from diagnose_miss's internal (private,
    # underscore-prefixed) keys, then strip them before saving/printing.
    availability = {
        "category": pct("_has_category_meta"),
        "brand": pct("_has_brand_meta"),
        "department": pct("_has_department_meta"),
        "material": pct("_has_material_meta"),
        "color": pct("_has_color_meta"),
        "price": pct("_has_price_meta"),
    }
    for row in miss_rows:
        for key in list(row.keys()):
            if key.startswith("_has_"):
                del row[key]

    # ---- query coverage (how many final query terms hit the target text) -
    coverage_buckets = Counter()
    for row in miss_rows:
        n = row["matching_query_terms"]
        if n == 0:
            coverage_buckets["0"] += 1
        elif n == 1:
            coverage_buckets["1"] += 1
        elif n <= 3:
            coverage_buckets["2-3"] += 1
        else:
            coverage_buckets["4+"] += 1

    summary = {
        "total_sessions": len(samples),
        "hit_count": hit_count,
        "miss_count": n_miss,
        "rank_buckets_overall": {b: overall_buckets.get(b, 0) for b in bucket_order},
        "rank_buckets_by_scenario": {
            scenario: {b: scenario_buckets[scenario].get(b, 0) for b in bucket_order}
            for scenario in sorted(scenario_buckets)
        },
        "failure_cause_counts": dict(cause_counts),
        "average_target_rank_where_retrieved": round(avg_retrieved_rank, 2) if avg_retrieved_rank else None,
        "attribute_availability_pct_on_misses": availability,
        "query_coverage_on_misses": {
            "0_matching_terms": coverage_buckets.get("0", 0),
            "1_matching_term": coverage_buckets.get("1", 0),
            "2_3_matching_terms": coverage_buckets.get("2-3", 0),
            "4_plus_matching_terms": coverage_buckets.get("4+", 0),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # ---- print report ------------------------------------------------
    print(f"Total sessions: {len(samples)}")
    print(f"Hit@10: {hit_count}")
    print(f"Misses: {n_miss}")
    print()
    print("Miss target locations:")
    print(f"Rank 11-20:  {overall_buckets.get('11-20', 0)}")
    print(f"Rank 21-50:  {overall_buckets.get('21-50', 0)}")
    print(f"Rank 51-100: {overall_buckets.get('51-100', 0)}")
    print(f"Rank 101-500: {overall_buckets.get('101-500', 0)}")
    print(f"Not Top500: {overall_buckets.get('not_top500', 0)}")
    print()
    for scenario in sorted(scenario_buckets):
        counts = scenario_buckets[scenario]
        print(scenario.upper())
        print(f"misses: {sum(counts.values())}")
        print(f"11-20: {counts.get('11-20', 0)}")
        print(f"21-50: {counts.get('21-50', 0)}")
        print(f"51-100: {counts.get('51-100', 0)}")
        print(f"101-500: {counts.get('101-500', 0)}")
        print(f">500: {counts.get('not_top500', 0)}")
        print()
    print("Likely cause counts:")
    for cause in ("NEAR_MISS_RANKING", "RETRIEVED_BUT_LOW", "RETRIEVAL_FAILURE",
                  "CATEGORY_MISMATCH", "BRAND_MISMATCH", "COLOR_MISMATCH",
                  "MATERIAL_MISMATCH", "STALE_STATE_SUSPECTED"):
        print(f"{cause}: {cause_counts.get(cause, 0)}")
    print()
    if avg_retrieved_rank:
        print(f"Average target rank where retrieved (<=500): {avg_retrieved_rank:.2f}")
    print()
    print("Attribute availability on misses:")
    for key, value in availability.items():
        print(f"Missed targets with {key} metadata: {value}%")
    print()
    print("Query coverage on misses (final query terms hitting target's indexed text):")
    print(f"0 matching query terms: {coverage_buckets.get('0', 0)}")
    print(f"1 matching term: {coverage_buckets.get('1', 0)}")
    print(f"2-3 matching terms: {coverage_buckets.get('2-3', 0)}")
    print(f"4+ matching terms: {coverage_buckets.get('4+', 0)}")
    print()
    print(f"CSV written to: {out_csv}")
    print(f"Summary written to: {out_summary}")


if __name__ == "__main__":
    main()
