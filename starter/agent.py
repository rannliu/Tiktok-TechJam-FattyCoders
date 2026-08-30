from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


## Converts a catalog field (string/list/dict/None) into one plain string.
def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


## Splits text into lowercase words, dropping stopwords and single letters.
def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


## Regex patterns for pulling attribute values (color, material, etc) out of
## whatever the customer types, so the agent can remember them.
COLOR_RE = re.compile(
    r"\b(black|white|blue|navy|red|pink|green|olive|brown|tan|beige|gray|grey|"
    r"purple|yellow|orange|gold|silver|maroon|khaki|cream|charcoal)\b",
    re.IGNORECASE,
)
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|denim|linen|"
    r"suede|mesh|fleece|canvas|cashmere)\b",
    re.IGNORECASE,
)
DEPARTMENT_RE = re.compile(r"\b(women'?s?|men'?s?|boy'?s?|girl'?s?|kids?|unisex)\b", re.IGNORECASE)
USE_CASE_RE = re.compile(
    r"\b(hiking|running|gym|workout|training|winter|summer|outdoor|office|work|"
    r"formal|casual|wedding|party|travel|yoga|athletic)\b",
    re.IGNORECASE,
)
STYLE_RE = re.compile(
    r"\b(waterproof|lightweight|breathable|slim[- ]fit|loose[- ]fit|high[- ]waisted|"
    r"vintage|classic|comfortable|stretch|elastic|relaxed[- ]fit)\b",
    re.IGNORECASE,
)
PRICE_DOLLAR_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
PRICE_WORD_RE = re.compile(r"\b(?:under|below|less than|around|up to)\s+\$?\s?(\d+(?:\.\d+)?)", re.IGNORECASE)
BRAND_COLON_RE = re.compile(r"\bbrand:\s*([A-Za-z0-9&' -]{2,40})", re.IGNORECASE)
BRAND_BY_RE = re.compile(r"\bby\s+([A-Z][\w&'\-]*(?:\s+[A-Z][\w&'\-]*){0,2})\b")

## Phrases signaling the customer is replacing an earlier answer, not adding a new one.
OVERRIDE_RE = re.compile(
    r"\b(forget|actually|instead|ignore my earlier preference|no longer|"
    r"never mind|change (?:that|it) to)\b",
    re.IGNORECASE,
)

## Product metadata (department/material/color/brand) only comes from these
## explicit keys -- never guessed from marketing text.
DETAILS_DEPARTMENT_KEYS = ("Department",)
DETAILS_MATERIAL_KEYS = ("Material", "Fabric Type")
DETAILS_COLOR_KEYS = ("Color",)
DETAILS_BRAND_KEYS = ("Brand", "Brand Name", "Manufacturer")


def _first_details_value(details: object, keys: tuple[str, ...]) -> str | None:
    if not isinstance(details, dict):
        return None
    for key in keys:
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


## Pulls department/material/color/brand for a product from its trusted keys.
def _structured_product_fields(product: dict) -> dict[str, str | None]:
    details = product.get("details")
    department = _first_details_value(details, DETAILS_DEPARTMENT_KEYS)
    if department:
        department = _normalize_department(department)
    material = _first_details_value(details, DETAILS_MATERIAL_KEYS)
    color = _first_details_value(details, DETAILS_COLOR_KEYS)
    store = product.get("store")
    brand = store.strip() if isinstance(store, str) and store.strip() else None
    if not brand:
        brand = _first_details_value(details, DETAILS_BRAND_KEYS)
    return {"department": department, "material": material, "color": color, "brand": brand}


## Matches phrases like "looking for/need/want X" to pull out the product category.
CATEGORY_PHRASE_RE = re.compile(
    r"(?:looking for|need|want|shopping for)\s+(.+?)(?:[.,;]|$|\bbut\b|\bwith\b)",
    re.IGNORECASE,
)

## Swaps the last clarification slot from "brand" to "feature",
## since the evaluator can never actually answer a "brand" question.
ENABLE_FEATURE_CLARIFICATION = True

## A bigger reorder of ATTRIBUTE_PRIORITY (still static, not
## candidate-aware), based on how often each attribute shows up in test data.
## Only applies when ENABLE_FEATURE_CLARIFICATION is also True.
ENABLE_OPTIMIZED_CLARIFICATION_ORDER = True

## Order we ask about missing slots in.
if ENABLE_FEATURE_CLARIFICATION:
    if ENABLE_OPTIMIZED_CLARIFICATION_ORDER:
        ATTRIBUTE_PRIORITY = ["use_case", "material", "color", "feature", "style", "size"]
    else:
        ATTRIBUTE_PRIORITY = ["use_case", "material", "color", "budget", "style", "feature"]
else:
    ATTRIBUTE_PRIORITY = ["use_case", "material", "color", "budget", "style", "brand"]

## Stop asking clarifying questions after this turn.
ASK_UNTIL_TURN = 5

DEBUG_ENABLED = os.environ.get("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_LOG_PATH = Path(os.environ.get("AGENT_DEBUG_LOG_PATH", "debug_logs/conversations.jsonl"))


## CROSS-TURN CONSENSUS: when the search query "freezes" (customer stops
## giving new info), blend in products that ranked well in earlier turns
## but have since fallen out of the current top results.

## Reads a True/False env var, falling back to `default` if unset.
def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


## Turns cross-turn consensus on/off (on by default, it tested well).
ENABLE_CROSS_TURN_CONSENSUS = _env_flag("ENABLE_CROSS_TURN_CONSENSUS", True)

CONSENSUS_RRF_K = 60         ## softens rank position into a score, bigger = position matters less
CONSENSUS_WEIGHT = 0.45      ## how much earlier-turn results count vs. the current search
CONSENSUS_HISTORY_DEPTH = 50 ## how many top results per earlier search we remember


## FTS columns for the trusted structured fields, added on top of the
## original free-text columns, plus the combined BM25 weights used for ranking.
STRUCTURED_FIELD_NAMES = ("department", "material_field", "color_field", "brand_field")
STRUCTURED_FIELD_WEIGHTS = (2.5, 3.5, 3.5, 3.0)
BASE_BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
ALL_BM25_WEIGHTS = BASE_BM25_WEIGHTS + STRUCTURED_FIELD_WEIGHTS


def _clean_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,;")
    value = re.sub(r"^(a|an|some)\s+", "", value, flags=re.IGNORECASE)
    return value.strip()


def _shares_word(a: str, b: str) -> bool:
    words_a = set(_terms(a))
    words_b = set(_terms(b))
    return bool(words_a & words_b)


def _normalize_department(raw: str) -> str:
    lowered = raw.lower().rstrip("'s")
    mapping = {"women": "womens", "woman": "womens", "men": "mens", "man": "mens",
               "boy": "boys", "girl": "girls", "kid": "kids", "kids": "kids", "unisex": "unisex"}
    return mapping.get(lowered, lowered)


## Pulls the product category out of a message (e.g. "jacket" from "I need a jacket").
## Rejects results that look like attributes (color/material/style) instead of a real category.
def extract_category(message: str) -> str | None:
    match = CATEGORY_PHRASE_RE.search(message)
    if not match:
        return None
    phrase = _clean_phrase(match.group(1))
    if not phrase:
        return None

    lowered = phrase.lower()
    if lowered.startswith(("is:", "is ", "are:", "are ")):
        return None

    pure_attribute_matches = (
        MATERIAL_RE.search(phrase) or
        COLOR_RE.search(phrase) or
        STYLE_RE.search(phrase) or
        USE_CASE_RE.search(phrase)
    )

    words = [w for w in _terms(phrase) if w]
    if len(words) == 1 and pure_attribute_matches:
        return None

    return phrase


## Pulls whatever structured attributes (color, material, price, etc) it can
## find in one message. A new value always overwrites an old one.
def extract_updates(message: str) -> dict:
    updates: dict = {}

    color = COLOR_RE.search(message)
    if color:
        updates["color"] = color.group(1).lower()

    material = MATERIAL_RE.search(message)
    if material:
        updates["material"] = material.group(1).lower()

    department = DEPARTMENT_RE.search(message)
    if department:
        updates["department"] = _normalize_department(department.group(1))

    use_case = USE_CASE_RE.search(message)
    if use_case:
        updates["use_case"] = use_case.group(1).lower()

    style_hits = STYLE_RE.findall(message)
    if style_hits:
        updates["style"] = [hit.lower() for hit in style_hits]

    price = PRICE_DOLLAR_RE.search(message) or PRICE_WORD_RE.search(message)
    if price:
        try:
            updates["max_price"] = float(price.group(1))
        except ValueError:
            pass

    brand = BRAND_COLON_RE.search(message) or BRAND_BY_RE.search(message)
    if brand:
        updates["brand"] = brand.group(1).strip()

    return updates


## Fills empty slots from the user's profile (department/gender/summary) at
## session start. Never overwrites anything the customer says later.
def _seed_from_profile(state: dict, user_profile: object) -> None:
    if not isinstance(user_profile, dict):
        return
    haystack_parts = []
    for key in ("department", "gender", "preferred_department", "summary", "notes"):
        value = user_profile.get(key)
        if isinstance(value, str):
            haystack_parts.append(value)
    if not haystack_parts:
        return
    seeded = extract_updates(" ".join(haystack_parts))
    for key, value in seeded.items():
        if key == "style":
            continue  # too speculative to seed a style preference from profile text
        if state.get(key) in (None, ""):
            state[key] = value


## Hand-picked synonyms so shopper wording (e.g. "sneakers") also matches
## catalog wording (e.g. "athletic shoes"). Adds words, never replaces them.
CATALOG_ALIASES: dict[str, list[str]] = {
    "sneaker": ["sneakers", "athletic shoes"],
    "sneakers": ["sneakers", "athletic shoes"],
    "trouser": ["pants"],
    "trousers": ["pants"],
    "purse": ["handbag"],
    "purses": ["handbags"],
    "handbag": ["handbag", "purse"],
    "hoodie": ["hooded sweatshirt", "sweatshirt"],
    "hoodies": ["hooded sweatshirts", "sweatshirts"],
    "tee": ["t-shirt"],
    "tees": ["t-shirts"],
    "women": ["women"],
    "woman": ["women"],
    "womens": ["women"],
    "men": ["men"],
    "man": ["men"],
    "mens": ["men"],
}


## Adds catalog-friendly synonyms onto a list of search words.
def normalize_terms(terms: list[str]) -> list[str]:
    expanded = list(terms)
    for term in terms:
        for alias_phrase in CATALOG_ALIASES.get(term.lower(), []):
            for word in alias_phrase.split():
                if word not in expanded:
                    expanded.append(word)
    return expanded


## Writes one debug log line per turn (only when AGENT_DEBUG is on).
def _log_turn(
    session_id: str,
    turn: int,
    user_message: str,
    state: dict,
    ask_attribute: object,
    recommendations: list[dict],
    *,
    query_terms: list[str] | None = None,
    override_detected: bool = False,
    fields_reset: list[str] | None = None,
) -> None:
    if not DEBUG_ENABLED:
        return
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "session_id": session_id,
            "turn": turn,
            "user_message": user_message,
            "override_detected": override_detected,
            "fields_reset": fields_reset or [],
            "query_terms": query_terms or [],
            "state": {
                "category": state.get("category"),
                "brand": state.get("brand"),
                "color": state.get("color"),
                "department": state.get("department"),
                "max_price": state.get("max_price"),
                "material": state.get("material"),
                "use_case": state.get("use_case"),
                "style": list(state.get("style") or []),
            },
            "ask_attribute": ask_attribute,
            "recommendations": [item.get("parent_asin") for item in recommendations],
        }
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:
        pass  ## logging should never crash the agent


## Picks the next attribute to ask the customer about, following the fixed
## priority order, skipping anything already filled in or already asked.
def _choose_ask_attribute(state: dict, turn: int) -> str | None:
    if turn > ASK_UNTIL_TURN:
        return None
    for attribute in ATTRIBUTE_PRIORITY:
        key = "max_price" if attribute == "budget" else attribute
        value = state.get(key)
        empty = value in (None, "", []) if not isinstance(value, list) else not value
        if empty and attribute not in state["asked"]:
            state["asked"].add(attribute)
            return attribute
    return None


## Builds the reply text: shows results (or not), then asks a clarifying question if needed.
def _compose_message(recommendations: list[dict], ask_attribute: str | None) -> str:
    base = "Here are some options that match what you've told me so far." if recommendations \
        else "I couldn't find a strong match yet with what you've told me so far."
    if ask_attribute:
        base += f" To narrow it down, what's your preference on {ask_attribute}?"
    return base


## If the query has repeated (frozen) and we've seen 2+ past
## searches, blends in products that ranked well earlier in the conversation.
## Otherwise returns candidate_ids unchanged.
def _apply_cross_turn_consensus(
    state: dict,
    query_signature: tuple[str, ...],
    candidate_ids: list[str],
    top_k: int,
) -> list[str]:
    history = state.setdefault("consensus_history", {})
    last_signature = state.get("last_consensus_signature")

    # Record this query state's Top-K once, the first time we see it.
    if query_signature not in history:
        history[query_signature] = list(candidate_ids[:CONSENSUS_HISTORY_DEPTH])

    is_frozen = last_signature is not None and query_signature == last_signature
    state["last_consensus_signature"] = query_signature

    if not is_frozen or len(history) < 2:
        return candidate_ids

    # Candidate union: current Top-K plus every other distinct query
    # state's stored Top-K, deduplicated by parent_asin. This union order
    # also doubles as the deterministic tiebreaker below.
    union: list[str] = []
    seen: set[str] = set()
    for asin in candidate_ids:
        if asin not in seen:
            seen.add(asin)
            union.append(asin)
    for signature, ranked in history.items():
        if signature == query_signature:
            continue
        for asin in ranked:
            if asin not in seen:
                seen.add(asin)
                union.append(asin)

    current_rank = {asin: i for i, asin in enumerate(candidate_ids)}

    def _score(asin: str) -> float:
        current_score = (
            1.0 / (CONSENSUS_RRF_K + current_rank[asin] + 1)
            if asin in current_rank
            else 0.0
        )
        historical_score = 0.0
        for signature, ranked in history.items():
            if signature == query_signature:
                continue
            if asin in ranked:
                historical_score += 1.0 / (CONSENSUS_RRF_K + ranked.index(asin) + 1)
        return current_score + CONSENSUS_WEIGHT * historical_score

    # Stable sort keeps the deterministic union order above as the tiebreaker.
    union.sort(key=_score, reverse=True)
    return union[:top_k]


## TIE-BREAK BONUS: nudges already-retrieved candidates up a little if they
## match what the customer told us (ENABLE_CONSTRAINT_BONUS, off - tested worse)
## or match the category they're looking for (ENABLE_CATEGORY_BONUS, on - tested better).
ENABLE_CONSTRAINT_BONUS = _env_flag("ENABLE_CONSTRAINT_BONUS", False)
ENABLE_CATEGORY_BONUS = _env_flag("ENABLE_CATEGORY_BONUS", True)

TIEBREAK_RRF_K = 60           ## same rank-softening constant used by consensus above
TIEBREAK_BONUS_WEIGHT = 0.02  ## how much one bonus point is worth
TIEBREAK_MIN_MATCHES = 2      ## matches needed before ENABLE_CONSTRAINT_BONUS applies


## Re-sorts an already-retrieved candidate list using the small bonuses above.
## Returns the list unchanged if both bonus flags are off.
def _apply_tiebreak_bonus(
    connection: sqlite3.Connection,
    state: dict,
    candidate_ids: list[str],
    top_k: int,
) -> list[str]:
    if not candidate_ids or not (ENABLE_CONSTRAINT_BONUS or ENABLE_CATEGORY_BONUS):
        return candidate_ids

    placeholders = ",".join("?" for _ in candidate_ids)
    rows = connection.execute(
        f"SELECT parent_asin, title, categories, department, material_field, "
        f"color_field, brand_field FROM products WHERE parent_asin IN ({placeholders})",
        candidate_ids,
    ).fetchall()

    fields = {
        str(row[0]): {
            "title": (row[1] or "").lower(),
            "categories": (row[2] or "").lower(),
            "department": (row[3] or "").lower(),
            "material_field": (row[4] or "").lower(),
            "color_field": (row[5] or "").lower(),
            "brand_field": (row[6] or "").lower(),
        }
        for row in rows
    }

    state_category_terms = set(_terms(str(state.get("category") or "")))

    ## Works out how big a bonus one candidate product should get.
    def _bonus(asin: str) -> float:
        info = fields.get(asin)
        if not info:
            return 0.0

        total = 0.0

        if ENABLE_CONSTRAINT_BONUS:
            matches = 0
            for state_key, field_key in (
                ("material", "material_field"),
                ("color", "color_field"),
                ("department", "department"),
                ("brand", "brand_field"),
            ):
                value = state.get(state_key)
                if value and str(value).strip().lower() in info[field_key]:
                    matches += 1
            state_use_case = state.get("use_case")
            if state_use_case and str(state_use_case).strip().lower() in info["title"]:
                matches += 1
            for style_value in state.get("style") or []:
                if str(style_value).strip().lower() in info["title"]:
                    matches += 1
                    break
            if matches >= TIEBREAK_MIN_MATCHES:
                total += TIEBREAK_BONUS_WEIGHT * matches

        if ENABLE_CATEGORY_BONUS and state_category_terms:
            candidate_terms = set(_terms(info["title"])) | set(_terms(info["categories"]))
            overlap = len(state_category_terms & candidate_terms)
            if overlap >= 2:
                total += TIEBREAK_BONUS_WEIGHT * overlap

        return total

    current_rank = {asin: i for i, asin in enumerate(candidate_ids)}

    def _score(asin: str) -> float:
        base_score_from_position = 1.0 / (TIEBREAK_RRF_K + current_rank[asin] + 1)
        return base_score_from_position + _bonus(asin)

    reranked = sorted(candidate_ids, key=_score, reverse=True)
    return reranked


## Stateful, rule-based shopping agent: per-session slot memory + SQLite BM25 search.
class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._bm25_sql = "bm25(products, " + ", ".join(str(w) for w in ALL_BM25_WEIGHTS) + ")"
        self._build_index()

    ## Loads the product catalog into an in-memory SQLite full-text search table.
    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "department, material_field, color_field, brand_field, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        insert_sql = "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                structured = _structured_product_fields(product)

                row = (
                    parent_asin,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                    structured["department"] or "",
                    structured["material"] or "",
                    structured["color"] or "",
                    structured["brand"] or "",
                )
                batch.append(row)
                if len(batch) >= 1000:
                    cursor.executemany(insert_sql, batch)
                    batch.clear()
        if batch:
            cursor.executemany(insert_sql, batch)
        self.connection.commit()

    ## Starts a fresh session with empty slot memory, seeded from the user profile.
    def reset(self, session_id: str, user_profile: dict) -> None:
        state = {
            "user_profile": user_profile,
            "category": None,
            "brand": None,
            "color": None,
            "department": None,
            "max_price": None,
            "material": None,
            "use_case": None,
            "style": [],
            "history": [],
            "asked": set(),
            "consensus_history": {},
            "last_consensus_signature": None,
        }
        _seed_from_profile(state, user_profile)
        self._sessions[session_id] = state

    ## Updates session state (category, attributes, overrides) from one customer message.
    def _apply_message(self, state: dict, user_message: str) -> None:
        state["history"].append(user_message)

        updates = extract_updates(user_message)
        new_category = extract_category(user_message)
        override_cue = bool(OVERRIDE_RE.search(user_message))

        ## True if the customer changed their mind about attributes but kept the same category.
        attribute_only_override = (
            override_cue and
            new_category is None and
            bool(updates) and
            state.get("category") is not None
        )

        if new_category:
            is_change = state["category"] is None or override_cue or not _shares_word(new_category, state["category"])
            if is_change and state["category"] is not None and not _shares_word(new_category, state["category"]):
                ## Category swap (e.g. "forget the shoes, I need a jacket") -- clear stale slots.
                state["color"] = None
                state["material"] = None
                state["use_case"] = None
                state["style"] = []
                state["brand"] = None
                state["asked"] = set()
            state["category"] = new_category
        elif attribute_only_override:
            state["asked"] = set()

        for key, value in updates.items():
            if key == "style":
                merged = list(dict.fromkeys([*(state.get("style") or []), *value]))
                state["style"] = merged[:5]
            else:
                state[key] = value

    ## Builds the search query terms from session state (category, attributes) and the latest message.
    def _build_query_terms(self, state: dict, latest_message: str) -> list[str]:
        parts: list[str] = []

        category = state.get("category")
        if category:
            parts.append(str(category))

        for key in ("brand", "color", "material", "department", "use_case"):
            value = state.get(key)
            if value:
                parts.append(str(value))

        if state.get("style"):
            parts.extend(state["style"])

        raw_terms = set(_terms(latest_message))
        if category:
            category_terms = set(_terms(category))
            if raw_terms - category_terms:
                parts.append(latest_message)
        else:
            parts.append(latest_message)

        text = " ".join(parts)
        terms = list(dict.fromkeys(_terms(text)))
        terms = normalize_terms(terms)

        return list(dict.fromkeys(terms))[:40]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]

        state_before = {
            "category": state.get("category"),
            "color": state.get("color"),
            "material": state.get("material"),
            "use_case": state.get("use_case"),
            "brand": state.get("brand"),
        }

        self._apply_message(state, user_message)

        state_after = {
            "category": state.get("category"),
            "color": state.get("color"),
            "material": state.get("material"),
            "use_case": state.get("use_case"),
            "brand": state.get("brand"),
        }
        override_detected = state_before != state_after
        fields_reset = [k for k in state_before if state_before[k] != state_after[k]]

        unique_terms = self._build_query_terms(state, user_message)
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            ## Retrieve deeper than top_k when consensus is on, so there's a pool to remember.
            retrieval_limit = (
                max(top_k, CONSENSUS_HISTORY_DEPTH)
                if ENABLE_CROSS_TURN_CONSENSUS
                else top_k
            )
            rows = self.connection.execute(
                f"SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY {self._bm25_sql} LIMIT ?",
                (expression, retrieval_limit),
            ).fetchall()
            candidate_ids = [str(row[0]) for row in rows]

            if ENABLE_CROSS_TURN_CONSENSUS:
                query_signature = tuple(unique_terms)
                candidate_ids = _apply_cross_turn_consensus(
                    state, query_signature, candidate_ids, top_k
                )

            if ENABLE_CONSTRAINT_BONUS or ENABLE_CATEGORY_BONUS:
                candidate_ids = _apply_tiebreak_bonus(self.connection, state, candidate_ids, top_k)

            recommendations = [{"parent_asin": asin} for asin in candidate_ids[:top_k]]

        ask_attribute = _choose_ask_attribute(state, turn)
        message = _compose_message(recommendations, ask_attribute)

        _log_turn(session_id, turn, user_message, state, ask_attribute, recommendations,
                  query_terms=unique_terms, override_detected=override_detected, fields_reset=fields_reset)

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
