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


def _text(value: object) -> str:
    # Turn whatever shape a catalog field is (string / list / dict / None)
    # into one plain string we can feed into search or regexes.
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    # Lower-case word tokens, dropping stopwords and single letters.
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# Attribute extraction vocabulary
#
# These patterns are intentionally simple / rule-based (no ML, no embeddings).
# They are used to pull structured slot values out of whatever the simulated
# customer says on a given turn, so the agent can remember them across turns.
# ---------------------------------------------------------------------------

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

# Cues that signal the customer is changing/replacing something they said
# earlier, rather than simply adding a new fact.
OVERRIDE_RE = re.compile(
    r"\b(forget|actually|instead|ignore my earlier preference|no longer|"
    r"never mind|change (?:that|it) to)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Structured product metadata (department / material / color / brand).
#
# A value is only ever pulled from an explicit structured `details` key (or
# the `store` field for brand). We deliberately do NOT fall back to
# regex-scanning marketing text for these -- that would reintroduce exactly
# the "noisy incidental mention" problem this structured extraction exists
# to avoid (e.g. a shirt whose *description* happens to mention "leather
# boots" shouldn't count as a leather product).
#
# These structured fields are indexed into their own dedicated search
# columns (see Agent._build_index) with higher weight than the free-text
# columns, so a trustworthy structured hit beats the same word turning up
# incidentally in marketing copy. This is now a permanent part of the
# agent -- there used to be a flag for it, but it consistently helped, so
# it's no longer optional.
# ---------------------------------------------------------------------------
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


def _structured_product_fields(product: dict) -> dict[str, str | None]:
    """Pull the handful of explicit, structured attribute keys we trust.

    Returns a dict with department/material/color/brand, each either a
    clean short string or None if the product doesn't carry that key.
    """
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


# Phrase used to pull out a free-text "category" description, e.g.
# "I'm looking for Women's Shoes." / "I need a jacket" / "forget the shoes, I need a jacket"
CATEGORY_PHRASE_RE = re.compile(
    r"(?:looking for|need|want|shopping for)\s+(.+?)(?:[.,;]|$|\bbut\b|\bwith\b)",
    re.IGNORECASE,
)

# Order in which we probe for missing slots. Names match the evaluator's
# ALLOWED_ATTRIBUTES vocabulary so ask_attribute values are meaningful to the
# simulated customer (see local_evaluator.customer_reply / classify_constraint).
ATTRIBUTE_PRIORITY = ["use_case", "material", "color", "budget", "style", "brand"]

# How many turns we're willing to spend asking clarifying questions before we
# stop probing and just keep searching with whatever we've accumulated. This
# leaves the remaining turns free for the agent to "land" a hit.
ASK_UNTIL_TURN = 5

DEBUG_ENABLED = os.environ.get("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_LOG_PATH = Path(os.environ.get("AGENT_DEBUG_LOG_PATH", "debug_logs/conversations.jsonl"))


# ---------------------------------------------------------------------------
# Frozen configuration.
#
# Earlier experiments tried two extra features behind flags:
#   - an "intent router" that treated buying vs. browsing turns differently
#     (regressed Hit@10 badly in ablation testing -- removed completely)
#   - "profile assist" query injection that added shopper-profile words to
#     the search query (measured zero benefit -- removed completely)
# Both are gone now, along with their flags, so the agent can't accidentally
# turn them back on. What remains permanent from that round of testing is:
#   - structured BM25 fields (see ALL_BM25_WEIGHTS below)
#   - conservative catalog vocabulary normalization (see normalize_terms)
# There are no more agent behavior flags to set.
# ---------------------------------------------------------------------------

# Extra dedicated FTS columns added on top of the original Phase 2 columns
# (title, categories, features, details, store, description), in this order.
STRUCTURED_FIELD_NAMES = ("department", "material_field", "color_field", "brand_field")
# Weights for the new columns. Kept meaningfully above the raw text fields
# they're pulled out of (features=2.5, description=1.0, store=1.5) so a
# trustworthy structured hit outranks the same word appearing incidentally
# in marketing copy, without approaching title(6.0)/categories(4.0).
STRUCTURED_FIELD_WEIGHTS = (2.5, 3.5, 3.5, 3.0)
# Phase 2's original 7-column weighting -- left completely unchanged.
BASE_BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
# Combined weight list actually used to rank every search.
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


def extract_category(message: str) -> str | None:
    match = CATEGORY_PHRASE_RE.search(message)
    if not match:
        return None
    phrase = _clean_phrase(match.group(1))
    if not phrase:
        return None

    # Phase 2: Reject categories that look like attributes, not product types.
    # Patterns like "is: leather", "is: waterproof", etc. are garbage extractions.
    # Real categories are product types: "shoes", "jacket", "belt", "watch".

    lowered = phrase.lower()

    # If the phrase starts with "is:", "are:", "colon", or looks purely like an attribute,
    # it's probably misparsed and should be rejected.
    if lowered.startswith(("is:", "is ", "are:", "are ")):
        return None

    # Reject pure material, color, style, or use-case matches.
    # These are attributes, not categories.
    pure_attribute_matches = (
        MATERIAL_RE.search(phrase) or
        COLOR_RE.search(phrase) or
        STYLE_RE.search(phrase) or
        USE_CASE_RE.search(phrase)
    )

    # If the entire phrase is a single color/material/style word (no other words),
    # reject it as a garbage category.
    words = [w for w in _terms(phrase) if w]
    if len(words) == 1 and pure_attribute_matches:
        return None

    return phrase


def extract_updates(message: str) -> dict:
    """Pull whatever structured slot values we can find in a single message.

    Every slot here is single-valued and simply overwritten when a new value
    is found. That one rule gives us both "accumulation" (a slot that was
    empty gets filled) and "override" (a slot that already had a value gets
    replaced by the newer one) for free, without needing separate logic paths.
    """
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


def _seed_from_profile(state: dict, user_profile: object) -> None:
    """Best-effort, low-risk use of the (anonymized/aggregate) user_profile.

    We don't know the exact schema participants will receive, so instead of
    hardcoding field names we look for a couple of common, low-risk signals
    (department/gender, and any free-text summary) and run them through the
    same extractor used for chat turns. This never overwrites information the
    customer states explicitly later -- extract_updates() on real turns will
    take priority because it's applied after this seeding step.

    This is plain Phase 2 behavior (fills empty slots only, at session start)
    -- not the separate "profile assist" query-injection experiment that was
    tried and removed for showing no benefit.
    """
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


# ---------------------------------------------------------------------------
# Catalog vocabulary normalization (permanent).
#
# Shoppers and Amazon listings don't always use the same word for the same
# thing ("sneakers" vs. "athletic shoes"). This is a small, hand-picked
# lookup table -- NOT semantic search or ML -- that adds the catalog's usual
# wording next to (never instead of) the shopper's own wording, so the
# search has a better chance of matching both sides. Kept short on purpose:
# extra noisy search words can hurt ranking more than they help.
# ---------------------------------------------------------------------------
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


def normalize_terms(terms: list[str]) -> list[str]:
    """Add a small number of catalog-friendly synonyms to a list of words.

    For every word we recognize, we look up its known catalog equivalent(s)
    and add any new words from that phrase onto the end of the list. The
    shopper's original word is always kept too -- this only adds, never
    replaces.
    """
    expanded = list(terms)
    for term in terms:
        for alias_phrase in CATALOG_ALIASES.get(term.lower(), []):
            for word in alias_phrase.split():
                if word not in expanded:
                    expanded.append(word)
    return expanded


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
        # Debug logging must never affect scoring or crash the agent.
        pass


def _choose_ask_attribute(state: dict, turn: int) -> str | None:
    # The one and only clarification policy: Phase 2's fixed priority
    # order. An adaptive, candidate-aware version of this was tried and
    # removed -- this fixed order is what produced the best verified score.
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


def _compose_message(recommendations: list[dict], ask_attribute: str | None) -> str:
    base = "Here are some options that match what you've told me so far." if recommendations \
        else "I couldn't find a strong match yet with what you've told me so far."
    if ask_attribute:
        base += f" To narrow it down, what's your preference on {ask_attribute}?"
    return base


class Agent:
    """Stateful, rule-based baseline: per-session slot memory + SQLite BM25 retrieval.

    This is the frozen, cleaned-up version of the agent. It keeps:
      - Phase 2 conversational memory/state, attribute extraction, and
        category/attribute override handling (including stale-state cleanup)
      - Phase 2's fixed clarification order (the only clarification policy)
      - structured BM25 fields (department/material/color/brand get their
        own higher-weight search columns)
      - conservative catalog vocabulary normalization

    Two experimental features (a buying/browsing intent router, and a
    "profile assist" query-injection step) were tried, measured, and found
    to either regress the score badly or add no benefit. Both have been
    removed completely rather than just defaulted off, so there is nothing
    left in this file that could accidentally re-enable them.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._bm25_sql = "bm25(products, " + ", ".join(str(w) for w in ALL_BM25_WEIGHTS) + ")"
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        # title/categories/features/details/store/description are the
        # original Phase 2 free-text columns; the last four columns are the
        # trusted structured attributes (permanent, see module docstring above).
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

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
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
        }
        _seed_from_profile(state, user_profile)
        self._sessions[session_id] = state

    def _apply_message(self, state: dict, user_message: str) -> None:
        state["history"].append(user_message)

        # Extract structured updates first, before category extraction.
        updates = extract_updates(user_message)

        new_category = extract_category(user_message)
        override_cue = bool(OVERRIDE_RE.search(user_message))

        # Phase 2: Detect attribute-only overrides.
        # If we have an override cue but only extracted attributes (not a new category),
        # we should reset category-specific slots and clarification state, but keep
        # the existing category focus.
        attribute_only_override = (
            override_cue and
            new_category is None and
            bool(updates) and
            state.get("category") is not None
        )

        if new_category:
            is_change = state["category"] is None or override_cue or not _shares_word(new_category, state["category"])
            if is_change and state["category"] is not None and not _shares_word(new_category, state["category"]):
                # A genuine category swap (e.g. "forget the shoes, I need a jacket").
                # Category-specific slots are stale and should not carry over;
                # department/budget are treated as durable shopper-level facts.
                state["color"] = None
                state["material"] = None
                state["use_case"] = None
                state["style"] = []
                state["brand"] = None
                state["asked"] = set()
            state["category"] = new_category
        elif attribute_only_override:
            # Phase 2: Customer changed their mind about attributes while keeping
            # the same category (e.g., "Actually, I prefer black instead").
            # Reset clarification state so we can ask about the refined goal.
            state["asked"] = set()

        # Apply extracted updates.
        for key, value in updates.items():
            if key == "style":
                merged = list(dict.fromkeys([*(state.get("style") or []), *value]))
                state["style"] = merged[:5]
            else:
                state[key] = value

    def _build_query_terms(self, state: dict, latest_message: str) -> list[str]:
        """Build BM25 query terms from structured state and raw message.

        Phase 2: Separate category/product-type terms from attribute terms
        to prevent stale attributes from dominating retrieval. Finishes by
        running the (permanent) conservative catalog normalization step.
        """
        parts: list[str] = []

        # Category/product type gets priority weight through placement.
        category = state.get("category")
        if category:
            parts.append(str(category))

        # Core attributes: brand, color, material, department, use_case.
        # These are significant but secondary to category.
        for key in ("brand", "color", "material", "department", "use_case"):
            value = state.get(key)
            if value:
                parts.append(str(value))

        # Style preferences are lower-weight additions.
        if state.get("style"):
            parts.extend(state["style"])

        # Raw message adds recall for unmmodeled attributes and specific nouns.
        # But don't include it if it would just duplicate the category
        # (e.g., latest message is a customer reply with no new info).
        raw_terms = set(_terms(latest_message))
        if category:
            category_terms = set(_terms(category))
            # Only include raw message if it has content beyond the category.
            if raw_terms - category_terms:
                parts.append(latest_message)
        else:
            parts.append(latest_message)

        text = " ".join(parts)
        terms = list(dict.fromkeys(_terms(text)))

        # Expand a few words to their catalog spelling (e.g. "sneaker" also
        # searches "sneakers"/"athletic shoes"). Always on -- see
        # CATALOG_ALIASES / normalize_terms above.
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

        # Store state before applying message for override detection logging.
        state_before = {
            "category": state.get("category"),
            "color": state.get("color"),
            "material": state.get("material"),
            "use_case": state.get("use_case"),
            "brand": state.get("brand"),
        }

        self._apply_message(state, user_message)

        # Detect which fields were reset/changed.
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
            rows = self.connection.execute(
                f"SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY {self._bm25_sql} LIMIT ?",
                (expression, top_k),
            ).fetchall()
            candidate_ids = [str(row[0]) for row in rows]
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
