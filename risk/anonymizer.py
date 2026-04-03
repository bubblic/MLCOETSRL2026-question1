"""Entity anonymization and de-anonymization for the risk pipeline.

Replaces identifying entities (organizations, people, locations) and
absolute years with typed placeholders before LLM processing, and
restores original values in the final output.

Supports user-provided known entities for guaranteed detection and
optional spacy NER for broader automatic detection.  Falls back
gracefully to regex-only mode if spacy is not installed.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# spacy entity labels that map to our placeholder types.
_SPACY_LABEL_TO_TYPE: Dict[str, str] = {
    "ORG": "ORG",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
}

_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def _load_spacy_nlp() -> Any:
    """Load spacy English NER model, or return ``None`` if unavailable."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        logger.warning(
            "spacy or en_core_web_sm not available; "
            "falling back to regex-only anonymization"
        )
        return None


class EntityAnonymizer:
    """Anonymize and de-anonymize entity names and years in text.

    Args:
        known_entities: Optional user-provided entities for guaranteed
            detection. Each entry is a dict with ``"type"`` (ORG,
            PERSON, LOCATION) and ``"names"`` (list of surface forms,
            longest first).  If omitted, relies entirely on NER.
        use_ner: Whether to use spacy NER for additional detection.
            Falls back to ``False`` if spacy is not installed.
        ner_entity_types: spacy entity labels to anonymize.
            Defaults to ``{"ORG", "PERSON", "GPE", "LOC"}``.
        anonymize_years: Whether to replace absolute years (2000–2099)
            with relative fiscal-year markers (``FY_T``, ``FY_T-1``, …).
    """

    def __init__(
        self,
        known_entities: Optional[List[Dict[str, Any]]] = None,
        use_ner: bool = True,
        ner_entity_types: Optional[Set[str]] = None,
        anonymize_years: bool = True,
    ) -> None:
        self._known_entities = known_entities or []
        self._use_ner = use_ner
        self._ner_entity_types = ner_entity_types or {"ORG", "PERSON", "GPE", "LOC"}
        self._anonymize_years = anonymize_years

        # Built during anonymize_pages().
        # placeholder -> canonical original text
        self._entity_map: Dict[str, str] = {}
        # original text (lowered) -> placeholder
        self._reverse_lookup: Dict[str, str] = {}
        # Ordered (pattern, placeholder) pairs for text replacement.
        self._replacement_pairs: List[Tuple[re.Pattern, str]] = []
        # Per-type counters for placeholder numbering.
        self._type_counters: Dict[str, int] = {}
        # Year mapping: absolute year string -> relative marker.
        self._year_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize_pages(
        self,
        pages: Dict[int, Optional[str]],
    ) -> Dict[int, Optional[str]]:
        """Replace entities and years in all pages.

        Builds the internal entity and year maps, then applies
        replacements.  Must be called exactly once per instance.

        Returns:
            New pages dict with anonymized text.
        """
        # Collect all non-None page texts.
        all_text = "\n\n".join(
            text for text in pages.values() if text is not None
        )
        if not all_text.strip():
            return dict(pages)

        # Step 1: Register known entities.
        self._register_known_entities()

        # Step 2: NER detection (on original text).
        if self._use_ner:
            self._detect_and_register_ner_entities(all_text)

        # Step 3: Build ordered replacement list (longest first).
        self._build_replacement_pairs()

        # Step 4: Year map.
        if self._anonymize_years:
            self._build_year_map(all_text)

        # Step 5: Apply replacements to each page.
        result: Dict[int, Optional[str]] = {}
        for page_num, text in pages.items():
            if text is None:
                result[page_num] = None
            else:
                result[page_num] = self._apply_replacements(text)
        return result

    def deanonymize(self, data: Any) -> Any:
        """Recursively restore placeholders in an arbitrary JSON structure.

        Handles dicts, lists, and strings.  Other types pass through
        unchanged.
        """
        if isinstance(data, str):
            return self._deanonymize_string(data)
        if isinstance(data, dict):
            return {k: self.deanonymize(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.deanonymize(item) for item in data]
        return data

    def entity_map(self) -> Dict[str, str]:
        """Return the placeholder-to-original mapping.

        Includes both entity placeholders (``[ORG_1]``, …) and year
        placeholders (``FY_T``, ``FY_T-1``, …).
        """
        combined = dict(self._entity_map)
        for year_str, marker in self._year_map.items():
            combined[marker] = year_str
        return combined

    # ------------------------------------------------------------------
    # Known-entity registration
    # ------------------------------------------------------------------

    def _register_known_entities(self) -> None:
        """Register user-provided entities into the lookup tables."""
        for entry in self._known_entities:
            entity_type = entry["type"].upper()
            names: List[str] = entry["names"]
            if not names:
                continue
            # First name in the list is treated as canonical.
            canonical = names[0]
            placeholder = self._next_placeholder(entity_type)
            self._entity_map[placeholder] = canonical
            for name in names:
                self._reverse_lookup[name.lower()] = placeholder

    # ------------------------------------------------------------------
    # NER detection
    # ------------------------------------------------------------------

    def _detect_and_register_ner_entities(self, text: str) -> None:
        """Run spacy NER and register new entities not already known."""
        nlp = _load_spacy_nlp()
        if nlp is None:
            self._use_ner = False
            return

        # spacy has a max text length; process in chunks if needed.
        max_len = nlp.max_length
        chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

        seen: Dict[str, str] = {}  # lowered surface form -> entity_type
        for chunk in chunks:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ not in self._ner_entity_types:
                    continue
                surface = ent.text.strip()
                if len(surface) < 2:
                    continue
                low = surface.lower()
                # Skip if already covered by a known entity.
                if self._is_already_known(low):
                    continue
                entity_type = _SPACY_LABEL_TO_TYPE.get(
                    ent.label_, ent.label_
                )
                # Keep the longest surface form per lowered key.
                if low not in seen or len(surface) > len(seen[low]):
                    seen[low] = entity_type

        # Register NER entities (deterministic order).
        for low in sorted(seen, key=lambda k: (-len(k), k)):
            if self._is_already_known(low):
                continue
            entity_type = seen[low]
            placeholder = self._next_placeholder(entity_type)
            self._entity_map[placeholder] = low  # preserve casing later
            self._reverse_lookup[low] = placeholder

    def _is_already_known(self, lowered: str) -> bool:
        """Check if a surface form is already covered by an existing entity."""
        if lowered in self._reverse_lookup:
            return True
        # Also check if it is a substring of an already-known form.
        for known_low in self._reverse_lookup:
            if lowered in known_low or known_low in lowered:
                return True
        return False

    # ------------------------------------------------------------------
    # Replacement machinery
    # ------------------------------------------------------------------

    def _build_replacement_pairs(self) -> None:
        """Compile ordered (regex, placeholder) pairs, longest first."""
        # Group by placeholder so multiple surface forms share one entry.
        placeholder_to_forms: Dict[str, List[str]] = {}
        for low, placeholder in self._reverse_lookup.items():
            placeholder_to_forms.setdefault(placeholder, []).append(low)

        pairs: List[Tuple[int, re.Pattern, str]] = []
        for placeholder, forms in placeholder_to_forms.items():
            for form in forms:
                escaped = re.escape(form)
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\w)",
                    re.IGNORECASE,
                )
                pairs.append((len(form), pattern, placeholder))

        # Sort by form length descending for longest-match-first.
        pairs.sort(key=lambda t: -t[0])
        self._replacement_pairs = [(p, ph) for _, p, ph in pairs]

    def _build_year_map(self, text: str) -> None:
        """Detect years in text and build absolute-to-relative mapping."""
        years = [int(m) for m in _YEAR_PATTERN.findall(text)]
        if not years:
            return
        # Fiscal year = most frequently occurring year.
        counter = Counter(years)
        fiscal_year = counter.most_common(1)[0][0]

        unique_years = sorted(set(years), reverse=True)
        for year in unique_years:
            offset = year - fiscal_year
            if offset == 0:
                marker = "FY_T"
            elif offset > 0:
                marker = f"FY_T+{offset}"
            else:
                marker = f"FY_T{offset}"  # e.g. FY_T-1
            self._year_map[str(year)] = marker

    def _apply_replacements(self, text: str) -> str:
        """Apply all entity and year replacements to a text string."""
        # Entity replacements (longest first).
        for pattern, placeholder in self._replacement_pairs:
            text = pattern.sub(placeholder, text)
        # Year replacements.
        if self._year_map:
            # Replace longest year strings first (all are 4 digits, so
            # order by year value descending to avoid partial overlap).
            for year_str in sorted(self._year_map, key=int, reverse=True):
                marker = self._year_map[year_str]
                text = re.sub(
                    r"\b" + re.escape(year_str) + r"\b",
                    marker,
                    text,
                )
        return text

    def _deanonymize_string(self, text: str) -> str:
        """Restore all placeholders in a single string."""
        # Entity placeholders.
        for placeholder, original in self._entity_map.items():
            text = text.replace(placeholder, original)
        # Year placeholders.
        for year_str, marker in self._year_map.items():
            text = text.replace(marker, year_str)
        return text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_placeholder(self, entity_type: str) -> str:
        """Return the next placeholder for a given entity type."""
        count = self._type_counters.get(entity_type, 0) + 1
        self._type_counters[entity_type] = count
        return f"[{entity_type}_{count}]"
