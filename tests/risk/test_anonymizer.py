"""Tests for entity anonymization module."""

from unittest.mock import patch, MagicMock

import pytest

from risk.anonymizer import EntityAnonymizer


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

KNOWN_ENTITIES_EVERGRANDE = [
    {
        "type": "ORG",
        "names": [
            "China Evergrande Group",
            "Evergrande Group",
            "Evergrande",
            "Hengda Real Estate Group",
        ],
    },
]

SAMPLE_PAGES = {
    1: (
        "China Evergrande Group Annual Report 2021\n"
        "Registered in Shenzhen, People's Republic of China"
    ),
    2: (
        "The Board of Directors of Evergrande Group confirms that "
        "Mr. Hui Ka Yan resigned as Chairman in 2022. "
        "Hengda Real Estate Group is a subsidiary."
    ),
    3: None,
    4: (
        "In 2020, EVERGRANDE reported revenue of RMB 507.2 billion. "
        "By 2021, revenue had declined significantly."
    ),
}


# -----------------------------------------------------------------------
# Regex-only anonymization (no NER)
# -----------------------------------------------------------------------

class TestRegexAnonymization:
    def test_known_entity_replaced(self):
        anon = EntityAnonymizer(
            known_entities=KNOWN_ENTITIES_EVERGRANDE,
            use_ner=False,
            anonymize_years=False,
        )
        result = anon.anonymize_pages(SAMPLE_PAGES)
        assert "Evergrande" not in result[1]
        assert "[ORG_1]" in result[1]

    def test_multiple_surface_forms_same_placeholder(self):
        anon = EntityAnonymizer(
            known_entities=KNOWN_ENTITIES_EVERGRANDE,
            use_ner=False,
            anonymize_years=False,
        )
        result = anon.anonymize_pages(SAMPLE_PAGES)
        # All forms map to the same [ORG_1]
        assert "Hengda Real Estate Group" not in result[2]
        assert "[ORG_1]" in result[2]

    def test_case_insensitive(self):
        anon = EntityAnonymizer(
            known_entities=KNOWN_ENTITIES_EVERGRANDE,
            use_ner=False,
            anonymize_years=False,
        )
        result = anon.anonymize_pages(SAMPLE_PAGES)
        # "EVERGRANDE" in page 4 should also be replaced
        assert "EVERGRANDE" not in result[4]

    def test_longest_match_priority(self):
        """'China Evergrande Group' should be matched before 'Evergrande'."""
        anon = EntityAnonymizer(
            known_entities=KNOWN_ENTITIES_EVERGRANDE,
            use_ner=False,
            anonymize_years=False,
        )
        result = anon.anonymize_pages(SAMPLE_PAGES)
        # Page 1 has "China Evergrande Group" — should be one placeholder,
        # not "China [ORG_1] Group".
        assert "China [ORG_1]" not in result[1]
        assert "[ORG_1] Annual Report" in result[1]

    def test_word_boundary_awareness(self):
        """Should not match inside other words."""
        pages = {1: "The Nevergrandesque tower stands tall."}
        anon = EntityAnonymizer(
            known_entities=[{"type": "ORG", "names": ["Evergrande"]}],
            use_ner=False,
            anonymize_years=False,
        )
        result = anon.anonymize_pages(pages)
        # "Nevergrandesque" should NOT be affected
        assert "Nevergrandesque" in result[1]

    def test_multiple_entity_types_separate_numbering(self):
        anon = EntityAnonymizer(
            known_entities=[
                {"type": "ORG", "names": ["Acme Corp"]},
                {"type": "PERSON", "names": ["John Doe"]},
                {"type": "LOCATION", "names": ["Springfield"]},
            ],
            use_ner=False,
            anonymize_years=False,
        )
        pages = {1: "John Doe works at Acme Corp in Springfield."}
        result = anon.anonymize_pages(pages)
        assert "[PERSON_1]" in result[1]
        assert "[ORG_1]" in result[1]
        assert "[LOCATION_1]" in result[1]
        # Each type has its own counter
        emap = anon.entity_map()
        assert emap["[ORG_1]"] == "Acme Corp"
        assert emap["[PERSON_1]"] == "John Doe"
        assert emap["[LOCATION_1]"] == "Springfield"

    def test_none_pages_preserved(self):
        anon = EntityAnonymizer(
            known_entities=KNOWN_ENTITIES_EVERGRANDE,
            use_ner=False,
            anonymize_years=False,
        )
        result = anon.anonymize_pages(SAMPLE_PAGES)
        assert result[3] is None

    def test_empty_pages(self):
        anon = EntityAnonymizer(use_ner=False, anonymize_years=False)
        result = anon.anonymize_pages({})
        assert result == {}

    def test_no_known_entities_regex_only_is_noop(self):
        """Without known entities and without NER, text is unchanged."""
        anon = EntityAnonymizer(use_ner=False, anonymize_years=False)
        pages = {1: "Some text about a company."}
        result = anon.anonymize_pages(pages)
        assert result[1] == pages[1]


# -----------------------------------------------------------------------
# Year anonymization
# -----------------------------------------------------------------------

class TestYearAnonymization:
    def test_fiscal_year_detected_as_most_frequent(self):
        pages = {
            1: "In 2021, revenue was $100M. The 2021 annual report shows growth.",
            2: "Compared to 2020, the 2019 baseline was lower.",
        }
        anon = EntityAnonymizer(
            use_ner=False,
            anonymize_years=True,
        )
        result = anon.anonymize_pages(pages)
        # 2021 appears twice → fiscal year → FY_T
        assert "FY_T" in result[1]
        assert "2021" not in result[1]
        # 2020 → FY_T-1
        assert "FY_T-1" in result[2]
        assert "2020" not in result[2]
        # 2019 → FY_T-2
        assert "FY_T-2" in result[2]

    def test_year_map_in_entity_map(self):
        pages = {1: "Annual report for 2022 and 2021."}
        anon = EntityAnonymizer(
            use_ner=False,
            anonymize_years=True,
        )
        anon.anonymize_pages(pages)
        emap = anon.entity_map()
        # Both years should appear in the map
        assert any("2022" in v for v in emap.values())
        assert any("2021" in v for v in emap.values())

    def test_non_year_numbers_preserved(self):
        pages = {1: "Revenue was $507.2 billion in 2021. Page 42 of 100."}
        anon = EntityAnonymizer(
            use_ner=False,
            anonymize_years=True,
        )
        result = anon.anonymize_pages(pages)
        assert "507.2" in result[1]
        assert "42" in result[1]
        assert "100" in result[1]

    def test_years_disabled(self):
        pages = {1: "In 2021, the company reported losses."}
        anon = EntityAnonymizer(
            use_ner=False,
            anonymize_years=False,
        )
        result = anon.anonymize_pages(pages)
        assert "2021" in result[1]

    def test_future_years(self):
        pages = {1: "Projections for 2023 and 2024 based on 2022 data."}
        anon = EntityAnonymizer(use_ner=False, anonymize_years=True)
        result = anon.anonymize_pages(pages)
        # 2022 appears once but it's the only baseline; all get mapped
        assert "2022" not in result[1]
        assert "2023" not in result[1]
        assert "2024" not in result[1]


# -----------------------------------------------------------------------
# De-anonymization
# -----------------------------------------------------------------------

class TestDeanonymize:
    def _make_anonymizer(self):
        anon = EntityAnonymizer(
            known_entities=[
                {"type": "ORG", "names": ["Acme Corp"]},
                {"type": "PERSON", "names": ["Jane Doe"]},
            ],
            use_ner=False,
            anonymize_years=True,
        )
        anon.anonymize_pages({
            1: "Jane Doe is CEO of Acme Corp. Report for 2022.",
        })
        return anon

    def test_deanonymize_string(self):
        anon = self._make_anonymizer()
        result = anon.deanonymize("[PERSON_1] is CEO of [ORG_1].")
        assert result == "Jane Doe is CEO of Acme Corp."

    def test_deanonymize_nested_dict(self):
        anon = self._make_anonymizer()
        data = {
            "category": "director_changes",
            "extraction": {"name": "[PERSON_1]", "role": "CEO"},
        }
        result = anon.deanonymize(data)
        assert result["extraction"]["name"] == "Jane Doe"
        assert result["extraction"]["role"] == "CEO"

    def test_deanonymize_nested_list(self):
        anon = self._make_anonymizer()
        data = [{"name": "[PERSON_1]"}, {"org": "[ORG_1]"}]
        result = anon.deanonymize(data)
        assert result[0]["name"] == "Jane Doe"
        assert result[1]["org"] == "Acme Corp"

    def test_deanonymize_year_placeholders(self):
        anon = self._make_anonymizer()
        result = anon.deanonymize("Revenue declined from FY_T-1 to FY_T.")
        # Should contain the actual years
        assert "FY_T" not in result

    def test_deanonymize_non_string_values(self):
        anon = self._make_anonymizer()
        assert anon.deanonymize(42) == 42
        assert anon.deanonymize(None) is None
        assert anon.deanonymize(True) is True

    def test_deanonymize_preserves_non_placeholder_text(self):
        anon = self._make_anonymizer()
        text = "No placeholders here."
        assert anon.deanonymize(text) == text


# -----------------------------------------------------------------------
# NER fallback
# -----------------------------------------------------------------------

class TestNerFallback:
    def test_graceful_fallback_without_spacy(self):
        """When spacy is unavailable, falls back to regex-only."""
        with patch("risk.anonymizer._load_spacy_nlp", return_value=None):
            anon = EntityAnonymizer(
                known_entities=KNOWN_ENTITIES_EVERGRANDE,
                use_ner=True,
                anonymize_years=False,
            )
            result = anon.anonymize_pages(SAMPLE_PAGES)
            # Known entities still get replaced
            assert "Evergrande" not in result[1]
            assert "[ORG_1]" in result[1]

    def test_ner_entities_merged_with_known(self):
        """NER-detected entities that overlap with known ones are skipped."""
        mock_nlp = MagicMock()
        mock_nlp.max_length = 1_000_000

        # Simulate spacy detecting "Evergrande" as ORG
        mock_ent = MagicMock()
        mock_ent.text = "Evergrande"
        mock_ent.label_ = "ORG"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        with patch("risk.anonymizer._load_spacy_nlp", return_value=mock_nlp):
            anon = EntityAnonymizer(
                known_entities=KNOWN_ENTITIES_EVERGRANDE,
                use_ner=True,
                anonymize_years=False,
            )
            anon.anonymize_pages(SAMPLE_PAGES)
            # Should only have [ORG_1] — NER "Evergrande" merged with known
            emap = anon.entity_map()
            org_entries = [k for k in emap if k.startswith("[ORG_")]
            assert len(org_entries) == 1

    def test_ner_detects_new_entities(self):
        """NER-detected entities not in known list get their own placeholder."""
        mock_nlp = MagicMock()
        mock_nlp.max_length = 1_000_000

        mock_ent_person = MagicMock()
        mock_ent_person.text = "Hui Ka Yan"
        mock_ent_person.label_ = "PERSON"
        mock_ent_loc = MagicMock()
        mock_ent_loc.text = "Shenzhen"
        mock_ent_loc.label_ = "GPE"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent_person, mock_ent_loc]
        mock_nlp.return_value = mock_doc

        with patch("risk.anonymizer._load_spacy_nlp", return_value=mock_nlp):
            anon = EntityAnonymizer(
                known_entities=KNOWN_ENTITIES_EVERGRANDE,
                use_ner=True,
                anonymize_years=False,
            )
            result = anon.anonymize_pages(SAMPLE_PAGES)
            emap = anon.entity_map()
            # Should have ORG, PERSON, and LOCATION entries
            assert any(k.startswith("[ORG_") for k in emap)
            assert any(k.startswith("[PERSON_") for k in emap)
            assert any(k.startswith("[LOCATION_") for k in emap)
            # Verify the text was actually anonymized
            assert "Hui Ka Yan" not in result[2]
            assert "Shenzhen" not in result[1]


# -----------------------------------------------------------------------
# Round-trip
# -----------------------------------------------------------------------

class TestRoundTrip:
    def test_anonymize_then_deanonymize_recovers_content(self):
        anon = EntityAnonymizer(
            known_entities=[
                {"type": "ORG", "names": ["Acme Corp"]},
                {"type": "PERSON", "names": ["Alice Smith"]},
            ],
            use_ner=False,
            anonymize_years=True,
        )
        original_pages = {
            1: "Alice Smith joined Acme Corp in 2022.",
            2: "The 2021 report was reviewed by Alice Smith.",
        }
        anonymized = anon.anonymize_pages(original_pages)

        # Verify anonymization happened
        assert "Alice Smith" not in anonymized[1]
        assert "Acme Corp" not in anonymized[1]
        assert "2022" not in anonymized[1]

        # De-anonymize recovers original text
        restored_1 = anon.deanonymize(anonymized[1])
        assert "Alice Smith" in restored_1
        assert "Acme Corp" in restored_1
        assert "2022" in restored_1

    def test_roundtrip_nested_structure(self):
        anon = EntityAnonymizer(
            known_entities=[{"type": "ORG", "names": ["BigBank"]}],
            use_ner=False,
            anonymize_years=False,
        )
        anon.anonymize_pages({1: "BigBank issued a report."})

        nested = {
            "findings": [
                {"text": "[ORG_1] reported losses.", "severity": "high"},
            ],
            "memo": "Risk rating for [ORG_1]: HIGH",
        }
        restored = anon.deanonymize(nested)
        assert restored["findings"][0]["text"] == "BigBank reported losses."
        assert restored["memo"] == "Risk rating for BigBank: HIGH"
        assert restored["findings"][0]["severity"] == "high"


# -----------------------------------------------------------------------
# Entity map
# -----------------------------------------------------------------------

class TestEntityMap:
    def test_entity_map_contains_all_entries(self):
        anon = EntityAnonymizer(
            known_entities=[
                {"type": "ORG", "names": ["Foo Inc"]},
                {"type": "PERSON", "names": ["Bob"]},
            ],
            use_ner=False,
            anonymize_years=True,
        )
        anon.anonymize_pages({1: "Bob at Foo Inc in 2023."})
        emap = anon.entity_map()
        assert "[ORG_1]" in emap
        assert "[PERSON_1]" in emap
        # Year entry
        assert any("2023" in v for v in emap.values())

    def test_entity_map_canonical_is_first_name(self):
        """The canonical name should be the first in the names list."""
        anon = EntityAnonymizer(
            known_entities=[
                {"type": "ORG", "names": ["Full Company Name", "ShortName"]},
            ],
            use_ner=False,
            anonymize_years=False,
        )
        anon.anonymize_pages({1: "ShortName is part of Full Company Name."})
        assert anon.entity_map()["[ORG_1]"] == "Full Company Name"
