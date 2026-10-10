"""Tests for src/esco_vocab.py - the skills-only ESCO vocabulary."""

from __future__ import annotations

import pandas as pd
import pytest

import esco_vocab
from esco_vocab import (
    ESCOVocabulary,
    extract_skills_from_text,
    find_esco_skills_file,
    load_esco_vocabulary,
    select_skill_rows,
)

SKILL = "http://data.europa.eu/esco/skill/"
OCC = "http://data.europa.eu/esco/occupation/"


def skills_frame() -> pd.DataFrame:
    """A miniature ESCO ``skills_en.csv`` (real column layout)."""
    return pd.DataFrame(
        [
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": SKILL + "py",
                "skillType": "knowledge",
                "preferredLabel": "Python",
                "altLabels": "python programming\nPython 3",
                "hiddenLabels": "pyhton",
                "description": "Programming language",
            },
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": SKILL + "ml",
                "skillType": "skill/competence",
                "preferredLabel": "machine learning",
                "altLabels": "ML\nstatistical learning",
                "hiddenLabels": "",
                "description": "",
            },
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": SKILL + "pm",
                "skillType": "skill/competence",
                "preferredLabel": "manage projects",
                "altLabels": "project management",
                "hiddenLabels": "",
                "description": "",
            },
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": SKILL + "c",
                "skillType": "knowledge",
                "preferredLabel": "C",
                "altLabels": "",
                "hiddenLabels": "",
                "description": "",
            },
        ]
    )


@pytest.fixture()
def vocabulary() -> ESCOVocabulary:
    vocab = ESCOVocabulary()
    vocab.build_from_dataframe(skills_frame())
    return vocab


def labels(found):
    return {item["preferred_label"] for item in found}


# ----------------------------------------------------------------------
# What is matched
# ----------------------------------------------------------------------

def test_matches_preferred_and_alternative_labels(vocabulary):

    found = extract_skills_from_text(
        "Built ML pipelines in Python 3; strong project management.",
        vocabulary,
    )

    assert labels(found) == {"Python", "machine learning", "manage projects"}


def test_skill_at_the_end_of_a_sentence_still_matches(vocabulary):

    # Regression: the Step 5 tokenizer kept the trailing "." on the last
    # word, so "... machine learning." never matched "machine learning".
    assert labels(vocabulary.extract("I know Python.")) == {"Python"}
    assert labels(vocabulary.extract("Skills: machine learning.")) == {
        "machine learning"
    }
    assert esco_vocab.tokenize("node.js, c++ and .NET.") == [
        "node.js", "c++", "and", ".net",
    ]


def test_results_carry_uri_and_are_deduplicated(vocabulary):

    found = vocabulary.extract("python python PYTHON programming")

    assert len(found) == 1
    assert found[0]["concept_uri"] == SKILL + "py"


def test_multiword_terms_match_as_phrases(vocabulary):

    assert labels(vocabulary.extract("statistical learning methods")) == {
        "machine learning"
    }
    assert vocabulary.extract("statistical methods learning") == []


def test_hidden_labels_are_not_used(vocabulary):

    assert vocabulary.extract("pyhton developer") == []


def test_single_character_terms_are_ignored(vocabulary):

    # ESCO lists "C" as a skill; matching a lone letter would fire on
    # grades, initials and bullets, so it is deliberately dropped.
    assert vocabulary.extract("Grade: C. Section C. Plan A") == []
    assert vocabulary.skipped_short_terms >= 1


def test_uris_are_never_matching_text(vocabulary):

    text = "see http://data.europa.eu/esco/skill/py for details"

    assert vocabulary.extract(text) == []


def test_preferred_label_wins_over_another_skills_synonym():

    frame = pd.DataFrame(
        [
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": SKILL + "a",
                "preferredLabel": "data analysis",
                "altLabels": "analytics",
            },
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": SKILL + "b",
                "preferredLabel": "analytics",
                "altLabels": "",
            },
        ]
    )

    vocab = ESCOVocabulary()
    vocab.build_from_dataframe(frame)

    found = vocab.extract("worked in analytics")

    assert [item["concept_uri"] for item in found] == [SKILL + "b"]


def test_lookup_term_and_label_for(vocabulary):

    assert vocabulary.lookup_term("Machine-Learning") is not None
    assert vocabulary.lookup_term("quantum basket weaving") is None
    assert vocabulary.label_for(SKILL + "ml") == "machine learning"


# ----------------------------------------------------------------------
# Non-skill ESCO data must never become "skills"
# ----------------------------------------------------------------------

def test_non_skill_rows_in_a_mixed_table_are_dropped():

    mixed = pd.concat(
        [
            skills_frame(),
            pd.DataFrame(
                [
                    {
                        "conceptType": "Occupation",
                        "conceptUri": OCC + "dev",
                        "preferredLabel": "software developer",
                        "altLabels": "programmer",
                    },
                    {
                        "conceptType": "SkillGroup",
                        "conceptUri": SKILL + "S1",
                        "preferredLabel": "communication, collaboration",
                        "altLabels": "",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    vocab = ESCOVocabulary()
    vocab.build_from_dataframe(mixed)

    assert vocab.dropped_rows == 2
    assert vocab.extract("experienced software developer and programmer") == []
    assert vocab.skill_count == 4


def test_occupations_table_is_rejected():

    occupations = pd.DataFrame(
        [
            {
                "conceptType": "Occupation",
                "conceptUri": OCC + "dev",
                "preferredLabel": "software developer",
                "altLabels": "programmer",
            }
        ]
    )

    with pytest.raises(ValueError):
        select_skill_rows(occupations)


def test_uri_marker_filter_when_no_concept_type_column():

    frame = pd.DataFrame(
        [
            {"conceptUri": SKILL + "py", "preferredLabel": "Python"},
            {"conceptUri": OCC + "dev", "preferredLabel": "software developer"},
        ]
    )

    selected, stats = select_skill_rows(frame)

    assert list(selected["preferredLabel"]) == ["Python"]
    assert stats["dropped_rows"] == 1


def test_loader_reads_only_the_skills_file(tmp_path, monkeypatch):
    """
    data/external holds many ESCO CSVs. Only skills_en.csv may feed the
    vocabulary, no matter what other files (or columns) are present.
    """

    external = tmp_path / "external"
    external.mkdir()

    skills_frame().to_csv(external / "skills_en.csv", index=False)

    pd.DataFrame(
        [
            {
                "conceptType": "Occupation",
                "conceptUri": OCC + "dev",
                "preferredLabel": "software developer",
                "altLabels": "programmer",
                "title": "Software developer",
                "name": "dev",
            }
        ]
    ).to_csv(external / "occupations_en.csv", index=False)

    pd.DataFrame(
        {"Level 0 URI": [SKILL + "S"], "Level 0 preferred term": ["languages"]}
    ).to_csv(external / "skillsHierarchy_en.csv", index=False)

    monkeypatch.setattr(
        esco_vocab,
        "ESCO_SKILLS_FILE_CANDIDATES",
        [tmp_path / "processed" / "esco_skills_clean.csv",
         external / "skills_en.csv"],
    )

    vocab = load_esco_vocabulary()

    assert vocab.source.endswith("skills_en.csv")
    assert vocab.skill_count == 4
    assert vocab.extract("software developer, programmer, languages") == []
    assert labels(vocab.extract("python")) == {"Python"}


def test_processed_clean_file_has_priority(tmp_path, monkeypatch):

    processed = tmp_path / "processed"
    external = tmp_path / "external"
    processed.mkdir()
    external.mkdir()

    clean = skills_frame().rename(
        columns={
            "conceptType": "concepttype",
            "conceptUri": "concepturi",
            "preferredLabel": "preferredlabel",
            "altLabels": "altlabels",
        }
    )

    clean.to_csv(processed / "esco_skills_clean.csv", index=False)
    skills_frame().to_csv(external / "skills_en.csv", index=False)

    candidates = [
        processed / "esco_skills_clean.csv",
        external / "skills_en.csv",
    ]

    assert find_esco_skills_file(candidates) == candidates[0]

    vocab = load_esco_vocabulary(candidates[0])

    assert labels(vocab.extract("machine learning")) == {"machine learning"}


def test_missing_skills_file_raises_with_instructions(tmp_path, monkeypatch):

    # An external folder that has occupations but NO skills file.
    external = tmp_path / "external"
    external.mkdir()

    pd.DataFrame(
        [{"conceptType": "Occupation", "conceptUri": OCC + "x",
          "preferredLabel": "x"}]
    ).to_csv(external / "occupations_en.csv", index=False)

    monkeypatch.setattr(
        esco_vocab,
        "ESCO_SKILLS_FILE_CANDIDATES",
        [tmp_path / "processed" / "esco_skills_clean.csv",
         external / "skills_en.csv"],
    )

    with pytest.raises(FileNotFoundError) as info:
        load_esco_vocabulary()

    assert "skills_en.csv" in str(info.value)


def test_fingerprint_changes_with_vocabulary_content():

    a = ESCOVocabulary()
    a.build_from_dataframe(skills_frame())

    b = ESCOVocabulary()
    b.build_from_dataframe(skills_frame().iloc[:2])

    assert a.fingerprint != b.fingerprint
