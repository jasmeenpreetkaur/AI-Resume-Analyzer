"""
Tests for src/06_map_skills_to_esco.py - candidate selection must be
deterministic and must look at ALL of a skill's tokens.

The old fuzzy stage took ``next(iter(query_tokens))`` from a *set* (a
different token in different Python processes, because string hashing is
randomised) and kept the first 200 labels containing it in file order.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

SCRIPT = SRC / "06_map_skills_to_esco.py"

TARGET_URI = "http://data.europa.eu/esco/skill/target"


def load_module():

    spec = importlib.util.spec_from_file_location("map_skills_to_esco", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def build_vocabulary(module):
    """
    One real target plus 400 decoys that all contain the very common
    token "management". A one-token candidate search that happens to pick
    "management" would keep only the first 200 decoys and never see the
    target.
    """

    rows = [
        {
            "conceptUri": f"http://data.europa.eu/esco/skill/decoy{i}",
            "preferredLabel": f"decoy{i:03d} management",
            "altLabels": "",
        }
        for i in range(400)
    ]

    rows.append(
        {
            "conceptUri": TARGET_URI,
            "preferredLabel": "kubernetes cluster management",
            "altLabels": "",
        }
    )

    vocabulary = module.ESCOVocabulary()
    vocabulary.build(pd.DataFrame(rows))

    return vocabulary


def test_fuzzy_stage_finds_the_target_whatever_the_token_order():

    module = load_module()

    mapper = module.SkillMapper(build_vocabulary(module))

    # Typo in the last word: no exact or token match, so it has to be
    # found by the fuzzy stage. It shares "kubernetes" and "cluster" with
    # the target but only the rare tokens identify it.
    result = mapper.map_skill("kubernetes cluster managment")

    assert result["esco_skill_uri"] == TARGET_URI
    assert result["match_method"] == "fuzzy_similarity"


def test_candidates_rank_rare_tokens_first_and_ignore_set_order():

    module = load_module()

    vocabulary = build_vocabulary(module)

    ranked = vocabulary.candidate_labels(
        module.tokenize("kubernetes cluster managment")
    )

    # The target shares two rare tokens, every decoy shares one common
    # token ("managment" is a typo, so it matches nothing).
    assert ranked[0] == "kubernetes cluster management"

    # And it is a pure function of the vocabulary and the query.
    assert ranked == vocabulary.candidate_labels(
        module.tokenize("managment cluster kubernetes")
    )


def test_no_shared_token_means_no_candidates():

    module = load_module()

    vocabulary = build_vocabulary(module)

    assert vocabulary.candidate_labels({"zzzz", "qqqq"}) == []

    assert module.SkillMapper(vocabulary).map_skill("zzzz qqqq")[
        "match_method"
    ] == "unmapped"


def test_mapping_is_identical_across_hash_seeds(tmp_path):
    """Run the mapper in fresh processes with different PYTHONHASHSEEDs."""

    probe = tmp_path / "probe.py"

    probe.write_text(
        "import importlib.util, json, sys\n"
        "import pandas as pd\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "rows = [{'conceptUri': f'u{i}', 'preferredLabel': l, 'altLabels': ''}\n"
        "        for i, l in enumerate(['data analysis', 'data visualisation',\n"
        "                               'machine learning', 'machine translation',\n"
        "                               'project management', 'risk management'])]\n"
        "v = m.ESCOVocabulary(); v.build(pd.DataFrame(rows))\n"
        "mapper = m.SkillMapper(v)\n"
        "out = [mapper.map_skill(s) for s in\n"
        "       ['data analyses', 'machine learnin', 'management of projects',\n"
        "        'visualisation of data', 'risk management planning']]\n"
        "print(json.dumps(out, sort_keys=True))\n",
        encoding="utf-8",
    )

    outputs = []

    for seed in ("0", "1", "2", "12345"):

        env = dict(os.environ, PYTHONHASHSEED=seed)

        done = subprocess.run(
            [sys.executable, str(probe)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert done.returncode == 0, done.stderr

        outputs.append(json.loads(done.stdout.strip().splitlines()[-1]))

    assert all(output == outputs[0] for output in outputs)


def test_job_skills_from_the_text_are_merged_with_the_listed_skills():
    """
    "Python Developer" whose structured skill list omits Python must still
    have the Python skill Step 5 extracted from its text (the live app
    reads the text, so the offline pool has to as well).
    """

    module = load_module()

    rows = [
        {"conceptUri": "http://esco/sql", "preferredLabel": "sql", "altLabels": ""},
        {"conceptUri": "http://esco/python", "preferredLabel": "python", "altLabels": ""},
    ]

    vocabulary = module.ESCOVocabulary()
    vocabulary.build(pd.DataFrame(rows))

    mapper = module.SkillMapper(vocabulary)

    jobs = pd.DataFrame(
        {
            "job_id": [1, 2, 3],
            "job_skills": ["['sql']", "['sql']", "[]"],
            "esco_extracted_skills": [
                json.dumps([{"concept_uri": "http://esco/python"}]),
                json.dumps([{"concept_uri": "http://esco/sql"}]),   # duplicate
                "not json",                                          # unreadable
            ],
        }
    )

    merged, report = module.process_dataset(
        jobs, "job_skills", mapper, "jobs", extracted_column="esco_extracted_skills"
    )

    sets = [set(json.loads(cell)) for cell in merged["esco_mapped_skill_uris"]]

    assert sets[0] == {"http://esco/sql", "http://esco/python"}
    assert sets[1] == {"http://esco/sql"}          # no duplicate
    assert sets[2] == set()                        # unreadable cell ignored
    assert report["uris_added_from_text_extraction"] == 1

    # Without the extra column the old behaviour is unchanged.
    plain, plain_report = module.process_dataset(jobs, "job_skills", mapper, "jobs")

    assert set(json.loads(plain["esco_mapped_skill_uris"][0])) == {"http://esco/sql"}
    assert plain_report["uris_added_from_text_extraction"] == 0
