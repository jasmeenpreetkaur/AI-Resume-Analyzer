"""
AI RESUME ANALYZER - STEP 6.5
IMPROVED ESCO SKILL MAPPING

Purpose:
    Map extracted resume/job skills to standardized ESCO skills.

Matching strategy:
    1. Exact normalized preferred-label match
    2. Exact normalized alternative-label match
    3. Conservative text normalization
    4. Token-based similarity
    5. Fuzzy matching with confidence threshold

Important:
    - Original extracted skills are preserved.
    - ESCO URIs are preserved.
    - Every mapping records its method and confidence.
    - Low-confidence matches are NOT forced.
    - No hardcoded skills.
    - Original datasets remain untouched.

Inputs:
    data/processed/resumes_skills_extracted.csv
    data/processed/jobs_skills_extracted.csv
    data/processed/esco_skills_clean.csv

Outputs:
    data/processed/resumes_esco_mapped.csv
    data/processed/jobs_esco_mapped.csv
    reports/esco_mapping_report.json
"""

from __future__ import annotations

import ast
import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESUME_INPUT = PROCESSED_DIR / "resumes_skills_extracted.csv"
JOB_INPUT = PROCESSED_DIR / "jobs_skills_extracted.csv"
ESCO_INPUT = PROCESSED_DIR / "esco_skills_clean.csv"

RESUME_OUTPUT = PROCESSED_DIR / "resumes_esco_mapped.csv"
JOB_OUTPUT = PROCESSED_DIR / "jobs_esco_mapped.csv"

REPORT_OUTPUT = REPORTS_DIR / "esco_mapping_report.json"

# Conservative thresholds.
# Exact matches do not need a threshold.
TOKEN_MATCH_THRESHOLD = 0.88
FUZZY_MATCH_THRESHOLD = 0.92

MAX_FUZZY_CANDIDATES = 10

LOG_EVERY = 100


# ======================================================================
# LOGGING
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    """
    Conservative normalization.

    We intentionally do NOT destroy human-readable text.
    Normalization is only used internally for matching.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("&", " and ")

    text = re.sub(r"[/_|]+", " ", text)

    text = re.sub(r"[-]+", " ", text)

    text = re.sub(r"[^\w\s+#.]", " ", text, flags=re.UNICODE)

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def tokenize(text: str) -> set[str]:
    if not text:
        return set()

    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1
    }


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def token_similarity(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    if union == 0:
        return 0.0

    jaccard = intersection / union

    containment = intersection / min(
        len(a_tokens),
        len(b_tokens)
    )

    return max(jaccard, containment)


# ======================================================================
# LIST PARSING
# ======================================================================

def parse_skill_list(value: Any) -> list[str]:
    """
    Safely convert stored skill representations into Python lists.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Python-list representation
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]
        except Exception:
            pass

    # JSON representation
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]
    except Exception:
        pass

    # Fallback
    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


# ======================================================================
# COLUMN DETECTION
# ======================================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str
) -> str:

    normalized = {
        normalize_text(col).replace(" ", "").replace("_", ""): col
        for col in df.columns
    }

    for candidate in candidates:

        key = normalize_text(candidate).replace(
            " ", ""
        ).replace("_", "")

        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not detect {description}. "
        f"Available columns: {list(df.columns)}"
    )


# ======================================================================
# ESCO VOCABULARY
# ======================================================================

class ESCOVocabulary:

    def __init__(self):
        self.skills: dict[str, dict[str, Any]] = {}

        self.preferred_lookup: dict[str, list[str]] = defaultdict(list)

        self.alternative_lookup: dict[str, list[str]] = defaultdict(list)

        self.all_labels: dict[str, list[str]] = defaultdict(list)

        self.label_tokens: dict[str, set[str]] = {}

        self.unique_labels: list[str] = []

    def build(self, df: pd.DataFrame) -> None:

        uri_col = find_column(
            df,
            ["conceptUri", "concept_uri", "uri"],
            "ESCO URI column"
        )

        preferred_col = find_column(
            df,
            ["preferredLabel", "preferred_label", "label"],
            "ESCO preferred label column"
        )

        alt_col = None

        try:
            alt_col = find_column(
                df,
                ["altLabels", "alt_labels", "alternative_labels"],
                "ESCO alternative labels column"
            )
        except ValueError:
            logger.info(
                "No ESCO alternative-label column detected."
            )

        logger.info(
            f"Detected ESCO URI column: {uri_col}"
        )

        logger.info(
            f"Detected ESCO preferred label column: {preferred_col}"
        )

        if alt_col:
            logger.info(
                f"Detected ESCO alternative labels column: {alt_col}"
            )

        for _, row in df.iterrows():

            uri = str(row[uri_col]).strip()
            preferred = str(row[preferred_col]).strip()

            if not uri or not preferred:
                continue

            skill = {
                "uri": uri,
                "preferred_label": preferred
            }

            self.skills[uri] = skill

            normalized_preferred = normalize_text(preferred)

            if normalized_preferred:

                self.preferred_lookup[
                    normalized_preferred
                ].append(uri)

                self.all_labels[
                    normalized_preferred
                ].append(uri)

                self.label_tokens[
                    normalized_preferred
                ] = tokenize(preferred)

            if alt_col:

                alt_value = row[alt_col]

                if not pd.isna(alt_value):

                    alternatives = str(
                        alt_value
                    ).splitlines()

                    for alternative in alternatives:

                        alternative = alternative.strip()

                        if not alternative:
                            continue

                        normalized_alt = normalize_text(
                            alternative
                        )

                        if not normalized_alt:
                            continue

                        self.alternative_lookup[
                            normalized_alt
                        ].append(uri)

                        self.all_labels[
                            normalized_alt
                        ].append(uri)

                        self.label_tokens[
                            normalized_alt
                        ] = tokenize(alternative)

        self.unique_labels = list(
            self.all_labels.keys()
        )

        logger.info(
            f"Unique ESCO skills: {len(self.skills)}"
        )

        logger.info(
            f"Unique ESCO matching labels: "
            f"{len(self.unique_labels)}"
        )


# ======================================================================
# SKILL MAPPER
# ======================================================================

class SkillMapper:

    def __init__(self, vocabulary: ESCOVocabulary):

        self.vocab = vocabulary

        self.stats = {
            "total": 0,
            "mapped": 0,
            "unmapped": 0,
            "preferred_exact": 0,
            "alternative_exact": 0,
            "token_similarity": 0,
            "fuzzy_similarity": 0
        }

    def _make_result(
        self,
        original: str,
        uri: str | None,
        method: str,
        score: float
    ) -> dict[str, Any]:

        if uri is None:

            self.stats["unmapped"] += 1

            return {
                "original_skill": original,
                "esco_skill_uri": "",
                "esco_skill_label": "",
                "match_method": "unmapped",
                "match_score": 0.0
            }

        self.stats["mapped"] += 1

        if method == "preferred_exact":
            self.stats["preferred_exact"] += 1

        elif method == "alternative_exact":
            self.stats["alternative_exact"] += 1

        elif method == "token_similarity":
            self.stats["token_similarity"] += 1

        elif method == "fuzzy_similarity":
            self.stats["fuzzy_similarity"] += 1

        return {
            "original_skill": original,
            "esco_skill_uri": uri,
            "esco_skill_label": self.vocab.skills[
                uri
            ]["preferred_label"],
            "match_method": method,
            "match_score": round(score, 4)
        }

    def map_skill(
        self,
        skill: str
    ) -> dict[str, Any]:

        self.stats["total"] += 1

        original = str(skill).strip()

        if not original:

            return self._make_result(
                original,
                None,
                "unmapped",
                0.0
            )

        normalized = normalize_text(original)

        if not normalized:

            return self._make_result(
                original,
                None,
                "unmapped",
                0.0
            )

        # --------------------------------------------------------------
        # 1. EXACT PREFERRED LABEL
        # --------------------------------------------------------------

        candidates = self.vocab.preferred_lookup.get(
            normalized,
            []
        )

        if candidates:

            return self._make_result(
                original,
                candidates[0],
                "preferred_exact",
                1.0
            )

        # --------------------------------------------------------------
        # 2. EXACT ALTERNATIVE LABEL
        # --------------------------------------------------------------

        candidates = self.vocab.alternative_lookup.get(
            normalized,
            []
        )

        if candidates:

            return self._make_result(
                original,
                candidates[0],
                "alternative_exact",
                1.0
            )

        # --------------------------------------------------------------
        # 3. TOKEN SIMILARITY
        # --------------------------------------------------------------

        best_uri = None
        best_label = None
        best_score = 0.0

        query_tokens = tokenize(normalized)

        if query_tokens:

            for label in self.vocab.unique_labels:

                label_tokens = self.vocab.label_tokens.get(
                    label,
                    set()
                )

                if not label_tokens:
                    continue

                intersection = len(
                    query_tokens & label_tokens
                )

                if intersection == 0:
                    continue

                score = token_similarity(
                    normalized,
                    label
                )

                if score > best_score:

                    best_score = score
                    best_label = label

                    uris = self.vocab.all_labels.get(
                        label,
                        []
                    )

                    if uris:
                        best_uri = uris[0]

        if (
            best_uri
            and best_score >= TOKEN_MATCH_THRESHOLD
        ):

            return self._make_result(
                original,
                best_uri,
                "token_similarity",
                best_score
            )

        # --------------------------------------------------------------
        # 4. FUZZY SIMILARITY
        # --------------------------------------------------------------

        fuzzy_best_uri = None
        fuzzy_best_score = 0.0

        # Use token candidates where possible.
        candidate_labels = []

        first_token = next(
            iter(query_tokens),
            None
        )

        if first_token:

            for label in self.vocab.unique_labels:

                if first_token in self.vocab.label_tokens.get(
                    label,
                    set()
                ):

                    candidate_labels.append(label)

        # If no useful candidates exist, don't scan everything.
        candidate_labels = candidate_labels[
            :MAX_FUZZY_CANDIDATES * 20
        ]

        for label in candidate_labels:

            score = similarity(
                normalized,
                label
            )

            if score > fuzzy_best_score:

                uris = self.vocab.all_labels.get(
                    label,
                    []
                )

                if uris:

                    fuzzy_best_score = score
                    fuzzy_best_uri = uris[0]

        if (
            fuzzy_best_uri
            and fuzzy_best_score >= FUZZY_MATCH_THRESHOLD
        ):

            return self._make_result(
                original,
                fuzzy_best_uri,
                "fuzzy_similarity",
                fuzzy_best_score
            )

        # --------------------------------------------------------------
        # 5. UNMAPPED
        # --------------------------------------------------------------

        return self._make_result(
            original,
            None,
            "unmapped",
            0.0
        )


# ======================================================================
# DATASET PROCESSING
# ======================================================================

def process_dataset(
    df: pd.DataFrame,
    skill_column: str,
    mapper: SkillMapper,
    dataset_name: str
) -> tuple[pd.DataFrame, dict[str, Any]]:

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Mapping {dataset_name}")
    logger.info("=" * 70)

    output = df.copy()

    mapped_skill_lists = []
    mapping_details = []

    total_rows = len(output)
    rows_with_mapped_skills = 0

    total_skills = 0
    mapped_skills = 0

    for index, value in enumerate(
        output[skill_column],
        start=1
    ):

        skills = parse_skill_list(value)

        row_mappings = []

        for skill in skills:

            total_skills += 1

            result = mapper.map_skill(skill)

            row_mappings.append(result)

            if result["esco_skill_uri"]:
                mapped_skills += 1

        mapped = [
            item["esco_skill_uri"]
            for item in row_mappings
            if item["esco_skill_uri"]
        ]

        labels = [
            item["esco_skill_label"]
            for item in row_mappings
            if item["esco_skill_label"]
        ]

        if mapped:
            rows_with_mapped_skills += 1

        mapped_skill_lists.append(
            json.dumps(
                mapped,
                ensure_ascii=False
            )
        )

        mapping_details.append(
            json.dumps(
                row_mappings,
                ensure_ascii=False
            )
        )

        if (
            index % LOG_EVERY == 0
            or index == total_rows
        ):

            logger.info(
                f"{dataset_name} progress: "
                f"{index}/{total_rows}"
            )

    output["esco_mapped_skill_uris"] = mapped_skill_lists

    output["esco_skill_mapping_details"] = mapping_details

    coverage = (
        mapped_skills / total_skills
        if total_skills
        else 0.0
    )

    report = {
        "dataset": dataset_name,
        "rows": total_rows,
        "rows_with_mapped_skills": rows_with_mapped_skills,
        "rows_without_mapped_skills": (
            total_rows - rows_with_mapped_skills
        ),
        "total_extracted_skills": total_skills,
        "mapped_skills": mapped_skills,
        "unmapped_skills": (
            total_skills - mapped_skills
        ),
        "skill_mapping_coverage": round(
            coverage,
            4
        )
    }

    return output, report


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    ensure_directories()

    logger.info("=" * 70)
    logger.info(
        "AI RESUME ANALYZER - STEP 6.5 IMPROVED ESCO MAPPING"
    )
    logger.info("=" * 70)

    # --------------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------------

    required_files = [
        RESUME_INPUT,
        JOB_INPUT,
        ESCO_INPUT
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("Loading datasets")
    logger.info("=" * 70)

    resumes = pd.read_csv(
        RESUME_INPUT
    )

    jobs = pd.read_csv(
        JOB_INPUT
    )

    esco = pd.read_csv(
        ESCO_INPUT
    )

    logger.info(
        f"Resumes: {resumes.shape}"
    )

    logger.info(
        f"Jobs: {jobs.shape}"
    )

    logger.info(
        f"ESCO skills: {esco.shape}"
    )

    # --------------------------------------------------------------
    # Detect skill columns
    # --------------------------------------------------------------

    resume_skill_column = find_column(
        resumes,
        [
            "extracted_skills",
            "skill_labels",
            "resume_skills"
        ],
        "resume skill column"
    )

    job_skill_column = find_column(
        jobs,
        [
            "job_skills",
            "esco_extracted_skills",
            "skill_labels"
        ],
        "job skill column"
    )

    logger.info(
        f"Resume skill column: {resume_skill_column}"
    )

    logger.info(
        f"Job skill column: {job_skill_column}"
    )

    # --------------------------------------------------------------
    # Build ESCO vocabulary
    # --------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("Building ESCO vocabulary")
    logger.info("=" * 70)

    vocabulary = ESCOVocabulary()

    vocabulary.build(esco)

    # --------------------------------------------------------------
    # Create mapper
    # --------------------------------------------------------------

    mapper = SkillMapper(
        vocabulary
    )

    # --------------------------------------------------------------
    # Map resumes
    # --------------------------------------------------------------

    resume_output, resume_report = process_dataset(
        resumes,
        resume_skill_column,
        mapper,
        "resumes"
    )

    # --------------------------------------------------------------
    # Reset mapper statistics for jobs
    # --------------------------------------------------------------

    mapper.stats = {
        "total": 0,
        "mapped": 0,
        "unmapped": 0,
        "preferred_exact": 0,
        "alternative_exact": 0,
        "token_similarity": 0,
        "fuzzy_similarity": 0
    }

    # --------------------------------------------------------------
    # Map jobs
    # --------------------------------------------------------------

    job_output, job_report = process_dataset(
        jobs,
        job_skill_column,
        mapper,
        "jobs"
    )

    # --------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------

    resume_output.to_csv(
        RESUME_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    job_output.to_csv(
        JOB_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info("")
    logger.info(
        f"Saved resume mappings: {RESUME_OUTPUT}"
    )

    logger.info(
        f"Saved job mappings: {JOB_OUTPUT}"
    )

    # --------------------------------------------------------------
    # Build report
    # --------------------------------------------------------------

    report = {
        "step": "6.5",
        "description": "Improved ESCO skill mapping",

        "inputs": {
            "resume_file": str(RESUME_INPUT),
            "job_file": str(JOB_INPUT),
            "esco_file": str(ESCO_INPUT)
        },

        "outputs": {
            "resume_file": str(RESUME_OUTPUT),
            "job_file": str(JOB_OUTPUT)
        },

        "esco": {
            "unique_skills": len(
                vocabulary.skills
            ),
            "unique_matching_labels": len(
                vocabulary.unique_labels
            )
        },

        "matching_strategy": [
            "exact preferred label",
            "exact alternative label",
            "token similarity",
            "conservative fuzzy similarity",
            "unmapped when confidence is insufficient"
        ],

        "resume": resume_report,
        "jobs": job_report,

        "thresholds": {
            "token_similarity": TOKEN_MATCH_THRESHOLD,
            "fuzzy_similarity": FUZZY_MATCH_THRESHOLD
        }
    }

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    logger.info(
        f"Report saved: {REPORT_OUTPUT}"
    )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 6.5 COMPLETE")
    logger.info("=" * 70)

    logger.info(
        "Resume mapped skills: "
        f"{resume_report['mapped_skills']}/"
        f"{resume_report['total_extracted_skills']} "
        f"("
        f"{resume_report['skill_mapping_coverage'] * 100:.2f}%"
        ")"
    )

    logger.info(
        "Job mapped skills: "
        f"{job_report['mapped_skills']}/"
        f"{job_report['total_extracted_skills']} "
        f"("
        f"{job_report['skill_mapping_coverage'] * 100:.2f}%"
        ")"
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()