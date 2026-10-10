"""
AI Resume Analyzer - Step 3
Resume PDF Parser

Pipeline:
PDF -> Text Extraction (+ OCR for scanned pages) -> Text Cleaning
    -> Section Detection -> JSONL

Text extraction is the shared ``pdf_text`` module, the same code the
Streamlit app and Step 13 use. Pages with no usable text layer
(scanned / image-only resumes) are read with OCR when an OCR engine is
installed; every record says whether OCR was used and how reliable the
text is.

The JSONL written here is consumed by Step 4
(04_prepare_text_data.py), which uses it as the resume text for the rest
of the offline pipeline.

No ML, embeddings, LLM, ESCO matching, or ATS scoring is performed here.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pdf_text import (
        PDFExtraction,
        clean_text,
        extract_pdf_text as extract_pdf,
        ocr_engine_status,
    )
except ImportError as exc:
    print(f"Could not import the shared PDF module: {exc}")
    print("Install PyMuPDF using: pip install pymupdf")
    sys.exit(1)


# ============================================================
# PROJECT PATH DISCOVERY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESUME_DATASET_DIR = DATA_DIR / "resume_dataset" / "data"

PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "resume_parsed"

REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = OUTPUT_DIR / "resumes_parsed.jsonl"
REPORT_FILE = REPORT_DIR / "resume_parsing_report.json"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# SECTION DEFINITIONS
# ============================================================

# These are section aliases, not hardcoded resume contents.
# Different resumes use different names for the same section.

SECTION_ALIASES = {
    "summary": {
        "summary",
        "professional summary",
        "career summary",
        "profile",
        "professional profile",
        "personal profile",
        "about me",
        "about"
    },

    "objective": {
        "objective",
        "career objective",
        "professional objective",
        "career goal"
    },

    "education": {
        "education",
        "educational background",
        "academic background",
        "academic qualifications",
        "qualifications"
    },

    "experience": {
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "professional background"
    },

    "skills": {
        "skills",
        "skill",
        "key skills",
        "core skills",
        "core competencies",
        "competencies"
    },

    "technical_skills": {
        "technical skills",
        "technical skill",
        "technology skills",
        "technical competencies",
        "technical expertise"
    },

    "projects": {
        "projects",
        "academic projects",
        "personal projects",
        "key projects",
        "project experience"
    },

    "certifications": {
        "certifications",
        "certificates",
        "certification",
        "professional certifications"
    },

    "achievements": {
        "achievements",
        "accomplishments",
        "honors",
        "honours"
    },

    "internships": {
        "internship",
        "internships",
        "internship experience"
    },

    "publications": {
        "publications",
        "research publications",
        "papers",
        "research papers"
    },

    "awards": {
        "awards",
        "awards and honors",
        "honours and awards"
    },

    "languages": {
        "languages",
        "language",
        "language skills"
    },

    "interests": {
        "interests",
        "hobbies",
        "hobbies and interests"
    }
}


# Build reverse lookup once.
SECTION_LOOKUP = {
    alias: section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


# ============================================================
# PDF DISCOVERY
# ============================================================

def discover_pdf_files(root: Path) -> List[Path]:
    """
    Recursively discover all PDF files.

    No category names or PDF counts are hardcoded.
    """

    if not root.exists():
        raise FileNotFoundError(
            f"Resume dataset directory does not exist: {root}"
        )

    pdf_files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    return pdf_files


# ============================================================
# CATEGORY DETECTION
# ============================================================

def get_category(pdf_path: Path, root: Path) -> Optional[str]:
    """
    Determine category from the PDF's directory relative to
    the resume dataset root.

    Example:

        data/resume_dataset/data/ENGINEERING/file.pdf

    -> ENGINEERING

    No category names are hardcoded.
    """

    try:
        relative_path = pdf_path.relative_to(root)
    except ValueError:
        return None

    parent_parts = relative_path.parts[:-1]

    if not parent_parts:
        return None

    # The first directory under the dataset root is treated
    # as the category.
    return parent_parts[0]


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================
#
# Extraction and cleaning live in pdf_text.py (shared with the app and
# Step 13), including the OCR fallback for scanned pages.


# ============================================================
# SECTION HEADER NORMALIZATION
# ============================================================

def normalize_section_header(line: str) -> str:
    """
    Normalize a possible section heading for matching.
    """

    line = line.strip().lower()

    # Remove common decorative characters.
    line = re.sub(r"^[\W_]+|[\W_]+$", "", line)

    # Normalize separators.
    line = re.sub(r"[_\-:]+", " ", line)

    # Normalize whitespace.
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def detect_section(line: str) -> Optional[str]:
    """
    Detect whether a line represents a known resume section.

    Conservative rule-based detection is used to reduce
    false positives.
    """

    normalized = normalize_section_header(line)

    if not normalized:
        return None

    if normalized in SECTION_LOOKUP:
        return SECTION_LOOKUP[normalized]

    return None


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into detected sections.

    Text before the first detected section is intentionally
    not forced into a section.
    """

    lines = text.splitlines()

    sections: Dict[str, List[str]] = {}

    current_section: Optional[str] = None

    for line in lines:

        detected = detect_section(line)

        if detected:

            current_section = detected

            if current_section not in sections:
                sections[current_section] = []

            continue

        if current_section is not None:
            sections[current_section].append(line)

    cleaned_sections = {}

    for section_name, section_lines in sections.items():

        section_text = "\n".join(section_lines)

        section_text = clean_text(section_text)

        if section_text:
            cleaned_sections[section_name] = section_text

    return cleaned_sections


# ============================================================
# RESUME RECORD
# ============================================================

def _display_path(pdf_path: Path) -> str:

    try:
        return str(pdf_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(pdf_path)


def _failed_record(
    pdf_path: Path,
    category: Optional[str],
    reason: str,
    *,
    pdf_status: str = "failed",
    page_count: int = 0,
    detail: Optional[Dict] = None
) -> Dict:

    record = {
        "resume_id": pdf_path.stem,
        "pdf_path": _display_path(pdf_path),
        "category": category,
        "extraction_status": "failed",
        "pdf_status": pdf_status,
        "failure_reason": reason,
        "page_count": page_count,
        "raw_text": "",
        "sections": {},
        "character_count": 0,
        "word_count": 0,
        "extraction_quality": "empty",
        "ocr_used": False,
        "ocr_engine": None,
        "ocr_pages": 0,
        "pages_needing_ocr": 0,
        "mean_ocr_confidence": None,
        "extraction_warnings": []
    }

    if detail:
        record.update(detail)

    return record


def build_resume_record(
    pdf_path: Path,
    root: Path
) -> Dict:

    category = get_category(pdf_path, root)

    try:

        extraction = extract_pdf(pdf_path)

    except Exception as exc:

        return _failed_record(pdf_path, category, str(exc))

    if not extraction.usable:

        # Includes scanned PDFs when no OCR engine is installed: the
        # reason says so instead of a generic "no extractable text".
        return _failed_record(
            pdf_path,
            category,
            extraction.user_message() or "No extractable text",
            pdf_status=extraction.status,
            page_count=extraction.page_count,
            detail=extraction.to_record()
        )

    cleaned = extraction.text

    sections = extract_sections(cleaned)

    return {
        "resume_id": pdf_path.stem,
        "pdf_path": _display_path(pdf_path),
        "category": category,
        "extraction_status": "success",
        "pdf_status": extraction.status,
        "page_count": extraction.page_count,
        "raw_text": cleaned,
        "sections": sections,
        "character_count": len(cleaned),
        "word_count": len(cleaned.split()),
        **extraction.to_record()
    }


# ============================================================
# REPORT GENERATION
# ============================================================

def create_report(
    pdf_files: List[Path],
    records: List[Dict]
) -> Dict:

    total = len(records)

    successful = [
        record
        for record in records
        if record["extraction_status"] == "success"
    ]

    failed = [
        record
        for record in records
        if record["extraction_status"] == "failed"
    ]

    total_pages = sum(
        record.get("page_count", 0)
        for record in records
    )

    total_words = sum(
        record.get("word_count", 0)
        for record in successful
    )

    category_counts = Counter(
        record["category"]
        for record in records
        if record.get("category")
    )

    section_counts = Counter()

    for record in successful:

        for section in record.get("sections", {}):
            section_counts[section] += 1

    ocr_records = [
        record
        for record in records
        if record.get("ocr_used")
    ]

    return {
        "total_pdfs_discovered": len(pdf_files),

        "total_results": total,

        "success": len(successful),

        "failed": len(failed),

        "success_percentage": (
            round((len(successful) / total) * 100, 2)
            if total
            else 0
        ),

        "total_pages": total_pages,

        "average_pages_per_resume": (
            round(total_pages / total, 2)
            if total
            else 0
        ),

        "average_words_per_successful_resume": (
            round(total_words / len(successful), 2)
            if successful
            else 0
        ),

        "pdf_status_counts": dict(
            sorted(
                Counter(
                    record.get("pdf_status", "unknown")
                    for record in records
                ).items()
            )
        ),

        "ocr": {
            "engine_status": ocr_engine_status(),
            "pdfs_read_with_ocr": len(ocr_records),
            "ocr_pages": sum(
                record.get("ocr_pages", 0)
                for record in ocr_records
            ),
            "low_quality_extractions": sum(
                1
                for record in records
                if record.get("extraction_quality") == "low"
            ),
            "scanned_pdfs_without_ocr_engine": sum(
                1
                for record in records
                if record.get("pdf_status") == "ocr_unavailable"
            )
        },

        "category_distribution": dict(
            sorted(category_counts.items())
        ),

        "section_detection_frequency": dict(
            sorted(
                section_counts.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ),

        "failed_pdfs": [
            {
                "pdf_path": record["pdf_path"],
                "reason": record.get(
                    "failure_reason",
                    "Unknown"
                )
            }
            for record in failed
        ],

        "output_file": str(
            OUTPUT_FILE.relative_to(PROJECT_ROOT)
        ),

        "report_file": str(
            REPORT_FILE.relative_to(PROJECT_ROOT)
        )
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    pdf_files: List[Path],
    records: List[Dict]
) -> None:

    if len(pdf_files) != len(records):
        raise RuntimeError(
            "Validation failed: "
            f"{len(pdf_files)} PDFs discovered but "
            f"{len(records)} results generated."
        )

    for record in records:

        required_fields = {
            "resume_id",
            "pdf_path",
            "category",
            "extraction_status",
            "page_count",
            "raw_text",
            "sections",
            "character_count",
            "word_count"
        }

        missing = required_fields - record.keys()

        if missing:
            raise RuntimeError(
                f"Missing fields in result: {missing}"
            )

        if record["extraction_status"] == "success":

            if not record["raw_text"].strip():
                raise RuntimeError(
                    f"Successful resume has empty text: "
                    f"{record['pdf_path']}"
                )

            if record["page_count"] <= 0:
                raise RuntimeError(
                    f"Invalid page count: "
                    f"{record['pdf_path']}"
                )

    logger.info("Validation: PASSED")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    print("=" * 70)
    print("AI RESUME ANALYZER - STEP 3 RESUME PARSING")
    print("=" * 70)

    logger.info(
        "Project root: %s",
        PROJECT_ROOT
    )

    logger.info(
        "Resume dataset directory: %s",
        RESUME_DATASET_DIR
    )

    ocr_status = ocr_engine_status()

    if ocr_status["selected"]:
        logger.info(
            "OCR engine for scanned pages: %s",
            ocr_status["selected"]
        )
    else:
        logger.warning(
            "No OCR engine available: scanned / image-only PDFs will be "
            "reported as failed. %s",
            ocr_status["hint"]
        )

    if not RESUME_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Resume dataset directory not found:\n"
            f"{RESUME_DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Discover PDFs
    # --------------------------------------------------------

    logger.info("Discovering PDF files...")

    pdf_files = discover_pdf_files(
        RESUME_DATASET_DIR
    )

    logger.info(
        "PDFs discovered: %d",
        len(pdf_files)
    )

    if not pdf_files:
        raise RuntimeError(
            "No PDF resumes were discovered."
        )

    # --------------------------------------------------------
    # Parse PDFs
    # --------------------------------------------------------

    records = []

    for index, pdf_path in enumerate(pdf_files, start=1):

        logger.info(
            "[%d/%d] Processing: %s",
            index,
            len(pdf_files),
            pdf_path.name
        )

        record = build_resume_record(
            pdf_path,
            RESUME_DATASET_DIR
        )

        records.append(record)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        pdf_files,
        records
    )

    # --------------------------------------------------------
    # Save JSONL
    # --------------------------------------------------------

    logger.info(
        "Writing parsed resumes..."
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = create_report(
        pdf_files,
        records
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    successful = report["success"]
    failed = report["failed"]

    print()
    print("=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)

    print(f"Total PDFs:       {report['total_pdfs_discovered']}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(
        f"Success rate:     "
        f"{report['success_percentage']}%"
    )

    print(
        f"Total pages:      "
        f"{report['total_pages']}"
    )

    print(
        f"Average pages:    "
        f"{report['average_pages_per_resume']}"
    )

    print(
        f"Average words:    "
        f"{report['average_words_per_successful_resume']}"
    )

    print(
        f"Read with OCR:    "
        f"{report['ocr']['pdfs_read_with_ocr']}"
    )

    print(
        f"Low quality text: "
        f"{report['ocr']['low_quality_extractions']}"
    )

    print(
        f"Scanned, no OCR:  "
        f"{report['ocr']['scanned_pdfs_without_ocr_engine']}"
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("Report:")
    print(REPORT_FILE)

    print()
    print("Section detection:")

    for section, count in report[
        "section_detection_frequency"
    ].items():

        print(
            f"  {section}: {count}"
        )

    print()
    print("Validation: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()