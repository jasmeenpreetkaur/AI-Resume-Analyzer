"""
======================================================================
AI RESUME ANALYZER - SHARED PDF TEXT EXTRACTION (WITH OCR FALLBACK)
======================================================================

One implementation of "PDF -> text" used by every part of the project:

    Step 3  (03_resume_parser.py)               offline PDF parsing
    Step 13 (13_explainable_rag_recommendation) command-line analysis
    Step 14 (14_app.py)                          Streamlit upload

Why this module exists
----------------------
Scanned / image-only resumes have no text layer. Plain PyMuPDF
``page.get_text()`` returns an empty string for them, which used to
surface as a generic "no extractable text" failure.

This module extracts the embedded text layer page by page and decides,
*per page*, whether the text layer is usable. Pages that have little or
no text (or garbled text, or a full-page image with only a token amount
of text on top) are rendered to an image and sent to an OCR engine.

It never silently returns unreliable text: every result carries a
``status`` and ``quality`` plus human-readable ``warnings``, so callers
can tell "clean digital PDF" from "OCR with low confidence" from
"scanned PDF and no OCR engine installed".

OCR engines (first available is used; see ``RESUME_OCR_ENGINE``)
---------------------------------------------------------------
    tesseract  pytesseract + the Tesseract binary      (TESSERACT_CMD optional)
    pymupdf    PyMuPDF's built-in Tesseract integration (needs tessdata)
    rapidocr   rapidocr-onnxruntime (pure pip install, no system binary)

Environment variables
---------------------
    RESUME_OCR_ENGINE   auto (default) | tesseract | pymupdf | rapidocr | none
    RESUME_OCR_LANG     Tesseract language code(s), default "eng"
    TESSERACT_CMD       full path to the tesseract executable (optional)
======================================================================
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

try:
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover - older PyMuPDF
    try:
        import fitz as pymupdf  # type: ignore
    except ImportError:  # pragma: no cover
        pymupdf = None  # type: ignore

logger = logging.getLogger(__name__)


# ======================================================================
# CONFIGURATION
# ======================================================================

# A page whose text layer has fewer alphanumeric characters than this is
# treated as "no usable text layer".
MIN_PAGE_ALNUM = 40

# A page that is mostly one big image and carries less text than this is
# treated as a scan that only has a token text layer (watermark, footer,
# page number...). A normal digital resume page has far more than this.
SCAN_TEXT_THRESHOLD = 200
SCAN_IMAGE_COVERAGE = 0.60

# Resumes are short. Never OCR more pages than this in one call.
DEFAULT_MAX_OCR_PAGES = 8

OCR_DPI = 300

# Below this many alphabetic words the extraction is flagged low quality.
MIN_GOOD_WORDS = 40

# Mean OCR confidence (0-100) below which the result is flagged.
LOW_OCR_CONFIDENCE = 55.0

OCR_INSTALL_HINT = (
    "Install an OCR engine to read scanned resumes: either "
    "`pip install pytesseract pillow` plus the Tesseract program "
    "(https://github.com/tesseract-ocr/tesseract), or the pure-Python "
    "option `pip install rapidocr-onnxruntime`."
)


# ======================================================================
# RESULT TYPES
# ======================================================================

@dataclass
class PageExtraction:
    page_number: int                      # 1-based
    method: str                           # "text" | "ocr" | "empty"
    chars: int                            # alphanumeric characters kept
    needed_ocr: bool = False
    ocr_engine: Optional[str] = None
    ocr_confidence: Optional[float] = None
    note: str = ""


@dataclass
class PDFExtraction:
    text: str
    page_count: int
    pages: list[PageExtraction] = field(default_factory=list)

    # success      good text (digital or OCR)
    # low_quality  text found but sparse / garbled / low OCR confidence
    # ocr_unavailable  scanned pages and no OCR engine could be used
    # empty        nothing readable even after OCR
    # failed       file could not be opened (corrupt, encrypted, ...)
    status: str = "failed"
    quality: str = "empty"                # "good" | "low" | "empty"
    ocr_used: bool = False
    ocr_engine: Optional[str] = None
    ocr_pages: int = 0
    pages_needing_ocr: int = 0
    mean_ocr_confidence: Optional[float] = None
    word_count: int = 0
    warnings: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def usable(self) -> bool:
        """True when ``text`` is worth using (possibly with warnings)."""
        return self.status in ("success", "low_quality") and bool(self.text)

    def user_message(self) -> str:
        """One short sentence suitable for a UI."""
        if self.status == "failed":
            return self.reason or "The PDF could not be read."
        if self.status == "ocr_unavailable":
            return (
                "This looks like a scanned (image-only) PDF and no OCR engine "
                "is available. " + OCR_INSTALL_HINT
            )
        if self.status == "empty":
            return self.reason or "No readable text was found in this PDF."
        if self.warnings:
            return " ".join(self.warnings)
        return ""

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable summary (no text) for reports / JSONL."""
        return {
            "extraction_quality": self.quality,
            "ocr_used": self.ocr_used,
            "ocr_engine": self.ocr_engine,
            "ocr_pages": self.ocr_pages,
            "pages_needing_ocr": self.pages_needing_ocr,
            "mean_ocr_confidence": self.mean_ocr_confidence,
            "extraction_warnings": list(self.warnings),
        }


# ======================================================================
# TEXT CLEANING  (moved here from Step 3 so every caller preprocesses
# PDF text identically)
# ======================================================================

def clean_text(text: str) -> str:
    """
    Clean PDF-extracted text while preserving useful structure.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = text.replace("\x00", " ")
    text = text.replace("\x0c", "\n")  # form feed emitted by OCR engines

    lines = [line.strip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    previous_blank = False

    for line in lines:

        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _alnum_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


_WORD_RE = re.compile(r"[^\W\d_]{2,}")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _looks_garbled(text: str) -> bool:
    """
    Detect a text layer that exists but is not real text
    (broken font encodings, "(cid:123)" placeholders, U+FFFD).
    """

    if not text:
        return False

    if text.count("(cid:") >= 3:
        return True

    length = max(len(text), 1)

    return text.count("�") / length > 0.02


# ======================================================================
# OCR ENGINES
# ======================================================================

OCRResult = tuple  # (text: str, confidence: float | None)


def _render_png(page, dpi: int) -> bytes:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return pixmap.tobytes("png")


def _ocr_lang() -> str:
    return os.getenv("RESUME_OCR_LANG", "eng").strip() or "eng"


# ---- tesseract through pytesseract -----------------------------------

def _tesseract_available() -> tuple[bool, str]:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False, "pytesseract and/or pillow not installed"

    command = os.getenv("TESSERACT_CMD", "").strip()

    if command:
        pytesseract.pytesseract.tesseract_cmd = command

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:  # TesseractNotFoundError and friends
        return False, f"Tesseract executable not found ({exc})"

    return True, "ok"


def _ocr_tesseract(page, dpi: int) -> OCRResult:
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(_render_png(page, dpi)))

    data = pytesseract.image_to_data(
        image,
        lang=_ocr_lang(),
        output_type=pytesseract.Output.DICT,
    )

    lines: dict[tuple[int, int, int], list[str]] = {}
    order: list[tuple[int, int, int]] = []
    confidences: list[float] = []

    for index, raw in enumerate(data["text"]):

        word = (raw or "").strip()

        if not word:
            continue

        key = (
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index],
        )

        if key not in lines:
            lines[key] = []
            order.append(key)

        lines[key].append(word)

        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0

        if confidence >= 0:
            confidences.append(confidence)

    out: list[str] = []
    previous_block = None

    for key in order:

        if previous_block is not None and key[0] != previous_block:
            out.append("")

        out.append(" ".join(lines[key]))
        previous_block = key[0]

    mean_confidence = (
        sum(confidences) / len(confidences) if confidences else None
    )

    return "\n".join(out), mean_confidence


# ---- tesseract built into PyMuPDF -------------------------------------

def _pymupdf_ocr_available() -> tuple[bool, str]:

    if pymupdf is None:
        return False, "PyMuPDF not installed"

    if not hasattr(pymupdf, "get_tessdata"):
        return False, "this PyMuPDF build has no OCR support"

    try:
        tessdata = pymupdf.get_tessdata()
    except Exception as exc:
        return False, f"tessdata not found ({exc})"

    if not tessdata:
        return False, "tessdata not found (set TESSDATA_PREFIX)"

    return True, "ok"


def _ocr_pymupdf(page, dpi: int) -> OCRResult:
    textpage = page.get_textpage_ocr(
        language=_ocr_lang(),
        dpi=dpi,
        full=True,
    )

    return page.get_text("text", textpage=textpage), None


# ---- RapidOCR (pip only) ------------------------------------------------

_RAPIDOCR_ENGINE: Any = None


def _rapidocr_available() -> tuple[bool, str]:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False, "rapidocr-onnxruntime not installed"

    return True, "ok"


def _ocr_rapidocr(page, dpi: int) -> OCRResult:
    global _RAPIDOCR_ENGINE

    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    if _RAPIDOCR_ENGINE is None:
        _RAPIDOCR_ENGINE = RapidOCR()

    pixmap = page.get_pixmap(dpi=dpi, alpha=False)

    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )

    result, _ = _RAPIDOCR_ENGINE(image)

    if not result:
        return "", None

    boxes = []

    for box, text, score in result:

        text = (text or "").strip()

        if not text:
            continue

        ys = [point[1] for point in box]
        xs = [point[0] for point in box]

        boxes.append(
            {
                "text": text,
                "score": float(score),
                "x": min(xs),
                "y": (min(ys) + max(ys)) / 2.0,
                "h": max(ys) - min(ys),
            }
        )

    if not boxes:
        return "", None

    heights = sorted(item["h"] for item in boxes)
    median_height = heights[len(heights) // 2] or 1.0

    # Group fragments into visual lines by vertical proximity.
    boxes.sort(key=lambda item: (item["y"], item["x"]))

    grouped: list[list[dict]] = []

    for item in boxes:

        if (
            grouped
            and abs(item["y"] - grouped[-1][0]["y"]) <= 0.6 * median_height
        ):
            grouped[-1].append(item)
        else:
            grouped.append([item])

    lines = [
        " ".join(
            part["text"]
            for part in sorted(line, key=lambda item: item["x"])
        )
        for line in grouped
    ]

    mean_confidence = 100.0 * sum(
        item["score"] for item in boxes
    ) / len(boxes)

    return "\n".join(lines), mean_confidence


# ---- registry -------------------------------------------------------------

@dataclass(frozen=True)
class OCREngine:
    name: str
    check: Callable[[], tuple[bool, str]]
    run: Callable[[Any, int], OCRResult]


_ENGINES = [
    OCREngine("tesseract", _tesseract_available, _ocr_tesseract),
    OCREngine("pymupdf", _pymupdf_ocr_available, _ocr_pymupdf),
    OCREngine("rapidocr", _rapidocr_available, _ocr_rapidocr),
]

_AVAILABILITY_CACHE: dict[str, tuple[bool, str]] = {}


def _engine_available(engine: OCREngine) -> tuple[bool, str]:

    cached = _AVAILABILITY_CACHE.get(engine.name)

    if cached is not None:
        return cached

    try:
        result = engine.check()
    except Exception as exc:  # defensive: never break extraction
        result = (False, f"availability check failed ({exc})")

    _AVAILABILITY_CACHE[engine.name] = result

    return result


def reset_ocr_cache() -> None:
    """Forget cached engine availability (used by tests / after installs)."""
    global _RAPIDOCR_ENGINE
    _AVAILABILITY_CACHE.clear()
    _RAPIDOCR_ENGINE = None


def select_ocr_engine(preference: Optional[str] = None) -> Optional[OCREngine]:
    """
    Return the OCR engine to use, or None when OCR is disabled or no
    engine is available.
    """

    preference = (
        preference
        or os.getenv("RESUME_OCR_ENGINE", "auto")
        or "auto"
    ).strip().lower()

    if preference in ("none", "off", "false", "0"):
        return None

    candidates = [
        engine
        for engine in _ENGINES
        if preference in ("auto", engine.name)
    ]

    for engine in candidates:
        if _engine_available(engine)[0]:
            return engine

    return None


def ocr_engine_status() -> dict[str, Any]:
    """Diagnostics for a UI / log: which engines are usable and why not."""

    selected = select_ocr_engine()

    return {
        "selected": selected.name if selected else None,
        "engines": {
            engine.name: {
                "available": _engine_available(engine)[0],
                "detail": _engine_available(engine)[1],
            }
            for engine in _ENGINES
        },
        "hint": None if selected else OCR_INSTALL_HINT,
    }


# ======================================================================
# PAGE ANALYSIS
# ======================================================================

def _image_coverage(page) -> float:
    """Fraction of the page area covered by embedded images (0..1)."""

    try:
        infos = page.get_image_info()
    except Exception:
        return 0.0

    page_area = page.rect.width * page.rect.height

    if page_area <= 0:
        return 0.0

    covered = 0.0

    for info in infos:

        try:
            rect = pymupdf.Rect(info["bbox"]) & page.rect
        except Exception:
            continue

        if rect.is_empty:
            continue

        covered += rect.width * rect.height

    return min(1.0, covered / page_area)


def _page_needs_ocr(native_text: str, page) -> tuple[bool, str]:

    alnum = _alnum_count(native_text)

    if alnum < MIN_PAGE_ALNUM:
        return True, "no usable text layer"

    if _looks_garbled(native_text):
        return True, "garbled text layer"

    if (
        alnum < SCAN_TEXT_THRESHOLD
        and _image_coverage(page) >= SCAN_IMAGE_COVERAGE
    ):
        return True, "full-page image with only a token text layer"

    return False, ""


# ======================================================================
# MAIN ENTRY POINT
# ======================================================================

PDFSource = Union[bytes, bytearray, str, "os.PathLike[str]"]


def _open_document(source: PDFSource):

    if isinstance(source, (bytes, bytearray)):
        return pymupdf.open(stream=bytes(source), filetype="pdf")

    return pymupdf.open(str(source))


def extract_pdf_text(
    source: PDFSource,
    *,
    ocr: Optional[str] = None,
    max_ocr_pages: int = DEFAULT_MAX_OCR_PAGES,
    dpi: int = OCR_DPI,
) -> PDFExtraction:
    """
    Extract text from a PDF (bytes or path), using OCR for pages that
    have no usable text layer.

    ``ocr`` overrides ``RESUME_OCR_ENGINE`` ("auto", "tesseract",
    "pymupdf", "rapidocr" or "none").

    This function does not raise for bad input; inspect ``status``.
    """

    if pymupdf is None:
        raise RuntimeError(
            "PyMuPDF is not installed. Install it with: pip install pymupdf"
        )

    try:
        document = _open_document(source)
    except Exception as exc:
        return PDFExtraction(
            text="",
            page_count=0,
            status="failed",
            quality="empty",
            reason=f"Could not open the PDF: {exc}",
        )

    try:

        if getattr(document, "needs_pass", False):

            # Some PDFs are "encrypted" with an empty user password.
            if not document.authenticate(""):
                return PDFExtraction(
                    text="",
                    page_count=len(document),
                    status="failed",
                    quality="empty",
                    reason="The PDF is password-protected.",
                )

        page_count = len(document)

        if page_count == 0:
            return PDFExtraction(
                text="",
                page_count=0,
                status="failed",
                quality="empty",
                reason="The PDF has no pages.",
            )

        engine = select_ocr_engine(ocr)

        pages: list[PageExtraction] = []
        page_texts: list[str] = []
        warnings: list[str] = []

        ocr_pages = 0
        pages_needing_ocr = 0
        pages_unresolved = 0
        confidences: list[float] = []

        for index in range(page_count):

            page = document[index]

            try:
                native = page.get_text("text") or ""
            except Exception:
                native = ""

            needs_ocr, why = _page_needs_ocr(native, page)

            native_alnum = _alnum_count(native)

            chosen_text = native
            method = "text" if native_alnum else "empty"
            used_engine: Optional[str] = None
            page_confidence: Optional[float] = None
            note = ""

            if needs_ocr:

                pages_needing_ocr += 1
                note = why

                if engine is None:

                    pages_unresolved += 1
                    note += "; no OCR engine available"

                elif ocr_pages >= max_ocr_pages:

                    pages_unresolved += 1
                    note += f"; OCR page limit ({max_ocr_pages}) reached"

                else:

                    try:
                        ocr_text, page_confidence = engine.run(page, dpi)
                    except Exception as exc:
                        ocr_text = ""
                        pages_unresolved += 1
                        note += f"; OCR failed ({exc})"
                        logger.warning(
                            "OCR (%s) failed on page %d: %s",
                            engine.name,
                            index + 1,
                            exc,
                        )
                    else:

                        ocr_pages += 1
                        used_engine = engine.name

                        if _alnum_count(ocr_text) > native_alnum:

                            chosen_text = ocr_text
                            method = "ocr"

                            if page_confidence is not None:
                                confidences.append(page_confidence)

                        else:

                            used_engine = None
                            note += "; OCR found no more text than the text layer"

            chosen_alnum = _alnum_count(chosen_text)

            if chosen_alnum == 0:
                method = "empty"

            pages.append(
                PageExtraction(
                    page_number=index + 1,
                    method=method,
                    chars=chosen_alnum,
                    needed_ocr=needs_ocr,
                    ocr_engine=used_engine,
                    ocr_confidence=(
                        round(page_confidence, 1)
                        if page_confidence is not None
                        else None
                    ),
                    note=note,
                )
            )

            if chosen_text.strip():
                page_texts.append(chosen_text)

    finally:
        document.close()

    text = clean_text("\n\n".join(page_texts))
    words = _word_count(text)

    ocr_pages_applied = sum(1 for page in pages if page.method == "ocr")

    mean_confidence = (
        round(sum(confidences) / len(confidences), 1)
        if confidences
        else None
    )

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    if ocr_pages_applied:
        warnings.append(
            f"{ocr_pages_applied} of {page_count} page(s) were read with OCR "
            f"({engine.name if engine else 'unknown'}); recognised text may "
            f"contain errors."
        )

    if mean_confidence is not None and mean_confidence < LOW_OCR_CONFIDENCE:
        warnings.append(
            f"OCR confidence is low ({mean_confidence:.0f}/100); "
            f"skill detection may be incomplete."
        )

    if pages_unresolved and text:
        warnings.append(
            f"{pages_unresolved} page(s) looked scanned but could not be "
            f"read; the results use only the readable pages."
        )

    if text and words < MIN_GOOD_WORDS:
        warnings.append(
            f"Only {words} readable words were found; results may be "
            f"unreliable."
        )

    # ------------------------------------------------------------------
    # Status / quality
    # ------------------------------------------------------------------

    reason = ""

    if not text:

        quality = "empty"

        if pages_needing_ocr and engine is None:
            status = "ocr_unavailable"
            reason = (
                "The PDF has no text layer (it looks scanned) and no OCR "
                "engine is available. " + OCR_INSTALL_HINT
            )
        elif pages_needing_ocr and pages_unresolved == pages_needing_ocr:
            status = "empty"
            reason = (
                "The PDF looks scanned but OCR could not read it "
                "(see warnings / logs)."
            )
        else:
            status = "empty"
            reason = "No readable text was found in this PDF."

    else:

        low = (
            words < MIN_GOOD_WORDS
            or pages_unresolved > 0
            or (
                mean_confidence is not None
                and mean_confidence < LOW_OCR_CONFIDENCE
            )
            or _looks_garbled(text)
        )

        quality = "low" if low else "good"
        status = "low_quality" if low else "success"

    return PDFExtraction(
        text=text,
        page_count=page_count,
        pages=pages,
        status=status,
        quality=quality,
        ocr_used=ocr_pages_applied > 0,
        ocr_engine=(engine.name if engine and ocr_pages_applied else None),
        ocr_pages=ocr_pages_applied,
        pages_needing_ocr=pages_needing_ocr,
        mean_ocr_confidence=mean_confidence,
        word_count=words,
        warnings=warnings,
        reason=reason,
    )
