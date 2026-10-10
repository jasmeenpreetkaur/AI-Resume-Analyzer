"""Tests for src/pdf_text.py - text extraction and OCR fallback."""

from __future__ import annotations

import pymupdf
import pytest

import pdf_text
from pdf_text import extract_pdf_text, ocr_engine_status, select_ocr_engine


def _ocr_available() -> bool:
    pdf_text.reset_ocr_cache()
    return select_ocr_engine("auto") is not None


needs_ocr = pytest.mark.skipif(
    not _ocr_available(),
    reason="no OCR engine installed (pytesseract+Tesseract, PyMuPDF OCR or rapidocr)",
)


def test_digital_pdf_uses_text_layer_and_no_ocr(digital_pdf):

    result = extract_pdf_text(digital_pdf)

    assert result.status == "success"
    assert result.quality == "good"
    assert not result.ocr_used
    assert result.ocr_pages == 0
    assert all(page.method == "text" for page in result.pages)
    assert "machine learning" in result.text.lower()


def test_accepts_a_path(tmp_path, digital_pdf):

    path = tmp_path / "resume.pdf"
    path.write_bytes(digital_pdf)

    assert extract_pdf_text(path).status == "success"
    assert extract_pdf_text(str(path)).status == "success"


def test_scanned_pdf_without_ocr_is_reported_not_silently_empty(scanned_pdf):

    result = extract_pdf_text(scanned_pdf, ocr="none")

    assert result.status == "ocr_unavailable"
    assert not result.usable
    assert result.text == ""
    assert result.pages_needing_ocr == 1
    # The message must tell the user what to do.
    assert "OCR" in result.user_message()
    assert "pip install" in result.user_message()


@needs_ocr
def test_scanned_pdf_is_read_with_ocr(scanned_pdf):

    pdf_text.reset_ocr_cache()

    result = extract_pdf_text(scanned_pdf)

    assert result.usable
    assert result.ocr_used
    assert result.ocr_pages == 1
    assert result.ocr_engine in {"tesseract", "pymupdf", "rapidocr"}
    assert result.pages[0].method == "ocr"

    text = result.text.lower()

    for expected in ("python", "machine learning", "data analysis"):
        assert expected in text

    # OCR output must carry a visible caveat.
    assert any("OCR" in warning for warning in result.warnings)


@needs_ocr
def test_mixed_pdf_ocrs_only_the_scanned_pages(
    make_digital_pdf, make_scanned_pdf, resume_lines
):

    digital = pymupdf.open(stream=make_digital_pdf(resume_lines), filetype="pdf")
    scanned = pymupdf.open(stream=make_scanned_pdf(resume_lines), filetype="pdf")

    digital.insert_pdf(scanned)

    mixed = digital.tobytes()

    digital.close()
    scanned.close()

    result = extract_pdf_text(mixed)

    assert result.page_count == 2
    assert [page.method for page in result.pages] == ["text", "ocr"]
    assert result.ocr_pages == 1
    assert result.usable


@needs_ocr
def test_ocr_page_limit_is_respected(make_scanned_pdf, resume_lines):

    pdf = make_scanned_pdf(resume_lines, pages=3)

    result = extract_pdf_text(pdf, max_ocr_pages=1)

    assert result.ocr_pages == 1
    assert result.pages_needing_ocr == 3
    # Pages that could not be OCR'd must degrade quality, not be hidden.
    assert result.status == "low_quality"
    assert any("could not be read" in w for w in result.warnings)


def test_blank_pdf_is_empty_not_a_crash():

    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()

    result = extract_pdf_text(data, ocr="none")

    assert not result.usable
    assert result.status in {"empty", "ocr_unavailable"}


def test_corrupt_pdf_fails_cleanly():

    result = extract_pdf_text(b"this is not a pdf")

    assert result.status == "failed"
    assert not result.usable
    assert result.reason


def test_password_protected_pdf_fails_cleanly(digital_pdf):

    document = pymupdf.open(stream=digital_pdf, filetype="pdf")
    protected = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="owner",
    )
    document.close()

    result = extract_pdf_text(protected)

    assert result.status == "failed"
    assert "password" in result.reason.lower()


def test_very_short_text_is_flagged_low_quality(make_digital_pdf):

    pdf = make_digital_pdf(["Jane Doe, Data Scientist. Python and SQL expert."])

    result = extract_pdf_text(pdf, ocr="none")

    assert result.status == "low_quality"
    assert result.usable
    assert any("readable words" in warning for warning in result.warnings)


def test_garbled_text_layer_detection():

    assert pdf_text._looks_garbled("(cid:12)(cid:13)(cid:14) abc")
    assert pdf_text._looks_garbled("�" * 10 + "a" * 90)
    assert not pdf_text._looks_garbled("A perfectly normal resume line.")
    assert not pdf_text._looks_garbled("")


def test_clean_text_normalises_whitespace_and_form_feeds():

    cleaned = pdf_text.clean_text("a  b\r\n\r\n\r\n\r\nc\x0cd\xa0e")

    assert "\r" not in cleaned
    assert "\x0c" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "a b" in cleaned


def test_engine_status_reports_every_engine():

    status = ocr_engine_status()

    assert set(status["engines"]) == {"tesseract", "pymupdf", "rapidocr"}
    assert "selected" in status


def test_engine_preference_none_disables_ocr():

    assert select_ocr_engine("none") is None
