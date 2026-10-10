"""
Shared pytest configuration and fixtures.

Run from the project root:

    python -m pytest tests -q
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


RESUME_LINES = [
    "Jane Doe - Data Scientist",
    "Experience",
    "Built machine learning models in Python and SQL for customer churn.",
    "Deployed Docker containers on AWS and managed data pipelines.",
    "Skills",
    "Python, SQL, machine learning, data analysis, project management,",
    "communication, teamwork, statistics, Tableau and Excel dashboards.",
    "Education",
    "Bachelor of Science in Computer Science with focus on data mining.",
    "Led a team of five engineers to deliver analytics products on time.",
    "Presented quarterly results to senior management and stakeholders.",
    "Certifications and projects include natural language processing work.",
]


def _digital_pdf_bytes(lines: list[str], repeat: int = 1) -> bytes:
    import pymupdf

    document = pymupdf.open()

    page = document.new_page()

    y = 72
    for _ in range(repeat):
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16

    data = document.tobytes()
    document.close()

    return data


def _scanned_page_png(lines: list[str]) -> bytes:
    """Render text to an image (no text layer) like a scanner would."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1654, 2339  # A4 at 200 dpi
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default(size=38)

    y = 150
    for line in lines:
        draw.text((120, y), line, fill="black", font=font)
        y += 70

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def _scanned_pdf_bytes(lines: list[str], pages: int = 1) -> bytes:
    import pymupdf

    png = _scanned_page_png(lines)

    document = pymupdf.open()

    for _ in range(pages):
        page = document.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=png)

    data = document.tobytes()
    document.close()

    return data


@pytest.fixture(scope="session")
def resume_lines():
    return list(RESUME_LINES)


@pytest.fixture(scope="session")
def digital_pdf(resume_lines):
    return _digital_pdf_bytes(resume_lines)


@pytest.fixture(scope="session")
def scanned_pdf(resume_lines):
    return _scanned_pdf_bytes(resume_lines)


@pytest.fixture(scope="session")
def make_digital_pdf():
    return _digital_pdf_bytes


@pytest.fixture(scope="session")
def make_scanned_pdf():
    return _scanned_pdf_bytes
