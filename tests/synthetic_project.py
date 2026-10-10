"""
Builds a tiny, fully synthetic project (raw inputs only) so the real
pipeline scripts can be run end to end in a temporary directory.

Contains deliberate traps:
  * ESCO rows that are NOT skills (occupations) - must never become skills
  * a single-character "skill" - must be ignored
  * image-only (scanned) PDFs, one of them with no CSV row at all
  * resumes whose PDF text differs from the CSV text
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pandas as pd

SKILLS = {
    "INFORMATION-TECHNOLOGY": ["python", "sql", "machine learning", "data analysis", "docker", "java"],
    "ACCOUNTANT": ["accounting", "financial reporting", "bookkeeping", "tax preparation", "auditing"],
    "HEALTHCARE": ["patient care", "nursing", "medication administration", "first aid", "infection control"],
    "TEACHER": ["lesson planning", "classroom management", "curriculum development", "student assessment"],
}
GENERIC = ["communication", "teamwork", "project management", "leadership"]
JOB_CATEGORY = {
    "INFORMATION-TECHNOLOGY": "Information Technology",
    "ACCOUNTANT": "Accountant",
    "HEALTHCARE": "Healthcare",
    "TEACHER": "Teacher",
}
TITLES = {
    "INFORMATION-TECHNOLOGY": ["Python Developer", "Data Analyst"],
    "ACCOUNTANT": ["Senior Accountant", "Tax Associate"],
    "HEALTHCARE": ["Registered Nurse", "Clinical Assistant"],
    "TEACHER": ["High School Teacher", "Private Tutor"],
}

RESUMES_PER_CATEGORY = 6
JOBS_PER_CATEGORY = 10
SCANNED_CSV_IDS = (17, 18)      # image-only PDF AND a CSV row
PDF_ONLY_DIGITAL_ID = 9001
PDF_ONLY_SCANNED_ID = 9002
CSV_ONLY_IDS = (23, 24)         # CSV row, no PDF


def _digital_pdf(lines):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    data = doc.tobytes()
    doc.close()
    return data


def _scanned_pdf(lines):
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=38)
    y = 150
    for line in lines:
        draw.text((120, y), line, fill="black", font=font)
        y += 70
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buffer.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


def build_project(root: Path, seed: int = 7) -> dict:
    rng = random.Random(seed)

    for sub in ("data/external", "data/jobs", "data/resume_dataset/Resume"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # ---- ESCO ---------------------------------------------------------
    rows = []

    def esco(kind, label, alt=""):
        rows.append(
            {
                "conceptType": kind,
                "conceptUri": f"http://data.europa.eu/esco/{'skill' if kind == 'KnowledgeSkillCompetence' else 'occupation'}/{len(rows):04d}",
                "preferredLabel": label,
                "altLabels": alt,
                "hiddenLabels": "",
                "description": f"about {label}",
            }
        )

    for skill in sum(SKILLS.values(), []) + GENERIC:
        esco("KnowledgeSkillCompetence", skill, "python programming" if skill == "python" else "")
    esco("KnowledgeSkillCompetence", "c")
    esco("Occupation", "teacher")
    esco("Occupation", "nurse")

    pd.DataFrame(rows).to_csv(root / "data/external/skills_en.csv", index=False)

    # ---- resumes ------------------------------------------------------
    def resume_text(category, rid):
        own = rng.sample(SKILLS[category], 3)
        gen = rng.sample(GENERIC, 2)
        return "\n".join(
            [
                f"Candidate {rid} - {JOB_CATEGORY[category]} professional",
                "Skills",
                ", ".join(own + gen) + ".",
                "Experience",
                f"Applied {own[0]} and {own[1]} daily; strong {gen[0]} and {gen[1]}.",
                "Education",
                "Bachelor degree with honours.",
            ]
        )

    records, rid = [], 1
    for category in SKILLS:
        for _ in range(RESUMES_PER_CATEGORY):
            records.append(
                {"ID": rid, "Resume_str": resume_text(category, rid),
                 "Resume_html": "<p>x</p>", "Category": category}
            )
            rid += 1

    resumes = pd.DataFrame(records)
    resumes.to_csv(root / "data/resume_dataset/Resume/Resume.csv", index=False)

    def write_pdf(category, resume_id, text, scanned):
        folder = root / "data/resume_dataset/data" / category
        folder.mkdir(parents=True, exist_ok=True)
        lines = text.split("\n")
        data = (_scanned_pdf if scanned else _digital_pdf)(lines)
        (folder / f"{resume_id}.pdf").write_bytes(data)

    for _, row in resumes.iterrows():
        if row.ID in CSV_ONLY_IDS:
            continue
        write_pdf(row.Category, row.ID, row.Resume_str, row.ID in SCANNED_CSV_IDS)

    write_pdf("INFORMATION-TECHNOLOGY", PDF_ONLY_DIGITAL_ID,
              resume_text("INFORMATION-TECHNOLOGY", PDF_ONLY_DIGITAL_ID), False)
    write_pdf("TEACHER", PDF_ONLY_SCANNED_ID,
              resume_text("TEACHER", PDF_ONLY_SCANNED_ID), True)

    # ---- jobs ---------------------------------------------------------
    jobs, jid = [], 1
    for category in SKILLS:
        for _ in range(JOBS_PER_CATEGORY):
            title = rng.choice(TITLES[category])
            own = rng.sample(SKILLS[category], 3)
            gen = rng.sample(GENERIC, 1)
            jobs.append(
                {
                    "job_id": jid,
                    "category": JOB_CATEGORY[category],
                    "job_title": title,
                    "job_description": f"We are hiring a {title}. Requirements: {', '.join(own + gen)}.",
                    "job_skill_set": json.dumps(own + gen).replace('"', "'"),
                    "company": f"Company {jid % 5}",
                    "location": "Remote",
                }
            )
            jid += 1

    pd.DataFrame(jobs).to_csv(root / "data/jobs/all_job_post.csv", index=False)

    # ---- the resume analysed by Step 13 -------------------------------
    lines = ["Alex Kim - Data Analyst", "Skills",
             "python, sql, machine learning, data analysis, communication.",
             "Experience", "Built python and sql pipelines.", "Education", "BSc."]
    (root / "data/test_resume.pdf").write_bytes(_digital_pdf(lines))
    (root / "data/test_resume_scanned.pdf").write_bytes(_scanned_pdf(lines))

    return {"resumes": len(resumes), "jobs": len(jobs)}
