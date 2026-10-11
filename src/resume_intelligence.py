"""Evidence-grounded, rule-based full-resume section extraction.

This module deliberately does not use an LLM to invent facts. It segments the
full extracted resume text into likely sections, keeps source snippets, and
marks uncertain/missing sections as unknown rather than fabricating them.
"""
from __future__ import annotations

import re
from typing import Any

SECTION_ALIASES = {
    "summary": ("summary", "professional summary", "profile", "career objective", "objective", "about me"),
    "skills": ("skills", "technical skills", "core skills", "technical competencies", "tools and technologies"),
    "projects": ("projects", "academic projects", "personal projects", "selected projects", "research projects"),
    "experience": ("experience", "work experience", "professional experience", "employment history", "work history"),
    "internships": ("internship", "internships", "research internship", "industrial training", "training"),
    "education": ("education", "academic background", "educational qualifications", "qualification"),
    "certifications": ("certifications", "certificates", "licenses and certifications", "courses and certifications"),
    "achievements": ("achievements", "awards", "honors", "accomplishments", "competitions", "hackathons"),
    "publications_research": ("publications", "research", "research experience", "papers", "patents"),
    "leadership_volunteering": ("leadership", "volunteering", "volunteer experience", "positions of responsibility", "extracurricular"),
    "languages": ("languages", "language proficiency"),
}

HEADER_LOOKUP = {
    re.sub(r"[^a-z0-9]+", "", alias.lower()): key
    for key, aliases in SECTION_ALIASES.items()
    for alias in aliases
}
HEADER_RE = re.compile(r"^\s{0,8}([A-Za-z][A-Za-z0-9 &/()'’.,+-]{1,55})\s*:?\s*$")
BULLET_RE = re.compile(r"^\s*(?:[•●▪◦‣*-]|\d+[.)])\s+(.*)$")
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\.?\s+\d{4}\b|\b(?:19|20)\d{2}\s*(?:[-–—]|to)\s*(?:(?:19|20)\d{2}|present|current)\b",
    re.I,
)
TECH_RE = re.compile(
    r"\b(Python|Java|C\+\+|C#|JavaScript|TypeScript|SQL|R|Go|Rust|React|Angular|"
    r"Node\.?js|Django|Flask|FastAPI|Streamlit|TensorFlow|PyTorch|Keras|scikit-learn|"
    r"Pandas|NumPy|OpenCV|YOLO(?:v\d+)?|Docker|Kubernetes|AWS|Azure|GCP|Git|GitHub|"
    r"PostgreSQL|MySQL|MongoDB|Redis|Power BI|Tableau|Excel|Linux|HTML|CSS|"
    r"LangChain|FAISS|Hugging Face|Transformers|REST API|RAG|LLM|NLP|Computer Vision)\b",
    re.I,
)

def _norm_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())

def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" \t•●▪◦‣*-")

def _unique(values: list[str]) -> list[str]:
    out, seen = [], set()
    for value in values:
        value = _clean_line(value)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out

def _find_header(line: str) -> str | None:
    candidate = line.strip().strip(":").strip()
    key = _norm_header(candidate)
    if key in HEADER_LOOKUP:
        return HEADER_LOOKUP[key]
    # Accept common numbered headers, e.g. "1. Projects"
    candidate = re.sub(r"^\d+[\).]\s*", "", candidate)
    return HEADER_LOOKUP.get(_norm_header(candidate))

def _split_sections(text: str) -> tuple[dict[str, list[str]], list[str]]:
    sections: dict[str, list[str]] = {"unclassified": []}
    current = "unclassified"
    lines = text.replace("\x00", " ").splitlines()
    recognized = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        header = _find_header(line)
        if header:
            current = header
            sections.setdefault(current, [])
            recognized.add(current)
            continue
        # Avoid mistaking ordinary short sentences for section titles.
        sections.setdefault(current, []).append(line)
    return sections, sorted(recognized)

def _entries(lines: list[str]) -> list[dict[str, Any]]:
    """Group bullets and following wrapped lines into evidence-backed entries."""
    entries: list[str] = []
    active = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        bullet = BULLET_RE.match(line)
        if bullet:
            if active:
                entries.append(active.strip())
            active = bullet.group(1).strip()
        elif active:
            # A new date/title-looking line often starts a separate entry.
            if DATE_RE.search(line) and len(active) > 45:
                entries.append(active.strip())
                active = line
            else:
                active += " " + line
        else:
            active = line
    if active:
        entries.append(active.strip())
    return [{"text": e, "evidence": e[:500]} for e in _unique(entries)]

def _extract_technologies(text: str) -> list[str]:
    return _unique([m.group(0) for m in TECH_RE.finditer(text)])

def _extract_project_fields(entry: str) -> dict[str, Any]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|[;\n]", entry) if s.strip()]
    tech = _extract_technologies(entry)
    contribution_patterns = re.compile(
        r"\b(develop(?:ed)?|implement(?:ed)?|design(?:ed)?|build(?:t)?|creat(?:ed)?|"
        r"train(?:ed)?|analy[sz](?:ed|ing)|deploy(?:ed)?|test(?:ed)?|integrat(?:ed)?|"
        r"lead|contribut(?:ed|ing)|responsible for)\b", re.I
    )
    outcome_patterns = re.compile(
        r"\b(achiev(?:ed|ing)|improv(?:ed|ement)|reduc(?:ed|tion)|increas(?:ed|e)|"
        r"accuracy|precision|recall|f1|mAP|score|users|downloads|deployed|published)\b",
        re.I
    )
    contribution = [s for s in sentences if contribution_patterns.search(s)]
    outcomes = [s for s in sentences if outcome_patterns.search(s)]
    return {
        "name_or_entry": entry[:180],
        "objective_or_problem": sentences[0] if sentences else "",
        "technologies": tech,
        "stated_contribution": contribution[:5],
        "documented_outcomes": outcomes[:4],
        "deployment_or_testing_evidence": [s for s in sentences if re.search(r"\b(deploy|test|production|hosted|live demo)\b", s, re.I)][:4],
        "evidence": entry[:700],
        "confidence": "medium" if len(sentences) > 1 else "low",
    }

def _extract_experience_fields(entry: str) -> dict[str, Any]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|[;\n]", entry) if s.strip()]
    dates = _unique(DATE_RE.findall(entry))
    return {
        "entry": entry[:220],
        "dates_detected": dates,
        "technologies": _extract_technologies(entry),
        "responsibilities_or_accomplishments": sentences[:6],
        "evidence": entry[:700],
        "confidence": "medium" if len(sentences) > 1 else "low",
    }

def analyze_resume_text(text: str) -> dict[str, Any]:
    """Return a stable, evidence-backed profile extracted from the entire text."""
    text = (text or "").strip()
    if not text:
        return {
            "sections": {}, "section_status": {}, "projects": [], "experience": [],
            "internships": [], "education": [], "certifications": [], "skills": [],
            "technologies": [], "achievements": [], "publications_research": [],
            "leadership_volunteering": [], "languages": [], "quality_notes": [
                "No usable resume text was available for full-resume analysis."
            ], "text_characters": 0,
        }
    raw_sections, recognized = _split_sections(text)
    normalized = {key: _entries(raw_sections.get(key, [])) for key in SECTION_ALIASES}
    all_lines = [line.strip() for line in text.splitlines() if line.strip()]
    # The resume parser may miss headings. Keep unclassified text visible as evidence.
    normalized["unclassified"] = _entries(raw_sections.get("unclassified", []))
    projects = [_extract_project_fields(x["text"]) for x in normalized["projects"]]
    experience = [_extract_experience_fields(x["text"]) for x in normalized["experience"]]
    internships = [_extract_experience_fields(x["text"]) for x in normalized["internships"]]
    section_status = {}
    for key, aliases in SECTION_ALIASES.items():
        if key in recognized and normalized[key]:
            section_status[key] = "extracted"
        elif key in recognized:
            section_status[key] = "heading_found_but_content_empty"
        else:
            section_status[key] = "not_detected"
    quality_notes = []
    if len(text) < 250:
        quality_notes.append("Very little text was extracted; results may be incomplete.")
    if not recognized:
        quality_notes.append("No recognized section headings were found; content is shown as unclassified where possible.")
    if section_status["projects"] == "not_detected":
        quality_notes.append("Projects section was not detected. This does not prove the resume has no projects.")
    if section_status["experience"] == "not_detected" and section_status["internships"] == "not_detected":
        quality_notes.append("Experience/internship sections were not detected; check whether headings were parsed correctly.")
    if section_status["certifications"] == "not_detected":
        quality_notes.append("Certifications section was not detected. Completion status cannot be inferred.")
    skills_text = " ".join(x["text"] for x in normalized["skills"])
    technologies = _extract_technologies(text)
    # Keep skill section evidence separate from technologies spotted elsewhere.
    skills = _unique([x["text"] for x in normalized["skills"]])
    return {
        "sections": normalized,
        "section_status": section_status,
        "projects": projects,
        "experience": experience,
        "internships": internships,
        "education": normalized["education"],
        "certifications": normalized["certifications"],
        "skills": skills,
        "technologies": technologies,
        "achievements": normalized["achievements"],
        "publications_research": normalized["publications_research"],
        "leadership_volunteering": normalized["leadership_volunteering"],
        "languages": normalized["languages"],
        "summary": normalized["summary"],
        "quality_notes": quality_notes,
        "text_characters": len(text),
        "recognized_section_count": len(recognized),
        "unclassified_entry_count": len(normalized["unclassified"]),
    }
