"""
======================================================================
AI RESUME ANALYZER - LIVE JOB POSTINGS
======================================================================

Fetches CURRENT job postings from official, documented job APIs and
returns them in the same column layout as ``data/jobs/all_job_post.csv``
so the rest of the project (job_ranker -> matching_core.MatchScorer ->
skill gaps -> explanations) works on them unchanged.

Providers (official APIs only; scraping LinkedIn/Indeed etc. violates
their terms and is deliberately not supported):

  Adzuna    https://developer.adzuna.com/docs/search
            Needs a free key pair in .env:
                ADZUNA_APP_ID=...
                ADZUNA_APP_KEY=...
            Broad coverage across fields and countries (including "in").
            LIMITATION (stated in Adzuna's docs): only a SNIPPET of each
            description is returned, so skill evidence is thinner.
            Terms: link users to ``redirect_url`` and credit Adzuna.

  Remotive  https://github.com/remotive-com/remote-jobs-api
            No key. Remote jobs only, weighted toward tech. Full HTML
            descriptions. Terms: link back to the Remotive URL and name
            Remotive as the source; their guidance is at most ~4 fetches a
            day and more than 2 requests a minute are blocked - so the
            whole feed is cached on disk and filtered locally.

Safety rules implemented here:
  * secrets (app_id / app_key) never appear in logs, cache files or error
    messages (``redact_secrets``);
  * every network failure becomes a short, user-presentable
    ``LiveJobsError``; if an older cached answer exists it is used instead
    and the user is told;
  * responses are cached on disk (data/cache/live) to save API quota.
======================================================================
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIVE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "live"

REQUEST_TIMEOUT = 20

ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"

ADZUNA_CACHE_TTL = 12 * 3600
REMOTIVE_CACHE_TTL = 6 * 3600

USER_AGENT = "AIResumeAnalyzer/1.0 (student project)"

MAX_QUERIES = 4
MAX_QUERY_CHARS = 100

# Countries documented for Adzuna. The API itself is the authority: an
# unsupported code comes back as an error that is reported to the user.
ADZUNA_COUNTRY_HINTS = ["in", "us", "gb", "au", "ca", "de", "fr", "nl", "pl", "br", "za"]

# Same layout as data/jobs/all_job_post.csv, plus live-only columns.
JOB_COLUMNS = [
    "job_id",
    "category",
    "job_title",
    "job_description",
    "job_skill_set",
    "company",
    "location",
    "url",
    "source",
    "posted",
    "description_truncated",
    "salary",
    "provider_category",
]

ATTRIBUTION = {
    "Adzuna": ("Jobs by Adzuna", "https://www.adzuna.com"),
    "Remotive": ("Remote jobs from Remotive", "https://remotive.com"),
}

_QUERY_STOPWORDS = {"and", "or", "the", "of", "for", "in", "at", "to", "a", "an", "with"}


class LiveJobsError(RuntimeError):
    """A failure whose message is safe and sensible to show to the user."""


# ======================================================================
# SECRETS
# ======================================================================

_SECRET_PARAM = re.compile(
    r"(app_key|app_id|api_key|apikey|key|token|secret)=([^&\s'\"]+)",
    re.IGNORECASE,
)


def redact_secrets(text: Any, secrets: Sequence[str] = ()) -> str:
    """Remove credentials from any text before it is logged or shown."""
    value = str(text)

    value = _SECRET_PARAM.sub(lambda m: f"{m.group(1)}=***", value)

    for secret in secrets:
        if secret and len(secret) >= 4:
            value = value.replace(secret, "***")

    return value


# ======================================================================
# TEXT CLEANING
# ======================================================================

class _TextExtractor(HTMLParser):

    BLOCK = {
        "p", "br", "li", "ul", "ol", "div", "tr", "table",
        "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
    }
    SKIP = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n- " if tag == "li" else "\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(value: Any) -> str:
    """HTML (or plain text with entities) -> clean readable text."""
    if value is None:
        return ""

    text = str(value)

    if "<" in text and ">" in text:
        extractor = _TextExtractor()

        try:
            extractor.feed(text)
            extractor.close()
            text = "".join(extractor.parts)
        except Exception:  # malformed markup: fall back to tag removal
            text = re.sub(r"<[^>]+>", " ", text)

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_queries(queries: Sequence[str]) -> List[str]:
    """Distinct, trimmed, length-limited search phrases (order preserved)."""
    cleaned: List[str] = []
    seen = set()

    for query in queries:
        text = re.sub(r"\s+", " ", strip_html(query)).strip()[:MAX_QUERY_CHARS]

        if len(text) < 2 or text.lower() in seen:
            continue

        seen.add(text.lower())
        cleaned.append(text)

    return cleaned[:MAX_QUERIES]


def _query_tokens(queries: Sequence[str]) -> List[str]:
    tokens: List[str] = []

    for query in queries:
        for token in re.findall(r"[a-z0-9+#]+", query.lower()):
            if len(token) >= 3 and token not in _QUERY_STOPWORDS and token not in tokens:
                tokens.append(token)

    return tokens


# ======================================================================
# HTTP + DISK CACHE
# ======================================================================

def _http_get(
    url: str,
    params: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Any:
    """GET -> parsed JSON. Every failure becomes a safe ``LiveJobsError``."""
    import requests

    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        raise LiveJobsError("the request timed out") from None
    except requests.exceptions.RequestException as exc:
        # Only the exception TYPE: its text can contain the full URL (and keys).
        raise LiveJobsError(f"network error ({type(exc).__name__})") from None

    status = response.status_code

    if status in (401, 403):
        raise LiveJobsError("the service rejected the credentials")
    if status == 404:
        raise LiveJobsError("not found (check the country code and endpoint)")
    if status == 429:
        raise LiveJobsError("the service's rate limit was reached")
    if status >= 500:
        raise LiveJobsError("the service is temporarily unavailable")
    if status != 200:
        raise LiveJobsError(f"unexpected response (HTTP {status})")

    try:
        return response.json()
    except ValueError:
        raise LiveJobsError("the service returned an unreadable response") from None


def _cache_file(cache_dir: Path, provider: str, key_parts: Mapping[str, Any]) -> Path:
    digest = hashlib.sha1(
        json.dumps(key_parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]

    return cache_dir / f"{provider}_{digest}.json"


def _read_cache(path: Path) -> Optional[dict]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(record, dict) and "fetched_at" in record and "payload" in record:
            return record
    except (OSError, ValueError):
        pass

    return None


def _write_cache(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write live-jobs cache: %s", exc)


def _age_text(seconds: float) -> str:
    if seconds < 90:
        return "moments"
    if seconds < 5400:
        return f"{int(seconds // 60)} minutes"
    return f"{seconds / 3600:.0f} hours"


def _cached_payload(
    path: Path,
    ttl: float,
    fetch: Callable[[], Any],
    notes: List[str],
    label: str,
    now: Callable[[], float] = time.time,
) -> Any:
    """Fresh cache -> use it. Otherwise fetch; on failure fall back to a stale cache."""
    cached = _read_cache(path)

    if cached is not None and now() - cached["fetched_at"] <= ttl:
        return cached["payload"]

    try:
        payload = fetch()
    except LiveJobsError as exc:

        if cached is not None:
            notes.append(
                f"{label}: live request failed ({exc}); showing results saved "
                f"{_age_text(now() - cached['fetched_at'])} ago."
            )
            return cached["payload"]

        raise

    _write_cache(path, payload)

    return payload


# ======================================================================
# PROVIDERS
# ======================================================================

def adzuna_credentials(env: Optional[Mapping[str, str]] = None) -> Optional[tuple]:
    env = os.environ if env is None else env

    app_id = (env.get("ADZUNA_APP_ID") or "").strip()
    app_key = (env.get("ADZUNA_APP_KEY") or "").strip()

    return (app_id, app_key) if app_id and app_key else None


def provider_status(env: Optional[Mapping[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """Which providers can be used right now, and why not."""
    return {
        "Adzuna": {
            "available": adzuna_credentials(env) is not None,
            "note": "broad coverage, all fields"
            if adzuna_credentials(env)
            else "add ADZUNA_APP_ID and ADZUNA_APP_KEY to .env (free key at developer.adzuna.com)",
        },
        "Remotive": {
            "available": True,
            "note": "no key needed; remote jobs only, mostly tech",
        },
    }


def _salary_text(low: Any, high: Any) -> str:
    try:
        low_value = float(low) if low not in (None, "") else None
        high_value = float(high) if high not in (None, "") else None
    except (TypeError, ValueError):
        return ""

    if low_value and high_value and round(low_value) != round(high_value):
        return f"{low_value:,.0f} - {high_value:,.0f}"

    value = low_value or high_value

    return f"{value:,.0f}" if value else ""


def normalize_adzuna(item: Mapping[str, Any]) -> Optional[dict]:
    """One Adzuna result -> the project's job row (None if unusable)."""
    title = strip_html(item.get("title"))
    url = str(item.get("redirect_url") or "").strip()

    if not title or not url:
        return None

    company = item.get("company") or {}
    location = item.get("location") or {}
    category = item.get("category") or {}

    return {
        "job_id": f"adzuna:{item.get('id', url)}",
        "category": "",
        "job_title": title,
        "job_description": strip_html(item.get("description")),
        "job_skill_set": "",
        "company": strip_html(company.get("display_name") if isinstance(company, Mapping) else ""),
        "location": strip_html(location.get("display_name") if isinstance(location, Mapping) else ""),
        "url": url,
        "source": "Adzuna",
        "posted": str(item.get("created") or ""),
        "description_truncated": True,  # Adzuna returns a snippet only
        "salary": _salary_text(item.get("salary_min"), item.get("salary_max")),
        "provider_category": strip_html(category.get("label") if isinstance(category, Mapping) else ""),
    }


def normalize_remotive(item: Mapping[str, Any]) -> Optional[dict]:
    """One Remotive listing -> the project's job row (None if unusable)."""
    title = strip_html(item.get("title"))
    url = str(item.get("url") or "").strip()

    if not title or not url:
        return None

    return {
        "job_id": f"remotive:{item.get('id', url)}",
        "category": "",
        "job_title": title,
        "job_description": strip_html(item.get("description")),
        "job_skill_set": "",
        "company": strip_html(item.get("company_name")),
        "location": strip_html(item.get("candidate_required_location")) or "Remote",
        "url": url,
        "source": "Remotive",
        "posted": str(item.get("publication_date") or ""),
        "description_truncated": False,
        "salary": strip_html(item.get("salary")),
        "provider_category": strip_html(item.get("category")),
    }


def search_adzuna(
    queries: Sequence[str],
    *,
    country: str,
    where: str,
    credentials: tuple,
    per_page: int,
    pages: int,
    http_get: Callable,
    cache_dir: Path,
    notes: List[str],
) -> List[dict]:
    country = re.sub(r"[^a-z]", "", (country or "").lower())

    if len(country) != 2:
        raise LiveJobsError("Adzuna needs a two-letter country code (for example 'in' or 'us')")

    app_id, app_key = credentials
    rows: List[dict] = []
    failures: List[str] = []

    for query in queries:
        for page in range(1, max(1, pages) + 1):

            params = {
                "app_id": app_id,
                "app_key": app_key,
                "what": query,
                "results_per_page": max(1, min(int(per_page), 50)),
                "content-type": "application/json",
            }

            if where:
                params["where"] = where

            url = f"{ADZUNA_BASE}/{country}/search/{page}"

            # The cache key never contains the credentials.
            path = _cache_file(
                cache_dir,
                "adzuna",
                {"country": country, "page": page, "what": query, "where": where,
                 "per_page": params["results_per_page"]},
            )

            try:
                payload = _cached_payload(
                    path,
                    ADZUNA_CACHE_TTL,
                    lambda: http_get(url, params, None, REQUEST_TIMEOUT),
                    notes,
                    f"Adzuna '{query}'",
                )
            except LiveJobsError as exc:
                failures.append(redact_secrets(exc, credentials))
                break  # later pages of this query would fail the same way

            results = payload.get("results") if isinstance(payload, dict) else None

            if not isinstance(results, list):
                failures.append("unexpected response format")
                break

            for item in results:
                if isinstance(item, Mapping):
                    row = normalize_adzuna(item)
                    if row:
                        rows.append(row)

            if len(results) < params["results_per_page"]:
                break  # no further pages

    if not rows and failures:
        raise LiveJobsError("; ".join(sorted(set(failures))))

    if failures:
        notes.append("Adzuna: some searches failed (" + "; ".join(sorted(set(failures))) + ").")

    return rows


def search_remotive(
    queries: Sequence[str],
    *,
    http_get: Callable,
    cache_dir: Path,
    notes: List[str],
    max_jobs: int = 200,
) -> List[dict]:
    """
    Whole Remotive feed (cached for hours, as their terms advise), filtered
    locally to postings whose title or category mentions a query word.
    """
    path = _cache_file(cache_dir, "remotive", {"feed": "all"})

    payload = _cached_payload(
        path,
        REMOTIVE_CACHE_TTL,
        lambda: http_get(REMOTIVE_URL, None, None, REQUEST_TIMEOUT),
        notes,
        "Remotive",
    )

    listings = payload.get("jobs") if isinstance(payload, dict) else None

    if not isinstance(listings, list):
        raise LiveJobsError("unexpected response format")

    rows = [row for item in listings if isinstance(item, Mapping)
            for row in [normalize_remotive(item)] if row]

    tokens = _query_tokens(queries)

    if tokens:
        rows = [
            row for row in rows
            if any(
                token in f"{row['job_title']} {row['provider_category']}".lower()
                for token in tokens
            )
        ]

    # The feed is newest-first; keep the freshest.
    return rows[:max_jobs]


# ======================================================================
# ORCHESTRATION
# ======================================================================

@dataclass
class LiveJobsResult:
    jobs: pd.DataFrame
    notes: List[str] = field(default_factory=list)
    providers_used: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)


def _role_key(row: dict) -> str:
    return "|".join(
        re.sub(r"\W+", " ", str(row[column]).lower()).strip()
        for column in ("job_title", "company", "location")
    )


def _dedupe(rows: List[dict]) -> List[dict]:
    """
    Remove repeated postings without merging distinct vacancies:
      * the same URL is always the same posting;
      * the same title + employer + location from DIFFERENT providers is the
        same job syndicated twice. Within one provider it is kept, because an
        employer can have several real openings with the same title.
    """
    seen_urls: set = set()
    role_sources: Dict[str, set] = {}
    unique: List[dict] = []

    for row in rows:

        url_key = row["url"].split("?")[0].lower()

        if url_key in seen_urls:
            continue

        role = _role_key(row)

        sources = role_sources.get(role)

        if sources is not None and row["source"] not in sources:
            continue

        seen_urls.add(url_key)
        role_sources.setdefault(role, set()).add(row["source"])
        unique.append(row)

    return unique


def fetch_live_jobs(
    queries: Sequence[str],
    *,
    providers: Optional[Sequence[str]] = None,
    country: str = "in",
    where: str = "",
    per_page: int = 50,
    pages: int = 1,
    http_get: Optional[Callable] = None,
    cache_dir: Path = LIVE_CACHE_DIR,
    env: Optional[Mapping[str, str]] = None,
) -> LiveJobsResult:
    """
    Current postings for the search phrases, from every requested provider
    that is usable. Raises ``LiveJobsError`` (safe to show) when nothing
    could be fetched at all; partial failures become ``notes``/``errors``.
    """
    http = http_get or _http_get

    phrases = clean_queries(queries)

    if not phrases:
        raise LiveJobsError("no job title or role was available to search for")

    status = provider_status(env)

    requested = [p for p in (providers or status.keys()) if p in status]

    notes: List[str] = []
    errors: Dict[str, str] = {}
    rows: List[dict] = []
    used: List[str] = []

    for provider in requested:

        if not status[provider]["available"]:
            errors[provider] = f"not configured - {status[provider]['note']}"
            continue

        credentials = adzuna_credentials(env) if provider == "Adzuna" else ()

        try:
            if provider == "Adzuna":
                found = search_adzuna(
                    phrases,
                    country=country,
                    where=where.strip(),
                    credentials=credentials,
                    per_page=per_page,
                    pages=pages,
                    http_get=http,
                    cache_dir=cache_dir,
                    notes=notes,
                )
            else:
                found = search_remotive(
                    phrases, http_get=http, cache_dir=cache_dir, notes=notes
                )
        except LiveJobsError as exc:
            errors[provider] = redact_secrets(exc, credentials or ())
            continue

        if found:
            rows.extend(found)
            used.append(provider)
        else:
            notes.append(f"{provider}: no postings matched {', '.join(phrases)}.")

    rows = _dedupe(rows)

    if not rows:
        parts = [f"{name}: {message}" for name, message in errors.items()]
        parts += [note for note in notes if "no postings matched" in note]
        detail = "; ".join(parts)
        raise LiveJobsError(
            "No live postings could be retrieved"
            + (f" ({detail})" if detail else f" for: {', '.join(phrases)}")
        )

    if "Adzuna" in used:
        notes.append(
            "Adzuna provides only a short snippet of each posting, so skill "
            "evidence for those jobs can be limited - open the posting for the "
            "full requirements."
        )

    frame = pd.DataFrame(rows, columns=JOB_COLUMNS).reset_index(drop=True)

    return LiveJobsResult(jobs=frame, notes=notes, providers_used=used, errors=errors)
