"""Tests for src/live_jobs.py (no network: HTTP is mocked with the providers' documented formats)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_jobs as lj  # noqa: E402


# ---- payloads in the formats shown in the official documentation ------------

def adzuna_item(job_id="129698749", title="Javascript Developer", url="http://adzuna.co.uk/jobs/land/ad/1"):
    return {
        "salary_min": 50000,
        "location": {"__CLASS__": "Adzuna::API::Response::Location",
                     "area": ["UK", "South East England"], "display_name": "Marlow, Buckinghamshire"},
        "salary_is_predicted": 0,
        "description": "JavaScript Developer Corporate &hellip; experience with <strong>python</strong>",
        "created": "2013-11-08T18:07:39Z",
        "redirect_url": url,
        "title": title,
        "category": {"label": "IT Jobs", "tag": "it-jobs"},
        "id": job_id,
        "salary_max": 55000,
        "company": {"display_name": "Corporate Project Solutions"},
        "contract_type": "permanent",
    }


def remotive_item(job_id=123, title="Lead Developer", category="Software Development", url=None):
    return {
        "id": job_id,
        "url": url or f"https://remotive.com/remote-jobs/software-dev/lead-developer-{job_id}",
        "title": title,
        "company_name": "Remotive",
        "category": category,
        "job_type": "full_time",
        "publication_date": "2020-02-15T10:23:26",
        "candidate_required_location": "Worldwide",
        "salary": "$40,000 - $50,000",
        "description": "<p>We use <b>Python</b> and SQL.</p><ul><li>Build APIs</li><li>Write tests</li></ul><script>x()</script>",
    }


class FakeHTTP:
    """Records calls; answers by URL."""

    def __init__(self, adzuna=None, remotive=None, fail=None):
        self.calls = []
        self.adzuna = adzuna if adzuna is not None else {"results": [adzuna_item()]}
        self.remotive = remotive if remotive is not None else {"job-count": 1, "jobs": [remotive_item()]}
        self.fail = fail

    def __call__(self, url, params=None, headers=None, timeout=20):
        self.calls.append((url, dict(params or {})))
        if self.fail:
            raise self.fail
        return self.adzuna if "adzuna" in url else self.remotive


ENV = {"ADZUNA_APP_ID": "id12345", "ADZUNA_APP_KEY": "SECRETKEY999"}


# ---- text cleaning ----------------------------------------------------------

def test_strip_html_handles_lists_scripts_and_entities():
    text = lj.strip_html(remotive_item()["description"])
    assert "Python" in text and "SQL" in text
    assert "- Build APIs" in text and "- Write tests" in text
    assert "x()" not in text  # script content removed
    assert "<" not in text


def test_strip_html_plain_text_with_entities_and_highlight_tags():
    assert lj.strip_html("<strong>Python</strong> developer") == "Python developer"
    assert lj.strip_html("Corporate &hellip; role") == "Corporate … role"
    assert lj.strip_html(None) == ""


def test_redact_secrets_in_urls_and_values():
    url = "https://api.adzuna.com/x?app_id=abc12&app_key=SECRETKEY999&what=python"
    redacted = lj.redact_secrets(url, ["SECRETKEY999"])
    assert "SECRETKEY999" not in redacted and "abc12" not in redacted
    assert "what=python" in redacted
    assert "SECRETKEY999" not in lj.redact_secrets("failed with SECRETKEY999 inside", ["SECRETKEY999"])


def test_clean_queries_dedupes_limits_and_trims():
    queries = ["  Software   Developer ", "software developer", "x", "<b>Data Analyst</b>",
               "Chef", "Pilot", "Nurse"]
    cleaned = lj.clean_queries(queries)
    assert cleaned[:2] == ["Software Developer", "Data Analyst"]
    assert len(cleaned) == lj.MAX_QUERIES
    assert "x" not in cleaned


# ---- normalisation ----------------------------------------------------------

def test_normalize_adzuna_follows_documented_fields():
    row = lj.normalize_adzuna(adzuna_item())
    assert row["job_id"] == "adzuna:129698749"
    assert row["job_title"] == "Javascript Developer"
    assert row["company"] == "Corporate Project Solutions"
    assert row["location"] == "Marlow, Buckinghamshire"
    assert row["url"].startswith("http://adzuna.co.uk")
    assert row["source"] == "Adzuna"
    assert row["description_truncated"] is True
    assert row["salary"] == "50,000 - 55,000"
    assert "python" in row["job_description"] and "<" not in row["job_description"]
    assert set(row) == set(lj.JOB_COLUMNS)


def test_normalize_skips_unusable_items():
    assert lj.normalize_adzuna({"title": "No link"}) is None
    assert lj.normalize_adzuna({"redirect_url": "http://x", "title": ""}) is None
    assert lj.normalize_remotive({"title": "No url"}) is None


def test_normalize_remotive_follows_documented_fields():
    row = lj.normalize_remotive(remotive_item())
    assert row["source"] == "Remotive" and row["description_truncated"] is False
    assert row["location"] == "Worldwide" and row["company"] == "Remotive"
    assert row["provider_category"] == "Software Development"
    assert lj.normalize_remotive({**remotive_item(), "candidate_required_location": ""})["location"] == "Remote"


# ---- providers --------------------------------------------------------------

def test_adzuna_request_shape_and_result(tmp_path):
    http = FakeHTTP()
    result = lj.fetch_live_jobs(
        ["python developer"], providers=["Adzuna"], country="IN", where="Delhi",
        http_get=http, cache_dir=tmp_path, env=ENV,
    )
    url, params = http.calls[0]
    assert url == "https://api.adzuna.com/v1/api/jobs/in/search/1"
    assert params["what"] == "python developer" and params["where"] == "Delhi"
    assert params["results_per_page"] == 50 and params["app_id"] == "id12345"
    assert list(result.jobs.columns) == lj.JOB_COLUMNS
    assert len(result.jobs) == 1 and result.providers_used == ["Adzuna"]
    assert any("snippet" in note for note in result.notes)


def test_credentials_never_written_to_cache(tmp_path):
    lj.fetch_live_jobs(["python"], providers=["Adzuna"], http_get=FakeHTTP(), cache_dir=tmp_path, env=ENV)
    for path in tmp_path.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "SECRETKEY999" not in text and "id12345" not in text


def test_adzuna_second_search_is_served_from_cache(tmp_path):
    http = FakeHTTP()
    kwargs = dict(providers=["Adzuna"], http_get=http, cache_dir=tmp_path, env=ENV)
    lj.fetch_live_jobs(["python"], **kwargs)
    lj.fetch_live_jobs(["python"], **kwargs)
    assert len(http.calls) == 1


def test_adzuna_pages_until_results_run_out(tmp_path):
    full = {"results": [adzuna_item(job_id=str(i), url=f"http://a/{i}") for i in range(2)]}
    http = FakeHTTP(adzuna=full)
    result = lj.fetch_live_jobs(
        ["python"], providers=["Adzuna"], per_page=2, pages=3,
        http_get=http, cache_dir=tmp_path, env=ENV,
    )
    assert [call[0][-1] for call in http.calls] == ["1", "2", "3"]
    assert len(result.jobs) == 2  # identical pages dedupe by URL to the same two jobs


def test_distinct_vacancies_from_one_provider_are_not_merged(tmp_path):
    # same title, employer and location, different postings -> both are kept
    items = [adzuna_item(job_id="1", url="http://a/1"), adzuna_item(job_id="2", url="http://a/2")]
    result = lj.fetch_live_jobs(["developer"], providers=["Adzuna"],
                                http_get=FakeHTTP(adzuna={"results": items}), cache_dir=tmp_path, env=ENV)
    assert len(result.jobs) == 2


def test_same_job_syndicated_across_providers_is_merged(tmp_path):
    adz = adzuna_item(title="Lead Developer", url="http://adzuna/9")
    adz["company"] = {"display_name": "Remotive"}
    adz["location"] = {"display_name": "Worldwide"}
    http = FakeHTTP(adzuna={"results": [adz]}, remotive={"jobs": [remotive_item(1, "Lead Developer")]})
    result = lj.fetch_live_jobs(["developer"], http_get=http, cache_dir=tmp_path, env=ENV)
    assert len(result.jobs) == 1


def test_adzuna_requires_valid_country(tmp_path):
    with pytest.raises(lj.LiveJobsError, match="two-letter country"):
        lj.fetch_live_jobs(["python"], providers=["Adzuna"], country="india",
                           http_get=FakeHTTP(), cache_dir=tmp_path, env=ENV)


def test_remotive_feed_cached_once_and_filtered_locally(tmp_path):
    feed = {"job-count": 3, "jobs": [
        remotive_item(1, "Python Developer"),
        remotive_item(2, "Customer Support Agent", "Customer Service"),
        remotive_item(3, "Sales Lead", "Sales"),
    ]}
    http = FakeHTTP(remotive=feed)
    kwargs = dict(providers=["Remotive"], http_get=http, cache_dir=tmp_path, env={})

    first = lj.fetch_live_jobs(["python developer"], **kwargs)
    second = lj.fetch_live_jobs(["sales"], **kwargs)

    assert len(http.calls) == 1  # one feed fetch serves both searches (rate limits)
    assert http.calls[0][1] == {}  # no query sent to the API
    assert list(first.jobs["job_title"]) == ["Python Developer"]
    assert list(second.jobs["job_title"]) == ["Sales Lead"]


def test_providers_are_combined_and_duplicates_removed(tmp_path):
    same_url = "https://remotive.com/remote-jobs/software-dev/lead-developer-1"
    http = FakeHTTP(
        adzuna={"results": [adzuna_item(url="http://adzuna/1"), adzuna_item(job_id="2", title="Lead Developer", url=same_url)]},
        remotive={"jobs": [remotive_item(1, "Lead Developer", url=same_url)]},
    )
    result = lj.fetch_live_jobs(["developer"], http_get=http, cache_dir=tmp_path, env=ENV)
    assert sorted(result.providers_used) == ["Adzuna", "Remotive"]
    assert result.jobs["url"].str.lower().is_unique


def test_missing_adzuna_key_is_reported_not_crashed(tmp_path):
    http = FakeHTTP()
    result = lj.fetch_live_jobs(["developer"], http_get=http, cache_dir=tmp_path, env={})
    assert result.providers_used == ["Remotive"]
    assert "not configured" in result.errors["Adzuna"]
    with pytest.raises(lj.LiveJobsError):
        lj.fetch_live_jobs(["developer"], providers=["Adzuna"], http_get=http, cache_dir=tmp_path, env={})


def test_total_failure_message_mentions_every_provider(tmp_path):
    with pytest.raises(lj.LiveJobsError) as caught:
        lj.fetch_live_jobs(["astronaut"], http_get=FakeHTTP(), cache_dir=tmp_path, env={})
    message = str(caught.value)
    assert "Adzuna: not configured" in message and "Remotive: no postings matched" in message


def test_no_queries_is_a_clean_error(tmp_path):
    with pytest.raises(lj.LiveJobsError, match="no job title"):
        lj.fetch_live_jobs(["", " "], http_get=FakeHTTP(), cache_dir=tmp_path, env=ENV)


# ---- failure handling -------------------------------------------------------

def test_error_messages_never_contain_the_key(tmp_path):
    http = FakeHTTP(fail=lj.LiveJobsError("boom https://x?app_key=SECRETKEY999&app_id=id12345"))
    with pytest.raises(lj.LiveJobsError) as caught:
        lj.fetch_live_jobs(["python"], providers=["Adzuna"], http_get=http, cache_dir=tmp_path, env=ENV)
    assert "SECRETKEY999" not in str(caught.value) and "id12345" not in str(caught.value)


def test_stale_cache_is_used_when_the_service_fails(tmp_path):
    ok = FakeHTTP()
    kwargs = dict(providers=["Adzuna"], cache_dir=tmp_path, env=ENV)
    lj.fetch_live_jobs(["python"], http_get=ok, **kwargs)

    # age the cache beyond its TTL
    for path in tmp_path.iterdir():
        record = json.loads(path.read_text(encoding="utf-8"))
        record["fetched_at"] = time.time() - 3 * lj.ADZUNA_CACHE_TTL
        path.write_text(json.dumps(record), encoding="utf-8")

    down = FakeHTTP(fail=lj.LiveJobsError("the service is temporarily unavailable"))
    result = lj.fetch_live_jobs(["python"], http_get=down, **kwargs)

    assert len(result.jobs) == 1
    assert any("showing results saved" in note for note in result.notes)


def test_total_failure_raises_a_safe_error_when_nothing_is_cached(tmp_path):
    down = FakeHTTP(fail=lj.LiveJobsError("the service's rate limit was reached"))
    with pytest.raises(lj.LiveJobsError, match="rate limit"):
        lj.fetch_live_jobs(["python"], providers=["Adzuna"], http_get=down, cache_dir=tmp_path, env=ENV)


def test_malformed_payload_is_handled(tmp_path):
    with pytest.raises(lj.LiveJobsError):
        lj.fetch_live_jobs(["python"], providers=["Adzuna"], http_get=FakeHTTP(adzuna={"oops": 1}),
                           cache_dir=tmp_path, env=ENV)
    with pytest.raises(lj.LiveJobsError):
        lj.fetch_live_jobs(["python"], providers=["Remotive"], http_get=FakeHTTP(remotive=["bad"]),
                           cache_dir=tmp_path, env={})


# ---- the real HTTP helper: status-code mapping ------------------------------

class FakeResponse:
    def __init__(self, status, body=None, bad_json=False):
        self.status_code, self._body, self._bad = status, body, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


@pytest.mark.parametrize(
    "status, expected",
    [(401, "rejected the credentials"), (403, "rejected the credentials"), (404, "not found"),
     (429, "rate limit"), (503, "temporarily unavailable"), (418, "HTTP 418")],
)
def test_http_status_codes_become_safe_messages(monkeypatch, status, expected):
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(status))
    with pytest.raises(lj.LiveJobsError, match=expected):
        lj._http_get("https://api.adzuna.com/x", {"app_key": "SECRETKEY999"})


def test_http_timeout_and_network_errors_do_not_leak_urls(monkeypatch):
    requests = pytest.importorskip("requests")

    def timeout(*a, **k):
        raise requests.exceptions.Timeout("GET https://x?app_key=SECRETKEY999 timed out")

    monkeypatch.setattr(requests, "get", timeout)
    with pytest.raises(lj.LiveJobsError) as caught:
        lj._http_get("https://api.adzuna.com/x")
    assert "SECRETKEY999" not in str(caught.value) and "timed out" in str(caught.value)

    def refused(*a, **k):
        raise requests.exceptions.ConnectionError("https://x?app_key=SECRETKEY999")

    monkeypatch.setattr(requests, "get", refused)
    with pytest.raises(lj.LiveJobsError) as caught:
        lj._http_get("https://api.adzuna.com/x")
    assert "SECRETKEY999" not in str(caught.value) and "ConnectionError" in str(caught.value)


def test_http_success_and_bad_json(monkeypatch):
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, {"results": []}))
    assert lj._http_get("https://x") == {"results": []}
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, bad_json=True))
    with pytest.raises(lj.LiveJobsError, match="unreadable"):
        lj._http_get("https://x")
