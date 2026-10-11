# AI Resume Analyzer

This repository builds a resume-to-job retrieval and ranking pipeline, an explainability CLI, and a Streamlit application. It uses resume/job text similarity, lexical overlap, and ESCO skill evidence. The rankings are **not predictions of hiring success**.

## Data required

The source ZIP does not include source datasets. Place these files in the project before running the real-data pipeline:

- `data/resume_dataset/Resume/Resume.csv`
- `data/jobs/all_job_post.csv`
- `data/external/skills_en.csv`
- PDF resumes under the data folders expected by `src/03_resume_parser.py`
- Any other ESCO source CSVs used by your local data-cleaning configuration

The first three paths are checked by the runner. Each step checks its own intermediate input and reports the expected path if a required file is missing.

## Run the pipeline

From the project root:

```bash
python -m pip install -r requirements.txt
python src/run_pipeline.py --dry-run
python src/run_pipeline.py
```

Optional proxy-model training runs Step 10b before Step 10:

```bash
python src/run_pipeline.py --with-model
```

The model target is a same-category proxy, not a human hiring label. `--force-model` explicitly bypasses Step 10b's model-quality gate and should only be used deliberately. To also run the final recommendation materialization step:

```bash
python src/run_pipeline.py --with-final-recommendations
```

## Human relevance review

Until independent human judgments validate a scorer version, the app labels scores as **experimental similarity** and does not describe a result as suitable for a person. Export the review template from Step 10's ranking output:

```bash
python src/11b_evaluate_human_judgments.py --export data/evaluation/step10_matching_predictions.csv
```

The exported CSV includes source resume text and job descriptions when the Step 4 text tables are available; use those contents, not just scores or titles, to judge relevance. Treat the CSV as sensitive candidate data and share it only with authorized reviewers. Have at least two independent reviewers label `human_relevance` as `0` (not relevant), `1` (partly/potentially relevant), or `2` (clearly relevant) and fill `reviewer_id`. Each reviewer may label each resume-job pair only once. Merge the reviewers' annotated rows into one CSV. Evaluate the merged CSV:

```bash
python src/11b_evaluate_human_judgments.py --evaluate data/evaluation/human_judgment_template.csv
```

Suitability wording remains disabled unless the sample-size, reviewer, coverage, relevance-metric, and scorer-fingerprint gates all pass **and** the operator supplies explicit sign-off:

```bash
python src/11b_evaluate_human_judgments.py --evaluate data/evaluation/human_judgment_template.csv --approve-suitability-wording
```

Approval is tied to the active scorer fingerprint. Changes to scoring or retrieval code, or to the loaded model, require a fresh Step 10 run and new judgments.

## Verify

```bash
# Syntax/source sanity checks
python src/verify_pipeline.py --mode source

# Full offline test suite; uses synthetic fixtures, not the real datasets
python src/verify_pipeline.py --mode tests

# Check live input files and output schemas after a real-data run
python src/verify_pipeline.py --mode deployment
```

The deployment verifier checks file presence, schema, and current scorer metadata. It does not establish recommendation quality, fairness, or hiring validity. Those require human judgment and suitable evaluation data.

## Configuration and secrets

Copy `.env.example` to `.env` only if environment-based configuration is needed, and add your own secrets locally. **Never commit or share `.env` or API credentials.** The project archive used for delivery excludes `.env`, virtual environments, generated datasets, and embedding/model caches.
