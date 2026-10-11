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
python src/run_pipeline.py --model auto
```

`--model auto` reuses an existing model if one is installed. Choose explicitly when changing that behavior: `--model train` runs Step 10b; `--model omit` moves an existing model aside (rather than deleting it) and uses the formula scorer. Step 11's automatic diagnostics run after Step 10 in all cases. The older `--with-model` flag remains as an alias for `--model train`. The legacy `--with-final-recommendations` option is accepted for compatibility; Step 12 runs during every full pipeline execution.

```bash
python src/run_pipeline.py --model train
python src/run_pipeline.py --model omit
```

The model target is a same-category proxy, not a human hiring label. `--force-model` explicitly bypasses Step 10b's model-quality gate and should only be used deliberately. To require successful model installation, add `--require-model`; to also run the final recommendation materialization step:

```bash
python src/run_pipeline.py --model auto --with-final-recommendations --verify-deployment
```

`--verify-deployment` explicitly requests the parity check after Step 12 (verification also runs by default unless `--skip-verify` is supplied). The runner records each command, exit code, duration, model decision, active scorer fingerprint, and claim metadata in `reports/pipeline_run_report.json`.

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

### Optional deeper pooled-relevance study

The added `src/11b_human_relevance_judgments.py` workflow evaluates the ranking formula, trained model, semantic-only, skill-only, and category-proxy baselines over a pooled candidate sample. It reports per-resume Spearman correlation, nDCG/precision, reviewer agreement, bootstrap comparisons, and whether the same-category training proxy tracks human relevance.

```bash
# Export a shuffled sample of candidate pairs from the Step 9 dataset
python src/11b_human_relevance_judgments.py export --resumes 40
# Fill relevance (0=not relevant through 3=strong fit) and annotator in the CSV
python src/11b_human_relevance_judgments.py evaluate --judgments data/evaluation/human_judgments_template.csv
# Only after reviewing the metrics, explicitly request the suitability-wording gate
python src/11b_human_relevance_judgments.py evaluate --judgments data/evaluation/human_judgments_template.csv --approve-suitability-wording
```

This is a complementary evaluation to the top-10-focused review workflow above. Statistical support alone does not unlock the app's suitability wording: the evaluation must be sufficient, the scorer fingerprint must match, and the explicit approval flag must be supplied.

## Verify

```bash
# Syntax/source sanity checks
python src/verify_pipeline.py --mode source

# Full offline test suite; uses synthetic fixtures, not the real datasets
python src/verify_pipeline.py --mode tests

# Check live input files and output schemas after a real-data run
python src/verify_pipeline.py --mode deployment

# Verify the active model/scorer against Step 10 and Step 12 artifacts
python src/verify_deployment.py --mode artifacts
# Or run materialization and the deployment verifier from the pipeline runner
python src/run_pipeline.py --with-final-recommendations --verify-deployment
```

The verifiers check source integrity, file/schema presence, and current scorer fingerprints across pipeline outputs. The deployment verifier is adapted to the current app/ranker interfaces. Neither verifier establishes recommendation quality, fairness, or hiring validity; those require human judgment and suitable evaluation data. The on-disk `EmbeddingStore` caches normalized text vectors by model and text so Step 7 and Step 9 can reuse encodings rather than repeatedly encoding the same strings.

## Configuration and secrets

Copy `.env.example` to `.env` only if environment-based configuration is needed, and add your own secrets locally. **Never commit or share `.env` or API credentials.** The project archive used for delivery excludes `.env`, virtual environments, generated datasets, and embedding/model caches.## Merged Streamlit app and live job search

The merged application is `src/14_app.py`. It keeps the existing resume-to-job retrieval/ranking workflow and adds live job retrieval (Adzuna and Remotive), occupation inference, and structured full-resume evidence sections. The ranking scores are experimental similarity scores, not predictions of hiring success.

### App configuration

Create or edit `.env` in the project root (the same folder as `requirements.txt`) when using provider APIs:

```dotenv
ADZUNA_APP_ID=your_actual_app_id
ADZUNA_APP_KEY=your_actual_app_key
GROQ_API_KEY=your_actual_groq_key
GROQ_MODEL=openai/gpt-oss-20b
```

Adzuna needs both the application ID and key. Groq is optional; the app has a deterministic fallback for explanations. Remotive does not require an API key. Never commit `.env` or share API credentials.

### Install OCR for scanned resumes

The requirements file now includes RapidOCR for image-only/scanned PDFs. From PowerShell in the project root, activate your virtual environment if you use one, then run:

```powershell
python -m pip install -r requirements.txt
```

If you already installed the other dependencies and only need to add OCR:

```powershell
python -m pip install rapidocr-onnxruntime
```

Restart Streamlit after installing. If the app still reports that no OCR engine is installed, the existing `src/pdf_text.py` must actually detect and call RapidOCR; dependency installation by itself cannot add OCR support to a parser that does not use it. Tesseract is another supported option but requires installing the separate Tesseract program on Windows.

### Launch the merged app

From the project root:

```powershell
python -m streamlit run src/14_app.py
```

In live-job mode, the target job title is the role preference; the category selector is reserved for offline/local-dataset filtering, avoiding duplicate role inputs.

### Merged app helper tests

```powershell
python -m pytest -q src/test_live_jobs.py src/test_occupation_inference.py src/test_resume_intelligence.py
```

These helper tests use mocked HTTP and synthetic data. They do not validate provider credentials or replace the repository's pipeline/deployment verification commands above.

### Integration note

Copy the updated app/helper files from the merged bundle into the existing repository's `src/` folder, and use the updated `requirements.txt` and this README in the project root. Keep your other pipeline modules and data files. Do not replace `candidate-retrieval.py` or `run_pipeline.py`; they are not part of this merged app change. A full end-to-end app run still needs to be checked against your local modules, datasets, model artifacts, and credentials.
