from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEARCH_DIRS = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "datasets",
    PROJECT_ROOT / "dataset",
]

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

TEXT_KEYWORDS = [
    "resume",
    "cv",
    "curriculum",
    "description",
    "job_description",
    "job_description_text",
    "posting",
    "summary",
    "profile",
    "experience",
    "education",
    "skills",
    "requirements",
    "responsibilities",
]

ID_KEYWORDS = [
    "resume_id",
    "cv_id",
    "job_id",
    "id",
]

TITLE_KEYWORDS = [
    "title",
    "job_title",
    "occupation",
    "position",
    "role",
]


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_file(path):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)

        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        if path.suffix.lower() == ".json":
            return pd.read_json(path)

        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)

    except Exception as e:
        print(f"ERROR loading {path.name}: {e}")

    return None


def classify_column(column):
    name = str(column).lower().strip()

    if any(k in name for k in TEXT_KEYWORDS):
        return "TEXT"

    if any(k in name for k in TITLE_KEYWORDS):
        return "TITLE"

    if any(k == name or k in name for k in ID_KEYWORDS):
        return "ID"

    return "OTHER"


def inspect_dataframe(path, df):
    print_header(f"FILE: {path}")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")

    for column in df.columns:
        kind = classify_column(column)

        dtype = str(df[column].dtype)
        non_null = int(df[column].notna().sum())

        print(
            f"  [{kind:5}] "
            f"{str(column):40} "
            f"dtype={dtype:12} "
            f"non_null={non_null:,}"
        )

    text_candidates = [
        c for c in df.columns
        if classify_column(c) in {"TEXT", "TITLE"}
    ]

    if text_candidates:
        print("\nPotential text/title columns:")

        for column in text_candidates:
            series = df[column].dropna().astype(str)

            if len(series) == 0:
                continue

            lengths = series.str.len()

            print(
                f"  {column}: "
                f"avg_length={lengths.mean():.1f}, "
                f"max_length={lengths.max():,}"
            )

            print("    Example:")
            print("    " + series.iloc[0][:300].replace("\n", " "))

    else:
        print("\nNO obvious text/title columns detected.")


def main():
    print_header("AI RESUME ANALYZER - RAW TEXT SOURCE INSPECTION")

    print(f"Project root: {PROJECT_ROOT}")

    files = []

    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                files.append(path)

    files = sorted(set(files))

    if not files:
        print("\nNo supported datasets found.")
        print("Expected CSV / Excel / JSON / Parquet files under:")
        for directory in SEARCH_DIRS:
            print(f"  {directory}")
        return

    print(f"\nFound {len(files)} dataset files.")

    for path in files:
        print_header(f"INSPECTING: {path.name}")

        df = load_file(path)

        if df is None:
            continue

        inspect_dataframe(path, df)

    print_header("INSPECTION COMPLETE")

    print(
        """
Next step:
1. Find the file containing actual resume text.
2. Find the file containing actual job descriptions.
3. Do NOT modify Step 9 yet.
4. Use the detected column names to build the real semantic pipeline.
"""
    )


if __name__ == "__main__":
    main()