from __future__ import annotations

from pathlib import Path

import pandas as pd

_df = None


def load_dataset() -> int:
    global _df

    base_dir = Path(__file__).resolve().parent
    postings_path = base_dir / "linkedin_job_postings.csv"
    skills_path = base_dir / "job_skills.csv"
    summary_path = base_dir / "job_summary.csv"

    required_files = [postings_path, skills_path, summary_path]
    missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]
    if missing_files:
        for missing_file in missing_files:
            print(f"Advertencia: archivo no encontrado: {missing_file}")
        _df = None
        return 0

    try:
        postings = pd.read_csv(postings_path)
        skills = pd.read_csv(skills_path)
        summary = pd.read_csv(summary_path)

        merged = postings.merge(skills, on="job_link", how="left")
        merged = merged.merge(summary, on="job_link", how="left")
        merged = merged[
            [
                "job_link",
                "job_title",
                "company",
                "job_location",
                "job_level",
                "job_type",
                "job_skills",
                "job_summary",
            ]
        ].copy()
        merged = merged.fillna("")

        merged["_title"] = merged["job_title"].astype(str).str.lower()
        merged["_skills"] = merged["job_skills"].astype(str).str.lower()
        merged["_summary"] = merged["job_summary"].astype(str).str.lower()

        _df = merged
        return int(len(_df))
    except Exception as e:
        print(f"Advertencia: no se pudo cargar el dataset: {e}")
        _df = None
        return 0


def search_jobs(keywords: str, top_n: int = 5) -> list[dict]:
    if _df is None or _df.empty:
        return []

    if not keywords or not keywords.strip():
        return []

    tokens = [token.lower() for token in keywords.split() if len(token) > 2]
    if not tokens:
        return []

    score = pd.Series(0, index=_df.index, dtype="int64")

    for token in tokens:
        score += _df["_title"].str.contains(token, regex=False, na=False).astype(int) * 3
        score += _df["_skills"].str.contains(token, regex=False, na=False).astype(int) * 2
        score += _df["_summary"].str.contains(token, regex=False, na=False).astype(int) * 1

    matches = _df.loc[score > 0].copy()
    if matches.empty:
        return []

    matches["score"] = score[score > 0]
    top = matches.nlargest(top_n, "score")

    results = []
    for _, row in top.iterrows():
        results.append(
            {
                "job_title": str(row["job_title"]),
                "company": str(row["company"]),
                "job_location": str(row["job_location"]),
                "job_level": str(row["job_level"]),
                "job_type": str(row["job_type"]),
                "job_skills": str(row["job_skills"])[:150],
                "job_link": str(row["job_link"]),
            }
        )
    return results


def dataset_loaded() -> bool:
    return _df is not None and len(_df) > 0


def dataset_size() -> int:
    return len(_df) if _df is not None else 0
