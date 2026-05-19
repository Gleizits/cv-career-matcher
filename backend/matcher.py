from __future__ import annotations

import time
from pathlib import Path
import shutil

import pandas as pd

COLUMNS = [
    "job_link",
    "job_title",
    "company",
    "job_location",
    "job_level",
    "job_type",
    "job_skills",
    "job_summary",
]
_jobs_df: pd.DataFrame | None = None


def load_dataset() -> int:
    global _jobs_df

    started_at = time.perf_counter()
    base_dir = Path(__file__).resolve().parent
    postings_path = base_dir / "linkedin_job_postings.csv"
    skills_path = base_dir / "job_skills.csv"
    summary_path = base_dir / "job_summary.csv"

    required_files = [postings_path, skills_path, summary_path]
    missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]
    if missing_files:
        try:
            import kagglehub

            path = Path(kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024"))
            for target_file in required_files:
                if target_file.exists():
                    continue
                candidates = list(path.rglob(target_file.name))
                if candidates:
                    shutil.copy(candidates[0], target_file)
        except Exception as e:
            print(f"Advertencia: no se pudo descargar el dataset desde Kaggle: {e}")
            _jobs_df = None
            return 0

        missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]
        if missing_files:
            for missing_file in missing_files:
                print(f"Advertencia: archivo no encontrado: {missing_file}")
            _jobs_df = None
            return 0

    try:
        postings = pd.read_csv(postings_path)
        skills = pd.read_csv(skills_path)
        summary = pd.read_csv(summary_path)

        merged = postings.merge(skills, on="job_link", how="left")
        merged = merged.merge(summary, on="job_link", how="left")
        merged = merged[COLUMNS].copy()
        merged = merged.fillna("")

        merged["_title"] = merged["job_title"].astype(str).str.lower()
        merged["_skills"] = merged["job_skills"].astype(str).str.lower()
        merged["_summary"] = merged["job_summary"].astype(str).str.lower()

        _jobs_df = merged
        total_jobs = int(len(_jobs_df))
        elapsed = time.perf_counter() - started_at
        print(f"Dataset cargado en {elapsed:.1f}s — {total_jobs:,} empleos")
        return total_jobs
    except Exception as e:
        print(f"Advertencia: no se pudo cargar el dataset: {e}")
        _jobs_df = None
        return 0


def search_jobs(keywords: str, top_n: int = 5) -> list[dict[str, str]]:
    if _jobs_df is None or _jobs_df.empty:
        return []

    if not keywords or not keywords.strip():
        return []

    tokens: list[str] = [token.lower() for token in keywords.split() if len(token) > 2]
    if not tokens:
        return []

    try:
        score = pd.Series(0, index=_jobs_df.index, dtype="int64")
        for token in tokens:
            score += _jobs_df["_title"].str.contains(token, regex=False, na=False).astype(int) * 3
            score += _jobs_df["_skills"].str.contains(token, regex=False, na=False).astype(int) * 2
            score += _jobs_df["_summary"].str.contains(token, regex=False, na=False).astype(int) * 1
        matches = _jobs_df.loc[score > 0].copy()
        if matches.empty:
            return []
        matches["score"] = score[score > 0]
        top = matches.nlargest(top_n, "score")
        results: list[dict[str, str]] = []
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
    except Exception as e:
        print(f"Error al buscar empleos en el dataset: {e}")
        return []


def dataset_loaded() -> bool:
    return _jobs_df is not None and len(_jobs_df) > 0


def dataset_size() -> int:
    return len(_jobs_df) if _jobs_df is not None else 0
