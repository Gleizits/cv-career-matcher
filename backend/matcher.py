from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

DATASET_FILE = "job_postings.csv"
DATASET_ID = "arshkon/linkedin-job-postings"

COLUMN_MAP = {
    "title": "job_title",
    "company_name": "company",
    "location": "job_location",
    "formatted_experience_level": "job_level",
    "formatted_work_type": "job_type",
    "skills_desc": "job_skills",
    "description": "job_summary",
    "job_posting_url": "job_link",
}

SEARCH_COLUMNS = ["_title", "_skills", "_summary"]
SEARCH_WEIGHTS = {"_title": 3, "_skills": 2, "_summary": 1}

_jobs_df: pd.DataFrame | None = None


def _download_dataset() -> bool:
    base_dir = Path(__file__).resolve().parent
    target_path = base_dir / DATASET_FILE
    if target_path.exists():
        return True
    try:
        import kagglehub

        dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
        candidates = list(dataset_path.rglob(DATASET_FILE))
        if not candidates:
            print(f"Advertencia: no se encontró {DATASET_FILE} en el dataset descargado.")
            return False
        shutil.copy(candidates[0], target_path)
        return True
    except Exception as exc:
        print(f"Advertencia: no se pudo descargar el dataset desde Kaggle: {exc}")
        return False


def load_dataset() -> int:
    global _jobs_df

    started_at = time.time()
    base_dir = Path(__file__).resolve().parent
    postings_path = base_dir / DATASET_FILE

    if not _download_dataset() or not postings_path.exists():
        _jobs_df = None
        return 0

    try:
        postings = pd.read_csv(postings_path, low_memory=False)
        postings = postings[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
        postings = postings[list(COLUMN_MAP.values())].copy()
        postings = postings.fillna("")

        postings["_title"] = postings["job_title"].astype(str).str.lower()
        postings["_skills"] = postings["job_skills"].astype(str).str.lower()
        postings["_summary"] = postings["job_summary"].astype(str).str.lower()

        _jobs_df = postings
        elapsed = time.time() - started_at
        print(f"Dataset cargado en {elapsed:.1f}s — {len(_jobs_df):,} empleos")
        return len(_jobs_df)
    except Exception as exc:
        print(f"Advertencia: no se pudo cargar el dataset: {exc}")
        _jobs_df = None
        return 0


def search_jobs(keywords: str, top_n: int = 5) -> list[dict[str, str]]:
    if _jobs_df is None or _jobs_df.empty:
        return []

    if not keywords or not keywords.strip():
        return []

    tokens = [token.lower() for token in keywords.split() if len(token) > 2]
    if not tokens:
        return []

    try:
        score = pd.Series(0, index=_jobs_df.index, dtype="int64")
        for token in tokens:
            for column in SEARCH_COLUMNS:
                weight = SEARCH_WEIGHTS.get(column, 1)
                score += _jobs_df[column].str.contains(token, regex=False, na=False).astype(int) * weight
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
    except Exception as exc:
        print(f"Error al buscar empleos en el dataset: {exc}")
        return []


def dataset_loaded() -> bool:
    return _jobs_df is not None and len(_jobs_df) > 0


def dataset_size() -> int:
    return len(_jobs_df) if _jobs_df is not None else 0
