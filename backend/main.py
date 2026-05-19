from __future__ import annotations

import base64
import json
import os
from typing import Any

import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import APIError, OpenAI

import matcher

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

app = FastAPI(title="CV Career Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_system_prompt(sector: str) -> str:
    return f"""Eres un experto en RRHH y reclutamiento con acceso al mercado laboral de LinkedIn 2024.
Analizas CVs y recomiendas puestos de trabajo.
Sector preferido del candidato: {sector}

Responde ÚNICAMENTE con JSON válido, sin markdown, sin texto adicional, sin backticks:
{{
  "nombre": "nombre detectado en el CV o 'Profesional' si no aparece",
  "iniciales": "2 letras mayúsculas del nombre",
  "titulo_perfil": "título profesional principal del candidato en una línea",
  "habilidades": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "scores": {{
    "experiencia": 75,
    "habilidades_tecnicas": 80,
    "formacion": 70,
    "mercado_laboral": 85
  }},
  "puestos": [
    {{
      "titulo": "Título exacto del puesto recomendado",
      "empresa_tipo": "tipo de empresa ideal (startup, corporativo, consultora, ONG, etc.)",
      "match": 92,
      "nivel": "Senior|Mid|Junior|Lead",
      "modalidad": "Remoto|Híbrido|Presencial",
      "salario_rango": "$X,000 - $Y,000 USD/año",
      "tags": ["tag1", "tag2", "tag3"],
      "razon": "Explicación clara de por qué este puesto encaja con el perfil",
      "keywords_dataset": "3 a 5 palabras clave en inglés para buscar en el dataset de LinkedIn"
    }}
  ]
}}
Genera exactamente 5 puestos ordenados por match de mayor a menor.
Los valores de match son enteros entre 60 y 99.
Los scores son enteros entre 50 y 99."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text() for page in document).strip()
    except Exception as e:
        print(f"Error al extraer texto del PDF: {e}")
        raise HTTPException(status_code=400, detail="El archivo PDF no es válido") from e


def image_to_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def call_deepseek(system_prompt: str, user_content: Any) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1500,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        raw = (response.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except APIError as e:
        print(f"Error con DeepSeek API: {e}")
        raise HTTPException(status_code=500, detail=f"Error con DeepSeek API: {str(e)}") from e
    except json.JSONDecodeError as e:
        print(f"DeepSeek no devolvió JSON válido: {e}")
        raise HTTPException(status_code=500, detail="DeepSeek no devolvió JSON válido") from e


async def read_cv_input(cv_text: str, cv_file: UploadFile | None) -> tuple[str, str, str]:
    resolved_text = cv_text.strip()
    image_media_type = ""
    image_b64 = ""
    if cv_file is None:
        return resolved_text, image_media_type, image_b64
    file_bytes = await cv_file.read()
    if cv_file.content_type == "application/pdf":
        resolved_text = extract_text_from_pdf(file_bytes)
    elif cv_file.content_type and cv_file.content_type.startswith("image/"):
        image_media_type = cv_file.content_type
        image_b64 = image_to_base64(file_bytes)
    return resolved_text, image_media_type, image_b64


def build_user_content(cv_text: str, image_media_type: str, image_b64: str) -> Any:
    if image_b64:
        return [
            {"type": "image_url", "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": "Analiza este CV y responde con el JSON solicitado."},
        ]
    return f"CV a analizar:\n\n{cv_text}\n\nResponde SOLO con el JSON."


def enrich_with_dataset_matches(result: dict[str, Any]) -> dict[str, Any]:
    puestos = result.get("puestos", [])
    if isinstance(puestos, list):
        for puesto in puestos:
            if isinstance(puesto, dict):
                keywords = str(puesto.get("keywords_dataset", ""))
                puesto["ds_matches"] = matcher.search_jobs(keywords, top_n=3)
    result["dataset_info"] = {"loaded": matcher.dataset_loaded(), "rows": matcher.dataset_size()}
    return result


@app.on_event("startup")
def startup_event() -> None:
    """Carga el dataset al iniciar la API."""
    rows = matcher.load_dataset()
    if rows > 0:
        print(f"Dataset cargado: {rows} empleos")
    else:
        print("Dataset no encontrado, continuar sin él")


@app.post("/analyze")
async def analyze_cv(
    cv_text: str = Form(default=""),
    cv_file: UploadFile | None = File(default=None),
    sector: str = Form(default="Tech"),
) -> JSONResponse:
    """Analiza un CV y devuelve recomendaciones laborales."""
    resolved_text, media_type, image_b64 = await read_cv_input(cv_text, cv_file)
    if not resolved_text and not image_b64:
        raise HTTPException(status_code=400, detail="Sube un CV o pega el texto")
    system_prompt = build_system_prompt(sector)
    user_content = build_user_content(resolved_text, media_type, image_b64)
    deepseek_result = call_deepseek(system_prompt, user_content)
    result = enrich_with_dataset_matches(deepseek_result)
    return JSONResponse(content=result)


@app.get("/health")
def health() -> dict[str, Any]:
    """Devuelve el estado de salud del servicio."""
    return {
        "status": "ok",
        "model": "deepseek-chat",
        "dataset_loaded": matcher.dataset_loaded(),
        "dataset_rows": matcher.dataset_size(),
    }


@app.get("/dataset-status")
def dataset_status() -> dict[str, Any]:
    """Informa si el dataset está cargado y su tamaño."""
    rows = matcher.dataset_size()
    return {
        "loaded": matcher.dataset_loaded(),
        "rows": rows,
        "message": f"Dataset listo con {rows:,} empleos" if rows > 0 else "Sin dataset cargado",
    }


# ============================================================
# CÓMO USAR ESTE PROYECTO
# ============================================================
# 1. Instalar dependencias:
#       pip install -r requirements.txt
#
# 2. Crear el archivo .env en esta misma carpeta:
#       DEEPSEEK_API_KEY=sk-tu-key-aqui
#    Obtén tu key gratis en: https://build.nvidia.com/deepseek-ai/deepseek-v4-flash
#
# 3. Descargar el dataset de Kaggle (gratis, requiere cuenta):
#    https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024
#    Colocar estos 3 archivos en la carpeta backend/:
#       - linkedin_job_postings.csv
#       - job_skills.csv
#       - job_summary.csv
#
# 4. Correr el servidor:
#       uvicorn main:app --reload --port 8000
#
# 5. Probar que funciona:
#       http://localhost:8000/health
#
# ENDPOINTS DISPONIBLES:
#   POST /analyze        ← analiza un CV y retorna puestos recomendados
#   GET  /health         ← verifica que el servidor está corriendo
#   GET  /dataset-status ← cuántos empleos tiene el dataset cargado
# ============================================================
