from __future__ import annotations

import os
import json
import base64
from typing import Any

import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError, OpenAI

import matcher

MODEL = "deepseek-ai/deepseek-v4-flash"
MAX_TOKENS = 16384
TEMPERATURE = 1

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)

app = FastAPI(title="CV Career Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")


@app.on_event("startup")
def startup_event() -> None:
    """Carga el dataset al iniciar la API."""
    rows = matcher.load_dataset()
    if rows > 0:
        print(f"Dataset cargado: {rows} empleos")
    else:
        print("Dataset no encontrado, continuar sin él")


def build_system_prompt() -> str:
    return """
Eres un experto en recursos humanos y reclutamiento con acceso
al mercado laboral global de LinkedIn 2024.

Recibirás el contenido de un CV. Tu tarea es:
1. Analizar toda la información del CV (experiencia, habilidades,
   formación, logros, tecnologías, idiomas, etc.)
2. Determinar automáticamente el perfil profesional del candidato
3. Recomendar los 5 puestos de trabajo más adecuados para ese perfil
   basándote únicamente en lo que dice el CV, sin asumir preferencias

Responde ÚNICAMENTE con JSON válido, sin markdown, sin texto adicional:
{
  "nombre": "nombre detectado en el CV o Profesional si no aparece",
  "iniciales": "2 letras mayúsculas",
  "titulo_perfil": "título profesional principal detectado del CV",
  "sector_detectado": "sector profesional detectado automáticamente",
  "anos_experiencia": "número de años de experiencia estimado como string",
  "habilidades": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "scores": {
    "experiencia": 75,
    "habilidades_tecnicas": 80,
    "formacion": 70,
    "mercado_laboral": 85
  },
  "puestos": [
    {
      "titulo": "Título exacto del puesto recomendado",
      "empresa_tipo": "tipo de empresa ideal",
      "match": 92,
      "nivel": "Senior|Mid|Junior|Lead",
      "modalidad": "Remoto|Híbrido|Presencial",
      "salario_rango": "$X,000 - $Y,000 USD/año",
      "tags": ["tag1", "tag2", "tag3"],
      "razon": "Explicación de por qué este puesto encaja con el perfil detectado en el CV",
      "keywords_dataset": "3 a 5 palabras clave en inglés para buscar en dataset de LinkedIn"
    }
  ]
}
Genera exactamente 5 puestos ordenados por match de mayor a menor.
Los valores de match son enteros entre 60 y 99.
Los scores son enteros entre 50 y 99.
"""


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text() for page in document).strip()
    except Exception as exc:
        print(f"Error al extraer texto del PDF: {exc}")
        return ""


def encode_image(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def build_user_content(cv_text: str, img_b64: str, media_type: str) -> list[dict[str, Any]] | str:
    if img_b64:
        return [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_b64}"}},
            {"type": "text", "text": "Analiza este CV y responde con el JSON solicitado."},
        ]
    return f"CV a analizar:\n\n{cv_text}\n\nResponde SOLO con el JSON."


def call_deepseek(system_prompt: str, user_content: Any) -> dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.95,
            extra_body={
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "high",
                }
            },
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        raw = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta.content is not None:
                raw += delta.content

        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except APIError as exc:
        print(f"Error con NVIDIA API: {exc}")
        raise HTTPException(status_code=500, detail=f"Error con NVIDIA API: {str(exc)}") from exc
    except json.JSONDecodeError as exc:
        print(f"La API no devolvió JSON válido: {exc}")
        raise HTTPException(status_code=500, detail="La API no devolvió JSON válido") from exc
    except Exception as exc:
        print(f"Error inesperado al llamar a la API: {exc}")
        raise HTTPException(status_code=500, detail="Error inesperado al procesar la solicitud") from exc


def enrich_with_dataset(result: dict[str, Any]) -> dict[str, Any]:
    puestos = result.get("puestos", [])
    if isinstance(puestos, list):
        for puesto in puestos:
            if isinstance(puesto, dict):
                keywords = str(puesto.get("keywords_dataset", ""))
                puesto["ds_matches"] = matcher.search_jobs(keywords, top_n=3)
    result["dataset_info"] = {"loaded": matcher.dataset_loaded(), "rows": matcher.dataset_size()}
    return result


@app.get("/")
def serve_frontend() -> FileResponse:
    """Sirve el frontend principal."""
    return FileResponse("index.html")


@app.post("/analyze")
async def analyze_cv(
    cv_text: str = Form(default=""),
    cv_file: UploadFile = File(default=None),
) -> JSONResponse:
    """Analiza un CV y retorna puestos de trabajo recomendados."""
    img_b64, media_type = "", ""

    if cv_file:
        file_bytes = await cv_file.read()
        if cv_file.content_type == "application/pdf":
            cv_text = extract_pdf_text(file_bytes)
        elif cv_file.content_type and cv_file.content_type.startswith("image/"):
            img_b64 = encode_image(file_bytes)
            media_type = cv_file.content_type

    if not cv_text and not img_b64:
        raise HTTPException(status_code=400, detail="Sube un CV o pega el texto")

    system_prompt = build_system_prompt()
    user_content = build_user_content(cv_text, img_b64, media_type)
    result = call_deepseek(system_prompt, user_content)
    result = enrich_with_dataset(result)

    return JSONResponse(content=result)


@app.get("/health")
def health() -> dict[str, Any]:
    """Devuelve el estado de salud del servicio."""
    return {
        "status": "ok",
        "model": MODEL,
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