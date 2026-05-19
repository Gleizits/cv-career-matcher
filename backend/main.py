from __future__ import annotations

import base64
import json
import os

import fitz
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI

import matcher

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)

app = FastAPI(title="CV Career Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    rows = matcher.load_dataset()
    if rows > 0:
        print(f"Dataset cargado: {rows} empleos")
    else:
        print("Dataset no encontrado, continuar sin él")


@app.post("/analyze")
async def analyze_cv(
    cv_text: str = Form(default=""),
    cv_file: UploadFile = File(default=None),
    sector: str = Form(default="Tech"),
):
    image_media_type = ""
    image_b64 = ""

    try:
        if cv_file is not None:
            if cv_file.content_type == "application/pdf":
                bytes_data = await cv_file.read()
                document = fitz.open(stream=bytes_data, filetype="pdf")
                extracted_text = []
                for page in document:
                    extracted_text.append(page.get_text())
                cv_text = "\n".join(extracted_text).strip()
            elif cv_file.content_type and cv_file.content_type.startswith("image/"):
                bytes_data = await cv_file.read()
                image_b64 = base64.b64encode(bytes_data).decode("utf-8")
                image_media_type = cv_file.content_type

        if not cv_text.strip() and not image_b64:
            raise HTTPException(status_code=400, detail="Sube un CV o pega el texto")

        system_prompt = f"""Eres un experto en RRHH y reclutamiento con acceso al mercado laboral de LinkedIn 2024.
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

        if image_b64:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": "Analiza este CV y responde con el JSON solicitado.",
                },
            ]
        else:
            user_content = f"CV a analizar:\n\n{cv_text}\n\nResponde SOLO con el JSON."

        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1500,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        raw = response.choices[0].message.content or ""
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500, detail=f"DeepSeek no devolvió JSON válido: {raw}"
            ) from e

        puestos = result.get("puestos", [])
        if isinstance(puestos, list):
            for puesto in puestos:
                if isinstance(puesto, dict):
                    keywords = str(puesto.get("keywords_dataset", ""))
                    puesto["ds_matches"] = matcher.search_jobs(keywords, top_n=3)

        result["dataset_info"] = {
            "loaded": matcher.dataset_loaded(),
            "rows": matcher.dataset_size(),
        }

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"Error con DeepSeek API: {str(e)}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="DeepSeek no devolvió JSON válido") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "deepseek-chat",
        "dataset_loaded": matcher.dataset_loaded(),
        "dataset_rows": matcher.dataset_size(),
    }


@app.get("/dataset-status")
def dataset_status():
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
#    Obtén tu key gratis en: https://platform.deepseek.com
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
