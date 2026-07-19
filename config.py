"""Configuração central do Tech Challenge Fase 4."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DADOS = ROOT / "dados"
VIDEO_DIR = DADOS / "video"
AUDIO_DIR = DADOS / "audio"
VITAIS_DIR = DADOS / "sinais_vitais"
PRESCRICOES_DIR = DADOS / "prescricoes"
SAIDAS_DIR = DADOS / "saidas"

# Azure — obrigatório para o pipeline de áudio (sem fallback local)
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "brazilsouth")
AZURE_TEXT_ANALYTICS_ENDPOINT = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT", "")
AZURE_TEXT_ANALYTICS_KEY = os.getenv("AZURE_TEXT_ANALYTICS_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
