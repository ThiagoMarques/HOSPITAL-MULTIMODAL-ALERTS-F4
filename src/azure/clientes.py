"""Clientes Azure Cognitive Services — Speech e Text Analytics.

Sem fallback local: se as credenciais faltarem ou a API falhar, a execução aborta.
"""
from __future__ import annotations

from dataclasses import dataclass

import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

import config


@dataclass
class TranscricaoResultado:
    texto: str
    motivo_cancelamento: str | None = None


def _exige(valor: str, nome: str) -> str:
    if not (valor or "").strip():
        raise RuntimeError(
            f"Credencial Azure ausente: {nome}. "
            "Preencha o arquivo .env (veja .env.example) e tente de novo."
        )
    return valor.strip()


def criar_speech_config() -> speechsdk.SpeechConfig:
    key = _exige(config.AZURE_SPEECH_KEY, "AZURE_SPEECH_KEY")
    region = _exige(config.AZURE_SPEECH_REGION, "AZURE_SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = "pt-BR"
    return speech_config


def criar_text_analytics_client() -> TextAnalyticsClient:
    endpoint = _exige(config.AZURE_TEXT_ANALYTICS_ENDPOINT, "AZURE_TEXT_ANALYTICS_ENDPOINT")
    if "SEU_RECURSO" in endpoint:
        raise RuntimeError(
            "AZURE_TEXT_ANALYTICS_ENDPOINT ainda está com o placeholder do .env.example. "
            "Substitua pelo endpoint real do recurso Language no portal Azure."
        )
    key = _exige(config.AZURE_TEXT_ANALYTICS_KEY, "AZURE_TEXT_ANALYTICS_KEY")
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def transcrever_arquivo_wav(caminho_wav: str) -> TranscricaoResultado:
    """Transcreve um WAV (PCM) com Azure Speech (reconhecimento contínuo)."""
    speech_config = criar_speech_config()
    audio_config = speechsdk.audio.AudioConfig(filename=caminho_wav)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    partes: list[str] = []
    done = False
    erro: str | None = None

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            texto = (evt.result.text or "").strip()
            if texto:
                partes.append(texto)

    def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        nonlocal erro, done
        cancellation = evt.result.cancellation_details
        if cancellation.reason == speechsdk.CancellationReason.Error:
            erro = f"{cancellation.reason}: {cancellation.error_details}"
        done = True

    def on_session_stopped(_: speechsdk.SessionEventArgs) -> None:
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    recognizer.start_continuous_recognition()
    # espera o fim da sessão
    import time

    while not done:
        time.sleep(0.25)
    recognizer.stop_continuous_recognition()

    if erro:
        raise RuntimeError(f"Azure Speech falhou: {erro}")

    texto = " ".join(partes).strip()
    if not texto:
        raise RuntimeError(
            "Azure Speech não retornou texto. "
            "Verifique o áudio (WAV PCM 16 kHz mono) e a região/chave."
        )
    return TranscricaoResultado(texto=texto)


def analisar_texto(texto: str) -> dict:
    """Sentimento + frases-chave via Azure Text Analytics Language."""
    client = criar_text_analytics_client()
    # API aceita documentos; cortamos pedaços grandes se necessário
    docs = [texto[:5000]] if len(texto) > 5000 else [texto]

    sent = client.analyze_sentiment(documents=docs, language="pt")[0]
    if sent.is_error:
        raise RuntimeError(f"Text Analytics (sentimento) falhou: {sent.error}")

    keys = client.extract_key_phrases(documents=docs, language="pt")[0]
    if keys.is_error:
        raise RuntimeError(f"Text Analytics (frases-chave) falhou: {keys.error}")

    return {
        "sentimento": sent.sentiment,
        "scores": {
            "positivo": round(sent.confidence_scores.positive, 4),
            "neutro": round(sent.confidence_scores.neutral, 4),
            "negativo": round(sent.confidence_scores.negative, 4),
        },
        "frases_chave": list(keys.key_phrases),
    }
