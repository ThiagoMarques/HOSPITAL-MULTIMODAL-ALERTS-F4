"""Pipeline de áudio clínico — somente Azure (Speech + Text Analytics)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from src.azure.clientes import analisar_texto, transcrever_arquivo_wav

# Termos ligados ao caso PAC-JOELHO-001 / tendinopatia patelar
TERMOS_CRITICOS = [
    "joelho",
    "tendinite",
    "tendinopatia",
    "tendão",
    "tendao",
    "patelar",
    "patela",
    "saltador",
    "dor",
    "inflamação",
    "inflamacao",
    "inchaço",
    "inchaco",
    "extensor",
    "quadríceps",
    "quadriceps",
    "chute",
    "salto",
    "corrida",
    "frenagem",
    "fisioterapia",
    "reabilitação",
    "reabilitacao",
    "agachamento",
    "lesão",
    "lesao",
]


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg não encontrado. Instale imageio-ffmpeg (pip install imageio-ffmpeg)."
    )


def audio_para_wav_pcm(origem: Path, destino: Path) -> Path:
    """Converte qualquer áudio suportado para WAV PCM 16 kHz mono (Azure Speech)."""
    cmd = [
        _ffmpeg(),
        "-y",
        "-i",
        str(origem),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destino),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return destino


def extrair_termos_criticos(texto: str) -> list[str]:
    achados: list[str] = []
    baixo = texto.casefold()
    for termo in TERMOS_CRITICOS:
        if termo.casefold() in baixo:
            achados.append(termo)
    # únicos preservando ordem
    vistos: set[str] = set()
    out: list[str] = []
    for t in achados:
        k = t.casefold()
        if k not in vistos:
            vistos.add(k)
            out.append(t)
    return out


def analisar_audio(caminho: Path) -> dict[str, Any]:
    if not caminho.exists():
        raise FileNotFoundError(caminho)

    with tempfile.TemporaryDirectory(prefix="azure_speech_") as tmp:
        wav = Path(tmp) / "audio_16k.wav"
        audio_para_wav_pcm(caminho, wav)
        transcricao = transcrever_arquivo_wav(str(wav))

    texto = re.sub(r"\s+", " ", transcricao.texto).strip()
    nlp = analisar_texto(texto)
    termos = extrair_termos_criticos(texto)

    return {
        "paciente_id": "PAC-JOELHO-001",
        "tema": "Tendinopatia patelar / consulta sobre joelho",
        "arquivo": caminho.name,
        "caminho": str(caminho),
        "provedor": {
            "transcricao": "Azure Speech Service",
            "nlp": "Azure Text Analytics (Language)",
            "fallback_local": False,
        },
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "transcricao": texto,
        "sentimento": nlp["sentimento"],
        "scores_sentimento": nlp["scores"],
        "frases_chave_azure": nlp["frases_chave"],
        "termos_criticos_clinicos": termos,
        "alerta_clinico": bool(termos),
    }


def salvar_relatorio(resultado: dict[str, Any], saida_dir: Path | None = None) -> tuple[Path, Path]:
    saida_dir = saida_dir or config.SAIDAS_DIR
    saida_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(resultado["arquivo"]).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = saida_dir / f"relatorio_audio_{stem}_{ts}.json"
    txt_path = saida_dir / f"relatorio_audio_{stem}_{ts}.txt"

    json_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    linhas = [
        "RELATÓRIO DE ÁUDIO — Azure Speech + Text Analytics",
        f"Arquivo: {resultado['arquivo']}",
        f"Paciente: {resultado['paciente_id']}",
        f"Gerado em: {resultado['gerado_em']}",
        "",
        f"Sentimento: {resultado['sentimento']} | scores={resultado['scores_sentimento']}",
        f"Termos clínicos: {resultado['termos_criticos_clinicos']}",
        f"Frases-chave (Azure): {resultado['frases_chave_azure']}",
        "",
        "Transcrição:",
        resultado["transcricao"],
    ]
    txt_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return json_path, txt_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Análise de áudio clínico (somente Azure)")
    parser.add_argument(
        "--audio",
        type=Path,
        default=config.AUDIO_DIR / "amostra_consulta_joelho.mp3",
    )
    args = parser.parse_args(argv)
    resultado = analisar_audio(args.audio)
    json_path, txt_path = salvar_relatorio(resultado)
    preview = {k: v for k, v in resultado.items() if k != "transcricao"}
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print("\nTranscrição (trecho):")
    print(resultado["transcricao"][:500], "...")
    print(f"\nRelatórios:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
