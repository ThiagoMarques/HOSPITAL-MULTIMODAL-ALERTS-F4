"""Fusão multimodal e geração de alertas para a equipe clínica.

Junta saídas de vídeo, áudio e anomalias (vitais/prescrições) num pacote único
para revisão humana (HITL). LLM opcional via OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

ORDEM_SEV = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAIXA": 3, "INFO": 4}


def _latest(glob_pat: str, saida_dir: Path | None = None) -> Path | None:
    saida_dir = saida_dir or config.SAIDAS_DIR
    arquivos = sorted(saida_dir.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return arquivos[0] if arquivos else None


def _carregar_json(caminho: Path) -> dict[str, Any]:
    return json.loads(caminho.read_text(encoding="utf-8"))


def _strip_pesado(video: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in video.items() if k not in {"serie_temporal", "amostra_desvios"}}


def _strip_audio(audio: dict[str, Any]) -> dict[str, Any]:
    out = dict(audio)
    if "transcricao" in out and isinstance(out["transcricao"], str):
        t = out["transcricao"]
        out["transcricao_trecho"] = t[:400] + ("…" if len(t) > 400 else "")
        del out["transcricao"]
    return out


def _strip_anomalias(anom: dict[str, Any]) -> dict[str, Any]:
    return {
        "paciente_id": anom.get("paciente_id"),
        "resumo": anom.get("resumo"),
        "metodos": anom.get("metodos"),
        "alertas": anom.get("alertas", []),
        "gerado_em": anom.get("gerado_em"),
    }


def obter_video(
    *,
    reprocessar: bool,
    sample_every: int,
    max_frames: int | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """Retorna (video_alerta, video_ok_opcional, fontes)."""
    fontes: list[str] = []
    video_ok = None

    if not reprocessar:
        p_anom = _latest("relatorio_video_*anomalia*.json")
        p_ok = _latest("relatorio_video_*ok*.json")
        if p_anom:
            fontes.append(f"cache:{p_anom.name}")
            video_anom = _strip_pesado(_carregar_json(p_anom))
            if p_ok:
                fontes.append(f"cache:{p_ok.name}")
                video_ok = _strip_pesado(_carregar_json(p_ok))
            return video_anom, video_ok, fontes

    from src.video.pipeline import analisar_video, salvar_relatorio

    caminho_anom = config.VIDEO_DIR / "amostra_anomalia.mp4"
    caminho_ok = config.VIDEO_DIR / "amostra_ok.mp4"
    res_anom = analisar_video(caminho_anom, sample_every=sample_every, max_frames=max_frames)
    jp, _ = salvar_relatorio(res_anom)
    fontes.append(f"processado:{jp.name}")
    video_anom = _strip_pesado(res_anom)

    if caminho_ok.exists():
        res_ok = analisar_video(caminho_ok, sample_every=sample_every, max_frames=max_frames)
        jp2, _ = salvar_relatorio(res_ok)
        fontes.append(f"processado:{jp2.name}")
        video_ok = _strip_pesado(res_ok)

    return video_anom, video_ok, fontes


def obter_audio(*, reprocessar: bool, audio_path: Path | None) -> tuple[dict[str, Any], list[str]]:
    fontes: list[str] = []
    audio_path = audio_path or (config.AUDIO_DIR / "amostra_consulta_joelho.mp3")

    if not reprocessar:
        p = _latest("relatorio_audio_*.json")
        if p:
            fontes.append(f"cache:{p.name}")
            return _strip_audio(_carregar_json(p)), fontes

    from src.audio.pipeline import analisar_audio, salvar_relatorio

    res = analisar_audio(audio_path)
    jp, _ = salvar_relatorio(res)
    fontes.append(f"processado:{jp.name}")
    return _strip_audio(res), fontes


def obter_anomalias(*, reprocessar: bool) -> tuple[dict[str, Any], list[str]]:
    fontes: list[str] = []
    if not reprocessar:
        p = _latest("relatorio_anomalias_*.json")
        if p:
            fontes.append(f"cache:{p.name}")
            return _strip_anomalias(_carregar_json(p)), fontes

    from src.anomalias.detector import analisar_anomalias, salvar_relatorio

    res = analisar_anomalias()
    jp, _ = salvar_relatorio(res)
    fontes.append(f"processado:{jp.name}")
    return _strip_anomalias(res), fontes


def _severidade_video(video: dict[str, Any]) -> str:
    cls = str(video.get("classificacao", "")).upper()
    if cls == "ANOMALO":
        pct = float(video.get("percentual_desvio") or 0)
        return "ALTA" if pct >= 30 else "MEDIA"
    if cls == "DENTRO_DO_PADRAO":
        return "INFO"
    return "BAIXA"


def _severidade_audio(audio: dict[str, Any]) -> str:
    if audio.get("alerta_clinico"):
        sent = str(audio.get("sentimento", "")).lower()
        if sent in {"negative", "negativo"}:
            return "MEDIA"
        return "BAIXA"
    return "INFO"


def _max_sev(sevs: list[str]) -> str:
    return min(sevs, key=lambda s: ORDEM_SEV.get(s, 9))


def _redigir_mensagem(
    paciente_id: str,
    sev: str,
    video: dict[str, Any],
    audio: dict[str, Any],
    anom: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    linhas = [
        f"ALERTA MULTIMODAL — {paciente_id} — severidade {sev}",
        "Revisão humana necessária (sistema de apoio; não substitui conduta clínica).",
        "",
        "Vídeo:",
        f"  Arquivo: {video.get('video') or video.get('arquivo')}",
        f"  Classificação: {video.get('classificacao')} | "
        f"desvio≈{video.get('percentual_desvio')}%",
        "",
        "Áudio (consulta / educação clínica):",
        f"  Sentimento Azure: {audio.get('sentimento')} | "
        f"termos: {audio.get('termos_criticos_clinicos')}",
        f"  Trecho: {audio.get('transcricao_trecho', '')[:220]}…",
        "",
        "Sinais vitais / prescrições:",
        f"  Total de alertas: {anom.get('resumo', {}).get('n_alertas_total')}",
        f"  Por severidade: {anom.get('resumo', {}).get('por_severidade')}",
    ]
    for a in (anom.get("alertas") or [])[:5]:
        if a.get("fonte") == "sinais_vitais":
            linhas.append(
                f"  - [{a.get('severidade')}] {a.get('timestamp')} vitais "
                f"{a.get('flags_clinicos')} FC={a.get('valores', {}).get('fc_bpm')}"
            )
        else:
            linhas.append(
                f"  - [{a.get('severidade')}] {a.get('timestamp')} "
                f"{a.get('medicamento')} ({a.get('status')})"
            )
    linhas.extend(
        [
            "",
            "Recomendação operacional: correlacionar sessão de exercício com forma inadequada, "
            "picos fisiológicos e revisões de receita (opioide/corticoide) antes de ajustar conduta.",
        ]
    )
    texto = "\n".join(linhas)
    meta = {"llm_usado": False, "modelo_llm": None, "provedor_llm": None}

    api_key = (config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return texto, meta

    try:
        from openai import OpenAI

        modelo = config.LLM_MODEL or "gpt-4o-mini"
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você redige alertas clínicos curtos em português para equipe de "
                        "fisioterapia/enfermagem. Não invente dados. Não substitua o médico. "
                        "Use só o que está no contexto."
                    ),
                },
                {
                    "role": "user",
                    "content": "Reescreva de forma clara e objetiva (máx. 180 palavras):\n\n" + texto,
                },
            ],
            temperature=0.2,
            max_tokens=400,
        )
        redigido = (resp.choices[0].message.content or "").strip()
        if redigido:
            meta = {
                "llm_usado": True,
                "modelo_llm": modelo,
                "provedor_llm": "OpenAI API",
            }
            return (
                redigido
                + "\n\n---\n(Base factual do sistema — multimodal)\n"
                + texto,
                meta,
            )
    except ImportError:
        return texto, meta
    except Exception:
        return texto, meta

    return texto, meta


def montar_alerta(
    video: dict[str, Any],
    audio: dict[str, Any],
    anomalias: dict[str, Any],
    *,
    video_ok: dict[str, Any] | None = None,
    fontes: list[str] | None = None,
) -> dict[str, Any]:
    sev_video = _severidade_video(video)
    sev_audio = _severidade_audio(audio)
    sevs_anom = [a.get("severidade", "BAIXA") for a in anomalias.get("alertas") or []]
    sev_anom = _max_sev(sevs_anom) if sevs_anom else "INFO"
    severidade = _max_sev([sev_video, sev_audio, sev_anom])

    paciente = (
        anomalias.get("paciente_id")
        or audio.get("paciente_id")
        or "PAC-JOELHO-001"
    )

    mensagem, meta_llm = _redigir_mensagem(paciente, severidade, video, audio, anomalias)

    modalidades_risco = []
    if sev_video not in {"INFO"}:
        modalidades_risco.append("video")
    if sev_audio not in {"INFO"}:
        modalidades_risco.append("audio")
    if sevs_anom:
        modalidades_risco.append("vitais_prescricoes")

    return {
        "tipo": "alerta_multimodal",
        "paciente_id": paciente,
        "tema": "Tendinopatia patelar — fusão vídeo + áudio + anomalias",
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "severidade": severidade,
        "hitl": True,
        "aviso": "Sistema de apoio à decisão. Não substitui avaliação da equipe clínica.",
        "modalidades_com_sinal": modalidades_risco,
        "resumo_por_modalidade": {
            "video": {
                "classificacao": video.get("classificacao"),
                "percentual_desvio": video.get("percentual_desvio"),
                "severidade_derivada": sev_video,
                "referencia_ok": (video_ok or {}).get("classificacao") if video_ok else None,
            },
            "audio": {
                "sentimento": audio.get("sentimento"),
                "termos_criticos_clinicos": audio.get("termos_criticos_clinicos"),
                "alerta_clinico": audio.get("alerta_clinico"),
                "severidade_derivada": sev_audio,
                "provedor": audio.get("provedor"),
            },
            "anomalias": {
                "n_alertas": anomalias.get("resumo", {}).get("n_alertas_total"),
                "por_severidade": anomalias.get("resumo", {}).get("por_severidade"),
                "severidade_derivada": sev_anom,
            },
        },
        "redacao": meta_llm,
        "mensagem_para_equipe": mensagem,
        "evidencias": {
            "video": video,
            "video_ok": video_ok,
            "audio": audio,
            "anomalias": anomalias,
        },
        "fontes_entrada": fontes or [],
    }


def salvar_alerta(alerta: dict[str, Any], saida_dir: Path | None = None) -> tuple[Path, Path]:
    saida_dir = saida_dir or config.SAIDAS_DIR
    saida_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = saida_dir / f"alerta_{ts}.json"
    txt_path = saida_dir / f"alerta_{ts}.txt"
    json_path.write_text(json.dumps(alerta, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(alerta.get("mensagem_para_equipe", "") + "\n", encoding="utf-8")
    return json_path, txt_path


def orquestrar(
    *,
    reprocessar: bool = False,
    audio_path: Path | None = None,
    sample_every: int = 2,
    max_frames: int | None = None,
) -> dict[str, Any]:
    fontes: list[str] = []
    video, video_ok, f_v = obter_video(
        reprocessar=reprocessar, sample_every=sample_every, max_frames=max_frames
    )
    fontes.extend(f_v)
    audio, f_a = obter_audio(reprocessar=reprocessar, audio_path=audio_path)
    fontes.extend(f_a)
    anom, f_n = obter_anomalias(reprocessar=reprocessar)
    fontes.extend(f_n)

    alerta = montar_alerta(video, audio, anom, video_ok=video_ok, fontes=fontes)
    return alerta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Orquestra alerta multimodal")
    parser.add_argument(
        "--reprocessar",
        action="store_true",
        help="Refaz vídeo/áudio/anomalias em vez de usar o último JSON em dados/saidas/.",
    )
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    alerta = orquestrar(
        reprocessar=args.reprocessar,
        audio_path=args.audio,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
    )
    jp, tp = salvar_alerta(alerta)
    preview = {k: v for k, v in alerta.items() if k != "evidencias"}
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"\nAlerta:\n  {jp}\n  {tp}")


if __name__ == "__main__":
    main()
