"""
Ponto de entrada do fluxo multimodal — Tech Challenge Fase 4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config


def checar_ambiente() -> None:
    for d in (
        config.VIDEO_DIR,
        config.AUDIO_DIR,
        config.VITAIS_DIR,
        config.PRESCRICOES_DIR,
        config.SAIDAS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

    print("HOSPITAL-MULTIMODAL-ALERTS-F4 — ambiente")
    print(f"  raiz: {config.ROOT}")
    print(f"  videos: {list(config.VIDEO_DIR.glob('*'))}")
    print(f"  audio: {list(config.AUDIO_DIR.glob('*'))}")
    print(f"  vitais: {list(config.VITAIS_DIR.glob('*.csv'))}")
    print(f"  prescricoes: {list(config.PRESCRICOES_DIR.glob('*.csv'))}")


def cmd_video(video: Path, sample_every: int, max_frames: int | None) -> None:
    from src.video.pipeline import analisar_video, salvar_relatorio

    resumo = analisar_video(video, sample_every=sample_every, max_frames=max_frames)
    preview = {k: v for k, v in resumo.items() if k not in {"serie_temporal", "amostra_desvios"}}
    json_path, txt_path = salvar_relatorio(resumo)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"\nRelatórios:\n  {json_path}\n  {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoramento multimodal — Fase 4")
    parser.add_argument("--checar", action="store_true", help="Valida pastas e dados.")
    parser.add_argument("--video", type=Path, help="Analisa um vídeo.")
    parser.add_argument(
        "--videos-amostra",
        action="store_true",
        help="Analisa amostra_ok.mp4 e amostra_anomalia.mp4.",
    )
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.checar or (not args.video and not args.videos_amostra):
        checar_ambiente()
        if not args.video and not args.videos_amostra:
            return

    if args.videos_amostra:
        resultados = []
        for nome in ("amostra_ok.mp4", "amostra_anomalia.mp4"):
            caminho = config.VIDEO_DIR / nome
            print(f"\n=== {nome} ===")
            from src.video.pipeline import analisar_video, salvar_relatorio

            resumo = analisar_video(
                caminho, sample_every=args.sample_every, max_frames=args.max_frames
            )
            preview = {
                k: v for k, v in resumo.items() if k not in {"serie_temporal", "amostra_desvios"}
            }
            json_path, txt_path = salvar_relatorio(resumo)
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            print(f"\nRelatórios:\n  {json_path}\n  {txt_path}")
            resultados.append(preview)

        print("\n=== Comparativo ===")
        for r in resultados:
            print(
                f"  {r['video']}: {r['classificacao']} | "
                f"desvio={r['percentual_desvio']}% | "
                f"ADM E/D={r['amplitude_joelho_esq']}/{r['amplitude_joelho_dir']}"
            )
        return

    if args.video:
        cmd_video(args.video, args.sample_every, args.max_frames)


if __name__ == "__main__":
    main()
