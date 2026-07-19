"""
Ponto de entrada do fluxo multimodal.

Os módulos em src/ processam vídeo, áudio e anomalias. Por enquanto só valida o ambiente.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config


def checar_ambiente() -> None:
    faltando = []
    for d in (
        config.VIDEO_DIR,
        config.AUDIO_DIR,
        config.VITAIS_DIR,
        config.PRESCRICOES_DIR,
        config.SAIDAS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
        if not any(d.iterdir()):
            faltando.append(str(d.relative_to(config.ROOT)))

    print("TECH-CHALLENGE-FASE-4 — ambiente OK")
    print(f"  raiz: {config.ROOT}")
    if faltando:
        print("  pastas ainda vazias:")
        for p in faltando:
            print(f"    - {p}")
    else:
        print("  dados de amostra encontrados.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoramento multimodal — Fase 4")
    parser.add_argument(
        "--checar",
        action="store_true",
        help="Valida pastas e config.",
    )
    args = parser.parse_args()

    if args.checar or True:
        # Sem argumentos, o default é só checar o ambiente.
        checar_ambiente()


if __name__ == "__main__":
    main()
