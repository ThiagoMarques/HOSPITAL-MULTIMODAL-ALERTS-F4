"""Pipeline de análise de vídeo clínico com MediaPipe Pose (Tasks API).

Foco: reabilitação de joelho — ângulos do joelho, assimetria e desvios
em relação a um padrão esperado (sessão Long Arc Quad / carga controlada).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mediapipe import Image as MpImage
from mediapipe import ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmark,
    RunningMode,
)

import config

# Landmarks relevantes para joelho / aparelho extensor
_HIP_L = PoseLandmark.LEFT_HIP
_KNEE_L = PoseLandmark.LEFT_KNEE
_ANKLE_L = PoseLandmark.LEFT_ANKLE
_HIP_R = PoseLandmark.RIGHT_HIP
_KNEE_R = PoseLandmark.RIGHT_KNEE
_ANKLE_R = PoseLandmark.RIGHT_ANKLE

DEFAULT_MODEL = config.ROOT / "models" / "pose_landmarker_lite.task"


@dataclass
class FrameMetrics:
    frame_idx: int
    time_s: float
    knee_angle_l: float | None
    knee_angle_r: float | None
    asymmetry_deg: float | None
    desvio: bool
    motivo: str


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo em graus no vértice b (a-b-c)."""
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _lm_xy(landmarks, idx: PoseLandmark) -> np.ndarray:
    lm = landmarks[int(idx)]
    return np.array([lm.x, lm.y], dtype=np.float64)


def knee_angles(landmarks) -> tuple[float | None, float | None]:
    try:
        left = _angle_3pts(_lm_xy(landmarks, _HIP_L), _lm_xy(landmarks, _KNEE_L), _lm_xy(landmarks, _ANKLE_L))
        right = _angle_3pts(_lm_xy(landmarks, _HIP_R), _lm_xy(landmarks, _KNEE_R), _lm_xy(landmarks, _ANKLE_R))
        return left, right
    except Exception:
        return None, None


def classificar_desvio(
    knee_l: float | None,
    knee_r: float | None,
    *,
    prev_l: float | None = None,
    prev_r: float | None = None,
    flexao_bilateral: float = 100.0,
    jerk_limiar: float = 35.0,
) -> tuple[bool, str]:
    """Heurística para reabilitação de joelho (Long Arc Quad vs forma inadequada).

    Assimetria isolada NÃO é desvio: no Long Arc Quad um joelho estende de cada vez.
    Desvios relevantes no MVP:
    - Flexão bilateral acentuada (ambos joelhos < limiar) → padrão de agachamento
    - Variação brusca frame-a-frame (jerk) → movimento pouco controlado
    """
    if knee_l is None or knee_r is None:
        return False, "pose_incompleta"

    if knee_l <= flexao_bilateral and knee_r <= flexao_bilateral:
        return True, f"flexao_bilateral_L{knee_l:.1f}_R{knee_r:.1f}"

    if prev_l is not None and prev_r is not None:
        jerk = max(abs(knee_l - prev_l), abs(knee_r - prev_r))
        if jerk >= jerk_limiar:
            return True, f"movimento_brusco_{jerk:.1f}deg"

    return False, "ok"


def analisar_video(
    video_path: Path,
    *,
    model_path: Path = DEFAULT_MODEL,
    max_frames: int | None = None,
    sample_every: int = 2,
) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo MediaPipe não encontrado: {model_path}. "
            "Baixe pose_landmarker_lite.task para models/."
        )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    metrics: list[FrameMetrics] = []
    angles_l: list[float] = []
    angles_r: list[float] = []
    frames_com_pose = 0
    frame_idx = 0
    prev_l: float | None = None
    prev_r: float | None = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                ts_ms = int((frame_idx / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                knee_l = knee_r = None
                desvio = False
                motivo = "sem_pose"
                if result.pose_landmarks:
                    frames_com_pose += 1
                    landmarks = result.pose_landmarks[0]
                    knee_l, knee_r = knee_angles(landmarks)
                    desvio, motivo = classificar_desvio(
                        knee_l, knee_r, prev_l=prev_l, prev_r=prev_r
                    )
                    if knee_l is not None:
                        angles_l.append(knee_l)
                        prev_l = knee_l
                    if knee_r is not None:
                        angles_r.append(knee_r)
                        prev_r = knee_r

                metrics.append(
                    FrameMetrics(
                        frame_idx=frame_idx,
                        time_s=round(frame_idx / fps, 3),
                        knee_angle_l=None if knee_l is None else round(knee_l, 2),
                        knee_angle_r=None if knee_r is None else round(knee_r, 2),
                        asymmetry_deg=(
                            None
                            if knee_l is None or knee_r is None
                            else round(abs(knee_l - knee_r), 2)
                        ),
                        desvio=desvio,
                        motivo=motivo,
                    )
                )

            frame_idx += 1

    cap.release()

    desvios = [m for m in metrics if m.desvio]
    amostrados = len(metrics) or 1
    adm_l = (max(angles_l) - min(angles_l)) if len(angles_l) >= 2 else 0.0
    adm_r = (max(angles_r) - min(angles_r)) if len(angles_r) >= 2 else 0.0

    # Classificação relativa ao padrão esperado (Long Arc Quad sentado):
    # - ADM alta + ângulos médios intermediários → reabilitação controlada
    # - Joelhos quase estendidos o tempo todo (em pé) → diverge do padrão esperado
    # - Muitos eventos de flexão bilateral/brusca → anômalo
    pct_desvio = 100.0 * len(desvios) / amostrados
    media_l = float(np.mean(angles_l)) if angles_l else 0.0
    media_r = float(np.mean(angles_r)) if angles_r else 0.0

    if pct_desvio >= 12.0:
        classificacao = "ANOMALO"
    elif media_l >= 160.0 and media_r >= 160.0:
        # Em pé / pernas estendidas — não corresponde à sessão Long Arc Quad
        classificacao = "ANOMALO"
    elif max(adm_l, adm_r) >= 25.0:
        classificacao = "DENTRO_DO_PADRAO"
    else:
        classificacao = "INCONCLUSIVO"

    resumo = {
        "video": str(video_path.name),
        "caminho": str(video_path),
        "paciente_id": "PAC-JOELHO-001",
        "tema": "Tendinopatia patelar / reabilitação de joelho",
        "modelo": "MediaPipe PoseLandmarker lite",
        "padrao_esperado": "long_arc_quad_carga_controlada",
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "fps": fps,
        "frames_totais": total or frame_idx,
        "frames_amostrados": len(metrics),
        "frames_com_pose": frames_com_pose,
        "taxa_deteccao_pose": round(frames_com_pose / amostrados, 3),
        "angulo_joelho_esq_medio": round(float(np.mean(angles_l)), 2) if angles_l else None,
        "angulo_joelho_dir_medio": round(float(np.mean(angles_r)), 2) if angles_r else None,
        "angulo_joelho_esq_std": round(float(np.std(angles_l)), 2) if angles_l else None,
        "angulo_joelho_dir_std": round(float(np.std(angles_r)), 2) if angles_r else None,
        "amplitude_joelho_esq": round(adm_l, 2),
        "amplitude_joelho_dir": round(adm_r, 2),
        "frames_com_desvio": len(desvios),
        "percentual_desvio": round(pct_desvio, 2),
        "classificacao": classificacao,
        "motivos_desvio": _contar_motivos(desvios),
        "amostra_desvios": [asdict(m) for m in desvios[:20]],
        "serie_temporal": [asdict(m) for m in metrics],
    }
    return resumo


def _contar_motivos(desvios: list[FrameMetrics]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in desvios:
        if m.motivo.startswith("flexao_bilateral"):
            key = "flexao_bilateral"
        elif m.motivo.startswith("movimento_brusco"):
            key = "movimento_brusco"
        else:
            key = m.motivo
        out[key] = out.get(key, 0) + 1
    return out


def salvar_relatorio(resumo: dict[str, Any], saida_dir: Path | None = None) -> tuple[Path, Path]:
    saida_dir = saida_dir or config.SAIDAS_DIR
    saida_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(resumo["video"]).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = saida_dir / f"relatorio_video_{stem}_{ts}.json"
    txt_path = saida_dir / f"relatorio_video_{stem}_{ts}.txt"

    # JSON completo (série temporal incluída)
    json_path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")

    # Relatório legível curto
    linhas = [
        "RELATÓRIO DE ANÁLISE DE VÍDEO",
        f"Vídeo: {resumo['video']}",
        f"Paciente: {resumo['paciente_id']} — {resumo['tema']}",
        f"Modelo: {resumo['modelo']}",
        f"Gerado em: {resumo['gerado_em']}",
        "",
        f"Frames amostrados: {resumo['frames_amostrados']}",
        f"Taxa de detecção de pose: {resumo['taxa_deteccao_pose']}",
        f"Ângulo médio joelho E/D: {resumo['angulo_joelho_esq_medio']} / {resumo['angulo_joelho_dir_medio']}",
        f"Frames com desvio: {resumo['frames_com_desvio']} ({resumo['percentual_desvio']}%)",
        f"Classificação: {resumo['classificacao']}",
        f"Motivos: {resumo['motivos_desvio']}",
        "",
        "Interpretação clínica (MVP):",
        "- DENTRO_DO_PADRAO: ADM relevante e ângulos compatíveis com Long Arc Quad sentado.",
        "- ANOMALO: flexão bilateral/brusca OU postura em pé com joelhos quase estendidos",
        "  (diverge do padrão esperado de reabilitação controlada do aparelho extensor).",
        "- Assimetria isolada NÃO conta como desvio (esperada em extensão unilateral).",
    ]
    txt_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return json_path, txt_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Análise de vídeo clínico (MediaPipe Pose)")
    parser.add_argument("--video", type=Path, required=True, help="Caminho do vídeo")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    resumo = analisar_video(
        args.video,
        model_path=args.model,
        max_frames=args.max_frames,
        sample_every=args.sample_every,
    )
    # série temporal grande — no print só o resumo
    preview = {k: v for k, v in resumo.items() if k != "serie_temporal"}
    json_path, txt_path = salvar_relatorio(resumo)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"\nRelatórios:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
