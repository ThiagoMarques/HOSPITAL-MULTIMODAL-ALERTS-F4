"""Detecção de anomalias em sinais vitais e prescrições.

Vitais: Isolation Forest + z-score + limiares clínicos.
Prescrições: regras sobre introdução de opioide/corticoide sistêmico fora do plano
conservador da tendinopatia patelar.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config

COLS_VITAIS = ["fc_bpm", "pas_mmhg", "pad_mmhg", "spo2_pct", "dor_eva_0_10"]

# Limiares clínicos didáticos (não são protocolo hospitalar real)
LIMIARES = {
    "fc_alta": 120,
    "fc_baixa": 50,
    "pas_alta": 150,
    "pad_alta": 95,
    "spo2_baixa": 94,
    "dor_alta": 8,
}

# Medicamentos que, ativos no contexto de tendinopatia patelar inicial, merecem alerta
MEDS_ALERTA = {
    "tramadol": {
        "severidade": "ALTA",
        "motivo": "Opioide introduzido sem falha documentada de analgesia multimodal "
        "no plano conservador da tendinopatia patelar.",
    },
    "prednisona": {
        "severidade": "ALTA",
        "motivo": "Corticoide sistêmico sem justificativa clara; risco de enfraquecimento "
        "tendíneo / ruptura em tendinopatia.",
    },
    "prednisolona": {
        "severidade": "ALTA",
        "motivo": "Corticoide sistêmico sem justificativa clara; risco de enfraquecimento "
        "tendíneo / ruptura em tendinopatia.",
    },
}


def _severidade_vital(row: pd.Series, flags: list[str]) -> str:
    if any(f in flags for f in ("fc_baixa", "spo2_baixa")) and row["spo2_pct"] < 92:
        return "CRITICA"
    if "dor_alta" in flags and row["dor_eva_0_10"] >= 9:
        return "ALTA"
    if len(flags) >= 3:
        return "ALTA"
    if len(flags) >= 2 or "fc_alta" in flags or "pas_alta" in flags:
        return "MEDIA"
    return "BAIXA"


def _flags_clinicos(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if row["fc_bpm"] >= LIMIARES["fc_alta"]:
        flags.append("fc_alta")
    if row["fc_bpm"] <= LIMIARES["fc_baixa"]:
        flags.append("fc_baixa")
    if row["pas_mmhg"] >= LIMIARES["pas_alta"]:
        flags.append("pas_alta")
    if row["pad_mmhg"] >= LIMIARES["pad_alta"]:
        flags.append("pad_alta")
    if row["spo2_pct"] <= LIMIARES["spo2_baixa"]:
        flags.append("spo2_baixa")
    if row["dor_eva_0_10"] >= LIMIARES["dor_alta"]:
        flags.append("dor_alta")
    return flags


def detectar_vitais(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    X = df[COLS_VITAIS].astype(float).values

    # Isolation Forest — contamination estimada pelos outliers sintéticos (~4/23)
    n = len(df)
    contamination = min(0.25, max(0.1, 4 / max(n, 1)))
    modelo = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    pred = modelo.fit_predict(X)  # -1 = anomalia
    scores = modelo.decision_function(X)  # menor = mais anômalo

    # Z-score por coluna (sessão como população)
    z = (df[COLS_VITAIS] - df[COLS_VITAIS].mean()) / df[COLS_VITAIS].std(ddof=0).replace(0, 1)
    z_abs_max = z.abs().max(axis=1)

    alertas: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        flags = _flags_clinicos(row)
        is_iforest = bool(pred[i] == -1)
        is_z = bool(z_abs_max.iloc[i] >= 2.5)
        if not (is_iforest or is_z or flags):
            continue

        metodos = []
        if is_iforest:
            metodos.append("isolation_forest")
        if is_z:
            metodos.append("zscore")
        if flags:
            metodos.append("limiar_clinico")

        severidade = _severidade_vital(row, flags) if flags else ("MEDIA" if is_iforest else "BAIXA")
        alertas.append(
            {
                "fonte": "sinais_vitais",
                "timestamp": row["timestamp"].isoformat(sep=" "),
                "paciente_id": row.get("paciente_id", ""),
                "severidade": severidade,
                "metodos": metodos,
                "flags_clinicos": flags,
                "valores": {c: float(row[c]) for c in COLS_VITAIS},
                "isolation_forest_score": round(float(scores[i]), 4),
                "zscore_abs_max": round(float(z_abs_max.iloc[i]), 3),
                "contexto_sessao": str(row.get("contexto_sessao", "")),
                "nota_clinica": str(row.get("nota_clinica", "")),
                "anomalia_esperada_dataset": str(row.get("anomalia_esperada", "")),
            }
        )

    # Ordena: CRITICA > ALTA > MEDIA > BAIXA, depois timestamp
    ordem = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAIXA": 3}
    alertas.sort(key=lambda a: (ordem.get(a["severidade"], 9), a["timestamp"]))

    esperadas = int((df["anomalia_esperada"].astype(str).str.lower() == "sim").sum()) if "anomalia_esperada" in df.columns else None
    return {
        "arquivo": csv_path.name,
        "n_registros": int(n),
        "contamination_iforest": round(contamination, 3),
        "n_alertas": len(alertas),
        "n_anomalias_esperadas_dataset": esperadas,
        "alertas": alertas,
    }


def detectar_prescricoes(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    alertas: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        med = str(row.get("medicamento", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        med_key = med.casefold().split()[0] if med else ""

        # Só alerta introdução ativa de meds de risco
        regra = None
        for chave, meta in MEDS_ALERTA.items():
            if chave in med.casefold():
                regra = meta
                med_key = chave
                break

        if regra is None:
            continue
        if status != "ativo":
            continue

        alertas.append(
            {
                "fonte": "prescricoes",
                "timestamp": str(row.get("data", "")),
                "paciente_id": str(row.get("paciente_id", "")),
                "severidade": regra["severidade"],
                "metodos": ["regra_prescricao"],
                "medicamento": med,
                "dose": str(row.get("dose", "")),
                "via": str(row.get("via", "")),
                "frequencia": str(row.get("frequencia", "")),
                "status": status,
                "motivo": regra["motivo"],
                "motivo_clinico_dataset": str(row.get("motivo_clinico", "")),
                "anomalia_esperada_dataset": str(row.get("anomalia_esperada", "")),
            }
        )

    ordem = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAIXA": 3}
    alertas.sort(key=lambda a: (ordem.get(a["severidade"], 9), a["timestamp"]))

    esperadas = (
        int((df["anomalia_esperada"].astype(str).str.lower() == "sim").sum())
        if "anomalia_esperada" in df.columns
        else None
    )
    return {
        "arquivo": csv_path.name,
        "n_registros": int(len(df)),
        "n_alertas": len(alertas),
        "n_anomalias_esperadas_dataset": esperadas,
        "alertas": alertas,
    }


def analisar_anomalias(
    vitais_path: Path | None = None,
    prescricoes_path: Path | None = None,
) -> dict[str, Any]:
    vitais_path = vitais_path or (config.VITAIS_DIR / "sinais_vitais_PAC-JOELHO-001.csv")
    prescricoes_path = prescricoes_path or (
        config.PRESCRICOES_DIR / "prescricoes_PAC-JOELHO-001.csv"
    )

    vitais = detectar_vitais(vitais_path)
    presc = detectar_prescricoes(prescricoes_path)
    todos = vitais["alertas"] + presc["alertas"]
    ordem = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAIXA": 3}
    todos.sort(key=lambda a: (ordem.get(a["severidade"], 9), str(a.get("timestamp", ""))))

    por_sev: dict[str, int] = {}
    for a in todos:
        por_sev[a["severidade"]] = por_sev.get(a["severidade"], 0) + 1

    return {
        "paciente_id": "PAC-JOELHO-001",
        "tema": "Detecção de anomalias — vitais e prescrições",
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "metodos": {
            "vitais": ["isolation_forest", "zscore", "limiar_clinico"],
            "prescricoes": ["regra_prescricao_opioide_corticoide"],
        },
        "resumo": {
            "n_alertas_total": len(todos),
            "por_severidade": por_sev,
            "n_alertas_vitais": vitais["n_alertas"],
            "n_alertas_prescricoes": presc["n_alertas"],
        },
        "vitais": vitais,
        "prescricoes": presc,
        "alertas": todos,
    }


def salvar_relatorio(
    resultado: dict[str, Any], saida_dir: Path | None = None
) -> tuple[Path, Path]:
    saida_dir = saida_dir or config.SAIDAS_DIR
    saida_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = saida_dir / f"relatorio_anomalias_{ts}.json"
    txt_path = saida_dir / f"relatorio_anomalias_{ts}.txt"

    json_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    linhas = [
        "RELATÓRIO DE ANOMALIAS — vitais + prescrições",
        f"Paciente: {resultado['paciente_id']}",
        f"Gerado em: {resultado['gerado_em']}",
        f"Total de alertas: {resultado['resumo']['n_alertas_total']}",
        f"Por severidade: {resultado['resumo']['por_severidade']}",
        "",
        "=== ALERTAS ===",
    ]
    for a in resultado["alertas"]:
        if a["fonte"] == "sinais_vitais":
            linhas.append(
                f"[{a['severidade']}] {a['timestamp']} | vitais | "
                f"flags={a.get('flags_clinicos')} | "
                f"FC={a['valores']['fc_bpm']} PAS={a['valores']['pas_mmhg']} "
                f"SpO2={a['valores']['spo2_pct']} dor={a['valores']['dor_eva_0_10']} | "
                f"metodos={a['metodos']}"
            )
        else:
            linhas.append(
                f"[{a['severidade']}] {a['timestamp']} | prescricão | "
                f"{a.get('medicamento')} {a.get('dose')} ({a.get('status')}) | "
                f"{a.get('motivo', '')[:80]}"
            )
    txt_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return json_path, txt_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detecção de anomalias (vitais + prescrições)")
    parser.add_argument("--vitais", type=Path, default=None)
    parser.add_argument("--prescricoes", type=Path, default=None)
    args = parser.parse_args(argv)
    resultado = analisar_anomalias(args.vitais, args.prescricoes)
    json_path, txt_path = salvar_relatorio(resultado)
    preview = {
        k: v
        for k, v in resultado.items()
        if k not in {"vitais", "prescricoes", "alertas"}
    }
    preview["alertas_resumo"] = [
        {
            "severidade": a["severidade"],
            "fonte": a["fonte"],
            "timestamp": a.get("timestamp"),
            "detalhe": a.get("flags_clinicos") or a.get("medicamento"),
        }
        for a in resultado["alertas"]
    ]
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"\nRelatórios:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
