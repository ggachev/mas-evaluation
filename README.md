# Evaluation autonomer Coding-Agenten

Dieses Repository enthält die vollständige Evaluierungs-Pipeline der Masterarbeit.
Es bewertet Agenten-Trajektorien von vier Systemen (OpenHands, SWE-agent, Live-SWE-agent, MetaGPT)
mit deterministischen Metriken und LLM-as-a-Judge.

Detaillierte Dokumentation der Analyse-Skripte und Abbildungen:
→ [`metrics_evaluation/README.md`](metrics_evaluation/README.md)

---

## Verzeichnisstruktur

```
mas-evaluation/
├── agent_systems/               # Quellcode der Agentensysteme (Referenz-Implementierungen)
│   ├── openhands/
│   ├── SWE-agent/
│   ├── live-swe-agent/
│   ├── metagpt/
│   ├── SWE-bench/
│   ├── mini-swe-agent/
│   └── chatdev/
│
├── logs/                        # Rohe Agenten-Trajektorien (Log-Dateien)
│   ├── openhands/
│   ├── swe-agent/
│   ├── live-swe-agent/
│   └── metagpt/
│
├── diffs/                       # Generierte Patches je Agent und Issue
│   ├── openhands/
│   ├── swe-agent/
│   ├── live-swe-agent/
│   └── metagpt/
│
├── swe_bench_verified_issues/   # Die 15 ausgewählten SWE-bench Verified Issues
│
├── metrics_evaluation/          # Evaluierungs-Pipeline (Hauptmodul)
│   ├── metrics_evaluation.py    # Haupt-Evaluationsskript
│   ├── batch_evaluation.py      # Stapelverarbeitung aller Issues
│   ├── consolidate_results.py   # Fasst eval_*.json zu consolidated_results.csv zusammen
│   ├── agent_parsers/           # Parser für die verschiedenen Log-Formate
│   ├── descriptive_analysis.py  # Deskriptive Analyse + Plots (RQ1, RQ3)
│   ├── annotation_analysis.py   # Visualisierung manueller Annotationen (RQ2)
│   ├── spearman_correlation.py  # Spearman-Korrelation Auto vs. Mensch (RQ2)
│   ├── kappa_sample_rate_comparison.py  # Weighted Kappa SR + Cross-Model (RQ4, RQ5)
│   ├── auc_predictor_analysis.py        # AUC-Prädiktor-Analyse (RQ4.3)
│   ├── plot_spearman_sr_rq4.py          # Spearman SR=1 vs. SR=5 (RQ4.2)
│   ├── manual_annotations.csv   # Manuelle Expertenbewertungen (Rater 1)
│   ├── evaluation_results/      # Ergebnisse, CSVs und Abbildungen
│   │   ├── 1_step_gptoss120b/   # GPT-OSS-120b, SR=1 (Standard-Konfiguration)
│   │   ├── 1_step_qwen3_235b/   # Qwen3-235b, SR=1
│   │   ├── 1_step_gpt4omini_8b/ # GPT-4o-mini-8b, SR=1
│   │   ├── default_gptoss120b/  # GPT-OSS-120b, SR=5 (Reduzierte Konfiguration)
│   │   ├── default_qwen3_235b/  # Qwen3-235b, SR=5
│   │   └── default_gpt4omini_8b/# GPT-4o-mini-8b, SR=5
│   └── README.md                # Detaillierte Dokumentation
│
└── figures/                     # TikZ-Abbildungen (Kapitel 2–4)
    ├── build_figure.sh          # Build-Skript → erzeugt PNG + PGF
    ├── fig_blackbox_vs_glassbox.tex
    ├── fig_agent_architecture.tex
    ├── fig_agent_evaluation.tex
    ├── fig_eval_pipeline.tex
    ├── fig_stratified_sample.tex
    ├── fig_pipeline_flow.tex
    └── fig_prompt_anatomy.tex
```

---

## Setup

```bash
cd mas-evaluation/metrics_evaluation
python3 -m venv venv
source venv/bin/activate
pip install openai pandas scipy scikit-learn matplotlib sentence-transformers

export HELMHOLTZ_API_KEY="<key>"
# oder
export OPENAI_API_KEY="<key>"
```

---

## Evaluation ausführen

```bash
cd mas-evaluation

# Einzelne Trajektorie
python metrics_evaluation/metrics_evaluation.py <trajectory_file> --agent OpenHands

# Stapelverarbeitung
python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands/logs
python metrics_evaluation/batch_evaluation.py --agent SWE-Agent --logs-dir logs/swe-agent
python metrics_evaluation/batch_evaluation.py --agent live-swe-agent --logs-dir logs/live-swe-agent
python metrics_evaluation/batch_evaluation.py --agent MetaGPT --logs-dir logs/metagpt --mas --global-plan

# Ergebnisse konsolidieren
cd metrics_evaluation
python consolidate_results.py
```

---

## Metriken

### Kategorie 1: Ergebnisse und Kosten

| Metrik | Typ | Beschreibung |
|--------|-----|--------------|
| M1.1 Task Success Rate | Manuell | Binärer Erfolg aus manuellen Labels |
| M1.2 Resource Efficiency | Deterministisch | Kosten, Token, Dauer, Schrittanzahl |

### Kategorie 2: Strategie und Navigation

| Metrik | Typ | Beschreibung |
|--------|-----|--------------|
| M2.1 Loop Detection | Deterministisch | Hash-basierte Erkennung wiederholter Sequenzen |
| M2.2 Trajectory Efficiency | LLM-Judge | Effizienz des Lösungswegs |
| M2.3 Global Strategy Consistency | LLM-Judge | Planformulierung und -einhaltung (nur MetaGPT) |
| M2.4 Stepwise Reasoning Quality | LLM-Judge | Logische Qualität je Schritt |
| M2.5 Role Adherence | LLM-Judge | Einhaltung der Agenten-Rolle |

### Kategorie 3: Werkzeuge

| Metrik | Typ | Beschreibung |
|--------|-----|--------------|
| M3.1 Tool Selection Quality | LLM-Judge | Angemessenheit der Werkzeugwahl |
| M3.2 Tool Execution Success | LLM-Judge | Technische Ausführungsrate |
| M3.3 Tool Usage Efficiency | Deterministisch | Kontext-Verschmutzungs-Messung |

### Kategorie 4: Wissen und Kontext

| Metrik | Typ | Beschreibung |
|--------|-----|--------------|
| M4.1 Context Utilization | LLM-Judge | Konsistenz im Sliding-Window |

### Kategorie 5: Multi-Agenten-Systeme (nur MetaGPT)

| Metrik | Typ | Beschreibung |
|--------|-----|--------------|
| M5.1 Communication Efficiency | LLM-Judge | Signal-Rausch-Verhältnis der Kommunikation |
| M5.2 Information Diversity | Embeddings | Diversität der Agenten-Nachrichten |
| M5.3 Path Redundancy | Deterministisch | Ping-Pong-Mustererkennung |
| M5.4 Agent Invocation Distribution | Deterministisch | Arbeitsverteilung (Shannon-Entropie) |

---

## Ausgabeformat

Ergebnisse werden in `evaluation_results/` gespeichert:

```json
{
  "meta": {
    "agent": "SWE-Agent",
    "task": "scikit-learn__scikit-learn-12585",
    "timestamp": "2025-12-28 17:23:44",
    "is_multi_agent_system": false,
    "llm_judge_model": "GPT-OSS-120b"
  },
  "metric_1_1_task_success_rate": {"success": true, "source": "manual_labels"},
  "metric_1_2_resource_efficiency": {"total_cost_usd": 0.017, "total_tokens": 68524},
  "metric_2_2_trajectory_efficiency": {"score": 0.85, "reasoning": "..."},
  "metric_5_1_communication_efficiency": "N/A - Single Agent"
}
```

---

## Unterstützte Agenten-Formate

| Agent | Format | Besonderheit |
|-------|--------|--------------|
| OpenHands | JSON (`history`-Array) | Kosten aus `metrics.accumulated_cost` |
| SWE-Agent | `.traj` (JSON) + `.config.yaml` | Aufgabe aus `problem_statement.text` |
| Live-SWE-Agent | `.traj` + `.config.yaml` | Ähnlich SWE-Agent |
| MetaGPT | `.txt` / `.log` | Multi-Agenten-Erkennung aus `AgentName(Role)`-Mustern |

---

## LLM-Judge Konfiguration

```python
BASE_URL_JUDGE = "https://api.helmholtz-blablador.fz-juelich.de/v1"
MODEL_JUDGE    = "1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
MODEL_EMBEDDING = "text-embedding-3-small"
CONTEXT_WINDOW_SIZE = 131000
```
