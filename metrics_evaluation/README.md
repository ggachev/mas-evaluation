# Metrics Evaluation

Evaluierungs-Pipeline für autonome Coding-Agenten (Masterarbeit).
Bewertet Agenten-Trajektorien mit deterministischen Metriken und LLM-as-a-Judge.

---

## Verzeichnisstruktur

```
metrics_evaluation/
├── metrics_evaluation.py         # Haupt-Evaluationsskript (einzelne Trajektorie)
├── batch_evaluation.py           # Stapelverarbeitung aller Issues
├── consolidate_results.py        # Fasst eval_*.json zu consolidated_results.csv zusammen
├── evaluation_data_models.py     # Datenklassen: StandardStep, EvaluationTrace
├── evaluation_prompts.py         # LLM-Judge Prompts (aktuelle Version)
├── evaluation_prompts_v1.py      # Backup: ursprüngliche Prompts
├── evaluation_prompts_v2.py      # Alternative Prompts-Version
├── review_trace.py               # Hilfsskript zur Anzeige von Trajektorien
│
├── agent_parsers/
│   ├── openhands_parser.py       # Parser für OpenHands JSON-Logs
│   ├── sweagent_parser.py        # Parser für SWE-Agent .traj-Dateien
│   ├── metagpt_parser.py         # Parser für MetaGPT .txt/.log-Dateien
│   └── live_sweagent_parser.py   # Parser für Live-SWE-Agent
│
├── descriptive_analysis.py       # Deskriptive Analyse + Plots (RQ1, RQ3)
├── annotation_analysis.py        # Visualisierung manueller Annotationen (RQ2)
├── spearman_correlation.py       # Spearman-Korrelation Auto vs. Mensch (RQ2)
├── kappa_sample_rate_comparison.py  # Weighted Kappa: SR-Stabilität + Cross-Model (RQ4, RQ5)
├── auc_predictor_analysis.py     # AUC-Prädiktor-Analyse (RQ4.3)
├── plot_spearman_sr_rq4.py       # Spearman SR=1 vs. SR=5 Vergleich (RQ4.2)
│
├── manual_annotations.csv        # Manuelle Expertenbewertungen (Rater 1)
├── manual_annotations_2.csv      # Manuelle Expertenbewertungen (Rater 2)
├── Labeling_Guide.md             # Bewertungskriterien für alle Metriken
│
└── evaluation_results/
    ├── eval_*.json               # Einzelergebnisse je Agent + Issue
    ├── 1_step_gptoss120b/        # GPT-OSS-120b, SR=1 (Standard)
    ├── 1_step_qwen3_235b/        # Qwen3-235b, SR=1
    ├── 1_step_gpt4omini_8b/      # GPT-4o-mini-8b, SR=1
    ├── default_gptoss120b/       # GPT-OSS-120b, SR=5 (Reduziert)
    ├── default_qwen3_235b/       # Qwen3-235b, SR=5
    └── default_gpt4omini_8b/     # GPT-4o-mini-8b, SR=5
```

Jedes `evaluation_results/<run>/`-Verzeichnis enthält:
- `consolidated_results.csv` — aggregierte Scores aller Agenten und Issues
- `spearman_correlation_results.csv` — Spearman-ρ je Metrik vs. Gold Standard
- Abbildungen als `.png`, `.pdf`, `.pgf` und `.tex`

---

## Setup

```bash
cd mas-evaluation/metrics_evaluation
python3 -m venv venv
source venv/bin/activate
pip install pandas scipy scikit-learn matplotlib sentence-transformers
```

API-Schlüssel als Umgebungsvariable setzen:

```bash
export HELMHOLTZ_API_KEY=<key>
# oder
export OPENAI_API_KEY=<key>
```

---

## Evaluation ausführen

### Einzelne Trajektorie

```bash
python3 metrics_evaluation.py <trajectory_file> --agent OpenHands
python3 metrics_evaluation.py <file.traj> --agent SWE-Agent
python3 metrics_evaluation.py <file.txt> --agent MetaGPT --mas --global-plan
```

### Stapelverarbeitung (alle Issues)

```bash
# OpenHands
python3 batch_evaluation.py --agent OpenHands --logs-dir ../logs/openhands/logs

# SWE-Agent
python3 batch_evaluation.py --agent SWE-Agent --logs-dir ../logs/swe-agent

# MetaGPT (Multi-Agent + Global Plan)
python3 batch_evaluation.py --agent MetaGPT --logs-dir ../logs/metagpt --mas --global-plan

# live-swe-agent
python3 batch_evaluation.py --agent live-swe-agent --logs-dir ../logs/live-swe-agent

# Mit angepasster Sampling-Rate
python3 batch_evaluation.py --agent SWE-Agent --logs-dir ../logs/swe-agent --sample-rate 1
```

### Ergebnisse konsolidieren

```bash
python3 consolidate_results.py
```

Liest alle `eval_*.json` im aktuellen Verzeichnis und schreibt `consolidated_results.csv`.

### Trajektorie anzeigen

```bash
python3 review_trace.py --agent OpenHands --task scikit-learn__scikit-learn-12585
```

---

## Analyse und Validierung

### Deskriptive Analyse (RQ1, RQ3)

```bash
source venv/bin/activate
python3 descriptive_analysis.py
```

### Manuelle Annotationen visualisieren (RQ2)

```bash
python3 annotation_analysis.py
```

### Spearman-Korrelation Auto vs. Mensch (RQ2)

```bash
python3 spearman_correlation.py
```

Liest `evaluation_results/consolidated_results.csv` und `manual_annotations.csv`.
Schreibt `spearman_correlation_results.csv` in das aktuelle Ergebnisverzeichnis.

### Weighted Kappa: SR-Stabilität + Cross-Model (RQ4.2, RQ5)

```bash
python3 kappa_sample_rate_comparison.py
```

### AUC-Prädiktor-Analyse (RQ4.3)

```bash
python3 auc_predictor_analysis.py
```

Liest aus `evaluation_results/1_step_gptoss120b/consolidated_results.csv`.

### Spearman SR=1 vs. SR=5 (RQ4.2)

```bash
source venv/bin/activate
python3 plot_spearman_sr_rq4.py
```

---

## Abbildungen

Alle Skripte speichern Abbildungen als `.png`, `.pdf`, `.pgf` und `.tex`.
Die `.tex`-Datei enthält `\includegraphics[width=\linewidth]{name.pdf}` und kann
in LaTeX direkt per `\input{name.tex}` eingebunden werden.

### Abbildungen je Skript

| Skript | Erzeugte Abbildungen |
|--------|----------------------|
| `descriptive_analysis.py` | `cost_benefit_scatter`, `radar_chart_agents`, `boxplots_per_metric`, `boxplots_per_metric_4x2`, `agent_comparison_heatmap`, `mas_metrics_detail`, `mas_metrics_detail_2x2`, `success_vs_failure_comparison`, `metric_correlation_matrix` |
| `spearman_correlation.py` | `correlation_scatterplots`, `correlation_barchart` |
| `annotation_analysis.py` | `annotation_distribution`, `annotation_distribution_by_agent`, `annotation_heatmap_by_agent`, `annotation_spread_heatmap`, `interrater_agreement` |
| `plot_spearman_sr_rq4.py` | `spearman_sr_rq4` |
| `auc_predictor_analysis.py` | `auc_predictor_plot` |
| `kappa_sample_rate_comparison.py` | `kappa_sample_rate_comparison`, `spearman_sample_rate_comparison`, `kappa_cross_model_default_(sr=5)`, `kappa_cross_model_1_step_(sr=1)` |

---

## TikZ-Abbildungen (Kapitel 2–4)

Die konzeptuellen Abbildungen der Arbeit liegen als TikZ-Quellcode in `../figures/`.

```bash
cd ../figures

./build_figure.sh fig_blackbox_vs_glassbox
./build_figure.sh fig_agent_architecture
./build_figure.sh fig_agent_evaluation
./build_figure.sh fig_eval_pipeline
./build_figure.sh fig_stratified_sample
./build_figure.sh fig_pipeline_flow
./build_figure.sh fig_prompt_anatomy
```

`build_figure.sh` erzeugt je Abbildung eine `.png` (300 dpi) und eine `.pgf`-Datei.

| Datei | Inhalt |
|-------|--------|
| `fig_blackbox_vs_glassbox` | Black-Box vs. Glass-Box Vergleich, symmetrisches Layout mit Trennlinie mittig |
| `fig_agent_architecture` | LLM-Agenten-Architektur, 4 Spalten (Wang 2024) |
| `fig_agent_evaluation` | Evaluationsdimensionen (Siegmund 2025), 7 Fragen mit geraden Pfeilen |
| `fig_eval_pipeline` | 3-Spalten-Pipeline (Ausführung → Transformation → Analyse), waagerechte Pfeile mit `\|-`-Trick |
| `fig_stratified_sample` | Pyramide Easy/Medium/Hard, Hard-Label innen, N=15 Aufgaben |
| `fig_pipeline_flow` | Flowchart mit 6 Schritten und TikZ-Icons |
| `fig_prompt_anatomy` | Sandwich-Blockdiagramm System-Prompt, alle Boxen gleich hell, Text bündig ausgerichtet |

---

## CLI-Argumente (batch_evaluation.py)

| Argument | Standard | Beschreibung |
|----------|----------|--------------|
| `--agent` | Pflicht | `OpenHands`, `SWE-Agent`, `MetaGPT`, `live-swe-agent` |
| `--logs-dir` | Pflicht | Verzeichnis mit Trajektorie-Dateien |
| `--mas` | `false` | Multi-Agenten-Modus aktivieren |
| `--global-plan` | `false` | Metrik M2.3 aktivieren (nur sinnvoll für MetaGPT) |
| `--sample-rate` | `5` | Sampling-Rate für M3.1 und M3.2 |
| `--context-window-steps` | `8` | Fenstergröße für M4.1 |
| `--context-sample-rate` | `4` | Sampling-Rate für M4.1 |
| `--start-from` | `1` | Ab Issue N starten |
| `--only-issue` | – | Nur ein bestimmtes Issue auswerten |
| `--prompts-file` | – | Pfad zu alternativer Prompts-Datei |
