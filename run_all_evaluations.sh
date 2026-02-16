#!/bin/bash
set -e  # Script bricht ab, sobald ein Befehl fehlschlägt

echo "🚀 Starte Automatische Metrics Evaluation"

############################
# OpenHands
############################
echo "▶ Agent: OpenHands"

python metrics_evaluation/batch_evaluation.py --agent OpenHands --logs-dir logs/openhands/logs --prompts-file metrics_evaluation/evaluation_prompts.py

############################
# SWE-Agent
############################
echo "▶ Agent: SWE-Agent"

python metrics_evaluation/batch_evaluation.py --agent SWE-Agent --logs-dir logs/swe-agent --prompts-file metrics_evaluation/evaluation_prompts.py

############################
# live-swe-agent
############################
echo "▶ Agent: live-swe-agent"

python metrics_evaluation/batch_evaluation.py --agent live-swe-agent --logs-dir logs/live-swe-agent --prompts-file metrics_evaluation/evaluation_prompts.py

############################
# MetaGPT (MAS)
############################
echo "▶ Agent: MetaGPT (MAS)"

python metrics_evaluation/batch_evaluation.py --agent MetaGPT --logs-dir logs/metagpt --mas --global-plan --prompts-file metrics_evaluation/evaluation_prompts.py

echo "✅ Alle Evaluations erfolgreich abgeschlossen"

