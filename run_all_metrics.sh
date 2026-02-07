#!/bin/bash
set -e  # Script bricht ab, sobald ein Befehl fehlschlägt

echo "🚀 Starte Metrics Evaluation"

############################
# OpenHands
############################
echo "▶ Agent: OpenHands"

python metrics_evaluation/metrics_evaluation.py logs/openhands/logs/issue_6/sympy__sympy-13480.json --agent OpenHands
python metrics_evaluation/metrics_evaluation.py logs/openhands/logs/issue_4/matplotlib__matplotlib-22719.json --agent OpenHands
python metrics_evaluation/metrics_evaluation.py logs/openhands/logs/issue_1/scikit-learn__scikit-learn-12585.json --agent OpenHands
python metrics_evaluation/metrics_evaluation.py logs/openhands/logs/issue_10/django__django-16901.json --agent OpenHands

############################
# SWE-Agent
############################
echo "▶ Agent: SWE-Agent"

python metrics_evaluation/metrics_evaluation.py logs/swe-agent/issue_6/sympy__sympy-13480.traj --agent SWE-Agent
python metrics_evaluation/metrics_evaluation.py logs/swe-agent/issue_4/matplotlib__matplotlib-22719.traj --agent SWE-Agent
python metrics_evaluation/metrics_evaluation.py logs/swe-agent/issue_1/scikit-learn__scikit-learn-12585.traj --agent SWE-Agent
python metrics_evaluation/metrics_evaluation.py logs/swe-agent/issue_10/django__django-16901.traj --agent SWE-Agent

############################
# live-swe-agent
############################
echo "▶ Agent: live-swe-agent"

python metrics_evaluation/metrics_evaluation.py logs/live-swe-agent/issue_6/sympy__sympy-13480.traj.json --agent live-swe-agent
python metrics_evaluation/metrics_evaluation.py logs/live-swe-agent/issue_4/matplotlib__matplotlib-22719.traj.json --agent live-swe-agent
python metrics_evaluation/metrics_evaluation.py logs/live-swe-agent/issue_1/scikit-learn__scikit-learn-12585.traj.json --agent live-swe-agent
python metrics_evaluation/metrics_evaluation.py logs/live-swe-agent/issue_10/django__django-16901.traj.json --agent live-swe-agent

############################
# MetaGPT (MAS)
############################
echo "▶ Agent: MetaGPT (MAS)"

python metrics_evaluation/metrics_evaluation.py logs/metagpt/issue_6/sympy__sympy-13480.txt --agent MetaGPT --mas
python metrics_evaluation/metrics_evaluation.py logs/metagpt/issue_4/matplotlib__matplotlib-22719.txt --agent MetaGPT --mas
python metrics_evaluation/metrics_evaluation.py logs/metagpt/issue_1/scikit-learn__scikit-learn-12585.txt --agent MetaGPT --mas
python metrics_evaluation/metrics_evaluation.py logs/metagpt/issue_10/django__django-16901.txt --agent MetaGPT --mas

echo "✅ Alle Evaluations erfolgreich abgeschlossen"

