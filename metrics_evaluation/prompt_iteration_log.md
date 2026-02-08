# Prompt-Iterationslog: M3.1, M2.3, M2.5

## Ausgangslage (V1 — Original-Prompts)

### Spearman-Korrelation (alle 60 Traces)

| Metrik | ρ | p-Wert | N | Interpretation |
|--------|-----|--------|---|----------------|
| **M3.1** (Tool Selection) | -0.02 | 0.85 | 41 | Very Weak — praktisch keine Korrelation |
| **M2.3** (Global Strategy) | 0.18 | 0.48 | 8 | Very Weak |
| **M2.5** (Role Adherence) | 0.27 | 0.03 | 41 | Weak |

### Ziele

| Metrik | Ziel-ρ | Dev-Set-Kriterium |
|--------|--------|-------------------|
| M3.1 | ≥ 0.35 | ≥ 12/16 Traces innerhalb ±1.5 Likert |
| M2.3 | ≥ 0.35 | ≥ 75% der vergleichbaren Traces innerhalb ±1.5 Likert |
| M2.5 | ≥ 0.40 | ≥ 12/16 Traces innerhalb ±1.5 Likert |

### Identifizierte Root-Causes

**M3.1 (ρ = -0.02):**
1. "Hallucinated Tool"-Fehlklassifikation: `submit` und `str_replace_editor` als halluziniert gewertet (0.0 statt 0.2)
2. Syntax-Fehler mit Tool Selection verwechselt (M3.2-Problem fälschlicherweise M3.1 zugeordnet)
3. Fehlende Trajektorie-Perspektive: Einzelschritte isoliert bewertet
4. Outcome-Disconnect: MetaGPT bekommt hohe Einzelscores, obwohl Workflow scheitert

**M2.3 (ρ = 0.18):**
1. Zu generöse Plan-Erkennung: `Plan.append_task` automatisch als Plan gewertet
2. Loops nicht als Adherence-Problem erkannt
3. Kleines N=8 durch wenig erkannte Pläne

**M2.5 (ρ = 0.27):**
1. Ceiling-Effekt: Fast alle Auto-Scores ≈ 1.0
2. "What's next?"-Fragen am Ende nicht als Violation erkannt
3. "User-Hilfe"-Anfragen (MetaGPT) nicht erkannt

### Dev-Set

4 Issues × 4 Agenten = 16 Traces:
- `sympy__sympy-13480` (issue_6)
- `matplotlib__matplotlib-22719` (issue_4)
- `scikit-learn__scikit-learn-12585` (issue_1)
- `django__django-16901` (issue_10)

---

## Iteration 1 (V1 → V2)

### M3.1 — Änderungen

| Problem | Lösung |
|---------|--------|
| Hallucinated-Fehlklassifikation | CRITICAL DISTINCTIONS Block: Agent-spezifische Commands (submit, str_replace_editor, Plan.append_task, etc.) und Standard-Shell-Commands sind REAL, auch bei falscher Syntax |
| Syntax = M3.1 Problem | Explizite Trennung: "Syntax correctness and parameter errors are evaluated separately (metric 3.2). Here, evaluate ONLY whether the CHOICE of tool was tactically appropriate" |
| Binäres Scoring | Granularere Beschreibungen: 0.1-0.3 = "wrong tool category, or real tool misapplied fundamentally", 0.0 NUR für "command genuinely does not exist" |

**Neuer Prompt-Kern:**
```
CRITICAL DISTINCTIONS:
- "Wrong parameters" or "misused tool" is NOT "hallucinated tool".
  A tool that EXISTS but is called with wrong arguments is POOR usage (0.1-0.3),
  not HALLUCINATION (0.0).
- Agent-specific commands (submit, str_replace_editor, ...) ARE real tools
  even if used incorrectly.
- Only classify as "Hallucinated" (0.0) if the command genuinely does not exist.
```

### M2.3 — Änderungen

| Problem | Lösung |
|---------|--------|
| Zu generöse Plan-Erkennung | Strenge Definition: "A real plan must contain SPECIFIC ordered steps" |
| `Plan.append_task` = Plan | "A single tool call like Plan.append_task with a generic description is NOT a structured plan" |
| Loops ignoriert | "5+ identical tool errors without strategy change = LOW adherence" |
| Falsches Problem = OK | "Working on a completely WRONG problem = LOW adherence" |

### M2.5 — Änderungen

| Problem | Lösung |
|---------|--------|
| Keine spezifischen Violations | Empirische "COMMON VIOLATIONS TO WATCH FOR" Liste hinzugefügt |
| "What's next?" nicht erkannt | Autonomous Operation Breach als Kategorie A definiert (0.3-0.6) |
| User-Hilfe ignoriert | "Can you help me?" explizit als MODERATE to SEVERE gelistet |
| Ceiling-Effekt | 6-stufige Scoring-Skala (1.0, 0.9, 0.7-0.8, 0.5-0.6, 0.3-0.4, 0.0-0.2) |

### Ergebnisse Iteration 1

**Dev-Set (16 Traces):**

| Metrik | Within ±1.5 | Prozent | Ziel |
|--------|------------|---------|------|
| M3.1 | 13/16 | 81% | ≥75% ✅ |
| M2.3 | 1/1 | 100% | ≥75% ✅ (aber N zu klein!) |
| M2.5 | 15/16 | 94% | ≥75% ✅ |

**Spearman-Korrelation (Mixed V1/V2 — 16 Dev-Traces V2, 44 noch V1):**

| Metrik | ρ (V1) | ρ (Mixed) | Δρ |
|--------|--------|-----------|-----|
| M3.1 | -0.02 | 0.07 | +0.09 |
| M2.3 | 0.18 | -0.06 | -0.24 |
| M2.5 | 0.27 | 0.35 | +0.08 |

**Erkannte Probleme:**

1. **M3.1**: MetaGPT-Inflation — `Plan.append_task` bekommt immer 1.0 ("planning tool correct for planning"), auch wenn Plan auf halluziniertes Problem zielt. Sampling (3/15 Aktionen bei django) verpasst schlechte Steps.
2. **M2.3**: Prompt ZU streng — fast alle Scores `null` (kein Plan erkannt). OpenHands und MetaGPT haben implizite Pläne, die ignoriert werden. N sinkt auf 1.
3. **M2.5**: "What's next?" erkannt aber zu hart bestraft — OpenHands django bekommt 0.55 (Likert 3.2) für eine End-of-Task-Frage, Manual gibt 5.

---

## Iteration 2 (V2 → V3)

### M3.1 — Änderungen

| Problem | Lösung |
|---------|--------|
| MetaGPT arbeitet am falschen Problem, Tool-Kategorie aber "korrekt" | Neues Kriterium #1 "Task Relevance": "An action that addresses a completely WRONG problem is Poor (0.1-0.3)" |
| `Plan.append_task` immer Optimal | Neues Kriterium #5 "Planning vs. Doing": "A planning action is only Optimal if the plan content is specific and relevant. A vague or misdirected plan is Suboptimal (0.4-0.6)" |
| Wiederholte Fehler nicht bestraft | Kriterium #4 erweitert: "If the SAME tool already failed with similar errors, selecting it AGAIN without meaningful parameter changes is Suboptimal (0.4-0.5)" |

**Neuer Prompt-Kern (Kriterien):**
```
1. Task Relevance: Is this action working on the CORRECT problem?
   An action that addresses a completely WRONG problem is Poor (0.1-0.3).
2. Tactical Fit: Is this the right CATEGORY of tool?
3. Efficiency: Could a simpler tool achieve the same result?
4. Redundancy: Same tool already failed? → Suboptimal (0.4-0.5).
5. Planning vs. Doing: Vague/misdirected plan → Suboptimal (0.4-0.6).
```

### M2.3 — Änderungen

| Problem | Lösung |
|---------|--------|
| Fast alle Scores `null` | Plan-Definition gelockert: EXPLICIT und IMPLICIT Pläne akzeptiert |
| `Plan.append_task` = kein Plan | "In multi-agent systems, using Plan.append_task to create multiple ordered sub-tasks counts as a plan" |
| Implizite Workflows ignoriert | "If the agent demonstrates a phased approach (investigate → implement → test), this counts as having a plan" |

**Neuer Prompt-Kern (Plan Existence):**
```
Plans can be EXPLICIT or IMPLICIT:
- EXPLICIT: Numbered/ordered steps stated upfront.
- IMPLICIT via task delegation: Plan.append_task with multiple ordered sub-tasks.
- IMPLICIT via structured workflow: Phased approach through early actions.
- NOT a plan: A single vague intention without follow-up structure.
```

### M2.5 — Änderungen

| Problem | Lösung |
|---------|--------|
| "What's next?" am Ende = 0.55 (zu hart) | Severity nach Kontext differenziert: End-of-Task = MINOR (0.7-0.8), Mid-Task = MODERATE (0.3-0.5), Wiederholt = SEVERE (0.2-0.3) |

**Neuer Prompt-Kern (Autonomous Operation Breach):**
```
SEVERITY depends on WHEN and HOW the question occurs:
- Mid-task help requests: → MODERATE to SEVERE (0.3-0.5)
- End-of-task conversational closing ("What's next?"): → MINOR (0.7-0.8)
- Repeated mid-task questions (3+ times): → SEVERE (0.2-0.3)
```

### Ergebnisse Iteration 2

**Dev-Set (16 Traces):**

| Metrik | Within ±1.5 | Prozent | Δ vs. Iter 1 |
|--------|------------|---------|-------------|
| M3.1 | 13/16 | 81% | = (MISSes verbessert) |
| M2.3 | 5/7 | 71% | ↑ (N: 1→7, aber 2 MISSes) |
| M2.5 | 16/16 | 100% | ↑ (von 94%) |

**M3.1 MetaGPT-MISSes Verbesserung:**

| Case | Iter 1 Δ | Iter 2 Δ | Verbesserung |
|------|----------|----------|-------------|
| MG matplotlib | +2.51 | +1.74 | -0.77 |
| MG scikit-learn | +2.39 | +1.93 | -0.46 |
| MG django | +4.00 | +1.80 | -2.20 |

**M2.3 neue MISSes:**
- OpenHands matplotlib: Auto=4.60, Manual=3, Δ=+1.60 (Adherence zu generös: 0.9 = "High")
- MetaGPT matplotlib: Auto=4.60, Manual=3, Δ=+1.60 (gleicher Grund)

**M2.5 Fix:**
- OpenHands django: 0.55 → 0.78 (Likert 3.20 → 4.12), Δ: -1.80 → -0.88 ✅

**Erkannte Probleme:**

1. **M2.3**: Judge gibt 0.9 ("High adherence") für implizite Pläne, obwohl Manual 3/5 gibt. Die geprunte Trace zeigt nicht alle Fehler (z.B. 8 identische Edit-Failures bei MetaGPT matplotlib).

---

## Iteration 3 (V3 → V4) — Nur M2.3

### M2.3 — Änderungen

| Problem | Lösung |
|---------|--------|
| Implizite Pläne bekommen zu hohe Adherence (0.9) | Cap für implizite Pläne: "If the plan was only IMPLICIT, the maximum score is 0.7 (MEDIUM)" |
| Tool-Failures ignoriert bei Adherence | "Any evidence of repeated tool failures indicates MEDIUM adherence at best, not HIGH" |
| HIGH zu leicht erreichbar | "HIGH (0.8-1.0): ONLY for explicit plans with near-flawless execution" |

**Neuer Prompt-Kern (Adherence):**
```
- HIGH (0.8-1.0): ONLY for explicit plans with near-flawless execution.
- MEDIUM (0.4-0.7): If plan was IMPLICIT → max 0.7 even if execution decent.
- LOW (0.1-0.3): Plan abandoned, stuck in repetitive failures.
IMPORTANT: Repeated tool failures or excessive steps = MEDIUM at best.
```

### M3.1, M2.5 — Keine Änderungen

M3.1: Verbleibende MetaGPT-Inflation ist architekturbedingt (Per-Step-Evaluation + Sampling), nicht durch Prompts lösbar.
M2.5: Bereits 100% — keine Änderung nötig.

### Ergebnisse Iteration 3

**Dev-Set (16 Traces):**

| Metrik | Within ±1.5 | Prozent | Δ vs. Iter 2 |
|--------|------------|---------|-------------|
| M3.1 | 13/16 | 81% | = |
| M2.3 | 7/8 | 88% | ↑ (von 71%) |
| M2.5 | 16/16 | 100% | = |

**M2.3 Verbesserung (matplotlib-MISSes gelöst):**

| Case | Iter 2 Auto (Likert) | Iter 3 Auto (Likert) | Manual | Δ Iter 3 |
|------|---------------------|---------------------|--------|----------|
| OH matplotlib | 4.60 | 3.40 | 3 | +0.40 ✅ |
| MG matplotlib | 4.60 | 3.60 | 3 | +0.60 ✅ |
| OH scikit-learn | 4.60 | 3.40 | 5 | -1.60 MISS (neu) |

**Neuer MISS**: OpenHands scikit-learn (Auto=3.40, Manual=5). Der Implicit-Plan-Cap drückt einen gut ausgeführten impliziten Plan zu stark. Dies ist ein Randfall — der Agent hatte keinen expliziten Plan, führte aber fast perfekt aus. Der Cap auf 0.7 begrenzt das korrekt per Definition, kollidiert aber mit der großzügigeren manuellen Bewertung.

---

## Gesamtübersicht: Fortschritt über alle Iterationen

### Dev-Set Within ±1.5 Likert

| Metrik | V1 (Original) | Iter 1 (V2) | Iter 2 (V3) | Iter 3 (V4) |
|--------|:---:|:---:|:---:|:---:|
| **M3.1** | n/a | 13/16 (81%) | 13/16 (81%) | 13/16 (81%) |
| **M2.3** | n/a | 1/1 (100%*) | 5/7 (71%) | 7/8 (88%) |
| **M2.5** | n/a | 15/16 (94%) | 16/16 (100%) | 16/16 (100%) |

*\* N=1 — statistisch nicht aussagekräftig*

### M3.1 MetaGPT-MISSes über Iterationen (Δ in Likert-Punkten)

| Case | Iter 1 | Iter 2 | Iter 3 |
|------|--------|--------|--------|
| MG matplotlib | +2.51 | +1.74 | +1.51 |
| MG scikit-learn | +2.39 | +1.93 | +2.16 |
| MG django | +4.00 | +1.80 | +1.73 |

### Zusammenfassung der Prompt-Änderungen

| Version | M3.1 | M2.3 | M2.5 |
|---------|------|------|------|
| **V1 (Original)** | Overkill, Wrong Tool, Fragile, Redundant | Plan vorhanden? → Adherence | Constraint Violations, Persona, Boundary |
| **V2 (Iter 1)** | + Hallucination-Klarstellung, + Syntax≠Selection | + Strenge Plan-Definition (nur explizit) | + Empirische Violations-Liste, + "What's next?" |
| **V3 (Iter 2)** | + Task Relevance, + Planning vs. Doing, + Repeated Failure | + Implizite Pläne akzeptiert | + Severity nach Kontext (End-of-Task = Minor) |
| **V4 (Iter 3)** | = (keine Änderung) | + Implicit-Plan-Cap (max 0.7), + Failure=Medium | = (keine Änderung) |
