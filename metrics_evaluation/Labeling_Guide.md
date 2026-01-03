# Labeling Guide für die Manuelle Annotation

Dieses Dokument dient als Leitfaden für die manuelle Bewertung ("Gold Standard") der Agenten-Traces. Bitte bewerte jeden Durchlauf (Trace) ganzheitlich auf einer Skala von 1 bis 5.

## Bewertungsskala (Generisch)
*   **1 - Sehr Schlecht:** Der Agent versagt komplett in diesem Aspekt. (z.B. nur Syntaxfehler, totaler Logikverlust).
*   **2 - Schlecht:** Überwiegend fehlerhaft oder ineffizient, aber mit erkennbarem Ansatz.
*   **3 - Mittelmäßig:** Akzeptable Leistung mit klaren Schwächen. Durchschnitt.
*   **4 - Gut:** Überwiegend kompetent, nur kleine Fehler.
*   **5 - Exzellent:** Nahezu perfektes Verhalten in diesem Aspekt.

---

## Metrik-Definitionen

### M1.1 Success (Bool)
*   **0 (Nein):** Task nicht gelöst (Tests fail).
*   **1 (Ja):** Task gelöst (Tests pass).
*   *Hinweis:* Dies ist meist objektiv aus den Logs (`pytest` output) ersichtlich.

### M2.2 Trajectory Efficiency (Effizienz des Lösungsweges)
Bewertet, wie zielgerichtet der Agent vorging.
*   **1:** Agent irrt völlig planlos umher, liest wahllos Dateien, dreht sich im Kreis.
*   **3:** Agent findet den Weg, macht aber viele unnötige Zwischenschritte (z.B. unnötige Tests, redundantes Lesen).
*   **5:** Agent navigiert direkt zur Lösung ("Laser-Fokus"), jeder Schritt bringt ihn dem Ziel näher.

### M2.3 Global Strategy (Strategische Planung)
Bewertet, ob ein erkennbarer Gesamtplan existiert und eingehalten wird.
*   **1:** Kein Plan erkennbar, rein reaktives "Wursteln".
*   **3:** Plan vorhanden, aber Agent verliert ihn zwischendurch aus den Augen oder passt ihn nicht an.
*   **5:** Agent erstellt initialen Plan, arbeitet ihn ab und aktualisiert ihn sinnvoll bei neuen Erkenntnissen.

### M2.4 Reasoning Quality (Logische Schlüssigkeit)
Bewertet die Qualität der `Thought` -> `Action` Kette. Macht der Schritt Sinn?
*   **1:** Halluzinationen, Non-Sequiturs (Gedanke passt null zur Aktion).
*   **3:** Meist logisch, aber manchmal voreilige Schlüsse oder Missverständnisse der Observation.
*   **5:** Messerscharfe Logik. Jede Aktion ist perfekt aus der vorherigen Beobachtung abgeleitet.

### M2.5 Role Adherence (Rollentreue)
Bewertet, ob der Agent sich an seine System-Instruktionen hält.
*   **1:** Agent vergisst seine Rolle, fragt den User nach Hilfe (obwohl er autonom sein soll), verweigert Coding.
*   **5:** Agent bleibt strikt "in Character" (z.B. als Senior Engineer), befolgt alle Formatvorgaben.

### M3.1 Tool Selection (Werkzeugwahl)
Bewertet die *taktische* Wahl des Tools.
*   **1:** Wählt völlig falsche Tools (z.B. `edit` statt `read` zum Lesen).
*   **3:** Wählt funktionierende, aber ineffiziente Tools (z.B. liest riesige Datei komplett statt `grep` zu nutzen).
*   **5:** Wählt immer das optimale, effizienteste Tool für das Teilproblem.

### M3.2 Tool Execution Quality (Technische Ausführung)
Bewertet die *technische* Kompetenz (Syntaxfehler, Crashes).
*   **1:** Ständige Syntaxfehler, falsche Argumente, "Command not found".
*   **5:** Jeder Befehl sitzt beim ersten Versuch. Keine Syntaxfehler.

### M4.1 Context Utilization (Kontext-Nutzung)
Bewertet, ob der Agent Informationen im Gedächtnis behält.
*   **1:** Vergisst Dinge, die er 2 Schritte vorher gelesen hat. Liest dieselbe Datei 3-mal.
*   **5:** Erinnert sich perfekt an alle Details, nutzt Wissen aus früheren Schritten effizient.

### M5.1 Communication (Nur für Multi-Agent Systeme)
Bewertet die Qualität der Kommunikation zwischen den Agenten.
*   **1:** Sinnloses "Ping-Pong", leere Nachrichten, endloses Danke-Sagen.
*   **5:** Hochdichte Informationsübertragung, klare Arbeitsverteilung, konstruktives Feedback.
*   *Lasse dieses Feld leer für Single-Agent Systeme (SWE-agent, OpenHands).*
