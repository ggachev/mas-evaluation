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
*   *Hinweis:* Dies ist meist objektiv aus den Logs (nach der SWE Bench Evaluation Harness) ersichtlich.

### M2.2 Trajectory Efficiency (Effizienz des Lösungsweges)
Bewertet, wie zielgerichtet der Agent vorging.
*   **1:** Agent irrt völlig planlos umher, liest wahllos Dateien, dreht sich im Kreis.
*   **2:** Gelegentlich zielgerichtete Schritte erkennbar, aber häufige Abschweifungen und redundante Aktionen ohne Systematik.
*   **3:** Agent findet den Weg, macht aber viele unnötige Zwischenschritte (z.B. unnötige Tests, redundantes Lesen).
*   **4:** Agent findet den Lösungsweg mit nur wenigen unnötigen Umwegen; gelegentliche Redundanzen, aber insgesamt strukturiertes Vorgehen.
*   **5:** Agent navigiert direkt zur Lösung ("Laser-Fokus"), jeder Schritt bringt ihn dem Ziel näher.

### M2.3 Global Strategy (Strategische Planung)
Bewertet, ob ein erkennbares strategisches Vorgehen existiert (explizit oder implizit) und eingehalten wird.
*   **1 - Planlos:** Kein Muster erkennbar, rein reaktives, chaotisches "Wursteln", Sprünge zwischen Ideen ohne Abschluss.
*   **2 - Schwacher Ansatz:** Strategie ist ansatzweise erkennbar, bricht aber frühzeitig zusammen oder ist stark inkonsistent; Agent beginnt mit einer Idee, wechselt unkontrolliert zu anderen.
*   **3 - Implizite/Mittlere Strategie:** Der Agent schreibt keinen expliziten Plan, folgt aber einem klaren, methodischen Schema (z.B. erst Analyse, dann Reproduktion, dann Fix). ODER: Plan vorhanden, aber Ausführung unsauber.
*   **4 - Guter Plan mit kleinen Abweichungen:** Plan vorhanden und weitgehend eingehalten; gelegentliche Abweichungen, die aber das Gesamtbild nicht stören.
*   **5 - Exzellente Strategie:** Agent erstellt einen **expliziten** initialen Plan (ToDo-Liste), arbeitet ihn Schritt für Schritt ab und aktualisiert ihn sinnvoll bei neuen Erkenntnissen.
*   **N/A / null:** Nur verwenden, wenn der Trace so kurz ist, dass kein Verhalten bewertet werden kann.

### M2.4 Reasoning Quality (Logische Schlüssigkeit)
Bewertet die Qualität der `Thought` -> `Action` Kette. Macht der Schritt Sinn?
*   **1:** Halluzinationen, Non-Sequiturs (Gedanke passt null zur Aktion).
*   **2:** Häufige logische Lücken; Agent zieht wiederholt falsche Schlüsse aus Observations oder handelt ohne erkennbare Begründung.
*   **3:** Meist logisch, aber manchmal voreilige Schlüsse oder Missverständnisse der Observation.
*   **4:** Überwiegend schlüssiges Reasoning mit seltenen Ungenauigkeiten; kleine logische Sprünge, die aber nicht zu Fehlern führen.
*   **5:** Messerscharfe Logik. Jede Aktion ist perfekt aus der vorherigen Beobachtung abgeleitet.

### M2.5 Role Adherence (Rollentreue)
Bewertet, ob der Agent sich an seine System-Instruktionen hält.
*   **1:** Schwere Verstöße: Agent vergisst seine Rolle, verweigert Arbeit oder fragt wiederholt den User während der Aufgabe nach Hilfe ("Autonomy Breach").
*   **2:** Mehrfache mittlere Verstöße oder ein schwerwiegender Verstoß, der die Autonomie merklich beeinträchtigt; Agent weicht häufig vom erwarteten Rollenprofil ab.
*   **3:** Mittlere Verstöße: Gelegentliche, aber nicht kritische Rückfragen oder unnötige soziale Floskeln, die aber die Autonomie nicht gefährden.
*   **4:** Einzelner, einmaliger kleiner Verstoß (z.B. eine kurze soziale Floskel oder einmalige Rückfrage); das Gesamtverhalten bleibt rollentreu und autonom.
*   **5:** Perfekt: Agent bleibt strikt "in Character" (z.B. als Senior Engineer), arbeitet vollständig autonom ohne User-Interaktion.

### M3.1 Tool Selection (Werkzeugwahl)
Bewertet die *taktische* Wahl des Tools.
*   **1:** Wählt völlig falsche Tools (z.B. `edit` statt `read` zum Lesen).
*   **2:** Häufig suboptimale oder falsche Tool-Wahl; nur gelegentlich trifft der Agent die richtige Entscheidung, oft entsteht erheblicher Mehraufwand.
*   **3:** Wählt funktionierende, aber ineffiziente Tools (z.B. liest riesige Datei komplett statt `grep` zu nutzen).
*   **4:** Überwiegend gute Tool-Wahl; in einzelnen Situationen wäre ein effizienteres Tool verfügbar gewesen, ohne dass dadurch Fehler entstehen.
*   **5:** Wählt immer das optimale, effizienteste Tool für das Teilproblem.

### M3.2 Tool Execution Quality (Technische Ausführung)
Bewertet die *technische* Zuverlässigkeit und Erfolgsrate der ausgeführten Befehle.
*   **1:** Hohe Fehlerrate (>50%). Ständige Syntaxfehler, Crashes oder "Command not found".
*   **2:** Fehlerrate ca. 25–50%; häufige technische Fehler, die den Fortschritt deutlich verlangsamen; Agent braucht mehrere Versuche zur Korrektur.
*   **3:** Solide Ausführung mit gelegentlichen Fehlern (z.B. falsche Parameter), die der Agent aber korrigiert.
*   **4:** Niedrige Fehlerrate (<10%); gelegentlich ein kleiner technischer Fehler (z.B. falsches Argument), der sofort und korrekt behoben wird.
*   **5:** Perfekte Ausführung (nahe 100%). Jeder Befehl sitzt beim ersten Versuch technisch korrekt (auch wenn das Ergebnis inhaltlich leer sein mag).

### M4.1 Context Utilization (Kontext-Nutzung)
Bewertet die Konsistenz des Agenten mit dem sichtbaren Kontext.
*   **1:** Halluzination/Widerspruch: Agent behauptet Dinge, die explizit im Kontext widerlegt wurden (z.B. "Datei existiert nicht", obwohl er sie gerade gelesen hat) oder erfindet Fakten.
*   **2:** Häufige Inkonsistenzen: Agent ignoriert wichtige Kontextinformationen wiederholt oder zieht aus vorangegangenen Observations falsche Schlüsse, ohne sich zu korrigieren.
*   **3:** Ineffizienz: Agent ignoriert Informationen, was zu unnötigen Schritten führt (z.B. liest Datei nochmal), aber ohne logischen Widerspruch.
*   **4:** Überwiegend konsistente Kontextnutzung; seltene Ungenauigkeiten (z.B. einmaliges redundantes Lesen) ohne logische Widersprüche.
*   **5:** Konsistent: Agent nutzt alle verfügbaren Informationen logisch korrekt und widerspruchsfrei.

---

## Kategorie 5: Multi-Agenten-Metriken (MAS)
*Hinweis:* Diese Metriken sind nur für Multi-Agenten-Systeme (z.B. MetaGPT, ChatDev) relevant.
*   **Für Single-Agent Systeme (SWE-agent, OpenHands, Live-SWE-Agent):** Bitte trage hier konsequent **"N/A - Single Agent"** ein.

### M5.1 Communication Efficiency
Bewertet die Qualität der Kommunikation zwischen den Agenten.
*   **1:** Sinnloses "Ping-Pong", leere Nachrichten, endloses Danke-Sagen.
*   **2:** Überwiegend rauschbehaftete Kommunikation; Nachrichten enthalten zwar gelegentlich relevante Informationen, sind aber mehrheitlich redundant oder ohne klaren Zweck.
*   **3:** Gemischte Qualität; relevante Inhalte und Rauschen halten sich die Waage; Arbeitsverteilung erkennbar, aber nicht konsequent.
*   **4:** Überwiegend informationsreiche Kommunikation mit gelegentlichen Wiederholungen oder Floskeln; Arbeitsverteilung klar, Feedback konstruktiv.
*   **5:** Hochdichte Informationsübertragung, klare Arbeitsverteilung, konstruktives Feedback.
