import os
import json

# Konfiguration
MODEL_NAME = "openhands"
OUTPUT_FILE = "preds.json"

# Mapping von Ordnernamen zu Instance IDs
# Format: "Ordnername": "Instance ID"
INSTANCE_MAPPING = {
    "issue_1": "scikit-learn__scikit-learn-12585",
    "issue_2": "pallets__flask-5014",
    "issue_3": "psf__requests-2317",
    "issue_4": "matplotlib__matplotlib-22719",
    "issue_5": "django__django-11099",
    "issue_6": "sympy__sympy-13480",
    "issue_7": "matplotlib__matplotlib-20676",
    "issue_8": "pylint-dev__pylint-4970",
    "issue_9": "pytest-dev__pytest-5262",
    "issue_10": "django__django-16901",
    "issue_11": "scikit-learn__scikit-learn-10297",
    "issue_12": "sympy__sympy-20801",
    "issue_13": "matplotlib__matplotlib-23299",
    "issue_14": "django__django-15128",
    "issue_15": "sympy__sympy-18199"
}

def create_preds_json():
    predictions = []
    
    print(f"Suche nach diff.txt Dateien in {len(INSTANCE_MAPPING)} Ordnern...")

    for folder_name, instance_id in INSTANCE_MAPPING.items():
        diff_path = os.path.join(folder_name, "diff.txt")
        
        # Prüfen, ob der Ordner und die Datei existieren
        if os.path.exists(diff_path):
            try:
                with open(diff_path, "r", encoding="utf-8") as f:
                    patch_content = f.read()
                
                # Eintrag für preds.json erstellen
                entry = {
                    "instance_id": instance_id,
                    "model_patch": patch_content,
                    "model_name_or_path": MODEL_NAME
                }
                predictions.append(entry)
                print(f"✅ {folder_name}: Patch für {instance_id} geladen.")
                
            except Exception as e:
                print(f"❌ {folder_name}: Fehler beim Lesen von {diff_path}: {e}")
        else:
            print(f"⚠️  {folder_name}: Keine 'diff.txt' gefunden! (Überspringe {instance_id})")

    # JSON Datei schreiben
    if predictions:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4)
        print(f"\n🎉 Erfolg! {len(predictions)} Einträge wurden in '{OUTPUT_FILE}' gespeichert.")
    else:
        print("\n❌ Keine gültigen Einträge gefunden. 'preds.json' wurde nicht erstellt.")

if __name__ == "__main__":
    create_preds_json()
