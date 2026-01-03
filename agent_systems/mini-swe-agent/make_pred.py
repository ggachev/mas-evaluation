import json

with open('last_swebench_single_run.traj.json') as f:
    data = json.load(f)

# Instance-ID bestimmen (z.B. "astropy__astropy-12907")
instance_id = "scikit-learn__scikit-learn-10844"
patch = data['info']['submission']

pred = {
    "instance_id": instance_id,
    "model_patch": patch,
    "model_name_or_path": "openai/gpt-5"
}

with open('all_preds.jsonl', 'w') as out:
    out.write(json.dumps(pred) + '\n')
print("✅ predictions-Datei erstellt")
