
import os, json

def merge_description_files(results_dir: str):
    """
    Findet alle results_*.json in results_dir/descriptions
    und schreibt full_descriptions.json.
    """
    desc_dir = os.path.join(results_dir, "descriptions")
    output_path = os.path.join(desc_dir, "full_descriptions.json")

    full = {}
    count = 0
    for fn in os.listdir(desc_dir):
        if fn.startswith("results_") and fn.endswith(".json"):
            with open(os.path.join(desc_dir, fn)) as f:
                part = json.load(f)
            for file_path, episodes in part.items():
                full.setdefault(file_path, {})
                for eid, data in episodes.items():
                    full[file_path][eid] = data
                    count += 1

    print(f"[MergeDesc] Inserted {count} entries → {output_path}")
    with open(output_path, "w") as f:
        json.dump(full, f)