import json, os
from tqdm import tqdm

def merge_bbox_files(results_dir: str):
    """
    Findet alle results_*_bboxes.json in results_dir/bboxes
    und schreibt full_bboxes.json.
    """
    bbox_dir = os.path.join(results_dir, "bboxes")
    output_path = os.path.join(bbox_dir, "full_bboxes.json")

    full = {}
    count = 0
    for fn in os.listdir(bbox_dir):
        if fn.startswith("results_") and fn.endswith(".json"):
            with open(os.path.join(bbox_dir, fn)) as f:
                part = json.load(f)
            for file_path, episodes in part.items():
                full.setdefault(file_path, {})
                for eid_str, ep_json in episodes.items():
                    eid = ep_json["episode_id"]
                    full[file_path][eid] = ep_json
                    count += 1

    print(f"[MergeBBoxes] Inserted {count} entries → {output_path}")
    with open(output_path, "w") as f:
        json.dump(full, f)