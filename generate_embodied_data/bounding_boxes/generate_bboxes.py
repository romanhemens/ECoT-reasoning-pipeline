import json
import os
import time
import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from generate_embodied_data.bounding_boxes.utils import NumpyFloatValuesEncoder, post_process_caption


def generate_bounding_boxes(
    tfds_name: str,
    start: int,
    end: int,
    device: str,
    data_dir: str,
    result_dir: str,
    descriptions_path: str,
):
    """
    Generates bounding boxes using gDINO for the given split of the dataset.
    Saves results to JSON file per split.
    """

    print(f"[BBoxes] Loading dataset '{tfds_name}' (split {start}%:{end}%)...")
    ds = tfds.load(
        tfds_name,
        data_dir=data_dir,
        split=f"train[{start}%:{end}%]",
    )
    print("[BBoxes] Dataset loaded.")

    print("[BBoxes] Loading descriptions...")
    with open(descriptions_path, "r") as f:
        results_json = json.load(f)
    print("[BBoxes] Descriptions loaded.")

    model_id = "IDEA-Research/grounding-dino-base"
    print(f"[BBoxes] Loading gDINO model ({model_id}) on device {device}...")
    processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 128, "longest_edge": 128})
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print("[BBoxes] Model loaded.")

    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.2

    bbox_results_json = {}
    output_file = os.path.join(result_dir, f"results_{start}_{end}_bboxes.json")

    for ep_idx, episode in enumerate(ds):
        episode_id = episode["episode_metadata"]["episode_id"].numpy()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"[BBoxes] Split {split_id} processing episode {ep_idx}: {episode_id} ({file_path})")

        if file_path not in bbox_results_json:
            bbox_results_json[file_path] = {}

        description = results_json[file_path][str(episode_id)]["caption"]

        bboxes_list = []
        for step_idx, step in enumerate(episode["steps"]):
            if step_idx == 0:
                lang_instruction = step["language_instruction"].numpy().decode()

            image = Image.fromarray(step["observation"]["image_0"].numpy())
            inputs = processor(
                images=image,
                text=post_process_caption(description, lang_instruction),
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]],
            )[0]

            logits, phrases, boxes = (
                results["scores"].cpu().numpy(),
                results["labels"],
                results["boxes"].cpu().numpy(),
            )

            bboxes = []
            for lg, p, b in zip(logits, phrases, boxes):
                b = list(b.astype(int))
                lg = round(lg, 5)
                bboxes.append((lg, p, b))
                break  # only take first match per image

            bboxes_list.append(bboxes)

        bbox_results_json[file_path][str(ep_idx)] = {
            "episode_id": int(episode_id),
            "file_path": file_path,
            "bboxes": bboxes_list,
        }

        with open(output_file, "w") as f:
            json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)