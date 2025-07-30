# generate.py
import os
import argparse
import tensorflow_datasets as tfds
from generate_embodied_data.bounding_boxes.generate_descriptions import generate_descriptions
from generate_embodied_data.bounding_boxes.generate_bboxes import generate_bounding_boxes
from generate_embodied_data.primitive_movements import extract_primitives_for_all_episodes
from generate_embodied_data.gripper_positions import compute_gripper_positions
from generate_embodied_data.full_reasonings import generate_reasonings
from generate_embodied_data.bounding_boxes.merge_descriptions import merge_description_files
from generate_embodied_data.bounding_boxes.merge_bboxes import merge_bbox_files


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning per RLDS episode.")
    parser.add_argument("--tfds_name", type=str, required=True, help="e.g. 'libero_10'")
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/tensorflow_datasets"), help="Path to the TFDS data directory.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()


    print("Step 1: Generating descriptions...")
    if os.path.isfile(os.path.join(args.output_dir, "descriptions", "full_descriptions.json")):
        print("Descriptions already generated. Skipping step 1.")
        # Hier kannst du optional eine Funktion aufrufen, die die bereits generierten Beschreibungen verarbeitet
        # z.B. merge_description_files(args.output_dir)
        # merge_description_files(os.path.join(args.output_dir, "descriptions"))
    else:

        generate_descriptions(
            tfds_name=args.tfds_name,
            start=args.start,
            end=args.end,
            device=args.device,
            hf_token=args.hf_token,
            results_path=os.path.join(args.output_dir, "descriptions"),
        )

        print("Merging descriptions…")
        merge_description_files(os.path.join(args.output_dir, "descriptions"))


    print("Step 2: Extract bounding boxes...")
    if os.path.isfile(os.path.join(args.output_dir, "bboxes", "full_bboxes.json")):
        print("Bounding boxes already generated. Skipping step 2.")
    else:

        generate_bounding_boxes(
            tfds_name=args.tfds_name,
            start=args.start,
            end=args.end,
            device=args.device,
            data_dir=args.data_dir,  # optional, falls du nicht den Standard-TFDS-Cache nutzt
            result_dir=os.path.join(args.output_dir, "bboxes"),
            descriptions_path=os.path.join(args.output_dir, "descriptions", "full_descriptions.json"),
        )

        print("Merging bounding boxes…")
        merge_bbox_files(args.output_dir)


    print("Step 3: Compute motion primitives...")
    if os.path.isfile(os.path.join(args.output_dir, "primitive", "primitive_movement.json")):
        print("Primitive movement already generated. Skipping step 3.")
    else:
        primitive_data = extract_primitives_for_all_episodes(
            tfds_name=args.tfds_name,
            data_dir=args.data_dir,
            start=args.start,
            end=args.end,
            verbose=True,
        )

        # Option: Ergebnisse speichern
        import json
        os.makedirs(os.path.join(args.output_dir, "primitive"), exist_ok=True)
        with open(os.path.join(args.output_dir, "primitive", "primitive_movement.json"), "w") as f:
            json.dump(primitive_data, f, indent=2)


    print("Step 4: Compute gripper position...")
    if os.path.isfile(os.path.join(args.output_dir, "gripper", "gripper_positions.json")):
        print("Gripper position already generated. Skipping step 4.")
    else:
        gripper_positions = compute_gripper_positions(
            tfds_name=args.tfds_name,
            data_dir=args.data_dir,
            start=args.start,
            end=args.end,
            device=args.device,
        )

        # Optional speichern
        import json
        os.makedirs(os.path.join(args.output_dir, "gripper"), exist_ok=True)
        with open(os.path.join(args.output_dir, "gripper", "gripper_positions.json"), "w") as f:
            json.dump(gripper_positions, f, indent=2)

    stop = True
    if not stop:
        print("Step 5: Generate plans and subtasks...")
        DEFAULT_DATA_DIR = os.path.expanduser("~/tensorflow_datasets")
        builder = tfds.builder_from_directory(
            builder_dir=DEFAULT_DATA_DIR,
            config=args.tfds_name)
        episode_ids = range(args.start, args.end)

        # NOTE the generator expects the captions.json file to be present in the working directory
        # The captions should be generated using the script in
        # scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py
        generate_reasonings(builder, episode_ids)
    else:
        print("Skipping step 5: Generating plans and subtasks.")

if __name__ == "__main__":
    main()