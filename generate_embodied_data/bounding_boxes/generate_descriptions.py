def generate_descriptions(tfds_name, tfds_dir, start, end, device, hf_token, results_path):
    import json
    import os
    import warnings
    import tensorflow_datasets as tfds
    import torch
    from PIL import Image
    from tqdm import tqdm
    from generate_embodied_data.bounding_boxes.utils import NumpyFloatValuesEncoder
    from prismatic import load

    vlm_model_id = "prism-dinosiglip+7b"
    warnings.filterwarnings("ignore")

    # Load split
    ds, ds_info = tfds.load(
        tfds_name,
        split=f"train[{start}%:{end}%]",
        data_dir=tfds_dir,
        with_info=True,
    )

    # Load Prismatic VLM
    print(f"Loading Prismatic VLM ({vlm_model_id})...")
    vlm = load(vlm_model_id, hf_token=hf_token)
    vlm = vlm.to(device, dtype=torch.bfloat16)

    results_json_path = os.path.join(results_path, f"descriptions.json")

    def create_user_prompt(lang_instruction):
        prompt = "Briefly describe the things in this scene and their spatial relations to each other."
        lang_instruction = lang_instruction.strip().rstrip(".")
        if " " in lang_instruction:
            prompt = f"The robot task is: '{lang_instruction}.' " + prompt
        return prompt

    results_json = {}
    for episode in tqdm(ds):
        #print("Available metadata keys:", episode.get("episode_metadata", {}).keys())

        try:
            episode_id = episode["episode_metadata"]["episode_id"].numpy()
        except KeyError:
            print(f"[WARN] 'episode_id' not found. Using fallback.")
            episode_id = str(episode["episode_metadata"]["file_path"].numpy().decode().split("/")[-1].split(".")[0])

        file_path = episode["episode_metadata"]["file_path"].numpy().decode()

        for step in episode["steps"]:
            lang_instruction = step["language_instruction"].numpy().decode()
            
            image = Image.fromarray(step["observation"]["image"].numpy())

            user_prompt = create_user_prompt(lang_instruction)
            prompt_builder = vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=user_prompt)
            prompt_text = prompt_builder.get_prompt()
            torch.manual_seed(0)
            caption = vlm.generate(
                image, 
                prompt_text, 
                do_sample=True, 
                temperature=0.4, 
                max_new_tokens=64)
            break

        episode_json = {
            "episode_id": str(episode_id), 
            "file_path": file_path, 
            "caption": caption
            }
        
        if file_path not in results_json:
            results_json[file_path] = {}

        results_json[file_path][str(episode_id)] = episode_json

        # Make sure the directory exists
        os.makedirs(os.path.dirname(results_json_path), exist_ok=True)

        with open(results_json_path, "w") as f:
            json.dump(results_json, f, cls=NumpyFloatValuesEncoder)