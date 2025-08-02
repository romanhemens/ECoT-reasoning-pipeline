import json
import os
import re
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from dotenv import load_dotenv
import openai
from google import genai

class Gemini:
    def __init__(self, model_name="gemini-1.5-flash"):

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")
        
        self.model = genai.Client(api_key=api_key).models

        self.model_name = model_name

    def safe_call(self, f):
        while True:
            try:
                return f()
            except Exception as e:
                print("Error in Gemini call:", e)
                time.sleep(5)

    def generate(self, prompt):
        response = self.safe_call(lambda: self.model.generate_content(
            model=self.model_name,
            contents=prompt
        ))
        
        return response.text
'''class DeepSeek:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key="OPEMAI_API_KEY",  # Set your OpenAI API key here
            base_url="https://api.groq.com/openai/v1"  # Groq endpoint
        )
        self.model_name = "deepseek-r1-distill-llama-70b"

    def safe_call(self, f):
        while True:
            try:
                return f()
            except openai.RateLimitError:
                time.sleep(5)
            except Exception as e:
                print("Error in DeepSeek call:", e)
                time.sleep(5)

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "You're a helpful reasoning agent for robotics task planning."},
            {"role": "user", "content": prompt},
        ]

        full_response = ""
        for i in range(8):
            response = self.safe_call(lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                stream=False
            ))

            chunk = response.choices[0].message.content
            full_response += chunk

            if "FINISHED" in full_response:
                print(f"n_retries: {i}")
                return full_response

            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": "Truncated, please continue."})

        return None'''


def build_prompt(features, language_instruction, caption=None, list_only_moves=False):
    structured_features = "{\n"

    print("features:", features)
    print("keys:", features.keys())
    keys = list(features.keys())

    if list_only_moves:
        n = len(features["move_primitive"])
    else:
        # oder: n = min(len(features[k]) for k in keys)
        n = min(len(features[k]) for k in keys)

    # 3. String aufbauen
    structured_features = "{\n"
    for i in range(n):
        if list_only_moves:
            instr = features["move_primitive"][i]
            structured_features += f'    {i}: "{instr}"\n'
        else:
            structured_features += f'    {i}: ' + "{\n"
            for key in keys:
                val = features[key][i]
                if isinstance(val, str):
                    val = f'"{val}"'
                structured_features += f'        "{key}": {val},\n'
            structured_features += "    },\n"

    structured_features += "}"
    
    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "move_primitive" describes the move that is about to be executed,\n'
            '- "gripper_position" denotes the location of the gripper in the 256x256 image observation'
        )

    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    break_line = ""  # for line formatting

    return f"""# Annotate the training trajectory with reasoning

## Specification of the experimental setup

You're an expert reinforcement learning researcher. You've trained an optimal policy for controlling a robotic arm. The
robot successfully completed a task specified by the instruction: "{language_instruction}". For that purpose, the
robotic arm executed a sequence of actions. Consecutive moves that were executed are the following:


python
trajectory_features = {structured_features}



{features_desc}

{caption}## Your objective

I want you to annotate the given trajectory with reasoning. That is, for each step, I need to know not only {
break_line}which action should be chosen, but importantly what reasoning justifies that action choice. I want you to {
break_line}be descriptive and include all the relevant information available. The reasoning should include the task {
break_line}to complete, the remaining high-level steps, the high-level movements that should be executed and why they {
break_line}are required, the premises that allow inferring the direction of each move, including the locations of {
break_line}relevant objects, possible obstacles or difficulties to avoid, and any other relevant justification.

### Begin by describing the task

Start by giving an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, {
break_line}the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe {
break_line}the high-level movements that were most likely executed, based on the task that was completed and the {
break_line}primitive movements that were executed. Then, for each high-level movement write the interval of steps that {
break_line}movement consists of. Also, for each high-level movement write a justification for why it should be {
break_line}executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight {
break_line}all the relevant details, but ensure that your description is consistent with the trajectory that was {
break_line}executed, specified by the features listed above in the trajectory_features dictionary.

### List the reasonings for each step

Finally, for each step describe the reasoning that allows to determine the correct action. For each step describe the {
break_line}remaining part of the objective, the current progress, the objects that are still relevant for determining {
break_line}the plan, and the plan for the next steps, based on the available features. Start the reasoning from a high {
break_line}level and gradually add finer features. I need you to be descriptive and very precise. Ensure that the {
break_line}reasoning is consistent with the task and the executed trajectory. Write the answer for this part as a {
break_line}Python-executable dictionary. For every step in the initial trajectory there should be exactly one separate {
break_line}item of the form <step id>:<reasoning>. Do not group the answers. The final dictionary should have exactly {
break_line}the same set of integer keys as the dictionary of features provided in the trajectory_features dictionary {
break_line}above. The reasoning should be a single string that describes the reasoning in natural language and {
break_line}includes all the required features.

Each reasoning string should have the following form:
- Describe the full task that remains to be completed (but only describe what remains), and place it inside a {
break_line}tag <task>.
- Describe the complete high-level plan for completing the remaining task (the list of remaining high-level steps), {
break_line}and place it inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it {
break_line}inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence {
break_line}that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Describe the current primitive movement of the arm that needs to be executed, and place it inside a tag <move>.
- Describe why the chosen movement should be executed now and which features of the current environment influence that {
break_line}decision. Place it inside a tag <move_reason>.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- For each step on the trajectory, describe the reasoning that leads to determining the correct action. The reasoning {
break_line}should be descriptive and precise. You should provide exactly one reasoning string for each step on the {
break_line}trajectory specified by trajectory_features.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""


def find_task_occurrences(input_string, tags):
    pattern = r"(\d+):"
    for tag in tags:
        pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"

    matches = re.findall(pattern, input_string)
    return matches


def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    matches = find_task_occurrences(reasoning_output, tags)

    for match in matches:
        trajectory[int(match[0])] = dict(zip(tags, match[1:]))

    return trajectory


def get_reasoning_dict(features, metadata, lm):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    prompt = build_prompt(features, language_instruction, caption=caption, list_only_moves=True)
    print("metadata:", metadata, "\nprompt:", prompt)

    reasoning_output = lm.generate(prompt)

    print("reasoning:", reasoning_output)

    return extract_reasoning_dict(reasoning_output)




def build_single_reasoning(ep_id, episode, file_path, ds, lm, captions, primitives, grippers):
    
    ft = dict()
    # get states
    ft["state_3d"] = [list(step["observation"]["state"][:3].numpy()) for step in episode["steps"]]

    # get primitive moves
    move_primitives = primitives[str(file_path)][str(ep_id)]["primitives"]
    ft["move_primitive"] = [move for move in move_primitives]

    # get gripper positions
    ft["gripper_positions"] = grippers[str(file_path)][str(ep_id)]

    mt = {
        "episode_id": ep_id,
        "file_path": file_path,
        "n_steps": len(episode["steps"]),
        "language_instruction": str(next(iter(episode["steps"]))["language_instruction"].numpy().decode())
    }

    mt["caption"] = captions[mt["file_path"]][mt["episode_id"]]["caption"]
    
    reasoning = get_reasoning_dict(ft, mt, lm)
    entry = {"reasoning": reasoning, "features": ft, "metadata": mt}

    return entry


def generate_reasonings(
    tfds_name: str,
    data_dir: str = None,
    start: int = 0,
    end: int = 5,
    device: str = "cuda", 
    output_dir = str,
):
    reasonings = dict()
    #lm = DeepSeek()
    lm = Gemini()

    ds = tfds.load(
        tfds_name,
        data_dir=data_dir,
        split=f"train[{start}%:{end}%]",
    )

    save_path = os.path.join(output_dir, "reasonings.json")

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    with open(os.path.join(output_dir, "descriptions", "full_descriptions.json"), "r") as captions_file:
        captions_dict = json.load(captions_file)
    with open(os.path.join(output_dir, "primitive", "primitive_movement.json"), "r") as f:
        primitives = json.load(f)
    with open(os.path.join(output_dir, "gripper", "gripper_positions.json"), "r") as f:
        grippers = json.load(f)
    

    for ep_idx, episode in enumerate(ds):

        try:
            ep_id = episode["episode_metadata"]["episode_id"].numpy()
        except KeyError:
            print(f"[WARN] 'episode_id' not found. Using fallback.")
            ep_id = str(episode["episode_metadata"]["file_path"].numpy().decode().split("/")[-1].split(".")[0])
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()


        entry = build_single_reasoning(ep_id, episode, file_path, ds, lm, captions_dict, primitives, grippers)

        if entry["metadata"]["file_path"] in reasonings.keys():
            reasonings[entry["metadata"]["file_path"]][entry["metadata"]["episode_id"]] = entry
        else:
            reasonings[entry["metadata"]["file_path"]] = {entry["metadata"]["episode_id"]: entry}

        print("computed reasoning:", entry)

    with open(save_path, "w") as out_f:
        json.dump(reasonings, out_f)