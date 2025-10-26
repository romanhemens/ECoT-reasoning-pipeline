# Reasoning Chain

**Reasoning Chain** is a research-oriented framework for generating structured reasoning and motion data from RLDS (Reinforcement Learning Dataset Standard) episodes involving robotic manipulators. 
It orientates itself on [Embodied Chain of Thought](https://github.com/MichalZawalski/embodied-CoT/tree/main?tab=readme-ov-file) and leverages their repository into an easy-to-use pipeline. 
The pipeline processes datasets such as [**LIBERO**](https://github.com/Lifelong-Robot-Learning/LIBERO) or [**BRIDGE**](https://rail-berkeley.github.io/bridgedata/) to produce high-quality annotations, reasoning traces, and motion analyses.

---

## Prerequisites

- Linux operating system  
- Python ≥ 3.10  
- CUDA-capable GPU (recommended for deep learning components)  
- Bash-compatible shell  

---

## Installation

1. **Clone the repository**

```bash
git clone <REPO_URL>
cd reasoning-chain
```


2. **Set up the Python environment**

```bach
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables**

```bash
cp .env.example .env
```

Edit the ```.env``` file and specify:

- ```HF_TOKEN```: HuggingFace authentication token
- ```GEMINI_API_KEY```: Google Gemini API key
- ```DATA_DIR```: Path to TensorFlow datasets
- ```OUTPUT_DIR```: Output directory for generted results
- ```DATASET_NAME```: Dataset identifier (e.g. libero_10)
- ```DEVICE```: Compute device, e.g. cuda or cpu


## Usage

The main entry point is main.py, which orchestrates all processing stages:

```bash
python main.py \
  --tfds_name <DATASET_NAME> \
  --data_dir <DATA_DIR> \
  --output_dir <OUTPUT_DIR> \
  --hf_token <HF_TOKEN> \
  --device <DEVICE> \
  --start <START> \
  --end <END>
```

**Example:**

```bash
python main.py \
  --tfds_name libero_10 \
  --data_dir ~/tensorflow_datasets \
  --output_dir output \
  --hf_token <your_token> \
  --device cuda \
  --start 0 \
  --end 100
```

## Workflow

1. **Description Generation**
    
    Produces structured natural language scene descriptions for each episode.

2. **Primitive Movement Extraction**

    Analyzes robotic arm trajectories to identify and segment motion primitives.

3. **Gripper Position Estimation**
    Utilizes visual data and deep learning models to localize gripper positions.

4. **Reasoning Generation**
    Produces detailed reasoning annotations, providing semantic explanations for each action or state transition.


## Advanced Functionality

- **Bounding Box Generation**

    Available under ```generate_embodied_data/bounding_boxes/generate_bboxes.py```

- **Merging Intermediate Files**

    Use ```merge_description_files``` and ``merge_bbox_files`` to combine partial outputs across processing runs.


## Troubleshooting

- Verify that all API keys and dataset paths are correctly specified.

- Depending on dataset size and available compute, runtime may range from several minutes to several hours.

## Licence

## Contact

For questions, feedback, or research collaborations, please open an issue on GitHub or contact the repository maintainers.