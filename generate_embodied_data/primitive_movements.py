import numpy as np
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as R

# ----------------------------
# Quaternion to Euler conversion
# ----------------------------
def quaternion_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
    """
    r = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return roll, pitch, yaw


# ----------------------------
# Movement description
# ----------------------------
def describe_move(move_vec):
    """
    Generate human-readable movement description from movement vector.
    """
    names = [
        {-1: "backward", 0: None, 1: "forward"},                   # x
        {-1: "right", 0: None, 1: "left"},                         # y
        {-1: "down", 0: None, 1: "up"},                            # z
        {-1: "tilt left", 0: None, 1: "tilt right"},               # roll
        {-1: "tilt down", 0: None, 1: "tilt up"},                  # pitch
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},  # yaw
        {-1: "close gripper", 0: None, 1: "open gripper"},         # gripper
    ]

    move_labels = [names[i][move_vec[i]] for i in range(len(move_vec))]
    move_labels = [label for label in move_labels if label is not None]

    if not move_labels:
        return "stop"
    
    return "move " + ", ".join(move_labels)


# ----------------------------
# Movement classification
# ----------------------------
def classify_movement(move, translation_threshold=0.01, gripper_threshold=0.01, rotation_threshold=0.01):
    """
    Classify the movement between two states as a movement primitive.
    """
    diff_trans = move[-1][:3] - move[0][:3]

    # Extract quaternions
    q0 = move[0][3:7]
    q1 = move[-1][3:7]

    # Compute Euler angles from quaternions
    r0 = np.array(quaternion_to_euler(*q0))
    r1 = np.array(quaternion_to_euler(*q1))

    diff_rot = r1 - r0
    gripper_diff = move[-1][7] - move[0][7]

    # Combine translation, rotation, and gripper difference
    diff = np.concatenate([diff_trans, diff_rot, [gripper_diff]])

    move_vec = np.zeros(7, dtype=int)

    # Translation
    move_vec[0:3] = (
        1 * (diff[0:3] > translation_threshold) -
        1 * (diff[0:3] < -translation_threshold)
    )

    # Rotation (roll, pitch, yaw)
    move_vec[3:6] = (
        1 * (diff[3:6] > rotation_threshold) -
        1 * (diff[3:6] < -rotation_threshold)
    )

    # Gripper
    if diff[6] > gripper_threshold:
        move_vec[6] = 1  # open
    elif diff[6] < -gripper_threshold:
        move_vec[6] = -1  # close

    return describe_move(move_vec), move_vec


# ----------------------------
# Episode-level primitive extraction
# ----------------------------
def extract_move_primitives_from_episode(episode):
    """
    Extract primitive movements from a single episode.
    """
    steps = list(episode["steps"])

    states = np.array([step["observation"]["state"] for step in steps])
    actions = [step["action"][:3].numpy() for step in steps]

    move_trajs = [states[i : i + 4] for i in range(len(states) - 1)]
    primitives = [classify_movement(move) for move in move_trajs]
    primitives.append(primitives[-1])  # duplicate last to match length

    move_actions = {}
    for (move, _), action in zip(primitives, actions):
        if move in move_actions:
            move_actions[move].append(action)
        else:
            move_actions[move] = [action]

    return {
        "primitive_descriptions": [p[0] for p in primitives],
        "primitive_vectors": [p[1].tolist() for p in primitives],
        "move_action_map": move_actions,
    }


# ----------------------------
# Dataset-level primitive extraction
# ----------------------------
def extract_primitives_for_all_episodes(
    tfds_name: str,
    data_dir: str = None,
    start: int = 0,
    end: int = 5,
    verbose: bool = True,
):
    """
    Extract movement primitives for all episodes in the dataset split.
    """
    ds = tfds.load(
        tfds_name,
        data_dir=data_dir,
        split=f"train[{start}%:{end}%]",
    )

    result = {}
    for ep_idx, episode in enumerate(ds):
        try:
            ep_id = episode["episode_metadata"]["episode_id"].numpy()
        except KeyError:
            print(f"[WARN] 'episode_id' not found. Using fallback.")
            ep_id = str(episode["episode_metadata"]["file_path"].numpy().decode().split("/")[-1].split(".")[0])

        file_path = episode["episode_metadata"]["file_path"].numpy().decode()

        primitives = extract_move_primitives_from_episode(episode)

        if file_path not in result:
            result[file_path] = {}

        result[file_path][str(ep_id)] = {
            "primitives": primitives["primitive_descriptions"],
            "vectors": primitives["primitive_vectors"],
        }

        if verbose:
            print(f"Episode {ep_idx}: {file_path} (ID {ep_id}) processed")

    return result
