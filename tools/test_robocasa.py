import argparse
import collections
import copy
import h5py
import imageio
import json
import math
import numpy as np
import pathlib
import random
import torch
import tqdm

import robocasa
import robosuite

import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geopredict import GeoPredict
from data_processing.data_transform import RobocasaInputTransform, RobocasaOutputTransform
from utils.utils import move_to_device, build_logger


ROBOCASA_DUMMY_ACTION = [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0]
ROBOCASA_ENV_RESOLUTION = 256
ROBOCASA_ENV_META_UPDATE_DICT = {
    "env_kwargs": {
        "generative_textures": None,
        "scene_split": None,
        "style_ids": None,
        "layout_ids": None,
        "layout_and_style_ids": [
            [
                1,
                1
            ],
            [
                2,
                2
            ],
            [
                4,
                4
            ],
            [
                6,
                9
            ],
            [
                7,
                10
            ]
        ],
        "randomize_cameras": False,
        "obj_instance_split": "B"
    }
}
ROBOCASA_MAX_STEPS = {
    'OpenDoubleDoor': 1000,
    'CloseDoubleDoor': 700,
    'PnPCounterToSink': 700,
    'PnPSinkToCounter': 500,
    'CoffeeSetupMug': 600,
    'CoffeeServeMug': 600,
    'PnPCounterToMicrowave': 600, 
    'PnPMicrowaveToCounter': 500,
    'TurnOnMicrowave': 500, 
    'TurnOffMicrowave': 500, 
    'CloseDrawer': 500, 
    'OpenDrawer': 500, 
    'TurnSinkSpout': 500,
    'CloseSingleDoor': 500,
    'OpenSingleDoor': 500,
    'PnPCounterToStove': 500,
    'PnPStoveToCounter': 500, 
    'TurnOffStove': 500, 
    'TurnOnStove': 500,
    'PnPCounterToCab': 500,
    'PnPCabToCounter': 500,
    'TurnOffSinkFaucet': 500, 
    'TurnOnSinkFaucet': 500,
    'CoffeePressButton': 300,
}
ROBOCASA_TASKS = [
    'OpenDoubleDoor', 'CloseDoubleDoor', 'PnPCounterToSink', 'PnPSinkToCounter',
    'CoffeeSetupMug', 'CoffeeServeMug', 'PnPCounterToMicrowave', 'PnPMicrowaveToCounter',
    'TurnOnMicrowave', 'TurnOffMicrowave', 'CloseDrawer', 'OpenDrawer',
    'TurnSinkSpout', 'CloseSingleDoor', 'OpenSingleDoor', 'PnPCounterToStove',
    'PnPStoveToCounter', 'TurnOffStove', 'TurnOnStove', 'PnPCounterToCab',
    'PnPCabToCounter', 'TurnOffSinkFaucet', 'TurnOnSinkFaucet', 'CoffeePressButton',
]


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


def deep_update(d, u):
    """
    Copied from https://stackoverflow.com/a/3233356
    """
    import collections
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _get_robocasa_env(task, resolution, seed, task_data_path):
    task_file = osp.join(os.path.expanduser(task_data_path), f"{task}.hdf5")

    env_meta = get_env_metadata_from_dataset(dataset_path=task_file)
    deep_update(env_meta, ROBOCASA_ENV_META_UPDATE_DICT)

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["seed"] = seed

    update_kwargs = dict(
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        renderer="mjviewer",
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env_kwargs.update(update_kwargs)

    env = robocasa.make(**env_kwargs)

    return env


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_keypoints(env, body_pos, body_rot):
    ori_trans = np.array([-0.5, -0.8, -0.0], dtype=np.float32)

    keypoint = None
    for j in range(1, 9):
        pos_name = "gripper0_right_eef" if j == 8 else f"robot0_link{j}"
        pos = env.sim.data.get_body_xpos(pos_name)
        pos = body_rot.T @ (pos - body_pos)
        pos = pos - ori_trans
        if keypoint is None:
            keypoint = pos
        else:
            keypoint = np.hstack((keypoint, pos))
    
    return keypoint.reshape(8, 3)


def eval_robocasa(args, model, logger):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    task_name = ROBOCASA_TASKS[args.suite_idx]
    logger.info(f"Task name: {task_name}")

    input_transform = RobocasaInputTransform()
    output_transform = RobocasaOutputTransform()

    # Start evaluation
    max_steps = ROBOCASA_MAX_STEPS[task_name]
    env = _get_robocasa_env(task_name, ROBOCASA_ENV_RESOLUTION, args.seed, args.task_data_path)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        # Reset environment
        env.unset_ep_meta()
        env.reset()
        task_description = env.get_ep_meta().get("lang", "dummy")
        logger.info(f"Task: {task_description}")

        body_id = env.sim.model.body_name2id('mobilebase0_support')
        body_pos = env.sim.data.xpos[body_id]
        body_rot = env.sim.data.xmat[body_id].reshape(3, 3)

        # Setup
        action_plan = collections.deque()
        t = 0
        replay_images = []
        replay_wrist_images = []
        his_kpts = np.zeros((1000, 8, 3), dtype=np.float32)
        his_len = 0
        
        logger.info(f"Starting episode {task_episodes+1}...")
        while t < max_steps + args.num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(ROBOCASA_DUMMY_ACTION)
                t += 1
                continue

            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            left_img = np.ascontiguousarray(env.sim.render(
                width=ROBOCASA_ENV_RESOLUTION, height=ROBOCASA_ENV_RESOLUTION, camera_name='robot0_agentview_left')[::-1])  # 256x256x3, uint8
            right_img = np.ascontiguousarray(env.sim.render(
                width=ROBOCASA_ENV_RESOLUTION, height=ROBOCASA_ENV_RESOLUTION, camera_name='robot0_agentview_right')[::-1])  # 256x256x3, uint8
            wrist_img = np.ascontiguousarray(env.sim.render(
                width=ROBOCASA_ENV_RESOLUTION, height=ROBOCASA_ENV_RESOLUTION, camera_name='robot0_eye_in_hand')[::-1])  # 256x256x3, uint8

            # Save preprocessed image for replay video
            replay_images.append(left_img)
            replay_wrist_images.append(wrist_img)

            if his_len == 0:
                input_len = 1
            else:
                input_len = his_len

            if not action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/left_image": left_img,  # 256x256x3, uint8
                    "observation/right_image": right_img,  # 256x256x3, uint8
                    "observation/wrist_image": wrist_img,  # 256x256x3, uint8
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],  # 3 dims
                            _quat2axisangle(obs["robot0_eef_quat"]),  # 3 dims
                            obs["robot0_gripper_qpos"],  # 2 dims
                        )
                    ),  # 8, float64
                    "prompt": str(task_description),  # N words, string
                    "resize_size": args.resize_size,  # int
                    "his_kpts": his_kpts,
                    "his_len": input_len,
                }

                # Query model to get action
                with torch.no_grad():
                    inputs = copy.deepcopy(element)
                    inputs = input_transform(inputs)
                    inputs = move_to_device(inputs, next(model.parameters()).device)
                    outputs = {
                        "state": inputs["state"].detach().cpu(),
                        "actions": model.sample_actions(inputs).detach().cpu(),
                    }
                    outputs = output_transform(outputs)
                    action_chunk = outputs["actions"]

                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            obs, _, _, _ = env.step(action.tolist())
            his_kpts[his_len] = get_keypoints(env, body_pos, body_rot)
            his_len += 1
            done = env._check_success()
            if done:
                task_successes += 1
                break
            t += 1

        task_episodes += 1

        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{episode_idx}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{episode_idx}_wrist.mp4",
            [np.asarray(x) for x in replay_wrist_images],
            fps=20,
        )

        # Log current results
        logger.info(f"Success: {done}")
        logger.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")

    # Log final results
    logger.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
    
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()

    ################ evaluation setting
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--suite_idx', default=0, type=int)
    parser.add_argument('--video_out_path', default="./data/robocasa", type=str)
    parser.add_argument('--num_trials_per_task', default=50, type=int)
    parser.add_argument('--num_steps_wait', default=10, type=int)
    parser.add_argument('--resize_size', default=224, type=int)
    parser.add_argument('--replan_steps', default=5, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--weight_path', default="./ckpts/GeoPredict_robocasa.pth", type=str)
    parser.add_argument('--task_data_path', default="./ckpts/robocasa", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    print(f"Using GPU {os.environ['CUDA_VISIBLE_DEVICES']}...")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    logger = build_logger(filepath=osp.join(args.video_out_path, f"log_{ROBOCASA_TASKS[args.suite_idx]}.txt"), verbose=False, mode="w")

    model = GeoPredict(embed_dtype=torch.float32)
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=True)
    model = model.cuda()
    model.eval()

    eval_robocasa(args, model, logger)
