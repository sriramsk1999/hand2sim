from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse
import os

import numpy as np
from tqdm import tqdm

from datasets.generic import GenericDataset
from datasets.hoi4d import HOI4DDataset
from environments.isaacsim_env import IsaacSimRetargetEnv
from environments.robohive_env import RoboHiveRetargetEnv


def main(
    env_name,
    dataset_name,
    embodiment,
    seed,
    input_path,
    output_path,
):
    np.random.seed(seed)
    base_path = input_path
    os.makedirs(output_path, exist_ok=True)

    # Create the dataset we're using as a source
    if dataset_name == "hoi4d":
        dataset = HOI4DDataset(base_path)
    elif dataset_name == "generic":
        dataset = GenericDataset(base_path)
    else:
        raise NotImplementedError

    # And the environment we're using as a target
    if env_name == "robohive":
        environment = RoboHiveRetargetEnv(seed, embodiment)
    elif env_name == "isaacsim":
        environment = IsaacSimRetargetEnv(seed, embodiment, simulation_app)
    else:
        raise NotImplementedError

    # Retarget the trajectory for the specified environment
    trajectory, hand_actions, valid_idxs = dataset.get_trajectory(
        environment.init_ee_pose,
        environment.ee_range,
        environment.align_transform,
        embodiment,
    )
    horizon = trajectory.shape[0]

    # Render images for visualization
    sim_imgs = []
    for i_step in tqdm(range(horizon)):
        image = environment.step_and_render(
            i_step, trajectory[i_step], hand_actions[i_step]
        )
        sim_imgs.append(image)

    sim_imgs = np.array(sim_imgs)

    output_name = base_path.replace("/", "_") + ".mp4"
    dataset.write_real_sim_video(
        sim_imgs,
        dataset.real_img_path,
        valid_idxs,
        f"{output_path}/{env_name}-{embodiment}-{output_name}",
        dataset.viz_fps,
    )
    environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retarget a trajectory from a human video to a robot trajectory in simulation"
    )

    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        default="isaacsim",
        choices=["isaacsim", "robohive"],
        help="Environment to load",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="hoi4d",
        choices=["hoi4d", "generic"],
        help="Dataset to load",
    )
    parser.add_argument(
        "-emb",
        "--embodiment",
        type=str,
        default="pjaw",
        choices=["pjaw", "allegro"],
        help="Embodiment to retarget to (pjaw - parallel jaw gripper)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-ip", "--input_path", type=str, help="Input data for real2sim", required=True
    )
    parser.add_argument(
        "-op",
        "--output_path",
        type=str,
        default="real2sim_viz",
        help="Directory to store real2sim viz",
    )

    args = parser.parse_args()
    main(
        args.env_name,
        args.dataset,
        args.embodiment,
        args.seed,
        args.input_path,
        args.output_path,
    )
