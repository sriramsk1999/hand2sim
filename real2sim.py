import argparse
import os

import numpy as np
from tqdm import tqdm

from datasets.generic import GenericDataset
from datasets.hoi4d import HOI4DDataset


def main(
    env_name,
    dataset_name,
    seed,
    input_path,
    output_path,
):
    np.random.seed(seed)
    base_path = input_path
    os.makedirs(output_path, exist_ok=True)

    if dataset_name == "hoi4d":
        dataset = HOI4DDataset(base_path)
    elif dataset_name == "generic":
        dataset = GenericDataset(base_path)
    else:
        raise NotImplementedError

    if env_name == "robohive":
        from environments.robohive_env import RoboHiveRetargetEnv

        environment = RoboHiveRetargetEnv(seed)
    elif env_name == "isaacsim":
        from environments.isaacsim_env import IsaacSimRetargetEnv

        environment = IsaacSimRetargetEnv(seed)

    trajectory, valid_idxs = dataset.get_trajectory(environment.init_ee_pose)
    horizon = trajectory.shape[0]

    sim_imgs = []
    for i_step in tqdm(range(horizon)):
        image = environment.step_and_render(i_step, trajectory[i_step])
        sim_imgs.append(image)

    environment.close()
    sim_imgs = np.array(sim_imgs)

    output_name = base_path.replace("/", "_") + ".mp4"
    dataset.write_real_sim_video(
        sim_imgs,
        dataset.real_img_path,
        valid_idxs,
        f"{output_path}/{output_name}",
        dataset.viz_fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retarget a trajectory from a human video to a robot trajectory in simulation"
    )

    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        default="robohive",
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
    main(args.env_name, args.dataset, args.seed, args.input_path, args.output_path)
