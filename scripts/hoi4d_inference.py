from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse
import os

import numpy as np
from tqdm import tqdm
import h5py

from datasets.hoi4d import HOI4DDataset, HOI4DDatasetWrapper
from environments.isaacsim_env import IsaacSimRetargetEnv
from environments.robohive_env import RoboHiveRetargetEnv


def main(
    env_name,
    embodiment,
    seed,
    hoi4d_root,
    output_path,
):
    np.random.seed(seed)
    os.makedirs(output_path, exist_ok=True)

    dataset = HOI4DDatasetWrapper(hoi4d_root)

    # And the environment we're using as a target
    if env_name == "robohive":
        environment = RoboHiveRetargetEnv(seed, embodiment)
    elif env_name == "isaacsim":
        environment = IsaacSimRetargetEnv(seed, embodiment, simulation_app)
    else:
        raise NotImplementedError

    for i in tqdm(range(len(dataset))):
        item = dataset[i]

        # Retarget the trajectory for the specified environment
        try:
            trajectory, hand_actions, valid_idxs = item.get_trajectory(
                environment.init_ee_pose,
                environment.ee_range,
                environment.align_transform,
                embodiment,
            )
        except Exception as e:
            print(
                f"Could not process: {item.base_path}, due to the following exception: {e}"
            )
            continue
        horizon = trajectory.shape[0]

        # Render images for visualization
        sim_imgs, qpos_arr, success_arr = [], [], []
        for i_step in tqdm(range(horizon)):
            image, qpos, success = environment.step_and_render(
                i_step, trajectory[i_step], hand_actions[i_step]
            )
            sim_imgs.append(image)
            qpos_arr.append(qpos)
            success_arr.append(success)

        qpos_arr, success_arr = np.array(qpos_arr), np.array(success_arr)
        cam_name_start_idx = item.base_path.find("ZY2021")
        save_name = item.base_path[cam_name_start_idx:].replace("/", "_")
        with h5py.File(f"{output_path}/{save_name}.h5", "w") as hf:
            hf["ee_pos"] = trajectory
            hf["qpos"] = qpos_arr
            hf["success"] = success_arr
            hf["vaild_idxs"] = valid_idxs
            hf["environment"] = env_name
            hf["embodiment"] = embodiment

        # Visualize
        if i % 100 == 0:
            sim_imgs = np.array(sim_imgs)
            item.write_real_sim_video(
                sim_imgs,
                item.real_img_path,
                valid_idxs,
                f"{output_path}/{env_name}-{embodiment}-{save_name}.mp4",
                item.viz_fps,
            )
        environment.reset()

    environment.close()


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
        "--hoi4d_root",
        type=str,
        required=True,
        help="Path to HOI4D dataset",
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
        "-op",
        "--output_path",
        type=str,
        default="hoi4d_hand2sim",
        help="Directory to store retargeted trajectories and visualizations",
    )

    args = parser.parse_args()
    main(
        args.env_name,
        args.embodiment,
        args.seed,
        args.hoi4d_root,
        args.output_path,
    )
