import argparse
import os

import gymnasium as gym
import numpy as np
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.quat_math import quat2mat
from tqdm import tqdm

from hoi4d import get_hoi4d_trajectory
from utils import set_initial_ee_target, write_real_sim_video


def main(
    env_name,
    seed,
    goal_site,
    teleop_site,
    input_path,
    output_path,
):
    base_path = input_path
    os.makedirs(output_path, exist_ok=True)

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)

    env.env.mujoco_render_frames = False
    goal_sid = env.sim.model.site_name2id(goal_site)

    translation = [0.4, 0, 1.1]
    rotation = [0, 0, 0, 1]
    set_initial_ee_target(env, goal_sid, translation, rotation)

    env.reset()
    curr_pos = env.sim.model.site_pos[goal_sid]
    curr_quat = env.sim.model.site_quat[goal_sid]
    ee_pose = np.eye(4)
    ee_pose[:3, :3] = quat2mat(curr_quat)
    ee_pose[:3, 3] = curr_pos
    trajectory, valid_idxs = get_hoi4d_trajectory(base_path, ee_pose)
    horizon = trajectory.shape[0]
    env.reset()

    # recover init state
    _ = env.forward()
    act = np.zeros(env.action_space.shape)
    gripper_state = 0
    sim_imgs = []

    for i_step in tqdm(range(horizon)):
        curr_pos = env.sim.model.site_pos[goal_sid]
        curr_pos[:] = trajectory[i_step][:3]
        # update rot
        curr_quat = env.sim.model.site_quat[goal_sid]
        curr_quat[:] = trajectory[i_step][3:7]
        # update gripper
        gripper_state = trajectory[i_step][7]

        # get action using IK
        ik_result = qpos_from_site_pose(
            physics=env.sim,
            site_name=teleop_site,
            target_pos=curr_pos,
            target_quat=curr_quat,
            inplace=False,
            regularization_strength=1.0,
        )
        if ik_result.success == False:
            print(
                f"IK(t:{i_step}):: Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}"
            )
        else:
            act[:7] = ik_result.qpos[:7]
            act[7:] = gripper_state
            if env.normalize_act:
                act = env.env.robot.normalize_actions(act)

        _ = env.step(act)
        sim_imgs.append(env.get_exteroception()["rgb:left_cam:240x424:2d"])

    sim_imgs = np.array(sim_imgs)
    output_name = base_path.replace("/", "_") + ".mp4"
    write_real_sim_video(
        sim_imgs, base_path, valid_idxs, f"{output_path}/{output_name}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retarget a trajectory from a human video to a robot trajectory in simulation"
    )

    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        default="rpFrankaRobotiqData-v0",
        help="Environment to load",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="hoi4d",
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
        "-gs",
        "--goal_site",
        type=str,
        default="ee_target",
        help="Site that updates as goal using inputs",
    )
    parser.add_argument(
        "-ts",
        "--teleop_site",
        type=str,
        default="end_effector",
        help="Site used for teleOp/target for IK",
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
        args.seed,
        args.goal_site,
        args.teleop_site,
        args.input_path,
        args.output_path,
    )
