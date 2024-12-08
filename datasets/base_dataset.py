from glob import glob

import cv2
import numpy as np
from robohive.utils.quat_math import euler2mat, mat2quat


class BaseDataset:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_trajectory(self, ee_pose):
        """
        Load the trajectory at `base_path`
        TODO: Currently just handles one trajectory
        """
        valid_idxs, camWorld2hand, handObjectContact = self.load_trajectory()
        trajectory = self.retarget_hand_trajectory(
            camWorld2hand, ee_pose, handObjectContact
        )
        return trajectory, valid_idxs

    def load_trajectory(self):
        raise NotImplementedError("To be implemented in subclass")

    def retarget_hand_trajectory(self, camWorld2hand, robotWorld2ee, handObjectContact):
        """
        Align hand coordinates with end effector.
        retarget the 4x4 extrinsics to quaternion/translation/gripper state
        """
        robotWorld2camWorld = robotWorld2ee @ np.linalg.inv(camWorld2hand[0])
        robotWorld2hand = robotWorld2camWorld @ camWorld2hand

        # An arbitrary rotation to align the trajectory correctly with robot
        align_rotation = np.eye(4)
        align_rotation[:3, :3] = euler2mat((0, np.pi, 0))
        robotWorld2hand = robotWorld2hand @ align_rotation

        # Target ranges based on the reachable space of the robot in a RoboHive env
        # Manually estimated with some teleop
        target_x = np.array([0.2, 0.5])
        target_y = np.array([-0.25, 0.25])
        target_z = np.array([1, 1.3])

        # Scale and shift the translation to fit into the robot workspace
        for i, (min_target, max_target) in enumerate([target_x, target_y, target_z]):
            v_min = np.min(robotWorld2hand[:, i, -1])
            v_max = np.max(robotWorld2hand[:, i, -1])
            robotWorld2hand[:, i, -1] = min_target + (
                (robotWorld2hand[:, i, -1] - v_min)
                * (max_target - min_target)
                / (v_max - v_min)
            )

        robot_trajectory_quat = mat2quat(robotWorld2hand[:, :3, :3])
        robot_trajectory_pos = robotWorld2hand[:, :3, 3]

        robot_gripper_state = handObjectContact[:, None]
        trajectory = np.concatenate(
            [robot_trajectory_pos, robot_trajectory_quat, robot_gripper_state], axis=-1
        )
        return trajectory

    def write_real_sim_video(
        self, sim_imgs, real_img_path, valid_idxs, output_path, fps
    ):
        """
        Visualize sim video and real video side-by-side.
        """
        sim_imgs = np.array([cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in sim_imgs])

        real_imgs = np.array([cv2.imread(i) for i in sorted(glob(real_img_path))])
        real_imgs = real_imgs[valid_idxs]
        real_imgs = np.array(
            [cv2.resize(i, (sim_imgs.shape[2], sim_imgs.shape[1])) for i in real_imgs]
        )
        real_and_sim = np.array(
            [cv2.hconcat([i, j]) for i, j in zip(real_imgs, sim_imgs)]
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (real_and_sim.shape[2], real_and_sim.shape[1])
        )

        for frame in real_and_sim:
            out.write(frame)
        out.release()
