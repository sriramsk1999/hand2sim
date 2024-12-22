from glob import glob

import cv2
import numpy as np
from robohive.utils.quat_math import mat2quat
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


class BaseDataset:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_trajectory(self, ee_pose, ee_range, align_transform):
        """
        Load the trajectory at `base_path`
        """
        valid_idxs, camWorld2hand, handObjectContact = self.load_trajectory()
        trajectory = self.retarget_hand_trajectory(
            camWorld2hand, ee_pose, handObjectContact, ee_range, align_transform
        )
        smooth_trajectory = self.smooth_trajectory(trajectory)
        return smooth_trajectory, valid_idxs

    def load_trajectory(self):
        raise NotImplementedError("To be implemented in subclass")

    def smooth_trajectory(self, trajectory, window=5, poly_order=3):
        """
        Smooth trajectory using Savitzky-Golay for positions/gripper
        and Slerp for quaternions

        Args:
        - trajectory: numpy array of shape (N, 8)
            - First 3: xyz positions
            - Next 4: quaternion (w,x,y,z)
            - Last: gripper position
        - window: smoothing window size (must be odd)
        - poly_order: polynomial order for fitting

        Returns:
        Smoothed trajectory
        """
        # Ensure window is odd
        window = window if window % 2 == 1 else window + 1
        half_window = window // 2

        smoothed = np.zeros_like(trajectory)

        # Smooth positions and gripper
        for col in [0, 1, 2, 7]:
            smoothed[:, col] = savgol_filter(
                trajectory[:, col], window_length=window, polyorder=poly_order
            )

        # Smooth quaternions using Slerp
        for i in range(len(trajectory)):
            # Define window bounds
            start = max(0, i - half_window)
            end = min(len(trajectory), i + half_window + 1)

            # Extract window quaternions
            window_quats = trajectory[start:end, 3:7]

            # Convert to rotation objects
            rots = Rotation.from_quat(window_quats)

            # Average rotations
            avg_rot = rots.mean()
            avg_quat = avg_rot.as_quat()
            # prefer positive w
            if avg_quat[0] < 0:
                avg_quat *= -1

            # Convert back to quaternion
            smoothed[i, 3:7] = avg_quat
        return smoothed

    def retarget_hand_trajectory(
        self, camWorld2hand, robotWorld2ee, handObjectContact, ee_range, align_transform
    ):
        """
        Align hand coordinates with end effector.
        retarget the 4x4 extrinsics to quaternion/translation/gripper state
        """
        robotWorld2camWorld = robotWorld2ee @ np.linalg.inv(camWorld2hand[0])
        robotWorld2hand = robotWorld2camWorld @ camWorld2hand
        robotWorld2hand = align_transform(robotWorld2hand)

        # Target ranges based on the reachable space of the robot
        # Manually estimated with some teleop
        target_x = ee_range[:2]
        target_y = ee_range[2:4]
        target_z = ee_range[4:]

        # Scale and shift the translation to fit into the robot workspace
        for i, (min_target, max_target) in enumerate([target_x, target_y, target_z]):
            v_min = np.min(robotWorld2hand[:, i, -1])
            v_max = np.max(robotWorld2hand[:, i, -1])
            robotWorld2hand[:, i, -1] = min_target + (
                (robotWorld2hand[:, i, -1] - v_min)
                * (max_target - min_target)
                / (v_max - v_min)
            )

        robot_trajectory_quat = mat2quat(robotWorld2hand[:, :3, :3])  # w,x,y,z
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
