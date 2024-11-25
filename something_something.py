from glob import glob

import cv2
import h5py
import numpy as np
from robohive.utils.quat_math import euler2mat, mat2quat, quat2mat
from utils import read_video_from_path


class SomethingSomethingDataset:
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
        """loads a trajectory from something-something
        - validIdxs - mask of frames in which a hand is visible
        - camWorld2hand - The trajectory of the hand wrt /world/ camera (first frame of video)
        - hoiContact - Binary array indicating hand/object contact in every frame

        This function uses the camera trajectory and hand pose in each frame
        to compute the camWorld2hand trajectory. Parameters are estimated
        using various off-the-shelf models [Monst3r, hamer, ContactHands]
        """
        cam_trajectory = np.loadtxt(f"{self.base_path}/monst3r/pred_traj.txt")

        vid_name = self.base_path.split("/")[-1]
        orig_size = read_video_from_path(f"{self.base_path}/{vid_name}.webm").shape[
            1:-1
        ]
        orig_size = np.array(orig_size)[::-1]

        cam_xyz = cam_trajectory[:, 1:4]
        cam_quat_wxyz = cam_trajectory[:, 4:]
        cam_rot_mat = quat2mat(cam_quat_wxyz)

        camWorld2cam = np.zeros((cam_trajectory.shape[0], 4, 4))
        camWorld2cam[:, :3, :3] = cam_rot_mat
        camWorld2cam[:, :3, 3] = cam_xyz
        camWorld2cam[:, 3, 3] = 1

        K = np.loadtxt(f"{self.base_path}/monst3r/pred_intrinsics.txt")[0].reshape(3, 3)

        with h5py.File(f"{self.base_path}/hamer.h5") as hf:
            validIdxs = np.asarray(hf["frame_idxs"])
            handRotations = np.asarray(hf["rotations"])
            wristKeypoint = np.asarray(hf["wrist_keypoints"])

        fnames = sorted(glob(f"{self.base_path}/monst3r/frame_*.npy"))
        depths = np.array([np.load(f) for f in fnames])

        keypoints = self.map_keypoint_to_resized_img(
            wristKeypoint, orig_size, depths[0].shape
        )
        mask1 = keypoints.min(1) > 0
        mask2 = keypoints.max(1) < 224
        validHamerMask = np.logical_and(mask1, mask2)

        validIdxs = validIdxs[validHamerMask]
        keypoints = keypoints[validHamerMask]
        handRotations = handRotations[validHamerMask]

        depths = depths[validIdxs]
        keypoint_depth = depths[
            np.arange(depths.shape[0]), keypoints[:, 1], keypoints[:, 0]
        ]

        K_inv = np.linalg.inv(K)
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        keypoints_normalized = keypoints_homogeneous @ K_inv.T
        handTranslations = keypoints_normalized * keypoint_depth[:, np.newaxis]

        cam2hand = np.zeros((handTranslations.shape[0], 4, 4))
        cam2hand[:, :3, :3] = handRotations
        cam2hand[:, :3, 3] = handTranslations
        cam2hand[:, 3, 3] = 1

        hoiContact = np.load(f"{self.base_path}/contactHands.npy")[validIdxs]

        camWorld2hand = camWorld2cam[validIdxs] @ cam2hand
        return validIdxs, camWorld2hand, hoiContact

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

    def map_keypoint_to_resized_img(self, keypoint, orig_size, new_size):
        """Map hand keypoint to resized image.
        Reverses the preprocessing for Monst3r.
        In Monst3r the shorter side to resized to the new size to
        maintain the aspect ratio and then the image is center cropped."""

        # Scale factor based on resizing shorter side to 224
        scale = max(new_size / orig_size)

        # Resized dimensions
        resized_orig = orig_size * scale

        # Map keypoint to resized (uncropped) coordinates
        resized_kp = keypoint * scale

        # Calculate crop offsets (one will be 0)
        offset = (resized_orig - new_size) / 2

        # Adjust for crop to get final keypoints
        resized_kp = resized_kp - offset

        return resized_kp.round().astype(int)

    def write_real_sim_video(self, sim_imgs, base_path, valid_idxs, output_path):
        """
        Visualize sim video and real video side-by-side.
        """
        sim_imgs = np.array([cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in sim_imgs])

        real_imgs = np.array(
            [cv2.imread(i) for i in sorted(glob(f"{base_path}/monst3r/frame*png"))]
        )
        real_imgs = real_imgs[valid_idxs]
        real_imgs = np.array(
            [cv2.resize(i, (sim_imgs.shape[2], sim_imgs.shape[1])) for i in real_imgs]
        )
        real_and_sim = np.array(
            [cv2.hconcat([i, j]) for i, j in zip(real_imgs, sim_imgs)]
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, 5, (real_and_sim.shape[2], real_and_sim.shape[1])
        )

        for frame in real_and_sim:
            out.write(frame)
        out.release()
