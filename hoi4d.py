import os
import pickle
from glob import glob

import cv2
import numpy as np
import open3d as o3d
from robohive.utils.quat_math import euler2mat, mat2quat


class HOI4DDataset:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_trajectory(self, ee_pose):
        """
        Load the trajectory at the base_path
        TODO: Currently just handles one trajectory
        """
        valid_idxs, camWorld2hand, handObjectContact = self.load_hoi4d_trajectory(
            self.base_path
        )
        trajectory = self.retarget_hand_trajectory(
            camWorld2hand, ee_pose, handObjectContact
        )
        return trajectory, valid_idxs

    def load_hoi4d_trajectory(self, base_path):
        """loads a test trajectory.
        - validIdxs - mask of frames in which a hand is visible
        - camWorld2hand - The trajectory of the hand wrt /world/ camera (first frame of video)
        - hoiContact - Binary array indicating hand/object contact in every frame

        This function uses the camera trajectory and hand pose in each frame
        to compute the camWorld2hand trajectory. validIdxs is given in the
        dataset and hoiContact is computed by checking if the hand segmask
        overlaps with the object segmask.
        """
        cam_trajectory = o3d.io.read_pinhole_camera_trajectory(
            f"{base_path}/3Dseg/output.log"
        )
        K = cam_trajectory.parameters[0].intrinsic.intrinsic_matrix
        cam2camWorld = np.array([i.extrinsic for i in cam_trajectory.parameters])
        camWorld2cam = np.linalg.inv(cam2camWorld)

        cam2hand = []
        # in some frames, the hand might not be visible, filter these out
        validIdxs = np.array(
            sorted([int(i.split(".")[0]) for i in os.listdir(f"{base_path}/handpose")])
        )
        for idx in validIdxs:
            f = open(f"{base_path}/handpose/{idx}.pickle", "rb")
            data = pickle.load(f)
            pose = np.eye(4)
            global_rot = data["poseCoeff"][:3]
            global_rot = cv2.Rodrigues(global_rot)[0]
            pose[:3, :3] = global_rot
            pose[:3, 3] = data["trans"]
            cam2hand.append(pose)
            f.close()
        cam2hand = np.array(cam2hand)

        seg_dir = f"{base_path}/2Dseg/shift_mask/"
        if not os.path.exists(seg_dir):
            seg_dir = f"{base_path}/2Dseg/mask/"

        segmasks = np.array([cv2.imread(i) for i in sorted(glob(f"{seg_dir}/*png"))])

        camWorld2hand = camWorld2cam[validIdxs] @ cam2hand
        hoiContact = self.check_hand_object_contact(segmasks)[validIdxs]
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

    def check_hand_object_contact(self, segmasks):
        """
        Check for contact between a hand mask (in green) and any colored objects
        in a list of segmentation masks. Specifically for HOI4D data
        """
        handObjectContact = []
        lower_green = np.array([0, 128, 0])  # Lower bound for green in BGR
        upper_green = np.array([100, 255, 100])  # Upper bound for green in BGR
        kernel = np.ones((5, 5), np.uint8)

        for segmask in segmasks:
            hand_mask = cv2.inRange(segmask, lower_green, upper_green)
            non_black_mask = cv2.cvtColor(segmask, cv2.COLOR_BGR2GRAY)
            non_black_mask[non_black_mask > 0] = (
                255  # Set all non-black pixels to white
            )
            non_black_mask[non_black_mask == 0] = 0
            object_mask = segmask > 0
            object_mask = cv2.bitwise_and(non_black_mask, cv2.bitwise_not(hand_mask))

            hand_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
            object_dilated = cv2.dilate(object_mask, kernel, iterations=1)
            overlap = cv2.bitwise_and(hand_dilated, object_dilated)

            if np.any(overlap):
                handObjectContact.append(1)
            else:
                handObjectContact.append(0)
        return np.array(handObjectContact)
