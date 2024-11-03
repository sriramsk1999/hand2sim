import os
import pickle

import cv2
import numpy as np
import open3d as o3d
from robohive.utils.quat_math import euler2mat, mat2quat


def load_hoi4d_trajectory(base_path):
    """loads a test trajectory from a hardcoded path.
    The minimum requirement is the hand pose in the
    camera frame (cam2hand) and the camera pose in the world (world2cam).
    """
    cam_trajectory = o3d.io.read_pinhole_camera_trajectory(
        f"{base_path}/3Dseg/output.log"
    )
    K = cam_trajectory.parameters[0].intrinsic.intrinsic_matrix
    cam2camWorld = np.array([i.extrinsic for i in cam_trajectory.parameters])
    camWorld2cam = np.linalg.inv(cam2camWorld)

    cam2hand = []
    # in some frames, the hand might not be visible, filter these out
    idxs = np.array(
        sorted([int(i.split(".")[0]) for i in os.listdir(f"{base_path}/handpose")])
    )
    for idx in idxs:
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

    camWorld2hand = camWorld2cam[idxs] @ cam2hand
    return idxs, camWorld2hand


def retarget_hand_trajectory(camWorld2hand, robotWorld2ee):
    """
    Align hand coordinates with end effector.
    retarget the 4x4 extrinsics to quaternion/translation/gripper state
    """
    scale_factor = 0.75
    # TODO: Scale translation more intelligently
    # Scale down the translation of the trajectory
    # so that end effector does not go out of reach of the robot
    camWorld2hand[:, :3, 3] *= scale_factor

    robotWorld2camWorld = robotWorld2ee @ np.linalg.inv(camWorld2hand[0])
    robotWorld2hand = robotWorld2camWorld @ camWorld2hand

    # An arbitrary rotation to align the trajectory correctly with robot
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = euler2mat((0, np.pi, 0))
    robotWorld2hand = robotWorld2hand @ align_rotation

    robot_trajectory_quat = mat2quat(robotWorld2hand[:, :3, :3])
    robot_trajectory_pos = robotWorld2hand[:, :3, 3]

    # TODO: Handle gripper state properly
    robot_gripper_state = np.zeros((robotWorld2hand.shape[0], 1))
    trajectory = np.concatenate(
        [robot_trajectory_pos, robot_trajectory_quat, robot_gripper_state], axis=-1
    )
    return trajectory
