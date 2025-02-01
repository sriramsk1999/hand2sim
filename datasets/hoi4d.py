import os
import pickle
from glob import glob

import cv2
import numpy as np
import open3d as o3d
import torch
from manopth.manolayer import ManoLayer
from utils import add_back_legacy_types_numpy

from datasets.base_dataset import BaseDataset

MANO_ROOT = "mano/models"


class HOI4DDatasetWrapper:
    """A wrapper for the HOI4DDataset class, used to iterate over the entire
    HOI4D dataset whereas HOI4DDataset processes data for a single clip.

    Can probably refactor into a single class ....
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.videos = sorted(glob(f"{self.dataset_dir}/**/image.mp4", recursive=True))
        self.intrinsic_dict = self.load_intrinsics()

    def __len__(self):
        return len(self.videos)

    def load_intrinsics(self):
        """
        Load camera intrinsics for HOI4D. A bit hacky.
        Camera intrinsics are expected to be placed one level above hoi4d dir
        with the directory name "camera_params"
        """
        intrinsic_dict = {}
        for cam_name in [
            "ZY20210800001",
            "ZY20210800002",
            "ZY20210800003",
            "ZY20210800004",
        ]:
            cam_param_pathname = f"{self.dataset_dir}/../camera_params/{cam_name}/"
            intrinsic_dict[cam_name] = np.load(f"{cam_param_pathname}/intrin.npy")
        return intrinsic_dict

    def __getitem__(self, idx):
        vid_path = self.videos[idx]

        dir_name = os.path.dirname(os.path.dirname(vid_path))
        cam_name_start_idx = dir_name.find("ZY2021")
        video_id = dir_name[cam_name_start_idx:]
        cam_name = video_id[:13]
        K = self.intrinsic_dict[cam_name]
        handpose_path = (
            f"{self.dataset_dir}/../handpose/refinehandpose_right/{video_id}"
        )
        item = HOI4DDataset(dir_name, handpose_path=handpose_path, intrinsics=K)
        return item


class HOI4DDataset(BaseDataset):
    """
    Dataset class for *one* event from the HOI4D dataset
    Use the wrapper class to iterate over the entire dataset.
    TODO: Refactor into a single dataset class
    """

    def __init__(self, base_path, handpose_path=None, intrinsics=None):
        super().__init__(base_path)
        self.real_img_path = f"{base_path}/align_rgb/*jpg"
        self.handpose_path = (
            handpose_path
            if handpose_path is not None
            else f"{self.base_path}/handpose/"
        )
        self.viz_fps = 30
        add_back_legacy_types_numpy()
        self.mano_layer = ManoLayer(mano_root="mano/models", use_pca=False, ncomps=45)
        # One of the 4 cameras from the dataset, others are pretty similar
        self.K = (
            intrinsics
            if intrinsics is not None
            else np.array(
                [
                    [1.0602955e03, 0.0000000e00, 9.7152105e02],
                    [0.0000000e00, 1.0615068e03, 5.2326190e02],
                    [0.0000000e00, 0.0000000e00, 1.0000000e00],
                ]
            )
        )

    def load_trajectory(self):
        """loads a trajectory from hoi4d
        - valid_idxs - mask of frames in which a hand is visible
        - camWorld2hand - The trajectory of the hand wrt /world/ camera (first frame of video)
        - hand_object_contact - Binary array indicating hand/object contact in every frame
        - hand_pose3d - 3D hand pose, extracted using MANO

        This function uses the camera trajectory and hand pose in each frame
        to compute the camWorld2hand trajectory. valid_idxs is given in the
        dataset and hand_object_contact is computed by checking if the hand segmask
        overlaps with the object segmask.
        """
        cam_trajectory = o3d.io.read_pinhole_camera_trajectory(
            f"{self.base_path}/3Dseg/output.log"
        )
        cam2camWorld = np.array([i.extrinsic for i in cam_trajectory.parameters])
        camWorld2cam = np.linalg.inv(cam2camWorld)

        cam2hand = []
        thetas, betas = [], []
        # in some frames, the hand might not be visible, filter these out
        valid_idxs = np.array(
            sorted([int(i.split(".")[0]) for i in os.listdir(f"{self.handpose_path}")])
        )
        for idx in valid_idxs:
            f = open(f"{self.handpose_path}/{idx}.pickle", "rb")
            data = pickle.load(f)
            pose = np.eye(4)
            global_rot = data["poseCoeff"][:3]

            thetas.append(data["poseCoeff"])
            betas.append(data["beta"])

            global_rot = cv2.Rodrigues(global_rot)[0]
            pose[:3, :3] = global_rot
            pose[:3, 3] = data["trans"]
            cam2hand.append(pose)
            f.close()
        cam2hand = np.array(cam2hand)
        camWorld2hand = camWorld2cam[valid_idxs] @ cam2hand

        # Hand-object Contact
        seg_dir = f"{self.base_path}/2Dseg/shift_mask/"
        if not os.path.exists(seg_dir):
            seg_dir = f"{self.base_path}/2Dseg/mask/"

        segmasks = np.array([cv2.imread(i) for i in sorted(glob(f"{seg_dir}/*png"))])
        hand_object_contact = self.check_hand_object_contact(segmasks)[valid_idxs]

        # Hand poses
        thetas, betas = torch.from_numpy(np.array(thetas)), torch.from_numpy(
            np.array(betas)
        )
        hand_verts, hand_joints = self.mano_layer(thetas, betas)
        hand_joint_angles = thetas.numpy()

        return valid_idxs, camWorld2hand, hand_object_contact, hand_joint_angles

    def check_hand_object_contact(self, segmasks):
        """
        Check for contact between a hand mask (in green) and any colored objects
        in a list of segmentation masks. Specifically for HOI4D data
        """
        hand_object_contact = []
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
                hand_object_contact.append(1)
            else:
                hand_object_contact.append(0)
        return np.array(hand_object_contact)
