import os
import pickle
from glob import glob

import cv2
import numpy as np
import open3d as o3d

from datasets.base_dataset import BaseDataset


class HOI4DDataset(BaseDataset):
    def __init__(self, base_path):
        super().__init__(base_path)
        self.real_img_path = f"{base_path}/align_rgb/*jpg"
        self.viz_fps = 30

    def load_trajectory(self):
        """loads a trajectory from hoi4d
        - validIdxs - mask of frames in which a hand is visible
        - camWorld2hand - The trajectory of the hand wrt /world/ camera (first frame of video)
        - hoiContact - Binary array indicating hand/object contact in every frame

        This function uses the camera trajectory and hand pose in each frame
        to compute the camWorld2hand trajectory. validIdxs is given in the
        dataset and hoiContact is computed by checking if the hand segmask
        overlaps with the object segmask.
        """
        cam_trajectory = o3d.io.read_pinhole_camera_trajectory(
            f"{self.base_path}/3Dseg/output.log"
        )
        K = np.array(
            [
                [1.0602955e03, 0.0000000e00, 9.7152105e02],
                [0.0000000e00, 1.0615068e03, 5.2326190e02],
                [0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
        cam2camWorld = np.array([i.extrinsic for i in cam_trajectory.parameters])
        camWorld2cam = np.linalg.inv(cam2camWorld)

        cam2hand = []
        # in some frames, the hand might not be visible, filter these out
        validIdxs = np.array(
            sorted(
                [int(i.split(".")[0]) for i in os.listdir(f"{self.base_path}/handpose")]
            )
        )
        for idx in validIdxs:
            f = open(f"{self.base_path}/handpose/{idx}.pickle", "rb")
            data = pickle.load(f)
            pose = np.eye(4)
            global_rot = data["poseCoeff"][:3]
            global_rot = cv2.Rodrigues(global_rot)[0]
            pose[:3, :3] = global_rot
            pose[:3, 3] = data["trans"]
            cam2hand.append(pose)
            f.close()
        cam2hand = np.array(cam2hand)

        seg_dir = f"{self.base_path}/2Dseg/shift_mask/"
        if not os.path.exists(seg_dir):
            seg_dir = f"{self.base_path}/2Dseg/mask/"

        segmasks = np.array([cv2.imread(i) for i in sorted(glob(f"{seg_dir}/*png"))])

        camWorld2hand = camWorld2cam[validIdxs] @ cam2hand
        hoiContact = self.check_hand_object_contact(segmasks)[validIdxs]
        return validIdxs, camWorld2hand, hoiContact

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
