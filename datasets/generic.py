from glob import glob

import h5py
import numpy as np
from robohive.utils.quat_math import quat2mat
from utils import read_video_from_path

from datasets.base_dataset import BaseDataset


class GenericDataset(BaseDataset):
    def __init__(self, base_path):
        super().__init__(base_path)
        self.monst3r_dim = 224
        self.real_img_path = f"{base_path}/monst3r/frame*png"
        self.viz_fps = 5

    def load_trajectory(self):
        """loads a trajectory from generic video
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
        mask2 = keypoints.max(1) < self.monst3r_dim
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
