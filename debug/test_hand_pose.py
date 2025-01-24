import numpy as np

from manopth.manolayer import ManoLayer
from manopth import demo
from utils import add_back_legacy_types_numpy
import torch

add_back_legacy_types_numpy()
mano_layer = ManoLayer(mano_root="mano/models", use_pca=False, ncomps=45)

thetas = torch.zeros(1, 48)
thetas[:, 45:] = np.pi / 2
betas = torch.zeros(1, 10)
hand_verts, hand_joints = mano_layer(thetas, betas)

demo.display_hand(
    {"verts": hand_verts, "joints": hand_joints},
    mano_faces=mano_layer.th_faces,
)
