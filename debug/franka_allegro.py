from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot
import omni.isaac.core.utils.stage as stage_utils

import numpy as np
from tqdm import tqdm

from manopth.manolayer import ManoLayer
from manopth import demo
from utils import add_back_legacy_types_numpy
import torch


def joint_angle(A, B, C):
    """Calculate joint angle between 3 3D keypoints A/B/C"""
    BA = A - B
    BC = C - B
    dot_product = np.dot(BA, BC)
    magnitudes = np.linalg.norm(BA) * np.linalg.norm(BC)
    angle_rad = np.arccos(dot_product / magnitudes)
    return angle_rad


def map_angle_to_allegro(angle, lower, upper):
    return lower + angle * (upper - lower) / (torch.pi / 2)


add_back_legacy_types_numpy()
mano_layer = ManoLayer(mano_root="mano/models", use_pca=False, ncomps=45)

thetas = torch.zeros(1, 48)
thetas[:, 3:6] = 1
betas = torch.zeros(1, 10)
# thetas = torch.from_numpy(np.load("theta.npy")[None])
# betas = torch.from_numpy(np.load("beta.npy")[None])
hand_verts, hand_joints = mano_layer(thetas, betas)

demo.display_hand(
    {"verts": hand_verts, "joints": hand_joints},
    mano_faces=mano_layer.th_faces,
)

retargeted_angles = {
    "index_joint_0": None,
    "middle_joint_0": None,
    "ring_joint_0": None,
    "thumb_joint_0": torch.linalg.norm(thetas[:, 39:42]),
    "index_joint_1": torch.linalg.norm(thetas[:, 3:6]),
    "middle_joint_1": torch.linalg.norm(thetas[:, 12:15]),
    "ring_joint_1": torch.linalg.norm(thetas[:, 30:33]),
    "thumb_joint_1": None,
    "index_joint_2": torch.linalg.norm(thetas[:, 6:9]),
    "middle_joint_2": torch.linalg.norm(thetas[:, 15:18]),
    "ring_joint_2": torch.linalg.norm(thetas[:, 33:36]),
    "thumb_joint_2": torch.linalg.norm(thetas[:, 42:45]),
    "index_joint_3": torch.linalg.norm(thetas[:, 9:12]),
    "middle_joint_3": torch.linalg.norm(thetas[:, 18:21]),
    "ring_joint_3": torch.linalg.norm(thetas[:, 36:39]),
    "thumb_joint_3": torch.linalg.norm(thetas[:, 45:48]),
}

# retargeted_angles = {
#     "index_joint_0": None,
#     "middle_joint_0": None,
#     "ring_joint_0": None,
#     "thumb_joint_0": joint_angle(hand_joint[0], hand_joint[1], hand_joint[2]),
#     "index_joint_1": joint_angle(hand_joint[0], hand_joint[5], hand_joint[6]),
#     "middle_joint_1": joint_angle(hand_joint[0], hand_joint[9], hand_joint[10]),
#     "ring_joint_1": joint_angle(hand_joint[0], hand_joint[13], hand_joint[14]),
#     "thumb_joint_1": None,
#     "index_joint_2": joint_angle(hand_joint[5], hand_joint[6], hand_joint[7]),
#     "middle_joint_2": joint_angle(hand_joint[9], hand_joint[10], hand_joint[11]),
#     "ring_joint_2": joint_angle(hand_joint[13], hand_joint[14], hand_joint[15]),
#     "thumb_joint_2": joint_angle(hand_joint[1], hand_joint[2], hand_joint[3]),
#     "index_joint_3": joint_angle(hand_joint[6], hand_joint[7], hand_joint[8]),
#     "middle_joint_3": joint_angle(hand_joint[10], hand_joint[11], hand_joint[12]),
#     "ring_joint_3": joint_angle(hand_joint[14], hand_joint[15], hand_joint[16]),
#     "thumb_joint_3": joint_angle(hand_joint[2], hand_joint[3], hand_joint[4]),
# }


# Create a new world
world = World()
# Add ground plane
world.scene.add_default_ground_plane()

franka_allegro_usd_path = "environments/franka_allegro.usd"
franka_allegro_prim_path = "/World"
stage_utils.add_reference_to_stage(franka_allegro_usd_path, franka_allegro_prim_path)
franka_allegro = Robot(
    prim_path=franka_allegro_prim_path,
    name="franka_allegro",
)
franka_articulation = Articulation(franka_allegro.prim_path)

# Reset the world
world.reset()

franka_articulation.initialize()
franka_allegro.initialize()

trajectory_len = 1000
# t1 = np.random.random(16)
# t2 = np.random.random(16)
# trajectory = np.linspace(t1, t2, trajectory_len)

joint_properties = franka_articulation.dof_properties
lower = joint_properties["lower"]
upper = joint_properties["upper"]

initial_position = franka_articulation.get_joint_positions()
final_position = initial_position.copy()

final_position[10] = retargeted_angles["thumb_joint_0"]
final_position[11] = map_angle_to_allegro(
    retargeted_angles["index_joint_1"], lower[11], upper[11]
)
final_position[12] = retargeted_angles["middle_joint_1"]
final_position[13] = retargeted_angles["ring_joint_1"]
final_position[15] = retargeted_angles["index_joint_2"]
final_position[16] = retargeted_angles["middle_joint_2"]
final_position[17] = retargeted_angles["ring_joint_2"]
final_position[18] = retargeted_angles["thumb_joint_2"]
final_position[19] = retargeted_angles["index_joint_3"]
final_position[20] = retargeted_angles["middle_joint_3"]
final_position[21] = retargeted_angles["ring_joint_3"]
final_position[22] = retargeted_angles["thumb_joint_3"]
trajectory = np.linspace(initial_position, final_position, trajectory_len)

action = ArticulationAction()
for i in tqdm(range(trajectory_len)):
    action.joint_positions = trajectory[i]

    franka_allegro.apply_action(action)

    # Step simulation
    world.step(render=True)

simulation_app.close()
