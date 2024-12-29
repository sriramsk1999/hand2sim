from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot
import omni.isaac.core.utils.stage as stage_utils

import numpy as np
from tqdm import tqdm

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
t1 = np.random.random(16)
t2 = np.random.random(16)
trajectory = np.linspace(t1, t2, trajectory_len)

action = ArticulationAction()
for i in tqdm(range(trajectory_len)):
    # Keep arm stationary, move hand randomly
    curr_joint_positions = franka_articulation.get_joint_positions()
    curr_joint_positions[7:] = trajectory[i]
    action.joint_positions = curr_joint_positions

    franka_allegro.apply_action(action)

    # Step simulation
    world.step(render=True)

simulation_app.close()
