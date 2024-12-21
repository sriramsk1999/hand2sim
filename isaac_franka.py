from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.franka import Franka, KinematicsSolver
from omni.isaac.sensor import Camera
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_random_trajectory(trajectory_len):
    """Generate a random reachable position for the end effector"""
    # Define workspace limits for the end effector
    x_range = (0.3, 0.7)  # Forward/backward
    y_range = (-0.4, 0.4)  # Left/right
    z_range = (0.2, 0.8)  # Up/down

    pos1 = np.array(
        [random.uniform(*x_range), random.uniform(*y_range), random.uniform(*z_range)]
    )

    pos2 = np.array(
        [random.uniform(*x_range), random.uniform(*y_range), random.uniform(*z_range)]
    )

    # Keep the end effector oriented downward
    # Rotation in quaternion (w, x, y, z)
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    gripper = np.array([0, 0])

    position1 = np.concatenate([pos1, orientation, gripper])
    position2 = np.concatenate([pos2, orientation, gripper])

    trajectory = np.linspace(position1, position2, trajectory_len)
    return trajectory


SAVE_DIR = "isaac_viz"
os.makedirs(SAVE_DIR, exist_ok=True)


# Create a new world
world = World()
# Add ground plane
world.scene.add_default_ground_plane()
# Add the Franka robot + articulation + IK solver
franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))
franka_articulation = Articulation(franka.prim_path)
franka_IK = KinematicsSolver(franka_articulation)

camera = Camera(
    prim_path="/World/camera",
    position=np.array([-1, 1.5, 1]),
    orientation=np.array([1, 0, 0.25, -0.35]),
    resolution=(320, 240),
)
camera.set_focal_length(3)
world.scene.add(camera)

# Reset the world
world.reset()

franka_articulation.initialize()
camera.initialize()
trajectory_len = 100

trajectory = generate_random_trajectory(trajectory_len)

for i in tqdm(range(trajectory_len)):
    action, success = franka_IK.compute_inverse_kinematics(
        trajectory[i, :3], trajectory[i, 3:]
    )
    franka.apply_action(action)

    # Step simulation
    world.step(render=True)

    rgb_data = camera.get_rgb()
    if rgb_data.shape[0] != 0:
        plt.imsave(f"{SAVE_DIR}/{str(i).zfill(5)}.png", rgb_data)

simulation_app.close()
