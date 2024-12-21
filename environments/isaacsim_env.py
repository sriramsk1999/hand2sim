import numpy as np
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.franka import Franka, KinematicsSolver
from omni.isaac.sensor import Camera


class IsaacSimRetargetEnv:
    def __init__(self, seed, simulation_app):
        self.seed = seed
        self.resolution = (320, 240)

        world = World()
        world.scene.add_default_ground_plane()

        # Add the Franka robot + articulation + IK solver
        franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))
        franka_articulation = Articulation(franka.prim_path)
        franka_IK = KinematicsSolver(franka_articulation)
        franka_IK_solver = franka_IK.get_kinematics_solver()
        franka_IK_solver.bfgs_max_iterations = 200

        camera = Camera(
            prim_path="/World/camera",
            position=np.array([-1, 1.5, 1]),
            orientation=np.array([1, 0, 0.25, -0.35]),
            resolution=self.resolution,
        )
        camera.set_focal_length(3)
        world.scene.add(camera)

        # Reset the world
        world.reset()

        franka_articulation.initialize()
        camera.initialize()

        self.franka = franka
        self.franka_IK = franka_IK
        self.camera = camera
        self.world = world
        self.simulation_app = simulation_app
        self.init_ee_pose = np.eye(4)
        ee_pos, ee_rot = franka_IK.compute_end_effector_pose()
        self.init_ee_pose[:3, 3] = ee_pos
        self.init_ee_pose[:3, :3] = ee_rot

        self.ee_range = np.array([0.2, 0.5, -0.25, 0.25, 0.2, 0.5])

    def step_and_render(self, step_num, next_pose):
        action, success = self.franka_IK.compute_inverse_kinematics(
            next_pose[:3], next_pose[3:7]
        )
        if success == False:
            print(f"IK(t:{step_num}):: Status:{success}")

        self.franka.apply_action(action)

        # Step simulation
        self.world.step(render=True)

        rgb_data = self.camera.get_rgb()
        if rgb_data.shape[0] == 0:
            rgb_data = np.zeros((self.resolution[1], self.resolution[0], 3))
        return rgb_data.astype(np.uint8)

    def close(self):
        self.simulation_app.close()
