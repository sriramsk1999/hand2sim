import numpy as np
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka, KinematicsSolver
from omni.isaac.sensor import Camera
from omni.isaac.core.robots import Robot
import omni.isaac.core.utils.stage as stage_utils


class IsaacSimRetargetEnv:
    def __init__(self, seed, embodiment, simulation_app):
        self.seed = seed
        self.resolution = (320, 240)
        self.embodiment = embodiment

        world = World()
        world.scene.add_default_ground_plane()

        franka = self.setup_embodiment(embodiment, world)
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
        franka.initialize()
        camera.initialize()

        self.franka = franka
        self.franka_IK = franka_IK
        self.hand_action = ArticulationAction()
        self.camera = camera
        self.world = world
        self.simulation_app = simulation_app
        self.init_ee_pose = np.eye(4)
        ee_pos, ee_rot = franka_IK.compute_end_effector_pose()
        self.init_ee_pose[:3, 3] = ee_pos
        self.init_ee_pose[:3, :3] = ee_rot

        # min-max range of x/y/z values signifying the valid range of end effector positions
        self.ee_range = np.array([0.3, 0.6, -0.25, 0.25, 0.3, 0.6])

    def setup_embodiment(self, embodiment, world):
        if embodiment == "pjaw":
            # Add the Franka robot + articulation + IK solver
            franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))
            self.gripper = franka.gripper
        elif embodiment == "allegro":
            franka_allegro_usd_path = "environments/franka_allegro.usd"
            franka_allegro_prim_path = "/World"
            stage_utils.add_reference_to_stage(
                franka_allegro_usd_path, franka_allegro_prim_path
            )
            franka = Robot(
                prim_path=franka_allegro_prim_path,
                name="franka_allegro",
            )
        else:
            raise NotImplementedError(f"Embodiment {embodiment} not supported.")
        return franka

    def align_transform(self, robotWorld2hand):
        # Flip signs of y/z translation. Not entirely sure why this was needed.
        robotWorld2hand[:, 1, -1] *= -1
        robotWorld2hand[:, 2, -1] *= -1
        return robotWorld2hand

    def step_and_render(self, step_num, ee_pose, hand_action):
        # Keep the tolerance relatively high to avoid jerky movements
        # TODO: Find a better way to enforce smoothness for the solver
        action, success = self.franka_IK.compute_inverse_kinematics(
            ee_pose[:3],
            ee_pose[3:7],
            position_tolerance=1e-2,
            orientation_tolerance=5e-2,
        )

        if success:
            self.franka.apply_action(action)
            self.apply_hand_action(self.embodiment, hand_action)
        else:
            print(f"IK(t:{step_num}):: Status:{success}")

        # Step simulation
        self.world.step(render=True)

        rgb_data = self.camera.get_rgb()
        if rgb_data.shape[0] == 0:  # The first 2-3 images are an empty array
            rgb_data = np.zeros((self.resolution[1], self.resolution[0], 3))
        return rgb_data.astype(np.uint8)

    def apply_hand_action(self, embodiment, hand_action):
        if embodiment == "pjaw":
            # Valid values in [0,0.05] where 0.05 means open.
            # Incoming labels are in {0, 1} where 1 is closed.
            # Repeat for left/right finger
            self.hand_action.joint_positions = 0.05 - (
                np.array([hand_action, hand_action]) / 20
            )
            self.gripper.apply_action(self.hand_action)
        elif embodiment == "allegro":
            pass
        else:
            raise NotImplementedError

    def close(self):
        self.simulation_app.close()
