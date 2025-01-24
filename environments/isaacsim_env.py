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
        self.franka_articulation = franka_articulation
        self.joint_properties = franka_articulation.dof_properties

        # min-max range of x/y/z values signifying the valid range of end effector positions
        self.ee_range = np.array([0.3, 0.6, -0.25, 0.25, 0.3, 0.6])

    def setup_embodiment(self, embodiment, world):
        """
        Load the required embodiment into the simulation.
        """
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
            self.apply_action(self.embodiment, action, hand_action)
        else:
            print(f"IK(t:{step_num}):: Status:{success}")

        # Step simulation
        self.world.step(render=True)

        rgb_data = self.camera.get_rgb()
        if rgb_data.shape[0] == 0:  # The first 2-3 images are an empty array
            rgb_data = np.zeros((self.resolution[1], self.resolution[0], 3))
        return rgb_data.astype(np.uint8)

    def apply_action(self, embodiment, action, hand_action):
        """
        Apply action to robot joints based on solved IK pose
        Similarly retarget the hand action as well.
        """
        if embodiment == "pjaw":
            # Valid values in [0,0.05] where 0.05 means open.
            # Incoming labels are in {0, 1} where 1 is closed.
            # Repeat for left/right finger
            self.hand_action.joint_positions = 0.05 - (
                np.array([hand_action, hand_action]) / 20
            )
            self.franka.apply_action(action)
            self.gripper.apply_action(self.hand_action)
        elif embodiment == "allegro":
            allegro_action = ArticulationAction()
            allegro_action.joint_positions = np.concatenate(
                [action.joint_positions, np.zeros(16)]
            )

            lower = self.joint_properties["lower"]
            upper = self.joint_properties["upper"]

            # Assuming norm of angles lie in [0, pi/2], scale to lower/upper bound of allegro joint
            def map_angle_to_allegro(angle, lower, upper):
                return lower + angle * (upper - lower) / (np.pi / 2)

            # Skip retargeting the abduction joints, except for the thumb.
            # Take the norm of each joint's 3 joint angles. Since the finger joints have
            # one primary DoF, taking the norm should be a decent approximation for
            # the magnitude of actuation.
            # Indexes for theta angles:
            # 0-3 -> global rotation, used for end effector IK
            # 3-12 -> index finger joints
            # 12-21 -> middle finger
            # 21-30 -> pinky finger (pinky and ring are swapped?)
            # 30-39 -> ring finger
            # 39-48 -> thumb
            retargeted_angles = {
                "index_joint_0": None,
                "middle_joint_0": None,
                "ring_joint_0": None,
                "thumb_joint_0": np.linalg.norm(hand_action[39:42]),
                "index_joint_1": np.linalg.norm(hand_action[3:6]),
                "middle_joint_1": np.linalg.norm(hand_action[12:15]),
                "ring_joint_1": np.linalg.norm(hand_action[30:33]),
                "thumb_joint_1": None,
                "index_joint_2": np.linalg.norm(hand_action[6:9]),
                "middle_joint_2": np.linalg.norm(hand_action[15:18]),
                "ring_joint_2": np.linalg.norm(hand_action[33:36]),
                "thumb_joint_2": np.linalg.norm(hand_action[42:45]),
                "index_joint_3": np.linalg.norm(hand_action[9:12]),
                "middle_joint_3": np.linalg.norm(hand_action[18:21]),
                "ring_joint_3": np.linalg.norm(hand_action[36:39]),
                "thumb_joint_3": np.linalg.norm(hand_action[45:48]),
            }

            # Indexes for Franka-Allegro joints
            # 0-7 -> Franka arm
            # 7/8/9/10 - index/middle/ring/thumb abduct
            # 11/12/13/14 - index/middle/ring/thumb knuckle
            # 15/16/17/18 - index/middle/ring/thumb joint2
            # 19/20/21/22 - index/middle/ring/thumb joint3
            finger_mappings = [
                (10, "thumb_joint_0"),
                (11, "index_joint_1"),
                (12, "middle_joint_1"),
                (13, "ring_joint_1"),
                (15, "index_joint_2"),
                (16, "middle_joint_2"),
                (17, "ring_joint_2"),
                (18, "thumb_joint_2"),
                (19, "index_joint_3"),
                (20, "middle_joint_3"),
                (21, "ring_joint_3"),
                (22, "thumb_joint_3"),
            ]

            for joint_index, joint_name in finger_mappings:
                allegro_action.joint_positions[joint_index] = map_angle_to_allegro(
                    retargeted_angles[joint_name],
                    lower[joint_index],
                    upper[joint_index],
                )
            self.franka.apply_action(allegro_action)
        else:
            raise NotImplementedError

    def close(self):
        self.simulation_app.close()
