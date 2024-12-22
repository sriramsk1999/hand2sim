import gymnasium as gym
import numpy as np
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.quat_math import quat2mat, euler2mat
from utils import set_initial_ee_target


class RoboHiveRetargetEnv:
    def __init__(self, seed):
        goal_site = "ee_target"  # Site that updates as goal using inputs
        teleop_site = "end_effector"  # Site used for teleOp/target for IK
        # seed and load environments
        env = gym.make("rpFrankaRobotiqData-v0")
        env.seed(seed)

        env.env.mujoco_render_frames = False
        goal_sid = env.sim.model.site_name2id(goal_site)

        # Place end effector in a more suitable location / orientation
        translation = [0.4, 0, 1.1]
        rotation = [0, 0, 0, 1]
        set_initial_ee_target(env, goal_sid, translation, rotation)

        env.reset()
        curr_pos = env.sim.model.site_pos[goal_sid]
        curr_quat = env.sim.model.site_quat[goal_sid]
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = quat2mat(curr_quat)
        ee_pose[:3, 3] = curr_pos

        # recover init state
        _ = env.forward()

        self.env = env
        self.init_ee_pose = ee_pose
        self.goal_sid = goal_sid
        self.teleop_site = teleop_site
        self.action_shape = env.action_space.shape

        # min-max range of x/y/z values signifying the valid range of end effector positions
        # Manually estimated through some teleop ...
        self.ee_range = np.array([0.2, 0.5, -0.25, 0.25, 1, 1.3])

    def align_transform(self, robotWorld2hand):
        """Rotate about y to replay correctly in robohive"""
        align_rotation = np.eye(4)
        align_rotation[:3, :3] = euler2mat((0, np.pi, 0))
        robotWorld2hand = robotWorld2hand @ align_rotation
        return robotWorld2hand

    def step_and_render(self, step_num, next_pose):
        """
        Step the simulation with the next pose and return the rendered image.
        """
        curr_pos = self.env.sim.model.site_pos[self.goal_sid]
        curr_pos[:] = next_pose[:3]
        # update rot
        curr_quat = self.env.sim.model.site_quat[self.goal_sid]
        curr_quat[:] = next_pose[3:7]
        # update gripper
        gripper_state = next_pose[7]

        # get action using IK
        ik_result = qpos_from_site_pose(
            physics=self.env.sim,
            site_name=self.teleop_site,
            target_pos=curr_pos,
            target_quat=curr_quat,
            inplace=False,
            regularization_strength=1.0,
        )

        act = np.zeros(self.action_shape)
        if ik_result.success:
            act[:7] = ik_result.qpos[:7]
            act[7:] = gripper_state
            if self.env.normalize_act:
                act = self.env.env.robot.normalize_actions(act)
            _ = self.env.step(act)
        else:
            print(
                f"IK(t:{step_num}):: Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}"
            )

        return self.env.get_exteroception()["rgb:left_cam:240x424:2d"]

    def close(self):
        self.env.close()
