# import xlerobot_single # imports your robot and registers it
# # imports the demo_robot example script and lets you test your new robot
# import mani_skill.examples.demo_robot as demo_robot_script
# demo_robot_script.main()

from copy import deepcopy
import sapien
import sapien.physx as physx
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


@register_agent()
class Lekiwi(BaseAgent):
    uid = "lekiwi"
    urdf_path = "/home/ubuntu/code/LeKiwi/urdf/LeKiwi.urdf"

    # urdf_config = dict(
    #     _materials=dict(
    #         gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
    #     ),
    #     link=dict(
    #         Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
    #         Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
    #     ),
    # )

    # keyframes = dict(
    #     rest=Keyframe(
    #         qpos=np.array([0, -1.5708, 1.5708, 0.66, 0, -1.1]),
    #         pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
    #     ),
    #     zero=Keyframe(
    #         qpos=np.array([0.0] * 6),
    #         pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
    #     ),
    # )
    def __init__(self, *args, **kwargs):

        self.arm_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]

        self.arm_stiffness = 2e4
        self.arm_damping = 1e2
        self.arm_force_limit = 250

        self.gripper_joint_names = [
            "gripper",
        ]
        self.gripper_stiffness = 50
        self.gripper_damping = 1e2
        self.gripper_force_limit = 2.8

        self.ee_link_name = "Fixed_Jaw"

        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ]
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # For SO100 gripper, we use a regular PDJointPosController instead of mimic controller
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            -20,  # closed position - update this value if needed
            20,  # open position - update this value if needed
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )

        # Create a single-arm controller config
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                base=base_pd_joint_vel,
                arm1=arm_pd_joint_delta_pos,
                gripper1=gripper_pd_joint_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["Fixed_Jaw"]
        self.finger2_link = self.robot.links_map["Moving_Jaw"]
        self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
        self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

    @property
    def tcp_pos(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["Fixed_Jaw"]
        self.finger2_link = self.robot.links_map["Moving_Jaw"]
        self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
        self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[:, :-1]  # exclude the gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)