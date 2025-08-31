# import xlerobot_single # imports your robot and registers it
# # imports the demo_robot example script and lets you test your new robot
# import mani_skill.examples.demo_robot as demo_robot_script
# demo_robot_script.main()

import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
import copy



@register_agent()
class LeKiwi(BaseAgent):
    '''
    LeKiwi robot agent file for ManiSkill environment. Inspired by the the so100 file here: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/so100/so_100.py
    '''

    uid = "lekiwi"
    urdf_path = "/home/ubuntu/code/LeKiwi/urdf/LeKiwi.urdf"
    fix_root_link = False # Don't fix the root link

    # TODO: Add urdf config later for material properties

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            qvel=np.array(
                [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            pose=sapien.Pose(p=[0.0, 0.0, 0.05]),
        )
    )

    arm_joint_names = ['shoulder_pan',
                       'shoulder_lift',
                       'elbow_flex',
                       'wrist_flex',
                       'wrist_roll'
                       ]
    gripper_joint_names = ['gripper']

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint for joint in self.arm_joint_names],
            lower=None,
            upper=None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint for joint in self.arm_joint_names],
            lower=[-0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            upper=[0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
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

