import lekiwi

import mani_skill.examples.demo_robot as demo_robot_script
# demo_robot_script.main()

import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import URDFLoader
loader = URDFLoader()
loader.set_scene(ManiSkillScene())
robot = loader.load("/home/ubuntu/code/LeKiwi/urdf/LeKiwi.urdf")
print(robot.active_joints_map)