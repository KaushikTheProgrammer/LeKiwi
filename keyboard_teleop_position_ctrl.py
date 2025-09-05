import gymnasium as gym
import numpy as np
import sapien
import pygame
import time

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import lekiwi

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode. Use 'pd_joint_pos' for position control."""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def get_mapped_joints(robot):
    """
    Get the current joint positions from the robot and map them correctly to the target joints.

    The mapping is:
    - full_joints[0,2] → current_joints[0,1] (base x position and base rotation)
    - full_joints[3,6,9,11,13] → current_joints[2,3,4,5,6] (first arm joints)
    - full_joints[4,7,10,12,14] → current_joints[7,8,9,10,11] (second arm joints)

    Returns:
        np.ndarray: Mapped joint positions with shape matching the target_joints
    """
    if robot is None:
        return np.zeros(8)  # Default size for action

    # Get full joint positions
    full_joints = robot.get_qpos()

    # Convert tensor to numpy array if needed
    if hasattr(full_joints, 'numpy'):
        full_joints = full_joints.numpy()

    # Handle case where it's a 2D tensor/array
    if full_joints.ndim > 1:
        full_joints = full_joints.squeeze()

    # Create the mapped joints array with correct size
    mapped_joints = np.zeros(9)

    # Map the joints according to the specified mapping
    if len(full_joints) >= 8:
        mapped_joints = full_joints.copy()

    return mapped_joints

def main(args: Args):
    pygame.init()

    screen_width, screen_height = 600, 750
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Control Window - Use keys to move")
    font = pygame.font.SysFont(None, 24)

    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    action = env.action_space.sample() if env.action_space is not None else None
    action = np.zeros_like(action)


    # Initialize target joint positions with zeros
    target_joints = np.zeros_like(action)

    target_base_x = 0.0
    current_base_x = 0.0
    base_x_error = 0.0
    base_x_kp = 2.0      # Proportional gain for base x control
    base_x_max_vel = 1.0 # Maximum base x velocity
    base_x_velocity = 0.0

    # Yaw position control variables
    target_yaw = 0.0  # Limits are -2pi to 2pi
    current_yaw = 0.0
    yaw_error = 0.0
    yaw_kp = 2.0      # Proportional gain for yaw control
    yaw_max_vel = 1.0 # Maximum yaw velocity

    target_shoulder_lift = 0.0 # Limits are 0.1 (inwards to robot) to -pi (outwards from robot)
    current_shoulder_lift = 0.0
    shoulder_lift_error = 0.0
    shoulder_lift_kp = 1.0      # Proportional gain for shoulder lift control
    shoulder_lift_max_vel = 1.0 # Maximum shoulder lift velocity
    shoulder_lift_velocity = 0.0

    target_elbow_flex = 0.0 # Limits are pi(fully extended) and 0.03(fully retracted)
    current_elbow_flex = 0.0
    elbow_flex_error = 0.0
    elbow_flex_kp = 1.0      # Proportional gain for elbow flex control
    elbow_flex_max_vel = 1.0 # Maximum elbow flex velocity
    elbow_flex_velocity = 0.0

    target_wrist_flex = 0.0 # Limits are -0.3 to pi
    current_wrist_flex = 0.0
    wrist_flex_error = 0.0
    wrist_flex_kp = 1.0      # Proportional gain for wrist flex control
    wrist_flex_max_vel = 1.0 # Maximum wrist flex velocity
    wrist_flex_velocity = 0.0

    target_wrist_roll = 0.0 # Limits are -pi/2 to pi/2
    current_wrist_roll = 0.0
    wrist_roll_error = 0.0
    wrist_roll_kp = 1.0      # Proportional gain for wrist roll control
    wrist_roll_max_vel = 1.0 # Maximum wrist roll velocity
    wrist_roll_velocity = 0.0

    target_gripper = -1.5 # Gripper range is -pi/2 (open) to 0.1 (closed)
    current_gripper = 0.0
    gripper_error = 0.0
    gripper_kp = 1.0      # Proportional gain for gripper control
    gripper_max_vel = 1.0 # Maximum gripper velocity
    gripper_velocity = 0.0

    # Get initial joint positions if available
    current_joints = np.zeros_like(action)
    robot = None

    # Try to get the robot instance for direct access
    if hasattr(env.unwrapped, "agent"):
        robot = env.unwrapped.agent.robot
    elif hasattr(env.unwrapped, "agents") and len(env.unwrapped.agents) > 0:
        robot = env.unwrapped.agents[0]  # Get the first robot if multiple exist


    # Get the correctly mapped joints
    current_joints = get_mapped_joints(robot)

    # Ensure target_joints is a numpy array with the same shape as current_joints
    target_joints = np.zeros_like(current_joints)

    # Add step counter for warmup phase
    step_counter = 0
    warmup_steps = 50

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                return

        keys = pygame.key.get_pressed()

        # Update target joint positions based on key presses - only after warmup
        if step_counter >= warmup_steps:
            # Base forward/backward - direct control
            if keys[pygame.K_w]:
                target_base_x = 0.1 # Forward
            elif keys[pygame.K_s]:
                target_base_x = -0.1  # Backward
            else:
                target_base_x = 0.0

            # Yaw position control - increment/decrement target yaw position
            if keys[pygame.K_a]:
                target_yaw += 0.1  # Increment target yaw (turn left)
            elif keys[pygame.K_d]:
                target_yaw -= 0.1  # Decrement target yaw (turn right)

            if keys[pygame.K_i]:
                target_shoulder_lift += 0.1
            if keys[pygame.K_o]:
                target_shoulder_lift -= 0.1

            if keys[pygame.K_k]:
                target_elbow_flex += 0.1
            if keys[pygame.K_l]:
                target_elbow_flex -= 0.1

            if keys[pygame.K_n]:
                target_wrist_flex += 0.1
            if keys[pygame.K_m]:
                target_wrist_flex -= 0.1

            if keys[pygame.K_COMMA]:
                target_wrist_roll += 0.1
            if keys[pygame.K_PERIOD]:
                target_wrist_roll -= 0.1

            if keys[pygame.K_LEFT]:
                target_gripper += 0.1
            if keys[pygame.K_RIGHT]:
                target_gripper -= 0.1



            current_yaw = current_joints[2]
            current_shoulder_lift = current_joints[3]
            current_elbow_flex = current_joints[4]
            current_wrist_flex = current_joints[5]
            current_wrist_roll = current_joints[6]
            current_gripper = current_joints[7]

            base_x_error = target_base_x - current_base_x
            yaw_error = target_yaw - current_yaw
            shoulder_lift_error = target_shoulder_lift - current_shoulder_lift
            elbow_flex_error = target_elbow_flex - current_elbow_flex
            wrist_flex_error = target_wrist_flex - current_wrist_flex
            wrist_roll_error = target_wrist_roll - current_wrist_roll
            gripper_error = target_gripper - current_gripper

            base_x_velocity = base_x_kp * base_x_error
            yaw_velocity = yaw_kp * yaw_error
            shoulder_lift_input = shoulder_lift_kp * shoulder_lift_error
            elbow_flex_input = elbow_flex_kp * elbow_flex_error
            wrist_flex_input = wrist_flex_kp * wrist_flex_error
            wrist_roll_input = wrist_roll_kp * wrist_roll_error
            gripper_input = gripper_kp * gripper_error

        current_joints = get_mapped_joints(robot)


        # Simple P controller for arm joints only (not base)
        if step_counter < warmup_steps:
            action = np.zeros_like(action)
        else:
            action[0] = base_x_velocity
            action[1] = yaw_velocity
            action[2] = shoulder_lift_input
            action[3] = elbow_flex_input
            action[4] = wrist_flex_input
            action[5] = wrist_roll_input
            action[6] = gripper_input


        screen.fill((0, 0, 0))

        text = font.render("Controls:", True, (255, 255, 255))
        screen.blit(text, (10, 10))

        # Add warmup status to display
        if step_counter < warmup_steps:
            warmup_text = font.render(f"WARMUP: {step_counter}/{warmup_steps} steps", True, (255, 0, 0))
            screen.blit(warmup_text, (300, 10))

        control_texts = [
            "W/S: Forward/Backward (joint[0])",
            "A/D: Yaw Position Control (joint[2])",
            "Y/U: Arm joint[3] (+/-)",
            "8/I: Arm joint[4] (+/-)",
            "9/O: Arm joint[5] (+/-)",
            "0/P: Arm joint[6] (+/-)",
            "-/[: Arm joint[7] (+/-)",
            "R: Reset targets to current"
        ]

        col_height = len(control_texts) // 2 + len(control_texts) % 2
        for i, txt in enumerate(control_texts):
            col = 0 if i < col_height else 1
            row = i if i < col_height else i - col_height
            ctrl_text = font.render(txt, True, (255, 255, 255))
            screen.blit(ctrl_text, (10 + col * 200, 40 + row * 25))

        # Display full joints (before mapping)
        y_pos = 40 + col_height * 30 + 10

        # Get full joint positions
        full_joints = robot.get_qpos() if robot is not None else np.zeros(8)

        # Convert tensor to numpy array if needed
        if hasattr(full_joints, 'numpy'):
            full_joints = full_joints.numpy()

        # Handle case where it's a 2D tensor/array
        if full_joints.ndim > 1:
            full_joints = full_joints.squeeze()

        # Display full joints in two rows
        full_joints_text1 = font.render(
            f"Full Joints (1-8): {np.round(full_joints[:8], 2)}",
            True, (255, 150, 0)
        )
        screen.blit(full_joints_text1, (10, y_pos))
        y_pos += 25


        # Display current joint positions in three logical groups
        # Group 1: Base control [0,1]
        base_joints = current_joints[0:2]
        base_text = font.render(
            f"Base [0,1]: {np.round(base_joints, 2)}",
            True, (255, 255, 0)
        )
        screen.blit(base_text, (10, y_pos))

        # Group 2: First arm [2,3,4,5,6]
        y_pos += 25
        arm1_joints = current_joints[2:7]
        arm1_text = font.render(
            f"Arm 1 [2,3,4,5,6]: {np.round(arm1_joints, 2)}",
            True, (255, 255, 0)
        )
        screen.blit(arm1_text, (10, y_pos))


        # Group 1: Base control [0,2] (forward/backward and yaw)
        y_pos += 25
        base_targets = np.array([target_joints[0], target_yaw])  # Forward/backward and yaw target
        base_target_text = font.render(
            f"Base Target [0,yaw]: {np.round(base_targets, 2)}",
            True, (0, 255, 0)
        )
        screen.blit(base_target_text, (10, y_pos))

        # Group 2: First arm [3,4,5,6,7]
        y_pos += 25
        arm1_targets = target_joints[2:8]
        arm1_target_text = font.render(
            f"Arm 1 Target [3,4,5,6,7]: {np.round(arm1_targets, 2)}",
            True, (0, 255, 0)
        )
        screen.blit(arm1_target_text, (10, y_pos))

        pygame.display.flip()

        # action = np.zeros_like(action)
        obs, reward, terminated, truncated, info = env.step(action)
        step_counter += 1

        if args.render_mode is not None:
            env.render()

        time.sleep(0.01)

        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break

    pygame.quit()
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
