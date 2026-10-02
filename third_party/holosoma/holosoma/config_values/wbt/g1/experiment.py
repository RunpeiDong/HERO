from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=3.0)),  # Separate environments for readable rollout videos.
    ),
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=3.0)),  # Separate environments for readable rollout videos.
    ),
)

# --- Object-aware ACTOR variants (policy sees object/EE info) -------------------
# A: object 6DoF pose (robot frame) into the actor observation.
g1_29dof_wbt_fast_sac_obj_pose = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_actor,
)

# B: HERO-style end-effector 6DoF residual into the actor observation.
g1_29dof_wbt_fast_sac_ee_residual = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_ee_residual_actor,
)

# C: object-pose actor obs + general HOI contact reward (hands should hold the object).
g1_29dof_wbt_fast_sac_contact = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_actor,
    reward=reward.g1_29dof_wbt_reward_w_object_contact,
)

# D: object 6DoF pose AND HERO-style EE 6DoF residual into the actor observation (combined).
g1_29dof_wbt_fast_sac_obj_pose_ee = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_residual_actor,
)

# FlashSAC with the object-pose and end-effector-residual observation configuration.


g1_29dof_wbt_flash_sac_obj_pose_ee = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    training=replace(
        g1_29dof_wbt_fast_sac_obj_pose_ee.training,
        name="g1_29dof_wbt_flash_sac_obj_pose_ee",
    ),
    algo=algo.flash_sac,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config,
                       scene=replace(simulator.isaacsim.config.scene, env_spacing=3.0)),
    ),
)

# Add scaled base-frame hand contact forces to the actor observations (+6 dimensions).


g1_29dof_wbt_fast_sac_obj_pose_ee_contactforce = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_contactforce,
)

# Include current and future reference frames: actor dimension 181 + 58 * num_future.


g1_29dof_wbt_fast_sac_obj_pose_ee_future1 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_future1,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_future4 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_future4,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_future8 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_future8,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_future16 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_future16,
)

# Stack the 181-dimensional actor observation over history_length frames.


g1_29dof_wbt_fast_sac_obj_pose_ee_hist1 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_hist1,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_hist4 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_hist4,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_hist8 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_hist8,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_hist16 = replace(
    g1_29dof_wbt_fast_sac_w_object,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_hist16,
)

# Train a single policy over all clips in motion_dir.


g1_29dof_wbt_fast_sac_omni = replace(
    g1_29dof_wbt_fast_sac_w_object,
    command=command.g1_29dof_wbt_command_w_object_multitask,
)
g1_29dof_wbt_fast_sac_omni_obj_pose = replace(
    g1_29dof_wbt_fast_sac_obj_pose,
    command=command.g1_29dof_wbt_command_w_object_multitask,
)
g1_29dof_wbt_fast_sac_omni_ee_residual = replace(
    g1_29dof_wbt_fast_sac_ee_residual,
    command=command.g1_29dof_wbt_command_w_object_multitask,
)
g1_29dof_wbt_fast_sac_omni_obj_pose_ee = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    command=command.g1_29dof_wbt_command_w_object_multitask,
)


g1_29dof_wbt_fast_sac_omni_obj_pose_ee_omnifix = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    command=command.g1_29dof_wbt_command_w_object_multitask,
    reward=reward.g1_29dof_wbt_reward_w_object_omnifix,
)

# PPO configuration for multi-clip object tracking.


_ppo_algo = replace(
    algo.ppo,
    config=replace(
        algo.ppo.config,
        num_learning_iterations=30000,
        num_learning_epochs=5,
        save_interval=2000,
        entropy_coef=0.005,
        init_noise_std=1.0,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        init_at_random_ep_len=True,
        empirical_normalization=True,
        use_symmetry=False,
        actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
        critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
    ),
)
g1_29dof_wbt_ppo_omni_obj_pose_ee = replace(
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee,
    algo=_ppo_algo,
)
# Relax object termination thresholds to 1.0 m and 1.2 rad.


g1_29dof_wbt_fast_sac_omni_obj_pose_ee_relaxterm = replace(
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee,
    termination=termination.g1_29dof_wbt_termination_relaxed,
)
# object-BLIND actor obs (motion-only), closest to OmniRetarget's "policy blind to object" obs.
g1_29dof_wbt_ppo_omni = replace(
    g1_29dof_wbt_fast_sac_omni,
    algo=_ppo_algo,
)
# PPO object tracking from a single motion clip.


g1_29dof_wbt_ppo_obj_pose_ee = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,   # single-clip w_object command + obj_pose_ee obs
    algo=_ppo_algo,
)
# Single-clip PPO with motion-only actor observations.
g1_29dof_wbt_ppo_blind = replace(
    g1_29dof_wbt_fast_sac_w_object,      # single-clip w_object command + motion-only actor obs
    algo=_ppo_algo,
)

# Use the rubber-hand robot geometry.

g1_29dof_wbt_fast_sac_obj_pose_ee_rubber = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    robot=replace(robot.g1_29dof_w_object_rubberhand,
                  asset=replace(robot.g1_29dof_w_object_rubberhand.asset, enable_self_collisions=True),
                  object=replace(robot.g1_29dof_w_object_rubberhand.object,
                                 object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
                  init_state=replace(robot.g1_29dof_w_object_rubberhand.init_state, pos=[0.0, 0.0, 0.76])),
)
g1_29dof_wbt_fast_sac_w_object_rubber = replace(
    g1_29dof_wbt_fast_sac_w_object,
    robot=replace(robot.g1_29dof_w_object_rubberhand,
                  asset=replace(robot.g1_29dof_w_object_rubberhand.asset, enable_self_collisions=True),
                  object=replace(robot.g1_29dof_w_object_rubberhand.object,
                                 object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
                  init_state=replace(robot.g1_29dof_w_object_rubberhand.init_state, pos=[0.0, 0.0, 0.76])),
)

# Use collision meshes on the rubber-hand links.


g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    robot=replace(robot.g1_29dof_w_object_rubberhand_collision,
                  asset=replace(robot.g1_29dof_w_object_rubberhand_collision.asset, enable_self_collisions=True),
                  object=replace(robot.g1_29dof_w_object_rubberhand_collision.object,
                                 object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
                  init_state=replace(robot.g1_29dof_w_object_rubberhand_collision.init_state, pos=[0.0, 0.0, 0.76])),
)
g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol,
    command=command.g1_29dof_wbt_command_w_object_multitask,
)

# Use 16 pre-decomposed convex hand meshes from OmniContact_sim2sim.


g1_29dof_wbt_fast_sac_obj_pose_ee_16convex = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    robot=replace(robot.g1_29dof_w_object_rubberhand_16convex,
                  asset=replace(robot.g1_29dof_w_object_rubberhand_16convex.asset, enable_self_collisions=True),
                  object=replace(robot.g1_29dof_w_object_rubberhand_16convex.object,
                                 object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
                  init_state=replace(robot.g1_29dof_w_object_rubberhand_16convex.init_state, pos=[0.0, 0.0, 0.76])),
)
# Use the three-box paddle hand geometry from OmniContact_sim2sim.

g1_29dof_wbt_fast_sac_obj_pose_ee_paddle3box = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    robot=replace(robot.g1_29dof_w_object_paddle3box,
                  asset=replace(robot.g1_29dof_w_object_paddle3box.asset, enable_self_collisions=True),
                  object=replace(robot.g1_29dof_w_object_paddle3box.object,
                                 object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
                  init_state=replace(robot.g1_29dof_w_object_paddle3box.init_state, pos=[0.0, 0.0, 0.76])),
)

# Add privileged proprioception and projected gravity to multi-clip object tracking.


g1_29dof_wbt_fast_sac_omni_priv_proprio_projgrav_rubbercol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol,
    command=command.g1_29dof_wbt_command_w_object_multitask,
    observation=observation.g1_29dof_wbt_observation_priv_proprio_projgrav,
)
g1_29dof_wbt_fast_sac_omni_obj_pose_ee_priv_proprio_projgrav_rubbercol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol,
    command=command.g1_29dof_wbt_command_w_object_multitask,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_priv_proprio_projgrav,
)

# Use a convex box collision shape at each palm.


_boxcol_robot = replace(
    robot.g1_29dof_w_object_rubberhand_boxcol,
    asset=replace(robot.g1_29dof_w_object_rubberhand_boxcol.asset, enable_self_collisions=True),
    object=replace(robot.g1_29dof_w_object_rubberhand_boxcol.object,
                   object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
    init_state=replace(robot.g1_29dof_w_object_rubberhand_boxcol.init_state, pos=[0.0, 0.0, 0.76]),
)
# Separate environments by 3 m to keep rollout videos readable.


_boxcol_sim_spaced = replace(
    simulator.isaacsim,
    config=replace(simulator.isaacsim.config,
                   scene=replace(simulator.isaacsim.config.scene, env_spacing=3.0)),
)
g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee, robot=_boxcol_robot, simulator=_boxcol_sim_spaced,
)
g1_29dof_wbt_fast_sac_w_object_boxcol = replace(
    g1_29dof_wbt_fast_sac_w_object, robot=_boxcol_robot, simulator=_boxcol_sim_spaced,
)
# Actor observation variants with a fully privileged critic.

g1_29dof_wbt_fast_sac_head_residual_boxcol = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    observation=observation.g1_29dof_wbt_observation_head_residual_actor,
)
#   blind + EE-residual only (hands, no object pose)
g1_29dof_wbt_fast_sac_ee_only_boxcol = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    observation=observation.g1_29dof_wbt_observation_ee_only_actor,
)
#   object-pose + EE-residual + head_residual (full VR-3-point on top of object-aware)
g1_29dof_wbt_fast_sac_obj_pose_ee_head_boxcol = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_head_actor,
)

# Add projected gravity to actor observations to expose base roll and pitch.


g1_29dof_wbt_fast_sac_w_object_boxcol_projgrav = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    observation=observation.g1_29dof_wbt_observation_projgrav,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_projgrav = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_projgrav,
)

# Stack four past actor-observation frames.


g1_29dof_wbt_fast_sac_w_object_boxcol_hist4 = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    observation=observation.g1_29dof_wbt_observation_blind_hist4,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_hist4 = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_hist4,
)

# PPO variants with the same robot and observation configurations.


g1_29dof_wbt_ppo_obj_pose_ee_boxcol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol,
    algo=_ppo_algo,
)
g1_29dof_wbt_ppo_w_object_boxcol = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    algo=_ppo_algo,
)
# PPO variants with a fully privileged critic.
g1_29dof_wbt_ppo_head_residual_boxcol = replace(
    g1_29dof_wbt_fast_sac_head_residual_boxcol, algo=_ppo_algo,
)
g1_29dof_wbt_ppo_ee_only_boxcol = replace(
    g1_29dof_wbt_fast_sac_ee_only_boxcol, algo=_ppo_algo,
)
g1_29dof_wbt_ppo_obj_pose_ee_head_boxcol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_head_boxcol, algo=_ppo_algo,
)
# PPO actor observations with projected gravity.

g1_29dof_wbt_ppo_obj_pose_ee_boxcol_projgrav = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_projgrav,
    algo=_ppo_algo,
)

# Add motion-reference position, body pose, and base linear velocity to actor observations.


g1_29dof_wbt_fast_sac_priv_proprio_boxcol = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    observation=observation.g1_29dof_wbt_observation_priv_proprio,
)
g1_29dof_wbt_ppo_priv_proprio_boxcol = replace(
    g1_29dof_wbt_fast_sac_priv_proprio_boxcol,
    algo=_ppo_algo,
)
# Combine object pose and end-effector residuals with privileged proprioception.

g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_priv_proprio,
)
# PPO with the same observation configuration.
g1_29dof_wbt_ppo_obj_pose_ee_priv_proprio_boxcol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol,
    algo=_ppo_algo,
)

# Expand the object-mass randomization range to 2–12 kg.


import copy as _copy  # noqa: E402

_rand_mass_2_12 = _copy.deepcopy(randomization.g1_29dof_wbt_randomization_w_object)
_rand_mass_2_12.setup_terms["randomize_object_rigid_body_mass_startup"].params[
    "mass_distribution_params"
] = [2.0, 12.0]
# Object tracking with mass randomization in [2, 12] kg.
g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol_mass212 = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol,
    randomization=_rand_mass_2_12,
)

# Expose object mass to the actor and ramp the sampled mass range from 2–3 kg to 2–8 kg.


_rand_mass_2_3 = _copy.deepcopy(randomization.g1_29dof_wbt_randomization_w_object)
_rand_mass_2_3.setup_terms["randomize_object_rigid_body_mass_startup"].params[
    "mass_distribution_params"
] = [2.0, 3.0]
g1_29dof_wbt_fast_sac_boxcol_mass_curriculum = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_priv_proprio_physics,
    curriculum=curriculum.g1_29dof_wbt_curriculum_mass,
    randomization=_rand_mass_2_3,
)

# PPO variant with object-mass observations and curriculum.


g1_29dof_wbt_ppo_boxcol_mass_curriculum = replace(
    g1_29dof_wbt_fast_sac_boxcol_mass_curriculum,
    algo=_ppo_algo,
)

g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol_omnifix = replace(
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol,
    reward=reward.g1_29dof_wbt_reward_w_object_omnifix,
)

# Keep hand links as separate contact bodies so contact rewards use the hand surface.


_handbody_robot = replace(
    robot.g1_29dof_w_object_rubberhand_collision_handbody,
    asset=replace(robot.g1_29dof_w_object_rubberhand_collision_handbody.asset, enable_self_collisions=True),
    object=replace(robot.g1_29dof_w_object_rubberhand_collision_handbody.object,
                   object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
    init_state=replace(robot.g1_29dof_w_object_rubberhand_collision_handbody.init_state, pos=[0.0, 0.0, 0.76]),
)
# single-clip obj_pose_ee, handbody hand-contact reward
g1_29dof_wbt_fast_sac_obj_pose_ee_handbody = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    robot=_handbody_robot,
    reward=reward.g1_29dof_wbt_reward_w_object_handcontact,
)
# multitask obj_pose_ee, handbody + omnifix-handcontact (boosted obj-ori + hand-anchored contact)
g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_handbody,
    command=command.g1_29dof_wbt_command_w_object_multitask,
)
g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody_omnifix = replace(
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody,
    reward=reward.g1_29dof_wbt_reward_w_object_omnifix_handcontact,
)
# PPO with hand contact bodies and hand-anchored contact rewards.


g1_29dof_wbt_ppo_obj_pose_ee_handbody = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_handbody,
    algo=_ppo_algo,
)

# Actor observation variants with collision-mesh hands.


_rubbercol_robot = replace(
    robot.g1_29dof_w_object_rubberhand_collision,
    asset=replace(robot.g1_29dof_w_object_rubberhand_collision.asset, enable_self_collisions=True),
    object=replace(robot.g1_29dof_w_object_rubberhand_collision.object,
                   object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"),
    init_state=replace(robot.g1_29dof_w_object_rubberhand_collision.init_state, pos=[0.0, 0.0, 0.76]),
)
# blind (motion-only actor, critic-only object) with collision-mesh hand
g1_29dof_wbt_fast_sac_w_object_rubbercol = replace(
    g1_29dof_wbt_fast_sac_w_object, robot=_rubbercol_robot,
)
# Multi-clip PPO with a motion-only actor and collision-mesh hands.


g1_29dof_wbt_ppo_omni_rubbercol = replace(
    g1_29dof_wbt_fast_sac_w_object_rubbercol,
    command=command.g1_29dof_wbt_command_w_object_multitask,
    algo=_ppo_algo,
)
# Multi-clip PPO variants with a fully privileged critic.


g1_29dof_wbt_ppo_omni_obj_pose_ee_rubbercol = replace(
    g1_29dof_wbt_ppo_omni_rubbercol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_residual_actor,
)
#   priv_proprio: BLIND actor + the 4 non-object privileged proprio groups (no object in actor)
g1_29dof_wbt_ppo_omni_priv_proprio_rubbercol = replace(
    g1_29dof_wbt_ppo_omni_rubbercol,
    observation=observation.g1_29dof_wbt_observation_priv_proprio,
)
#   objposeee + priv_proprio: actor sees object/EE AND the privileged proprio bundle
g1_29dof_wbt_ppo_omni_obj_pose_ee_priv_proprio_rubbercol = replace(
    g1_29dof_wbt_ppo_omni_rubbercol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_priv_proprio,
)
# A: obj_pose actor obs, collision-mesh hand
g1_29dof_wbt_fast_sac_obj_pose_rubbercol = replace(
    g1_29dof_wbt_fast_sac_obj_pose, robot=_rubbercol_robot,
)
# B: ee_residual actor obs, collision-mesh hand
g1_29dof_wbt_fast_sac_ee_residual_rubbercol = replace(
    g1_29dof_wbt_fast_sac_ee_residual, robot=_rubbercol_robot,
)
# C: obj_pose + HOI contact reward, collision-mesh hand
g1_29dof_wbt_fast_sac_contact_rubbercol = replace(
    g1_29dof_wbt_fast_sac_contact, robot=_rubbercol_robot,
)

# Evaluate object rewards and termination in the robot-relative reference frame.


g1_29dof_wbt_fast_sac_w_object_rubbercol_retgt = replace(
    g1_29dof_wbt_fast_sac_w_object_rubbercol,
    reward=reward.g1_29dof_wbt_reward_w_object_retgt,
    termination=termination.g1_29dof_wbt_termination_retgtobj,
)
# Object-pose and end-effector-residual observations with a robot-relative object reference.
g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol_retgt = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol,
    reward=reward.g1_29dof_wbt_reward_w_object_retgt,
    termination=termination.g1_29dof_wbt_termination_retgtobj,
)
# Use half-sphere hands with robot-relative object rewards and termination.


g1_29dof_wbt_fast_sac_w_object_halfsphere_retgt = replace(
    g1_29dof_wbt_fast_sac_w_object,
    robot=replace(
        robot.g1_29dof_w_object_halfsphere,
        asset=replace(
            robot.g1_29dof_w_object_halfsphere.asset,
            enable_self_collisions=True,
        ),
        object=g1_29dof_wbt_fast_sac_w_object.robot.object,
        init_state=g1_29dof_wbt_fast_sac_w_object.robot.init_state,
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_retgt,
    termination=termination.g1_29dof_wbt_termination_retgtobj,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_halfsphere_retgt = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    robot=g1_29dof_wbt_fast_sac_w_object_halfsphere_retgt.robot,
    reward=reward.g1_29dof_wbt_reward_w_object_retgt,
    termination=termination.g1_29dof_wbt_termination_retgtobj,
)

# Multi-clip tracking with rubber hands; configure transitions and phase sampling through the CLI.


g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubber,
    command=command.g1_29dof_wbt_command_w_object_multitask,
)
# Combine rubber hands with object-orientation and hand-contact rewards.

g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber_omnifix = replace(
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber,
    reward=reward.g1_29dof_wbt_reward_w_object_omnifix,
)


# Hold an eight-frame reference chunk between replanning steps.


g1_29dof_wbt_fast_sac_chunk8_s0_boxcol = replace(
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    observation=observation.g1_29dof_wbt_observation_chunk8_s0,
)
g1_29dof_wbt_fast_sac_obj_pose_ee_chunk8_s0_boxcol = replace(
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol,
    observation=observation.g1_29dof_wbt_observation_obj_pose_ee_chunk8_s0,
)

__all__ = [
    "g1_29dof_wbt_fast_sac_chunk8_s0_boxcol",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_chunk8_s0_boxcol",
    "g1_29dof_wbt",
    "g1_29dof_wbt_ppo_omni_obj_pose_ee",
    "g1_29dof_wbt_ppo_omni",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_fast_sac_obj_pose",
    "g1_29dof_wbt_fast_sac_ee_residual",
    "g1_29dof_wbt_fast_sac_contact",
    "g1_29dof_wbt_fast_sac_obj_pose_ee",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future1",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future4",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future8",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future16",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist1",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist4",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist8",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist16",
    "g1_29dof_wbt_fast_sac_omni",
    "g1_29dof_wbt_fast_sac_omni_obj_pose",
    "g1_29dof_wbt_fast_sac_omni_ee_residual",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_omnifix",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_relaxterm",
    "g1_29dof_wbt_ppo_obj_pose_ee",
    "g1_29dof_wbt_ppo_blind",
    "g1_29dof_wbt_ppo_obj_pose_ee_boxcol",
    "g1_29dof_wbt_ppo_w_object_boxcol",
    "g1_29dof_wbt_ppo_obj_pose_ee_boxcol_projgrav",
    "g1_29dof_wbt_ppo_obj_pose_ee_handbody",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_rubber",
    "g1_29dof_wbt_fast_sac_w_object_rubber",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber_omnifix",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_16convex",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_paddle3box",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol",
    "g1_29dof_wbt_fast_sac_w_object_boxcol",
    "g1_29dof_wbt_fast_sac_w_object_boxcol_projgrav",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_projgrav",
    "g1_29dof_wbt_fast_sac_w_object_boxcol_hist4",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_hist4",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol_omnifix",
    "g1_29dof_wbt_fast_sac_w_object_rubbercol",
    "g1_29dof_wbt_fast_sac_obj_pose_rubbercol",
    "g1_29dof_wbt_fast_sac_ee_residual_rubbercol",
    "g1_29dof_wbt_fast_sac_contact_rubbercol",
    "g1_29dof_wbt_fast_sac_w_object_rubbercol_retgt",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol_retgt",
    "g1_29dof_wbt_fast_sac_w_object_halfsphere_retgt",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_halfsphere_retgt",
    "g1_29dof_wbt_fast_sac_obj_pose_ee_handbody",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody",
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody_omnifix",
    "g1_29dof_wbt_w_object",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""
