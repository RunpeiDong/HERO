from __future__ import annotations

from dataclasses import field
from typing import Any, List, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""

    min_noise_std: float | None = None
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = None
    """Minimum mean noise standard deviation."""


@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization to actor and critic observations."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    # False exports a plain actor_obs policy; True also embeds the motion corpus.


    export_motion_in_onnx: bool = True
    """Whether ``PPO.export`` may bake the motion reference into the ONNX graph."""


@dataclass(frozen=True)
class PPODualModuleDictConfig:
    """Modules for HERO dual-actor, dual-critic PPO.

    Both actors read actor_obs and both critics read critic_obs. Actions concatenate
    lower-body and upper-body outputs in DOF order."""

    actor_lower: ModuleConfig
    """Lower-body actor (legs + waist, 15 dof on G1-29)."""

    actor_upper: ModuleConfig
    """Upper-body actor (arms, 14 dof on G1-29)."""

    critic_lower: ModuleConfig
    """Critic for the ``lower_body`` reward group."""

    critic_upper: ModuleConfig
    """Critic for the ``upper_body`` reward group."""


@dataclass(frozen=True)
class PPODualConfig(PPOConfig):
    """Configuration for HERO dual-actor, dual-critic PPO.

    init_noise_std controls the lower actor; init_noise_std_upper controls the upper actor."""

    module_dict: PPODualModuleDictConfig  # type: ignore[assignment]
    """Four modules: actor_lower, actor_upper, critic_lower, critic_upper."""

    init_noise_std_upper: float = 0.6
    """Initial noise std of the upper-body actor (``init_noise_std`` is used for the lower-body actor)."""

    action_split: tuple[int, int] = (15, 14)
    """(num lower-body actions, num upper-body actions); contiguous slices of the dof order, must sum to actions_dim."""

    export_motion_in_onnx: bool = False
    """PPODual never bakes the motion corpus into the ONNX graph (kept for API symmetry with PPOConfig)."""

    history_layout_export: str = "frame_major_hero_v1"
    """History layout of the exported ONNX inputs: ``frame_major_hero_v1`` (HERO sim2real contract,
    permutation applied inside the graph) or ``term_major_holosoma_v1`` (holosoma's native layout, identity)."""

    hero_std_clamp_max: float | None = None
    """Upper clamp of both actors' exploration std (``None`` = unclamped, the paper recipe). The std is projected
    under the bound after every update and when a checkpoint is loaded; ``scripts/train.py --std-clamp-max`` sets it."""


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    logging_interval: int = 100
    """the interval to log the metrics"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class FlashSACConfig:
    """FlashSAC (https://github.com/Holiday-Robot/FlashSAC, arXiv 2604.04539).

    Off-policy SAC with hyperspherical weight normalization, residual BatchNorm
    blocks, a categorical critic tied to a running reward normalizer, and
    zeta-distributed time-correlated exploration noise. Defaults follow the
    upstream IsaacLab recipe (scripts/run_isaaclab.sh), adapted to holosoma's
    per-env replay buffer.
    """

    num_learning_iterations: int = 160000
    """total env steps per env (one rollout step per iteration)"""

    buffer_size: int = 1024
    """replay buffer size per environment"""

    num_steps: int = 3
    """n-step return length (upstream IsaacLab uses n_step=3)"""

    gamma: float = 0.99
    """discount factor"""

    tau: float = 0.01
    """target critic EMA coefficient (upstream critic_target_update_tau)"""

    batch_size: int = 8192
    """global batch size sampled from the replay buffer per update"""

    learning_starts: int = 25
    """iterations before updates start (25 * 4096 envs ~= upstream 100K buffer_min)"""

    num_updates: int = 2
    """gradient updates per env step (upstream updates_per_interaction_step)"""

    actor_update_period: int = 2
    """actor/temperature update every N critic updates"""

    learning_rate_init: float = 3e-4
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 1.5e-4
    lr_warmup_steps: int = 0
    """warmup steps for the lr schedule (upstream IsaacLab: effectively 0)"""

    lr_decay_steps: int = 0
    """total lr schedule length in update steps; 0 = num_learning_iterations * num_updates"""

    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_arch: str = "mlp"
    """actor trunk: "mlp" (residual blocks), "transformer" (attention over obs tokens),
    or "resmlp" (ResMLP: linear token mixing + affine norm + LayerScale)"""
    actor_num_tokens: int = 4
    """token count for actor_arch="transformer"; cost grows steeply with this"""
    actor_num_heads: int = 4
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256

    num_bins: int = 101
    """number of categorical critic bins"""

    normalize_reward: bool = True
    """scale rewards by running return std; the critic support is +-g_max"""

    g_max: float = 5.0
    """normalized return bound; also the categorical critic support bound"""

    temp_initial_value: float = 0.0005
    """Initial entropy temperature, calibrated for actions scaled by the full joint range."""

    temp_target_sigma: float = 0.05
    """Target-entropy standard deviation as a fraction of the full physical joint range.

    The target is H* = 0.5 * d * log(2 * pi * e * sigma**2)."""

    head_init: str = "safe"
    """actor head init: "safe" = zero head gains -> initial action = default pose
    (required with holosoma full-range action scaling: upstream unit-norm heads
    emit near-limit actions at init and the humanoid dies in ~3 steps);
    "upstream" = exact upstream FlashSAC init."""

    init_std: float = 0.08
    """initial policy std under head_init="safe". 0.08 matches the FastSAC twin
    (log_std zero-init at bounds [-5,0] -> e^-2.5). 0.15 in full-range tanh
    space is enough physical noise to prevent survival from ever emerging."""

    zeta_mu: float = 2.0
    zeta_max: int = 16
    """truncated-zeta exploration noise repetition (mu, max repeat length)"""

    max_grad_norm: float = 0.0
    """gradient clipping (0 = off; upstream does not clip)"""

    compile: bool = True
    """whether to torch.compile the update functions"""

    amp: bool = True
    amp_dtype: str = "bf16"

    save_interval: int = 20000
    logging_interval: int = 100

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class PPODualAlgoConfig:
    """Algorithm wrapper for HERO dual-actor, dual-critic PPO."""

    _target_: str
    """Target algorithm class (dotted path, e.g. ``hero_isaacsim.agents.ppo_dual.ppo_dual.PPODual``)."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPODualConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FlashSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FlashSACConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = Union[PPOConfig, PPODualConfig, FastSACConfig, FlashSACConfig]

AlgoConfig = Union[PPOAlgoConfig, PPODualAlgoConfig, FastSACAlgoConfig, FlashSACAlgoConfig]
