#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        resize_shape: (H, W) shape to resize images to as a preprocessing step for the vision
            backbone. If None, no resizing is done and the original image resolution is used.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape
            (crop_h = int(resize_shape[0] * crop_ratio), likewise for width).
            Set to 1.0 to disable cropping. Only takes effect when resize_shape is not None.
        crop_shape: (H, W) shape to crop images to. When resize_shape is set and crop_ratio < 1.0,
            this is computed automatically. Can also be set directly for legacy configs that use
            crop-only (without resize). If None and no derivation applies, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center
            crop in eval mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # --- Virtual views (config-driven crops of the source cameras) ---
    # When set, the conditioning views are built from these specs instead of
    # the raw image_features: {view_name: {source: <image feature key>,
    # crop: [y0, x0, y1, x1] | null}}. Crops are pixel boxes on the SOURCE
    # frame; every view is then resized to resize_shape before the shared
    # encoder, so mixed crop sizes are fine. Runs identically at train and
    # deploy (serialized with the checkpoint) — no dataset rebuild needed.
    virtual_views: dict[str, dict] | None = None

    # --- Relative actions (OpenPI DeltaActions pattern) ---
    # Train on action - current_state offsets instead of absolute targets.
    # Requires state and action to share the same layout/names (e.g. an
    # ee6d dataset where observation.state == action space). The conversion
    # runs AFTER normalization (and inverts BEFORE unnormalization), so the
    # offsets are in normalized units and the algebra is exact. Deployment:
    # RTC engine is relative-aware; the sync engine rejects relative policies.
    use_relative_actions: bool = False
    # Names to keep absolute (token match against action_feature_names).
    relative_exclude_joints: list[str] = field(default_factory=list)
    # Per-dim action names (from the dataset card) used to build the mask.
    action_feature_names: list[str] | None = None

    # --- UMI-faithful SE(3) relative poses (arXiv 2402.10329, PD2) ---
    # Re-express observation.state AND action windows in the frame of the
    # LAST observation step (the anchor): after the transform the anchor
    # state is exactly identity ([0,0,0, 1,0,0,0,1,0]), earlier obs steps
    # carry velocity information, and actions are SE(3) transforms w.r.t.
    # the anchor (UMI's "relative trajectory"). Makes the policy invariant
    # to the world/tracking frame — no train/rollout frame calibration.
    # Requires: observation.state and action are both 9D [xyz + rot6d cols].
    # The transform runs INSIDE the model on raw (unnormalized) poses — the
    # processor pipeline skips normalization for these two keys, and
    # positions are scaled by 1/se3_pos_scale in place of dataset stats
    # (rot6d components are already in [-1, 1]). Mutually exclusive with
    # use_relative_actions.
    use_se3_relative: bool = False
    # Meters mapped to 1.0 normalized unit for relative positions. Should
    # comfortably cover the largest within-horizon displacement in the data.
    # Ignored when `use_se3_normalize` is on.
    se3_pos_scale: float = 0.2
    # Original-repo-faithful alternative to se3_pos_scale: per-dim MIN_MAX
    # normalization of the RELATIVIZED state/action to [-1, 1], with stats
    # fitted on the relativized dataset (compute_se3_rel_stats.py). Bounds
    # the diffusion space exactly like original diffusion_policy/UMI, so
    # x0 clipping (`clip_sample`) is valid again in SE(3) mode.
    use_se3_normalize: bool = False
    # UMI-FAITHFUL normalization (2026-08-18). When True, the ONLY scaling
    # applied to poses is a MIN_MAX on the RELATIVIZED POSITION (dims 0:3);
    # rot6d is passed through untouched everywhere, and the ABSOLUTE action is
    # not normalized at all. Matches the original repo, which range-normalizes
    # position + gripper but uses an IDENTITY normalizer for rotation
    # (diffusion_policy/dataset/umi_dataset.py get_normalizer).
    #
    # WHY THIS EXISTS: with it False (the pre-2026-08-18 behaviour) the ABSOLUTE
    # action is MIN_MAX-normalized by the processor before the model relativizes
    # it. Per-dim affine scaling destroys rot6d — components reach ~8 where a
    # rotation column caps at 1 — and the Gram-Schmidt inside pose9_to_mat then
    # silently projects that back onto SO(3), which is NOT invertible. Measured
    # end-to-end round-trip error of the action targets: position 0.0000 mm but
    # rotation 2.89 deg median / 8.76 deg max. Observation.state was always
    # exempt, so the two were relativized in mismatched spaces.
    #
    # Kept default False so every checkpoint trained before this date keeps
    # loading and evaluating exactly as it was trained.
    se3_normalize_pos_only: bool = False
    # Real-Time Chunking prefix guidance at INFERENCE (Black et al., ported to
    # the epsilon/DDIM parametrization; see RTCProcessor.guide_x0). When True,
    # predict_action_chunk consumes the engine-provided prev_chunk_left_over:
    # each DDIM step nudges the clean-chunk estimate toward the previous
    # chunk's leftover, weighted by the prefix schedule (frozen executing rows
    # -> 1, ramp across execution_horizon, 0 beyond). Consecutive chunks then
    # agree by construction and splice_mode "none" is the natural pairing.
    # Requires use_se3_relative (the prefix is re-relativized to the current
    # anchor). Costs no extra denoiser calls (~elementwise ops per step).
    # Read per replan -> live-togglable. Knobs (execution_horizon,
    # max_guidance_weight, prefix_attention_schedule) come from rtc_config,
    # which the rollout stack aliases onto this config at deploy.
    rtc_guidance: bool = False
    # True (default): the correction is the reference implementation's true
    # vector-Jacobian product back through the denoiser (Jacobian-filtered,
    # stays on the learned action manifold; ~2x per-step cost while guided).
    # False: the identity-VJP shortcut (raw weighted x0 error, s clamped to
    # 1) — cheap, but carved off-manifold seam cliffs live on 2026-08-22.
    rtc_guidance_vjp: bool = True
    # --- train-time low-pass on the ACTION TARGETS (zero-phase FIR) ---
    # Teaches the policy to emit smooth trajectories natively, instead of
    # filtering at deploy (which would add phase lag). Applied in WORLD space
    # before SE(3) relativization. The action window is fetched with
    # `action_lpf_pad` extra rows on each side and cropped after filtering,
    # so the result is identical to filtering the whole episode offline — no
    # window-edge artifacts. Observation state is deliberately NOT filtered:
    # at deploy it comes from raw FK, so leaving it raw keeps train/deploy
    # consistent. 0 = off.
    action_lpf_hz: float = 0.0
    action_lpf_pad: int = 12
    # Sample rate of the action sequence (= dataset fps). Must be set
    # explicitly when action_lpf_hz > 0; the policy config has no fps of
    # its own and guessing it would silently mis-tune the cutoff.
    action_lpf_fps: float = 0.0
    # {"state_min": [9], "state_max": [9], "action_min": [9],
    #  "action_max": [9]} — raw relativized units (meters / rot6d).
    se3_rel_stats: dict | None = None

    # --- EMA of model weights (original diffusion_policy trains with EMA
    # and evaluates the EMA copy; diffusers EMAModel power schedule) ---
    use_ema: bool = False
    ema_power: float = 0.75
    ema_inv_gamma: float = 1.0
    ema_max_decay: float = 0.9999

    # --- Current-as-touch additions (arXiv 2607.03529) ---
    # Extra STATE-type input features concatenated to observation.state in the
    # global conditioning (e.g. ["observation.current"]). They receive the same
    # n_obs_steps window and are normalized like any input feature.
    extra_state_keys: list[str] = field(default_factory=list)
    # Auxiliary smoothed-current regularization: weight > 0 adds a small MLP
    # head predicting `current_aux_key` (an input feature, so normalized by the
    # preprocessor, but NEVER fed to the conditioning) from the global cond;
    # total loss += weight * MSE. Inference is unaffected (head unused).
    current_aux_weight: float = 0.0
    current_aux_key: str = "observation.current_smooth"

    # --- Progress head (per-subtask completion signal) ---
    # weight > 0 adds a small MLP head on the global conditioning predicting
    # `progress_aux_key`: steps-till-completion normalized to [-1, 0]
    # (a per-frame dataset column; -1 = episode start, 0 = final frame).
    # Same contract as current_aux: the key must be a declared input feature
    # (so the preprocessor normalizes it) but is NEVER fed to the
    # conditioning. At inference generate_actions stashes the prediction on
    # the policy as `_last_progress_norm`; the RTC engine reads it per
    # replan for stage-transition detection.
    progress_aux_weight: float = 0.0
    progress_aux_key: str = "observation.progress"

    # --- Pixel head (2D end-effector localization in a camera view) ---
    # weight > 0 adds a small MLP head on the global conditioning predicting
    # `pixel_aux_key` (e.g. the vive tracker's normalized (u,v) in the head
    # camera; one 2-vector per obs step). AMODAL: trained on every frame,
    # including when the point is occluded/masked/out of frame; frames whose
    # `pixel_aux_valid_key` is <= 0 (post-normalization) are dropped from the
    # loss (far-out-of-frame). Same contract as progress_aux: both keys must
    # be declared input features but are NEVER fed to the conditioning.
    pixel_aux_weight: float = 0.0
    pixel_aux_key: str = "observation.puck_uv"
    pixel_aux_valid_key: str = "observation.puck_valid"

    # --- Train-time image augmentation (applied ONLY under self.training,
    # in NORMALIZED image units, after view stacking; never at deploy) ---
    aug_glare_p: float = 0.0          # per-sample probability of glare blobs
    aug_glare_max_blobs: int = 2
    aug_glare_amp: float = 0.8        # max blob amplitude (normalized units)
    aug_glare_cam_index: int = -1     # index into image_features order; -1 = all
    aug_glare_rect: list[float] | None = None  # [x0,y0,x1,y1] normalized; keeps
                                               # glare off baked-masked regions
    aug_noise_p: float = 0.0          # per-sample probability of pixel noise
    aug_noise_std: float = 0.10       # gaussian std (normalized units)
    # --- train-only JITTERED BLACK MASK (per Andrew 2026-08-14): the tray
    # blackout is applied at TRAIN TIME on unmasked images, with the rect
    # randomly shifted so the mask edge cannot become a spatial landmark.
    # Deploy applies the STATIC base rect (adapter-side). Rect + jitter in
    # normalized [0,1] image coords; jitter fractions are of W/H.
    aug_mask_p: float = 0.0            # 0 = off (dataset presumed pre-masked)
    aug_mask_rect: tuple[float, float, float, float] | None = None
    aug_mask_cam_index: int | None = None   # which view gets the mask
    aug_mask_jitter: tuple[float, float] = (0.10, 0.10)   # +-frac of W, H
    aug_mask_edge_jitter: float = 0.016
    # Fill value for the masked region. The aug runs AFTER mean/std
    # normalization, so writing 0 means the ImageNet MEAN colour (mid-grey),
    # NOT black — a ~2-sigma mismatch against a deploy path that blacks raw
    # pixels (measured 2026-08-17). "black" writes (0 - mean)/std per channel
    # so the model sees true black, matching a raw-black deploy mask.
    aug_mask_fill: str = "zero"          # "zero" (legacy grey) | "black"     # extra +-frac per edge

    # Inference: reuse ONE sampling latent for the whole rollout (set on
    # reset(), reused every predict_action_chunk). Consecutive replans become
    # correlated draws instead of independent samples -> far less chunk-to-
    # chunk disagreement at RTC splices. No effect on training.
    frozen_inference_noise: bool = False

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # Denoiser selection: "unet" (default, DiffusionConditionalUnet1d) or
    # "dit" (DiffusionTransformerDenoiser — adaLN-zero transformer, same
    # (x, timestep, global_cond) interface).
    denoiser: str = "unet"
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # DiT (only used when denoiser == "dit"). Defaults sized to roughly
    # match the (512, 1024, 2048) UNet's denoiser parameter count.
    dit_dim: int = 1024
    dit_depth: int = 14
    dit_heads: int = 16
    dit_ffn_mult: int = 4
    dit_dropout: float = 0.0
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.use_se3_relative and self.use_relative_actions:
            raise ValueError(
                "`use_se3_relative` and `use_relative_actions` are mutually "
                "exclusive — the SE(3) transform subsumes the delta-actions one."
            )
        if self.use_se3_relative and self.se3_pos_scale <= 0:
            raise ValueError(f"`se3_pos_scale` must be > 0. Got {self.se3_pos_scale}.")
        if self.action_lpf_hz > 0:
            if self.action_lpf_fps <= 0:
                raise ValueError("`action_lpf_hz` requires `action_lpf_fps` (dataset fps).")
            if self.action_lpf_hz >= self.action_lpf_fps / 2:
                raise ValueError(
                    f"`action_lpf_hz` ({self.action_lpf_hz}) must be below Nyquist "
                    f"({self.action_lpf_fps / 2})."
                )
        if self.use_se3_normalize:
            if not self.use_se3_relative:
                raise ValueError("`use_se3_normalize` requires `use_se3_relative`.")
            required = {"state_min", "state_max", "action_min", "action_max"}
            if not isinstance(self.se3_rel_stats, dict) or not required <= set(self.se3_rel_stats):
                raise ValueError(
                    f"`use_se3_normalize` needs `se3_rel_stats` with keys {sorted(required)} "
                    "(fit them with compute_se3_rel_stats.py)."
                )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.use_se3_relative:
            state_dim = self.robot_state_feature.shape[0] if self.robot_state_feature else None
            action_dim = self.action_feature.shape[0] if self.action_feature else None
            if state_dim != 9 or action_dim != 9:
                raise ValueError(
                    "`use_se3_relative` requires 9D [xyz + rot6d] observation.state "
                    f"and action. Got state={state_dim}, action={action_dim}."
                )

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        lo = 1 - self.n_obs_steps
        hi = lo + self.horizon
        if self.action_lpf_hz > 0:
            # padding for the filter, PLUS (n_obs_steps - 1) extra rows on the
            # left so the observation-state timesteps are inside this window
            # too: dataset action[j] == state[j+1], so state at delta d is
            # action at delta d-1. Filtering both from one fetch avoids
            # over-fetching observation keys (which would decode extra video).
            lo -= self.action_lpf_pad + (self.n_obs_steps - 1)
            hi += self.action_lpf_pad
        return list(range(lo, hi))

    @property
    def reward_delta_indices(self) -> None:
        return None
