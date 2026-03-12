from __future__ import annotations

import dataclasses
import functools
import logging
import os
import pathlib
import platform

from flax.training import common_utils
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.pi0_fast as pi0_fast
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as checkpoints
import openpi.training.config as training_config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as weight_loaders
from clanker_constants import VAL_EPISODES
from clanker_dataset import ClankerDatasetConfig
from clanker_dataset import ModelClankerDataset
from clanker_dataset import ModelDataLoader
from clanker_dataset import RawClankerDataset
from clanker_dataset import TorchBatchLoader
from clanker_dataset import train_episode_indices
from clanker_config import LeRobotClankerDataConfig

os.environ.setdefault("OPENPI_DATA_HOME", str(pathlib.Path("./.cache/openpi").resolve()))

PROJECT_NAME = "openpi"
RUN_NAME = "h5_repaired_v1"

REPO_ID = "rspcunningham/clanker0-teleop"

ACTION_DIM = 4
ACTION_HORIZON = 5
MAX_TOKEN_LEN = 180
PALIGEMMA_VARIANT = "gemma_2b_lora"

SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_TRAIN_STEPS = 2000
LOG_INTERVAL = 20
EVAL_INTERVAL = 100
SAVE_INTERVAL = 100
KEEP_PERIOD = 500
FSDP_DEVICES = 1

OVERWRITE = True
RESUME = False
WANDB_ENABLED = True

ASSETS_BASE_DIR = "./assets"
CHECKPOINT_BASE_DIR = "./checkpoints"

WARMUP_STEPS = 100
PEAK_LR = 1.0e-5
DECAY_STEPS = NUM_TRAIN_STEPS
END_LR = 1.0e-6
ADAM_B1 = 0.9
ADAM_B2 = 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-10
CLIP_GRAD_NORM = 1.0

EMA_DECAY = None

MODEL_CONFIG = pi0_fast.Pi0FASTConfig(
    action_dim=ACTION_DIM,
    action_horizon=ACTION_HORIZON,
    max_token_len=MAX_TOKEN_LEN,
    paligemma_variant=PALIGEMMA_VARIANT,
)
FREEZE_FILTER = MODEL_CONFIG.get_freeze_filter()
WEIGHT_LOADER = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params")
LR_SCHEDULE = optax.warmup_cosine_decay_schedule(
    init_value=PEAK_LR / (WARMUP_STEPS + 1),
    peak_value=PEAK_LR,
    warmup_steps=WARMUP_STEPS,
    decay_steps=DECAY_STEPS,
    end_value=END_LR,
)
TX = optax.chain(
    optax.clip_by_global_norm(CLIP_GRAD_NORM),
    optax.adamw(
        LR_SCHEDULE,
        b1=ADAM_B1,
        b2=ADAM_B2,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    ),
)


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(enabled: bool, project_name: str, exp_name: str, config_dict: dict, checkpoint_dir, *, resuming: bool) -> None:
    if not enabled:
        wandb.init(mode="disabled")
        return

    if resuming:
        run_id = (checkpoint_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=project_name)
        return

    wandb.init(name=exp_name, config=config_dict, project=project_name)
    (checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)


def _to_wandb_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return image


def load_weights_and_validate(loader: weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    model_config,
    freeze_filter,
    ema_decay,
    tx,
    weight_loader,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
):
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(nnx.All(nnx.Param, nnx.Not(freeze_filter)))),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = load_weights_and_validate(weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(freeze_filter, ema_decay, rng: at.KeyArrayLike, state: training_utils.TrainState, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model, rng: at.KeyArrayLike, observation, actions):
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    diff_state = nnx.DiffState(0, trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: ema_decay * old + (1 - ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def eval_step(rng: at.KeyArrayLike, state: training_utils.TrainState, batch):
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()

    observation, actions = batch
    eval_rng = jax.random.fold_in(rng, state.step)
    loss = jnp.mean(model.compute_loss(eval_rng, observation, actions, train=False))
    return {"val_loss": loss}


def build_data_loader_config() -> training_config.TrainConfig:
    return training_config.TrainConfig(
        name="pi0_fast_clanker",
        exp_name=RUN_NAME,
        project_name=PROJECT_NAME,
        model=MODEL_CONFIG,
        weight_loader=WEIGHT_LOADER,
        freeze_filter=FREEZE_FILTER,
        data=LeRobotClankerDataConfig(
            repo_id=REPO_ID,
            base_config=training_config.DataConfig(prompt_from_task=True),
        ),
        ema_decay=EMA_DECAY,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_train_steps=NUM_TRAIN_STEPS,
        log_interval=LOG_INTERVAL,
        save_interval=SAVE_INTERVAL,
        keep_period=KEEP_PERIOD,
        assets_base_dir=ASSETS_BASE_DIR,
        checkpoint_base_dir=CHECKPOINT_BASE_DIR,
        overwrite=OVERWRITE,
        resume=RESUME,
        wandb_enabled=WANDB_ENABLED,
        fsdp_devices=FSDP_DEVICES,
        seed=SEED,
    )


def create_episode_loader(
    loader_config: training_config.TrainConfig,
    episodes: list[int],
    *,
    sharding_spec,
    shuffle: bool,
    num_batches: int | None = None,
):
    data_config = loader_config.data.create(loader_config.assets_dirs, loader_config.model)
    raw_dataset = RawClankerDataset(
        ClankerDatasetConfig(
            repo_id=data_config.repo_id,
            action_horizon=loader_config.model.action_horizon,
            max_token_len=loader_config.model.max_token_len,
            action_dim=loader_config.model.action_dim,
            use_quantile_norm=data_config.use_quantile_norm,
        ),
        episodes,
    )
    dataset = ModelClankerDataset(
        raw_dataset,
        data_config.norm_stats,
        loader_config.model,
        use_quantile_norm=data_config.use_quantile_norm,
    )
    local_batch_size = loader_config.batch_size // jax.process_count()
    if num_batches is None and not shuffle:
        num_batches = max(1, len(dataset) // local_batch_size)
    torch_loader = TorchBatchLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding_spec,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=loader_config.num_workers,
        seed=loader_config.seed,
    )
    return ModelDataLoader(data_config, torch_loader)


def main() -> None:
    loader_config = build_data_loader_config()
    checkpoint_dir = loader_config.checkpoint_dir
    train_episodes = train_episode_indices(REPO_ID, VAL_EPISODES)
    val_episodes = list(VAL_EPISODES)

    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Checkpoint dir: {checkpoint_dir}")
    logging.info(f"Train episodes: {train_episodes}")
    logging.info(f"Val episodes: {val_episodes}")

    if BATCH_SIZE % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {BATCH_SIZE} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str((checkpoint_dir.parent / ".jax_cache").resolve()))

    rng = jax.random.key(SEED)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(FSDP_DEVICES)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=KEEP_PERIOD,
        overwrite=OVERWRITE,
        resume=RESUME,
    )
    init_wandb(
        WANDB_ENABLED,
        PROJECT_NAME,
        RUN_NAME,
        {
            "repo_id": REPO_ID,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "max_token_len": MAX_TOKEN_LEN,
            "batch_size": BATCH_SIZE,
            "num_train_steps": NUM_TRAIN_STEPS,
            "eval_interval": EVAL_INTERVAL,
            "save_interval": SAVE_INTERVAL,
            "paligemma_variant": PALIGEMMA_VARIANT,
        },
        checkpoint_dir,
        resuming=resuming,
    )

    train_loader = create_episode_loader(
        loader_config,
        train_episodes,
        sharding_spec=data_sharding,
        shuffle=True,
    )
    val_loader = create_episode_loader(
        loader_config,
        val_episodes,
        sharding_spec=data_sharding,
        shuffle=False,
    )
    data_iter = iter(train_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    images_to_log = [
        wandb.Image(
            np.concatenate([_to_wandb_image(np.array(img[i])) for img in batch[0].images.values()], axis=1)
        )
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(
        MODEL_CONFIG,
        FREEZE_FILTER,
        EMA_DECAY,
        TX,
        WEIGHT_LOADER,
        init_rng,
        mesh,
        resume=resuming,
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = checkpoints.restore_state(checkpoint_manager, train_state, train_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, FREEZE_FILTER, EMA_DECAY),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        eval_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, NUM_TRAIN_STEPS),
        initial=start_step,
        total=NUM_TRAIN_STEPS,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % LOG_INTERVAL == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if step % EVAL_INTERVAL == 0:
                val_infos = []
                for val_batch in val_loader:
                    with sharding.set_mesh(mesh):
                        val_infos.append(peval_step(train_rng, train_state, val_batch))
                stacked_val_infos = common_utils.stack_forest(val_infos)
                reduced_val_info = jax.device_get(jax.tree.map(jnp.mean, stacked_val_infos))
                reduced_info.update(reduced_val_info)
            pbar.write("Step {}: {}".format(step, ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())))
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (step % SAVE_INTERVAL == 0 and step > start_step) or step == NUM_TRAIN_STEPS - 1:
            checkpoints.save_state(checkpoint_manager, train_state, train_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
