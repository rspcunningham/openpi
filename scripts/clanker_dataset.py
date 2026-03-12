from __future__ import annotations

import dataclasses
import os
import pathlib
import warnings

import einops
import jax
import numpy as np
import torch

from openpi.models import model as model_lib
from openpi.models.tokenizer import FASTTokenizer
from openpi.shared import image_tools
import openpi.shared.normalize as normalize

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover - depends on local environment
    LeRobotDataset = None
    _LEROBOT_IMPORT_ERROR = exc
else:
    _LEROBOT_IMPORT_ERROR = None

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
    module=r"torchvision\.io\._video_deprecation_warning",
)


def _require_new_lerobot() -> None:
    if LeRobotDataset is None:
        raise ImportError(
            "This Clanker pipeline expects the newer LeRobot API "
            "(`from lerobot.datasets.lerobot_dataset import LeRobotDataset`). "
            "Update/install LeRobot in this environment before running the scripts."
        ) from _LEROBOT_IMPORT_ERROR


def _default_lerobot_root() -> pathlib.Path:
    return pathlib.Path(os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _normalize_array(x: np.ndarray, stats: normalize.NormStats, *, use_quantiles: bool) -> np.ndarray:
    if use_quantiles:
        assert stats.q01 is not None and stats.q99 is not None
        return 2.0 * (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) - 1.0
    return (x - stats.mean) / (stats.std + 1e-6)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    image = image_tools.resize_with_pad(image, 224, 224)
    return image


@dataclasses.dataclass(frozen=True)
class ClankerDatasetConfig:
    repo_id: str
    action_horizon: int
    max_token_len: int
    action_dim: int = 4
    root: pathlib.Path = dataclasses.field(default_factory=_default_lerobot_root)
    video_backend: str = "pyav"
    camera_key: str = "observation.images.front"
    state_key: str = "observation.state"
    action_key: str = "action"
    use_quantile_norm: bool = True


def discover_episode_indices(repo_id: str, *, root: pathlib.Path | None = None) -> list[int]:
    _require_new_lerobot()
    dataset = LeRobotDataset(repo_id, root=root or _default_lerobot_root(), force_cache_sync=False)
    return sorted({int(idx) for idx in dataset.hf_dataset["episode_index"]})


def train_episode_indices(repo_id: str, val_episodes: tuple[int, ...], *, root: pathlib.Path | None = None) -> list[int]:
    val_set = set(val_episodes)
    return [episode for episode in discover_episode_indices(repo_id, root=root) if episode not in val_set]


class RawClankerDataset:
    def __init__(self, cfg: ClankerDatasetConfig, episodes: list[int], *, include_image_prompt: bool = True):
        _require_new_lerobot()
        self.cfg = cfg
        self.include_image_prompt = include_image_prompt
        self.ds = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
            episodes=episodes,
            video_backend=cfg.video_backend,
            force_cache_sync=False,
        )
        self.episode_index_col = self.ds.hf_dataset["episode_index"]

    def __len__(self) -> int:
        return len(self.ds)

    def _build_action_chunk(self, idx: int) -> np.ndarray:
        chunk_actions: list[np.ndarray] = []
        episode_index = int(self.episode_index_col[idx])

        for t in range(self.cfg.action_horizon):
            future_idx = idx + t
            if future_idx >= len(self.ds):
                break
            if int(self.episode_index_col[future_idx]) != episode_index:
                break
            future_sample = self.ds[future_idx]
            chunk_actions.append(np.asarray(future_sample[self.cfg.action_key], dtype=np.float32))

        if not chunk_actions:
            return np.zeros((self.cfg.action_horizon, self.cfg.action_dim), dtype=np.float32)

        chunk = np.stack(chunk_actions, axis=0)
        if chunk.shape[0] < self.cfg.action_horizon:
            pad = np.zeros((self.cfg.action_horizon - chunk.shape[0], chunk.shape[-1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | str]:
        sample = self.ds[idx]
        state = np.asarray(sample[self.cfg.state_key], dtype=np.float32)
        actions_abs = self._build_action_chunk(idx)
        actions_delta = actions_abs - state[None, :]
        result: dict[str, np.ndarray | str] = {
            "state": state,
            "actions": actions_delta,
        }
        if self.include_image_prompt:
            prompt = sample.get("task", "")
            if not isinstance(prompt, str):
                prompt = ""
            result["image"] = _parse_image(sample[self.cfg.camera_key])
            result["prompt"] = prompt
        return result


class ModelClankerDataset:
    def __init__(
        self,
        raw_dataset: RawClankerDataset,
        norm_stats: dict[str, normalize.NormStats],
        model_config,
        *,
        use_quantile_norm: bool,
    ):
        self.raw_dataset = raw_dataset
        self.norm_stats = norm_stats
        self.model_config = model_config
        self.use_quantile_norm = use_quantile_norm
        self.tokenizer = FASTTokenizer(model_config.max_token_len)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        sample = self.raw_dataset[idx]
        state = _normalize_array(sample["state"], self.norm_stats["state"], use_quantiles=self.use_quantile_norm)
        actions = _normalize_array(sample["actions"], self.norm_stats["actions"], use_quantiles=self.use_quantile_norm)

        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(sample["prompt"], state, actions)

        arm_image = np.asarray(sample["image"])
        zero_image = np.zeros_like(arm_image)
        return {
            "image": {
                "base_0_rgb": zero_image,
                "base_1_rgb": zero_image,
                "wrist_0_rgb": arm_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.True_,
                "wrist_0_rgb": np.True_,
            },
            "state": state.astype(np.float32),
            "tokenized_prompt": tokens.astype(np.int32),
            "tokenized_prompt_mask": token_mask.astype(np.bool_),
            "token_ar_mask": ar_mask.astype(np.int32),
            "token_loss_mask": loss_mask.astype(np.bool_),
            "actions": actions.astype(np.float32),
        }


class ModelDataLoader:
    def __init__(self, data_config, torch_loader):
        self._data_config = data_config
        self._torch_loader = torch_loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        for batch in self._torch_loader:
            yield model_lib.Observation.from_dict(batch), batch["actions"]


def _collate_fn(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


class TorchBatchLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding=None,
        as_jax_arrays: bool = True,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        self._as_jax_arrays = as_jax_arrays
        self._sharding = sharding
        if as_jax_arrays and sharding is None:
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            drop_last=True,
            generator=generator,
        )

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                if self._as_jax_arrays:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield batch
