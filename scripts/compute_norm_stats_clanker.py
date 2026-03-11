from __future__ import annotations

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
from clanker_constants import VAL_EPISODES
from clanker_dataset import ClankerDatasetConfig
from clanker_dataset import RawClankerDataset
from clanker_dataset import TorchBatchLoader
from clanker_dataset import train_episode_indices
from clanker_config import make_config


def main() -> None:
    config = make_config()
    data_config = config.data.create(config.assets_dirs, config.model)
    train_episodes = train_episode_indices(data_config.repo_id, VAL_EPISODES)
    dataset = RawClankerDataset(
        ClankerDatasetConfig(
            repo_id=data_config.repo_id,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            action_dim=config.model.action_dim,
        ),
        train_episodes,
        include_image_prompt=False,
    )

    num_batches = len(dataset) // config.batch_size
    loader = TorchBatchLoader(
        dataset,
        local_batch_size=config.batch_size,
        as_jax_arrays=False,
        num_workers=config.num_workers,
        shuffle=False,
        num_batches=num_batches,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, {key: value.get_statistics() for key, value in stats.items()})


if __name__ == "__main__":
    main()
