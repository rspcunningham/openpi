from __future__ import annotations

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from clanker_config import make_config
from clanker_constants import VAL_EPISODES
from clanker_dataset import ClankerDatasetConfig, ModelClankerDataset, RawClankerDataset, train_episode_indices
from openpi.models import model as model_lib
from openpi.models.tokenizer import FASTTokenizer
from openpi.policies import policy_config
import openpi.shared.normalize as normalize


TRAINED_CHECKPOINT = pathlib.Path("./checkpoints/pi0_fast_clanker/default/199")
BASE_CHECKPOINT = pathlib.Path("./.cache/openpi/openpi-assets/checkpoints/pi0_fast_base")
SAMPLE_INDEX = 0


def _extract_action_text(decoded: str) -> str:
    if "Action: " not in decoded:
        return "<missing Action: prefix>"
    action_text = decoded.split("Action: ", 1)[1]
    if "|" in action_text:
        action_text = action_text.split("|", 1)[0]
    return action_text


def _decode_tokens(tokenizer: FASTTokenizer, tokens: np.ndarray) -> str:
    tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
    valid = tokens != 0
    if valid.any():
        tokens = tokens[: np.flatnonzero(valid)[-1] + 1]
    return tokenizer._paligemma_tokenizer.decode(tokens.tolist())


def _infer_raw_tokens(policy, obs: dict[str, np.ndarray | str]) -> np.ndarray:
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = model_lib.Observation.from_dict(inputs)
    policy._rng, sample_rng = jax.random.split(policy._rng)
    raw_tokens = policy._sample_actions(sample_rng, observation)
    return np.asarray(raw_tokens[0], dtype=np.int32)


def main() -> None:
    config = make_config()
    dataset_cfg = ClankerDatasetConfig(
        repo_id=config.data.repo_id,
        action_horizon=config.model.action_horizon,
        max_token_len=config.model.max_token_len,
        action_dim=config.model.action_dim,
    )
    train_eps = train_episode_indices(dataset_cfg.repo_id, VAL_EPISODES, root=dataset_cfg.root)

    norm_stats = normalize.load(TRAINED_CHECKPOINT / "assets" / dataset_cfg.repo_id)
    raw_dataset = RawClankerDataset(dataset_cfg, train_eps, include_image_prompt=True)
    model_dataset = ModelClankerDataset(
        raw_dataset,
        norm_stats,
        config.model,
        use_quantile_norm=dataset_cfg.use_quantile_norm,
    )

    raw_item = raw_dataset[SAMPLE_INDEX]
    model_item = model_dataset[SAMPLE_INDEX]

    obs = {
        "observation/images/front": raw_item["image"],
        "observation/state": raw_item["state"],
        "prompt": raw_item["prompt"],
    }

    tokenizer = FASTTokenizer(config.model.max_token_len)
    target_tokens = model_item["tokenized_prompt"][model_item["tokenized_prompt_mask"]]
    target_decoded = _decode_tokens(tokenizer, target_tokens)

    trained_policy = policy_config.create_trained_policy(
        config,
        TRAINED_CHECKPOINT,
        default_prompt=None,
    )
    base_config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, paligemma_variant="gemma_2b"),
        freeze_filter=None,
    )
    base_policy = policy_config.create_trained_policy(
        base_config,
        BASE_CHECKPOINT,
        default_prompt=None,
        norm_stats=norm_stats,
    )

    trained_tokens = _infer_raw_tokens(trained_policy, obs)
    base_tokens = _infer_raw_tokens(base_policy, obs)

    print("prompt:", raw_item["prompt"])
    print("state:", np.asarray(raw_item["state"]).tolist())
    print()
    print("TARGET FULL:")
    print(target_decoded)
    print("TARGET ACTION SUFFIX:")
    print(_extract_action_text(target_decoded))
    print()
    print("FINETUNED RAW TOKENS:")
    print(trained_tokens.tolist())
    trained_decoded = _decode_tokens(tokenizer, trained_tokens)
    print("FINETUNED FULL:")
    print(trained_decoded)
    print("FINETUNED ACTION SUFFIX:")
    print(_extract_action_text(trained_decoded))
    try:
        trained_actions = tokenizer.extract_actions(
            trained_tokens, config.model.action_horizon, config.model.action_dim
        )
        print("FINETUNED DECODED FIRST ACTION:", trained_actions[0].tolist())
    except Exception as exc:  # pragma: no cover - debugging script
        print("FINETUNED ACTION DECODE FAILED:", exc)
    print()
    print("BASE RAW TOKENS:")
    print(base_tokens.tolist())
    base_decoded = _decode_tokens(tokenizer, base_tokens)
    print("BASE FULL:")
    print(base_decoded)
    print("BASE ACTION SUFFIX:")
    print(_extract_action_text(base_decoded))
    try:
        base_actions = tokenizer.extract_actions(base_tokens, config.model.action_horizon, config.model.action_dim)
        print("BASE DECODED FIRST ACTION:", base_actions[0].tolist())
    except Exception as exc:  # pragma: no cover - debugging script
        print("BASE ACTION DECODE FAILED:", exc)


if __name__ == "__main__":
    main()
