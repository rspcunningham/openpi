from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from clanker_config import make_config
from clanker_constants import VAL_EPISODES
from clanker_dataset import ClankerDatasetConfig, RawClankerDataset, train_episode_indices
from openpi.models import model as model_lib
from openpi.models.tokenizer import FASTTokenizer
from openpi.policies import policy_config
import openpi.shared.normalize as normalize


CHECKPOINT_DIR = pathlib.Path("./checkpoints/pi0_fast_clanker/h5_repaired_v1/1100")
TRAIN_SAMPLES = 8
VAL_SAMPLES = 8
PRINT_EXAMPLES = 3


def _infer_raw_tokens(policy, obs: dict[str, np.ndarray | str]) -> np.ndarray:
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = model_lib.Observation.from_dict(inputs)
    policy._rng, sample_rng = jax.random.split(policy._rng)
    raw_tokens = policy._sample_actions(sample_rng, observation)
    return np.asarray(raw_tokens[0], dtype=np.int32)


def _decode_text(tokenizer: FASTTokenizer, tokens: np.ndarray) -> str:
    tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
    if (eos_idx := np.flatnonzero(tokens == 1)).size > 0:
        tokens = tokens[: eos_idx[0] + 1]
    if (non_pad_idx := np.flatnonzero(tokens != 0)).size > 0:
        tokens = tokens[: non_pad_idx[-1] + 1]
    return tokenizer._paligemma_tokenizer.decode(tokens.tolist())


def _extract_action_suffix(decoded_text: str) -> str:
    if "Action: " not in decoded_text:
        return "<missing Action: prefix>"
    suffix = decoded_text.split("Action: ", 1)[1]
    if "|" in suffix:
        suffix = suffix.split("|", 1)[0]
    return suffix


def _eval_split(name: str, dataset: RawClankerDataset, policy, tokenizer: FASTTokenizer, limit: int) -> None:
    successes = 0
    first_action_l1_errors: list[float] = []
    examples_printed = 0

    print(f"\n== {name} ==")
    print(f"num_examples={min(limit, len(dataset))}")

    for idx in range(min(limit, len(dataset))):
        item = dataset[idx]
        obs = {
            "observation/images/front": item["image"],
            "observation/state": item["state"],
            "prompt": item["prompt"],
        }
        raw_tokens = _infer_raw_tokens(policy, obs)
        decoded_text = _decode_text(tokenizer, raw_tokens)
        target_first_action = np.asarray(item["actions"][0] + item["state"], dtype=np.float32)

        try:
            pred_actions = policy.infer(obs)["actions"]
            pred_first_action = np.asarray(pred_actions[0], dtype=np.float32)
            successes += 1
            first_action_l1_errors.append(float(np.mean(np.abs(pred_first_action - target_first_action))))
            decode_ok = True
        except Exception as exc:
            pred_first_action = None
            decode_ok = False
            error_text = str(exc)

        if examples_printed < PRINT_EXAMPLES:
            print(f"\n[{name} sample {idx}]")
            print(f"prompt: {item['prompt']!r}")
            print(f"state: {np.asarray(item['state']).tolist()}")
            print(f"target_first_action_abs: {target_first_action.tolist()}")
            print(f"generated_suffix: {_extract_action_suffix(decoded_text)}")
            if decode_ok:
                print(f"pred_first_action_abs: {pred_first_action.tolist()}")
                print(f"first_action_l1: {first_action_l1_errors[-1]:.6f}")
            else:
                print(f"decode_failed: {error_text}")
            examples_printed += 1

    success_rate = successes / max(1, min(limit, len(dataset)))
    mean_l1 = float(np.mean(first_action_l1_errors)) if first_action_l1_errors else float("nan")
    print(f"\n{name}_decode_success_rate={success_rate:.3f}")
    print(f"{name}_mean_first_action_l1={mean_l1:.6f}")


def main() -> None:
    config = make_config()
    dataset_cfg = ClankerDatasetConfig(
        repo_id=config.data.repo_id,
        action_horizon=config.model.action_horizon,
        max_token_len=config.model.max_token_len,
        action_dim=config.model.action_dim,
    )

    norm_stats = normalize.load(CHECKPOINT_DIR / "assets" / dataset_cfg.repo_id)
    train_episodes = train_episode_indices(dataset_cfg.repo_id, VAL_EPISODES, root=dataset_cfg.root)
    val_episodes = list(VAL_EPISODES)

    train_dataset = RawClankerDataset(dataset_cfg, train_episodes, include_image_prompt=True)
    val_dataset = RawClankerDataset(dataset_cfg, val_episodes, include_image_prompt=True)

    policy = policy_config.create_trained_policy(
        config,
        CHECKPOINT_DIR,
        default_prompt=None,
        norm_stats=norm_stats,
    )
    tokenizer = FASTTokenizer(config.model.max_token_len)

    print(f"checkpoint={CHECKPOINT_DIR}")
    print(f"action_horizon={config.model.action_horizon}")
    print(f"action_dim={config.model.action_dim}")

    _eval_split("train", train_dataset, policy, tokenizer, TRAIN_SAMPLES)
    _eval_split("val", val_dataset, policy, tokenizer, VAL_SAMPLES)


if __name__ == "__main__":
    main()
