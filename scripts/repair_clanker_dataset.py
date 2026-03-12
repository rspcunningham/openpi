from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_stats


def _make_action_array(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=values.shape[1])


def _rewrite_actions_in_table(table: pa.Table) -> tuple[pa.Table, np.ndarray]:
    data = table.to_pydict()
    states = np.asarray(data["observation.state"], dtype=np.float32)
    episodes = np.asarray(data["episode_index"], dtype=np.int64)

    shifted = np.empty_like(states)
    shifted[:-1] = states[1:]
    shifted[-1] = states[-1]

    # Keep episode boundaries intact by padding the terminal frame of each episode
    # with its own state rather than crossing into the next episode.
    boundary = episodes[:-1] != episodes[1:]
    shifted[:-1][boundary] = states[:-1][boundary]

    action_idx = table.schema.get_field_index("action")
    new_table = table.set_column(action_idx, "action", _make_action_array(shifted))
    return new_table, shifted


def _rewrite_dataset(root: Path) -> dict:
    data_files = sorted((root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")

    action_stats = RunningQuantileStats(quantile_list=[0.01, 0.10, 0.50, 0.90, 0.99])
    total_rows = 0

    for path in data_files:
        table = pq.read_table(path)
        new_table, shifted_actions = _rewrite_actions_in_table(table)
        pq.write_table(new_table, path)
        action_stats.update(shifted_actions)
        total_rows += len(shifted_actions)
        logging.info("Rewrote %s (%d rows)", path.relative_to(root), len(shifted_actions))

    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in action_stats.get_statistics().items()
    }
    write_stats(stats, root)
    logging.info("Updated action stats in %s", stats_path)
    return {"rows": total_rows, "files": len(data_files)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite Clanker LeRobot dataset actions to next-state targets.")
    parser.add_argument("--repo-id", default="rspcunningham/clanker0-teleop")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the repaired dataset back to the Hugging Face dataset repo after rewriting it locally.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional branch to push to instead of the default branch.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary working directory instead of deleting it.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset = LeRobotDataset(args.repo_id, force_cache_sync=False)
    src_root = Path(dataset.root)
    temp_dir = Path(tempfile.mkdtemp(prefix="repair-clanker-", dir="/tmp"))
    work_root = temp_dir / src_root.name
    shutil.copytree(src_root, work_root)
    logging.info("Copied dataset from %s to %s", src_root, work_root)

    result = _rewrite_dataset(work_root)
    logging.info("Repaired %d rows across %d parquet files", result["rows"], result["files"])

    if args.push:
        repaired = LeRobotDataset(args.repo_id, root=work_root, force_cache_sync=False)
        repaired.push_to_hub(branch=args.branch)
        logging.info("Pushed repaired dataset to %s", args.repo_id if args.branch is None else f"{args.repo_id}:{args.branch}")

    if args.keep_temp:
        logging.info("Keeping repaired dataset copy at %s", work_root)
    else:
        shutil.rmtree(temp_dir)
        logging.info("Deleted temporary directory %s", temp_dir)


if __name__ == "__main__":
    main()
