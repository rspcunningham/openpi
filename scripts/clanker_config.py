from __future__ import annotations

import dataclasses
import os
import pathlib

import openpi.models.pi0_fast as pi0_fast
from openpi.policies import clanker_policy
import openpi.training.config as training_config
import openpi.training.optimizer as optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms

os.environ.setdefault("OPENPI_DATA_HOME", str(pathlib.Path("./.cache/openpi").resolve()))


@dataclasses.dataclass(frozen=True)
class LeRobotClankerDataConfig(training_config.DataConfigFactory):
    """LeRobot data config for rspcunningham/clanker0-teleop."""

    @staticmethod
    def repack_transforms() -> transforms.Group:
        return transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation.images.front": "observation/images/front",
                        "observation.state": "observation/state",
                        "action": "actions",
                    }
                )
            ]
        )

    def create(
        self, assets_dirs: pathlib.Path, model_config: training_config._model.BaseModelConfig
    ) -> training_config.DataConfig:
        data_transforms = transforms.Group(
            inputs=[clanker_policy.ClankerInputs(model_type=model_config.model_type)],
            outputs=[clanker_policy.ClankerOutputs(action_dim=model_config.action_dim)],
        )

        # Clanker actions are absolute target joint angles; convert them to delta actions for pi0-style training.
        delta_action_mask = transforms.make_bool_mask(4)
        data_transforms = data_transforms.push(
            inputs=[transforms.DeltaActions(delta_action_mask)],
            outputs=[transforms.AbsoluteActions(delta_action_mask)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms(),
            data_transforms=data_transforms,
            model_transforms=training_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def make_config() -> training_config.TrainConfig:
    model = pi0_fast.Pi0FASTConfig(
        action_dim=4,
        action_horizon=5,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    )

    return training_config.TrainConfig(
        name="pi0_fast_clanker",
        exp_name="default",
        model=model,
        data=LeRobotClankerDataConfig(
            repo_id="rspcunningham/clanker0-teleop",
            base_config=training_config.DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        optimizer=optimizer.AdamW(),
        lr_schedule=optimizer.CosineDecaySchedule(),
        batch_size=8,
        num_workers=2,
        num_train_steps=30_000,
        save_interval=500,
        keep_period=2_000,
        assets_base_dir="./assets",
        checkpoint_base_dir="./checkpoints",
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        fsdp_devices=1,
    )


def with_overrides(config: training_config.TrainConfig, **updates) -> training_config.TrainConfig:
    """Small helper for editing the code-defined config in one place from a script."""
    return dataclasses.replace(config, **updates)
