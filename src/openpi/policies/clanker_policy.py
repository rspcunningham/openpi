import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ClankerInputs(transforms.DataTransformFn):
    """Map the Clanker LeRobot schema to the pi0 / pi0-fast observation format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        arm_image = _parse_image(data["observation/images/front"])
        zero_image = np.zeros_like(arm_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image = {
                    "base_0_rgb": zero_image,
                    "left_wrist_0_rgb": arm_image,
                    "right_wrist_0_rgb": zero_image,
                }
                image_mask = {
                    "base_0_rgb": np.False_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.False_,
                }
            case _model.ModelType.PI0_FAST:
                image = {
                    "base_0_rgb": zero_image,
                    "base_1_rgb": zero_image,
                    "wrist_0_rgb": arm_image,
                }
                # FAST models do not mask out padded cameras in the existing repo adapters.
                image_mask = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": image,
            "image_mask": image_mask,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ClankerOutputs(transforms.DataTransformFn):
    action_dim: int = 4

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
