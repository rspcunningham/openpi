from __future__ import annotations

import logging
import socket

import numpy as np

from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from clanker_config import make_config


CHECKPOINT_DIR = "./checkpoints/pi0_fast_clanker/h5_repaired_v1/1999"
PORT = 8000
DEFAULT_PROMPT: str | None = None
RECORD = False
DEBUG_INFERENCE = True


class DebugPolicy(policy_lib.BasePolicy):
    def __init__(self, policy: policy_lib.BasePolicy):
        self._policy = policy

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        raw_state = np.asarray(obs.get("observation/state", []), dtype=np.float32)
        prompt = obs.get("prompt")
        logging.info("Inference input state=%s prompt=%r", raw_state.tolist(), prompt)
        outputs = self._policy.infer(obs, noise=noise)
        first_action = np.asarray(outputs.get("actions", []), dtype=np.float32)
        if first_action.ndim >= 2 and first_action.shape[0] > 0:
            logging.info("Inference output first_action=%s", first_action[0].tolist())
        else:
            logging.info("Inference output actions=%s", first_action.tolist())
        return outputs

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


def main() -> None:
    config = make_config()
    policy = policy_config.create_trained_policy(
        config,
        CHECKPOINT_DIR,
        default_prompt=DEFAULT_PROMPT,
    )

    if RECORD:
        policy = policy_lib.PolicyRecorder(policy, "policy_records")
    if DEBUG_INFERENCE:
        policy = DebugPolicy(policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=PORT,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
