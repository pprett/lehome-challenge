"""OpenPI websocket policy adapter for LeHome evaluation.

This policy delegates inference to a policy server that implements the
``openpi_client`` websocket protocol (for example ``nn_control/serve_policy.py``).
It converts LeHome observations into the server's metadata-advertised wire format,
maintains the requested observation horizon, and consumes returned action chunks.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry


# Prefer an installed openpi_client, but make the sibling checkout usable too.
try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    sibling_openpi_client = repo_root.parent / "nc_openpi" / "packages" / "openpi-client" / "src"
    if sibling_openpi_client.exists():
        sys.path.insert(0, str(sibling_openpi_client))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy


@PolicyRegistry.register("openpi_ws")
class OpenPIWebsocketPolicy(BasePolicy):
    """Policy adapter for OpenPI-compatible websocket policy servers.

    The server is expected to send metadata on connect and return
    ``{"action": np.ndarray}`` from ``infer``. If the action has shape
    ``(n_action_steps, action_dim)``, actions are cached and replayed one at a
    time before the next websocket inference request.
    """

    # Explicit server-key -> LeHome observation-key mapping.
    # Keep this hard-coded for now so a metadata/spec change fails loudly instead
    # of silently picking the wrong camera or low-dimensional observation.
    SERVER_TO_LEHOME_OBS_KEY = {
        "/image/robot/cam/top_rgb": "observation.images.top_rgb",        
        "/image/robot/cam/left_rgb": "observation.images.left_rgb",
        "/image/robot/cam/right_rgb": "observation.images.right_rgb",
        "/lowdim/robot/follower/state/qpos": "observation.state",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: Optional[int] = 9999,
        api_key: Optional[str] = None,
        n_obs_steps: Optional[int] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Convenience: allow --policy_path ws://host:port for this policy.
        if model_path and model_path.startswith("ws"):
            host = model_path
            port = None

        self.client = WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self.metadata = self.client.get_server_metadata()
        self.n_obs_steps = int(n_obs_steps or self.metadata.get("n_obs_steps") or 1)
        self._history: deque[Dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)
        self._action_chunk: List[np.ndarray] = []
        self._chunk_idx = 0

        print(f"[OpenPIWebsocketPolicy] Connected to {host}:{port}; metadata={self.metadata}")

    def reset(self):
        """Clear local buffers and notify the client policy."""
        self._history.clear()
        self._action_chunk = []
        self._chunk_idx = 0
        self.client.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Return the next action for a LeHome observation."""
        self._history.append({k: v.copy() for k, v in observation.items() if isinstance(v, np.ndarray)})

        if self._chunk_idx >= len(self._action_chunk):
            request = self._build_openpi_observation(observation)
            response = self.client.infer(request)
            action = response.get("action")
            if action is None:
                raise RuntimeError(f"OpenPI websocket response did not contain 'action': {response}")

            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 1:
                self._action_chunk = [action]
            elif action.ndim == 2:
                self._action_chunk = [action[i] for i in range(action.shape[0])]
            else:
                raise ValueError(f"Expected action shape (D,) or (T,D), got {action.shape}")
            self._chunk_idx = 0

        out = self._action_chunk[self._chunk_idx]
        self._chunk_idx += 1
        return out.astype(np.float32, copy=False)

    def _build_openpi_observation(self, current_observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        specs = self.metadata.get("observation_specs") or []
        return {spec["key"]: self._array_for_spec(spec) for spec in specs}


    def _array_for_spec(self, spec: dict) -> np.ndarray:
        key = spec["key"]
        source_key = self._mapped_lehome_key(key)
        spec_type = str(spec.get("type", ""))
        if spec_type == "image":
            return self._stack_history(source_key, np.uint8)
        if spec_type == "depth":
            return self._stack_history(source_key, np.uint16)
        if spec_type.startswith("lowdim"):
            return self._stack_history(source_key, np.float32)
        raise ValueError(f"Unsupported OpenPI observation spec type for '{key}': {spec_type}")

    def _stack_history(self, source_key: str, dtype) -> np.ndarray:
        if not self._history:
            raise RuntimeError("Observation history is empty")
        frames = [obs[source_key] for obs in self._history if source_key in obs]
        if not frames:
            raise KeyError(f"Observation key '{source_key}' is required by websocket policy but is missing")
        while len(frames) < self.n_obs_steps:
            frames.insert(0, frames[0])
        arr = np.stack(frames[-self.n_obs_steps :], axis=0).astype(dtype, copy=False)
        if dtype == np.uint8 and arr.ndim == 4 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr)

    def _mapped_lehome_key(self, server_key: str) -> str:
        try:
            return self.SERVER_TO_LEHOME_OBS_KEY[server_key]
        except KeyError:
            known = ", ".join(sorted(self.SERVER_TO_LEHOME_OBS_KEY))
            raise KeyError(
                f"No explicit LeHome observation mapping for server key '{server_key}'. "
                f"Add it to OpenPIWebsocketPolicy.SERVER_TO_LEHOME_OBS_KEY. Known keys: {known}"
            ) from None

    @staticmethod
    def _as_list(value, fallback: list[str]) -> list[str]:
        if value is None:
            return fallback
        if isinstance(value, str):
            return [value]
        return list(value)
