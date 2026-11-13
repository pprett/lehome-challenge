from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import mixed_autonomy as ma


@dataclass(frozen=True)
class MixedAutonomyEvaluationConfig:
    client: ma.ClientConfig
    sidecar: ma.SidecarConfig


def build_mixed_autonomy_evaluation_config(
    *,
    task_description: str,
    is_bimanual: bool,
) -> MixedAutonomyEvaluationConfig:
    """Build the Mixed Autonomy manifest for one LeHome evaluation process."""

    joint_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    components = (("left_arm", "left_"), ("right_arm", "right_")) if is_bimanual else (("arm", ""),)
    signals: list[ma.StateSignal] = []
    for component, joint_prefix in components:
        signal_name = f"{component}_joints"
        signals.extend(
            (
                ma.JointStateSignal(
                    name=signal_name,
                    entity_path=f"robot/{component}",
                    joint_names=tuple(f"{joint_prefix}{name}" for name in joint_names),
                ),
                ma.TransformSignal(
                    name=f"{component}_end_effector_pose",
                    frame_id="world",
                    child_frame_id=f"{component}/gripper_link",
                    role="end_effector",
                    producer=ma.StateSignalProducer.FORWARD_KINEMATICS,
                ),
            )
        )

    sidecar = ma.SidecarConfig.from_env()
    if sidecar.backend_addr is None:
        raise RuntimeError(
            "Mixed Autonomy relay is not configured. Set MA_BACKEND_ADDR=<relay-host>:4433 "
            "before starting evaluation. For the development stack, also set MA_SERVER_NAME, "
            "MA_TENANT_ID, MA_ROBOT_ID, MA_AUTH_TOKEN or MA_AUTH_TOKEN_FILE, and MA_CA_CERT."
        )
    if sidecar.auth_token is None and sidecar.auth_token_file is None:
        raise RuntimeError(
            "Mixed Autonomy authentication is not configured. Set MA_AUTH_TOKEN or "
            "MA_AUTH_TOKEN_FILE before starting evaluation."
        )

    embodiment_id = "bimanual_so101" if is_bimanual else "single_so101"
    task = ma.TaskSpec(
        task_id=os.environ.get("MA_TASK_ID")
        or f"lehome-{hashlib.sha256(task_description.encode()).hexdigest()[:12]}",
        instruction=task_description,
    )
    client = ma.ClientConfig(
        instance_name=embodiment_id,
        embodiment_id=embodiment_id,
        state_signals=tuple(signals),
        control_capabilities=(ma.TextSteerabilityControlCapability(),),
        current_task=task,
        task_capabilities=ma.TaskCapabilities(
            can_report=True,
            can_accept_assignments=False,
            can_request_teleoperation=False,
        ),
        cameras=(
            ma.CameraConfig(
                name="front",
                width=640,
                height=480,
                pixel_format=ma.PixelFormat.RGB8,
                monitoring_profile=ma.MediaProfile(
                    width=320,
                    height=240,
                    fps=5,
                    jpeg_quality=60,
                ),
                teleop_profile=ma.MediaProfile(
                    width=640,
                    height=480,
                    fps=30,
                    jpeg_quality=80,
                ),
            ),
        ),
    )
    return MixedAutonomyEvaluationConfig(client=client, sidecar=sidecar)
