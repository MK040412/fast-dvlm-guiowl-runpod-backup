#!/usr/bin/env python
"""Run AndroidWorld with the GUI-Owl/Fast-dVLM HTTP policy server."""

from __future__ import annotations

from collections.abc import Sequence
import os
import sys

from absl import app
from absl import flags
from absl import logging

from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.env import env_launcher

sys.path.insert(0, os.path.dirname(__file__))
from guiowl_androidworld_agent import GuiOwlAgent  # noqa: E402

logging.set_verbosity(logging.WARNING)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = "none"


def _find_adb_directory() -> str:
    candidates = [
        "/workspace/android-sdk/platform-tools/adb",
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        "/opt/android-sdk/platform-tools/adb",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return "adb"


_ADB_PATH = flags.DEFINE_string("adb_path", _find_adb_directory(), "Path to adb.")
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup",
    False,
    "Run AndroidWorld one-time emulator setup.",
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer("console_port", 5554, "Emulator console port.")
_SUITE_FAMILY = flags.DEFINE_enum(
    "suite_family",
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    "Suite family.",
)
_TASK_RANDOM_SEED = flags.DEFINE_integer("task_random_seed", 30, "Task RNG seed.")
_TASKS = flags.DEFINE_list("tasks", None, "Comma-separated task list.")
_N_TASK_COMBINATIONS = flags.DEFINE_integer("n_task_combinations", 1, "Task variants.")
_FIXED_TASK_SEED = flags.DEFINE_boolean("fixed_task_seed", False, "Use same task seed.")
_CHECKPOINT_DIR = flags.DEFINE_string("checkpoint_dir", "", "Resume/output checkpoint dir.")
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "/opt/androidworld_eval/runs",
    "Output root.",
)
_SERVER_URL = flags.DEFINE_string(
    "guiowl_server_url",
    "http://127.0.0.1:8123",
    "GUI-Owl policy server URL.",
)


def main(argv: Sequence[str]) -> None:
    del argv
    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    task_registry = registry.TaskRegistry()
    suite = suite_utils.create_suite(
        task_registry.get_registry(family=_SUITE_FAMILY.value),
        n_task_combinations=_N_TASK_COMBINATIONS.value,
        seed=_TASK_RANDOM_SEED.value,
        tasks=_TASKS.value,
        use_identical_params=_FIXED_TASK_SEED.value,
    )
    suite.suite_family = _SUITE_FAMILY.value
    agent = GuiOwlAgent(env, server_url=_SERVER_URL.value)
    agent.transition_pause = None

    checkpoint_dir = (
        _CHECKPOINT_DIR.value
        if _CHECKPOINT_DIR.value
        else checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)
    )
    print(f"Starting AndroidWorld eval with guiowl_dvlm -> {checkpoint_dir}")
    suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
    )
    print(f"Finished AndroidWorld eval. Wrote to {checkpoint_dir}")
    env.close()


if __name__ == "__main__":
    app.run(main)
