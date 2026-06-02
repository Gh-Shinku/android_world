# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line arguments.
"""

import argparse
from collections.abc import Sequence
import logging
import os
from pathlib import Path

from android_world import benchmark_state as benchmark_state_lib
from android_world import checkpointer as checkpointer_lib
from android_world import config as run_config_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import factory as agent_factory
from android_world.env import env_launcher

logging.getLogger().setLevel(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _parse_tasks(value: str) -> list[str] | None:
  if not value:
    return None
  tasks = [task.strip() for task in value.split(',') if task.strip()]
  return tasks or None


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command line parser for benchmark runs."""
  parser = argparse.ArgumentParser(description='Run Android World eval suite.')

  parser.add_argument(
      '--config',
      default=argparse.SUPPRESS,
      help='Path to a resolved RunConfig JSON. Explicit CLI args override it.',
  )
  parser.add_argument(
      '--adb_path',
      default=argparse.SUPPRESS,
      help=(
          'Path to adb. Defaults to ANDROID_WORLD_ADB_PATH, or lets the '
          'environment layer resolve adb when empty.'
      ),
  )
  parser.add_argument(
      '--perform_emulator_setup',
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help=(
          'Whether to perform emulator setup. This must be done once before '
          'running Android World.'
      ),
  )
  parser.add_argument(
      '--console_port',
      type=int,
      default=argparse.SUPPRESS,
      help='The console port of the running Android device.',
  )
  parser.add_argument(
      '--grpc_port',
      type=int,
      default=argparse.SUPPRESS,
      help='The gRPC port of the running Android emulator.',
  )

  parser.add_argument(
      '--suite_family',
      choices=[
          registry.TaskRegistry.ANDROID_WORLD_FAMILY,
          registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
          registry.TaskRegistry.MINIWOB_FAMILY,
          registry.TaskRegistry.ANDROID_FAMILY,
          registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
      ],
      default=argparse.SUPPRESS,
      help='Suite family to run. See registry.py for more information.',
  )
  parser.add_argument(
      '--task_random_seed',
      type=int,
      default=argparse.SUPPRESS,
      help='Random seed for task randomness.',
  )
  parser.add_argument(
      '--tasks',
      type=_parse_tasks,
      default=argparse.SUPPRESS,
      help=(
          'Comma-separated list of tasks to run in the given suite family. If '
          'omitted, run all tasks in the suite family.'
      ),
  )
  parser.add_argument(
      '--first_k_tasks',
      type=int,
      default=argparse.SUPPRESS,
      help='Run only the first K task templates after suite filtering.',
  )
  parser.add_argument(
      '--n_task_combinations',
      type=int,
      default=argparse.SUPPRESS,
      help='Number of task instances to run for each task template.',
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=argparse.SUPPRESS,
      help='Maximum number of agent steps per episode. If 0, use complexity.',
  )
  parser.add_argument(
      '--fixed_task_seed',
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help='Use the same task seed across task combinations.',
  )

  parser.add_argument(
      '--checkpoint_dir',
      default=argparse.SUPPRESS,
      help='Directory to save checkpoints and resume evaluation from.',
  )
  parser.add_argument(
      '--output_path',
      default=argparse.SUPPRESS,
      help='Path to save results to when checkpoint_dir is not provided.',
  )
  parser.add_argument(
      '--prompt_data_out',
      default=argparse.SUPPRESS,
      help='JSONL path to write T3A action-selection prompt component data.',
  )
  parser.add_argument(
      '--runtime_prompt_out',
      default=argparse.SUPPRESS,
      help='JSONL path to write exact runtime T3A action-selection prompts.',
  )
  parser.add_argument(
      '--benchmark_state',
      default=argparse.SUPPRESS,
      help=(
          'Line-based path containing TaskName_instance -> true/false state. '
          'If set, only instances marked false are run.'
      ),
  )
  parser.add_argument(
      '--benchmark_state_init_from',
      default=argparse.SUPPRESS,
      help='Checkpoint run directory used to initialize benchmark_state.',
  )
  parser.add_argument(
      '--benchmark_state_autosave',
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help='Whether to save benchmark_state after every episode.',
  )

  parser.add_argument(
      '--agent_name',
      default=argparse.SUPPRESS,
      help='Agent name.',
  )
  parser.add_argument(
      '--llm_model_name',
      default=argparse.SUPPRESS,
      help='Model name for OpenAI-compatible LLM backends.',
  )
  parser.add_argument(
      '--llm_api_base_url',
      default=argparse.SUPPRESS,
      help='Base URL for OpenAI-compatible chat completions APIs.',
  )
  parser.add_argument(
      '--llm_api_key_env',
      default=argparse.SUPPRESS,
      help='Environment variable containing the LLM backend API key.',
  )
  parser.add_argument(
      '--llm_config_path',
      default=argparse.SUPPRESS,
      help='Path to a JSON config file for provider-specific LLM settings.',
  )
  parser.add_argument(
      '--ui_state_mode',
      choices=['legacy', 'compiled'],
      default=argparse.SUPPRESS,
      help='UI state representation used by T3A/M3A agents.',
  )
  parser.add_argument(
      '--ui_state_include_system_ui',
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help='Whether compiled UI state keeps pure system UI surfaces.',
  )
  parser.add_argument(
      '--ui_state_include_invisible',
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help='Whether compiled UI state keeps invisible elements.',
  )
  return parser


def _apply_arg_overrides(
    config: run_config_lib.RunConfig,
    overrides: argparse.Namespace | dict[str, object],
) -> None:
  values = vars(overrides) if isinstance(overrides, argparse.Namespace) else overrides

  if 'adb_path' in values:
    config.env.adb_path = str(values['adb_path'])
  if 'perform_emulator_setup' in values:
    config.env.perform_emulator_setup = bool(values['perform_emulator_setup'])
  if 'console_port' in values:
    config.env.console_port = int(values['console_port'])
  if 'grpc_port' in values:
    config.env.grpc_port = int(values['grpc_port'])

  if 'suite_family' in values:
    config.suite.family = str(values['suite_family'])
  if 'task_random_seed' in values:
    config.suite.task_random_seed = int(values['task_random_seed'])
  if 'tasks' in values:
    config.suite.tasks = values['tasks']  # type: ignore[assignment]
  if 'first_k_tasks' in values:
    config.suite.first_k_tasks = int(values['first_k_tasks'])
  if 'n_task_combinations' in values:
    config.suite.n_task_combinations = int(values['n_task_combinations'])
  if 'max_steps' in values:
    config.suite.max_steps = int(values['max_steps']) or None
  if 'fixed_task_seed' in values:
    config.suite.fixed_task_seed = bool(values['fixed_task_seed'])

  if 'checkpoint_dir' in values:
    config.output.checkpoint_dir = str(values['checkpoint_dir'])
  if 'output_path' in values:
    config.output.output_path = str(values['output_path'])
  if 'prompt_data_out' in values:
    config.output.prompt_data_out = str(values['prompt_data_out'])
  if 'runtime_prompt_out' in values:
    config.output.runtime_prompt_out = str(values['runtime_prompt_out'])
  if 'benchmark_state' in values:
    config.output.benchmark_state = str(values['benchmark_state'])
  if 'benchmark_state_init_from' in values:
    config.output.benchmark_state_init_from = str(
        values['benchmark_state_init_from']
    )
  if 'benchmark_state_autosave' in values:
    config.output.benchmark_state_autosave = bool(
        values['benchmark_state_autosave']
    )

  if 'agent_name' in values:
    config.agent.name = str(values['agent_name'])
  if 'llm_model_name' in values:
    config.agent.llm.model_name = str(values['llm_model_name'])
  if 'llm_api_base_url' in values:
    config.agent.llm.api_base_url = str(values['llm_api_base_url'])
  if 'llm_api_key_env' in values:
    config.agent.llm.api_key_env = str(values['llm_api_key_env'])
  if 'ui_state_mode' in values:
    config.agent.ui_state_mode = str(values['ui_state_mode'])
  if 'ui_state_include_system_ui' in values:
    config.agent.ui_state_include_system_ui = bool(
        values['ui_state_include_system_ui']
    )
  if 'ui_state_include_invisible' in values:
    config.agent.ui_state_include_invisible = bool(
        values['ui_state_include_invisible']
    )


def _build_run_config(args: argparse.Namespace) -> run_config_lib.RunConfig:
  values = vars(args)
  if 'config' in values:
    config = run_config_lib.RunConfig.from_json(str(values['config']))
  else:
    config = run_config_lib.RunConfig()
  _apply_arg_overrides(config, args)

  if 'llm_config_path' in values:
    config.agent.llm = run_config_lib.LLMConfig.from_provider_json(
        str(values['llm_config_path'])
    )
  return config


def _get_benchmark_state(
    output_config: run_config_lib.OutputConfig,
) -> benchmark_state_lib.BenchmarkState | None:
  if not output_config.benchmark_state:
    return None

  state_path = Path(output_config.benchmark_state).resolve()
  if not state_path.exists():
    if not output_config.benchmark_state_init_from:
      raise ValueError(
          '--benchmark_state does not exist. Provide '
          '--benchmark_state_init_from to initialize it.'
      )
    state = benchmark_state_lib.state_from_run_dir(
        Path(output_config.benchmark_state_init_from).resolve()
    )
    benchmark_state_lib.save_state(state_path, state)
    print(f'Initialized benchmark state with {len(state)} records: {state_path}')

  return benchmark_state_lib.BenchmarkState(state_path)


def _resolve_checkpoint_dir(output_config: run_config_lib.OutputConfig) -> str:
  if output_config.checkpoint_dir:
    return output_config.checkpoint_dir
  return checkpointer_lib.create_run_directory(output_config.output_path)


def _build_suite(suite_config: run_config_lib.SuiteConfig) -> suite_utils.Suite:
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=suite_config.family),
      n_task_combinations=suite_config.n_task_combinations,
      seed=suite_config.task_random_seed,
      tasks=suite_config.tasks,
      use_identical_params=suite_config.fixed_task_seed,
  )
  if suite_config.first_k_tasks:
    suite = suite_utils.Suite(list(suite.items())[: suite_config.first_k_tasks])
  suite.suite_family = suite_config.family
  return suite


def _main(config: run_config_lib.RunConfig) -> None:
  """Runs eval suite and gets rewards back."""
  checkpoint_dir = _resolve_checkpoint_dir(config.output)
  config.output.checkpoint_dir = checkpoint_dir
  config.to_json(Path(checkpoint_dir) / 'run_config.json')

  benchmark_state = _get_benchmark_state(config.output)
  env = env_launcher.load_and_setup_env(
      console_port=config.env.console_port,
      emulator_setup=config.env.perform_emulator_setup,
      adb_path=config.env.adb_path,
      grpc_port=config.env.grpc_port,
  )
  try:
    suite = _build_suite(config.suite)
    print('Initializing agent...')
    agent = agent_factory.create_agent(config.agent, env, config.suite.family)

    print(
        f'Starting eval with agent {config.agent.name} and writing to'
        f' {checkpoint_dir}'
    )
    suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
        max_n_steps=config.suite.max_steps,
        prompt_data_out=config.output.prompt_data_out,
        runtime_prompt_out=config.output.runtime_prompt_out,
        benchmark_state=benchmark_state,
        benchmark_state_autosave=config.output.benchmark_state_autosave,
    )
    print(
        f'Finished running agent {config.agent.name} on {config.suite.family}'
        f' family. Wrote to {checkpoint_dir}.'
    )
  finally:
    env.close()


def main(argv: Sequence[str] | None = None) -> None:
  args = build_arg_parser().parse_args(argv)
  _main(_build_run_config(args))


if __name__ == '__main__':
  main()
