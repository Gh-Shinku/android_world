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
command-line flags.
"""

from collections.abc import Sequence
import json
import os

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.env import interface

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      # os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      # os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
      os.path.abspath("/home/zyt/sda_ws/programs/android/platform-tools/adb")
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)
_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port',
    8554,
    'The gRPC port of the running Android emulator.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_FIRST_K_TASKS = flags.DEFINE_integer(
    'first_k_tasks',
    0,
    'Run only the first K task templates after suite filtering. If 0, run all.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)
_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    0,
    'Maximum number of agent steps per episode. If 0, use task complexity.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('./runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)
_PROMPT_DATA_OUT = flags.DEFINE_string(
    'prompt_data_out',
    '',
    'JSONL path to write T3A action-selection prompt component data.',
)
_RUNTIME_PROMPT_OUT = flags.DEFINE_string(
    'runtime_prompt_out',
    '',
    'JSONL path to write exact runtime T3A action-selection prompts.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'm3a_gpt4v', help='Agent name.')
_LLM_MODEL_NAME = flags.DEFINE_string(
    'llm_model_name',
    'gpt-4-turbo-2024-04-09',
    'Model name for OpenAI-compatible LLM backends.',
)
_LLM_API_BASE_URL = flags.DEFINE_string(
    'llm_api_base_url',
    'https://api.openai.com/v1',
    'Base URL for OpenAI-compatible chat completions APIs.',
)
_LLM_API_KEY_ENV = flags.DEFINE_string(
    'llm_api_key_env',
    'OPENAI_API_KEY',
    'Environment variable containing the API key for the LLM backend.',
)
_LLM_CONFIG_PATH = flags.DEFINE_string(
    'llm_config_path',
    '',
    'Path to a JSON config file for provider-specific LLM settings.',
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


def _load_llm_config() -> dict[str, object]:
  if not _LLM_CONFIG_PATH.value:
    return {}
  with open(_LLM_CONFIG_PATH.value, 'r', encoding='utf-8') as f:
    return json.load(f)


def _get_openai_compatible_wrapper(
    config: dict[str, object],
) -> infer.Gpt4Wrapper:
  if not config:
    return infer.Gpt4Wrapper(
        _LLM_MODEL_NAME.value,
        api_key_env=_LLM_API_KEY_ENV.value,
        api_base_url=_LLM_API_BASE_URL.value,
    )

  api_key = config.get('api_key')
  if api_key == '':
    api_key = None
  if api_key is not None and not isinstance(api_key, str):
    raise ValueError('LLM config field "api_key" must be a string.')
  temperature = config.get('temperature', 0.0)
  if temperature is not None:
    temperature = float(temperature)
  max_tokens = config.get('max_tokens', 1000)
  if max_tokens is not None:
    max_tokens = int(max_tokens)
  extra_body = config.get('extra_body', {})
  if not isinstance(extra_body, dict):
    raise ValueError('LLM config field "extra_body" must be an object.')
  extra_request_kwargs = config.get('extra_request_kwargs', {})
  if not isinstance(extra_request_kwargs, dict):
    raise ValueError(
        'LLM config field "extra_request_kwargs" must be an object.'
    )
  if extra_request_kwargs.get('stream'):
    raise ValueError('Streaming responses are not supported by Android World.')

  model_name = config.get('model')
  if model_name is None:
    model_name = 'gpt-4-turbo-2024-04-09'
  api_key_env = config.get('api_key_env')
  if api_key_env is None:
    api_key_env = 'OPENAI_API_KEY'
  api_base_url = config.get('base_url')
  if api_base_url is None:
    api_base_url = 'https://api.openai.com/v1'

  return infer.Gpt4Wrapper(
      model_name=str(model_name),
      api_key=api_key,
      api_key_env=str(api_key_env),
      api_base_url=str(api_base_url),
      max_retry=int(config.get('max_retry', 3)),
      max_tokens=max_tokens,
      temperature=temperature,
      extra_body=extra_body,
      extra_request_kwargs=extra_request_kwargs,
  )


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  llm_config = _load_llm_config()
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  # Gemini.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    agent = m3a.M3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 't3a_gemini_gcp':
    agent = t3a.T3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  # GPT.
  elif _AGENT_NAME.value == 't3a_gpt4':
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  # OpenAI-compatible APIs.
  elif _AGENT_NAME.value == 't3a_openai_compatible':
    agent = t3a.T3A(env, _get_openai_compatible_wrapper(llm_config))
  elif _AGENT_NAME.value == 'm3a_openai_compatible':
    agent = m3a.M3A(env, _get_openai_compatible_wrapper(llm_config))
  # SeeAct.
  elif _AGENT_NAME.value == 'seeact':
    agent = seeact.SeeAct(env)

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
      grpc_port=_GRPC_PORT.value,
  )

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  if _FIRST_K_TASKS.value:
    suite = suite_utils.Suite(list(suite.items())[: _FIRST_K_TASKS.value])
  suite.suite_family = _SUITE_FAMILY.value

  agent = _get_agent(env, _SUITE_FAMILY.value)

  if _SUITE_FAMILY.value.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
      max_n_steps=_MAX_STEPS.value or None,
      prompt_data_out=_PROMPT_DATA_OUT.value,
      runtime_prompt_out=_RUNTIME_PROMPT_OUT.value,
  )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
