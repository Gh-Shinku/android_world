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

"""Tests for run.py config pipeline."""

import json
import os
import tempfile
import unittest
from unittest import mock

from android_world import config as run_config_lib
import run as run_module


class RunPipelineTest(unittest.TestCase):

  def test_parser_parses_tasks_and_boolean_args(self):
    args = run_module.build_arg_parser().parse_args([
        '--tasks=TaskA,TaskB',
        '--fixed_task_seed',
        '--no-benchmark_state_autosave',
    ])

    self.assertEqual(args.tasks, ['TaskA', 'TaskB'])
    self.assertTrue(args.fixed_task_seed)
    self.assertFalse(args.benchmark_state_autosave)

  def test_config_override_only_updates_selected_fields(self):
    config = run_config_lib.RunConfig(
        suite=run_config_lib.SuiteConfig(
            family='android_world', task_random_seed=1, max_steps=9
        ),
        agent=run_config_lib.AgentConfig(
            name='m3a_openai_compatible',
            llm=run_config_lib.LLMConfig(model_name='base-model'),
            ui_state_mode='legacy',
        ),
    )

    args = run_module.build_arg_parser().parse_args([
        '--suite_family=miniwob',
        '--ui_state_mode=compiled',
    ])
    run_module._apply_arg_overrides(config, args)

    self.assertEqual(config.suite.family, 'miniwob')
    self.assertEqual(config.suite.task_random_seed, 1)
    self.assertEqual(config.suite.max_steps, 9)
    self.assertEqual(config.agent.llm.model_name, 'base-model')
    self.assertEqual(config.agent.ui_state_mode, 'compiled')

  def test_build_config_from_defaults_normalizes_max_steps_zero(self):
    args = run_module.build_arg_parser().parse_args(['--max_steps=0'])

    config = run_module._build_run_config(args)

    self.assertIsNone(config.suite.max_steps)
    self.assertEqual(config.agent.name, 'm3a_gpt4v')

  def test_build_config_loads_provider_json(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      llm_config_path = os.path.join(temp_dir, 'llm.json')
      with open(llm_config_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'model': 'provider-model',
                'base_url': 'http://localhost:8000/v1',
                'api_key_env': 'LOCAL_KEY',
                'max_tokens': 32,
                'temperature': 0.1,
            },
            f,
        )
      args = run_module.build_arg_parser().parse_args([
          f'--llm_config_path={llm_config_path}',
          '--llm_model_name=ignored-model',
      ])

      config = run_module._build_run_config(args)

    self.assertEqual(config.agent.llm.model_name, 'provider-model')
    self.assertEqual(config.agent.llm.api_base_url, 'http://localhost:8000/v1')
    self.assertEqual(config.agent.llm.api_key_env, 'LOCAL_KEY')
    self.assertEqual(config.agent.llm.max_tokens, 32)
    self.assertEqual(config.agent.llm.temperature, 0.1)

  def test_config_file_with_explicit_arg_override(self):
    base = run_config_lib.RunConfig(
        suite=run_config_lib.SuiteConfig(
            family='android_world', task_random_seed=7
        ),
        agent=run_config_lib.AgentConfig(
            name='m3a_openai_compatible',
            ui_state_mode='legacy',
        ),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      config_path = os.path.join(temp_dir, 'run_config.json')
      base.to_json(config_path)
      args = run_module.build_arg_parser().parse_args([
          f'--config={config_path}',
          '--ui_state_mode=compiled',
      ])

      config = run_module._build_run_config(args)

    self.assertEqual(config.suite.family, 'android_world')
    self.assertEqual(config.suite.task_random_seed, 7)
    self.assertEqual(config.agent.name, 'm3a_openai_compatible')
    self.assertEqual(config.agent.ui_state_mode, 'compiled')

  def test_main_writes_resolved_config_and_passes_pipeline_args(self):
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    checkpoint_dir = os.path.join(temp_dir.name, 'run')
    config = run_config_lib.RunConfig(
        env=run_config_lib.EnvConfig(
            adb_path='/tmp/adb',
            perform_emulator_setup=True,
            console_port=5556,
            grpc_port=8555,
        ),
        suite=run_config_lib.SuiteConfig(
            family='android_world',
            task_random_seed=7,
            max_steps=3,
        ),
        agent=run_config_lib.AgentConfig(name='random_agent'),
        output=run_config_lib.OutputConfig(
            checkpoint_dir=checkpoint_dir,
            prompt_data_out='/tmp/prompts.jsonl',
            runtime_prompt_out='/tmp/runtime.jsonl',
            benchmark_state_autosave=False,
        ),
    )
    env = mock.Mock()
    suite = mock.Mock()
    agent = mock.Mock()

    with mock.patch.object(run_module, '_build_suite', return_value=suite):
      with mock.patch.object(
          run_module.env_launcher, 'load_and_setup_env', return_value=env
      ) as mock_load_env:
        with mock.patch.object(
            run_module.agent_factory, 'create_agent', return_value=agent
        ) as mock_create_agent:
          with mock.patch.object(
              run_module, '_get_benchmark_state', return_value=None
          ):
            with mock.patch.object(run_module.suite_utils, 'run') as mock_run:
              run_module._main(config)

    with open(f'{checkpoint_dir}/run_config.json', 'r', encoding='utf-8') as f:
      saved_config = json.load(f)
    self.assertEqual(saved_config['output']['checkpoint_dir'], checkpoint_dir)
    self.assertEqual(saved_config['agent']['name'], 'random_agent')
    mock_load_env.assert_called_once_with(
        console_port=5556,
        emulator_setup=True,
        adb_path='/tmp/adb',
        grpc_port=8555,
    )
    mock_create_agent.assert_called_once_with(
        config.agent, env, config.suite.family
    )
    mock_run.assert_called_once()
    _, _, kwargs = mock_run.mock_calls[0]
    self.assertEqual(kwargs['max_n_steps'], 3)
    self.assertEqual(kwargs['prompt_data_out'], '/tmp/prompts.jsonl')
    self.assertEqual(kwargs['runtime_prompt_out'], '/tmp/runtime.jsonl')
    self.assertFalse(kwargs['benchmark_state_autosave'])
    env.close.assert_called_once()


if __name__ == '__main__':
  unittest.main()
