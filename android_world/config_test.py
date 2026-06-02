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

"""Tests for typed Android World run config."""

import json

from absl.testing import absltest
from android_world import config as run_config_lib


class RunConfigTest(absltest.TestCase):

  def test_json_round_trip(self):
    config = run_config_lib.RunConfig(
        env=run_config_lib.EnvConfig(adb_path='/tmp/adb', console_port=5556),
        suite=run_config_lib.SuiteConfig(
            family='miniwob', tasks=['TaskA'], max_steps=7
        ),
        agent=run_config_lib.AgentConfig(
            name='t3a_openai_compatible',
            llm=run_config_lib.LLMConfig(
                model_name='local-model',
                api_base_url='http://localhost:8000/v1',
                api_key_env='LOCAL_KEY',
                max_tokens=None,
                temperature=None,
                extra_body={'top_k': 8},
            ),
            ui_state_mode='compiled',
        ),
        output=run_config_lib.OutputConfig(
            checkpoint_dir='/tmp/run', benchmark_state='/tmp/state.txt'
        ),
    )
    path = self.create_tempfile('run_config.json').full_path

    config.to_json(path)
    loaded = run_config_lib.RunConfig.from_json(path)

    self.assertEqual(loaded, config)

  def test_max_steps_zero_normalizes_to_none(self):
    config = run_config_lib.RunConfig.from_dict({'suite': {'max_steps': 0}})

    self.assertIsNone(config.suite.max_steps)

  def test_legacy_provider_json_mapping(self):
    path = self.create_tempfile('llm.json').full_path
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(
          {
              'model': 'deepseek-chat',
              'base_url': 'https://example.test/v1',
              'api_key_env': 'DEEPSEEK_API_KEY',
              'api_key': '',
              'max_retry': 4,
              'max_tokens': 4096,
              'temperature': 0.2,
              'extra_body': {'reasoning': False},
              'extra_request_kwargs': {'timeout': 60},
          },
          f,
      )

    config = run_config_lib.LLMConfig.from_provider_json(path)

    self.assertEqual(config.model_name, 'deepseek-chat')
    self.assertEqual(config.api_base_url, 'https://example.test/v1')
    self.assertEqual(config.api_key_env, 'DEEPSEEK_API_KEY')
    self.assertIsNone(config.api_key)
    self.assertEqual(config.max_retry, 4)
    self.assertEqual(config.max_tokens, 4096)
    self.assertEqual(config.temperature, 0.2)
    self.assertEqual(config.extra_body, {'reasoning': False})
    self.assertEqual(config.extra_request_kwargs, {'timeout': 60})


if __name__ == '__main__':
  absltest.main()
