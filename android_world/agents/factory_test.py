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

"""Tests for agent factory."""

from unittest import mock

from absl.testing import absltest
from android_world import config as run_config_lib
from android_world.agents import factory


class FakeAgent:

  def __init__(self, name='T3A'):
    self.name = name
    self.transition_pause = 1.0
    self.additional_guidelines = None

  def set_task_guidelines(self, task_guidelines):
    self.additional_guidelines = task_guidelines


class AgentFactoryTest(absltest.TestCase):

  def test_expected_agent_names_registered(self):
    self.assertContainsSubset(
        [
            'human_agent',
            'random_agent',
            'seeact',
            't3a_gpt4',
            'm3a_gpt4v',
            't3a_gemini_gcp',
            'm3a_gemini_gcp',
            't3a_openai_compatible',
            'm3a_openai_compatible',
            't3a_ui_state_openai_compatible',
            'm3a_ui_state_openai_compatible',
        ],
        factory.registered_agent_names(),
    )

  def test_openai_compatible_injects_compiled_ui_state_provider(self):
    agent = FakeAgent('T3A')
    with mock.patch.object(factory, 'create_llm_wrapper') as mock_llm:
      with mock.patch.object(factory.t3a, 'T3A', return_value=agent) as mock_t3a:
        config = run_config_lib.AgentConfig(
            name='t3a_openai_compatible',
            ui_state_mode='compiled',
        )

        created = factory.create_agent(config, env=mock.Mock(), family='android')

    self.assertIs(created, agent)
    mock_llm.assert_called_once_with(config.llm)
    self.assertIsNotNone(mock_t3a.call_args.kwargs['ui_state_provider'])
    self.assertEqual(created.transition_pause, None)
    self.assertEqual(created.name, 't3a_openai_compatible')

  def test_ui_state_agent_name_enables_compiled_provider(self):
    agent = FakeAgent('M3A')
    with mock.patch.object(factory, 'create_llm_wrapper'):
      with mock.patch.object(factory.m3a, 'M3A', return_value=agent) as mock_m3a:
        config = run_config_lib.AgentConfig(
            name='m3a_ui_state_openai_compatible',
            ui_state_mode='legacy',
        )

        factory.create_agent(config, env=mock.Mock(), family='android')

    self.assertIsNotNone(mock_m3a.call_args.kwargs['ui_state_provider'])

  def test_miniwob_sets_pause_and_guidelines(self):
    agent = FakeAgent('SeeAct')
    with mock.patch.object(factory.seeact, 'SeeAct', return_value=agent):
      config = run_config_lib.AgentConfig(name='seeact')

      created = factory.create_agent(config, env=mock.Mock(), family='miniwob')

    self.assertEqual(created.transition_pause, factory.MINIWOB_TRANSITION_PAUSE)
    self.assertEqual(
        created.additional_guidelines, factory.MINIWOB_ADDITIONAL_GUIDELINES
    )
    self.assertEqual(created.name, 'seeact')


if __name__ == '__main__':
  absltest.main()
