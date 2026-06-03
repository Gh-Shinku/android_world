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

from typing import Any
from unittest import mock
from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import m3a
from android_world.env import adb_utils
from android_world.ui_state import provider as ui_state_provider
from android_world.utils import test_utils
import numpy as np


class MockMultimodalLlmWrapper(infer.MultimodalLlmWrapper):
  """Mock multimodal LLM wrapper for testing."""

  def __init__(self, mock_responses: list[tuple[str, Any]]):
    self.mock_responses = mock_responses
    self.index = 0

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Any]:
    if self.index < len(self.mock_responses):
      index = self.index
      self.index += 1
      return self.mock_responses[index][0], None, self.mock_responses[index][1]
    else:
      return infer.ERROR_CALLING_LLM, None, None


class FakeCompiledUiStateProvider:

  def build(self, state, *, screen_size, app_name='', activity=''):
    del state, screen_size, app_name, activity
    return ui_state_provider.UiStateView(
        mode='compiled',
        prompt_text='Compiled UI State\n  Actions:\n    A0 click button "Done"',
        action_map={},
    )


class M3AInteractionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_get_orientation = mock.patch.object(
        adb_utils,
        'get_orientation',
    ).start()
    self.mock_get_physical_frame_boundary = mock.patch.object(
        adb_utils,
        'get_physical_frame_boundary',
    ).start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_step_method_with_completion(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        'test raw response',
    )])
    self.mock_get_orientation.return_value = 0
    self.mock_get_physical_frame_boundary.return_value = [0, 0, 100, 100]
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step_data = agent.step(goal)
    self.assertTrue(step_data.done)

  def test_legacy_mode_records_raw_prompt(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        'test raw response',
    )])
    agent = m3a.M3A(env, llm)

    step_data = agent.step('do something')

    prompt_compare = step_data.data['prompt_compare']
    self.assertEqual(prompt_compare['actual_mode'], 'legacy')
    self.assertEqual(prompt_compare['actual_prompt_field'], 'action_prompt')
    self.assertEmpty(prompt_compare['alternative_prompts'])

  def test_compiled_mode_records_raw_and_compiled_prompts(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        'test raw response',
    )])
    agent = m3a.M3A(
        env, llm, ui_state_provider=FakeCompiledUiStateProvider()
    )

    step_data = agent.step('do something')

    prompt_compare = step_data.data['prompt_compare']
    self.assertEqual(prompt_compare['actual_mode'], 'compiled')
    self.assertEqual(prompt_compare['actual_prompt_field'], 'action_prompt')
    raw_prompt = prompt_compare['alternative_prompts']['raw']
    self.assertNotEqual(
        raw_prompt, step_data.data['action_prompt']
    )
    self.assertIn(
        'Here is a list of detailed information',
        raw_prompt,
    )
    self.assertIn('Here is the compiled UI state',
                  step_data.data['action_prompt'])

  def test_compiled_mode_uses_compiled_ui_state_for_summary_prompt(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([
        (
            (
                "Reason: answer question.\nAction: {'action_type': 'answer',"
                " 'text': 'fake answer.'}"
            ),
            'test raw response',
        ),
        ('fake summary', 'test raw response'),
    ])
    agent = m3a.M3A(
        env, llm, ui_state_provider=FakeCompiledUiStateProvider()
    )

    step_data = agent.step('do something')

    self.assertFalse(step_data.done)
    self.assertEqual(step_data.data['summary_ui_state_mode'], 'compiled')
    self.assertIn('Compiled UI State', step_data.data['summary_prompt'])
    self.assertIn('Here is the compiled UI state',
                  step_data.data['action_prompt'])

  def test_step_method_with_invalid_action_output(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        'Output in incorrect format.',
        'test raw response',
    )])
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step_data = agent.step(goal)

    self.assertFalse(step_data.done)
    self.assertIn(
        'Output for action selection is not in the correct format',
        step_data.data['summary'],
    )

  def test_history_recording(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([
        (
            (
                "Reason: answer question.\nAction: {'action_type': 'answer',"
                " 'text': 'fake answer.'}"
            ),
            'test raw response',
        ),
        (
            'fake summary',
            'test raw response',
        ),
        (
            (
                "Reason: completed.\nAction: {'action_type': 'status',"
                " 'goal_status': 'complete'}"
            ),
            'test raw response',
        ),
    ])
    self.mock_get_orientation.side_effect = [0, 0, 0]
    self.mock_get_physical_frame_boundary.side_effect = [
        [0, 0, 100, 100],
        [0, 0, 100, 100],
        [0, 0, 100, 100],
    ]
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step1_data = agent.step(goal)
    self.assertFalse(step1_data.done)
    self.assertIn('fake summary', step1_data.data['summary'])

    step2_data = agent.step(goal)
    self.assertTrue(step2_data.done)
    self.assertLen(agent.history, 2)

  def test_step_logs_progress_stages(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([
        (
            (
                "Reason: answer question.\nAction: {'action_type': 'answer',"
                " 'text': 'fake answer.'}"
            ),
            'test raw response',
        ),
        (
            'fake summary',
            'test raw response',
        ),
    ])
    self.mock_get_orientation.side_effect = [0, 0]
    self.mock_get_physical_frame_boundary.side_effect = [
        [0, 0, 100, 100],
        [0, 0, 100, 100],
    ]
    agent = m3a.M3A(env, llm)

    with mock.patch.object(m3a.progress, 'log') as mock_log:
      step_data = agent.step('do something')

    self.assertFalse(step_data.done)
    stages = [call.args[0] for call in mock_log.call_args_list]
    self.assertIn('llm_action', stages)
    self.assertIn('execute_action', stages)
    self.assertIn('llm_summary', stages)


if __name__ == '__main__':
  absltest.main()
