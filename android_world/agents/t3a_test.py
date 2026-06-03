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
from android_world.agents import t3a
from android_world.ui_state import provider as ui_state_provider
from android_world.utils import test_utils


class MockLlmWrapper(infer.LlmWrapper):
  """Mock LLM wrapper for testing."""

  def __init__(self, mock_responses: list[tuple[str, Any]]):
    self.mock_responses = mock_responses
    self.index = 0

  def predict(
      self,
      text_prompt: str,
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


class T3AInteractionTest(absltest.TestCase):

  def test_step_method_with_completion(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])
    agent = t3a.T3A(env, mock_llm)

    goal = "do something"
    step_data = agent.step(goal)

    self.assertTrue(step_data.done)

  def test_runtime_prompt_logger_records_action_prompt_before_llm_call(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])
    agent = t3a.T3A(env, mock_llm)
    records = []
    agent.set_runtime_prompt_logger(lambda **kwargs: records.append(kwargs))

    step_data = agent.step("do something")

    self.assertTrue(step_data.done)
    self.assertLen(records, 1)
    self.assertEqual(records[0]['prompt_kind'], 't3a_action_selection')
    self.assertEqual(records[0]['step_number'], 0)
    self.assertEqual(records[0]['goal'], 'do something')
    self.assertEqual(records[0]['prompt'], step_data.data['action_prompt'])

  def test_legacy_mode_records_raw_prompt(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])
    agent = t3a.T3A(env, mock_llm)

    step_data = agent.step("do something")

    prompt_compare = step_data.data['prompt_compare']
    self.assertEqual(prompt_compare['actual_mode'], 'legacy')
    self.assertEqual(prompt_compare['actual_prompt_field'], 'action_prompt')
    self.assertEmpty(prompt_compare['alternative_prompts'])

  def test_compiled_mode_records_raw_and_compiled_prompts(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])
    agent = t3a.T3A(
        env, mock_llm, ui_state_provider=FakeCompiledUiStateProvider()
    )
    records = []
    agent.set_runtime_prompt_logger(lambda **kwargs: records.append(kwargs))

    step_data = agent.step("do something")

    prompt_compare = step_data.data['prompt_compare']
    self.assertEqual(prompt_compare['actual_mode'], 'compiled')
    self.assertEqual(prompt_compare['actual_prompt_field'], 'action_prompt')
    raw_prompt = prompt_compare['alternative_prompts']['raw']
    self.assertNotEqual(
        raw_prompt, step_data.data['action_prompt']
    )
    self.assertIn('Here is a list of descriptions for some UI elements',
                  raw_prompt)
    self.assertIn('Here is the compiled UI state',
                  step_data.data['action_prompt'])
    self.assertLen(records, 1)
    self.assertEqual(
        records[0]['prompt_compare'], step_data.data['prompt_compare']
    )

  def test_compiled_mode_uses_compiled_ui_state_for_summary_prompt(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([
        (
            (
                "Reason: answer.\nAction: {'action_type': 'answer',"
                " 'text': 'mock_response'}"
            ),
            "fake_response_1",
        ),
        ("fake_summary", "fake_response_2"),
    ])
    agent = t3a.T3A(
        env, mock_llm, ui_state_provider=FakeCompiledUiStateProvider()
    )

    step_data = agent.step("do something")

    self.assertFalse(step_data.done)
    self.assertEqual(step_data.data['summary_ui_state_mode'], 'compiled')
    self.assertIn('Compiled UI State', step_data.data['summary_prompt'])
    self.assertIn('Here is the compiled UI state', step_data.data['action_prompt'])

  def test_history_recording(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([
        (
            (
                "Reason: completed.\nAction: {'action_type': 'answer',"
                " 'text': 'mock_response'}"
            ),
            "fake_response_1",
        ),
        (
            "fake_summary",
            "fake_response_1",
        ),
        (
            (
                "Reason: completed.\nAction: {'action_type': 'status',"
                " 'goal_status': 'complete'}"
            ),
            "fake_response_2",
        ),
    ])
    agent = t3a.T3A(env, mock_llm)

    goal = "do something"
    step1_data = agent.step(goal)
    self.assertFalse(step1_data.done)

    step2_data = agent.step(goal)
    self.assertTrue(step2_data.done)
    self.assertLen(agent.history, 2)

  def test_step_logs_progress_stages(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([
        (
            (
                "Reason: answer.\nAction: {'action_type': 'answer',"
                " 'text': 'mock_response'}"
            ),
            "fake_response_1",
        ),
        (
            "fake_summary",
            "fake_response_2",
        ),
    ])
    agent = t3a.T3A(env, mock_llm)

    with mock.patch.object(t3a.progress, 'log') as mock_log:
      step_data = agent.step("do something")

    self.assertFalse(step_data.done)
    stages = [call.args[0] for call in mock_log.call_args_list]
    self.assertIn('llm_action', stages)
    self.assertIn('execute_action', stages)
    self.assertIn('llm_summary', stages)


if __name__ == "__main__":
  absltest.main()
