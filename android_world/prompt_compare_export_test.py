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

import numpy as np
from absl.testing import absltest
from android_world import constants
from android_world import prompt_compare_export
from android_world.env import representation_utils


def _element(text: str) -> representation_utils.UIElement:
  return representation_utils.UIElement(
      text=text,
      class_name='android.widget.TextView',
      bbox_pixels=representation_utils.BoundingBox(0, 50, 0, 50),
      is_clickable=True,
      is_enabled=True,
      is_visible=True,
  )


def _pixels() -> np.ndarray:
  return np.zeros((100, 100, 3), dtype=np.uint8)


def _t3a_episode(**episode_data_overrides):
  episode_data = {
      'before_screenshot': [_pixels()],
      'after_screenshot': [_pixels()],
      'before_element_list': [[_element('Before')]],
      'after_element_list': [[_element('After')]],
      'action_output': [
          'Reason: Need to tap the visible item.\n'
          'Action: {"action_type": "click", "index": 0}'
      ],
      'summary': ['Tapped the item.'],
  }
  episode_data.update(episode_data_overrides)
  return {
      constants.EpisodeConstants.GOAL: 'Tap the item',
      constants.EpisodeConstants.TASK_TEMPLATE: 'TestTask',
      constants.EpisodeConstants.AGENT_NAME: 'T3A',
      constants.EpisodeConstants.SCREEN_CONFIG: {'width': 100, 'height': 100},
      constants.EpisodeConstants.EPISODE_DATA: episode_data,
  }


class PromptCompareExportTest(absltest.TestCase):

  def test_t3a_step_exports_raw_and_compiled_action_prompts(self):
    exported = prompt_compare_export.export_prompt_compare([_t3a_episode()])

    self.assertEqual('Tap the item', exported['goal'])
    step = exported['steps'][0]
    self.assertIn('Here is a list of descriptions', step['action_prompt_raw'])
    self.assertIn('Before', step['action_prompt_raw'])
    self.assertIn('Here is the compiled UI state', step['action_prompt_compiled'])

  def test_summary_prompts_are_exported_when_action_and_states_exist(self):
    step = prompt_compare_export.export_prompt_compare([_t3a_episode()])[
        'steps'
    ][0]

    self.assertIn('Summary of this step', step['summary_prompt_raw'])
    self.assertIn('Before', step['summary_prompt_raw'])
    self.assertIn('After', step['summary_prompt_raw'])
    self.assertIn('Summary of this step', step['summary_prompt_compiled'])
    self.assertIn('This is the action you picked', step['summary_prompt_compiled'])

  def test_missing_fields_emit_empty_strings_without_raising(self):
    exported = prompt_compare_export.export_prompt_compare([
        _t3a_episode(
            before_screenshot=[None],
            after_screenshot=[None],
            action_output=['not parseable'],
        )
    ])

    self.assertEqual(
        {
            'action_prompt_raw': '',
            'action_prompt_compiled': '',
            'summary_prompt_raw': '',
            'summary_prompt_compiled': '',
        },
        exported['steps'][0],
    )

  def test_multiple_episodes_export_as_list(self):
    exported = prompt_compare_export.export_prompt_compare([
        _t3a_episode(),
        _t3a_episode(),
    ])

    self.assertIsInstance(exported, list)
    self.assertLen(exported, 2)


if __name__ == '__main__':
  absltest.main()
