#!/usr/bin/env python3
"""Tests for retrofit_readable_raw_compiled_prompts."""

from __future__ import annotations

from absl.testing import absltest

from scripts import retrofit_readable_raw_compiled_prompts as retrofit


def _element(label: str = 'Camera') -> dict:
  return {
      'text': label,
      'content_description': label,
      'class_name': 'android.widget.TextView',
      'bbox': None,
      'bbox_pixels': {'x_min': 0, 'x_max': 320, 'y_min': 0, 'y_max': 640},
      'hint_text': None,
      'is_checked': False,
      'is_checkable': False,
      'is_clickable': True,
      'is_editable': False,
      'is_enabled': True,
      'is_focused': False,
      'is_focusable': True,
      'is_long_clickable': False,
      'is_scrollable': False,
      'is_selected': False,
      'is_visible': True,
      'package_name': 'com.example',
      'resource_name': 'com.example:id/camera',
  }


class RetrofitReadableRawCompiledPromptsTest(absltest.TestCase):

  def test_retrofits_compiled_t3a_step(self):
    value = [{
        'goal': 'Take one photo.',
        'agent_name': 't3a_openai_compatible',
        'episode_data': {
            'steps': [{
                'ui_state_mode': 'compiled',
                'before_element_list': [_element()],
                'before_ui_state_text': 'Compiled UI',
                'action_prompt': 'Compiled prompt',
                'summary': 'Clicked Camera.',
            }],
        },
    }]

    result = retrofit.retrofit_value(value)
    step = result[0]['episode_data']['steps'][0]
    prompt_compare = step['prompt_compare']

    self.assertEqual(result[0]['retrofit_status'], 'ok')
    self.assertEqual(prompt_compare['actual_mode'], 'compiled')
    self.assertEqual(prompt_compare['actual_prompt_field'], 'action_prompt')
    self.assertIn(
        'Take one photo.', prompt_compare['alternative_prompts']['raw']
    )
    self.assertIn('UI element 0:', prompt_compare['alternative_prompts']['raw'])

  def test_missing_before_element_list_marks_incomplete(self):
    value = {
        'goal': 'Take one photo.',
        'agent_name': 't3a_openai_compatible',
        'episode_data': {
            'steps': [{
                'ui_state_mode': 'compiled',
                'before_ui_state_text': 'Compiled UI',
                'action_prompt': 'Compiled prompt',
            }],
        },
    }

    result = retrofit.retrofit_value(value)

    self.assertEqual(result['retrofit_status'], 'incomplete')
    self.assertIn('missing before_element_list', result['retrofit_errors'][0])


if __name__ == '__main__':
  absltest.main()
