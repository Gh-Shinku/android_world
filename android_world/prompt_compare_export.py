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

"""Exports raw/compiled prompt comparisons from checkpoint episodes."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

from android_world.agents import m3a_utils
from android_world import constants
from android_world.env import interface
from android_world.env import representation_utils
from android_world.ui_state import compiler
from android_world.ui_state import ir
import numpy as np

Episode = dict[str, Any]


def export_prompt_compare(
    episodes: list[Episode],
) -> dict[str, Any] | list[dict[str, Any]]:
  """Builds the prompt comparison sidecar payload.

  Args:
    episodes: Episodes loaded from or about to be saved to a checkpoint.

  Returns:
    A single episode object when one episode is supplied, otherwise a list of
    episode objects.
  """
  exported = [_episode_prompt_compare(episode) for episode in episodes]
  if len(exported) == 1:
    return exported[0]
  return exported


def write_prompt_compare(episodes: list[Episode], filename: str) -> None:
  """Writes prompt comparison JSON for checkpoint episodes."""
  with open(filename, 'w', encoding='utf-8') as f:
    json.dump(export_prompt_compare(episodes), f, ensure_ascii=False, indent=2)


def _episode_prompt_compare(episode: Episode) -> dict[str, Any]:
  goal = _string_or_empty(episode.get(constants.EpisodeConstants.GOAL))
  task_template = _string_or_empty(
      episode.get(constants.EpisodeConstants.TASK_TEMPLATE)
  )
  episode_data = episode.get(constants.EpisodeConstants.EPISODE_DATA)
  if not isinstance(episode_data, dict):
    episode_data = {}

  agent_kind = _agent_kind(episode.get(constants.EpisodeConstants.AGENT_NAME))
  screen_size = _screen_size(episode, episode_data)
  additional_guidelines = _additional_guidelines(episode)

  summaries: list[str] = []
  steps = []
  for step_index in range(_step_count(episode_data)):
    step = _build_step_compare(
        episode_data,
        step_index,
        agent_kind,
        goal,
        summaries,
        screen_size,
        additional_guidelines,
    )
    steps.append(step)

    summary = _get_step_value(episode_data.get('summary'), step_index)
    if isinstance(summary, str):
      if agent_kind == 'm3a':
        summaries.append('Step ' + str(len(summaries) + 1) + '- ' + summary)
      else:
        summaries.append('Step ' + str(len(summaries) + 1) + ': ' + summary)

  return {
      'goal': goal,
      'task_template': task_template,
      'steps': steps,
  }


def _build_step_compare(
    episode_data: dict[str, Any],
    step_index: int,
    agent_kind: str,
    goal: str,
    history: list[str],
    screen_size: tuple[int, int],
    additional_guidelines: list[str] | None,
) -> dict[str, str]:
  step = {
      'action_prompt_raw': '',
      'action_prompt_compiled': '',
      'summary_prompt_raw': '',
      'summary_prompt_compiled': '',
  }

  before_elements = _before_elements(episode_data, step_index, agent_kind)
  before_pixels = _before_pixels(episode_data, step_index, agent_kind)
  if isinstance(before_elements, list) and before_pixels is not None:
    raw_before = _raw_ui_state(agent_kind, before_elements, screen_size)
    if raw_before:
      step['action_prompt_raw'] = _raw_action_prompt(
          agent_kind, goal, history, raw_before, additional_guidelines
      )
    compiled_before = _compiled_ui_state(before_pixels, before_elements, screen_size)
    if compiled_before:
      step['action_prompt_compiled'] = _compiled_action_prompt(
          agent_kind, goal, history, compiled_before, additional_guidelines
      )

  after_elements = _after_elements(episode_data, step_index, agent_kind)
  after_pixels = _after_pixels(episode_data, step_index, agent_kind)
  reason, action = _reason_action(episode_data, step_index)
  if (
      isinstance(before_elements, list)
      and before_pixels is not None
      and isinstance(after_elements, list)
      and after_pixels is not None
      and reason
      and action
  ):
    raw_before = _raw_ui_state(agent_kind, before_elements, screen_size)
    raw_after = _raw_ui_state(agent_kind, after_elements, screen_size)
    if raw_before and raw_after:
      step['summary_prompt_raw'] = _raw_summary_prompt(
          agent_kind, goal, action, reason, raw_before, raw_after
      )

    compiled_before = _compiled_ui_state(before_pixels, before_elements, screen_size)
    compiled_after = _compiled_ui_state(after_pixels, after_elements, screen_size)
    if compiled_before and compiled_after:
      step['summary_prompt_compiled'] = _raw_summary_prompt(
          agent_kind, goal, action, reason, compiled_before, compiled_after
      )

  return step


def _agent_kind(agent_name: Any) -> str:
  if isinstance(agent_name, str) and 'm3a' in agent_name.lower():
    return 'm3a'
  return 't3a'


def _additional_guidelines(episode: Episode) -> list[str] | None:
  guidelines = episode.get('additional_guidelines')
  if isinstance(guidelines, list) and all(
      isinstance(guideline, str) for guideline in guidelines
  ):
    return guidelines
  return None


def _screen_size(
    episode: Episode, episode_data: dict[str, Any]
) -> tuple[int, int]:
  screen_config = episode.get(constants.EpisodeConstants.SCREEN_CONFIG)
  if isinstance(screen_config, dict):
    width = screen_config.get('width')
    height = screen_config.get('height')
    if width and height:
      return (int(width), int(height))

  for key in ('before_screenshot', 'raw_screenshot'):
    value = episode_data.get(key)
    pixels = _get_step_value(value, 0)
    if isinstance(pixels, np.ndarray) and pixels.ndim >= 2:
      return (int(pixels.shape[1]), int(pixels.shape[0]))

  return (1080, 2400)


def _step_count(episode_data: dict[str, Any]) -> int:
  lengths = []
  for value in episode_data.values():
    if _is_step_sequence(value):
      lengths.append(len(value))
  return max(lengths, default=0)


def _is_step_sequence(value: Any) -> bool:
  if isinstance(value, (str, bytes, dict)):
    return False
  return isinstance(value, Sequence)


def _get_step_value(value: Any, step_index: int) -> Any:
  if _is_step_sequence(value) and step_index < len(value):
    return value[step_index]
  return None


def _before_elements(
    episode_data: dict[str, Any], step_index: int, agent_kind: str
) -> Any:
  if agent_kind == 'm3a':
    return _get_step_value(episode_data.get('before_ui_elements'), step_index)
  return _get_step_value(episode_data.get('before_element_list'), step_index)


def _after_elements(
    episode_data: dict[str, Any], step_index: int, agent_kind: str
) -> Any:
  if agent_kind == 'm3a':
    return _get_step_value(episode_data.get('after_ui_elements'), step_index)
  return _get_step_value(episode_data.get('after_element_list'), step_index)


def _before_pixels(
    episode_data: dict[str, Any], step_index: int, agent_kind: str
) -> Any:
  key = 'raw_screenshot' if agent_kind == 'm3a' else 'before_screenshot'
  return _get_step_value(episode_data.get(key), step_index)


def _after_pixels(
    episode_data: dict[str, Any], step_index: int, agent_kind: str
) -> Any:
  if agent_kind == 'm3a':
    after_pixels = _get_step_value(episode_data.get('after_screenshot'), step_index)
    if after_pixels is not None:
      return after_pixels
    return _get_step_value(episode_data.get('after_screenshot_with_som'), step_index)
  return _get_step_value(episode_data.get('after_screenshot'), step_index)


def _raw_ui_state(
    agent_kind: str,
    ui_elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
) -> str:
  try:
    if agent_kind == 'm3a':
      from android_world.agents import m3a as m3a_agent  # pylint: disable=g-import-not-at-top

      return m3a_agent._generate_ui_elements_description_list(  # pylint: disable=protected-access
          ui_elements, screen_size
      )
    from android_world.agents import t3a as t3a_agent  # pylint: disable=g-import-not-at-top

    return t3a_agent._generate_ui_elements_description_list_full(  # pylint: disable=protected-access
        ui_elements, screen_size
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return ''


def _compiled_ui_state(
    pixels: np.ndarray,
    ui_elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
) -> str:
  if pixels is None:
    return ''
  try:
    state = interface.State(pixels=pixels, forest=None, ui_elements=ui_elements)
    screen_ir = _screen_ir_from_ui_elements(state.ui_elements, screen_size)
    return compiler.UiStateCompiler().compile_screen(screen_ir).prompt
  except Exception:  # pylint: disable=broad-exception-caught
    return ''


def _screen_ir_from_ui_elements(
    ui_elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
) -> ir.ScreenIR:
  surface_id = 's0'
  elements = {}
  for index, ui_element in enumerate(ui_elements):
    element_id = f'e{index}'
    text = ui_element.text or ''
    description = ui_element.content_description or ''
    hint = ui_element.hint_text or ''
    role = _role_from_ui_element(ui_element)
    label = text or description or hint or _resource_key(ui_element) or role
    elements[element_id] = ir.ElementIR(
        id=element_id,
        surface_id=surface_id,
        parent_id=None,
        child_ids=[],
        role=role,
        label=label,
        bounds=_bounds(ui_element),
        state=_state_from_ui_element(ui_element),
        ops=_ops_from_ui_element(ui_element),
        text=text,
        description=description,
        hint=hint,
        resource_key=_resource_key(ui_element),
        source_ref=ir.SourceRef(
            source_type='ui_elements',
            source_id=f'element:{index}',
            attrs={'index': index},
        ),
    )
  return ir.ScreenIR(
      version='screen_ir_v1',
      app_name='',
      activity='',
      surfaces={
          surface_id: ir.SurfaceIR(
              id=surface_id,
              kind='app',
              bounds=[0, 0, int(screen_size[0]), int(screen_size[1])],
              root_ids=list(elements.keys()),
          )
      },
      elements=elements,
      metadata={'source': 'prompt_compare_export'},
  )


def _bounds(ui_element: representation_utils.UIElement) -> list[int]:
  bbox = ui_element.bbox_pixels
  if bbox is None:
    return [0, 0, 0, 0]
  return [
      int(bbox.x_min),
      int(bbox.y_min),
      int(bbox.x_max),
      int(bbox.y_max),
  ]


def _resource_key(ui_element: representation_utils.UIElement) -> str:
  resource_name = ui_element.resource_name or ui_element.resource_id or ''
  package_name = ui_element.package_name or ''
  if package_name and resource_name.startswith(f'{package_name}:id/'):
    return resource_name.removeprefix(f'{package_name}:id/')
  if ':id/' in resource_name:
    return resource_name.split(':id/', maxsplit=1)[1]
  return resource_name


def _role_from_ui_element(ui_element: representation_utils.UIElement) -> str:
  class_name = (ui_element.class_name or '').lower()
  if ui_element.is_editable:
    return 'input'
  if ui_element.is_scrollable:
    return 'scroll'
  if 'button' in class_name:
    return 'button'
  if 'text' in class_name:
    return 'text'
  if ui_element.is_clickable:
    return 'button'
  if 'image' in class_name:
    return 'image'
  if 'list' in class_name or 'recycler' in class_name:
    return 'list'
  return ui_element.class_name or 'node'


def _state_from_ui_element(
    ui_element: representation_utils.UIElement,
) -> set[str]:
  state = set()
  for name, value in (
      ('checkable', ui_element.is_checkable),
      ('checked', ui_element.is_checked),
      ('enabled', ui_element.is_enabled),
      ('focusable', ui_element.is_focusable),
      ('focused', ui_element.is_focused),
      ('selected', ui_element.is_selected),
      ('visible', ui_element.is_visible),
  ):
    if value:
      state.add(name)
  return state


def _ops_from_ui_element(ui_element: representation_utils.UIElement) -> list[str]:
  ops = []
  if ui_element.is_editable:
    ops.append('input_text')
  if ui_element.is_scrollable:
    ops.append('scroll')
  if ui_element.is_clickable or ui_element.is_checkable:
    ops.append('click')
  if ui_element.is_long_clickable:
    ops.append('long_press')
  return ops


def _raw_action_prompt(
    agent_kind: str,
    goal: str,
    history: list[str],
    raw_ui_state: str,
    additional_guidelines: list[str] | None,
) -> str:
  try:
    if agent_kind == 'm3a':
      from android_world.agents import m3a as m3a_agent  # pylint: disable=g-import-not-at-top

      return m3a_agent._action_selection_prompt(  # pylint: disable=protected-access
          goal, history, raw_ui_state, additional_guidelines
      )
    from android_world.agents import t3a as t3a_agent  # pylint: disable=g-import-not-at-top

    return t3a_agent._action_selection_prompt(  # pylint: disable=protected-access
        goal, history, raw_ui_state, additional_guidelines
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return ''


def _compiled_action_prompt(
    agent_kind: str,
    goal: str,
    history: list[str],
    compiled_ui_state: str,
    additional_guidelines: list[str] | None,
) -> str:
  try:
    if agent_kind == 'm3a':
      from android_world.agents import m3a as m3a_agent  # pylint: disable=g-import-not-at-top

      return m3a_agent._compiled_action_selection_prompt(  # pylint: disable=protected-access
          goal, history, compiled_ui_state, additional_guidelines
      )
    from android_world.agents import t3a as t3a_agent  # pylint: disable=g-import-not-at-top

    return t3a_agent._compiled_action_selection_prompt(  # pylint: disable=protected-access
        goal, history, compiled_ui_state, additional_guidelines
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return ''


def _raw_summary_prompt(
    agent_kind: str,
    goal: str,
    action: str,
    reason: str,
    before_ui_state: str,
    after_ui_state: str,
) -> str:
  try:
    if agent_kind == 'm3a':
      from android_world.agents import m3a as m3a_agent  # pylint: disable=g-import-not-at-top

      return m3a_agent._summarize_prompt(  # pylint: disable=protected-access
          action, reason, goal, before_ui_state, after_ui_state
      )
    from android_world.agents import t3a as t3a_agent  # pylint: disable=g-import-not-at-top

    return t3a_agent._summarize_prompt(  # pylint: disable=protected-access
        goal, action, reason, before_ui_state, after_ui_state
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return ''


def _reason_action(
    episode_data: dict[str, Any], step_index: int
) -> tuple[str, str]:
  action_output = _get_step_value(episode_data.get('action_output'), step_index)
  if not isinstance(action_output, str):
    return '', ''
  try:
    reason, action = m3a_utils.parse_reason_action_output(action_output)
  except Exception:  # pylint: disable=broad-exception-caught
    return '', ''
  return _string_or_empty(reason), _string_or_empty(action)


def _string_or_empty(value: Any) -> str:
  return value if isinstance(value, str) else ''
