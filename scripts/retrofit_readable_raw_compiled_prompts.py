#!/usr/bin/env python3
"""Adds raw/compiled prompt fields to existing readable episode JSON files."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from android_world.agents import m3a
from android_world.agents import t3a
from android_world.env import representation_utils


def _bbox(value: Any) -> representation_utils.BoundingBox | None:
  if not isinstance(value, dict):
    return None
  keys = ('x_min', 'x_max', 'y_min', 'y_max')
  if not all(key in value for key in keys):
    return None
  return representation_utils.BoundingBox(
      value['x_min'], value['x_max'], value['y_min'], value['y_max']
  )


def _ui_element(value: Any) -> representation_utils.UIElement | None:
  if not isinstance(value, dict):
    return None
  fields = {field.name for field in dataclasses.fields(representation_utils.UIElement)}
  kwargs = {key: value.get(key) for key in fields if key in value}
  kwargs['bbox'] = _bbox(value.get('bbox'))
  kwargs['bbox_pixels'] = _bbox(value.get('bbox_pixels'))
  return representation_utils.UIElement(**kwargs)


def _ui_elements(values: Any) -> list[representation_utils.UIElement]:
  if not isinstance(values, list):
    return []
  return [
      element
      for value in values
      if (element := _ui_element(value)) is not None
  ]


def _infer_screen_size(
    elements: list[representation_utils.UIElement],
) -> tuple[int, int]:
  max_x = 0
  max_y = 0
  for element in elements:
    if element.bbox_pixels is None:
      continue
    max_x = max(max_x, int(element.bbox_pixels.x_max))
    max_y = max(max_y, int(element.bbox_pixels.y_max))
  return max_x or 320, max_y or 640


def _agent_family(agent_name: str) -> str:
  normalized = agent_name.lower()
  if normalized.startswith('m3a') or '_m3a' in normalized:
    return 'm3a'
  return 't3a'


def _history(agent_family: str, summaries: list[str]) -> list[str]:
  if agent_family == 'm3a':
    return [f'Step {index + 1}- {summary}' for index, summary in enumerate(summaries)]
  return [f'Step {index + 1}: {summary}' for index, summary in enumerate(summaries)]


def _raw_ui_state_text(
    agent_family: str,
    elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
) -> str:
  if agent_family == 'm3a':
    return m3a._generate_ui_elements_description_list(elements, screen_size)  # pylint: disable=protected-access
  return t3a._generate_ui_elements_description_list_full(elements, screen_size)  # pylint: disable=protected-access


def _raw_action_prompt(
    agent_family: str,
    *,
    goal: str,
    summaries: list[str],
    raw_ui_state_text: str,
) -> str:
  history = _history(agent_family, summaries)
  if agent_family == 'm3a':
    return m3a._action_selection_prompt(goal, history, raw_ui_state_text)  # pylint: disable=protected-access
  return t3a._action_selection_prompt(goal, history, raw_ui_state_text)  # pylint: disable=protected-access


def _ensure_prompt_compare(step: dict[str, Any], actual: str) -> dict[str, Any]:
  prompt_compare = step.get('prompt_compare')
  if not isinstance(prompt_compare, dict):
    prompt_compare = {}
    step['prompt_compare'] = prompt_compare
  prompt_compare.setdefault('actual_mode', actual)
  prompt_compare.setdefault('actual_prompt_field', 'action_prompt')
  prompt_compare.setdefault('alternative_prompts', {})
  return prompt_compare


def _retrofit_episode(episode: dict[str, Any]) -> dict[str, Any]:
  agent_family = _agent_family(str(episode.get('agent_name') or ''))
  goal = str(episode.get('goal') or '')
  episode_data = episode.get('episode_data')
  if not isinstance(episode_data, dict):
    episode['retrofit_status'] = 'incomplete'
    episode['retrofit_errors'] = ['missing episode_data']
    return episode
  steps = episode_data.get('steps')
  if not isinstance(steps, list):
    episode['retrofit_status'] = 'incomplete'
    episode['retrofit_errors'] = ['missing episode_data.steps']
    return episode

  errors = []
  summaries: list[str] = []
  for step_index, step in enumerate(steps):
    if not isinstance(step, dict):
      errors.append(f'step {step_index}: not an object')
      continue
    raw_elements_value = step.get('before_element_list')
    if raw_elements_value is None:
      raw_elements_value = step.get('before_ui_elements')
    elements = _ui_elements(raw_elements_value)
    if not goal:
      errors.append(f'step {step_index}: missing goal')
    if not elements:
      errors.append(
          f'step {step_index}: missing before_element_list/before_ui_elements'
      )
    if goal and elements and step.get('ui_state_mode') == 'compiled':
      screen_size = _infer_screen_size(elements)
      raw_ui_state = _raw_ui_state_text(agent_family, elements, screen_size)
      prompt_compare = _ensure_prompt_compare(
          step, str(step.get('ui_state_mode') or 'legacy')
      )
      alternatives = prompt_compare.get('alternative_prompts')
      if isinstance(alternatives, dict):
        alternatives.setdefault(
            'raw',
            _raw_action_prompt(
              agent_family,
              goal=goal,
              summaries=summaries,
              raw_ui_state_text=raw_ui_state,
            ),
        )

    if step.get('ui_state_mode') == 'compiled':
      _ensure_prompt_compare(step, 'compiled')
    elif step.get('action_prompt'):
      _ensure_prompt_compare(step, 'legacy')

    summary = step.get('summary')
    summaries.append(str(summary) if summary is not None else '')

  episode['retrofit_status'] = 'ok' if not errors else 'incomplete'
  if errors:
    episode['retrofit_errors'] = errors
  else:
    episode.pop('retrofit_errors', None)
  return episode


def retrofit_value(value: Any) -> Any:
  if isinstance(value, list):
    return [
        _retrofit_episode(episode) if isinstance(episode, dict) else episode
        for episode in value
    ]
  if isinstance(value, dict):
    return _retrofit_episode(value)
  return value


def _input_files(path: Path) -> list[Path]:
  if path.is_dir():
    return sorted(path.rglob('*.readable.json'))
  return [path]


def _output_path(input_path: Path, root: Path, output_dir: Path | None) -> Path:
  if output_dir is not None:
    if root.is_dir():
      return output_dir / input_path.relative_to(root)
    return output_dir / input_path.name
  if root.is_dir():
    return root / 'retrofitted' / input_path.relative_to(root)
  return input_path.with_name(
      input_path.name.removesuffix('.readable.json')
      + '.with_prompts.readable.json'
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Retrofit readable JSON files with raw/compiled prompt fields.'
  )
  parser.add_argument('path', type=Path, help='A .readable.json file or directory.')
  parser.add_argument(
      '--output-dir',
      type=Path,
      default=None,
      help='Output directory. Defaults to non-overwriting paths next to input.',
  )
  args = parser.parse_args()

  input_files = _input_files(args.path)
  if not input_files:
    raise FileNotFoundError(f'No .readable.json files found under {args.path}')
  for input_path in input_files:
    value = json.loads(input_path.read_text(encoding='utf-8'))
    output_value = retrofit_value(value)
    output_path = _output_path(input_path, args.path, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'wrote {output_path}')


if __name__ == '__main__':
  main()
