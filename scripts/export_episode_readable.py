#!/usr/bin/env python3
"""Exports Android World checkpoint files in a readable format."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from PIL import Image


IMAGE_KEYS = {
    'raw_screenshot',
    'pixels',
    'screenshot',
    'before_screenshot',
    'after_screenshot',
    'before_screenshot_with_som',
    'after_screenshot_with_som',
}


def _is_image(value: Any) -> bool:
  return (
      isinstance(value, np.ndarray)
      and value.ndim in (2, 3)
      and value.dtype == np.uint8
  )


def _jsonable(value: Any) -> Any:
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return {
        'type': 'ndarray',
        'shape': list(value.shape),
        'dtype': str(value.dtype),
    }
  if dataclasses.is_dataclass(value):
    return _jsonable(dataclasses.asdict(value))
  if hasattr(value, 'as_dict') and callable(value.as_dict):
    try:
      return _jsonable(value.as_dict())
    except TypeError:
      pass
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  return repr(value)


def _save_image(value: np.ndarray, assets_dir: Path, name: str) -> str:
  assets_dir.mkdir(parents=True, exist_ok=True)
  path = assets_dir / f'{name}.png'
  Image.fromarray(value).save(path)
  return path.name


def _episode_step_count(episode_data: dict[str, Any]) -> int:
  for values in episode_data.values():
    if isinstance(values, list):
      return len(values)
  return 0


def _step_value(values: Any, step_idx: int) -> Any:
  if isinstance(values, list) and step_idx < len(values):
    return values[step_idx]
  return None


def _convert_episode(
    episode: dict[str, Any],
    episode_idx: int,
    assets_dir: Path,
) -> dict[str, Any]:
  converted = {}
  for key, value in episode.items():
    if key == 'episode_data':
      continue
    converted[key] = _jsonable(value)

  episode_data = episode.get('episode_data') or {}
  if not isinstance(episode_data, dict):
    converted['episode_data'] = _jsonable(episode_data)
    return converted

  steps = []
  step_count = _episode_step_count(episode_data)
  for step_idx in range(step_count):
    step = {}
    for key, values in episode_data.items():
      value = _step_value(values, step_idx)
      if key in IMAGE_KEYS and _is_image(value):
        filename = _save_image(
            value,
            assets_dir,
            f'episode_{episode_idx:03d}_step_{step_idx:03d}_{key}',
        )
        step[key] = f'{assets_dir.name}/{filename}'
      else:
        step[key] = _jsonable(value)
    steps.append(step)

  converted['episode_data'] = {
      'num_steps': step_count,
      'steps': steps,
  }
  return converted


def _short(value: Any, max_len: int = 500) -> str:
  if value is None:
    return ''
  if not isinstance(value, str):
    value = json.dumps(_jsonable(value), ensure_ascii=False)
  value = value.strip()
  if len(value) <= max_len:
    return value
  return value[:max_len] + '...'


def _write_markdown(
    episodes: list[dict[str, Any]],
    output_path: Path,
) -> None:
  lines = [f'# {output_path.stem}', '']
  for episode_idx, episode in enumerate(episodes):
    lines.extend([
        f'## Episode {episode_idx}',
        '',
        f'- Goal: {_short(episode.get("goal"))}',
        f'- Task: {_short(episode.get("task_template"))}',
        f'- Agent: {_short(episode.get("agent_name"))}',
        f'- Success: {_short(episode.get("is_successful"))}',
        f'- Episode length: {_short(episode.get("episode_length"))}',
        f'- Runtime: {_short(episode.get("run_time"))}',
        '',
    ])

    episode_data = episode.get('episode_data') or {}
    steps = episode_data.get('steps', []) if isinstance(episode_data, dict) else []
    for step_idx, step in enumerate(steps):
      lines.extend([f'### Step {step_idx}', ''])
      for image_key in sorted(IMAGE_KEYS):
        image_path = step.get(image_key)
        if image_path:
          lines.extend([
              f'**{image_key}**',
              '',
              f'![episode {episode_idx} step {step_idx} {image_key}]({image_path})',
              '',
          ])
      for key in (
          'action',
          'action_output_json',
          'action_output',
          'action_reason',
          'summary',
          'action_description',
          'step_number',
      ):
        if key in step and step[key] not in (None, ''):
          lines.extend([f'- {key}:', '', '```text', _short(step[key]), '```', ''])

      ui_keys = [
          key for key in step
          if key.endswith('ui_elements')
          or key.endswith('element_list')
          or key == 'ui_elements'
          or key == 'elements'
      ]
      for key in ui_keys:
        value = step.get(key)
        if isinstance(value, list):
          lines.append(f'- {key}: {len(value)} items')
      lines.append('')

  output_path.write_text('\n'.join(lines), encoding='utf-8')


def export_file(path: Path) -> None:
  with gzip.open(path, 'rb') as f:
    episodes = pickle.load(f)
  if not isinstance(episodes, list):
    episodes = [episodes]

  assets_dir = path.with_suffix('').with_suffix('').with_name(
      path.name.removesuffix('.pkl.gz') + '_assets'
  )
  converted = [
      _convert_episode(episode, i, assets_dir)
      for i, episode in enumerate(episodes)
  ]

  json_path = path.with_name(path.name.removesuffix('.pkl.gz') + '.readable.json')
  json_path.write_text(
      json.dumps(converted, ensure_ascii=False, indent=2),
      encoding='utf-8',
  )

  md_path = path.with_name(path.name.removesuffix('.pkl.gz') + '.summary.md')
  _write_markdown(converted, md_path)
  print(f'wrote {json_path}')
  print(f'wrote {md_path}')
  if assets_dir.exists():
    print(f'wrote assets under {assets_dir}')


def _iter_inputs(path: Path) -> list[Path]:
  if path.is_dir():
    return sorted(path.glob('*.pkl.gz'))
  return [path]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'path',
      type=Path,
      help='Path to a .pkl.gz checkpoint file or a directory containing them.',
  )
  args = parser.parse_args()

  for input_path in _iter_inputs(args.path):
    export_file(input_path)


if __name__ == '__main__':
  main()
