#!/usr/bin/env python3
"""Exports Android World checkpoint files in a readable format."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw


IMAGE_KEYS = {
    'raw_screenshot',
    'pixels',
    'screenshot',
    'before_screenshot',
    'after_screenshot',
    'before_screenshot_with_som',
    'after_screenshot_with_som',
}


def load_tokenizer(model_or_path: str | None) -> Any:
  if not model_or_path:
    return None
  try:
    from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise RuntimeError(
        'The --tokenizer option requires transformers in the active environment.'
    ) from exc
  return AutoTokenizer.from_pretrained(model_or_path)


def _token_count(tokenizer: Any, text: str) -> int | None:
  if tokenizer is None:
    return None
  return len(tokenizer(text, add_special_tokens=False)['input_ids'])


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


def _parse_action_payload(value: Any) -> dict[str, Any] | None:
  if isinstance(value, dict):
    return value
  if not isinstance(value, str):
    return None
  match = re.search(r'Action:\s*(\{.*?\})\s*$', value, flags=re.DOTALL)
  if not match:
    match = re.search(r'(\{[^{}]*"action_type"[^{}]*\})', value, flags=re.DOTALL)
  if not match:
    return None
  try:
    parsed = json.loads(match.group(1))
  except json.JSONDecodeError:
    return None
  return parsed if isinstance(parsed, dict) else None


def _selected_action_payload(step: dict[str, Any]) -> dict[str, Any] | None:
  payload = _parse_action_payload(step.get('action_output_json'))
  if payload is not None:
    return payload
  return _parse_action_payload(step.get('action_output'))


def _source_attrs(action: dict[str, Any]) -> dict[str, Any]:
  source_ref = action.get('source_ref')
  if isinstance(source_ref, dict):
    attrs = source_ref.get('attrs')
    if isinstance(attrs, dict):
      return attrs
  return {}


def _compact_action(action_id: str, action: dict[str, Any]) -> dict[str, Any]:
  attrs = _source_attrs(action)
  return {
      'id': action_id,
      'op': action.get('op'),
      'element_id': action.get('element_id'),
      'label': action.get('label'),
      'role': action.get('role'),
      'bounds': action.get('bounds'),
      'enabled': action.get('enabled'),
      'state': action.get('state'),
      'class_name': attrs.get('class_name'),
      'resource_name': attrs.get('resource_name'),
      'package_name': attrs.get('package_name'),
      'source_id': (
          action.get('source_ref', {}).get('source_id')
          if isinstance(action.get('source_ref'), dict)
          else None
      ),
  }


def _action_debug_line(action_id: str, action: dict[str, Any]) -> str:
  compact = _compact_action(action_id, action)
  for key in ('resource_name', 'class_name', 'package_name', 'source_id'):
    if compact.get(key) is None and action.get(key) is not None:
      compact[key] = action.get(key)
  bounds = compact.get('bounds') or []
  bounds_text = ','.join(str(value) for value in bounds)
  pieces = [
      f'{compact["id"]}',
      str(compact.get('op') or ''),
      str(compact.get('role') or ''),
      f'"{compact.get("label") or ""}"',
  ]
  if compact.get('resource_name'):
    pieces.append(f'resource={compact["resource_name"]}')
  if compact.get('class_name'):
    pieces.append(f'class={compact["class_name"]}')
  if bounds_text:
    pieces.append(f'bounds=[{bounds_text}]')
  return ' '.join(piece for piece in pieces if piece)


def _add_action_debug(step: dict[str, Any]) -> None:
  action_map = step.get('before_action_map')
  if not isinstance(action_map, dict):
    return
  step['action_map_summary'] = [
      _action_debug_line(action_id, action)
      for action_id, action in sorted(action_map.items())
      if isinstance(action, dict)
  ]
  action_payload = _selected_action_payload(step)
  if action_payload is not None:
    step['selected_action_payload'] = action_payload
  target = action_payload.get('target') if isinstance(action_payload, dict) else None
  if isinstance(target, str) and isinstance(action_map.get(target), dict):
    step['selected_target'] = _compact_action(target, action_map[target])


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    bounds: list[int],
    *,
    outline: tuple[int, int, int],
    width: int,
    label: str,
) -> None:
  if len(bounds) != 4:
    return
  x0, y0, x1, y1 = [int(value) for value in bounds]
  draw.rectangle((x0, y0, x1, y1), outline=outline, width=width)
  text_y = max(0, y0 - 12)
  draw.rectangle((x0, text_y, min(x0 + 8 * len(label) + 6, x1 + 120), text_y + 12), fill=outline)
  draw.text((x0 + 3, text_y), label, fill=(255, 255, 255))


def _save_action_overlay(
    screenshot: np.ndarray,
    action_map: dict[str, Any],
    selected_target_id: str | None,
    resolved_action: dict[str, Any] | None,
    assets_dir: Path,
    name: str,
) -> str:
  assets_dir.mkdir(parents=True, exist_ok=True)
  image = Image.fromarray(screenshot).convert('RGB')
  draw = ImageDraw.Draw(image)
  for action_id, action in sorted(action_map.items()):
    if not isinstance(action, dict):
      continue
    bounds = action.get('bounds')
    if isinstance(bounds, list):
      color = (52, 120, 246)
      width = 2
      if action_id == selected_target_id:
        color = (220, 38, 38)
        width = 4
      _draw_rect(draw, bounds, outline=color, width=width, label=action_id)
  if isinstance(resolved_action, dict):
    bounds = resolved_action.get('target_bounds')
    if isinstance(bounds, list):
      _draw_rect(draw, bounds, outline=(22, 163, 74), width=2, label='resolved')
    x = resolved_action.get('x')
    y = resolved_action.get('y')
    if x is not None and y is not None:
      x = int(x)
      y = int(y)
      draw.line((x - 8, y, x + 8, y), fill=(22, 163, 74), width=2)
      draw.line((x, y - 8, x, y + 8), fill=(22, 163, 74), width=2)
  path = assets_dir / f'{name}.png'
  image.save(path)
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


def _model_name_from_raw_response(value: Any) -> str | None:
  if value is None:
    return None
  model = getattr(value, 'model', None)
  if model:
    return str(model)
  if isinstance(value, dict) and value.get('model'):
    return str(value['model'])
  if hasattr(value, 'model_dump') and callable(value.model_dump):
    try:
      dumped = value.model_dump()
    except Exception:  # pylint: disable=broad-exception-caught
      dumped = None
    if isinstance(dumped, dict) and dumped.get('model'):
      return str(dumped['model'])
  return None


def _episode_backend_model(episode_data: dict[str, Any]) -> str | None:
  for key in ('action_raw_response', 'summary_raw_response'):
    values = episode_data.get(key)
    if not isinstance(values, list):
      continue
    for value in values:
      model = _model_name_from_raw_response(value)
      if model:
        return model
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
  converted['backend_model_name'] = _episode_backend_model(episode_data)

  steps = []
  step_count = _episode_step_count(episode_data)
  for step_idx in range(step_count):
    step = {}
    raw_step = {}
    for key, values in episode_data.items():
      value = _step_value(values, step_idx)
      raw_step[key] = value
      if key in IMAGE_KEYS and _is_image(value):
        filename = _save_image(
            value,
            assets_dir,
            f'episode_{episode_idx:03d}_step_{step_idx:03d}_{key}',
        )
        step[key] = f'{assets_dir.name}/{filename}'
      else:
        step[key] = _jsonable(value)
    _add_action_debug(step)
    before_screenshot = raw_step.get('before_screenshot')
    action_map = step.get('before_action_map')
    selected = step.get('selected_action_payload')
    selected_target_id = (
        selected.get('target') if isinstance(selected, dict) else None
    )
    resolved_action = step.get('resolved_action')
    if (
        _is_image(before_screenshot)
        and isinstance(action_map, dict)
        and action_map
    ):
      filename = _save_action_overlay(
          before_screenshot,
          action_map,
          selected_target_id if isinstance(selected_target_id, str) else None,
          resolved_action if isinstance(resolved_action, dict) else None,
          assets_dir,
          f'episode_{episode_idx:03d}_step_{step_idx:03d}_before_actions_overlay',
      )
      step['before_actions_overlay'] = f'{assets_dir.name}/{filename}'
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


def _full_text(value: Any) -> str:
  if value is None:
    return ''
  if isinstance(value, str):
    return value
  return json.dumps(_jsonable(value), ensure_ascii=False, indent=2)


def _result_label(value: Any) -> str:
  try:
    return 'success' if float(value) > 0.5 else 'fail'
  except (TypeError, ValueError):
    return 'unknown'


def _image_link(
    lines: list[str],
    label: str,
    image_path: str | None,
    episode_idx: int,
    step_idx: int,
) -> None:
  lines.extend([f'*{label}*:', ''])
  if image_path:
    lines.extend([
        f'![episode {episode_idx} step {step_idx} {label}]({image_path})',
        '',
    ])
  else:
    lines.extend(['not available', ''])


def _first_image_path(step: dict[str, Any], keys: tuple[str, ...]) -> str | None:
  for key in keys:
    value = step.get(key)
    if isinstance(value, str) and value:
      return value
  return None


def _add_prompt_block(
    lines: list[str],
    title: str,
    prompt: str,
    *,
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
) -> None:
  lines.extend([f'### {title}', ''])
  if tokenizer is not None and prompt:
    tokens = _token_count(tokenizer, prompt)
    lines.extend([
        f'- tokens: {tokens}',
        f'- tokenizer: {tokenizer_name}',
        '',
    ])
  lines.append('```text')
  lines.append(prompt if prompt else 'not available')
  lines.extend(['```', ''])


def _prompt_compare(step: dict[str, Any]) -> dict[str, Any]:
  prompt_compare = step.get('prompt_compare')
  return prompt_compare if isinstance(prompt_compare, dict) else {}


def _actual_prompt_mode(step: dict[str, Any]) -> str:
  prompt_compare = _prompt_compare(step)
  mode = prompt_compare.get('actual_mode') or step.get('ui_state_mode')
  return str(mode) if mode else ''


def _alternative_prompt(step: dict[str, Any], key: str) -> str:
  prompt_compare = _prompt_compare(step)
  alternatives = prompt_compare.get('alternative_prompts')
  if isinstance(alternatives, dict) and alternatives.get(key):
    return _full_text(alternatives[key])

  # Compatibility for older readable JSONs that used flattened fields.
  legacy_key = {
      'raw': 'raw_action_prompt',
      'compiled': 'compiled_action_prompt',
  }.get(key)
  return _full_text(step.get(legacy_key)) if legacy_key else ''


def _write_markdown(
    episodes: list[dict[str, Any]],
    output_path: Path,
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
) -> None:
  lines = [f'# {output_path.stem} Summary', '']
  for episode_idx, episode in enumerate(episodes):
    agent = _short(episode.get('agent_name')) or 'unknown_agent'
    model = _short(episode.get('backend_model_name')) or 'unknown_model'
    lines.extend([
        f'- Goal: {_short(episode.get("goal"))}',
        f'- Task: {_short(episode.get("task_template"))}',
        f'- Agent: {agent} | {model}',
        f'- Result: {_result_label(episode.get("is_successful"))}',
        f'- Runtime: {_short(episode.get("run_time"))}',
        '',
    ])

    episode_data = episode.get('episode_data') or {}
    steps = episode_data.get('steps', []) if isinstance(episode_data, dict) else []
    for step_idx, step in enumerate(steps):
      lines.extend([f'## Step {step_idx}', ''])

      lines.extend(['### action_map', '', '```text'])
      action_summary = step.get('action_map_summary')
      if isinstance(action_summary, list) and action_summary:
        lines.extend(str(line) for line in action_summary)
      else:
        lines.append('not available')
      lines.extend(['```', ''])
      overlay_path = step.get('before_actions_overlay')
      if isinstance(overlay_path, str) and overlay_path:
        _image_link(lines, 'overlay', overlay_path, episode_idx, step_idx)

      prompt = _full_text(step.get('action_prompt'))
      actual_mode = _actual_prompt_mode(step)
      raw_prompt = (
          prompt if actual_mode == 'legacy' else _alternative_prompt(step, 'raw')
      )
      compiled_prompt = (
          prompt
          if actual_mode == 'compiled'
          else _alternative_prompt(step, 'compiled')
      )
      _add_prompt_block(
          lines,
          'prompt sent to model',
          prompt,
          tokenizer=tokenizer,
          tokenizer_name=tokenizer_name,
      )
      _add_prompt_block(
          lines,
          'raw prompt',
          raw_prompt,
          tokenizer=tokenizer,
          tokenizer_name=tokenizer_name,
      )
      _add_prompt_block(
          lines,
          'compiled prompt',
          compiled_prompt,
          tokenizer=tokenizer,
          tokenizer_name=tokenizer_name,
      )

      lines.extend(['### response', '', '```text'])
      response = _full_text(step.get('action_output'))
      lines.append(response if response else 'not available')
      lines.extend(['```', ''])

      if step.get('selected_target'):
        lines.extend(['### selected target', '', '```text'])
        lines.append(
            _action_debug_line(
                str(step['selected_target'].get('id')),
                step['selected_target'],
            )
        )
        lines.extend(['```', ''])
      if step.get('resolved_action'):
        lines.extend(['### resolved action', '', '```json'])
        lines.append(json.dumps(step['resolved_action'], ensure_ascii=False, indent=2))
        lines.extend(['```', ''])

      before_raw = _first_image_path(
          step,
          ('before_screenshot', 'raw_screenshot', 'before_screenshot_with_som'),
      )
      lines.extend(['### before screenshot', ''])
      _image_link(
          lines,
          'raw',
          before_raw,
          episode_idx,
          step_idx,
      )

      after_raw = _first_image_path(
          step,
          ('after_screenshot', 'after_screenshot_with_som'),
      )
      lines.extend(['### after screenshot', ''])
      _image_link(lines, 'raw', after_raw, episode_idx, step_idx)

  output_path.write_text('\n'.join(lines), encoding='utf-8')


def export_file(
    path: Path,
    output_dir: Path | None = None,
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
) -> None:
  with gzip.open(path, 'rb') as f:
    episodes = pickle.load(f)
  if not isinstance(episodes, list):
    episodes = [episodes]

  output_base_dir = output_dir or path.parent
  output_base_dir.mkdir(parents=True, exist_ok=True)
  stem = path.name.removesuffix('.pkl.gz')
  assets_dir = output_base_dir / f'{stem}_assets'
  converted = [
      _convert_episode(episode, i, assets_dir)
      for i, episode in enumerate(episodes)
  ]

  json_path = output_base_dir / f'{stem}.readable.json'
  json_path.write_text(
      json.dumps(converted, ensure_ascii=False, indent=2),
      encoding='utf-8',
  )

  md_path = output_base_dir / f'{stem}.summary.md'
  _write_markdown(
      converted,
      md_path,
      tokenizer=tokenizer,
      tokenizer_name=tokenizer_name,
  )
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
  parser.add_argument(
      '--output-dir',
      type=Path,
      default=None,
      help='Directory to write readable exports. Defaults next to input.',
  )
  parser.add_argument(
      '--tokenizer',
      default=None,
      help='Optional Hugging Face tokenizer model/path for action prompt tokens.',
  )
  args = parser.parse_args()
  tokenizer = load_tokenizer(args.tokenizer)

  for input_path in _iter_inputs(args.path):
    export_file(
        input_path,
        args.output_dir,
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
    )


if __name__ == '__main__':
  main()
