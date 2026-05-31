#!/usr/bin/env python3
"""Builds candidate UI-state IRs from raw Android accessibility forests."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

from google.protobuf import text_format

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from android_env.proto.a11y import android_accessibility_forest_pb2
from android_world.env import representation_utils


SYSTEM_UI_PACKAGE = 'com.android.systemui'
DEFAULT_INPUT_DIR = Path('data/A11y')
DEFAULT_OUTPUT_DIR = Path('data/ui_state_ir_candidates')


def _read_forest(path: Path):
  forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
  text_format.Parse(path.read_text(encoding='utf-8'), forest)
  return forest


def _read_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    return {}
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError:
    return {}
  return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
      encoding='utf-8',
  )


def _write_text(path: Path, value: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(value, encoding='utf-8')


def _bbox_to_dict(bbox: Any) -> dict[str, int]:
  return {
      'x_min': int(bbox.x_min),
      'x_max': int(bbox.x_max),
      'y_min': int(bbox.y_min),
      'y_max': int(bbox.y_max),
  }


def _rect_to_list(rect: Any) -> list[int]:
  return [int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)]


def _rect_nonzero(rect: Any) -> bool:
  return any((rect.left, rect.top, rect.right, rect.bottom))


def _strip_none(value: Any) -> Any:
  if isinstance(value, dict):
    stripped_dict = {
        key: stripped
        for key, item in value.items()
        if (stripped := _strip_none(item)) is not None
    }
    return stripped_dict or None
  if isinstance(value, list):
    stripped_list = [
        stripped
        for item in value
        if (stripped := _strip_none(item)) is not None
    ]
    return stripped_list or None
  if value == '':
    return None
  return value


def _short_class_name(class_name: str) -> str:
  if not class_name:
    return ''
  for prefix in ('android.widget.', 'android.view.'):
    if class_name.startswith(prefix):
      return class_name.removeprefix(prefix)
  return class_name.rsplit('.', maxsplit=1)[-1]


def _short_resource_name(resource_name: str, package_name: str) -> str:
  if not resource_name:
    return ''
  if package_name and resource_name.startswith(f'{package_name}:id/'):
    return resource_name.removeprefix(f'{package_name}:id/')
  if ':id/' in resource_name:
    return resource_name.split(':id/', maxsplit=1)[1]
  return resource_name


def _window_type_name(window: Any) -> str:
  enum_desc = window.DESCRIPTOR.fields_by_name['window_type'].enum_type
  value = enum_desc.values_by_number.get(window.window_type)
  return value.name if value is not None else str(window.window_type)


def _node_flags(node: Any) -> list[str]:
  flags = []
  mapping = (
      ('checkable', node.is_checkable),
      ('checked', node.is_checked),
      ('clickable', node.is_clickable),
      ('editable', node.is_editable),
      ('enabled', node.is_enabled),
      ('focusable', node.is_focusable),
      ('focused', node.is_focused),
      ('long_clickable', node.is_long_clickable),
      ('password', node.is_password),
      ('scrollable', node.is_scrollable),
      ('selected', node.is_selected),
      ('visible', node.is_visible_to_user),
  )
  for name, enabled in mapping:
    if enabled:
      flags.append(name)
  return flags


def _node_action_ids(node: Any) -> list[int]:
  return [int(action.id) for action in node.actions if action.id]


def _element_dict(element: representation_utils.UIElement) -> dict[str, Any]:
  value = dataclasses.asdict(element)
  if element.bbox is not None:
    value['bbox'] = _bbox_to_dict(element.bbox)
  if element.bbox_pixels is not None:
    value['bbox_pixels'] = _bbox_to_dict(element.bbox_pixels)
  return _strip_none(value)


def _baseline_ui_elements(forest: Any, include_invisible: bool) -> list[dict[str, Any]]:
  elements = representation_utils.forest_to_ui_elements(
      forest,
      exclude_invisible_elements=not include_invisible,
  )
  return [
      {'index': index, **_element_dict(element)}
      for index, element in enumerate(elements)
  ]


def _baseline_text(elements: list[dict[str, Any]]) -> str:
  lines = ['Here is a list of descriptions for some UI elements on the current screen:']
  for element in elements:
    fields = ', '.join(
        f'{key}={value!r}'
        for key, value in element.items()
        if key != 'index'
    )
    lines.append(f'UI element {element["index"]}: UIElement({fields})')
  return '\n'.join(lines) + '\n'


def _window_packages(window: Any) -> set[str]:
  return {node.package_name for node in window.tree.nodes if node.package_name}


def _is_system_ui_window(window: Any) -> bool:
  packages = _window_packages(window)
  return bool(packages) and packages <= {SYSTEM_UI_PACKAGE}


def _window_header(window: Any, *, compact: bool) -> dict[str, Any]:
  keymap = {
      'id': 'id',
      'type': 'type',
      'bounds': 'bounds',
      'active': 'active',
      'focused': 'focused',
      'package_names': 'packages',
      'nodes': 'nodes',
  }
  if compact:
    keymap = {
        'id': 'id',
        'type': 'ty',
        'bounds': 'b',
        'active': 'act',
        'focused': 'foc',
        'package_names': 'pkgs',
        'nodes': 'ch',
    }
  value = {
      keymap['id']: int(window.id),
      keymap['type']: _window_type_name(window),
      keymap['bounds']: _rect_to_list(window.bounds_in_screen),
      keymap['active']: window.is_active,
      keymap['focused']: window.is_focused,
      keymap['package_names']: sorted(_window_packages(window)),
  }
  return _strip_none(value)


def _node_full(node: Any, children: list[dict[str, Any]]) -> dict[str, Any]:
  value = {
      'id': int(node.unique_id),
      'class_name': node.class_name,
      'text': node.text,
      'content_description': node.content_description,
      'hint_text': node.hint_text,
      'package_name': node.package_name,
      'resource_name': node.view_id_resource_name,
      'bounds': _rect_to_list(node.bounds_in_screen),
      'flags': _node_flags(node),
      'actions': _node_action_ids(node),
      'children': children,
  }
  return _strip_none(value)


def _node_compact(node: Any, children: list[dict[str, Any]]) -> dict[str, Any]:
  value = {
      'id': int(node.unique_id),
      'c': _short_class_name(node.class_name),
      't': node.text,
      'd': node.content_description,
      'h': node.hint_text,
      'pkg': node.package_name,
      'r': _short_resource_name(node.view_id_resource_name, node.package_name),
      'b': _rect_to_list(node.bounds_in_screen),
      'flags': _node_flags(node),
      'ch': children,
  }
  return _strip_none(value)


def _node_is_meaningful(node: Any, include_invisible: bool) -> bool:
  if not include_invisible and not node.is_visible_to_user:
    return False
  return bool(
      node.text
      or node.content_description
      or node.hint_text
      or node.view_id_resource_name
      or node.is_clickable
      or node.is_editable
      or node.is_scrollable
      or node.is_checkable
      or node.is_long_clickable
  )


def _child_map(window: Any) -> dict[int, list[int]]:
  return {
      int(node.unique_id): [int(child_id) for child_id in node.child_ids]
      for node in window.tree.nodes
  }


def _root_ids(window: Any) -> list[int]:
  node_ids = {int(node.unique_id) for node in window.tree.nodes}
  child_ids = {
      int(child_id)
      for node in window.tree.nodes
      for child_id in node.child_ids
  }
  roots = sorted(node_ids - child_ids)
  if roots:
    return roots
  if 0 in node_ids:
    return [0]
  return sorted(node_ids)


def _kept_node_ids(window: Any, include_invisible: bool) -> set[int]:
  nodes = {int(node.unique_id): node for node in window.tree.nodes}
  children = _child_map(window)
  keep: set[int] = {
      node_id
      for node_id, node in nodes.items()
      if _node_is_meaningful(node, include_invisible)
  }

  def visit(node_id: int) -> bool:
    child_kept = False
    for child_id in children.get(node_id, []):
      child_kept = visit(child_id) or child_kept
    if child_kept:
      keep.add(node_id)
    return node_id in keep

  for root_id in _root_ids(window):
    visit(root_id)
  return keep


def _build_tree(
    window: Any,
    *,
    keep_ids: set[int] | None,
    compact: bool,
) -> list[dict[str, Any]]:
  nodes = {int(node.unique_id): node for node in window.tree.nodes}
  children = _child_map(window)

  def build(node_id: int) -> dict[str, Any] | None:
    if keep_ids is not None and node_id not in keep_ids:
      return None
    node = nodes[node_id]
    child_values = [
        built
        for child_id in children.get(node_id, [])
        if (built := build(child_id)) is not None
    ]
    return _node_compact(node, child_values) if compact else _node_full(node, child_values)

  return [
      built
      for root_id in _root_ids(window)
      if (built := build(root_id)) is not None
  ]


def _compact_tree_v0(forest: Any) -> dict[str, Any]:
  windows = []
  for window in forest.windows:
    value = _window_header(window, compact=False)
    value['nodes'] = _build_tree(window, keep_ids=None, compact=False)
    windows.append(value)
  return {'strategy': 'compact_tree_v0', 'windows': windows}


def _compact_tree_v1(
    forest: Any,
    *,
    include_system_ui: bool,
    include_invisible: bool,
) -> dict[str, Any]:
  windows = []
  for window in forest.windows:
    if not include_system_ui and _is_system_ui_window(window):
      continue
    keep_ids = _kept_node_ids(window, include_invisible)
    if not keep_ids:
      keep_ids = set(_root_ids(window))
    nodes = _build_tree(window, keep_ids=keep_ids, compact=True)
    if not nodes:
      continue
    value = _window_header(window, compact=True)
    value['ch'] = nodes
    windows.append(value)
  return {
      'strategy': 'compact_tree_v1',
      'windows': windows,
  }


def _raw_stats(forest: Any, raw_text: str) -> dict[str, Any]:
  windows = list(forest.windows)
  nodes = [node for window in windows for node in window.tree.nodes]
  return {
      'raw_text_chars': len(raw_text),
      'window_count': len(windows),
      'node_count': len(nodes),
      'visible_node_count': sum(1 for node in nodes if node.is_visible_to_user),
      'leaf_node_count': sum(1 for node in nodes if not node.child_ids),
      'text_node_count': sum(1 for node in nodes if node.text),
      'content_description_node_count': sum(
          1 for node in nodes if node.content_description
      ),
      'clickable_node_count': sum(1 for node in nodes if node.is_clickable),
      'packages': sorted({node.package_name for node in nodes if node.package_name}),
      'window_types': [_window_type_name(window) for window in windows],
  }


def _count_nodes(value: Any) -> int:
  if isinstance(value, dict):
    count = 1 if ('id' in value and ('c' in value or 'class_name' in value)) else 0
    return count + sum(_count_nodes(item) for item in value.values())
  if isinstance(value, list):
    return sum(_count_nodes(item) for item in value)
  return 0


def _count_fields(value: Any) -> int:
  if isinstance(value, dict):
    return len(value) + sum(_count_fields(item) for item in value.values())
  if isinstance(value, list):
    return sum(_count_fields(item) for item in value)
  return 0


def _json_text(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _load_tokenizer(model: str | None):
  if model is None:
    return None
  try:
    from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise SystemExit(
        'The --model option requires transformers in the active environment.'
    ) from exc
  return AutoTokenizer.from_pretrained(model, local_files_only=True)


def _token_count(tokenizer: Any, text: str) -> int | None:
  if tokenizer is None:
    return None
  return len(tokenizer(text, add_special_tokens=False)['input_ids'])


def _variant_metric(
    name: str,
    value: Any,
    *,
    raw_chars: int,
    baseline_chars: int,
    tokenizer: Any,
) -> dict[str, Any]:
  text = value if isinstance(value, str) else _json_text(value)
  metric = {
      'chars': len(text),
      'lines': text.count('\n') + 1 if text else 0,
      'nodes': _count_nodes(value),
      'fields': _count_fields(value),
      'compression_vs_raw_chars': (
          round(len(text) / raw_chars, 6) if raw_chars else None
      ),
      'compression_vs_baseline_chars': (
          round(len(text) / baseline_chars, 6) if baseline_chars else None
      ),
  }
  tokens = _token_count(tokenizer, text)
  if tokens is not None:
    metric['tokens'] = tokens
  return {'variant': name, **metric}


def _build_metrics(
    *,
    raw_text: str,
    baseline_text: str,
    variants: dict[str, Any],
    raw_stats: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
  baseline_chars = len(baseline_text)
  metrics = {
      'raw_stats': raw_stats,
      'variants': [],
  }
  metrics['variants'].append(
      _variant_metric(
          'raw_protobuf_text',
          raw_text,
          raw_chars=len(raw_text),
          baseline_chars=baseline_chars,
          tokenizer=tokenizer,
      )
  )
  metrics['variants'].append(
      _variant_metric(
          'baseline_ui_elements_text',
          baseline_text,
          raw_chars=len(raw_text),
          baseline_chars=baseline_chars,
          tokenizer=tokenizer,
      )
  )
  for name, value in variants.items():
    metrics['variants'].append(
        _variant_metric(
            name,
            value,
            raw_chars=len(raw_text),
            baseline_chars=baseline_chars,
            tokenizer=tokenizer,
        )
    )
  return metrics


def _app_dirs(input_dir: Path, app_names: list[str] | None) -> list[Path]:
  if app_names:
    return [input_dir / app_name for app_name in app_names]
  return sorted(path for path in input_dir.iterdir() if path.is_dir())


def _build_one(
    app_dir: Path,
    output_dir: Path,
    *,
    tokenizer: Any,
    include_system_ui: bool,
    include_invisible: bool,
    write_text: bool,
) -> dict[str, Any]:
  tree_path = app_dir / 'a11y_tree.txt'
  app_name = app_dir.name
  if not tree_path.is_file():
    raise FileNotFoundError(f'Missing {tree_path}')

  raw_text = tree_path.read_text(encoding='utf-8')
  forest = _read_forest(tree_path)
  metadata = _read_json(app_dir / 'metadata.json')
  raw_stats = _raw_stats(forest, raw_text)
  baseline = _baseline_ui_elements(forest, include_invisible)
  baseline_text = _baseline_text(baseline)
  compact_v0 = _compact_tree_v0(forest)
  compact_v1 = _compact_tree_v1(
      forest,
      include_system_ui=include_system_ui,
      include_invisible=include_invisible,
  )

  app_output_dir = output_dir / app_name
  _write_json(app_output_dir / 'raw_stats.json', raw_stats)
  _write_json(app_output_dir / 'baseline_ui_elements.json', baseline)
  _write_json(app_output_dir / 'compact_tree_v0.json', compact_v0)
  _write_json(app_output_dir / 'compact_tree_v1.json', compact_v1)
  if write_text:
    _write_text(app_output_dir / 'baseline_ui_elements.txt', baseline_text)
    _write_text(
        app_output_dir / 'compact_tree_v0.txt',
        json.dumps(compact_v0, ensure_ascii=False, indent=2) + '\n',
    )
    _write_text(
        app_output_dir / 'compact_tree_v1.txt',
        json.dumps(compact_v1, ensure_ascii=False, indent=2) + '\n',
    )

  metrics = _build_metrics(
      raw_text=raw_text,
      baseline_text=baseline_text,
      variants={
          'baseline_ui_elements_json': baseline,
          'compact_tree_v0_json': compact_v0,
          'compact_tree_v1_json': compact_v1,
      },
      raw_stats=raw_stats,
      tokenizer=tokenizer,
  )
  _write_json(app_output_dir / 'metrics.json', metrics)

  return {
      'app_name': app_name,
      'source_tree_path': str(tree_path),
      'output_dir': str(app_output_dir),
      'status': 'ok',
      'metadata': metadata,
      'summary': {
          'raw_nodes': raw_stats['node_count'],
          'baseline_elements': len(baseline),
          'compact_tree_v1_nodes': _count_nodes(compact_v1),
          'compact_tree_v1_chars': next(
              item['chars']
              for item in metrics['variants']
              if item['variant'] == 'compact_tree_v1_json'
          ),
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Build UI-state IR candidates from raw A11y tree exports.'
  )
  parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument('--apps', nargs='+', help='App directory names to process.')
  parser.add_argument(
      '--model',
      help='Optional local Hugging Face model/tokenizer path for token metrics.',
  )
  parser.add_argument(
      '--include-system-ui',
      action='store_true',
      help='Keep pure com.android.systemui windows in compact_tree_v1.',
  )
  parser.add_argument(
      '--include-invisible',
      action='store_true',
      help='Keep invisible nodes in baseline and compact_tree_v1.',
  )
  parser.add_argument(
      '--no-write-text',
      action='store_true',
      help='Only write JSON files, not prompt-friendly text files.',
  )
  parser.add_argument('--fail-fast', action='store_true')
  args = parser.parse_args()

  tokenizer = _load_tokenizer(args.model)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  records = []
  for app_dir in _app_dirs(args.input_dir, args.apps):
    try:
      record = _build_one(
          app_dir,
          args.output_dir,
          tokenizer=tokenizer,
          include_system_ui=args.include_system_ui,
          include_invisible=args.include_invisible,
          write_text=not args.no_write_text,
      )
      print(f'[{app_dir.name}] ok')
    except Exception as exc:  # pylint: disable=broad-exception-caught
      record = {
          'app_name': app_dir.name,
          'source_tree_path': str(app_dir / 'a11y_tree.txt'),
          'status': 'error',
          'error': f'{type(exc).__name__}: {exc}',
      }
      print(f'[{app_dir.name}] error: {record["error"]}')
      if args.fail_fast:
        raise
    records.append(record)

  manifest = {
      'input_dir': str(args.input_dir),
      'output_dir': str(args.output_dir),
      'model': args.model,
      'include_system_ui': args.include_system_ui,
      'include_invisible': args.include_invisible,
      'records': records,
  }
  _write_json(args.output_dir / 'manifest.json', manifest)
  print(f'wrote {args.output_dir / "manifest.json"}')


if __name__ == '__main__':
  main()
