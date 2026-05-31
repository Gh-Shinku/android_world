#!/usr/bin/env python3
"""Offline wrapper for Android World UI State Compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from google.protobuf import text_format

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from android_env.proto.a11y import android_accessibility_forest_pb2
from android_world.ui_state import compiler
from android_world.ui_state import ir
from android_world.ui_state.adapters import a11y


DEFAULT_INPUT_DIR = Path('data/A11y')
DEFAULT_OUTPUT_DIR = Path('data/ui_state_prompt_ir')


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


def _compile_one(
    app_dir: Path,
    output_dir: Path,
    *,
    ui_compiler: compiler.UiStateCompiler,
) -> dict[str, Any]:
  app_name = app_dir.name
  tree_path = app_dir / 'a11y_tree.txt'
  if not tree_path.is_file():
    raise FileNotFoundError(f'Missing {tree_path}')

  metadata = _read_json(app_dir / 'metadata.json')
  forest = _read_forest(tree_path)
  screen_ir = a11y.forest_to_screen_ir(
      app_name=app_name,
      activity=metadata.get('foreground_activity') or '',
      forest=forest,
      metadata={
          'source': metadata.get('a11y_source') or 'a11y',
          'source_tree': metadata.get('a11y_tree') or 'a11y_tree.txt',
          'screenshot': metadata.get('screenshot') or 'screenshot.png',
      },
  )
  compiled = ui_compiler.compile_screen(screen_ir)
  report = dict(compiled.compile_report)
  report['source_tree_path'] = str(tree_path)

  app_output_dir = output_dir / app_name
  _write_text(app_output_dir / 'compiled_prompt.txt', compiled.prompt)
  _write_json(app_output_dir / 'screen_ir.json', ir.to_jsonable(compiled.screen_ir))
  _write_json(
      app_output_dir / 'optimized_ir.json', ir.to_jsonable(compiled.optimized_ir)
  )
  _write_json(app_output_dir / 'compiled_ir.json', compiled.compiled_ir)
  _write_json(app_output_dir / 'action_map.json', compiled.action_map)
  _write_json(
      app_output_dir / 'sufficiency_report.json', compiled.sufficiency_report
  )
  _write_json(app_output_dir / 'compile_report.json', report)
  return {
      'app_name': app_name,
      'status': 'ok',
      'output_dir': str(app_output_dir),
      'summary': report,
  }


def _app_dirs(input_dir: Path, app_names: list[str] | None) -> list[Path]:
  if app_names:
    return [input_dir / app_name for app_name in app_names]
  return sorted(path for path in input_dir.iterdir() if path.is_dir())


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Compile raw A11y forests into prompt-oriented UI-state IR.'
  )
  parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument('--apps', nargs='+', help='App directory names to process.')
  parser.add_argument('--include-system-ui', action='store_true')
  parser.add_argument('--include-invisible', action='store_true')
  parser.add_argument('--fail-fast', action='store_true')
  args = parser.parse_args()

  ui_compiler = compiler.UiStateCompiler(
      compiler.UiStateCompilerConfig(
          include_system_ui=args.include_system_ui,
          include_invisible=args.include_invisible,
      )
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  records = []
  for app_dir in _app_dirs(args.input_dir, args.apps):
    try:
      record = _compile_one(app_dir, args.output_dir, ui_compiler=ui_compiler)
      print(f'[{app_dir.name}] ok')
    except Exception as exc:  # pylint: disable=broad-exception-caught
      record = {
          'app_name': app_dir.name,
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
      'include_system_ui': args.include_system_ui,
      'include_invisible': args.include_invisible,
      'records': records,
  }
  _write_json(args.output_dir / 'manifest.json', manifest)
  print(f'wrote {args.output_dir / "manifest.json"}')


if __name__ == '__main__':
  main()
