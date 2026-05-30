#!/usr/bin/env python3
"""Collects raw Android World app accessibility trees and screenshots."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

from google.protobuf import text_format
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from android_world.env import adb_utils
from android_world.env import env_launcher
from android_world.env.setup_device import setup


_APPS = setup._APPS  # pylint: disable=protected-access


def _default_adb_path() -> str:
  candidates = [
      os.environ.get('ADB_PATH'),
      shutil.which('adb'),
      '/home/zyt/sda_ws/programs/android/platform-tools/adb',
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  for candidate in candidates:
    if candidate and Path(candidate).is_file():
      return candidate
  raise EnvironmentError(
      'adb not found. Pass --adb_path or set ADB_PATH to the adb binary.'
  )


def _safe_name(value: str) -> str:
  value = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip().lower())
  return value.strip('_') or 'app'


def _save_screenshot(pixels: np.ndarray, output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(pixels).save(output_path)


def _is_blank_screenshot(pixels: np.ndarray) -> bool:
  if pixels.size == 0:
    return True
  return float(np.std(pixels)) < 1.0


def _screenshot_file_is_blank(screenshot_path: Path) -> bool:
  try:
    return _is_blank_screenshot(np.asarray(Image.open(screenshot_path)))
  except Exception:  # pylint: disable=broad-exception-caught
    return True


def _collection_complete(tree_path: Path, screenshot_path: Path) -> bool:
  if (
      not tree_path.is_file()
      or tree_path.stat().st_size == 0
      or not screenshot_path.is_file()
      or screenshot_path.stat().st_size == 0
  ):
    return False
  return not _screenshot_file_is_blank(screenshot_path)


def _remove_blank_existing_outputs(tree_path: Path, screenshot_path: Path) -> None:
  if screenshot_path.is_file() and _screenshot_file_is_blank(screenshot_path):
    screenshot_path.unlink()
    if tree_path.is_file():
      tree_path.unlink()


def _app_package_name(app: type) -> str:
  try:
    return app.package_name()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise RuntimeError(
        f'Cannot resolve package name for {app.app_name}'
    ) from exc


def _launch_app(app: type, env: Any) -> None:
  package_name = _app_package_name(app)
  try:
    response = adb_utils.issue_generic_request(
        [
            'shell',
            'monkey',
            '-p',
            package_name,
            '-candroid.intent.category.LAUNCHER',
            '1',
        ],
        env.controller,
        timeout_sec=30,
    )
    adb_utils.check_ok(
        response, f'Failed to launch {app.app_name} package {package_name}.'
    )
    output = response.generic.output.decode(errors='replace').lower()
    if 'no activities found to run' in output or 'monkey aborted' in output:
      raise RuntimeError(output.strip())
  except Exception as monkey_exc:  # pylint: disable=broad-exception-caught
    try:
      adb_utils.launch_app(app.app_name, env.controller)
    except Exception as launch_exc:  # pylint: disable=broad-exception-caught
      raise RuntimeError(
          f'Failed to launch {app.app_name} with package launcher and '
          'activity fallback.'
      ) from launch_exc
    activity, _ = adb_utils.get_current_activity(env.controller)
    if not activity or not activity.startswith(package_name):
      raise RuntimeError(
          f'Failed to launch {app.app_name}. Package launcher failed with '
          f'{type(monkey_exc).__name__}: {monkey_exc}; activity fallback '
          f'foreground is {activity!r}.'
      )


def _forest_to_text(forest: Any) -> str:
  if forest is None:
    return ''
  try:
    return text_format.MessageToString(forest)
  except text_format.Error:
    return str(forest)


def _write_text(path: Path, contents: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(contents, encoding='utf-8')


def _write_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
      encoding='utf-8',
  )


def _read_existing_manifest(output_dir: Path) -> list[dict[str, Any]]:
  manifest_path = output_dir / 'manifest.json'
  if not manifest_path.is_file():
    return []
  try:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
  except json.JSONDecodeError:
    return []
  if not isinstance(manifest, list):
    return []
  return [record for record in manifest if isinstance(record, dict)]


def _merge_records(
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  merged = {
      record.get('app_name'): record
      for record in existing_records
      if record.get('app_name')
  }
  for record in new_records:
    merged[record['app_name']] = record
  return list(merged.values())


def _selected_apps(app_names: list[str] | None):
  if not app_names:
    return _APPS

  selected = []
  app_by_name = {app.app_name: app for app in _APPS}
  missing = []
  for app_name in app_names:
    app = app_by_name.get(app_name)
    if app is None:
      missing.append(app_name)
    else:
      selected.append(app)
  if missing:
    raise ValueError(
        'Unknown app(s): '
        + ', '.join(missing)
        + '. Known apps: '
        + ', '.join(sorted(app_by_name))
    )
  return tuple(selected)


def _get_valid_state(env: Any, timeout_sec: float, poll_sec: float):
  deadline = time.time() + timeout_sec
  last_state = None
  while True:
    last_state = env.get_state(wait_to_stabilize=False)
    if not _is_blank_screenshot(last_state.pixels):
      return last_state
    if time.time() >= deadline:
      raise RuntimeError(
          'Captured screenshot stayed blank. The app may have failed to render '
          'or crashed during launch.'
      )
    time.sleep(poll_sec)


def _build_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
  rows = []
  for record in records:
    app_name = html.escape(record['app_name'])
    status = html.escape(record['status'])
    app_dir = html.escape(record['directory'])
    screenshot = html.escape(record.get('screenshot', ''))
    tree = html.escape(record.get('a11y_tree', ''))
    error = html.escape(record.get('error', ''))
    if record.get('screenshot') and record.get('a11y_tree'):
      preview = (
          f'<img src="{app_dir}/{screenshot}" alt="{app_name} screenshot">'
          f'<iframe src="{app_dir}/{tree}" title="{app_name} a11y tree"></iframe>'
      )
    else:
      preview = f'<pre class="error">{error}</pre>'
    rows.append(
        '<section>'
        f'<h2>{app_name} <span>{status}</span></h2>'
        f'<div class="pair">{preview}</div>'
        '</section>'
    )

  contents = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Android World raw A11y trees</title>
<style>
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 24px;
  color: #202124;
  background: #f8fafc;
}
h1 {
  font-size: 24px;
  margin: 0 0 16px;
}
section {
  border-top: 1px solid #d8dee9;
  padding: 20px 0;
}
h2 {
  font-size: 18px;
  margin: 0 0 12px;
}
h2 span {
  color: #5f6368;
  font-size: 13px;
  font-weight: 500;
}
.pair {
  display: grid;
  grid-template-columns: minmax(260px, 360px) minmax(420px, 1fr);
  gap: 16px;
  align-items: start;
}
img {
  width: 100%;
  height: auto;
  border: 1px solid #d8dee9;
  background: white;
}
iframe {
  width: 100%;
  height: 720px;
  border: 1px solid #d8dee9;
  background: white;
}
.error {
  white-space: pre-wrap;
  color: #b3261e;
}
@media (max-width: 900px) {
  .pair {
    grid-template-columns: 1fr;
  }
}
</style>
</head>
<body>
<h1>Android World raw A11y trees</h1>
""" + '\n'.join(rows) + """
</body>
</html>
"""
  _write_text(output_dir / 'index.html', contents)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          'Collect raw get_a11y_tree accessibility forests and matching '
          'screenshots for Android World benchmark apps.'
      )
  )
  parser.add_argument('--console_port', type=int, default=5554)
  parser.add_argument('--grpc_port', type=int, default=8554)
  parser.add_argument('--adb_path', default=_default_adb_path())
  parser.add_argument('--output_dir', type=Path, default=Path('data/A11y'))
  parser.add_argument(
      '--apps',
      nargs='+',
      help='Optional app_name list. Defaults to every Android World app.',
  )
  parser.add_argument(
      '--launch_delay_sec',
      type=float,
      default=3.0,
      help='Seconds to wait after launching each app before capture.',
  )
  parser.add_argument(
      '--valid_capture_timeout_sec',
      type=float,
      default=10.0,
      help='Seconds to wait for a non-blank screenshot before failing an app.',
  )
  parser.add_argument(
      '--valid_capture_poll_sec',
      type=float,
      default=1.0,
      help='Seconds between non-blank screenshot checks.',
  )
  parser.add_argument(
      '--wait_to_stabilize',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='Wait for the UI to stabilize before saving the screenshot/tree.',
  )
  parser.add_argument(
      '--freeze_datetime',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='Freeze Android datetime through env setup.',
  )
  parser.add_argument(
      '--setup_apps',
      action='store_true',
      help='Install and run Android World app setup before collecting.',
  )
  parser.add_argument(
      '--resume',
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          'Skip apps that already have non-empty a11y_tree.txt and '
          'screenshot.png outputs.'
      ),
  )
  parser.add_argument(
      '--recollect_apps',
      nargs='+',
      default=[],
      help=(
          'App names to collect again even when --resume would skip existing '
          'outputs.'
      ),
  )
  parser.add_argument(
      '--fail_fast',
      action='store_true',
      help='Stop on the first app collection failure.',
  )
  args = parser.parse_args()

  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  selected_apps = _selected_apps(args.apps)
  recollect_app_names = {
      app.app_name for app in _selected_apps(args.recollect_apps)
  }

  print(
      'connecting to emulator '
      f'console_port={args.console_port} grpc_port={args.grpc_port} '
      f'adb_path={args.adb_path}',
      flush=True,
  )
  env = env_launcher.load_and_setup_env(
      console_port=args.console_port,
      grpc_port=args.grpc_port,
      adb_path=args.adb_path,
      emulator_setup=args.setup_apps,
      freeze_datetime=args.freeze_datetime,
  )

  records: list[dict[str, Any]] = []
  try:
    adb_utils.press_home_button(env.controller)
    env.hide_automation_ui()

    for app in selected_apps:
      app_dir_name = _safe_name(app.app_name)
      app_dir = output_dir / app_dir_name
      tree_path = app_dir / 'a11y_tree.txt'
      screenshot_path = app_dir / 'screenshot.png'
      metadata_path = app_dir / 'metadata.json'

      if (
          args.resume
          and app.app_name not in recollect_app_names
          and _collection_complete(tree_path, screenshot_path)
      ):
        records.append({
            'app_name': app.app_name,
            'status': 'skipped',
            'directory': app_dir_name,
            'a11y_tree': tree_path.name,
            'screenshot': screenshot_path.name,
        })
        print(f'[{app.app_name}] skip existing {app_dir}', flush=True)
        continue

      _remove_blank_existing_outputs(tree_path, screenshot_path)
      print(f'[{app.app_name}] launch', flush=True)
      try:
        _launch_app(app, env)
        time.sleep(args.launch_delay_sec)

        if args.wait_to_stabilize:
          env.get_state(wait_to_stabilize=True)
        state = _get_valid_state(
            env,
            timeout_sec=args.valid_capture_timeout_sec,
            poll_sec=args.valid_capture_poll_sec,
        )
        # state.forest is the raw AndroidAccessibilityForest returned by
        # get_a11y_tree via the A11Y_FORWARDER_APP wrapper.
        _write_text(tree_path, _forest_to_text(state.forest))
        _save_screenshot(state.pixels, screenshot_path)

        activity, _ = adb_utils.get_current_activity(env.controller)
        metadata = {
            'app_name': app.app_name,
            'app_class': app.__name__,
            'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'foreground_activity': activity,
            'a11y_source': 'get_a11y_tree(A11Y_FORWARDER_APP)',
            'a11y_tree': tree_path.name,
            'screenshot': screenshot_path.name,
        }
        _write_json(metadata_path, metadata)

        records.append({
            'app_name': app.app_name,
            'status': 'ok',
            'directory': app_dir_name,
            'a11y_tree': tree_path.name,
            'screenshot': screenshot_path.name,
        })
        print(f'[{app.app_name}] wrote {app_dir}', flush=True)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        error = f'{type(exc).__name__}: {exc}'
        records.append({
            'app_name': app.app_name,
            'status': 'error',
            'directory': app_dir_name,
            'error': error,
        })
        _write_json(metadata_path, {
            'app_name': app.app_name,
            'app_class': app.__name__,
            'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'error': error,
        })
        print(f'[{app.app_name}] error: {error}', flush=True)
        if args.fail_fast:
          raise
      finally:
        adb_utils.press_home_button(env.controller)

    records = _merge_records(_read_existing_manifest(output_dir), records)
    _write_json(output_dir / 'manifest.json', records)
    _build_index(output_dir, records)
    print(f'wrote manifest and index to {output_dir}', flush=True)
  finally:
    try:
      env.close()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f'env close warning: {type(exc).__name__}: {exc}', flush=True)


if __name__ == '__main__':
  main()
