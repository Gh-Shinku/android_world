#!/usr/bin/env python3
"""Installs and sets up all Android World apps on a running emulator."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import shutil

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from android_world.env import env_launcher
from android_world.env import adb_utils
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


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Install and initialize all Android World apps.'
  )
  parser.add_argument('--console_port', type=int, default=5554)
  parser.add_argument('--grpc_port', type=int, default=8554)
  parser.add_argument('--adb_path', default=_default_adb_path())
  args = parser.parse_args()

  print(
      'connecting to emulator '
      f'console_port={args.console_port} grpc_port={args.grpc_port} '
      f'adb_path={args.adb_path} '
      f'ANDROID_ADB_SERVER_PORT={os.environ.get("ANDROID_ADB_SERVER_PORT", "")}',
      flush=True,
  )
  env = env_launcher.load_and_setup_env(
      console_port=args.console_port,
      grpc_port=args.grpc_port,
      adb_path=args.adb_path,
      emulator_setup=False,
      freeze_datetime=False,
  )
  try:
    adb_utils.press_home_button(env.controller)
    adb_utils.set_root_if_needed(env.controller)
    print('installing and setting up all Android World apps', flush=True)
    for app in _APPS:
      print(f'[{app.app_name}] install start', flush=True)
      setup.maybe_install_app(app, env)
      print(f'[{app.app_name}] setup start', flush=True)
      setup.setup_app(app, env)
      print(f'[{app.app_name}] done', flush=True)
    print('all apps setup complete')
  finally:
    try:
      env.close()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f'env close warning: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
  main()
