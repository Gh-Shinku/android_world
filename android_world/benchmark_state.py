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

"""Persistent benchmark pass/fail state."""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import queue
import re
from typing import Any, Mapping

from android_world import checkpointer
from android_world import constants


SUCCESS = 'true'
FAIL = 'false'
VALID_STATUSES = frozenset((SUCCESS, FAIL))
STATE_LINE_RE = re.compile(
    r'^\s*"(?P<key>(?:[^"\\]|\\.)*)"\s*:\s*"(?P<status>true|false)"\s*(?P<comment>#.*)?$'
)


def instance_name(task_template: str, instance_id: int) -> str:
  return f'{task_template}{checkpointer.INSTANCE_SEPARATOR}{instance_id}'


def is_successful_episode(episode: Mapping[str, Any]) -> bool:
  if episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is not None:
    return False
  try:
    return float(episode.get(constants.EpisodeConstants.IS_SUCCESSFUL, 0.0)) > 0.5
  except (TypeError, ValueError):
    return False


def status_for_episode(episode: Mapping[str, Any]) -> str:
  return SUCCESS if is_successful_episode(episode) else FAIL


def _unquote(value: str) -> str:
  return json.loads(f'"{value}"')


def _quote(value: str) -> str:
  return json.dumps(value, ensure_ascii=False)


def _strip_comment(line: str) -> str:
  in_string = False
  escaped = False
  for index, char in enumerate(line):
    if escaped:
      escaped = False
      continue
    if char == '\\' and in_string:
      escaped = True
      continue
    if char == '"':
      in_string = not in_string
      continue
    if char == '#' and not in_string:
      return line[:index]
  return line


def _load_json_state(text: str, path: Path) -> dict[str, str]:
  raw = json.loads(text)
  if not isinstance(raw, dict):
    raise ValueError(f'Benchmark state must be a mapping: {path}')
  state: dict[str, str] = {}
  for key, value in raw.items():
    if value == 'success':
      value = SUCCESS
    elif value == 'fail':
      value = FAIL
    if value not in VALID_STATUSES:
      raise ValueError(
          f'Invalid benchmark state for {key}: {value!r}. '
          f'Expected one of {sorted(VALID_STATUSES)}.'
      )
    state[str(key)] = str(value)
  return state


def load_state(path: Path | str) -> dict[str, str]:
  path = Path(path)
  if not path.exists():
    return {}
  text = path.read_text(encoding='utf-8')
  stripped = text.lstrip()
  if stripped.startswith('{'):
    return _load_json_state(text, path)

  state: dict[str, str] = {}
  for line_number, line in enumerate(text.splitlines(), start=1):
    if not line.strip() or line.lstrip().startswith('#'):
      continue
    match = STATE_LINE_RE.match(line)
    if not match:
      uncommented = _strip_comment(line).strip()
      if not uncommented:
        continue
      raise ValueError(
          f'Invalid benchmark state line {line_number} in {path}: {line!r}'
      )
    state[_unquote(match.group('key'))] = match.group('status')
  return state


def save_state(path: Path | str, state: Mapping[str, str]) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = [f'{_quote(key)}: {_quote(status)}' for key, status in state.items()]
  path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def save_state_preserving_comments(path: Path | str, state: Mapping[str, str]) -> None:
  path = Path(path)
  if not path.exists() or path.read_text(encoding='utf-8').lstrip().startswith('{'):
    save_state(path, state)
    return

  original_lines = path.read_text(encoding='utf-8').splitlines()
  seen: set[str] = set()
  output_lines: list[str] = []
  for line in original_lines:
    match = STATE_LINE_RE.match(line)
    if not match:
      output_lines.append(line)
      continue
    key = _unquote(match.group('key'))
    if key not in state:
      output_lines.append(line)
      continue
    comment = match.group('comment') or ''
    separator = ' ' if comment else ''
    output_lines.append(f'{_quote(key)}: {_quote(state[key])}{separator}{comment}')
    seen.add(key)

  for key, status in state.items():
    if key not in seen:
      output_lines.append(f'{_quote(key)}: {_quote(status)}')

  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text('\n'.join(output_lines) + '\n', encoding='utf-8')


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
  with gzip.open(path, 'rb') as f:
    episodes = pickle.load(f)
  if not isinstance(episodes, list):
    episodes = [episodes]
  return [episode for episode in episodes if isinstance(episode, dict)]


def _state_from_checkpoint(checkpoint_path: Path) -> dict[str, str]:
  checkpoint_key = checkpoint_path.name.removesuffix('.pkl.gz')
  try:
    episodes = _load_checkpoint(checkpoint_path)
  except Exception:  # pylint: disable=broad-exception-caught
    return {checkpoint_key: FAIL}

  state: dict[str, str] = {}
  if not episodes:
    state[checkpoint_key] = FAIL
    return state
  for episode_index, episode in enumerate(episodes):
    task_template = episode.get(constants.EpisodeConstants.TASK_TEMPLATE)
    instance_id = episode.get(constants.EpisodeConstants.INSTANCE_ID)
    if task_template is not None and instance_id is not None:
      try:
        key = instance_name(str(task_template), int(instance_id))
      except (TypeError, ValueError):
        key = checkpoint_key
    elif len(episodes) == 1:
      key = checkpoint_key
    else:
      key = f'{checkpoint_key}{checkpointer.INSTANCE_SEPARATOR}{episode_index}'
    state[key] = status_for_episode(episode)
  return state


def _checkpoint_worker(path: str, output_queue: mp.Queue) -> None:
  output_queue.put(_state_from_checkpoint(Path(path)))


def _state_from_checkpoint_with_timeout(
    checkpoint_path: Path,
    timeout_s: float,
) -> dict[str, str]:
  output_queue: mp.Queue = mp.Queue(maxsize=1)
  process = mp.Process(
      target=_checkpoint_worker,
      args=(str(checkpoint_path), output_queue),
  )
  process.start()
  process.join(timeout_s)
  checkpoint_key = checkpoint_path.name.removesuffix('.pkl.gz')
  if process.is_alive():
    process.terminate()
    process.join(1)
    if process.is_alive():
      process.kill()
      process.join()
    return {checkpoint_key: FAIL}

  try:
    return output_queue.get_nowait()
  except queue.Empty:
    return {checkpoint_key: FAIL}


def state_from_run_dir(run_dir: Path, timeout_s: float = 30.0) -> dict[str, str]:
  """Builds benchmark state from a checkpoint directory."""
  if not run_dir.is_dir():
    raise ValueError(f'Run directory does not exist: {run_dir}')

  state: dict[str, str] = {}
  for checkpoint_path in sorted(run_dir.glob('*.pkl.gz')):
    checkpoint_state = _state_from_checkpoint_with_timeout(
        checkpoint_path,
        timeout_s,
    )
    state.update(checkpoint_state)
  return state


class BenchmarkState:
  """Mutable benchmark state file used to run only failed instances."""

  def __init__(
      self, path: Path | str, state: Mapping[str, str] | None = None
  ) -> None:
    self.path = Path(path)
    self.state = dict(state) if state is not None else load_state(self.path)

  def should_run(self, key: str) -> bool:
    return self.state.get(key) == FAIL

  def mark_episode(self, key: str, episode: Mapping[str, Any]) -> str:
    status = status_for_episode(episode)
    self.state[key] = status
    return status

  def save(self) -> None:
    save_state_preserving_comments(self.path, self.state)
