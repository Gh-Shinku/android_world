#!/usr/bin/env python3
"""Validates prompt component export against saved Android World episodes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

from android_world import suite_utils
from android_world.agents import t3a


def _load_first_episode(path: Path) -> dict:
  with gzip.open(path, 'rb') as f:
    episodes = pickle.load(f)
  if not isinstance(episodes, list) or not episodes:
    raise ValueError(f'{path} does not contain a non-empty episode list.')
  return episodes[0]


def _validate_hashes(episode: dict) -> None:
  item = suite_utils._create_prompt_data_item(episode, None)  # pylint: disable=protected-access
  if item is None:
    raise ValueError('Could not create prompt data item.')

  mismatches = 0
  steps = item['steps']
  goal = item['goal']
  guidelines = item['additional_guidelines']
  for step in steps:
    history = [
        f'Step {i + 1}: ' + steps[i]['summary']
        for i in range(len(step['history_summaries']))
    ]
    prompt = t3a._action_selection_prompt(  # pylint: disable=protected-access
        goal,
        history,
        step['before_elements_description'],
        guidelines,
    )
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if prompt_hash != step['action_prompt_sha256']:
      mismatches += 1

  print(item['task_template'], len(steps))
  print('mismatches', mismatches)
  if mismatches:
    raise AssertionError(f'Found {mismatches} prompt hash mismatches.')


def _validate_jsonl_smoke(episode: dict, output_path: Path | None) -> None:
  if output_path is None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl')
    output_path = Path(tmp.name)
    tmp.close()

  existing_records = [
      {'task_template': 'AAAExistingTask', 'steps': []},
      {'task_template': 'ZZZExistingTask', 'steps': []},
  ]
  output_path.write_text(
      ''.join(json.dumps(record) + '\n' for record in existing_records),
      encoding='utf-8',
  )
  suite_utils._append_prompt_data_jsonl(  # pylint: disable=protected-access
      episode,
      str(output_path),
      None,
  )
  lines = output_path.read_text(encoding='utf-8').splitlines()
  if len(lines) != 3:
    raise AssertionError(f'Expected three JSONL lines, found {len(lines)}.')

  records = [json.loads(line) for line in lines]
  task_names = [record['task_template'] for record in records]
  if task_names != sorted(task_names):
    raise AssertionError(f'Records are not sorted by task name: {task_names}')

  item = next(
      record for record in records
      if record['task_template'] not in ('AAAExistingTask', 'ZZZExistingTask')
  )
  first_step = item['steps'][0]
  if 'action_output' in first_step:
    raise AssertionError('action_output should not be exported.')
  if 'before_elements' not in first_step:
    raise AssertionError('before_elements missing from exported step.')

  print(item['task_template'], len(item['steps']))
  print(f'wrote {output_path}')


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint', type=Path)
  parser.add_argument(
      '--mode',
      choices=('hash', 'jsonl'),
      required=True,
  )
  parser.add_argument('--output', type=Path, default=None)
  args = parser.parse_args()

  episode = _load_first_episode(args.checkpoint)
  if args.mode == 'hash':
    _validate_hashes(episode)
  else:
    _validate_jsonl_smoke(episode, args.output)


if __name__ == '__main__':
  main()
