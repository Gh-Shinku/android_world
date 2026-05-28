#!/usr/bin/env python3
"""Reconstructs T3A action-selection prompts from prompt-data JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_world import constants
from android_world.agents import t3a

_PROMPT_KIND = 't3a_action_selection'


def _sha256(text: str) -> str:
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  records = []
  with path.open('r', encoding='utf-8') as f:
    for line_number, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        records.append(json.loads(line))
      except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON on {path}:{line_number}: {e}') from e
  return records


def _history_strings(step: dict[str, Any]) -> list[str]:
  summaries = step.get('history_summaries') or []
  if not isinstance(summaries, list):
    raise ValueError('step.history_summaries must be a list when present.')
  return [f'Step {i + 1}: {summary}' for i, summary in enumerate(summaries)]


def _reconstruct_prompt(record: dict[str, Any], step: dict[str, Any]) -> str:
  return t3a._action_selection_prompt(  # pylint: disable=protected-access
      record.get(constants.EpisodeConstants.GOAL) or '',
      _history_strings(step),
      step.get('before_elements_description') or '',
      record.get('additional_guidelines'),
  )


def _reconstruct_records(
    prompt_data_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  reconstructed = []
  task_occurrences: dict[str, int] = {}
  for record_index, record in enumerate(prompt_data_records):
    task_template = record.get(constants.EpisodeConstants.TASK_TEMPLATE) or ''
    occurrence_index = task_occurrences.get(task_template, 0)
    task_occurrences[task_template] = occurrence_index + 1
    steps = record.get('steps')
    if not isinstance(steps, list):
      raise ValueError(
          f'Prompt data record {record_index} for {task_template} has no steps.'
      )
    for step in steps:
      prompt = _reconstruct_prompt(record, step)
      prompt_sha256 = _sha256(prompt)
      expected_sha256 = step.get('action_prompt_sha256')
      reconstructed.append({
          'prompt_data_record_index': record_index,
          'task_occurrence_index': occurrence_index,
          constants.EpisodeConstants.TASK_TEMPLATE: task_template,
          constants.EpisodeConstants.INSTANCE_ID: record.get(
              constants.EpisodeConstants.INSTANCE_ID
          ),
          constants.EpisodeConstants.GOAL: record.get(
              constants.EpisodeConstants.GOAL
          ),
          constants.STEP_NUMBER: step.get(constants.STEP_NUMBER),
          'prompt_kind': _PROMPT_KIND,
          'prompt_sha256': prompt_sha256,
          'expected_action_prompt_sha256': expected_sha256,
          'matches_exported_hash': (
              expected_sha256 is None or prompt_sha256 == expected_sha256
          ),
          'prompt': prompt,
      })
  return reconstructed


def _runtime_key(record: dict[str, Any]) -> tuple[Any, ...]:
  instance_id = record.get(constants.EpisodeConstants.INSTANCE_ID)
  if instance_id is None:
    instance_id = record.get('task_occurrence_index')
  return (
      record.get(constants.EpisodeConstants.TASK_TEMPLATE),
      instance_id,
      record.get(constants.STEP_NUMBER),
      record.get('prompt_kind'),
  )


def _load_runtime_records(runtime_prompt_path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
  records = [
      record for record in _read_jsonl(runtime_prompt_path)
      if record.get('prompt_kind') == _PROMPT_KIND
  ]
  task_occurrences: dict[str, int] = {}
  for record in records:
    task_template = record.get(constants.EpisodeConstants.TASK_TEMPLATE) or ''
    if record.get(constants.EpisodeConstants.INSTANCE_ID) is None:
      record['task_occurrence_index'] = task_occurrences.get(task_template, 0)
      task_occurrences[task_template] = record['task_occurrence_index'] + 1
  keyed = {}
  for record in records:
    key = _runtime_key(record)
    if key in keyed:
      raise ValueError(f'Duplicate runtime prompt key: {key}')
    keyed[key] = record
  return keyed


def _write_readable(records: list[dict[str, Any]], output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('w', encoding='utf-8') as f:
    for index, record in enumerate(records):
      if index:
        f.write('\n')
      f.write('=' * 88 + '\n')
      f.write(
          'task_template={task} instance_id={instance} step_number={step} '
          'sha256={sha}\n'.format(
              task=record.get(constants.EpisodeConstants.TASK_TEMPLATE),
              instance=record.get(constants.EpisodeConstants.INSTANCE_ID),
              step=record.get(constants.STEP_NUMBER),
              sha=record.get('prompt_sha256'),
          )
      )
      f.write('=' * 88 + '\n')
      f.write(record['prompt'])
      if not record['prompt'].endswith('\n'):
        f.write('\n')


def _write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('w', encoding='utf-8') as f:
    for record in records:
      manifest_record = {k: v for k, v in record.items() if k != 'prompt'}
      f.write(json.dumps(manifest_record, ensure_ascii=False) + '\n')


def _compare_runtime(
    reconstructed_records: list[dict[str, Any]],
    runtime_prompt_path: Path,
) -> int:
  runtime_records = _load_runtime_records(runtime_prompt_path)
  mismatches = 0
  for record in reconstructed_records:
    key = _runtime_key(record)
    runtime_record = runtime_records.get(key)
    if runtime_record is None:
      print(f'MISSING runtime prompt for {key}')
      mismatches += 1
      continue
    runtime_hash = runtime_record.get('prompt_sha256')
    if runtime_hash is None and isinstance(runtime_record.get('prompt'), str):
      runtime_hash = _sha256(runtime_record['prompt'])
    if runtime_hash != record['prompt_sha256']:
      print(
          'MISMATCH {key}: runtime={runtime} reconstructed={reconstructed}'
          .format(
              key=key,
              runtime=runtime_hash,
              reconstructed=record['prompt_sha256'],
          )
      )
      mismatches += 1
  return mismatches


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('prompt_data', type=Path)
  parser.add_argument(
      '--output',
      type=Path,
      required=True,
      help='Readable text file to write reconstructed prompts to.',
  )
  parser.add_argument(
      '--manifest-output',
      type=Path,
      default=None,
      help='JSONL manifest path. Defaults to OUTPUT with .manifest.jsonl.',
  )
  parser.add_argument(
      '--runtime-prompts',
      type=Path,
      default=None,
      help='Optional runtime prompt JSONL to hash-compare against.',
  )
  args = parser.parse_args()

  prompt_data_records = _read_jsonl(args.prompt_data)
  reconstructed_records = _reconstruct_records(prompt_data_records)
  if not reconstructed_records:
    raise ValueError(f'No reconstructable prompts found in {args.prompt_data}.')

  manifest_output = args.manifest_output
  if manifest_output is None:
    manifest_output = args.output.with_suffix(args.output.suffix + '.manifest.jsonl')
  _write_readable(reconstructed_records, args.output)
  _write_manifest(reconstructed_records, manifest_output)

  exported_mismatches = sum(
      1 for record in reconstructed_records
      if not record['matches_exported_hash']
  )
  runtime_mismatches = 0
  if args.runtime_prompts is not None:
    runtime_mismatches = _compare_runtime(
        reconstructed_records, args.runtime_prompts
    )

  print(f'wrote {args.output}')
  print(f'wrote {manifest_output}')
  print(f'reconstructed {len(reconstructed_records)} prompts')
  print(f'exported hash mismatches: {exported_mismatches}')
  if args.runtime_prompts is not None:
    print(f'runtime hash mismatches: {runtime_mismatches}')

  if exported_mismatches or runtime_mismatches:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
