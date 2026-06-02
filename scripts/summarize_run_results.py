#!/usr/bin/env python3
"""Summarizes partial Android World run checkpoints.

This is intended for interrupted runs where ``suite_utils.process_episodes`` did
not get a chance to print a final table. It only reads checkpoint files and does
not contact the emulator or an LLM provider.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gzip
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import queue
import re
from typing import Any


CHECKPOINT_SUFFIX = '.pkl.gz'

GOAL = 'goal'
TASK_TEMPLATE = 'task_template'
INSTANCE_ID = 'instance_id'
IS_SUCCESSFUL = 'is_successful'
EPISODE_LENGTH = 'episode_length'
RUN_TIME = 'run_time'
EXCEPTION_INFO = 'exception_info'
AGENT_NAME = 'agent_name'
FINISH_DTIME = 'finish_dtime'
AUX_DATA = 'aux_data'

SUMMARY_BASENAME = 'partial_summary'


@dataclasses.dataclass(frozen=True)
class FileSummary:
  checkpoint: str
  task_template: str | None
  instance_id: int | None
  goal: str | None
  agent_name: str | None
  status: str
  success_score: float | None
  episode_length: float | None
  run_time_s: float | None
  finish_dtime: str | None
  error_kind: str | None
  error_tail: str | None


def _safe_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    numeric = float(value)
  except (TypeError, ValueError):
    return None
  if math.isnan(numeric):
    return None
  return numeric


def _safe_int(value: Any) -> int | None:
  if value is None:
    return None
  try:
    numeric = int(value)
  except (TypeError, ValueError):
    return None
  return numeric


def _safe_str(value: Any) -> str | None:
  if value is None:
    return None
  if isinstance(value, str):
    return value
  if isinstance(value, dt.datetime):
    return value.isoformat(sep=' ')
  return str(value)


def _error_tail(exception_info: Any) -> str | None:
  if not exception_info:
    return None
  lines = [line.strip() for line in str(exception_info).splitlines() if line.strip()]
  if not lines:
    return None
  return lines[-1]


def _error_kind(exception_info: Any) -> str | None:
  if not exception_info:
    return None
  text = str(exception_info)
  lower_text = text.lower()
  if 'insufficient_quota' in lower_text or 'quota' in lower_text:
    return 'quota'
  if 'rate_limit' in lower_text or 'ratelimit' in lower_text:
    return 'rate_limit'
  if re.search(r'\b429\b', text):
    return 'http_429'
  if 'deadlineexceeded' in text or 'timeout' in lower_text:
    return 'timeout'
  if 'connectionerror' in text or 'connection error' in lower_text:
    return 'connection_error'
  if 'permission denial' in lower_text or 'securityexception' in lower_text:
    return 'android_permission'
  if 'error calling llm in action selection phase' in lower_text:
    return 'llm_action_error'
  if 'error calling llm in summarization phase' in lower_text:
    return 'llm_summary_error'
  tail = _error_tail(exception_info)
  return tail or 'unknown_error'


def _classify_episode(episode: dict[str, Any]) -> str:
  if episode.get(EXCEPTION_INFO) is not None:
    return 'runtime_error'
  success_score = _safe_float(episode.get(IS_SUCCESSFUL))
  if success_score is None:
    return 'unknown'
  if success_score > 0.5:
    return 'success'
  return 'task_failure'


def _episode_to_summary(path: Path, episode: dict[str, Any]) -> FileSummary:
  return FileSummary(
      checkpoint=path.name,
      task_template=_safe_str(episode.get(TASK_TEMPLATE)),
      instance_id=_safe_int(episode.get(INSTANCE_ID)),
      goal=_safe_str(episode.get(GOAL)),
      agent_name=_safe_str(episode.get(AGENT_NAME)),
      status=_classify_episode(episode),
      success_score=_safe_float(episode.get(IS_SUCCESSFUL)),
      episode_length=_safe_float(episode.get(EPISODE_LENGTH)),
      run_time_s=_safe_float(episode.get(RUN_TIME)),
      finish_dtime=_safe_str(episode.get(FINISH_DTIME)),
      error_kind=_error_kind(episode.get(EXCEPTION_INFO)),
      error_tail=_error_tail(episode.get(EXCEPTION_INFO)),
  )


def _load_checkpoint(path: Path) -> list[FileSummary]:
  with gzip.open(path, 'rb') as f:
    episodes = pickle.load(f)
  if not isinstance(episodes, list):
    episodes = [episodes]

  summaries = []
  for episode in episodes:
    if not isinstance(episode, dict):
      summaries.append(FileSummary(
          checkpoint=path.name,
          task_template=None,
          instance_id=None,
          goal=None,
          agent_name=None,
          status='unreadable',
          success_score=None,
          episode_length=None,
          run_time_s=None,
          finish_dtime=None,
          error_kind='not_dict_episode',
          error_tail=repr(type(episode)),
      ))
      continue
    summaries.append(_episode_to_summary(path, episode))
  return summaries


def _worker(path: str, output_queue: mp.Queue) -> None:
  try:
    output_queue.put({
        'ok': True,
        'summaries': [dataclasses.asdict(item) for item in _load_checkpoint(Path(path))],
    })
  except Exception as exc:  # pylint: disable=broad-exception-caught
    output_queue.put({
        'ok': False,
        'error_kind': type(exc).__name__,
        'error_tail': str(exc),
    })


def _load_checkpoint_with_timeout(path: Path, timeout_s: float) -> list[FileSummary]:
  output_queue: mp.Queue = mp.Queue(maxsize=1)
  process = mp.Process(target=_worker, args=(str(path), output_queue))
  process.start()
  process.join(timeout_s)
  if process.is_alive():
    process.terminate()
    process.join(1)
    if process.is_alive():
      process.kill()
      process.join()
    return [FileSummary(
        checkpoint=path.name,
        task_template=path.name.removesuffix(CHECKPOINT_SUFFIX),
        instance_id=None,
        goal=None,
        agent_name=None,
        status='unreadable',
        success_score=None,
        episode_length=None,
        run_time_s=None,
        finish_dtime=None,
        error_kind='timeout',
        error_tail=f'timed out after {timeout_s:g}s while reading checkpoint',
    )]

  try:
    result = output_queue.get_nowait()
  except queue.Empty:
    return [FileSummary(
        checkpoint=path.name,
        task_template=path.name.removesuffix(CHECKPOINT_SUFFIX),
        instance_id=None,
        goal=None,
        agent_name=None,
        status='unreadable',
        success_score=None,
        episode_length=None,
        run_time_s=None,
        finish_dtime=None,
        error_kind='worker_exit_without_result',
        error_tail=f'exit code {process.exitcode}',
    )]

  if not result['ok']:
    return [FileSummary(
        checkpoint=path.name,
        task_template=path.name.removesuffix(CHECKPOINT_SUFFIX),
        instance_id=None,
        goal=None,
        agent_name=None,
        status='unreadable',
        success_score=None,
        episode_length=None,
        run_time_s=None,
        finish_dtime=None,
        error_kind=result.get('error_kind') or 'load_error',
        error_tail=result.get('error_tail'),
    )]

  return [FileSummary(**item) for item in result['summaries']]


def _iter_checkpoints(run_dir: Path) -> list[Path]:
  return sorted(
      run_dir.glob(f'*{CHECKPOINT_SUFFIX}'),
      key=lambda path: path.name,
  )


def _counts_by(items: list[FileSummary], field_name: str) -> dict[str, int]:
  counts: dict[str, int] = {}
  for item in items:
    value = getattr(item, field_name)
    key = str(value) if value else ''
    counts[key] = counts.get(key, 0) + 1
  return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _mean(values: list[float]) -> float | None:
  if not values:
    return None
  return sum(values) / len(values)


def _aggregate(items: list[FileSummary], run_dir: Path) -> dict[str, Any]:
  attempted = len(items)
  runtime_errors = [item for item in items if item.status == 'runtime_error']
  unreadable = [item for item in items if item.status == 'unreadable']
  evaluated = [
      item for item in items
      if item.status in ('success', 'task_failure', 'unknown')
  ]
  successes = [item for item in items if item.status == 'success']
  failures = [item for item in items if item.status == 'task_failure']
  known_evaluated = [
      item for item in evaluated if item.status in ('success', 'task_failure')
  ]
  runtimes = [
      item.run_time_s for item in items if item.run_time_s is not None
  ]
  lengths = [
      item.episode_length for item in evaluated
      if item.episode_length is not None
  ]

  success_rate_evaluated = (
      len(successes) / len(known_evaluated) if known_evaluated else None
  )
  success_rate_attempted = len(successes) / attempted if attempted else None

  return {
      'run_dir': str(run_dir),
      'checkpoint_files': len(_iter_checkpoints(run_dir)),
      'episodes': attempted,
      'status_counts': _counts_by(items, 'status'),
      'error_kind_counts': _counts_by(runtime_errors + unreadable, 'error_kind'),
      'evaluated_episodes': len(evaluated),
      'known_evaluated_episodes': len(known_evaluated),
      'successes': len(successes),
      'task_failures': len(failures),
      'runtime_errors': len(runtime_errors),
      'unreadable_checkpoints': len(unreadable),
      'success_rate_evaluated': success_rate_evaluated,
      'success_rate_attempted': success_rate_attempted,
      'total_runtime_s': sum(runtimes),
      'mean_episode_length_evaluated': _mean(lengths),
  }


def _jsonable(value: Any) -> Any:
  if isinstance(value, float) and math.isnan(value):
    return None
  return value


def _write_json(path: Path, aggregate: dict[str, Any], items: list[FileSummary]) -> None:
  payload = {
      'summary': aggregate,
      'episodes': [dataclasses.asdict(item) for item in items],
  }
  path.write_text(
      json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable),
      encoding='utf-8',
  )


def _write_csv(path: Path, items: list[FileSummary]) -> None:
  with path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[field.name for field in dataclasses.fields(FileSummary)],
    )
    writer.writeheader()
    for item in items:
      writer.writerow(dataclasses.asdict(item))


def _fmt_rate(value: float | None) -> str:
  if value is None:
    return 'n/a'
  return f'{value:.3f}'


def _fmt_number(value: float | None) -> str:
  if value is None:
    return 'n/a'
  return f'{value:.1f}'


def _write_markdown(path: Path, aggregate: dict[str, Any], items: list[FileSummary]) -> None:
  lines = [
      f'# {path.parent.name} Partial Summary',
      '',
      f'- Episodes: {aggregate["episodes"]}',
      f'- Evaluated episodes: {aggregate["evaluated_episodes"]}',
      f'- Known evaluated episodes: {aggregate["known_evaluated_episodes"]}',
      f'- Successes: {aggregate["successes"]}',
      f'- Task failures: {aggregate["task_failures"]}',
      f'- Runtime errors: {aggregate["runtime_errors"]}',
      f'- Unreadable checkpoints: {aggregate["unreadable_checkpoints"]}',
      f'- Success rate over evaluated: {_fmt_rate(aggregate["success_rate_evaluated"])}',
      f'- Success rate over attempted: {_fmt_rate(aggregate["success_rate_attempted"])}',
      f'- Total runtime seconds: {_fmt_number(aggregate["total_runtime_s"])}',
      '',
      '## Status Counts',
      '',
  ]
  for status, count in aggregate['status_counts'].items():
    lines.append(f'- {status or "unknown"}: {count}')

  lines.extend(['', '## Error Kinds', ''])
  error_counts = aggregate['error_kind_counts']
  if error_counts:
    for kind, count in error_counts.items():
      lines.append(f'- {kind or "unknown"}: {count}')
  else:
    lines.append('- none')

  lines.extend([
      '',
      '## Episodes',
      '',
      '| checkpoint | status | score | steps | runtime_s | error_kind |',
      '| --- | --- | ---: | ---: | ---: | --- |',
  ])
  for item in items:
    lines.append(
        '| {checkpoint} | {status} | {score} | {steps} | {runtime} | {error} |'.format(
            checkpoint=item.checkpoint,
            status=item.status,
            score=_fmt_number(item.success_score),
            steps=_fmt_number(item.episode_length),
            runtime=_fmt_number(item.run_time_s),
            error=item.error_kind or '',
        )
    )

  path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _print_summary(aggregate: dict[str, Any]) -> None:
  print(f'Run: {aggregate["run_dir"]}')
  print(f'Checkpoint files: {aggregate["checkpoint_files"]}')
  print(f'Episodes: {aggregate["episodes"]}')
  print('Status counts:')
  for status, count in aggregate['status_counts'].items():
    print(f'  {status or "unknown"}: {count}')
  print('Error kind counts:')
  if aggregate['error_kind_counts']:
    for kind, count in aggregate['error_kind_counts'].items():
      print(f'  {kind or "unknown"}: {count}')
  else:
    print('  none')
  print(f'Success rate over evaluated: {_fmt_rate(aggregate["success_rate_evaluated"])}')
  print(f'Success rate over attempted: {_fmt_rate(aggregate["success_rate_attempted"])}')
  print(f'Total runtime seconds: {_fmt_number(aggregate["total_runtime_s"])}')


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Summarize partial Android World run checkpoint results.'
  )
  parser.add_argument(
      '--run-dir',
      type=Path,
      required=True,
      help='Run directory containing *.pkl.gz checkpoint files.',
  )
  parser.add_argument(
      '--timeout-s',
      type=float,
      default=30.0,
      help='Per-checkpoint read timeout in seconds.',
  )
  parser.add_argument(
      '--output-dir',
      type=Path,
      default=None,
      help='Directory for partial_summary.{json,csv,md}; defaults to run dir.',
  )
  parser.add_argument(
      '--no-write',
      action='store_true',
      help='Only print the summary; do not write output files.',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  run_dir = args.run_dir
  if not run_dir.is_dir():
    raise ValueError(f'Run directory does not exist: {run_dir}')

  checkpoints = _iter_checkpoints(run_dir)
  if not checkpoints:
    raise ValueError(f'No *{CHECKPOINT_SUFFIX} files found under {run_dir}')

  items: list[FileSummary] = []
  for checkpoint in checkpoints:
    items.extend(_load_checkpoint_with_timeout(checkpoint, args.timeout_s))

  items.sort(key=lambda item: item.checkpoint)
  aggregate = _aggregate(items, run_dir)
  _print_summary(aggregate)

  if args.no_write:
    return

  output_dir = args.output_dir or run_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_json(output_dir / f'{SUMMARY_BASENAME}.json', aggregate, items)
  _write_csv(output_dir / f'{SUMMARY_BASENAME}.csv', items)
  _write_markdown(output_dir / f'{SUMMARY_BASENAME}.md', aggregate, items)
  print(f'Wrote {output_dir / f"{SUMMARY_BASENAME}.json"}')
  print(f'Wrote {output_dir / f"{SUMMARY_BASENAME}.csv"}')
  print(f'Wrote {output_dir / f"{SUMMARY_BASENAME}.md"}')


if __name__ == '__main__':
  main()
