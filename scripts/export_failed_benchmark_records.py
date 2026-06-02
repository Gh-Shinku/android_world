#!/usr/bin/env python3
"""Exports readable files for benchmark records still marked fail."""

from __future__ import annotations

import argparse
from pathlib import Path

from android_world import benchmark_state
from scripts import export_episode_readable


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--benchmark-state',
      type=Path,
      required=True,
      help='File containing TaskName_instance -> true/false benchmark state.',
  )
  parser.add_argument(
      '--checkpoint-dir',
      type=Path,
      required=True,
      help='Directory containing *.pkl.gz checkpoints.',
  )
  parser.add_argument(
      '--output-dir',
      type=Path,
      default=None,
      help='Readable export directory. Defaults to checkpoint-dir/failed_readable.',
  )
  parser.add_argument(
      '--missing-ok',
      action='store_true',
      help='Skip fail records whose checkpoint file is missing.',
  )
  parser.add_argument(
      '--tokenizer',
      default=None,
      help='Optional Hugging Face tokenizer model/path for action prompt tokens.',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  state = benchmark_state.load_state(args.benchmark_state)
  output_dir = args.output_dir or args.checkpoint_dir / 'failed_readable'
  fail_keys = [
      key for key, status in state.items() if status == benchmark_state.FAIL
  ]
  tokenizer = export_episode_readable.load_tokenizer(args.tokenizer)

  exported = 0
  missing: list[Path] = []
  review_lines = ['# Failed Benchmark Review', '']
  for key in fail_keys:
    checkpoint_path = args.checkpoint_dir / f'{key}.pkl.gz'
    if not checkpoint_path.exists():
      missing.append(checkpoint_path)
      continue
    task_output_dir = output_dir / key
    export_episode_readable.export_file(
        checkpoint_path,
        task_output_dir,
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
    )
    summary_path = task_output_dir / f'{key}.summary.md'
    readable_path = task_output_dir / f'{key}.readable.json'
    review_lines.extend([
        f'## {key}',
        '',
        f'- Summary: [{summary_path.name}]({key}/{summary_path.name})',
        f'- Readable JSON: [{readable_path.name}]({key}/{readable_path.name})',
        '',
    ])
    exported += 1

  if missing and not args.missing_ok:
    preview = '\n'.join(str(path) for path in missing[:10])
    suffix = '' if len(missing) <= 10 else f'\n... {len(missing) - 10} more'
    raise FileNotFoundError(f'Missing fail checkpoints:\n{preview}{suffix}')

  print(f'Fail records: {len(fail_keys)}')
  print(f'Exported: {exported}')
  print(f'Missing: {len(missing)}')
  print(f'Output: {output_dir}')
  if exported:
    review_path = output_dir / 'failed_review.md'
    review_path.write_text('\n'.join(review_lines) + '\n', encoding='utf-8')
    print(f'Review: {review_path}')


if __name__ == '__main__':
  main()
