#!/usr/bin/env python3
"""Initializes a benchmark pass/fail state file from checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from android_world import benchmark_state


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--run-dir',
      type=Path,
      required=True,
      help='Run directory containing *.pkl.gz checkpoint files.',
  )
  parser.add_argument(
      '--output',
      type=Path,
      required=True,
      help='Path to write TaskName_instance -> true/false benchmark state.',
  )
  parser.add_argument(
      '--overwrite',
      action='store_true',
      help='Overwrite output if it already exists.',
  )
  parser.add_argument(
      '--timeout-s',
      type=float,
      default=30.0,
      help='Per-checkpoint read timeout in seconds.',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.output.exists() and not args.overwrite:
    raise ValueError(f'Output already exists: {args.output}')

  state = benchmark_state.state_from_run_dir(args.run_dir, timeout_s=args.timeout_s)
  benchmark_state.save_state(args.output, state)
  success_count = sum(
      1 for status in state.values() if status == benchmark_state.SUCCESS
  )
  fail_count = sum(1 for status in state.values() if status == benchmark_state.FAIL)
  print(f'Wrote {args.output}')
  print(f'Success: {success_count}')
  print(f'Fail: {fail_count}')


if __name__ == '__main__':
  main()
