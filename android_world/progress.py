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

"""Console progress logging helpers for long-running benchmark steps."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
import contextvars
import time
from typing import Any


PrintFn = Callable[..., None]

_TASK = contextvars.ContextVar[str | None]('progress_task', default=None)
_INSTANCE_ID = contextvars.ContextVar[int | None](
    'progress_instance_id', default=None
)


@contextlib.contextmanager
def context(
    *,
    task: str | None = None,
    instance_id: int | None = None,
) -> Iterator[None]:
  """Applies task context to nested progress logs."""
  task_token = _TASK.set(task if task is not None else _TASK.get())
  instance_token = _INSTANCE_ID.set(
      instance_id if instance_id is not None else _INSTANCE_ID.get()
  )
  try:
    yield
  finally:
    _INSTANCE_ID.reset(instance_token)
    _TASK.reset(task_token)


def log(
    stage: str,
    event: str,
    *,
    task: str | None = None,
    instance_id: int | None = None,
    step: int | None = None,
    elapsed: float | None = None,
    print_fn: PrintFn = print,
    **details: Any,
) -> None:
  """Prints a single flushed progress log line."""
  if task is None:
    task = _TASK.get()
  if instance_id is None:
    instance_id = _INSTANCE_ID.get()
  fields = ['[progress]']
  if task:
    fields.append(str(task))
  if instance_id is not None:
    fields.append(f'instance={instance_id}')
  if step is not None:
    fields.append(f'step={step}')
  fields.extend([event, stage])
  if elapsed is not None:
    fields.append(f'elapsed={elapsed:.2f}s')
  for key, value in details.items():
    if value is not None:
      fields.append(f'{key}={value}')
  line = ' '.join(fields)
  try:
    print_fn(line, flush=True)
  except TypeError:
    print_fn(line)


@contextlib.contextmanager
def span(
    stage: str,
    *,
    task: str | None = None,
    instance_id: int | None = None,
    step: int | None = None,
    print_fn: PrintFn = print,
    **details: Any,
) -> Iterator[None]:
  """Logs start/done/error messages with elapsed time for a stage."""
  log(
      stage,
      'start',
      task=task,
      instance_id=instance_id,
      step=step,
      print_fn=print_fn,
      **details,
  )
  start = time.monotonic()
  try:
    yield
  except Exception as e:
    log(
        stage,
        'error',
        task=task,
        instance_id=instance_id,
        step=step,
        elapsed=time.monotonic() - start,
        print_fn=print_fn,
        error=type(e).__name__,
        **details,
    )
    raise
  else:
    log(
        stage,
        'done',
        task=task,
        instance_id=instance_id,
        step=step,
        elapsed=time.monotonic() - start,
        print_fn=print_fn,
        **details,
    )
