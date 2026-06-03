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

"""Tests for progress logging helpers."""

from absl.testing import absltest
from android_world import progress


class ProgressTest(absltest.TestCase):

  def test_span_logs_start_done_and_flushes(self):
    calls = []

    def fake_print(*args, **kwargs):
      calls.append((args, kwargs))

    with progress.span(
        'llm_action',
        task='Task',
        instance_id=1,
        step=2,
        print_fn=fake_print,
    ):
      pass

    lines = [args[0] for args, _ in calls]
    self.assertIn('[progress] Task instance=1 step=2 start llm_action', lines[0])
    self.assertIn('[progress] Task instance=1 step=2 done llm_action', lines[1])
    self.assertIn('elapsed=', lines[1])
    self.assertTrue(all(kwargs.get('flush') for _, kwargs in calls))

  def test_span_logs_error(self):
    calls = []

    def fake_print(*args, **kwargs):
      calls.append((args, kwargs))

    with self.assertRaises(ValueError):
      with progress.span('execute_action', step=3, print_fn=fake_print):
        raise ValueError('bad action')

    error_line = calls[-1][0][0]
    self.assertIn('[progress] step=3 error execute_action', error_line)
    self.assertIn('error=ValueError', error_line)

  def test_log_supports_print_fn_without_flush(self):
    lines = []

    def fake_print(line):
      lines.append(line)

    progress.log('step', 'start', step=1, print_fn=fake_print)

    self.assertEqual(lines, ['[progress] step=1 start step'])

  def test_context_applies_task_and_instance(self):
    lines = []

    def fake_print(line, **kwargs):
      del kwargs
      lines.append(line)

    with progress.context(task='TaskA', instance_id=3):
      progress.log('llm_action', 'start', step=2, print_fn=fake_print)

    self.assertEqual(
        lines,
        ['[progress] TaskA instance=3 step=2 start llm_action'],
    )


if __name__ == '__main__':
  absltest.main()
