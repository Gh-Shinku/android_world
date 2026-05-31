"""Resolves compiled UI-state action targets into executable JSONAction."""

from __future__ import annotations

import copy
from typing import Any

from android_world.env import json_action


def _center(bounds: list[int]) -> tuple[int, int]:
  return int((bounds[0] + bounds[2]) / 2), int((bounds[1] + bounds[3]) / 2)


class CompiledActionResolver:
  """Converts target-based compiled actions into coordinate JSON actions."""

  def __init__(self, action_map: dict[str, dict[str, Any]]):
    self._action_map = action_map

  def resolve(self, action: json_action.JSONAction) -> json_action.JSONAction:
    if not action.target:
      return action
    if action.target not in self._action_map:
      raise ValueError(f'Unknown UI-state target: {action.target}')

    target = self._action_map[action.target]
    bounds = target.get('bounds')
    if not bounds:
      raise ValueError(f'UI-state target has no bounds: {action.target}')

    resolved = copy.deepcopy(action)
    resolved.index = None
    resolved.target = None
    resolved.x, resolved.y = _center(bounds)
    resolved.target_bounds = [int(value) for value in bounds]
    expected_ops = {
        json_action.CLICK: 'click',
        json_action.LONG_PRESS: 'long_press',
        json_action.INPUT_TEXT: 'input_text',
        json_action.SCROLL: 'scroll',
    }
    if target.get('op') and resolved.action_type in expected_ops:
      if expected_ops[resolved.action_type] != target['op']:
        raise ValueError(
            f'Action {resolved.action_type} is incompatible with target '
            f'{action.target} ({target["op"]}).'
        )
    return resolved
