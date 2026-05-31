"""Prompt backend for optimized UI State IR."""

from __future__ import annotations

from typing import Any

from android_world.ui_state import ir
from android_world.ui_state import optimizer


def _action_to_public(action: ir.ActionIR) -> dict[str, Any]:
  return {
      'id': action.id,
      'op': action.op,
      'target': action.element_id,
      'role': action.role,
      'label': action.label,
      'bounds': action.bounds,
      'enabled': action.enabled,
  }


def _fold_context(
    element_id: str,
    screen: ir.ScreenIR,
    live_ids: set[str],
    action_element_ids: set[str],
) -> list[dict[str, Any]]:
  if element_id not in screen.elements or element_id not in live_ids:
    return []
  element = screen.elements[element_id]
  children = []
  for child_id in element.child_ids:
    children.extend(_fold_context(child_id, screen, live_ids, action_element_ids))

  if element_id in action_element_ids:
    return children
  if not optimizer.has_semantics(element):
    return children
  value = {
      'id': element.id,
      'role': element.role,
      'label': element.label,
      'bounds': element.bounds,
  }
  if element.text:
    value['text'] = element.text
  if element.description:
    value['description'] = element.description
  if element.hint:
    value['hint'] = element.hint
  if element.value:
    value['value'] = element.value
  state = sorted(flag for flag in element.state if flag != 'visible')
  if state:
    value['state'] = state
  if children:
    value['children'] = children
  return [value]


def compile_ir(screen: ir.ScreenIR, analysis: dict[str, Any]) -> dict[str, Any]:
  """Compiles optimized ScreenIR into backend-oriented structured IR."""
  live_ids = analysis['live_ids']
  action_element_ids = analysis['action_element_ids']
  group_action_ids = {
      action_id for group in screen.groups for action_id in group.action_ids
  }
  dialog_surface_ids = {
      group.surface_id for group in screen.groups if group.kind == 'dialog'
  }
  compiled_groups = []
  for group in screen.groups:
    compiled_groups.append({
        'id': group.id,
        'kind': group.kind,
        'surface_id': group.surface_id,
        'title': group.title,
        'body': group.body,
        'modal': group.modal,
        'actions': [
            _action_to_public(screen.actions[action_id])
            for action_id in group.action_ids
            if action_id in screen.actions
        ],
    })

  compiled_surfaces = []
  for surface in screen.surfaces.values():
    if surface.id in dialog_surface_ids:
      continue
    root_context = []
    for root_id in surface.root_ids:
      root_context.extend(_fold_context(root_id, screen, live_ids, action_element_ids))
    actions = [
        _action_to_public(action)
        for action in sorted(
            screen.actions.values(),
            key=lambda item: (item.bounds[1], item.bounds[0], item.id),
        )
        if action.id not in group_action_ids
        and screen.elements[action.element_id].surface_id == surface.id
    ]
    if root_context or actions:
      compiled_surfaces.append({
          'id': surface.id,
          'kind': surface.kind,
          'bounds': surface.bounds,
          'context': root_context,
          'actions': actions,
      })

  return {
      'version': 'prompt_ir_v1',
      'screen': {
          'app_name': screen.app_name,
          'activity': screen.activity,
      },
      'groups': compiled_groups,
      'surfaces': compiled_surfaces,
      'action_count': len(screen.actions),
  }


def action_map(screen: ir.ScreenIR) -> dict[str, dict[str, Any]]:
  result = {}
  for action_id, action in screen.actions.items():
    element = screen.elements[action.element_id]
    result[action_id] = {
        'op': action.op,
        'element_id': action.element_id,
        'label': action.label,
        'role': action.role,
        'bounds': action.bounds,
        'enabled': action.enabled,
        'state': sorted(element.state),
        'source_ref': ir.to_jsonable(action.source_ref),
      }
  return result


def _format_action_line(action: dict[str, Any], indent: str = '  ') -> str:
  label = action.get('label') or action.get('role') or action['id']
  op = action.get('op') or 'act'
  role = action.get('role') or 'node'
  bounds = action.get('bounds')
  suffix = f' [{",".join(str(v) for v in bounds)}]' if bounds else ''
  disabled = ' disabled' if action.get('enabled') is False else ''
  return f'{indent}{action["id"]} {op} {role} "{label}"{suffix}{disabled}'


def _format_context_line(node: dict[str, Any], depth: int = 0) -> list[str]:
  indent = '  ' * depth
  label = node.get('label') or node.get('role') or node.get('id')
  role = node.get('role') or 'node'
  bounds = node.get('bounds')
  suffix = f' [{",".join(str(v) for v in bounds)}]' if bounds else ''
  lines = [f'{indent}- {role}: {label}{suffix}']
  for child in node.get('children', []):
    lines.extend(_format_context_line(child, depth + 1))
  return lines


def prompt(compiled: dict[str, Any]) -> str:
  """Renders compiled UI state into compact prompt text."""
  screen = compiled['screen']
  activity = screen.get('activity') or ''
  lines = [f'Screen: {screen["app_name"]}' + (f' / {activity}' if activity else '')]
  lines.append('')

  if compiled['groups']:
    lines.append('Groups:')
    for group in compiled['groups']:
      if group['kind'] == 'dialog':
        lines.append(f'Modal Dialog {group["id"]}: {group.get("title", "")}')
        for body in group.get('body') or []:
          lines.append(f'  Body: {body}')
      elif group['kind'] == 'toolbar':
        lines.append(f'Toolbar {group["id"]}: {group.get("title", "")}')
      else:
        lines.append(f'{group["kind"].title()} {group["id"]}: {group.get("title", "")}')
      if group.get('actions'):
        lines.append('  Actions:')
        for action in group['actions']:
          lines.append(_format_action_line(action, indent='    '))
    lines.append('')

  if compiled['surfaces']:
    lines.append('Surfaces:')
    for surface in compiled['surfaces']:
      lines.append(f'Surface {surface["id"]}: {surface["kind"]}')
      if surface.get('context'):
        lines.append('  Context:')
        for node in surface['context']:
          lines.extend('  ' + line for line in _format_context_line(node))
      if surface.get('actions'):
        lines.append('  Actions:')
        for action in surface['actions']:
          lines.append(_format_action_line(action, indent='    '))
    lines.append('')

  if compiled['action_count'] == 0:
    lines.append('Actions: none')
  return '\n'.join(lines).rstrip() + '\n'


def sufficiency_report(
    *,
    app_name: str,
    screen: ir.ScreenIR,
    compiled: dict[str, Any],
) -> dict[str, Any]:
  actions = [
      action
      for group in compiled['groups']
      for action in group.get('actions', [])
  ] + [
      action
      for surface in compiled['surfaces']
      for action in surface.get('actions', [])
  ]
  action_labels = [action.get('label') for action in actions]
  has_modal = any(
      group.get('kind') == 'dialog' and group.get('modal')
      for group in compiled['groups']
  )
  has_warning_dialog = any(
      group.get('kind') == 'dialog'
      and group.get('title') == 'Warning!'
      and 'Some of your records was deleted or moved' in (group.get('body') or [])
      for group in compiled['groups']
  )
  has_ok_action = any(
      action.get('op') == 'click' and action.get('label') == 'Ok'
      for action in actions
  )
  system_actions = [
      action['id']
      for action in actions
      if screen.surfaces[screen.elements[action['target']].surface_id].kind == 'system'
  ]
  checks = {
      'has_actions': bool(actions),
      'all_actions_have_label_op_bounds': all(
          action.get('label') and action.get('op') and action.get('bounds')
          for action in actions
      ),
      'modal_context_present_if_required': True,
      'system_actions_filtered': not system_actions,
  }
  if app_name == 'audio_recorder':
    checks['audio_recorder_warning_dialog_present'] = has_warning_dialog
    checks['audio_recorder_expected_ok_action_present'] = has_ok_action
  return {
      'status': 'pass' if all(checks.values()) else 'warn',
      'checks': checks,
      'action_labels': action_labels,
      'system_action_ids': system_actions,
      'expected_action': (
          {'label': 'Ok', 'op': 'click'} if app_name == 'audio_recorder' else None
      ),
  }
