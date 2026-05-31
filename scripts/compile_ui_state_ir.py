#!/usr/bin/env python3
"""Compiles raw Android accessibility forests into source-agnostic UI prompts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

from google.protobuf import text_format

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from android_env.proto.a11y import android_accessibility_forest_pb2
from ui_state_ir import ActionIR
from ui_state_ir import ElementIR
from ui_state_ir import GroupIR
from ui_state_ir import ScreenIR
from ui_state_ir import SourceRef
from ui_state_ir import SurfaceIR
from ui_state_ir import to_jsonable


DEFAULT_INPUT_DIR = Path('data/A11y')
DEFAULT_OUTPUT_DIR = Path('data/ui_state_prompt_ir')
SYSTEM_UI_PACKAGE = 'com.android.systemui'


def _read_forest(path: Path):
  forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
  text_format.Parse(path.read_text(encoding='utf-8'), forest)
  return forest


def _read_json(path: Path) -> dict[str, Any]:
  if not path.is_file():
    return {}
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError:
    return {}
  return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
      encoding='utf-8',
  )


def _write_text(path: Path, value: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(value, encoding='utf-8')


def _rect_to_list(rect: Any) -> list[int]:
  return [int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)]


def _window_type_name(window: Any) -> str:
  enum_desc = window.DESCRIPTOR.fields_by_name['window_type'].enum_type
  value = enum_desc.values_by_number.get(window.window_type)
  return value.name if value is not None else str(window.window_type)


def _short_class_name(class_name: str) -> str:
  if not class_name:
    return ''
  for prefix in ('android.widget.', 'android.view.'):
    if class_name.startswith(prefix):
      return class_name.removeprefix(prefix)
  return class_name.rsplit('.', maxsplit=1)[-1]


def _short_resource_name(resource_name: str, package_name: str) -> str:
  if not resource_name:
    return ''
  if package_name and resource_name.startswith(f'{package_name}:id/'):
    return resource_name.removeprefix(f'{package_name}:id/')
  if ':id/' in resource_name:
    return resource_name.split(':id/', maxsplit=1)[1]
  return resource_name


def _state_from_a11y_node(node: Any) -> set[str]:
  result = set()
  pairs = (
      ('checkable', node.is_checkable),
      ('checked', node.is_checked),
      ('enabled', node.is_enabled),
      ('focusable', node.is_focusable),
      ('focused', node.is_focused),
      ('password', node.is_password),
      ('selected', node.is_selected),
      ('visible', node.is_visible_to_user),
  )
  for name, enabled in pairs:
    if enabled:
      result.add(name)
  return result


def _ops_from_a11y_node(node: Any) -> list[str]:
  ops = []
  if node.is_clickable or node.is_checkable:
    ops.append('click')
  if node.is_long_clickable:
    ops.append('long_press')
  if node.is_editable:
    ops.append('input_text')
  if node.is_scrollable:
    ops.append('scroll')
  return ops


def _role_from_a11y_node(node: Any) -> str:
  class_name = _short_class_name(node.class_name).lower()
  if node.is_editable:
    return 'input'
  if node.is_scrollable:
    return 'scroll'
  if 'button' in class_name or node.is_clickable:
    return 'button'
  if 'text' in class_name:
    return 'text'
  if 'image' in class_name:
    return 'image'
  if 'list' in class_name or 'recycler' in class_name:
    return 'list'
  return class_name or 'node'


def _surface_kind(window_type: str, package_names: list[str]) -> str:
  if package_names and set(package_names) <= {SYSTEM_UI_PACKAGE}:
    return 'system'
  if window_type in {'TYPE_INPUT_METHOD', 'TYPE_INPUT_METHOD_DIALOG'}:
    return 'keyboard'
  return 'app'


def _node_label(
    *,
    text: str,
    description: str,
    hint: str,
    resource_key: str,
    role: str,
) -> str:
  for value in (text, description, hint, resource_key):
    if value:
      return value
  return role


def _root_node_ids(window: Any) -> list[int]:
  node_ids = {int(node.unique_id) for node in window.tree.nodes}
  child_ids = {
      int(child_id)
      for node in window.tree.nodes
      for child_id in node.child_ids
  }
  return sorted(node_ids - child_ids) or ([0] if 0 in node_ids else sorted(node_ids))


def _forest_to_screen_ir(
    *,
    app_name: str,
    metadata: dict[str, Any],
    forest: Any,
) -> ScreenIR:
  surfaces: dict[str, SurfaceIR] = {}
  elements: dict[str, ElementIR] = {}
  node_to_element_id: dict[tuple[str, int], str] = {}
  parent_by_element_id: dict[str, str] = {}

  for surface_index, window in enumerate(forest.windows):
    source_window_id = int(window.id)
    surface_id = f's{surface_index}'
    package_names = sorted({
        node.package_name
        for node in window.tree.nodes
        if node.package_name
    })
    window_type = _window_type_name(window)
    root_source_ids = _root_node_ids(window)
    for node in window.tree.nodes:
      node_to_element_id[(surface_id, int(node.unique_id))] = (
          f'e{len(node_to_element_id)}'
      )
    root_ids = [
        node_to_element_id[(surface_id, source_id)]
        for source_id in root_source_ids
        if (surface_id, source_id) in node_to_element_id
    ]
    surfaces[surface_id] = SurfaceIR(
        id=surface_id,
        kind=_surface_kind(window_type, package_names),
        bounds=_rect_to_list(window.bounds_in_screen),
        root_ids=root_ids,
        modal=False,
        z_order=surface_index,
        package_names=package_names,
        source_ref=SourceRef(
            source_type='a11y',
            source_id=f'window:{source_window_id}',
            attrs={
                'window_id': source_window_id,
                'window_type': window_type,
                'active': bool(window.is_active),
                'focused': bool(window.is_focused),
            },
        ),
    )

    for node in window.tree.nodes:
      element_id = node_to_element_id[(surface_id, int(node.unique_id))]
      child_ids = [
          node_to_element_id[(surface_id, int(child_id))]
          for child_id in node.child_ids
          if (surface_id, int(child_id)) in node_to_element_id
      ]
      for child_id in child_ids:
        parent_by_element_id[child_id] = element_id
      role = _role_from_a11y_node(node)
      resource_key = _short_resource_name(
          node.view_id_resource_name, node.package_name
      )
      text = node.text or ''
      description = node.content_description or ''
      hint = node.hint_text or ''
      elements[element_id] = ElementIR(
          id=element_id,
          surface_id=surface_id,
          parent_id=None,
          child_ids=child_ids,
          role=role,
          label=_node_label(
              text=text,
              description=description,
              hint=hint,
              resource_key=resource_key,
              role=role,
          ),
          bounds=_rect_to_list(node.bounds_in_screen),
          state=_state_from_a11y_node(node),
          ops=_ops_from_a11y_node(node),
          text=text,
          description=description,
          hint=hint,
          resource_key=resource_key,
          source_ref=SourceRef(
              source_type='a11y',
              source_id=f'window:{source_window_id}/node:{int(node.unique_id)}',
              attrs={
                  'window_id': source_window_id,
                  'node_id': int(node.unique_id),
                  'class_name': _short_class_name(node.class_name),
                  'package_name': node.package_name or None,
                  'resource_name': node.view_id_resource_name or None,
              },
          ),
      )

  for element_id, parent_id in parent_by_element_id.items():
    elements[element_id].parent_id = parent_id

  return ScreenIR(
      version='screen_ir_v1',
      app_name=app_name,
      activity=metadata.get('foreground_activity') or '',
      surfaces=surfaces,
      elements=elements,
      metadata={
          'source': metadata.get('a11y_source') or 'a11y',
          'source_tree': metadata.get('a11y_tree') or 'a11y_tree.txt',
          'screenshot': metadata.get('screenshot') or 'screenshot.png',
      },
  )


def _element_has_semantics(element: ElementIR) -> bool:
  return bool(
      element.text
      or element.description
      or element.hint
      or element.value
  )


def _is_action_target(element: ElementIR, include_invisible: bool) -> bool:
  if not include_invisible and 'visible' not in element.state:
    return False
  return bool(element.ops)


def _is_context_target(element: ElementIR, include_invisible: bool) -> bool:
  if not include_invisible and 'visible' not in element.state:
    return False
  return _element_has_semantics(element)


def _ancestor_ids(element: ElementIR, elements: dict[str, ElementIR]) -> list[str]:
  result = []
  parent_id = element.parent_id
  while parent_id and parent_id in elements:
    result.append(parent_id)
    parent_id = elements[parent_id].parent_id
  return result


def _descendant_action_count(
    element_id: str,
    elements: dict[str, ElementIR],
    action_element_ids: set[str],
) -> int:
  total = 1 if element_id in action_element_ids else 0
  for child_id in elements[element_id].child_ids:
    if child_id in elements:
      total += _descendant_action_count(child_id, elements, action_element_ids)
  return total


def _live_element_ids(
    screen: ScreenIR,
    *,
    include_system_ui: bool,
    include_invisible: bool,
) -> tuple[set[str], set[str], set[str]]:
  candidate_ids = {
      element_id
      for element_id, element in screen.elements.items()
      if include_system_ui or screen.surfaces[element.surface_id].kind != 'system'
  }
  action_ids = {
      element_id
      for element_id in candidate_ids
      if _is_action_target(screen.elements[element_id], include_invisible)
  }
  context_ids = {
      element_id
      for element_id in candidate_ids
      if _is_context_target(screen.elements[element_id], include_invisible)
  }
  live = set(action_ids) | set(context_ids)
  for element_id in list(live):
    for ancestor_id in _ancestor_ids(screen.elements[element_id], screen.elements):
      if ancestor_id in candidate_ids:
        live.add(ancestor_id)
  for surface in screen.surfaces.values():
    if include_system_ui or surface.kind != 'system':
      live.update(element_id for element_id in surface.root_ids if element_id in screen.elements)
  return live, action_ids, context_ids


def _assign_actions(
    screen: ScreenIR,
    action_element_ids: set[str],
) -> dict[str, str]:
  element_to_action_id = {}
  sorted_ids = sorted(
      action_element_ids,
      key=lambda element_id: (
          screen.elements[element_id].bounds[1],
          screen.elements[element_id].bounds[0],
          element_id,
      ),
  )
  for index, element_id in enumerate(sorted_ids):
    element = screen.elements[element_id]
    action_id = f'A{index}'
    element_to_action_id[element_id] = action_id
    screen.actions[action_id] = ActionIR(
        id=action_id,
        op=element.ops[0],
        element_id=element_id,
        label=element.label,
        role=element.role,
        bounds=element.bounds,
        enabled='enabled' in element.state,
        source_ref=element.source_ref,
    )
  return element_to_action_id


def _nodes_under(
    root_id: str,
    elements: dict[str, ElementIR],
    keep_ids: set[str] | None = None,
) -> list[ElementIR]:
  result = []
  stack = [root_id]
  while stack:
    element_id = stack.pop(0)
    if element_id not in elements:
      continue
    if keep_ids is None or element_id in keep_ids:
      result.append(elements[element_id])
    stack[0:0] = [
        child_id for child_id in elements[element_id].child_ids if child_id in elements
    ]
  return result


def _detect_dialog(
    screen: ScreenIR,
    surface: SurfaceIR,
    live_ids: set[str],
    action_element_ids: set[str],
    element_to_action_id: dict[str, str],
) -> GroupIR | None:
  if surface.kind == 'system':
    return None
  all_nodes = []
  for root_id in surface.root_ids:
    all_nodes.extend(_nodes_under(root_id, screen.elements, live_ids))
  actions = [
      node for node in all_nodes
      if node.id in action_element_ids and node.id in element_to_action_id
  ]
  semantic_nodes = [
      node for node in all_nodes
      if node.id not in action_element_ids and _element_has_semantics(node)
  ]
  text_nodes = [node for node in semantic_nodes if node.text or node.description]
  width = max(0, surface.bounds[2] - surface.bounds[0])
  height = max(0, surface.bounds[3] - surface.bounds[1])
  looks_modal = height < 520 or width < 320
  if not actions or not text_nodes or not looks_modal:
    return None
  surface.modal = True
  surface.kind = 'dialog'
  surface.title = text_nodes[0].label
  return GroupIR(
      id='D0',
      kind='dialog',
      surface_id=surface.id,
      title=text_nodes[0].label,
      body=[node.label for node in text_nodes[1:] if node.label != text_nodes[0].label],
      element_ids=[node.id for node in semantic_nodes],
      action_ids=[element_to_action_id[node.id] for node in actions],
      modal=True,
  )


def _detect_toolbar(
    screen: ScreenIR,
    surface: SurfaceIR,
    live_ids: set[str],
    action_element_ids: set[str],
    element_to_action_id: dict[str, str],
) -> GroupIR | None:
  if surface.kind != 'app':
    return None
  top_limit = surface.bounds[1] + 96
  nodes = [
      element for element in screen.elements.values()
      if element.surface_id == surface.id
      and element.id in live_ids
      and element.bounds[1] <= top_limit
      and 'visible' in element.state
  ]
  actions = [
      element for element in nodes
      if element.id in action_element_ids and element.id in element_to_action_id
  ]
  title_nodes = [
      element for element in nodes
      if element.id not in action_element_ids and (element.text or element.description)
  ]
  if not actions and not title_nodes:
    return None
  return GroupIR(
      id='T0',
      kind='toolbar',
      surface_id=surface.id,
      title=title_nodes[0].label if title_nodes else '',
      element_ids=[element.id for element in title_nodes],
      action_ids=[element_to_action_id[element.id] for element in actions],
  )


def _detect_groups(
    screen: ScreenIR,
    live_ids: set[str],
    action_element_ids: set[str],
    element_to_action_id: dict[str, str],
) -> None:
  groups = []
  for surface in screen.surfaces.values():
    dialog = _detect_dialog(
        screen, surface, live_ids, action_element_ids, element_to_action_id
    )
    if dialog is not None:
      groups.append(dialog)
      continue
    toolbar = _detect_toolbar(
        screen, surface, live_ids, action_element_ids, element_to_action_id
    )
    if toolbar is not None:
      groups.append(toolbar)
  counters: dict[str, int] = {}
  for group in groups:
    prefix = {'dialog': 'D', 'toolbar': 'T'}.get(group.kind, 'G')
    index = counters.get(prefix, 0)
    counters[prefix] = index + 1
    group.id = f'{prefix}{index}'
  screen.groups = groups


def _prune_to_live(
    screen: ScreenIR,
    live_ids: set[str],
    *,
    include_system_ui: bool,
) -> None:
  screen.elements = {
      element_id: element
      for element_id, element in screen.elements.items()
      if element_id in live_ids
  }
  for element in screen.elements.values():
    element.child_ids = [
        child_id for child_id in element.child_ids if child_id in screen.elements
    ]
    if element.parent_id not in screen.elements:
      element.parent_id = None

  grouped_surface_ids = {group.surface_id for group in screen.groups}
  live_surface_ids = {
      element.surface_id for element in screen.elements.values()
  } | grouped_surface_ids
  screen.surfaces = {
      surface_id: surface
      for surface_id, surface in screen.surfaces.items()
      if surface_id in live_surface_ids
      and (include_system_ui or surface.kind != 'system')
  }
  for surface in screen.surfaces.values():
    surface.root_ids = [
        root_id for root_id in surface.root_ids if root_id in screen.elements
    ]


def _fold_context(
    element_id: str,
    screen: ScreenIR,
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
  if not _element_has_semantics(element):
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


def _optimized_screen(
    screen: ScreenIR,
    *,
    include_system_ui: bool,
    include_invisible: bool,
) -> tuple[ScreenIR, dict[str, Any]]:
  optimized = copy.deepcopy(screen)
  live_ids, action_element_ids, context_ids = _live_element_ids(
      optimized,
      include_system_ui=include_system_ui,
      include_invisible=include_invisible,
  )
  element_to_action_id = _assign_actions(optimized, action_element_ids)
  _detect_groups(optimized, live_ids, action_element_ids, element_to_action_id)
  _prune_to_live(
      optimized,
      live_ids,
      include_system_ui=include_system_ui,
  )
  optimized.metadata['optimizer'] = {
      'live_element_ids': sorted(live_ids),
      'action_element_ids': sorted(action_element_ids),
      'context_element_ids': sorted(context_ids),
      'include_system_ui': include_system_ui,
      'include_invisible': include_invisible,
  }
  return optimized, {
      'live_ids': live_ids,
      'action_element_ids': action_element_ids,
      'context_ids': context_ids,
      'element_to_action_id': element_to_action_id,
  }


def _action_to_public(action: ActionIR) -> dict[str, Any]:
  return {
      'id': action.id,
      'op': action.op,
      'target': action.element_id,
      'role': action.role,
      'label': action.label,
      'bounds': action.bounds,
      'enabled': action.enabled,
  }


def _compiled_ir(screen: ScreenIR, analysis: dict[str, Any]) -> dict[str, Any]:
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


def _action_map(screen: ScreenIR) -> dict[str, dict[str, Any]]:
  action_map = {}
  for action_id, action in screen.actions.items():
    element = screen.elements[action.element_id]
    action_map[action_id] = {
        'op': action.op,
        'element_id': action.element_id,
        'label': action.label,
        'role': action.role,
        'bounds': action.bounds,
        'enabled': action.enabled,
        'state': sorted(element.state),
        'source_ref': to_jsonable(action.source_ref),
      }
  return action_map


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


def _compiled_prompt(compiled: dict[str, Any]) -> str:
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


def _sufficiency_report(
    *,
    app_name: str,
    screen: ScreenIR,
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
  has_modal = any(group.get('kind') == 'dialog' and group.get('modal') for group in compiled['groups'])
  modal_required = has_modal
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
      'modal_context_present_if_required': (not modal_required) or has_modal,
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


def _compile_one(
    app_dir: Path,
    output_dir: Path,
    *,
    include_system_ui: bool,
    include_invisible: bool,
) -> dict[str, Any]:
  app_name = app_dir.name
  tree_path = app_dir / 'a11y_tree.txt'
  if not tree_path.is_file():
    raise FileNotFoundError(f'Missing {tree_path}')

  metadata = _read_json(app_dir / 'metadata.json')
  forest = _read_forest(tree_path)
  screen = _forest_to_screen_ir(
      app_name=app_name,
      metadata=metadata,
      forest=forest,
  )
  optimized, analysis = _optimized_screen(
      screen,
      include_system_ui=include_system_ui,
      include_invisible=include_invisible,
  )
  compiled = _compiled_ir(optimized, analysis)
  prompt = _compiled_prompt(compiled)
  action_map = _action_map(optimized)
  sufficiency = _sufficiency_report(
      app_name=app_name,
      screen=optimized,
      compiled=compiled,
  )

  app_output_dir = output_dir / app_name
  _write_text(app_output_dir / 'compiled_prompt.txt', prompt)
  _write_json(app_output_dir / 'screen_ir.json', to_jsonable(screen))
  _write_json(app_output_dir / 'optimized_ir.json', to_jsonable(optimized))
  _write_json(app_output_dir / 'compiled_ir.json', compiled)
  _write_json(app_output_dir / 'action_map.json', action_map)
  _write_json(app_output_dir / 'sufficiency_report.json', sufficiency)

  report = {
      'app_name': app_name,
      'source_tree_path': str(tree_path),
      'raw_surface_count': len(screen.surfaces),
      'raw_element_count': len(screen.elements),
      'live_element_count': len(analysis['live_ids']),
      'context_element_count': len(analysis['context_ids']),
      'action_count': len(optimized.actions),
      'eliminated_element_count': max(0, len(screen.elements) - len(analysis['live_ids'])),
      'group_count': len(optimized.groups),
      'groups': [
          {
              'id': group.id,
              'kind': group.kind,
              'action_count': len(group.action_ids),
          }
          for group in optimized.groups
      ],
      'prompt_chars': len(prompt),
      'sufficiency_status': sufficiency['status'],
  }
  _write_json(app_output_dir / 'compile_report.json', report)
  return {
      'app_name': app_name,
      'status': 'ok',
      'output_dir': str(app_output_dir),
      'summary': report,
  }


def _app_dirs(input_dir: Path, app_names: list[str] | None) -> list[Path]:
  if app_names:
    return [input_dir / app_name for app_name in app_names]
  return sorted(path for path in input_dir.iterdir() if path.is_dir())


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Compile raw A11y forests into source-agnostic UI-state prompts.'
  )
  parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument('--apps', nargs='+', help='App directory names to process.')
  parser.add_argument(
      '--include-system-ui',
      action='store_true',
      help='Keep pure system UI surfaces.',
  )
  parser.add_argument(
      '--include-invisible',
      action='store_true',
      help='Keep invisible elements as live candidates.',
  )
  parser.add_argument('--fail-fast', action='store_true')
  args = parser.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)
  records = []
  for app_dir in _app_dirs(args.input_dir, args.apps):
    try:
      record = _compile_one(
          app_dir,
          args.output_dir,
          include_system_ui=args.include_system_ui,
          include_invisible=args.include_invisible,
      )
      print(f'[{app_dir.name}] ok')
    except Exception as exc:  # pylint: disable=broad-exception-caught
      record = {
          'app_name': app_dir.name,
          'status': 'error',
          'error': f'{type(exc).__name__}: {exc}',
      }
      print(f'[{app_dir.name}] error: {record["error"]}')
      if args.fail_fast:
        raise
    records.append(record)

  manifest = {
      'input_dir': str(args.input_dir),
      'output_dir': str(args.output_dir),
      'include_system_ui': args.include_system_ui,
      'include_invisible': args.include_invisible,
      'records': records,
  }
  _write_json(args.output_dir / 'manifest.json', manifest)
  print(f'wrote {args.output_dir / "manifest.json"}')


if __name__ == '__main__':
  main()
