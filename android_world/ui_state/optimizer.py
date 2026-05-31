"""Middle-end optimizations for source-agnostic UI State IR."""

from __future__ import annotations

import copy
from typing import Any

from android_world.ui_state import ir


def _has_semantics(element: ir.ElementIR) -> bool:
  return bool(
      element.text
      or element.description
      or element.hint
      or element.value
  )


def _is_action_target(element: ir.ElementIR, include_invisible: bool) -> bool:
  if not include_invisible and 'visible' not in element.state:
    return False
  return bool(element.ops)


def _is_context_target(element: ir.ElementIR, include_invisible: bool) -> bool:
  if not include_invisible and 'visible' not in element.state:
    return False
  return _has_semantics(element)


def _ancestor_ids(
    element: ir.ElementIR,
    elements: dict[str, ir.ElementIR],
) -> list[str]:
  result = []
  parent_id = element.parent_id
  while parent_id and parent_id in elements:
    result.append(parent_id)
    parent_id = elements[parent_id].parent_id
  return result


def descendant_action_count(
    element_id: str,
    elements: dict[str, ir.ElementIR],
    action_element_ids: set[str],
) -> int:
  total = 1 if element_id in action_element_ids else 0
  for child_id in elements[element_id].child_ids:
    if child_id in elements:
      total += descendant_action_count(child_id, elements, action_element_ids)
  return total


def live_element_ids(
    screen: ir.ScreenIR,
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
      live.update(
          element_id for element_id in surface.root_ids if element_id in screen.elements
      )
  return live, action_ids, context_ids


def assign_actions(
    screen: ir.ScreenIR,
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
    screen.actions[action_id] = ir.ActionIR(
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
    elements: dict[str, ir.ElementIR],
    keep_ids: set[str] | None = None,
) -> list[ir.ElementIR]:
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
    screen: ir.ScreenIR,
    surface: ir.SurfaceIR,
    live_ids: set[str],
    action_element_ids: set[str],
    element_to_action_id: dict[str, str],
) -> ir.GroupIR | None:
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
      if node.id not in action_element_ids and _has_semantics(node)
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
  return ir.GroupIR(
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
    screen: ir.ScreenIR,
    surface: ir.SurfaceIR,
    live_ids: set[str],
    action_element_ids: set[str],
    element_to_action_id: dict[str, str],
) -> ir.GroupIR | None:
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
  return ir.GroupIR(
      id='T0',
      kind='toolbar',
      surface_id=surface.id,
      title=title_nodes[0].label if title_nodes else '',
      element_ids=[element.id for element in title_nodes],
      action_ids=[element_to_action_id[element.id] for element in actions],
  )


def detect_groups(
    screen: ir.ScreenIR,
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
    screen: ir.ScreenIR,
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


def optimize(
    screen: ir.ScreenIR,
    *,
    include_system_ui: bool = False,
    include_invisible: bool = False,
) -> tuple[ir.ScreenIR, dict[str, Any]]:
  """Runs UI State middle-end optimizations."""
  optimized = copy.deepcopy(screen)
  live_ids, action_element_ids, context_ids = live_element_ids(
      optimized,
      include_system_ui=include_system_ui,
      include_invisible=include_invisible,
  )
  element_to_action_id = assign_actions(optimized, action_element_ids)
  detect_groups(optimized, live_ids, action_element_ids, element_to_action_id)
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


def has_semantics(element: ir.ElementIR) -> bool:
  return _has_semantics(element)
