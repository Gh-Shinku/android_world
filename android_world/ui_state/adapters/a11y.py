"""Android accessibility forest adapter for UI State IR."""

from __future__ import annotations

from typing import Any

from android_world.ui_state import ir


SYSTEM_UI_PACKAGE = 'com.android.systemui'


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


def _state_from_node(node: Any) -> set[str]:
  result = set()
  for name, enabled in (
      ('checkable', node.is_checkable),
      ('checked', node.is_checked),
      ('enabled', node.is_enabled),
      ('focusable', node.is_focusable),
      ('focused', node.is_focused),
      ('password', node.is_password),
      ('selected', node.is_selected),
      ('visible', node.is_visible_to_user),
  ):
    if enabled:
      result.add(name)
  return result


def _ops_from_node(node: Any) -> list[str]:
  ops = []
  class_name = _short_class_name(node.class_name).lower()
  resource_key = _short_resource_name(
      node.view_id_resource_name, node.package_name
  )
  if node.is_editable:
    ops.append('input_text')
  if node.is_scrollable:
    ops.append('scroll')
  is_ambiguous_text_click = (
      'text' in class_name
      and node.is_clickable
      and not node.is_checkable
      and not node.content_description
      and resource_key.startswith('txt_')
  )
  if (node.is_clickable or node.is_checkable) and not is_ambiguous_text_click:
    ops.append('click')
  if node.is_long_clickable:
    ops.append('long_press')
  return ops


def _role_from_node(node: Any) -> str:
  class_name = _short_class_name(node.class_name).lower()
  if node.is_editable:
    return 'input'
  if node.is_scrollable:
    return 'scroll'
  if 'button' in class_name:
    return 'button'
  if 'text' in class_name:
    return 'text'
  if node.is_clickable:
    return 'button'
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


def _label(
    *,
    text: str,
    description: str,
    hint: str,
    resource_key: str,
    role: str,
) -> str:
  if not text and resource_key.startswith('btn_'):
    return resource_key
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


def forest_to_screen_ir(
    *,
    app_name: str,
    activity: str,
    forest: Any,
    metadata: dict[str, Any] | None = None,
) -> ir.ScreenIR:
  """Converts an AndroidAccessibilityForest into source-agnostic ScreenIR."""
  metadata = metadata or {}
  surfaces: dict[str, ir.SurfaceIR] = {}
  elements: dict[str, ir.ElementIR] = {}
  node_to_element_id: dict[tuple[str, int], str] = {}
  parent_by_element_id: dict[str, str] = {}

  for surface_index, window in enumerate(forest.windows):
    source_window_id = int(window.id)
    surface_id = f's{surface_index}'
    package_names = sorted({
        node.package_name for node in window.tree.nodes if node.package_name
    })
    window_type = _window_type_name(window)
    for node in window.tree.nodes:
      node_to_element_id[(surface_id, int(node.unique_id))] = (
          f'e{len(node_to_element_id)}'
      )
    root_ids = [
        node_to_element_id[(surface_id, source_id)]
        for source_id in _root_node_ids(window)
        if (surface_id, source_id) in node_to_element_id
    ]
    surfaces[surface_id] = ir.SurfaceIR(
        id=surface_id,
        kind=_surface_kind(window_type, package_names),
        bounds=_rect_to_list(window.bounds_in_screen),
        root_ids=root_ids,
        z_order=surface_index,
        package_names=package_names,
        source_ref=ir.SourceRef(
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
      role = _role_from_node(node)
      resource_key = _short_resource_name(
          node.view_id_resource_name, node.package_name
      )
      text = node.text or ''
      description = node.content_description or ''
      hint = node.hint_text or ''
      elements[element_id] = ir.ElementIR(
          id=element_id,
          surface_id=surface_id,
          parent_id=None,
          child_ids=child_ids,
          role=role,
          label=_label(
              text=text,
              description=description,
              hint=hint,
              resource_key=resource_key,
              role=role,
          ),
          bounds=_rect_to_list(node.bounds_in_screen),
          state=_state_from_node(node),
          ops=_ops_from_node(node),
          text=text,
          description=description,
          hint=hint,
          resource_key=resource_key,
          source_ref=ir.SourceRef(
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

  return ir.ScreenIR(
      version='screen_ir_v1',
      app_name=app_name,
      activity=activity,
      surfaces=surfaces,
      elements=elements,
      metadata=dict(metadata),
  )
