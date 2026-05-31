"""Source-agnostic UI state IR."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class SourceRef:
  source_type: str
  source_id: str
  attrs: dict[str, Any]


@dataclasses.dataclass
class ElementIR:
  id: str
  surface_id: str
  parent_id: str | None
  child_ids: list[str]
  role: str
  label: str
  bounds: list[int]
  state: set[str]
  ops: list[str]
  text: str = ''
  description: str = ''
  hint: str = ''
  value: str = ''
  resource_key: str = ''
  source_ref: SourceRef | None = None


@dataclasses.dataclass
class SurfaceIR:
  id: str
  kind: str
  bounds: list[int]
  root_ids: list[str]
  title: str = ''
  modal: bool = False
  z_order: int = 0
  package_names: list[str] = dataclasses.field(default_factory=list)
  source_ref: SourceRef | None = None


@dataclasses.dataclass
class ActionIR:
  id: str
  op: str
  element_id: str
  label: str
  role: str
  bounds: list[int]
  enabled: bool
  source_ref: SourceRef | None = None


@dataclasses.dataclass
class GroupIR:
  id: str
  kind: str
  surface_id: str
  title: str = ''
  body: list[str] = dataclasses.field(default_factory=list)
  element_ids: list[str] = dataclasses.field(default_factory=list)
  action_ids: list[str] = dataclasses.field(default_factory=list)
  modal: bool = False


@dataclasses.dataclass
class ScreenIR:
  version: str
  app_name: str
  activity: str
  surfaces: dict[str, SurfaceIR]
  elements: dict[str, ElementIR]
  actions: dict[str, ActionIR] = dataclasses.field(default_factory=dict)
  groups: list[GroupIR] = dataclasses.field(default_factory=list)
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CompiledUiState:
  screen_ir: ScreenIR
  optimized_ir: ScreenIR
  compiled_ir: dict[str, Any]
  prompt: str
  action_map: dict[str, dict[str, Any]]
  compile_report: dict[str, Any]
  sufficiency_report: dict[str, Any]


def to_jsonable(value: Any, *, include_source_refs: bool = True) -> Any:
  """Converts IR dataclasses into JSON-compatible values."""
  if dataclasses.is_dataclass(value):
    return to_jsonable(
        dataclasses.asdict(value), include_source_refs=include_source_refs
    )
  if isinstance(value, dict):
    result = {}
    for key, item in value.items():
      if key == 'source_ref' and not include_source_refs:
        continue
      converted = to_jsonable(item, include_source_refs=include_source_refs)
      if converted not in ({}, [], None, ''):
        result[key] = converted
    return result
  if isinstance(value, list):
    return [
        converted
        for item in value
        if (converted := to_jsonable(item, include_source_refs=include_source_refs))
        not in ({}, [], None, '')
    ]
  if isinstance(value, set):
    return sorted(value)
  return value
