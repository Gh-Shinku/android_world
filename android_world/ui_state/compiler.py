"""Public UI State Compiler entry points."""

from __future__ import annotations

import dataclasses
from typing import Any

from android_world.env import interface
from android_world.ui_state import backend
from android_world.ui_state import ir
from android_world.ui_state import optimizer
from android_world.ui_state.adapters import a11y


@dataclasses.dataclass(frozen=True)
class UiStateCompilerConfig:
  include_system_ui: bool = False
  include_invisible: bool = False


class UiStateCompiler:
  """Compiles environment State into prompt-oriented UI state."""

  def __init__(self, config: UiStateCompilerConfig | None = None):
    self.config = config or UiStateCompilerConfig()

  def compile_state(
      self,
      state: interface.State,
      *,
      app_name: str = '',
      activity: str = '',
      metadata: dict[str, Any] | None = None,
  ) -> ir.CompiledUiState:
    metadata = metadata or {}
    screen_ir = a11y.forest_to_screen_ir(
        app_name=app_name,
        activity=activity,
        forest=state.forest,
        metadata=metadata,
    )
    return self.compile_screen(screen_ir)

  def compile_screen(self, screen_ir: ir.ScreenIR) -> ir.CompiledUiState:
    optimized_ir, analysis = optimizer.optimize(
        screen_ir,
        include_system_ui=self.config.include_system_ui,
        include_invisible=self.config.include_invisible,
    )
    compiled_ir = backend.compile_ir(optimized_ir, analysis)
    prompt = backend.prompt(compiled_ir)
    action_map = backend.action_map(optimized_ir)
    sufficiency = backend.sufficiency_report(
        app_name=screen_ir.app_name,
        screen=optimized_ir,
        compiled=compiled_ir,
    )
    report = {
        'app_name': screen_ir.app_name,
        'raw_surface_count': len(screen_ir.surfaces),
        'raw_element_count': len(screen_ir.elements),
        'live_element_count': len(analysis['live_ids']),
        'context_element_count': len(analysis['context_ids']),
        'action_count': len(optimized_ir.actions),
        'eliminated_element_count': max(
            0, len(screen_ir.elements) - len(analysis['live_ids'])
        ),
        'group_count': len(optimized_ir.groups),
        'groups': [
            {
                'id': group.id,
                'kind': group.kind,
                'action_count': len(group.action_ids),
            }
            for group in optimized_ir.groups
        ],
        'prompt_chars': len(prompt),
        'sufficiency_status': sufficiency['status'],
    }
    return ir.CompiledUiState(
        screen_ir=screen_ir,
        optimized_ir=optimized_ir,
        compiled_ir=compiled_ir,
        prompt=prompt,
        action_map=action_map,
        compile_report=report,
        sufficiency_report=sufficiency,
    )
