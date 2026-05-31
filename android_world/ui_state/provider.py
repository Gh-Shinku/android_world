"""UI state providers used by agents."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from android_world.env import interface
from android_world.env import representation_utils
from android_world.ui_state import compiler
from android_world.ui_state import ir


@dataclasses.dataclass
class UiStateView:
  mode: str
  prompt_text: str
  action_map: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
  compiled: ir.CompiledUiState | None = None


class LegacyUiStateProvider:
  """Adapter for existing UIElement-list prompt formatting."""

  mode = 'legacy'

  def __init__(
      self,
      formatter: Callable[
          [list[representation_utils.UIElement], tuple[int, int]], str
      ],
  ):
    self._formatter = formatter

  def build(
      self,
      state: interface.State,
      *,
      screen_size: tuple[int, int],
      app_name: str = '',
      activity: str = '',
  ) -> UiStateView:
    del app_name, activity
    return UiStateView(
        mode=self.mode,
        prompt_text=self._formatter(state.ui_elements, screen_size),
    )


class CompiledUiStateProvider:
  """Builds UI state with the UI State Compiler."""

  mode = 'compiled'

  def __init__(
      self,
      config: compiler.UiStateCompilerConfig | None = None,
  ):
    self._compiler = compiler.UiStateCompiler(config)

  def build(
      self,
      state: interface.State,
      *,
      screen_size: tuple[int, int],
      app_name: str = '',
      activity: str = '',
  ) -> UiStateView:
    del screen_size
    compiled = self._compiler.compile_state(
        state,
        app_name=app_name,
        activity=activity,
        metadata={},
    )
    return UiStateView(
        mode=self.mode,
        prompt_text=compiled.prompt,
        action_map=compiled.action_map,
        compiled=compiled,
    )
