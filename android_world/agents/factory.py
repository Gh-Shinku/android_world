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

"""Agent factory for Android World run configuration."""

from __future__ import annotations

from collections.abc import Callable

from android_world import config as run_config_lib
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import interface
from android_world.ui_state import compiler as ui_state_compiler
from android_world.ui_state import provider as ui_state_provider


AgentFactory = Callable[
    [run_config_lib.AgentConfig, interface.AsyncEnv],
    base_agent.EnvironmentInteractingAgent,
]

_REGISTRY: dict[str, AgentFactory] = {}

MINIWOB_TRANSITION_PAUSE = 0.2
MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


def register_agent(name: str, factory: AgentFactory) -> None:
  if name in _REGISTRY:
    raise ValueError(f'Agent already registered: {name}')
  _REGISTRY[name] = factory


def registered_agent_names() -> tuple[str, ...]:
  return tuple(sorted(_REGISTRY))


def create_agent(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Creates and configures an environment-interacting agent."""
  if config.name not in _REGISTRY:
    raise ValueError(f'Unknown agent: {config.name}')
  agent = _REGISTRY[config.name](config, env)
  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(MINIWOB_ADDITIONAL_GUIDELINES)
  if family and family.startswith('miniwob'):
    agent.transition_pause = MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None
  agent.name = config.name
  return agent


def create_llm_wrapper(
    config: run_config_lib.LLMConfig,
) -> infer.Gpt4Wrapper:
  return infer.Gpt4Wrapper(
      model_name=config.model_name,
      api_key=config.api_key,
      api_key_env=config.api_key_env,
      api_base_url=config.api_base_url,
      max_retry=config.max_retry,
      max_tokens=config.max_tokens,
      temperature=config.temperature,
      extra_body=config.extra_body,
      extra_request_kwargs=config.extra_request_kwargs,
  )


def _compiled_ui_state_provider(
    config: run_config_lib.AgentConfig,
) -> ui_state_provider.CompiledUiStateProvider:
  return ui_state_provider.CompiledUiStateProvider(
      ui_state_compiler.UiStateCompilerConfig(
          include_system_ui=config.ui_state_include_system_ui,
          include_invisible=config.ui_state_include_invisible,
      )
  )


def _use_compiled_ui_state(config: run_config_lib.AgentConfig) -> bool:
  return config.ui_state_mode == 'compiled' or config.name in (
      't3a_ui_state_openai_compatible',
      'm3a_ui_state_openai_compatible',
  )


def _ui_state_provider_for(
    config: run_config_lib.AgentConfig,
) -> ui_state_provider.CompiledUiStateProvider | None:
  if _use_compiled_ui_state(config):
    return _compiled_ui_state_provider(config)
  return None


def _human_agent(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  del config
  return human_agent.HumanAgent(env)


def _random_agent(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  del config
  return random_agent.RandomAgent(env)


def _seeact_agent(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  del config
  return seeact.SeeAct(env)


def _m3a_gemini(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  return m3a.M3A(
      env,
      infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest'),
      ui_state_provider=_ui_state_provider_for(config),
  )


def _t3a_gemini(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  return t3a.T3A(
      env,
      infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest'),
      ui_state_provider=_ui_state_provider_for(config),
  )


def _m3a_gpt4(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  return m3a.M3A(
      env,
      infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'),
      ui_state_provider=_ui_state_provider_for(config),
  )


def _t3a_gpt4(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  return t3a.T3A(
      env,
      infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'),
      ui_state_provider=_ui_state_provider_for(config),
  )


def _m3a_openai_compatible(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  return m3a.M3A(
      env,
      create_llm_wrapper(config.llm),
      ui_state_provider=_ui_state_provider_for(config),
  )


def _t3a_openai_compatible(
    config: run_config_lib.AgentConfig,
    env: interface.AsyncEnv,
) -> base_agent.EnvironmentInteractingAgent:
  return t3a.T3A(
      env,
      create_llm_wrapper(config.llm),
      ui_state_provider=_ui_state_provider_for(config),
  )


register_agent('human_agent', _human_agent)
register_agent('random_agent', _random_agent)
register_agent('seeact', _seeact_agent)
register_agent('m3a_gemini_gcp', _m3a_gemini)
register_agent('t3a_gemini_gcp', _t3a_gemini)
register_agent('m3a_gpt4v', _m3a_gpt4)
register_agent('t3a_gpt4', _t3a_gpt4)
register_agent('m3a_openai_compatible', _m3a_openai_compatible)
register_agent('t3a_openai_compatible', _t3a_openai_compatible)
register_agent('m3a_ui_state_openai_compatible', _m3a_openai_compatible)
register_agent('t3a_ui_state_openai_compatible', _t3a_openai_compatible)
