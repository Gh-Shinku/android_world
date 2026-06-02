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

"""Typed configuration for Android World benchmark runs."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from android_world import registry


@dataclasses.dataclass
class EnvConfig:
  adb_path: str = dataclasses.field(
      default_factory=lambda: os.environ.get('ANDROID_WORLD_ADB_PATH', '')
  )
  perform_emulator_setup: bool = False
  console_port: int = 5554
  grpc_port: int = 8554


@dataclasses.dataclass
class SuiteConfig:
  family: str = registry.TaskRegistry.ANDROID_WORLD_FAMILY
  task_random_seed: int = 30
  tasks: list[str] | None = None
  first_k_tasks: int = 0
  n_task_combinations: int = 1
  max_steps: int | None = None
  fixed_task_seed: bool = False

  def __post_init__(self) -> None:
    if self.max_steps == 0:
      self.max_steps = None


@dataclasses.dataclass
class LLMConfig:
  model_name: str = 'gpt-4-turbo-2024-04-09'
  api_base_url: str = 'https://api.openai.com/v1'
  api_key_env: str = 'OPENAI_API_KEY'
  api_key: str | None = None
  max_retry: int = 3
  max_tokens: int | None = 1000
  temperature: float | None = 0.0
  extra_body: dict[str, Any] = dataclasses.field(default_factory=dict)
  extra_request_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def from_provider_json(cls, path: str | os.PathLike[str]) -> 'LLMConfig':
    with open(path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    if not isinstance(data, dict):
      raise ValueError('LLM provider config must be a JSON object.')
    return cls.from_provider_dict(data)

  @classmethod
  def from_provider_dict(cls, data: dict[str, Any]) -> 'LLMConfig':
    extra_body = data.get('extra_body', {})
    if not isinstance(extra_body, dict):
      raise ValueError('LLM config field "extra_body" must be an object.')
    extra_request_kwargs = data.get('extra_request_kwargs', {})
    if not isinstance(extra_request_kwargs, dict):
      raise ValueError(
          'LLM config field "extra_request_kwargs" must be an object.'
      )
    if extra_request_kwargs.get('stream'):
      raise ValueError('Streaming responses are not supported by Android World.')

    api_key = data.get('api_key')
    if api_key == '':
      api_key = None
    if api_key is not None and not isinstance(api_key, str):
      raise ValueError('LLM config field "api_key" must be a string.')

    temperature = data.get('temperature', 0.0)
    if temperature is not None:
      temperature = float(temperature)
    max_tokens = data.get('max_tokens', 1000)
    if max_tokens is not None:
      max_tokens = int(max_tokens)

    return cls(
        model_name=str(data.get('model', 'gpt-4-turbo-2024-04-09')),
        api_base_url=str(data.get('base_url', 'https://api.openai.com/v1')),
        api_key_env=str(data.get('api_key_env', 'OPENAI_API_KEY')),
        api_key=api_key,
        max_retry=int(data.get('max_retry', 3)),
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=dict(extra_body),
        extra_request_kwargs=dict(extra_request_kwargs),
    )


@dataclasses.dataclass
class AgentConfig:
  name: str = 'm3a_gpt4v'
  llm: LLMConfig = dataclasses.field(default_factory=LLMConfig)
  ui_state_mode: str = 'legacy'
  ui_state_include_system_ui: bool = False
  ui_state_include_invisible: bool = False


@dataclasses.dataclass
class OutputConfig:
  checkpoint_dir: str = ''
  output_path: str = './runs'
  prompt_data_out: str = ''
  runtime_prompt_out: str = ''
  benchmark_state: str = ''
  benchmark_state_init_from: str = ''
  benchmark_state_autosave: bool = True


@dataclasses.dataclass
class RunConfig:
  env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
  suite: SuiteConfig = dataclasses.field(default_factory=SuiteConfig)
  agent: AgentConfig = dataclasses.field(default_factory=AgentConfig)
  output: OutputConfig = dataclasses.field(default_factory=OutputConfig)

  @classmethod
  def from_json(cls, path: str | os.PathLike[str]) -> 'RunConfig':
    with open(path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    if not isinstance(data, dict):
      raise ValueError('Run config must be a JSON object.')
    return cls.from_dict(data)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'RunConfig':
    env = _dataclass_from_dict(EnvConfig, data.get('env', {}))
    suite = _dataclass_from_dict(SuiteConfig, data.get('suite', {}))
    agent_raw = data.get('agent', {})
    if not isinstance(agent_raw, dict):
      raise ValueError('AgentConfig config must be a JSON object.')
    llm_data = agent_raw.get('llm', {})
    llm = _dataclass_from_dict(LLMConfig, llm_data)
    agent_data = dict(agent_raw)
    agent_data['llm'] = llm
    agent = _dataclass_from_dict(AgentConfig, agent_data)
    output = _dataclass_from_dict(OutputConfig, data.get('output', {}))
    return cls(env=env, suite=suite, agent=agent, output=output)

  def to_json(self, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(dataclasses.asdict(self), f, indent=2, sort_keys=True)
      f.write('\n')

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _dataclass_from_dict(cls: type[Any], data: Any) -> Any:
  if not isinstance(data, dict):
    raise ValueError(f'{cls.__name__} config must be a JSON object.')
  field_names = {field.name for field in dataclasses.fields(cls)}
  return cls(**{key: value for key, value in data.items() if key in field_names})
