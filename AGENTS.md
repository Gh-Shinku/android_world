# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `android_world/`. Agent implementations are in
`android_world/agents/`, emulator and controller logic in `android_world/env/`,
task definitions and validators in `android_world/task_evals/`, and shared
helpers in `android_world/utils/`.

Primary entry points:
- `run.py`: benchmark suite runs and GUI Agent data collection.
- `minimal_task_runner.py`: single-task experiments.
- `scripts/install_all_apps.py`: install and initialize Android World apps on a running emulator.
- `scripts/export_episode_readable.py`: convert checkpoint `.pkl.gz` episodes into readable JSON/Markdown/assets.
- `scripts/validate_prompt_data_export.py`: validate exported T3A prompt component records.
- `scripts/reconstruct_prompts_from_jsonl.py`: reconstruct readable prompts from prompt JSONL.

Provider examples live in `configs/`, documentation in `docs/`, Android app
assets/tests under `apps/`, `assets/`, and `android_world/google/`.

Generated outputs such as `runs/`, `logs/`, readable episode exports, runtime
prompt dumps, and prompt datasets are local artifacts and should not be
committed unless explicitly sanitized.

## Local Environment

Use the configured miniconda Python environments through `Makefile` variables
rather than assuming `python` on PATH.

Default local variables:
- `ANDROID-WORLD_PYTHON`: Python for Android World collection/eval.
- `NLP_PYTHON`: Python for downstream dataset/KV-analysis scripts.
- `CONSOLE_PORT`: emulator console port, default `5554`.
- `GRPC_PORT`: emulator gRPC port, default `8554`.

If changing commands or adding a repeatable workflow, add or update a Makefile
target so the exact command can be rerun later.

Android environment is in qemu, refer to the following Makefile targets to boot it.

```bash
make emulator             # Start the headless Android emulator
make check-emulator       # Check boot status
```

## Git Commit Guidelines

Recent commits use short imperative summaries, for example
`Add DeepSeek backend and readable episode export tooling`. Keep the subject
line concise and explain behavior changes in the body when needed.
