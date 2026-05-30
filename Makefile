ANDROID-WORLD_PYTHON ?= /mnt/sda/zyt/miniconda3/envs/android_world/bin/python
NLP_PYTHON ?= /mnt/sda/zyt/miniconda3/envs/nlp/bin/python

CHECKPOINT ?= runs/run_20260528T214558244817/AudioRecorderRecordAudio_0.pkl.gz
PROMPT_JSONL ?= ./runs/android_world_data.jsonl
UI_STATE_DATASET ?= ui-state/data/processed/prompt_dataset.jsonl
UI_STATE_MODEL ?= /mnt/sda/zyt/models/Llama-3.2-3B-Instruct
UI_STATE_RUN ?= ui-state/runs/20260528_225849

# TASK ?= ContactsAddContact
FIRST_K_TASKS ?= 0
TASKS_ARG = $(if $(TASK),--tasks=$(TASK),)
CHECKPOINT_DIR ?= runs/debug_contacts
PROMPT_DATA_OUT ?= runs/prompts.jsonl
LLM_CONFIG ?= configs/bailian.example.json
# LLM_CONFIG ?= configs/deepseek.example.json
CONSOLE_PORT ?= 5554
GRPC_PORT ?= 8554
MAX_STEPS ?= 20

.PHONY: help test prompt-hash prompt-jsonl-smoke \
	reconstruct-prompts export-pkl install-package collect \
	ui-state-dataset ui-state-kv ui-state-kv-resume ui-state-kv-test \
	ui-state-analyze ui-state-analyze-test \
	install-apps emulator check-emulator estimate-kv collect-a11y

help:
	@echo "Targets:"
	@echo "  make prompt-hash           Verify exported prompt components rebuild hashes"
	@echo "  make prompt-jsonl-smoke    Write one prompt JSONL item from CHECKPOINT"
	@echo "  make reconstruct-prompts   Rebuild readable prompts from PROMPT_DATA_OUT"
	@echo "  make ui-state-dataset      Build ui-state prompt dataset from PROMPT_DATA_OUT"
	@echo "  make ui-state-kv           Run Llama KV capture over ui-state dataset"
	@echo "  make ui-state-kv-resume    Resume latest ui-state KV capture run"
	@echo "  make ui-state-kv-test      Run one-sample ui-state KV smoke test"
	@echo "  make ui-state-analyze      Run Phase 2 KV reuse analysis"
	@echo "  make ui-state-analyze-test Run one-pair Phase 2 analysis smoke test"
	@echo "  make export-pkl Convert CHECKPOINT to readable JSON/Markdown"
	@echo "  make install-package     Reinstall this project into site-packages"
	@echo "  make collect    Run a short live Android World DeepSeek task"
	@echo "  make install-apps          Install and initialize all Android World apps"

estimate-kv:
	$(NLP_PYTHON) scripts/estimate_kv_cache_size.py \
		$(UI_STATE_MODEL) scripts/ui.txt

prompt-hash:
	$(ANDROID-WORLD_PYTHON) scripts/validate_prompt_data_export.py \
		$(CHECKPOINT) \
		--mode hash

prompt-jsonl-smoke:
	$(ANDROID-WORLD_PYTHON) scripts/validate_prompt_data_export.py \
		$(CHECKPOINT) \
		--mode jsonl \
		--output $(PROMPT_JSONL)

reconstruct-prompts:
	$(ANDROID-WORLD_PYTHON) scripts/reconstruct_prompts_from_jsonl.py \
		$(PROMPT_DATA_OUT) \
		--output runs/reconstructed_prompts.txt

ui-state-dataset:
	$(ANDROID-WORLD_PYTHON) ui-state/scripts/build_prompt_dataset.py \
		--input $(PROMPT_DATA_OUT) \
		--output-dir ui-state/data/processed

ui-state-kv:
	$(NLP_PYTHON) ui-state/scripts/run_kv_capture.py \
		--dataset $(UI_STATE_DATASET) \
		--model $(UI_STATE_MODEL)

ui-state-kv-resume:
	$(NLP_PYTHON) ui-state/scripts/run_kv_capture.py \
		--resume

ui-state-kv-test:
	$(NLP_PYTHON) ui-state/scripts/run_kv_capture.py \
		--dataset $(UI_STATE_DATASET) \
		--model $(UI_STATE_MODEL) \
		--max-samples 10

ui-state-analyze:
	$(NLP_PYTHON) ui-state/scripts/analyze_kv_reuse.py \
		--run-dir $(UI_STATE_RUN)

ui-state-analyze-test:
	$(NLP_PYTHON) ui-state/scripts/analyze_kv_reuse.py \
		--run-dir $(UI_STATE_RUN) \
		--max-pairs 1

export-pkl:
	$(ANDROID-WORLD_PYTHON) scripts/export_episode_readable.py $(CHECKPOINT)

install-package:
	$(ANDROID-WORLD_PYTHON) -m pip install --no-deps --no-build-isolation -e .

collect:
	$(ANDROID-WORLD_PYTHON) run.py \
		--suite_family=android_world \
		--first_k_tasks=$(FIRST_K_TASKS) \
		--agent_name=t3a_openai_compatible \
		--llm_config_path=$(LLM_CONFIG) \
		--console_port=$(CONSOLE_PORT) \
		--grpc_port=$(GRPC_PORT) \
		--max_steps=$(MAX_STEPS) \
		--prompt_data_out=$(PROMPT_DATA_OUT) \
		--runtime_prompt_out=runs/runtime_prompts.jsonl

install-apps:
	@$(ANDROID-WORLD_PYTHON) scripts/install_all_apps.py \
		--console_port=$(CONSOLE_PORT) \
		--grpc_port=$(GRPC_PORT)

emulator:
	@mkdir -p logs
	@xvfb-run -a emulator \
		-avd AndroidWorldAvd \
		-no-snapshot \
		-no-metrics \
		-no-window \
		-gpu swiftshader_indirect \
		-ports 5554,5555 \
		-grpc $(GRPC_PORT) \
		-verbose \
		> ./logs/androidworld-emulator-$(shell date +%Y%m%d-%H%M%S).log 2>&1

check-emulator:
	@BOOT_STATUS=$$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r'); \
	if [ "$$BOOT_STATUS" = "1" ]; then \
		echo "ok"; \
	else \
		echo "not boot"; \
	fi

collect-a11y:
	@$(ANDROID-WORLD_PYTHON) scripts/collect_raw_a11y_trees.py \
    --console_port 5554 \
    --grpc_port 8554
