ANDROID-WORLD_PYTHON ?= /mnt/sda/zyt/miniconda3/envs/android_world/bin/python
NLP_PYTHON ?= /mnt/sda/zyt/miniconda3/envs/nlp/bin/python

CHECKPOINT ?= runs/run_20260528T214558244817/AudioRecorderRecordAudio_0.pkl.gz
RUN_DIR ?= runs/run_20260601T225149154046
PROMPT_JSONL ?= ./runs/android_world_data.jsonl
UI_STATE_DATASET ?= ui-state/data/processed/prompt_dataset.jsonl
UI_STATE_MODEL ?= /mnt/sda/zyt/models/Llama-3.2-3B-Instruct
UI_STATE_RUN ?= ui-state/runs/20260528_225849
UI_STATE_A11Y_INPUT ?= data/A11y
UI_STATE_IR_OUTPUT ?= data/ui_state_prompt_ir
UI_STATE_IR_APPS ?=
UI_STATE_IR_ARGS ?=
UI_STATE_IR_APP_ARGS = $(if $(strip $(UI_STATE_IR_APPS)),--apps $(UI_STATE_IR_APPS),)
UI_STATE_MODE ?= legacy
BENCHMARK_STATE ?=
BENCHMARK_STATE_PATH = $(if $(strip $(BENCHMARK_STATE)),$(BENCHMARK_STATE),runs/ui_state_benchmark_state.txt)
BENCHMARK_STATE_INIT_FROM ?=
BENCHMARK_EXPORT_DIR ?=
BENCHMARK_EXPORT_DIR_ARG = $(if $(strip $(BENCHMARK_EXPORT_DIR)),--output-dir $(BENCHMARK_EXPORT_DIR),)
BENCHMARK_STATE_ARG = $(if $(strip $(BENCHMARK_STATE)),--benchmark_state=$(BENCHMARK_STATE),)
BENCHMARK_INIT_FROM_ARG = $(if $(strip $(BENCHMARK_STATE_INIT_FROM)),--benchmark_state_init_from=$(BENCHMARK_STATE_INIT_FROM),)
TOKENIZER ?=
TOKENIZER_ARG = $(if $(strip $(TOKENIZER)),--tokenizer $(TOKENIZER),)
bool_true = $(filter true True TRUE 1 yes Yes YES,$(strip $(1)))

TASK ?= ContactsAddContact
FIRST_K_TASKS ?= 0
TASKS_ARG = $(if $(TASK),--tasks=$(TASK),)
N_TASK_COMBINATIONS ?= 1
TASK_RANDOM_SEED ?= 30
FIXED_TASK_SEED ?= false
FIXED_TASK_SEED_ARG = $(if $(call bool_true,$(FIXED_TASK_SEED)),--fixed_task_seed,)
AGENT_NAME ?= t3a_openai_compatible
CHECKPOINT_DIR ?= runs/debug_contacts
PROMPT_DATA_OUT ?= runs/prompts.jsonl
RUNTIME_PROMPT_OUT ?= runs/runtime_prompts.jsonl
LLM_CONFIG ?= configs/bailian.example.json
# LLM_CONFIG ?= configs/deepseek.example.json
CONSOLE_PORT ?= 5554
GRPC_PORT ?= 8554
MAX_STEPS ?= 20

.PHONY: help test prompt-hash prompt-jsonl-smoke \
	reconstruct-prompts export-pkl summarize-run install-package collect \
	benchmark-state-init benchmark-rerun-fail benchmark-export-fail \
	ui-state-dataset ui-state-kv ui-state-kv-resume ui-state-kv-test \
	ui-state-analyze ui-state-analyze-test \
	install-apps emulator check-emulator screenshot estimate-kv collect-a11y \
	ir ir-clean compile-ui-state-ir compile-ui-state-ir-clean compile \
	collect-ui-state compare-ui-state t3a-compiled-smoke

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
	@echo "  make summarize-run        Summarize partial run checkpoints from RUN_DIR"
	@echo "  make benchmark-state-init Initialize success/fail benchmark state from RUN_DIR"
	@echo "  make benchmark-rerun-fail Run only fail records from BENCHMARK_STATE"
	@echo "  make benchmark-export-fail Export readable files for fail records"
	@echo "  make install-package     Reinstall this project into site-packages"
	@echo "  make collect              Run a short live Android World DeepSeek task"
	@echo "  make t3a-compiled-smoke   Run T3A compiled mode with 3 random tasks"
	@echo "  make install-apps          Install and initialize all Android World apps"
	@echo "  make collect-a11y          Collect raw A11y trees and screenshots"
	@echo "  make ir                    Build exploratory UI-state IR candidates"
	@echo "  make compile-ui-state-ir   Compile raw A11y trees into prompt-oriented UI IR"
	@echo "  make collect-ui-state      Run collection with UI State Compiler enabled"
	@echo "  make screenshot            Capture emulator screenshot"

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
	$(ANDROID-WORLD_PYTHON) scripts/export_episode_readable.py $(CHECKPOINT) $(TOKENIZER_ARG)

summarize-run:
	$(ANDROID-WORLD_PYTHON) scripts/summarize_run_results.py \
		--run-dir $(RUN_DIR)

benchmark-state-init:
	$(ANDROID-WORLD_PYTHON) scripts/init_benchmark_state.py \
		--run-dir $(RUN_DIR) \
		--output $(BENCHMARK_STATE_PATH) \
		--overwrite

benchmark-rerun-fail:
	@$(MAKE) collect FIRST_K_TASKS=0 BENCHMARK_STATE=$(BENCHMARK_STATE_PATH)

benchmark-export-fail:
	$(ANDROID-WORLD_PYTHON) scripts/export_failed_benchmark_records.py \
		--benchmark-state $(BENCHMARK_STATE_PATH) \
		--checkpoint-dir $(CHECKPOINT_DIR) \
		--missing-ok \
		$(BENCHMARK_EXPORT_DIR_ARG) \
		$(TOKENIZER_ARG)

install-package:
	$(ANDROID-WORLD_PYTHON) -m pip install --no-deps --no-build-isolation -e .

collect:
	$(ANDROID-WORLD_PYTHON) run.py \
		--suite_family=android_world \
		--first_k_tasks=$(FIRST_K_TASKS) \
		--n_task_combinations=$(N_TASK_COMBINATIONS) \
		--task_random_seed=$(TASK_RANDOM_SEED) \
		$(FIXED_TASK_SEED_ARG) \
		--agent_name=$(AGENT_NAME) \
		--llm_config_path=$(LLM_CONFIG) \
		--console_port=$(CONSOLE_PORT) \
		--grpc_port=$(GRPC_PORT) \
		--max_steps=$(MAX_STEPS) \
		--ui_state_mode=$(UI_STATE_MODE) \
		--prompt_data_out=$(PROMPT_DATA_OUT) \
		--runtime_prompt_out=$(RUNTIME_PROMPT_OUT) \
		$(BENCHMARK_STATE_ARG)

collect-ui-state:
	@$(MAKE) collect UI_STATE_MODE=compiled

compare-ui-state:
	@$(MAKE) collect UI_STATE_MODE=legacy PROMPT_DATA_OUT=runs/prompts_legacy.jsonl RUNTIME_PROMPT_OUT=runs/runtime_prompts_legacy.jsonl
	@$(MAKE) collect UI_STATE_MODE=compiled PROMPT_DATA_OUT=runs/prompts_compiled.jsonl RUNTIME_PROMPT_OUT=runs/runtime_prompts_compiled.jsonl

t3a-compiled-smoke:
	@$(MAKE) collect AGENT_NAME=t3a_openai_compatible UI_STATE_MODE=compiled FIRST_K_TASKS=0

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

SCREENSHOT_DIR ?= data/screenshots
screenshot:
	@mkdir -p $(SCREENSHOT_DIR)
	@TS=$$(date +%Y%m%d-%H%M%S); \
	adb exec-out screencap -p > $(SCREENSHOT_DIR)/screenshot_$$TS.png; \
	echo "saved to $(SCREENSHOT_DIR)/screenshot_$$TS.png"

collect-a11y:
	@$(ANDROID-WORLD_PYTHON) scripts/collect_raw_a11y_trees.py \
		--console_port $(CONSOLE_PORT) \
		--grpc_port $(GRPC_PORT)

ir:
	@$(ANDROID-WORLD_PYTHON) scripts/build_ui_state_ir_candidates.py \
		--model $(UI_STATE_MODEL)

ir-clean:
	@rm -rf data/ui_state_ir_candidates

compile-ui-state-ir:
	@$(ANDROID-WORLD_PYTHON) scripts/compile_ui_state_ir.py --input-dir $(UI_STATE_A11Y_INPUT) --output-dir $(UI_STATE_IR_OUTPUT) $(UI_STATE_IR_APP_ARGS) $(UI_STATE_IR_ARGS)

compile-ui-state-ir-clean:
	@rm -rf $(UI_STATE_IR_OUTPUT)

compile: compile-ui-state-ir
