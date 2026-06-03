ANDROID_WORLD_PYTHON ?= /mnt/sda/zyt/miniconda3/envs/android_world/bin/python
CONSOLE_PORT ?= 5554
GRPC_PORT ?= 8554
SCREENSHOT_DIR ?= data/screenshots

# ---- Benchmark ----
RUN_CONFIG ?= configs/runs/default.json
ARGS ?=

# ---- Post-processing ----
RUN_DIR ?= runs/latest
CHECKPOINT ?=
TOKENIZER ?=
BENCHMARK_STATE ?= runs/benchmark_state.txt

.PHONY: help \
	emulator check-emulator screenshot install-apps install-package \
	run run-debug run-compiled \
	summarize export export-fail state-init state-rerun

help:
	@echo "Infrastructure:"
	@echo "  make emulator          Start the headless Android emulator"
	@echo "  make check-emulator    Check whether the emulator has finished booting"
	@echo "  make screenshot        Capture a screenshot from the running emulator"
	@echo "  make install-apps      Install and initialize all Android World apps"
	@echo "  make install-package   Reinstall this project into site-packages"
	@echo ""
	@echo "Benchmark (uses run.py --config):"
	@echo "  make run               Run benchmark with $(RUN_CONFIG)"
	@echo "  make run-debug         Quick debug: 3 tasks, 10-step budget"
	@echo "  make run-compiled      Run with compiled UI state mode"
	@echo ""
	@echo "  Override defaults:  make run RUN_CONFIG=configs/runs/compiled.json"
	@echo "  Pass extra args:    make run ARGS='--first_k_tasks=5 --max_steps=15'"
	@echo ""
	@echo "Post-processing:"
	@echo "  make summarize         Summarize partial run from RUN_DIR"
	@echo "  make export            Export CHECKPOINT to readable JSON/Markdown"
	@echo "  make export-fail       Export readable files for records marked fail"
	@echo "  make state-init        Initialize benchmark state from RUN_DIR"
	@echo "  make state-rerun       Re-run only instances marked fail"
	@echo ""
	@echo "UI-state analysis:  cd ui-state && make help"

## ---- Infrastructure ----

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
		> ./logs/androidworld-emulator-$$(date +%Y%m%d-%H%M%S).log 2>&1

check-emulator:
	@BOOT_STATUS=$$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r'); \
	if [ "$$BOOT_STATUS" = "1" ]; then \
		echo "ok"; \
	else \
		echo "not boot"; \
	fi

screenshot:
	@mkdir -p $(SCREENSHOT_DIR)
	@TS=$$(date +%Y%m%d-%H%M%S); \
	adb exec-out screencap -p > $(SCREENSHOT_DIR)/screenshot_$$TS.png; \
	echo "saved to $(SCREENSHOT_DIR)/screenshot_$$TS.png"

install-apps:
	$(ANDROID_WORLD_PYTHON) scripts/install_all_apps.py \
		--console_port=$(CONSOLE_PORT) \
		--grpc_port=$(GRPC_PORT)

install-package:
	$(ANDROID_WORLD_PYTHON) -m pip install --no-deps --no-build-isolation -e .

## ---- Benchmark ----

run:
	$(ANDROID_WORLD_PYTHON) run.py --config $(RUN_CONFIG) $(ARGS)

run-debug:
	$(ANDROID_WORLD_PYTHON) run.py --config configs/runs/debug.json $(ARGS)

run-compiled:
	$(ANDROID_WORLD_PYTHON) run.py --config configs/runs/compiled.json $(ARGS)

## ---- Post-processing ----

summarize:
	$(ANDROID_WORLD_PYTHON) scripts/summarize_run_results.py \
		--run-dir $(RUN_DIR)

export:
	$(ANDROID_WORLD_PYTHON) scripts/export_episode_readable.py $(CHECKPOINT) \
		$(and $(TOKENIZER),--tokenizer $(TOKENIZER))

export-fail:
	$(ANDROID_WORLD_PYTHON) scripts/export_failed_benchmark_records.py \
		--benchmark-state $(BENCHMARK_STATE) \
		--checkpoint-dir $(RUN_DIR) \
		--missing-ok \
		$(and $(TOKENIZER),--tokenizer $(TOKENIZER))

state-init:
	$(ANDROID_WORLD_PYTHON) scripts/init_benchmark_state.py \
		--run-dir $(RUN_DIR) \
		--output $(BENCHMARK_STATE) \
		--overwrite

state-rerun:
	$(ANDROID_WORLD_PYTHON) run.py --config $(RUN_CONFIG) \
		--benchmark_state=$(BENCHMARK_STATE) $(ARGS)
