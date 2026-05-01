const state = {
  benchmarkModels: [],
  liveModels: [],
  examples: [],
  exampleOffset: 0,
  selectedMode: "zero-shot",
};

const els = {
  selectedModeLabel: document.querySelector("#selected-mode-label"),
  version: document.querySelector("#app-version"),
  source: document.querySelector("#result-source"),
  modeTabs: document.querySelector("#benchmark-mode-tabs"),
  modeDescription: document.querySelector("#benchmark-mode-description"),
  leaderboard: document.querySelector("#leaderboard"),
  modelSelect: document.querySelector("#model-select"),
  exampleSelect: document.querySelector("#example-select"),
  question: document.querySelector("#question-text"),
  table: document.querySelector("#example-table"),
  reload: document.querySelector("#reload-examples"),
  run: document.querySelector("#run-button"),
  prediction: document.querySelector("#prediction-sql"),
  reference: document.querySelector("#reference-sql"),
  runMeta: document.querySelector("#run-meta"),
  verdict: document.querySelector("#verdict"),
  checkExact: document.querySelector("#check-exact"),
  checkValid: document.querySelector("#check-valid"),
  checkExec: document.querySelector("#check-exec"),
  chartExec: document.querySelector("#chart-exec"),
  chartValid: document.querySelector("#chart-valid"),
  chartLatency: document.querySelector("#chart-latency"),
  benchmarkSummary: document.querySelector("#benchmark-summary"),
  benchmarkGrid: document.querySelector("#benchmark-grid"),
  runtimeSummary: document.querySelector("#runtime-summary"),
  runtimeGrid: document.querySelector("#runtime-grid"),
};

const METRIC_DEFINITIONS = {
  exact_match: "Percentage of predictions whose normalized SQL exactly matches the reference query.",
  execution_accuracy: "Percentage of predictions that return the same result as the reference SQL when executed against the WikiSQL table.",
  valid_sql: "Percentage of generated outputs that can be parsed and executed as SQL for the example table.",
  bleu: "N-gram overlap between generated SQL and reference SQL. Useful as an auxiliary similarity metric, not as proof of semantic correctness.",
  rouge_l: "Longest-common-subsequence overlap between generated SQL and reference SQL. Useful as an auxiliary sequence similarity metric.",
  token_f1: "Token-level precision and recall combined into an F1 score between the generated SQL and the reference SQL.",
  latency: "Average model generation time per example, measured in seconds.",
  eval_loss: "Validation loss reported by the trainer during the post-training evaluation pass.",
  best_epoch: "Training epoch where the best validation loss was observed. This is the selected checkpoint point, not a proof of absolute convergence.",
  best_step: "Optimizer step of the selected checkpoint with the best validation loss.",
  best_metric: "Best validation loss used to select the final checkpoint. Lower is better.",
  training_time: "Total wall-clock time spent by the fine-tuning job, measured from trainer start to trainer end.",
  train_steps_per_second: "Training throughput reported by Hugging Face Trainer as optimizer steps completed per second. This is speed, not total steps.",
  cpu_utilization: "Estimated process CPU utilization during fine-tuning, normalized by available CPU cores.",
  gpu_utilization: "Mean and peak GPU utilization sampled with nvidia-smi during fine-tuning.",
  gpu_memory: "Mean and peak GPU memory allocated during fine-tuning, sampled from the training GPU.",
  ram_usage: "Peak resident memory used by the training process.",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function pct(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function seconds(value) {
  return `${Number(value || 0).toFixed(2)}s`;
}

function shortName(name) {
  return name.split("/").pop();
}

async function loadBenchmarkModes() {
  const data = await api("/api/benchmark-modes");
  els.modeTabs.innerHTML = data.modes
    .map((mode) => {
      const active = mode.id === state.selectedMode ? " active" : "";
      const unavailable = mode.available ? "" : " pending";
      const status = mode.available ? "Available" : "Pending";
      return `
        <button
          type="button"
          class="mode-tab${active}${unavailable}"
          data-mode="${mode.id}"
          aria-pressed="${mode.id === state.selectedMode}"
          title="${status}: ${mode.description}"
        >
          <span>${mode.label}</span>
          <small>${status}</small>
        </button>
      `;
    })
    .join("");
}

async function loadVersion() {
  const data = await api("/api/version");
  els.version.textContent = `Version ${data.version}`;
}

async function loadBenchmarks() {
  const mode = state.selectedMode || "zero-shot";
  const data = await api(`/api/benchmarks?mode=${encodeURIComponent(mode)}`);
  els.selectedModeLabel.textContent = data.label || "Benchmark run";
  els.modeDescription.textContent = data.available
    ? data.description || ""
    : `${data.description || ""} · results pending`;
  state.benchmarkModels = data.models;
  els.source.textContent = data.is_demo
    ? "Demo results"
    : data.source
      ? `Results: ${data.source}`
      : "No results";

  if (!data.available || !data.models.length) {
    els.leaderboard.innerHTML = `
      <article class="model-card">
        <header>
          <div>
            <h2>${data.label || "Benchmark"}</h2>
            <small>${data.message || "Results pending"}</small>
          </div>
        </header>
      </article>
    `;
    renderChart(els.chartExec, [], "execution_accuracy", pct);
    renderChart(els.chartValid, [], "sql_validity", pct);
    renderLatencyChart(els.chartLatency, []);
    renderBenchmarkDetails(data.benchmark, data.dataset);
    renderRuntime(data.runtime);
    return;
  }

  els.leaderboard.innerHTML = data.models
    .map((model) => {
      const metrics = model.metrics;
      const trainerMetrics = model.training?.trainer_eval_metrics
        || data.benchmark?.fine_tuning?.trainer_eval_metrics
        || null;
      const hasTrainerMetrics = trainerMetrics && trainerMetrics.eval_exact_match !== undefined;
      const score = hasTrainerMetrics
        ? pct(trainerMetrics.eval_exact_match)
        : pct(metrics.execution_accuracy);
      return `
        <article class="model-card">
          <header>
            <div>
              <h2>${shortName(model.name)}</h2>
              <small>${model.role}</small>
            </div>
            <div class="score">${score}</div>
          </header>
          ${hasTrainerMetrics ? renderTrainerMetrics(trainerMetrics) : ""}
          <div class="metric-row aux"><span>${metricLabel("exact_match", "Benchmark exact match")}</span><strong>${pct(metrics.exact_match)}</strong></div>
          <div class="metric-row aux"><span>${metricLabel("execution_accuracy", "Benchmark execution")}</span><strong>${pct(metrics.execution_accuracy)}</strong></div>
          <div class="metric-row aux"><span>${metricLabel("valid_sql", "Benchmark valid SQL")}</span><strong>${pct(metrics.sql_validity)}</strong></div>
          <div class="metric-row aux"><span>${metricLabel("latency", "Benchmark latency")}</span><strong>${seconds(metrics.latency_seconds_per_example)}</strong></div>
          ${hasTrainerMetrics ? renderTrainingResources(trainerMetrics) : ""}
        </article>
      `;
    })
    .join("");

  renderChart(els.chartExec, data.models, "execution_accuracy", pct);
  renderChart(els.chartValid, data.models, "sql_validity", pct);
  renderLatencyChart(els.chartLatency, data.models);
  renderBenchmarkDetails(data.benchmark, data.dataset);
  renderRuntime(data.runtime);
}

function renderTrainerMetrics(metrics) {
  return `
    <div class="metric-row"><span>${metricLabel("exact_match", "Training eval exact match")}</span><strong>${pct(metrics.eval_exact_match)}</strong></div>
    <div class="metric-row"><span>${metricLabel("bleu", "Training eval BLEU")}</span><strong>${pct(metrics.eval_bleu)}</strong></div>
    <div class="metric-row"><span>${metricLabel("rouge_l", "Training eval ROUGE-L")}</span><strong>${pct(metrics.eval_rouge_l)}</strong></div>
    <div class="metric-row"><span>${metricLabel("token_f1", "Training eval Token F1")}</span><strong>${pct(metrics.eval_token_f1)}</strong></div>
    <div class="metric-row aux"><span>${metricLabel("eval_loss", "Training eval loss")}</span><strong>${number(metrics.eval_loss)}</strong></div>
  `;
}

function renderTrainingResources(metrics) {
  const trainMetrics = metrics.train_metrics || {};
  const resources = metrics.resource_metrics || {};
  if (!Object.keys(trainMetrics).length && !Object.keys(resources).length) {
    return "";
  }
  return `
    <div class="resource-block">
      <h4>Training Runtime & Resources</h4>
      <div class="metric-row resource"><span>${metricLabel("best_epoch", "Best eval-loss epoch")}</span><strong>${number(metrics.epoch)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("best_step", "Best eval-loss step")}</span><strong>${integer(stepFromCheckpoint(metrics.best_model_checkpoint) ?? metrics.global_step)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("best_metric", "Best eval loss")}</span><strong>${number(metrics.best_metric ?? metrics.eval_loss)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("training_time", "Fine-tuning wall time")}</span><strong>${minutes(resources.training_wall_time_minutes)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("training_time", "Trainer runtime")}</span><strong>${minutesFromSeconds(trainMetrics.train_runtime)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("train_steps_per_second", "Training speed")}</span><strong>${stepsPerSecond(trainMetrics.train_steps_per_second)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("gpu_utilization", "GPU utilization mean / peak")}</span><strong>${percent(resources.gpu_utilization_mean_percent)} / ${percent(resources.gpu_utilization_peak_percent)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("gpu_memory", "GPU memory mean / peak")}</span><strong>${mb(resources.gpu_memory_used_mean_mb)} / ${mb(resources.gpu_memory_used_peak_mb)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("cpu_utilization", "CPU utilization")}</span><strong>${percent(resources.cpu_utilization_estimated_percent)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("ram_usage", "Peak RAM RSS")}</span><strong>${mb(resources.process_max_rss_mb)}</strong></div>
    </div>
  `;
}

function metricLabel(metricKey, label) {
  const description = METRIC_DEFINITIONS[metricKey];
  if (!description) return label;
  return `<span class="metric-help" tabindex="0" data-tooltip="${escapeAttr(description)}">${label}</span>`;
}

function escapeAttr(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderChart(target, models, metric, formatter) {
  if (!models.length) {
    target.innerHTML = `<p class="empty-state">No data available.</p>`;
    return;
  }
  target.innerHTML = models
    .map((model) => {
      const value = model.metrics[metric] || 0;
      return `
        <div class="bar-row">
          <div class="bar-label" title="${model.name}">${shortName(model.name)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, value * 100)}%"></div></div>
          <strong>${formatter(value)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderLatencyChart(target, models) {
  if (!models.length) {
    target.innerHTML = `<p class="empty-state">No data available.</p>`;
    return;
  }
  const maxLatency = Math.max(
    ...models.map((model) => model.metrics.latency_seconds_per_example || 0.01),
  );
  target.innerHTML = models
    .map((model) => {
      const value = model.metrics.latency_seconds_per_example || 0;
      const width = Math.max(2, (value / maxLatency) * 100);
      return `
        <div class="bar-row">
          <div class="bar-label" title="${model.name}">${shortName(model.name)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${seconds(value)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderRuntime(runtime = {}) {
  const cloudRun = runtime.cloud_run || {};
  els.runtimeSummary.textContent = `${runtime.runtime || "runtime"} · ${runtime.device || "device"} · ${runtime.generated_at || "no timestamp"}`;
  const rows = [
    ["Training service", runtime.training_service],
    ["Region", runtime.region],
    ["Machine type", runtime.machine_type],
    ["Image", runtime.container_image],
    ["Platform", runtime.platform],
    ["Python", runtime.python_version],
    ["PyTorch", runtime.torch_version],
    ["CUDA", runtime.cuda_available ? `Yes (${runtime.cuda_version || "n/a"})` : "No"],
    ["GPU count", formatGpuCount(runtime)],
    ["GPU type", runtime.accelerator_type || runtime.device],
    ["GPU", formatGpu(runtime)],
    ["Device", runtime.device],
    ["CPU", runtime.cpu_count],
    ["RAM", runtime.memory_total_gb ? `${runtime.memory_total_gb} GB` : "n/a"],
    ["Cloud Run", cloudRun.revision || cloudRun.service || "n/a"],
  ];
  els.runtimeGrid.innerHTML = rows
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value ?? "n/a"}</dd></div>`)
    .join("");
}

function formatGpuCount(runtime = {}) {
  return runtime.accelerator_count
    ?? runtime.cuda_device_count
    ?? (runtime.cuda_available ? 1 : 0);
}

function formatGpu(runtime = {}) {
  const count = formatGpuCount(runtime);
  if (!count) return "0";
  const type = runtime.accelerator_type || runtime.device || "GPU";
  return `${count} x ${type}`;
}

function renderBenchmarkDetails(benchmark = {}, dataset = {}) {
  const generation = benchmark.generation || {};
  const fineTuning = benchmark.fine_tuning || {};
  const trainerMetrics = fineTuning.trainer_eval_metrics || {};
  const trainMetrics = trainerMetrics.train_metrics || {};
  const resources = trainerMetrics.resource_metrics || {};
  const sampleSize = benchmark.sample_size ?? dataset?.sample_size;
  const callsPerModel = benchmark.calls_per_model ?? sampleSize;
  const totalCalls = benchmark.total_model_calls ?? (
    callsPerModel && benchmark.models_evaluated
      ? callsPerModel * benchmark.models_evaluated
      : null
  );
  const generationMode = generation.do_sample === false
    ? "deterministic"
    : generation.temperature !== undefined
      ? `temperature ${generation.temperature}`
      : "n/a";

  els.benchmarkSummary.textContent = [
    benchmark.mode || "benchmark",
    benchmark.dataset || dataset?.name || "dataset",
    benchmark.split || dataset?.split || "split",
  ].filter(Boolean).join(" · ");

  const rows = [
    ["Task", benchmark.task || "NL-to-SQL"],
    ["Runner", benchmark.runner],
    ["Framework", benchmark.framework || benchmark.planned_framework],
    ["Dataset", benchmark.dataset || dataset?.name],
    ["Split", benchmark.split || dataset?.split],
    ["Benchmark sample size", formatNullable(sampleSize)],
    ["Benchmark calls/model", formatNullable(callsPerModel)],
    ["Benchmark total calls", formatNullable(totalCalls)],
    ["Fine-tuning", fineTuning.technique],
    ["Training split", fineTuning.train_split],
    ["Training limit", fineTuning.train_limit === null ? "Full split" : formatNullable(fineTuning.train_limit)],
    ["Training examples", formatNullable(fineTuning.train_examples)],
    ["Max training epochs", formatNullable(fineTuning.epochs)],
    ["Training eval examples", formatNullable(fineTuning.eval_examples)],
    [metricLabel("best_epoch", "Best eval-loss epoch"), number(trainerMetrics.epoch)],
    [metricLabel("best_step", "Best eval-loss step"), integer(stepFromCheckpoint(trainerMetrics.best_model_checkpoint) ?? trainerMetrics.global_step)],
    [metricLabel("best_metric", "Best eval loss"), number(trainerMetrics.best_metric ?? trainerMetrics.eval_loss)],
    [metricLabel("training_time", "Fine-tuning wall time"), minutes(resources.training_wall_time_minutes)],
    [metricLabel("training_time", "Trainer runtime"), minutesFromSeconds(trainMetrics.train_runtime)],
    [metricLabel("train_steps_per_second", "Training speed"), stepsPerSecond(trainMetrics.train_steps_per_second)],
    [metricLabel("cpu_utilization", "CPU utilization"), percent(resources.cpu_utilization_estimated_percent)],
    [metricLabel("gpu_utilization", "GPU utilization mean / peak"), `${percent(resources.gpu_utilization_mean_percent)} / ${percent(resources.gpu_utilization_peak_percent)}`],
    [metricLabel("gpu_memory", "GPU memory mean / peak"), `${mb(resources.gpu_memory_used_mean_mb)} / ${mb(resources.gpu_memory_used_peak_mb)}`],
    [metricLabel("ram_usage", "Peak RAM RSS"), mb(resources.process_max_rss_mb)],
    [metricLabel("exact_match", "Trainer eval EM"), trainerMetrics.eval_exact_match !== undefined ? pct(trainerMetrics.eval_exact_match) : null],
    [metricLabel("bleu", "Trainer eval BLEU"), trainerMetrics.eval_bleu !== undefined ? pct(trainerMetrics.eval_bleu) : null],
    [metricLabel("rouge_l", "Trainer eval ROUGE-L"), trainerMetrics.eval_rouge_l !== undefined ? pct(trainerMetrics.eval_rouge_l) : null],
    [metricLabel("token_f1", "Trainer eval Token F1"), trainerMetrics.eval_token_f1 !== undefined ? pct(trainerMetrics.eval_token_f1) : null],
    ["Models", formatNullable(benchmark.models_evaluated)],
    ["Sampling", benchmark.sample_strategy],
    ["Seed", formatNullable(benchmark.seed)],
    ["Max tokens", formatNullable(benchmark.max_new_tokens)],
    ["Max input", formatNullable(benchmark.max_source_length)],
    ["Generation", generationMode],
    ["Aux metrics", "BLEU · ROUGE-L · Token F1"],
  ];
  els.benchmarkGrid.innerHTML = rows
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value ?? "n/a"}</dd></div>`)
    .join("");
}

function formatNullable(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  return value;
}

function number(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(3);
}

function integer(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function stepsPerSecond(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Number(value).toFixed(3)} steps/s`;
}

function stepFromCheckpoint(checkpoint) {
  const match = String(checkpoint || "").match(/checkpoint-(\d+)/);
  return match ? Number(match[1]) : null;
}

function minutes(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return durationFromSeconds(Number(value) * 60);
}

function minutesFromSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return durationFromSeconds(Number(value));
}

function durationFromSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const totalSeconds = Math.max(0, Math.round(Number(value)));
  const hours = Math.floor(totalSeconds / 3600);
  const minutesValue = Math.floor((totalSeconds % 3600) / 60);
  const secondsValue = totalSeconds % 60;
  const paddedMinutes = String(minutesValue).padStart(hours ? 2 : 1, "0");
  const paddedSeconds = String(secondsValue).padStart(2, "0");
  if (hours) return `${hours}h ${paddedMinutes}m ${paddedSeconds}s`;
  if (minutesValue) return `${minutesValue}m ${paddedSeconds}s`;
  return `${secondsValue}s`;
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Number(value).toFixed(1)}%`;
}

function mb(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const numeric = Number(value);
  if (numeric >= 1024) return `${(numeric / 1024).toFixed(2)} GB`;
  return `${numeric.toFixed(0)} MB`;
}

async function loadModels() {
  const data = await api("/api/models");
  state.liveModels = data.models;
  renderModelOptions();
}

function renderModelOptions() {
  const models = liveModelsForSelectedMode();
  els.modelSelect.innerHTML = models
    .map((model) => `<option value="${model.id}">${shortName(model.name)}</option>`)
    .join("");
  els.modelSelect.disabled = models.length === 0;
  els.run.disabled = models.length === 0;
  if (!models.length) {
    els.runMeta.textContent = "No live model available for this technique yet";
  }
}

function liveModelsForSelectedMode() {
  if (state.selectedMode === "zero-shot") {
    return state.liveModels.filter((model) => !model.is_fine_tuned);
  }
  return state.liveModels.filter((model) => model.peft_method === state.selectedMode);
}

async function loadExamples() {
  const data = await api(`/api/examples?limit=12&offset=${state.exampleOffset}`);
  state.examples = data.examples;
  els.exampleSelect.innerHTML = data.examples
    .map((example) => {
      const label = `#${example.index} · ${example.question}`;
      return `<option value="${example.index}">${label}</option>`;
    })
    .join("");
  renderSelectedExample();
}

function selectedExample() {
  const index = Number(els.exampleSelect.value);
  return state.examples.find((example) => example.index === index) || state.examples[0];
}

function renderSelectedExample() {
  const example = selectedExample();
  if (!example) return;

  els.question.textContent = example.question;
  els.reference.textContent = example.reference_sql;
  renderTable(example);
  resetResult();
}

function renderTable(example) {
  const headers = example.columns
    .map((column, index) => `<th>${column}<br><small>${example.types[index] || "text"}</small></th>`)
    .join("");
  const rows = example.sample_rows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell ?? ""}</td>`).join("")}</tr>`)
    .join("");
  els.table.innerHTML = `<thead><tr>${headers}</tr></thead><tbody>${rows}</tbody>`;
}

function resetResult() {
  els.prediction.textContent = "-- Run a model to see the prediction";
  els.runMeta.textContent = "No run yet";
  els.verdict.textContent = "Pending";
  els.verdict.className = "verdict";
  setCheck(els.checkExact, "--", null);
  setCheck(els.checkValid, "--", null);
  setCheck(els.checkExec, "--", null);
}

function setCheck(element, label, ok) {
  element.classList.remove("ok", "bad");
  if (ok === true) element.classList.add("ok");
  if (ok === false) element.classList.add("bad");
  element.querySelector("strong").textContent = label;
}

async function runGeneration() {
  const example = selectedExample();
  if (!example) return;

  els.run.disabled = true;
  els.run.textContent = "Generating...";
  els.runMeta.textContent = "Loading model and generating SQL";

  try {
    const result = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        model_id: els.modelSelect.value,
        example_index: example.index,
        peft_method: state.selectedMode,
      }),
    });

    els.prediction.textContent = result.prediction || "-- No output";
    els.reference.textContent = result.reference;
    els.runMeta.textContent = `${shortName(result.model.name)} · ${seconds(result.latency_seconds)}`;

    const ok = result.execution_match;
    els.verdict.textContent = ok ? "Match" : "Mismatch";
    els.verdict.className = `verdict ${ok ? "ok" : "bad"}`;
    setCheck(els.checkExact, result.exact_match ? "Yes" : "No", result.exact_match);
    setCheck(els.checkValid, result.valid_sql ? "Yes" : "No", result.valid_sql);
    setCheck(els.checkExec, result.execution_match ? "Yes" : "No", result.execution_match);
  } catch (error) {
    els.prediction.textContent = `-- Error: ${error.message}`;
    els.runMeta.textContent = "Inference could not be completed";
    els.verdict.textContent = "Error";
    els.verdict.className = "verdict bad";
  } finally {
    els.run.disabled = false;
    els.run.textContent = "Generate SQL";
  }
}

function bindEvents() {
  els.modeTabs.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-mode]");
    if (!button) return;
    state.selectedMode = button.dataset.mode;
    await loadBenchmarkModes();
    await loadBenchmarks();
    renderModelOptions();
    resetResult();
  });
  els.exampleSelect.addEventListener("change", renderSelectedExample);
  els.run.addEventListener("click", runGeneration);
  els.reload.addEventListener("click", async () => {
    state.exampleOffset += 12;
    await loadExamples();
  });
}

async function boot() {
  bindEvents();
  await loadVersion();
  await loadBenchmarkModes();
  await Promise.all([loadBenchmarks(), loadModels(), loadExamples()]);
}

boot().catch((error) => {
  els.source.textContent = `Error: ${error.message}`;
});
