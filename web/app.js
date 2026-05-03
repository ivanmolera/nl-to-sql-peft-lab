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
  analysisTitle: document.querySelector("#analysis-title"),
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
  latency: "Average model generation time per example, displayed in milliseconds for benchmark runs.",
  eval_loss: "Validation loss reported by the trainer during the post-training evaluation pass.",
  best_epoch: "Training epoch where the best validation loss was observed. Fractional epochs are expected: 2.41 means slightly more than two full passes over the training split.",
  best_step: "Optimizer step of the selected checkpoint with the best validation loss.",
  best_metric: "Best validation loss used to select the final checkpoint. Lower is better.",
  training_time: "External wall-clock measurement around the training loop, collected by the project resource monitor. It is used together with CPU, GPU, and RAM sampling.",
  trainer_runtime: "Internal Hugging Face Trainer train_runtime metric for trainer.train(). It should be close to wall time, but comes from the Trainer logs rather than the external resource monitor.",
  train_steps_per_second: "Training throughput reported by Hugging Face Trainer as optimizer steps completed per second. This is speed, not total steps.",
  estimated_training_cost: "Estimated Vertex AI custom training cost for this run, based on job duration, machine type, GPU count, boot disk, and the versioned pricing table in the repository. It is not a Cloud Billing invoice.",
  cost_per_execution_accuracy: "Estimated training cost divided by execution accuracy percentage points. Lower means more execution accuracy per training dollar.",
  cpu_utilization: "Estimated process CPU utilization during fine-tuning, normalized by available CPU cores.",
  gpu_utilization: "Mean and peak GPU utilization sampled with nvidia-smi during fine-tuning.",
  gpu_memory: "Mean and peak GPU memory allocated during fine-tuning, sampled from the training GPU.",
  ram_usage: "Peak resident memory used by the training process.",
  total_parameters: "Total number of parameters in the base model before PEFT adapters are applied.",
  trainable_parameters: "Number of parameters updated during fine-tuning. For PEFT runs this should be a small fraction of the base model.",
  trainable_ratio: "Percentage of model parameters that were updated during fine-tuning.",
};

const MODEL_PARAMETER_COUNTS = {
  "t5-small": 60_506_624,
  "google-t5/t5-small": 60_506_624,
  "gpt2": 124_439_808,
  "openai-community/gpt2": 124_439_808,
  "smollm2-135m-instruct": 135_000_000,
  "HuggingFaceTB/SmolLM2-135M-Instruct": 135_000_000,
  "qwen2.5-coder-0.5b-instruct": 500_000_000,
  "Qwen/Qwen2.5-Coder-0.5B-Instruct": 500_000_000,
};

const TRAINABLE_PARAMETER_FALLBACKS = {
  "t5-small": {
    qlora: 589_824,
    bitfit: 512,
    "prefix-tuning": 983_040,
    ia3: 43_008,
  },
  "smollm2-135m-instruct": {
    qlora: 921_600,
    bitfit: 0,
    "prefix-tuning": 230_400,
  },
  "qwen2.5-coder-0.5b-instruct": {
    qlora: 1_081_344,
    bitfit: 27_648,
    "prefix-tuning": 122_880,
  },
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

function milliseconds(value) {
  return `${Math.round(Number(value || 0) * 1000)} ms`;
}

function shortName(name) {
  return name.split("/").pop();
}

function displayModelName(model) {
  const rawName = model.base_model_name || model.base_model_id || model.name || model.id || "";
  const baseName = rawName.replace(/\s+\+\s+(QLoRA|BitFit|Prefix Tuning|IA3)$/i, "");
  return shortName(baseName);
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
  els.analysisTitle.textContent = `Comparative Analysis (${data.label || "Benchmark run"})`;
  els.modeDescription.textContent = data.available
    ? data.description || ""
    : `${data.description || ""} · results pending`;
  const models = sortModelsByParameters(data.models || []);
  state.benchmarkModels = models;
  els.source.textContent = data.is_demo
    ? "Demo results"
    : data.source
      ? `Results: ${data.source}`
      : "No results";

  if (!data.available || !models.length) {
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

  els.leaderboard.innerHTML = models
    .map((model) => {
      const metrics = model.metrics;
      const trainerMetrics = trainerMetricsForModel(model, data);
      const hasTrainerMetrics = hasValues(trainerMetrics);
      const score = pct(metrics.execution_accuracy ?? metrics.exact_match);
      return `
        <article class="model-card">
          <header>
            <div>
              <h2>${displayModelName(model)}</h2>
              <small>${model.role}</small>
            </div>
            <div class="score">${score}</div>
          </header>
          ${renderParameterProfile(model, trainerMetrics, mode)}
          ${hasTrainerMetrics ? renderTrainerMetrics(trainerMetrics) : ""}
          ${renderBenchmarkMetrics(metrics)}
          ${hasTrainerMetrics ? renderTrainingResources(trainerMetrics, model.training, metrics) : ""}
        </article>
      `;
    })
    .join("");

  renderChart(els.chartExec, models, "execution_accuracy", pct);
  renderChart(els.chartValid, models, "sql_validity", pct);
  renderLatencyChart(els.chartLatency, models);
  renderBenchmarkDetails(data.benchmark, data.dataset);
  renderRuntime(data.runtime);
}

function renderBenchmarkMetrics(metrics) {
  const rows = [
    metrics.exact_match !== undefined
      ? [metricLabel("exact_match", "Benchmark exact match"), pct(metrics.exact_match)]
      : null,
    metrics.sql_validity !== undefined
      ? [metricLabel("valid_sql", "Benchmark valid SQL"), pct(metrics.sql_validity)]
      : null,
    metrics.execution_accuracy !== undefined
      ? [metricLabel("execution_accuracy", "Benchmark execution"), pct(metrics.execution_accuracy)]
      : null,
    metrics.bleu !== undefined
      ? [metricLabel("bleu", "Benchmark BLEU"), pct(metrics.bleu)]
      : null,
    metrics.rouge_l !== undefined
      ? [metricLabel("rouge_l", "Benchmark ROUGE-L"), pct(metrics.rouge_l)]
      : null,
    metrics.token_f1 !== undefined
      ? [metricLabel("token_f1", "Benchmark Token F1"), pct(metrics.token_f1)]
      : null,
    metrics.latency_seconds_per_example !== undefined
      ? [metricLabel("latency", "Benchmark latency"), milliseconds(metrics.latency_seconds_per_example)]
      : null,
  ].filter(Boolean);

  return rows
    .map(([label, value]) => `<div class="metric-row aux"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderParameterProfile(model, trainerMetrics, mode) {
  const profile = parameterProfileForModel(model, trainerMetrics, mode);
  const trainableValue = profile.trainableParameters === null
    ? profile.trainableFallbackLabel
    : `${compactInteger(profile.trainableParameters)} (${parameterPercent(profile.trainableRatio)})`;
  const trainableClass = profile.trainableParameters === null ? " muted-value" : "";
  return `
    <div class="parameter-block">
      <div class="metric-row parameter"><span>${metricLabel("total_parameters", "Total parameters")}</span><strong>${compactInteger(profile.totalParameters)}</strong></div>
      <div class="metric-row parameter"><span>${metricLabel("trainable_parameters", "Fine-tuned parameters")}</span><strong class="${trainableClass.trim()}">${trainableValue}</strong></div>
    </div>
  `;
}

function parameterProfileForModel(model, trainerMetrics, mode) {
  const parameterMetrics = trainerMetrics?.parameter_metrics || {};
  const catalogTotal = parameterCount(model);
  const artifactTotal = parameterMetrics.total_parameters
    ?? model.parameter_metrics?.total_parameters;
  const totalParameters = Number(
    catalogTotal !== Number.MAX_SAFE_INTEGER
      ? catalogTotal
      : artifactTotal ?? 0,
  );
  const trainerTrainable = parameterMetrics.trainable_parameters
    ?? model.parameter_metrics?.trainable_parameters;

  if (trainerTrainable !== undefined) {
    const trainableParameters = Number(trainerTrainable);
    return {
      totalParameters,
      trainableParameters,
      trainableRatio: totalParameters
        ? trainableParameters / totalParameters
        : parameterMetrics.trainable_parameter_ratio ?? 0,
      source: "trainer",
    };
  }

  if (mode === "zero-shot") {
    return {
      totalParameters,
      trainableParameters: 0,
      trainableRatio: 0,
      source: "base",
    };
  }

  const estimate = estimatedTrainableParameters(model, mode);
  if (estimate !== null) {
    return {
      totalParameters,
      trainableParameters: estimate,
      trainableRatio: totalParameters ? estimate / totalParameters : 0,
      source: "estimate",
    };
  }

  return {
    totalParameters,
    trainableParameters: null,
    trainableRatio: null,
    trainableFallbackLabel: "Pending artifact",
    source: "pending",
  };
}

function trainerMetricsForModel(model, data) {
  const modelTrainerMetrics = model.training?.trainer_eval_metrics;
  if (hasValues(modelTrainerMetrics)) return modelTrainerMetrics;

  const aggregateTrainerMetrics = data.benchmark?.fine_tuning?.trainer_eval_metrics;
  if (data.models?.length === 1 && hasValues(aggregateTrainerMetrics)) {
    return aggregateTrainerMetrics;
  }
  return null;
}

function hasValues(value) {
  return !!value && Object.keys(value).length > 0;
}

function sortModelsByParameters(models) {
  return [...models].sort((left, right) => {
    const byParams = parameterCount(left) - parameterCount(right);
    if (byParams !== 0) return byParams;
    return shortName(left.name || left.id).localeCompare(shortName(right.name || right.id));
  });
}

function parameterCount(model) {
  const candidates = [
    model.id,
    model.name,
    model.base_model_name,
    model.base_model_id,
  ];
  for (const candidate of candidates) {
    const count = MODEL_PARAMETER_COUNTS[candidate];
    if (count !== undefined) return count;
  }
  const text = candidates.filter(Boolean).join(" ").toLowerCase();
  if (text.includes("qwen2.5")) return 500_000_000;
  if (text.includes("smollm2")) return 135_000_000;
  if (text.includes("gpt2")) return 124_439_808;
  if (text.includes("t5-small")) return 60_506_624;
  return Number.MAX_SAFE_INTEGER;
}

function estimatedTrainableParameters(model, mode) {
  const family = modelFamilyKey(model);
  return TRAINABLE_PARAMETER_FALLBACKS[family]?.[mode] ?? null;
}

function modelFamilyKey(model) {
  const text = [
    model.id,
    model.name,
    model.base_model_name,
    model.base_model_id,
  ].filter(Boolean).join(" ").toLowerCase();
  if (text.includes("t5-small")) return "t5-small";
  if (text.includes("smollm2")) return "smollm2-135m-instruct";
  if (text.includes("qwen2.5") || text.includes("qwen2-5")) return "qwen2.5-coder-0.5b-instruct";
  if (text.includes("gpt2")) return "gpt2";
  return "";
}

function renderTrainerMetrics(metrics) {
  const rows = [
    metrics.eval_loss !== undefined
      ? [metricLabel("eval_loss", "Training eval loss"), number(metrics.eval_loss), "aux"]
      : null,
  ].filter(Boolean);

  return rows
    .map(([label, value, className = ""]) => (
      `<div class="metric-row ${className}"><span>${label}</span><strong>${value}</strong></div>`
    ))
    .join("");
}

function renderTrainingResources(metrics, training = {}, benchmarkMetrics = {}) {
  const trainMetrics = metrics.train_metrics || {};
  const resources = metrics.resource_metrics || {};
  const cost = training.cost_estimate || {};
  if (!Object.keys(trainMetrics).length && !Object.keys(resources).length) {
    return "";
  }
  const costPerExecutionPoint = cost.estimated_total_usd !== undefined && benchmarkMetrics.execution_accuracy
    ? Number(cost.estimated_total_usd) / (Number(benchmarkMetrics.execution_accuracy) * 100)
    : null;
  return `
    <div class="resource-block">
      <h4>Training Runtime & Resources</h4>
      <div class="metric-row resource"><span>${metricLabel("best_epoch", "Best eval-loss epoch")}</span><strong>${epochValue(metrics.epoch)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("best_step", "Best eval-loss step")}</span><strong>${integer(stepFromCheckpoint(metrics.best_model_checkpoint) ?? metrics.global_step)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("best_metric", "Best eval loss")}</span><strong>${number(metrics.best_metric ?? metrics.eval_loss)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("training_time", "Fine-tuning wall time")}</span><strong>${minutes(resources.training_wall_time_minutes)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("estimated_training_cost", "Estimated training cost")}</span><strong>${money(cost.estimated_total_usd, cost.estimated_total_eur)}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("cost_per_execution_accuracy", "Cost / execution accuracy point")}</span><strong>${money(costPerExecutionPoint, eurCost(costPerExecutionPoint, cost.usd_to_eur))}</strong></div>
      <div class="metric-row resource"><span>${metricLabel("trainer_runtime", "Trainer runtime")}</span><strong>${minutesFromSeconds(trainMetrics.train_runtime)}</strong></div>
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
          <div class="bar-label" title="${escapeAttr(model.name)}">${displayModelName(model)}</div>
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
          <div class="bar-label" title="${escapeAttr(model.name)}">${displayModelName(model)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${milliseconds(value)}</strong>
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
    ["Selected PEFT technique", fineTuning.technique || benchmark.mode],
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

function epochValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Number(value).toFixed(2)} epochs`;
}

function integer(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function compactInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const numeric = Number(value);
  if (numeric >= 1_000_000_000) return `${(numeric / 1_000_000_000).toFixed(2)}B`;
  if (numeric >= 1_000_000) return `${(numeric / 1_000_000).toFixed(1)}M`;
  if (numeric >= 1_000) return `${(numeric / 1_000).toFixed(1)}K`;
  return numeric.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function parameterPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const numeric = Number(value) * 100;
  if (numeric > 0 && numeric < 0.01) return "<0.01%";
  return `${numeric.toFixed(numeric < 1 ? 2 : 1)}%`;
}

function stepsPerSecond(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Number(value).toFixed(3)} steps/s`;
}

function usd(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `$${Number(value).toFixed(2)}`;
}

function eur(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `€${Number(value).toFixed(2)}`;
}

function money(usdValue, eurValue) {
  if (usdValue === null || usdValue === undefined || Number.isNaN(Number(usdValue))) return "n/a";
  return eurValue === null || eurValue === undefined || Number.isNaN(Number(eurValue))
    ? usd(usdValue)
    : `${usd(usdValue)} · ${eur(eurValue)}`;
}

function eurCost(usdValue, usdToEur) {
  if (usdValue === null || usdValue === undefined || Number.isNaN(Number(usdValue))) return null;
  if (usdToEur === null || usdToEur === undefined || Number.isNaN(Number(usdToEur))) return null;
  return Number(usdValue) * Number(usdToEur);
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
  state.liveModels = sortModelsByParameters(data.models || []);
  renderModelOptions();
}

function renderModelOptions() {
  const models = sortModelsByParameters(liveModelsForSelectedMode());
  els.modelSelect.innerHTML = models
    .map((model) => `<option value="${model.id}">${displayModelName(model)}</option>`)
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
    els.runMeta.textContent = `${displayModelName(result.model)} · ${seconds(result.latency_seconds)}`;

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
