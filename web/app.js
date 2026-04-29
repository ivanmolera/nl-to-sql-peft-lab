const state = {
  models: [],
  examples: [],
  exampleOffset: 0,
};

const els = {
  source: document.querySelector("#result-source"),
  modeSelect: document.querySelector("#benchmark-mode-select"),
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
  els.modeSelect.innerHTML = data.modes
    .map((mode) => {
      const suffix = mode.available ? "" : " · pendiente";
      return `<option value="${mode.id}">${mode.label}${suffix}</option>`;
    })
    .join("");
}

async function loadBenchmarks() {
  const mode = els.modeSelect.value || "zero-shot";
  const data = await api(`/api/benchmarks?mode=${encodeURIComponent(mode)}`);
  els.modeDescription.textContent = data.available
    ? data.description || ""
    : `${data.description || ""} · resultados pendientes de generar`;
  state.models = data.models;
  els.source.textContent = data.is_demo
    ? "Resultados demo"
    : data.source
      ? `Resultados: ${data.source}`
      : "Sin resultados";

  if (!data.available || !data.models.length) {
    els.leaderboard.innerHTML = `
      <article class="model-card">
        <header>
          <div>
            <h2>${data.label || "Benchmark"}</h2>
            <small>${data.message || "Resultados pendientes de generar"}</small>
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
      return `
        <article class="model-card">
          <header>
            <div>
              <h2>${shortName(model.name)}</h2>
              <small>${model.role}</small>
            </div>
            <div class="score">${pct(metrics.execution_accuracy)}</div>
          </header>
          <div class="metric-row"><span>Exact match</span><strong>${pct(metrics.exact_match)}</strong></div>
          <div class="metric-row"><span>SQL válido</span><strong>${pct(metrics.sql_validity)}</strong></div>
          <div class="metric-row"><span>Latencia</span><strong>${seconds(metrics.latency_seconds_per_example)}</strong></div>
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

function renderChart(target, models, metric, formatter) {
  if (!models.length) {
    target.innerHTML = `<p class="empty-state">Sin datos disponibles.</p>`;
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
    target.innerHTML = `<p class="empty-state">Sin datos disponibles.</p>`;
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
  els.runtimeSummary.textContent = `${runtime.runtime || "runtime"} · ${runtime.device || "device"} · ${runtime.generated_at || "sin timestamp"}`;
  const rows = [
    ["Imagen", runtime.container_image],
    ["Plataforma", runtime.platform],
    ["Python", runtime.python_version],
    ["PyTorch", runtime.torch_version],
    ["CUDA", runtime.cuda_available ? `Sí (${runtime.cuda_version || "n/a"})` : "No"],
    ["Dispositivo", runtime.device],
    ["CPU", runtime.cpu_count],
    ["RAM", runtime.memory_total_gb ? `${runtime.memory_total_gb} GB` : "n/a"],
    ["Cloud Run", cloudRun.revision || cloudRun.service || "n/a"],
  ];
  els.runtimeGrid.innerHTML = rows
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value ?? "n/a"}</dd></div>`)
    .join("");
}

function renderBenchmarkDetails(benchmark = {}, dataset = {}) {
  const generation = benchmark.generation || {};
  const sampleSize = benchmark.sample_size ?? dataset?.sample_size;
  const callsPerModel = benchmark.calls_per_model ?? sampleSize;
  const totalCalls = benchmark.total_model_calls ?? (
    callsPerModel && benchmark.models_evaluated
      ? callsPerModel * benchmark.models_evaluated
      : null
  );
  const generationMode = generation.do_sample === false
    ? "determinista"
    : generation.temperature !== undefined
      ? `temperature ${generation.temperature}`
      : "n/a";

  els.benchmarkSummary.textContent = [
    benchmark.mode || "benchmark",
    benchmark.dataset || dataset?.name || "dataset",
    benchmark.split || dataset?.split || "split",
  ].filter(Boolean).join(" · ");

  const rows = [
    ["Tarea", benchmark.task || "NL-to-SQL"],
    ["Runner", benchmark.runner],
    ["Framework", benchmark.framework || benchmark.planned_framework],
    ["Dataset", benchmark.dataset || dataset?.name],
    ["Split", benchmark.split || dataset?.split],
    ["Muestra", formatNullable(sampleSize)],
    ["Llamadas/modelo", formatNullable(callsPerModel)],
    ["Llamadas totales", formatNullable(totalCalls)],
    ["Modelos", formatNullable(benchmark.models_evaluated)],
    ["Muestreo", benchmark.sample_strategy],
    ["Seed", formatNullable(benchmark.seed)],
    ["Max tokens", formatNullable(benchmark.max_new_tokens)],
    ["Max entrada", formatNullable(benchmark.max_source_length)],
    ["Generación", generationMode],
  ];
  els.benchmarkGrid.innerHTML = rows
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value ?? "n/a"}</dd></div>`)
    .join("");
}

function formatNullable(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  return value;
}

async function loadModels() {
  const data = await api("/api/models");
  els.modelSelect.innerHTML = data.models
    .map((model) => `<option value="${model.id}">${shortName(model.name)}</option>`)
    .join("");
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
  els.prediction.textContent = "-- Ejecuta un modelo para ver la predicción";
  els.runMeta.textContent = "Sin ejecución todavía";
  els.verdict.textContent = "Pendiente";
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
  els.run.textContent = "Generando...";
  els.runMeta.textContent = "Cargando modelo y generando SQL";

  try {
    const result = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        model_id: els.modelSelect.value,
        example_index: example.index,
      }),
    });

    els.prediction.textContent = result.prediction || "-- Sin salida";
    els.reference.textContent = result.reference;
    els.runMeta.textContent = `${shortName(result.model.name)} · ${seconds(result.latency_seconds)}`;

    const ok = result.execution_match;
    els.verdict.textContent = ok ? "Coincide" : "No coincide";
    els.verdict.className = `verdict ${ok ? "ok" : "bad"}`;
    setCheck(els.checkExact, result.exact_match ? "Sí" : "No", result.exact_match);
    setCheck(els.checkValid, result.valid_sql ? "Sí" : "No", result.valid_sql);
    setCheck(els.checkExec, result.execution_match ? "Sí" : "No", result.execution_match);
  } catch (error) {
    els.prediction.textContent = `-- Error: ${error.message}`;
    els.runMeta.textContent = "No se pudo completar la inferencia";
    els.verdict.textContent = "Error";
    els.verdict.className = "verdict bad";
  } finally {
    els.run.disabled = false;
    els.run.textContent = "Generar SQL";
  }
}

function bindEvents() {
  els.modeSelect.addEventListener("change", loadBenchmarks);
  els.exampleSelect.addEventListener("change", renderSelectedExample);
  els.run.addEventListener("click", runGeneration);
  els.reload.addEventListener("click", async () => {
    state.exampleOffset += 12;
    await loadExamples();
  });
}

async function boot() {
  bindEvents();
  await loadBenchmarkModes();
  await Promise.all([loadBenchmarks(), loadModels(), loadExamples()]);
}

boot().catch((error) => {
  els.source.textContent = `Error: ${error.message}`;
});
