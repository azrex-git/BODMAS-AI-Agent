/**
 * BODMAS Tutor Agent Frontend Controller
 * Retro Green CRT / HUD interface for Gemma 4 MoE
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const eqInput = document.getElementById("equation-input");
  const charCount = document.getElementById("char-count");
  const statusPill = document.getElementById("status-pill");
  const btnSolve = document.getElementById("btn-solve");
  const btnEval = document.getElementById("btn-eval");
  const btnCloseEval = document.getElementById("btn-close-eval");
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const modelSelect = document.getElementById("model-select");

  // State Screens
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const resultsState = document.getElementById("results-state");
  const evalView = document.getElementById("eval-view");
  const loadingStatusText = document.getElementById("loading-status-text");

  // Telemetry HUD Items
  const telLatency = document.querySelector("#tel-latency .tel-value");
  const telTokens = document.querySelector("#tel-tokens .tel-value");
  const telSchema = document.querySelector("#tel-schema .tel-badge");
  const telModel = document.querySelector("#tel-model .tel-value");

  // Results Components
  const finalAnswerVal = document.getElementById("final-answer-val");
  const stagesMatrixContainer = document.getElementById("stages-matrix-container");
  const stepsFlowContainer = document.getElementById("steps-flow-container");
  const tutorTitle = document.getElementById("tutor-title");
  const tutorOverview = document.getElementById("tutor-overview");
  const tutorStepsList = document.getElementById("tutor-steps-list");
  const tutorEncouragement = document.getElementById("tutor-encouragement");
  const jsonCodeViewer = document.getElementById("json-code-viewer");
  const jsonToggle = document.getElementById("json-toggle");
  const evalTableOutput = document.getElementById("eval-table-output");

  // Settings Modal
  const settingsModal = document.getElementById("settings-modal");
  const btnSettings = document.getElementById("btn-settings");
  const btnCloseSettings = document.getElementById("btn-close-settings");
  const btnSaveKey = document.getElementById("btn-save-key");
  const btnClearKey = document.getElementById("btn-clear-key");
  const apiKeyInput = document.getElementById("api-key-input");
  const keyStatusText = document.getElementById("key-status-text");

  // Local Storage Key Management
  const STORAGE_KEY = "gemini_api_key_override";
  let userApiKey = localStorage.getItem(STORAGE_KEY) || "";
  if (userApiKey) {
    apiKeyInput.value = userApiKey;
  }

  // Check Backend Health
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        if (data.api_key_configured || userApiKey) {
          keyStatusText.textContent = "● GEMINI_API_KEY detected. Gemma 4 active.";
          keyStatusText.style.color = "var(--hud-green)";
        } else {
          keyStatusText.textContent = "○ No API Key set. Running in local emulator mode.";
          keyStatusText.style.color = "var(--hud-text-muted)";
        }
      }
    } catch {
      keyStatusText.textContent = "⚠ Server offline";
    }
  }
  checkHealth();

  // Character Count & Input Status
  eqInput.addEventListener("input", () => {
    const len = eqInput.value.length;
    charCount.textContent = `${len} Chars`;
    statusPill.textContent = len > 0 ? "EXPRESSION READY" : "AWAITING INPUT";
  });

  // Preset Chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      eqInput.value = chip.getAttribute("data-eq");
      eqInput.dispatchEvent(new Event("input"));
      eqInput.focus();
    });
  });

  // Temperature Slider
  tempSlider.addEventListener("input", () => {
    tempVal.textContent = parseFloat(tempSlider.value).toFixed(2);
  });

  // Screen Switcher
  function showView(viewName) {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultsState.classList.add("hidden");
    evalView.classList.add("hidden");

    if (viewName === "empty") emptyState.classList.remove("hidden");
    if (viewName === "loading") loadingState.classList.remove("hidden");
    if (viewName === "results") resultsState.classList.remove("hidden");
    if (viewName === "eval") evalView.classList.remove("hidden");
  }

  // Toggle JSON Viewer
  jsonToggle.addEventListener("click", () => {
    const isHidden = jsonCodeViewer.classList.contains("hidden");
    const indicator = jsonToggle.querySelector(".toggle-indicator");
    if (isHidden) {
      jsonCodeViewer.classList.remove("hidden");
      indicator.textContent = "[COLLAPSE]";
    } else {
      jsonCodeViewer.classList.add("hidden");
      indicator.textContent = "[EXPAND]";
    }
  });

  // Settings Modal Handlers
  btnSettings.addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
  });
  btnCloseSettings.addEventListener("click", () => {
    settingsModal.classList.add("hidden");
  });
  btnSaveKey.addEventListener("click", () => {
    const key = apiKeyInput.value.trim();
    if (key) {
      localStorage.setItem(STORAGE_KEY, key);
      userApiKey = key;
    } else {
      localStorage.removeItem(STORAGE_KEY);
      userApiKey = "";
    }
    settingsModal.classList.add("hidden");
    checkHealth();
  });
  btnClearKey.addEventListener("click", () => {
    apiKeyInput.value = "";
    localStorage.removeItem(STORAGE_KEY);
    userApiKey = "";
    checkHealth();
  });

  // Benchmark Eval Handlers
  btnEval.addEventListener("click", async () => {
    showView("eval");
    evalTableOutput.textContent = "RUNNING GEMMA 4 BENCHMARK EVALUATION HARNESS...\nExecuting 6 Edge-Case Test Vectors...";

    try {
      const res = await fetch("/api/eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: modelSelect.value,
          api_key: userApiKey || undefined,
        }),
      });

      const data = await res.json();
      renderBenchmarkReport(data);
    } catch (err) {
      evalTableOutput.textContent = `BENCHMARK ERROR: ${err.message}`;
    }
  });

  btnCloseEval.addEventListener("click", () => {
    showView("empty");
  });

  function renderBenchmarkReport(data) {
    let report = "";
    report += "==========================================================================================\n";
    report += " ▮ BODMAS TUTOR AGENT: GEMMA 4 BENCHMARK EVALUATION HARNESS ▮\n";
    report += ` Model: ${data.model} | Tests: ${data.total_tests} | Mode: Multi-Stage Pipeline\n`;
    report += "==========================================================================================\n";
    report += `Test ID  | ${"Expression".padEnd(26)} | ${"Expected".padEnd(10)} | ${"Actual".padEnd(10)} | ${"Status".padEnd(8)} | Latency | Tokens\n`;
    report += "------------------------------------------------------------------------------------------\n";

    data.results.forEach((r) => {
      const statusStr = r.passed ? "PASSED" : "FAILED";
      const eqTrunc = (r.equation || "").slice(0, 24).padEnd(26);
      const expTrunc = String(r.expected).slice(0, 9).padEnd(10);
      const actTrunc = String(r.actual).slice(0, 9).padEnd(10);
      const latStr = `${r.latency_ms.toFixed(1)}ms`.padStart(7);
      const tokStr = String(r.tokens).padStart(6);

      report += `${r.id.padEnd(8)} | ${eqTrunc} | ${expTrunc} | ${actTrunc} | ${statusStr.padEnd(8)} | ${latStr} | ${tokStr}\n`;
    });

    report += "------------------------------------------------------------------------------------------\n";
    report += ` SUMMARY: ${data.passed_tests}/${data.total_tests} Passed (${data.pass_percentage}%) | Avg Latency: ${data.avg_latency_ms} ms | Total Tokens: ${data.total_tokens}\n`;
    report += "==========================================================================================\n";

    evalTableOutput.textContent = report;
  }

  // Solve Button Action
  btnSolve.addEventListener("click", async () => {
    const equation = eqInput.value.trim();
    if (!equation) {
      alert("Please enter an arithmetic equation first.");
      eqInput.focus();
      return;
    }

    btnSolve.disabled = true;
    showView("loading");
    loadingStatusText.textContent = "INITIALIZING GEMMA 4 MoE MULTI-STAGE PIPELINE...";

    const s1 = document.getElementById("stage-step-1");
    const s2 = document.getElementById("stage-step-2");
    const s3 = document.getElementById("stage-step-3");
    s1.classList.add("active");
    s2.classList.remove("active");
    s3.classList.remove("active");

    const timer1 = setTimeout(() => {
      s2.classList.add("active");
      loadingStatusText.textContent = "EXECUTING STAGE 2: STEP-BY-STEP ALGEBRAIC REDUCTION...";
    }, 450);

    const timer2 = setTimeout(() => {
      s3.classList.add("active");
      loadingStatusText.textContent = "EXECUTING STAGE 3: FRIENDLY TUTOR EXPLAINER...";
    }, 900);

    try {
      const response = await fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          equation: equation,
          tutor_temperature: parseFloat(tempSlider.value),
          model_id: modelSelect.value,
          api_key: userApiKey || undefined,
        }),
      });

      clearTimeout(timer1);
      clearTimeout(timer2);

      const result = await response.json();
      renderExecutionResult(result);
    } catch (err) {
      clearTimeout(timer1);
      clearTimeout(timer2);
      alert(`API Execution Error: ${err.message}`);
      showView("empty");
    } finally {
      btnSolve.disabled = false;
    }
  });

  // Render Execution Result
  function renderExecutionResult(data) {
    showView("results");

    // Update Telemetry Bar
    const tel = data.telemetry || {};
    telLatency.textContent = `${tel.total_latency_ms || 0} ms`;
    telTokens.textContent = `${tel.total_tokens || 0} (In: ${tel.input_tokens || 0} | Out: ${tel.output_tokens || 0})`;
    telModel.textContent = tel.model_used || modelSelect.value;

    if (tel.schema_valid) {
      telSchema.textContent = "VALID JSON";
      telSchema.style.borderColor = "var(--hud-green)";
      telSchema.style.color = "var(--hud-green)";
    } else {
      telSchema.textContent = "NON-STANDARD";
      telSchema.style.borderColor = "var(--hud-red)";
      telSchema.style.color = "var(--hud-red)";
    }

    // Populate JSON Viewer
    jsonCodeViewer.textContent = JSON.stringify(data, null, 2);

    // Guardrail / Syntax Rejection Case
    if (data.status === "invalid_input") {
      finalAnswerVal.textContent = "SYNTAX ERROR";
      finalAnswerVal.style.color = "var(--hud-red)";
      stepsFlowContainer.innerHTML = `
        <div class="step-card" style="border-color: var(--hud-red)">
          <div class="step-header">
            <span class="rule-badge" style="background: var(--hud-red); color: #fff;">STAGE 1 GUARDRAIL</span>
            <span class="step-index">HALTED</span>
          </div>
          <div class="step-equation-row">
            <span style="color: #ff9999;">Equation rejected: ${data.clean_equation || eqInput.value}</span>
          </div>
          <div style="color: #ffb3b3; font-size: 11px; margin-top: 4px;">
            Reason: ${data.error || "Expression does not conform to valid BODMAS syntax."}
          </div>
        </div>
      `;
      tutorTitle.textContent = "Equation Needs Correction";
      tutorOverview.textContent =
        "The parser detected a mathematical syntax issue. Please review the parentheses and operators before recalculating.";
      tutorStepsList.innerHTML = "";
      tutorEncouragement.textContent = "Tip: Make sure all brackets are closed and operators have numbers on both sides!";
      return;
    }

    // Valid Algebraic Result
    finalAnswerVal.textContent = data.final_answer || "--";
    finalAnswerVal.style.color = "var(--hud-green)";

    // Render Steps Flow
    stepsFlowContainer.innerHTML = "";
    const steps = data.reduction_steps || [];
    if (steps.length === 0) {
      stepsFlowContainer.innerHTML = `<div class="step-card"><div>No reduction steps required.</div></div>`;
    } else {
      steps.forEach((step, idx) => {
        const card = document.createElement("div");
        card.className = "step-card";
        card.innerHTML = `
          <div class="step-header">
            <span class="rule-badge">[${step.rule || "BODMAS"}]</span>
            <span class="step-index">STEP ${step.step_index || idx + 1} OF ${steps.length} // PASS #${idx + 1}</span>
          </div>
          <div class="step-equation-row">
            <span>Evaluating Target:</span>
            <span class="focus-highlight">${step.focus || ""}</span>
            <span class="step-result-badge">➔ ${step.result || ""}</span>
          </div>
          <div style="color: var(--hud-text-muted); font-size: 11.5px; margin-top: 6px;">
            Rewritten Equation: <span style="color: #fff; font-weight: 700;">${step.current_equation || ""}</span>
          </div>
        `;
        stepsFlowContainer.appendChild(card);
      });
    }

    // Render Tutor Section
    const tutor = data.tutor_explanation || {};
    tutorTitle.textContent = tutor.title || `BODMAS Solution: ${data.clean_equation}`;
    tutorOverview.textContent = tutor.explanation || tutor.overview || "Systematic reduction applying the order of operations.";
    tutorEncouragement.textContent = tutor.encouragement || "Great work! Applying BODMAS guarantees correct answers every time.";

    tutorStepsList.innerHTML = "";
    if (tutor.tip) {
      const tipDiv = document.createElement("div");
      tipDiv.className = "tutor-tip";
      tipDiv.innerHTML = `💡 TUTOR TIP: ${tutor.tip}`;
      tutorStepsList.appendChild(tipDiv);
    }
    const explanations = tutor.step_explanations || [];
    explanations.forEach((exp) => {
      const expDiv = document.createElement("div");
      expDiv.className = "tutor-step-item";
      expDiv.innerHTML = `
        <div class="tutor-step-heading">${exp.heading || `Step ${exp.step_index}`}</div>
        <div class="tutor-step-text">${exp.explanation || ""}</div>
        ${exp.tip ? `<div class="tutor-tip">💡 TIP: ${exp.tip}</div>` : ""}
      `;
      tutorStepsList.appendChild(expDiv);
    });
  }
});
