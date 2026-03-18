/**
 * DMsim Frontend Application
 * Step-by-step wizard for configuring and running decision market simulations.
 */

const API_BASE = "";

// ── State ────────────────────────────────────────────────────────────────────

const state = {
  currentStep: 1,
  personas: [],
};

// ── DOM Helpers ──────────────────────────────────────────────────────────────

function $(selector) { return document.querySelector(selector); }
function $$(selector) { return document.querySelectorAll(selector); }

// ── Stepper Navigation ──────────────────────────────────────────────────────

function goToStep(step) {
  state.currentStep = step;

  // Update step panels
  $$(".step").forEach(function (el) { el.classList.remove("active"); });
  var stepEl = $("#step-" + step);
  if (stepEl) stepEl.classList.add("active");

  // Update stepper indicators
  $$(".stepper-step").forEach(function (el) {
    var s = parseInt(el.getAttribute("data-step"));
    el.classList.remove("active", "completed");
    if (s === step) el.classList.add("active");
    else if (s < step) el.classList.add("completed");
  });
}

// ── Persona Management ──────────────────────────────────────────────────────

function addPersona(name, description) {
  state.personas.push({
    name: name || "",
    description: description || "",
  });
  renderPersonas();
}

function removePersona(index) {
  state.personas.splice(index, 1);
  renderPersonas();
}

function updatePersona(index, field, value) {
  state.personas[index][field] = value;
}

function renderPersonas() {
  var list = $("#persona-list");
  list.innerHTML = "";

  state.personas.forEach(function (persona, i) {
    var card = document.createElement("div");
    card.className = "persona-card";
    card.setAttribute("data-index", i);
    card.innerHTML =
      '<div class="persona-header">' +
        '<span class="persona-number">Agent ' + (i + 1) + '</span>' +
        '<button class="btn btn-danger remove-persona-btn" data-index="' + i + '">Remove</button>' +
      '</div>' +
      '<div class="form-group">' +
        '<label>Name / Role</label>' +
        '<input type="text" class="persona-name" data-index="' + i + '" value="' + escapeAttr(persona.name) + '" placeholder="e.g., Conflicted Council Member">' +
      '</div>' +
      '<div class="form-group">' +
        '<label>Personality &amp; Background</label>' +
        '<textarea class="persona-desc" data-index="' + i + '" rows="3" placeholder="Describe this agent\'s holdings, biases, knowledge, and perspective...">' + escapeHtml(persona.description) + '</textarea>' +
      '</div>';
    list.appendChild(card);
  });

  // Re-bind events
  $$(".remove-persona-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      removePersona(parseInt(btn.getAttribute("data-index")));
    });
  });

  $$(".persona-name").forEach(function (input) {
    input.addEventListener("input", function () {
      updatePersona(parseInt(input.getAttribute("data-index")), "name", input.value);
    });
  });

  $$(".persona-desc").forEach(function (input) {
    input.addEventListener("input", function () {
      updatePersona(parseInt(input.getAttribute("data-index")), "description", input.value);
    });
  });
}

// ── Results Rendering ───────────────────────────────────────────────────────

function renderLoading() {
  $("#results-content").innerHTML =
    '<div class="loading-overlay">' +
      '<div class="spinner"></div>' +
      '<div class="loading-text">Generating agent profiles via LLM...</div>' +
    '</div>';
}

function renderError(message) {
  $("#results-content").innerHTML =
    '<div class="error-box">' + escapeHtml(message) + '</div>';
}

function renderResults(profiles) {
  if (!profiles || profiles.length === 0) {
    renderError("No profiles were generated.");
    return;
  }

  var html = '<div class="panel"><h2 class="panel-title">Generated Agent Profiles</h2>';
  html += '<div style="overflow-x:auto;">';
  html += '<table class="results-table">';
  html += '<thead><tr>' +
    '<th>Agent</th>' +
    '<th style="text-align:right;" class="val-action-a">&theta;<sub>A</sub></th>' +
    '<th style="text-align:right;" class="val-action-b">&theta;<sub>B</sub></th>' +
    '<th style="text-align:right;" class="val-action-a">p<sub>A</sub></th>' +
    '<th style="text-align:right;" class="val-action-b">p<sub>B</sub></th>' +
    '</tr></thead>';
  html += '<tbody>';

  var sumTA = 0, sumTB = 0, sumPA = 0, sumPB = 0;

  profiles.forEach(function (p, i) {
    sumTA += p.theta_A;
    sumTB += p.theta_B;
    sumPA += p.p_A;
    sumPB += p.p_B;

    html += '<tr>' +
      '<td class="agent-name">' + escapeHtml(p.name) + '</td>' +
      '<td style="text-align:right;"><input class="cell-edit val-action-a" data-idx="' + i + '" data-field="theta_A" value="' + p.theta_A.toFixed(1) + '"></td>' +
      '<td style="text-align:right;"><input class="cell-edit val-action-b" data-idx="' + i + '" data-field="theta_B" value="' + p.theta_B.toFixed(1) + '"></td>' +
      '<td style="text-align:right;"><input class="cell-edit val-action-a" data-idx="' + i + '" data-field="p_A" value="' + p.p_A.toFixed(3) + '"></td>' +
      '<td style="text-align:right;"><input class="cell-edit val-action-b" data-idx="' + i + '" data-field="p_B" value="' + p.p_B.toFixed(3) + '"></td>' +
      '</tr>';
    html += '<tr><td colspan="5" class="rationale-text">' + escapeHtml(p.rationale) + '</td></tr>';
  });

  var n = profiles.length;
  html += '<tr class="summary-row">' +
    '<td>Mean (Consensus)</td>' +
    '<td style="text-align:right;" class="val-action-a" id="mean-theta-a">' + (sumTA / n).toFixed(1) + '</td>' +
    '<td style="text-align:right;" class="val-action-b" id="mean-theta-b">' + (sumTB / n).toFixed(1) + '</td>' +
    '<td style="text-align:right;" class="val-action-a" id="mean-p-a">' + (sumPA / n).toFixed(3) + '</td>' +
    '<td style="text-align:right;" class="val-action-b" id="mean-p-b">' + (sumPB / n).toFixed(3) + '</td>' +
    '</tr>';

  html += '</tbody></table></div></div>';

  $("#results-content").innerHTML = html;

  // Bind editable cells to sync with lastProfiles
  $$(".cell-edit").forEach(function (input) {
    input.addEventListener("change", function () {
      var idx = parseInt(input.getAttribute("data-idx"));
      var field = input.getAttribute("data-field");
      var val = parseFloat(input.value);
      if (!isNaN(val) && lastProfiles && lastProfiles[idx]) {
        lastProfiles[idx][field] = val;
        updateMeans();
      }
    });
  });
}

function updateMeans() {
  if (!lastProfiles || lastProfiles.length === 0) return;
  var n = lastProfiles.length;
  var sTA = 0, sTB = 0, sPA = 0, sPB = 0;
  lastProfiles.forEach(function (p) {
    sTA += p.theta_A; sTB += p.theta_B;
    sPA += p.p_A; sPB += p.p_B;
  });
  var el;
  el = $("#mean-theta-a"); if (el) el.textContent = (sTA / n).toFixed(1);
  el = $("#mean-theta-b"); if (el) el.textContent = (sTB / n).toFixed(1);
  el = $("#mean-p-a"); if (el) el.textContent = (sPA / n).toFixed(3);
  el = $("#mean-p-b"); if (el) el.textContent = (sPB / n).toFixed(3);
}

function valClass(v) {
  if (v > 0) return "val-positive";
  if (v < 0) return "val-negative";
  return "val-neutral";
}

function beliefClass(v) {
  return v > 0.5 ? "val-belief-high" : "val-belief-low";
}

// ── VCGR Simulation ─────────────────────────────────────────────────────────

var lastProfiles = null;

function showVCGRPanel() {
  var panel = $("#vcgr-panel");
  if (panel) panel.style.display = "";
}

function runVCGR() {
  if (!lastProfiles || lastProfiles.length === 0) return;

  var budgetVal = parseFloat($("#vcgr-budget").value);
  var deltaVal = parseFloat($("#vcgr-delta").value);
  var budget = isNaN(budgetVal) ? 50 : budgetVal;
  var delta = isNaN(deltaVal) ? 1 : deltaVal;

  var body = {
    profiles: lastProfiles.map(function (p) {
      return {
        name: p.name,
        theta_A: p.theta_A,
        theta_B: p.theta_B,
        p_A: p.p_A,
        p_B: p.p_B,
      };
    }),
    budget: budget,
    delta: delta,
  };

  $("#vcgr-results").innerHTML =
    '<div class="loading-overlay" style="padding:1.5rem;">' +
      '<div class="spinner"></div>' +
      '<div class="loading-text">Running VCGR simulation...</div>' +
    '</div>';

  fetch(API_BASE + "/api/vcgr-simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(function (res) {
      if (!res.ok) {
        return res.json().then(function (data) {
          throw new Error(data.detail || "Server error " + res.status);
        });
      }
      return res.json();
    })
    .then(function (data) {
      renderVCGRResults(data);
    })
    .catch(function (err) {
      $("#vcgr-results").innerHTML =
        '<div class="error-box">VCGR simulation failed: ' + escapeHtml(err.message) + '</div>';
    });
}

function renderVCGRResults(data) {
  var allocationLabel = data.allocation ? "Action A" : "Action B";
  var allocationClass = data.allocation ? "val-action-a" : "val-action-b";
  var deltaClass = data.delta > 0 ? "val-positive" : (data.delta < 0 ? "val-negative" : "val-neutral");

  var html = '<div class="vcgr-summary">' +
    '<div class="vcgr-stat">' +
      '<span class="vcgr-stat-label">Decision</span>' +
      '<span class="vcgr-stat-value ' + allocationClass + '">' + allocationLabel + '</span>' +
    '</div>' +
    '<div class="vcgr-stat">' +
      '<span class="vcgr-stat-label">Sum of Reports</span>' +
      '<span class="vcgr-stat-value ' + (data.sum_reports >= 0 ? 'val-action-a' : 'val-action-b') + '">' + data.sum_reports.toFixed(2) + '</span>' +
    '</div>' +
    '<div class="vcgr-stat">' +
      '<span class="vcgr-stat-label">Budget (c)</span>' +
      '<span class="vcgr-stat-value">' + data.budget.toFixed(1) + '</span>' +
    '</div>' +
    '<div class="vcgr-stat">' +
      '<span class="vcgr-stat-label">&Delta;</span>' +
      '<span class="vcgr-stat-value ' + deltaClass + '">' + data.delta.toFixed(1) + '</span>' +
    '</div>' +
  '</div>';

  html += '<div style="overflow-x:auto;">';
  html += '<table class="results-table">';
  html += '<thead><tr>' +
    '<th>Agent</th>' +
    '<th style="text-align:right;">Optimal Report (m<sub>i</sub>)</th>' +
    '<th style="text-align:right;">Transfer (t<sub>i</sub>)</th>' +
    '<th style="text-align:right;">Reward (r<sub>i</sub>)</th>' +
    '<th style="text-align:right;">Payoff (&pi;<sub>i</sub>)</th>' +
    '</tr></thead>';
  html += '<tbody>';

  data.agents.forEach(function (a) {
    html += '<tr>' +
      '<td class="agent-name">' + escapeHtml(a.name) + '</td>' +
      '<td style="text-align:right;" class="' + (a.optimal_report >= 0 ? 'val-action-a' : 'val-action-b') + '">' + a.optimal_report.toFixed(2) + '</td>' +
      '<td style="text-align:right;" class="' + valClass(a.transfer) + '">' + a.transfer.toFixed(2) + '</td>' +
      '<td style="text-align:right;" class="' + valClass(a.reward) + '">' + a.reward.toFixed(2) + '</td>' +
      '<td style="text-align:right;" class="' + valClass(a.payoff) + '">' + a.payoff.toFixed(2) + '</td>' +
      '</tr>';
  });

  html += '</tbody></table></div>';

  $("#vcgr-results").innerHTML = html;
}

// ── API Call ─────────────────────────────────────────────────────────────────

function generateProfiles() {
  var context = $("#context-input").value.trim();
  var actionA = $("#action-a-input").value.trim();
  var actionB = $("#action-b-input").value.trim();

  var body = {
    context: context,
    action_a: actionA,
    action_b: actionB,
    personas: state.personas.map(function (p) {
      return { name: p.name, description: p.description };
    }),
  };

  // Populate the context summary for step 4
  $("#summary-context").textContent = context;
  $("#summary-action-a").textContent = actionA;
  $("#summary-action-b").textContent = actionB;

  goToStep(4);
  renderLoading();

  fetch(API_BASE + "/api/generate-profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(function (res) {
      if (!res.ok) {
        return res.json().then(function (data) {
          throw new Error(data.detail || "Server error " + res.status);
        });
      }
      return res.json();
    })
    .then(function (data) {
      lastProfiles = data.profiles;
      renderResults(data.profiles);
      showVCGRPanel();
    })
    .catch(function (err) {
      renderError("Failed to generate profiles: " + err.message);
    });
}

// ── Validation ──────────────────────────────────────────────────────────────

function validateStep(step) {
  if (step === 1) {
    return $("#context-input").value.trim().length > 0;
  }
  if (step === 2) {
    return (
      $("#action-a-input").value.trim().length > 0 &&
      $("#action-b-input").value.trim().length > 0
    );
  }
  if (step === 3) {
    return (
      state.personas.length > 0 &&
      state.personas.every(function (p) {
        return p.name.trim().length > 0 && p.description.trim().length > 0;
      })
    );
  }
  return true;
}

// ── Escape Utilities ────────────────────────────────────────────────────────

function escapeHtml(str) {
  var div = document.createElement("div");
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function escapeAttr(str) {
  return str.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Load Defaults ───────────────────────────────────────────────────────────

function loadDefaults() {
  fetch(API_BASE + "/api/defaults")
    .then(function (res) { return res.json(); })
    .then(function (data) {
      $("#context-input").value = data.context;
      $("#action-a-input").value = data.action_a;
      $("#action-b-input").value = data.action_b;
      state.personas = data.personas.map(function (p) {
        return { name: p.name, description: p.description };
      });
      renderPersonas();
    })
    .catch(function () {
      // If API is not running, seed with one empty persona
      addPersona("", "");
    });
}

// ── Init ─────────────────────────────────────────────────────────────────────

function init() {
  // Navigation buttons
  $("#btn-to-step2").addEventListener("click", function () {
    if (validateStep(1)) goToStep(2);
  });
  $("#btn-to-step3").addEventListener("click", function () {
    if (validateStep(2)) goToStep(3);
  });
  $("#btn-back-to-1").addEventListener("click", function () { goToStep(1); });
  $("#btn-back-to-2").addEventListener("click", function () { goToStep(2); });
  $("#btn-back-to-3").addEventListener("click", function () { goToStep(3); });

  $("#btn-generate").addEventListener("click", function () {
    if (validateStep(3)) generateProfiles();
  });

  $("#btn-restart").addEventListener("click", function () {
    lastProfiles = null;
    $("#vcgr-panel").style.display = "none";
    $("#vcgr-results").innerHTML = "";
    goToStep(1);
  });

  $("#btn-vcgr-run").addEventListener("click", runVCGR);

  // Add persona button
  $("#add-persona-btn").addEventListener("click", function () {
    addPersona("", "");
  });

  // Load defaults from API
  loadDefaults();
}

// Wait for DOM
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

// Export for testing (Node.js)
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    state: state,
    goToStep: goToStep,
    addPersona: addPersona,
    removePersona: removePersona,
    updatePersona: updatePersona,
    validateStep: validateStep,
    valClass: valClass,
    beliefClass: beliefClass,
    escapeHtml: escapeHtml,
    escapeAttr: escapeAttr,
    renderResults: renderResults,
    renderError: renderError,
    renderLoading: renderLoading,
    generateProfiles: generateProfiles,
    runVCGR: runVCGR,
    renderVCGRResults: renderVCGRResults,
    updateMeans: updateMeans,
    showVCGRPanel: showVCGRPanel,
    lastProfiles: lastProfiles,
    init: init,
  };
}
