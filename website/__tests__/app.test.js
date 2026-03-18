/**
 * @jest-environment jsdom
 */

// Set up a minimal DOM that mirrors index.html before loading app.js
function setupDOM() {
  document.body.innerHTML = `
    <div class="app-container">
      <nav class="stepper" id="stepper">
        <div class="stepper-step active" data-step="1"><div class="stepper-dot">1</div><span>Scenario</span></div>
        <div class="stepper-line"></div>
        <div class="stepper-step" data-step="2"><div class="stepper-dot">2</div><span>Decision</span></div>
        <div class="stepper-line"></div>
        <div class="stepper-step" data-step="3"><div class="stepper-dot">3</div><span>Agents</span></div>
        <div class="stepper-line"></div>
        <div class="stepper-step" data-step="4"><div class="stepper-dot">4</div><span>Results</span></div>
      </nav>

      <section class="step active" id="step-1">
        <textarea id="context-input"></textarea>
        <button id="btn-to-step2">Next</button>
      </section>

      <section class="step" id="step-2">
        <textarea id="action-a-input"></textarea>
        <textarea id="action-b-input"></textarea>
        <button id="btn-to-step3">Next</button>
        <button id="btn-back-to-1">Back</button>
      </section>

      <section class="step" id="step-3">
        <div class="persona-list" id="persona-list"></div>
        <button id="add-persona-btn">+ Add Agent</button>
        <button id="btn-generate">Generate</button>
        <button id="btn-back-to-2">Back</button>
      </section>

      <section class="step" id="step-4">
        <div id="results-content"></div>
        <button id="btn-back-to-3">Back</button>
        <button id="btn-restart">Restart</button>
      </section>
    </div>
  `;
}

let app;

beforeEach(() => {
  setupDOM();
  // Mock fetch to prevent network calls during init
  global.fetch = jest.fn(() =>
    Promise.reject(new Error("no api"))
  );
  // Fresh require each time
  jest.resetModules();
  app = require("../assets/js/app");
});

afterEach(() => {
  delete global.fetch;
});

// ── Stepper Navigation ──────────────────────────────────────────────────────

describe("goToStep", () => {
  test("shows the correct step panel", () => {
    app.goToStep(2);
    expect(document.querySelector("#step-1").classList.contains("active")).toBe(false);
    expect(document.querySelector("#step-2").classList.contains("active")).toBe(true);
  });

  test("marks previous steps as completed", () => {
    app.goToStep(3);
    const steps = document.querySelectorAll(".stepper-step");
    expect(steps[0].classList.contains("completed")).toBe(true);
    expect(steps[1].classList.contains("completed")).toBe(true);
    expect(steps[2].classList.contains("active")).toBe(true);
    expect(steps[3].classList.contains("active")).toBe(false);
  });

  test("updates currentStep state", () => {
    app.goToStep(4);
    expect(app.state.currentStep).toBe(4);
  });
});

// ── Persona Management ──────────────────────────────────────────────────────

describe("persona management", () => {
  beforeEach(() => {
    // Clear personas added by loadDefaults fallback
    app.state.personas = [];
    document.querySelector("#persona-list").innerHTML = "";
  });

  test("addPersona adds to state and renders card", () => {
    app.addPersona("Analyst", "A neutral risk analyst.");
    expect(app.state.personas).toHaveLength(1);
    expect(app.state.personas[0].name).toBe("Analyst");
    expect(document.querySelectorAll(".persona-card")).toHaveLength(1);
  });

  test("addPersona with empty values creates blank persona", () => {
    app.addPersona("", "");
    expect(app.state.personas).toHaveLength(1);
    expect(app.state.personas[0].name).toBe("");
  });

  test("removePersona removes from state and re-renders", () => {
    app.addPersona("A", "desc A");
    app.addPersona("B", "desc B");
    app.removePersona(0);
    expect(app.state.personas).toHaveLength(1);
    expect(app.state.personas[0].name).toBe("B");
    expect(document.querySelectorAll(".persona-card")).toHaveLength(1);
  });

  test("updatePersona modifies the correct field", () => {
    app.addPersona("Old", "old desc");
    app.updatePersona(0, "name", "New");
    expect(app.state.personas[0].name).toBe("New");
  });

  test("multiple personas render with correct indices", () => {
    app.addPersona("Agent 1", "desc 1");
    app.addPersona("Agent 2", "desc 2");
    app.addPersona("Agent 3", "desc 3");
    const cards = document.querySelectorAll(".persona-card");
    expect(cards).toHaveLength(3);
    expect(cards[0].querySelector(".persona-number").textContent).toBe("Agent 1");
    expect(cards[2].querySelector(".persona-number").textContent).toBe("Agent 3");
  });
});

// ── Validation ──────────────────────────────────────────────────────────────

describe("validateStep", () => {
  test("step 1 fails when context is empty", () => {
    document.querySelector("#context-input").value = "";
    expect(app.validateStep(1)).toBe(false);
  });

  test("step 1 passes when context has content", () => {
    document.querySelector("#context-input").value = "Some context";
    expect(app.validateStep(1)).toBe(true);
  });

  test("step 2 fails when action A is empty", () => {
    document.querySelector("#action-a-input").value = "";
    document.querySelector("#action-b-input").value = "Reject";
    expect(app.validateStep(2)).toBe(false);
  });

  test("step 2 fails when action B is empty", () => {
    document.querySelector("#action-a-input").value = "Approve";
    document.querySelector("#action-b-input").value = "";
    expect(app.validateStep(2)).toBe(false);
  });

  test("step 2 passes when both actions are filled", () => {
    document.querySelector("#action-a-input").value = "Approve";
    document.querySelector("#action-b-input").value = "Reject";
    expect(app.validateStep(2)).toBe(true);
  });

  test("step 3 fails when no personas", () => {
    app.state.personas = [];
    expect(app.validateStep(3)).toBe(false);
  });

  test("step 3 fails when persona has empty name", () => {
    app.state.personas = [{ name: "", description: "desc" }];
    expect(app.validateStep(3)).toBe(false);
  });

  test("step 3 fails when persona has empty description", () => {
    app.state.personas = [{ name: "Agent", description: "" }];
    expect(app.validateStep(3)).toBe(false);
  });

  test("step 3 passes with valid personas", () => {
    app.state.personas = [
      { name: "Analyst", description: "A neutral risk analyst." },
      { name: "Depositor", description: "Large depositor with $1M." },
    ];
    expect(app.validateStep(3)).toBe(true);
  });
});

// ── Utility Functions ───────────────────────────────────────────────────────

describe("valClass", () => {
  test("returns val-positive for positive values", () => {
    expect(app.valClass(10)).toBe("val-positive");
  });

  test("returns val-negative for negative values", () => {
    expect(app.valClass(-5)).toBe("val-negative");
  });

  test("returns val-neutral for zero", () => {
    expect(app.valClass(0)).toBe("val-neutral");
  });
});

describe("beliefClass", () => {
  test("returns val-belief-high for values > 0.5", () => {
    expect(app.beliefClass(0.7)).toBe("val-belief-high");
  });

  test("returns val-belief-low for values <= 0.5", () => {
    expect(app.beliefClass(0.3)).toBe("val-belief-low");
    expect(app.beliefClass(0.5)).toBe("val-belief-low");
  });
});

describe("escapeHtml", () => {
  test("escapes HTML special characters", () => {
    expect(app.escapeHtml('<script>alert("xss")</script>')).toBe(
      '&lt;script&gt;alert("xss")&lt;/script&gt;'
    );
  });

  test("leaves plain text unchanged", () => {
    expect(app.escapeHtml("hello world")).toBe("hello world");
  });
});

describe("escapeAttr", () => {
  test("escapes attribute special characters", () => {
    expect(app.escapeAttr('value "with" <quotes>')).toBe(
      "value &quot;with&quot; &lt;quotes&gt;"
    );
  });
});

// ── Results Rendering ───────────────────────────────────────────────────────

describe("renderResults", () => {
  test("renders a table with agent profiles", () => {
    const profiles = [
      { name: "Analyst", theta_A: 10.0, theta_B: -20.0, p_A: 0.6, p_B: 0.8, rationale: "I trust the data." },
      { name: "Depositor", theta_A: -30.0, theta_B: 40.0, p_A: 0.3, p_B: 0.9, rationale: "Protect my deposits." },
    ];
    app.renderResults(profiles);

    const content = document.querySelector("#results-content");
    expect(content.querySelector(".results-table")).not.toBeNull();

    const rows = content.querySelectorAll("tbody tr");
    // 2 agents * 2 rows each (data + rationale) + 1 summary row = 5
    expect(rows).toHaveLength(5);

    // Check agent names appear
    expect(content.textContent).toContain("Analyst");
    expect(content.textContent).toContain("Depositor");

    // Check rationale appears
    expect(content.textContent).toContain("I trust the data.");
    expect(content.textContent).toContain("Protect my deposits.");

    // Check mean consensus row
    expect(content.textContent).toContain("Mean (Consensus)");
  });

  test("renders error for empty profiles", () => {
    app.renderResults([]);
    expect(document.querySelector("#results-content .error-box")).not.toBeNull();
  });

  test("renders error for null profiles", () => {
    app.renderResults(null);
    expect(document.querySelector("#results-content .error-box")).not.toBeNull();
  });
});

describe("renderLoading", () => {
  test("shows spinner and loading text", () => {
    app.renderLoading();
    const content = document.querySelector("#results-content");
    expect(content.querySelector(".spinner")).not.toBeNull();
    expect(content.textContent).toContain("Generating agent profiles");
  });
});

describe("renderError", () => {
  test("shows error message", () => {
    app.renderError("Something went wrong");
    const box = document.querySelector("#results-content .error-box");
    expect(box).not.toBeNull();
    expect(box.textContent).toBe("Something went wrong");
  });

  test("escapes HTML in error messages", () => {
    app.renderError('<script>alert("xss")</script>');
    const box = document.querySelector("#results-content .error-box");
    expect(box.innerHTML).not.toContain("<script>");
  });
});

// ── API Integration ─────────────────────────────────────────────────────────

describe("generateProfiles", () => {
  test("sends correct request and renders results on success", async () => {
    // Set up form values
    document.querySelector("#context-input").value = "Test context";
    document.querySelector("#action-a-input").value = "Action A";
    document.querySelector("#action-b-input").value = "Action B";
    app.state.personas = [
      { name: "Agent 1", description: "Desc 1" },
    ];

    const mockProfiles = [
      { name: "Agent 1", theta_A: 25.0, theta_B: -10.0, p_A: 0.7, p_B: 0.4, rationale: "Test rationale" },
    ];

    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ profiles: mockProfiles }),
      })
    );

    app.generateProfiles();

    // Should navigate to step 4
    expect(app.state.currentStep).toBe(4);

    // Wait for async fetch
    await new Promise((r) => setTimeout(r, 50));

    // Verify fetch was called correctly
    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/generate-profiles",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
    );

    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.context).toBe("Test context");
    expect(body.action_a).toBe("Action A");
    expect(body.action_b).toBe("Action B");
    expect(body.personas).toHaveLength(1);

    // Results should be rendered
    const content = document.querySelector("#results-content");
    expect(content.textContent).toContain("Agent 1");
    expect(content.textContent).toContain("Test rationale");
  });

  test("renders error on API failure", async () => {
    document.querySelector("#context-input").value = "ctx";
    document.querySelector("#action-a-input").value = "A";
    document.querySelector("#action-b-input").value = "B";
    app.state.personas = [{ name: "X", description: "Y" }];

    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: false,
        status: 502,
        json: () => Promise.resolve({ detail: "LLM down" }),
      })
    );

    app.generateProfiles();
    await new Promise((r) => setTimeout(r, 50));

    const content = document.querySelector("#results-content");
    expect(content.querySelector(".error-box")).not.toBeNull();
    expect(content.textContent).toContain("LLM down");
  });

  test("renders error on network failure", async () => {
    document.querySelector("#context-input").value = "ctx";
    document.querySelector("#action-a-input").value = "A";
    document.querySelector("#action-b-input").value = "B";
    app.state.personas = [{ name: "X", description: "Y" }];

    global.fetch = jest.fn(() => Promise.reject(new Error("Network error")));

    app.generateProfiles();
    await new Promise((r) => setTimeout(r, 50));

    const content = document.querySelector("#results-content");
    expect(content.querySelector(".error-box")).not.toBeNull();
    expect(content.textContent).toContain("Network error");
  });
});

// ── Button Click Integration ────────────────────────────────────────────────

describe("button navigation", () => {
  beforeEach(() => {
    // Re-init to bind event listeners
    global.fetch = jest.fn(() => Promise.reject(new Error("no api")));
    app.init();
  });

  test("next button from step 1 requires context", () => {
    document.querySelector("#context-input").value = "";
    document.querySelector("#btn-to-step2").click();
    expect(app.state.currentStep).toBe(1);
  });

  test("next button from step 1 advances with content", () => {
    document.querySelector("#context-input").value = "Some scenario";
    document.querySelector("#btn-to-step2").click();
    expect(app.state.currentStep).toBe(2);
  });

  test("back button from step 2 returns to step 1", () => {
    app.goToStep(2);
    document.querySelector("#btn-back-to-1").click();
    expect(app.state.currentStep).toBe(1);
  });

  test("restart button returns to step 1", () => {
    app.goToStep(4);
    document.querySelector("#btn-restart").click();
    expect(app.state.currentStep).toBe(1);
  });

  test("add persona button creates a new persona card", () => {
    app.goToStep(3);
    document.querySelector("#add-persona-btn").click();
    expect(app.state.personas.length).toBeGreaterThanOrEqual(1);
    expect(document.querySelectorAll(".persona-card").length).toBeGreaterThanOrEqual(1);
  });
});
