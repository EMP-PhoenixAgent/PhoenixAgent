/* =========================================================================
   Phoenix Agent — Frontend logic
   Bridges the Tauri command/event API to the chat UI.
   ========================================================================= */

// Tauri injects its API into window.__TAURI__ at runtime.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

// ----- DOM shortcuts --------------------------------------------------------
const $ = (id) => document.getElementById(id);

const setupScreen = $("setup-screen");
const unlockScreen = $("unlock-screen");
const chatScreen = $("chat-screen");
const passphraseInput = $("passphrase-input");
const unlockBtn = $("unlock-btn");
const unlockError = $("unlock-error");
const recoverBtn = $("recover-btn");

const setupPassphrase = $("setup-passphrase");
const setupConfirm = $("setup-confirm");
const setupBtn = $("setup-btn");
const setupError = $("setup-error");

const chatMessages = $("chat-messages");
const messageInput = $("message-input");
const sendBtn = $("send-btn");
const modelSelect = $("model-select");
const modelPickerBtn = $("model-picker-btn");
const modelPopup = $("model-popup");
const healthSummary = $("health-summary");

// Sidebar elements (Science Workbench).
const sidebar = $("sidebar");
const sidebarResizer = $("sidebar-resizer");
const modelsNavItem = document.querySelector('.nav-item[data-panel="models"]');
const skillsNavItem = document.querySelector('.nav-item[data-panel="skills"]');
const workdirDisplay = $("workdir-display");
const workdirRow = $("workdir-row");
const profileNewBtn = $("profile-new-btn");
const profileHomeBtn = $("profile-home-btn");
// Sidebar console (output log + input line) + its height drag handle.
const consoleOutput = $("console-output");
const consoleInput = $("console-input");
const consoleResizer = $("console-resizer");
const consoleSection = $("console-section");
const modelsPanel = $("models-panel");
const modelsCloseBtn = $("models-close-btn");
// Circular download progress ring on the Models nav item (far right of the
// button) — thin ring with the completion percentage inside.
const navEta = $("nav-eta");
const navEtaRing = navEta ? navEta.querySelector(".nav-eta-ring") : null;
const navEtaPct = navEta ? navEta.querySelector(".nav-eta-pct") : null;
// Models panel v0.5 — runner tabs (AmberCore | Ollama) + shared model list
const runnerBox = $("runner-box");
const paneAmberCore = $("pane-ambercore");
const paneOllama = $("pane-ollama");
const mlTitle = $("ml-title");
const mlAcquireAmberCore = $("ml-acquire-ambercore");
const mlAcquireOllama = $("ml-acquire-ollama");
// Which runner the settings pane + model list show ("ambercore" | "ollama");
// persisted across sessions like the console height.
let activeRunner = "ambercore";
const icDir = $("ic-dir");
const icDirClear = $("ic-dir-clear");
const icSearchBtn = $("ic-search-btn");
// Model search modal.
const modelSearchModal = $("model-search-modal");
const msQuery = $("ms-query");
const msSearchBtn = $("ms-search-btn");
const msCloseBtn = $("ms-close-btn");
const msResults = $("ms-results");
const msVram = $("ms-vram");
const msFilterSize = $("ms-filter-size");
const msFilterQuant = $("ms-filter-quant");
const msFilterParams = $("ms-filter-params");
let msActiveSource = "huggingface";
let msLastResponse = null; // last search — the filters re-slice it, no re-fetch
const mlPullBars = $("ml-pull-bars");
const icList = $("ic-list");
const olInstallBtn = $("ol-install-btn");
const olPull = $("ol-pull");
const olPullBtn = $("ol-pull-btn");
const olList = $("ol-list");
const prBox = $("provider-box");
const prName = $("pr-name");
const prKey = $("pr-key");
const prUrl = $("pr-url");
const prRegisterBtn = $("pr-register-btn");
const prList = $("pr-list");
// Skills panel elements.
const skillsPanel = $("skills-panel");
const skillsList = $("skills-list");
const skillsCloseBtn = $("skills-close-btn");
const skillNewBtn = $("skill-new-btn");
// The skill form lives inside its own modal overlay (#skill-form-overlay).
const skillForm = $("skill-form-overlay");
const skillFormTitle = $("skill-form-title");
const skillFormName = $("skill-form-name");
const skillFormDesc = $("skill-form-desc");
const skillFormBody = $("skill-form-body");
const skillFormSave = $("skill-form-save");
const skillFormCancel = $("skill-form-cancel");
// Sub-Agents (Panel 6)
const subagentsNavItem = document.querySelector('.nav-item[data-panel="subagents"]');
const subagentsPanel = $("subagents-panel");
const subagentsList = $("subagents-list");
const subagentsCloseBtn = $("subagents-close-btn");
const subagentNewBtn = $("subagent-new-btn");
const subagentForm = $("subagent-form-overlay");
const subagentFormTitle = $("subagent-form-title");
const subagentFormName = $("subagent-form-name");
const subagentFormDesc = $("subagent-form-desc");
const subagentFormModel = $("subagent-form-model");
const subagentFormPersona = $("subagent-form-persona");
const subagentFormCancel = $("subagent-form-cancel");
const subagentFormSave = $("subagent-form-save");
let editingSubAgentId = null; // null = creating new, number = editing
const skillSearchInput = $("skill-search-input");
const skillSearchBtn = $("skill-search-btn");
const skillSearchResults = $("skill-search-results");
// Tools panel elements.
const toolsNavItem = document.querySelector('.nav-item[data-panel="tools"]');
const toolsPanel = $("tools-panel");
const toolsList = $("tools-list");
const toolsCloseBtn = $("tools-close-btn");
const toolNewBtn = $("tool-new-btn");
// The tool form lives inside its own modal overlay (#tool-form-overlay). We
// toggle the overlay's visibility; the inner #tool-form card holds the fields.
const toolForm = $("tool-form-overlay");
const toolFormTitle = $("tool-form-title");
const toolFormName = $("tool-form-name");
const toolFormDesc = $("tool-form-desc");
const toolFormInterpreter = $("tool-form-interpreter");
const toolFormKind = $("tool-form-kind");
const toolFormSchema = $("tool-form-schema");
const toolFormBody = $("tool-form-body");
const toolFormSave = $("tool-form-save");
const toolFormCancel = $("tool-form-cancel");
const toolSearchInput = $("tool-search-input");
const toolSearchBtn = $("tool-search-btn");
const toolSearchResults = $("tool-search-results");
let editingToolId = null; // null = creating new, number = editing
// Context panel elements.
const contextNavItem = document.querySelector('.nav-item[data-panel="context"]');
const contextPanel = $("context-panel");
const contextList = $("context-list");
const contextCloseBtn = $("context-close-btn");
const contextNewBtn = $("context-new-btn");
// The context form lives inside its own modal overlay (#context-form-overlay).
const contextForm = $("context-form-overlay");
const contextFormTitle = $("context-form-title");
const contextFormName = $("context-form-name");
const contextFormDesc = $("context-form-desc");
const contextFormBody = $("context-form-body");
const contextFormSave = $("context-form-save");
const contextFormCancel = $("context-form-cancel");
let editingContextId = null; // null = creating new, number = editing
// Memory panel (Panel 5: MCP connections) elements.
const memoryNavItem = document.querySelector('.nav-item[data-panel="memory"]');
const memoryPanel = $("memory-panel");
const memoryList = $("memory-list");
const memoryCloseBtn = $("memory-close-btn");
const memoryNewBtn = $("memory-new-btn");
// The memory form lives inside its own modal overlay (#memory-form-overlay).
const memoryForm = $("memory-form-overlay");
const memoryFormTitle = $("memory-form-title");
const memoryFormName = $("memory-form-name");
const memoryFormDesc = $("memory-form-desc");
const memoryFormTransport = $("memory-form-transport");
const memoryFormCommand = $("memory-form-command");
const memoryFormArgs = $("memory-form-args");
const memoryFormSave = $("memory-form-save");
const memoryFormCancel = $("memory-form-cancel");
const memoryFormTest = $("memory-form-test");
let editingMemoryId = null; // null = creating new, number = editing
// Main menu / configuration window elements.
// Main menu / configuration window elements. The lock icon on the health bar
// is the sole entry point to the menu (password manager + settings).
const configMenuBtn = $("config-menu-btn");
const configModal = $("config-modal");
const configCloseBtn = $("config-close-btn");
// Change launch password (Card 1).
const launchPassForm = $("launch-pass-form");
const lpOld = $("lp-old");
const lpNew = $("lp-new");
const lpConfirm = $("lp-confirm");
const lpStatus = $("lp-status");
// Change database password (Card 2).
const cpForm = $("change-passphrase-form");
const cpOld = $("cp-old");
const cpNew = $("cp-new");
const cpConfirm = $("cp-confirm");
const cpLaunch = $("cp-launch");
const cpStatus = $("cp-status");
// 2FA setup/disable.
const totpEnabledView = $("totp-enabled-view");
const totpDisabledView = $("totp-disabled-view");
const totpSetupView = $("totp-setup-view");
const totpAccount = $("totp-account");
const totpEnableBtn = $("totp-enable-btn");
const totpDisableBtn = $("totp-disable-btn");
const totpQr = $("totp-qr");
const totpSecretDisplay = $("totp-secret-display");
const totpConfirmCode = $("totp-confirm-code");
const totpConfirmBtn = $("totp-confirm-btn");
const totpCancelBtn = $("totp-cancel-btn");
const totpSetupStatus = $("totp-setup-status");
// Holds the in-progress TOTP setup (secret + otpauth) between enable→confirm.
let pendingTotp = null;

let currentModel = "";
let isAgentBusy = false;
let streamingBubble = null; // the assistant bubble currently receiving deltas
let workingBlock = null; // the inline working indicator currently receiving reasoning
let toolCards = {}; // index -> tool-card element for the current turn's batch
let activeProfileId = null; // tracked across unlock/profile switch
let editingSkillId = null;  // null = creating new, number = editing

// ----- Markdown config ------------------------------------------------------
if (typeof marked !== "undefined") {
  marked.setOptions({ breaks: true, gfm: true });
}

// ----- Boot -----------------------------------------------------------------
async function init() {
  const ready = await invoke("is_initialized");

  // Set up listeners immediately so events aren't missed after unlock/setup.
  setupListeners();

  // NOTE: this tree ships as the encapsulated launcher — there is no dev
  // auto-unlock here (no bypass and no hardcoded password in the shipped JS).
  // Developer convenience lives in the old dev tree / typing the password.
  if (!ready) {
    // First run — show setup screen.
    setupScreen.classList.add("active");
    setupPassphrase.focus();
  } else {
    // Returning user — show unlock screen (launch password gate).
    unlockScreen.classList.add("active");
    passphraseInput.focus();
    // Show the "recover via 2FA" link only if 2FA is enabled.
    try {
      const has2fa = await invoke("has_totp");
      if (recoverBtn) recoverBtn.hidden = !has2fa;
    } catch (e) {
      console.warn("has_totp check failed:", e);
    }
  }
}

// ----- Setup (first run) ----------------------------------------------------
async function doSetup() {
  const launchPassword = setupPassphrase.value;
  const confirm = setupConfirm.value;
  if (!launchPassword || !confirm) return;

  setupBtn.disabled = true;
  setupBtn.textContent = "Creating…";
  setupError.textContent = "";

  try {
    const result = await invoke("setup", {
      launchPassword,
      confirmLaunchPassword: confirm,
    });
    currentModel = result.model;

    // Switch to chat.
    setupScreen.classList.remove("active");
    chatScreen.classList.add("active");

    await populateModels();
    await loadSidebar(result);
    await loadTodo();

    addSystemMessage(`Welcome! Encrypted memory created. Working in: ${result.project_path}`);
    logsNotices();
    messageInput.focus();
  } catch (e) {
    setupError.textContent = String(e);
    setupBtn.disabled = false;
    setupBtn.textContent = "Create & Launch";
  }
}

// ----- Unlock (launch password gate) ----------------------------------------
async function doUnlock() {
  const launchPassword = passphraseInput.value;
  if (!launchPassword) return;

  unlockBtn.disabled = true;
  unlockBtn.textContent = "Unlocking…";
  unlockError.textContent = "";

  try {
    const result = await invoke("unlock", { launchPassword });
    currentModel = result.model;
    modelSelect.value = result.model;

    // Switch screens.
    unlockScreen.classList.remove("active");
    chatScreen.classList.add("active");

    // Load models + sidebar.
    await populateModels();
    await loadSidebar(result);
    await loadTodo();

    // Hardware check-up at launch — preloads the Telemetry tab baseline.
    refreshTelemetryTab();

    // Add welcome message.
    addSystemMessage(`Ready. Working in: ${result.project_path}`);
    logsNotices();
    refreshSessionRail();

    messageInput.focus();

    // Seven: show the alpha Chronos invitation once (until dismissed).
    try { if (await invoke("should_show_alpha_popup")) $("alpha-popup").hidden = false; } catch { /* ignore */ }
  } catch (e) {
    unlockError.textContent = String(e);
    unlockBtn.disabled = false;
    unlockBtn.textContent = "Unlock";
  }
}

// ----- Model selector -------------------------------------------------------
/** Cache of the models currently offered by the active backend (for the popup). */
let pickerModels = [];

/** Load the active backend's models and render both the hidden native select
 *  and the custom upward-opening popup. */
async function populateModels() {
  let models = [];
  try {
    models = await invoke("list_models");
  } catch (e) {
    console.warn("Model list failed:", e);
  }
  pickerModels = models;

  // Keep the hidden native select in sync (some code reads .value).
  modelSelect.innerHTML = "";
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === currentModel) opt.selected = true;
    modelSelect.appendChild(opt);
  }

  // Render the custom popup.
  renderModelPopup();
  // Button label = current model (or a placeholder) + open/close caret.
  setModelBtnLabel();
}

/** Render the custom popup list from `pickerModels`. */
function renderModelPopup() {
  if (!modelPopup) return;
  modelPopup.innerHTML = "";
  if (pickerModels.length === 0) {
    const empty = document.createElement("div");
    empty.className = "model-opt-empty";
    empty.textContent = "(no models — switch backend / pull a model)";
    modelPopup.appendChild(empty);
    return;
  }
  for (const m of pickerModels) {
    const opt = document.createElement("div");
    opt.className = "model-opt" + (m === currentModel ? " active" : "");
    opt.textContent = m;
    opt.title = m;
    opt.addEventListener("click", () => {
      modelPopup.hidden = true;
      switchModel(m);
    });
    modelPopup.appendChild(opt);
  }
}

/** Set the picker button label: current model + a caret that flips when open. */
function setModelBtnLabel() {
  if (!modelPickerBtn) return;
  const name = currentModel ? currentModel : "Model";
  const open = !!(modelPopup && !modelPopup.hidden);
  modelPickerBtn.textContent = name + (open ? " ▴" : " ▾");
}

/** Toggle the popup open/closed. */
function toggleModelPopup() {
  if (!modelPopup) return;
  modelPopup.hidden = !modelPopup.hidden;
  if (!modelPopup.hidden) renderModelPopup();
  setModelBtnLabel();
}

modelPickerBtn?.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleModelPopup();
});
// Click outside closes the popup. Capture phase + target guard: nothing in
// the DOM can stopPropagation and leave it stuck open, and clicks on the
// button/popup itself are excluded so toggling and selecting still work.
document.addEventListener("click", (e) => {
  if (!modelPopup || modelPopup.hidden) return;
  if (modelPickerBtn?.contains(e.target) || modelPopup.contains(e.target)) return;
  modelPopup.hidden = true;
  setModelBtnLabel();
}, true);
modelPopup?.addEventListener("click", (e) => e.stopPropagation());
// Escape closes the popup.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modelPopup && !modelPopup.hidden) {
    modelPopup.hidden = true;
    setModelBtnLabel();
  }
});

modelSelect?.addEventListener("change", (e) => {
  switchModel(e.target.value);
});

// ----- Mode selector (Plan/Think/Auto) — 32×32 button beside Send -----
const modeSelector = $("mode-selector");
const modeFace = $("mode-face");
const modeFaceIcon = $("mode-face-icon");
const modePopout = $("mode-popout");
let currentMode = "think"; // tracked for mode-gated UI (to-do editing)

function setActiveMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".mode-btn").forEach((b) => {
    const active = b.dataset.mode === mode;
    b.classList.toggle("active", active);
    if (active) {
      // Mirror the active option's icon onto the face; the tooltip names it.
      modeFaceIcon.innerHTML = b.querySelector("svg")?.outerHTML || "";
      modeFace.title = `Reasoning mode: ${b.textContent.trim()} — click to change`;
    }
  });
  renderTodo();
}
function setModePopout(open) {
  modePopout.hidden = !open;
  modeSelector.classList.toggle("open", open);
  modeFace.setAttribute("aria-expanded", String(open));
}
modeFace.addEventListener("click", () => setModePopout(modePopout.hidden));
document.addEventListener("click", (e) => {
  if (!modeSelector.contains(e.target)) setModePopout(false);
});
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const m = btn.dataset.mode;
    invoke("set_mode", { mode: m })
      .then(() => {
        setActiveMode(m);
        setModePopout(false);
      })
      .catch((e) => addSystemMessage(`Error: ${e}`));
  });
});
// Restore the active mode on load (default Think).
invoke("get_mode").then(setActiveMode).catch(() => setActiveMode("think"));

// ----- To-do list (the shared plan panel + drag-and-drop editor modal) --------
// The agent maintains it via the `update_todo` tool (events arrive as
// `todo_updated`); the 📋 title opens the editor modal for manual changes
// (Plan mode only). The floating panel itself is display-only. The modal is a
// card-based flow builder that compiles to the plan markdown:
//   - [ ] Task            main task
//   - [x] ~~Task~~        done (strikethrough)
//     - [ ] Sub-task      nested (2-space indent)
//       - 🔧 `tool`       tool for the sub-task above
//     - 🔧 `tool`         tool directly under a Task = whole-task scope
//   - 🤖 sub-agent: Name  the NEXT task is delegated to this sub-agent
//   - ⏹ STOP              movable sub-agent stop marker
const todoPanel = $("todo-panel");
const todoBody = $("todo-body");
const todoOpenBtn = $("todo-open-btn");
const todoCollapseBtn = $("todo-collapse-btn");
const todoModal = $("todo-modal");
const todoModalCloseBtn = $("todo-modal-close-btn");
const todoFlow = $("todo-flow");
const todoPreview = $("todo-preview");
const todoSaveBtn = $("todo-save-btn");
const todoCancelBtn = $("todo-cancel-btn");
let todoMarkdown = "";
let flowItems = [];       // structured plan (see comment above)
let flowIdSeq = 1;
let flowHint = "";        // transient guidance shown at the top of the flow
let flowHintTimer = null;
let todoSubAgents = [];   // loaded when the modal opens (picker data)
let todoTools = [];       // enabled user tool names + built-ins
const BUILTIN_TOOLS = ["read_file", "write_file", "edit_file", "list_dir", "grep", "run_command", "delegate", "update_todo"];

function renderTodo() {
  const empty = !todoMarkdown.trim();
  // Visible when there's a plan to show, or when the user could write one
  // (Plan mode shows the panel with an empty-state hint).
  todoPanel.hidden = empty && currentMode !== "plan";
  // The "invisible wall": while the to-do overlay is up, reserve the chat's
  // right side so messages never flow under it (see styles.css todo-open).
  document.getElementById("main-column")?.classList.toggle("todo-open", !todoPanel.hidden);
  if (todoPanel.hidden) return;
  todoBody.innerHTML = empty
    ? '<div id="todo-empty">No plan yet — the agent will fill this in while working, or open the editor (📋 above) in Plan mode.</div>'
    : marked.parse(todoMarkdown);
}

async function loadTodo() {
  try {
    todoMarkdown = await invoke("get_todo");
  } catch (e) {
    todoMarkdown = "";
  }
  renderTodo();
}

// ---- markdown ⇄ flow conversion ---------------------------------------------

function parseFlow(md) {
  const items = [];
  let lastTask = null;
  let lastSub = null;
  for (const raw of md.split(/\r?\n/)) {
    const line = raw.trimEnd();
    if (!line.trim()) continue;
    let m;
    if ((m = line.match(/^##\s+(.+)$/))) {
      items.push({ kind: "heading", text: m[1], _id: flowIdSeq++ });
      lastTask = lastSub = null;
    } else if ((m = line.match(/^-\s+🤖\s*(?:sub-agent:)?\s*(.+)$/i))) {
      items.push({ kind: "subagent", name: m[1].trim(), _id: flowIdSeq++ });
      lastTask = lastSub = null;
    } else if (/^-\s+⏹/.test(line)) {
      items.push({ kind: "stop", _id: flowIdSeq++ });
    } else if ((m = line.match(/^(\s*)-\s+\[([ xX])\]\s+(.*)$/))) {
      const done = m[2].toLowerCase() === "x";
      let text = m[3].trim();
      const strike = text.match(/^~~(.*)~~$/);
      if (strike) text = strike[1];
      if (m[1].length >= 2 && lastTask) {
        lastSub = { text, done, tool: null, _id: flowIdSeq++ };
        lastTask.subtasks.push(lastSub);
      } else {
        lastTask = { kind: "task", text, done, tool: null, subtasks: [], _id: flowIdSeq++ };
        lastSub = null;
        items.push(lastTask);
      }
    } else if ((m = line.match(/^(\s*)-\s+🔧\s+`?([A-Za-z0-9_\-]+)`?/))) {
      const tool = m[2];
      if (m[1].length >= 4 && lastSub) lastSub.tool = tool;
      else if (lastTask) lastTask.tool = tool;
    } else if ((m = line.match(/^#\s+(.+)$/))) {
      items.push({ kind: "heading", text: m[1], _id: flowIdSeq++ });
      lastTask = lastSub = null;
    } else {
      items.push({ kind: "note", text: line, _id: flowIdSeq++ });
    }
  }
  return items;
}

function compileFlow() {
  const out = [];
  for (const it of flowItems) {
    if (it.kind === "heading") out.push(`## ${it.text}`);
    else if (it.kind === "note") out.push(it.text);
    else if (it.kind === "subagent") out.push(`- 🤖 sub-agent: ${it.name}`);
    else if (it.kind === "stop") out.push("- ⏹ STOP");
    else if (it.kind === "task") {
      out.push(it.done ? `- [x] ~~${it.text}~~` : `- [ ] ${it.text}`);
      if (it.tool) out.push(`  - 🔧 \`${it.tool}\``);
      for (const st of it.subtasks) {
        out.push(st.done ? `  - [x] ~~${st.text}~~` : `  - [ ] ${st.text}`);
        if (st.tool) out.push(`    - 🔧 \`${st.tool}\``);
      }
    }
  }
  return out.join("\n");
}

// ---- flow rendering ----------------------------------------------------------

function setFlowHint(msg) {
  flowHint = msg;
  clearTimeout(flowHintTimer);
  flowHintTimer = setTimeout(() => {
    flowHint = "";
    renderTodoFlow();
  }, 2600);
  renderTodoFlow();
}

function refreshTodoPreview() {
  const md = compileFlow();
  todoPreview.innerHTML = md.trim()
    ? marked.parse(md)
    : '<div id="todo-empty">Preview — your plan renders here.</div>';
}

/** The innermost item "above" a gap: the last sub-task of the task above it,
 *  or that task itself when it has no sub-tasks. */
function innermostAbove(gapIndex) {
  for (let i = gapIndex - 1; i >= 0; i--) {
    const it = flowItems[i];
    if (it.kind === "task") {
      return it.subtasks.length ? it.subtasks[it.subtasks.length - 1] : it;
    }
  }
  return null;
}

function nearestTaskAbove(gapIndex) {
  for (let i = gapIndex - 1; i >= 0; i--) {
    if (flowItems[i].kind === "task") return flowItems[i];
  }
  return null;
}

function toolPicker(target, onPick) {
  const sel = document.createElement("select");
  sel.className = "flow-picker";
  const names = [...BUILTIN_TOOLS, ...todoTools.filter((t) => !BUILTIN_TOOLS.includes(t))];
  const opts = [target.tool ? "✕ remove tool" : "Select a tool…", ...names];
  for (const n of opts) {
    const o = document.createElement("option");
    o.textContent = n;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    const v = sel.value;
    onPick(v.startsWith("✕") ? null : v.startsWith("Select") ? target.tool : v);
  });
  return sel;
}

function makeGap(index) {
  const gap = document.createElement("div");
  gap.className = "flow-gap";
  gap.addEventListener("dragover", (e) => {
    e.preventDefault();
    gap.classList.add("over");
  });
  gap.addEventListener("dragleave", () => gap.classList.remove("over"));
  gap.addEventListener("drop", (e) => {
    e.preventDefault();
    gap.classList.remove("over");
    handleFlowDrop(e, index);
  });
  return gap;
}

function handleFlowDrop(e, index) {
  const payload = e.dataTransfer.getData("text/plain");
  if (payload.startsWith("new:")) insertCard(payload.slice(4), index);
  else if (payload.startsWith("move:")) moveFlowItem(parseInt(payload.slice(5), 10), index);
}

/** Insert a palette card at gap `index` (flowItems.length = append). */
function insertCard(card, index) {
  if (card === "task") {
    flowItems.splice(index, 0, { kind: "task", text: "New task", done: false, tool: null, subtasks: [], _id: flowIdSeq++, _editing: true });
  } else if (card === "subtask") {
    const task = nearestTaskAbove(index);
    if (!task) return setFlowHint("Drop a Sub-task under a Task.");
    task.subtasks.push({ text: "New sub-task", done: false, tool: null, _id: flowIdSeq++, _editing: true });
  } else if (card === "tool") {
    const target = innermostAbove(index);
    if (!target || (target.kind !== "task" && !target.text)) return setFlowHint("Drop a Tool below a task or sub-task.");
    target._picking = true;
  } else if (card === "subagent") {
    const next = flowItems[index];
    if (!next || next.kind !== "task") return setFlowHint("Place the Sub-agent card ABOVE a Task.");
    flowItems.splice(index, 0, { kind: "subagent", name: "", _id: flowIdSeq++, _picking: true });
    if (todoSubAgents.length === 0) {
      flowItems.splice(index, 1);
      return setFlowHint("No sub-agents defined — create one in the Sub-Agents panel first.");
    }
    // The card also spawns a movable stop marker right after the delegated task.
    let stopAt = index + 2; // subagent + task
    flowItems.splice(stopAt, 0, { kind: "stop", _id: flowIdSeq++ });
  }
  renderTodoFlow();
}

/** Move an existing top-level item (incl. the stop marker) to gap `index`. */
function moveFlowItem(id, index) {
  const idx = flowItems.findIndex((it) => it._id === id);
  if (idx === -1) return;
  const item = flowItems[idx];
  let target = index;
  if (idx < target) target--;
  // Validate sub-agent placement: it must sit above a main Task.
  const probe = flowItems.filter((_, i) => i !== idx);
  probe.splice(target, 0, item);
  if (item.kind === "subagent") {
    const next = probe[target + 1];
    if (!next || next.kind !== "task") return setFlowHint("A Sub-agent card must sit above a Task.");
  }
  flowItems.splice(idx, 1);
  flowItems.splice(target, 0, item);
  renderTodoFlow();
}

function textEditor(value, onCommit) {
  const input = document.createElement("input");
  input.className = "flow-text-input";
  input.value = value;
  const commit = () => onCommit(input.value.trim() || value);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    if (e.key === "Escape") { e.stopImmediatePropagation(); input.value = value; input.blur(); }
  });
  input.addEventListener("blur", commit);
  setTimeout(() => { input.focus(); input.select(); }, 0);
  return input;
}

function renderTodoFlow() {
  todoFlow.innerHTML = "";
  if (flowHint) {
    const hint = document.createElement("div");
    hint.id = "todo-flow-hint";
    hint.textContent = flowHint;
    todoFlow.appendChild(hint);
  }
  if (flowItems.length === 0 && !flowHint) {
    const empty = document.createElement("div");
    empty.id = "todo-flow-hint";
    empty.textContent = "Empty plan — drag a card here (or click one above).";
    todoFlow.appendChild(empty);
  }
  const appendGap = (i) => todoFlow.appendChild(makeGap(i));

  flowItems.forEach((it, i) => {
    appendGap(i);
    const block = document.createElement("div");
    block.className = "flow-block";
    const row = document.createElement("div");
    row.className = "flow-item";

    const mkDel = () => {
      const b = document.createElement("button");
      b.className = "flow-del";
      b.textContent = "✕";
      b.title = "Remove";
      b.addEventListener("click", () => {
        flowItems.splice(i, 1);
        renderTodoFlow();
      });
      return b;
    };
    const mkCheck = (obj, owner) => {
      const c = document.createElement("input");
      c.type = "checkbox";
      c.className = "flow-check";
      c.checked = obj.done;
      c.title = "Done";
      c.addEventListener("change", () => {
        obj.done = c.checked;
        owner.classList.toggle("done", obj.done);
        refreshTodoPreview();
      });
      return c;
    };
    const mkText = (obj) => {
      if (obj._editing) {
        return textEditor(obj.text, (v) => {
          obj.text = v;
          obj._editing = false;
          renderTodoFlow();
        });
      }
      const s = document.createElement("span");
      s.className = "flow-text";
      s.textContent = obj.text;
      s.title = "Click to edit";
      s.addEventListener("click", () => {
        obj._editing = true;
        renderTodoFlow();
      });
      return s;
    };

    if (it.kind === "heading" || it.kind === "note") {
      row.classList.add("heading");
      row.appendChild(mkText(it));
      row.appendChild(mkDel());
      block.draggable = true;
      block.appendChild(row);
    } else if (it.kind === "subagent") {
      row.classList.add("subagent");
      if (it._picking || !it.name) {
        const sel = document.createElement("select");
        sel.className = "flow-picker";
        for (const sa of todoSubAgents) {
          const o = document.createElement("option");
          o.textContent = sa.name;
          sel.appendChild(o);
        }
        sel.addEventListener("change", () => {
          it.name = sel.value;
          it._picking = false;
          renderTodoFlow();
        });
        row.appendChild(sel);
      } else {
        const chip = document.createElement("span");
        chip.className = "flow-chip agent";
        chip.textContent = `🤖 ${it.name} → next task`;
        chip.title = "Click to change the sub-agent";
        chip.addEventListener("click", () => {
          it._picking = true;
          renderTodoFlow();
        });
        row.appendChild(chip);
      }
      row.appendChild(mkDel());
      block.draggable = true;
      block.appendChild(row);
    } else if (it.kind === "stop") {
      row.classList.add("stop");
      row.textContent = "⏹ STOP — drag between tasks";
      row.title = "The sub-agent stops when the plan reaches this marker";
      block.draggable = true;
      block.appendChild(row);
    } else {
      // task
      row.classList.add("task");
      if (it.done) row.classList.add("done");
      row.appendChild(mkCheck(it, row));
      row.appendChild(mkText(it));
      if (it._picking) {
        row.appendChild(toolPicker(it, (v) => {
          it.tool = v;
          it._picking = false;
          renderTodoFlow();
        }));
      } else if (it.tool) {
        const chip = document.createElement("span");
        chip.className = "flow-chip";
        chip.textContent = `🔧 ${it.tool} · whole task`;
        chip.title = "Used throughout this task — click to change";
        chip.addEventListener("click", () => {
          it._picking = true;
          renderTodoFlow();
        });
        row.appendChild(chip);
      }
      row.appendChild(mkDel());
      block.appendChild(row);
      if (it.subtasks.length) {
        const subs = document.createElement("div");
        subs.className = "flow-subs";
        for (const st of it.subtasks) {
          const sr = document.createElement("div");
          sr.className = "flow-sub-row" + (st.done ? " done" : "");
          sr.appendChild(mkCheck(st, sr));
          sr.appendChild(mkText(st));
          if (st._picking) {
            sr.appendChild(toolPicker(st, (v) => {
              st.tool = v;
              st._picking = false;
              renderTodoFlow();
            }));
          } else if (st.tool) {
            const chip = document.createElement("span");
            chip.className = "flow-chip";
            chip.textContent = `🔧 ${st.tool}`;
            chip.title = "Tool for this sub-task — click to change";
            chip.addEventListener("click", () => {
              st._picking = true;
              renderTodoFlow();
            });
            sr.appendChild(chip);
          }
          const del = document.createElement("button");
          del.className = "flow-del";
          del.textContent = "✕";
          del.addEventListener("click", () => {
            it.subtasks = it.subtasks.filter((s) => s._id !== st._id);
            renderTodoFlow();
          });
          sr.appendChild(del);
          subs.appendChild(sr);
        }
        block.appendChild(subs);
      }
      block.draggable = true;
    }

    block.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", `move:${it._id}`);
      e.dataTransfer.effectAllowed = "move";
      block.classList.add("dragging");
    });
    block.addEventListener("dragend", () => block.classList.remove("dragging"));
    // Dropping a Tool/Sub-task card ON a task row = the gap right after it.
    block.addEventListener("dragover", (e) => e.preventDefault());
    block.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      handleFlowDrop(e, i + 1);
    });
    todoFlow.appendChild(block);
  });
  appendGap(flowItems.length);
  refreshTodoPreview();
}

// ---- palette: drag sources + click-to-append fallback ------------------------

document.querySelectorAll(".palette-card").forEach((card) => {
  card.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("text/plain", `new:${card.dataset.card}`);
    e.dataTransfer.effectAllowed = "copy";
  });
  card.addEventListener("click", () => insertCard(card.dataset.card, flowItems.length));
});

// ---- modal lifecycle ---------------------------------------------------------

async function openTodoModal() {
  if (currentMode !== "plan") {
    addSystemMessage("Switch to Plan mode to edit the plan manually.");
    return;
  }
  // Picker data: defined sub-agents + enabled user tools.
  try { todoSubAgents = await invoke("list_sub_agents"); } catch (e) { todoSubAgents = []; }
  try {
    const rows = await invoke("list_tools_for_active_profile");
    todoTools = (rows || []).map((r) => r.name);
  } catch (e) { todoTools = []; }
  flowItems = parseFlow(todoMarkdown);
  todoModal.hidden = false;
  renderTodoFlow();
}
function closeTodoModal() {
  todoModal.hidden = true;
}

todoOpenBtn.addEventListener("click", openTodoModal);
todoModalCloseBtn.addEventListener("click", closeTodoModal);
todoCancelBtn.addEventListener("click", closeTodoModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !todoModal.hidden) closeTodoModal();
});
todoSaveBtn.addEventListener("click", async () => {
  todoSaveBtn.disabled = true;
  try {
    const md = compileFlow();
    await invoke("set_todo", { markdown: md });
    todoMarkdown = md;
    closeTodoModal();
    renderTodo();
  } catch (e) {
    addSystemMessage(`Error saving the to-do list: ${e}`);
  } finally {
    todoSaveBtn.disabled = false;
  }
});
todoCollapseBtn.addEventListener("click", () => {
  const collapsed = todoPanel.classList.toggle("collapsed");
  todoCollapseBtn.textContent = collapsed ? "▸" : "▾";
  todoCollapseBtn.title = collapsed ? "Expand" : "Collapse";
});

// ----- Send message ---------------------------------------------------------
async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || isAgentBusy) return;

  // Slash commands (client-side).
  if (text.startsWith("/")) {
    handleSlashCommand(text);
    messageInput.value = "";
    return;
  }

  // Add user bubble.
  addUserMessage(text);
  messageInput.value = "";
  isAgentBusy = true;
  sendBtn.disabled = true;
  streamingBubble = null; // will be created on first delta
  workingBlock = null;
  toolCards = {};
  setPhase("Working…");
  refreshHealthLabel(); // flip the health-bar busy indicator immediately

  try {
    await invoke("send_message", { text });
  } catch (e) {
    addSystemMessage(`Error: ${e}`);
    isAgentBusy = false;
    sendBtn.disabled = false;
    refreshHealthLabel();
  }
}

function handleSlashCommand(text) {
  const [cmd, ...rest] = text.split(" ");
  const arg = rest.join(" ").trim();
  switch (cmd) {
    case "/help":
      addSystemMessage(
        "Commands: /model <name>, /new, /clear, /cc (compact the conversation into a context file + new session), /learn (update the agent's project memory: what's to do & what's done), /context-resume, /help"
      );
      break;
    case "/new":
      invoke("new_session").catch((e) => addSystemMessage(`Error: ${e}`));
      chatMessages.innerHTML = "";
      sessionViews.clear(); // fresh chat — past-session views close with it
      refreshSessionRail();
      addSystemMessage("New session started.");
      break;
    case "/clear":
      chatMessages.innerHTML = "";
      break;
    case "/model":
      if (arg) {
        switchModel(arg);
      } else {
        addSystemMessage(`Current model: ${currentModel}`);
      }
      break;
    case "/cc":
      // Context Compacting: summarize the whole conversation into a context
      // file, then start a fresh session (the history is replaced by the
      // compact — nothing deleted, the old session stays in the DB).
      addSystemMessage("Compacting context — summarizing the conversation…");
      invoke("compact_context")
        .then((name) => {
          if (!name) return;
          chatMessages.innerHTML = "";
          sessionViews.clear(); // the chat was cleared — past views go with it
          refreshSessionRail(); // dash flips to green (learned + compacted)
          addSystemMessage(
            `Context compacted into “${name}” — new session started with the compact loaded.`
          );
        })
        .catch((e) => addSystemMessage(`Error: ${e}`));
      break;
    case "/learn":
      addSystemMessage("Updating project memory — what's to do & what's done…");
      invoke("learn")
        .then((r) => {
          if (r) addSystemMessage(r);
          refreshSessionRail(); // dash flips to orange/green
        })
        .catch((e) => addSystemMessage(`Error: ${e}`));
      break;
    case "/context-resume":
      addSystemMessage("Preparing a project résumé from your profile & context files…");
      invoke("context_resume").catch((e) => addSystemMessage(`Error: ${e}`));
      break;
    default:
      addSystemMessage(`Unknown command: ${cmd}`);
  }
}

// ----- Event listeners ------------------------------------------------------
function setupListeners() {
  // Agent events.
  listen("agent-event", (event) => {
    const payload = event.payload;
    switch (payload.type) {
      case "assistant_reasoning": {
        if (!workingBlock) {
          workingBlock = ensureWorkingBlock();
        }
        const content = workingBlock.querySelector(".thinking-body .content");
        content.dataset.raw = (content.dataset.raw || "") + payload.delta;
        content.textContent += payload.delta;
        updateThinkingMeta(workingBlock);
        setPhaseTitle("Thinking…");
        scrollToBottom();
        break;
      }
      case "assistant_delta": {
        // The visible answer begins → fold the thinking block away (auto-collapse).
        endReasoningSegment();
        if (!streamingBubble) {
          streamingBubble = addAssistantMessage("");
        }
        streamingBubble.querySelector(".content").textContent += payload.delta;
        setPhase("Answering…");
        scrollToBottom();
        break;
      }
      case "assistant_message": {
        endReasoningSegment();
        if (streamingBubble) {
          // Finalize: render markdown.
          const content = streamingBubble.querySelector(".content");
          content.innerHTML = renderMarkdown(payload.text);
          streamingBubble = null;
        } else if (payload.text) {
          addAssistantMessage(payload.text);
        }
        // A new assistant message starts a fresh tool batch (indices reset to 0).
        toolCards = {};
        scrollToBottom();
        break;
      }
      case "tool_started": {
        endReasoningSegment();
        streamingBubble = null; // stop streaming into an assistant bubble
        createToolActivityCard(payload.index, payload.name, payload.args);
        setPhase(`Running ${toolLabel(payload.name)}…`);
        break;
      }
      case "tool_needs_approval": {
        endReasoningSegment();
        // Render the card (if not already) and transition it to approval state.
        if (!toolCards[payload.index]) {
          createToolActivityCard(payload.index, payload.name, payload.args);
        }
        setToolActivityApproval(payload.index, payload.name);
        setPhase(`Needs approval: ${toolLabel(payload.name)}`);
        break;
      }
      case "tool_finished": {
        finishToolActivityCard(payload.index, payload.success, payload.result, payload.duration_ms);
        break;
      }
      case "tool_denied": {
        if (toolCards[payload.index]) {
          denyToolActivityCard(payload.index);
        } else {
          addSystemMessage(`Tool denied: ${payload.name}`);
        }
        break;
      }
      case "subagent_started": {
        const card = toolCards[payload.index];
        if (card) {
          const body = card.querySelector(".activity-body");
          body.innerHTML = "";
          body.appendChild(activitySubagentBody(payload.name, payload.model, payload.task));
          const out = document.createElement("div");
          out.className = "sa-out";
          body.appendChild(out);
          setActivityStatus(card, "working…", "live");
        }
        setPhase(`Delegating to ${payload.name}…`);
        break;
      }
      case "subagent_delta": {
        const out = toolCards[payload.index]?.querySelector(".sa-out");
        if (out) {
          out.textContent += payload.text;
          scrollToBottom();
        }
        break;
      }
      case "subagent_finished": {
        const card = toolCards[payload.index];
        if (card) setActivityStatus(card, "✓ done", "done");
        break;
      }
      case "turn_done": {
        isAgentBusy = false;
        sendBtn.disabled = false;
        endReasoningSegment();
        streamingBubble = null;
        toolCards = {};
        setPhase(null);
        refreshHealthLabel(); // idle indicator + final metrics right away
        refreshSessionRail(); // a new session row may have appeared
        autosaveSeal(); // task finished — carry it in the exe
        break;
      }
      case "error": {
        const msg = payload.message || payload.error || JSON.stringify(payload);
        addSystemMessage(`Error: ${msg}`);
        addModelActivity({
          kind: "error",
          title: "Error — I need your help",
          status: "turn failed",
          state: "attention",
          body: activityErrorBody(
            msg,
            "The agent stopped this turn. Check the console in the sidebar, or send the request again."
          ),
        });
        isAgentBusy = false;
        sendBtn.disabled = false;
        endReasoningSegment();
        streamingBubble = null;
        setPhase(null);
        refreshHealthLabel();
        break;
      }
      case "status": {
        const msg = payload.message || payload.status || JSON.stringify(payload);
        addSystemMessage(msg);
        break;
      }
      case "todo_updated": {
        // The agent replaced the shared plan (via the update_todo tool).
        todoMarkdown = payload.markdown || "";
        renderTodo();
        // If the editor modal is open, rebuild the flow from the new plan.
        if (!todoModal.hidden) {
          flowItems = parseFlow(todoMarkdown);
          renderTodoFlow();
        }
        break;
      }
    }
  });

  // Health updates.
  listen("health-update", (event) => {
    updateHealth(event.payload);
  });

  // Model/route changes (set_model + the Models-panel Run buttons all funnel
  // through `apply_model` on the backend). Keeps the chat selector and the
  // Models panel in sync in BOTH directions.
  listen("model-changed", (event) => {
    const route = event.payload;
    if (!route || !route.model) return;
    syncModelFromRoute(route);
    // If the Models panel is open, move the "Running" indicator + box glow.
    if (modelsPanel && !modelsPanel.hidden) refreshModelsPanel();
    refreshHealthLabel();
  });
}

// ----- Message helpers ------------------------------------------------------
function addUserMessage(text) {
  const div = document.createElement("div");
  // No role badge: the right-anchored mirrored glass already reads as "you".
  div.className = "message user";
  div.innerHTML = `<div class="content">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addAssistantMessage(text) {
  const div = document.createElement("div");
  div.className = "message assistant";
  div.innerHTML = `<div class="role-badge">Phoenix</div><div class="content">${renderMarkdown(text)}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

// ----- Sidebar console -------------------------------------------------------
// Every functional message + error lands here now (workdir/profile changes,
// VRAM verdicts, failures…) — the chat stays conversation-only. The console
// doubles as a user shell in the active workdir.

/** Bounded console command history for ↑/↓ navigation. */
const consoleHistory = [];
let consoleHistoryIdx = 0;

/** Append a line to the sidebar console. `kind`: info | warn | err | cmd | dim. */
function logConsole(text, kind = "info") {
  if (!consoleOutput) return;
  const line = document.createElement("div");
  line.className = `console-line c-${kind}`;
  line.textContent = text;
  consoleOutput.appendChild(line);
  while (consoleOutput.childElementCount > 500) {
    consoleOutput.firstElementChild.remove();
  }
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

/** Run the console input as a shell command in the active workdir. */
async function runConsoleCommand() {
  const command = consoleInput.value.trim();
  if (!command) return;
  consoleInput.value = "";
  consoleHistory.push(command);
  if (consoleHistory.length > 50) consoleHistory.shift();
  consoleHistoryIdx = consoleHistory.length;
  if (command.toLowerCase() === "clear" || command.toLowerCase() === "cls") {
    consoleOutput.innerHTML = "";
    return;
  }
  logConsole(`❯ ${command}`, "cmd");
  try {
    const out = await invoke("console_run", { command });
    if (out.stdout) logConsole(out.stdout.replace(/\n+$/, ""), "info");
    if (out.stderr) logConsole(out.stderr.replace(/\n+$/, ""), "err");
    if (out.exit_code !== 0) logConsole(`(exit code ${out.exit_code})`, "err");
    if (!out.stdout && !out.stderr && out.exit_code === 0) logConsole("(no output)", "dim");
  } catch (e) {
    logConsole(String(e), "err");
  }
}

/**
 * Functional message — workdir/profile changes, VRAM verdicts, errors, tool
 * denials… Since the console rework these render in the SIDEBAR CONSOLE
 * (color-coded by severity), not as chat bubbles.
 */
function addSystemMessage(text) {
  const kind = /^error/i.test(text)
    ? "err"
    : /REFUSED|Low VRAM|failed|Failed|timed out|denied/i.test(text)
      ? "warn"
      : "info";
  logConsole(text, kind);
}

// ----- Reasoning / tool pipeline UI ----------------------------------------
// Builds the visible step-by-step pipeline: an inline working indicator (the
// phase label + live reasoning stream, riding the pipeline's tail), collapsible
// thinking chips it leaves behind, expandable tool cards (correlated
// start↔finish), and nested sub-agent cards. All reuse the glass / role-tint /
// flame-glow language.

/** Live phase label, shown in the inline working indicator inside the chat.
 *  `null` (or the answering phase, whose stream is visible on its own) ends
 *  the indicator; any other label (re)creates it at the END of the chat — the
 *  pipeline's tail — so it always sits where the next event will land. */
function setPhase(label) {
  if (!label || label === "Answering…") {
    removeWorkingBlock();
    return;
  }
  const block = ensureWorkingBlock();
  setPhaseTitle(label);
}

/** Update the working block's header label (no-op when titles match). */
function setPhaseTitle(label) {
  if (!workingBlock) return;
  const title = workingBlock.querySelector(".thinking-title");
  if (title.textContent !== label) title.textContent = label;
}

/** Human label for a tool name. */
function toolLabel(name) {
  return (name || "tool").replace(/_/g, " ");
}

/** The inline working indicator: a thinking-block variant whose header is a
 *  pulsing dot + the live phase label, and whose body streams the model's
 *  reasoning so the user can monitor it in real time. It rides the tail of the
 *  chat pipeline (appendChild MOVES an existing block to the end, so it always
 *  follows the latest thinking chip / tool card). */
function ensureWorkingBlock() {
  if (!workingBlock) {
    const block = document.createElement("div");
    // Same header anatomy as the activity cards (icon chip + title + meta +
    // chevron) so collapsed rows align across the whole family. Starts
    // collapsed — click to watch the stream.
    block.className = "thinking-block working active collapsed";
    block.innerHTML = `
      <div class="thinking-header">
        <span class="activity-icon">◷</span>
        <span class="thinking-title">Working…</span>
        <span class="thinking-meta"></span>
        <span class="chevron">▸</span>
      </div>
      <div class="thinking-body"><div class="content"></div></div>`;
    block.querySelector(".thinking-header").addEventListener("click", () => {
      block.classList.toggle("collapsed");
      block.querySelector(".chevron").textContent = block.classList.contains("collapsed") ? "▸" : "▾";
    });
    workingBlock = block;
  }
  chatMessages.appendChild(workingBlock);
  scrollToBottom();
  return workingBlock;
}

/** Drop the inline working indicator (turn over, or answering — the answer
 *  stream is its own indicator). */
function removeWorkingBlock() {
  if (workingBlock) {
    workingBlock.remove();
    workingBlock = null;
  }
}

/** Update the "N lines · M chars" meta on a thinking block while it streams. */
function updateThinkingMeta(block) {
  const raw = block.querySelector(".thinking-body .content").dataset.raw || "";
  const lines = raw.split("\n").length;
  const chars = raw.length;
  block.querySelector(".thinking-meta").textContent = `${lines} line${lines === 1 ? "" : "s"} · ${chars} chars`;
}

/** Freeze the current reasoning segment: with content it becomes the standard
 *  collapsed "Thinking · N lines" chip IN PLACE (pipeline order preserved);
 *  empty it stays the live working indicator — the next phase event re-rides
 *  it to the chat's tail. */
function endReasoningSegment() {
  if (!workingBlock) return;
  const block = workingBlock;
  const content = block.querySelector(".thinking-body .content");
  const raw = content.dataset.raw || "";
  if (!raw.trim()) return; // nothing reasoned — keep the live indicator
  workingBlock = null;
  block.classList.remove("active", "working");
  block.classList.add("collapsed");
  // The ◷ icon chip is already in place from ensureWorkingBlock — just
  // retitle it as the resting "Thinking" chip.
  const title = block.querySelector(".thinking-title");
  if (title) title.textContent = "Thinking";
  content.innerHTML = renderMarkdown(raw);
  updateThinkingMeta(block);
  block.querySelector(".chevron").textContent = "▸";
}

// ----- Tool pipeline → activity cards ----------------------------------------
// Real agent events render through the activity-card family: file tools →
// explore cards, write/edit → code cards (language badge + REAL ± counts
// derived from the call args), run_command → terminal, delegate → sub-agent,
// unknown/user tools → heuristics (web/image/3D by name) or a generic task
// card. Cards correlate by tool index, exactly like the old tool cards.

/** Built-in tool → activity kind + title. */
const TOOL_KINDS = {
  read_file: { kind: "explore", title: "Reading file" },
  list_dir: { kind: "explore", title: "Exploring files" },
  grep: { kind: "explore", title: "Searching files" },
  write_file: { kind: "code", title: "Writing code" },
  edit_file: { kind: "code", title: "Editing code" },
  run_command: { kind: "terminal", title: "Running task" },
  delegate: { kind: "subagent", title: "Running sub-agent" },
  update_todo: { kind: "terminal", title: "Updating plan" },
};

/** Heuristic kind for user-defined tools (name-based). */
function guessToolKind(name) {
  const n = (name || "").toLowerCase();
  if (/web|browse|fetch|url|http/.test(n)) return { kind: "web", title: "Browsing the web" };
  if (/image|draw|paint|diffuse/.test(n)) return { kind: "image", title: "Generating image" };
  if (/3d|mesh|cad/.test(n)) return { kind: "model3d", title: "Generating 3D model" };
  return { kind: "terminal", title: `Running ${toolLabel(name)}` };
}

/** Body for a tool's live phase, from its parsed args. */
function buildToolBody(kind, name, args) {
  switch (kind) {
    case "explore":
      return activityFilesBody(String(args.path || args.pattern || "."), []);
    case "code": {
      const path = String(args.path || "");
      const snippet = [];
      if (args.find) String(args.find).split("\n").slice(0, 6).forEach((l) => snippet.push(`- ${l}`));
      if (args.replace) String(args.replace).split("\n").slice(0, 6).forEach((l) => snippet.push(`+ ${l}`));
      if (args.content) String(args.content).split("\n").slice(0, 6).forEach((l) => snippet.push(`+ ${l}`));
      return activityCodeBody(path, snippet.length ? snippet : ["(new file)"]);
    }
    case "terminal":
      return activityTermBody(String(args.command || toolLabel(name)), []);
    case "subagent":
      return activitySubagentBody(String(args.sub_agent || "sub-agent"), "", String(args.task || ""));
    case "web":
      return activityWebBody(String(args.query || args.url || args.pattern || ""), []);
    case "image":
    case "model3d":
      return activityProgressBody(0, "starting…");
    default:
      return null;
  }
}

/** Create the activity card for a tool call (running state). */
function createToolActivityCard(index, name, argsJson) {
  let args = {};
  try { args = JSON.parse(argsJson || "{}"); } catch { /* keep {} */ }
  const spec = TOOL_KINDS[name] || guessToolKind(name);
  let plus = 0;
  let minus = 0;
  if (spec.kind === "code") {
    // Real diff counts from the call args: edit = replace vs find lines,
    // write = new content lines.
    if (args.find) minus = String(args.find).split("\n").length;
    if (args.replace) plus = String(args.replace).split("\n").length;
    if (args.content) plus = String(args.content).split("\n").length;
  }
  const card = addModelActivity({
    kind: spec.kind,
    title: spec.title,
    status: "running…",
    state: "live",
    lang: spec.kind === "code" ? String(args.path || "") : undefined,
    plus,
    minus,
    body: buildToolBody(spec.kind, name, args),
  });
  card.dataset.toolName = name;
  toolCards[index] = card;
  return card;
}

/** Transition a card to its approval state with inline Approve/Deny. */
function setToolActivityApproval(index, name) {
  const card = toolCards[index];
  if (!card) return;
  setActivityStatus(card, "needs approval", "attention");
  card.querySelector(".activity-body").appendChild(
    activityQuestionBody(
      `Approve “${name}”? The agent is waiting to continue.`,
      () => invoke("approve", { index }).catch((e) => addSystemMessage(`Error: ${e}`)),
      () => invoke("deny", { index }).catch((e) => addSystemMessage(`Error: ${e}`))
    )
  );
  scrollToBottom();
}

/** Fill the result, mark done/failed, and collapse. */
function finishToolActivityCard(index, success, result, durationMs) {
  let card = toolCards[index];
  if (!card) {
    // Defensive: a finish without a start (e.g. denied before start).
    card = createToolActivityCard(index, "?", "{}");
  }
  const dur = durationMs != null ? ` · ${formatDuration(durationMs)}` : "";
  setActivityStatus(card, success ? `✓ done${dur}` : `✗ failed`, "done");
  card.classList.toggle("failed", !success);
  card.querySelector(".activity-question")?.remove();
  card.querySelector(".t-cursor")?.remove();
  const pre = document.createElement("pre");
  pre.className = "activity-result";
  pre.textContent = String(result ?? "").slice(0, 2000);
  card.querySelector(".activity-body").appendChild(pre);
  card.classList.add("collapsed");
  card.querySelector(".chevron").textContent = "▸";
  scrollToBottom();
}

/** Mark a card denied. */
function denyToolActivityCard(index) {
  const card = toolCards[index];
  if (!card) return;
  setActivityStatus(card, "✗ denied", "done");
  card.classList.add("failed");
  card.querySelector(".activity-question")?.remove();
  card.classList.add("collapsed");
  card.querySelector(".chevron").textContent = "▸";
}

// ----- Model activity cards ---------------------------------------------------
// The message family for "what the model is doing right now" — the states the
// model reports to the user while it works: exploring files, writing code,
// running a terminal task, asking validation, erroring out, generating images
// / 3D models. One anatomy (icon chip + title + status pill + collapsible
// body, see the .activity CSS family) with a per-kind accent; each card's
// body is composed by the caller from the small activity*Body helpers.
// "Thinking" stays its own component (.thinking-block).

/** Per-kind look: glyph icon (tinted by the kind's accent) + default title. */
const ACTIVITY_KINDS = {
  explore: { icon: "☰", label: "Exploring files" },
  code: { icon: "✎", label: "Writing code" },
  terminal: { icon: ">_", label: "Running task" },
  approval: { icon: "?", label: "Validation needed" },
  error: { icon: "⚠", label: "Error" },
  image: { icon: "▦", label: "Generating image" },
  model3d: { icon: "⬢", label: "Generating 3D model" },
  web: { icon: "◎", label: "Browsing the web" },
  subagent: { icon: "✦", label: "Running sub-agent" },
};

/** 16×16 colored language badges (inline SVG, brand-colored). Shield shape
 *  for HTML/CSS mimics the official marks; the rest are rounded monogram
 *  tiles in the language's brand color. */
const LANG_BADGES = {
  rust: { bg: "#CE422B", fg: "#FFFFFF", label: "Rs" },
  javascript: { bg: "#F7DF1E", fg: "#000000", label: "JS" },
  typescript: { bg: "#3178C6", fg: "#FFFFFF", label: "TS" },
  python: { bg: "#3776AB", fg: "#FFFFFF", label: "Py" },
  html: { bg: "#E34F26", fg: "#FFFFFF", label: "5", shield: true },
  css: { bg: "#1572B6", fg: "#FFFFFF", label: "3", shield: true },
  json: { bg: "#6E7B8B", fg: "#FFFFFF", label: "{}", size: 7 },
  c: { bg: "#A8B9CC", fg: "#0E1420", label: "C" },
  cpp: { bg: "#00599C", fg: "#FFFFFF", label: "C++", size: 7 },
  go: { bg: "#00ADD8", fg: "#FFFFFF", label: "Go" },
  shell: { bg: "#89E051", fg: "#0B2E13", label: ">_", size: 7.5, mono: true },
  markdown: { bg: "#083FA1", fg: "#FFFFFF", label: "M" },
  yaml: { bg: "#CB171E", fg: "#FFFFFF", label: "Y" },
  code: { bg: "#787878", fg: "#E8E8E8", label: "</>", size: 6.5 },
};

/** Map a file path/extension to a LANG_BADGES key (null when unknown). */
function detectLangFromPath(path) {
  const ext = (path || "").split(".").pop().toLowerCase();
  const map = {
    rs: "rust", js: "javascript", mjs: "javascript", cjs: "javascript",
    jsx: "javascript", ts: "typescript", tsx: "typescript", py: "python",
    html: "html", htm: "html", css: "css", json: "json", c: "c", h: "c",
    cpp: "cpp", cc: "cpp", cxx: "cpp", hpp: "cpp", go: "go",
    sh: "shell", bash: "shell", ps1: "shell", bat: "shell", cmd: "shell",
    md: "markdown", markdown: "markdown", yaml: "yaml", yml: "yaml",
  };
  return map[ext] || null;
}

/** Build the 16×16 colored SVG badge for a language key. */
function langBadgeSvg(lang) {
  const b = LANG_BADGES[lang];
  if (!b) return "";
  const fs = b.size || 8.5;
  const fam = b.mono ? "'Cascadia Code', monospace" : "'Exo 2', sans-serif";
  const shape = b.shield
    ? `<path d="M2 1h12l-1.1 12L8 15.4 3.1 13z" fill="${b.bg}"/>`
    : `<rect width="16" height="16" rx="3" fill="${b.bg}"/>`;
  return `<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${lang}">${shape}<text x="8" y="${b.shield ? 10.5 : 11.5}" text-anchor="middle" font-family="${fam}" font-size="${fs}" font-weight="700" fill="${b.fg}">${b.label}</text></svg>`;
}

/**
 * Add a model-activity card to the chat. `state`: "live" (in progress — the
 * icon chip pulses) | "attention" (needs the user: approval, error) |
 * "done". `body` is an element built by the activity*Body helpers (null for
 * a header-only card). Code cards additionally take `lang` (badge key or a
 * file path to detect from — colored 16×16 language icon before the status
 * pill) and `plus`/`minus` (starting diff counts shown after the pill).
 * Returns the card so callers can update it.
 */
function addModelActivity({ kind, title, status, state = "live", body, lang, plus = 0, minus = 0 }) {
  const spec = ACTIVITY_KINDS[kind] || {};
  const card = document.createElement("div");
  // Cards start COLLAPSED (compact one-line rows); expanding grows the card.
  card.className = `activity kind-${kind} ${state} collapsed`;
  card.innerHTML = `
    <div class="activity-header">
      <span class="activity-icon">${spec.icon || "●"}</span>
      <span class="activity-title">${escapeHtml(title || spec.label || "Working…")}</span>
      <span class="activity-lang-slot"></span>
      <span class="activity-status"></span>
      <span class="activity-diff"></span>
      <span class="chevron">▸</span>
    </div>
    <div class="activity-body"></div>`;
  if (status) card.querySelector(".activity-status").textContent = status;
  const langKey = lang ? (LANG_BADGES[lang] ? lang : detectLangFromPath(lang)) : null;
  if (langKey) {
    const slot = card.querySelector(".activity-lang-slot");
    slot.className = "activity-lang";
    slot.title = langKey;
    slot.innerHTML = langBadgeSvg(langKey);
    card.dataset.lang = langKey;
  }
  if (plus || minus) setCodeDiff(card, plus, minus);
  card.querySelector(".activity-header").addEventListener("click", () => {
    card.classList.toggle("collapsed");
    card.querySelector(".chevron").textContent = card.classList.contains("collapsed") ? "▸" : "▾";
  });
  if (body) card.querySelector(".activity-body").appendChild(body);
  chatMessages.appendChild(card);
  scrollToBottom();
  return card;
}

/** Set the code card's +/- diff counters absolutely. */
function setCodeDiff(card, plus, minus) {
  if (!card) return;
  card.dataset.plus = String(plus);
  card.dataset.minus = String(minus);
  const diff = card.querySelector(".activity-diff");
  diff.innerHTML =
    `<span class="d-add">${plus > 0 ? `+${plus}` : ""}</span>` +
    `<span class="d-del">${minus > 0 ? `-${minus}` : ""}</span>`;
  diff.hidden = !(plus > 0 || minus > 0);
}

/** Increment the code card's diff counters as the model keeps editing. */
function bumpCodeDiff(card, dPlus = 0, dMinus = 0) {
  if (!card) return;
  setCodeDiff(card, (+(card.dataset.plus || 0)) + dPlus, (+(card.dataset.minus || 0)) + dMinus);
}

/** Update a card's status pill / state class as the activity progresses. */
function setActivityStatus(card, status, state) {
  if (!card) return;
  if (status != null) card.querySelector(".activity-status").textContent = status;
  if (state) {
    card.classList.remove("live", "attention", "done");
    card.classList.add(state);
  }
}

/** File-tree body for explore cards: a path chip + mono entry rows. */
function activityFilesBody(path, entries) {
  const div = document.createElement("div");
  div.className = "activity-files";
  div.innerHTML = `
    <div class="activity-path">${escapeHtml(path)}</div>
    ${entries
      .map(
        (e) => `
    <div class="activity-file${e.dir ? " dir" : ""}">
      <span>${e.dir ? "▸" : "·"} ${escapeHtml(e.name)}${e.dir ? "/" : ""}</span>
      <span class="activity-file-size">${escapeHtml(e.size || "")}</span>
    </div>`
      )
      .join("")}`;
  return div;
}

/** Code body: file chip + colored +/- diff snippet. */
function activityCodeBody(file, snippet) {
  const div = document.createElement("div");
  div.className = "activity-code";
  div.innerHTML = `
    <div class="activity-path">${escapeHtml(file)}</div>
    <pre>${snippet
      .map((l) => {
        const cls = l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : "";
        return `<span class="dl-${cls}">${escapeHtml(l)}</span>`;
      })
      .join("\n")}</pre>`;
  return div;
}

/** Terminal body: accent prompt + output lines + blinking cursor. */function activityTermBody(command, lines) {
  const div = document.createElement("div");
  div.className = "activity-term";
  div.innerHTML = `
    <pre><span class="t-prompt">$</span> ${escapeHtml(command)}
${lines.map((l) => `<span class="t-out">${escapeHtml(l)}</span>`).join("\n")}
<span class="t-cursor">▌</span></pre>`;
  return div;
}

/** Web browsing body: a URL/search chip + result rows with favicon-style
 *  letter dots. `pages`: [{ fav, title, domain }]. */
function activityWebBody(url, pages) {
  const div = document.createElement("div");
  div.className = "activity-web";
  div.innerHTML = `
    <div class="activity-path">${escapeHtml(url)}</div>
    <div class="activity-pages">
      ${(pages || [])
        .map(
          (p) => `
      <div class="activity-page">
        <span class="p-fav">${escapeHtml((p.fav || "•").slice(0, 1))}</span>
        <span class="p-title">${escapeHtml(p.title || "")}</span>
        <span class="p-domain">${escapeHtml(p.domain || "")}</span>
      </div>`
        )
        .join("")}
    </div>`;
  return div;
}

/** Sub-agent body: who was spawned (name + model chips) and the task it was
 *  given, quoted. */
function activitySubagentBody(agent, model, task) {
  const div = document.createElement("div");
  div.className = "activity-subagent";
  div.innerHTML = `
    <div class="sa-row">
      <span class="activity-path">${escapeHtml(agent)}</span>
      <span class="sa-model">${escapeHtml(model)}</span>
    </div>
    <div class="sa-task">“${escapeHtml(task)}”</div>`;
  return div;
}

/** Generation progress body: bar + percentage + context meta. */
function activityProgressBody(pct, meta) {
  const div = document.createElement("div");
  div.className = "activity-progress";
  const p = Math.max(0, Math.min(100, pct));
  div.innerHTML = `
    <div class="activity-bar"><div class="activity-bar-fill" style="width:${p}%"></div></div>
    <div class="activity-meta"><span class="activity-pct">${Math.round(p)}%</span><span class="activity-sub">${escapeHtml(meta || "")}</span></div>`;
  return div;
}

/** Question body for approval cards: text + Approve/Deny. */
function activityQuestionBody(text, onApprove, onDeny) {
  const div = document.createElement("div");
  div.className = "activity-question";
  div.innerHTML = `
    <div class="activity-question-text">${escapeHtml(text)}</div>
    <div class="approval-actions">
      <button class="btn-approve">Approve</button>
      <button class="btn-deny">Deny</button>
    </div>`;
  div.querySelector(".btn-approve").addEventListener("click", () => onApprove?.());
  div.querySelector(".btn-deny").addEventListener("click", () => onDeny?.());
  return div;
}

/** Error body: the error text + a dim "what would help" hint. */
function activityErrorBody(message, hint) {
  const div = document.createElement("div");
  div.className = "activity-error";
  div.innerHTML = `
    <pre>${escapeHtml(message)}</pre>
    ${hint ? `<div class="activity-hint">💡 ${escapeHtml(hint)}</div>` : ""}`;
  return div;
}


/** Pretty-print a JSON string; fall back to the raw string if it isn't JSON. */
function prettyJson(str) {
  if (!str) return "";
  try {
    return JSON.stringify(JSON.parse(str), null, 2);
  } catch {
    return str;
  }
}

/** Format a millisecond duration compactly. */
function formatDuration(ms) {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

// ----- Health bar -----------------------------------------------------------
function updateHealth(state) {
  const components = [
    ["ollama", state.ollama],
    ["model", state.model],
    ["database", state.database],
    ["ripgrep", state.ripgrep],
    ["shell", state.shell],
  ];

  let healthy = 0;
  for (const [key, status] of components) {
    const item = $(`health-${key}`);
    if (!item) continue;
    const dot = item.querySelector(".health-dot");
    const statusKey = status.status; // "ok" | "down" | "checking" | "unknown"
    dot.className = `health-dot ${statusKey}`;
    const detail = status.detail || "";
    item.title = detail;
    if (statusKey === "ok") healthy++;
  }

  healthSummary.textContent = `${healthy}/5`;
  healthSummary.style.color = healthy === 5
    ? "var(--health-green)"
    : healthy === 0
      ? "var(--health-red)"
      : "var(--health-yellow)";

  // Drive the Models nav status dot from the ollama + model probes: green only
  // when Ollama is up AND the active model is pulled.
  const modelsDot = $("dot-models");
  if (modelsDot) {
    const ollamaOk = state.ollama && state.ollama.status === "ok";
    const modelOk = state.model && state.model.status === "ok";
    modelsDot.className = `status-dot ${ollamaOk && modelOk ? "ok" : "down"}`;
  }

  // Relabel the first health item to match the active backend/provider. The
  // label ships as "Ollama" but switches to "AmberCore" or the provider name.
  refreshHealthLabel();
}

/** Relabel the first health-bar item to reflect the active route. Also suffix
 *  the model item's label with the active model name + live runtime metrics
 *  (T/s · TTFT · TBT · generating/idle) so they're visible at a glance. */
async function refreshHealthLabel() {
  let route = null;
  try { route = await invoke("get_active_route"); } catch { /* pre-unlock */ }
  if (!route) return;
  const backendItem = $("health-ollama");
  if (backendItem) {
    const dot = backendItem.querySelector(".health-dot");
    const label = route.kind === "cloud" ? (providerNameCache[route.provider_id] || "Provider") : (route.backend === "ambercore" ? "AmberCore" : "Ollama");
    backendItem.innerHTML = "";
    if (dot) backendItem.appendChild(dot);
    backendItem.appendChild(document.createTextNode(" " + label));
  }
  const modelItem = $("health-model");
  if (modelItem) {
    const dot = modelItem.querySelector(".health-dot");
    const modelLabel = route.model ? `Model: ${route.model}` : "Model";
    modelItem.innerHTML = "";
    if (dot) modelItem.appendChild(dot);
    modelItem.appendChild(document.createTextNode(" " + modelLabel));
  }
  // Live metrics — centered in the health bar, white + bold (dispatch layer,
  // works for every backend). The first thing users monitor.
  const metricsBar = $("health-metrics-bar");
  if (metricsBar) {
    let stats = null;
    try { stats = await invoke("get_runtime_metrics"); } catch { /* pre-unlock */ }
    if (stats) {
      const parts = [];
      if (stats.tokens_per_sec != null) parts.push(`${Number(stats.tokens_per_sec).toFixed(1)} T/s`);
      if (stats.ttft_ms != null) parts.push(`TTFT ${(stats.ttft_ms / 1000).toFixed(1)} s`);
      if (stats.tbt_avg_ms != null) parts.push(`TBT ${Number(stats.tbt_avg_ms).toFixed(1)} ms`);
      const busy = !!(stats.busy || isAgentBusy);
      metricsBar.innerHTML =
        (parts.length ? escapeHtml(parts.join(" · ")) : "") +
        (parts.length ? " · " : "") +
        `<span class="metrics-busy${busy ? " busy" : ""}">${busy ? "● generating" : "○ idle"}</span>`;
      metricsBar.hidden = false;
    } else {
      metricsBar.hidden = true;
    }
  }
}

/** Cache of provider id -> name, so the health label can show the provider name
 *  without an extra invoke on every health tick. Refreshed when the Models
 *  panel opens. */
const providerNameCache = {};

// ----- Utilities ------------------------------------------------------------
function renderMarkdown(text) {
  if (typeof marked !== "undefined" && text) {
    try {
      const html = marked.parse(text);
      // Highlight code blocks after rendering.
      setTimeout(() => {
        document.querySelectorAll(".message.assistant pre code").forEach((block) => {
          if (typeof hljs !== "undefined") hljs.highlightElement(block);
        });
      }, 10);
      return html;
    } catch (e) {
      return escapeHtml(text);
    }
  }
  return escapeHtml(text);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

/** Format a byte count as a short human-readable string. */
function humanBytes(n) {
  const GB = 1024 ** 3, MB = 1024 ** 2, KB = 1024;
  if (n >= GB) return `${(n / GB).toFixed(1)} GB`;
  if (n >= MB) return `${Math.round(n / MB)} MB`;
  if (n >= KB) return `${Math.round(n / KB)} KB`;
  return `${n} B`;
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ----- Sidebar: models / profiles / workdir -------------------------------

/** Populate the sidebar (profile selector + workdir display) after unlock. */
async function loadSidebar(unlockResult) {
  restoreSidebarWidth();
  restoreConsoleHeight();
  await loadWorkdir();
  await loadProfiles(unlockResult?.active_profile);
  // Track the active profile id for skills enable/disable.
  if (unlockResult?.active_profile && unlockResult.active_profile.id != null) {
    activeProfileId = unlockResult.active_profile.id;
  }
  updateSkillsDot();
  updateToolsDot();
  updateContextDot();
  updateMemoryDot();
}

/** Live-switch the active model from the under-Send selector or /model. */
async function switchModel(model) {
  currentModel = model;
  modelSelect.value = model;
  setModelBtnLabel();
  renderModelPopup();
  addSystemMessage(`Switching model to ${model}…`);
  try {
    await invoke("set_model", { model });
    // The backend's `model-changed` event (emitted by set_model) re-syncs the
    // Models panel's "Running" indicator; nothing else to do here.
  } catch (e) {
    addSystemMessage(`Model switch failed: ${e}`);
  }
}

/** Re-sync the chat-side model selector from the backend's active route —
 *  the single source of truth. Used after Models-panel Run buttons and the
 *  `model-changed` event so both selectors always show the same model. */
async function syncModelFromRoute(route) {
  if (!route) {
    try { route = await invoke("get_active_route"); } catch { return; }
  }
  if (!route || !route.model) return;
  currentModel = route.model;
  modelSelect.value = route.model;
  // Re-list (the backend may have changed) + re-render the popup so the green
  // highlight lands on the right row, and update the button label.
  await populateModels();
}

/** Load + render the ACTIVE PROFILE's working directory. Blank profile →
 *  the "select a directory" placeholder (the row opens the native picker). */
async function loadWorkdir() {
  try {
    const wd = await invoke("get_workdir");
    if (wd && wd.trim()) {
      workdirDisplay.textContent = wd;
      workdirDisplay.title = wd;
      workdirDisplay.classList.remove("placeholder");
    } else {
      workdirDisplay.textContent = "Please select a directory for this profile!";
      workdirDisplay.title = "Click to open the directory picker";
      workdirDisplay.classList.add("placeholder");
    }
  } catch (e) {
    console.warn("Workdir load failed:", e);
  }
}

/** Load the profile list into the cache and track the active one. (The
 *  selector dropdown is gone — 🏠 jumps to Default, ＋ creates + activates.)
 *  `activeProfile` is the unlock result's profile when available. */
async function loadProfiles(activeProfile) {
  try {
    const profiles = await invoke("list_profiles");
    if (activeProfile && activeProfile.id != null) {
      activeProfileId = activeProfile.id;
    } else if (profiles.find((p) => p.is_default)) {
      activeProfileId = profiles.find((p) => p.is_default).id;
    } else {
      activeProfileId = profiles[0]?.id ?? null;
    }
  } catch (e) {
    console.warn("Profile load failed:", e);
  }
}

/** Open the Models panel overlay and refresh all three boxes. */
async function openModelsPanel() {
  modelsPanel.hidden = false;
  await refreshModelsPanel();
}

/** Refresh all three boxes + the active-box highlight. */
async function refreshModelsPanel() {
  let route = { kind: "local", backend: "ollama", provider_id: null, model: "" };
  try { route = await invoke("get_active_route"); } catch { /* pre-unlock */ }
  highlightActiveBox(route);
  // AmberCore
  try { const dir = await invoke("get_ambercore_directory"); if (dir) icDir.value = dir; } catch { /* ignore */ }
  await refreshAmberCoreRemote();
  renderAmberCore(route);
  // Ollama
  renderOllama(route);
  // Provider API
  renderProviders(route);
  // Refresh the health-bar label so it reflects the new active backend/model.
  refreshHealthLabel();
}

/** Highlight the active route with the flame glow: the matching runner tab
 *  (AmberCore | Ollama) or the Provider API box for cloud routes. */
function highlightActiveBox(route) {
  runnerBox.querySelectorAll(".runner-tab").forEach((tab) => {
    const matches =
      route.kind === "local" &&
      ((tab.dataset.runner === "ambercore" && route.backend === "ambercore") ||
        (tab.dataset.runner === "ollama" && route.backend === "ollama"));
    tab.classList.toggle("active-route", matches);
  });
  prBox.classList.toggle("active", route.kind === "cloud");
}

/** Switch the models panel's runner: the settings pane, the model-list title,
 *  the acquire row (Find models vs Ollama pull) and the visible list all
 *  follow the selected tab. Persisted so the panel reopens as it was left. */
function setRunnerTab(runner) {
  activeRunner = runner === "ollama" ? "ollama" : "ambercore";
  localStorage.setItem("phoenix.runnerTab", activeRunner);
  runnerBox.querySelectorAll(".runner-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.runner === activeRunner);
  });
  paneAmberCore.hidden = activeRunner !== "ambercore";
  paneOllama.hidden = activeRunner !== "ollama";
  mlTitle.textContent =
    activeRunner === "ollama" ? "Ollama models" : "AmberCore models";
  mlAcquireAmberCore.hidden = activeRunner !== "ambercore";
  mlAcquireOllama.hidden = activeRunner !== "ollama";
  icList.hidden = activeRunner !== "ambercore";
  olList.hidden = activeRunner !== "ollama";
}

/** Render the AmberCore model list (blue box). */
async function renderAmberCore(route) {
  icList.innerHTML = '<li class="panel-loading">Loading AmberCore models…</li>';
  let models = [];
  try {
    models = await invoke("list_ambercore_models");
  } catch (e) {
    icList.innerHTML = `<li class="panel-loading">Failed to load: ${escapeHtml(String(e))}</li>`;
    return;
  }
  if (models.length === 0) {
    icList.innerHTML = '<li class="panel-loading">No AmberCore models found. Pull one with "Find models…".</li>';
    return;
  }
  icList.innerHTML = "";
  const active = route.kind === "local" && route.backend === "ambercore";
  for (const m of models) {
    const isActive = active && m.name === route.model;
    const li = document.createElement("li");
    li.className = "mp-row";
    li.innerHTML =
      `<span class="mp-name" title="${escapeHtml(m.size)}">${escapeHtml(m.name)}</span>` +
      `<span class="mp-meta">${escapeHtml(m.quantization)}</span>` +
      `<span class="mp-sep">|</span>` +
      `<span class="mp-meta">${escapeHtml(m.downloaded_at)}</span>` +
      `<button class="btn-run">${isActive ? "Running" : "Run"}</button>` +
      `<button class="btn-del" title="Delete model + tokenizer">🗑</button>`;
    li.querySelector(".btn-run").addEventListener("click", () => runAmberCore(m.name));
    li.querySelector(".btn-del").addEventListener("click", () => deleteAmberCoreModel(m.name));
    if (isActive) li.querySelector(".btn-run").style.borderColor = "var(--phoenix-warm)";
    icList.appendChild(li);
  }
}

/** Pull a GGUF model (and its auto-detected tokenizer) from a URL into the
 *  AmberCore models directory — the shared path behind the search modal's
 *  Pull buttons. Progress UI is event-driven (see the pull-bars helpers), so
 *  several pulls can run at once, each with its own bar. */
async function pullAmberCoreFromUrl(url) {
  try {
    const tag = await invoke("pull_ambercore_model", { url, tokenizerUrl: null });
    addSystemMessage(`Pulled AmberCore model: ${tag} (model + tokenizer ready)`);
    await renderAmberCore(await invoke("get_active_route"));
    return true;
  } catch (e) {
    addSystemMessage(`AmberCore pull failed: ${e}`);
    return false;
  }
}

// ----- Models-button download progress ring -----------------------------------
//
// Tracks each active pull's (completed, total) and shows a thin circular
// progress ring with the completion percentage inside it at the far right of
// the Models nav item — visible even with the panel closed, which is the
// point. The percentage replaced the earlier time-ETA text (byte rates swing
// too much to be a stable readout).

const pullEta = new Map(); // id -> { total, completed }

/** Feed a progress sample (bytes) for one pull; refresh the nav indicator. */
function etaTrack(id, completed, total) {
  let st = pullEta.get(id);
  if (!st) {
    st = { total: total ?? null, completed };
    pullEta.set(id, st);
  } else {
    st.completed = completed;
    if (total != null) st.total = total;
  }
  updateNavEta();
}

function etaDrop(id) {
  pullEta.delete(id);
  updateNavEta();
}

/** Aggregate all active pulls into the ring; hide when idle. */
function updateNavEta() {
  if (!navEta) return;
  const active = [...pullEta.values()];
  if (!active.length) {
    navEta.hidden = true;
    return;
  }
  let done = 0, all = 0, known = 0;
  for (const st of active) {
    if (st.total != null) {
      done += st.completed;
      all += st.total;
      known++;
    }
  }
  const pct = all > 0 ? (done / all) * 100 : 0;
  navEta.hidden = false;
  if (navEtaRing) {
    navEtaRing.style.setProperty("--eta-pct", known ? pct.toFixed(1) : "0");
  }
  if (navEtaPct) {
    navEtaPct.textContent = known ? `${Math.round(pct)}%` : "";
  }
}

// ----- Session history rail ---------------------------------------------------
// Thin vertical dash list at the chat's left edge: one dash per PAST session
// (newest top), color = status — green = learned + compacted, orange =
// learned only, red = wild (never learned). Clicking a dash loads that
// session read-only into a collapsible container above the live chat, in
// chronological order (oldest top). The newest session is skipped — it is
// the live one.

const sessionRail = $("session-rail");
let sessionSummaries = [];
const sessionMsgCache = new Map(); // id -> messages
const sessionViews = new Map(); // id -> container element (currently open)

function sessionDashClass(s) {
  if (s.learned_at && s.compacted_at) return "learned compacted";
  if (s.learned_at) return "learned";
  return ""; // wild — red by default
}

/** Re-fetch sessions and re-render the rail (colors/order/actives). */
async function refreshSessionRail() {
  if (!sessionRail) return;
  try {
    sessionSummaries = await invoke("list_sessions");
  } catch {
    return; // pre-unlock or transient — keep the rail as-is
  }
  const viewable = sessionSummaries.slice(1); // [0] is the live session
  sessionRail.innerHTML = "";
  // The LIVE session rides at the top as an amber, non-clickable dash — the
  // trail must be visible from the very first conversation (and the upcoming
  // log system anchors on the rail), not only once past sessions exist.
  const live = sessionSummaries[0];
  if (live) {
    const liveDash = document.createElement("div");
    liveDash.className = "session-dash live";
    liveDash.title = `${live.title} · current session`;
    sessionRail.appendChild(liveDash);
  }
  sessionRail.hidden = !live && viewable.length === 0;
  for (const s of viewable) {
    const dash = document.createElement("div");
    dash.className = `session-dash ${sessionDashClass(s)}`.trim();
    dash.dataset.sessionId = String(s.id);
    const when = (s.updated_at || "").replace("T", " ").slice(0, 16);
    dash.title = `${s.title} · ${when} · ${s.message_count} messages`;
    if (sessionViews.has(String(s.id))) dash.classList.add("active");
    dash.addEventListener("click", () => toggleSessionView(String(s.id)));
    sessionRail.appendChild(dash);
  }
}

/** Toggle one past session's read-only container in the chat. */
async function toggleSessionView(id) {
  const key = String(id);
  const existing = sessionViews.get(key);
  if (existing) {
    existing.remove();
    sessionViews.delete(key);
    sessionRail
      ?.querySelector(`.session-dash[data-session-id="${key}"]`)
      ?.classList.remove("active");
    return;
  }
  let msgs = sessionMsgCache.get(key);
  if (!msgs) {
    try {
      msgs = await invoke("load_session_messages", { sessionId: Number(key) });
      sessionMsgCache.set(key, msgs);
    } catch (e) {
      addSystemMessage(`Error: ${e}`);
      return;
    }
  }
  const summary = sessionSummaries.find((s) => String(s.id) === key);
  const when = (summary?.updated_at || "").replace("T", " ").slice(0, 16);
  const status = summary?.learned_at && summary?.compacted_at
    ? '<span class="pill compacted">learned · compacted</span>'
    : summary?.learned_at
      ? '<span class="pill learned">learned</span>'
      : '<span class="pill wild">wild</span>';
  const view = document.createElement("div");
  view.className = "session-view";
  view.dataset.sessionId = key;
  view.dataset.createdAt = summary?.created_at || "";
  view.innerHTML = `
    <div class="session-view-header">
      <span class="session-view-title">${escapeHtml(summary?.title || `Session #${key}`)}</span>
      <span class="session-view-meta">${escapeHtml(when)} · ${msgs.length} messages · ${escapeHtml(summary?.model || "")}</span>
      ${status}
      <span class="chevron">▾</span>
    </div>
    <div class="session-view-body"></div>`;
  const body = view.querySelector(".session-view-body");
  for (const m of msgs) {
    if (m.role === "user") {
      const d = document.createElement("div");
      d.className = "message user";
      d.innerHTML = `<div class="content">${escapeHtml(m.content)}</div>`;
      body.appendChild(d);
    } else if (m.role === "assistant") {
      const d = document.createElement("div");
      d.className = "message assistant";
      d.innerHTML = `<div class="content">${renderMarkdown(m.content)}</div>`;
      body.appendChild(d);
    } else if (m.role === "tool") {
      const d = document.createElement("div");
      d.className = "sv-tool";
      d.textContent = `⚙ ${m.tool_name || "tool"} — ${(m.content || "").split("\n")[0].slice(0, 90)}`;
      body.appendChild(d);
    }
  }
  view.querySelector(".session-view-header").addEventListener("click", () => {
    view.classList.toggle("collapsed");
    view.querySelector(".chevron").textContent = view.classList.contains("collapsed") ? "▸" : "▾";
  });
  // Chronological placement: oldest on top among the leading session-view
  // block, always above every live element.
  const views = [...chatMessages.querySelectorAll(":scope > .session-view")];
  let target = null;
  for (const v of views) {
    if ((v.dataset.createdAt || "") > view.dataset.createdAt) { target = v; break; }
  }
  const firstLive = [...chatMessages.children].find((c) => !c.classList.contains("session-view"));
  chatMessages.insertBefore(view, target || firstLive || null);
  sessionViews.set(key, view);
  sessionRail
    ?.querySelector(`.session-dash[data-session-id="${key}"]`)
    ?.classList.add("active");
}

// ----- Per-pull download bars -------------------------------------------------
//
// One bar row per concurrent pull, keyed by the pull id the backend stamps on
// every progress event (the GGUF file stem / Ollama model name). Rows live in
// #ml-pull-bars OUTSIDE the model lists, so list re-renders never clobber
// them; a terminal done/error event retires the row.

/** Find (or create) the bar row for a pull id. */
function pullBarItem(id) {
  const escaped = (window.CSS && CSS.escape) ? CSS.escape(id) : id.replace(/"/g, '\\"');
  let row = mlPullBars.querySelector(`.mp-pull-item[data-pull-id="${escaped}"]`);
  if (!row) {
    row = document.createElement("div");
    row.className = "mp-pull-item";
    row.dataset.pullId = id;
    const name = document.createElement("span");
    name.className = "mp-pull-name";
    name.title = id;
    name.textContent = id;
    const prog = document.createElement("div");
    prog.className = "mp-progress";
    prog.innerHTML = '<div class="mp-progress-bar"></div><span class="mp-progress-text"></span>';
    row.append(name, prog);
    mlPullBars.appendChild(row);
    mlPullBars.hidden = false;
  }
  return row;
}

/** Update a pull's bar: `pct` 0-100 (or null → indeterminate) + status text. */
function setPullBar(id, pct, text) {
  const row = pullBarItem(id);
  const bar = row.querySelector(".mp-progress-bar");
  const label = row.querySelector(".mp-progress-text");
  if (bar) bar.style.setProperty("--mp-pct", pct == null ? "100%" : `${pct}%`);
  if (label) label.textContent = text;
}

// ----- Encapsulated auto-save -------------------------------------------------
// After every completed agent task (and finished model pull), seal the state
// back into the exe: a power break or crash can then never lose more than
// the current unfinished task. Best-effort and silent.
let autosaveTimer = null;
function autosaveSeal() {
  if (autosaveTimer) return;
  autosaveTimer = setTimeout(() => {
    autosaveTimer = null;
    invoke("seal_capsule").catch(() => {});
  }, 1200);
}

/** Retire a pull's bar (done or failed): mark it, then remove it shortly.
 * This is ALSO the single retirement point for the nav ring — drop the ETA
 * entry so the aggregate ring vanishes when no pull is active. */
function retirePullBar(id, ok, text) {
  etaDrop(id);
  autosaveSeal();
  const escaped = (window.CSS && CSS.escape) ? CSS.escape(id) : id.replace(/"/g, '\\"');
  const row = mlPullBars.querySelector(`.mp-pull-item[data-pull-id="${escaped}"]`);
  if (!row) return;
  row.classList.add(ok ? "done" : "error");
  const bar = row.querySelector(".mp-progress-bar");
  const label = row.querySelector(".mp-progress-text");
  if (bar) bar.style.setProperty("--mp-pct", "100%");
  if (label && text) label.textContent = text;
  setTimeout(() => {
    row.remove();
    if (!mlPullBars.children.length) mlPullBars.hidden = true;
  }, ok ? 1200 : 3200);
}

/** AmberCore pull progress → bar row. Payload: { id, phase: model|tokenizer|
 *  done|error, completed, total, tag?, error? }. */
function onAmberCorePullProgress(payload) {
  const id = String(payload?.id ?? "pull");
  const phase = payload?.phase;
  if (phase === "done") {
    retirePullBar(id, true, `✓ ${payload?.tag ?? "done"}`);
    return;
  }
  if (phase === "error") {
    retirePullBar(id, false, `✗ failed`);
    return;
  }
  const label = phase === "tokenizer" ? "Tokenizer" : "Model";
  const total = payload?.total ?? null;
  const completed = payload?.completed ?? 0;
  etaTrack(id, completed, total);
  const pct = total ? Math.min(100, Math.round((completed / total) * 100)) : null;
  setPullBar(
    id,
    pct,
    total
      ? `${label} · ${pct}% · ${humanBytes(completed)} / ${humanBytes(total)}`
      : `${label} · ${humanBytes(completed)} downloaded`
  );
}

/** Ollama pull progress → bar row. Payload: { id, line } where line is
 *  `ollama pull`'s NDJSON ({"status","completed","total"}), or { id, phase:
 *  done|error } terminal events. */
function onOllamaPullProgress(payload) {
  const id = String(payload?.id ?? "pull");
  const phase = payload?.phase;
  if (phase === "done") {
    retirePullBar(id, true, "✓ done");
    return;
  }
  if (phase === "error") {
    retirePullBar(id, false, "✗ failed");
    return;
  }
  const line = String(payload?.line ?? "");
  let parsed = null;
  try { parsed = JSON.parse(line); } catch { /* human-readable status line */ }
  if (parsed && typeof parsed.completed === "number" && typeof parsed.total === "number" && parsed.total > 0) {
    etaTrack(id, parsed.completed, parsed.total);
    const pct = Math.min(100, Math.round((parsed.completed / parsed.total) * 100));
    setPullBar(id, pct, `${pct}% · ${humanBytes(parsed.completed)} / ${humanBytes(parsed.total)}`);
  } else {
    const status = (parsed?.status ?? line).slice(0, 60);
    setPullBar(id, null, status);
  }
}

/** Delete an AmberCore model — the bin button next to Run. Removes the GGUF,
 *  its sibling tokenizer, the per-model pull folder, and the manifest entry
 *  (after unloading any replica holding the file), behind a confirmation. */
async function deleteAmberCoreModel(name) {
  if (!window.confirm(`Delete AmberCore model "${name}" and its tokenizer from disk? This cannot be undone.`)) return;
  try {
    await invoke("delete_ambercore_model", { name });
    addSystemMessage(`Deleted AmberCore model: ${name} (model + tokenizer removed)`);
    await renderAmberCore(await invoke("get_active_route"));
  } catch (e) {
    addSystemMessage(`AmberCore delete failed: ${e}`);
  }
}

// ----- Model search modal ----------------------------------------------------

/** Same resident-size estimate the engine's pre-flight check uses:
 *  file × 1.5 + 300 MiB (weights + KV cache + activations + overhead). */
function estimatedResidentMb(sizeGb) {
  return sizeGb * 1024 * 1.5 + 300;
}

/** Open the model search modal and auto-search the most-downloaded GGUFs
 *  (empty query) — the fit coloring shows what the user's GPU can run. */
async function openModelSearch() {
  modelSearchModal.hidden = false;
  setMsSource(msActiveSource);
  await runModelSearch();
}

/** Switch the search source (Hugging Face / Civitai) and re-run. */
function setMsSource(source) {
  msActiveSource = source;
  document.querySelectorAll(".ms-source").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.source === source);
  });
  msQuery.placeholder = source === "civitai"
    ? "Search Civitai (image-gen models — browsable, not AmberCore-runnable)…"
    : "Search GGUF models (e.g. qwen3 8b)…";
}

/** Run a search against the active source, rebuild the filter options, and
 *  render (the backend already drops models that would spill > 1 GB). */
async function runModelSearch() {
  msResults.innerHTML = '<div class="ms-empty">Searching…</div>';
  try {
    const resp = await invoke("search_models", {
      query: msQuery.value.trim(),
      source: msActiveSource,
    });
    msLastResponse = resp;
    buildMsFilterOptions();
    applyMsFilters();
  } catch (e) {
    msLastResponse = null;
    msResults.innerHTML = `<div class="ms-empty">Search failed: ${escapeHtml(String(e))}</div>`;
  }
}

/** (Re)build the quant + params filter options from the current results. */
function buildMsFilterOptions() {
  const fill = (select, values) => {
    const current = select.value;
    select.innerHTML = `<option value="">${select.title.startsWith("Filter by quant") ? "Any quant" : "Any params"}</option>`;
    for (const v of values) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    }
    if ([...select.options].some((o) => o.value === current)) select.value = current;
  };
  const quants = [...new Set((msLastResponse?.results || []).map((m) => m.quant).filter(Boolean))]
    .sort((a, b) => Number(a.match(/(\d+)/)?.[1] ?? 0) - Number(b.match(/(\d+)/)?.[1] ?? 0));
  const params = [...new Set((msLastResponse?.results || []).map((m) => m.params).filter(Boolean))]
    .sort((a, b) => parseFloat(a) - parseFloat(b));
  fill(msFilterQuant, quants);
  fill(msFilterParams, params);
}

/** Re-render the current results through the size/quant/params filters. */
function applyMsFilters() {
  if (!msLastResponse) return;
  const free = msLastResponse.vram_free_mb; // MiB, null when no GPU / unqueryable
  const total = msLastResponse.vram_total_mb; // MiB — the GPU's MAX VRAM
  msVram.textContent = total != null
    ? `GPU: ${Math.round(total / 1024 * 10) / 10} GB VRAM — green cards fit entirely, orange spill ≤ 1 GB.`
    : "No GPU detected — fit coloring unavailable.";
  const maxSize = parseFloat(msFilterSize?.value || "");
  const list = (msLastResponse.results || []).filter((m) => {
    if (maxSize && m.size_gb > maxSize) return false;
    if (msFilterQuant?.value && m.quant !== msFilterQuant.value) return false;
    if (msFilterParams?.value && m.params !== msFilterParams.value) return false;
    return true;
  });
  renderMsCards(list, free);
}

/** Render the result cards: [name + size / params + quant] + Pull button,
 *  glass-green when the model fits the GPU entirely, glass-orange when it
 *  would spill (≤ 1 GB — bigger spills are filtered out server-side). */
function renderMsCards(list, free) {
  msResults.innerHTML = "";
  if (!list.length) {
    msResults.innerHTML = '<div class="ms-empty">No models match — try another search or loosen the filters.</div>';
    return;
  }
  for (const m of list) {
    const known = m.size_gb > 0; // unknown-size cards get no fit tint
    const fits = free != null && known && estimatedResidentMb(m.size_gb) <= free;
    const card = document.createElement("div");
    card.className = "ms-card" + (free == null || !known ? "" : fits ? " fits" : " spills");
    card.title = m.url;
    const params = m.params || "—";
    const quant = m.quant || "—";
    card.innerHTML = `
      <div class="ms-card-main">
        <div class="ms-row ms-row-top">
          <span class="ms-name">${escapeHtml(m.name)}</span>
          <span class="ms-size">${known ? Number(m.size_gb).toFixed(1) + " GB" : "size —"}</span>
        </div>
        <div class="ms-row ms-row-bottom">
          <span class="ms-params">${escapeHtml(params)} params</span>
          <span class="ms-sep">·</span>
          <span class="ms-quant">${escapeHtml(quant)}</span>
          ${m.ambercore_compatible ? "" : '<span class="ms-badge" title="Civitai hosts image-generation models — AmberCore runs text-model GGUFs only">image model</span>'}
        </div>
      </div>`;
    const pullBtn = document.createElement("button");
    pullBtn.className = "ms-pull btn-secondary";
    pullBtn.textContent = "Pull";
    if (!m.ambercore_compatible) {
      pullBtn.disabled = true;
      pullBtn.title = "Image-gen models can't run on AmberCore";
    } else {
      pullBtn.addEventListener("click", async () => {
        // Keep the modal open so several models can be pulled at once — each
        // pull gets its own bar in the Models panel (see the pull-bars
        // helpers); the card's button reflects only its own pull.
        pullBtn.disabled = true;
        pullBtn.textContent = "Pulling…";
        const ok = await pullAmberCoreFromUrl(m.url);
        pullBtn.textContent = ok ? "Pulled ✓" : "Pull";
        if (!ok) pullBtn.disabled = false;
        else setTimeout(() => { pullBtn.textContent = "Pull"; pullBtn.disabled = false; }, 2500);
      });
    }
    card.appendChild(pullBtn);
    msResults.appendChild(card);
  }
}

/** Start AmberCore + switch to a model (the "Run" semantics). */
async function runAmberCore(modelTag) {
  addSystemMessage(`Starting the embedded AmberCore engine and switching to ${modelTag}…`);
  try {
    await invoke("run_ambercore", { modelTag });
    await refreshModelsPanel();
    await syncModelFromRoute();
  } catch (e) {
    addSystemMessage(`AmberCore run failed: ${e}`);
  }
}

/** Persist the AmberCore custom directory when changed. */
async function setAmberCoreDir() {
  const dir = icDir.value.trim();
  try { await invoke("set_ambercore_directory", { dir: dir || null }); } catch (e) {
    addSystemMessage(`Failed to set directory: ${e}`);
  }
}

/** Link a remote AmberCore server (e.g. an AmberCore-Server on a private machine). */
async function connectAmberCoreRemote() {
  const urlEl = $("ic-remote-url");
  const url = (urlEl?.value || "").trim();
  if (!url) return;
  try {
    await invoke("connect_ambercore_remote", { url });
    addSystemMessage(`Linked remote AmberCore server: ${url}`);
    await refreshModelsPanel();
  } catch (e) {
    addSystemMessage(`Remote connect failed: ${e}`);
  }
}

/** Switch AmberCore back to local mode (Phoenix runs `ambercore serve` itself). */
async function useLocalAmberCore() {
  try {
    await invoke("use_local_ambercore");
    addSystemMessage("Switched back to local AmberCore.");
    const urlEl = $("ic-remote-url");
    if (urlEl) urlEl.value = "";
    await refreshModelsPanel();
  } catch (e) {
    addSystemMessage(`Switch failed: ${e}`);
  }
}

/** Reflect the saved remote/local mode + URL in the AmberCore box UI. */
async function refreshAmberCoreRemote() {
  try {
    const st = await invoke("get_ambercore_status");
    const urlEl = $("ic-remote-url");
    const localBtn = $("ic-local-btn");
    if (urlEl && !urlEl.value && st.remote) urlEl.value = st.url;
    if (localBtn) localBtn.hidden = !st.remote;
  } catch { /* pre-unlock — ignore */ }
}

/** Render the Ollama model list (yellow box). */
async function renderOllama(route) {
  olList.innerHTML = '<li class="panel-loading">Loading Ollama models…</li>';
  let models = [];
  try {
    models = await invoke("list_ollama_models");
  } catch (e) {
    olList.innerHTML = `<li class="panel-loading">Ollama not running. Click "Install Ollama" or start it, then pull a model.</li>`;
    return;
  }
  if (models.length === 0) {
    olList.innerHTML = '<li class="panel-loading">No Ollama models. Pull one above (e.g. qwen2.5-coder:7b).</li>';
    return;
  }
  olList.innerHTML = "";
  const active = route.kind === "local" && route.backend === "ollama";
  for (const m of models) {
    const isActive = active && m.name === route.model;
    const li = document.createElement("li");
    li.className = "mp-row";
    li.innerHTML =
      `<span class="mp-name">${escapeHtml(m.name)}</span>` +
      `<span class="mp-sep">|</span>` +
      `<span class="mp-meta">${escapeHtml(m.downloaded_at)}</span>` +
      `<button class="btn-run">${isActive ? "Running" : "Run"}</button>` +
      `<button class="btn-del" title="Delete model">🗑</button>`;
    li.querySelector(".btn-run").addEventListener("click", () => runOllama(m.name));
    li.querySelector(".btn-del").addEventListener("click", () => deleteOllamaModel(m.name));
    if (isActive) li.querySelector(".btn-run").style.borderColor = "var(--phoenix-warm)";
    olList.appendChild(li);
  }
}

/** Delete an Ollama model from its store — the bin button next to Run. */
async function deleteOllamaModel(name) {
  if (!window.confirm(`Delete Ollama model "${name}" from disk? This cannot be undone.`)) return;
  try {
    await invoke("delete_ollama_model", { name });
    addSystemMessage(`Deleted Ollama model: ${name}`);
    await renderOllama(await invoke("get_active_route"));
  } catch (e) {
    addSystemMessage(`Ollama delete failed: ${e}`);
  }
}

/** Pull an Ollama-hosted model via `ollama pull`. Progress streams into the
 *  shared per-pull bars (see onOllamaPullProgress). */
async function pullOllama() {
  const name = olPull.value.trim();
  if (!name) { addSystemMessage("Enter a model name first."); return; }
  olPullBtn.disabled = true;
  try {
    await invoke("pull_ollama_model", { name });
    addSystemMessage(`Pulled Ollama model: ${name}`);
    olPull.value = "";
    await renderOllama(await invoke("get_active_route"));
  } catch (e) {
    addSystemMessage(`Ollama pull failed: ${e}`);
  } finally {
    olPullBtn.disabled = false;
  }
}

/** Auto-install Ollama. */
async function installOllama() {
  addSystemMessage("Installing Ollama…");
  olInstallBtn.disabled = true;
  try {
    const path = await invoke("install_ollama");
    addSystemMessage(`Ollama installed from ${path}. You can now pull models.`);
  } catch (e) {
    addSystemMessage(`Ollama install failed: ${e}`);
  } finally {
    olInstallBtn.disabled = false;
  }
}

/** Start Ollama + switch to a model (the "Run" semantics). */
async function runOllama(model) {
  addSystemMessage(`Starting Ollama and switching to ${model}…`);
  try {
    await invoke("run_ollama", { model });
    await refreshModelsPanel();
    await syncModelFromRoute();
  } catch (e) {
    addSystemMessage(`Ollama run failed: ${e}`);
  }
}

/** Render the registered providers list (red box). */
async function renderProviders(route) {
  prList.innerHTML = '<li class="panel-loading">Loading providers…</li>';
  let providers = [];
  try {
    providers = await invoke("list_providers");
  } catch (e) {
    prList.innerHTML = `<li class="panel-loading">Failed to load: ${escapeHtml(String(e))}</li>`;
    return;
  }
  // Cache provider names so the health-bar label can show the active provider.
  for (const p of providers) providerNameCache[p.id] = p.name;
  if (providers.length === 0) {
    prList.innerHTML = '<li class="panel-loading">No providers registered. Add one above.</li>';
    return;
  }
  prList.innerHTML = "";
  for (const p of providers) {
    // Usage is fetched per-row (best-effort; shows "—" if unavailable).
    let usage = "—";
    try { usage = `${await invoke("provider_usage_last_hour", { providerId: p.id })} tok/h`; } catch { /* ignore */ }
    const isActive = route.kind === "cloud" && route.provider_id === p.id;
    const li = document.createElement("li");
    li.className = "mp-row";
    li.innerHTML =
      `<span class="mp-name">${escapeHtml(p.name)}</span>` +
      `<span class="mp-key" title="Hover to reveal">${escapeHtml(p.api_key_masked)}</span>` +
      `<span class="mp-sep">|</span>` +
      `<span class="mp-usage">${usage}</span>` +
      `<button class="btn-run">${isActive ? "Connected" : "Run"}</button>`;
    li.querySelector(".btn-run").addEventListener("click", () => runProvider(p.id));
    if (isActive) li.querySelector(".btn-run").style.borderColor = "var(--phoenix-warm)";
    prList.appendChild(li);
  }
}

/** Register a new cloud provider. */
async function registerProvider() {
  const name = prName.value.trim();
  const apiKey = prKey.value.trim();
  const baseUrl = prUrl.value.trim();
  if (!name || !apiKey) { addSystemMessage("Provider name and API key are required."); return; }
  try {
    await invoke("create_provider", { name, baseUrl: baseUrl || "https://api.openai.com", apiKey });
    addSystemMessage(`Registered provider: ${name}`);
    prName.value = ""; prKey.value = ""; prUrl.value = "";
    await renderProviders(await invoke("get_active_route"));
  } catch (e) {
    addSystemMessage(`Register failed: ${e}`);
  }
}

/** Switch the active route to a cloud provider (the "Run" semantics). */
async function runProvider(providerId) {
  addSystemMessage("Connecting to cloud provider…");
  try {
    await invoke("run_provider", { providerId });
    await refreshModelsPanel();
    await syncModelFromRoute();
  } catch (e) {
    addSystemMessage(`Provider connection failed: ${e}`);
  }
}

/** Create a new profile via prompt, then activate it. */
async function createNewProfile() {
  const name = window.prompt("Profile name:");
  if (!name || !name.trim()) return;
  try {
    const id = await invoke("create_profile", { name: name.trim() });
    await switchToProfile(id);
    addSystemMessage(`Created profile "${name.trim()}".`);
  } catch (e) {
    addSystemMessage(`Create profile failed: ${e}`);
  }
}

/** Switch the active profile by id and refresh everything profile-scoped. */
async function switchToProfile(id) {
  if (!id) return;
  try {
    const p = await invoke("switch_profile", { id });
    activeProfileId = p.id;
    updateSkillsDot();
    updateToolsDot();
    updateContextDot();
    updateMemoryDot();
    await loadWorkdir(); // workdir is profile-scoped — follow the switch
    addSystemMessage(`Profile switched to "${p.name}".`);
  } catch (e) {
    addSystemMessage(`Profile switch failed: ${e}`);
  }
}

/** Jump straight to the Default profile (the 🏠 button). */
async function goDefaultProfile() {
  try {
    const profiles = await invoke("list_profiles");
    const def = profiles.find((p) => p.is_default) || profiles[0];
    if (!def) return;
    if (activeProfileId === def.id) {
      await loadWorkdir();
      return;
    }
    await switchToProfile(def.id);
  } catch (e) {
    addSystemMessage(`Switch to default profile failed: ${e}`);
  }
}

/** Open the OS native directory picker and apply the choice to the active
 *  profile. Cancelling the dialog is a no-op. */
async function changeWorkdir() {
  const path = await window.__TAURI__.dialog.open({
    directory: true,
    title: "Select a directory for this profile",
  });
  if (!path || typeof path !== "string") return;
  try {
    await invoke("set_workdir", { path });
    await loadWorkdir();
    addSystemMessage(`Working directory set to ${path}.`);
  } catch (e) {
    addSystemMessage(`Workdir change failed: ${e}`);
  }
}

// ----- Skills panel -------------------------------------------------------

/** Open the Skills panel and refresh the skill list. */
async function openSkillsPanel() {
  skillsPanel.hidden = false;
  switchSkillsTab("mine");
  await refreshSkills();
}

/** Refresh the "My Skills" list from the backend. */
async function refreshSkills() {
  skillsList.innerHTML = '<div class="panel-loading">Loading skills…</div>';
  let rows = [];
  try {
    rows = await invoke("list_skills_for_active_profile");
  } catch (e) {
    skillsList.innerHTML = `<div class="panel-loading">Failed to load: ${escapeHtml(String(e))}</div>`;
    return;
  }
  if (rows.length === 0) {
    skillsList.innerHTML = '<div class="panel-loading">No skills yet. Create one or search GitHub.</div>';
    updateSkillsDot();
    return;
  }
  skillsList.innerHTML = "";
  for (const r of rows) {
    skillsList.appendChild(buildSkillRow(r));
  }
  updateSkillsDot();
}

/** Build a single skill row (toggle + name + desc + edit/delete). */
function buildSkillRow(r) {
  const row = document.createElement("div");
  row.className = "skill-row" + (r.enabled ? " enabled" : "");
  const id = r.id;
  const sourceTag = r.source === "github"
    ? '<span class="skill-source-tag">github</span>'
    : "";
  row.innerHTML = `
    <div class="skill-meta">
      <div class="skill-name">${escapeHtml(r.name)} ${sourceTag}</div>
      <div class="skill-desc">${escapeHtml(r.description || "(no description)")}</div>
    </div>
    <div class="skill-actions">
      <label class="toggle" title="Enable for this profile">
        <input type="checkbox" data-toggle ${r.enabled ? "checked" : ""}/>
        <span class="toggle-slider"></span>
      </label>
      <button class="skill-icon-btn" data-edit title="Edit">✎</button>
      <button class="skill-icon-btn" data-delete title="Delete">🗑</button>
    </div>`;
  // Toggle enable/disable.
  row.querySelector("[data-toggle]").addEventListener("change", async (e) => {
    try {
      await invoke("set_skill_enabled", { skillId: id, enabled: e.target.checked });
      row.classList.toggle("enabled", e.target.checked);
      updateSkillsDot();
    } catch (err) {
      addSystemMessage(`Toggle skill failed: ${err}`);
      e.target.checked = !e.target.checked; // revert
    }
  });
  row.querySelector("[data-edit]").addEventListener("click", () => openSkillForm(r));
  row.querySelector("[data-delete]").addEventListener("click", async () => {
    if (!window.confirm(`Delete skill "${r.name}"? This cannot be undone.`)) return;
    try {
      await invoke("delete_skill", { id });
      await refreshSkills();
      addSystemMessage(`Deleted skill "${r.name}".`);
    } catch (err) {
      addSystemMessage(`Delete skill failed: ${err}`);
    }
  });
  return row;
}

/** Switch between the My Skills / Search tabs. */
function switchSkillsTab(tab) {
  for (const btn of skillsPanel.querySelectorAll(".tab-btn")) {
    btn.classList.toggle("active", btn.dataset.skillsTab === tab);
  }
  $("skills-tab-mine").hidden = tab !== "mine";
  $("skills-tab-search").hidden = tab !== "search";
}

/** "+ New skill" button handler. If a skill is currently open in the form (with
 *  a name filled in), save it first — then open a fresh, empty form. This lets
 *  the user chain-create skills without losing the one being edited. */
async function startNewSkill() {
  const formOpen = !skillForm.hidden;
  const hasName = skillFormName.value.trim().length > 0;
  if (formOpen && hasName) {
    await saveSkillForm();
    // Only proceed to a fresh form if the save succeeded (form is now hidden).
    if (!skillForm.hidden) return;
  }
  openSkillForm(null);
}

/** Open the new/edit skill form. Pass a skill object to edit, omit to create. */
function openSkillForm(skill) {
  editingSkillId = skill ? skill.id : null;
  skillFormTitle.textContent = skill ? `Edit: ${skill.name}` : "New skill";
  skillFormName.value = skill ? skill.name : "";
  skillFormDesc.value = skill ? skill.description : "";
  skillFormBody.value = skill ? skill.body : "";
  skillForm.hidden = false;
  skillFormName.focus();
}

/** Save the skill form (create or update). */
async function saveSkillForm() {
  const name = skillFormName.value.trim();
  const description = skillFormDesc.value.trim();
  const body = skillFormBody.value;
  if (!name) {
    addSystemMessage("Skill name is required.");
    return;
  }
  try {
    if (editingSkillId != null) {
      await invoke("update_skill", { id: editingSkillId, name, description, body });
      addSystemMessage(`Updated skill "${name}".`);
    } else {
      await invoke("create_skill", { name, description, body });
      addSystemMessage(`Created skill "${name}".`);
    }
    skillForm.hidden = true;
    await refreshSkills();
  } catch (e) {
    addSystemMessage(`Save skill failed: ${e}`);
  }
}

/** Run a GitHub skill search and render results. */
async function searchGithubSkills() {
  const query = skillSearchInput.value.trim();
  if (!query) return;
  skillSearchResults.innerHTML = '<div class="panel-loading">Searching GitHub…</div>';
  let hits = [];
  try {
    hits = await invoke("search_github_skills", { query });
  } catch (e) {
    skillSearchResults.innerHTML = `<div class="panel-loading">${escapeHtml(String(e))}</div>`;
    return;
  }
  if (hits.length === 0) {
    skillSearchResults.innerHTML = '<div class="panel-loading">No results.</div>';
    return;
  }
  skillSearchResults.innerHTML = "";
  for (const h of hits) {
    const row = document.createElement("div");
    row.className = "skill-search-result";
    row.innerHTML = `
      <span class="search-name" title="${escapeHtml(h.html_url)}">${escapeHtml(h.name)}</span>
      <button class="btn-secondary" data-install>Install</button>`;
    row.querySelector("[data-install]").addEventListener("click", async () => {
      const btn = row.querySelector("[data-install]");
      btn.disabled = true;
      btn.textContent = "Installing…";
      try {
        // Derive a name + empty description; user can edit after install.
        const name = h.path.split("/").pop().replace(/\.md$/i, "");
        await invoke("install_github_skill", { name, description: `from ${h.repo}`, rawUrl: h.raw_url });
        addSystemMessage(`Installed skill "${name}" from GitHub.`);
        btn.textContent = "Installed ✓";
      } catch (e) {
        addSystemMessage(`Install failed: ${e}`);
        btn.disabled = false;
        btn.textContent = "Install";
      }
    });
    skillSearchResults.appendChild(row);
  }
}

/** Update the Skills nav status dot: green if any skill enabled. */
async function updateSkillsDot() {
  const dot = $("dot-skills");
  if (!dot) return;
  try {
    const rows = await invoke("list_skills_for_active_profile");
    const anyEnabled = rows.some((r) => r.enabled);
    dot.className = `status-dot ${anyEnabled ? "ok" : "down"}`;
  } catch {
    dot.className = "status-dot down";
  }
}

// ----- Tools panel -------------------------------------------------------

/** Open the Tools panel and refresh the tool list. */
async function openToolsPanel() {
  toolsPanel.hidden = false;
  switchToolsTab("mine");
  await refreshTools();
}

/** Refresh the "My Tools" list from the backend. */
async function refreshTools() {
  toolsList.innerHTML = '<div class="panel-loading">Loading tools…</div>';
  let rows = [];
  try {
    rows = await invoke("list_tools_for_active_profile");
  } catch (e) {
    toolsList.innerHTML = `<div class="panel-loading">Failed to load: ${escapeHtml(String(e))}</div>`;
    return;
  }
  if (rows.length === 0) {
    toolsList.innerHTML = '<div class="panel-loading">No tools yet. Create one, or search GitHub. Tools run scripts (python/node/sh) with the model\'s args on stdin.</div>';
    updateToolsDot();
    return;
  }
  toolsList.innerHTML = "";
  for (const r of rows) {
    toolsList.appendChild(buildToolRow(r));
  }
  updateToolsDot();
}

/** Build a single tool row (toggle + name + desc + edit/delete). */
function buildToolRow(r) {
  const row = document.createElement("div");
  row.className = "skill-row" + (r.enabled ? " enabled" : "");
  const id = r.id;
  const sourceTag = r.source === "github"
    ? '<span class="skill-source-tag">github</span>'
    : "";
  const kindTag = `<span class="skill-source-tag">${escapeHtml(r.tool_kind || "write")}</span>`;
  row.innerHTML = `
    <div class="skill-meta">
      <div class="skill-name">${escapeHtml(r.name)} ${kindTag} ${sourceTag}</div>
      <div class="skill-desc">${escapeHtml(r.description || "(no description)")} · ${escapeHtml(r.interpreter)}</div>
    </div>
    <div class="skill-actions">
      <label class="toggle" title="Enable for this profile">
        <input type="checkbox" data-toggle ${r.enabled ? "checked" : ""}/>
        <span class="toggle-slider"></span>
      </label>
      <button class="skill-icon-btn" data-edit title="Edit">✎</button>
      <button class="skill-icon-btn" data-delete title="Delete">🗑</button>
    </div>`;
  row.querySelector("[data-toggle]").addEventListener("change", async (e) => {
    try {
      await invoke("set_tool_enabled", { toolId: id, enabled: e.target.checked });
      row.classList.toggle("enabled", e.target.checked);
      updateToolsDot();
    } catch (err) {
      addSystemMessage(`Toggle tool failed: ${err}`);
      e.target.checked = !e.target.checked; // revert
    }
  });
  row.querySelector("[data-edit]").addEventListener("click", () => openToolForm(r));
  row.querySelector("[data-delete]").addEventListener("click", async () => {
    if (!window.confirm(`Delete tool "${r.name}"? This removes it from all profiles.`)) return;
    try {
      await invoke("delete_tool", { id });
      await refreshTools();
      addSystemMessage(`Deleted tool "${r.name}".`);
    } catch (err) {
      addSystemMessage(`Delete tool failed: ${err}`);
    }
  });
  return row;
}

/** Switch between the My Tools / Search tabs. */
function switchToolsTab(tab) {
  for (const btn of toolsPanel.querySelectorAll(".tab-btn")) {
    btn.classList.toggle("active", btn.dataset.toolsTab === tab);
  }
  $("tools-tab-mine").hidden = tab !== "mine";
  $("tools-tab-search").hidden = tab !== "search";
}

/** "+ New tool" button handler. If a tool is currently open in the form (with
 *  a name filled in), save it first — then open a fresh, empty form. Mirrors
 *  the Skills panel's chain-create behavior. */
async function startNewTool() {
  const formOpen = !toolForm.hidden;
  const hasName = toolFormName.value.trim().length > 0;
  if (formOpen && hasName) {
    await saveToolForm();
    if (!toolForm.hidden) return; // save failed — leave the current form open
  }
  openToolForm(null);
}

/** Open the new/edit tool form. Pass a tool object to edit, omit to create. */
function openToolForm(tool) {
  editingToolId = tool ? tool.id : null;
  toolFormTitle.textContent = tool ? `Edit: ${tool.name}` : "New tool";
  toolFormName.value = tool ? tool.name : "";
  toolFormDesc.value = tool ? tool.description : "";
  toolFormInterpreter.value = tool ? tool.interpreter : "sh";
  toolFormKind.value = tool ? tool.tool_kind : "write";
  toolFormSchema.value = tool ? tool.params_schema : '{"type":"object","properties":{},"required":[]}';
  toolFormBody.value = tool ? tool.script_body : "";
  toolForm.hidden = false;
  toolFormName.focus();
}

/** Save the tool form (create or update). */
async function saveToolForm() {
  const name = toolFormName.value.trim();
  const description = toolFormDesc.value.trim();
  const interpreter = toolFormInterpreter.value;
  const tool_kind = toolFormKind.value;
  const params_schema = toolFormSchema.value;
  const script_body = toolFormBody.value;
  if (!name) {
    addSystemMessage("Tool name is required.");
    return;
  }
  // Validate the schema is valid JSON.
  try {
    JSON.parse(params_schema || "{}");
  } catch {
    addSystemMessage("Parameters schema must be valid JSON.");
    return;
  }
  try {
    if (editingToolId != null) {
      await invoke("update_tool", { id: editingToolId, name, description, interpreter, scriptBody: script_body, paramsSchema: params_schema, toolKind: tool_kind });
      addSystemMessage(`Updated tool "${name}".`);
    } else {
      await invoke("create_tool", { name, description, interpreter, scriptBody: script_body, paramsSchema: params_schema, toolKind: tool_kind });
      addSystemMessage(`Created tool "${name}".`);
    }
    toolForm.hidden = true;
    await refreshTools();
  } catch (e) {
    addSystemMessage(`Save tool failed: ${e}`);
  }
}

/** Run a GitHub tool search and render results. Clicking a result loads its
 * body into a new tool form (user fills in metadata). */
async function searchGithubTools() {
  const query = toolSearchInput.value.trim();
  if (!query) return;
  toolSearchResults.innerHTML = '<div class="panel-loading">Searching GitHub…</div>';
  let hits = [];
  try {
    hits = await invoke("search_github_tools", { query });
  } catch (e) {
    toolSearchResults.innerHTML = `<div class="panel-loading">${escapeHtml(String(e))}</div>`;
    return;
  }
  if (hits.length === 0) {
    toolSearchResults.innerHTML = '<div class="panel-loading">No results.</div>';
    return;
  }
  toolSearchResults.innerHTML = "";
  for (const h of hits) {
    const row = document.createElement("div");
    row.className = "skill-search-result";
    row.innerHTML = `
      <span class="search-name" title="${escapeHtml(h.html_url)}">${escapeHtml(h.name)}</span>
      <button class="btn-secondary" data-load>Load body</button>`;
    row.querySelector("[data-load]").addEventListener("click", async () => {
      const btn = row.querySelector("[data-load]");
      btn.disabled = true;
      btn.textContent = "Fetching…";
      try {
        const body = await invoke("prefetch_github_tool", { rawUrl: h.raw_url });
        // Open a new-tool form pre-filled with the body; user completes metadata.
        switchToolsTab("mine");
        openToolForm(null);
        toolFormBody.value = body;
        // Guess interpreter from extension.
        const ext = h.path.split(".").pop().toLowerCase();
        const interp = { py: "python", js: "node", sh: "sh", ps1: "powershell" }[ext] || "sh";
        toolFormInterpreter.value = interp;
        toolFormName.value = h.path.split("/").pop().replace(/\.(py|js|sh|ps1)$/i, "");
        toolFormDesc.value = `from ${h.repo}`;
        addSystemMessage(`Loaded tool body from GitHub. Review and Save.`);
      } catch (e) {
        addSystemMessage(`Fetch failed: ${e}`);
        btn.disabled = false;
        btn.textContent = "Load body";
      }
    });
    toolSearchResults.appendChild(row);
  }
}

/** Update the Tools nav status dot: green if any tool enabled. */
async function updateToolsDot() {
  const dot = $("dot-tools");
  if (!dot) return;
  try {
    const rows = await invoke("list_tools_for_active_profile");
    const anyEnabled = rows.some((r) => r.enabled);
    dot.className = `status-dot ${anyEnabled ? "ok" : "down"}`;
  } catch {
    dot.className = "status-dot down";
  }
}

// ----- Context panel -----------------------------------------------------

/** Open the Context panel and refresh the list. */
async function openContextPanel() {
  contextPanel.hidden = false;
  await refreshContext();
}

/** Refresh the context list from the backend. */
async function refreshContext() {
  contextList.innerHTML = '<div class="panel-loading">Loading context…</div>';
  let rows = [];
  try {
    rows = await invoke("list_context_for_active_profile");
  } catch (e) {
    contextList.innerHTML = `<div class="panel-loading">Failed to load: ${escapeHtml(String(e))}</div>`;
    return;
  }
  if (rows.length === 0) {
    contextList.innerHTML = '<div class="panel-loading">No context files yet. Add facts about your project (e.g. "we use PostgreSQL", "tests run via pnpm test") — the model treats them as ground truth.</div>';
    updateContextDot();
    return;
  }
  contextList.innerHTML = "";
  for (const r of rows) {
    contextList.appendChild(buildContextRow(r));
  }
  updateContextDot();
}

/** Build a single context row (toggle + name + desc + edit/delete). */
function buildContextRow(r) {
  const row = document.createElement("div");
  row.className = "skill-row" + (r.enabled ? " enabled" : "");
  const id = r.id;
  row.innerHTML = `
    <div class="skill-meta">
      <div class="skill-name">${escapeHtml(r.name)}</div>
      <div class="skill-desc">${escapeHtml(r.description || "(no description)")}</div>
    </div>
    <div class="skill-actions">
      <label class="toggle" title="Enable for this profile">
        <input type="checkbox" data-toggle ${r.enabled ? "checked" : ""}/>
        <span class="toggle-slider"></span>
      </label>
      <button class="skill-icon-btn" data-edit title="Edit">✎</button>
      <button class="skill-icon-btn" data-delete title="Delete">🗑</button>
    </div>`;
  row.querySelector("[data-toggle]").addEventListener("change", async (e) => {
    try {
      await invoke("set_context_enabled", { contextId: id, enabled: e.target.checked });
      row.classList.toggle("enabled", e.target.checked);
      updateContextDot();
    } catch (err) {
      addSystemMessage(`Toggle context failed: ${err}`);
      e.target.checked = !e.target.checked; // revert
    }
  });
  row.querySelector("[data-edit]").addEventListener("click", () => openContextForm(r));
  row.querySelector("[data-delete]").addEventListener("click", async () => {
    if (!window.confirm(`Delete context "${r.name}"? This removes it from all profiles.`)) return;
    try {
      await invoke("delete_context", { id });
      await refreshContext();
      addSystemMessage(`Deleted context "${r.name}".`);
    } catch (err) {
      addSystemMessage(`Delete context failed: ${err}`);
    }
  });
  return row;
}

/** Open the new/edit context form. Pass a context object to edit, omit to create. */
function openContextForm(context) {
  editingContextId = context ? context.id : null;
  contextFormTitle.textContent = context ? `Edit: ${context.name}` : "New context";
  contextFormName.value = context ? context.name : "";
  contextFormDesc.value = context ? context.description : "";
  contextFormBody.value = context ? context.body : "";
  contextForm.hidden = false;
  contextFormName.focus();
}

/** Save the context form (create or update). */
async function saveContextForm() {
  const name = contextFormName.value.trim();
  const description = contextFormDesc.value.trim();
  const body = contextFormBody.value;
  if (!name) {
    addSystemMessage("Context name is required.");
    return;
  }
  try {
    if (editingContextId != null) {
      await invoke("update_context", { id: editingContextId, name, description, body });
      addSystemMessage(`Updated context "${name}".`);
    } else {
      await invoke("create_context", { name, description, body });
      addSystemMessage(`Created context "${name}".`);
    }
    contextForm.hidden = true;
    await refreshContext();
  } catch (e) {
    addSystemMessage(`Save context failed: ${e}`);
  }
}

/** Update the Context nav status dot: green if any context enabled. */
async function updateContextDot() {
  const dot = $("dot-context");
  if (!dot) return;
  try {
    const rows = await invoke("list_context_for_active_profile");
    const anyEnabled = rows.some((r) => r.enabled);
    dot.className = `status-dot ${anyEnabled ? "ok" : "down"}`;
  } catch {
    dot.className = "status-dot down";
  }
}

// ---- Memory panel (Panel 5: MCP connections) ---------------------------

/** Open the Memory panel and refresh the list. */
async function openMemoryPanel() {
  memoryPanel.hidden = false;
  await refreshMemory();
}

/** Refresh the connection list from the backend. */
async function refreshMemory() {
  memoryList.innerHTML = '<div class="panel-loading">Loading connections…</div>';
  let rows = [];
  try {
    rows = await invoke("list_memory_for_active_profile");
  } catch (e) {
    memoryList.innerHTML = `<div class="panel-loading">Failed to load: ${escapeHtml(String(e))}</div>`;
    return;
  }
  if (rows.length === 0) {
    memoryList.innerHTML = '<div class="panel-loading">No MCP connections yet. Add an MCP server (stdio process or HTTP endpoint) to expose its tools to the agent.</div>';
    updateMemoryDot();
    return;
  }
  memoryList.innerHTML = "";
  for (const r of rows) {
    memoryList.appendChild(buildMemoryRow(r));
  }
  updateMemoryDot();
}

/** Build a single memory row (toggle + name + transport chip + edit/delete). */
function buildMemoryRow(r) {
  const row = document.createElement("div");
  row.className = "skill-row" + (r.enabled ? " enabled" : "");
  const id = r.id;
  const transport = r.transport || "stdio";
  row.innerHTML = `
    <div class="skill-meta">
      <div class="skill-name">${escapeHtml(r.name)} <span class="skill-source-tag">${escapeHtml(transport)}</span></div>
      <div class="skill-desc">${escapeHtml(r.description || r.command || "(no description)")}</div>
    </div>
    <div class="skill-actions">
      <label class="toggle" title="Enable for this profile">
        <input type="checkbox" data-toggle ${r.enabled ? "checked" : ""}/>
        <span class="toggle-slider"></span>
      </label>
      <button class="skill-icon-btn" data-edit title="Edit">✎</button>
      <button class="skill-icon-btn" data-delete title="Delete">🗑</button>
    </div>`;
  row.querySelector("[data-toggle]").addEventListener("change", async (e) => {
    try {
      await invoke("set_memory_enabled", { memoryId: id, enabled: e.target.checked });
      row.classList.toggle("enabled", e.target.checked);
      updateMemoryDot();
    } catch (err) {
      addSystemMessage(`Toggle connection failed: ${err}`);
      e.target.checked = !e.target.checked; // revert
    }
  });
  row.querySelector("[data-edit]").addEventListener("click", () => openMemoryForm(r));
  row.querySelector("[data-delete]").addEventListener("click", async () => {
    if (!window.confirm(`Delete connection "${r.name}"? This removes it from all profiles.`)) return;
    try {
      await invoke("delete_memory", { id });
      await refreshMemory();
      addSystemMessage(`Deleted connection "${r.name}".`);
    } catch (err) {
      addSystemMessage(`Delete connection failed: ${err}`);
    }
  });
  return row;
}

/** Open the new/edit connection form. Pass a row to edit, omit to create. */
function openMemoryForm(r) {
  editingMemoryId = r ? r.id : null;
  memoryFormTitle.textContent = r ? `Edit: ${r.name}` : "New connection";
  memoryFormName.value = r ? r.name : "";
  memoryFormDesc.value = r ? r.description : "";
  memoryFormTransport.value = r ? (r.transport || "stdio") : "stdio";
  memoryFormCommand.value = r ? r.command : "";
  memoryFormArgs.value = r ? r.args_json : "[]";
  memoryForm.hidden = false;
  memoryFormName.focus();
}

/** Probe the unsaved connection: connect + list tools, without saving. */
async function testMemoryForm() {
  const transport = memoryFormTransport.value;
  const command = memoryFormCommand.value.trim();
  const argsJson = memoryFormArgs.value.trim() || "[]";
  if (!command) {
    addSystemMessage("Enter a command/URL before testing.");
    return;
  }
  memoryFormTest.disabled = true;
  memoryFormTest.textContent = "Testing…";
  try {
    const result = await invoke("test_memory_connection", { transport, command, argsJson });
    if (result.ok) {
      addSystemMessage(`Connection OK — ${result.tool_count} tool${result.tool_count === 1 ? "" : "s"} available.`);
    } else {
      addSystemMessage(`Connection failed: ${result.error || "unknown error"}`);
    }
  } catch (e) {
    addSystemMessage(`Test connection failed: ${e}`);
  } finally {
    memoryFormTest.disabled = false;
    memoryFormTest.textContent = "Test";
  }
}

/** Save the connection form (create or update). */
async function saveMemoryForm() {
  const name = memoryFormName.value.trim();
  const description = memoryFormDesc.value.trim();
  const transport = memoryFormTransport.value;
  const command = memoryFormCommand.value.trim();
  const argsJson = memoryFormArgs.value.trim() || "[]";
  if (!name) {
    addSystemMessage("Connection name is required.");
    return;
  }
  // Validate args is parseable JSON if non-empty.
  try {
    JSON.parse(argsJson);
  } catch {
    addSystemMessage("Args must be valid JSON (or empty).");
    return;
  }
  try {
    if (editingMemoryId != null) {
      await invoke("update_memory", { id: editingMemoryId, name, description, transport, command, argsJson });
      addSystemMessage(`Updated connection "${name}".`);
    } else {
      await invoke("create_memory", { name, description, transport, command, argsJson });
      addSystemMessage(`Created connection "${name}".`);
    }
    memoryForm.hidden = true;
    await refreshMemory();
  } catch (e) {
    addSystemMessage(`Save connection failed: ${e}`);
  }
}

/** Update the Memory nav status dot: green if any connection enabled. */
async function updateMemoryDot() {
  const dot = $("dot-memory");
  if (!dot) return;
  try {
    const rows = await invoke("list_memory_for_active_profile");
    const anyEnabled = rows.some((r) => r.enabled);
    dot.className = `status-dot ${anyEnabled ? "ok" : "down"}`;
  } catch {
    dot.className = "status-dot down";
  }
}

/** Restore the saved sidebar width (persisted in localStorage). */
function restoreSidebarWidth() {
  const saved = localStorage.getItem("phoenix.sidebarWidth");
  if (saved) sidebar.style.width = saved;
}

/** Clamp a pixel width to the sidebar's min/max defined in CSS. */
function clampSidebarWidth(px) {
  const min = 180;
  const max = Math.round(window.innerWidth * 0.45);
  return Math.max(min, Math.min(max, px));
}

/** Restore the saved console height (persisted in localStorage). */
function restoreConsoleHeight() {
  const saved = localStorage.getItem("phoenix.consoleHeight");
  if (saved) consoleSection.style.height = saved;
}

/** Clamp the console height: at least the input row + ~2 lines, never past
 *  the sidebar's other content (nav + workdir + the profile row itself). */
function clampConsoleHeight(px) {
  const min = 70;
  const max = Math.max(min, sidebar.clientHeight - 240);
  return Math.max(min, Math.min(max, px));
}

// Console height drag — the profile row rides on top of the handle (see
// #sidebar-bottom in index.html) and moves with it as the console resizes.
(() => {
  let dragging = false;
  consoleResizer?.addEventListener("mousedown", (e) => {
    dragging = true;
    consoleResizer.classList.add("dragging");
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    // The console spans from the drag point down to the sidebar's bottom.
    const rect = sidebar.getBoundingClientRect();
    const px = clampConsoleHeight(rect.bottom - e.clientY);
    consoleSection.style.height = `${px}px`;
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    consoleResizer.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    localStorage.setItem("phoenix.consoleHeight", consoleSection.style.height);
  });
})();

// ----- Event bindings -------------------------------------------------------
unlockBtn.addEventListener("click", doUnlock);
passphraseInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doUnlock();
});

/** Recover a forgotten launch password via 2FA. Prompts for a TOTP code and a
 *  new launch password; on success the backend boots the runtime and returns
 *  the same result as unlock, so we transition to the chat screen. */
async function recoverLaunch() {
  const totpCode = window.prompt("Enter your current 6-digit 2FA code:");
  if (!totpCode) return;
  const newPass = window.prompt("Set a NEW launch password (min 8 chars):");
  if (!newPass) return;
  if (newPass.length < 8) { unlockError.textContent = "New launch password must be at least 8 characters."; return; }
  unlockBtn.disabled = true;
  unlockBtn.textContent = "Recovering…";
  unlockError.textContent = "";
  try {
    const result = await invoke("recover_launch_via_totp", { totpCode, newLaunchPassword: newPass });
    currentModel = result.model;
    modelSelect.value = result.model;
    unlockScreen.classList.remove("active");
    chatScreen.classList.add("active");
    await populateModels();
    await loadSidebar(result);
    await loadTodo();
    addSystemMessage(`Access recovered. Set a new launch password. Working in: ${result.project_path}`);
    messageInput.focus();
  } catch (e) {
    unlockError.textContent = String(e);
    unlockBtn.disabled = false;
    unlockBtn.textContent = "Unlock";
  }
}
recoverBtn?.addEventListener("click", recoverLaunch);

setupBtn.addEventListener("click", doSetup);
setupConfirm.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSetup();
});

sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ----- Main menu: launch password + DB password + TOTP 2FA --------------

/** Open the main-menu window and refresh the 2FA view. */
async function openConfigModal() {
  switchConfigTab("security");
  await refreshTotpView();
  configModal.hidden = false;
}

/** Switch between Security / Telemetry / Logs / About tabs. */
function switchConfigTab(tab) {
  for (const btn of configModal.querySelectorAll(".tab-btn")) {
    if (btn.disabled) continue;
    btn.classList.toggle("active", btn.dataset.configTab === tab);
  }
  $("config-tab-security").hidden = tab !== "security";
  $("config-tab-telemetry").hidden = tab !== "telemetry";
  $("config-tab-logs").hidden = tab !== "logs";
  $("config-tab-about").hidden = tab !== "about";
  if (tab === "telemetry") refreshTelemetryTab();
  if (tab === "logs") refreshLogsTab();
}

// ----- Main menu: Logs tab (session-log maintenance) -----------------------

/** Load retention settings + folder stats into the Logs tab. */
async function refreshLogsTab() {
  try {
    const s = await invoke("logs_stats");
    $("logs-sessions").textContent = s.sessions;
    $("logs-size").textContent = s.size_text;
    $("logs-dir").textContent = s.logs_dir;
    $("logs-dir").title = s.logs_dir;
  } catch { /* keep placeholders */ }
  try {
    const cfg = await invoke("logs_settings_get");
    $("logs-auto-clean").checked = !!cfg.auto_clean;
    $("logs-keep-days").value = String(cfg.keep_days || 0);
  } catch { /* defaults */ }
}

/** Persist the maintenance toggles (config.toml — sealed with the capsule). */
async function saveLogsSettings() {
  try {
    await invoke("logs_settings_set", {
      autoClean: $("logs-auto-clean").checked,
      keepDays: Number($("logs-keep-days").value) || 0,
    });
    addSystemMessage("Logs: maintenance settings saved.");
  } catch (e) { addSystemMessage(`Error: ${e}`); }
}

/** Two-click confirmation for destructive buttons (the webview has no native
 *  confirm): first click arms the button for 3.5 s, second click fires. */
function armConfirm(btn, action) {
  if (btn.dataset.armed === "1") {
    btn.dataset.armed = "";
    btn.textContent = btn.dataset.label;
    action();
    return;
  }
  btn.dataset.label = btn.textContent;
  btn.dataset.armed = "1";
  btn.textContent = "Really? Click again";
  setTimeout(() => {
    if (btn.dataset.armed === "1") {
      btn.dataset.armed = "";
      btn.textContent = btn.dataset.label;
    }
  }, 3500);
}

/** Delete run files per scope, then refresh the readout. */
async function logsDelete(scope) {
  try {
    const removed = await invoke("logs_delete", { scope, files: null });
    $("logs-cleanup-status").textContent = "";
    addSystemMessage(`Logs: removed ${removed} session file${removed === 1 ? "" : "s"}.`);
    refreshLogsTab();
  } catch (e) {
    $("logs-cleanup-status").textContent = String(e);
  }
}

/** One-shot notices from the session-log system (first creation + size nudge). */
async function logsNotices() {
  try {
    const s = await invoke("logs_stats");
    if (s.first_creation) {
      addSystemMessage(
        `Logs: session logs now live in ${s.logs_dir} — open LogsExplorer.html any time ` +
        "(strict metadata only, never your conversations)."
      );
    }
    if (s.nudge && !sessionStorage.getItem("pa-logs-nudge")) {
      sessionStorage.setItem("pa-logs-nudge", "1");
      addSystemMessage("Logs: the folder passed 100 MB — clean it from Main Menu → Logs.");
    }
  } catch { /* pre-init — ignore */ }
}

function wireLogsTab() {
  $("logs-auto-clean")?.addEventListener("change", saveLogsSettings);
  $("logs-keep-days")?.addEventListener("change", saveLogsSettings);
  $("logs-reveal-btn")?.addEventListener("click", () => {
    invoke("logs_reveal_folder").catch((e) => addSystemMessage(`Error: ${e}`));
  });
  $("logs-explorer-btn")?.addEventListener("click", () => {
    invoke("logs_open_explorer").catch((e) => addSystemMessage(`Error: ${e}`));
  });
  $("logs-del-clean-btn")?.addEventListener("click", function () { armConfirm(this, () => logsDelete("clean")); });
  $("logs-del-all-btn")?.addEventListener("click", function () { armConfirm(this, () => logsDelete("all")); });
}
wireLogsTab();

/**
 * Populate the Telemetry tab's environment baseline from the launch hardware
 * check-up (CPU/cores/RAM/OS snapshot + active compute backend + live GPU
 * reading when one is in use).
 */
async function refreshTelemetryTab() {
  const $set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  try {
    const hw = await invoke("get_hardware_status");
    const gpu = hw.gpu;
    $set("tele-hardware", gpu?.name || "CPU only (no GPU backend)");
    $set("tele-vram", gpu?.vram_total_mb
      ? `${((gpu.vram_used_mb ?? 0) / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GB`
      : "—");
    $set("tele-cpu", hw.cpu
      ? `${String(hw.cpu).trim()} · ${hw.cpu_cores ?? "?"} cores`
      : "—");
    $set("tele-ram", hw.ram_total_mb ? `${(hw.ram_total_mb / 1024).toFixed(1)} GB` : "—");
    $set("tele-backend", hw.backend === "cuda" ? "CUDA (GPU)" : "CPU");
    $set("tele-quant", currentModel || "—");
  } catch (e) {
    $set("tele-backend", "unavailable");
  }
}

/** Refresh the 2FA card: show enabled vs disabled view. */
async function refreshTotpView() {
  let enabled = false;
  try {
    enabled = await invoke("has_totp");
  } catch (e) {
    console.warn("has_totp failed:", e);
  }
  totpEnabledView.hidden = !enabled;
  totpDisabledView.hidden = enabled;
  totpSetupView.hidden = true;
  pendingTotp = null;
}

/** Handle "Change launch password" (Card 1) form submit. Re-wraps the DB key;
 *  does NOT rekey the DB, so it's instant and risk-free. */
async function changeLaunchPassword(e) {
  e.preventDefault();
  lpStatus.textContent = "";
  const current = lpOld.value;
  const next = lpNew.value;
  const confirm = lpConfirm.value;
  if (!current || !next) return;

  const submitBtn = $("lp-submit");
  submitBtn.disabled = true;
  submitBtn.textContent = "Changing…";
  try {
    await invoke("set_launch_password", {
      currentPassword: current,
      newPassword: next,
      confirm,
    });
    lpStatus.style.color = "var(--health-green)";
    lpStatus.textContent = "✅ Launch password changed.";
    lpOld.value = "";
    lpNew.value = "";
    lpConfirm.value = "";
    addSystemMessage("Launch password changed.");
  } catch (err) {
    lpStatus.style.color = "var(--health-red)";
    lpStatus.textContent = String(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Change launch password";
  }
}

/** Handle "Change database password" (Card 2) form submit. Re-encrypts the DB
 *  and re-wraps the key under the launch password. Requires the runtime to
 *  reboot, so the UI may pause briefly. */
async function changePassphrase(e) {
  e.preventDefault();
  cpStatus.textContent = "";
  const currentDb = cpOld.value;
  const newDb = cpNew.value;
  const confirm = cpConfirm.value;
  const launch = cpLaunch.value;
  if (!currentDb || !newDb || !launch) return;

  const submitBtn = $("cp-submit");
  submitBtn.disabled = true;
  submitBtn.textContent = "Re-encrypting…";
  try {
    await invoke("change_passphrase", {
      currentDbPassword: currentDb,
      newDbPassword: newDb,
      confirm,
      launchPassword: launch,
    });
    cpStatus.style.color = "var(--health-green)";
    cpStatus.textContent = "✅ Database password changed. Key re-wrapped.";
    cpOld.value = "";
    cpNew.value = "";
    cpConfirm.value = "";
    cpLaunch.value = "";
    addSystemMessage("Database password changed and DB re-encrypted.");
  } catch (err) {
    cpStatus.style.color = "var(--health-red)";
    cpStatus.textContent = String(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Change database password";
  }
}

/** Begin 2FA setup: ask the backend for a secret + otpauth URL, render a QR. */
async function enableTotp() {
  const account = (totpAccount.value || "").trim() || "phoenix-agent";
  totpSetupStatus.textContent = "";
  try {
    const setup = await invoke("setup_totp", { account });
    pendingTotp = setup;
    totpSecretDisplay.textContent = setup.secret_b32;
    // Render a QR from the otpauth URL via the QR Server API (offline-friendly:
    // it just encodes the string into an <img>; the secret is in the URL).
    // For a fully-local build you'd swap in a tiny JS QR lib; this keeps deps at zero.
    const url = encodeURIComponent(setup.otpauth_url);
    totpQr.innerHTML = `<img src="https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${url}" alt="TOTP QR" />`;
    totpDisabledView.hidden = true;
    totpSetupView.hidden = false;
    totpConfirmCode.focus();
  } catch (err) {
    totpSetupStatus.style.color = "var(--health-red)";
    totpSetupStatus.textContent = String(err);
  }
}

/** Confirm 2FA setup with a live code, persisting it as active. */
async function confirmTotp() {
  const code = (totpConfirmCode.value || "").trim();
  if (code.length !== 6) {
    totpSetupStatus.style.color = "var(--health-red)";
    totpSetupStatus.textContent = "Enter the 6-digit code from your app.";
    return;
  }
  if (!pendingTotp) {
    totpSetupStatus.textContent = "Start setup first.";
    return;
  }
  try {
    await invoke("confirm_totp", { code });
    pendingTotp = null;
    await refreshTotpView();
    addSystemMessage("Two-factor authentication enabled. A code is now required at unlock.");
  } catch (err) {
    totpSetupStatus.style.color = "var(--health-red)";
    totpSetupStatus.textContent = String(err);
  }
}

/** Cancel an in-progress 2FA setup (discards the pending secret). */
function cancelTotpSetup() {
  pendingTotp = null;
  totpConfirmCode.value = "";
  totpSetupStatus.textContent = "";
  totpSetupView.hidden = true;
  totpDisabledView.hidden = false;
}

/** Disable 2FA (requires being unlocked). */
async function disableTotp() {
  if (!window.confirm("Disable two-factor authentication? You'll only need your passphrase to unlock.")) return;
  try {
    await invoke("disable_totp");
    await refreshTotpView();
    addSystemMessage("Two-factor authentication disabled.");
  } catch (err) {
    addSystemMessage(`Disable 2FA failed: ${err}`);
  }
}

// ----- Sidebar bindings ----------------------------------------------------

// Models nav item → open the models panel.
modelsNavItem?.addEventListener("click", () => {
  if (!modelsNavItem.classList.contains("disabled")) openModelsPanel();
});
modelsCloseBtn?.addEventListener("click", () => { modelsPanel.hidden = true; });
// Click outside the panel (on the overlay backdrop) closes it.
modelsPanel?.addEventListener("click", (e) => {
  if (e.target === modelsPanel) modelsPanel.hidden = true;
});
// Models panel v0.5 — runner tabs + shared model list + Provider API wiring.
// Runner tabs: switching swaps the settings pane, the list title, the acquire
//  row (Find models / Ollama pull) and the visible model list.
runnerBox?.querySelectorAll(".runner-tab").forEach((tab) => {
  tab.addEventListener("click", () => setRunnerTab(tab.dataset.runner));
});
// Restore the last-selected runner (defaults to AmberCore).
setRunnerTab(localStorage.getItem("phoenix.runnerTab") || "ambercore");
// Model search modal: open from the list container's Find-models row, search,
// switch source, and re-slice the results through the size/quant/params filters.
icSearchBtn?.addEventListener("click", openModelSearch);
msSearchBtn?.addEventListener("click", runModelSearch);
msQuery?.addEventListener("keydown", (e) => { if (e.key === "Enter") runModelSearch(); });
msCloseBtn?.addEventListener("click", () => { modelSearchModal.hidden = true; });
[msFilterSize, msFilterQuant, msFilterParams].forEach((sel) => {
  sel?.addEventListener("change", applyMsFilters);
});
document.querySelectorAll(".ms-source").forEach((btn) => {
  btn.addEventListener("click", () => { setMsSource(btn.dataset.source); runModelSearch(); });
});
icDirClear?.addEventListener("click", async () => {
  icDir.value = "";
  await setAmberCoreDir();
});
icDir?.addEventListener("change", setAmberCoreDir);
icDir?.addEventListener("keydown", (e) => { if (e.key === "Enter") setAmberCoreDir(); });
$("ic-remote-btn")?.addEventListener("click", connectAmberCoreRemote);
$("ic-remote-url")?.addEventListener("keydown", (e) => { if (e.key === "Enter") connectAmberCoreRemote(); });
$("ic-local-btn")?.addEventListener("click", useLocalAmberCore);
olPullBtn?.addEventListener("click", pullOllama);
olPull?.addEventListener("keydown", (e) => { if (e.key === "Enter") pullOllama(); });
olInstallBtn?.addEventListener("click", installOllama);
prRegisterBtn?.addEventListener("click", registerProvider);
// Pull-progress events streamed from the backend — one bar per concurrent
// pull (AmberCore and Ollama alike), keyed by the pull id.
listen("ambercore-pull-progress", (e) => onAmberCorePullProgress(e.payload));
listen("ollama-pull-progress", (e) => onOllamaPullProgress(e.payload));

// Profile home (Default) + new-profile button.
profileNewBtn?.addEventListener("click", createNewProfile);
profileHomeBtn?.addEventListener("click", goDefaultProfile);

// Workdir: the whole row opens the native directory picker.
workdirRow?.addEventListener("click", changeWorkdir);

// Sidebar console: Enter runs, ↑/↓ walks history.
consoleInput?.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    runConsoleCommand();
  } else if (ev.key === "ArrowUp") {
    if (consoleHistoryIdx > 0) {
      consoleHistoryIdx -= 1;
      consoleInput.value = consoleHistory[consoleHistoryIdx];
      ev.preventDefault();
    }
  } else if (ev.key === "ArrowDown") {
    if (consoleHistoryIdx < consoleHistory.length - 1) {
      consoleHistoryIdx += 1;
      consoleInput.value = consoleHistory[consoleHistoryIdx];
    } else {
      consoleHistoryIdx = consoleHistory.length;
      consoleInput.value = "";
    }
  }
});

// Skills panel: nav open, close, tabs, new/edit form, search.
skillsNavItem?.addEventListener("click", () => {
  if (!skillsNavItem.classList.contains("disabled")) openSkillsPanel();
});
skillsCloseBtn?.addEventListener("click", () => { skillsPanel.hidden = true; });
skillsPanel?.addEventListener("click", (e) => {
  if (e.target === skillsPanel) skillsPanel.hidden = true;
});
for (const btn of skillsPanel.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => switchSkillsTab(btn.dataset.skillsTab));
}
skillNewBtn?.addEventListener("click", startNewSkill);
skillFormCancel?.addEventListener("click", () => { skillForm.hidden = true; });
skillFormSave?.addEventListener("click", saveSkillForm);
// Click the backdrop (outside the card) closes the skill form modal.
skillForm?.addEventListener("click", (e) => { if (e.target === skillForm) skillForm.hidden = true; });
skillSearchBtn?.addEventListener("click", searchGithubSkills);
skillSearchInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") searchGithubSkills();
});

// Tools panel: nav open, close, tabs, new/edit form, search.
toolsNavItem?.addEventListener("click", () => {
  if (!toolsNavItem.classList.contains("disabled")) openToolsPanel();
});
toolsCloseBtn?.addEventListener("click", () => { toolsPanel.hidden = true; });
toolsPanel?.addEventListener("click", (e) => {
  if (e.target === toolsPanel) toolsPanel.hidden = true;
});
for (const btn of toolsPanel.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => switchToolsTab(btn.dataset.toolsTab));
}
toolNewBtn?.addEventListener("click", startNewTool);
toolFormCancel?.addEventListener("click", () => { toolForm.hidden = true; });
toolFormSave?.addEventListener("click", saveToolForm);
// Click the backdrop (outside the card) closes the tool form modal.
toolForm?.addEventListener("click", (e) => { if (e.target === toolForm) toolForm.hidden = true; });
toolSearchBtn?.addEventListener("click", searchGithubTools);
toolSearchInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") searchGithubTools();
});

// Context panel: nav open, close, new/edit form.
contextNavItem?.addEventListener("click", () => {
  if (!contextNavItem.classList.contains("disabled")) openContextPanel();
});
contextCloseBtn?.addEventListener("click", () => { contextPanel.hidden = true; });
contextPanel?.addEventListener("click", (e) => {
  if (e.target === contextPanel) contextPanel.hidden = true;
});
contextNewBtn?.addEventListener("click", () => openContextForm(null));
contextFormCancel?.addEventListener("click", () => { contextForm.hidden = true; });
contextFormSave?.addEventListener("click", saveContextForm);
// Click the backdrop (outside the card) closes the context form modal.
contextForm?.addEventListener("click", (e) => { if (e.target === contextForm) contextForm.hidden = true; });

// Memory panel: nav open, close, new/edit form, test connection.
memoryNavItem?.addEventListener("click", () => {
  if (!memoryNavItem.classList.contains("disabled")) openMemoryPanel();
});

// ----- Sub-Agents panel (Panel 6) -----
async function openSubAgentsPanel() {
  subagentsPanel.hidden = false;
  await refreshSubAgents();
}

async function refreshSubAgents() {
  subagentsList.innerHTML = '<div class="panel-loading">Loading sub-agents…</div>';
  let rows = [];
  try {
    rows = await invoke("list_sub_agents");
  } catch (e) {
    subagentsList.innerHTML = `<div class="panel-loading">Failed to load: ${escapeHtml(String(e))}</div>`;
    return;
  }
  if (rows.length === 0) {
    subagentsList.innerHTML = '<div class="panel-loading">No sub-agents yet. Create one (e.g. a Mathematician).</div>';
    return;
  }
  subagentsList.innerHTML = "";
  for (const r of rows) subagentsList.appendChild(buildSubAgentRow(r));
}

function buildSubAgentRow(r) {
  const row = document.createElement("div");
  row.className = "skill-row enabled";
  const modelTag = r.model
    ? `<span class="skill-source-tag">${escapeHtml(r.model)}</span>`
    : "";
  row.innerHTML = `
    <div class="skill-meta">
      <div class="skill-name">${escapeHtml(r.name)} ${modelTag}</div>
      <div class="skill-desc">${escapeHtml(r.description || "(no description)")}</div>
    </div>
    <div class="skill-actions">
      <button class="skill-icon-btn" data-edit title="Edit">✎</button>
      <button class="skill-icon-btn" data-delete title="Delete">🗑</button>
    </div>`;
  row.querySelector("[data-edit]").addEventListener("click", () => openSubAgentForm(r));
  row.querySelector("[data-delete]").addEventListener("click", async () => {
    if (!window.confirm(`Delete sub-agent "${r.name}"?`)) return;
    try {
      await invoke("delete_sub_agent", { id: r.id });
      await refreshSubAgents();
      addSystemMessage(`Deleted sub-agent "${r.name}".`);
    } catch (err) {
      addSystemMessage(`Delete failed: ${err}`);
    }
  });
  return row;
}

function startNewSubAgent() {
  openSubAgentForm(null);
}

/** Fill the sub-agent form's model selector with every model the active
 *  backend recognizes (same source as the chat picker). `selected` is the
 *  sub-agent's saved tag; a tag no longer listed (e.g. pulled from another
 *  backend) is kept as an extra option so editing never silently loses it. */
async function populateSubAgentModelSelect(selected) {
  let models = [];
  try {
    models = await invoke("list_models");
  } catch (e) {
    console.warn("Sub-agent model list failed:", e);
  }
  subagentFormModel.innerHTML = "";
  const active = document.createElement("option");
  active.value = "";
  active.textContent = "(active model) — follows the chat's selection";
  subagentFormModel.appendChild(active);
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    subagentFormModel.appendChild(opt);
  }
  if (selected && !models.includes(selected)) {
    const stale = document.createElement("option");
    stale.value = selected;
    stale.textContent = `${selected} (not currently listed)`;
    subagentFormModel.appendChild(stale);
  }
  subagentFormModel.value = selected || "";
}

async function openSubAgentForm(sa) {
  editingSubAgentId = sa ? sa.id : null;
  subagentFormTitle.textContent = sa ? `Edit: ${sa.name}` : "New sub-agent";
  subagentFormName.value = sa ? sa.name : "";
  subagentFormDesc.value = sa ? sa.description : "";
  await populateSubAgentModelSelect(sa ? sa.model : "");
  subagentFormPersona.value = sa ? sa.persona : "";
  subagentForm.hidden = false;
  subagentFormName.focus();
}

async function saveSubAgentForm() {
  const name = subagentFormName.value.trim();
  const description = subagentFormDesc.value.trim();
  const model = subagentFormModel.value.trim();
  const persona = subagentFormPersona.value;
  if (!name) {
    addSystemMessage("Sub-agent name is required.");
    return;
  }
  try {
    if (editingSubAgentId != null) {
      await invoke("update_sub_agent", { id: editingSubAgentId, name, description, persona, model });
      addSystemMessage(`Updated sub-agent "${name}".`);
    } else {
      await invoke("create_sub_agent", { name, description, persona, model });
      addSystemMessage(`Created sub-agent "${name}".`);
    }
    subagentForm.hidden = true;
    await refreshSubAgents();
  } catch (e) {
    addSystemMessage(`Save sub-agent failed: ${e}`);
  }
}

subagentsNavItem?.addEventListener("click", () => openSubAgentsPanel());
subagentsCloseBtn?.addEventListener("click", () => { subagentsPanel.hidden = true; });
subagentsPanel?.addEventListener("click", (e) => {
  if (e.target === subagentsPanel) subagentsPanel.hidden = true;
});
subagentNewBtn?.addEventListener("click", startNewSubAgent);
subagentFormCancel?.addEventListener("click", () => { subagentForm.hidden = true; });
subagentFormSave?.addEventListener("click", saveSubAgentForm);
memoryCloseBtn?.addEventListener("click", () => { memoryPanel.hidden = true; });
memoryPanel?.addEventListener("click", (e) => {
  if (e.target === memoryPanel) memoryPanel.hidden = true;
});
memoryNewBtn?.addEventListener("click", () => openMemoryForm(null));
memoryFormCancel?.addEventListener("click", () => { memoryForm.hidden = true; });
memoryFormSave?.addEventListener("click", saveMemoryForm);
// Click the backdrop (outside the card) closes the memory form modal.
memoryForm?.addEventListener("click", (e) => { if (e.target === memoryForm) memoryForm.hidden = true; });
memoryFormTest?.addEventListener("click", testMemoryForm);

// Sidebar resize (mouse drag on the handle).
(() => {
  let dragging = false;
  sidebarResizer?.addEventListener("mousedown", (e) => {
    dragging = true;
    sidebarResizer.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const px = clampSidebarWidth(e.clientX);
    sidebar.style.width = `${px}px`;
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    sidebarResizer.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    localStorage.setItem("phoenix.sidebarWidth", sidebar.style.width);
  });
})();

// ----- Main menu bindings ------------------------------------------------
// The lock icon on the health bar opens the main-menu window.
configMenuBtn?.addEventListener("click", openConfigModal);
configCloseBtn?.addEventListener("click", () => { configModal.hidden = true; });
configModal?.addEventListener("click", (e) => {
  if (e.target === configModal) configModal.hidden = true;
});
for (const btn of configModal.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => {
    if (!btn.disabled) switchConfigTab(btn.dataset.configTab);
  });
}

// ----- About tab: inner Wiki sub-tabs (Phoenix / Features / Security / AmberCore) -----
// Isolated from the outer switchConfigTab() by using a distinct .wiki-tab-btn
// class + data-wiki-tab, so the main-menu tab logic never touches these.
(() => {
  const about = $("config-tab-about");
  if (!about) return;
  about.querySelectorAll(".wiki-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.wikiTab;
      about.querySelectorAll(".wiki-tab-btn").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      about.querySelectorAll(".wiki-sub").forEach((sub) => {
        sub.hidden = sub.id !== `wiki-sub-${tab}`;
      });
      about.scrollTop = 0; // jump the scroll area back to the top on switch
    });
  });
})();

// Chronos test protocol launcher + warning modal.
$("chronos-run-btn")?.addEventListener("click", () => {
  const m = $("chronos-modal");
  if (m) m.hidden = false;
});
$("chronos-cancel")?.addEventListener("click", () => {
  const m = $("chronos-modal");
  if (m) m.hidden = true;
});
// Click the backdrop closes the Chronos warning modal.
$("chronos-modal")?.addEventListener("click", (e) => {
  if (e.target === $("chronos-modal")) $("chronos-modal").hidden = true;
});
// "Run Protocol" — runs the full Chronos test protocol and submits the results
// to the central website. The submission endpoint is under construction, so for
// now this records intent + reports that the connection will be made later.
$("chronos-run-confirm")?.addEventListener("click", async () => {
  $("chronos-modal").hidden = true;
  addSystemMessage("Chronos test protocol: the submission website is under construction — the protocol will run and send results once the connection is wired.");
});

// Seven: alpha Chronos invitation pop-up.
$("alpha-run-btn")?.addEventListener("click", () => {
  $("alpha-popup").hidden = true;
  const m = $("chronos-modal");
  if (m) m.hidden = false; // open the Chronos / Prometheus confirmation
});
$("alpha-later-btn")?.addEventListener("click", () => { $("alpha-popup").hidden = true; });
$("alpha-close-btn")?.addEventListener("click", () => { $("alpha-popup").hidden = true; });
$("alpha-never-btn")?.addEventListener("click", async () => {
  try { await invoke("dismiss_alpha_popup"); } catch (e) { addSystemMessage(`Error: ${e}`); }
  $("alpha-popup").hidden = true;
});
$("alpha-popup")?.addEventListener("click", (e) => {
  if (e.target === $("alpha-popup")) $("alpha-popup").hidden = true;
});
launchPassForm?.addEventListener("submit", changeLaunchPassword);
cpForm?.addEventListener("submit", changePassphrase);
totpEnableBtn?.addEventListener("click", enableTotp);
totpConfirmBtn?.addEventListener("click", confirmTotp);
totpCancelBtn?.addEventListener("click", cancelTotpSetup);
totpDisableBtn?.addEventListener("click", disableTotp);

/** Dev-only demo (`#activity-demo` in the URL): replay the 10 activity
 *  states through the REAL components, one every ~1.3s — used to eyeball
 *  message spacing and styling in the running app. No-op unless the hash
 *  opts in; never part of a normal session. */
async function playActivityDemo() {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const showThinking = (text) => {
    workingBlock = null;
    const block = ensureWorkingBlock();
    setPhaseTitle("Thinking");
    const c = block.querySelector(".thinking-body .content");
    c.dataset.raw = text;
    c.textContent = text;
    updateThinkingMeta(block);
  };
  const steps = [
    () => addUserMessage("Fix the model error from yesterday and clean up the old model folders."),
    () => addModelActivity({
      kind: "explore", title: "Exploring files", status: "scanning…",
      body: activityFilesBody("N:\\Phoenix Agent\\phoenix-agent", [
        { name: "ambercore", dir: true },
        { name: "src", dir: true },
        { name: "frontend", dir: true },
        { name: "app.js", size: "212 KB" },
        { name: "Cargo.toml", size: "3 KB" },
      ]),
    }),
    () => addModelActivity({
      kind: "code", title: "Writing code", status: "editing…",
      lang: "ambercore/src/model/qwen35.rs", plus: 5, minus: 1,
      body: activityCodeBody("ambercore/src/model/qwen35.rs", [
        "-        let out = t.broadcast_as((rep, heads, len, d))?;",
        "+        let out = t.reshape((1, k_heads, len, d))?",
        "+            .broadcast_as((rep, k_heads, len, d))?",
        "+            .contiguous()?",
        "+            .reshape((v_heads, len, d))?;",
        "         Ok(out)",
      ]),
    }),
    () => addModelActivity({
      kind: "terminal", title: "Running task", status: "running…",
      body: activityTermBody("cargo test --release", [
        "running 88 tests",
        "test model::qwen35::kv_head_grow_layouts ... ok",
      ]),
    }),
    () => addModelActivity({
      kind: "approval", title: "Validation needed", status: "waiting for you", state: "attention",
      body: activityQuestionBody(
        "I'm about to delete 3 old model folders in models/ (~12.4 GB total). Proceed?"
      ),
    }),
    () => addModelActivity({
      kind: "error", title: "Error — I need your help", status: "blocked", state: "attention",
      body: activityErrorBody(
        "model error: sample: A weight is negative, too large or not a valid number",
        "The GGUF looks corrupted — I can re-pull it (2.4 GB), or you point me at another file."
      ),
    }),
    () => addModelActivity({
      kind: "image", title: "Generating image", status: "diffusing…",
      body: activityProgressBody(64, "logo-concept-3.png · 1024×1024 · step 19/30"),
    }),
    () => addModelActivity({
      kind: "model3d", title: "Generating 3D model", status: "texturing…",
      body: activityProgressBody(40, "spaceship.glb · phase 2/3 — baking textures"),
    }),
    () => addModelActivity({
      kind: "web", title: "Browsing the web", status: "reading…",
      body: activityWebBody("search: qwen 3.5 broadcast repeat fix", [
        { fav: "g", title: "ggml_repeat_4d — tiling semantics", domain: "github.com" },
        { fav: "g", title: "llama.cpp qwen35.cpp — GDN head grow", domain: "github.com" },
        { fav: "h", title: "candle broadcast_as — stride-0 views", domain: "hf.co" },
      ]),
    }),
    () => addModelActivity({
      kind: "subagent", title: "Running sub-agent", status: "researching…",
      body: activitySubagentBody(
        "Researcher",
        "Qwen3.5-0.8B · local",
        "Find the exact ggml_repeat_4d tiling semantics in llama.cpp and cite the file + line."
      ),
    }),
    () => showThinking(
      "The error is in the GDN head grow — the 4B is the first model where n_v ≠ n_k, " +
      "so the broadcast path never ran on a validated model. ACRoad §6b says: check the " +
      "copy-axis placement before touching anything else…"
    ),
  ];
  for (const step of steps) {
    step();
    await sleep(1300);
  }
}

// Start.
init();

// Demo trigger: only when the URL hash opts in (dev harness / manual check).
if (location.hash === "#activity-demo") {
  (async () => {
    for (let i = 0; i < 100 && !(chatScreen && chatScreen.classList.contains("active")); i++) {
      await new Promise((r) => setTimeout(r, 200));
    }
    await new Promise((r) => setTimeout(r, 400));
    await playActivityDemo();
  })();
}

// Preview/debug hook — lets the browser harness (frontend/_preview.html)
// render sample UI states through the REAL code paths. Never used by the
// app itself; harmless if removed.
window.__phoenix = {
  chat: () => chatMessages,
  addUserMessage, addAssistantMessage, addSystemMessage,
  addModelActivity, setActivityStatus, setCodeDiff, bumpCodeDiff,
  activityFilesBody, activityCodeBody, activityTermBody, activityWebBody,
  activitySubagentBody, activityProgressBody, activityQuestionBody, activityErrorBody,
  showThinking(text) {
    workingBlock = null;
    const block = ensureWorkingBlock();
    setPhaseTitle("Thinking");
    const content = block.querySelector(".thinking-body .content");
    content.dataset.raw = text;
    content.textContent = text;
    updateThinkingMeta(block);
    return block;
  },
};

