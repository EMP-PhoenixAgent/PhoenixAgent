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
const agentPhase = $("agent-phase");
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
const workdirChangeBtn = $("workdir-change-btn");
const profileSelect = $("profile-select");
const profileNewBtn = $("profile-new-btn");
const modelsPanel = $("models-panel");
const modelsCloseBtn = $("models-close-btn");
// Models panel v0.5 — AmberCore / Ollama / Provider API
const icBox = $("ambercore-box");
const icDir = $("ic-dir");
const icDirClear = $("ic-dir-clear");
const icUrl = $("ic-url");
const icTokenizerUrl = $("ic-tokenizer-url");
const icPullBtn = $("ic-pull-btn");
const icProgress = $("ic-progress");
const icList = $("ic-list");
const olBox = $("ollama-box");
const olInstallBtn = $("ol-install-btn");
const olPull = $("ol-pull");
const olPullBtn = $("ol-pull-btn");
const olProgress = $("ol-progress");
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
let streamingThinking = null; // the thinking block currently receiving reasoning
let toolCards = {}; // index -> tool-card element for the current turn's batch
let subAgentBlocks = {}; // index -> sub-agent block nested in the delegate card
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

  // Dev convenience: in debug builds (`cargo tauri dev`), auto-unlock with the
  // known dev launch password so iteration doesn't require retyping it every
  // restart. `is_dev` is false in release builds, so this never ships.
  const dev = await invoke("is_dev").catch(() => false);

  if (!ready) {
    // First run — show setup screen.
    setupScreen.classList.add("active");
    setupPassphrase.focus();
  } else if (dev) {
    // Returning user, dev build — skip the unlock screen automatically.
    passphraseInput.value = "PhoenixAgent";
    await doUnlock();
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

    addSystemMessage(`Welcome! Encrypted memory created. Working in: ${result.project_path}`);
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

    // Hardware check-up at launch — preloads the Telemetry tab baseline.
    refreshTelemetryTab();

    // Add welcome message.
    addSystemMessage(`Ready. Working in: ${result.project_path}`);

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

// ----- Mode selector (Plan/Think/Auto) above the send button -----
function setActiveMode(mode) {
  document.querySelectorAll(".mode-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode)
  );
}
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const m = btn.dataset.mode;
    invoke("set_mode", { mode: m })
      .then(() => setActiveMode(m))
      .catch((e) => addSystemMessage(`Error: ${e}`));
  });
});
// Restore the active mode on load (default Auto).
invoke("get_mode").then(setActiveMode).catch(() => {});

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
  streamingThinking = null;
  toolCards = {};
  subAgentBlocks = {};
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
      addSystemMessage("Commands: /model <name>, /new, /clear, /learn, /context-resume, /help");
      break;
    case "/new":
      invoke("new_session").catch((e) => addSystemMessage(`Error: ${e}`));
      chatMessages.innerHTML = "";
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
    case "/learn":
      addSystemMessage("Compacting this conversation into a memory note…");
      invoke("learn")
        .then((r) => { if (r) addSystemMessage(r); })
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
        if (!streamingThinking) {
          streamingThinking = createThinkingBlock();
        }
        const content = streamingThinking.querySelector(".thinking-body .content");
        content.dataset.raw = (content.dataset.raw || "") + payload.delta;
        content.textContent += payload.delta;
        updateThinkingMeta(streamingThinking);
        scrollToBottom();
        break;
      }
      case "assistant_delta": {
        // The visible answer begins → fold the thinking block away (auto-collapse).
        finalizeThinking();
        if (!streamingBubble) {
          streamingBubble = addAssistantMessage("");
        }
        streamingBubble.querySelector(".content").textContent += payload.delta;
        setPhase("Answering…");
        scrollToBottom();
        break;
      }
      case "assistant_message": {
        finalizeThinking();
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
        subAgentBlocks = {};
        scrollToBottom();
        break;
      }
      case "tool_started": {
        finalizeThinking();
        streamingBubble = null; // stop streaming into an assistant bubble
        createToolCard(payload.index, payload.name, payload.args);
        setPhase(`Running ${toolLabel(payload.name)}…`);
        break;
      }
      case "tool_needs_approval": {
        finalizeThinking();
        // Render the card (if not already) and transition it to approval state.
        if (!toolCards[payload.index]) {
          createToolCard(payload.index, payload.name, payload.args);
        }
        setToolApproval(payload.index, payload.name);
        setPhase(`Needs approval: ${toolLabel(payload.name)}`);
        break;
      }
      case "tool_finished": {
        finishToolCard(payload.index, payload.success, payload.result, payload.duration_ms);
        break;
      }
      case "tool_denied": {
        if (toolCards[payload.index]) {
          finishToolCard(payload.index, false, "Denied by user.", null);
        } else {
          addSystemMessage(`Tool denied: ${payload.name}`);
        }
        break;
      }
      case "subagent_started": {
        if (toolCards[payload.index]) {
          addSubAgentBlock(payload.index, payload.name, payload.model, payload.task);
        }
        setPhase(`Delegating to ${payload.name}…`);
        break;
      }
      case "subagent_delta": {
        const block = subAgentBlocks[payload.index];
        if (block) {
          block.querySelector(".subagent-text").textContent += payload.text;
          scrollToBottom();
        }
        break;
      }
      case "subagent_reasoning": {
        const block = subAgentBlocks[payload.index];
        if (block) {
          const rc = block.querySelector(".subagent-reasoning .content");
          rc.dataset.raw = (rc.dataset.raw || "") + payload.text;
          rc.textContent += payload.text;
          block.querySelector(".subagent-reasoning").hidden = false;
          scrollToBottom();
        }
        break;
      }
      case "subagent_finished": {
        const block = subAgentBlocks[payload.index];
        if (block) {
          block.querySelector(".subagent-status").textContent = "✓ done";
          block.classList.add("done");
        }
        break;
      }
      case "turn_done": {
        isAgentBusy = false;
        sendBtn.disabled = false;
        finalizeThinking();
        streamingBubble = null;
        toolCards = {};
        subAgentBlocks = {};
        setPhase(null);
        refreshHealthLabel(); // idle indicator + final metrics right away
        break;
      }
      case "error": {
        const msg = payload.message || payload.error || JSON.stringify(payload);
        addSystemMessage(`Error: ${msg}`);
        isAgentBusy = false;
        sendBtn.disabled = false;
        finalizeThinking();
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
  div.className = "message user";
  div.innerHTML = `<div class="role-badge">You</div><div class="content">${escapeHtml(text)}</div>`;
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

function addSystemMessage(text) {
  const div = document.createElement("div");
  div.className = "message system";
  div.innerHTML = `<div class="content">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

// ----- Reasoning / tool pipeline UI ----------------------------------------
// Builds the visible step-by-step pipeline: a collapsible thinking block,
// expandable tool cards (correlated start↔finish), nested sub-agent cards, and
// a live phase pill. All reuse the glass / role-tint / flame-glow language.

/** Show / update / hide the live phase pill. `null` hides it. */
function setPhase(label) {
  if (!agentPhase) return;
  if (!label) {
    agentPhase.hidden = true;
    return;
  }
  agentPhase.hidden = false;
  agentPhase.innerHTML = `<span class="phase-dot"></span><span class="phase-label">${escapeHtml(label)}</span>`;
}

/** Icon (emoji) for a tool name. */
function toolIcon(name) {
  const map = {
    read_file: "📖", write_file: "✍️", edit_file: "✎", list_dir: "📁",
    grep: "🔍", run_command: "⚙️", delegate: "🤖",
  };
  return map[name] || "🛠";
}

/** Human label for a tool name. */
function toolLabel(name) {
  return (name || "tool").replace(/_/g, " ");
}

/** Create an expanded, glowing thinking block that receives reasoning deltas. */
function createThinkingBlock() {
  const block = document.createElement("div");
  block.className = "thinking-block active";
  block.innerHTML = `
    <div class="thinking-header">
      <span class="thinking-icon">◷</span>
      <span class="thinking-title">Thinking</span>
      <span class="thinking-meta">…</span>
      <span class="chevron">▾</span>
    </div>
    <div class="thinking-body"><div class="content"></div></div>`;
  block.querySelector(".thinking-header").addEventListener("click", () => {
    block.classList.toggle("collapsed");
    block.querySelector(".chevron").textContent = block.classList.contains("collapsed") ? "▸" : "▾";
  });
  chatMessages.appendChild(block);
  scrollToBottom();
  return block;
}

/** Update the "N lines · M chars" meta on a thinking block while it streams. */
function updateThinkingMeta(block) {
  const raw = block.querySelector(".thinking-body .content").dataset.raw || "";
  const lines = raw.split("\n").length;
  const chars = raw.length;
  block.querySelector(".thinking-meta").textContent = `${lines} line${lines === 1 ? "" : "s"} · ${chars} chars`;
}

/** Fold the streaming thinking block away (auto-collapse) and render markdown. */
function finalizeThinking() {
  if (!streamingThinking) return;
  const block = streamingThinking;
  streamingThinking = null;
  const content = block.querySelector(".thinking-body .content");
  const raw = content.dataset.raw || "";
  block.classList.remove("active");
  if (raw.trim()) {
    content.innerHTML = renderMarkdown(raw);
    updateThinkingMeta(block);
    block.classList.add("collapsed");
    block.querySelector(".chevron").textContent = "▸";
  } else {
    // Nothing was actually reasoned — drop the empty block entirely.
    block.remove();
  }
}

/** Create a tool card in "running" state and register it for its index. */
function createToolCard(index, name, argsJson) {
  const card = document.createElement("div");
  card.className = "tool-card running";
  card.dataset.index = String(index);
  card.innerHTML = `
    <div class="tool-card-header">
      <span class="tool-icon">${toolIcon(name)}</span>
      <span class="tool-name">${escapeHtml(name)}</span>
      <span class="tool-status">running…</span>
      <span class="tool-duration"></span>
      <span class="chevron">▾</span>
    </div>
    <div class="tool-card-body">
      <div class="tool-section tool-args">
        <span class="tool-section-label">ARGS</span>
        <pre>${escapeHtml(prettyJson(argsJson))}</pre>
      </div>
      <div class="tool-section tool-result" hidden>
        <span class="tool-section-label">RESULT</span>
        <pre></pre>
      </div>
      <div class="tool-approval" hidden></div>
      <div class="subagent-stack"></div>
    </div>`;
  card.querySelector(".tool-card-header").addEventListener("click", () => {
    card.classList.toggle("collapsed");
    card.querySelector(".chevron").textContent = card.classList.contains("collapsed") ? "▸" : "▾";
  });
  chatMessages.appendChild(card);
  toolCards[index] = card;
  scrollToBottom();
  return card;
}

/** Transition a tool card into its approval state with inline Approve/Deny. */
function setToolApproval(index, name) {
  const card = toolCards[index];
  if (!card) return;
  card.classList.remove("running");
  card.classList.add("approval");
  card.querySelector(".tool-status").textContent = "needs approval";
  const ap = card.querySelector(".tool-approval");
  ap.hidden = false;
  ap.innerHTML = `
    <div class="approval-text">⚠ Approve <b>${escapeHtml(name)}</b>?</div>
    <div class="approval-actions">
      <button class="btn-approve">Approve</button>
      <button class="btn-deny">Deny</button>
    </div>`;
  ap.querySelector(".btn-approve").addEventListener("click", () => {
    invoke("approve", { index }).catch((e) => addSystemMessage(`Error: ${e}`));
  });
  ap.querySelector(".btn-deny").addEventListener("click", () => {
    invoke("deny", { index }).catch((e) => addSystemMessage(`Error: ${e}`));
  });
  scrollToBottom();
}

/** Fill a tool card's result, mark it done/failed, and collapse it. */
function finishToolCard(index, success, result, durationMs) {
  let card = toolCards[index];
  if (!card) {
    // Defensive: a finish without a start (e.g. denied before start).
    card = createToolCard(index, "?", "{}");
  }
  card.classList.remove("running", "approval");
  card.classList.add(success ? "done" : "failed");
  card.querySelector(".tool-status").textContent = success ? "✓ done" : "✗ failed";
  const res = card.querySelector(".tool-result");
  res.hidden = false;
  res.querySelector("pre").textContent = result ?? "";
  if (durationMs != null) {
    card.querySelector(".tool-duration").textContent = formatDuration(durationMs);
  }
  // Collapse on finish (click header to re-expand).
  card.classList.add("collapsed");
  card.querySelector(".chevron").textContent = "▸";
  const ap = card.querySelector(".tool-approval");
  if (ap) ap.hidden = true; // decision made
  scrollToBottom();
}

/** Add a nested sub-agent block inside the delegate tool card. */
function addSubAgentBlock(index, name, model, task) {
  const card = toolCards[index];
  if (!card) return;
  const stack = card.querySelector(".subagent-stack");
  const block = document.createElement("div");
  block.className = "subagent";
  block.innerHTML = `
    <div class="subagent-header">
      <span class="subagent-badge">🤖</span>
      <span class="subagent-name">${escapeHtml(name)}</span>
      <span class="subagent-model">${escapeHtml(model)}</span>
      <span class="subagent-status">working…</span>
    </div>
    ${task ? `<div class="subagent-task">${escapeHtml(task)}</div>` : ""}
    <details class="subagent-reasoning" hidden>
      <summary>thinking…</summary>
      <div class="content"></div>
    </details>
    <div class="subagent-text"></div>`;
  stack.appendChild(block);
  subAgentBlocks[index] = block;
  scrollToBottom();
  return block;
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
    // Live metrics chip from the dispatch layer (works for every backend).
    let stats = null;
    try { stats = await invoke("get_runtime_metrics"); } catch { /* pre-unlock */ }
    if (stats) {
      const chip = document.createElement("span");
      chip.className = "health-metrics";
      const parts = [];
      if (stats.tokens_per_sec != null) parts.push(`${Number(stats.tokens_per_sec).toFixed(1)} T/s`);
      if (stats.ttft_ms != null) parts.push(`TTFT ${Math.round(stats.ttft_ms)} ms`);
      if (stats.tbt_avg_ms != null) parts.push(`TBT ${Number(stats.tbt_avg_ms).toFixed(1)} ms`);
      const busy = !!(stats.busy || isAgentBusy);
      chip.innerHTML =
        (parts.length ? ` · ${escapeHtml(parts.join(" · "))}` : "") +
        ` <span class="metrics-busy${busy ? " busy" : ""}">${busy ? "● generating" : "○ idle"}</span>`;
      modelItem.appendChild(chip);
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

/** Load + render the working directory display. */
async function loadWorkdir() {
  try {
    const wd = await invoke("get_workdir");
    workdirDisplay.textContent = wd || "—";
    workdirDisplay.title = wd || "";
  } catch (e) {
    console.warn("Workdir load failed:", e);
  }
}

/** Load profiles into the selector and mark the active one. */
async function loadProfiles(activeProfile) {
  try {
    const profiles = await invoke("list_profiles");
    profileSelect.innerHTML = "";
    if (profiles.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "(no profiles)";
      profileSelect.appendChild(opt);
      return;
    }
    let activeId = null;
    if (activeProfile && activeProfile.id != null) {
      activeId = activeProfile.id;
    } else if (profiles.find((p) => p.is_default)) {
      activeId = profiles.find((p) => p.is_default).id;
    } else {
      activeId = profiles[0].id;
    }
    for (const p of profiles) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name + (p.is_default ? " (default)" : "");
      if (p.id === activeId) opt.selected = true;
      profileSelect.appendChild(opt);
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

/** Highlight the box that matches the active route with the flame glow. */
function highlightActiveBox(route) {
  icBox.classList.toggle("active", route.kind === "local" && route.backend === "ambercore");
  olBox.classList.toggle("active", route.kind === "local" && route.backend === "ollama");
  prBox.classList.toggle("active", route.kind === "cloud");
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
    icList.innerHTML = '<li class="panel-loading">No AmberCore models found. Pull one by URL above.</li>';
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
      `<button class="btn-run">${isActive ? "Running" : "Run"}</button>`;
    li.querySelector(".btn-run").addEventListener("click", () => runAmberCore(m.name));
    if (isActive) li.querySelector(".btn-run").style.borderColor = "var(--phoenix-warm)";
    icList.appendChild(li);
  }
}

/** Pull a GGUF model (and its tokenizer) from a URL into the AmberCore models directory. */
async function pullAmberCore() {
  const url = icUrl.value.trim();
  if (!url) { addSystemMessage("Enter a model URL first."); return; }
  const tokenizerUrl = icTokenizerUrl?.value.trim() || null;
  icProgress.hidden = false;
  icProgress.querySelector(".mp-progress-text").textContent = "Starting download…";
  icProgress.querySelector(".mp-progress-bar").style.setProperty("--mp-pct", "0%");
  icPullBtn.disabled = true;
  try {
    const tag = await invoke("pull_ambercore_model", { url, tokenizerUrl });
    addSystemMessage(`Pulled AmberCore model: ${tag} (model + tokenizer ready)`);
    icUrl.value = "";
    if (icTokenizerUrl) icTokenizerUrl.value = "";
    await renderAmberCore(await invoke("get_active_route"));
  } catch (e) {
    addSystemMessage(`AmberCore pull failed: ${e}`);
  } finally {
    icProgress.hidden = true;
    icPullBtn.disabled = false;
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
      `<button class="btn-run">${isActive ? "Running" : "Run"}</button>`;
    li.querySelector(".btn-run").addEventListener("click", () => runOllama(m.name));
    if (isActive) li.querySelector(".btn-run").style.borderColor = "var(--phoenix-warm)";
    olList.appendChild(li);
  }
}

/** Pull an Ollama-hosted model via `ollama pull`. */
async function pullOllama() {
  const name = olPull.value.trim();
  if (!name) { addSystemMessage("Enter a model name first."); return; }
  olProgress.hidden = false;
  olProgress.querySelector(".mp-progress-text").textContent = "Pulling…";
  olPullBtn.disabled = true;
  try {
    await invoke("pull_ollama_model", { name });
    addSystemMessage(`Pulled Ollama model: ${name}`);
    olPull.value = "";
    await renderOllama(await invoke("get_active_route"));
  } catch (e) {
    addSystemMessage(`Ollama pull failed: ${e}`);
  } finally {
    olProgress.hidden = true;
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

/** Create a new profile via prompt, then refresh the selector. */
async function createNewProfile() {
  const name = window.prompt("Profile name:");
  if (!name || !name.trim()) return;
  try {
    const id = await invoke("create_profile", { name: name.trim() });
    await loadProfiles();
    // Auto-switch to the freshly created profile.
    profileSelect.value = String(id);
    await onProfileChange();
    addSystemMessage(`Created profile "${name.trim()}".`);
  } catch (e) {
    addSystemMessage(`Create profile failed: ${e}`);
  }
}

/** Handle a profile selector change. */
async function onProfileChange() {
  const id = Number(profileSelect.value);
  if (!id) return;
  try {
    const p = await invoke("switch_profile", { id });
    activeProfileId = p.id;
    updateSkillsDot();
    updateToolsDot();
    updateContextDot();
    updateMemoryDot();
    addSystemMessage(`Profile switched to "${p.name}".`);
  } catch (e) {
    addSystemMessage(`Profile switch failed: ${e}`);
  }
}

/** Prompt for a new working directory and apply it live. */
async function changeWorkdir() {
  const path = window.prompt("Working directory path:", workdirDisplay.textContent);
  if (!path || !path.trim()) return;
  try {
    await invoke("set_workdir", { path: path.trim() });
    await loadWorkdir();
    addSystemMessage(`Working directory set to ${path.trim()}.`);
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

/** Switch between Security / Telemetry / About tabs. */
function switchConfigTab(tab) {
  for (const btn of configModal.querySelectorAll(".tab-btn")) {
    if (btn.disabled) continue;
    btn.classList.toggle("active", btn.dataset.configTab === tab);
  }
  $("config-tab-security").hidden = tab !== "security";
  $("config-tab-telemetry").hidden = tab !== "telemetry";
  $("config-tab-about").hidden = tab !== "about";
  if (tab === "telemetry") refreshTelemetryTab();
}

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
// Models panel v0.5 — AmberCore / Ollama / Provider API wiring.
icPullBtn?.addEventListener("click", pullAmberCore);
icUrl?.addEventListener("keydown", (e) => { if (e.key === "Enter") pullAmberCore(); });
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
// Pull-progress events streamed from the backend.
listen("ambercore-pull-progress", (e) => {
  const { completed, total } = e.payload;
  const label = e.payload?.phase === "tokenizer" ? "Tokenizer" : "Model";
  const pct = total ? Math.min(100, Math.round((completed / total) * 100)) : 0;
  const bar = icProgress.querySelector(".mp-progress-bar");
  if (bar) bar.style.setProperty("--mp-pct", `${pct}%`);
  const txt = icProgress.querySelector(".mp-progress-text");
  if (txt) txt.textContent = total
    ? `${label} · ${pct}% · ${humanBytes(completed)} / ${humanBytes(total)}`
    : `${label} · ${humanBytes(completed)} downloaded`;
});
listen("ollama-pull-progress", (e) => {
  const txt = olProgress.querySelector(".mp-progress-text");
  if (txt) txt.textContent = String(e.payload?.line ?? "").slice(0, 80);
});

// Profile selector + new-profile button.
profileSelect?.addEventListener("change", onProfileChange);
profileNewBtn?.addEventListener("click", createNewProfile);

// Workdir change.
workdirChangeBtn?.addEventListener("click", changeWorkdir);

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

function openSubAgentForm(sa) {
  editingSubAgentId = sa ? sa.id : null;
  subagentFormTitle.textContent = sa ? `Edit: ${sa.name}` : "New sub-agent";
  subagentFormName.value = sa ? sa.name : "";
  subagentFormDesc.value = sa ? sa.description : "";
  subagentFormModel.value = sa ? sa.model : "";
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

// Start.
init();
