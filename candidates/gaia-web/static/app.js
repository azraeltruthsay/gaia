/**
 * GAIA Mission Control — Dashboard JavaScript
 *
 * Modules: Navigation, Chat, System State, Blueprints, Audio, Voice, Hooks
 * No build step — vanilla JS + D3 from CDN.
 */

// ── Configuration ────────────────────────────────────────────────────────────

const POLL_INTERVAL = 10_000; // system state poll every 10s
const CHAT_TIMEOUT = 300_000; // 5 minute timeout for chat
const HOOKS_POLL_INTERVAL = 15_000; // hooks status poll every 15s

// ── Navigation ──────────────────────────────────────────────────────────────

let currentView = "dashboard";
let hooksPollTimer = null;

function switchView(viewName) {
  currentView = viewName;
  // Update tabs
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.view === viewName);
  });
  // Update views
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("active", v.dataset.view === viewName);
  });
  // Start/stop hooks polling based on active view
  if (viewName === "hooks") {
    hookRefreshAll();
    if (!hooksPollTimer) {
      hooksPollTimer = setInterval(hookRefreshAll, HOOKS_POLL_INTERVAL);
    }
  } else if (hooksPollTimer) {
    clearInterval(hooksPollTimer);
    hooksPollTimer = null;
  }
}

// Bind tab clicks
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

// ── Chat Panel ───────────────────────────────────────────────────────────────

const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

function addChatMessage(text, type) {
  const el = document.createElement("div");
  el.className = `chat-msg ${type}`;
  if (type === "gaia" && typeof marked !== "undefined") {
    el.innerHTML = marked.parse(text);
  } else {
    el.textContent = text;
  }
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChat() {
  const text = chatInput.value.trim();
  if (!text) return;

  addChatMessage(text, "user");
  chatInput.value = "";
  chatSend.disabled = true;

  try {
    const resp = await fetch(
      `/process_user_input?user_input=${encodeURIComponent(text)}`,
      { method: "POST", signal: AbortSignal.timeout(CHAT_TIMEOUT) }
    );
    const data = await resp.json();
    if (resp.ok) {
      addChatMessage(data.response || "(no response)", "gaia");
    } else {
      addChatMessage(`Error ${resp.status}: ${data.detail || data.response || "unknown"}`, "system");
    }
  } catch (err) {
    addChatMessage(`Connection error: ${err.message}`, "system");
  } finally {
    chatSend.disabled = false;
    chatInput.focus();
  }
}

chatSend.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});

// ── System State Panel ───────────────────────────────────────────────────────

const serviceGridEl = document.getElementById("service-grid");
const discordBadgeEl = document.getElementById("discord-badge");

// Display name mapping for cleaner labels
const SERVICE_LABELS = {
  "gaia-core": "Core",
  "gaia-web": "Web",
  "gaia-orchestrator": "Orchestrator",
  "gaia-prime": "Prime",
  "gaia-mcp": "MCP",
  "gaia-study": "Study",
  "gaia-core-candidate": "Core (cand.)",
  "discord": "Discord",
};

function statusClass(status) {
  if (status === "online" || status === "ready") return "ok";
  if (status === "offline" || status === "not_started" || status === "not_available") return "error";
  if (status === "connecting" || status === "timeout") return "warn";
  if (typeof status === "string" && status.startsWith("error")) return "error";
  return "";
}

function renderServiceGrid(services) {
  serviceGridEl.innerHTML = "";

  // Sleep state card first (special — fetched from sleep endpoint)
  const sleepCard = document.createElement("div");
  sleepCard.className = "state-card state-card-sleep";
  sleepCard.id = "sleep-state-card";
  sleepCard.innerHTML = `
    <div class="state-label">Sleep State</div>
    <div id="sleep-state" class="state-value">--</div>
  `;
  serviceGridEl.appendChild(sleepCard);

  // GPU owner card
  const gpuCard = document.createElement("div");
  gpuCard.className = "state-card";
  gpuCard.id = "gpu-owner-card";
  gpuCard.innerHTML = `
    <div class="state-label">GPU Owner</div>
    <div id="gpu-owner" class="state-value">--</div>
  `;
  serviceGridEl.appendChild(gpuCard);

  // Service cards
  for (const svc of services) {
    const card = document.createElement("div");
    const cls = statusClass(svc.status);
    card.className = `state-card${svc.candidate ? " state-card-candidate" : ""}`;

    const label = SERVICE_LABELS[svc.id] || svc.id;
    const latency = svc.latency_ms != null ? `${svc.latency_ms}ms` : "";
    const candidateBadge = svc.candidate ? `<span class="state-candidate-badge">CAND</span>` : "";

    // Discord gets special treatment
    let extra = "";
    if (svc.id === "discord" && svc.discord) {
      const d = svc.discord;
      if (d.user) extra = `<div class="state-extra">${d.user} · ${d.guilds} guild${d.guilds !== 1 ? "s" : ""}</div>`;
    }

    card.innerHTML = `
      <div class="state-label">${label} ${candidateBadge}</div>
      <div class="state-value ${cls}">${svc.status}</div>
      ${latency ? `<div class="state-latency">${latency}</div>` : ""}
      ${extra}
    `;
    serviceGridEl.appendChild(card);
  }
}

async function pollSystemState() {
  // Fetch all service statuses
  try {
    const resp = await fetch("/api/system/services");
    if (resp.ok) {
      const services = await resp.json();
      renderServiceGrid(services);

      // Update Discord badge in header
      const discord = services.find((s) => s.id === "discord");
      if (discord) {
        const cls = statusClass(discord.status);
        discordBadgeEl.textContent = `discord: ${discord.status}`;
        discordBadgeEl.className = `badge badge-${cls}`;
        if (discord.discord?.latency_ms) {
          discordBadgeEl.title = `Discord: ${discord.discord.user || "?"} (${discord.discord.latency_ms}ms)`;
        }
      }
    }
  } catch {
    // Service endpoint not available — fall back
    serviceGridEl.innerHTML = '<div class="state-card"><div class="state-value error">services unreachable</div></div>';
  }

  // Sleep / core status (for sleep state + GPU owner cards)
  try {
    const resp = await fetch("/api/system/sleep");
    if (resp.ok) {
      const data = await resp.json();
      const state = data.state || data.sleep_state || "unknown";
      const stateMap = { active: "ok", asleep: "warn", drowsy: "warn", waking: "ok" };
      const el = document.getElementById("sleep-state");
      if (el) {
        el.textContent = state;
        el.className = `state-value ${stateMap[state] || ""}`;
      }
      if (data.gpu_owner) {
        const gpuEl = document.getElementById("gpu-owner");
        if (gpuEl) {
          gpuEl.textContent = data.gpu_owner;
          gpuEl.className = "state-value ok";
        }
      }
    }
  } catch {
    // sleep endpoint unavailable
  }

  // Orchestrator status (for GPU owner if not from sleep)
  try {
    const resp = await fetch("/api/system/status");
    if (resp.ok) {
      const data = await resp.json();
      if (data.gpu_owner) {
        const gpuEl = document.getElementById("gpu-owner");
        if (gpuEl && gpuEl.textContent === "--") {
          gpuEl.textContent = data.gpu_owner;
          gpuEl.className = "state-value ok";
        }
      }
    }
  } catch {
    // orchestrator unavailable
  }
}

// ── Graph Visualization ──────────────────────────────────────────────────────

let graphData = null;
let currentGraphView = "service"; // "service" or "component"
let currentComponentServiceId = null;

const graphTitleEl = document.getElementById("graph-title");
const graphBackEl = document.getElementById("graph-back");

// Back button → return to service graph
graphBackEl.addEventListener("click", backToServiceGraph);

function backToServiceGraph() {
  currentGraphView = "service";
  currentComponentServiceId = null;
  graphTitleEl.textContent = "Service Graph";
  graphBackEl.style.display = "none";
  if (graphData) renderGraph(graphData);
}

async function loadGraph() {
  try {
    const resp = await fetch("/api/blueprints/graph?include_candidates=true");
    if (!resp.ok) return;
    graphData = await resp.json();
    renderGraph(graphData);
    updateStatusBar(graphData);
  } catch {
    // graph not available yet
  }
}

function updateStatusBar(data) {
  document.getElementById("service-count").textContent = `${data.blueprint_count} services`;
  const pendingEl = document.getElementById("pending-count");
  pendingEl.textContent = `${data.pending_review_count} pending`;
  pendingEl.className = data.pending_review_count > 0 ? "badge badge-warn" : "badge";
}

function renderGraph(data) {
  const container = document.getElementById("graph-container");
  const svg = d3.select("#graph-svg");
  svg.selectAll("*").remove();

  const width = container.clientWidth;
  const height = container.clientHeight;
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  if (!data.nodes || data.nodes.length === 0) return;

  // Create a group for zoomable content
  const g = svg.append("g").attr("class", "graph-root");

  // Add zoom behavior
  const zoom = d3.zoom()
    .scaleExtent([0.3, 5])
    .on("zoom", (event) => {
      g.attr("transform", event.transform);
    });
  svg.call(zoom);

  // Double-click to reset zoom
  svg.on("dblclick.zoom", () => {
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
  });

  // Build node/link data for D3
  const nodeMap = {};
  const nodes = data.nodes.map((n) => {
    const node = { ...n };
    nodeMap[n.id] = node;
    return node;
  });

  const links = (data.edges || [])
    .filter((e) => nodeMap[e.from_service] && nodeMap[e.to_service])
    .map((e) => ({
      source: e.from_service,
      target: e.to_service,
      transport_type: e.transport_type,
      status: e.status,
      description: e.description,
      has_fallback: e.has_fallback,
    }));

  // Force simulation
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(90))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(40));

  // Tooltip
  let tooltip = d3.select(".tooltip");
  if (tooltip.empty()) {
    tooltip = d3.select("body").append("div").attr("class", "tooltip").style("display", "none");
  }

  // Edges
  const edgeStyleMap = {
    http_rest: "",
    websocket: "6,3",
    event: "4,4",
    sse: "2,2",
    mcp: "1,3",
    direct_call: "8,4",
    grpc: "3,6",
  };

  const link = g.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("class", (d) => `graph-edge ${d.status}`)
    .attr("stroke-dasharray", (d) => edgeStyleMap[d.transport_type] || "")
    .attr("stroke-width", (d) => d.has_fallback ? 2.5 : 1.5)
    .on("mouseover", (event, d) => {
      tooltip.style("display", "block")
        .html(`<strong>${d.source.id || d.source} → ${d.target.id || d.target}</strong><br>${d.transport_type} · ${d.description || ""}`)
        .style("left", `${event.pageX + 10}px`)
        .style("top", `${event.pageY - 10}px`);
    })
    .on("mouseout", () => tooltip.style("display", "none"));

  // Nodes
  const nodeColor = (d) => {
    if (d.blueprint_status === "live") return "#4caf50";
    return "#ffa726";
  };

  const node = g.append("g")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("class", "graph-node")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
    )
    .on("click", (event, d) => {
      selectBlueprint(d.id);
    });

  node.append("circle")
    .attr("r", (d) => 10 + (d.interface_count || 0) * 1.5)
    .attr("fill", nodeColor)
    .attr("stroke", (d) => d.genesis ? "#e94560" : nodeColor(d))
    .attr("stroke-dasharray", (d) => d.genesis ? "3,3" : "")
    .attr("opacity", 0.85);

  node.append("text")
    .attr("dy", (d) => 10 + (d.interface_count || 0) * 1.5 + 14)
    .text((d) => d.id.replace("gaia-", ""));

  // Tick
  sim.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    node.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });
}

// ── Component Graph (drill-down) ────────────────────────────────────────────

async function loadComponentGraph(serviceId) {
  try {
    const resp = await fetch(`/api/blueprints/${serviceId}/components`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.components || data.components.length === 0) return;

    currentGraphView = "component";
    currentComponentServiceId = serviceId;
    graphTitleEl.textContent = `${serviceId} — Architecture`;
    graphBackEl.style.display = "inline-block";

    renderComponentGraph(data);
  } catch {
    // component graph not available
  }
}

function renderComponentGraph(data) {
  const container = document.getElementById("graph-container");
  const svg = d3.select("#graph-svg");
  svg.selectAll("*").remove();

  const width = container.clientWidth;
  const height = container.clientHeight;
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  if (!data.components || data.components.length === 0) return;

  // Create a group for zoomable content
  const g = svg.append("g").attr("class", "graph-root");

  // Add zoom behavior
  const zoom = d3.zoom()
    .scaleExtent([0.3, 5])
    .on("zoom", (event) => {
      g.attr("transform", event.transform);
    });
  svg.call(zoom);

  // Double-click to reset zoom
  svg.on("dblclick.zoom", () => {
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
  });

  // Build node/link data
  const nodeMap = {};
  const nodes = data.components.map((c) => {
    const node = { ...c };
    nodeMap[c.id] = node;
    return node;
  });

  const links = (data.edges || [])
    .filter((e) => nodeMap[e.from_component] && nodeMap[e.to_component])
    .map((e) => ({
      source: e.from_component,
      target: e.to_component,
      label: e.label,
      transport: e.transport,
      data_flow: e.data_flow,
    }));

  // Force simulation — tighter than service graph
  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(110))
    .force("charge", d3.forceManyBody().strength(-400))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(35));

  // Tooltip
  let tooltip = d3.select(".tooltip");
  if (tooltip.empty()) {
    tooltip = d3.select("body").append("div").attr("class", "tooltip").style("display", "none");
  }

  // Edges
  const link = g.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "component-edge")
    .on("mouseover", (event, d) => {
      tooltip.style("display", "block")
        .html(`<strong>${d.label}</strong>${d.data_flow ? `<br>Data: ${d.data_flow}` : ""}`)
        .style("left", `${event.pageX + 10}px`)
        .style("top", `${event.pageY - 10}px`);
    })
    .on("mouseout", () => tooltip.style("display", "none"));

  // Edge labels (abbreviated, placed at midpoint)
  const edgeLabels = g.append("g")
    .selectAll("text")
    .data(links)
    .join("text")
    .attr("class", "component-edge-label")
    .text((d) => d.data_flow ? d.data_flow.substring(0, 24) : "");

  // Component color palette (blue-purple range)
  const colorScale = d3.scaleOrdinal()
    .range(["#5c6bc0", "#7986cb", "#4fc3f7", "#4dd0e1", "#4db6ac", "#81c784", "#aed581"]);

  // Nodes
  const node = g.append("g")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("class", "component-node")
    .call(d3.drag()
      .on("start", (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
    )
    .on("click", (event, d) => {
      event.stopPropagation();
      showComponentDetail(d);
    });

  node.append("circle")
    .attr("r", (d) => 8 + (d.interface_count || 0) * 1.2)
    .attr("fill", (d) => colorScale(d.id))
    .attr("stroke", (d) => d3.color(colorScale(d.id)).darker(0.5))
    .attr("stroke-width", 2)
    .attr("opacity", 0.9);

  node.append("text")
    .attr("dy", (d) => 8 + (d.interface_count || 0) * 1.2 + 14)
    .text((d) => d.label);

  // Tick
  sim.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    edgeLabels
      .attr("x", (d) => (d.source.x + d.target.x) / 2)
      .attr("y", (d) => (d.source.y + d.target.y) / 2 - 4);
    node.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });
}

function showComponentDetail(comp) {
  const detailEl = document.getElementById("blueprint-detail");

  const srcList = (comp.source_files || []).map((f) =>
    `<li><code>${f}</code></li>`
  ).join("");

  const classList = (comp.key_classes || []).map((c) =>
    `<li><code>${c}</code></li>`
  ).join("");

  const funcList = (comp.key_functions || []).map((f) =>
    `<li><code>${f}</code></li>`
  ).join("");

  const ifaceList = [
    ...(comp.exposes_interfaces || []).map((i) => `<li>exposes: <code>${i}</code></li>`),
    ...(comp.consumes_interfaces || []).map((i) => `<li>consumes: <code>${i}</code></li>`),
  ].join("");

  detailEl.innerHTML = `
    <div class="component-detail">
      <h4>${comp.label}</h4>
      <p class="comp-desc">${comp.description}</p>
      ${classList ? `<h4>Key Classes</h4><ul>${classList}</ul>` : ""}
      ${funcList ? `<h4>Key Functions</h4><ul>${funcList}</ul>` : ""}
      ${srcList ? `<h4>Source Files</h4><ul>${srcList}</ul>` : ""}
      ${ifaceList ? `<h4>Interfaces</h4><ul>${ifaceList}</ul>` : ""}
    </div>
  `;
}

// ── Blueprint Panel ──────────────────────────────────────────────────────────

const blueprintListEl = document.getElementById("blueprint-list");
const blueprintDetailEl = document.getElementById("blueprint-detail");
let selectedBlueprintId = null;

async function loadBlueprintList() {
  try {
    const resp = await fetch("/api/blueprints");
    if (!resp.ok) return;
    const list = await resp.json();
    renderBlueprintList(list);
  } catch {
    blueprintListEl.innerHTML = '<div class="chat-msg system">Failed to load blueprints</div>';
  }
}

function renderBlueprintList(list) {
  blueprintListEl.innerHTML = "";
  list.sort((a, b) => a.id.localeCompare(b.id));
  for (const bp of list) {
    const card = document.createElement("div");
    card.className = `bp-card${bp.id === selectedBlueprintId ? " selected" : ""}`;
    card.dataset.id = bp.id;

    const statusClass = bp.status === "live" ? "live" : "candidate";
    let badges = `<span class="bp-badge ${statusClass}">${bp.status}</span>`;
    if (bp.genesis) badges += `<span class="bp-badge genesis">genesis</span>`;

    card.innerHTML = `
      <div class="bp-card-left">
        <div class="bp-card-id">${bp.id}</div>
        <div class="bp-card-role">${bp.role}</div>
      </div>
      <div class="bp-card-badges">${badges}</div>
    `;
    card.addEventListener("click", () => selectBlueprint(bp.id));
    blueprintListEl.appendChild(card);
  }
}

async function selectBlueprint(serviceId) {
  selectedBlueprintId = serviceId;

  // Highlight card
  document.querySelectorAll(".bp-card").forEach((c) => {
    c.classList.toggle("selected", c.dataset.id === serviceId);
  });

  blueprintDetailEl.innerHTML = "<em>Loading...</em>";

  // Load blueprint markdown + component graph in parallel
  const mdPromise = fetch(`/api/blueprints/${serviceId}/markdown`)
    .then(async (resp) => {
      if (!resp.ok) {
        blueprintDetailEl.innerHTML = `<em>Error ${resp.status}</em>`;
        return;
      }
      const md = await resp.text();
      if (typeof marked !== "undefined") {
        blueprintDetailEl.innerHTML = marked.parse(md);
      } else {
        blueprintDetailEl.textContent = md;
      }
    })
    .catch((err) => {
      blueprintDetailEl.innerHTML = `<em>Failed: ${err.message}</em>`;
    });

  const graphPromise = loadComponentGraph(serviceId);

  await Promise.all([mdPromise, graphPromise]);
}

// ── Audio Processing Widget ──────────────────────────────────────────────────

const AUDIO_ENDPOINT = "/api/audio";  // proxied through gaia-web, or direct
const AUDIO_WS_URL = (() => {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  // Try direct gaia-audio WebSocket (port 8080/8081)
  return `${proto}//${location.hostname}:8081/status/ws`;
})();

const audioStateBadge = document.getElementById("audio-state-badge");
const audioGpuBadge = document.getElementById("audio-gpu-badge");
const audioMuteBtn = document.getElementById("audio-mute-btn");
const audioVramFill = document.getElementById("audio-vram-fill");
const audioVramText = document.getElementById("audio-vram-text");
const audioTranscriptLog = document.getElementById("audio-transcript-log");
const audioEventLog = document.getElementById("audio-event-log");

let audioWs = null;
let audioMuted = false;
let audioSttLatencies = [];
let audioTtsLatencies = [];
const VRAM_BUDGET = 5600;

function updateAudioState(state) {
  if (!audioStateBadge) return;
  audioStateBadge.textContent = state;
  audioStateBadge.className = `audio-state-badge state-${state}`;
}

function updateAudioGpu(mode, vramMb) {
  if (audioGpuBadge) audioGpuBadge.textContent = `GPU: ${mode}`;
  if (audioVramFill) audioVramFill.style.width = `${Math.min(100, (vramMb / VRAM_BUDGET) * 100)}%`;
  if (audioVramText) audioVramText.textContent = `${Math.round(vramMb)} / ${VRAM_BUDGET} MB`;
}

function addAudioLogEntry(container, text, type, timestamp) {
  if (!container) return;
  const entry = document.createElement("div");
  entry.className = `audio-log-entry ${type}`;
  const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
  entry.innerHTML = `<span class="audio-log-time">${timeStr}</span><span class="audio-log-text">${escapeHtml(text)}</span>`;
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;

  // Keep last 50 entries
  while (container.children.length > 50) container.removeChild(container.firstChild);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function drawSparkline(canvasId, values, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || values.length < 2) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const max = Math.max(...values, 1);
  const step = w / (values.length - 1);

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  values.forEach((v, i) => {
    const x = i * step;
    const y = h - (v / max) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Latest value label
  if (values.length > 0) {
    const last = values[values.length - 1];
    ctx.fillStyle = color;
    ctx.font = "9px monospace";
    ctx.fillText(`${Math.round(last)}ms`, w - 40, 10);
  }
}

function handleAudioEvent(event) {
  const { event_type, detail, latency_ms, timestamp } = event;

  // Determine log type for coloring
  let logType = "";
  if (event_type.startsWith("stt")) logType = "stt";
  else if (event_type.startsWith("tts")) logType = "tts";
  else if (event_type.startsWith("gpu")) logType = "gpu";
  else if (event_type === "error") logType = "error";

  // Add to event log
  const eventText = `[${event_type}] ${detail}${latency_ms > 0 ? ` (${Math.round(latency_ms)}ms)` : ""}`;
  addAudioLogEntry(audioEventLog, eventText, logType, timestamp);

  // Update transcript log for STT completions
  if (event_type === "stt_complete" && detail) {
    addAudioLogEntry(audioTranscriptLog, detail, "stt", timestamp);
  }

  // Update state badge based on event
  if (event_type === "stt_start") updateAudioState("transcribing");
  else if (event_type === "tts_start") updateAudioState("synthesizing");
  else if (event_type === "stt_complete" || event_type === "tts_complete") updateAudioState("idle");
  else if (event_type === "mute") { updateAudioState("muted"); audioMuted = true; updateMuteBtn(); }
  else if (event_type === "unmute") { updateAudioState("idle"); audioMuted = false; updateMuteBtn(); }

  // Track latencies for sparklines
  if (event_type === "stt_complete" && latency_ms > 0) {
    audioSttLatencies.push(latency_ms);
    if (audioSttLatencies.length > 20) audioSttLatencies.shift();
    drawSparkline("audio-stt-sparkline", audioSttLatencies, "#4fc3f7");
  }
  if (event_type === "tts_complete" && latency_ms > 0) {
    audioTtsLatencies.push(latency_ms);
    if (audioTtsLatencies.length > 20) audioTtsLatencies.shift();
    drawSparkline("audio-tts-sparkline", audioTtsLatencies, "#ffa726");
  }
}

function handleAudioSnapshot(snap) {
  updateAudioState(snap.state || "idle");
  updateAudioGpu(snap.gpu_mode || "idle", snap.vram_used_mb || 0);
  audioMuted = snap.muted || false;
  updateMuteBtn();

  if (snap.stt_latencies) {
    audioSttLatencies = snap.stt_latencies.slice(-20);
    drawSparkline("audio-stt-sparkline", audioSttLatencies, "#4fc3f7");
  }
  if (snap.tts_latencies) {
    audioTtsLatencies = snap.tts_latencies.slice(-20);
    drawSparkline("audio-tts-sparkline", audioTtsLatencies, "#ffa726");
  }

  // Populate event log from snapshot
  if (snap.events) {
    for (const e of snap.events.slice(-20)) {
      handleAudioEvent(e);
    }
  }
}

function updateMuteBtn() {
  if (!audioMuteBtn) return;
  audioMuteBtn.textContent = audioMuted ? "Unmute" : "Mute";
  audioMuteBtn.classList.toggle("muted", audioMuted);
}

function connectAudioWs() {
  if (audioWs && audioWs.readyState <= 1) return; // Already connected or connecting

  try {
    audioWs = new WebSocket(AUDIO_WS_URL);

    audioWs.onopen = () => {
      addAudioLogEntry(audioEventLog, "WebSocket connected", "", null);
    };

    audioWs.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.state !== undefined) {
          // Full snapshot
          handleAudioSnapshot(data);
        } else if (data.event_type && data.event_type !== "keepalive") {
          // Individual event
          handleAudioEvent(data);
        }
      } catch { /* ignore parse errors */ }
    };

    audioWs.onclose = () => {
      audioWs = null;
      // Retry in 5s
      setTimeout(connectAudioWs, 5000);
    };

    audioWs.onerror = () => {
      audioWs = null;
    };
  } catch {
    // WebSocket not available — fall back to polling
    setTimeout(pollAudioStatus, 2000);
  }
}

async function pollAudioStatus() {
  // Fallback polling if WebSocket is unavailable
  if (audioWs && audioWs.readyState === 1) return; // WS is active

  try {
    const resp = await fetch(`http://${location.hostname}:8081/status`);
    if (resp.ok) {
      const snap = await resp.json();
      handleAudioSnapshot(snap);
    }
  } catch {
    // gaia-audio not available
    updateAudioState("idle");
    if (audioGpuBadge) audioGpuBadge.textContent = "GPU: offline";
  }
}

// Mute button handler
if (audioMuteBtn) {
  audioMuteBtn.addEventListener("click", async () => {
    const action = audioMuted ? "unmute" : "mute";
    try {
      await fetch(`http://${location.hostname}:8081/${action}`, { method: "POST" });
    } catch {
      addAudioLogEntry(audioEventLog, `Failed to ${action}`, "error", null);
    }
  });
}

// ── Voice Auto-Answer Widget ─────────────────────────────────────────────────

const voiceStateBadge = document.getElementById("voice-state-badge");
const voiceChannelName = document.getElementById("voice-channel-name");
const voiceDuration = document.getElementById("voice-duration");
const voiceDisconnectBtn = document.getElementById("voice-disconnect-btn");
const voiceUserList = document.getElementById("voice-user-list");

function updateVoiceState(state) {
  if (!voiceStateBadge) return;
  voiceStateBadge.textContent = state;
  voiceStateBadge.className = "voice-state-badge state-" + state;
}

function formatDuration(seconds) {
  if (!seconds || seconds <= 0) return "--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

async function pollVoiceStatus() {
  try {
    const resp = await fetch("/api/voice/status");
    if (!resp.ok) return;
    const data = await resp.json();
    updateVoiceState(data.state || "disconnected");
    if (voiceChannelName) voiceChannelName.textContent = data.channel_name || "--";
    if (voiceDuration) voiceDuration.textContent = formatDuration(data.duration_seconds);
    if (voiceDisconnectBtn) voiceDisconnectBtn.style.display = data.connected ? "" : "none";
  } catch {
    updateVoiceState("disconnected");
  }
}

async function loadVoiceUsers() {
  try {
    const resp = await fetch("/api/voice/users");
    if (!resp.ok) return;
    const users = await resp.json();
    renderVoiceUsers(users);
  } catch {
    // Voice API not available
  }
}

function renderVoiceUsers(users) {
  if (!voiceUserList) return;
  voiceUserList.innerHTML = "";
  for (const user of users) {
    const card = document.createElement("div");
    card.className = "voice-user-card" + (user.whitelisted ? " whitelisted" : "");
    card.innerHTML = `
      <span class="voice-user-name">${escapeHtml(user.name)}</span>
      <span class="voice-user-badge ${user.whitelisted ? 'on' : 'off'}">${user.whitelisted ? 'auto-answer' : 'off'}</span>
    `;
    card.addEventListener("click", () => toggleVoiceWhitelist(user.user_id, user.whitelisted));
    voiceUserList.appendChild(card);
  }
  if (users.length === 0) {
    voiceUserList.innerHTML = '<div style="font-size:10px;color:var(--text-muted);padding:8px;">No users seen yet. GAIA will populate this list from Discord activity.</div>';
  }
}

async function toggleVoiceWhitelist(userId, currentlyWhitelisted) {
  try {
    if (currentlyWhitelisted) {
      await fetch(`/api/voice/whitelist/${userId}`, { method: "DELETE" });
    } else {
      await fetch("/api/voice/whitelist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });
    }
    // Refresh list
    await loadVoiceUsers();
  } catch {
    // Silently fail
  }
}

if (voiceDisconnectBtn) {
  voiceDisconnectBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/voice/disconnect", { method: "POST" });
      await pollVoiceStatus();
    } catch {
      // Silently fail
    }
  });
}

// ── Hooks / Commands Panel ───────────────────────────────────────────────────

const hookSleepState = document.getElementById("hook-sleep-state");
const hookSleepCycle = document.getElementById("hook-sleep-cycle");
const hookSleepUptime = document.getElementById("hook-sleep-uptime");
const hookGpuOwner = document.getElementById("hook-gpu-owner");
const hookGpuVram = document.getElementById("hook-gpu-vram");
const hookCodexInput = document.getElementById("hook-codex-input");
const hookCodexResults = document.getElementById("hook-codex-results");
const hookActionLog = document.getElementById("hook-action-log");

function hookLogEntry(action, result, isError) {
  if (!hookActionLog) return;
  // Clear placeholder
  const ph = hookActionLog.querySelector(".hook-log-placeholder");
  if (ph) ph.remove();

  const entry = document.createElement("div");
  entry.className = `hook-log-entry ${isError ? "error" : "success"}`;
  const time = new Date().toLocaleTimeString();
  entry.innerHTML = `<span class="hook-log-time">${time}</span><span class="hook-log-action">${escapeHtml(action)}</span><span class="hook-log-result">${escapeHtml(result)}</span>`;
  hookActionLog.appendChild(entry);
  hookActionLog.scrollTop = hookActionLog.scrollHeight;
  while (hookActionLog.children.length > 50) hookActionLog.removeChild(hookActionLog.firstChild);
}

async function hookRefreshSleep() {
  try {
    const resp = await fetch("/api/hooks/sleep/status");
    if (resp.ok) {
      const data = await resp.json();
      if (hookSleepState) hookSleepState.textContent = data.state || data.sleep_state || "unknown";
      if (hookSleepCycle) hookSleepCycle.textContent = data.cycle_count != null ? `#${data.cycle_count}` : "--";
      if (hookSleepUptime) {
        const up = data.uptime_seconds || data.uptime;
        hookSleepUptime.textContent = up != null ? formatDuration(up) : "--";
      }
    }
  } catch {
    if (hookSleepState) hookSleepState.textContent = "unreachable";
  }
}

async function hookRefreshGpu() {
  try {
    const resp = await fetch("/api/hooks/gpu/status");
    if (resp.ok) {
      const data = await resp.json();
      if (hookGpuOwner) hookGpuOwner.textContent = data.owner || data.gpu_owner || "none";
      if (hookGpuVram) {
        const used = data.vram_used_mb || data.used_mb;
        const total = data.vram_total_mb || data.total_mb;
        hookGpuVram.textContent = used != null && total != null ? `${Math.round(used)} / ${Math.round(total)} MB` : "--";
      }
    }
  } catch {
    if (hookGpuOwner) hookGpuOwner.textContent = "unreachable";
  }
}

async function hookRefreshAll() {
  await Promise.all([hookRefreshSleep(), hookRefreshGpu()]);
}

async function hookAction(endpoint) {
  hookLogEntry(endpoint, "sending...", false);
  try {
    const resp = await fetch(`/api/hooks/${endpoint}`, { method: "POST" });
    const data = await resp.json();
    if (resp.ok) {
      hookLogEntry(endpoint, JSON.stringify(data).substring(0, 120), false);
    } else {
      hookLogEntry(endpoint, `Error ${resp.status}: ${data.error || data.detail || "unknown"}`, true);
    }
  } catch (err) {
    hookLogEntry(endpoint, `Connection error: ${err.message}`, true);
  }
  // Refresh status after action
  setTimeout(hookRefreshAll, 1000);
}

async function hookCodexSearch() {
  if (!hookCodexInput || !hookCodexResults) return;
  const query = hookCodexInput.value.trim();
  if (!query) return;

  hookCodexResults.innerHTML = '<div class="hook-codex-placeholder">Searching...</div>';
  hookLogEntry("codex/search", `query: "${query}"`, false);

  try {
    const resp = await fetch("/api/hooks/codex/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      hookCodexResults.innerHTML = `<div class="hook-codex-placeholder">Error: ${escapeHtml(err.error || err.detail || "unknown")}</div>`;
      hookLogEntry("codex/search", `Error ${resp.status}`, true);
      return;
    }
    const data = await resp.json();
    const results = data.results || data;
    if (!Array.isArray(results) || results.length === 0) {
      hookCodexResults.innerHTML = '<div class="hook-codex-placeholder">No results found.</div>';
      hookLogEntry("codex/search", "0 results", false);
      return;
    }
    hookCodexResults.innerHTML = "";
    for (const r of results) {
      const div = document.createElement("div");
      div.className = "hook-codex-result";
      const score = r.score != null ? r.score.toFixed(3) : "";
      const title = r.title || r.key || r.id || "untitled";
      const snippet = r.snippet || r.text || r.content || "";
      div.innerHTML = `
        ${score ? `<span class="hook-codex-result-score">${score}</span>` : ""}
        <div class="hook-codex-result-title">${escapeHtml(title)}</div>
        <div class="hook-codex-result-snippet">${escapeHtml(snippet.substring(0, 200))}</div>
      `;
      hookCodexResults.appendChild(div);
    }
    hookLogEntry("codex/search", `${results.length} results`, false);
  } catch (err) {
    hookCodexResults.innerHTML = `<div class="hook-codex-placeholder">Connection error: ${escapeHtml(err.message)}</div>`;
    hookLogEntry("codex/search", `Connection error: ${err.message}`, true);
  }
}

// Codex search on Enter key
if (hookCodexInput) {
  hookCodexInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      hookCodexSearch();
    }
  });
}

// ── Initialization ───────────────────────────────────────────────────────────

async function init() {
  addChatMessage("Mission Control online. Type a message to talk to GAIA.", "system");

  // Load initial data
  await Promise.all([pollSystemState(), loadGraph(), loadBlueprintList()]);

  // Start polling
  setInterval(pollSystemState, POLL_INTERVAL);

  // Audio: try WebSocket, fall back to polling
  connectAudioWs();
  setInterval(pollAudioStatus, 5000);

  // Voice: load users and start polling
  await loadVoiceUsers();
  pollVoiceStatus();
  setInterval(pollVoiceStatus, POLL_INTERVAL);
  setInterval(loadVoiceUsers, 30000);  // Refresh user list every 30s

  // Handle graph resize — re-render whichever view is active
  window.addEventListener("resize", () => {
    if (currentGraphView === "component" && currentComponentServiceId) {
      loadComponentGraph(currentComponentServiceId);
    } else if (graphData) {
      renderGraph(graphData);
    }
  });
}

init();
