/**
 * GAIA Mission Control — Dashboard JavaScript
 *
 * Three modules: Chat, System State, Blueprints
 * No build step — vanilla JS + D3 from CDN.
 */

// ── Configuration ────────────────────────────────────────────────────────────

const POLL_INTERVAL = 10_000; // system state poll every 10s
const CHAT_TIMEOUT = 300_000; // 5 minute timeout for chat

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

const sleepStateEl = document.getElementById("sleep-state");
const gpuOwnerEl = document.getElementById("gpu-owner");
const orchStatusEl = document.getElementById("orch-status");
const coreStatusEl = document.getElementById("core-status");

function setStateValue(el, text, status) {
  el.textContent = text;
  el.className = `state-value ${status}`;
}

async function pollSystemState() {
  // Sleep / core status
  try {
    const resp = await fetch("/api/system/sleep");
    if (resp.ok) {
      const data = await resp.json();
      const state = data.state || data.sleep_state || "unknown";
      const stateMap = { active: "ok", asleep: "warn", drowsy: "warn", waking: "ok" };
      setStateValue(sleepStateEl, state, stateMap[state] || "");
      setStateValue(coreStatusEl, "online", "ok");

      if (data.gpu_owner) {
        setStateValue(gpuOwnerEl, data.gpu_owner, "ok");
      }
    } else {
      setStateValue(coreStatusEl, "error", "error");
      setStateValue(sleepStateEl, "--", "");
    }
  } catch {
    setStateValue(coreStatusEl, "offline", "error");
    setStateValue(sleepStateEl, "--", "");
  }

  // Orchestrator status
  try {
    const resp = await fetch("/api/system/status");
    if (resp.ok) {
      setStateValue(orchStatusEl, "online", "ok");
      const data = await resp.json();
      if (data.gpu_owner) {
        setStateValue(gpuOwnerEl, data.gpu_owner, "ok");
      }
    } else {
      setStateValue(orchStatusEl, `error (${resp.status})`, "error");
    }
  } catch {
    setStateValue(orchStatusEl, "offline", "error");
  }
}

// ── Graph Visualization ──────────────────────────────────────────────────────

let graphData = null;

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

  const link = svg.append("g")
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

  const node = svg.append("g")
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

  try {
    const resp = await fetch(`/api/blueprints/${serviceId}/markdown`);
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
  } catch (err) {
    blueprintDetailEl.innerHTML = `<em>Failed: ${err.message}</em>`;
  }
}

// ── Initialization ───────────────────────────────────────────────────────────

async function init() {
  addChatMessage("Mission Control online. Type a message to talk to GAIA.", "system");

  // Load initial data
  await Promise.all([pollSystemState(), loadGraph(), loadBlueprintList()]);

  // Start polling
  setInterval(pollSystemState, POLL_INTERVAL);

  // Handle graph resize
  window.addEventListener("resize", () => {
    if (graphData) renderGraph(graphData);
  });
}

init();
