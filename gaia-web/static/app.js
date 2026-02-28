/**
 * GAIA Mission Control — Dashboard JavaScript
 *
 * Alpine.js component architecture:
 *   Stores: Alpine.store('nav'), Alpine.store('header')
 *   Dashboard: chatPanel(), systemPanel(), audioWidget(), voiceWidget(), blueprintPanel()
 *   Commands: hooksPanel()
 *   Files: fileBrowser()
 *   Knowledge: knowledgeBrowser()
 *   Terminal: terminalPanel() + xterm.js
 *   D3 graph + canvas sparklines remain imperative, called from Alpine lifecycle hooks.
 */

// ── Configuration ────────────────────────────────────────────────────────────

const POLL_INTERVAL = 10_000; // system state poll every 10s
const CHAT_TIMEOUT = 300_000; // 5 minute timeout for chat
const HOOKS_POLL_INTERVAL = 15_000; // hooks status poll every 15s

// ── Shared Utilities ──────────────────────────────────────────────────────────

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatDuration(seconds) {
  if (!seconds || seconds <= 0) return '--';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function drawSparkline(canvas, values, color) {
  if (!canvas || values.length < 2) return;
  const ctx = canvas.getContext('2d');
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
  if (values.length > 0) {
    const last = values[values.length - 1];
    ctx.fillStyle = color;
    ctx.font = '9px monospace';
    ctx.fillText(`${Math.round(last)}ms`, w - 40, 10);
  }
}

// ── Alpine Stores ─────────────────────────────────────────────────────────────

document.addEventListener('alpine:init', () => {
  Alpine.store('nav', {
    currentView: 'dashboard',
    tabs: [
      { id: 'dashboard', label: 'Dashboard' },
      { id: 'hooks', label: 'Commands' },
      { id: 'files', label: 'Files' },
      { id: 'knowledge', label: 'Knowledge' },
      { id: 'audio', label: 'Audio' },
      { id: 'terminal', label: 'Terminal' },
      { id: 'consent', label: 'Consent' },
      { id: 'logs', label: 'Logs' },
    ],
    switchView(viewName) {
      this.currentView = viewName;
    },
  });

  Alpine.store('header', {
    serviceCount: '-- services',
    pendingCount: '-- pending',
    pendingWarn: false,
    discordStatus: 'discord: --',
    discordClass: 'badge',
    discordTitle: 'Discord connectivity',
  });
});

// ── Chat Panel Component ──────────────────────────────────────────────────────

function chatPanel() {
  return {
    messages: [],
    input: '',
    sending: false,
    _nextId: 0,

    init() {
      this.addMessage('Mission Control online. Type a message to talk to GAIA.', 'system');
      this._connectAutoStream();
    },

    _connectAutoStream() {
      this._autoSSE = new EventSource('/api/autonomous/stream');
      this._autoSSE.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.text) {
            this.addMessage(data.text, 'gaia-auto');
          }
        } catch { /* ignore parse errors */ }
      };
      this._autoSSE.onerror = () => {
        this._autoSSE.close();
        setTimeout(() => this._connectAutoStream(), 5000);
      };
    },

    addMessage(text, type) {
      this.messages.push({ text, type, id: this._nextId++ });
      this.$nextTick(() => {
        const el = this.$refs.messages;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    renderMessage(msg) {
      if ((msg.type === 'gaia' || msg.type === 'gaia-auto') && typeof marked !== 'undefined') {
        return marked.parse(msg.text);
      }
      return escapeHtml(msg.text);
    },

    async send() {
      const text = this.input.trim();
      if (!text || this.sending) return;
      this.addMessage(text, 'user');
      this.input = '';
      this.sending = true;
      try {
        const resp = await fetch(`/process_user_input?user_input=${encodeURIComponent(text)}`, {
          method: 'POST',
          signal: AbortSignal.timeout(CHAT_TIMEOUT),
        });
        const data = await resp.json();
        if (resp.ok) {
          this.addMessage(data.response || '(no response)', 'gaia');
        } else {
          this.addMessage(`Error ${resp.status}: ${data.detail || data.response || 'unknown'}`, 'system');
        }
      } catch (err) {
        this.addMessage(`Connection error: ${err.message}`, 'system');
      } finally {
        this.sending = false;
      }
    },
  };
}

// ── System State Panel Component ──────────────────────────────────────────────

const SERVICE_LABELS = {
  'gaia-core': 'Core',
  'gaia-web': 'Web',
  'gaia-orchestrator': 'Orchestrator',
  'gaia-prime': 'Prime',
  'gaia-mcp': 'MCP',
  'gaia-study': 'Study',
  'gaia-audio': 'Audio',
  'gaia-core-candidate': 'Core (cand.)',
  'discord': 'Discord',
};

function systemPanel() {
  return {
    services: [],
    sleepState: '--',
    sleepStateClass: '',
    gpuOwner: '--',
    graphData: null,
    currentGraphView: 'service',
    currentComponentServiceId: null,
    graphTitle: 'Service Graph',
    showGraphBack: false,
    _pollTimer: null,

    init() {
      this.poll();
      this.loadGraph();
      this._pollTimer = setInterval(() => this.poll(), POLL_INTERVAL);
      this._resizeHandler = () => this.handleResize();
      window.addEventListener('resize', this._resizeHandler);
    },

    destroy() {
      if (this._pollTimer) clearInterval(this._pollTimer);
      if (this._resizeHandler) window.removeEventListener('resize', this._resizeHandler);
    },

    serviceLabel(id) {
      return SERVICE_LABELS[id] || id;
    },

    statusClass(status) {
      if (status === 'online' || status === 'ready') return 'ok';
      if (status === 'offline' || status === 'not_started' || status === 'not_available') return 'error';
      if (status === 'connecting' || status === 'timeout') return 'warn';
      if (typeof status === 'string' && status.startsWith('error')) return 'error';
      return '';
    },

    async poll() {
      await Promise.all([this.pollServices(), this.pollSleep(), this.pollOrchestrator()]);
    },

    async pollServices() {
      try {
        const resp = await fetch('/api/system/services');
        if (resp.ok) {
          this.services = await resp.json();
          const header = Alpine.store('header');
          const discord = this.services.find(s => s.id === 'discord');
          if (discord) {
            header.discordStatus = `discord: ${discord.status}`;
            header.discordClass = `badge badge-${this.statusClass(discord.status)}`;
            if (discord.discord?.latency_ms) {
              header.discordTitle = `Discord: ${discord.discord.user || '?'} (${discord.discord.latency_ms}ms)`;
            }
          }
        }
      } catch {
        this.services = [];
      }
    },

    async pollSleep() {
      try {
        const resp = await fetch('/api/system/sleep');
        if (resp.ok) {
          const data = await resp.json();
          const state = data.state || data.sleep_state || 'unknown';
          const stateMap = { active: 'ok', asleep: 'warn', drowsy: 'warn', waking: 'ok' };
          this.sleepState = state;
          this.sleepStateClass = stateMap[state] || '';
          if (data.gpu_owner) this.gpuOwner = data.gpu_owner;
        }
      } catch { /* sleep endpoint unavailable */ }
    },

    async pollOrchestrator() {
      try {
        const resp = await fetch('/api/system/status');
        if (resp.ok) {
          const data = await resp.json();
          if (data.gpu_owner && this.gpuOwner === '--') {
            this.gpuOwner = data.gpu_owner;
          }
        }
      } catch { /* orchestrator unavailable */ }
    },

    // ── Graph ──────────────────────────────────────────────────────────

    async loadGraph() {
      try {
        const resp = await fetch('/api/blueprints/graph?include_candidates=true');
        if (!resp.ok) return;
        this.graphData = await resp.json();
        this.renderGraph(this.graphData);
        this.updateStatusBar(this.graphData);
      } catch { /* graph not available */ }
    },

    updateStatusBar(data) {
      const header = Alpine.store('header');
      header.serviceCount = `${data.blueprint_count} services`;
      header.pendingCount = `${data.pending_review_count} pending`;
      header.pendingWarn = data.pending_review_count > 0;
    },

    handleResize() {
      if (this.currentGraphView === 'component' && this.currentComponentServiceId) {
        this.loadComponentGraph(this.currentComponentServiceId);
      } else if (this.graphData) {
        this.renderGraph(this.graphData);
      }
    },

    backToServiceGraph() {
      this.currentGraphView = 'service';
      this.currentComponentServiceId = null;
      this.graphTitle = 'Service Graph';
      this.showGraphBack = false;
      if (this.graphData) this.renderGraph(this.graphData);
    },

    renderGraph(data) {
      const container = this.$refs.graphContainer;
      const svgEl = this.$refs.graphSvg;
      if (!container || !svgEl) return;
      const svg = d3.select(svgEl);
      svg.selectAll('*').remove();

      const width = container.clientWidth;
      const height = container.clientHeight;
      svg.attr('viewBox', `0 0 ${width} ${height}`);

      if (!data.nodes || data.nodes.length === 0) return;

      const g = svg.append('g').attr('class', 'graph-root');
      const zoom = d3.zoom()
        .scaleExtent([0.3, 5])
        .on('zoom', (event) => g.attr('transform', event.transform));
      svg.call(zoom);
      svg.on('dblclick.zoom', () => {
        svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
      });

      const nodeMap = {};
      const nodes = data.nodes.map(n => {
        const node = { ...n };
        nodeMap[n.id] = node;
        return node;
      });

      const links = (data.edges || [])
        .filter(e => nodeMap[e.from_service] && nodeMap[e.to_service])
        .map(e => ({
          source: e.from_service, target: e.to_service,
          transport_type: e.transport_type, status: e.status,
          description: e.description, has_fallback: e.has_fallback,
        }));

      const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(90))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide(40));

      let tooltip = d3.select('.tooltip');
      if (tooltip.empty()) {
        tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('display', 'none');
      }

      const edgeStyleMap = {
        http_rest: '', websocket: '6,3', event: '4,4',
        sse: '2,2', mcp: '1,3', direct_call: '8,4', grpc: '3,6',
      };

      const link = g.append('g').selectAll('line').data(links).join('line')
        .attr('class', d => `graph-edge ${d.status}`)
        .attr('stroke-dasharray', d => edgeStyleMap[d.transport_type] || '')
        .attr('stroke-width', d => d.has_fallback ? 2.5 : 1.5)
        .on('mouseover', (event, d) => {
          tooltip.style('display', 'block')
            .html(`<strong>${d.source.id || d.source} → ${d.target.id || d.target}</strong><br>${d.transport_type} · ${d.description || ''}`)
            .style('left', `${event.pageX + 10}px`)
            .style('top', `${event.pageY - 10}px`);
        })
        .on('mouseout', () => tooltip.style('display', 'none'));

      const nodeColor = d => d.blueprint_status === 'live' ? '#4caf50' : '#ffa726';

      const node = g.append('g').selectAll('g').data(nodes).join('g')
        .attr('class', 'graph-node')
        .call(d3.drag()
          .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
        )
        .on('click', (event, d) => {
          window.dispatchEvent(new CustomEvent('select-blueprint', { detail: { id: d.id } }));
        });

      node.append('circle')
        .attr('r', d => 10 + (d.interface_count || 0) * 1.5)
        .attr('fill', nodeColor)
        .attr('stroke', d => d.genesis ? '#e94560' : nodeColor(d))
        .attr('stroke-dasharray', d => d.genesis ? '3,3' : '')
        .attr('opacity', 0.85);

      node.append('text')
        .attr('dy', d => 10 + (d.interface_count || 0) * 1.5 + 14)
        .text(d => d.id.replace('gaia-', ''));

      sim.on('tick', () => {
        link
          .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
      });
    },

    // ── Component Graph (drill-down) ────────────────────────────────────

    async loadComponentGraph(serviceId) {
      try {
        const resp = await fetch(`/api/blueprints/${serviceId}/components`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.components || data.components.length === 0) return;
        this.currentGraphView = 'component';
        this.currentComponentServiceId = serviceId;
        this.graphTitle = `${serviceId} — Architecture`;
        this.showGraphBack = true;
        this.renderComponentGraph(data);
      } catch { /* component graph not available */ }
    },

    renderComponentGraph(data) {
      const container = this.$refs.graphContainer;
      const svgEl = this.$refs.graphSvg;
      if (!container || !svgEl) return;
      const svg = d3.select(svgEl);
      svg.selectAll('*').remove();

      const width = container.clientWidth;
      const height = container.clientHeight;
      svg.attr('viewBox', `0 0 ${width} ${height}`);

      if (!data.components || data.components.length === 0) return;

      const g = svg.append('g').attr('class', 'graph-root');
      const zoom = d3.zoom()
        .scaleExtent([0.3, 5])
        .on('zoom', (event) => g.attr('transform', event.transform));
      svg.call(zoom);
      svg.on('dblclick.zoom', () => {
        svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
      });

      const nodeMap = {};
      const nodes = data.components.map(c => {
        const node = { ...c };
        nodeMap[c.id] = node;
        return node;
      });

      const links = (data.edges || [])
        .filter(e => nodeMap[e.from_component] && nodeMap[e.to_component])
        .map(e => ({
          source: e.from_component, target: e.to_component,
          label: e.label, transport: e.transport, data_flow: e.data_flow,
        }));

      const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(110))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide(35));

      let tooltip = d3.select('.tooltip');
      if (tooltip.empty()) {
        tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('display', 'none');
      }

      const link = g.append('g').selectAll('line').data(links).join('line')
        .attr('class', 'component-edge')
        .on('mouseover', (event, d) => {
          tooltip.style('display', 'block')
            .html(`<strong>${d.label}</strong>${d.data_flow ? `<br>Data: ${d.data_flow}` : ''}`)
            .style('left', `${event.pageX + 10}px`)
            .style('top', `${event.pageY - 10}px`);
        })
        .on('mouseout', () => tooltip.style('display', 'none'));

      const edgeLabels = g.append('g').selectAll('text').data(links).join('text')
        .attr('class', 'component-edge-label')
        .text(d => d.data_flow ? d.data_flow.substring(0, 24) : '');

      const colorScale = d3.scaleOrdinal()
        .range(['#5c6bc0', '#7986cb', '#4fc3f7', '#4dd0e1', '#4db6ac', '#81c784', '#aed581']);

      const node = g.append('g').selectAll('g').data(nodes).join('g')
        .attr('class', 'component-node')
        .call(d3.drag()
          .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
        )
        .on('click', (event, d) => {
          event.stopPropagation();
          window.dispatchEvent(new CustomEvent('show-component-detail', { detail: d }));
        });

      node.append('circle')
        .attr('r', d => 8 + (d.interface_count || 0) * 1.2)
        .attr('fill', d => colorScale(d.id))
        .attr('stroke', d => d3.color(colorScale(d.id)).darker(0.5))
        .attr('stroke-width', 2)
        .attr('opacity', 0.9);

      node.append('text')
        .attr('dy', d => 8 + (d.interface_count || 0) * 1.2 + 14)
        .text(d => d.label);

      sim.on('tick', () => {
        link
          .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        edgeLabels
          .attr('x', d => (d.source.x + d.target.x) / 2)
          .attr('y', d => (d.source.y + d.target.y) / 2 - 4);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
      });
    },
  };
}

// ── Audio Widget Component ────────────────────────────────────────────────────

function audioWidget() {
  return {
    state: 'idle',
    muted: false,
    gpuMode: 'idle',
    vramUsed: 0,
    vramBudget: 5600,
    transcriptLog: [],
    eventLog: [],
    sttLatencies: [],
    ttsLatencies: [],
    _ws: null,
    _nextId: 0,
    _pollTimer: null,

    init() {
      this.connectWs();
      this._pollTimer = setInterval(() => this.pollStatus(), 5000);
    },

    destroy() {
      if (this._ws) { this._ws.close(); this._ws = null; }
      if (this._pollTimer) clearInterval(this._pollTimer);
    },

    get vramPercent() {
      return Math.min(100, (this.vramUsed / this.vramBudget) * 100);
    },

    get stateClass() {
      return `audio-state-badge state-${this.state}`;
    },

    get muteLabel() {
      return this.muted ? 'Unmute' : 'Mute';
    },

    addLogEntry(log, text, type, timestamp) {
      const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
      log.push({ text, type, time: timeStr, id: this._nextId++ });
      if (log.length > 50) log.splice(0, log.length - 50);
    },

    handleEvent(evt) {
      const { event_type, detail, latency_ms, timestamp } = evt;
      let logType = '';
      if (event_type.startsWith('stt')) logType = 'stt';
      else if (event_type.startsWith('tts')) logType = 'tts';
      else if (event_type.startsWith('gpu')) logType = 'gpu';
      else if (event_type === 'error') logType = 'error';

      const eventText = `[${event_type}] ${detail}${latency_ms > 0 ? ` (${Math.round(latency_ms)}ms)` : ''}`;
      this.addLogEntry(this.eventLog, eventText, logType, timestamp);

      if (event_type === 'stt_complete' && detail) {
        this.addLogEntry(this.transcriptLog, detail, 'stt', timestamp);
      }

      if (event_type === 'stt_start') this.state = 'transcribing';
      else if (event_type === 'tts_start') this.state = 'synthesizing';
      else if (event_type === 'stt_complete' || event_type === 'tts_complete') this.state = 'idle';
      else if (event_type === 'mute') { this.state = 'muted'; this.muted = true; }
      else if (event_type === 'unmute') { this.state = 'idle'; this.muted = false; }

      if (event_type === 'stt_complete' && latency_ms > 0) {
        this.sttLatencies.push(latency_ms);
        if (this.sttLatencies.length > 20) this.sttLatencies.shift();
        this.$nextTick(() => drawSparkline(this.$refs.sttSparkline, this.sttLatencies, '#4fc3f7'));
      }
      if (event_type === 'tts_complete' && latency_ms > 0) {
        this.ttsLatencies.push(latency_ms);
        if (this.ttsLatencies.length > 20) this.ttsLatencies.shift();
        this.$nextTick(() => drawSparkline(this.$refs.ttsSparkline, this.ttsLatencies, '#ffa726'));
      }
    },

    handleSnapshot(snap) {
      this.state = snap.state || 'idle';
      this.gpuMode = snap.gpu_mode || 'idle';
      this.vramUsed = snap.vram_used_mb || 0;
      this.muted = snap.muted || false;
      if (snap.stt_latencies) {
        this.sttLatencies = snap.stt_latencies.slice(-20);
        this.$nextTick(() => drawSparkline(this.$refs.sttSparkline, this.sttLatencies, '#4fc3f7'));
      }
      if (snap.tts_latencies) {
        this.ttsLatencies = snap.tts_latencies.slice(-20);
        this.$nextTick(() => drawSparkline(this.$refs.ttsSparkline, this.ttsLatencies, '#ffa726'));
      }
      if (snap.events) {
        for (const e of snap.events.slice(-20)) this.handleEvent(e);
      }
    },

    connectWs() {
      if (this._ws && this._ws.readyState <= 1) return;
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${location.hostname}:8081/status/ws`;
      try {
        this._ws = new WebSocket(url);
        this._ws.onopen = () => this.addLogEntry(this.eventLog, 'WebSocket connected', '', null);
        this._ws.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            if (data.state !== undefined) this.handleSnapshot(data);
            else if (data.event_type && data.event_type !== 'keepalive') this.handleEvent(data);
          } catch { /* ignore parse errors */ }
        };
        this._ws.onclose = () => { this._ws = null; setTimeout(() => this.connectWs(), 5000); };
        this._ws.onerror = () => { this._ws = null; };
      } catch {
        setTimeout(() => this.pollStatus(), 2000);
      }
    },

    async pollStatus() {
      if (this._ws && this._ws.readyState === 1) return;
      try {
        const resp = await fetch(`http://${location.hostname}:8081/status`);
        if (resp.ok) this.handleSnapshot(await resp.json());
      } catch {
        this.state = 'idle';
        this.gpuMode = 'offline';
      }
    },

    async toggleMute() {
      const action = this.muted ? 'unmute' : 'mute';
      try {
        await fetch(`http://${location.hostname}:8081/${action}`, { method: 'POST' });
      } catch {
        this.addLogEntry(this.eventLog, `Failed to ${action}`, 'error', null);
      }
    },
  };
}

// ── Voice Widget Component ────────────────────────────────────────────────────

function voiceWidget() {
  return {
    state: 'disconnected',
    channelName: '--',
    duration: '--',
    connected: false,
    users: [],
    _statusTimer: null,
    _usersTimer: null,

    init() {
      this.pollStatus();
      this.loadUsers();
      this._statusTimer = setInterval(() => this.pollStatus(), POLL_INTERVAL);
      this._usersTimer = setInterval(() => this.loadUsers(), 30000);
    },

    destroy() {
      if (this._statusTimer) clearInterval(this._statusTimer);
      if (this._usersTimer) clearInterval(this._usersTimer);
    },

    get stateClass() {
      return 'voice-state-badge state-' + this.state;
    },

    async pollStatus() {
      try {
        const resp = await fetch('/api/voice/status');
        if (!resp.ok) return;
        const data = await resp.json();
        this.state = data.state || 'disconnected';
        this.channelName = data.channel_name || '--';
        this.duration = formatDuration(data.duration_seconds);
        this.connected = data.connected || false;
      } catch {
        this.state = 'disconnected';
      }
    },

    async loadUsers() {
      try {
        const resp = await fetch('/api/voice/users');
        if (!resp.ok) return;
        this.users = await resp.json();
      } catch { /* voice API not available */ }
    },

    async toggleWhitelist(userId, currentlyWhitelisted) {
      try {
        if (currentlyWhitelisted) {
          await fetch(`/api/voice/whitelist/${userId}`, { method: 'DELETE' });
        } else {
          await fetch('/api/voice/whitelist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId }),
          });
        }
        await this.loadUsers();
      } catch { /* silently fail */ }
    },

    async disconnect() {
      try {
        await fetch('/api/voice/disconnect', { method: 'POST' });
        await this.pollStatus();
      } catch { /* silently fail */ }
    },
  };
}

// ── Audio Listener Panel Component ────────────────────────────────────────────

const AL_POLL_INTERVAL = 3000;

function audioListenerPanel() {
  return {
    status: null,
    mode: 'passive',
    saveAudio: true,
    compress: true,
    ingestSource: '',
    actionLog: [],
    _pollTimer: null,
    _nextId: 0,

    init() {
      this.$watch('$store.nav.currentView', (view) => {
        if (view === 'audio') {
          this.pollStatus();
          if (!this._pollTimer) {
            this._pollTimer = setInterval(() => this.pollStatus(), AL_POLL_INTERVAL);
          }
        } else if (this._pollTimer) {
          clearInterval(this._pollTimer);
          this._pollTimer = null;
        }
      });
    },

    destroy() {
      if (this._pollTimer) clearInterval(this._pollTimer);
    },

    get isCapturing() {
      return this.status?.capturing === true;
    },

    get isRunning() {
      return this.status?.running === true;
    },

    get statusBadge() {
      if (!this.status) return 'offline';
      if (this.status.capturing) return 'capturing';
      if (this.status.running) return 'idle';
      return 'offline';
    },

    get statusBadgeClass() {
      return 'al-badge al-badge-' + this.statusBadge;
    },

    get backend() {
      return this.status?.backend || '--';
    },

    get uptime() {
      return this.status?.uptime_seconds ? formatDuration(this.status.uptime_seconds) : '--';
    },

    get chunks() {
      return this.status?.transcript_buffer_size ?? 0;
    },

    get recordingFile() {
      if (!this.status?.recording_file) return '--';
      const parts = this.status.recording_file.split('/');
      return parts[parts.length - 1] || '--';
    },

    get transcriptLog() {
      return this.status?.transcript_log || [];
    },

    get transcriptText() {
      return this.transcriptLog.join('\n');
    },

    logEntry(action, result, isError) {
      this.actionLog.push({
        id: this._nextId++,
        time: new Date().toLocaleTimeString(),
        action,
        result,
        isError,
      });
      if (this.actionLog.length > 50) this.actionLog.splice(0, this.actionLog.length - 50);
    },

    async pollStatus() {
      try {
        const resp = await fetch('/api/audio/listener/status');
        if (resp.ok) {
          this.status = await resp.json();
        }
      } catch {
        this.status = null;
      }
    },

    async startListener() {
      this.logEntry('start', 'sending...', false);
      try {
        const resp = await fetch('/api/audio/listener/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mode: this.mode,
            save_audio: this.saveAudio,
            compress: this.compress,
          }),
        });
        const data = await resp.json();
        if (resp.ok) {
          this.logEntry('start', `ok (mode=${this.mode}, save=${this.saveAudio}, compress=${this.compress})`, false);
        } else {
          this.logEntry('start', `Error ${resp.status}: ${data.error || 'unknown'}`, true);
        }
      } catch (err) {
        this.logEntry('start', `Connection error: ${err.message}`, true);
      }
      setTimeout(() => this.pollStatus(), 1000);
    },

    async stopListener() {
      this.logEntry('stop', 'sending...', false);
      try {
        const resp = await fetch('/api/audio/listener/stop', { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
          this.logEntry('stop', 'ok', false);
        } else {
          this.logEntry('stop', `Error ${resp.status}: ${data.error || 'unknown'}`, true);
        }
      } catch (err) {
        this.logEntry('stop', `Connection error: ${err.message}`, true);
      }
      setTimeout(() => this.pollStatus(), 1000);
    },

    clearTranscript() {
      // Clear is local-only (transcript log comes from daemon status)
      if (this.status) {
        this.status.transcript_log = [];
      }
    },

    async ingestTranscript() {
      const text = this.transcriptText;
      if (!text.trim()) {
        this.logEntry('ingest', 'No transcript to send', true);
        return;
      }
      const source = this.ingestSource.trim() || 'Audio Listener';
      this.logEntry('ingest', `sending ${text.length} chars...`, false);
      try {
        const resp = await fetch('/api/audio/listener/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transcript: text, source }),
        });
        const data = await resp.json();
        if (resp.ok) {
          this.logEntry('ingest', `ok (${data.chars} chars sent to gaia-core)`, false);
        } else {
          this.logEntry('ingest', `Error ${resp.status}: ${data.error || 'unknown'}`, true);
        }
      } catch (err) {
        this.logEntry('ingest', `Connection error: ${err.message}`, true);
      }
    },
  };
}

// ── Audio Inbox Panel Component ────────────────────────────────────────────────

function audioInboxPanel() {
  return {
    status: null,
    _pollTimer: null,

    init() {
      this.refresh();
      this._pollTimer = setInterval(() => this.refresh(), 5000);
    },

    destroy() {
      if (this._pollTimer) clearInterval(this._pollTimer);
    },

    get inboxStateBadge() {
      if (!this.status) return 'offline';
      if (!this.status.running) return 'offline';
      return this.status.state || 'idle';
    },

    get queueDepth() {
      return this.status?.queue_depth ?? 0;
    },

    get filesProcessed() {
      return this.status?.files_processed ?? 0;
    },

    get inboxBusy() {
      return this.status?.state === 'processing';
    },

    async refresh() {
      try {
        const resp = await fetch('/api/audio/inbox/status');
        if (resp.ok) this.status = await resp.json();
      } catch {
        this.status = null;
      }
    },

    async processInbox() {
      try {
        await fetch('/api/audio/inbox/process', { method: 'POST' });
      } catch { /* ignore */ }
      setTimeout(() => this.refresh(), 1500);
    },
  };
}

// ── Blueprint Panel Component ─────────────────────────────────────────────────

function blueprintPanel() {
  return {
    blueprints: [],
    selectedId: null,
    detailHtml: '',

    init() {
      this.loadList();
    },

    async loadList() {
      try {
        const resp = await fetch('/api/blueprints');
        if (!resp.ok) return;
        this.blueprints = (await resp.json()).sort((a, b) => a.id.localeCompare(b.id));
      } catch {
        this.blueprints = [];
      }
    },

    async selectBlueprint(serviceId) {
      this.selectedId = serviceId;
      this.detailHtml = '<em>Loading...</em>';
      try {
        const resp = await fetch(`/api/blueprints/${serviceId}/markdown`);
        if (resp.ok) {
          const md = await resp.text();
          this.detailHtml = (typeof marked !== 'undefined') ? marked.parse(md) : escapeHtml(md);
        } else {
          this.detailHtml = `<em>Error ${resp.status}</em>`;
        }
      } catch (err) {
        this.detailHtml = `<em>Failed: ${err.message}</em>`;
      }
      // Trigger component graph in system panel
      window.dispatchEvent(new CustomEvent('load-component-graph', { detail: { id: serviceId } }));
    },

    showComponentDetail(comp) {
      const srcList = (comp.source_files || []).map(f => `<li><code>${escapeHtml(f)}</code></li>`).join('');
      const classList = (comp.key_classes || []).map(c => `<li><code>${escapeHtml(c)}</code></li>`).join('');
      const funcList = (comp.key_functions || []).map(f => `<li><code>${escapeHtml(f)}</code></li>`).join('');
      const ifaceList = [
        ...(comp.exposes_interfaces || []).map(i => `<li>exposes: <code>${escapeHtml(i)}</code></li>`),
        ...(comp.consumes_interfaces || []).map(i => `<li>consumes: <code>${escapeHtml(i)}</code></li>`),
      ].join('');

      this.detailHtml = `
        <div class="component-detail">
          <h4>${escapeHtml(comp.label)}</h4>
          <p class="comp-desc">${escapeHtml(comp.description)}</p>
          ${classList ? `<h4>Key Classes</h4><ul>${classList}</ul>` : ''}
          ${funcList ? `<h4>Key Functions</h4><ul>${funcList}</ul>` : ''}
          ${srcList ? `<h4>Source Files</h4><ul>${srcList}</ul>` : ''}
          ${ifaceList ? `<h4>Interfaces</h4><ul>${ifaceList}</ul>` : ''}
        </div>
      `;
    },
  };
}

// ── Hooks / Commands Panel Component ──────────────────────────────────────────

function hooksPanel() {
  return {
    sleepState: '--',
    sleepCycle: '--',
    sleepUptime: '--',
    gpuOwner: '--',
    gpuVram: '--',
    codexQuery: '',
    codexResults: [],
    codexSearching: false,
    codexSearched: false,
    actionLog: [],
    _pollTimer: null,
    _nextId: 0,

    init() {
      this.$watch('$store.nav.currentView', (view) => {
        if (view === 'hooks') {
          this.refreshAll();
          if (!this._pollTimer) {
            this._pollTimer = setInterval(() => this.refreshAll(), HOOKS_POLL_INTERVAL);
          }
        } else if (this._pollTimer) {
          clearInterval(this._pollTimer);
          this._pollTimer = null;
        }
      });
    },

    destroy() {
      if (this._pollTimer) clearInterval(this._pollTimer);
    },

    logEntry(action, result, isError) {
      this.actionLog.push({
        id: this._nextId++,
        time: new Date().toLocaleTimeString(),
        action,
        result,
        isError,
      });
      if (this.actionLog.length > 50) this.actionLog.splice(0, this.actionLog.length - 50);
    },

    async refreshSleep() {
      try {
        const resp = await fetch('/api/hooks/sleep/status');
        if (resp.ok) {
          const data = await resp.json();
          this.sleepState = data.state || data.sleep_state || 'unknown';
          this.sleepCycle = data.cycle_count != null ? `#${data.cycle_count}` : '--';
          const up = data.uptime_seconds || data.uptime;
          this.sleepUptime = up != null ? formatDuration(up) : '--';
        }
      } catch {
        this.sleepState = 'unreachable';
      }
    },

    async refreshGpu() {
      try {
        const resp = await fetch('/api/hooks/gpu/status');
        if (resp.ok) {
          const data = await resp.json();
          this.gpuOwner = data.owner || data.gpu_owner || 'none';
          const used = data.vram_used_mb || data.used_mb;
          const total = data.vram_total_mb || data.total_mb;
          this.gpuVram = used != null && total != null ? `${Math.round(used)} / ${Math.round(total)} MB` : '--';
        }
      } catch {
        this.gpuOwner = 'unreachable';
      }
    },

    async refreshAll() {
      await Promise.all([this.refreshSleep(), this.refreshGpu()]);
    },

    async action(endpoint) {
      this.logEntry(endpoint, 'sending...', false);
      try {
        const resp = await fetch(`/api/hooks/${endpoint}`, { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
          this.logEntry(endpoint, JSON.stringify(data).substring(0, 120), false);
        } else {
          this.logEntry(endpoint, `Error ${resp.status}: ${data.error || data.detail || 'unknown'}`, true);
        }
      } catch (err) {
        this.logEntry(endpoint, `Connection error: ${err.message}`, true);
      }
      setTimeout(() => this.refreshAll(), 1000);
    },

    async codexSearch() {
      const query = this.codexQuery.trim();
      if (!query) return;
      this.codexSearching = true;
      this.codexResults = [];
      this.logEntry('codex/search', `query: "${query}"`, false);
      try {
        const resp = await fetch('/api/hooks/codex/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: 5 }),
        });
        if (!resp.ok) {
          const err = await resp.json();
          this.logEntry('codex/search', `Error ${resp.status}`, true);
          this.codexSearching = false;
          this.codexSearched = true;
          return;
        }
        const data = await resp.json();
        const results = data.results || data;
        this.codexResults = Array.isArray(results) ? results : [];
        this.logEntry('codex/search', `${this.codexResults.length} results`, false);
      } catch (err) {
        this.logEntry('codex/search', `Connection error: ${err.message}`, true);
        this.codexResults = [];
      }
      this.codexSearching = false;
      this.codexSearched = true;
    },
  };
}

// ── File Browser Component ─────────────────────────────────────────────────

function fileBrowser() {
  return {
    roots: [],
    currentRoot: '',
    currentPath: '',
    entries: [],
    fileContent: null,
    fileName: '',
    fileExtension: '',
    filePath: '',
    loading: false,
    error: '',
    // Editor state
    editing: false,
    editContent: '',
    saving: false,

    init() {
      this.loadRoots();
    },

    get breadcrumbs() {
      if (!this.currentPath) return [];
      const parts = this.currentPath.split('/').filter(Boolean);
      return parts.map((name, i) => ({
        name,
        path: parts.slice(0, i + 1).join('/'),
      }));
    },

    get viewingFile() {
      return this.fileContent !== null;
    },

    get isWritableRoot() {
      const root = this.roots.find(r => r.name === this.currentRoot);
      return root ? root.writable : false;
    },

    async loadRoots() {
      try {
        const resp = await fetch('/api/files/roots');
        if (resp.ok) {
          this.roots = await resp.json();
          if (this.roots.length > 0 && !this.currentRoot) {
            this.currentRoot = this.roots[0].name;
            this.browse(this.currentRoot, '');
          }
        }
      } catch (err) {
        this.error = `Failed to load roots: ${err.message}`;
      }
    },

    async browse(root, path) {
      this.loading = true;
      this.error = '';
      this.fileContent = null;
      this.fileName = '';
      this.editing = false;
      this.currentRoot = root;
      this.currentPath = path;
      try {
        const resp = await fetch(`/api/files/browse/${root}/${path}`);
        if (resp.ok) {
          const data = await resp.json();
          this.entries = data.entries;
        } else {
          const err = await resp.json();
          this.error = err.detail || `Error ${resp.status}`;
          this.entries = [];
        }
      } catch (err) {
        this.error = `Connection error: ${err.message}`;
        this.entries = [];
      }
      this.loading = false;
    },

    async readFile(root, path) {
      this.loading = true;
      this.error = '';
      this.editing = false;
      this.filePath = path;
      try {
        const resp = await fetch(`/api/files/read/${root}/${path}`);
        if (resp.ok) {
          const data = await resp.json();
          this.fileContent = data.content;
          this.fileName = data.name;
          this.fileExtension = data.extension || '';
        } else {
          const err = await resp.json();
          this.error = err.detail || `Error ${resp.status}`;
        }
      } catch (err) {
        this.error = `Connection error: ${err.message}`;
      }
      this.loading = false;
    },

    clickEntry(entry) {
      const newPath = this.currentPath ? `${this.currentPath}/${entry.name}` : entry.name;
      if (entry.type === 'dir') {
        this.browse(this.currentRoot, newPath);
      } else {
        this.readFile(this.currentRoot, newPath);
      }
    },

    navigateUp() {
      const parts = this.currentPath.split('/').filter(Boolean);
      parts.pop();
      this.browse(this.currentRoot, parts.join('/'));
    },

    navigateBreadcrumb(path) {
      this.browse(this.currentRoot, path);
    },

    backToListing() {
      this.fileContent = null;
      this.fileName = '';
      this.fileExtension = '';
      this.editing = false;
    },

    changeRoot() {
      this.browse(this.currentRoot, '');
    },

    // ── Editor ────────────────────────────────────────────────────────
    enterEdit() {
      this.editContent = this.fileContent;
      this.editing = true;
    },

    cancelEdit() {
      this.editing = false;
    },

    async saveFile() {
      if (this.saving) return;
      this.saving = true;
      this.error = '';
      try {
        const resp = await fetch(`/api/files/write/${this.currentRoot}/${this.filePath}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: this.editContent }),
        });
        if (resp.ok) {
          this.fileContent = this.editContent;
          this.editing = false;
        } else {
          const err = await resp.json();
          this.error = err.detail || `Error ${resp.status}`;
        }
      } catch (err) {
        this.error = `Save failed: ${err.message}`;
      }
      this.saving = false;
    },

    formatSize(bytes) {
      if (bytes == null) return '';
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    },

    formatDate(ts) {
      if (!ts) return '';
      return new Date(ts * 1000).toLocaleString();
    },
  };
}

// ── Knowledge Browser Component ────────────────────────────────────────────

function knowledgeBrowser() {
  return {
    currentPath: '',
    entries: [],
    fileContent: null,
    renderedHtml: '',
    fileName: '',
    fileExtension: '',
    loading: false,
    error: '',

    init() {
      this.browse('');
    },

    get breadcrumbs() {
      if (!this.currentPath) return [];
      const parts = this.currentPath.split('/').filter(Boolean);
      return parts.map((name, i) => ({
        name,
        path: parts.slice(0, i + 1).join('/'),
      }));
    },

    get viewingFile() {
      return this.fileContent !== null;
    },

    get isMarkdown() {
      return this.fileExtension === '.md';
    },

    async browse(path) {
      this.loading = true;
      this.error = '';
      this.fileContent = null;
      this.renderedHtml = '';
      this.fileName = '';
      this.currentPath = path;
      try {
        const resp = await fetch(`/api/files/browse/knowledge/${path}`);
        if (resp.ok) {
          this.entries = (await resp.json()).entries;
        } else {
          const err = await resp.json();
          this.error = err.detail || `Error ${resp.status}`;
          this.entries = [];
        }
      } catch (err) {
        this.error = `Connection error: ${err.message}`;
        this.entries = [];
      }
      this.loading = false;
    },

    async readFile(path) {
      this.loading = true;
      this.error = '';
      try {
        const resp = await fetch(`/api/files/read/knowledge/${path}`);
        if (resp.ok) {
          const data = await resp.json();
          this.fileContent = data.content;
          this.fileName = data.name;
          this.fileExtension = data.extension || '';
          if (this.isMarkdown && typeof marked !== 'undefined') {
            this.renderedHtml = marked.parse(data.content);
          }
        } else {
          const err = await resp.json();
          this.error = err.detail || `Error ${resp.status}`;
        }
      } catch (err) {
        this.error = `Connection error: ${err.message}`;
      }
      this.loading = false;
    },

    clickEntry(entry) {
      const newPath = this.currentPath ? `${this.currentPath}/${entry.name}` : entry.name;
      if (entry.type === 'dir') {
        this.browse(newPath);
      } else {
        this.readFile(newPath);
      }
    },

    navigateUp() {
      const parts = this.currentPath.split('/').filter(Boolean);
      parts.pop();
      this.browse(parts.join('/'));
    },

    navigateBreadcrumb(path) {
      this.browse(path);
    },

    backToListing() {
      this.fileContent = null;
      this.renderedHtml = '';
      this.fileName = '';
      this.fileExtension = '';
    },

    formatSize(bytes) {
      if (bytes == null) return '';
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    },
  };
}

// ── Terminal Panel Component ───────────────────────────────────────────────

function terminalPanel() {
  return {
    containers: [],
    selectedContainer: '',
    connected: false,
    connecting: false,
    error: '',
    _term: null,
    _fitAddon: null,
    _ws: null,
    _resizeHandler: null,

    init() {
      this._initTerminal();
      this.loadContainers();
      this._resizeHandler = () => this._fit();
      window.addEventListener('resize', this._resizeHandler);
    },

    destroy() {
      this.disconnect();
      if (this._term) { this._term.dispose(); this._term = null; }
      if (this._resizeHandler) window.removeEventListener('resize', this._resizeHandler);
    },

    _initTerminal() {
      if (typeof Terminal === 'undefined') return;
      this._term = new Terminal({
        cursorBlink: true,
        fontSize: 13,
        fontFamily: '"Fira Code", "JetBrains Mono", "Cascadia Code", monospace',
        theme: {
          background: '#0d1117',
          foreground: '#c9d1d9',
          cursor: '#4fc3f7',
          selectionBackground: 'rgba(79,195,247,0.3)',
          black: '#484f58', red: '#ff7b72', green: '#7ee787', yellow: '#d29922',
          blue: '#79c0ff', magenta: '#d2a8ff', cyan: '#56d4dd', white: '#c9d1d9',
        },
      });

      if (typeof FitAddon !== 'undefined') {
        this._fitAddon = new FitAddon.FitAddon();
        this._term.loadAddon(this._fitAddon);
      }
      if (typeof WebLinksAddon !== 'undefined') {
        this._term.loadAddon(new WebLinksAddon.WebLinksAddon());
      }

      this.$nextTick(() => {
        const el = this.$refs.termContainer;
        if (el) {
          this._term.open(el);
          this._fit();
          this._term.writeln('\x1b[36mGAIA Terminal\x1b[0m — Select a container and click Connect.');
        }
      });
    },

    _fit() {
      if (this._fitAddon && this._term) {
        try { this._fitAddon.fit(); } catch { /* ignore */ }
      }
    },

    async loadContainers() {
      try {
        const resp = await fetch('/api/terminal/containers');
        if (resp.ok) {
          this.containers = await resp.json();
        } else {
          const err = await resp.json();
          this.error = err.error || `Error ${resp.status}`;
        }
      } catch (err) {
        this.error = `Failed to load containers: ${err.message}`;
      }
    },

    connect() {
      if (!this.selectedContainer || this.connecting || this.connected) return;
      this.connecting = true;
      this.error = '';

      if (this._term) {
        this._term.clear();
        this._term.writeln(`\x1b[33mConnecting to ${this.selectedContainer}...\x1b[0m`);
      }

      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${location.host}/api/terminal/ws?container=${encodeURIComponent(this.selectedContainer)}`;

      try {
        this._ws = new WebSocket(url);
        this._ws.binaryType = 'arraybuffer';

        this._ws.onopen = () => {
          this.connecting = false;
          this.connected = true;
          this._fit();
          // Forward terminal input to WebSocket
          this._term.onData((data) => {
            if (this._ws && this._ws.readyState === 1) {
              this._ws.send(new TextEncoder().encode(data));
            }
          });
        };

        this._ws.onmessage = (event) => {
          if (event.data instanceof ArrayBuffer) {
            this._term.write(new Uint8Array(event.data));
          } else {
            // JSON error messages
            try {
              const msg = JSON.parse(event.data);
              if (msg.error) {
                this._term.writeln(`\x1b[31mError: ${msg.error}\x1b[0m`);
                this.error = msg.error;
              }
            } catch {
              this._term.write(event.data);
            }
          }
        };

        this._ws.onclose = () => {
          this.connected = false;
          this.connecting = false;
          if (this._term) this._term.writeln('\r\n\x1b[33mDisconnected.\x1b[0m');
          this._ws = null;
        };

        this._ws.onerror = () => {
          this.error = 'WebSocket connection failed';
          this.connecting = false;
          this._ws = null;
        };
      } catch (err) {
        this.error = `Connection error: ${err.message}`;
        this.connecting = false;
      }
    },

    disconnect() {
      if (this._ws) {
        this._ws.close();
        this._ws = null;
      }
      this.connected = false;
    },
  };
}

/* ── Generation Stream Panel ────────────────────────────────────── */

function generationPanel() {
  return {
    entries: [],          // ring buffer of display entries
    maxEntries: 500,
    currentGenId: null,
    scrollLocked: true,   // auto-scroll to bottom
    roleFilter: '',       // '' = all, 'prime', 'lite'
    connected: false,
    _es: null,            // EventSource
    _currentBuf: '',      // accumulator for in-progress generation
    _inThink: false,      // inside <think> block

    init() {
      this._connect();
    },

    destroy() {
      if (this._es) { this._es.close(); this._es = null; }
    },

    _connect() {
      const url = '/api/generation/stream' + (this.roleFilter ? `?role=${this.roleFilter}` : '');
      if (this._es) this._es.close();
      this._es = new EventSource(url);
      this.connected = true;

      this._es.onmessage = (ev) => {
        try {
          const rec = JSON.parse(ev.data);
          this._handleRecord(rec);
        } catch (_) {}
      };

      this._es.onerror = () => {
        this.connected = false;
        // reconnect after 2s
        setTimeout(() => this._connect(), 2000);
      };
    },

    _handleRecord(rec) {
      if (rec.event === 'gen_start') {
        // Flush any pending buffer
        this._flushBuffer();
        this.currentGenId = rec.gen_id;
        this._currentBuf = '';
        this._inThink = false;
        this._push({
          type: 'header',
          genId: rec.gen_id,
          model: rec.model || '?',
          role: rec.role || '?',
          phase: rec.phase || '',
          ts: rec.ts,
        });
      } else if (rec.event === 'token') {
        this._currentBuf += (rec.t || '');
        // Re-render the current generation entry
        this._updateCurrentEntry();
      } else if (rec.event === 'gen_end') {
        this._flushBuffer();
        this._push({
          type: 'summary',
          genId: rec.gen_id,
          tokens: rec.tokens || 0,
          elapsed: rec.elapsed_ms || 0,
          ts: rec.ts,
        });
        this.currentGenId = null;
      }
    },

    _updateCurrentEntry() {
      // Find or create the 'stream' entry for the current gen
      const existing = this.entries.findIndex(e => e.type === 'stream' && e.genId === this.currentGenId);
      const parsed = this._parseThinkTags(this._currentBuf);
      if (existing >= 0) {
        this.entries[existing] = { ...this.entries[existing], html: parsed };
      } else {
        this._push({ type: 'stream', genId: this.currentGenId, html: parsed });
      }
      this._autoScroll();
    },

    _flushBuffer() {
      // nothing extra needed — buffer is rendered live
      this._currentBuf = '';
      this._inThink = false;
    },

    _parseThinkTags(text) {
      // Render <think> blocks with special styling
      return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/&lt;think&gt;/g, '<span class="gen-think">&lt;think&gt;')
        .replace(/&lt;\/think&gt;/g, '&lt;/think&gt;</span>');
    },

    _push(entry) {
      this.entries.push(entry);
      if (this.entries.length > this.maxEntries) {
        this.entries.splice(0, this.entries.length - this.maxEntries);
      }
      this._autoScroll();
    },

    _autoScroll() {
      if (!this.scrollLocked) return;
      this.$nextTick(() => {
        const el = this.$refs.genOutput;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    clear() { this.entries = []; },

    setFilter(role) {
      this.roleFilter = role;
      this.entries = [];
      this._connect();
    },

    handleScroll() {
      const el = this.$refs.genOutput;
      if (!el) return;
      // If user scrolled up, pause auto-scroll; if at bottom, resume
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      this.scrollLocked = atBottom;
    },
  };
}

/* ── Service Logs Panel (plain text log tail) ──────────────────── */

function serviceLogsPanel() {
  return {
    lines: [],
    maxLines: 1000,
    scrollLocked: true,
    connected: false,
    service: 'core',
    _es: null,
    // Level filter state
    mode: 'live',                  // 'live' (SSE stream) or 'filtered' (per-level ring files)
    levelFilters: { error: true, warning: true, info: false, debug: false },
    _filterLoading: false,

    init() {
      this.$nextTick(() => this._connect());
    },

    destroy() {
      if (this._es) { this._es.close(); this._es = null; }
    },

    switchService(svc) {
      if (svc === this.service && this.connected) return;
      this.service = svc;
      this.lines = [];
      if (this.mode === 'filtered') {
        this._fetchFiltered();
      } else {
        this._connect();
      }
    },

    toggleLevel(level) {
      this.levelFilters[level] = !this.levelFilters[level];
      // If any filter is active, switch to filtered mode
      const anyActive = Object.values(this.levelFilters).some(v => v);
      if (!anyActive) {
        // Reset to live if nothing selected
        this.levelFilters = { error: true, warning: true, info: false, debug: false };
      }
      if (this.mode === 'filtered') {
        this._fetchFiltered();
      }
    },

    setMode(m) {
      this.mode = m;
      this.lines = [];
      if (m === 'live') {
        this._connect();
      } else {
        if (this._es) { this._es.close(); this._es = null; this.connected = false; }
        this._fetchFiltered();
      }
    },

    async _fetchFiltered() {
      this._filterLoading = true;
      const active = Object.entries(this.levelFilters)
        .filter(([, v]) => v).map(([k]) => k);
      if (!active.length) { this.lines = []; this._filterLoading = false; return; }
      try {
        const resp = await fetch(
          `/api/logs/levels?service=${this.service}&levels=${active.join(',')}&limit=1000`
        );
        const data = await resp.json();
        this.lines = (data.lines || []).map(l => ({ text: l.text, level: l.level }));
        this._autoScroll();
      } catch (e) {
        console.error('Failed to fetch filtered logs:', e);
      }
      this._filterLoading = false;
    },

    _connect() {
      if (this._es) this._es.close();
      this._es = new EventSource(`/api/logs/stream?service=${this.service}`);
      this.connected = true;

      this._es.onmessage = (ev) => {
        try {
          const rec = JSON.parse(ev.data);
          this._pushLine(rec);
        } catch (_) {}
      };

      this._es.onerror = () => {
        this.connected = false;
        setTimeout(() => this._connect(), 3000);
      };
    },

    _pushLine(rec) {
      this.lines.push({
        text: rec.text || '',
        level: rec.level || 'info',
      });
      if (this.lines.length > this.maxLines) {
        this.lines.splice(0, this.lines.length - this.maxLines);
      }
      this._autoScroll();
    },

    _autoScroll() {
      if (!this.scrollLocked) return;
      this.$nextTick(() => {
        const el = this.$refs.logsOutput;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    clear() { this.lines = []; },

    handleScroll() {
      const el = this.$refs.logsOutput;
      if (!el) return;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      this.scrollLocked = atBottom;
    },
  };
}

// ── Consent Panel Component ────────────────────────────────────────────────

const CT_POLL_INTERVAL = 30_000;

function consentPanel() {
  return {
    library: null,
    tiers: [],
    selectedTier: '',
    currentTests: [],
    results: [],
    stats: null,
    selectedResult: null,
    runningTest: null,
    runningTier: false,
    actionLog: [],
    _pollTimer: null,
    _nextId: 0,

    init() {
      this.$watch('$store.nav.currentView', (view) => {
        if (view === 'consent') {
          this.loadLibrary();
          this.loadResults();
          this.loadStats();
          if (!this._pollTimer) {
            this._pollTimer = setInterval(() => {
              this.loadResults();
              this.loadStats();
            }, CT_POLL_INTERVAL);
          }
        } else if (this._pollTimer) {
          clearInterval(this._pollTimer);
          this._pollTimer = null;
        }
      });
    },

    destroy() {
      if (this._pollTimer) clearInterval(this._pollTimer);
    },

    logEntry(action, result, isError) {
      this.actionLog.push({
        id: this._nextId++,
        time: new Date().toLocaleTimeString(),
        action,
        result,
        isError,
      });
      if (this.actionLog.length > 50) this.actionLog.splice(0, this.actionLog.length - 50);
    },

    // ── Library ────────────────────────────────────────────────────

    async loadLibrary() {
      try {
        const resp = await fetch('/api/consent/library');
        if (resp.ok) {
          this.library = await resp.json();
          this.tiers = Object.entries(this.library.tiers || {}).map(([key, val]) => ({
            key,
            label: val.label,
            description: val.description,
            tests: val.tests || [],
          }));
          if (this.tiers.length > 0 && !this.selectedTier) {
            this.selectedTier = this.tiers[0].key;
          }
          this.updateCurrentTests();
        }
      } catch (err) {
        this.logEntry('loadLibrary', `Error: ${err.message}`, true);
      }
    },

    updateCurrentTests() {
      const tier = this.tiers.find(t => t.key === this.selectedTier);
      this.currentTests = tier ? tier.tests : [];
    },

    selectTier(tierKey) {
      this.selectedTier = tierKey;
      this.updateCurrentTests();
    },

    tierLabel(tierKey) {
      const tier = this.tiers.find(t => t.key === tierKey);
      return tier ? tier.label : tierKey;
    },

    // ── Run Tests ─────────────────────────────────────────────────

    async runTest(testId) {
      this.runningTest = testId;
      this.logEntry('run', `${testId} — sending...`, false);
      try {
        const resp = await fetch('/api/consent/test/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ test_id: testId }),
        });
        const data = await resp.json();
        if (resp.ok) {
          this.selectedResult = data;
          this.logEntry('run', `${testId} — ${data.classification}`, false);
          this.loadResults();
          this.loadStats();
        } else {
          this.logEntry('run', `${testId} — Error: ${data.error || resp.status}`, true);
        }
      } catch (err) {
        this.logEntry('run', `${testId} — Connection error: ${err.message}`, true);
      }
      this.runningTest = null;
    },

    async runAllTier() {
      this.runningTier = true;
      this.logEntry('run-tier', `${this.selectedTier} — starting...`, false);
      try {
        const resp = await fetch('/api/consent/test/run-tier', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tier: this.selectedTier }),
        });
        const data = await resp.json();
        if (resp.ok) {
          const n = data.results?.length || 0;
          this.logEntry('run-tier', `${this.selectedTier} — ${n} tests complete`, false);
          if (data.results?.length > 0) {
            this.selectedResult = data.results[data.results.length - 1];
          }
          this.loadResults();
          this.loadStats();
        } else {
          this.logEntry('run-tier', `Error: ${data.error || resp.status}`, true);
        }
      } catch (err) {
        this.logEntry('run-tier', `Connection error: ${err.message}`, true);
      }
      this.runningTier = false;
    },

    // ── Results ───────────────────────────────────────────────────

    async loadResults() {
      try {
        const resp = await fetch('/api/consent/results?limit=25');
        if (resp.ok) {
          const data = await resp.json();
          this.results = data.results || [];
        }
      } catch { /* silent */ }
    },

    async loadStats() {
      try {
        const resp = await fetch('/api/consent/results/stats');
        if (resp.ok) {
          this.stats = await resp.json();
        }
      } catch { /* silent */ }
    },

    async viewResult(resultId) {
      try {
        const resp = await fetch(`/api/consent/results/${encodeURIComponent(resultId)}`);
        if (resp.ok) {
          this.selectedResult = await resp.json();
        }
      } catch (err) {
        this.logEntry('view', `Error: ${err.message}`, true);
      }
    },

    // ── Acknowledgment (Phase 2) ──────────────────────────────────

    async acknowledgeResult() {
      if (!this.selectedResult) return;
      const resultId = this.selectedResult.result_id;
      this.logEntry('acknowledge', `${resultId} — sending...`, false);
      try {
        const resp = await fetch('/api/consent/acknowledge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ result_id: resultId }),
        });
        const data = await resp.json();
        if (resp.ok) {
          this.selectedResult = data;
          this.logEntry('acknowledge', `${resultId} — acknowledged`, false);
          this.loadResults();
        } else {
          this.logEntry('acknowledge', `Error: ${data.error || resp.status}`, true);
        }
      } catch (err) {
        this.logEntry('acknowledge', `Connection error: ${err.message}`, true);
      }
    },

    // ── Display helpers ───────────────────────────────────────────

    classificationBadge(cls) {
      const map = {
        engage: 'ct-badge-engage',
        refuse_with_reasoning: 'ct-badge-reasoning',
        refuse: 'ct-badge-refuse',
        refuse_and_flag: 'ct-badge-flag',
        error: 'ct-badge-error',
      };
      return 'ct-badge ' + (map[cls] || 'ct-badge-unknown');
    },

    classificationLabel(cls) {
      const map = {
        engage: 'engaged',
        refuse_with_reasoning: 'refuse + reasoning',
        refuse: 'refuse',
        refuse_and_flag: 'refuse + flag',
        error: 'error',
      };
      return map[cls] || cls;
    },

    get statsSummary() {
      if (!this.stats) return '--';
      return `${this.stats.total} tests`;
    },

    get statsEngaged() {
      if (!this.stats) return 0;
      return this.stats.reasoning_quality?.engaged || 0;
    },

    get statsRuleCiting() {
      if (!this.stats) return 0;
      return this.stats.reasoning_quality?.rule_citing || 0;
    },

    truncate(text, len) {
      if (!text) return '';
      return text.length > len ? text.substring(0, len) + '...' : text;
    },

    formatTime(ts) {
      if (!ts) return '--';
      try {
        return new Date(ts).toLocaleTimeString();
      } catch { return ts; }
    },
  };
}
