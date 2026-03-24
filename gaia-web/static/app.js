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

function timeAgo(isoString) {
  if (!isoString) return '';
  const diff = Date.now() - new Date(isoString).getTime();
  const secs = Math.floor(diff / 1000);
  if (secs < 60) return 'just now';
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function _genHex8() {
  return Math.random().toString(16).slice(2, 10);
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
    currentView: localStorage.getItem('gaia_active_tab') || 'chat',
    tabs: [
      { id: 'chat', label: 'Chat' },
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
      localStorage.setItem('gaia_active_tab', viewName);
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

  // ── Shared Chat Store (Multi-Conversation) ────────────────────────────
  // Single source of truth for all conversations, shared between Chat tab and Dashboard panel.
  const CONV_STORAGE_KEY = 'gaia_conversations';
  const OLD_CHAT_STORAGE_KEY = 'gaia_chat_history';
  const CHAT_MAX_STORED = 200;

  const chatStore = {
    conversations: {},
    activeId: null,
    sending: false,
    sidebarOpen: false,
    sidebarPinned: false,
    _nextMsgId: 0,
    _sseConnected: false,
    _autoSSE: null,

    // Context pool
    contextPool: { summaries: [], lastUpdated: null },

    // ── Getters ──

    getActive() {
      return this.conversations[this.activeId] || null;
    },

    getMessages() {
      const active = this.getActive();
      return active ? active.messages : [];
    },

    getSortedList() {
      return Object.values(this.conversations).sort((a, b) => {
        return new Date(b.updatedAt) - new Date(a.updatedAt);
      });
    },

    // ── Conversation CRUD ──

    createConversation(title) {
      const id = `conv_${_genHex8()}`;
      const sessionId = `web_${_genHex8()}`;
      const now = new Date().toISOString();
      this.conversations[id] = {
        id,
        sessionId,
        title: title || 'New conversation',
        messages: [],
        contextMode: 'pooled',
        joinedFrom: [],
        createdAt: now,
        updatedAt: now,
      };
      this.activeId = id;
      this._save();
      return id;
    },

    switchConversation(id) {
      if (this.conversations[id]) {
        this.activeId = id;
        this._save();
      }
    },

    deleteConversation(id) {
      delete this.conversations[id];
      const remaining = this.getSortedList();
      if (remaining.length > 0) {
        this.activeId = remaining[0].id;
      } else {
        this.createConversation('General');
      }
      this._save();
    },

    renameConversation(id, title) {
      const conv = this.conversations[id];
      if (conv) {
        conv.title = title;
        conv.updatedAt = new Date().toISOString();
        this._save();
      }
    },

    // ── Messages (operate on active conversation) ──

    addMessage(text, type) {
      const active = this.getActive();
      if (!active) return;
      active.messages.push({ text, type, id: this._nextMsgId++, ts: new Date().toISOString() });
      active.updatedAt = new Date().toISOString();
      // Trim to max stored
      if (active.messages.length > CHAT_MAX_STORED) {
        active.messages = active.messages.slice(-CHAT_MAX_STORED);
      }
      this._save();
    },

    clearHistory() {
      const active = this.getActive();
      if (!active) return;
      active.messages = [];
      this._nextMsgId = 0;
      active.updatedAt = new Date().toISOString();
      this._save();
      this.addMessage('Chat history cleared.', 'system');
    },

    // ── Session plumbing ──

    getSessionId() {
      const active = this.getActive();
      return active ? active.sessionId : `web_${_genHex8()}`;
    },

    // ── Context ──

    setContextMode(id, mode) {
      const conv = this.conversations[id];
      if (conv) {
        conv.contextMode = mode;
        this._save();
      }
    },

    joinContext(targetId, srcId) {
      const target = this.conversations[targetId];
      const src = this.conversations[srcId];
      if (!target || !src) return;
      if (!target.joinedFrom.includes(srcId)) {
        target.joinedFrom.push(srcId);
      }
      // Add a system message noting the join
      const prevActive = this.activeId;
      this.activeId = targetId;
      this.addMessage(`Context joined from "${src.title}"`, 'system');
      this.activeId = prevActive;
    },

    addToPool(convId) {
      const conv = this.conversations[convId];
      if (!conv || conv.messages.length === 0) return;
      // Generate a simple summary from recent messages
      const userMsgs = conv.messages.filter(m => m.type === 'user').map(m => m.text);
      const summary = userMsgs.slice(-5).join(' | ').slice(0, 200) || '(empty conversation)';
      this.contextPool.summaries.push({
        conversationId: convId,
        title: conv.title,
        summary,
        addedAt: new Date().toISOString(),
      });
      // Prune to 20 max
      if (this.contextPool.summaries.length > 20) {
        this.contextPool.summaries = this.contextPool.summaries.slice(-20);
      }
      this.contextPool.lastUpdated = new Date().toISOString();
      this._save();
      this.addMessage('Conversation added to context pool.', 'system');
    },

    // ── Auto-title ──

    autoTitle(convId, firstMsg) {
      const conv = this.conversations[convId];
      if (!conv) return;
      let title = firstMsg.trim();
      if (title.length > 40) {
        title = title.slice(0, 37) + '...';
      }
      conv.title = title;
      conv.updatedAt = new Date().toISOString();
      this._save();
    },

    // ── Persistence ──

    _save() {
      try {
        const data = {
          activeId: this.activeId,
          conversations: this.conversations,
          contextPool: this.contextPool,
        };
        localStorage.setItem(CONV_STORAGE_KEY, JSON.stringify(data));
      } catch { /* storage full or unavailable */ }
    },

    _load() {
      try {
        const raw = localStorage.getItem(CONV_STORAGE_KEY);
        if (raw) {
          const data = JSON.parse(raw);
          if (data.conversations && typeof data.conversations === 'object') {
            this.conversations = data.conversations;
            this.activeId = data.activeId || null;
            if (data.contextPool) this.contextPool = data.contextPool;
            // Recalculate _nextMsgId from all messages
            let maxId = 0;
            for (const conv of Object.values(this.conversations)) {
              for (const msg of (conv.messages || [])) {
                if ((msg.id ?? 0) > maxId) maxId = msg.id;
              }
            }
            this._nextMsgId = maxId + 1;
            return true;
          }
        }
      } catch { /* ignore corrupt storage */ }
      return false;
    },

    _migrate() {
      try {
        const raw = localStorage.getItem(OLD_CHAT_STORAGE_KEY);
        if (!raw) return false;
        const saved = JSON.parse(raw);
        if (!Array.isArray(saved) || saved.length === 0) return false;
        const id = `conv_${_genHex8()}`;
        const sessionId = `web_${_genHex8()}`;
        const now = new Date().toISOString();
        this.conversations[id] = {
          id,
          sessionId,
          title: 'General',
          messages: saved,
          contextMode: 'pooled',
          joinedFrom: [],
          createdAt: now,
          updatedAt: now,
        };
        this.activeId = id;
        this._nextMsgId = Math.max(...saved.map(m => m.id ?? 0)) + 1;
        // Remove old key
        localStorage.removeItem(OLD_CHAT_STORAGE_KEY);
        this._save();
        return true;
      } catch { return false; }
    },

    // ── SSE (messages go to active conversation) ──

    connectAutoStream() {
      if (this._sseConnected) return;
      this._sseConnected = true;
      const connect = () => {
        this._autoSSE = new EventSource('/api/autonomous/stream');
        this._autoSSE.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            if (data.text) this.addMessage(data.text, 'gaia-auto');
          } catch { /* ignore */ }
        };
        this._autoSSE.onerror = () => {
          this._autoSSE.close();
          this._sseConnected = false;
          setTimeout(() => { this._sseConnected = false; this.connectAutoStream(); }, 5000);
        };
      };
      connect();
    },

    async fetchGreeting() {
      try {
        const resp = await fetch('/api/system/sleep');
        if (resp.ok) {
          const data = await resp.json();
          const state = data.state || data.sleep_state || 'unknown';
          if (state === 'ASLEEP' || state === 'DROWSY') {
            this.addMessage("I'm currently resting. Send a message to wake me up.", 'gaia');
          } else {
            this.addMessage("I'm here. What would you like to talk about?", 'gaia');
          }
        }
      } catch { /* Core unreachable */ }
    },
  };

  // Initialize: load existing data, migrate old format, or create first conversation
  const loaded = chatStore._load();
  if (!loaded) {
    const migrated = chatStore._migrate();
    if (!migrated) {
      chatStore.createConversation('General');
      chatStore.addMessage('Mission Control online.', 'system');
      chatStore.fetchGreeting();
    }
  }
  // Ensure there's an active conversation
  if (!chatStore.activeId || !chatStore.conversations[chatStore.activeId]) {
    const list = Object.keys(chatStore.conversations);
    if (list.length > 0) {
      chatStore.activeId = list[0];
    } else {
      chatStore.createConversation('General');
      chatStore.addMessage('Mission Control online.', 'system');
      chatStore.fetchGreeting();
    }
  }
  chatStore.connectAutoStream();

  Alpine.store('chat', chatStore);
});

// ── Chat Panel Component ──────────────────────────────────────────────────────

function chatPanel() {
  return {
    input: '',

    init() {
      // Scroll to bottom on init
      this.$nextTick(() => {
        const el = this.$refs.messages;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    clearHistory() { Alpine.store('chat').clearHistory(); },

    formatTimestamp(ts) {
      if (!ts) return '';
      return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    renderMessage(msg) {
      if ((msg.type === 'gaia' || msg.type === 'gaia-auto') && typeof marked !== 'undefined') {
        return marked.parse(msg.text);
      }
      return escapeHtml(msg.text);
    },

    async send() {
      const store = Alpine.store('chat');
      const text = this.input.trim();
      if (!text || store.sending) return;

      // Auto-title on first user message in a new conversation
      const active = store.getActive();
      const isFirstUserMsg = active && active.messages.filter(m => m.type === 'user').length === 0;

      store.addMessage(text, 'user');
      this.input = '';
      store.sending = true;

      if (isFirstUserMsg && active) {
        store.autoTitle(active.id, text);
      }

      try {
        const resp = await fetch(`/process_user_input?user_input=${encodeURIComponent(text)}`, {
          method: 'POST',
          headers: { 'X-Session-ID': store.getSessionId() },
          signal: AbortSignal.timeout(CHAT_TIMEOUT),
        });
        if (!resp.ok) {
          try {
            const errData = await resp.json();
            store.addMessage(`Error ${resp.status}: ${errData.detail || errData.response || 'unknown'}`, 'system');
          } catch {
            store.addMessage(`Error ${resp.status}`, 'system');
          }
          return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let responseText = '';

        const messages = store.getMessages();
        const streamId = store._nextMsgId++;
        messages.push({ text: '', type: 'gaia streaming', id: streamId, ts: new Date().toISOString() });
        store._save();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const chunk = JSON.parse(line);
              if (chunk.type === 'token') {
                responseText += chunk.value || '';
                const msg = messages.find(m => m.id === streamId);
                if (msg) msg.text = responseText;
                this.$nextTick(() => {
                  const el = this.$refs.messages;
                  if (el) el.scrollTop = el.scrollHeight;
                });
              } else if (chunk.type === 'final') {
                responseText = chunk.value || responseText;
              } else if (chunk.type === 'error') {
                const msg = messages.find(m => m.id === streamId);
                if (msg) { msg.text = `Error: ${chunk.value || 'unknown'}`; msg.type = 'system'; }
                return;
              }
            } catch { /* skip */ }
          }
        }

        const finalMsg = messages.find(m => m.id === streamId);
        if (finalMsg) {
          finalMsg.type = 'gaia';
          finalMsg.text = responseText.trim() || '(no response)';
        }
        store._save();
      } catch (err) {
        Alpine.store('chat').addMessage(`Connection error: ${err.message}`, 'system');
      } finally {
        Alpine.store('chat').sending = false;
      }
    },
  };
}

// ── Neural Mind Map Component ─────────────────────────────────────────────────
// Unified brain — all three cognitive tiers mapped to anatomical regions
// Sagittal side-view: Frontal/Prime (left/top), Brain stem/Nano (bottom-right)

// Brain visualization uses a real anatomical SVG (Human-brain.svg from Wikimedia)
// loaded as an <image> element in the SVG, with our neural fibers overlaid.
// Original SVG viewBox: 0 0 1024 732 — we scale to fit our 280x400 viewport.
// Brain regions are positioned to match the anatomical illustration.
const BRAIN_SVG_URL = '/static/brain.svg';
const BRAIN_SVG_VIEWBOX = { w: 1024, h: 732 };  // original dimensions

// Legacy path data kept as fallback (not rendered when SVG image loads)
const BRAIN_OUTLINE =
  // Frontal pole (bottom-left, rounded)
  'M 32,215 C 28,205 25,192 24,178' +
  // Prefrontal cortex — ascending, slight orbital bulge
  ' C 23,162 25,145 30,128' +
  // Inferior frontal gyrus — small bump
  ' C 33,118 37,108 42,98' +
  // Middle frontal gyrus — gentle undulation ascending
  ' C 48,86 56,74 65,63' +
  // Superior frontal gyrus — reaching the dome
  ' C 74,53 85,44 97,37 C 108,31 120,28 132,26' +
  // Precentral gyrus — bump before central sulcus
  ' C 142,25 150,24 157,26 C 162,28 165,32 167,28' +
  // CENTRAL SULCUS — distinctive notch (key landmark)
  ' C 169,24 172,22 175,25' +
  // Postcentral gyrus — bump after central sulcus
  ' C 178,28 181,32 183,28 C 186,25 190,26 194,28' +
  // Superior parietal — continuing the dome
  ' C 202,32 210,38 217,47' +
  // Parieto-occipital transition — curving back and down
  ' C 224,56 230,68 234,82' +
  // Occipital lobe — narrowing toward back
  ' C 238,98 241,116 243,135 C 244,152 244,168 242,183' +
  // Occipital pole — slight point
  ' C 240,195 236,206 230,215' +
  // Pre-occipital notch — dip before cerebellum junction
  ' C 225,222 218,228 210,232' +
  // Inferior temporal/occipital boundary
  ' C 200,237 188,240 175,242' +
  // Temporal lobe underside — long flat curve forward
  ' C 158,244 140,244 122,242 C 104,240 86,236 72,230' +
  // Temporal pole — rounded front of temporal lobe
  ' C 58,224 46,218 38,212' +
  // Close — connect temporal pole back to frontal pole
  ' C 34,214 32,215 32,215 Z';

// Sylvian fissure — separates temporal from frontal/parietal
// Runs from anterior to posterior, angling upward
const BRAIN_SYLVIAN =
  'M 52,212 C 65,200 82,192 100,188' +
  ' C 120,184 142,184 162,188' +
  ' C 178,192 192,200 205,212';

// Central sulcus — deep groove from dome down toward Sylvian fissure
const BRAIN_CENTRAL_SULCUS =
  'M 171,25 C 169,40 166,58 162,78' +
  ' C 158,98 154,120 150,142' +
  ' C 147,158 144,172 142,185';

// Additional gyri texture — subtle cortical folds
const BRAIN_GYRI = [
  // Precentral sulcus
  'M 148,30 C 145,50 140,72 136,95 C 133,112 130,130 128,148',
  // Superior frontal sulcus
  'M 90,40 C 95,55 100,72 108,88 C 114,102 120,115 125,130',
  // Inferior frontal sulcus
  'M 50,105 C 62,98 78,92 95,90 C 110,88 125,90 138,95',
  // Intraparietal sulcus
  'M 190,32 C 195,50 200,72 205,95 C 208,112 210,130 210,148',
  // Superior temporal sulcus
  'M 62,222 C 80,214 100,208 125,206 C 148,205 170,208 190,216',
  // Parieto-occipital sulcus
  'M 220,52 C 224,72 228,98 230,125 C 232,148 232,170 228,190',
];

// Cerebellum — distinct foliated structure under occipital
const BRAIN_CEREBELLUM =
  'M 210,232 C 225,235 242,248 248,265' +
  ' C 253,280 250,298 240,310' +
  ' C 230,320 216,325 202,322' +
  ' C 188,318 178,308 174,295' +
  ' C 170,278 175,258 190,245' +
  ' C 196,240 204,236 210,232 Z';

// Cerebellum folia — horizontal striations (tree-of-life pattern)
const BRAIN_CEREBELLUM_FOLIA = [
  'M 182,260 C 198,255 220,256 242,264',
  'M 178,275 C 196,270 218,270 244,278',
  'M 178,288 C 196,283 218,284 242,292',
  'M 182,300 C 198,296 218,297 236,305',
  'M 190,310 C 205,307 220,308 232,314',
];

// Brain stem — pons + medulla, descending from center
const BRAIN_STEM =
  // Pons (wider, bulging)
  'M 172,280 C 168,285 165,295 166,308' +
  // Medulla (narrowing)
  ' C 167,322 170,340 172,358' +
  // Slight taper
  ' C 173,368 174,375 172,380' +
  // Return
  ' M 182,280 C 186,290 188,305 186,320' +
  ' C 184,338 180,355 178,370';

// Brain regions mapped to cognitive tiers and transformer layer ranges
// Anatomy: Nano=brainstem+cerebellum, Core=temporal+parietal+occipital, Prime=frontal+prefrontal+motor
// Brain regions positioned to match the Wikimedia anatomical SVG
// SVG scaled to ~280x200 with y-offset of 20. Brain faces LEFT.
// Scale factor: 280/1024 ≈ 0.273
const BRAIN_REGIONS = [
  // NANO — brain stem & cerebellum (bottom-right in the SVG)
  { name: 'Brain Stem',   tier: 'nano',  layerRange: [0, 8],   cx: 165, cy: 200, rx: 12, ry: 18 },
  { name: 'Cerebellum',   tier: 'nano',  layerRange: [8, 16],  cx: 210, cy: 175, rx: 28, ry: 20 },

  // CORE — temporal (bottom-center), parietal (top-center), occipital (back)
  { name: 'Temporal',     tier: 'core',  layerRange: [0, 8],   cx: 110, cy: 170, rx: 35, ry: 14 },
  { name: 'Parietal',     tier: 'core',  layerRange: [8, 16],  cx: 130, cy: 60,  rx: 28, ry: 20 },
  { name: 'Occipital',    tier: 'core',  layerRange: [16, 24], cx: 230, cy: 100, rx: 18, ry: 28 },

  // PRIME — frontal cortex (left side of brain — front)
  { name: 'Prefrontal',   tier: 'prime', layerRange: [0, 12],  cx: 30,  cy: 130, rx: 18, ry: 30 },
  { name: 'Motor Cortex', tier: 'prime', layerRange: [12, 24], cx: 80,  cy: 55,  rx: 25, ry: 18 },
  { name: 'Frontal',      tier: 'prime', layerRange: [24, 32], cx: 50,  cy: 90,  rx: 28, ry: 28 },
];

// Tier idle colors — anatomically coded
const TIER_IDLE_COLORS = {
  nano:  '#4fc3f7',  // cyan — fast reflexes
  core:  '#90caf9',  // light blue — operational
  prime: '#ce93d8',  // light purple — deep thought
};

// Concept color palette — distinct hues for named features
const CONCEPT_PALETTE = [
  '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
  '#2196f3', '#00bcd4', '#009688', '#4caf50', '#8bc34a',
  '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722',
  '#ef5350', '#ab47bc', '#5c6bc0', '#26a69a', '#66bb6a',
];
const _conceptColorMap = {};  // label -> color
let _conceptColorIdx = 0;
function _getConceptColor(label) {
  if (!label || label.startsWith('neuron_') || label.startsWith('feature_')) return null;
  if (_conceptColorMap[label]) return _conceptColorMap[label];
  _conceptColorMap[label] = CONCEPT_PALETTE[_conceptColorIdx % CONCEPT_PALETTE.length];
  _conceptColorIdx++;
  return _conceptColorMap[label];
}

// Pre-compute neuron fiber segments within brain regions for a specific tier
function _generateNeurons(tier, count) {
  const neurons = [];
  const tierRegions = BRAIN_REGIONS.filter(r => r.tier === tier);
  if (tierRegions.length === 0) return neurons;
  const perRegion = Math.ceil(count / tierRegions.length);
  // Use different seed per tier so positions don't overlap
  let seed = tier === 'nano' ? 42 : tier === 'core' ? 137 : 271;
  function rand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }

  for (const region of tierRegions) {
    for (let i = 0; i < perRegion && neurons.length < count; i++) {
      const angle = rand() * Math.PI * 2;
      const dist = Math.sqrt(rand());
      const cx = region.cx + Math.cos(angle) * dist * region.rx * 0.85;
      const cy = region.cy + Math.sin(angle) * dist * region.ry * 0.85;

      const fiberAngle = rand() * Math.PI;
      const fiberLen = 10 + rand() * 16; // slightly shorter for denser packing
      const x1 = cx - Math.cos(fiberAngle) * fiberLen / 2;
      const y1 = cy - Math.sin(fiberAngle) * fiberLen / 2;
      const x2 = cx + Math.cos(fiberAngle) * fiberLen / 2;
      const y2 = cy + Math.sin(fiberAngle) * fiberLen / 2;

      const rx1 = Math.round(x1 * 10) / 10;
      const ry1 = Math.round(y1 * 10) / 10;
      const rx2 = Math.round(x2 * 10) / 10;
      const ry2 = Math.round(y2 * 10) / 10;
      neurons.push({
        x: Math.round(cx * 10) / 10,
        y: Math.round(cy * 10) / 10,
        x1: rx1, y1: ry1,
        x2: rx2, y2: ry2,
        ox1: rx1, oy1: ry1,  // original endpoints for relaxation
        ox2: rx2, oy2: ry2,
        fiberLen: Math.round(fiberLen * 10) / 10,
        region: region.name,
        tier: tier,
        layerRange: region.layerRange,
        id: null, // assigned after merge
      });
    }
  }
  return neurons;
}

// Neuron counts per tier
const TIER_NEURON_COUNTS = { nano: 30, core: 45, prime: 60 };

function mindMapPanel() {
  return {
    live: true,
    hoveredFeature: null,
    activeConcepts: [],     // [{label, color, strength}] — for dynamic legend
    _es: null,
    _svg: null,
    _neurons: [],           // all neurons, all tiers, single array
    _neuronsByTier: {},     // tier -> [neuron refs] for fast lookup
    _activeNeurons: new Map(), // neuronId -> {label, strength, ...}
    _featureLabels: {},
    _colorScale: null,
    _reconnectTimer: null,
    _synapses: [],           // [{id, x, y, strength, pairKey, neuronA, neuronB, lastActive, tier}]
    _maxSynapses: 40,       // cap

    init() {
      this._colorScale = d3.scaleLinear()
        .domain([0, 5, 15])
        .range(['#4fc3f7', '#ffa726', '#e94560'])
        .clamp(true);

      // Generate neurons for all tiers into a single array with global IDs
      const allNeurons = [];
      this._neuronsByTier = {};
      for (const tier of ['nano', 'core', 'prime']) {
        const tierNeurons = _generateNeurons(tier, TIER_NEURON_COUNTS[tier]);
        this._neuronsByTier[tier] = [];
        for (const n of tierNeurons) {
          n.id = allNeurons.length;
          allNeurons.push(n);
          this._neuronsByTier[tier].push(n);
        }
      }
      this._neurons = allNeurons;

      this._loadAtlas();
      this.$nextTick(() => {
        this._initBrain();
        if (this.live) this._connect();
      });
    },

    destroy() {
      if (this._es) { this._es.close(); this._es = null; }
      if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    },

    _initBrain() {
      const container = document.getElementById('mindmap-unified');
      if (!container) return;

      const svg = d3.select(container).append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', '0 0 280 250')
        .attr('preserveAspectRatio', 'xMidYMid meet');

      // Zoom + pan
      const zoomG = svg.append('g').attr('class', 'zoom-group');
      svg.call(d3.zoom()
        .scaleExtent([0.5, 6])
        .on('zoom', (event) => zoomG.attr('transform', event.transform))
      ).on('dblclick.zoom', () => {
        svg.transition().duration(300).call(
          d3.zoom().scaleExtent([0.5, 6]).transform, d3.zoomIdentity);
      });

      // Glow filters — one per tier for color-coded glow
      const defs = svg.append('defs');
      for (const tier of ['nano', 'core', 'prime']) {
        const filter = defs.append('filter').attr('id', 'neuron-glow-' + tier)
          .attr('x', '-100%').attr('y', '-100%').attr('width', '300%').attr('height', '300%');
        filter.append('feGaussianBlur').attr('stdDeviation', 4).attr('result', 'glow');
        filter.append('feMerge').selectAll('feMergeNode')
          .data(['glow', 'SourceGraphic']).enter()
          .append('feMergeNode').attr('in', d => d);
      }

      // Brain anatomy — load real anatomical SVG as background image
      // The SVG (1024x732) is scaled to fit our viewBox (280x400)
      // We apply CSS filters to make it a dim wireframe-style backdrop
      zoomG.append('image')
        .attr('class', 'brain-image')
        .attr('href', BRAIN_SVG_URL)
        .attr('x', 0).attr('y', 20)
        .attr('width', 280)
        .attr('height', 280 * (BRAIN_SVG_VIEWBOX.h / BRAIN_SVG_VIEWBOX.w))
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .attr('opacity', 0.25);

      // Pathway layer (below neurons)
      zoomG.append('g').attr('class', 'pathways');

      // Synapse layer
      zoomG.append('g').attr('class', 'synapses');

      // Neuron fiber layer — all tiers in one group, color-coded
      const neuronGroup = zoomG.append('g').attr('class', 'neuron-group');
      const self = this;

      neuronGroup.selectAll('line.neuron-fiber')
        .data(this._neurons, d => d.id)
        .enter()
        .append('line')
        .attr('class', d => 'neuron-fiber idle tier-' + d.tier)
        .attr('x1', d => d.x1)
        .attr('y1', d => d.y1)
        .attr('x2', d => d.x2)
        .attr('y2', d => d.y2)
        .attr('stroke', d => TIER_IDLE_COLORS[d.tier])
        .attr('stroke-width', 1)
        .attr('stroke-linecap', 'round')
        .attr('opacity', 0.06)
        .on('mouseover', function(evt, d) {
          const active = self._activeNeurons.get(d.id);
          self.hoveredFeature = active ? {
            label: active.label,
            strength: active.strength,
            tier: d.tier.toUpperCase(),
            region: d.region,
            idx: d.id,
          } : {
            label: 'neuron_' + d.id,
            strength: 0,
            tier: d.tier.toUpperCase(),
            region: d.region,
            idx: d.id,
          };
        })
        .on('mouseout', function() {
          self.hoveredFeature = null;
        });

      // No text labels on the brain — color legend key is in the HTML above

      // Idle label (shown until first activity)
      zoomG.append('text')
        .attr('class', 'idle-label')
        .attr('x', 140).attr('y', 125)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--text-dim)')
        .attr('font-size', '10px')
        .attr('opacity', 0.5)
        .text('waiting for activity...');

      this._svg = svg;
    },

    _connect() {
      if (this._es) { this._es.close(); this._es = null; }
      // Show ALL neural activity — don't filter by session
      this._es = new EventSource('/api/activations/stream');

      this._es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.tier) {
            this._updateTier(data.tier, data);
          }
        } catch { /* skip malformed */ }
      };

      this._es.onerror = () => {
        if (this._es) this._es.close();
        this._es = null;
        if (this.live) {
          this._reconnectTimer = setTimeout(() => this._connect(), 3000);
        }
      };
    },

    _mapFeatureToNeuron(feature, tier) {
      const neurons = this._neuronsByTier[tier];
      if (!neurons || neurons.length === 0) return null;
      const layer = feature.layer != null ? feature.layer : 0;

      // Find neurons in the matching brain region for this layer
      const candidates = neurons.filter(n =>
        layer >= n.layerRange[0] && layer <= n.layerRange[1]
      );

      if (candidates.length === 0) {
        return neurons[feature.idx % neurons.length];
      }

      return candidates[feature.idx % candidates.length];
    },

    _updateTier(tier, data) {
      if (!data.features || !Array.isArray(data.features)) return;

      const svg = this._svg;
      if (!svg) return;

      // Hide idle label on first activity
      svg.select('.idle-label').attr('opacity', 0);

      const colorScale = this._colorScale;
      const neurons = this._neuronsByTier[tier];
      if (!neurons) return;
      const activeMap = this._activeNeurons;
      const idleColor = TIER_IDLE_COLORS[tier];

      // Track which neurons are active this frame
      const frameActiveIds = new Set();
      const frameActiveRegions = new Map();
      const frameFeaturesByIdx = new Map();

      for (const feat of data.features) {
        const label = feat.label || this._featureLabels[feat.idx] || ('feature_' + feat.idx);
        const neuron = this._mapFeatureToNeuron(feat, tier);
        if (!neuron) continue;

        frameActiveIds.add(neuron.id);
        activeMap.set(neuron.id, {
          label: label,
          strength: feat.strength,
          layer: feat.layer,
          featureIdx: feat.idx,
          tier: tier,
          timestamp: Date.now(),
        });

        const regionList = frameActiveRegions.get(neuron.region) || [];
        regionList.push(neuron.id);
        frameActiveRegions.set(neuron.region, regionList);

        const idxList = frameFeaturesByIdx.get(feat.idx) || [];
        idxList.push({ neuronId: neuron.id, layer: feat.layer, strength: feat.strength });
        frameFeaturesByIdx.set(feat.idx, idxList);
      }

      // Temporal co-occurrence tracker
      if (!this._cooccurrence) this._cooccurrence = new Map();
      const coMap = this._cooccurrence;
      const activeIds = Array.from(frameActiveIds);
      for (let i = 0; i < activeIds.length; i++) {
        for (let j = i + 1; j < activeIds.length; j++) {
          const key = Math.min(activeIds[i], activeIds[j]) + '-' + Math.max(activeIds[i], activeIds[j]);
          coMap.set(key, (coMap.get(key) || 0) + 1);
        }
      }
      if (coMap.size > 500) {
        for (const [k, v] of coMap) {
          if (v <= 1) coMap.delete(k);
          else coMap.set(k, Math.floor(v * 0.8));
        }
      }

      // Update neuron fiber visuals — only for this tier's neurons
      const fiberSel = svg.select('g.neuron-group').selectAll('line.tier-' + tier);
      const self = this;

      fiberSel.each(function(d) {
        const el = d3.select(this);
        const active = activeMap.get(d.id);

        if (frameActiveIds.has(d.id) && active) {
          const str = active.strength;
          const width = Math.max(2, Math.min(5, 1.5 + str * 0.25));
          // Concept color if this neuron has a named feature, else heat scale
          const conceptColor = _getConceptColor(active.label);
          const activeColor = conceptColor || colorScale(str);
          el.classed('idle', false)
            .classed('active', true)
            .classed('firing', true)
            .classed('glow', str > 5);
          el.attr('stroke', '#ffffff')
            .attr('stroke-width', width + 1.5)
            .attr('opacity', 1)
            .attr('filter', `url(#neuron-glow-${tier})`)
            .transition().duration(250)
            .attr('stroke', activeColor)
            .attr('stroke-width', width)
            .attr('opacity', 0.85)
            .attr('filter', str > 5 ? `url(#neuron-glow-${tier})` : null);

          // Stretch outer endpoint toward strongest connected synapse
          const mySynapses = self._synapses.filter(s => s.neuronA === d.id || s.neuronB === d.id);
          if (mySynapses.length > 0) {
            const strongest = mySynapses.reduce((a, b) => a.strength > b.strength ? a : b);
            const targetX = d.ox2 + (strongest.x - d.ox2) * 0.4;
            const targetY = d.oy2 + (strongest.y - d.oy2) * 0.4;
            el.transition().duration(300)
              .attr('x2', targetX)
              .attr('y2', targetY);
          }
        } else if (active && (Date.now() - active.timestamp) < 1500) {
          const elapsed = Date.now() - active.timestamp;
          const decay = 1 - (elapsed / 1500);
          const str = active.strength * decay;
          const width = Math.max(1, 1 + str * 0.15);
          const decayConceptColor = _getConceptColor(active.label);
          el.classed('glow', false)
            .classed('firing', decay > 0.3)
            .transition().duration(400)
            .attr('stroke', decayConceptColor || colorScale(str))
            .attr('stroke-width', width)
            .attr('opacity', Math.max(0.06, 0.9 * decay))
            .attr('filter', null);
          if (decay <= 0.05) {
            activeMap.delete(d.id);
          }
          // Relax endpoint back toward original during decay
          el.transition().duration(800)
            .attr('x2', d.ox2)
            .attr('y2', d.oy2);
        } else {
          if (activeMap.has(d.id)) activeMap.delete(d.id);
          el.classed('idle', true)
            .classed('active', false)
            .classed('firing', false)
            .classed('glow', false)
            .transition().duration(800)
            .attr('stroke', idleColor)
            .attr('stroke-width', 1)
            .attr('opacity', 0.06)
            .attr('filter', null)
            .attr('x2', d.ox2)
            .attr('y2', d.oy2);
        }
      });

      // Draw pathways (cross-tier pathways now visible in the unified brain)
      this._drawPathways(tier, frameActiveRegions, frameFeaturesByIdx, neurons);

      // Update and render synapses from co-occurrence data
      this._updateSynapses();
      this._renderSynapses();

      // Update active concepts legend — show currently-firing named features
      const conceptMap = new Map();
      for (const [nid, info] of activeMap) {
        if (info.label && !info.label.startsWith('neuron_') && !info.label.startsWith('feature_')) {
          const color = _getConceptColor(info.label);
          if (color) {
            const existing = conceptMap.get(info.label);
            if (!existing || info.strength > existing.strength) {
              conceptMap.set(info.label, { label: info.label, color, strength: info.strength });
            }
          }
        }
      }
      this.activeConcepts = Array.from(conceptMap.values())
        .sort((a, b) => b.strength - a.strength)
        .slice(0, 12);
    },

    _updateSynapses() {
      const coMap = this._cooccurrence;
      if (!coMap) return;
      const neurons = this._neurons;
      const synapses = this._synapses;

      // Create synapses for strong co-occurrences
      for (const [key, count] of coMap) {
        if (count < 3) continue; // need at least 3 co-firings
        const parts = key.split('-');
        const idA = parseInt(parts[0], 10);
        const idB = parseInt(parts[1], 10);
        const nA = neurons[idA];
        const nB = neurons[idB];
        if (!nA || !nB) continue;

        let existing = synapses.find(s => s.pairKey === key);
        if (existing) {
          existing.strength = Math.min(1, count / 15);
          existing.lastActive = Date.now();
          // Update position as midpoint of current neuron anchors
          existing.x = (nA.x + nB.x) / 2;
          existing.y = (nA.y + nB.y) / 2;
        } else if (synapses.length < this._maxSynapses) {
          synapses.push({
            id: 'syn-' + key,
            x: (nA.x + nB.x) / 2,
            y: (nA.y + nB.y) / 2,
            strength: Math.min(1, count / 15),
            pairKey: key,
            neuronA: idA,
            neuronB: idB,
            lastActive: Date.now(),
            tier: nA.tier,
            coCount: count,
          });
        }
      }

      // Remove weak/old synapses
      this._synapses = synapses.filter(s =>
        (Date.now() - s.lastActive) < 10000 && coMap.get(s.pairKey) > 1
      );

      // Keep strongest if over cap
      if (this._synapses.length > this._maxSynapses) {
        this._synapses.sort((a, b) => b.strength - a.strength);
        this._synapses = this._synapses.slice(0, this._maxSynapses);
      }
    },

    _renderSynapses() {
      const svg = this._svg;
      if (!svg) return;
      const self = this;

      const synSel = svg.select('g.synapses').selectAll('circle.synapse')
        .data(this._synapses, d => d.id);

      synSel.exit()
        .transition().duration(500)
        .attr('r', 0)
        .attr('opacity', 0)
        .remove();

      synSel.enter().append('circle')
        .attr('class', 'synapse')
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', 0)
        .attr('fill', '#ffffff')
        .attr('opacity', 0)
        .on('mouseover', function(evt, d) {
          const nA = self._neurons[d.neuronA];
          const nB = self._neurons[d.neuronB];
          const activeA = self._activeNeurons.get(d.neuronA);
          const activeB = self._activeNeurons.get(d.neuronB);
          const labelA = activeA ? activeA.label : ('neuron_' + d.neuronA);
          const labelB = activeB ? activeB.label : ('neuron_' + d.neuronB);
          self.hoveredFeature = {
            type: 'synapse',
            label: 'Synapse',
            tier: d.tier.toUpperCase(),
            region: 'Functional connectivity',
            strength: d.strength,
            idx: d.pairKey,
            neuronLabels: labelA + ' \u2194 ' + labelB,
            coCount: d.coCount || 0,
          };
        })
        .on('mouseout', function() {
          self.hoveredFeature = null;
        })
        .transition().duration(300)
        .attr('r', d => 2 + d.strength * 3)
        .attr('opacity', d => 0.3 + d.strength * 0.5);

      // Update existing synapses
      synSel
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .transition().duration(200)
        .attr('r', d => 2 + d.strength * 3)
        .attr('opacity', d => 0.3 + d.strength * 0.5);
    },

    _drawPathways(tier, frameActiveRegions, frameFeaturesByIdx, neurons) {
      const svg = this._svg;
      const colorScale = this._colorScale;
      const activeMap = this._activeNeurons;
      const coMap = this._cooccurrence || new Map();
      const pathways = [];

      // ── Type 1: Cross-layer depth pathways (solid) ──
      for (const [featIdx, entries] of frameFeaturesByIdx) {
        if (entries.length < 2) continue;
        entries.sort((a, b) => a.layer - b.layer);
        for (let i = 0; i < entries.length - 1; i++) {
          const nA = this._neurons.find(n => n.id === entries[i].neuronId);
          const nB = this._neurons.find(n => n.id === entries[i + 1].neuronId);
          if (nA && nB && nA.id !== nB.id) {
            pathways.push({
              x1: nA.x, y1: nA.y, x2: nB.x, y2: nB.y,
              strength: (entries[i].strength + entries[i + 1].strength) / 2,
              key: 'depth-' + tier + '-' + featIdx + '-' + entries[i].layer + '-' + entries[i + 1].layer,
              type: 'depth',
            });
          }
        }
      }

      // ── Type 2: Same-token co-activation (dotted) ──
      const regionNames = Array.from(frameActiveRegions.keys());
      for (let i = 0; i < regionNames.length; i++) {
        for (let j = i + 1; j < regionNames.length; j++) {
          const idsA = frameActiveRegions.get(regionNames[i]);
          const idsB = frameActiveRegions.get(regionNames[j]);
          let bestA = null, bestStrA = 0;
          for (const id of idsA) {
            const a = activeMap.get(id);
            if (a && a.strength > bestStrA) { bestStrA = a.strength; bestA = id; }
          }
          let bestB = null, bestStrB = 0;
          for (const id of idsB) {
            const a = activeMap.get(id);
            if (a && a.strength > bestStrB) { bestStrB = a.strength; bestB = id; }
          }
          if (bestA != null && bestB != null) {
            const nA = this._neurons.find(n => n.id === bestA);
            const nB = this._neurons.find(n => n.id === bestB);
            if (nA && nB) {
              const coKey = Math.min(bestA, bestB) + '-' + Math.max(bestA, bestB);
              const coCount = coMap.get(coKey) || 0;
              const temporalBoost = Math.min(1.0, coCount / 10);
              pathways.push({
                x1: nA.x, y1: nA.y, x2: nB.x, y2: nB.y,
                strength: (bestStrA + bestStrB) / 2,
                key: 'coactive-' + tier + '-' + regionNames[i] + '-' + regionNames[j],
                type: 'coactive',
                temporalBoost: temporalBoost,
              });
            }
          }
        }
      }

      // ── D3 update — scoped to this tier's pathways ──
      const allPathSel = svg.select('g.pathways').selectAll('line.pathway-' + tier)
        .data(pathways, d => d.key);

      allPathSel.exit()
        .transition().duration(500)
        .attr('opacity', 0)
        .remove();

      const enter = allPathSel.enter().append('line')
        .attr('class', d => 'pathway pathway-' + tier + ' ' + d.type)
        .attr('x1', d => d.x1).attr('y1', d => d.y1)
        .attr('x2', d => d.x2).attr('y2', d => d.y2)
        .attr('stroke-linecap', 'round')
        .attr('stroke-dasharray', d => d.type === 'coactive' ? '4,4' : 'none');

      enter.filter(d => d.type === 'depth')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 2.5)
        .attr('opacity', 0.8)
        .transition().duration(400)
        .attr('stroke', d => colorScale(d.strength))
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.5);

      enter.filter(d => d.type === 'coactive')
        .attr('stroke', '#8899aa')
        .attr('stroke-width', 0.8)
        .attr('opacity', 0)
        .transition().duration(300)
        .attr('opacity', d => 0.15 + (d.temporalBoost || 0) * 0.4);

      allPathSel
        .attr('x1', d => d.x1).attr('y1', d => d.y1)
        .attr('x2', d => d.x2).attr('y2', d => d.y2)
        .attr('stroke', d => d.type === 'depth' ? colorScale(d.strength) : '#8899aa')
        .attr('stroke-width', d => d.type === 'depth' ? 1.5 : 0.8)
        .attr('opacity', d => {
          if (d.type === 'depth') return 0.6;
          return 0.2 + (d.temporalBoost || 0) * 0.4;
        });
    },

    async _loadAtlas() {
      try {
        const resp = await fetch('/api/activations/atlas');
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.labels) {
            this._featureLabels = data.labels;
          }
        }
      } catch { /* atlas unavailable — use fallback labels */ }
    },

    toggleLive() {
      this.live = !this.live;
      if (this.live) {
        this._connect();
      } else {
        if (this._es) { this._es.close(); this._es = null; }
        if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
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
  'gaia-core-candidate': 'Core (cand.)',
  'discord': 'Discord',
};

function systemPanel() {
  return {
    services: [],
    sleepState: '--',
    sleepStateClass: '',
    gpuOwner: '--',
    alignment: '--',
    alignmentClass: '',
    cogMonitorText: '--',
    cogMonitorClass: '',
    registryText: '--',
    registryClass: '',
    registryDetail: '',
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
      await Promise.all([this.pollServices(), this.pollSleep(), this.pollOrchestrator(), this.pollAlignment(), this.pollCognitiveMonitor(), this.pollRegistry()]);
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

    async pollAlignment() {
      try {
        const resp = await fetch('/api/system/cognitive/status');
        if (resp.ok) {
          const data = await resp.json();
          this.alignment = data.alignment || 'UNKNOWN';
          const cls = { 'SELF_ALIGNED': 'ok', 'ALIGNED': 'ok', 'PARTIAL': 'warn', 'UNTRAINED': 'error', 'UNKNOWN': '' };
          this.alignmentClass = cls[this.alignment] || '';
        }
      } catch { this.alignment = '--'; }
    },

    async pollCognitiveMonitor() {
      try {
        const resp = await fetch('/api/system/cognitive/monitor');
        if (resp.ok) {
          const data = await resp.json();
          const lr = data.last_result;
          if (!lr) {
            this.cogMonitorText = 'pending';
            this.cogMonitorClass = '';
          } else if (lr.status === 'pass') {
            this.cogMonitorText = 'pass';
            this.cogMonitorClass = 'ok';
          } else if (lr.status === 'skipped') {
            this.cogMonitorText = 'skipped';
            this.cogMonitorClass = 'warn';
          } else {
            this.cogMonitorText = `fail (${data.consecutive_failures})`;
            this.cogMonitorClass = 'error';
          }
        }
      } catch { this.cogMonitorText = '--'; this.cogMonitorClass = ''; }
    },

    async pollRegistry() {
      try {
        const resp = await fetch('/api/system/registry/validation');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (data.status === 'not_compiled' || data.status === 'not_checked') {
          this.registryText = 'not compiled';
          this.registryClass = 'warn';
          this.registryDetail = '';
        } else if (data.status === 'clean') {
          this.registryText = `${data.services_covered} svc · ${data.edges} edges`;
          this.registryClass = 'ok';
          this.registryDetail = '';
        } else if (data.status === 'warnings') {
          this.registryText = `${data.services_covered} svc · ${data.orphaned_outbound} orphaned`;
          this.registryClass = 'warn';
          this.registryDetail = `${data.edges} edges`;
        } else if (data.status === 'error') {
          this.registryText = 'error';
          this.registryClass = 'error';
          this.registryDetail = '';
        } else {
          this.registryText = data.status || 'unknown';
          this.registryClass = '';
          this.registryDetail = '';
        }
      } catch { this.registryText = '--'; this.registryClass = ''; this.registryDetail = ''; }
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
          .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); /* keep d.fx/d.fy — node stays where dropped */ })
        )
        .on('click', (event, d) => {
          window.dispatchEvent(new CustomEvent('select-blueprint', { detail: { id: d.id } }));
        });

      // Log scale for node radius — prevents gateway services (gaia-web: 94 interfaces)
      // from dwarfing focused services (gaia-nano: 5 interfaces).
      // Base 15px, log scale with floor of 1 to avoid log(0).
      const nodeRadius = d => 15 + Math.log2(Math.max(d.interface_count || 1, 1)) * 8;

      node.append('circle')
        .attr('r', nodeRadius)
        .attr('fill', nodeColor)
        .attr('stroke', d => d.genesis ? '#e94560' : nodeColor(d))
        .attr('stroke-dasharray', d => d.genesis ? '3,3' : '')
        .attr('opacity', 0.85);

      node.append('text')
        .attr('dy', d => nodeRadius(d) + 14)
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
          .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); /* keep d.fx/d.fy — node stays where dropped */ })
        )
        .on('click', (event, d) => {
          event.stopPropagation();
          window.dispatchEvent(new CustomEvent('show-component-detail', { detail: d }));
        });

      const compRadius = d => 12 + Math.log2(Math.max(d.interface_count || 1, 1)) * 6;

      node.append('circle')
        .attr('r', compRadius)
        .attr('fill', d => colorScale(d.id))
        .attr('stroke', d => d3.color(colorScale(d.id)).darker(0.5))
        .attr('stroke-width', 2)
        .attr('opacity', 0.9);

      node.append('text')
        .attr('dy', d => compRadius(d) + 14)
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

// ── Lifecycle Mission Control Component ───────────────────────────────────────

function lifecyclePanel() {
  return {
    state: 'unknown',
    tiers: {},
    transitions: [],
    history: [],
    vramUsed: 0,
    vramTotal: 15833,
    vramSegments: [],
    freePct: 100,
    transitioning: false,
    transPhase: '',
    lastResult: null,
    _poll: null,

    init() {
      this.refresh();
      this._poll = setInterval(() => this.refresh(), 5000);
    },

    destroy() {
      if (this._poll) clearInterval(this._poll);
    },

    async refresh() {
      try {
        const resp = await fetch('/api/system/lifecycle/state');
        if (!resp.ok) return;
        const data = await resp.json();
        this.state = data.state || 'unknown';
        this.tiers = data.tiers || {};
        this.vramUsed = data.vram_used_mb || 0;
        this.vramTotal = data.vram_total_mb || 15833;
        this.freePct = Math.max(0, ((this.vramTotal - this.vramUsed) / this.vramTotal) * 100);

        // Build VRAM segments
        const segs = [];
        for (const [tier, info] of Object.entries(this.tiers)) {
          if (info.vram_mb > 0) {
            segs.push({
              tier,
              mb: info.vram_mb,
              pct: (info.vram_mb / this.vramTotal) * 100,
            });
          }
        }
        this.vramSegments = segs;

        // Check for transitioning state
        if (data.state === 'transitioning') {
          this.transitioning = true;
          this.transPhase = data.transition_phase || 'working';
        } else {
          this.transitioning = false;
        }

        // Fetch available transitions
        const tResp = await fetch('/api/system/lifecycle/transitions');
        if (tResp.ok) this.transitions = await tResp.json();

        // Fetch history
        const hResp = await fetch('/api/system/lifecycle/history');
        if (hResp.ok) this.history = await hResp.json();
      } catch { /* silent */ }
    },

    async doTransition(trigger, target) {
      this.transitioning = true;
      this.transPhase = 'requesting';
      this.lastResult = null;
      try {
        const body = { trigger, reason: 'dashboard' };
        if (target) body.target = target;
        const resp = await fetch('/api/system/lifecycle/transition', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        this.lastResult = await resp.json();
      } catch (e) {
        this.lastResult = { ok: false, error: e.message };
      }
      this.transitioning = false;
      this.refresh();
    },

    async reconcile() {
      this.transitioning = true;
      this.transPhase = 'reconciling';
      try {
        const resp = await fetch('/api/system/lifecycle/reconcile', { method: 'POST' });
        this.lastResult = await resp.json();
      } catch (e) {
        this.lastResult = { ok: false, error: e.message };
      }
      this.transitioning = false;
      this.refresh();
    },

    transitionLabel(t) {
      const labels = {
        wake_signal: 'Wake',
        idle_timeout: 'Sleep',
        voice_join: 'Listen',
        voice_leave: 'Stop Listening',
        escalation_needed: 'Focus (Prime)',
        task_complete: 'Relax (Core)',
        training_scheduled: 'Meditate (Train)',
        training_complete: 'Wake from Training',
        extended_idle: 'Deep Sleep',
        preempt: 'Preempt Training',
      };
      if (t.trigger === 'user_request') {
        const target = t.target || t.targets?.[0] || '';
        const tLabels = {
          awake: 'Wake',
          focusing: 'Focus (Prime)',
          sleep: 'Sleep',
          deep_sleep: 'Deep Sleep',
          meditation: 'Meditate',
          listening: 'Listen',
        };
        return tLabels[target] || target;
      }
      return labels[t.trigger] || t.trigger;
    },
  };
}


// ── Hooks / Commands Panel Component ──────────────────────────────────────────

function hooksPanel() {
  return {
    sleepState: '--',
    sleepCycle: '--',
    sleepUptime: '--',
    autoSleepEnabled: true,
    sleepThreshold: 30,
    wakeDiscordTyping: true,
    wakeWorkstationActivity: false,
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
        const [statusResp, configResp] = await Promise.all([
          fetch('/api/hooks/sleep/status'),
          fetch('/api/hooks/sleep/config'),
        ]);
        if (statusResp.ok) {
          const data = await statusResp.json();
          this.sleepState = data.state || data.sleep_state || 'unknown';
          this.sleepCycle = data.cycle_count != null ? `#${data.cycle_count}` : '--';
          const up = data.uptime_seconds || data.uptime;
          this.sleepUptime = up != null ? formatDuration(up) : '--';
          if (data.auto_sleep_enabled != null) this.autoSleepEnabled = data.auto_sleep_enabled;
          if (data.idle_threshold_minutes != null) this.sleepThreshold = data.idle_threshold_minutes;
        }
        if (configResp.ok) {
          const cfg = await configResp.json();
          if (cfg.auto_sleep_enabled != null) this.autoSleepEnabled = cfg.auto_sleep_enabled;
          if (cfg.idle_threshold_minutes != null) this.sleepThreshold = cfg.idle_threshold_minutes;
        }
      } catch {
        this.sleepState = 'unreachable';
      }
    },

    async toggleAutoSleep(enabled) {
      this.logEntry('sleep/toggle', `auto-sleep → ${enabled}`, false);
      try {
        const resp = await fetch('/api/hooks/sleep/toggle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled }),
        });
        const data = await resp.json();
        if (resp.ok) {
          this.autoSleepEnabled = data.auto_sleep_enabled;
          this.logEntry('sleep/toggle', JSON.stringify(data).substring(0, 120), false);
        } else {
          this.logEntry('sleep/toggle', `Error ${resp.status}: ${data.error || 'unknown'}`, true);
        }
      } catch (e) {
        this.logEntry('sleep/toggle', `Failed: ${e.message}`, true);
      }
    },

    async forceSleep() {
      this.logEntry('sleep/deep', 'entering deep sleep — unloading all models...', false);
      try {
        const resp = await fetch('/api/hooks/sleep/deep', { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
          this.logEntry('sleep/deep', JSON.stringify(data).substring(0, 150), false);
          this.sleepState = data.state || 'asleep';
        } else {
          this.logEntry('sleep/deep', `Error ${resp.status}: ${data.error || 'unknown'}`, true);
        }
      } catch (e) {
        this.logEntry('sleep/deep', `Failed: ${e.message}`, true);
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

    async refreshWakeConfig() {
      try {
        const resp = await fetch('/api/hooks/sleep/wake-config');
        if (resp.ok) {
          const data = await resp.json();
          if (data.discord_typing != null) this.wakeDiscordTyping = data.discord_typing;
          if (data.workstation_activity != null) this.wakeWorkstationActivity = data.workstation_activity;
        }
      } catch { /* ignore */ }
    },

    async toggleWakeTrigger(trigger, enabled) {
      this.logEntry('sleep/wake-toggle', `${trigger} → ${enabled}`, false);
      try {
        const resp = await fetch('/api/hooks/sleep/wake-toggle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ trigger, enabled }),
        });
        const data = await resp.json();
        if (resp.ok && data.wake_config) {
          this.wakeDiscordTyping = data.wake_config.discord_typing ?? this.wakeDiscordTyping;
          this.wakeWorkstationActivity = data.wake_config.workstation_activity ?? this.wakeWorkstationActivity;
          this.logEntry('sleep/wake-toggle', JSON.stringify(data).substring(0, 120), false);
        } else {
          this.logEntry('sleep/wake-toggle', `Error ${resp.status}: ${data.error || 'unknown'}`, true);
        }
      } catch (e) {
        this.logEntry('sleep/wake-toggle', `Failed: ${e.message}`, true);
      }
    },

    async refreshAll() {
      await Promise.all([this.refreshSleep(), this.refreshGpu(), this.refreshWakeConfig()]);
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

// ── Changelog Panel ──────────────────────────────────────────────────────

function changelogPanel() {
  return {
    entries: [],
    serviceFilter: '',
    typeFilter: '',

    async init() {
      await this.load();
    },

    async load() {
      try {
        const params = new URLSearchParams();
        if (this.serviceFilter) params.set('service', this.serviceFilter);
        if (this.typeFilter) params.set('type', this.typeFilter);
        params.set('limit', '100');
        const r = await fetch('/api/changelog?' + params.toString());
        if (r.ok) {
          const data = await r.json();
          this.entries = data.entries || [];
        }
      } catch (e) {}
    },

    typeIcon(type) {
      const icons = { feat: '+', fix: '!', refactor: '~', promote: '^', docs: '#', config: '%', manual: '*' };
      return icons[type] || '?';
    },
  };
}

// ── Chaos Monkey Panel ────────────────────────────────────────────────────

function chaosPanel() {
  return {
    config: { mode: 'triggered', drill_types: ['container', 'code'], schedule_interval_hours: 6 },
    serenity: { serene: false, score: 0, threshold: 5.0 },
    injecting: false,
    lastResult: '',

    async init() {
      await this.loadConfig();
      await this.loadSerenity();
      // Refresh serenity every 30s
      setInterval(() => this.loadSerenity(), 30000);
    },

    async loadConfig() {
      try {
        const r = await fetch('/api/chaos/config');
        if (r.ok) this.config = await r.json();
      } catch (e) {}
    },

    async saveConfig() {
      try {
        await fetch('/api/chaos/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.config),
        });
      } catch (e) {}
    },

    async loadSerenity() {
      try {
        const r = await fetch('/api/chaos/serenity');
        if (r.ok) this.serenity = await r.json();
      } catch (e) {}
    },

    async injectChaos() {
      if (this.injecting) return;
      this.injecting = true;
      this.lastResult = '';
      try {
        const r = await fetch('/api/chaos/inject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ drill_types: this.config.drill_types }),
        });
        const data = await r.json();
        if (data.error) {
          this.lastResult = 'Error: ' + data.error;
        } else {
          const results = data.drill_results || [];
          const summary = results.map(r => `${r.service}: ${r.status}`).join(', ');
          this.lastResult = summary || (data.drill_type ? `${data.drill_type} drill completed` : 'Done');
          await this.loadSerenity();
        }
      } catch (e) {
        this.lastResult = 'Request failed: ' + e.message;
      } finally {
        this.injecting = false;
      }
    },
  };
}

// ── Cognitive Battery Panel ────────────────────────────────────────────────

function cognitivePanel() {
  return {
    running: false,
    alignment: 'UNKNOWN',
    lastRun: {},
    failures: [],
    bySection: {},
    selectedSection: '',
    showResults: false,
    _timer: null,

    async init() {
      await this.pollStatus();
      this._timer = setInterval(() => this.pollStatus(), 15000);
    },

    destroy() {
      if (this._timer) clearInterval(this._timer);
    },

    async pollStatus() {
      try {
        const r = await fetch('/api/system/cognitive/status');
        if (r.ok) {
          const data = await r.json();
          this.running = data.running || false;
          this.alignment = data.alignment || 'UNKNOWN';
          if (data.last_run) this.lastRun = data.last_run;
        }
      } catch {}
      // If we have results, fetch full details for failures/sections
      if (this.lastRun.total && !this.bySection.architecture) {
        await this.fetchResults();
      }
    },

    async fetchResults() {
      try {
        const r = await fetch('/api/system/cognitive/results');
        if (r.ok) {
          const data = await r.json();
          this.failures = data.failures || [];
          this.bySection = data.by_section || {};
          this.alignment = data.alignment || this.alignment;
          if (data.summary) this.lastRun = data.summary;
        }
      } catch {}
    },

    async runBattery() {
      if (this.running) return;
      this.running = true;
      this.showResults = false;
      try {
        const body = {};
        if (this.selectedSection) body.section = this.selectedSection;
        await fetch('/api/system/cognitive/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        // Poll more frequently while running
        const pollUntilDone = setInterval(async () => {
          await this.pollStatus();
          if (!this.running) {
            clearInterval(pollUntilDone);
            await this.fetchResults();
          }
        }, 5000);
      } catch (e) {
        this.running = false;
      }
    },
  };
}


// ── Training Pipeline Panel ───────────────────────────────────────────────

function pipelinePanel() {
  return {
    pipelineStatus: '--',
    currentStage: '',
    alignmentStatus: '',
    pipelineRunning: false,
    dryRun: false,
    skipNano: false,
    _timer: null,

    async init() {
      await this.poll();
      this._timer = setInterval(() => this.poll(), 15000);
    },

    destroy() {
      if (this._timer) clearInterval(this._timer);
    },

    async poll() {
      try {
        const r = await fetch('/api/system/pipeline/status');
        if (r.ok) {
          const data = await r.json();
          this.alignmentStatus = data.alignment_status || '';
          // Find current/latest stage
          const stages = data.stages || {};
          let running = Object.entries(stages).find(([_, s]) => s.status === 'running');
          if (running) {
            this.pipelineStatus = 'Running';
            this.pipelineRunning = true;
            this.currentStage = running[0];
          } else {
            this.pipelineRunning = false;
            let completed = Object.entries(stages).filter(([_, s]) => s.status === 'completed');
            if (completed.length > 0) {
              this.pipelineStatus = 'Completed';
              this.currentStage = completed[completed.length - 1][0];
            } else {
              this.pipelineStatus = data.status || 'Idle';
              this.currentStage = '';
            }
          }
        }
      } catch {
        this.pipelineStatus = '--';
      }
    },

    async runPipeline() {
      if (this.pipelineRunning) return;
      this.pipelineRunning = true;
      this.pipelineStatus = 'Starting...';
      try {
        const opts = {};
        if (this.dryRun) opts.dry_run = true;
        if (this.skipNano) opts.skip_nano = true;
        const r = await fetch('/api/system/pipeline/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(opts),
        });
        const data = await r.json();
        if (data.ok) {
          this.pipelineStatus = 'Running';
          // Poll more frequently while running
          if (this._timer) clearInterval(this._timer);
          this._timer = setInterval(() => this.poll(), 5000);
        } else {
          this.pipelineStatus = data.error || 'Failed';
          this.pipelineRunning = false;
        }
      } catch (e) {
        this.pipelineStatus = 'Error: ' + e.message;
        this.pipelineRunning = false;
      }
    },

    async runSmoke() {
      if (this.pipelineRunning) return;
      this.pipelineRunning = true;
      this.pipelineStatus = 'Starting smoke...';
      try {
        const r = await fetch('/api/system/pipeline/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stage: 'COGNITIVE_SMOKE' }),
        });
        const data = await r.json();
        if (data.ok) {
          this.pipelineStatus = 'Running (smoke)';
          if (this._timer) clearInterval(this._timer);
          this._timer = setInterval(() => this.poll(), 5000);
        } else {
          this.pipelineStatus = data.error || 'Failed';
          this.pipelineRunning = false;
        }
      } catch (e) {
        this.pipelineStatus = 'Error: ' + e.message;
        this.pipelineRunning = false;
      }
    },
  };
}


// ── Doctor & Immunity Panel ─────────────────────────────────────────────────

function doctorPanel() {
  return {
    // Status from /status polling
    irritations: [],
    irritationCount: 0,
    alarms: [],
    alarmCount: 0,
    maintenanceActive: false,
    remediations: [],
    serenityScore: 0,
    serenityThreshold: 5.0,
    dissonanceCount: 0,
    dissonanceFiles: [],

    // Surgeon state
    surgeonApprovalRequired: false,
    surgeonQueue: [],
    surgeonHistory: [],

    // UI toggles
    showIrritations: false,
    showRemediations: false,
    showDissonance: false,
    showSurgeonHistory: false,
    expandedRepair: null,

    _timer: null,

    async init() {
      await this.pollDoctor();
      this._timer = setInterval(() => this.pollDoctor(), 10000);
    },

    destroy() {
      if (this._timer) clearInterval(this._timer);
    },

    async pollDoctor() {
      // Parallel fetch: doctor status + maintenance + surgeon config + surgeon queue + irritations + dissonance
      const fetches = await Promise.allSettled([
        fetch('/api/system/doctor/status'),     // 0: full doctor status (alarms, remediations, serenity, maintenance)
        fetch('/api/system/maintenance/status'), // 1: maintenance
        fetch('/api/system/surgeon/config'),     // 2: surgeon config
        fetch('/api/system/surgeon/queue'),      // 3: surgeon queue
        fetch('/api/system/irritations'),        // 4: irritations
        fetch('/api/system/dissonance'),         // 5: dissonance
      ]);

      // [0] Doctor status — alarms, remediations, serenity, maintenance
      if (fetches[0].status === 'fulfilled' && fetches[0].value.ok) {
        try {
          const data = await fetches[0].value.json();
          // Serenity
          if (data.serenity) {
            this.serenityScore = data.serenity.score ?? data.serenity.total ?? 0;
            this.serenityThreshold = data.serenity.threshold ?? 5.0;
          }
          // Alarms
          this.alarms = data.active_alarms || [];
          this.alarmCount = this.alarms.length;
          // Remediations
          this.remediations = data.recent_remediations || [];
          // Maintenance
          this.maintenanceActive = data.maintenance_mode || false;
        } catch {}
      }

      // [1] Maintenance (fallback / authoritative)
      if (fetches[1].status === 'fulfilled' && fetches[1].value.ok) {
        try {
          const data = await fetches[1].value.json();
          this.maintenanceActive = data.active || false;
        } catch {}
      }

      // [2] Surgeon config
      if (fetches[2].status === 'fulfilled' && fetches[2].value.ok) {
        try {
          const data = await fetches[2].value.json();
          this.surgeonApprovalRequired = data.approval_required || false;
        } catch {}
      }

      // [3] Surgeon queue
      if (fetches[3].status === 'fulfilled' && fetches[3].value.ok) {
        try {
          const data = await fetches[3].value.json();
          this.surgeonQueue = data.queue || [];
        } catch {}
      }

      // [4] Irritations
      if (fetches[4].status === 'fulfilled' && fetches[4].value.ok) {
        try {
          const data = await fetches[4].value.json();
          const list = data.irritations || [];
          this.irritationCount = list.length;
          if (this.showIrritations) {
            this.irritations = list.slice(-20);
          }
        } catch {}
      }

      // [5] Dissonance
      if (fetches[5].status === 'fulfilled' && fetches[5].value.ok) {
        try {
          const data = await fetches[5].value.json();
          const diverged = data.diverged || [];
          this.dissonanceCount = diverged.length;
          if (this.showDissonance) {
            this.dissonanceFiles = diverged.slice(0, 20);
          }
        } catch {}
      }
    },

    async toggleIrritations() {
      this.showIrritations = !this.showIrritations;
      if (this.showIrritations && this.irritations.length === 0) {
        try {
          const r = await fetch('/api/system/irritations');
          if (r.ok) {
            const data = await r.json();
            this.irritations = (data.irritations || []).slice(-20);
          }
        } catch {}
      }
    },

    async toggleDissonance() {
      this.showDissonance = !this.showDissonance;
      if (this.showDissonance && this.dissonanceFiles.length === 0) {
        try {
          const r = await fetch('/api/system/dissonance');
          if (r.ok) {
            const data = await r.json();
            this.dissonanceFiles = (data.diverged || []).slice(0, 20);
          }
        } catch {}
      }
    },

    async toggleMaintenance() {
      const endpoint = this.maintenanceActive
        ? '/api/system/maintenance/exit'
        : '/api/system/maintenance/enter';
      try {
        const r = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'dashboard', entered_by: 'dashboard' }),
        });
        if (r.ok) {
          const data = await r.json();
          this.maintenanceActive = !this.maintenanceActive;
        }
      } catch {}
    },

    async toggleSurgeonApproval() {
      try {
        const r = await fetch('/api/system/surgeon/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ approval_required: !this.surgeonApprovalRequired }),
        });
        if (r.ok) {
          const data = await r.json();
          this.surgeonApprovalRequired = data.approval_required;
        }
      } catch {}
    },

    async approveRepair(repairId) {
      try {
        const r = await fetch('/api/system/surgeon/approve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repair_id: repairId }),
        });
        if (r.ok) {
          // Remove from queue, refresh
          this.surgeonQueue = this.surgeonQueue.filter(p => p.repair_id !== repairId);
          this.expandedRepair = null;
          await this.fetchSurgeonHistory();
        }
      } catch {}
    },

    async rejectRepair(repairId) {
      try {
        const r = await fetch('/api/system/surgeon/reject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repair_id: repairId }),
        });
        if (r.ok) {
          this.surgeonQueue = this.surgeonQueue.filter(p => p.repair_id !== repairId);
          this.expandedRepair = null;
          await this.fetchSurgeonHistory();
        }
      } catch {}
    },

    async toggleSurgeonHistory() {
      this.showSurgeonHistory = !this.showSurgeonHistory;
      if (this.showSurgeonHistory) {
        await this.fetchSurgeonHistory();
      }
    },

    async fetchSurgeonHistory() {
      try {
        const r = await fetch('/api/system/surgeon/history');
        if (r.ok) {
          const data = await r.json();
          this.surgeonHistory = (data.history || []).slice(-20);
        }
      } catch {}
    },
  };
}


// ── Training Monitor Panel ──────────────────────────────────────────────────

function trainingMonitor() {
  return {
    state: 'idle',
    step: 0,
    totalSteps: 0,
    loss: null,
    avgLoss: null,
    elapsed: 0,
    eta: 0,
    speed: 0,
    adapter: '',
    lossHistory: [],
    subprocessAlive: false,
    stopReason: '',
    error: '',
    progress: 0,
    lastState: '',

    // UI
    showLog: false,
    logLines: [],
    _timer: null,
    _logSource: null,

    async init() {
      await this.pollTraining();
      this._adaptInterval();
    },

    destroy() {
      if (this._timer) clearInterval(this._timer);
      if (this._logSource) this._logSource.close();
    },

    _adaptInterval() {
      if (this._timer) clearInterval(this._timer);
      const interval = (this.state === 'training' || this.state === 'setup' || this.state === 'saving') ? 3000 : 15000;
      this._timer = setInterval(() => this.pollTraining(), interval);
    },

    async pollTraining() {
      const prevState = this.state;
      try {
        const r = await fetch('/api/system/training/progress');
        if (r.ok) {
          const raw = await r.json();
          // Merge manager + progress_file into flat data object
          const mgr = raw.manager || {};
          const pf = raw.progress_file || {};
          const data = { ...pf, ...mgr, ...raw };

          // Determine state: prefer manager state, then progress_file state
          const mgrState = mgr.state || '';
          const pfState = pf.state || '';
          // Manager "complete" + subprocess "failed" = show failed
          if (mgrState === 'complete' && pfState === 'failed') {
            this.state = 'failed';
          } else if (mgrState === 'training' || pfState === 'training') {
            this.state = 'training';
          } else if (mgrState === 'complete' || pfState === 'completed') {
            this.state = 'completed';
          } else if (mgrState === 'idle' && !pfState) {
            this.state = 'idle';
          } else {
            this.state = pfState || mgrState || 'idle';
          }

          this.step = pf.step || mgr.subprocess_step || 0;
          this.totalSteps = pf.total_steps || 0;
          this.loss = (pf.loss ?? mgr.subprocess_loss) || null;
          if (this.loss === 0.0 && this.state !== 'training') this.loss = null;
          this.elapsed = pf.elapsed_seconds || 0;
          this.eta = pf.estimated_remaining || 0;
          this.adapter = mgr.current_adapter || pf.adapter_dir?.split('/').pop() || '';
          this.subprocessAlive = mgr.subprocess_alive || false;
          this.stopReason = pf.stop_reason || '';
          this.error = pf.error || '';
          this.progress = mgr.progress ?? (this.totalSteps > 0 ? this.step / this.totalSteps : 0);

          // Loss history
          if (pf.loss_history && pf.loss_history.length > 0) {
            this.lossHistory = pf.loss_history;
          } else if (this.loss != null && this.loss > 0 && this.state === 'training') {
            // Accumulate from polling
            if (this.lossHistory.length === 0 || this.lossHistory[this.lossHistory.length - 1] !== this.loss) {
              this.lossHistory.push(this.loss);
              if (this.lossHistory.length > 200) this.lossHistory = this.lossHistory.slice(-200);
            }
          }

          // Compute average loss
          if (this.lossHistory.length > 0) {
            this.avgLoss = this.lossHistory.reduce((a, b) => a + b, 0) / this.lossHistory.length;
          }

          // Compute speed (seconds per step)
          if (this.step > 0 && this.elapsed > 0) {
            this.speed = this.elapsed / this.step;
          }

          // Draw sparkline after data update
          this.$nextTick(() => {
            if (this.$refs.lossCanvas && this.lossHistory.length >= 2) {
              this.drawLossSparkline(this.$refs.lossCanvas);
            }
          });
        }
      } catch {
        // Silently handle — endpoint may not be reachable
      }

      // Track last active state for idle display
      if (this.state !== 'idle' && this.state !== 'unknown') {
        this.lastState = this.state;
      }

      // Adapt polling interval when state changes
      if (prevState !== this.state) {
        this._adaptInterval();
        // Clear loss history on new training session
        if (this.state === 'training' && prevState === 'idle') {
          this.lossHistory = [];
          this.avgLoss = null;
        }
      }
    },

    toggleLog() {
      this.showLog = !this.showLog;
      if (this.showLog) {
        this._connectLog();
      } else {
        this._disconnectLog();
      }
    },

    _connectLog() {
      if (this._logSource) this._logSource.close();
      this.logLines = [];
      try {
        this._logSource = new EventSource('/api/logs/stream?service=study');
        this._logSource.onmessage = (e) => {
          const line = e.data;
          if (line) {
            this.logLines.push(line);
            if (this.logLines.length > 200) this.logLines = this.logLines.slice(-150);
            // Auto-scroll
            this.$nextTick(() => {
              const container = this.$refs.logContainer;
              if (container) container.scrollTop = container.scrollHeight;
            });
          }
        };
        this._logSource.onerror = () => {
          // Reconnect silently handled by EventSource
        };
      } catch {
        this.logLines = ['Failed to connect to log stream'];
      }
    },

    _disconnectLog() {
      if (this._logSource) {
        this._logSource.close();
        this._logSource = null;
      }
    },

    drawLossSparkline(canvas) {
      if (!canvas || this.lossHistory.length < 2) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const values = this.lossHistory;
      const max = Math.max(...values);
      const min = Math.min(...values);
      const range = max - min || 1;
      const step = w / (values.length - 1);

      // Fill area
      ctx.beginPath();
      ctx.fillStyle = 'rgba(33, 150, 243, 0.1)';
      values.forEach((v, i) => {
        const x = i * step;
        const y = h - ((v - min) / range) * (h - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.strokeStyle = '#42a5f5';
      ctx.lineWidth = 1.5;
      values.forEach((v, i) => {
        const x = i * step;
        const y = h - ((v - min) / range) * (h - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Current value label
      const last = values[values.length - 1];
      ctx.fillStyle = '#42a5f5';
      ctx.font = '9px monospace';
      ctx.fillText(last.toFixed(4), w - 48, 10);
    },

    formatDur(seconds) {
      return formatDuration(seconds);
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
    roleFilter: '',       // '' = all, 'prime', 'core', 'nano'
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
