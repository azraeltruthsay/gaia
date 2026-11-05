/* Palace report: everything in her memory, for review (GAIA_Project-oxfy). */
document.addEventListener('alpine:init', () => {
  Alpine.store('palace', {
    sections: [
      { id: 'people', label: 'People', count: 'people' },
      { id: 'lookups', label: 'Looked up', count: 'lookups' },
      { id: 'memories', label: 'Memories', count: 'memories' },
      { id: 'conversations', label: 'Conversations', count: 'conversations' },
      { id: 'works', label: 'Works', count: 'works' },
      { id: 'graph', label: 'Knowledge graph', count: 'facts' },
    ],
    section: 'people',
    counts: {},
    query: '',
    offset: 0,
    limit: 50,
    total: 0,
    items: [],
    worlds: [],
    factTypes: [],
    world: '',
    factType: '',
    includeEnded: false,
    transcript: null,
    loading: false,
    error: '',
    started: false,
    _seq: 0,

    start() {
      if (this.started) return;
      this.started = true;
      this.refresh();
    },
    async refresh() {
      try {
        const r = await fetch('/api/palace/overview');
        if (r.ok) this.counts = await r.json();
      } catch (e) { /* the list below reports the failure */ }
      await this.load();
    },
    open(section) {
      this.section = section;
      this.offset = 0;
      this.query = '';
      this.transcript = null;
      this.load();
    },
    search() {
      this.offset = 0;
      this.transcript = null;
      this.load();
    },
    async load() {
      const seq = ++this._seq;
      this.loading = true;
      this.error = '';
      const params = new URLSearchParams({ q: this.query, limit: this.limit, offset: this.offset });
      if (this.section === 'graph') {
        if (this.world) params.set('world', this.world);
        if (this.factType) params.set('fact_type', this.factType);
        if (this.includeEnded) params.set('include_ended', 'true');
      }
      try {
        const r = await fetch(`/api/palace/${this.section}?${params}`);
        if (!r.ok) throw new Error(`The server answered ${r.status}`);
        const data = await r.json();
        if (seq !== this._seq) return;
        this.items = data.items;
        this.total = data.total;
        if (data.worlds) this.worlds = data.worlds;
        if (data.fact_types) this.factTypes = data.fact_types;
      } catch (e) {
        if (seq !== this._seq) return;
        this.items = [];
        this.total = 0;
        this.error = `Couldn't load this section. ${e.message}`;
      }
      this.loading = false;
    },
    page(direction) {
      this.offset = Math.max(0, this.offset + direction * this.limit);
      this.load();
    },
    get range() {
      if (!this.total) return '';
      return `${this.offset + 1}–${Math.min(this.offset + this.limit, this.total)} of ${this.total}`;
    },
    async read(sessionId) {
      this.error = '';
      try {
        const r = await fetch(`/api/palace/conversations/${encodeURIComponent(sessionId)}`);
        if (!r.ok) throw new Error(`The server answered ${r.status}`);
        this.transcript = await r.json();
      } catch (e) {
        this.error = `Couldn't open that conversation. ${e.message}`;
      }
    },
    day(ts) { return (ts || '').slice(0, 10); },
    list(v) { return Array.isArray(v) ? v.join(', ') : (v || ''); },
  });
});
