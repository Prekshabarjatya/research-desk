/* Research Desk. Plain JS, no build step. All dynamic text goes through textContent or
   createTextNode, never innerHTML, so model-written text cannot inject markup. */
(() => {
  'use strict';

  const app = document.getElementById('app');
  const toasts = document.getElementById('toasts');
  const KEY = 'rd_token';
  const BASE = (window.RD_API_BASE || '').replace(/\/+$/, '');

  const store = {
    get() { try { return sessionStorage.getItem(KEY) || ''; } catch { return ''; } },
    set(v) { try { sessionStorage.setItem(KEY, v); } catch { /* private mode: stay in memory */ } },
    clear() { try { sessionStorage.removeItem(KEY); } catch { /* ignore */ } },
  };

  let token = store.get();
  const S = { runs: null, run: null, runId: null, sig: '', listTimer: 0, runTimer: 0, main: null, rail: null };

  /* ---------- helpers ---------- */

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  function toast(message, kind) {
    const t = h('div', { class: 'toast' + (kind === 'error' ? ' error' : ''), text: message });
    toasts.append(t);
    setTimeout(() => t.remove(), 5000);
  }

  function parseTs(s) {
    if (!s) return null;
    const d = new Date(String(s).replace(' ', 'T').replace(/(\.\d{3})\d+/, '$1'));
    return isNaN(d) ? null : d;
  }

  function relTime(s) {
    const d = parseTs(s);
    if (!d) return '';
    const sec = Math.max(0, (Date.now() - d.getTime()) / 1000);
    if (sec < 45) return 'just now';
    if (sec < 3600) return Math.round(sec / 60) + ' min ago';
    if (sec < 86400) return Math.round(sec / 3600) + ' h ago';
    return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
  }

  const clock = (s) => {
    const d = parseTs(s);
    return d ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
  };
  const sentence = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
  const words = (t) => (t.match(/[\p{L}\p{N}]+(?:['-][\p{L}\p{N}]+)*/gu) || []).length;
  const slug = (t) => (t || 'paper').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'paper';
  const clip = (t, n) => (t.length > n ? t.slice(0, n).replace(/\s+\S*$/, '') + '...' : t);

  // Best available title: the finished topic, else the proposed one, else a short form of the brief.
  function titleFor(r) {
    const res = r.result || {};
    const proposed = [...(r.progress || [])].reverse().find((p) => p.topic);
    return res.topic || (r.gate && r.gate.topic) || (proposed && proposed.topic) || clip(r.prompt, 110);
  }

  /* ---------- api ---------- */

  async function api(path, opts = {}) {
    const res = await fetch(BASE + path, {
      ...opts,
      headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
    });
    if (res.status === 401) {
      signOut('That token was not accepted.');
      throw new Error('unauthorized');
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
      } catch { /* keep statusText */ }
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  function stopTimers() {
    clearTimeout(S.listTimer);
    clearTimeout(S.runTimer);
  }

  function signOut(message) {
    stopTimers();
    token = '';
    store.clear();
    S.runs = S.run = null;
    renderLogin(message);
  }

  /* ---------- status vocabulary ---------- */

  function statusInfo(r) {
    const review = r.needs_human_review ?? (r.result && r.result.needs_human_review);
    switch (r.status) {
      case 'queued': return ['Queued', 'working'];
      case 'running': return ['Working', 'working'];
      case 'awaiting_approval': return ['Needs your approval', 'waiting'];
      case 'completed': return review ? ['Needs review', 'review'] : ['Complete', 'done'];
      case 'failed': return ['Failed', 'failed'];
      case 'cancelled': return ['Cancelled', 'muted'];
      default: return [r.status, 'muted'];
    }
  }

  const isActive = (r) => r.status === 'queued' || r.status === 'running';

  function friendlyError(err) {
    const e = err || '';
    if (/401|invalid api key|invalid_api_key/i.test(e)) return 'The model provider rejected the API key. Check GROQ_API_KEY on the server, then retry.';
    if (/413|too large|tokens per minute/i.test(e)) return 'A request was larger than the model provider allows per minute. Retrying after a short wait usually works.';
    if (/tokens per day|\bTPD\b/i.test(e)) {
      const wait = e.match(/try again in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)/i);
      return 'The daily token allowance for this model is used up.' + (wait && wait[1] ? ` The provider says to try again in about ${wait[1].replace(/\.\d+s/, 's')}.` : '') + ' A paid tier removes this limit.';
    }
    if (/429|rate limit/i.test(e)) return 'The model provider is rate limiting this account. Wait a minute, then retry.';
    if (/BudgetExceeded/.test(e)) return 'The run used more than its token limit, so it was stopped to control cost.';
    if (/SourceShortfall/.test(e)) return 'Not enough verifiable sources were found. The literature search may be rate limited, or the topic may be too narrow. Retry in a minute, or try a broader topic.';
    if (/All providers failed/.test(e)) return 'Every configured model provider failed for one step.';
    return 'Something went wrong while working on this paper.';
  }

  /* ---------- login ---------- */

  const WAKING = 'The server is waking up. It sleeps when idle, so this can take up to a minute.';

  // A free host sleeps when idle. Ping /health as soon as the sign-in page opens, so the wake-up
  // starts while the visitor is still typing, and say so if it is slow.
  function warmUp(status) {
    const slow = setTimeout(() => { if (!status.textContent) status.textContent = WAKING; }, 2500);
    fetch(BASE + '/health', { cache: 'no-store' })
      .then((r) => { if (r.ok && status.textContent === WAKING) status.textContent = ''; })
      .catch(() => { if (status.textContent === WAKING) status.textContent = 'Cannot reach the server yet. Try again in a moment.'; })
      .finally(() => clearTimeout(slow));
  }

  function renderLogin(message) {
    stopTimers();
    const input = h('input', { type: 'password', id: 'tok', name: 'token', autocomplete: 'current-password', required: true, 'aria-describedby': 'tok-hint' });
    const err = h('p', { class: 'field-error', role: 'alert', text: message || '' });
    const btn = h('button', { class: 'btn primary', type: 'submit', text: 'Continue' });
    const status = h('p', { class: 'hint', role: 'status', 'aria-live': 'polite' });
    const form = h('form', { onsubmit: async (e) => {
      e.preventDefault();
      btn.disabled = true;
      btn.textContent = 'Signing in...';
      status.textContent = 'Contacting the server. If it was idle, waking it can take up to a minute.';
      err.textContent = '';
      token = input.value.trim();
      try {
        await api('/runs');
        store.set(token);
        start();
      } catch (ex) {
        if (ex.message !== 'unauthorized') {
          err.textContent = 'Could not reach the server: ' + ex.message;
          btn.disabled = false;
          btn.textContent = 'Continue';
          status.textContent = '';
        }
      }
    } },
      h('h1', { text: 'Research Desk' }),
      h('p', { class: 'lede', text: 'Turn an assignment into a cited research paper. You approve the topic and the thesis; verified sources and drafting happen in between.' }),
      h('div', { class: 'field' },
        h('label', { for: 'tok', text: 'Access token' }),
        input,
        h('p', { class: 'hint', id: 'tok-hint', text: 'Ask whoever runs this server. It is kept only until you close this tab.' }),
        err),
      btn,
      status);
    app.replaceChildren(h('main', { id: 'main', class: 'login' }, form));
    document.title = 'Sign in | Research Desk';
    input.focus();
    warmUp(status);
  }

  /* ---------- shell ---------- */

  function start() {
    const rail = h('aside', { class: 'rail', 'aria-label': 'Library' });
    const main = h('main', { id: 'main', tabindex: '-1' });
    S.rail = rail;
    S.main = main;
    app.replaceChildren(h('div', { class: 'shell' }, rail, main));
    renderRail();
    loadRuns();
    renderMain(true);
  }

  async function loadRuns() {
    clearTimeout(S.listTimer);
    if (!document.hidden || S.runs === null) {  // background tabs pause polling, but still load once
      try {
        S.runs = await api('/runs');
        renderRail();
      } catch (ex) {
        if (ex.message !== 'unauthorized' && !S.runs) toast('Could not load your papers: ' + ex.message, 'error');
      }
    }
    if (token) S.listTimer = setTimeout(loadRuns, 6000);
  }

  function renderRail() {
    if (!S.rail) return;
    const list = h('ul', { class: 'papers-list' });
    if (S.runs === null) {
      list.append(h('li', {}, h('div', { class: 'skel w80' }), h('div', { class: 'skel w55' })));
    } else if (!S.runs.length) {
      list.append(h('li', { class: 'empty-state' }, h('strong', { text: 'No papers yet' }), 'Start with an assignment brief.'));
    } else {
      for (const r of S.runs) {
        const [label, tone] = statusInfo(r);
        const title = r.topic || r.prompt;
        list.append(h('li', {},
          h('a', { class: 'paper-link', href: '#/run/' + r.id, 'aria-current': S.runId === r.id ? 'page' : null },
            h('span', { class: 'paper-title', text: title }),
            h('span', { class: 'paper-meta' }, h('span', { class: 'st-' + tone, text: label }), h('span', { text: relTime(r.updated_at) })))));
      }
    }
    const details = h('details', { class: 'papers' },
      h('summary', { text: S.runs ? `Papers (${S.runs.length})` : 'Papers' }), list);
    if (window.matchMedia('(min-width: 821px)').matches || S.papersOpen) details.open = true;
    details.addEventListener('toggle', () => { S.papersOpen = details.open; });
    const scroll = S.rail.querySelector('.papers-list');
    const top = scroll ? scroll.scrollTop : 0;
    S.rail.replaceChildren(
      h('div', { class: 'brand' }, 'Research Desk', h('small', { text: 'Cited papers, verified sources' })),
      h('a', { class: 'btn primary', href: '#/new', text: 'New paper' }),
      details,
      h('div', { class: 'rail-foot' }, h('button', { class: 'btn quiet', type: 'button', onclick: () => signOut(''), text: 'Sign out' })));
    const fresh = S.rail.querySelector('.papers-list');
    if (fresh) fresh.scrollTop = top;
  }

  /* ---------- routing ---------- */

  function route() {
    const m = location.hash.match(/^#\/run\/([\w-]+)/);
    return m ? { view: 'run', id: m[1] } : { view: 'new' };
  }

  window.addEventListener('hashchange', () => { if (token) renderMain(true); });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && token) { loadRuns(); if (S.runId) loadRun(S.runId); }
  });

  function renderMain(focus) {
    clearTimeout(S.runTimer);
    const r = route();
    S.runId = r.view === 'run' ? r.id : null;
    S.sig = '';
    S.run = null;
    renderRail();
    if (r.view === 'new') {
      S.main.replaceChildren(newView());
      document.title = 'New paper | Research Desk';
    } else {
      S.main.replaceChildren(h('div', { class: 'col' }, h('div', { class: 'skel w60 tall' }), h('div', { class: 'skel w100 block' })));
      loadRun(r.id, focus);
    }
    if (focus) S.main.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }

  /* ---------- new paper ---------- */

  const EXAMPLE = 'Write a 1,500 word research paper on how remote work affects employee productivity. Use APA citations. Required sections: Introduction, Literature Review, Analysis, Conclusion.';

  function newView() {
    const ta = h('textarea', { id: 'brief', rows: '11', maxlength: '20000', 'aria-describedby': 'brief-hint', placeholder: '' });
    const count = h('div', { class: 'counter', 'aria-hidden': 'true', text: '0 / 20,000' });
    const err = h('p', { class: 'field-error', role: 'alert' });
    const go = h('button', { class: 'btn primary', type: 'submit', disabled: true, text: 'Start research' });
    const sync = () => {
      const n = ta.value.trim().length;
      count.textContent = `${ta.value.length.toLocaleString()} / 20,000`;
      go.disabled = n < 10;
    };
    ta.addEventListener('input', sync);
    const form = h('form', { class: 'new-form', onsubmit: async (e) => {
      e.preventDefault();
      go.disabled = true;
      err.textContent = '';
      try {
        const run = await api('/runs', { method: 'POST', body: JSON.stringify({ prompt: ta.value.trim() }) });
        await loadRuns();
        location.hash = '#/run/' + run.id;
      } catch (ex) {
        if (ex.message !== 'unauthorized') { err.textContent = ex.message; go.disabled = false; }
      }
    } },
      h('div', { class: 'field' },
        h('label', { for: 'brief', text: 'Assignment or research question' }),
        h('p', { class: 'hint', id: 'brief-hint', text: 'Paste the brief. Include the word limit, citation style and any required sections.' }),
        ta, count, err),
      h('div', { class: 'form-actions' }, go,
        h('button', { class: 'btn quiet', type: 'button', text: 'Use an example brief', onclick: () => { ta.value = EXAMPLE; sync(); ta.focus(); } })));

    const how = h('section', { class: 'how', 'aria-labelledby': 'how-h' },
      h('h2', { id: 'how-h', text: 'What happens next' }),
      h('ol', {},
        ...[
          ['Reads the brief', 'Pulls out length, style and required sections.'],
          ['Proposes a topic', 'You approve it or ask for a different one.'],
          ['Finds and verifies sources', 'Each reference is matched to a Crossref record.'],
          ['Proposes a thesis', 'You approve or edit it before any writing.'],
          ['Outlines, drafts and reviews', 'Checks length, sections and citations, then revises.'],
        ].map(([t, d]) => h('li', {}, h('strong', { text: t }), h('span', { text: d })))));

    return h('div', { class: 'col' },
      h('header', {}, h('h1', { class: 'page-title', text: 'New paper' }),
        h('p', { class: 'lede', text: 'Start with the assignment. The first draft usually takes a few minutes.' })),
      h('div', { class: 'new-grid' }, form, how));
  }

  /* ---------- run view ---------- */

  async function loadRun(id, focus) {
    clearTimeout(S.runTimer);
    if (S.runId !== id) return;
    if (!document.hidden || !S.run) {  // background tabs pause polling, but still load once
      try {
        const run = await api('/runs/' + id);
        if (S.runId !== id) return;
        S.run = run;
        const sig = JSON.stringify([run.status, (run.progress || []).length, run.updated_at]);
        if (sig !== S.sig) {
          S.sig = sig;
          renderRun(focus);
          loadRuns();
        }
      } catch (ex) {
        if (S.runId !== id || ex.message === 'unauthorized') return;
        if (!S.run) {
          S.main.replaceChildren(h('div', { class: 'col' },
            h('h1', { class: 'page-title', text: 'Paper not found' }),
            h('p', { class: 'lede', text: 'It may have been removed, or the link is wrong.' }),
            h('a', { class: 'btn', href: '#/new', text: 'Start a new paper' })));
          return;
        }
        toast('Connection problem: ' + ex.message, 'error');
      }
    }
    const status = S.run && S.run.status;
    const delay = status === 'queued' || status === 'running' ? 2000 : status === 'awaiting_approval' ? 5000 : 0;
    if (delay) S.runTimer = setTimeout(() => loadRun(id), delay);
  }

  const STAGES = ['Brief', 'Topic', 'Sources', 'Thesis', 'Outline', 'Draft', 'Review'];
  const NODE_STAGE = { analyst: 0, propose_topic: 1, approve_topic: 1, scout: 2, propose_thesis: 3, approve_thesis: 3, planner: 4, writer: 5, critic: 6 };
  const NEXT_NODE = { analyst: 'propose_topic', propose_topic: 'approve_topic', approve_topic: 'scout', scout: 'propose_thesis', propose_thesis: 'approve_thesis', approve_thesis: 'planner', planner: 'writer', writer: 'critic', critic: 'writer' };

  function stageStates(r) {
    const states = STAGES.map(() => 'todo');
    const caps = STAGES.map(() => '');
    const prog = r.progress || [];
    if (r.status === 'completed') return { states: states.map(() => 'done'), caps };
    let active = 0;
    if (r.status === 'awaiting_approval' && r.gate) active = r.gate.gate === 'topic' ? 1 : 3;
    else if (prog.length) {
      const last = prog[prog.length - 1].node;
      active = NODE_STAGE[NEXT_NODE[last] || last] ?? 0;
    }
    for (let i = 0; i < active; i++) states[i] = 'done';
    const drafts = prog.filter((p) => p.node === 'writer').length;
    if (r.status === 'awaiting_approval') { states[active] = 'waiting'; caps[active] = 'Waiting for you'; }
    else if (r.status === 'failed') { states[active] = 'failed'; caps[active] = 'Stopped'; }
    else if (r.status === 'cancelled') { caps[active] = 'Cancelled'; }
    else {
      states[active] = 'active';
      caps[active] = r.status === 'queued' && !prog.length ? 'Queued' : drafts > 1 && active >= 5 ? `Revision ${drafts - 1}` : 'In progress';
    }
    return { states, caps };
  }

  function renderRun(focus) {
    const r = S.run;
    const [label, tone] = statusInfo(r);
    const res = r.result || {};
    const title = titleFor(r);
    document.title = title.slice(0, 60) + ' | Research Desk';

    const { states, caps } = stageStates(r);
    const stages = h('ol', { class: 'stages', 'aria-label': 'Progress' },
      ...STAGES.map((name, i) => h('li', { 'data-state': states[i], 'aria-current': states[i] === 'active' || states[i] === 'waiting' ? 'step' : null },
        h('span', { class: 'name', text: name }), h('span', { class: 'cap', text: caps[i] }))));

    const canCancel = ['queued', 'running', 'awaiting_approval'].includes(r.status);
    const canDelete = ['completed', 'failed', 'cancelled', 'awaiting_approval'].includes(r.status);
    const actions = h('div', { class: 'run-actions' },
      r.status === 'failed' && h('button', { class: 'btn primary', type: 'button', text: 'Retry', onclick: () => act('retry') }),
      canCancel && h('button', { class: 'btn danger', type: 'button', text: 'Cancel run', onclick: () => {
        if (confirm('Cancel this run? Work done so far is kept, but it will not continue.')) act('cancel');
      } }),
      canDelete && h('button', { class: 'btn danger', type: 'button', text: 'Delete paper', onclick: removeRun }));

    const head = h('header', { class: 'run-head' },
      h('h1', { class: 'run-title', id: 'run-title', tabindex: '-1', text: title }),
      h('div', { class: 'status-line', 'aria-live': 'polite' },
        h('strong', { class: 'st-' + tone, text: label }),
        h('span', { text: 'Updated ' + relTime(r.updated_at) }),
        actions),
      h('details', { class: 'assignment' }, h('summary', { text: 'Assignment' }), h('p', { text: r.prompt })));

    const parts = [head, stages];
    if (r.status === 'awaiting_approval' && r.gate) parts.push(gatePanel(r));
    if (r.status === 'completed') parts.push(...completedParts(r));
    if (r.status === 'failed') parts.push(...failedParts(r));
    if (r.status === 'cancelled') parts.push(h('p', { class: 'lede', text: 'This run was cancelled. Anything produced before that is kept in the activity log.' }));
    parts.push(activity(r));
    S.main.replaceChildren(h('div', { class: 'col' }, ...parts));
    if (focus) document.getElementById('run-title')?.focus({ preventScroll: true });
  }

  async function act(what, body) {
    try {
      S.run = await api(`/runs/${S.runId}/${what}`, { method: 'POST', body: body ? JSON.stringify(body) : undefined });
      S.sig = '';
      loadRun(S.runId);
      loadRuns();
    } catch (ex) {
      if (ex.message !== 'unauthorized') toast(ex.message, 'error');
      throw ex;
    }
  }

  async function removeRun() {
    const id = S.runId;
    if (!confirm('Delete this paper permanently? The draft, activity log and saved progress are removed. This cannot be undone.')) return;
    try {
      await api('/runs/' + id, { method: 'DELETE' });
    } catch (ex) {
      if (ex.message !== 'unauthorized') toast(ex.message, 'error');
      return;
    }
    S.runs = (S.runs || []).filter((r) => r.id !== id);
    toast('Paper deleted.');
    location.hash = '#/new';
    loadRuns();
  }

  /* ---------- approval gates ---------- */

  function gatePanel(r) {
    const g = r.gate;
    const isTopic = g.gate === 'topic';
    const field = h('textarea', { id: 'gate-text', rows: isTopic ? '3' : '5' }, '');
    field.value = isTopic ? g.topic : g.thesis;
    const fb = h('textarea', { id: 'gate-fb', rows: '3' });
    const fbBox = h('div', { class: 'field', hidden: true },
      h('label', { for: 'gate-fb', text: 'What should change?' }), fb);
    const approve = h('button', { class: 'btn primary', type: 'button', text: isTopic ? 'Approve topic' : 'Approve thesis' });
    const decline = h('button', { class: 'btn', type: 'button', text: isTopic ? 'Ask for a different topic' : 'Ask for a different thesis' });
    const send = h('button', { class: 'btn primary', type: 'button', hidden: true, text: 'Send feedback' });
    const all = [approve, decline, send];
    const busy = (b) => all.forEach((x) => { x.disabled = b; });

    approve.addEventListener('click', async () => {
      busy(true);
      try { await act('approve', { approved: true, [isTopic ? 'topic' : 'thesis']: field.value.trim() }); } catch { busy(false); }
    });
    decline.addEventListener('click', () => { fbBox.hidden = false; send.hidden = false; decline.hidden = true; fb.focus(); });
    send.addEventListener('click', async () => {
      if (!fb.value.trim()) { toast('Say what should change so the next proposal improves.', 'error'); fb.focus(); return; }
      busy(true);
      try { await act('approve', { approved: false, feedback: fb.value.trim() }); } catch { busy(false); }
    });

    return h('section', { class: 'panel', 'aria-labelledby': 'gate-h' },
      h('h2', { id: 'gate-h', text: isTopic ? 'Approve the topic' : 'Approve the thesis' }),
      h('p', { class: 'hint', text: isTopic
        ? 'Sources are searched for this topic. Narrow it here if it is too broad.'
        : 'The paper will argue this. Edit it now, because everything after this builds on it.' }),
      h('div', { class: 'field' }, h('label', { for: 'gate-text', text: isTopic ? 'Topic' : 'Thesis statement' }), field),
      isTopic && g.queries && g.queries.length ? h('div', {}, h('h3', { class: 'section-h', text: 'Search queries' }),
        h('ul', { class: 'query-list' }, ...g.queries.map((q) => h('li', { text: q })))) : null,
      fbBox,
      h('div', { class: 'actions' }, approve, decline, send));
  }

  /* ---------- results ---------- */

  function splitPaper(md) {
    const [body] = md.split(/\n## References/);
    const refs = md.includes('\n## References') ? md.split(/\n## References/)[1].split(/\n{2,}/).map((s) => s.trim()).filter(Boolean) : [];
    return { body, refs };
  }

  function completedParts(r) {
    const res = r.result || {};
    const md = res.draft || '';
    const { body, refs } = splitPaper(md);
    const ok = res.approved;
    const issues = res.open_issues || [];
    const verdict = h('section', { class: 'verdict ' + (ok ? 'ok' : 'review') },
      h('h2', { text: ok ? 'Ready to read' : 'Needs your review' }),
      h('p', { text: ok
        ? 'The reviewer approved this draft.'
        : 'The reviewer still had concerns when the revision limit was reached. Check them before you rely on this draft.' }),
      !ok && issues.length ? h('ul', {}, ...issues.map((i) => h('li', { text: i }))) : null,
      res.note ? h('p', { text: res.note }) : null,
      h('p', { class: 'caveat', text: 'Every reference was matched to a Crossref record. Whether each sentence is faithful to its source has not been proven, so check quotes and numbers against the original papers.' }));

    const facts = h('dl', { class: 'facts' },
      ...[
        ['Words', words(body).toLocaleString()],
        ['References', String(refs.length)],
        ['Sources found', String(res.verified_sources ?? 0)],
        ['Drafts', String(res.revision_count ?? 0)],
        ['Tokens', (res.tokens_used ?? 0).toLocaleString()],
      ].map(([k, v]) => h('div', {}, h('dt', { text: k }), h('dd', { text: v }))));

    const title = res.topic || 'Paper';
    const toolbar = h('div', { class: 'toolbar' },
      h('button', { class: 'btn', type: 'button', text: 'Copy Markdown', onclick: async () => {
        try { await navigator.clipboard.writeText(md); toast('Copied to clipboard.'); } catch { toast('Copy was blocked by the browser.', 'error'); }
      } }),
      h('button', { class: 'btn', type: 'button', text: 'Download .md', onclick: () => {
        const url = URL.createObjectURL(new Blob([md], { type: 'text/markdown' }));
        const a = h('a', { href: url, download: slug(title) + '.md' });
        document.body.append(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } }),
      h('button', { class: 'btn', type: 'button', text: 'Print or save as PDF', onclick: () => window.print() }));

    return [verdict, facts, h('div', { class: 'no-print toolbar' }, toolbar), md ? paper(title, res, md) : null];
  }

  function failedParts(r) {
    const res = r.result || {};
    const verdict = h('section', { class: 'verdict fail' },
      h('h2', { text: 'This run stopped' }),
      h('p', { text: friendlyError(r.error) }),
      h('p', { class: 'caveat', text: 'Progress is saved. Retry continues from the last finished step, not from the beginning.' }),
      r.error ? h('details', { class: 'tech' }, h('summary', { text: 'Technical detail' }), h('pre', { class: 'err', text: r.error })) : null);
    return [verdict, res.draft ? h('h2', { class: 'section-h', text: 'Partial draft' }) : null, res.draft ? paper(res.topic || 'Partial draft', res, res.draft) : null];
  }

  /* ---------- manuscript rendering ---------- */

  function inline(text) {
    const out = [];
    const re = /(\*[^*\n]+\*|https?:\/\/[^\s]*[^\s.,;:)])/g;
    let last = 0;
    let m;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const t = m[0];
      out.push(t.startsWith('*') ? h('em', { text: t.slice(1, -1) }) : h('a', { href: t, target: '_blank', rel: 'noopener noreferrer', text: t }));
      last = m.index + t.length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  function paper(title, res, md) {
    const art = h('article', { class: 'paper' }, h('h1', { text: title }),
      h('p', { class: 'byline', text: 'Draft for review' }));
    for (const part of md.split(/^## /m).slice(1)) {
      const nl = part.indexOf('\n');
      const name = (nl === -1 ? part : part.slice(0, nl)).trim();
      const blocks = (nl === -1 ? '' : part.slice(nl + 1)).split(/\n{2,}/).map((s) => s.trim()).filter(Boolean);
      art.append(h('h2', { text: name }));
      for (const b of blocks) art.append(h('p', { class: name === 'References' ? 'ref' : null }, ...inline(b)));
    }
    return h('div', { class: 'sheet' }, art);
  }

  /* ---------- activity ---------- */

  const LABELS = { analyst: 'Brief', strategist: 'Topic', gate: 'Approval', scout: 'Sources', thesis: 'Thesis', planner: 'Outline', writer: 'Draft', critic: 'Review' };

  function tidy(text) {
    const m = text.match(/^hard issues \[(.*)\]$/);
    return m ? 'Automatic checks failed: ' + m[1].replace(/', '/g, '; ').replace(/^'|'$/g, '').replace(/\\'/g, "'") : text;
  }

  function activity(r) {
    const rows = [];
    for (const p of r.progress || []) {
      for (const line of p.log || []) {
        const i = line.indexOf(': ');
        const key = i === -1 ? '' : line.slice(0, i);
        rows.push({ at: p.at, label: LABELS[key] || sentence(key), text: sentence(tidy(i === -1 ? line : line.slice(i + 2))) });
      }
    }
    const list = h('ul', { class: 'activity' });
    if (!rows.length) list.append(h('li', { class: 'empty', text: r.status === 'queued' ? 'Waiting for a worker to pick this up.' : 'Nothing has finished yet.' }));
    for (const row of rows.reverse()) {
      list.append(h('li', {}, h('time', { text: clock(row.at) }), h('span', {}, h('b', { text: row.label }), row.text)));
    }
    return h('section', { 'aria-labelledby': 'act-h', class: 'no-print' }, h('h2', { class: 'section-h', id: 'act-h', text: 'Activity' }), list);
  }

  /* ---------- boot ---------- */

  if (token) start();
  else renderLogin('');
})();
