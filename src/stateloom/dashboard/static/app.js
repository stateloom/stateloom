// StateLoom Dashboard — Vanilla JS (no build step)

const API_BASE = '/api';
let ws = null;
let currentView = null;
let currentDetailSessionId = null;
let _sessionBillingMode = 'api';  // billing mode of the currently displayed session detail
let _toolStepCache = {};  // sessionId → { parentStep → [events] } for lazy-loaded tool sub-steps

// --- Pagination State ---
let _sessionsPage = 0;
const _sessionsPageSize = 50;
let _sessionsTotal = 0;
let _detailEventsLoaded = [];     // accumulated events for current session detail
let _detailEventsOffset = 0;      // current offset for event loading
let _detailEventsHasMore = false;  // whether more events are available
let _detailEventsPageSize = parseInt(localStorage.getItem('stateloom_detailEventsPageSize')) || 200;
const _toolStepPageSize = 20;

function _onPageSizeChange(value) {
    const newSize = parseInt(value, 10);
    if (isNaN(newSize) || newSize === _detailEventsPageSize) return;
    _detailEventsPageSize = newSize;
    localStorage.setItem('stateloom_detailEventsPageSize', String(newSize));
    if (currentDetailSessionId) {
        loadSessionDetail(currentDetailSessionId);
    }
}

// --- Toast Notifications ---
const _toastContainer = (() => {
    let el = document.getElementById('toast-container');
    if (!el) {
        el = document.createElement('div');
        el.id = 'toast-container';
        el.style.cssText =
            'position:fixed;top:16px;right:16px;z-index:10000;' +
            'display:flex;flex-direction:column;gap:8px;max-width:400px;';
        document.body.appendChild(el);
    }
    return el;
})();

function showToast(message, type = 'error', duration = 4000) {
    const toast = document.createElement('div');
    const colors = {
        error: 'background:#dc3545;color:#fff;',
        success: 'background:#28a745;color:#fff;',
        warning: 'background:#ffc107;color:#212529;',
        info: 'background:#17a2b8;color:#fff;',
    };
    toast.style.cssText =
        (colors[type] || colors.info) +
        'padding:10px 16px;border-radius:6px;font-size:13px;' +
        'box-shadow:0 4px 12px rgba(0,0,0,.3);opacity:0;' +
        'transition:opacity .2s;cursor:pointer;';
    toast.textContent = message;
    toast.addEventListener('click', () => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 200);
    });
    _toastContainer.appendChild(toast);
    requestAnimationFrame(() => { toast.style.opacity = '1'; });
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 200);
    }, duration);
}

// --- Navigation ---
document.querySelectorAll('[data-view]').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        switchView(link.dataset.view);
    });
});

function switchView(view) {
    currentView = view;
    // Don't overwrite hash for observability — loadObservability sets it with window param
    // Don't overwrite hash if it already starts with this view (preserves sub-tab suffix)
    const curHash = location.hash ? location.hash.slice(1).split('?')[0] : '';
    if (view !== 'observability' && view !== 'agent-detail') location.hash = view;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('[data-view]').forEach(l => l.classList.remove('active'));

    const viewEl = document.getElementById(`view-${view}`);
    const linkEl = document.querySelector(`[data-view="${view}"]`);
    if (viewEl) viewEl.classList.add('active');
    if (linkEl) linkEl.classList.add('active');

    // Load view data
    if (view === 'overview') loadOverview();
    else if (view === 'sessions') { _sessionsPage = 0; loadSessions(); }
    else if (view === 'models') loadModels();
    else if (view === 'experiments') loadExperiments();
    else if (view === 'compliance') loadCompliance();
    else if (view === 'security') loadSecurity();
    else if (view === 'observability') loadObservability();
    else if (view === 'agents') loadAgents();
    else if (view === 'jobs') loadJobs();
    else if (view === 'consensus') loadConsensus();
    else if (view === 'settings') loadSettings();
    else if (view === 'server-logs') loadServerLogs();
}

// --- WebSocket ---
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        document.getElementById('ws-status').textContent = 'Connected';
        document.getElementById('ws-status').className = 'ws-connected';
    };

    ws.onclose = () => {
        document.getElementById('ws-status').textContent = 'Disconnected';
        document.getElementById('ws-status').className = 'ws-disconnected';
        // Reconnect after 3s
        setTimeout(connectWebSocket, 3000);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'new_event') {
            const data = msg.data;
            if (data && data.type === 'hot_swap_progress') {
                updateHotSwapUI(data);
            } else {
                addEventToStream(data);
                if (currentView === 'overview') loadOverview();
            }
        } else if (msg.type === 'session_update') {
            if (currentView === 'sessions') loadSessions();
        } else if (msg.type === 'job_update') {
            if (currentView === 'jobs') loadJobs();
        }
    };
}

// --- Data Loading ---
async function fetchJSON(path, options) {
    try {
        const res = await fetch(`${API_BASE}${path}`, options);
        if (!res.ok) {
            const body = await res.text();
            let detail = '';
            try { detail = JSON.parse(body).detail || body; } catch { detail = body; }
            showToast(`API error (${res.status}): ${detail}`, 'error');
            return null;
        }
        return await res.json();
    } catch (err) {
        showToast(`Network error: ${err.message}`, 'error');
        return null;
    }
}

/** Like fetchJSON but silently returns null on errors (no toast). */
async function fetchJSONSilent(path, options) {
    try {
        const res = await fetch(`${API_BASE}${path}`, options);
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

async function loadOverview() {
    const stats = await fetchJSON('/stats');
    if (!stats) return;
    document.getElementById('stat-total-cost').textContent = `$${(stats.total_cost || 0).toFixed(4)}`;
    document.getElementById('stat-active-sessions').textContent = stats.active_sessions || 0;
    document.getElementById('stat-total-calls').textContent = stats.total_calls || 0;
    document.getElementById('stat-cache-savings').textContent = `$${(stats.total_cache_savings || 0).toFixed(4)}`;
    document.getElementById('stat-cloud-calls').textContent = stats.cloud_calls || 0;
    document.getElementById('stat-local-calls').textContent = stats.local_calls || 0;
    document.getElementById('stat-pii-detections').textContent = stats.total_pii_detections || 0;
    document.getElementById('stat-guardrail-detections').textContent = stats.total_guardrail_detections || 0;
    document.getElementById('stat-cache-hits').textContent = stats.total_cache_hits || 0;
}

async function loadSessions() {
    const offset = _sessionsPage * _sessionsPageSize;
    const [data, orgsData, teamsData] = await Promise.all([
        fetchJSON(`/sessions?limit=${_sessionsPageSize}&offset=${offset}`),
        fetchJSON('/organizations'),
        fetchJSON('/teams'),
    ]);
    if (!data) return;
    const tbody = document.getElementById('sessions-tbody');
    tbody.innerHTML = '';

    _sessionsTotal = data.total || 0;
    _updateSessionsPagination();

    if (!data.sessions || data.sessions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="empty-state">No sessions yet</td></tr>';
        return;
    }

    // Build name lookup maps
    const orgNames = {};
    const teamNames = {};
    if (orgsData && orgsData.organizations) {
        orgsData.organizations.forEach(o => { orgNames[o.id] = o.name || o.id; });
    }
    if (teamsData && teamsData.teams) {
        teamsData.teams.forEach(t => { teamNames[t.id] = t.name || t.id; });
    }

    data.sessions.forEach(s => {
        const tr = document.createElement('tr');
        tr.onclick = () => loadSessionDetail(s.id);
        const isReplay = s.id.startsWith('replay-');
        const replayBadge = isReplay ? ' <span class="replay-badge">REPLAY</span>' : '';
        const meta = s.metadata || {};
        const isAgent = meta.agent_slug || s.id.startsWith('agent-');
        const typeBadge = isAgent ? ' <span class="replay-badge" style="background:#6f42c1;color:#fff;">AGENT</span>' : '';
        const orgLabel = s.org_id ? escapeHtml(orgNames[s.org_id] || s.org_id) : '<span class="text-muted">\u2014</span>';
        const teamLabel = s.team_id ? escapeHtml(teamNames[s.team_id] || s.team_id) : '<span class="text-muted">\u2014</span>';
        const sBilling = s.billing_mode || (s.metadata || {}).billing_mode || 'api';
        const sCostCell = sBilling === 'subscription' ? '\u2014' : '$' + s.total_cost.toFixed(4);
        tr.innerHTML = `
            <td>${s.name ? '<strong>' + escapeHtml(s.name) + '</strong><br><span style="font-size:11px;color:#888;">' + escapeHtml(s.id) + '</span>' : escapeHtml(s.id)}${replayBadge}${typeBadge}</td>
            <td><span class="status-badge status-${escapeHtml(s.status)}">${escapeHtml(s.status)}</span></td>
            <td>${sCostCell}</td>
            <td>${s.total_tokens.toLocaleString()}</td>
            <td>${s.call_count}</td>
            <td>${s.cache_hits}</td>
            <td>${s.pii_detections}</td>
            <td>${orgLabel}</td>
            <td>${teamLabel}</td>
            <td>${new Date(s.started_at).toLocaleDateString([], {month:'short', day:'numeric'})} ${new Date(s.started_at).toLocaleTimeString()}</td>
        `;
        tbody.appendChild(tr);
    });
}

function _updateSessionsPagination() {
    const totalPages = Math.max(1, Math.ceil(_sessionsTotal / _sessionsPageSize));
    const currentPage = _sessionsPage + 1;
    const infoEl = document.getElementById('sessions-page-info');
    const prevBtn = document.getElementById('sessions-prev-btn');
    const nextBtn = document.getElementById('sessions-next-btn');
    if (infoEl) infoEl.textContent = `Page ${currentPage} of ${totalPages}`;
    if (prevBtn) prevBtn.disabled = _sessionsPage <= 0;
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages;
}

function sessionsPagePrev() {
    if (_sessionsPage > 0) {
        _sessionsPage--;
        loadSessions();
    }
}

function sessionsPageNext() {
    const totalPages = Math.ceil(_sessionsTotal / _sessionsPageSize);
    if (_sessionsPage + 1 < totalPages) {
        _sessionsPage++;
        loadSessions();
    }
}

function _updateEventsLoadMore() {
    const el = document.getElementById('events-load-more');
    if (el) el.style.display = _detailEventsHasMore ? 'flex' : 'none';
}

async function loadMoreEvents() {
    if (!currentDetailSessionId || !_detailEventsHasMore) return;
    const btn = document.querySelector('#events-load-more .load-more-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Loading\u2026'; }
    try {
        const eventsData = await fetchJSON(
            `/sessions/${encodeURIComponent(currentDetailSessionId)}/events?limit=${_detailEventsPageSize}&offset=${_detailEventsOffset}&exclude_types=compliance_audit&primary_only=true`
        );
        if (!eventsData) return;
        const newEvents = eventsData.events || [];
        _detailEventsLoaded = _detailEventsLoaded.concat(newEvents);
        _detailEventsOffset += newEvents.length;
        _detailEventsHasMore = !!eventsData.has_more;
        _updateEventsLoadMore();

        // Re-render flat events
        const container = document.getElementById('detail-events');
        if (container) {
            container.innerHTML = '';
            _detailEventsLoaded.forEach(e => {
                container.appendChild(createEventLine(e));
            });
        }

        // Re-render waterfall and step timeline
        renderWaterfallTimeline(_detailEventsLoaded, null);
        renderStepTimeline(_detailEventsLoaded);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Load More Events'; }
    }
}

async function loadSessionDetail(sessionId) {
    currentDetailSessionId = sessionId;
    _toolStepCache = {};  // clear tool sub-step cache on session switch
    // Encode session ID in URL for refresh/back navigation
    location.hash = 'session-detail/' + encodeURIComponent(sessionId);
    currentView = 'session-detail';
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('[data-view]').forEach(l => l.classList.remove('active'));
    const viewEl = document.getElementById('view-session-detail');
    if (viewEl) viewEl.classList.add('active');
    const pageSizeSelect = document.getElementById('events-page-size');
    if (pageSizeSelect) pageSizeSelect.value = String(_detailEventsPageSize);
    document.getElementById('detail-session-id').textContent = sessionId;

    const session = await fetchJSON(`/sessions/${sessionId}`);
    if (!session) return;

    // Session name
    const nameEl = document.getElementById('detail-session-name');
    if (session.name) {
        nameEl.textContent = session.name + ' — ';
        nameEl.style.display = '';
    } else {
        nameEl.style.display = 'none';
    }

    // Status
    document.getElementById('detail-status').innerHTML =
        `<span class="status-badge status-${escapeHtml(session.status)}">${escapeHtml(session.status)}</span>`;

    // Close Session button visibility
    const closeBtn = document.getElementById('close-session-btn');
    if (closeBtn) {
        closeBtn.style.display = (session.status === 'active' || session.status === 'suspended') ? '' : 'none';
    }

    // Cost & tokens
    const billingMode = session.billing_mode || (session.metadata || {}).billing_mode || 'api';
    _sessionBillingMode = billingMode;
    const isSub = billingMode === 'subscription';
    const costLabel = document.getElementById('detail-cost-label');
    const costSub = document.getElementById('detail-cost-sub');
    if (isSub) {
        costLabel.textContent = 'Total Cost';
        document.getElementById('detail-cost').textContent = '\u2014';
    } else {
        costLabel.textContent = 'Total Cost';
        document.getElementById('detail-cost').textContent = `$${session.total_cost.toFixed(4)}`;
    }
    costSub.style.display = 'none';
    document.getElementById('detail-tokens').textContent = session.total_tokens.toLocaleString();
    document.getElementById('detail-prompt-tokens').textContent = (session.total_prompt_tokens || 0).toLocaleString();
    document.getElementById('detail-completion-tokens').textContent = (session.total_completion_tokens || 0).toLocaleString();
    document.getElementById('detail-calls').textContent = session.call_count;
    document.getElementById('detail-cache-hits').textContent = session.cache_hits || 0;
    document.getElementById('detail-cache-savings').textContent = isSub ? '\u2014' : `$${(session.cache_savings || 0).toFixed(4)}`;
    document.getElementById('detail-pii-detections').textContent = session.pii_detections || 0;
    document.getElementById('detail-guardrail-detections').textContent = session.guardrail_detections || 0;


    // Timeout display
    const timeoutCard = document.getElementById('detail-timeout-card');
    if (session.timeout != null || session.idle_timeout != null) {
        timeoutCard.style.display = '';
        const parts = [];
        if (session.timeout != null) parts.push(session.timeout + 's session');
        if (session.idle_timeout != null) parts.push(session.idle_timeout + 's idle');
        document.getElementById('detail-timeout').textContent = parts.join(' / ');
    } else {
        timeoutCard.style.display = 'none';
    }

    // Org & Team — fetch once and reuse for budget bars below
    const detailOrgEl = document.getElementById('detail-org');
    const detailTeamEl = document.getElementById('detail-team');
    let _orgData = null;
    let _teamData = null;
    if (session.org_id) {
        _orgData = await fetchJSONSilent(`/organizations/${encodeURIComponent(session.org_id)}`);
        detailOrgEl.textContent = _orgData && _orgData.name ? _orgData.name : session.org_id;
    } else {
        detailOrgEl.innerHTML = '<span class="text-muted">\u2014</span>';
    }
    if (session.team_id) {
        _teamData = await fetchJSONSilent(`/teams/${encodeURIComponent(session.team_id)}`);
        detailTeamEl.textContent = _teamData && _teamData.name ? _teamData.name : session.team_id;
    } else {
        detailTeamEl.innerHTML = '<span class="text-muted">\u2014</span>';
    }

    // Duration — removed (session started_at/ended_at don't reflect true
    // wall-clock span for proxy sessions like Gemini CLI)
    const durationEl = document.getElementById('detail-duration');
    durationEl.textContent = '';

    // Experiment badge
    const expBadge = document.getElementById('detail-experiment-badge');
    if (session.experiment_id) {
        document.getElementById('detail-exp-id').textContent = session.experiment_id;
        document.getElementById('detail-exp-variant').textContent = session.variant || '-';
        expBadge.style.display = 'flex';
    } else {
        expBadge.style.display = 'none';
    }

    // Budget bar — session, team, and org levels
    // Hide budget bars for subscription sessions (cost tracking is incomplete)
    const budgetCard = document.getElementById('detail-budget-card');
    const hasBudget = !isSub && (session.budget || session.org_id || session.team_id);
    if (hasBudget) {
        // Session budget
        if (session.budget) {
            const rawPct = (session.total_cost / session.budget) * 100;
            const pct = Math.min(rawPct, 100);
            document.getElementById('detail-budget-text').textContent =
                `$${session.total_cost.toFixed(4)} / $${session.budget.toFixed(4)}`;
            const bar = document.getElementById('detail-budget-progress');
            bar.style.width = pct + '%';
            bar.className = rawPct > 90 ? 'budget-progress budget-danger' : 'budget-progress';
            document.getElementById('detail-budget-session-row').style.display = 'block';
        } else {
            document.getElementById('detail-budget-session-row').style.display = 'none';
        }
        // Team budget (reuse _teamData fetched above)
        const teamBudgetRow = document.getElementById('detail-budget-team-row');
        if (_teamData && _teamData.budget) {
            const teamCost = (_teamData.stats && _teamData.stats.total_cost) || _teamData.total_cost || 0;
            const teamRawPct = (teamCost / _teamData.budget) * 100;
            const teamPct = Math.min(teamRawPct, 100);
            document.getElementById('detail-budget-team-text').textContent =
                `$${teamCost.toFixed(6)} / $${_teamData.budget.toFixed(6)}  (${_teamData.name || session.team_id})`;
            const teamBar = document.getElementById('detail-budget-team-progress');
            teamBar.style.width = teamPct + '%';
            teamBar.className = teamRawPct > 90 ? 'budget-progress budget-danger' : 'budget-progress';
            teamBudgetRow.style.display = 'block';
        } else {
            teamBudgetRow.style.display = 'none';
        }
        // Org budget (reuse _orgData fetched above)
        const orgBudgetRow = document.getElementById('detail-budget-org-row');
        if (_orgData && _orgData.budget) {
            const orgCost = (_orgData.stats && _orgData.stats.total_cost) || _orgData.total_cost || 0;
            const orgRawPct = (orgCost / _orgData.budget) * 100;
            const orgPct = Math.min(orgRawPct, 100);
            document.getElementById('detail-budget-org-text').textContent =
                `$${orgCost.toFixed(6)} / $${_orgData.budget.toFixed(6)}  (${_orgData.name || session.org_id})`;
            const orgBar = document.getElementById('detail-budget-org-progress');
            orgBar.style.width = orgPct + '%';
            orgBar.className = orgRawPct > 90 ? 'budget-progress budget-danger' : 'budget-progress';
            orgBudgetRow.style.display = 'block';
        } else {
            orgBudgetRow.style.display = 'none';
        }
        budgetCard.style.display = 'block';
    } else {
        budgetCard.style.display = 'none';
    }

    // Replay source detection (replay sessions have IDs like "replay-<original-id>")
    const replaySourceEl = document.getElementById('detail-replay-source');
    const isReplaySession = sessionId.startsWith('replay-');
    if (isReplaySession) {
        const originalId = sessionId.replace(/^replay-/, '');
        replaySourceEl.innerHTML = `Replay of: <a onclick="loadSessionDetail('${escapeHtml(originalId)}')">${escapeHtml(originalId)}</a>`;
        replaySourceEl.style.display = 'block';
    } else {
        replaySourceEl.style.display = 'none';
    }

    // Events (fetch first — needed for replay panel step count)
    // Exclude compliance_audit events — they're internal bookkeeping and inflate the count.
    // primary_only=true filters out tool-continuation and CLI-internal LLM calls server-side,
    // attaching aggregated _tool_summary to parent events for lazy expansion.
    _detailEventsOffset = 0;
    _detailEventsLoaded = [];
    _detailEventsHasMore = false;
    const eventsData = await fetchJSON(`/sessions/${sessionId}/events?limit=${_detailEventsPageSize}&offset=0&exclude_types=compliance_audit&primary_only=true`);
    if (!eventsData) return;
    _detailEventsLoaded = eventsData.events || [];
    _detailEventsOffset = _detailEventsLoaded.length;
    _detailEventsHasMore = !!eventsData.has_more;
    _updateEventsLoadMore();
    const events = _detailEventsLoaded;
    const container = document.getElementById('detail-events');
    container.innerHTML = '';
    events.forEach(e => {
        container.appendChild(createEventLine(e));
    });

    // Recompute header stats from the fetched events instead of the session
    // object.  For active sessions, the session object may be stale (last saved
    // state), while the events list contains all persisted events including any
    // that arrived after the session was last saved.  This avoids mismatches
    // between header totals and waterfall/tool-summary totals.
    // Also exclude CLI-internal calls (quota checks, token counting) which
    // inflate totals (e.g. a simple "hey there" appears as 16K tokens).
    {
        let evPrompt = 0, evCompletion = 0, evCost = 0, evCalls = 0;
        for (const e of events) {
            if (e.event_type !== 'llm_call') continue;
            if (e.is_cli_internal) continue;  // skip hidden CLI overhead
            evPrompt += e.prompt_tokens || 0;
            evCompletion += e.completion_tokens || 0;
            evCost += e.cost || 0;
            evCalls++;
            // Include tool-continuation aggregates from server-provided _tool_summary
            if (e._tool_summary) {
                evCost += e._tool_summary.total_cost || 0;
                // total_tokens includes both prompt+completion combined
                const toolTokens = e._tool_summary.total_tokens || 0;
                evCompletion += toolTokens;  // approximate — no prompt/completion split available
            }
        }
        document.getElementById('detail-tokens').textContent = (evPrompt + evCompletion).toLocaleString();
        document.getElementById('detail-prompt-tokens').textContent = evPrompt.toLocaleString();
        document.getElementById('detail-completion-tokens').textContent = evCompletion.toLocaleString();
        document.getElementById('detail-calls').textContent = evCalls;
        document.getElementById('detail-cost').textContent = isSub ? '\u2014' : `$${evCost.toFixed(4)}`;
    }

    // Retry stats
    const retryEvents = events.filter(e => e.event_type === 'semantic_retry' && !e.resolved);
    const retriesEl = document.getElementById('detail-retries');
    retriesEl.textContent = retryEvents.length;
    retriesEl.style.color = retryEvents.length > 0 ? 'var(--danger)' : '';

    // Waterfall trace timeline
    renderWaterfallTimeline(events, session);

    // Step timeline (flat view, hidden by default)
    renderStepTimeline(events);

    // Parent link
    const parentLinkEl = document.getElementById('detail-parent-link');
    if (session.parent_session_id) {
        const anchor = document.getElementById('detail-parent-anchor');
        anchor.textContent = session.parent_session_id;
        anchor.onclick = () => loadSessionDetail(session.parent_session_id);
        parentLinkEl.style.display = 'inline-flex';
    } else {
        parentLinkEl.style.display = 'none';
    }

    // Children panel + flame graph
    const childrenPanel = document.getElementById('detail-children-panel');
    try {
        const childrenData = await fetchJSON(`/sessions/${sessionId}/children`);
        if (childrenData && childrenData.children && childrenData.children.length > 0) {
            document.getElementById('detail-children-count').textContent = `(${childrenData.total})`;
            renderFlameGraph(session, childrenData.children);
            renderChildrenList(childrenData.children);
            childrenPanel.style.display = 'block';
        } else {
            childrenPanel.style.display = 'none';
        }
    } catch (e) {
        childrenPanel.style.display = 'none';
    }

    // Feedback
    loadFeedback(sessionId);
}

// --- Flame Graph & Children Panel ---

const _FLAME_CHILD_COLORS = [
    'rgba(34, 211, 168, 0.45)',   // green
    'rgba(183, 148, 246, 0.45)',  // purple
    'rgba(240, 180, 41, 0.45)',   // yellow
    'rgba(236, 72, 153, 0.45)',   // pink
    'rgba(34, 211, 238, 0.45)',   // cyan
    'rgba(245, 101, 101, 0.45)',  // red
];

function renderFlameGraph(parentSession, children) {
    const container = document.getElementById('flame-graph');
    container.innerHTML = '';

    const parentStart = new Date(parentSession.started_at).getTime();
    const now = Date.now();
    const parentEnd = parentSession.ended_at ? new Date(parentSession.ended_at).getTime() : now;
    const totalDuration = Math.max(parentEnd - parentStart, 1);

    const barHeight = 28;
    const barGap = 6;
    const totalHeight = barHeight + barGap + (children.length * (barHeight + barGap));
    container.style.height = totalHeight + 'px';

    // Parent bar (full width)
    const parentBar = document.createElement('div');
    parentBar.className = 'flame-bar flame-bar-parent';
    parentBar.style.top = '0px';
    parentBar.style.left = '0';
    parentBar.style.width = '100%';
    const parentDurMs = parentEnd - parentStart;
    const flameCost = _sessionBillingMode === 'subscription' ? '' : ` &middot; $${parentSession.total_cost.toFixed(4)}`;
    parentBar.innerHTML =
        `<span class="flame-bar-label">${escapeHtml(parentSession.name || parentSession.id)}</span>` +
        `<span class="flame-bar-stats">${formatDuration(parentDurMs)}${flameCost}</span>`;
    if (!parentSession.ended_at && parentSession.status === 'active') {
        parentBar.classList.add('flame-bar-active');
    }
    container.appendChild(parentBar);

    // Child bars
    children.forEach((child, idx) => {
        const childStart = new Date(child.started_at).getTime();
        const childEnd = child.ended_at ? new Date(child.ended_at).getTime() : now;
        const offsetPct = Math.max(((childStart - parentStart) / totalDuration) * 100, 0);
        const widthPct = Math.max(((childEnd - childStart) / totalDuration) * 100, 1);
        const top = barHeight + barGap + idx * (barHeight + barGap);
        const color = _FLAME_CHILD_COLORS[idx % _FLAME_CHILD_COLORS.length];

        const bar = document.createElement('div');
        bar.className = 'flame-bar flame-bar-child';
        bar.style.top = top + 'px';
        bar.style.left = offsetPct + '%';
        bar.style.width = Math.min(widthPct, 100 - offsetPct) + '%';
        bar.style.background = color;
        const childDurMs = childEnd - childStart;
        const childCostStr = _sessionBillingMode === 'subscription' ? '' : ` &middot; $${child.total_cost.toFixed(4)}`;
        bar.innerHTML =
            `<span class="flame-bar-label">${escapeHtml(child.name || child.id)}</span>` +
            `<span class="flame-bar-stats">${formatDuration(childDurMs)}${childCostStr}</span>`;
        bar.onclick = () => loadSessionDetail(child.id);
        if (!child.ended_at && child.status === 'active') {
            bar.classList.add('flame-bar-active');
        }
        container.appendChild(bar);
    });
}

function renderChildrenList(children) {
    const container = document.getElementById('children-list');
    let html = '<table><thead><tr>' +
        '<th>Session ID</th><th>Status</th><th>Cost</th><th>Tokens</th><th>Calls</th><th>Duration</th>' +
        '</tr></thead><tbody>';
    children.forEach(child => {
        const durMs = child.ended_at
            ? new Date(child.ended_at) - new Date(child.started_at)
            : Date.now() - new Date(child.started_at);
        html += `<tr onclick="loadSessionDetail('${escapeHtml(child.id)}')">` +
            `<td>${escapeHtml(child.name || child.id)}</td>` +
            `<td><span class="status-badge status-${escapeHtml(child.status)}">${escapeHtml(child.status)}</span></td>` +
            `<td>${_sessionBillingMode === 'subscription' ? '\u2014' : '$' + child.total_cost.toFixed(4)}</td>` +
            `<td>${(child.total_tokens || 0).toLocaleString()}</td>` +
            `<td>${child.call_count}</td>` +
            `<td>${formatDuration(durMs)}</td>` +
            '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

function toggleChildrenPanel() {
    const body = document.getElementById('detail-children-body');
    const icon = document.getElementById('children-toggle-icon');
    body.classList.toggle('collapsed');
    icon.classList.toggle('collapsed');
}

function renderStepTimeline(events) {
    const timeline = document.getElementById('step-timeline');
    if (!events || events.length === 0) {
        timeline.style.display = 'none';
        return;
    }

    // Only show step timeline if flat mode is active (waterfall mode hides it)
    const waterfallBtn = document.querySelector('.waterfall-toggle-btn[data-mode="waterfall"]');
    const isWaterfallMode = waterfallBtn && waterfallBtn.classList.contains('active');
    timeline.style.display = isWaterfallMode ? 'none' : 'flex';

    timeline.innerHTML = '';

    const label = document.createElement('span');
    label.className = 'step-timeline-label';
    label.textContent = 'Steps';
    timeline.appendChild(label);

    events.forEach((e, idx) => {
        const dot = document.createElement('span');
        dot.className = 'step-dot ' + getStepDotClass(e);
        dot.textContent = idx + 1;
        dot.title = `Step ${idx + 1}: ${e.event_type}`;
        dot.addEventListener('click', () => {
            const eventLines = document.querySelectorAll('#detail-events .event-line');
            if (eventLines[idx]) {
                eventLines[idx].scrollIntoView({ behavior: 'smooth', block: 'center' });
                eventLines[idx].classList.add('expanded');
                // Brief highlight
                eventLines[idx].style.boxShadow = '0 0 12px rgba(34, 211, 238, 0.3)';
                setTimeout(() => { eventLines[idx].style.boxShadow = ''; }, 1500);
            }
        });
        timeline.appendChild(dot);
    });
}

function getStepDotClass(e) {
    switch (e.event_type) {
        case 'llm_call': return 'step-dot-llm';
        case 'cache_hit': return 'step-dot-cache';
        case 'shadow_draft': return 'step-dot-shadow';
        case 'pii_detection': return 'step-dot-pii';
        case 'local_routing': return e.routing_success ? 'step-dot-routing' : 'step-dot-error';
        case 'loop_detection':
        case 'budget_enforcement':
        case 'kill_switch': return 'step-dot-error';
        case 'blast_radius': return 'step-dot-error';
        case 'circuit_breaker': return e.state === 'open' ? 'step-dot-error' : 'step-dot-pii';
        case 'guardrail': return (e.action_taken || '').toLowerCase() === 'blocked' ? 'step-dot-error' : 'step-dot-pii';
        case 'rate_limit': return (e.rejected || e.timed_out) ? 'step-dot-error' : 'step-dot-pii';
        case 'semantic_retry': return e.resolved ? 'step-dot-cache' : 'step-dot-error';
        case 'checkpoint': return 'step-dot-cache';
        case 'session_lifecycle': return 'step-dot-error';
        default: return 'step-dot-default';
    }
}

function formatDuration(ms) {
    if (ms < 0) ms = 0;
    const totalSec = Math.floor(ms / 1000);
    if (totalSec < 60) return `${totalSec}s`;
    const min = Math.floor(totalSec / 60);
    const sec = totalSec % 60;
    if (min < 60) return `${min}m ${sec}s`;
    const hr = Math.floor(min / 60);
    const rm = min % 60;
    return `${hr}h ${rm}m`;
}

// --- Models (Cloud + Local) ---
async function loadModels() {
    await Promise.all([
        loadCloudModels(),
        loadProviderKeys(),
        loadLocalStatus(),
        loadDownloadedModels(),
        loadRecommendations(),
    ]);
}

function switchModelsTab(tab) {
    location.hash = tab === 'overview' ? 'models' : `models/${tab}`;
    document.querySelectorAll('#view-models .sub-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('#view-models .sub-tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`#view-models .sub-tab[onclick*="${tab}"]`).classList.add('active');
    document.getElementById(`models-tab-${tab}`).classList.add('active');
    if (tab === 'testing') loadModelTesting();
}

async function loadCloudModels() {
    const [config, stats, costData, cloudModels, licData] = await Promise.all([
        fetchJSON('/config'),
        fetchJSON('/stats'),
        fetchJSON('/stats/cost-by-model'),
        fetchJSON('/models/cloud'),
        fetchJSON('/license'),
    ]);

    // Check enterprise license for model override
    let modelOverrideLicensed = false;
    try {
        if (licData && licData.features && licData.features.model_override) {
            modelOverrideLicensed = licData.features.model_override.enabled;
        }
    } catch (e) { /* default unlicensed */ }

    // Populate model override dropdown
    const select = document.getElementById('models-default-model');
    const currentVal = config?.default_model || '';
    select.innerHTML = '<option value="">None</option>';
    if (cloudModels?.models) {
        const byProvider = {};
        cloudModels.models.forEach(m => {
            if (!byProvider[m.provider]) byProvider[m.provider] = [];
            byProvider[m.provider].push(m.model);
        });
        for (const [provider, modelList] of Object.entries(byProvider)) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = provider.charAt(0).toUpperCase() + provider.slice(1);
            modelList.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                optgroup.appendChild(opt);
            });
            select.appendChild(optgroup);
        }
    }
    select.value = currentVal;

    // Gate model override behind enterprise license
    const lockNote = document.getElementById('model-override-lock');
    if (!modelOverrideLicensed) {
        select.disabled = true;
        select.style.opacity = '0.45';
        if (lockNote) lockNote.style.display = 'block';
    } else {
        select.disabled = false;
        select.style.opacity = '1';
        if (lockNote) lockNote.style.display = 'none';
    }

    // Stat cards
    if (stats) {
        document.getElementById('models-cloud-spend').textContent = '$' + (stats.total_cost || 0).toFixed(4);
        document.getElementById('models-cloud-calls').textContent = stats.cloud_calls || 0;
    }

    // Top model (highest cost)
    const models = costData?.models || {};
    const topEntry = Object.entries(models).sort((a, b) => b[1] - a[1])[0];
    document.getElementById('models-top-model').textContent = topEntry ? topEntry[0] : '\u2014';

    // Cost by model bar chart
    renderCostByModel(costData);
}

async function saveDefaultModel() {
    const model = document.getElementById('models-default-model').value;
    const resp = await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ default_model: model }),
    });
    if (resp.status === 403) {
        showToast('Emergency model override requires an enterprise license', 'error');
        return;
    }
    const flash = document.getElementById('flash-default-model');
    flash.classList.add('visible');
    setTimeout(() => flash.classList.remove('visible'), 2000);
}

async function loadProviderKeys() {
    const data = await fetchJSON('/provider-keys');
    if (!data) return;
    for (const provider of ['openai', 'anthropic', 'google']) {
        const info = data[provider];
        const statusEl = document.getElementById(`key-status-${provider}`);
        if (!statusEl) continue;
        if (info && info.set) {
            statusEl.innerHTML = '<span style="color:var(--accent-green, #4ade80);">Configured</span>' +
                ` <button class="btn btn-sm" style="font-size:10px;padding:2px 6px;margin-left:6px;" onclick="removeProviderKey('${provider}')">Remove</button>`;
        } else {
            statusEl.textContent = 'Not configured';
            statusEl.style.color = 'var(--text-secondary)';
        }
    }
}

async function saveProviderKey(provider) {
    const input = document.getElementById(`key-${provider}`);
    const key = input.value.trim();
    if (!key) return;

    // Check if key already exists and confirm overwrite
    const data = await fetchJSON('/provider-keys');
    if (data && data[provider] && data[provider].set) {
        if (!confirm(`A ${provider} key is already configured. Overwrite it?`)) return;
    }

    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ [`provider_api_key_${provider}`]: key }),
    });
    input.value = '';
    showToast(`${provider} key saved`, 'success');
    loadProviderKeys();
    invalidateProviderKeysCache();
}

async function removeProviderKey(provider) {
    if (!confirm(`Remove the ${provider} API key?`)) return;
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ [`provider_api_key_${provider}`]: '' }),
    });
    showToast(`${provider} key removed`, 'success');
    loadProviderKeys();
    invalidateProviderKeysCache();
}

function renderCostByModel(data) {
    const container = document.getElementById('cost-by-model');
    if (!data || !data.models || Object.keys(data.models).length === 0) {
        container.innerHTML = '<div class="empty-state">No cost data yet</div>';
        return;
    }

    let html = '<div style="width: 100%; padding: 12px;">';
    const total = Object.values(data.models).reduce((a, b) => a + b, 0);
    Object.entries(data.models)
        .sort((a, b) => b[1] - a[1])
        .forEach(([model, cost]) => {
            const pct = total > 0 ? (cost / total * 100).toFixed(1) : 0;
            html += `
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: var(--text-primary);">${escapeHtml(model)}</span>
                        <span style="color: var(--accent-cyan); font-family: var(--font-mono);">$${cost.toFixed(4)} (${pct}%)</span>
                    </div>
                    <div style="background: var(--bg-tertiary); border-radius: 4px; height: 8px;">
                        <div style="background: var(--accent-cyan); border-radius: 4px; height: 8px; width: ${pct}%;"></div>
                    </div>
                </div>
            `;
        });
    html += '</div>';
    container.innerHTML = html;
}

async function loadPII() {
    const data = await fetchJSON('/pii');
    if (!data) return;
    const tbody = document.getElementById('pii-tbody');
    tbody.innerHTML = '';

    // Populate summary stats
    document.getElementById('pii-total').textContent = data.total || 0;
    document.getElementById('pii-sessions').textContent = data.sessions_affected || 0;

    // Type breakdown
    const typeEl = document.getElementById('pii-types-summary');
    if (data.by_type && Object.keys(data.by_type).length > 0) {
        typeEl.innerHTML = Object.entries(data.by_type)
            .map(([t, c]) => `<span class="pii-badge pii-type-${t}">${t}: ${c}</span>`)
            .join(' ');
    } else {
        typeEl.textContent = '—';
    }

    // Action breakdown
    const actionEl = document.getElementById('pii-actions-summary');
    if (data.by_action && Object.keys(data.by_action).length > 0) {
        actionEl.innerHTML = Object.entries(data.by_action)
            .map(([a, c]) => `<span class="pii-badge pii-action-${a}">${a}: ${c}</span>`)
            .join(' ');
    } else {
        actionEl.textContent = '—';
    }

    if (!data.detections || data.detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No PII detections</td></tr>';
        return;
    }

    loadPIIRules();

    data.detections.forEach(d => {
        const tr = document.createElement('tr');
        // Color-code action
        const actionClass = d.action === 'blocked' ? 'pii-action-blocked'
            : d.action === 'redacted' ? 'pii-action-redacted' : 'pii-action-logged';
        const preview = d.redacted_preview
            ? `<code class="pii-preview">${escapeHtml(d.redacted_preview)}</code>`
            : `<span class="dim">[${d.match_length || '?'} chars]</span>`;
        tr.innerHTML = `
            <td>${new Date(d.timestamp).toLocaleString()}</td>
            <td>${d.session_id}</td>
            <td><span class="pii-badge pii-type-${d.pii_type}">${d.pii_type}</span></td>
            <td>${preview}</td>
            <td><span class="pii-badge ${actionClass}">${d.action}</span></td>
            <td><code>${d.field}</code></td>
        `;
        tbody.appendChild(tr);
    });
}

async function restartServer() {
    const btn = document.getElementById('btn-restart');
    if (!confirm('Restart the StateLoom server? Active connections will be interrupted.')) return;
    btn.disabled = true;
    btn.textContent = 'Restarting...';
    try {
        await fetch('/api/restart', { method: 'POST' });
        // Server will restart — poll until it comes back
        setTimeout(async () => {
            for (let i = 0; i < 30; i++) {
                await new Promise(r => setTimeout(r, 1000));
                try {
                    const resp = await fetch('/api/stats');
                    if (resp.ok) {
                        location.reload();
                        return;
                    }
                } catch(e) { /* server still down */ }
            }
            btn.textContent = 'Restart Server';
            btn.disabled = false;
            alert('Server did not come back after 30 seconds. Check logs.');
        }, 1000);
    } catch(e) {
        btn.textContent = 'Restart Server';
        btn.disabled = false;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- PII Rules CRUD ---
async function loadPIIRules() {
    try {
        const data = await fetchJSON('/pii/rules');
        if (!data) return;
        document.getElementById('pii-enabled-toggle').checked = data.pii_enabled;
        document.getElementById('pii-default-mode-select').value = data.pii_default_mode || 'audit';
        const rules = data.rules || [];
        document.getElementById('pii-rule-count').textContent = rules.length;

        const tbody = document.getElementById('pii-rules-tbody');
        if (rules.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No rules configured</td></tr>';
        } else {
            tbody.innerHTML = '';
            rules.forEach(r => {
                const modeClass = 'pii-mode-' + (r.mode || 'audit');
                const failureAction = r.on_middleware_failure || '\u2014';
                const tr = document.createElement('tr');
                tr.style.cursor = 'default';
                tr.innerHTML = `
                    <td>${escapeHtml(r.pattern)}</td>
                    <td><span class="pii-badge ${escapeHtml(modeClass)}">${escapeHtml(r.mode || 'audit')}</span></td>
                    <td>${escapeHtml(failureAction)}</td>
                    <td><button class="btn btn-danger btn-sm" data-pattern="${escapeHtml(r.pattern)}">Delete</button></td>
                `;
                tr.querySelector('button').addEventListener('click', (e) => {
                    e.stopPropagation();
                    deletePIIRule(r.pattern);
                });
                tbody.appendChild(tr);
            });
        }
    } catch (e) { console.warn('Failed to load PII rules:', e); }
}

function showAddPIIRuleForm() {
    document.getElementById('pii-add-rule-form').style.display = 'block';
}

function hideAddPIIRuleForm() {
    document.getElementById('pii-add-rule-form').style.display = 'none';
    document.getElementById('pii-rule-pattern').value = 'email';
    document.getElementById('pii-rule-mode').value = 'audit';
    document.getElementById('pii-rule-on-failure').value = '';
}

async function addPIIRule() {
    const body = {
        pattern: document.getElementById('pii-rule-pattern').value,
        mode: document.getElementById('pii-rule-mode').value,
    };
    const onFailure = document.getElementById('pii-rule-on-failure').value;
    if (onFailure) body.on_middleware_failure = onFailure;

    await fetch(`${API_BASE}/pii/rules`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    hideAddPIIRuleForm();
    loadPIIRules();
}

async function deletePIIRule(pattern) {
    await fetch(`${API_BASE}/pii/rules/${encodeURIComponent(pattern)}`, { method: 'DELETE' });
    loadPIIRules();
}

async function clearAllPIIRules() {
    await fetch(`${API_BASE}/pii/rules`, { method: 'DELETE' });
    loadPIIRules();
}

async function togglePIIEnabled(enabled) {
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pii_enabled: enabled}),
    });
}

async function updatePIIDefaultMode(mode) {
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pii_default_mode: mode}),
    });
}

// --- Event Stream ---
function addEventToStream(data) {
    const stream = document.getElementById('event-stream');
    const empty = stream.querySelector('.empty-state');
    if (empty) empty.remove();

    const line = createEventLine(data);
    stream.insertBefore(line, stream.firstChild);

    // Cap at 200 events
    while (stream.children.length > 200) {
        stream.removeChild(stream.lastChild);
    }
}

function createEventLine(e) {
    const div = document.createElement('div');
    div.className = 'event-line';

    const time = new Date(e.timestamp).toLocaleTimeString();
    let summaryText = '';
    let typeClass = '';
    let detailHtml = '';

    switch (e.event_type) {
        case 'llm_call': {
            typeClass = ' event-llm';
            const _costStr = _sessionBillingMode === 'subscription' ? '\u2014' : '$' + (e.cost || 0).toFixed(4);
            summaryText = `[${time}] ${e.model || '?'} | ${e.total_tokens || 0} tok | ${_costStr} | ${(e.latency_ms || 0).toFixed(0)}ms`;
            const _detailCost = _sessionBillingMode === 'subscription' ? '\u2014' : '$' + (e.cost || 0).toFixed(6);
            detailHtml = buildDetailGrid({
                'Provider': e.provider || '?',
                'Model': e.model || '?',
                'Prompt Tokens': e.prompt_tokens || 0,
                'Completion Tokens': e.completion_tokens || 0,
                'Total Tokens': e.total_tokens || 0,
                'Cost': _detailCost,
                'Latency': (e.latency_ms || 0).toFixed(0) + 'ms',
                'Streaming': e.streaming ? 'Yes' : 'No',
            });
            break;
        }
        case 'cache_hit':
            typeClass = ' event-cache';
            if (_sessionBillingMode === 'subscription') {
                if (e.match_type === 'semantic' && e.similarity_score != null) {
                    summaryText = `[${time}] SEMANTIC CACHE HIT | sim=${(e.similarity_score * 100).toFixed(0)}%`;
                } else {
                    summaryText = `[${time}] CACHE HIT`;
                }
            } else if (e.match_type === 'semantic' && e.similarity_score != null) {
                summaryText = `[${time}] SEMANTIC CACHE HIT | sim=${(e.similarity_score * 100).toFixed(0)}% | saved $${(e.saved_cost || 0).toFixed(4)}`;
            } else {
                summaryText = `[${time}] CACHE HIT | saved $${(e.saved_cost || 0).toFixed(4)}`;
            }
            {
                const cacheDetail = {
                    'Original Model': e.model || e.original_model || '?',
                    'Saved Cost': _sessionBillingMode === 'subscription' ? '\u2014' : '$' + (e.saved_cost || 0).toFixed(6),
                    'Match Type': e.match_type || 'exact',
                };
                if (e.similarity_score != null) {
                    cacheDetail['Similarity Score'] = (e.similarity_score * 100).toFixed(1) + '%';
                }
                detailHtml = buildDetailGrid(cacheDetail);
            }
            break;
        case 'pii_detection':
            typeClass = ' event-pii';
            summaryText = `[${time}] PII ${(e.action_taken || 'detected').toUpperCase()} | type:${e.pii_type || '?'} | field:${e.pii_field || '?'}`;
            detailHtml = buildDetailGrid({
                'PII Type': e.pii_type || '?',
                'Field': e.pii_field || '?',
                'Mode': e.pii_mode || e.mode || '?',
                'Action': e.action_taken || 'detected',
                'Redacted Preview': e.redacted_preview || '\u2014',
            });
            break;
        case 'loop_detection':
            typeClass = ' event-loop';
            summaryText = `[${time}] LOOP DETECTED | calls:${e.repeat_count || 0}`;
            detailHtml = buildDetailGrid({
                'Repeat Count': e.repeat_count || 0,
                'Threshold': e.threshold || '?',
            });
            break;
        case 'budget_enforcement':
            typeClass = ' event-budget';
            const budgetLevel = e.budget_level || '';
            const levelLabel = budgetLevel ? ` (${budgetLevel})` : '';
            const actionLabel = e.budget_action || e.action || '?';
            summaryText = `[${time}] BUDGET${levelLabel} | ${actionLabel} | limit:$${(e.limit || 0).toFixed(4)} | spent:$${(e.spent || 0).toFixed(4)}`;
            detailHtml = buildDetailGrid({
                'Level': budgetLevel || 'session',
                'Limit': '$' + (e.limit || 0).toFixed(6),
                'Spent': '$' + (e.spent || 0).toFixed(6),
                'Action': actionLabel,
                'Status': actionLabel === 'hard_stop' ? 'Blocked — next call will be rejected' : 'Warning — call allowed',
            });
            break;
        case 'shadow_draft':
            typeClass = ' event-shadow';
            if (e.shadow_status === 'success') {
                const _savedPart = _sessionBillingMode === 'subscription' ? '' : ` | saved $${(e.cost_saved || 0).toFixed(4)}`;
                let shadowText = `[${time}] MODEL TEST | ${e.local_model || '?'} | ${(e.local_latency_ms || 0).toFixed(0)}ms | ${e.local_tokens || 0} tok${_savedPart}`;
                if (e.similarity_score != null) {
                    shadowText += ` | sim=${(e.similarity_score * 100).toFixed(0)}%`;
                }
                summaryText = shadowText;
                detailHtml = '<div class="shadow-compare">' +
                    '<div class="shadow-compare-col"><h5>Cloud (Production)</h5>' +
                    buildDetailGrid({
                        'Model': e.cloud_model || e.model || '?',
                        'Latency': (e.cloud_latency_ms || e.latency_ms || 0).toFixed(0) + 'ms',
                        'Tokens': e.cloud_tokens || e.total_tokens || '?',
                        'Cost': _sessionBillingMode === 'subscription' ? '\u2014' : '$' + (e.cloud_cost || e.cost || 0).toFixed(6),
                    }) + '</div>' +
                    '<div class="shadow-compare-col"><h5>Candidate Model</h5>' +
                    buildDetailGrid({
                        'Model': e.local_model || '?',
                        'Latency': (e.local_latency_ms || 0).toFixed(0) + 'ms',
                        'Tokens': e.local_tokens || 0,
                        'Cost Saved': _sessionBillingMode === 'subscription' ? '\u2014' : '$' + (e.cost_saved || 0).toFixed(6),
                        'Similarity': e.similarity_score != null ? (e.similarity_score * 100).toFixed(1) + '%' : '\u2014',
                    }) + '</div></div>';
                if (e.cloud_preview || e.local_preview || e.shadow_preview) {
                    detailHtml += '<div class="shadow-compare" style="margin-top:8px;">';
                    if (e.cloud_preview) {
                        detailHtml += '<div class="shadow-compare-col"><h5>Cloud Response</h5>' +
                            '<div style="padding:6px 10px;background:rgba(0,0,0,.15);border-radius:4px;font-size:11px;color:var(--text-secondary);white-space:pre-wrap;max-height:100px;overflow-y:auto;">' +
                            escapeHtml(e.cloud_preview) + '</div></div>';
                    }
                    const localText = e.local_preview || e.shadow_preview || '';
                    if (localText) {
                        detailHtml += '<div class="shadow-compare-col"><h5>Candidate Response</h5>' +
                            '<div style="padding:6px 10px;background:rgba(0,0,0,.15);border-radius:4px;font-size:11px;color:var(--text-secondary);white-space:pre-wrap;max-height:100px;overflow-y:auto;">' +
                            escapeHtml(localText) + '</div></div>';
                    }
                    detailHtml += '</div>';
                }
            } else {
                summaryText = `[${time}] MODEL TEST ${(e.shadow_status || 'error').toUpperCase()} | ${e.local_model || '?'}`;
                detailHtml = buildDetailGrid({
                    'Status': e.shadow_status || 'error',
                    'Candidate Model': e.local_model || '?',
                    'Error': e.error || e.shadow_error || '\u2014',
                });
            }
            break;
        case 'local_routing':
            if (e.routing_success) {
                typeClass = ' event-cache';
                const savedStr = _sessionBillingMode === 'subscription' ? '' : (e.estimated_cloud_cost > 0 ? ` | saved $${e.estimated_cloud_cost.toFixed(4)}` : '');
                summaryText = `[${time}] ROUTED LOCAL | ${e.local_model || '?'} | complexity:${(e.complexity_score || 0).toFixed(2)} | was:${e.original_cloud_model || '?'}${savedStr}`;
            } else {
                typeClass = ' event-loop';
                summaryText = `[${time}] ROUTE FAILED | ${e.local_model || '?'} | fallback to ${e.original_cloud_model || 'cloud'} | ${e.routing_reason || ''}`;
            }
            {
                const routingDetail = {
                    'Original Model': e.original_cloud_model || '?',
                    'Local Model': e.local_model || '?',
                    'Complexity Score': (e.complexity_score || 0).toFixed(3),
                    'Semantic Complexity': e.semantic_complexity != null ? e.semantic_complexity.toFixed(3) : '\u2014',
                    'Routing Threshold': e.routing_threshold != null ? e.routing_threshold.toFixed(3) : '\u2014',
                    'Budget Pressure': e.budget_pressure != null ? (e.budget_pressure * 100).toFixed(0) + '%' : '\u2014',
                    'Reason': e.routing_reason || '\u2014',
                    'Success': e.routing_success ? 'Yes' : 'No',
                    'Est. Cloud Cost': e.estimated_cloud_cost != null ? '$' + e.estimated_cloud_cost.toFixed(6) : '\u2014',
                };
                detailHtml = buildDetailGrid(routingDetail);
            }
            break;
        case 'kill_switch':
            typeClass = ' event-kill-switch';
            summaryText = `[${time}] KILL SWITCH | ${e.reason || 'active'} | model:${e.blocked_model || '?'} | provider:${e.blocked_provider || '?'}`;
            detailHtml = buildDetailGrid({
                'Reason': e.reason || 'active',
                'Blocked Model': e.blocked_model || '?',
                'Blocked Provider': e.blocked_provider || '?',
                'Matched Rule': e.matched_rule ? JSON.stringify(e.matched_rule) : '\u2014',
            });
            break;
        case 'blast_radius':
            typeClass = ' event-blast-radius';
            summaryText = `[${time}] BLAST RADIUS | ${e.trigger || '?'} | ${e.count || 0}/${e.threshold || 0} | ${e.action || 'paused'}${e.agent_id ? ' | agent:' + e.agent_id : ''}`;
            detailHtml = buildDetailGrid({
                'Trigger': e.trigger || '?',
                'Count': e.count || 0,
                'Threshold': e.threshold || 0,
                'Action': e.action || 'paused',
                'Agent ID': e.agent_id || '\u2014',
            });
            break;
        case 'circuit_breaker':
            typeClass = e.state === 'open' ? ' event-kill-switch' : ' event-blast-radius';
            summaryText = `[${time}] CIRCUIT BREAKER | ${e.provider || '?'} | ${e.state || '?'} | failures:${e.failure_count || 0}/${e.failure_threshold || 0}${e.fallback_model ? ' | fallback:' + e.fallback_model : ''}`;
            detailHtml = buildDetailGrid({
                'Provider': e.provider || '?',
                'State': e.state || '?',
                'Previous State': e.previous_state || '?',
                'Failures': `${e.failure_count || 0} / ${e.failure_threshold || 0}`,
                'Original Model': e.original_model || '\u2014',
                'Fallback Model': e.fallback_model || '\u2014',
                'Fallback Provider': e.fallback_provider || '\u2014',
                'Probe Success': e.probe_success != null ? String(e.probe_success) : '\u2014',
            });
            break;
        case 'guardrail':
            typeClass = (e.action_taken || '').toLowerCase() === 'blocked' ? ' event-kill-switch' : ' event-pii';
            summaryText = `[${time}] GUARDRAIL | ${e.rule_name || '?'} | ${e.category || '?'} | ${e.severity || '?'} | ${(e.action_taken || 'logged').toUpperCase()}`;
            detailHtml = buildDetailGrid({
                'Rule': e.rule_name || '?',
                'Category': e.category || '?',
                'Severity': e.severity || '?',
                'Score': e.score != null ? e.score.toFixed(3) : '\u2014',
                'Action': e.action_taken || 'logged',
                'Validator': e.validator_type || '?',
                'Phase': e.scan_phase || '?',
                'Violation Text': e.violation_text || '\u2014',
            });
            break;
        case 'checkpoint':
            typeClass = ' event-cache';
            summaryText = `[${time}] CHECKPOINT | ${e.label || 'unnamed'}${e.description ? ' | ' + e.description : ''}`;
            detailHtml = buildDetailGrid({
                'Label': e.label || '\u2014',
                'Description': e.description || '\u2014',
            });
            break;
        default:
            summaryText = `[${time}] ${e.event_type}`;
            // Build detail from all available fields
            const fields = {};
            for (const [k, v] of Object.entries(e)) {
                if (k !== 'timestamp' && k !== 'event_type' && k !== 'session_id' && v != null) {
                    fields[k] = String(v);
                }
            }
            if (Object.keys(fields).length > 0) {
                detailHtml = buildDetailGrid(fields);
            }
    }

    div.className += typeClass;

    // Build expandable structure
    const summary = document.createElement('div');
    summary.className = 'event-summary';
    const arrow = document.createElement('span');
    arrow.className = 'event-expand-icon';
    arrow.textContent = '\u25B6';
    summary.appendChild(arrow);
    const textSpan = document.createElement('span');
    textSpan.textContent = summaryText;
    summary.appendChild(textSpan);
    div.appendChild(summary);

    if (detailHtml) {
        const detail = document.createElement('div');
        detail.className = 'event-detail';
        detail.innerHTML = detailHtml;
        div.appendChild(detail);

        div.addEventListener('click', (evt) => {
            // Don't toggle when clicking inside the detail panel
            if (evt.target.closest('.event-detail')) return;
            div.classList.toggle('expanded');
        });
    }

    return div;
}

function buildDetailGrid(fields) {
    let html = '<div class="event-detail-grid">';
    for (const [key, value] of Object.entries(fields)) {
        html += `<span class="detail-key">${escapeHtml(key)}</span>`;
        html += `<span class="detail-value">${escapeHtml(String(value))}</span>`;
    }
    html += '</div>';
    return html;
}

// === Waterfall Trace Timeline ===

const _PRIMARY_EVENT_TYPES = new Set([
    'llm_call', 'cache_hit', 'tool_call', 'local_routing', 'checkpoint',
    'circuit_breaker', 'blast_radius', 'kill_switch', 'budget_enforcement',
    'loop_detection', 'guardrail', 'rate_limit', 'semantic_retry', 'shadow_draft',
    'session_lifecycle',
]);

function groupEventsIntoSteps(events) {
    const steps = [];
    let currentStep = null;
    let pendingSubEvents = [];  // sub-events that arrive before any primary

    // Filter out noise events from waterfall
    const _COMPLIANCE_ALERT_ACTIONS = new Set([
        'streaming_blocked', 'endpoint_blocked', 'endpoint_unknown',
        'pii_blocked', 'pii_redacted', 'budget_enforced',
        'rate_limit_enforced', 'blast_radius_triggered', 'kill_switch_activated',
    ]);
    const filtered = events.filter(e => {
        if (e.event_type === 'shadow_draft' && e.shadow_status === 'cancelled') return false;
        if (e.event_type === 'compliance_audit' && !_COMPLIANCE_ALERT_ACTIONS.has(e.action)) return false;
        // Filter out near-0ms LLM call noise (pipeline executions with no actual LLM activity).
        if (e.event_type === 'llm_call' && (e.latency_ms || 0) < 1 && !e.cost) return false;
        // Filter out CLI internal requests (quota checks, token counting — single user
        // message with no system prompt, detected server-side by CostTracker).
        if (e.event_type === 'llm_call' && e.is_cli_internal) return false;
        // Hide routing decisions that stayed on cloud (no local routing happened)
        if (e.event_type === 'local_routing' && !e.routing_success) return false;
        return true;
    });

    for (const event of filtered) {
        // Successful local_routing starts a new step; the follow-up llm_call
        // becomes a sub-event (it's just the cost record for the local response).
        // Failed local_routing (cloud decision info) is always a sub-event.
        const isLocalRouteSuccess = event.event_type === 'local_routing' && event.routing_success;
        const isLocalRouteInfo = event.event_type === 'local_routing' && !event.routing_success;

        // When a successful local_routing step is open, treat llm_call as sub-event
        const isLlmAfterLocalRoute = event.event_type === 'llm_call' &&
            currentStep && currentStep.primary.event_type === 'local_routing' &&
            currentStep.primary.routing_success;

        // PII blocked events are standalone steps (the call was prevented).
        // PII redacted/logged events are sub-events of the next LLM call.
        const isPiiBlocked = event.event_type === 'pii_detection' &&
            event.action_taken === 'blocked';

        const isPrimary = (isLocalRouteSuccess) || (isPiiBlocked) ||
            (!isLocalRouteInfo && !isLlmAfterLocalRoute &&
             _PRIMARY_EVENT_TYPES.has(event.event_type));

        // PII logged/redacted events precede their LLM call, so they should
        // queue in pendingSubEvents even when a previous step is open.
        const isPiiSubEvent = event.event_type === 'pii_detection' && !isPiiBlocked;

        if (isPrimary) {
            // Start a new step group, absorbing any pending sub-events
            currentStep = {
                step: steps.length + 1,
                primary: event,
                subEvents: [...pendingSubEvents],
                hasDurableCheckpoint: !!(event.has_cached_response),
                totalLatencyMs: event.event_type === 'shadow_draft' ? (event.local_latency_ms || 0) : (event.latency_ms || 0),
                totalCost: event.event_type === 'shadow_draft' ? (event.cost_saved || 0) : (event.cost || 0),
            };
            pendingSubEvents = [];
            steps.push(currentStep);
        } else if (isPiiSubEvent) {
            // PII events always defer to the next primary (their LLM call)
            pendingSubEvents.push(event);
        } else if (currentStep) {
            // Sub-event — attach to current step
            currentStep.subEvents.push(event);
        } else {
            // No primary yet — defer until next primary arrives
            pendingSubEvents.push(event);
        }
    }

    // If only sub-events remain with no primary, show them as a standalone step
    if (pendingSubEvents.length > 0 && steps.length === 0) {
        currentStep = {
            step: 1,
            primary: pendingSubEvents[0],
            subEvents: pendingSubEvents.slice(1),
            hasDurableCheckpoint: false,
            totalLatencyMs: pendingSubEvents[0].latency_ms || 0,
            totalCost: pendingSubEvents[0].cost || 0,
        };
        steps.push(currentStep);
    }

    return steps;
}

function getWaterfallTypeBadge(event) {
    const type = event.event_type;
    // Tool-continuation LLM calls render as [Tool] not [LLM]
    if (type === 'llm_call' && event.is_tool_continuation) {
        return `<span class="wf-type-badge wf-type-tool-summary">Tool</span>`;
    }
    const classMap = {
        'llm_call': 'wf-type-llm',
        'cache_hit': 'wf-type-cache',
        'local_routing': 'wf-type-local',
        'shadow_draft': 'wf-type-shadow',
        'pii_detection': 'wf-type-pii',
        'budget_enforcement': 'wf-type-budget',
        'kill_switch': 'wf-type-kill',
        'blast_radius': 'wf-type-blast',
        'circuit_breaker': 'wf-type-kill',
        'loop_detection': 'wf-type-loop',
        'semantic_retry': 'wf-type-retry',
        'checkpoint': 'wf-type-checkpoint',
        'rate_limit': 'wf-type-rate-limit',
        'guardrail': 'wf-type-pii',
        'session_lifecycle': 'wf-type-kill',
    };
    const cls = classMap[type] || 'wf-type-default';
    const labelMap = {
        'llm_call': 'LLM',
        'cache_hit': 'Cache',
        'local_routing': 'Local',
        'shadow_draft': 'Shadow',
        'pii_detection': 'PII',
        'budget_enforcement': 'Budget',
        'kill_switch': 'Kill',
        'blast_radius': 'Blast',
        'circuit_breaker': 'CB',
        'loop_detection': 'Loop',
        'semantic_retry': 'Retry',
        'checkpoint': 'Checkpoint',
        'rate_limit': 'Rate Limit',
        'tool_call': 'Tool',
        'guardrail': 'Guardrail',
        'session_lifecycle': event.action === 'cancelled' ? 'Cancelled' : 'Timeout',
    };
    const label = labelMap[type] || type;
    return `<span class="wf-type-badge ${cls}">${escapeHtml(label)}</span>`;
}

function getDurationBarClass(event) {
    const type = event.event_type;
    if (type === 'llm_call') return event.is_tool_continuation ? 'wf-bar-tool' : 'wf-bar-llm';
    if (type === 'tool_call') return 'wf-bar-tool';
    if (type === 'cache_hit') return 'wf-bar-cache';
    if (type === 'local_routing') return event.routing_success ? 'wf-bar-local' : 'wf-bar-llm';
    if (type === 'shadow_draft') return 'wf-bar-shadow';
    if (type === 'semantic_retry') return 'wf-bar-retry';
    if (type === 'pii_detection') return 'wf-bar-pii';
    return 'wf-bar-default';
}

function createDurationBarHtml(latencyMs, maxLatency, barClass) {
    if (!latencyMs || !maxLatency) {
        return '<span class="wf-duration-label">\u2014</span>';
    }
    const pct = Math.min((latencyMs / maxLatency) * 100, 100);
    const label = latencyMs >= 1000 ? (latencyMs / 1000).toFixed(2) + 's' : latencyMs.toFixed(0) + 'ms';
    return `<div style="display:flex;align-items:center;gap:6px;">` +
        `<div class="wf-duration-bar-track"><div class="wf-duration-bar-fill ${barClass}" style="width:${pct}%"></div></div>` +
        `<span class="wf-duration-label">${label}</span></div>`;
}

function formatWaterfallCost(event) {
    // Hide cost display for subscription sessions (token counts are incomplete
    // for streaming CLI sessions, making cost estimates unreliable).
    if (_sessionBillingMode === 'subscription') return '\u2014';
    if (event.event_type === 'cache_hit') {
        const saved = event.saved_cost || 0;
        return saved > 0 ? `<span style="color:var(--accent-green)">saved $${saved.toFixed(4)}</span>` : '\u2014';
    }
    if (event.event_type === 'shadow_draft') {
        const sim = event.similarity_score;
        if (sim != null) return `<span style="color:var(--accent-purple)">sim ${(sim * 100).toFixed(0)}%</span>`;
        return '\u2014';
    }
    const cost = event.cost || 0;
    if (cost > 0) return '$' + cost.toFixed(4);
    return '\u2014';
}

function getWaterfallStatus(event) {
    const type = event.event_type;
    if (type === 'llm_call') return '<span class="wf-status wf-status-ok">OK</span>';
    if (type === 'cache_hit') return '<span class="wf-status wf-status-cached">Cached</span>';
    if (type === 'local_routing') {
        return event.routing_success
            ? '<span class="wf-status wf-status-routed">Routed</span>'
            : '<span class="wf-status wf-status-ok">Cloud</span>';
    }
    if (type === 'pii_detection') {
        const action = (event.action_taken || '').toLowerCase();
        if (action === 'block' || action === 'blocked') return '<span class="wf-status wf-status-blocked">Blocked</span>';
        if (action === 'redact' || action === 'redacted') return '<span class="wf-status wf-status-redacted">Redacted</span>';
        return '<span class="wf-status wf-status-ok">Logged</span>';
    }
    if (type === 'semantic_retry') {
        return event.resolved
            ? '<span class="wf-status wf-status-resolved">Resolved</span>'
            : '<span class="wf-status wf-status-failed">Failed</span>';
    }
    if (type === 'shadow_draft') {
        return event.shadow_status === 'success'
            ? '<span class="wf-status wf-status-ok">OK</span>'
            : '<span class="wf-status wf-status-failed">Failed</span>';
    }
    if (type === 'budget_enforcement' || type === 'kill_switch' || type === 'blast_radius') {
        return '<span class="wf-status wf-status-blocked">Blocked</span>';
    }
    if (type === 'circuit_breaker') {
        return event.state === 'open'
            ? '<span class="wf-status wf-status-blocked">Tripped</span>'
            : '<span class="wf-status wf-status-failed">Failure ' + (event.failure_count || 0) + '/' + (event.failure_threshold || 0) + '</span>';
    }
    if (type === 'loop_detection') return '<span class="wf-status wf-status-blocked">Blocked</span>';
    if (type === 'guardrail') {
        const action = (event.action_taken || '').toLowerCase();
        if (action === 'blocked') return '<span class="wf-status wf-status-blocked">Blocked</span>';
        return '<span class="wf-status wf-status-ok">Logged</span>';
    }
    if (type === 'rate_limit') {
        if (event.rejected) return '<span class="wf-status wf-status-blocked">Rejected</span>';
        if (event.timed_out) return '<span class="wf-status wf-status-blocked">Timed Out</span>';
        if (event.queued) {
            const ms = Math.round(event.wait_ms || 0);
            return `<span class="wf-status wf-status-queued">Waited ${ms}ms</span>`;
        }
        return '<span class="wf-status wf-status-ok">Passed</span>';
    }
    if (type === 'checkpoint') return '<span class="wf-status wf-status-ok">OK</span>';
    if (type === 'session_lifecycle') {
        if (event.action === 'timed_out') return '<span class="wf-status wf-status-blocked">Timed Out</span>';
        if (event.action === 'cancelled') return '<span class="wf-status wf-status-blocked">Cancelled</span>';
        return '<span class="wf-status wf-status-blocked">' + escapeHtml(event.action || '') + '</span>';
    }
    return '<span class="wf-status wf-status-ok">OK</span>';
}

function getSubRowContext(event) {
    const type = event.event_type;
    if (type === 'pii_detection') return event.pii_type || '';
    if (type === 'shadow_draft') return event.local_model || '';
    if (type === 'semantic_retry') return `attempt ${event.attempt || '?'}/${event.max_attempts || '?'}`;
    if (type === 'blast_radius') return event.trigger || '';
    if (type === 'rate_limit') return event.team_id || '';
    if (type === 'guardrail') return event.rule_name || event.category || '';
    if (type === 'loop_detection') return `${event.repeat_count || 0} repeats`;
    return '';
}

function buildWaterfallDetail(event) {
    // Reuse createEventLine's detail logic by extracting detail fields
    const fields = {};
    for (const [k, v] of Object.entries(event)) {
        if (!['timestamp', 'event_type', 'session_id', 'event_id', 'id', 'step'].includes(k) && v != null) {
            fields[k] = String(v);
        }
    }
    if (Object.keys(fields).length === 0) return '';
    return '<div class="wf-detail-content">' + buildDetailGrid(fields) + '</div>';
}

function _collapseToolSteps(steps) {
    // With primary_only=true, tool-continuation events are excluded server-side.
    // The server attaches _tool_summary to each parent LLM call event.
    // This function reads those summaries and attaches them to the step objects.
    // It also handles the legacy path (tool continuations still in the array)
    // as a fallback when primary_only is not used.
    const result = [];
    let i = 0;
    while (i < steps.length) {
        const step = steps[i];

        // Server-provided tool summary (primary_only mode)
        if (step.primary._tool_summary) {
            step.tool_summary = step.primary._tool_summary;
            // collapsed_steps will be lazy-loaded on click
            step.collapsed_steps = null;
            result.push(step);
            i++;
            continue;
        }

        // Legacy fallback: scan for consecutive tool-continuation / tool_call steps
        const isToolStep = (s) => {
            if (s.primary.event_type === 'llm_call' && s.primary.is_tool_continuation) return true;
            if (s.primary.event_type === 'tool_call') return true;
            return false;
        };
        if (isToolStep(step)) {
            const run = [];
            const tcModel = step.primary.model || '';
            while (i < steps.length) {
                const s = steps[i];
                if (isToolStep(s)) {
                    run.push(s);
                    i++;
                } else if (s.primary.event_type === 'llm_call' && !s.primary.is_tool_continuation) {
                    let gapEnd = i;
                    let allDiffModel = true;
                    while (gapEnd < steps.length &&
                           steps[gapEnd].primary.event_type === 'llm_call' &&
                           !steps[gapEnd].primary.is_tool_continuation) {
                        if ((steps[gapEnd].primary.model || '') === tcModel) {
                            allDiffModel = false;
                        }
                        gapEnd++;
                    }
                    if (allDiffModel &&
                        gapEnd < steps.length &&
                        isToolStep(steps[gapEnd])) {
                        while (i < gapEnd) {
                            run.push(steps[i]);
                            i++;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            let totalTokens = 0, totalCost = 0, totalLatency = 0;
            for (const s of run) {
                totalTokens += (s.primary.prompt_tokens || 0) + (s.primary.completion_tokens || 0);
                totalCost += s.primary.cost || 0;
                totalLatency += s.primary.latency_ms || 0;
            }
            const summary = {
                total_tokens: totalTokens,
                total_cost: totalCost,
                count: run.length,
                total_latency_ms: totalLatency,
                model: run[0].primary.model || '',
            };
            if (result.length > 0) {
                const parent = result[result.length - 1];
                parent.tool_summary = summary;
                parent.collapsed_steps = run;
            } else {
                result.push({
                    step: 0,
                    primary: { event_type: 'tool_steps_summary', ...summary },
                    subEvents: [],
                    collapsed_steps: run,
                });
            }
        } else {
            result.push(step);
            i++;
        }
    }
    // Renumber steps
    let num = 1;
    for (const s of result) {
        s.step = num++;
    }
    return result;
}

function _groupByPromptId(steps) {
    // Groups steps that share the same user_prompt_id under the first
    // occurrence.  Gemini CLI sends the same user_prompt_id for the
    // parent call AND subagent delegations belonging to one user prompt.
    // Steps without a user_prompt_id pass through untouched.
    const result = [];
    // Map: user_prompt_id → index in result[] of the parent step
    const parentIndex = {};

    for (const step of steps) {
        const pid = _getPromptId(step);
        if (!pid) {
            result.push(step);
            continue;
        }

        if (!(pid in parentIndex)) {
            // First step with this prompt ID — it's the parent
            parentIndex[pid] = result.length;
            result.push(step);
        } else {
            // Subsequent step with same prompt ID — only merge if it's
            // actually a tool step. Regular LLM calls sharing the same
            // prompt ID (e.g. Gemini CLI flash-lite + flash-preview for
            // the same user turn) should stay as separate steps.
            const p = step.primary;
            const isToolStep = (p.is_tool_continuation) ||
                               (p.event_type === 'tool_call');
            if (!isToolStep) {
                result.push(step);
                continue;
            }

            const parent = result[parentIndex[pid]];

            // Accumulate this step (and its tool children) into the parent's
            // tool_summary / collapsed_steps, creating them if needed.
            if (!parent.collapsed_steps) {
                parent.collapsed_steps = [];
            }
            // Add this step itself as a collapsed child
            parent.collapsed_steps.push(step);
            // Also fold in any tool children this step already has
            if (step.collapsed_steps) {
                for (const child of step.collapsed_steps) {
                    parent.collapsed_steps.push(child);
                }
            }

            // Update or create tool_summary on the parent
            const childSteps = [step].concat(step.collapsed_steps || []);
            let addTokens = 0, addCost = 0, addLatency = 0, addCount = 0;
            for (const cs of childSteps) {
                const cp = cs.primary || cs;
                addTokens += (cp.prompt_tokens || 0) + (cp.completion_tokens || 0);
                addCost += cp.cost || 0;
                addLatency += cp.latency_ms || 0;
                addCount++;
            }

            if (parent.tool_summary) {
                parent.tool_summary.total_tokens += addTokens;
                parent.tool_summary.total_cost += addCost;
                parent.tool_summary.total_latency_ms += addLatency;
                parent.tool_summary.count += addCount;
            } else {
                parent.tool_summary = {
                    total_tokens: addTokens,
                    total_cost: addCost,
                    count: addCount,
                    total_latency_ms: addLatency,
                    model: step.primary.model || '',
                };
            }
        }
    }

    // Renumber
    let num = 1;
    for (const s of result) {
        s.step = num++;
    }
    return result;
}

function _getPromptId(step) {
    // Extract user_prompt_id from the step's primary event metadata.
    // Returns empty string if not present (non-Gemini calls).
    const p = step.primary;
    if (!p) return '';
    // Check metadata (stored by CostTracker)
    if (p.metadata && p.metadata.user_prompt_id) return p.metadata.user_prompt_id;
    // Check extra_json (how metadata is often serialized from DB)
    if (p.extra_json) {
        try {
            const extra = typeof p.extra_json === 'string' ? JSON.parse(p.extra_json) : p.extra_json;
            if (extra && extra.user_prompt_id) return extra.user_prompt_id;
        } catch (e) { /* ignore */ }
    }
    return '';
}

/**
 * Render tool child rows into the waterfall tbody.
 * @param {HTMLElement} tbody - The waterfall table body
 * @param {Array} childSteps - Array of step objects with .primary
 * @param {string} toolRowId - Base ID for the tool rows
 * @param {number} maxLatency - Max latency for duration bar scaling
 * @param {HTMLElement} [insertAfter] - If provided, insert rows after this element (lazy-load mode)
 */
function _renderToolChildRows(tbody, childSteps, toolRowId, maxLatency, insertAfter) {
    const fragment = document.createDocumentFragment();
    childSteps.forEach((cs, csIdx) => {
        const csPrimary = cs.primary;
        const csRowId = `${toolRowId}-${csIdx}`;
        const isLast = csIdx === childSteps.length - 1;
        const connector = isLast ? '\u2514\u2500' : '\u251C\u2500';
        const csBarClass = getDurationBarClass(csPrimary);
        // For tool children: show tool_name (ToolCallEvent) or prompt_preview
        // (tool-continuation LLMCallEvent which has tool names extracted)
        const isToolCont = csPrimary.event_type === 'llm_call' && csPrimary.is_tool_continuation;
        const isToolCall = csPrimary.event_type === 'tool_call';
        const toolLabel = csPrimary.tool_name || csPrimary.prompt_preview || '';
        const toolSecondary = (isToolCont || isToolCall)
            ? (csPrimary.model || '')
            : '';

        const csTr = document.createElement('tr');
        csTr.className = 'wf-row-tool-child';
        csTr.id = csRowId;
        csTr.innerHTML =
            `<td><span class="wf-tree-connector">${connector}</span></td>` +
            `<td>${getWaterfallTypeBadge(csPrimary)}</td>` +
            `<td style="color:var(--text-primary);font-size:11px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(toolLabel)}">${escapeHtml(toolLabel || '\u2014')}</td>` +
            `<td style="color:var(--text-secondary);font-size:11px;">${escapeHtml(toolSecondary)}</td>` +
            `<td>${createDurationBarHtml(csPrimary.latency_ms, maxLatency, csBarClass)}</td>` +
            `<td style="font-size:11px;">${formatWaterfallCost(csPrimary)}</td>` +
            `<td>${getWaterfallStatus(csPrimary)}</td>`;

        const csDetailRow = document.createElement('tr');
        csDetailRow.className = 'wf-row-detail';
        csDetailRow.id = csRowId + '-detail';
        csDetailRow.innerHTML = `<td colspan="7">${buildWaterfallDetail(csPrimary)}</td>`;

        csTr.addEventListener('click', (e) => {
            e.stopPropagation();
            csTr.classList.toggle('expanded');
            csDetailRow.classList.toggle('visible');
        });

        fragment.appendChild(csTr);
        fragment.appendChild(csDetailRow);
    });

    if (insertAfter && insertAfter.nextSibling) {
        // Insert after the summary row (for lazy-loaded rows)
        tbody.insertBefore(fragment, insertAfter.nextSibling);
    } else {
        tbody.appendChild(fragment);
    }
}

function _addToolLoadMoreRow(tbody, toolRowId, sessionId, parentStep, maxLatency, insertAfterSummary) {
    const loadMoreId = `${toolRowId}-loadmore`;
    // Remove existing load-more row if present
    const existing = document.getElementById(loadMoreId);
    if (existing) existing.remove();

    const tr = document.createElement('tr');
    tr.className = 'wf-row-tool-child visible';
    tr.id = loadMoreId;
    tr.innerHTML = `<td colspan="7" style="text-align:center;padding:8px 12px;">
        <button class="btn btn-sm load-more-btn" style="font-size:11px;">Load more tool steps</button>
    </td>`;

    const btn = tr.querySelector('button');
    btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        btn.disabled = true;
        btn.textContent = 'Loading\u2026';
        const cached = (_toolStepCache[sessionId] && _toolStepCache[sessionId][parentStep]) || [];
        const currentOffset = cached.length;
        const data = await fetchJSON(
            `/sessions/${encodeURIComponent(sessionId)}/events/tool-steps?parent_step=${parentStep}&limit=${_toolStepPageSize}&offset=${currentOffset}`
        );
        if (data && data.events && data.events.length > 0) {
            // Extend cache
            _toolStepCache[sessionId][parentStep] = cached.concat(data.events);
            // Build and render new child rows
            const childSteps = data.events.map(e2 => ({ primary: e2, subEvents: [] }));
            // Insert before the load-more row
            _renderToolChildRows(tbody, childSteps, `${toolRowId}-p${currentOffset}`, maxLatency, tr.previousElementSibling || insertAfterSummary);
            // Make the new rows visible
            tbody.querySelectorAll(`[id^="${toolRowId}-p${currentOffset}-"]`).forEach(el => {
                if (el.classList.contains('wf-row-tool-child')) el.classList.add('visible');
            });
            if (data.has_more) {
                btn.disabled = false;
                btn.textContent = 'Load more tool steps';
            } else {
                tr.remove();
            }
        } else {
            tr.remove();
        }
    });

    // Find the last tool child row to insert after
    const allToolChildren = tbody.querySelectorAll(`[id^="${toolRowId}-"]`);
    const lastChild = allToolChildren.length > 0 ? allToolChildren[allToolChildren.length - 1] : insertAfterSummary;
    if (lastChild && lastChild.nextSibling) {
        tbody.insertBefore(tr, lastChild.nextSibling);
    } else {
        tbody.appendChild(tr);
    }
}

function renderWaterfallTimeline(events, session) {
    const tbody = document.getElementById('waterfall-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!events || events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state" style="padding:32px;">No events yet</td></tr>';
        return;
    }

    try {
        const allSteps = groupEventsIntoSteps(events);

        // Collapse consecutive tool-continuation steps into summary rows
        const collapsedSteps = _collapseToolSteps(allSteps);

        // Group steps by user_prompt_id (Gemini CLI sends the same ID for
        // the parent call and subagent delegations belonging to one user prompt)
        const steps = _groupByPromptId(collapsedSteps);

        const isDurable = session && (session.durable || (session.metadata && session.metadata.durable));

        // Compute max latency for proportional bars.
        // Steps with nested tool runs show the combined duration (prompt + tools).
        let maxLatency = 0;
        for (const step of steps) {
            let lat = step.primary.latency_ms || 0;
            if (step.tool_summary) lat += step.tool_summary.total_latency_ms || 0;
            if (lat > maxLatency) maxLatency = lat;
            for (const sub of step.subEvents) {
                const sLat = sub.latency_ms || 0;
                if (sLat > maxLatency) maxLatency = sLat;
            }
        }
        // Ensure maxLatency has a floor to avoid division issues
        if (maxLatency <= 0) maxLatency = 1;

        for (const step of steps) {
            const primary = step.primary;
            const rowId = `wf-step-${step.step}`;

            // Checkpoint events render as divider rows
            if (primary.event_type === 'checkpoint') {
                const divRow = document.createElement('tr');
                divRow.className = 'wf-row-checkpoint';
                divRow.innerHTML = `<td colspan="7">` +
                    `<span class="wf-checkpoint-dot"></span>` +
                    `<span class="wf-checkpoint-label">${escapeHtml(primary.label || 'Checkpoint')}</span>` +
                    (primary.description ? ` <span style="color:var(--text-secondary);margin-left:8px;">${escapeHtml(primary.description)}</span>` : '') +
                    `</td>`;
                tbody.appendChild(divRow);
                continue;
            }

            // Fallback: standalone tool_steps_summary (edge case — tools with no parent)
            if (primary.event_type === 'tool_steps_summary') {
                const count = primary.count || 0;
                const tokens = (primary.total_tokens || 0).toLocaleString();
                const cost = _sessionBillingMode === 'subscription' ? '\u2014' : ((primary.total_cost || 0) > 0 ? '$' + primary.total_cost.toFixed(4) : '\u2014');
                const latMs = primary.total_latency_ms || 0;
                const latLabel = latMs >= 1000 ? (latMs / 1000).toFixed(1) + 's' : latMs.toFixed(0) + 'ms';
                const costPart = _sessionBillingMode === 'subscription' ? '' : ` \u00B7 ${cost}`;
                const summaryText = `${count} tool-use step${count !== 1 ? 's' : ''} \u00B7 ${tokens} tokens${costPart} \u00B7 ${latLabel}`;

                const summaryTr = document.createElement('tr');
                summaryTr.className = 'wf-row-tool-summary';
                summaryTr.id = rowId;
                summaryTr.innerHTML =
                    `<td><span class="wf-expand-icon">\u25B6</span> ${step.step}</td>` +
                    `<td><span class="wf-type-badge wf-type-tool-summary">Tools</span></td>` +
                    `<td colspan="3" style="color:var(--text-secondary);font-size:11px;">\u2514\u2500 ${escapeHtml(summaryText)}</td>` +
                    `<td style="font-size:11px;">${cost}</td>` +
                    `<td><span class="wf-status wf-status-ok">OK</span></td>`;
                tbody.appendChild(summaryTr);
                continue;
            }

            // Primary row — duration includes nested tool time when present
            const tr = document.createElement('tr');
            tr.className = 'wf-row-primary';
            tr.id = rowId;

            const model = String(primary.model || primary.local_model || primary.cloud_model || primary.original_model || primary.pii_type || '\u2014');
            const barClass = getDurationBarClass(primary);
            const durableMarker = (isDurable && primary.has_cached_response)
                ? '<span class="wf-durable-marker" title="Durable checkpoint"></span>' : '';
            const isShadow = primary.event_type === 'shadow_draft';
            let promptPreview = primary.prompt_preview || '';
            if (primary.event_type === 'session_lifecycle') {
                const action = primary.action || '';
                if (action === 'timed_out') {
                    const reason = (primary.reason || '').replace('_', ' ');
                    promptPreview = `${reason} — ${primary.elapsed || 0}s elapsed (limit ${primary.limit || 0}s)`;
                } else if (action === 'cancelled') {
                    promptPreview = 'Session cancelled';
                } else {
                    promptPreview = action.replace('_', ' ');
                }
            }
            const toolLatency = step.tool_summary ? (step.tool_summary.total_latency_ms || 0) : 0;
            const combinedLatency = (isShadow ? (primary.local_latency_ms || 0) : (primary.latency_ms || 0)) + toolLatency;

            tr.innerHTML =
                `<td><span class="wf-expand-icon">\u25B6</span> ${step.step}</td>` +
                `<td>${getWaterfallTypeBadge(primary)}${durableMarker}</td>` +
                `<td style="color:var(--text-primary);font-size:11px;">${escapeHtml(model)}</td>` +
                `<td style="color:var(--text-secondary);font-size:11px;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(promptPreview)}">${escapeHtml(promptPreview)}</td>` +
                `<td>${createDurationBarHtml(combinedLatency, maxLatency, barClass)}</td>` +
                `<td style="font-size:11px;">${formatWaterfallCost(primary)}</td>` +
                `<td>${getWaterfallStatus(primary)}</td>`;

            // Detail row for primary
            const detailRow = document.createElement('tr');
            detailRow.className = 'wf-row-detail';
            detailRow.id = rowId + '-detail';
            detailRow.innerHTML = `<td colspan="7">${buildWaterfallDetail(primary)}</td>`;

            tr.addEventListener('click', () => {
                tr.classList.toggle('expanded');
                detailRow.classList.toggle('visible');
            });

            tbody.appendChild(tr);
            tbody.appendChild(detailRow);

            // Sub-event rows
            step.subEvents.forEach((sub, idx) => {
                const isLast = idx === step.subEvents.length - 1 && !step.tool_summary;
                const connector = isLast ? '\u2514\u2500' : '\u251C\u2500';
                const subRowId = `${rowId}-sub-${idx}`;
                const subBarClass = getDurationBarClass(sub);

                const subTr = document.createElement('tr');
                subTr.className = 'wf-row-sub';
                subTr.id = subRowId;
                subTr.innerHTML =
                    `<td><span class="wf-tree-connector">${connector}</span></td>` +
                    `<td>${getWaterfallTypeBadge(sub)}</td>` +
                    `<td style="color:var(--text-secondary);font-size:11px;">${escapeHtml(String(getSubRowContext(sub)))}</td>` +
                    `<td></td>` +
                    `<td>${createDurationBarHtml(sub.latency_ms, maxLatency, subBarClass)}</td>` +
                    `<td style="font-size:11px;">${formatWaterfallCost(sub)}</td>` +
                    `<td>${getWaterfallStatus(sub)}</td>`;

                const subDetailRow = document.createElement('tr');
                subDetailRow.className = 'wf-row-detail';
                subDetailRow.id = subRowId + '-detail';
                subDetailRow.innerHTML = `<td colspan="7">${buildWaterfallDetail(sub)}</td>`;

                subTr.addEventListener('click', () => {
                    subTr.classList.toggle('expanded');
                    subDetailRow.classList.toggle('visible');
                });

                tbody.appendChild(subTr);
                tbody.appendChild(subDetailRow);
            });

            // Nested tool summary row (tool continuations triggered by this prompt)
            if (step.tool_summary) {
                const ts = step.tool_summary;
                const tsCount = ts.count || 0;
                const tsTokens = (ts.total_tokens || 0).toLocaleString();
                const tsCost = _sessionBillingMode === 'subscription' ? '\u2014' : ((ts.total_cost || 0) > 0 ? '$' + ts.total_cost.toFixed(4) : '\u2014');
                const tsLatMs = ts.total_latency_ms || 0;
                const tsLatLabel = tsLatMs >= 1000 ? (tsLatMs / 1000).toFixed(1) + 's' : tsLatMs.toFixed(0) + 'ms';
                const tsCostPart = _sessionBillingMode === 'subscription' ? '' : ` \u00B7 ${tsCost}`;
                const tsSummaryText = `${tsCount} tool-use step${tsCount !== 1 ? 's' : ''} \u00B7 ${tsTokens} tokens${tsCostPart} \u00B7 ${tsLatLabel}`;
                const toolRowId = `${rowId}-tools`;

                const toolSummaryTr = document.createElement('tr');
                toolSummaryTr.className = 'wf-row-tool-summary';
                toolSummaryTr.id = toolRowId;
                toolSummaryTr.innerHTML =
                    `<td><span class="wf-expand-icon">\u25B6</span></td>` +
                    `<td><span class="wf-type-badge wf-type-tool-summary">Tools</span></td>` +
                    `<td colspan="3" style="color:var(--text-secondary);font-size:11px;">\u2514\u2500 ${escapeHtml(tsSummaryText)}</td>` +
                    `<td style="font-size:11px;">${tsCost}</td>` +
                    `<td><span class="wf-status wf-status-ok">OK</span></td>`;

                tbody.appendChild(toolSummaryTr);

                // Lazy-load mode (primary_only): collapsed_steps is null,
                // child rows are fetched on demand from /events/tool-steps.
                // Legacy mode: collapsed_steps is a pre-populated array.
                const isLazy = step.collapsed_steps === null;

                if (!isLazy) {
                    // Legacy: render child rows upfront (hidden)
                    const childSteps = step.collapsed_steps || [];
                    _renderToolChildRows(tbody, childSteps, toolRowId, maxLatency);
                }

                // Capture the parent event step for lazy-fetch
                const parentEventStep = primary.step;

                // Toggle expansion on tool summary row click
                toolSummaryTr.addEventListener('click', async () => {
                    const isExpanding = !toolSummaryTr.classList.contains('expanded');
                    toolSummaryTr.classList.toggle('expanded');

                    if (isLazy && isExpanding) {
                        // Lazy-load: check cache first, then fetch
                        const sid = currentDetailSessionId;
                        const cacheKey = parentEventStep;
                        if (!_toolStepCache[sid]) _toolStepCache[sid] = {};

                        if (_toolStepCache[sid][cacheKey]) {
                            // Already fetched — just toggle visibility
                            const existing = tbody.querySelectorAll(`[id^="${toolRowId}-"]`);
                            existing.forEach(el => {
                                if (el.classList.contains('wf-row-tool-child')) {
                                    el.classList.toggle('visible');
                                }
                                if (!toolSummaryTr.classList.contains('expanded') && el.classList.contains('wf-row-detail')) {
                                    el.classList.remove('visible');
                                }
                            });
                        } else {
                            // Fetch tool sub-steps from server
                            const data = await fetchJSON(
                                `/sessions/${encodeURIComponent(sid)}/events/tool-steps?parent_step=${parentEventStep}&limit=${_toolStepPageSize}`
                            );
                            if (data && data.events) {
                                _toolStepCache[sid][cacheKey] = data.events;
                                // Create pseudo-step objects for rendering
                                const childSteps = data.events.map(e => ({
                                    primary: e,
                                    subEvents: [],
                                }));
                                _renderToolChildRows(tbody, childSteps, toolRowId, maxLatency, toolSummaryTr);
                                // Make them visible immediately
                                const newChildren = tbody.querySelectorAll(`[id^="${toolRowId}-"]`);
                                newChildren.forEach(el => {
                                    if (el.classList.contains('wf-row-tool-child')) {
                                        el.classList.add('visible');
                                    }
                                });
                                // Add "Load more tool steps" row if has_more
                                if (data.has_more) {
                                    _addToolLoadMoreRow(tbody, toolRowId, sid, parentEventStep, maxLatency, toolSummaryTr);
                                }
                            }
                        }
                    } else {
                        // Legacy mode or collapsing — toggle existing children
                        const children = tbody.querySelectorAll(`[id^="${toolRowId}-"]`);
                        children.forEach(el => {
                            if (el.classList.contains('wf-row-tool-child')) {
                                el.classList.toggle('visible');
                            }
                            if (!toolSummaryTr.classList.contains('expanded') && el.classList.contains('wf-row-detail')) {
                                el.classList.remove('visible');
                            }
                        });
                    }
                });
            }
        }
    } catch (err) {
        console.error('Waterfall render error:', err);
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state" style="padding:32px;">Error rendering waterfall timeline</td></tr>';
    }
}

function switchWaterfallMode(mode) {
    const waterfall = document.getElementById('waterfall-container');
    const flatTimeline = document.getElementById('step-timeline');
    const flatEvents = document.getElementById('detail-events');

    document.querySelectorAll('.waterfall-toggle-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    if (mode === 'waterfall') {
        if (waterfall) waterfall.style.display = '';
        if (flatTimeline) flatTimeline.style.display = 'none';
        if (flatEvents) flatEvents.style.display = 'none';
    } else {
        if (waterfall) waterfall.style.display = 'none';
        if (flatTimeline) flatTimeline.style.display = '';
        if (flatEvents) flatEvents.style.display = '';
    }
}

async function loadLocalStatus() {
    const [status, config] = await Promise.all([
        fetchJSON('/local/status'),
        fetchJSON('/config'),
    ]);

    const indicator = document.getElementById('local-ollama-status');
    if (status.available) {
        indicator.innerHTML = '<span class="status-online">Online</span>';
    } else {
        indicator.innerHTML = '<span class="status-offline">Offline</span>';
    }

    // Force-local and auto-route toggles
    document.getElementById('force-local-toggle').checked = config.auto_route_force_local || false;
    document.getElementById('auto-route-toggle').checked = config.auto_route_enabled || false;

    // Populate active model dropdown from downloaded models
    const activeSelect = document.getElementById('active-model-select');
    const currentActiveModel = config.local_model_default;
    activeSelect.innerHTML = '<option value="">None</option>';

    if (status.available) {
        try {
            const modelsData = await fetchJSON('/local/models');
            if (!modelsData) return;
            let matchedActive = '';
            (modelsData.models || []).forEach(m => {
                const name = m.name || m.model || '';
                const nameBase = name.replace(/:latest$/, '');
                // Active model dropdown
                const opt2 = document.createElement('option');
                opt2.value = name;
                opt2.textContent = name;
                if (name === currentActiveModel || nameBase === currentActiveModel) matchedActive = name;
                activeSelect.appendChild(opt2);
            });
            if (matchedActive) activeSelect.value = matchedActive;
        } catch (e) { console.warn('Failed to load local models for dropdowns:', e); }
    }
}

async function loadDownloadedModels() {
    const tbody = document.getElementById('local-models-tbody');
    try {
        const data = await fetchJSON('/local/models');
        if (!data) return;
        const models = data.models || [];

        if (models.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No models downloaded</td></tr>';
            return;
        }

        tbody.innerHTML = '';
        models.forEach(m => {
            const name = m.name || m.model || '?';
            const size = m.size ? (m.size / (1024 * 1024 * 1024)).toFixed(1) + ' GB' : '-';
            const modified = m.modified_at ? new Date(m.modified_at).toLocaleDateString() : '-';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${escapeHtml(name)}</td>
                <td>${escapeHtml(size)}</td>
                <td>${escapeHtml(modified)}</td>
                <td><button class="btn btn-danger btn-sm">Delete</button></td>
            `;
            tr.querySelector('button').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteModel(name);
            });
            tbody.appendChild(tr);
        });

        // Populate hot-swap dropdown from downloaded models
        const hotSwapSelect = document.getElementById('hot-swap-model');
        if (hotSwapSelect) {
            const currentVal = hotSwapSelect.value;
            hotSwapSelect.innerHTML = '<option value="">Select model...</option>';
            models.forEach(m => {
                const name = m.name || m.model || '?';
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                hotSwapSelect.appendChild(opt);
            });
            if (currentVal) hotSwapSelect.value = currentVal;
        }
    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="4" class="empty-state">Ollama unavailable</td></tr>';
    }
}

async function loadRecommendations() {
    const container = document.getElementById('recommended-models');
    const hwEl = document.getElementById('hw-summary');
    try {
        const data = await fetchJSON('/local/recommend');
        if (!data) return;
        const hw = data.hardware || {};
        const models = data.recommended_models || [];

        hwEl.style.display = 'block';
        const parts = [];
        if (hw.ram_gb) parts.push(`${hw.ram_gb} GB RAM`);
        if (hw.gpu_name) parts.push(hw.gpu_name);
        if (hw.disk_free_gb) parts.push(`${hw.disk_free_gb} GB free`);
        if (hw.arch) parts.push(hw.arch);
        hwEl.textContent = parts.join(' \u00b7 ');

        if (models.length === 0) {
            container.innerHTML = '<div class="empty-state">No recommendations available</div>';
            return;
        }

        container.innerHTML = '';
        models.forEach(m => {
            const tierClass = 'tier-' + (m.tier || 'light').replace('_', '-');
            const card = document.createElement('div');
            card.className = 'model-card';
            const modelName = m.model || m.name || '?';
            card.innerHTML = `
                <div class="model-name">
                    ${escapeHtml(modelName)}
                    <span class="tier-badge ${escapeHtml(tierClass)}">${escapeHtml((m.tier || '').replace('_', '-'))}</span>
                </div>
                <div class="model-meta">${m.size_gb ? escapeHtml(String(m.size_gb)) + ' GB' : ''} ${escapeHtml(m.parameters || '')}</div>
                <div class="model-desc">${escapeHtml(m.description || '')}</div>
                <div class="model-actions">
                    <button class="btn btn-primary btn-sm">Pull</button>
                </div>
            `;
            card.querySelector('button').addEventListener('click', function() {
                pullModel(modelName, this);
            });
            container.appendChild(card);
        });
    } catch (e) {
        hwEl.style.display = 'none';
        container.innerHTML = '<div class="empty-state">Could not load recommendations</div>';
    }
}

// --- Model Testing ---

async function loadModelTesting() {
    // Check enterprise license for model testing configuration
    let modelTestingLicensed = false;
    let modelTestingConfigurable = false;
    try {
        const licData = await fetchJSON('/license');
        if (licData && licData.features && licData.features.model_testing) {
            modelTestingLicensed = licData.features.model_testing.enabled;
        }
        modelTestingConfigurable = modelTestingLicensed && !!(licData && licData.valid);
    } catch (e) { /* license endpoint unavailable — default unlicensed */ }
    const banner = document.getElementById('mt-enterprise-banner');
    if (banner) banner.style.display = modelTestingLicensed ? 'none' : 'block';
    window._modelTestingLicensed = modelTestingLicensed;
    window._modelTestingConfigurable = modelTestingConfigurable;

    if (!modelTestingLicensed) {
        return;
    }

    // Unlicensed: lock configuration section, metrics still load
    if (!modelTestingConfigurable) {
        const cfgSection = document.getElementById('mt-config-section');
        if (cfgSection) {
            cfgSection.style.opacity = '0.45';
            cfgSection.style.pointerEvents = 'none';
            if (!document.getElementById('mt-config-lock-note')) {
                const note = document.createElement('p');
                note.id = 'mt-config-lock-note';
                note.style.cssText = 'color:var(--text-secondary);font-size:13px;margin:4px 0 12px;';
                note.textContent = 'Configuration requires an enterprise license. Set STATELOOM_LICENSE_KEY to unlock.';
                cfgSection.parentNode.insertBefore(note, cfgSection);
            }
        }
    } else {
        const [config] = await Promise.all([fetchJSON('/config')]);
        const toggle = document.getElementById('mt-toggle');
        if (toggle && config) toggle.checked = config.shadow_enabled || false;
        const rateSlider = document.getElementById('mt-sample-rate');
        const rateLabel = document.getElementById('mt-sample-label');
        if (rateSlider && config) {
            const pct = Math.round((config.shadow_sample_rate || 1.0) * 100);
            rateSlider.value = pct;
            if (rateLabel) rateLabel.textContent = pct + '%';
        }
        const maxTokensInput = document.getElementById('mt-max-tokens');
        if (maxTokensInput && config) maxTokensInput.value = config.shadow_max_context_tokens || 8192;

        // Similarity method selector
        const simSelect = document.getElementById('mt-similarity-method');
        const simStatus = document.getElementById('mt-similarity-status');
        if (simSelect && config) {
            const simVal = config.shadow_similarity_method || 'auto';
            simSelect.value = simVal;
            simSelect.dataset.prevValue = simVal;
        }
        if (simStatus) {
            try {
                const shadowCfg = await fetchJSON('/shadow/config');
                const semanticAvail = shadowCfg?.semantic_available;
                window._semanticAvailable = !!semanticAvail;
                if (semanticAvail) {
                    simStatus.innerHTML = '<span style="color:var(--accent-green, #4ade80);">sentence-transformers installed</span>';
                } else {
                    simStatus.innerHTML = '<span style="color:var(--accent-yellow, #facc15);">sentence-transformers not installed</span>' +
                        '<br><code style="font-size:11px;color:var(--text-secondary);">pip install sentence-transformers</code>';
                }
            } catch (e) {
                simStatus.textContent = '';
            }
        }

        // Populate model dropdown from downloaded local models
        try {
            const localData = await fetchJSON('/local/models');
            const select = document.getElementById('mt-model-select');
            if (select && localData) {
                const currentVal = config?.shadow_model || '';
                select.innerHTML = '<option value="">None</option>';
                (localData.models || []).forEach(m => {
                    const name = m.name || m.model || '?';
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = name;
                    select.appendChild(opt);
                });
                if (currentVal) select.value = currentVal;
            }
        } catch (e) { /* Ollama unavailable */ }
    }

    // Load metrics + readiness in parallel
    await Promise.all([loadMTMetrics(), loadMTReadiness()]);
}

async function loadMTMetrics() {
    try {
        const m = await fetchJSON('/shadow/metrics');
        if (!m) return;
        document.getElementById('mt-total-calls').textContent = m.total_calls || 0;
        document.getElementById('mt-success-rate').textContent =
            m.total_calls > 0 ? (m.success_rate * 100).toFixed(0) + '%' : '0%';
        document.getElementById('mt-cost-saved').textContent =
            '$' + (m.total_cost_saved || 0).toFixed(4);
        const simEl = document.getElementById('mt-avg-similarity');
        if (m.avg_similarity != null) {
            const pct = (m.avg_similarity * 100).toFixed(0);
            simEl.textContent = pct + '%';
            if (m.avg_similarity >= 0.7) simEl.style.color = 'var(--accent-green, #4ade80)';
            else if (m.avg_similarity >= 0.4) simEl.style.color = 'var(--accent-yellow, #facc15)';
            else simEl.style.color = 'var(--accent-red, #f87171)';
        } else {
            simEl.textContent = '\u2014';
            simEl.style.color = '';
        }

        // Filtering funnel
        const skipReasons = m.skip_reasons || {};
        const totalSkipped = Object.values(skipReasons).reduce((a, b) => a + b, 0);
        const tested = m.total_calls || 0;
        const eligible = tested + (skipReasons.sampling || 0);
        const totalCalls = eligible + totalSkipped - (skipReasons.sampling || 0);

        document.getElementById('mt-funnel-total').textContent = totalCalls;
        document.getElementById('mt-funnel-eligible').textContent = eligible;
        document.getElementById('mt-funnel-tested').textContent = tested;

        if (totalCalls > 0) {
            document.getElementById('mt-funnel-eligible-bar').style.width = Math.round(eligible / totalCalls * 100) + '%';
            document.getElementById('mt-funnel-tested-bar').style.width = Math.round(tested / totalCalls * 100) + '%';
        }

        // Skip reason breakdown
        const reasonsEl = document.getElementById('mt-skip-reasons');
        if (reasonsEl && totalSkipped > 0) {
            const labels = {
                tool_continuation: 'Tool calls',
                unsupported_features: 'Unsupported features',
                images: 'Image content',
                realtime_data: 'Realtime data',
                context_too_large: 'Context too large',
                sampling: 'Sampling',
            };
            const parts = Object.entries(skipReasons)
                .filter(([, v]) => v > 0)
                .map(([k, v]) => `${labels[k] || k}: ${v}`)
                .join(' · ');
            reasonsEl.textContent = 'Skipped: ' + parts;
        }
    } catch (e) { /* metrics unavailable */ }
}

async function loadMTReadiness() {
    try {
        const data = await fetchJSON('/shadow/readiness');
        if (!data || !data.models) return;
        const container = document.getElementById('mt-readiness');
        const models = data.models;
        if (Object.keys(models).length === 0) return;

        const emptyCard = document.getElementById('mt-readiness-empty-card');
        if (emptyCard) emptyCard.style.display = 'none';

        // Remove previous readiness cards (keep empty-state card)
        container.querySelectorAll('.mt-readiness-card').forEach(el => el.remove());

        for (const [model, info] of Object.entries(models)) {
            const card = document.createElement('div');
            card.className = 'stat-card mt-readiness-card';

            const scoreColor = info.score >= 80 ? 'var(--accent-green, #4ade80)' :
                               info.score >= 60 ? 'var(--accent-yellow, #facc15)' :
                               'var(--accent-red, #f87171)';

            const dist = info.similarity_distribution || {};
            const distTotal = (dist.excellent || 0) + (dist.good || 0) + (dist.fair || 0) + (dist.poor || 0);

            let distBar = '';
            if (distTotal > 0) {
                const pctE = (dist.excellent / distTotal * 100).toFixed(0);
                const pctG = (dist.good / distTotal * 100).toFixed(0);
                const pctF = (dist.fair / distTotal * 100).toFixed(0);
                const pctP = (dist.poor / distTotal * 100).toFixed(0);
                distBar = `<div class="mt-dist-bar">
                    <div class="mt-dist-seg mt-dist-excellent" style="width:${pctE}%" title="Excellent (≥90%): ${dist.excellent}"></div>
                    <div class="mt-dist-seg mt-dist-good" style="width:${pctG}%" title="Good (70-90%): ${dist.good}"></div>
                    <div class="mt-dist-seg mt-dist-fair" style="width:${pctF}%" title="Fair (40-70%): ${dist.fair}"></div>
                    <div class="mt-dist-seg mt-dist-poor" style="width:${pctP}%" title="Poor (<40%): ${dist.poor}"></div>
                </div>`;
            }

            card.innerHTML = `
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <span style="font-weight:600;font-size:15px;">${escapeHtml(model)}</span>
                    <span class="mt-score-badge" style="background:${scoreColor};color:#000;padding:4px 10px;border-radius:12px;font-weight:700;font-size:14px;">${info.score}/100</span>
                </div>
                <div style="font-size:13px;color:var(--text-secondary);margin-bottom:10px;">${escapeHtml(info.recommendation)}</div>
                <div class="mt-score-breakdown">
                    <span>Quality: ${info.quality}/40</span>
                    <span>Reliability: ${info.reliability}/30</span>
                    <span>Speed: ${info.speed}/20</span>
                    <span>Confidence: ${info.confidence}/10</span>
                </div>
                ${distBar}
                <div style="font-size:12px;color:var(--text-secondary);margin-top:8px;">
                    ${info.total_tests} tests · ${info.avg_similarity != null ? (info.avg_similarity * 100).toFixed(0) + '% avg sim' : 'no similarity data'}
                    ${info.success_rate != null ? ' · ' + (info.success_rate * 100).toFixed(0) + '% success' : ''}
                </div>
            `;
            container.appendChild(card);
        }
    } catch (e) { /* readiness unavailable */ }
}

async function toggleShadow(enabled) {
    if (!window._modelTestingConfigurable) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        const t = document.getElementById('mt-toggle'); if (t) t.checked = !enabled;
        return;
    }
    const res = await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shadow_enabled: enabled}),
    });
    if (res.status === 403) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        const toggle = document.getElementById('mt-toggle');
        if (toggle) toggle.checked = !enabled;
    }
}

async function updateShadowModel(model) {
    const select = document.getElementById('mt-model-select');
    const prevVal = select ? select.dataset.prevValue || '' : '';
    if (!window._modelTestingConfigurable) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (select) select.value = prevVal;
        return;
    }
    const res = await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shadow_model: model}),
    });
    if (res.status === 403) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (select) select.value = prevVal;
    } else if (select) {
        select.dataset.prevValue = model;
    }
}

async function updateMTSampleRate(rate) {
    const slider = document.getElementById('mt-sample-rate');
    const label = document.getElementById('mt-sample-label');
    const prevVal = slider ? slider.dataset.prevValue || '100' : '100';
    if (!window._modelTestingConfigurable) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (slider) { slider.value = prevVal; if (label) label.textContent = prevVal + '%'; }
        return;
    }
    const res = await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shadow_sample_rate: rate}),
    });
    if (res.status === 403) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (slider) { slider.value = prevVal; if (label) label.textContent = prevVal + '%'; }
    } else if (slider) {
        slider.dataset.prevValue = Math.round(rate * 100);
    }
}

async function updateMTMaxTokens(tokens) {
    const val = parseInt(tokens, 10);
    if (isNaN(val) || val < 256) return;
    const input = document.getElementById('mt-max-tokens');
    const prevVal = input ? input.dataset.prevValue || '8192' : '8192';
    if (!window._modelTestingConfigurable) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (input) input.value = prevVal;
        return;
    }
    const res = await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shadow_max_context_tokens: val}),
    });
    if (res.status === 403) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (input) input.value = prevVal;
    } else if (input) {
        input.dataset.prevValue = val;
    }
}

async function updateSimilarityMethod(method) {
    const select = document.getElementById('mt-similarity-method');
    const prevVal = select ? select.dataset.prevValue || 'auto' : 'auto';
    if (!window._modelTestingConfigurable) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (select) select.value = prevVal;
        return;
    }
    if (method === 'semantic' && !window._semanticAvailable) {
        showToast('sentence-transformers is not installed. Install with: pip install sentence-transformers', 'error');
        if (select) select.value = prevVal;
        return;
    }
    const res = await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shadow_similarity_method: method}),
    });
    if (res.status === 403) {
        showToast('Model Testing configuration requires an enterprise license', 'error');
        if (select) select.value = prevVal;
    } else if (select) {
        select.dataset.prevValue = method;
        showToast('Similarity method updated to ' + method, 'success');
    }
}

async function pullModel(model, btn) {
    btn.disabled = true;
    btn.textContent = '0%';

    // Helper: poll GET /local/pull/{model}/progress (pull thread runs server-side)
    function fallbackPoll() {
        const pollId = setInterval(async () => {
            try {
                const progress = await fetchJSON(`/local/pull/${encodeURIComponent(model)}/progress`);
                if (progress.status === 'pulling') {
                    btn.textContent = Math.round(progress.progress_pct || 0) + '%';
                    btn.title = progress.detail || '';
                } else if (progress.status === 'complete') {
                    clearInterval(pollId);
                    btn.textContent = 'Done!';
                    loadDownloadedModels();
                    loadLocalStatus();
                    setTimeout(() => { btn.disabled = false; btn.textContent = 'Pull'; }, 2000);
                } else if (progress.status === 'error') {
                    clearInterval(pollId);
                    btn.textContent = 'Error';
                    setTimeout(() => { btn.disabled = false; btn.textContent = 'Pull'; }, 3000);
                }
            } catch (err) {
                clearInterval(pollId);
                btn.disabled = false;
                btn.textContent = 'Pull';
            }
        }, 1000);
    }

    // Helper: process a parsed NDJSON object. Returns true if terminal.
    function handleData(data) {
        if (data.status === 'pulling') {
            btn.textContent = Math.round(data.progress_pct || 0) + '%';
            btn.title = data.detail || '';
            return false;
        } else if (data.status === 'complete') {
            btn.textContent = 'Done!';
            loadDownloadedModels();
            loadLocalStatus();
            setTimeout(() => { btn.disabled = false; btn.textContent = 'Pull'; }, 2000);
            return true;
        } else if (data.status === 'error') {
            btn.textContent = 'Error';
            btn.title = data.error || '';
            setTimeout(() => { btn.disabled = false; btn.textContent = 'Pull'; }, 3000);
            return true;
        }
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/local/pull`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: model}),
        });

        // Non-200 (e.g. 422 validation error) — fall back to polling
        if (!response.ok || !response.body) {
            console.warn('Pull POST returned', response.status, '— falling back to polling');
            fallbackPoll();
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let resolved = false;

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep partial trailing line

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue; // Skip keepalive
                let data;
                try { data = JSON.parse(trimmed); } catch (e) { continue; }
                if (handleData(data)) { resolved = true; return; }
            }
        }

        // Process any remaining data left in buffer
        if (!resolved && buffer.trim()) {
            try {
                const data = JSON.parse(buffer.trim());
                if (handleData(data)) { resolved = true; return; }
            } catch (e) { /* ignore parse error */ }
        }

        // Stream ended without terminal status — fall back to polling
        if (!resolved) {
            console.warn('Stream ended without completion status — falling back to polling');
            fallbackPoll();
        }

    } catch (e) {
        // Network error or body reading failed — fall back to polling
        console.warn('Streaming pull failed, falling back to polling:', e);
        fallbackPoll();
    }
}

async function deleteModel(model) {
    await fetch(`${API_BASE}/local/models/${encodeURIComponent(model)}`, {
        method: 'DELETE',
    });
    loadDownloadedModels();
    loadLocalStatus();
}

// --- Feedback ---
async function loadFeedback(sessionId) {
    const section = document.getElementById('detail-feedback-section');
    const data = await fetchJSON(`/sessions/${sessionId}/feedback`);
    if (!data) return;

    if (data.feedback) {
        const fb = data.feedback;
        section.innerHTML = `
            <div class="feedback-display">
                <h4>Feedback</h4>
                <div class="feedback-display-row">
                    <span class="fb-label">Rating</span>
                    <span class="fb-value"><span class="status-badge status-feedback-${fb.rating}">${fb.rating}</span></span>
                </div>
                ${fb.score !== null && fb.score !== undefined ? `
                <div class="feedback-display-row">
                    <span class="fb-label">Score</span>
                    <span class="fb-value" style="font-family:var(--font-mono)">${fb.score.toFixed(2)}</span>
                </div>` : ''}
                ${fb.comment ? `
                <div class="feedback-display-row">
                    <span class="fb-label">Comment</span>
                    <span class="fb-value">${escapeHtml(fb.comment)}</span>
                </div>` : ''}
                <div class="feedback-display-row">
                    <span class="fb-label">Submitted</span>
                    <span class="fb-value" style="font-size:12px;color:var(--text-muted)">${new Date(fb.created_at).toLocaleString()}</span>
                </div>
            </div>
        `;
    } else {
        section.innerHTML = `
            <div class="feedback-form" id="feedback-form">
                <h4>Submit Feedback</h4>
                <div class="feedback-rating-buttons">
                    <button class="feedback-btn" data-rating="success">Success</button>
                    <button class="feedback-btn" data-rating="failure">Failure</button>
                    <button class="feedback-btn" data-rating="partial">Partial</button>
                </div>
                <div class="feedback-score-row">
                    <label>Score</label>
                    <input type="checkbox" id="fb-score-toggle">
                    <input type="range" id="fb-score-slider" min="0" max="1" step="0.05" value="0.5" disabled>
                    <span class="score-val" id="fb-score-val">-</span>
                </div>
                <textarea class="feedback-textarea" id="fb-comment" placeholder="Optional comment..."></textarea>
                <button class="btn btn-primary btn-sm" id="fb-submit-btn" disabled>Submit</button>
            </div>
        `;
        // Bind events safely (no inline handlers)
        section.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', () => selectRating(btn));
        });
        const scoreToggle = document.getElementById('fb-score-toggle');
        if (scoreToggle) {
            scoreToggle.addEventListener('change', function() {
                document.getElementById('fb-score-slider').disabled = !this.checked;
                document.getElementById('fb-score-val').textContent = this.checked ? document.getElementById('fb-score-slider').value : '-';
            });
        }
        const scoreSlider = document.getElementById('fb-score-slider');
        if (scoreSlider) {
            scoreSlider.addEventListener('input', function() {
                document.getElementById('fb-score-val').textContent = this.value;
            });
        }
        const submitBtn = document.getElementById('fb-submit-btn');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => submitFeedback(sessionId));
        }
    }
}

function selectRating(btn) {
    document.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('feedback-btn-active'));
    btn.classList.add('feedback-btn-active');
    document.getElementById('fb-submit-btn').disabled = false;
}

async function submitFeedback(sessionId) {
    const activeBtn = document.querySelector('.feedback-btn.feedback-btn-active');
    if (!activeBtn) return;

    const rating = activeBtn.dataset.rating;
    const scoreEnabled = document.getElementById('fb-score-toggle').checked;
    const score = scoreEnabled ? parseFloat(document.getElementById('fb-score-slider').value) : null;
    const comment = document.getElementById('fb-comment').value;

    document.getElementById('fb-submit-btn').disabled = true;
    document.getElementById('fb-submit-btn').textContent = 'Submitting...';

    await fetch(`${API_BASE}/sessions/${sessionId}/feedback`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ rating, score, comment }),
    });

    loadFeedback(sessionId);
}

// --- Experiments ---
async function loadExperiments() {
    const data = await fetchJSON('/experiments');
    if (!data) return;
    const tbody = document.getElementById('experiments-tbody');
    tbody.innerHTML = '';

    if (!data.experiments || data.experiments.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No experiments yet</td></tr>';
        return;
    }

    data.experiments.forEach(e => {
        const totalAssignments = Object.values(e.assignment_counts || {}).reduce((a, b) => a + b, 0);
        const tr = document.createElement('tr');
        tr.onclick = () => loadExperimentDetail(e.id);
        tr.innerHTML = `
            <td>${escapeHtml(e.id.substring(0, 8))}</td>
            <td>${escapeHtml(e.name)}</td>
            <td><span class="status-badge status-${escapeHtml(e.status)}">${escapeHtml(e.status)}</span></td>
            <td>${escapeHtml(e.strategy)}</td>
            <td>${e.variant_count}</td>
            <td>${totalAssignments}</td>
            <td>${new Date(e.created_at).toLocaleDateString()}</td>
        `;
        tbody.appendChild(tr);
    });
}

async function loadExperimentDetail(id) {
    const data = await fetchJSON(`/experiments/${id}`);
    if (!data) return;
    const panel = document.getElementById('experiment-detail-panel');

    document.getElementById('exp-detail-name').textContent = data.name;
    const statusEl = document.getElementById('exp-detail-status');
    statusEl.textContent = data.status;
    statusEl.className = `status-badge status-${data.status}`;
    document.getElementById('exp-detail-desc').textContent = data.description || 'No description';

    const variantMetrics = (data.metrics && data.metrics.variants) || {};
    const container = document.getElementById('exp-detail-variants');
    container.innerHTML = '';

    (data.variants || []).forEach(v => {
        const vm = variantMetrics[v.name] || {};
        const card = document.createElement('div');
        card.className = 'variant-card';
        card.innerHTML = `
            <div class="variant-name">${escapeHtml(v.name)}</div>
            <div class="variant-weight">Weight: ${v.weight}</div>
            ${v.model ? `<div class="variant-model">${escapeHtml(v.model)}</div>` : ''}
            <div class="variant-metrics">
                <div class="variant-metric">
                    <span class="metric-label">Sessions</span>
                    <span class="metric-value">${vm.session_count || 0}</span>
                </div>
                <div class="variant-metric">
                    <span class="metric-label">Success Rate</span>
                    <span class="metric-value">${vm.success_rate != null ? (vm.success_rate * 100).toFixed(1) + '%' : '-'}</span>
                </div>
                <div class="variant-metric">
                    <span class="metric-label">Avg Cost</span>
                    <span class="metric-value">${vm.avg_cost != null ? '$' + vm.avg_cost.toFixed(4) : '-'}</span>
                </div>
                <div class="variant-metric">
                    <span class="metric-label">Assignments</span>
                    <span class="metric-value">${(data.assignment_counts || {})[v.name] || 0}</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    });

    // Action buttons based on experiment status
    const actionsEl = document.getElementById('exp-detail-actions');
    actionsEl.innerHTML = '';
    const btnStyle = 'padding:4px 12px;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:600;';
    if (data.status === 'draft') {
        actionsEl.innerHTML =
            `<button style="${btnStyle}background:#28a745;color:#fff;" onclick="expAction('${id}','start')">Start</button>`;
    } else if (data.status === 'running') {
        actionsEl.innerHTML =
            `<button style="${btnStyle}background:#ffc107;color:#212529;" onclick="expAction('${id}','pause')">Pause</button>` +
            `<button style="${btnStyle}background:#dc3545;color:#fff;" onclick="expAction('${id}','conclude')">Conclude</button>`;
    } else if (data.status === 'paused') {
        actionsEl.innerHTML =
            `<button style="${btnStyle}background:#28a745;color:#fff;" onclick="expAction('${id}','start')">Resume</button>` +
            `<button style="${btnStyle}background:#dc3545;color:#fff;" onclick="expAction('${id}','conclude')">Conclude</button>`;
    }

    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function expAction(id, action) {
    const res = await fetchJSON(`/experiments/${id}/${action}`, { method: 'POST' });
    if (res) {
        showToast(`Experiment ${action}${action.endsWith('e') ? 'd' : 'ed'} successfully`, 'success');
        loadExperiments();
        loadExperimentDetail(id);
    }
}

function closeExpDetail() {
    document.getElementById('experiment-detail-panel').style.display = 'none';
}

function switchExpTab(tab) {
    location.hash = tab === 'experiments' ? 'experiments' : `experiments/${tab}`;
    document.querySelectorAll('#view-experiments .sub-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('#view-experiments .sub-tab-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`#view-experiments .sub-tab[onclick*="${tab}"]`).classList.add('active');
    document.getElementById(`exp-tab-${tab}`).classList.add('active');

    if (tab === 'leaderboard') loadLeaderboard();
}

async function loadLeaderboard() {
    const data = await fetchJSON('/leaderboard');
    if (!data) return;
    const tbody = document.getElementById('leaderboard-tbody');
    tbody.innerHTML = '';

    if (!data.entries || data.entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No leaderboard data</td></tr>';
        return;
    }

    data.entries.forEach((e, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>#${i + 1}</td>
            <td>${escapeHtml(e.experiment_name || e.experiment_id)}</td>
            <td><span class="status-badge status-${escapeHtml(e.experiment_status || '')}">${escapeHtml(e.experiment_status || '-')}</span></td>
            <td>${escapeHtml(e.variant_name)}</td>
            <td>${e.success_rate != null ? (e.success_rate * 100).toFixed(1) + '%' : '-'}</td>
            <td>${e.avg_cost != null ? '$' + e.avg_cost.toFixed(4) : '-'}</td>
        `;
        tbody.appendChild(tr);
    });
}

// --- Safety ---
// --- Security ---

async function loadSecurity() {
    await Promise.all([
        loadKillSwitch(), loadBlastRadius(),
        loadAuditHooks(), loadVaultStatus(), loadSecurityEvents(), loadGuardrails()
    ]);
}

function switchSecurityTab(tab) {
    location.hash = tab === 'killswitch' ? 'security' : `security/${tab}`;
    document.querySelectorAll('#view-security .sub-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('#view-security .sub-tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`#view-security .sub-tab[onclick*="${tab}"]`).classList.add('active');
    const contentId = (tab === 'killswitch' || tab === 'blastradius')
        ? `safety-tab-${tab}` : `security-tab-${tab}`;
    document.getElementById(contentId).classList.add('active');
    if (tab === 'killswitch') loadKillSwitch();
    else if (tab === 'blastradius') loadBlastRadius();
    else if (tab === 'audithooks') loadAuditHooks();
    else if (tab === 'vault') loadVaultStatus();
    else if (tab === 'events') loadSecurityEvents();
    else if (tab === 'guardrails') loadGuardrails();
}

async function loadAuditHooks() {
    const data = await fetchJSON('/security');
    if (!data) return;
    const hooks = data.audit_hooks || {};
    document.getElementById('sec-hooks-toggle').checked = hooks.enabled || false;
    const modeEl = document.getElementById('sec-hooks-mode');
    if (modeEl) modeEl.value = hooks.mode || 'audit';
    document.getElementById('sec-event-count').textContent = hooks.event_count || 0;
    document.getElementById('sec-blocked-count').textContent = hooks.blocked_count || 0;
    renderDenyEvents(hooks.deny_events || []);
    renderAllowPaths(hooks.allow_paths || []);
}

function renderDenyEvents(events) {
    const el = document.getElementById('sec-deny-events');
    el.innerHTML = events.map(e => `<span class="tag">${e}<a href="#" onclick="removeDenyEvent('${e}');return false;">&times;</a></span>`).join('');
}

function renderAllowPaths(paths) {
    const el = document.getElementById('sec-allow-paths');
    el.innerHTML = paths.map(p => `<span class="tag">${p}<a href="#" onclick="removeAllowPath('${p}');return false;">&times;</a></span>`).join('');
}

async function toggleAuditHooks(enabled) {
    await fetchJSON('/security/audit-hooks/configure', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ enabled }) });
    await loadAuditHooks();
}

async function updateAuditHooksMode(mode) {
    await fetchJSON('/security/audit-hooks/configure', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ mode }) });
    await loadAuditHooks();
}

async function addDenyEvent() {
    const sel = document.getElementById('sec-deny-event-select');
    const evt = sel.value;
    const data = await fetchJSON('/security');
    const current = (data && data.audit_hooks && data.audit_hooks.deny_events) || [];
    if (!current.includes(evt)) current.push(evt);
    await fetchJSON('/security/audit-hooks/configure', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ deny_events: current }) });
    await loadAuditHooks();
}

async function removeDenyEvent(evt) {
    const data = await fetchJSON('/security');
    const current = ((data && data.audit_hooks && data.audit_hooks.deny_events) || []).filter(e => e !== evt);
    await fetchJSON('/security/audit-hooks/configure', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ deny_events: current }) });
    await loadAuditHooks();
}

async function addAllowPath() {
    const input = document.getElementById('sec-allow-path-input');
    const path = input.value.trim();
    if (!path) return;
    const data = await fetchJSON('/security');
    const current = (data && data.audit_hooks && data.audit_hooks.allow_paths) || [];
    if (!current.includes(path)) current.push(path);
    await fetchJSON('/security/audit-hooks/configure', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ allow_paths: current }) });
    input.value = '';
    await loadAuditHooks();
}

async function removeAllowPath(path) {
    const data = await fetchJSON('/security');
    const current = ((data && data.audit_hooks && data.audit_hooks.allow_paths) || []).filter(p => p !== path);
    await fetchJSON('/security/audit-hooks/configure', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ allow_paths: current }) });
    await loadAuditHooks();
}

async function loadVaultStatus() {
    const data = await fetchJSON('/security/vault');
    if (!data) return;
    document.getElementById('sec-vault-status').textContent = data.enabled ? 'Active' : 'Disabled';
    document.getElementById('sec-vault-count').textContent = data.key_count || 0;
    document.getElementById('sec-scrubbed-count').textContent = data.scrubbed_count || 0;
    const tbody = document.getElementById('sec-vault-tbody');
    const keys = data.keys || [];
    const scrubbed = data.scrubbed || [];
    tbody.innerHTML = keys.map(k => `<tr><td><code>${k}</code></td><td>${scrubbed.includes(k) ? '<span class="badge badge-warning">scrubbed</span>' : '<span class="badge badge-success">vaulted</span>'}</td></tr>`).join('');
}

async function addVaultSecret() {
    const nameEl = document.getElementById('sec-vault-name');
    const valueEl = document.getElementById('sec-vault-value');
    const name = nameEl.value.trim();
    const value = valueEl.value;
    if (!name || !value) return;
    await fetchJSON('/security/vault/store', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name, value }) });
    nameEl.value = '';
    valueEl.value = '';
    await loadVaultStatus();
}

async function loadSecurityEvents() {
    const data = await fetchJSON('/security/events');
    if (!data) return;
    const tbody = document.getElementById('sec-events-tbody');
    const events = data.events || [];
    tbody.innerHTML = events.reverse().map(e => {
        const ts = e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '';
        const badge = e.blocked ? '<span class="badge badge-danger">blocked</span>' : '<span class="badge badge-info">logged</span>';
        return `<tr><td>${ts}</td><td><code>${e.audit_event || ''}</code></td><td>${(e.detail || '').substring(0, 80)}</td><td>${badge}</td></tr>`;
    }).join('');
}

async function loadGuardrails() {
    const data = await fetchJSON('/security/guardrails');
    if (!data) return;
    const cfg = data.config || {};
    const stats = data.stats || {};

    document.getElementById('gr-status').textContent = cfg.enabled ? 'Enabled' : 'Disabled';
    document.getElementById('gr-status').style.color = cfg.enabled ? 'var(--accent)' : 'var(--text-secondary)';
    document.getElementById('gr-mode').textContent = cfg.enabled ? cfg.mode : '\u2014';
    document.getElementById('gr-total').textContent = stats.total || 0;
    document.getElementById('gr-blocked').textContent = stats.blocked || 0;
    document.getElementById('gr-heuristic').textContent = cfg.heuristic_enabled ? 'Enabled' : 'Disabled';
    const nliEl = document.getElementById('gr-nli');
    if (cfg.nli_enabled) {
        nliEl.textContent = cfg.nli_available ? 'Enabled' : 'Enabled (unavailable)';
        nliEl.style.color = cfg.nli_available ? 'var(--accent)' : 'var(--warning)';
    } else {
        nliEl.textContent = 'Disabled';
        nliEl.style.color = 'var(--text-secondary)';
    }
    document.getElementById('gr-nli-toggle').checked = cfg.nli_enabled || false;
    document.getElementById('gr-local-model').textContent = cfg.local_model_enabled ? (cfg.local_model || 'Enabled') : 'Disabled';
    document.getElementById('gr-output-scan').textContent = cfg.output_scanning_enabled ? 'Enabled' : 'Disabled';
    document.getElementById('gr-pattern-count').textContent = cfg.pattern_count || 0;

    // Render category breakdown
    const catEl = document.getElementById('gr-categories');
    const cats = stats.by_category || {};
    if (Object.keys(cats).length === 0) {
        catEl.innerHTML = '<span style="color:var(--text-secondary);font-size:13px;">No detections yet</span>';
    } else {
        catEl.innerHTML = Object.entries(cats).map(([cat, count]) =>
            `<div class="stat-card" style="min-width:100px;padding:12px;"><div class="stat-label">${cat}</div><div class="stat-value">${count}</div></div>`
        ).join('');
    }

    // Render severity breakdown
    const sevEl = document.getElementById('gr-severities');
    const sevs = stats.by_severity || {};
    const sevColors = { low: '#22c55e', medium: '#eab308', high: '#f97316', critical: '#ef4444' };
    if (Object.keys(sevs).length === 0) {
        sevEl.innerHTML = '<span style="color:var(--text-secondary);font-size:13px;">No detections yet</span>';
    } else {
        sevEl.innerHTML = Object.entries(sevs).map(([sev, count]) =>
            `<div class="stat-card" style="min-width:100px;padding:12px;border-left:3px solid ${sevColors[sev] || '#6b7280'};"><div class="stat-label">${sev}</div><div class="stat-value">${count}</div></div>`
        ).join('');
    }

    await loadGuardrailEvents();
}

async function loadGuardrailEvents() {
    const data = await fetchJSON('/security/guardrails/events?limit=50');
    if (!data) return;
    const tbody = document.getElementById('gr-events-tbody');
    const events = data.events || [];
    if (events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No guardrail violations recorded</td></tr>';
        return;
    }
    const sevBadge = (sev) => {
        const cls = { low: 'badge-success', medium: 'badge-warning', high: 'badge-danger', critical: 'badge-danger' };
        return `<span class="badge ${cls[sev] || 'badge-info'}">${sev}</span>`;
    };
    const actionBadge = (action) => {
        if (action === 'blocked') return '<span class="badge badge-danger">blocked</span>';
        return '<span class="badge badge-info">' + (action || 'logged') + '</span>';
    };
    tbody.innerHTML = events.reverse().map(e => {
        const ts = e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '';
        return `<tr>
            <td>${ts}</td>
            <td><code>${e.rule_name || ''}</code></td>
            <td>${e.category || ''}</td>
            <td>${sevBadge(e.severity)}</td>
            <td>${e.scan_phase || ''}</td>
            <td>${e.validator_type || ''}</td>
            <td>${actionBadge(e.action_taken)}</td>
        </tr>`;
    }).join('');
}

async function toggleNLI(enabled) {
    await fetchJSON('/security/guardrails/configure', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({nli_enabled: enabled})
    });
    await loadGuardrails();
}

async function loadKillSwitch() {
    const data = await fetchJSON('/kill-switch');
    if (!data) return;
    document.getElementById('ks-global-toggle').checked = data.active;
    document.getElementById('ks-response-mode').value = data.response_mode || 'error';
    document.getElementById('ks-rule-count').textContent = (data.rules || []).length;
    document.getElementById('ks-environment').textContent = data.environment || '\u2014';
    document.getElementById('ks-message').value = data.message || '';

    const tbody = document.getElementById('ks-rules-tbody');
    const rules = data.rules || [];
    if (rules.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No rules configured</td></tr>';
    } else {
        tbody.innerHTML = '';
        rules.forEach(r => {
            const tr = document.createElement('tr');
            tr.style.cursor = 'default';
            tr.innerHTML = `
                <td>${r.model ? escapeHtml(r.model) : '<span class="dim">any</span>'}</td>
                <td>${r.provider ? escapeHtml(r.provider) : '<span class="dim">any</span>'}</td>
                <td>${r.environment ? escapeHtml(r.environment) : '<span class="dim">any</span>'}</td>
                <td>${r.agent_version ? escapeHtml(r.agent_version) : '<span class="dim">any</span>'}</td>
                <td>${r.message ? escapeHtml(r.message) : '<span class="dim">\u2014</span>'}</td>
                <td>${r.reason ? escapeHtml(r.reason) : '<span class="dim">\u2014</span>'}</td>
            `;
            tbody.appendChild(tr);
        });
    }
}

async function toggleKillSwitch(active) {
    const endpoint = active ? '/kill-switch/activate' : '/kill-switch/deactivate';
    await fetch(`${API_BASE}${endpoint}`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
    loadKillSwitch();
}

async function updateKSResponseMode(mode) {
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ kill_switch_response_mode: mode }),
    });
}

async function updateKSMessage() {
    const msg = document.getElementById('ks-message').value;
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ kill_switch_message: msg }),
    });
    const flash = document.getElementById('flash-ks-message');
    flash.classList.add('visible');
    setTimeout(() => flash.classList.remove('visible'), 2000);
}

function showAddRuleForm() {
    document.getElementById('ks-add-rule-form').style.display = 'block';
}

function hideAddRuleForm() {
    document.getElementById('ks-add-rule-form').style.display = 'none';
    ['rule-model','rule-provider','rule-environment','rule-agent-version','rule-message','rule-reason']
        .forEach(id => document.getElementById(id).value = '');
}

async function addRule() {
    const body = {};
    const model = document.getElementById('rule-model').value.trim();
    const provider = document.getElementById('rule-provider').value.trim();
    const environment = document.getElementById('rule-environment').value.trim();
    const agentVersion = document.getElementById('rule-agent-version').value.trim();
    const message = document.getElementById('rule-message').value.trim();
    const reason = document.getElementById('rule-reason').value.trim();
    if (model) body.model = model;
    if (provider) body.provider = provider;
    if (environment) body.environment = environment;
    if (agentVersion) body.agent_version = agentVersion;
    if (message) body.message = message;
    if (reason) body.reason = reason;

    await fetch(`${API_BASE}/kill-switch/rules`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    hideAddRuleForm();
    loadKillSwitch();
}

async function clearAllRules() {
    await fetch(`${API_BASE}/kill-switch/rules`, { method: 'DELETE' });
    loadKillSwitch();
}

async function loadBlastRadius() {
    const [status, eventsData] = await Promise.all([
        fetchJSON('/blast-radius'),
        fetchJSON('/blast-radius/events?limit=50'),
    ]);

    document.getElementById('br-enabled').textContent = status.enabled ? 'Yes' : 'No';
    const pausedSessions = status.paused_sessions || [];
    const pausedAgents = status.paused_agents || [];
    document.getElementById('br-paused-sessions').textContent = pausedSessions.length;
    document.getElementById('br-paused-agents').textContent = pausedAgents.length;

    // Paused sessions list
    const sessionsList = document.getElementById('br-sessions-list');
    if (pausedSessions.length === 0) {
        sessionsList.innerHTML = '<div class="empty-state" style="padding:16px;">No paused sessions</div>';
    } else {
        sessionsList.innerHTML = '';
        pausedSessions.forEach(id => {
            const row = document.createElement('div');
            row.className = 'br-entity-row';
            row.innerHTML = `
                <span class="br-entity-id">${escapeHtml(id)}</span>
                <button class="btn btn-primary btn-sm">Unpause</button>
            `;
            row.querySelector('button').addEventListener('click', () => unpauseSession(id));
            sessionsList.appendChild(row);
        });
    }

    // Paused agents list
    const agentsList = document.getElementById('br-agents-list');
    if (pausedAgents.length === 0) {
        agentsList.innerHTML = '<div class="empty-state" style="padding:16px;">No paused agents</div>';
    } else {
        agentsList.innerHTML = '';
        pausedAgents.forEach(id => {
            const row = document.createElement('div');
            row.className = 'br-entity-row';
            row.innerHTML = `
                <span class="br-entity-id">${escapeHtml(id)}</span>
                <button class="btn btn-primary btn-sm">Unpause</button>
            `;
            row.querySelector('button').addEventListener('click', () => unpauseAgent(encodeURIComponent(id)));
            agentsList.appendChild(row);
        });
    }

    // Failure counts
    const sfCounts = status.session_failure_counts || {};
    const afCounts = status.agent_failure_counts || {};
    const sfEl = document.getElementById('br-session-failures');
    const afEl = document.getElementById('br-agent-failures');
    sfEl.textContent = Object.keys(sfCounts).length > 0
        ? Object.entries(sfCounts).map(([k,v]) => `${k}: ${v}`).join(', ')
        : '\u2014';
    afEl.textContent = Object.keys(afCounts).length > 0
        ? Object.entries(afCounts).map(([k,v]) => `${k}: ${v}`).join(', ')
        : '\u2014';

    // Events
    const eventsContainer = document.getElementById('br-events');
    const events = (eventsData && eventsData.events) || [];
    if (events.length === 0) {
        eventsContainer.innerHTML = '<div class="empty-state" style="padding:16px;">No blast radius events</div>';
    } else {
        eventsContainer.innerHTML = '';
        events.forEach(e => {
            eventsContainer.appendChild(createEventLine(e));
        });
    }
}

async function unpauseSession(id) {
    await fetch(`${API_BASE}/blast-radius/unpause-session/${encodeURIComponent(id)}`, { method: 'POST' });
    loadBlastRadius();
}

async function unpauseAgent(id) {
    await fetch(`${API_BASE}/blast-radius/unpause-agent/${id}`, { method: 'POST' });
    loadBlastRadius();
}

// --- Force Local / Auto-Route / Hot-Swap ---

async function toggleForceLocal(enabled) {
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({auto_route_force_local: enabled}),
    });
    // Force-local auto-enables auto_route and may auto-select a model — refresh UI
    const config = await fetchJSON('/config');
    if (config) {
        document.getElementById('auto-route-toggle').checked = config.auto_route_enabled || false;
        // Refresh active model dropdown in case one was auto-selected
        const activeSelect = document.getElementById('active-model-select');
        const currentActive = config.local_model_default || '';
        if (currentActive && activeSelect) {
            // Set the value — the option should already exist from loadLocalStatus
            activeSelect.value = currentActive;
            if (!activeSelect.value) {
                // Option not in dropdown yet — add it
                const opt = document.createElement('option');
                opt.value = currentActive;
                opt.textContent = currentActive;
                activeSelect.appendChild(opt);
                activeSelect.value = currentActive;
            }
        }
    }
}

async function toggleAutoRoute(enabled) {
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({auto_route_enabled: enabled}),
    });
    if (!enabled) {
        // Turning off auto-route also disables force-local
        document.getElementById('force-local-toggle').checked = false;
    }
}

async function updateActiveModel(model) {
    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({local_model_default: model}),
    });
}

async function startHotSwap() {
    const newModel = document.getElementById('hot-swap-model').value.trim();
    if (!newModel) return;
    const deleteOld = document.getElementById('hot-swap-delete-old').checked;
    const btn = document.getElementById('hot-swap-btn');
    btn.disabled = true;
    btn.textContent = 'Starting...';

    try {
        const res = await fetch(`${API_BASE}/local/hot-swap`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({new_model: newModel, delete_old: deleteOld}),
        });
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || 'Hot-swap failed');
            btn.disabled = false;
            btn.textContent = 'Start Hot-Swap';
            return;
        }
        // Show progress area and start polling
        document.getElementById('hot-swap-progress').style.display = 'block';
        pollHotSwap();
    } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Start Hot-Swap';
    }
}

function pollHotSwap() {
    const pollId = setInterval(async () => {
        try {
            const state = await fetchJSON('/local/hot-swap');
            updateHotSwapUI(state);
            if (!state.active && (state.phase === 'complete' || state.phase === 'error')) {
                clearInterval(pollId);
                setTimeout(() => {
                    document.getElementById('hot-swap-progress').style.display = 'none';
                    document.getElementById('hot-swap-btn').disabled = false;
                    document.getElementById('hot-swap-btn').textContent = 'Start Hot-Swap';
                    loadDownloadedModels();
                    loadLocalStatus();
                }, 3000);
            }
        } catch (e) {
            clearInterval(pollId);
        }
    }, 1000);
}

function updateHotSwapUI(state) {
    const progressEl = document.getElementById('hot-swap-progress');
    if (progressEl) progressEl.style.display = 'block';
    const phaseLabel = document.getElementById('hot-swap-phase-label');
    const bar = document.getElementById('hot-swap-bar');
    const pctEl = document.getElementById('hot-swap-pct');
    const phase = state.phase || '';
    const phaseName = phase === 'pulling' ? 'Pulling new model...'
        : phase === 'switching' ? 'Switching active model...'
        : phase === 'cleaning' ? 'Cleaning up old model...'
        : phase === 'complete' ? 'Hot-swap complete!'
        : phase === 'error' ? 'Error: ' + (state.error || '') : phase;
    if (phaseLabel) phaseLabel.textContent = phaseName;
    const pct = state.progress_pct || 0;
    if (bar) bar.style.width = (phase === 'complete' ? 100 : pct) + '%';
    if (pctEl) pctEl.textContent = phase === 'complete' ? '100%' : Math.round(pct) + '%';
    if (phase === 'error' && bar) bar.className = 'budget-progress budget-danger';
    else if (bar) bar.className = 'budget-progress';
}

// --- Settings ---
async function loadSettings() {
    const config = await fetchJSON('/config');
    if (!config) return;

    document.getElementById('settings-budget-session').value = config.budget_per_session != null ? config.budget_per_session : '';
    document.getElementById('settings-budget-global').value = config.budget_global != null ? config.budget_global : '';
    document.getElementById('settings-budget-action').value = config.budget_action || 'hard_stop';

    document.getElementById('settings-cache-enabled').textContent = config.cache_enabled ? 'Yes' : 'No';
    document.getElementById('settings-cache-max-size').value = config.cache_max_size || 1000;

    document.getElementById('settings-loop-enabled').checked = config.loop_detection_enabled || false;
    document.getElementById('settings-loop-threshold').value = config.loop_exact_threshold || 5;

    document.getElementById('settings-retention-days').value = config.store_retention_days || 30;

    document.getElementById('settings-console-output').textContent = config.console_output ? 'Yes' : 'No';
    document.getElementById('settings-console-verbose').checked = config.console_verbose || false;
}

async function saveSettings(section) {
    const body = {};

    if (section === 'budget') {
        const sessionVal = document.getElementById('settings-budget-session').value;
        body.budget_per_session = sessionVal === '' ? -1 : parseFloat(sessionVal);
        const globalVal = document.getElementById('settings-budget-global').value;
        body.budget_global = globalVal === '' ? -1 : parseFloat(globalVal);
        body.budget_action = document.getElementById('settings-budget-action').value;
    } else if (section === 'cache') {
        body.cache_max_size = parseInt(document.getElementById('settings-cache-max-size').value);
    } else if (section === 'loop') {
        body.loop_detection_enabled = document.getElementById('settings-loop-enabled').checked;
        body.loop_exact_threshold = parseInt(document.getElementById('settings-loop-threshold').value);
    } else if (section === 'storage') {
        body.store_retention_days = parseInt(document.getElementById('settings-retention-days').value);
    } else if (section === 'console') {
        body.console_verbose = document.getElementById('settings-console-verbose').checked;
    }

    await fetch(`${API_BASE}/config`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });

    // Flash "Saved!"
    const flash = document.getElementById(`flash-${section}`);
    flash.classList.add('visible');
    setTimeout(() => flash.classList.remove('visible'), 2000);
}

// --- Compliance ---
async function loadCompliance() {
    await Promise.all([loadPII(), loadPIIRules(), loadComplianceProfiles()]);
}

function switchComplianceTab(tab) {
    location.hash = tab === 'pii' ? 'compliance' : `compliance/${tab}`;
    document.querySelectorAll('#view-compliance .sub-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('#view-compliance .sub-tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`#view-compliance .sub-tab[onclick*="${tab}"]`).classList.add('active');
    document.getElementById(`compliance-tab-${tab}`).classList.add('active');

    if (tab === 'pii') { loadPII(); loadPIIRules(); }
    else if (tab === 'audit') loadComplianceAudit();
    else if (tab === 'orgs') loadComplianceOrgs();
}

async function loadComplianceProfiles() {
    const data = await fetchJSON('/compliance/profiles');
    const container = document.getElementById('compliance-profiles-grid');
    if (!data || !data.profiles) {
        container.innerHTML = '<div class="empty-state">No compliance profiles available</div>';
        return;
    }

    const activeGlobal = data.active_global || null;

    container.innerHTML = '';
    for (const [name, profile] of Object.entries(data.profiles)) {
        const badgeClass = 'compliance-badge compliance-' + name;
        const isActive = activeGlobal === name;
        const card = document.createElement('div');
        card.className = 'stat-card';
        card.style.border = isActive ? '2px solid var(--accent)' : '';
        card.innerHTML = `
            <div class="stat-label"><span class="${escapeHtml(badgeClass)}">${escapeHtml(name.toUpperCase())}</span>${isActive ? ' <span style="font-size:10px;color:var(--accent);">ACTIVE</span>' : ''}</div>
            <div style="margin-top:10px;font-size:12px;color:var(--text-secondary);line-height:1.8;">
                <div>Region: <strong style="color:var(--text-primary);">${escapeHtml(profile.region || 'global')}</strong></div>
                <div>Session TTL: <strong style="color:var(--text-primary);">${profile.session_ttl_days || 0} days</strong></div>
                <div>Cache TTL: <strong style="color:var(--text-primary);">${profile.cache_ttl_seconds || 0}s</strong></div>
                <div>Zero Retention: <strong style="color:var(--text-primary);">${profile.zero_retention_logs ? 'Yes' : 'No'}</strong></div>
                <div>PII Rules: <strong style="color:var(--text-primary);">${profile.pii_rules_count || 0}</strong></div>
                ${name === 'gdpr' ? `<div>Default Consent: <strong style="color:var(--text-primary);">${profile.default_consent ? 'Enabled' : 'Disabled'}</strong></div>` : ''}
            </div>
            <div style="margin-top:12px;">
                ${isActive
                    ? '<button class="btn btn-danger btn-sm" onclick="clearGlobalCompliance()">Disable</button>'
                    : `<button class="btn btn-primary btn-sm" onclick="setGlobalCompliance('${escapeHtml(name)}')">Enable</button>`}
                ${isActive && name === 'gdpr'
                    ? `<button class="btn btn-sm" style="margin-left:6px;" onclick="toggleGDPRDefaultConsent(${!profile.default_consent})">${profile.default_consent ? 'Revoke Default Consent' : 'Enable Default Consent'}</button>`
                    : ''}
            </div>
        `;
        container.appendChild(card);
    }
}

async function setGlobalCompliance(standard) {
    const result = await fetchJSON('/compliance/global', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({standard: standard}),
    });
    if (result) {
        showToast(`Enabled ${standard.toUpperCase()} compliance globally`, 'success');
        loadComplianceProfiles();
    }
}

async function clearGlobalCompliance() {
    const res = await fetch(API_BASE + '/compliance/global', {method: 'DELETE'});
    if (res.ok) {
        showToast('Global compliance profile disabled', 'success');
        loadComplianceProfiles();
    } else {
        showToast('Failed to clear compliance profile', 'error');
    }
}

async function toggleGDPRDefaultConsent(value) {
    const current = await fetchJSON('/compliance/global');
    if (!current || !current.compliance_profile) return;
    const cp = current.compliance_profile;
    const result = await fetchJSON('/compliance/global', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({standard: cp.standard, region: cp.region, default_consent: value}),
    });
    if (result) {
        showToast(value ? 'Default consent enabled' : 'Default consent revoked', 'success');
        loadComplianceProfiles();
    }
}

async function loadComplianceAudit() {
    const standard = document.getElementById('compliance-audit-standard').value;
    const orgId = document.getElementById('compliance-audit-org').value.trim();
    let query = '?limit=100';
    if (standard) query += `&standard=${encodeURIComponent(standard)}`;
    if (orgId) query += `&org_id=${encodeURIComponent(orgId)}`;

    const data = await fetchJSON(`/compliance/audit${query}`);
    const tbody = document.getElementById('compliance-audit-tbody');
    if (!data || !data.events || data.events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No audit events</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    data.events.forEach(e => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'default';
        const std = e.compliance_standard || '';
        const badgeClass = 'compliance-badge compliance-' + std;
        const hash = e.integrity_hash || '';
        const hashDisplay = hash.length > 12 ? hash.substring(0, 12) + '\u2026' : hash;
        tr.innerHTML = `
            <td>${new Date(e.timestamp).toLocaleString()}</td>
            <td><span class="${escapeHtml(badgeClass)}">${escapeHtml(std.toUpperCase())}</span></td>
            <td>${escapeHtml(e.action || '')}</td>
            <td>${escapeHtml(e.legal_rule || '\u2014')}</td>
            <td>${escapeHtml((e.target_type || '') + (e.target_id ? ':' + e.target_id : ''))}</td>
            <td>${escapeHtml(e.org_id || '\u2014')}</td>
            <td title="${escapeHtml(hash)}"><code>${escapeHtml(hashDisplay)}</code></td>
        `;
        tbody.appendChild(tr);
    });
}

async function submitPurge() {
    const userId = document.getElementById('purge-user-id').value.trim();
    if (!userId) {
        showToast('Please enter a user identifier', 'warning');
        return;
    }
    if (!confirm(`This will permanently delete all data for "${userId}". This action cannot be undone. Continue?`)) {
        return;
    }

    const standard = document.getElementById('purge-standard').value;
    const data = await fetchJSON('/compliance/purge', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_identifier: userId, standard: standard}),
    });

    if (!data) return;

    document.getElementById('purge-sessions-deleted').textContent = data.sessions_deleted || 0;
    document.getElementById('purge-events-deleted').textContent = data.events_deleted || 0;
    document.getElementById('purge-cache-deleted').textContent = data.cache_entries_deleted || 0;
    document.getElementById('purge-audit-id').textContent = data.audit_event_id || '\u2014';
    document.getElementById('purge-results').style.display = 'block';
    showToast('User data purged successfully', 'success');
}

async function loadComplianceOrgs() {
    const [data, teamsData] = await Promise.all([
        fetchJSON('/organizations'),
        fetchJSON('/teams'),
    ]);
    const container = document.getElementById('compliance-orgs-list');

    const orgs = (data && data.organizations) || [];
    const allTeams = (teamsData && teamsData.teams) || [];

    // Stats
    document.getElementById('orgs-stat-total').textContent = orgs.length;
    document.getElementById('orgs-stat-teams').textContent = allTeams.length;

    if (orgs.length === 0) {
        container.innerHTML = '<div class="empty-state">No organizations yet. Create one to get started.</div>';
        return;
    }

    // Group teams by org
    const teamsByOrg = {};
    allTeams.forEach(t => {
        if (!teamsByOrg[t.org_id]) teamsByOrg[t.org_id] = [];
        teamsByOrg[t.org_id].push(t);
    });

    container.innerHTML = '';
    for (const org of orgs) {
        const section = document.createElement('div');
        section.className = 'settings-section';
        const profileBadge = org.compliance_profile
            ? `<span class="compliance-badge compliance-${escapeHtml(org.compliance_profile.standard || '')}">${escapeHtml((org.compliance_profile.standard || '').toUpperCase())}</span>`
            : '';
        const statusBadge = `<span class="status-badge status-${escapeHtml(org.status)}">${escapeHtml(org.status)}</span>`;

        const orgTeams = teamsByOrg[org.id] || [];

        section.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <h3 style="margin:0;">${escapeHtml(org.name || org.id)}</h3>
                    ${statusBadge}
                    ${profileBadge}
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <select class="settings-select org-compliance-select" data-org-id="${escapeHtml(org.id)}" style="width:120px;">
                        <option value="">Set profile...</option>
                        <option value="gdpr">GDPR</option>
                        <option value="hipaa">HIPAA</option>
                        <option value="ccpa">CCPA</option>
                    </select>
                    <button class="btn btn-primary btn-sm org-compliance-btn" data-org-id="${escapeHtml(org.id)}">Apply</button>
                </div>
            </div>
            <div class="event-detail-grid" style="margin-bottom:12px;">
                <span class="detail-key">ID</span><span class="detail-value" style="font-size:11px;">${escapeHtml(org.id)}</span>
                <span class="detail-key">Cost</span><span class="detail-value">$${(org.total_cost || 0).toFixed(4)}</span>
                <span class="detail-key">Tokens</span><span class="detail-value">${(org.total_tokens || 0).toLocaleString()}</span>
                <span class="detail-key">Budget</span><span class="detail-value">${org.budget != null ? '$' + org.budget : '\u2014'}</span>
                <span class="detail-key">Created</span><span class="detail-value">${new Date(org.created_at).toLocaleDateString()}</span>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <h4 style="margin:0;font-size:13px;color:var(--text-secondary);">Teams (${orgTeams.length})</h4>
                <button class="btn btn-sm add-team-btn" data-org-id="${escapeHtml(org.id)}">+ Add Team</button>
            </div>
            <div class="add-team-form" data-org-id="${escapeHtml(org.id)}" style="display:none;margin-bottom:12px;padding:12px;background:var(--bg-secondary);border-radius:var(--radius-sm);border:1px solid var(--border);">
                <div style="display:flex;gap:8px;align-items:center;">
                    <input type="text" class="settings-input add-team-name" placeholder="Team name" style="width:180px;">
                    <input type="number" class="settings-input add-team-budget" placeholder="Budget (opt)" min="0" step="0.01" style="width:100px;">
                    <button class="btn btn-primary btn-sm submit-team-btn" data-org-id="${escapeHtml(org.id)}">Create</button>
                    <button class="btn btn-sm cancel-team-btn" data-org-id="${escapeHtml(org.id)}">Cancel</button>
                </div>
            </div>
            <div class="org-teams-container" data-org-id="${escapeHtml(org.id)}"></div>
        `;

        // Compliance apply
        section.querySelector('.org-compliance-btn').addEventListener('click', function() {
            const select = section.querySelector('.org-compliance-select');
            if (select.value) setOrgCompliance(org.id, select.value);
        });

        // Add team toggle
        section.querySelector('.add-team-btn').addEventListener('click', function() {
            const form = section.querySelector(`.add-team-form[data-org-id="${org.id}"]`);
            form.style.display = form.style.display === 'none' ? 'block' : 'none';
        });
        section.querySelector('.cancel-team-btn').addEventListener('click', function() {
            section.querySelector(`.add-team-form[data-org-id="${org.id}"]`).style.display = 'none';
        });
        section.querySelector('.submit-team-btn').addEventListener('click', function() {
            const form = section.querySelector(`.add-team-form[data-org-id="${org.id}"]`);
            const name = form.querySelector('.add-team-name').value.trim();
            const budgetVal = form.querySelector('.add-team-budget').value;
            if (!name) { showToast('Team name is required', 'warning'); return; }
            submitCreateTeam(org.id, name, budgetVal ? parseFloat(budgetVal) : null);
        });

        container.appendChild(section);
        renderOrgTeams(orgTeams, section.querySelector('.org-teams-container'), org.compliance_profile);
    }
}

function renderOrgTeams(teams, container, orgProfile) {
    if (teams.length === 0) {
        container.innerHTML = '<div style="font-size:12px;color:var(--text-muted);padding:8px 0;">No teams yet</div>';
        return;
    }

    let html = '<table class="data-table" style="margin-top:4px;"><thead><tr><th>Team</th><th>Status</th><th>Cost</th><th>Tokens</th><th>Budget</th><th>Rate Limit</th><th>Profile</th><th>Set Profile</th></tr></thead><tbody>';
    teams.forEach(t => {
        const ownProfile = t.compliance_profile;
        const effectiveProfile = ownProfile || orgProfile;
        const inherited = !ownProfile && orgProfile;
        const profileBadge = effectiveProfile
            ? `<span class="compliance-badge compliance-${escapeHtml(effectiveProfile.standard || '')}">${escapeHtml((effectiveProfile.standard || '').toUpperCase())}${inherited ? ' <span style="font-size:10px;opacity:0.7;">(inherited)</span>' : ''}</span>`
            : '<span class="compliance-none">None</span>';
        const tps = t.rate_limit_tps;
        const priority = t.rate_limit_priority || 0;
        const rateLimitBadge = tps != null
            ? `<span class="status-badge status-active" style="font-size:11px;">${tps} TPS</span>${priority > 0 ? ` <span style="font-size:10px;color:var(--text-muted);">P${priority}</span>` : ''}`
            : '<span style="font-size:11px;color:var(--text-muted);">\u2014</span>';
        html += `
            <tr style="cursor:default;">
                <td>${escapeHtml(t.name || t.id)}</td>
                <td><span class="status-badge status-${escapeHtml(t.status)}">${escapeHtml(t.status)}</span></td>
                <td>$${(t.total_cost || 0).toFixed(4)}</td>
                <td>${(t.total_tokens || 0).toLocaleString()}</td>
                <td>${t.budget != null ? '$' + t.budget : '\u2014'}</td>
                <td>${rateLimitBadge}</td>
                <td>${profileBadge}</td>
                <td>
                    <div style="display:flex;gap:6px;align-items:center;">
                        <select class="settings-select team-compliance-select" data-team-id="${escapeHtml(t.id)}" style="width:100px;">
                            <option value="">Select...</option>
                            <option value="gdpr">GDPR</option>
                            <option value="hipaa">HIPAA</option>
                            <option value="ccpa">CCPA</option>
                        </select>
                        <button class="btn btn-primary btn-sm team-compliance-btn" data-team-id="${escapeHtml(t.id)}">Apply</button>
                    </div>
                </td>
            </tr>
        `;
    });
    html += '</tbody></table>';
    container.innerHTML = html;

    container.querySelectorAll('.team-compliance-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const teamId = this.dataset.teamId;
            const select = container.querySelector(`.team-compliance-select[data-team-id="${teamId}"]`);
            if (select.value) setTeamCompliance(teamId, select.value);
        });
    });
}

function showCreateOrgForm() {
    document.getElementById('create-org-form').style.display = 'block';
}

function hideCreateOrgForm() {
    document.getElementById('create-org-form').style.display = 'none';
    document.getElementById('new-org-name').value = '';
    document.getElementById('new-org-budget').value = '';
}

async function submitCreateOrg() {
    const name = document.getElementById('new-org-name').value.trim();
    if (!name) { showToast('Organization name is required', 'warning'); return; }
    const budgetVal = document.getElementById('new-org-budget').value;
    const body = { name };
    if (budgetVal) body.budget = parseFloat(budgetVal);
    const res = await fetch(`${API_BASE}/organizations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (res.ok) {
        showToast('Organization created', 'success');
        hideCreateOrgForm();
        loadComplianceOrgs();
    } else {
        const b = await res.json().catch(() => ({}));
        showToast(b.detail || 'Failed to create organization', 'error');
    }
}

async function submitCreateTeam(orgId, name, budget) {
    const body = { org_id: orgId, name };
    if (budget != null) body.budget = budget;
    const res = await fetch(`${API_BASE}/teams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (res.ok) {
        showToast('Team created', 'success');
        loadComplianceOrgs();
    } else {
        const b = await res.json().catch(() => ({}));
        showToast(b.detail || 'Failed to create team', 'error');
    }
}

async function setOrgCompliance(orgId, standard) {
    const result = await fetchJSON(`/organizations/${encodeURIComponent(orgId)}/compliance`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({standard: standard}),
    });
    if (result) {
        showToast(`Set ${standard.toUpperCase()} profile for organization`, 'success');
        loadComplianceOrgs();
    }
}

async function setTeamCompliance(teamId, standard) {
    const result = await fetchJSON(`/teams/${encodeURIComponent(teamId)}/compliance`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({standard: standard}),
    });
    if (result) {
        showToast(`Set ${standard.toUpperCase()} profile for team`, 'success');
        loadComplianceOrgs();
    }
}

// --- Async Jobs ---
async function loadJobs() {
    const filter = document.getElementById('jobs-status-filter').value;
    const params = filter ? `?status=${filter}&limit=100` : '?limit=100';
    const [data, statsData] = await Promise.all([
        fetchJSON('/jobs' + params),
        fetchJSON('/jobs/stats'),
    ]);

    // Populate stats cards
    if (statsData) {
        const byStatus = statsData.by_status || {};
        document.getElementById('jobs-stat-total').textContent = statsData.total || 0;
        document.getElementById('jobs-stat-running').textContent = byStatus.running || 0;
        document.getElementById('jobs-stat-completed').textContent = byStatus.completed || 0;
        document.getElementById('jobs-stat-failed').textContent = byStatus.failed || 0;
    }

    const tbody = document.getElementById('jobs-tbody');
    if (!data || !data.jobs || data.jobs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="empty-state">No jobs yet</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    data.jobs.forEach(j => {
        const tr = document.createElement('tr');
        tr.onclick = () => loadJobDetail(j.id);
        const startedAt = j.started_at ? new Date(j.started_at).toLocaleTimeString() : '\u2014';
        const completedAt = j.completed_at ? new Date(j.completed_at).toLocaleTimeString() : '\u2014';
        const errorText = j.error ? escapeHtml(j.error.substring(0, 60)) : '\u2014';
        tr.innerHTML = `
            <td>${escapeHtml(j.id.substring(0, 12))}</td>
            <td><span class="status-badge status-${escapeHtml(j.status)}">${escapeHtml(j.status)}</span></td>
            <td>${escapeHtml(j.provider || '\u2014')}</td>
            <td>${escapeHtml(j.model || '\u2014')}</td>
            <td>${escapeHtml(j.team_id ? j.team_id.substring(0, 15) : '\u2014')}</td>
            <td>${escapeHtml(j.session_id ? j.session_id.substring(0, 15) : '\u2014')}</td>
            <td>${new Date(j.created_at).toLocaleTimeString()}</td>
            <td>${startedAt}</td>
            <td>${completedAt}</td>
            <td>${j.retry_count || 0}/${j.max_retries || 0}</td>
            <td>${errorText}</td>
        `;
        tbody.appendChild(tr);
    });
}

async function loadJobDetail(jobId) {
    const job = await fetchJSON(`/jobs/${encodeURIComponent(jobId)}`);
    if (!job) return;

    const panel = document.getElementById('job-detail-panel');
    document.getElementById('job-detail-title').textContent = 'Job: ' + job.id;
    const statusEl = document.getElementById('job-detail-status');
    statusEl.textContent = job.status;
    statusEl.className = `status-badge status-${job.status}`;

    let html = '<div class="event-detail-grid">';
    const fields = {
        'Job ID': job.id,
        'Status': job.status,
        'Provider': job.provider || '\u2014',
        'Model': job.model || '\u2014',
        'Session ID': job.session_id || '\u2014',
        'Org ID': job.org_id || '\u2014',
        'Team ID': job.team_id || '\u2014',
        'Created': job.created_at ? new Date(job.created_at).toLocaleString() : '\u2014',
        'Started': job.started_at ? new Date(job.started_at).toLocaleString() : '\u2014',
        'Completed': job.completed_at ? new Date(job.completed_at).toLocaleString() : '\u2014',
        'Retries': `${job.retry_count || 0} / ${job.max_retries || 0}`,
        'TTL': job.ttl_seconds ? job.ttl_seconds + 's' : '\u2014',
        'Webhook URL': job.webhook_url || '\u2014',
        'Webhook Status': job.webhook_status || '\u2014',
    };
    for (const [key, value] of Object.entries(fields)) {
        html += `<span class="detail-key">${escapeHtml(key)}</span>`;
        html += `<span class="detail-value">${escapeHtml(String(value))}</span>`;
    }
    html += '</div>';

    if (job.error) {
        html += '<h4 style="margin-top:16px;font-size:13px;color:var(--accent-red);">Error</h4>';
        html += `<div style="padding:10px 14px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:var(--radius-sm);font-family:var(--font-mono);font-size:12px;color:var(--accent-red);white-space:pre-wrap;margin-top:8px;">${escapeHtml(job.error)}</div>`;
        if (job.error_code) {
            html += `<div style="margin-top:6px;font-size:11px;color:var(--text-muted);">Error code: ${escapeHtml(job.error_code)}</div>`;
        }
    }

    if (job.result) {
        html += '<h4 style="margin-top:16px;font-size:13px;color:var(--accent-green);">Result</h4>';
        html += `<div style="padding:10px 14px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:var(--radius-sm);font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);white-space:pre-wrap;max-height:300px;overflow-y:auto;margin-top:8px;">${escapeHtml(JSON.stringify(job.result, null, 2))}</div>`;
    }

    if (job.metadata && Object.keys(job.metadata).length > 0) {
        html += '<h4 style="margin-top:16px;font-size:13px;color:var(--text-secondary);">Metadata</h4>';
        html += `<div style="padding:10px 14px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:var(--radius-sm);font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);white-space:pre-wrap;margin-top:8px;">${escapeHtml(JSON.stringify(job.metadata, null, 2))}</div>`;
    }

    if (job.session_id) {
        html += `<div style="margin-top:16px;"><button class="btn btn-primary btn-sm" onclick="loadSessionDetail('${escapeHtml(job.session_id)}')">View Session</button></div>`;
    }

    if (job.status === 'pending') {
        html += `<div style="margin-top:12px;"><button class="btn btn-danger btn-sm" onclick="cancelJob('${escapeHtml(job.id)}')">Cancel Job</button></div>`;
    }

    document.getElementById('job-detail-content').innerHTML = html;
    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeJobDetail() {
    document.getElementById('job-detail-panel').style.display = 'none';
}

async function cancelJob(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
    if (res.ok) {
        showToast('Job cancelled', 'success');
        closeJobDetail();
        loadJobs();
    } else {
        const body = await res.json().catch(() => ({}));
        showToast(body.detail || 'Failed to cancel job', 'error');
    }
}

// --- Agents ---
let _currentAgentId = null;
let _agentVersionsCache = [];

function switchAgentTab(tab) {
    location.hash = tab === 'list' ? 'agents' : `agents/${tab}`;

    // Toggle button active state
    document.getElementById('agents-btn-list').classList.toggle('active', tab === 'list');
    document.getElementById('agents-btn-create').classList.toggle('active', tab === 'create');

    // Toggle content panels
    document.getElementById('agents-tab-list').classList.toggle('active', tab === 'list');
    document.getElementById('agents-tab-create').classList.toggle('active', tab === 'create');

    if (tab === 'create') {
        loadTeamsForAgentForm();
        populateModelSelect('agent-form-model-select', '');
    } else if (tab === 'list') {
        loadAgents();
    }
}

async function loadAgents() {
    const [data, teamsData] = await Promise.all([
        fetchJSON('/agents'),
        fetchJSON('/teams'),
    ]);
    if (!data) return;
    const agents = data.agents || [];

    // Team name lookup
    const teamNames = {};
    if (teamsData && teamsData.teams) {
        teamsData.teams.forEach(t => { teamNames[t.id] = t.name || t.id; });
    }

    // Stats
    let active = 0, paused = 0, archived = 0;
    agents.forEach(a => {
        if (a.status === 'active') active++;
        else if (a.status === 'paused') paused++;
        else if (a.status === 'archived') archived++;
    });
    document.getElementById('agents-stat-total').textContent = agents.length;
    document.getElementById('agents-stat-active').textContent = active;
    document.getElementById('agents-stat-paused').textContent = paused;
    document.getElementById('agents-stat-archived').textContent = archived;

    // Table
    const tbody = document.getElementById('agents-tbody');
    tbody.innerHTML = '';
    if (agents.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No agents yet. Create one to get started.</td></tr>';
        return;
    }

    agents.forEach(a => {
        const tr = document.createElement('tr');
        tr.onclick = () => loadAgentDetail(a.id);
        tr.style.cursor = 'pointer';
        const teamLabel = a.team_id ? escapeHtml(teamNames[a.team_id] || a.team_id) : '<span class="text-muted">\u2014</span>';
        tr.innerHTML = `
            <td>${escapeHtml(a.slug)}</td>
            <td>${escapeHtml(a.name || '\u2014')}</td>
            <td><span class="status-badge status-${escapeHtml(a.status)}">${escapeHtml(a.status)}</span></td>
            <td>\u2014</td>
            <td>${teamLabel}</td>
            <td>\u2014</td>
            <td>${new Date(a.created_at).toLocaleDateString()}</td>
        `;
        tbody.appendChild(tr);
    });

    // Fetch active version info for model/version columns
    agents.forEach(async (a, idx) => {
        if (!a.active_version_id) return;
        const detail = await fetchJSON(`/agents/${encodeURIComponent(a.id)}`);
        if (!detail || !detail.active_version) return;
        const row = tbody.children[idx];
        if (!row) return;
        row.children[3].textContent = detail.active_version.model || '\u2014';
        row.children[5].textContent = 'v' + detail.active_version.version_number;
    });
}

async function loadAgentDetail(agentId) {
    _currentAgentId = agentId;
    // Navigate to agent detail URL
    location.hash = 'agents/detail/' + encodeURIComponent(agentId);

    // Show agent-detail view, hide others
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-agent-detail').classList.add('active');
    // Highlight Agents nav link
    document.querySelectorAll('[data-view]').forEach(l => l.classList.remove('active'));
    const agentsLink = document.querySelector('[data-view="agents"]');
    if (agentsLink) agentsLink.classList.add('active');

    const [agent, versionsData] = await Promise.all([
        fetchJSON(`/agents/${encodeURIComponent(agentId)}`),
        fetchJSON(`/agents/${encodeURIComponent(agentId)}/versions`),
    ]);
    if (!agent) return;

    // Resolve team and org names
    let teamName = agent.team_id || '\u2014';
    let orgName = agent.org_id || '\u2014';
    const [teamData, orgData] = await Promise.all([
        agent.team_id ? fetchJSONSilent(`/teams/${encodeURIComponent(agent.team_id)}`) : null,
        agent.org_id ? fetchJSONSilent(`/organizations/${encodeURIComponent(agent.org_id)}`) : null,
    ]);
    if (teamData && teamData.name) teamName = teamData.name;
    if (orgData && orgData.name) orgName = orgData.name;

    // Header
    document.getElementById('agent-detail-name').textContent = agent.name || agent.slug;
    const statusBadge = document.getElementById('agent-detail-status');
    statusBadge.textContent = agent.status;
    statusBadge.className = `status-badge status-${agent.status}`;

    // Info grid
    const infoGrid = document.getElementById('agent-detail-info');
    infoGrid.innerHTML = `
        <span class="detail-key">ID</span><span class="detail-value">${escapeHtml(agent.id)}</span>
        <span class="detail-key">Slug</span><span class="detail-value">${escapeHtml(agent.slug)}</span>
        <span class="detail-key">Team</span><span class="detail-value">${escapeHtml(teamName)}</span>
        <span class="detail-key">Org</span><span class="detail-value">${escapeHtml(orgName)}</span>
        <span class="detail-key">Description</span><span class="detail-value">${escapeHtml(agent.description || '\u2014')}</span>
        <span class="detail-key">Created</span><span class="detail-value">${new Date(agent.created_at).toLocaleString()}</span>
        <span class="detail-key">Updated</span><span class="detail-value">${new Date(agent.updated_at).toLocaleString()}</span>
    `;

    // Proxy URL
    const port = location.port || (location.protocol === 'https:' ? '443' : '80');
    const proxyUrl = `POST ${location.protocol}//${location.hostname}:${port}/v1/agents/${agent.slug}/chat/completions`;
    document.getElementById('agent-proxy-url').textContent = proxyUrl;

    // Active version
    const versionInfo = document.getElementById('agent-active-version-info');
    const promptSection = document.getElementById('agent-active-version-prompt');
    if (agent.active_version) {
        const v = agent.active_version;
        versionInfo.innerHTML = `
            <span class="detail-key">Version</span><span class="detail-value">v${v.version_number}</span>
            <span class="detail-key">Model</span><span class="detail-value">${escapeHtml(v.model)}</span>
            <span class="detail-key">Budget</span><span class="detail-value">${v.budget_per_session != null ? '$' + v.budget_per_session : '\u2014'}</span>
            <span class="detail-key">Created By</span><span class="detail-value">${escapeHtml(v.created_by || '\u2014')}</span>
            <span class="detail-key">Request Overrides</span><span class="detail-value">${v.request_overrides && Object.keys(v.request_overrides).length > 0 ? escapeHtml(JSON.stringify(v.request_overrides)) : '\u2014'}</span>
        `;
        if (v.system_prompt) {
            document.getElementById('agent-version-prompt-text').textContent = v.system_prompt;
            promptSection.style.display = 'block';
        } else {
            promptSection.style.display = 'none';
        }
    } else {
        versionInfo.innerHTML = '<span class="detail-value">No active version</span>';
        promptSection.style.display = 'none';
    }

    // Version history
    const versions = (versionsData && versionsData.versions) || [];
    _agentVersionsCache = versions;
    const vTbody = document.getElementById('agent-versions-tbody');
    vTbody.innerHTML = '';
    if (versions.length === 0) {
        vTbody.innerHTML = '<tr><td colspan="5" class="empty-state">No versions</td></tr>';
    } else {
        versions.forEach(v => {
            const tr = document.createElement('tr');
            const isActive = agent.active_version_id === v.id;
            tr.innerHTML = `
                <td>v${v.version_number}${isActive ? ' <span class="status-badge status-active">active</span>' : ''}</td>
                <td>${escapeHtml(v.model)}</td>
                <td>${new Date(v.created_at).toLocaleString()}</td>
                <td>${escapeHtml(v.created_by || '\u2014')}</td>
                <td>${isActive ? '\u2014' : `<button class="btn btn-primary btn-sm" onclick="activateVersion('${escapeHtml(agentId)}','${escapeHtml(v.id)}')">Activate</button>`}</td>
            `;
            vTbody.appendChild(tr);
        });
    }

    // Pause/Resume button
    const pauseBtn = document.getElementById('agent-pause-btn');
    if (agent.status === 'active') {
        pauseBtn.textContent = 'Pause';
        pauseBtn.onclick = () => toggleAgentStatus(agentId, 'paused');
    } else if (agent.status === 'paused') {
        pauseBtn.textContent = 'Resume';
        pauseBtn.onclick = () => toggleAgentStatus(agentId, 'active');
    } else {
        pauseBtn.style.display = 'none';
    }

    // Hide archive button for already-archived agents
    const archiveBtn = document.getElementById('agent-archive-btn');
    archiveBtn.style.display = agent.status === 'archived' ? 'none' : '';

    // Hide new version form
    document.getElementById('agent-new-version-form').style.display = 'none';

    // Hide one-time key banner
    document.getElementById('agent-vk-created').style.display = 'none';

    // Load virtual keys for this agent's team
    if (agent.team_id) loadAgentVirtualKeys(agentId, agent.team_id);
}

function closeAgentDetail() {
    _currentAgentId = null;
    switchView('agents');
}

async function activateVersion(agentId, versionId) {
    const res = await fetch(`${API_BASE}/agents/${encodeURIComponent(agentId)}/versions/${encodeURIComponent(versionId)}/activate`, { method: 'PUT' });
    if (res.ok) {
        showToast('Version activated', 'success');
        loadAgentDetail(agentId);
        loadAgents();
    } else {
        const body = await res.json().catch(() => ({}));
        showToast(body.detail || 'Failed to activate version', 'error');
    }
}

async function toggleAgentStatus(agentId, newStatus) {
    const res = await fetch(`${API_BASE}/agents/${encodeURIComponent(agentId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus }),
    });
    if (res.ok) {
        showToast(`Agent ${newStatus}`, 'success');
        loadAgentDetail(agentId);
        loadAgents();
    } else {
        const body = await res.json().catch(() => ({}));
        showToast(body.detail || 'Failed to update agent', 'error');
    }
}

function toggleAgentPause() {
    // Handled via onclick set in loadAgentDetail
}

async function archiveCurrentAgent() {
    if (!_currentAgentId) return;
    if (!confirm('Archive this agent? It will no longer be accessible via the proxy.')) return;
    const res = await fetch(`${API_BASE}/agents/${encodeURIComponent(_currentAgentId)}`, { method: 'DELETE' });
    if (res.ok) {
        showToast('Agent archived', 'success');
        closeAgentDetail();
        loadAgents();
    } else {
        const body = await res.json().catch(() => ({}));
        showToast(body.detail || 'Failed to archive agent', 'error');
    }
}

// --- Agent Virtual Keys ---
async function loadAgentVirtualKeys(agentId, teamId) {
    const data = await fetchJSON(`/virtual-keys?team_id=${encodeURIComponent(teamId)}`);
    const tbody = document.getElementById('agent-vk-tbody');
    tbody.innerHTML = '';
    const keys = ((data && data.virtual_keys) || []).filter(vk =>
        !vk.revoked && (!vk.agent_ids || vk.agent_ids.length === 0 || vk.agent_ids.includes(agentId))
    );
    if (keys.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No virtual keys for this team</td></tr>';
        return;
    }
    keys.forEach(vk => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><code style="font-size:11px;">${escapeHtml(vk.key_preview)}</code></td>
            <td>${escapeHtml(vk.name || '\u2014')}</td>
            <td>${new Date(vk.created_at).toLocaleDateString()}</td>
            <td><button class="btn btn-danger btn-sm" style="font-size:10px;padding:2px 8px;" onclick="revokeVirtualKey('${escapeHtml(vk.id)}')">Revoke</button></td>
        `;
        tbody.appendChild(tr);
    });
}

async function createAgentVirtualKey() {
    if (!_currentAgentId) return;
    // Get the agent's team_id
    const agent = await fetchJSON(`/agents/${encodeURIComponent(_currentAgentId)}`);
    if (!agent) return;

    const res = await fetch(`${API_BASE}/virtual-keys`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            team_id: agent.team_id,
            name: `key-for-${agent.slug}`,
            agent_ids: [agent.id],
        }),
    });
    if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        showToast(b.detail || 'Failed to create virtual key', 'error');
        return;
    }
    const result = await res.json();
    // Show the full key (one-time display)
    document.getElementById('agent-vk-full-key').textContent = result.key;
    document.getElementById('agent-vk-created').style.display = 'block';

    showToast('Virtual key created — copy it now!', 'success');
    loadAgentVirtualKeys(_currentAgentId, agent.team_id);
}

function copyVirtualKey() {
    const key = document.getElementById('agent-vk-full-key').textContent;
    navigator.clipboard.writeText(key).then(() => {
        showToast('Key copied to clipboard', 'success');
    });
}

async function revokeVirtualKey(keyId) {
    if (!confirm('Revoke this virtual key? This cannot be undone.')) return;
    const res = await fetch(`${API_BASE}/virtual-keys/${encodeURIComponent(keyId)}`, { method: 'DELETE' });
    if (res.ok) {
        showToast('Virtual key revoked', 'success');
        // Reload the list
        const agent = await fetchJSON(`/agents/${encodeURIComponent(_currentAgentId)}`);
        if (agent) loadAgentVirtualKeys(_currentAgentId, agent.team_id);
    } else {
        const b = await res.json().catch(() => ({}));
        showToast(b.detail || 'Failed to revoke key', 'error');
    }
}

function showNewVersionForm() {
    document.getElementById('agent-new-version-form').style.display = 'block';
    // Pre-fill model from current version
    const currentVersion = _agentVersionsCache.length > 0 ? _agentVersionsCache[0] : null;
    populateModelSelect('new-version-model-select', currentVersion ? currentVersion.model : '');
    document.getElementById('new-version-model-custom').style.display = 'none';
    document.getElementById('new-version-model-custom').value = '';
}

function hideNewVersionForm() {
    document.getElementById('agent-new-version-form').style.display = 'none';
}

async function submitNewVersion() {
    if (!_currentAgentId) return;
    const model = getSelectedModel('new-version');
    if (!model) { showToast('Model is required', 'warning'); return; }

    const overridesText = document.getElementById('new-version-request-overrides').value.trim();
    let requestOverrides = {};
    if (overridesText) {
        try { requestOverrides = JSON.parse(overridesText); }
        catch { showToast('Request overrides must be valid JSON', 'warning'); return; }
    }

    const budgetVal = document.getElementById('new-version-budget').value;

    const body = {
        model,
        system_prompt: document.getElementById('new-version-system-prompt').value,
        request_overrides: requestOverrides,
        budget_per_session: budgetVal ? parseFloat(budgetVal) : null,
        created_by: document.getElementById('new-version-created-by').value.trim(),
    };

    const res = await fetch(`${API_BASE}/agents/${encodeURIComponent(_currentAgentId)}/versions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (res.ok) {
        showToast('Version created', 'success');
        hideNewVersionForm();
        loadAgentDetail(_currentAgentId);
        loadAgents();
    } else {
        const respBody = await res.json().catch(() => ({}));
        showToast(respBody.detail || 'Failed to create version', 'error');
    }
}

let _cachedCloudModels = null;
let _cachedProviderKeys = null;

async function populateModelSelect(selectId, currentValue) {
    const select = document.getElementById(selectId);
    if (!select) return;

    // Fetch models and provider keys (cached)
    if (!_cachedCloudModels) {
        _cachedCloudModels = await fetchJSON('/models/cloud');
    }
    if (!_cachedProviderKeys) {
        _cachedProviderKeys = await fetchJSON('/provider-keys');
    }

    // Determine which providers have keys configured
    const configuredProviders = new Set();
    if (_cachedProviderKeys) {
        for (const [provider, info] of Object.entries(_cachedProviderKeys)) {
            if (info && info.set) configuredProviders.add(provider);
        }
    }

    select.innerHTML = '<option value="">Select model...</option>';

    if (_cachedCloudModels && _cachedCloudModels.models) {
        const byProvider = {};
        _cachedCloudModels.models.forEach(m => {
            if (!byProvider[m.provider]) byProvider[m.provider] = [];
            byProvider[m.provider].push(m.model);
        });

        // Show configured providers first, then unconfigured (disabled)
        const providers = Object.keys(byProvider);
        const sorted = providers.sort((a, b) => {
            const aSet = configuredProviders.has(a) ? 0 : 1;
            const bSet = configuredProviders.has(b) ? 0 : 1;
            return aSet - bSet || a.localeCompare(b);
        });

        for (const provider of sorted) {
            const modelList = byProvider[provider];
            const isConfigured = configuredProviders.has(provider);
            const optgroup = document.createElement('optgroup');
            const label = provider.charAt(0).toUpperCase() + provider.slice(1);
            optgroup.label = isConfigured ? label : `${label} (no API key)`;
            modelList.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                if (!isConfigured) opt.disabled = true;
                optgroup.appendChild(opt);
            });
            select.appendChild(optgroup);
        }
    }

    // Custom option at the end
    const customOpt = document.createElement('option');
    customOpt.value = '__custom__';
    customOpt.textContent = 'Custom...';
    select.appendChild(customOpt);

    if (currentValue) select.value = currentValue;

    // Show hint if no providers configured
    if (configuredProviders.size === 0) {
        const hint = select.parentElement.querySelector('.model-key-hint');
        if (!hint) {
            const span = document.createElement('span');
            span.className = 'model-key-hint';
            span.style.cssText = 'font-size:11px;color:var(--text-secondary);display:block;margin-top:4px;';
            span.innerHTML = 'No API keys configured. <a href="#models" style="color:var(--accent);">Set up keys in Models tab</a>';
            select.parentElement.appendChild(span);
        }
    } else {
        const hint = select.parentElement.querySelector('.model-key-hint');
        if (hint) hint.remove();
    }
}

// Invalidate cache when keys are saved so the model dropdown refreshes
function invalidateProviderKeysCache() {
    _cachedProviderKeys = null;
}

function onAgentModelSelect(prefix) {
    const select = document.getElementById(prefix + '-model-select');
    const custom = document.getElementById(prefix + '-model-custom');
    if (select.value === '__custom__') {
        custom.style.display = '';
        custom.focus();
    } else {
        custom.style.display = 'none';
        custom.value = '';
    }
}

function getSelectedModel(prefix) {
    const select = document.getElementById(prefix + '-model-select');
    if (select.value === '__custom__') {
        return document.getElementById(prefix + '-model-custom').value.trim();
    }
    return select.value;
}

async function loadTeamsForAgentForm() {
    const data = await fetchJSON('/teams');
    const select = document.getElementById('agent-form-team');
    select.innerHTML = '<option value="">Select team...</option>';
    if (data && data.teams) {
        data.teams.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.id;
            opt.textContent = t.name || t.id;
            select.appendChild(opt);
        });
    }
    // Always add "+ New Team" option at the end
    const newOpt = document.createElement('option');
    newOpt.value = '__new__';
    newOpt.textContent = '+ New Team';
    select.appendChild(newOpt);
}

function onAgentTeamSelect(value) {
    if (value === '__new__') {
        document.getElementById('agent-create-team-form').style.display = 'block';
        loadOrgsForTeamForm();
        // Reset the select to placeholder so it doesn't stay on "+ New Team"
        document.getElementById('agent-form-team').value = '';
    } else {
        document.getElementById('agent-create-team-form').style.display = 'none';
    }
}

async function loadOrgsForTeamForm() {
    const data = await fetchJSON('/organizations');
    const select = document.getElementById('agent-new-team-org');
    select.innerHTML = '<option value="">Select org...</option>';
    if (data && data.organizations) {
        data.organizations.forEach(o => {
            const opt = document.createElement('option');
            opt.value = o.id;
            opt.textContent = o.name || o.id;
            select.appendChild(opt);
        });
    }
    const newOpt = document.createElement('option');
    newOpt.value = '__new__';
    newOpt.textContent = '+ New Organization';
    select.appendChild(newOpt);
}

function onNewTeamOrgSelect(value) {
    document.getElementById('agent-new-org-row').style.display = value === '__new__' ? '' : 'none';
}

function hideCreateTeamForm() {
    document.getElementById('agent-create-team-form').style.display = 'none';
}

async function createTeamForAgent() {
    let orgId = document.getElementById('agent-new-team-org').value;
    const teamName = document.getElementById('agent-new-team-name').value.trim();

    if (!teamName) { showToast('Team name is required', 'warning'); return; }

    // Create org first if needed
    if (orgId === '__new__') {
        const orgName = document.getElementById('agent-new-org-name').value.trim();
        if (!orgName) { showToast('Organization name is required', 'warning'); return; }
        const orgRes = await fetch(`${API_BASE}/organizations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: orgName }),
        });
        if (!orgRes.ok) {
            const b = await orgRes.json().catch(() => ({}));
            showToast(b.detail || 'Failed to create organization', 'error');
            return;
        }
        const org = await orgRes.json();
        orgId = org.id;
    }

    if (!orgId) { showToast('Organization is required', 'warning'); return; }

    // Create team
    const teamRes = await fetch(`${API_BASE}/teams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ org_id: orgId, name: teamName }),
    });
    if (!teamRes.ok) {
        const b = await teamRes.json().catch(() => ({}));
        showToast(b.detail || 'Failed to create team', 'error');
        return;
    }
    const team = await teamRes.json();

    showToast('Team created', 'success');
    hideCreateTeamForm();

    // Reload teams dropdown and select the new team
    await loadTeamsForAgentForm();
    document.getElementById('agent-form-team').value = team.id;
}

async function createAgent() {
    const slug = document.getElementById('agent-form-slug').value.trim();
    const teamId = document.getElementById('agent-form-team').value;
    const model = getSelectedModel('agent-form');

    if (!slug) { showToast('Slug is required', 'warning'); return; }
    if (!teamId) { showToast('Team is required', 'warning'); return; }
    if (!model) { showToast('Model is required', 'warning'); return; }

    const overridesText = document.getElementById('agent-form-request-overrides').value.trim();
    let requestOverrides = {};
    if (overridesText) {
        try { requestOverrides = JSON.parse(overridesText); }
        catch { showToast('Request overrides must be valid JSON', 'warning'); return; }
    }

    const budgetVal = document.getElementById('agent-form-budget').value;

    const body = {
        slug,
        team_id: teamId,
        name: document.getElementById('agent-form-name').value.trim(),
        description: document.getElementById('agent-form-description').value.trim(),
        model,
        system_prompt: document.getElementById('agent-form-system-prompt').value,
        request_overrides: requestOverrides,
        budget_per_session: budgetVal ? parseFloat(budgetVal) : null,
    };

    const res = await fetch(`${API_BASE}/agents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (res.ok) {
        showToast('Agent created', 'success');
        // Clear form
        document.getElementById('agent-form-slug').value = '';
        document.getElementById('agent-form-name').value = '';
        document.getElementById('agent-form-description').value = '';
        document.getElementById('agent-form-model-select').value = '';
        document.getElementById('agent-form-model-custom').value = '';
        document.getElementById('agent-form-model-custom').style.display = 'none';
        document.getElementById('agent-form-system-prompt').value = '';
        document.getElementById('agent-form-request-overrides').value = '';
        document.getElementById('agent-form-budget').value = '';
        // Switch to list
        switchAgentTab('list');
        loadAgents();
    } else {
        const respBody = await res.json().catch(() => ({}));
        showToast(respBody.detail || 'Failed to create agent', 'error');
    }
}

// --- Server Logs (debug mode) ---
let _logsWs = null;
let _logsWsReconnectTimer = null;

function _formatLogTimestamp(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function _createLogEntryEl(entry) {
    const el = document.createElement('div');
    el.className = 'log-entry';
    el.dataset.level = entry.level;

    const ts = document.createElement('span');
    ts.className = 'log-timestamp';
    ts.textContent = _formatLogTimestamp(entry.timestamp);

    const lvl = document.createElement('span');
    lvl.className = 'log-level log-level-' + entry.level;
    lvl.textContent = entry.level;

    const logger = document.createElement('span');
    logger.className = 'log-logger';
    logger.textContent = entry.logger;
    logger.title = entry.logger;

    const msg = document.createElement('span');
    msg.className = 'log-message';
    msg.textContent = entry.message;

    el.appendChild(ts);
    el.appendChild(lvl);
    el.appendChild(logger);
    el.appendChild(msg);
    return el;
}

function _applyLogLevelFilter() {
    const filter = document.getElementById('logs-level-filter').value;
    const container = document.getElementById('server-logs-container');
    container.querySelectorAll('.log-entry').forEach(el => {
        el.style.display = (!filter || el.dataset.level === filter) ? '' : 'none';
    });
}

async function loadServerLogs() {
    const data = await fetchJSON('/debug');
    if (!data || !data.debug) {
        document.getElementById('server-logs-container').innerHTML =
            '<div class="empty-state">Debug mode is not enabled. Start the server with --debug to see logs.</div>';
        return;
    }

    const level = document.getElementById('logs-level-filter').value;
    const params = new URLSearchParams({ limit: '500' });
    if (level) params.set('level', level);

    const logsData = await fetchJSON('/logs?' + params.toString());
    if (!logsData) return;

    const container = document.getElementById('server-logs-container');
    container.innerHTML = '';

    // Add WS status bar
    const statusBar = document.createElement('div');
    statusBar.className = 'server-logs-ws-status';
    statusBar.id = 'logs-ws-status';
    statusBar.innerHTML = '<span class="ws-dot ws-dot-disconnected" id="logs-ws-dot"></span> Connecting...';
    container.appendChild(statusBar);

    const logs = logsData.logs || [];
    logs.forEach(entry => {
        container.appendChild(_createLogEntryEl(entry));
    });

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;

    // Start live streaming
    _connectLogsWebSocket();
}

function _connectLogsWebSocket() {
    if (_logsWs) {
        try { _logsWs.close(); } catch {}
        _logsWs = null;
    }
    if (_logsWsReconnectTimer) {
        clearTimeout(_logsWsReconnectTimer);
        _logsWsReconnectTimer = null;
    }

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    _logsWs = new WebSocket(`${protocol}//${location.host}/ws/logs`);

    _logsWs.onopen = () => {
        const dot = document.getElementById('logs-ws-dot');
        const status = document.getElementById('logs-ws-status');
        if (dot) { dot.className = 'ws-dot ws-dot-connected'; }
        if (status) { status.lastChild.textContent = ' Live streaming'; }
    };

    _logsWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'log') {
            const container = document.getElementById('server-logs-container');
            if (!container) return;
            const filter = document.getElementById('logs-level-filter').value;
            const el = _createLogEntryEl(msg.data);
            if (filter && msg.data.level !== filter) el.style.display = 'none';
            container.appendChild(el);

            // Cap DOM entries at 2000
            const entries = container.querySelectorAll('.log-entry');
            if (entries.length > 2000) entries[0].remove();

            // Auto-scroll
            const autoScroll = document.getElementById('logs-auto-scroll');
            if (autoScroll && autoScroll.checked) {
                container.scrollTop = container.scrollHeight;
            }
        }
    };

    _logsWs.onclose = () => {
        const dot = document.getElementById('logs-ws-dot');
        const status = document.getElementById('logs-ws-status');
        if (dot) { dot.className = 'ws-dot ws-dot-disconnected'; }
        if (status) { status.lastChild.textContent = ' Disconnected — reconnecting...'; }
        // Only reconnect if we're still on the server-logs view
        if (currentView === 'server-logs') {
            _logsWsReconnectTimer = setTimeout(_connectLogsWebSocket, 3000);
        }
    };

    _logsWs.onerror = () => {};
}

function _disconnectLogsWebSocket() {
    if (_logsWsReconnectTimer) {
        clearTimeout(_logsWsReconnectTimer);
        _logsWsReconnectTimer = null;
    }
    if (_logsWs) {
        try { _logsWs.close(); } catch {}
        _logsWs = null;
    }
}

async function clearServerLogs() {
    const res = await fetch(`${API_BASE}/logs`, { method: 'DELETE' });
    if (res.ok) {
        const container = document.getElementById('server-logs-container');
        // Keep the status bar, remove log entries
        container.querySelectorAll('.log-entry').forEach(el => el.remove());
        showToast('Logs cleared', 'success');
    }
}

// Disconnect logs WS when leaving the view
const _origSwitchView = switchView;
switchView = function(view) {
    if (currentView === 'server-logs' && view !== 'server-logs') {
        _disconnectLogsWebSocket();
    }
    _origSwitchView(view);
};

// Level filter change handler
document.addEventListener('DOMContentLoaded', () => {
    const filter = document.getElementById('logs-level-filter');
    if (filter) filter.addEventListener('change', () => {
        if (currentView === 'server-logs') _applyLogLevelFilter();
    });
});

// Check debug status on load and show/hide nav link
(async function checkDebugMode() {
    try {
        const res = await fetch(`${API_BASE}/debug`);
        if (res.ok) {
            const data = await res.json();
            if (data.debug) {
                const navItem = document.getElementById('nav-server-logs');
                if (navItem) navItem.style.display = '';
                // Re-register click handler for the dynamically shown link
                const link = navItem.querySelector('[data-view]');
                if (link) {
                    link.addEventListener('click', (e) => {
                        e.preventDefault();
                        switchView(link.dataset.view);
                    });
                }
            }
        }
    } catch {}
})();

// --- Init ---
connectWebSocket();

// Restore tab from URL hash, default to overview
function handleHash() {
    const raw = location.hash ? location.hash.slice(1) : 'overview';
    const [hash, query] = raw.split('?');
    const params = new URLSearchParams(query || '');

    if (hash === 'pii') { switchView('compliance'); return; }
    if (hash === 'costs' || hash === 'local') { switchView('models'); return; }
    if (hash === 'model-testing') { switchView('models'); setTimeout(() => switchModelsTab('testing'), 50); return; }
    if (hash === 'safety') { switchView('security'); return; }
    if (hash.startsWith('session-detail/')) {
        const sessionId = decodeURIComponent(hash.slice('session-detail/'.length));
        if (sessionId && sessionId !== currentDetailSessionId) {
            loadSessionDetail(sessionId);
        }
    } else if (hash.startsWith('agents/detail/')) {
        const agentId = decodeURIComponent(hash.slice('agents/detail/'.length));
        if (agentId && agentId !== _currentAgentId) {
            loadAgentDetail(agentId);
        }
    } else {
        // Parse view/subtab format (e.g. "compliance/orgs", "security/guardrails")
        const slashIdx = hash.indexOf('/');
        const view = slashIdx > 0 ? hash.slice(0, slashIdx) : hash;
        const subtab = slashIdx > 0 ? hash.slice(slashIdx + 1) : null;

        // Restore observability window from URL before switching view
        if (view === 'observability' && params.has('window')) {
            const sel = document.getElementById('obs-window');
            if (sel) sel.value = params.get('window');
        }
        if (view !== currentView) {
            switchView(view);
        }
        // Restore sub-tab after view is loaded
        if (subtab) {
            if (view === 'compliance') switchComplianceTab(subtab);
            else if (view === 'safety') {
                switchView('security');
                switchSecurityTab(subtab);
                return;
            }
            else if (view === 'security') switchSecurityTab(subtab);
            else if (view === 'models') switchModelsTab(subtab);
            else if (view === 'experiments') switchExpTab(subtab);
            else if (view === 'agents') switchAgentTab(subtab);
        }
    }
}
handleHash();

// Handle browser back/forward
window.addEventListener('hashchange', handleHash);

// Poll for updates every 5s
setInterval(() => {
    if (currentView === 'overview') loadOverview();
    else if (currentView === 'jobs') loadJobs();
    else if (currentView === 'session-detail' && currentDetailSessionId) {
        refreshSessionDetailEvents(currentDetailSessionId);
    }
}, 15000);

async function refreshSessionDetailEvents(sessionId) {
    try {
        // On periodic refresh, re-fetch from offset 0 up to current loaded count (or page size, whichever is larger)
        const fetchLimit = Math.max(_detailEventsOffset, _detailEventsPageSize);
        const eventsData = await fetchJSON(`/sessions/${sessionId}/events?limit=${fetchLimit}&offset=0&exclude_types=compliance_audit&primary_only=true`);
        if (!eventsData) return;
        const container = document.getElementById('detail-events');
        if (!container) return;
        const events = eventsData.events || [];
        const currentCount = container.children.length;
        if (eventsData.total !== currentCount) {
            _toolStepCache = {};  // clear tool sub-step cache on refresh
            _detailEventsLoaded = events;
            _detailEventsOffset = events.length;
            _detailEventsHasMore = !!eventsData.has_more;
            _updateEventsLoadMore();
            container.innerHTML = '';
            events.forEach(e => {
                container.appendChild(createEventLine(e));
            });
            renderStepTimeline(events);
            // Re-render waterfall view too
            renderWaterfallTimeline(events, null);
            // Recompute header stats from events for consistency
            let evPrompt = 0, evCompletion = 0, evCost = 0, evCalls = 0;
            for (const e of events) {
                if (e.event_type !== 'llm_call') continue;
                if (e.is_cli_internal) continue;
                evPrompt += e.prompt_tokens || 0;
                evCompletion += e.completion_tokens || 0;
                evCost += e.cost || 0;
                evCalls++;
                // Include tool-continuation aggregates from server-provided _tool_summary
                if (e._tool_summary) {
                    evCost += e._tool_summary.total_cost || 0;
                    evCompletion += e._tool_summary.total_tokens || 0;
                }
            }
            const tokensEl = document.getElementById('detail-tokens');
            if (tokensEl) tokensEl.textContent = (evPrompt + evCompletion).toLocaleString();
            const promptEl = document.getElementById('detail-prompt-tokens');
            if (promptEl) promptEl.textContent = evPrompt.toLocaleString();
            const compEl = document.getElementById('detail-completion-tokens');
            if (compEl) compEl.textContent = evCompletion.toLocaleString();
            const callsEl = document.getElementById('detail-calls');
            if (callsEl) callsEl.textContent = evCalls;
            const costEl = document.getElementById('detail-cost');
            if (costEl) costEl.textContent = _sessionBillingMode === 'subscription' ? '\u2014' : `$${evCost.toFixed(4)}`;
        }
    } catch (e) { /* ignore refresh errors */ }
}

// --- Export Session ---
async function exportSession() {
    const sessionId = currentDetailSessionId;
    if (!sessionId) {
        showToast('No session selected', 'warning');
        return;
    }
    try {
        const res = await fetch(
            `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/export?include_children=true`
        );
        if (!res.ok) {
            const body = await res.text();
            let detail = '';
            try { detail = JSON.parse(body).detail || body; } catch { detail = body; }
            showToast(`Export failed (${res.status}): ${detail}`, 'error');
            return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `session-${sessionId}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        showToast('Session exported successfully', 'success');
    } catch (err) {
        showToast(`Export error: ${err.message}`, 'error');
    }
}

// --- Close Session ---
async function closeSession() {
    const sessionId = currentDetailSessionId;
    if (!sessionId) {
        showToast('No session selected', 'warning');
        return;
    }
    try {
        const res = await fetch(
            `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/end`,
            { method: 'POST' }
        );
        if (!res.ok) {
            const body = await res.text();
            let detail = '';
            try { detail = JSON.parse(body).detail || body; } catch { detail = body; }
            showToast(`Close failed (${res.status}): ${detail}`, 'error');
            return;
        }
        showToast('Session closed successfully', 'success');
        loadSessionDetail(sessionId);
    } catch (err) {
        showToast(`Close error: ${err.message}`, 'error');
    }
}

// --- Observability Charts ---
const obsCharts = {};
let obsRefreshInterval = null;

const obsChartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
    },
    scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(148,163,184,0.08)' } },
        y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(148,163,184,0.08)' } },
    },
};

async function loadObservability() {
    const obsWin = document.getElementById('obs-window').value;

    // Persist selected window in URL hash so refresh preserves it
    const newHash = 'observability?window=' + obsWin;
    if (location.hash !== '#' + newHash) {
        history.replaceState(null, '', '#' + newHash);
    }

    const [timeseries, latency, breakdown] = await Promise.all([
        fetchJSON(`/observability/timeseries?window=${obsWin}`),
        fetchJSON(`/observability/latency?window=${obsWin}`),
        fetchJSON(`/observability/breakdown?window=${obsWin}`),
    ]);

    if (latency) {
        document.getElementById('obs-p50').textContent = latency.p50 ? latency.p50.toFixed(0) + 'ms' : '—';
        document.getElementById('obs-p90').textContent = latency.p90 ? latency.p90.toFixed(0) + 'ms' : '—';
        document.getElementById('obs-p95').textContent = latency.p95 ? latency.p95.toFixed(0) + 'ms' : '—';
        document.getElementById('obs-p99').textContent = latency.p99 ? latency.p99.toFixed(0) + 'ms' : '—';
    }

    if (timeseries) renderRequestRateChart(timeseries);
    if (latency) renderLatencyChart(latency);
    if (timeseries) renderCostChart(timeseries);
    if (timeseries) renderTokenChart(timeseries);
    if (breakdown) renderCacheChart(breakdown);
    if (breakdown) renderProviderHealthChart(breakdown);
    if (breakdown) renderTopModelsChart(breakdown);

    // Check Prometheus status
    try {
        const res = await fetch('/metrics');
        const text = await res.text();
        const el = document.getElementById('obs-prometheus-status');
        if (text.includes('stateloom_')) {
            el.innerHTML = '<span style="color:var(--accent-green,#4ade80);">Prometheus /metrics endpoint active</span>';
        } else {
            el.innerHTML = '<span style="color:var(--text-secondary);">Prometheus metrics not enabled (set metrics_enabled=True)</span>';
        }
    } catch (e) {
        document.getElementById('obs-prometheus-status').innerHTML =
            '<span style="color:var(--text-secondary);">Prometheus /metrics endpoint unavailable</span>';
    }

    // Set up auto-refresh
    if (obsRefreshInterval) clearInterval(obsRefreshInterval);
    obsRefreshInterval = setInterval(() => {
        if (currentView === 'observability') loadObservability();
    }, 30000);
}

function formatBucketTime(ts) {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function destroyChart(key) {
    if (obsCharts[key]) {
        obsCharts[key].destroy();
        obsCharts[key] = null;
    }
}

function renderRequestRateChart(data) {
    destroyChart('requestRate');
    const ctx = document.getElementById('obs-chart-request-rate');
    if (!ctx || !data.buckets) return;
    const labels = data.buckets.map(b => formatBucketTime(b.timestamp));
    const values = data.buckets.map(b => b.requests);
    obsCharts.requestRate = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Requests',
                data: values,
                borderColor: '#22d3ee',
                backgroundColor: 'rgba(34,211,238,0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 1,
            }],
        },
        options: { ...obsChartDefaults },
    });
}

function renderLatencyChart(data) {
    destroyChart('latency');
    const ctx = document.getElementById('obs-chart-latency');
    if (!ctx || !data.histogram_buckets) return;
    const labels = data.histogram_buckets.map(b => b.le >= 1000 ? (b.le / 1000) + 's' : b.le + 'ms');
    const values = data.histogram_buckets.map(b => b.count);
    obsCharts.latency = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Requests',
                data: values,
                backgroundColor: 'rgba(34,211,238,0.6)',
                borderColor: '#22d3ee',
                borderWidth: 1,
            }],
        },
        options: {
            ...obsChartDefaults,
            plugins: {
                ...obsChartDefaults.plugins,
                annotation: data.p95 ? {
                    annotations: {
                        p95Line: {
                            type: 'line', yMin: 0, yMax: 0,
                            label: { content: `p95: ${data.p95}ms`, display: true },
                        }
                    }
                } : undefined,
            },
        },
    });
}

function renderCostChart(data) {
    destroyChart('cost');
    const ctx = document.getElementById('obs-chart-cost');
    if (!ctx || !data.buckets) return;
    const labels = data.buckets.map(b => formatBucketTime(b.timestamp));
    let cumulative = 0;
    const values = data.buckets.map(b => { cumulative += b.cost; return cumulative; });
    obsCharts.cost = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Cumulative Cost ($)',
                data: values,
                borderColor: '#4ade80',
                backgroundColor: 'rgba(74,222,128,0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 1,
            }],
        },
        options: { ...obsChartDefaults },
    });
}

function renderTokenChart(data) {
    destroyChart('tokens');
    const ctx = document.getElementById('obs-chart-tokens');
    if (!ctx || !data.buckets) return;
    const labels = data.buckets.map(b => formatBucketTime(b.timestamp));
    const promptTokens = data.buckets.map(b => b.prompt_tokens);
    const completionTokens = data.buckets.map(b => b.completion_tokens);
    obsCharts.tokens = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Prompt',
                    data: promptTokens,
                    backgroundColor: 'rgba(34,211,238,0.6)',
                },
                {
                    label: 'Completion',
                    data: completionTokens,
                    backgroundColor: 'rgba(168,85,247,0.6)',
                },
            ],
        },
        options: {
            ...obsChartDefaults,
            scales: {
                ...obsChartDefaults.scales,
                x: { ...obsChartDefaults.scales.x, stacked: true },
                y: { ...obsChartDefaults.scales.y, stacked: true },
            },
        },
    });
}

function renderCacheChart(data) {
    destroyChart('cache');
    const ctx = document.getElementById('obs-chart-cache');
    if (!ctx || !data.cache) return;
    const hits = data.cache.hits || 0;
    const misses = data.cache.misses || 0;
    if (hits === 0 && misses === 0) return;
    obsCharts.cache = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Hits', 'Misses'],
            datasets: [{
                data: [hits, misses],
                backgroundColor: ['rgba(74,222,128,0.7)', 'rgba(248,113,113,0.7)'],
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
            },
        },
    });
}

function renderProviderHealthChart(data) {
    destroyChart('provider');
    const ctx = document.getElementById('obs-chart-provider');
    if (!ctx || !data.by_provider) return;
    const providers = Object.keys(data.by_provider);
    if (providers.length === 0) return;
    const requests = providers.map(p => data.by_provider[p].requests || 0);
    const costs = providers.map(p => data.by_provider[p].cost || 0);
    obsCharts.provider = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: providers,
            datasets: [{
                label: 'Requests',
                data: requests,
                backgroundColor: 'rgba(34,211,238,0.6)',
            }],
        },
        options: { ...obsChartDefaults },
    });
}

function renderTopModelsChart(data) {
    destroyChart('topModels');
    const ctx = document.getElementById('obs-chart-top-models');
    if (!ctx || !data.by_model) return;
    const models = Object.entries(data.by_model)
        .sort((a, b) => b[1].requests - a[1].requests)
        .slice(0, 10);
    if (models.length === 0) return;
    obsCharts.topModels = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(([m]) => m),
            datasets: [{
                label: 'Requests',
                data: models.map(([, v]) => v.requests),
                backgroundColor: 'rgba(34,211,238,0.6)',
            }],
        },
        options: {
            ...obsChartDefaults,
            indexAxis: 'y',
        },
    });
}

// --- Consensus ---
async function loadConsensus() {
    const data = await fetchJSON('/consensus-runs');
    if (!data) return;

    const runs = data.runs || [];
    document.getElementById('consensus-stat-total').textContent = runs.length;

    if (runs.length > 0) {
        const avgConf = runs.reduce((s, r) => s + r.confidence, 0) / runs.length;
        document.getElementById('consensus-stat-confidence').textContent = (avgConf * 100).toFixed(1) + '%';
        const totalCost = runs.reduce((s, r) => s + r.total_cost, 0);
        document.getElementById('consensus-stat-cost').textContent = '$' + totalCost.toFixed(4);
        const earlyCount = runs.filter(r => r.early_stopped).length;
        document.getElementById('consensus-stat-early').textContent = earlyCount;
    } else {
        document.getElementById('consensus-stat-confidence').textContent = '-';
        document.getElementById('consensus-stat-cost').textContent = '$0';
        document.getElementById('consensus-stat-early').textContent = '0';
    }

    const tbody = document.getElementById('consensus-tbody');
    tbody.innerHTML = '';

    if (runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No consensus runs yet</td></tr>';
        return;
    }

    runs.forEach(r => {
        const tr = document.createElement('tr');
        tr.onclick = () => loadConsensusDetail(r.session_id);
        const models = (r.models || []).join(', ');
        const dur = r.total_duration_ms > 1000
            ? (r.total_duration_ms / 1000).toFixed(1) + 's'
            : Math.round(r.total_duration_ms) + 'ms';
        tr.innerHTML = `
            <td title="${escapeHtml(r.session_id)}">${escapeHtml(r.session_id.substring(0, 12))}...</td>
            <td><span class="status-badge status-${escapeHtml(r.strategy)}">${escapeHtml(r.strategy)}</span></td>
            <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(models)}">${escapeHtml(models)}</td>
            <td>${r.total_rounds}</td>
            <td>${(r.confidence * 100).toFixed(1)}%</td>
            <td>$${r.total_cost.toFixed(4)}</td>
            <td>${dur}</td>
            <td>${new Date(r.timestamp).toLocaleString()}</td>
        `;
        tbody.appendChild(tr);
    });
}

async function loadConsensusDetail(sessionId) {
    const data = await fetchJSON(`/consensus-runs/${encodeURIComponent(sessionId)}`);
    if (!data) return;

    const panel = document.getElementById('consensus-detail-panel');
    const c = data.consensus || {};

    document.getElementById('consensus-detail-title').textContent = 'Session: ' + sessionId.substring(0, 16) + '...';
    const stratEl = document.getElementById('consensus-detail-strategy');
    stratEl.textContent = c.strategy || '';
    stratEl.className = 'status-badge status-' + (c.strategy || '');

    document.getElementById('consensus-detail-answer').textContent = c.final_answer_preview || '(none)';
    document.getElementById('consensus-detail-winner').textContent = c.winner_model || '-';
    document.getElementById('consensus-detail-aggregation').textContent = c.aggregation_method || '-';
    document.getElementById('consensus-detail-early').textContent = c.early_stopped ? 'Yes' : 'No';

    // Rounds
    const roundsContainer = document.getElementById('consensus-detail-rounds');
    roundsContainer.innerHTML = '';
    (data.rounds || []).forEach(rd => {
        const card = document.createElement('div');
        card.className = 'variant-card';
        const responses = (rd.responses_summary || []).map(rs =>
            `<div style="margin:4px 0;font-size:12px;">
                <strong>${escapeHtml(rs.model)}</strong> &mdash;
                Confidence: ${(rs.confidence * 100).toFixed(1)}% |
                Cost: $${(rs.cost || 0).toFixed(4)}
                <div style="color:var(--text-muted);margin-top:2px;">${escapeHtml((rs.content_preview || '').substring(0, 120))}${(rs.content_preview || '').length > 120 ? '...' : ''}</div>
            </div>`
        ).join('');
        card.innerHTML = `
            <div class="variant-name">Round ${rd.round_number}</div>
            <div class="variant-metrics">
                <div class="variant-metric">
                    <span class="metric-label">Agreement</span>
                    <span class="metric-value">${(rd.agreement_score * 100).toFixed(1)}%</span>
                </div>
                <div class="variant-metric">
                    <span class="metric-label">Cost</span>
                    <span class="metric-value">$${rd.round_cost.toFixed(4)}</span>
                </div>
                <div class="variant-metric">
                    <span class="metric-label">Duration</span>
                    <span class="metric-value">${rd.round_duration_ms > 1000 ? (rd.round_duration_ms / 1000).toFixed(1) + 's' : Math.round(rd.round_duration_ms) + 'ms'}</span>
                </div>
                <div class="variant-metric">
                    <span class="metric-label">Consensus</span>
                    <span class="metric-value">${rd.consensus_reached ? 'Yes' : 'No'}</span>
                </div>
            </div>
            <div style="margin-top:8px;">${responses}</div>
        `;
        roundsContainer.appendChild(card);
    });

    // Children
    const childTbody = document.getElementById('consensus-detail-children');
    childTbody.innerHTML = '';
    (data.children || []).forEach(ch => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td title="${escapeHtml(ch.id)}">${escapeHtml(ch.id.substring(0, 20))}...</td>
            <td>${escapeHtml(ch.name || '-')}</td>
            <td>$${(ch.total_cost || 0).toFixed(4)}</td>
            <td>${ch.total_tokens || 0}</td>
            <td><span class="status-badge status-${escapeHtml(ch.status)}">${escapeHtml(ch.status)}</span></td>
        `;
        childTbody.appendChild(tr);
    });

    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeConsensusDetail() {
    document.getElementById('consensus-detail-panel').style.display = 'none';
}

// --- Sidebar username (one-time fetch on page load) ---
(function loadSidebarUsername() {
    fetch(`${API_BASE}/v1/stats`).then(r => r.json()).then(data => {
        const el = document.getElementById('nav-username');
        if (el && data.username) {
            el.textContent = data.username;
            el.style.display = 'block';
        }
    }).catch(() => {});
})();

// --- Dev Mode License Banner ---
(function checkLicenseStatus() {
    fetch(`${API_BASE}/v1/license`).then(r => r.json()).then(data => {
        if (data.dev_mode && !data.valid) {
            const banner = document.getElementById('dev-mode-banner');
            if (banner) banner.style.display = 'block';
        }
    }).catch(() => {});
})();
