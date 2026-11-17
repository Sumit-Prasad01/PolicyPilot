const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');
const exampleButtons = Array.from(document.querySelectorAll('.example'));

let busy = false;

function escapeHtml(value = '') {
    return String(value).replace(/[&<>'"]/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[c]));
}

function formatText(value = '') {
    return escapeHtml(value).replace(/\n/g, '<br>');
}

function safeUrl(value = '') {
    try {
        const url = new URL(String(value), window.location.origin);
        return (url.protocol === 'http:' || url.protocol === 'https:') ? url.href : '';
    } catch {
        return '';
    }
}

function citationsHtml(citations = []) {
    citations = citations.filter(Boolean);
    if (!citations.length) return '';

    const items = citations.map(c => {
        const title = escapeHtml(c.title || c.url || 'Source');
        const url = safeUrl(c.url);
        return url
            ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a>`
            : title;
    });

    return `<div class="citations"><strong>Sources</strong><br>${items.join('<br>')}</div>`;
}

function addMessage(role, text, source = '', citations = []) {
    const wrap = document.createElement('div');
    wrap.className = `message ${role}`;
    wrap.innerHTML = `<div class="avatar">AI</div><div class="bubble">${formatText(text)}${source ? `<div class="answer-source">Source: ${escapeHtml(source)}</div>` : ''}${citationsHtml(citations)}</div>`;
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
}

function renderTrace(items = []) {
    trace.innerHTML = items.length
        ? items.map(x => `<div class="trace-item">${escapeHtml(x)}</div>`).join('')
        : '<div class="empty">No trace.</div>';
}

function setBusy(value) {
    busy = value;
    form.querySelector('button').disabled = value;
    exampleButtons.forEach(button => { button.disabled = value; });
}

async function askAgent(q) {
    if (busy) return;

    setBusy(true);
    addMessage('user', q);
    question.value = '';
    renderTrace(['Running LangGraph workflow...']);
    sourceUsed.textContent = 'Running';

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);

        addMessage('assistant', data.answer || 'No answer returned.', data.source_used, data.citations || []);
        renderTrace(data.trace || []);
        sourceUsed.textContent = data.source_used || 'unknown';
    } catch (e) {
        addMessage('assistant', `Error: ${e.message}`);
        renderTrace(['Request failed']);
        sourceUsed.textContent = 'Error';
    } finally {
        setBusy(false);
    }
}

form.addEventListener('submit', e => {
    e.preventDefault();
    const q = question.value.trim();
    if (q) askAgent(q);
});

question.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        form.requestSubmit();
    }
});

exampleButtons.forEach(button => {
    button.addEventListener('click', () => askAgent(button.textContent.trim()));
});

const modal = document.getElementById('uploadModal');
const uploadBtn = document.getElementById('uploadBtn');

document.getElementById('openUpload').onclick = () => modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick = () => modal.classList.add('hidden');

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') modal.classList.add('hidden');
});

uploadBtn.onclick = async () => {
    if (uploadBtn.disabled) return;

    const file = document.getElementById('fileInput').files[0];
    const key = document.getElementById('adminKey').value;
    const status = document.getElementById('uploadStatus');

    if (!file) {
        status.textContent = 'Choose a file first.';
        return;
    }

    status.textContent = 'Indexing document...';
    uploadBtn.disabled = true;

    const fd = new FormData();
    fd.append('file', file);

    try {
        const r = await fetch('/api/ingest', {
            method: 'POST',
            headers: { 'X-Admin-Key': key },
            body: fd
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || `Upload failed (${r.status})`);

        status.textContent = `Indexed ${d.file}: ${d.chunks} chunks.`;
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
    } finally {
        uploadBtn.disabled = false;
    }
};
