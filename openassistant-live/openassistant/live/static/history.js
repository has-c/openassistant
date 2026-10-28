'use strict';
let conversationId = null, historyReady = false, observing = false, nextChatPage = null;
let sendingEntry = null;
let historySearchTimer;
const chatQueues = new Map();
const pendingTitleAnimations = new Map();
function revealTitle(button, title) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const letters = typeof Intl.Segmenter === 'function'
    ? [...new Intl.Segmenter(undefined, {granularity:'grapheme'}).segment(title)].map(part => part.segment)
    : Array.from(title);
  button.setAttribute('aria-label', title);
  button.textContent = '';
  const visible = document.createElement('span'); visible.setAttribute('aria-hidden', 'true');
  button.append(visible);
  let index = 0;
  function nextLetter() {
    if (!button.isConnected) return;
    visible.textContent += letters[index++];
    if (index < letters.length) setTimeout(nextLetter, 35);
  }
  if (letters.length) nextLetter();
}
try { conversationId = localStorage.getItem('openassistant-conversation'); } catch { /* Storage optional. */ }
function selectConversation(id = null) {
  if (ignore || !historyReady) {
    return notice('Wait for the current chat change or stop to finish.');
  }
  microphone.disable(); pendingSpeech = null; clearAudio();
  historyReady = false; controls(true);
  $('history-status').textContent = 'Loading conversation…';
  send(id ? {type:'open_conversation', id} : {type:'new_conversation'});
}
// app.js uses deferred scripts, so register after it has created the call controls.
window.addEventListener('DOMContentLoaded', () => {
  const history = $('chat-history'), resize = $('resize-history');
  let preferredWidth = null, dragging = false;
  try { const saved = Number(localStorage.getItem('openassistant-sidebar-width')); if (saved > 0) preferredWidth = saved; } catch { /* Storage optional. */ }
  function bounds() {
    const available = document.querySelector('.workspace').clientWidth;
    return {min: Math.min(200, available), max: Math.max(200, Math.floor(available * (innerWidth <= 650 ? 0.9 : 0.5)))};
  }
  function applyWidth(width, persist = false) {
    const {min, max} = bounds();
    const value = Math.max(min, Math.min(max, Math.round(width)));
    document.querySelector('main').style.setProperty('--history-width', `${value}px`);
    resize.setAttribute('aria-valuemin', String(min));
    resize.setAttribute('aria-valuemax', String(max));
    resize.setAttribute('aria-valuenow', String(value));
    if (persist) {
      preferredWidth = value;
      try { localStorage.setItem('openassistant-sidebar-width', String(value)); } catch { /* Storage optional. */ }
    }
  }
  function refreshWidth() {
    if (preferredWidth !== null) applyWidth(preferredWidth);
    else {
      document.querySelector('main').style.removeProperty('--history-width');
      if (!history.hidden) {
        const {min, max} = bounds();
        resize.setAttribute('aria-valuemin', String(min));
        resize.setAttribute('aria-valuemax', String(max));
        resize.setAttribute('aria-valuenow', String(Math.round(history.getBoundingClientRect().width)));
      }
    }
  }
  resize.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    dragging = true; resize.setPointerCapture(event.pointerId); event.preventDefault();
  });
  resize.addEventListener('pointermove', event => {
    if (dragging) applyWidth(event.clientX - history.getBoundingClientRect().left);
  });
  resize.addEventListener('pointerup', event => {
    if (!dragging) return;
    dragging = false; applyWidth(event.clientX - history.getBoundingClientRect().left, true);
  });
  resize.addEventListener('pointercancel', () => { dragging = false; refreshWidth(); });
  resize.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') {
      preferredWidth = null;
      try { localStorage.removeItem('openassistant-sidebar-width'); } catch { /* Storage optional. */ }
      refreshWidth(); return;
    }
    applyWidth(history.getBoundingClientRect().width + (event.key === 'ArrowRight' ? 16 : -16), true);
  });
  window.addEventListener('resize', refreshWidth);
  function showHistory(open) {
    $('chat-history').hidden = !open;
    $('toggle-history').setAttribute('aria-expanded', String(open));
    if (open) refreshWidth();
    if (open && innerWidth <= 1100) showPanel(false);
  }
  showHistory(false);
  $('toggle-history').onclick = () => showHistory($('chat-history').hidden);
  $('close-history').onclick = () => { showHistory(false); $('toggle-history').focus(); };
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !document.querySelector('dialog[open]')) showHistory(false);
    if (document.querySelector('dialog[open]')) return;
    const target = event.target;
    const typing = target instanceof Element && (target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])') || target.isContentEditable);
    if (event.key.toLowerCase() === 'c' && !event.isComposing && !event.repeat &&
        !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && !typing &&
        !$('image-viewer').open) {
      event.preventDefault(); showHistory(true);
      ($('saved-chats').querySelector('li > button:not(.remove-chat)') || $('search-chats')).focus();
    }
    if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.isComposing && !event.repeat && !$('image-viewer').open) {
      const match = /^Digit([1-9])$/.exec(event.code);
      const button = match && $('saved-chats').querySelectorAll('li > button:not(.remove-chat)')[Number(match[1]) - 1];
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    }
    if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && (event.code === 'KeyH' || event.key.toLowerCase() === 'h')) {
      event.preventDefault(); showHistory(true);
      ($('saved-chats').querySelector('li > button:not(.remove-chat)') || $('search-chats')).focus();
    }
  });
  history.addEventListener('keydown', event => {
    if (!['ArrowUp', 'ArrowDown'].includes(event.key) || event.isComposing || event.altKey ||
        event.ctrlKey || event.metaKey || event.shiftKey) return;
    const buttons = [...$('saved-chats').querySelectorAll('li > button:not(.remove-chat)')];
    const current = buttons.indexOf(document.activeElement);
    if (current < 0 && document.activeElement !== $('search-chats')) return;
    if (!buttons.length) return;
    event.preventDefault();
    const next = current < 0 ? (event.key === 'ArrowDown' ? 0 : buttons.length - 1)
      : Math.max(0, Math.min(buttons.length - 1, current + (event.key === 'ArrowDown' ? 1 : -1)));
    buttons[next].focus();
  });
  $('new-chat').onclick = () => selectConversation();
  $('refresh-chats').onclick = () => refreshHistory();
  $('older-chats').onclick = () => refreshHistory(nextChatPage);
  $('search-chats').oninput = () => {
    clearTimeout(historySearchTimer);
    historySearchTimer = setTimeout(() => refreshHistory(), 200);
  };
});
function refreshHistory(offset = 0) {
  if (ws?.readyState === WebSocket.OPEN) {
    send({type:'conversations', offset, query:$('search-chats').value});
  }
}
function historyEvent(data) {
  if (data.type === 'conversation_hidden') {
    refreshHistory();
    return true;
  }
  if (data.type === 'title_changed') {
    const row = [...$('saved-chats').children].find(item => item.dataset.id === data.id);
    const button = row?.querySelector('button:not(.remove-chat)');
    if (button && !$('chat-history').hidden) pendingTitleAnimations.set(data.id, button.getAttribute('aria-label') || button.textContent);
    refreshHistory(); return true;
  }
  if (data.type === 'chat_status') {
    const row = [...$('saved-chats').children].find(row => row.dataset.id === data.id);
    if (row) renderChatStatus(row, data.task);
    else refreshHistory();
    return true;
  }
  if (data.type === 'conversations') {
    if ((data.query || '') !== $('search-chats').value) return true;
    const list = $('saved-chats');
    if (!data.offset) list.replaceChildren();
    for (const chat of data.chats) {
      if ([...list.children].some(row => row.dataset.id === chat.id)) continue;
      const row = document.createElement('li'); row.dataset.id = chat.id;
      if (list.children.length < 9) row.dataset.shortcut = String(list.children.length + 1);
      const button = document.createElement('button'); button.textContent = chat.title;
      const previousTitle = pendingTitleAnimations.get(chat.id);
      pendingTitleAnimations.delete(chat.id);
      button.title = `${chat.title} · ${new Date(chat.updated * 1000).toLocaleString()}`;
      button.setAttribute('aria-current', chat.id === conversationId ? 'page' : 'false');
      button.onclick = () => selectConversation(chat.id);
      const remove = document.createElement('button');
      remove.className = 'remove-chat'; remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${chat.title} from recent chats`);
      remove.title = 'Remove from recent chats'; remove.textContent = '×';
      remove.onclick = () => { remove.disabled = true; send({type:'hide_conversation', id:chat.id}); };
      const updated = document.createElement('time'); updated.className = 'chat-updated';
      const date = new Date(chat.updated * 1000), today = new Date();
      if (Number.isFinite(date.getTime())) {
        updated.dateTime = date.toISOString();
        updated.textContent = date.toDateString() === today.toDateString()
          ? `Today · ${date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}`
          : date.toLocaleDateString([], {day:'numeric', month:'short',
            ...(date.getFullYear() !== today.getFullYear() ? {year:'numeric'} : {})});
      }
      row.append(button, updated, remove); list.append(row);
      if (previousTitle !== undefined && previousTitle !== chat.title) revealTitle(button, chat.title);
      renderChatStatus(row, chat.task);
    }
    if (!list.children.length) {
      const empty = document.createElement('li'); empty.className = 'empty-chats';
      empty.textContent = $('search-chats').value ? 'No matching chats.' : 'No chats yet.';
      list.append(empty);
    }
    nextChatPage = data.next;
    $('older-chats').hidden = nextChatPage === null;
    return true;
  }
  if (data.type === 'conversation_error') {
    historyReady = false; controls(true);
    if (data.code === 'not_found') {
      conversationId = null;
      try { localStorage.removeItem('openassistant-conversation'); } catch { /* Storage optional. */ }
      $('history-status').textContent = ''; notice('');
      refreshHistory(); send({type:'new_conversation'});
      return true;
    }
    $('history-status').textContent = data.text;
    notice(data.text);
    return true;
  }
  if (data.type === 'conversation_busy') {
    if (sendingEntry) { textQueue.unshift(sendingEntry); sendingEntry = null; }
    queuePaused = true; busy = true; observing = true;
    notice(data.text + ' Your message is paused in the queue.'); controls(true);
    return true;
  }
  if (data.type === 'conversation') {
    if (conversationId !== data.id) {
      if (conversationId) {
        const entries = textQueue.map(entry => ({...entry,
          status:entry.status === 'sending' ? 'unconfirmed' : entry.status}));
        if (sendingEntry) entries.unshift({...sendingEntry, status:'unconfirmed'});
        chatQueues.set(conversationId, {entries, draft:$('prompt').value});
      }
      const saved = chatQueues.get(data.id);
      textQueue.splice(0, textQueue.length, ...(saved?.entries || []));
      $('prompt').value = saved?.draft || '';
      resizePrompt();
      sendingEntry = null;
      // A background chat's unsent queue never runs merely because it was reopened.
      queuePaused = textQueue.length > 0;
    }
    clearRequests();
    conversationId = data.id; historyReady = true; observing = !!data.observer;
    if (observing) { queuePaused = true; microphone.disable(); }
    try { localStorage.setItem('openassistant-conversation', data.id); } catch { /* Storage optional. */ }
    turn = data.turn; busy = data.busy; ignore = false; clearAudio();
    messages.clear(); clearImages(); $('transcript').replaceChildren();
    $('live-caption').hidden = true; $('live-response').hidden = true;
    $('task-panel').hidden = true; document.body.classList.remove('has-task');
    activities.clear(); $('activity-list').replaceChildren();
    taskPhase = 'idle'; taskStarted = null;
    $('status').textContent = data.busy ? 'Working' : 'Ready';
    const activeItems = new Set(data.busy ? (data.task?.responses || []).map(item => item.item) : []);
    for (const item of data.messages) {
      message(activeItems.has(item.id) ? `${turn}-${item.id}` : `saved-${item.id}`, item.who, item.text, false, item.images || []);
    }
    $('live-caption').hidden = true; $('live-response').hidden = true;
    if (!data.messages.length) {
      const empty = document.createElement('p'); empty.id = 'empty';
      $('transcript').append(empty);
    }
    $('history-status').textContent = data.busy ? (observing
      ? 'Viewing another browser’s task.' : 'Task running.')
      : data.truncated ? 'Showing recent messages.' : '';
    if (data.task) {
      const task = data.task;
      if (!observing) for (const request of task.pending_ui || []) renderRequest(request, false);
      taskStarted = task.started ? task.started * 1000 : Date.now();
      $('task-time').textContent = `${Math.max(0, Math.floor((Date.now() - taskStarted) / 1000))}s elapsed`;
      $('task-request').textContent = task.request || '';
      taskState(task.state || 'working', task.detail || 'Working');
      $('status').textContent = data.busy ? 'Working' : 'Ready';
      for (const item of task.activities || []) activity(item);
      // Native history owns completed replies. The snapshot fills in the active turn only.
      if (data.busy) {
        for (const item of task.responses || []) message(`${turn}-${item.item}`, 'OpenAssistant', item.text, false, item.images || []);
      } else if (task.state === 'finished') {
        const latest = [...data.messages].reverse().find(item => item.who === 'OpenAssistant');
        if (latest) { $('live-caption').textContent = latest.text; $('live-caption').hidden = false; }
      }
    } else if (data.busy) taskState('working', 'Watching a task started in another call.');
    if (innerWidth <= 1100) {
      $('chat-history').hidden = true; $('toggle-history').setAttribute('aria-expanded', 'false');
    }
    controls(true); refreshHistory(); return true;
  }
  return false;
}

function renderChatStatus(row, task) {
  row.querySelector('.chat-task')?.remove();
  const state = task?.state;
  if (!state) { delete row.dataset.task; return; }
  row.dataset.task = state;
  const label = document.createElement('span'); label.className = 'chat-task';
  const labels = {working:'Working',
    finished:'Finished', failed:'Task error', stopped:'Stopped', interrupted:'Interrupted'};
  label.textContent = labels[state] || state;
  label.title = task.detail || label.textContent;
  row.append(label);
}
