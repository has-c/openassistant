'use strict';
const $ = id => document.getElementById(id);
let ws, turn = 0, busy = false, muted = false, ignore = false;
let clientId = crypto.randomUUID();
try {
  clientId = localStorage.getItem('openassistant-client') || clientId;
  localStorage.setItem('openassistant-client', clientId);
} catch { /* Reconnection ownership requires browser storage. */ }
let player, audioUrl;
let joinedAt = null;
let pendingSpeech = null;
let leaving = false;
let autoConnect = false, reconnectTimer, reconnectDelay = 1000;
let connectionSerial = 0;
let taskPhase = 'idle', taskStarted = null;
const activities = new Map();
const textQueue = [];
let queuePaused = false;
function renderQueue() {
  const list = $('text-queue'); list.replaceChildren();
  for (const entry of textQueue) {
    const row = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = `${entry.status === 'sending' ? 'Sending guidance' : entry.status === 'unconfirmed' ? 'Delivery unconfirmed — check before retrying' : queuePaused ? 'Paused' : 'Queued'}: ${entry.text}`;
    row.append(label);
    const add = document.createElement('button'); add.textContent = busy ? 'Add now' : 'Send';
    add.disabled = entry.status !== 'queued' || ws?.readyState !== WebSocket.OPEN || ignore
      || observing || !historyReady;
    add.onclick = () => {
      if (busy) {
        entry.status = 'sending'; renderQueue();
        send({type:'steer', id:entry.id, text:entry.text});
      } else { queuePaused = false; dispatchQueued(entry); }
    };
    const remove = document.createElement('button'); remove.textContent = 'Remove';
    remove.disabled = entry.status === 'sending';
    remove.onclick = () => { textQueue.splice(textQueue.indexOf(entry), 1); renderQueue(); };
    row.append(add, remove); list.append(row);
  }
  list.hidden = !textQueue.length;
  $('send').setAttribute('aria-label', busy ? 'Queue message' : 'Send message');
  $('send').textContent = busy ? 'Queue' : '↑';
}
function dispatchQueued(entry = textQueue[0]) {
  if (!entry || queuePaused || busy || ignore || !historyReady || entry.status !== 'queued'
      || ws?.readyState !== WebSocket.OPEN) return;
  textQueue.splice(textQueue.indexOf(entry), 1);
  sendingEntry = entry;
  beginTask(entry.text); busy = true; syncMicrophone();
  send({type:'prompt', text:entry.text}); renderQueue();
}
function taskState(state, detail) {
  taskPhase = state;
  $('task-panel').hidden = false;
  $('live-response').hidden = false;
  document.body.classList.add('has-task');
  $('task-panel').dataset.state = state;
  $('task-state').textContent = {
    idle:'Ready', sending:'Sending…', working:'Working',
    stopping:'Stopping task…', stopped:'Task stopped',
    finished:'Turn finished', failed:'Task error', interrupted:'Task interrupted',
    disconnected:'Disconnected — task may still be working'
  }[state] || state;
  $('task-detail').textContent = detail;
  $('task-state').title = detail;

}
function beginTask(text) {
  taskStarted = Date.now(); activities.clear(); $('activity-list').replaceChildren();
  $('task-time').textContent = '0s elapsed';
  $('task-activity').hidden = true; $('task-activity').open = false;
  $('live-caption').hidden = true;
  $('activity-count').textContent = '0'; $('task-request').textContent = text;
  taskState('sending', 'Sending…');
}
function activity(data) {
  $('task-activity').hidden = false;
  const id = data.id || `update-${Date.now()}-${activities.size}`;
  let row = activities.get(id);
  if (!row) {
    row = document.createElement('li'); activities.set(id, row); $('activity-list').append(row);
    if (activities.size > 20) {
      const first = activities.keys().next().value; activities.get(first).remove(); activities.delete(first);
    }
  }
  row.textContent = `${data.label} — ${{running:'running', returned:'returned', failed:'failed or stopped', update:'update'}[data.state] || data.state}`;
  $('activity-count').textContent = String(activities.size);
}
function showPanel(open) {
  $('conversation').hidden = !open;
  $('toggle-panel').setAttribute('aria-expanded', String(open));
  if (open && innerWidth <= 1100) {
    $('chat-history').hidden = true;
    $('toggle-history').setAttribute('aria-expanded', 'false');
  }
}
$('toggle-panel').onclick = () => showPanel($('conversation').hidden);
$('close-panel').onclick = () => { showPanel(false); $('toggle-panel').focus(); };
$('shortcuts-help').onclick = () => $('shortcuts-dialog').showModal();
$('close-shortcuts').onclick = () => $('shortcuts-dialog').close();
document.addEventListener('keydown', event => {
  if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.repeat ||
      event.isComposing || (event.code !== 'KeyW' && event.key.toLowerCase() !== 'w')) return;
  const dialog = document.querySelector('dialog[open]');
  if (dialog) {
    event.preventDefault(); dialog.close(); return;
  }
  if (!$('conversation').hidden) {
    event.preventDefault(); $('close-panel').click(); return;
  }
  if (!$('chat-history').hidden) {
    event.preventDefault(); $('close-history').click();
  }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !document.querySelector('dialog[open]')) showPanel(false);
  if (event.key === 'Enter' && !event.defaultPrevented && !event.isComposing && !event.repeat &&
      !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && !$('prompt').disabled &&
      !document.querySelector('dialog[open]')) {
    const target = event.target;
    const control = target instanceof Element && (target.closest('button, a, input, textarea, select, summary, [contenteditable], [role="button"], [role="textbox"], [role="link"], [tabindex]:not([tabindex="-1"])') || target.isContentEditable);
    if (!control) { event.preventDefault(); $('prompt').focus(); }
  }
  if (!document.querySelector('dialog[open]') && event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.key.toLowerCase() === 'c') {
    event.preventDefault(); showPanel(true); $('close-panel').focus();
  }
});
setInterval(() => {
  if (joinedAt === null) return;
  const seconds = Math.floor((Date.now() - joinedAt) / 1000);
  $('elapsed').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  if (taskStarted && ['sending','working','stopping'].includes(taskPhase)) {
    $('task-time').textContent = `${Math.floor((Date.now() - taskStarted) / 1000)}s elapsed`;
  }
}, 1000);
const clips = [], messages = new Map();
const notice = text => $('notice').textContent = text;
const audioNotices = new Set(['Audio unavailable.', 'Click Play audio to allow voice playback.',
  'This browser could not play the audio. Captions remain available.',
  'Audio queue full; following captions.']);
function clearAudioNotice() { if (audioNotices.has($('notice').textContent)) notice(''); }
const microphone = new HandsFreeMic({
  onSpeechStart: () => {
    if (busy || player || clips.length) interrupt();
    $('status').textContent = 'Listening';
  },
  onSpeech: blob => {
    if (ws?.readyState !== WebSocket.OPEN) return microphone.disable();
    if (ignore) { pendingSpeech = blob; return; }
    submitSpeech(blob);
  },
  onStatus: state => {
    if (state === 'off') pendingSpeech = null;
    $('mic-label').textContent = state === 'off' ? 'Mic off' : 'Mic on';
    $('mic').setAttribute('aria-pressed', String(state !== 'off'));
    $('mic-state').textContent = {off:'Microphone off', listening:'Listening', waiting:'Waiting for OpenAssistant'}[state];
    if (state === 'listening' && !busy && !player) $('status').textContent = 'Listening';
    if (state === 'off' && !busy && !player) $('status').textContent = 'Ready';
  },
  onError: notice
});
function syncMicrophone() { microphone.setPlayback(!!player && !player.paused); }
function submitSpeech(blob) {
  beginTask('Transcribing…');
  busy = true; controls(true); $('status').textContent = 'Thinking';
  ws.send(blob);
  microphone.setPaused(false);
}
function interrupt() {
  if (ignore || ws?.readyState !== WebSocket.OPEN) return;
  queuePaused = true;
  ignore = true; clearAudio();
  if (!['finished','failed'].includes(taskPhase)) {
    taskState('stopping', 'Waiting for the agent to stop. Completed actions are not undone.');
  }
  send({type:'stop'});
}
fetch(new URL('client-config', location.href)).then(response => {
  if (!response.ok) throw new Error('Configuration unavailable');
  return response.json();
}).then(config => {
  $('key-fields').hidden = !config.access_required;
  $('token').required = config.access_required;
  $('join').disabled = false;
  autoConnect = !config.access_required;
  $('login').hidden = autoConnect;
  if (autoConnect) connect();
}).catch(() => { $('status').textContent = 'Connection unavailable'; notice('Could not load call settings. Refresh to try again.'); });
const send = data => {
  if (['stop','steer'].includes(data.type)) {
    data = {...data, conversation:conversationId};
    // Stop can be pressed before the server's start acknowledgement arrives.
    if (data.type !== 'stop') data.turn = turn;
  }
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
};
function clearAudio() {
  clips.length = 0;
  if (player) { player.pause(); player.src = ''; player = null; }
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null;
  syncMicrophone();
}
async function play() {
  if (player || muted || !clips.length) return;
  audioUrl = URL.createObjectURL(clips.shift());
  player = new Audio(audioUrl);
  player.playbackRate = 1.5;
  player.preservesPitch = true;
  player.onended = () => { player = null; URL.revokeObjectURL(audioUrl); audioUrl = null; play(); if (!clips.length && !player && !busy) { $('status').textContent = microphone.enabled ? 'Listening' : 'Ready'; controls(true); } syncMicrophone(); };
  player.onplaying = () => { clearAudioNotice(); $('status').textContent = 'Speaking'; $('interrupt').disabled = false; syncMicrophone(); };
  player.onerror = () => { clearAudio(); syncMicrophone(); notice('Audio unavailable.'); };
  syncMicrophone();
  try { await player.play(); } catch {
    $('sound-label').textContent = 'Play audio'; notice('Click Play audio to allow voice playback.');
  }
}
const imageUrls = new Map();
const imagePending = new Set();
function clearImages() {
  for (const url of imageUrls.values()) URL.revokeObjectURL(url);
  imageUrls.clear(); imagePending.clear();
  if ($('image-viewer').open) $('image-viewer').close();
}
function showImages(body, text, images) {
  body.replaceChildren();
  const pattern = /!?\[([^\]\n]{0,300})\]\((?:<([^>\n]+)>|(\/[^)\n]+))\)/g;
  let previous = 0;
  for (const match of text.matchAll(pattern)) {
    const image = images.find(entry => entry.path === (match[2] || match[3]));
    if (!image) continue;
    body.append(document.createTextNode(text.slice(previous, match.index)));
    const button = document.createElement('button');
    button.className = 'message-image'; button.dataset.imageId = image.id;
    button.setAttribute('aria-label', `Enlarge ${image.alt}`);
    const thumbnail = document.createElement('img'); thumbnail.alt = image.alt;
    if (imageUrls.has(image.id)) thumbnail.src = imageUrls.get(image.id);
    else if (!imagePending.has(image.id)) {
      imagePending.add(image.id); send({type:'get_image', id:image.id});
    }
    button.append(thumbnail);
    button.onclick = () => {
      if (!thumbnail.src) return;
      $('image-full').src = thumbnail.src; $('image-full').alt = thumbnail.alt;
      $('image-viewer').showModal();
    };
    body.append(button); previous = match.index + match[0].length;
  }
  body.append(document.createTextNode(text.slice(previous)));
}
$('close-image').onclick = () => $('image-viewer').close();
function message(id, who, text, append = false, images = []) {
  id = `${connectionSerial}-${id}`;
  $('empty')?.remove();
  if (!messages.has(id)) {
    const block = document.createElement('div'); block.className = 'message' + (who === 'You' ? ' user' : '');
    const label = document.createElement('strong'); label.textContent = who;
    const body = document.createElement('span'); block.append(label, body); $('transcript').append(block);
    messages.set(id, body);
    if (messages.size > 100) { const first = messages.keys().next().value; messages.get(first).parentElement.remove(); messages.delete(first); }
  }
  const body = messages.get(id);
  const raw = append ? (body.dataset.raw || '') + text : text;
  body.dataset.raw = raw;
  showImages(body, raw, images);
  $('live-caption').hidden = false;
  $('live-response').hidden = false;
  const caption = raw.replace(/!?\[([^\]\n]*)\]\((?:<\/[^>\n]+>|\/[^)\n]+)\)/g, '$1');
  $('live-caption').textContent = who === 'OpenAssistant' ? caption : `${who}: ${caption}`;
  $('transcript').scrollTop = $('transcript').scrollHeight;
}
function controls(connected) {
  for (const id of ['mic', 'sound', 'leave', 'prompt', 'send']) $(id).disabled = !connected;
  for (const id of ['prompt', 'send', 'mic']) $(id).disabled = !connected || !historyReady;
  $('mic').disabled ||= observing;
  $('interrupt').disabled = !connected || observing || !(busy || player || clips.length);
  for (const id of ['new-chat', 'search-chats', 'refresh-chats']) $(id).disabled = !connected;
  for (const button of $('saved-chats').querySelectorAll('button')) button.disabled = !connected;
  renderQueue();
}
function stopMic() { microphone.disable(); }
function connect() {
  if (ws && ws.readyState < 2) return;
  clearTimeout(reconnectTimer);
  leaving = false;
  $('login').hidden = autoConnect;
  $('status').textContent = 'Connecting…';
  $('connection').textContent = 'Connecting…';
  const token = $('token').value; $('token').value = '';
  const socketUrl = new URL('ws', location.href);
  socketUrl.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(socketUrl);
  ws.onopen = () => ws.send(JSON.stringify({token, client_id:clientId}));
  ws.onerror = () => notice('Could not connect. Check your connection and try again.');
  ws.onclose = event => {
    queuePaused = true;
    for (const entry of textQueue) if (entry.status === 'sending') entry.status = 'unconfirmed';
    joinedAt = null; document.body.classList.remove('connected');
    document.querySelector('.portrait h2').textContent = leaving ? 'Call ended' : 'Connection lost';
    $('live-caption').hidden = true;
    stopMic(true); clearAudio(); busy = false; ignore = false; pendingSpeech = null;
    historyReady = false; observing = false; controls(false);
    $('login').hidden = false; $('in-call').hidden = true;
    $('status').textContent = leaving ? 'Call ended' : 'Connection lost'; $('connection').textContent = 'Disconnected';
    notice(leaving ? 'Call ended.' : 'Connection lost. Reconnecting…');
    if (['sending','working','stopping'].includes(taskPhase)) {
      taskState('disconnected', 'Rejoin to see the latest progress. Closing the app does not stop an accepted task.');
    }
    if (autoConnect && !leaving && event.code !== 1008) {
      $('login').hidden = true;
      $('connection').textContent = 'Reconnecting…';
      notice('Reconnecting…');
      reconnectTimer = setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, 30000);
    }
  };
  ws.onmessage = event => {
    const data = JSON.parse(event.data);
    if (historyEvent(data)) return;
    if (data.conversation && data.conversation !== conversationId) return;
    if (data.type === 'image_data') {
      if (!imagePending.has(data.id) || !['image/png', 'image/jpeg', 'image/gif', 'image/webp'].includes(data.mime)) return;
      imagePending.delete(data.id);
      const bytes = Uint8Array.from(atob(data.data), char => char.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], {type:data.mime}));
      imageUrls.set(data.id, url);
      document.querySelectorAll('.message-image').forEach(button => {
        if (button.dataset.imageId === data.id) button.querySelector('img').src = url;
      });
      return;
    }
    if (data.type === 'steer_result') {
      const entry = textQueue.find(entry => entry.id === data.id);
      if (!entry) return;
      if (data.status === 'accepted') {
        textQueue.splice(textQueue.indexOf(entry), 1);
        message(`guidance-${entry.id}`, 'You · added to current task', entry.text);
      } else {
        entry.status = data.status === 'not_sent' ? 'queued' : 'unconfirmed';
        if (entry.status === 'unconfirmed') queuePaused = true;
        notice(data.status === 'not_sent' ? 'Kept in queue. OpenAssistant is not ready for guidance yet.'
          : 'Guidance delivery is unconfirmed. Check the reply before sending again.');
      }
      renderQueue();
      if (!busy) dispatchQueued();
      return;
    }
    if (data.type === 'ready') {
      reconnectDelay = 1000;
      connectionSerial += 1; turn = 0;
      joinedAt = Date.now(); $('elapsed').textContent = '00:00';
      document.body.classList.add('connected');
      document.querySelector('.portrait h2').textContent = 'OpenAssistant';
      $('login').hidden = true; $('in-call').hidden = false; $('name').textContent = data.name;
      $('connection').textContent = 'Connected'; $('status').textContent = 'Ready';
      notice(''); controls(true);
      refreshHistory();
      send(conversationId ? {type:'open_conversation', id:conversationId} : {type:'new_conversation'});
      return;
    }
    if (data.type === 'start' && !ignore) {
      observing = !!data.observer; sendingEntry = null;
      if (observing) { queuePaused = true; microphone.disable(); }
      turn = data.turn; busy = true; clearAudio(); syncMicrophone(); controls(true);
      taskState('working', 'Speaking over OpenAssistant stops this task. Sound off keeps it running.');
    }
    if (data.type === 'stopped') {
      observing = false;
      ignore = false; busy = false; $('status').textContent = microphone.enabled ? 'Listening' : 'Ready';
      if (!['finished','failed'].includes(taskPhase)) {
        taskState('stopped', 'Agent turn stopped. Completed actions are not undone; external jobs may still be running.');
        for (const row of activities.values()) {
          if (row.textContent.endsWith('— running')) row.textContent = row.textContent.replace(/running$/, 'outcome unconfirmed');
        }
      }
      controls(true);
      if (pendingSpeech) { const blob = pendingSpeech; pendingSpeech = null; submitSpeech(blob); }
      syncMicrophone(); return;
    }
    if (ignore || (data.turn && data.turn !== turn)) return;
    if (data.type === 'task') taskState(data.state, data.state === 'finished'
      ? 'The agent has finished this turn. Check its reply for results; audio may still be playing.'
      : data.state === 'failed' ? `${data.text} Check completed actions before retrying. External work may already have happened.`
      : data.text + '. Speaking over OpenAssistant stops the task; Sound off only mutes replies.');
    if (data.type === 'activity') activity(data);
    if (data.type === 'status') $('status').textContent = data.text;
    if (data.type === 'user') { $('task-request').textContent = data.text; message(`user-${turn}`, 'You', data.text); }
    if (data.type === 'delta') message(`${turn}-${data.item}`, 'OpenAssistant', data.text, true);
    if (data.type === 'message') { message(`${turn}-${data.item}`, 'OpenAssistant', data.text, false, data.images || []); if (data.images?.length) showPanel(true); }
    if (data.type === 'result' && ![...messages.keys()].some(key => key.startsWith(`${connectionSerial}-${turn}-`))) { message(`${turn}-result`, 'OpenAssistant', data.text, false, data.images || []); if (data.images?.length) showPanel(true); }
    if (data.type === 'audio' && !muted) {
      if (clips.length >= 16) { notice('Audio queue full; following captions.'); clearAudio(); return; }
      clips.push(new Blob([Uint8Array.from(atob(data.data), char => char.charCodeAt(0))], {type:'audio/ogg'})); play();
    }
    if (data.type === 'notice') notice(data.text);
    if (data.type === 'done') {
      if (taskPhase === 'failed') queuePaused = true;
      if (!player) $('status').textContent = microphone.enabled ? 'Listening' : 'Ready';
      controls(true); syncMicrophone();
    }
    if (data.type === 'turn_ready') {
      const wasObserving = observing;
      busy = false; observing = false; controls(true);
      if (wasObserving) send({type:'open_conversation', id:conversationId});
      refreshHistory(); dispatchQueued();
    }
  };
}
$('login').onsubmit = event => { event.preventDefault(); connect(); };
const promptHistories = new Map();
function resizePrompt() {
  const prompt = $('prompt');
  prompt.style.height = 'auto';
  prompt.style.height = `${Math.min(prompt.scrollHeight, 160)}px`;
}
$('prompt').addEventListener('input', resizePrompt);
function currentPromptHistory() {
  if (!promptHistories.has(conversationId)) promptHistories.set(conversationId, {items:[], index:0, draft:''});
  return promptHistories.get(conversationId);
}
$('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey && !event.altKey && !event.isComposing && !event.repeat) {
    event.preventDefault(); event.stopPropagation();
    const previousLength = textQueue.length;
    $('composer').requestSubmit();
    if (busy && textQueue.length > previousLength) {
      const button = $('text-queue').lastElementChild?.querySelector('button');
      if (button && !button.disabled) button.click();
    }
    return;
  }
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !event.altKey && !event.ctrlKey && !event.metaKey) {
    event.preventDefault(); $('composer').requestSubmit(); return;
  }
  if (event.isComposing || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  const history = currentPromptHistory();
  const prompt = $('prompt');
  if (event.key === 'ArrowUp' && !prompt.value.slice(0, prompt.selectionStart).includes('\n') && history.items.length && history.index > 0) {
    event.preventDefault();
    if (history.index === history.items.length) history.draft = $('prompt').value;
    $('prompt').value = history.items[--history.index];
    resizePrompt();
  } else if (event.key === 'ArrowDown' && !prompt.value.slice(prompt.selectionEnd).includes('\n') && history.index < history.items.length) {
    event.preventDefault();
    history.index++;
    $('prompt').value = history.index === history.items.length ? history.draft : history.items[history.index];
    resizePrompt();
  }
});
$('composer').onsubmit = event => {
  event.preventDefault();
  const text = $('prompt').value.trim();
  if (!text || ws?.readyState !== WebSocket.OPEN) return;
  if (textQueue.length >= 10) return notice('Queue full (10 messages). Remove one before adding more.');
  const history = currentPromptHistory();
  history.items.push(text); if (history.items.length > 100) history.items.shift();
  history.index = history.items.length; history.draft = '';
  textQueue.push({id:crypto.randomUUID(), text, status:'queued'});
  $('prompt').value = ''; resizePrompt(); renderQueue();
  if (!busy && !ignore && !queuePaused) dispatchQueued();
};
$('mic').onclick = async () => {
  if (microphone.enabled) return microphone.disable();
  syncMicrophone();
  await microphone.enable();
};
$('sound').onclick = async () => {
  if (player?.paused && !muted) {
    try { await player.play(); $('sound-label').textContent = 'Sound on'; } catch { notice('This browser could not play the audio. Captions remain available.'); }
    return;
  }
  muted = !muted; if (!muted) clearAudioNotice(); $('sound-label').textContent = muted ? 'Sound off' : 'Sound on'; $('sound').setAttribute('aria-pressed', String(muted)); if (muted) clearAudio(); syncMicrophone();
};
$('interrupt').onclick = () => { queuePaused = true; interrupt(); renderQueue(); };
$('leave').onclick = () => { leaving = true; clearTimeout(reconnectTimer); stopMic(true); clearAudio(); ws?.close(); };
window.addEventListener('pagehide', () => { clearTimeout(reconnectTimer); stopMic(true); clearAudio(); ws?.close(); autoConnect = false; });
window.addEventListener('pageshow', event => { if (event.persisted && !leaving) { autoConnect = !$('token').required; if (autoConnect) connect(); } });
