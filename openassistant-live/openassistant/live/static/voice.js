'use strict';

// Echo-cancelled utterances; keep listening while OpenAssistant replies so speech can interrupt.
class HandsFreeMic {
  constructor({onSpeech, onSpeechStart = () => {}, onStatus, onError}) {
    Object.assign(this, {onSpeech, onSpeechStart, onStatus, onError});
    this.enabled = false;
    this.paused = false;
    this.epoch = 0;
    this.playback = false;
  }

  setPlayback(active) { this.playback = active; }

  async enable() {
    if (this.enabled) return;
    const epoch = ++this.epoch;
    this.enabled = true;
    try {
      if (!navigator.mediaDevices || !window.MediaRecorder) throw new Error('unsupported');
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.context = new AudioContext();
      await this.context.resume();
      const stream = await navigator.mediaDevices.getUserMedia({audio: {
        echoCancellation: true, noiseSuppression: true, autoGainControl: true
      }});
      if (!this.enabled || epoch !== this.epoch) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      this.stream = stream;
      this.source = this.context.createMediaStreamSource(stream);
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 2048;
      this.source.connect(this.analyser);
      this.samples = new Float32Array(this.analyser.fftSize);
      stream.getAudioTracks()[0].onended = () => this.fail('Microphone disconnected. Turn it on to retry.');
      this.resume();
    } catch {
      if (epoch === this.epoch) this.fail('Microphone unavailable. Allow microphone access, or type a message.');
    }
  }

  disable() {
    this.enabled = false;
    this.epoch++;
    clearTimeout(this.restart);
    this.finish(false);
    this.stream?.getTracks().forEach(track => { track.onended = null; track.stop(); });
    this.stream = null;
    this.source?.disconnect();
    this.context?.close().catch(() => {});
    this.context = null;
    this.onStatus('off');
  }

  fail(message) { this.disable(); this.onError(message); }

  setPaused(paused) {
    this.paused = paused;
    if (paused) {
      clearTimeout(this.restart);
      this.finish(false);
      if (this.enabled) this.onStatus('waiting');
    } else this.resume();
  }

  resume() {
    clearTimeout(this.restart);
    if (!this.enabled || this.paused || !this.stream || this.recorder) return;
    this.restart = setTimeout(() => this.record(), 80);
  }

  record() {
    if (!this.enabled || this.paused || !this.stream || this.recorder) return;
    const chunks = [];
    const epoch = this.epoch;
    let size = 0, voiced = 0, lastVoice = 0, continuous = 0, speechStarted = false;
    let noise = 0.004;
    const started = performance.now();
    let recorder;
    try { recorder = new MediaRecorder(this.stream); }
    catch { this.fail('This browser could not start the microphone. You can still type.'); return; }
    this.recorder = recorder;
    recorder.submit = false;
    recorder.ondataavailable = event => {
      size += event.data.size;
      if (size > 9_000_000) this.fail('Voice segment too large. Turn the mic on to try again.');
      else chunks.push(event.data);
    };
    recorder.onerror = () => this.fail('Microphone recording failed. Turn it on to retry.');
    recorder.onstop = () => {
      if (this.recorder === recorder) this.recorder = null;
      if (this.enabled && !this.paused && epoch === this.epoch && recorder.submit && size > 0) {
        this.paused = true;
        this.onStatus('waiting');
        this.onSpeech(new Blob(chunks, {type: recorder.mimeType}));
      } else this.resume();
    };
    try { recorder.start(250); }
    catch { this.recorder = null; this.fail('Microphone recording could not start. You can still type.'); return; }
    this.onStatus('listening');
    this.tick = setInterval(() => {
      this.analyser.getFloatTimeDomainData(this.samples);
      let sum = 0;
      for (const sample of this.samples) sum += sample * sample;
      const level = Math.sqrt(sum / this.samples.length);
      const now = performance.now();
      const threshold = Math.max(this.playback ? 0.035 : 0.018, noise * 3);
      if (level > threshold) {
        voiced += 50; continuous += 50; lastVoice = now;
        if (!speechStarted && continuous >= 200) {
          speechStarted = true;
          this.onSpeechStart();
        }
      } else {
        continuous = 0;
        if (!speechStarted && !this.playback) noise = noise * 0.95 + level * 0.05;
      }
      // Ignore clicks; submit after a natural pause, without any send button.
      if (speechStarted && voiced >= 250 && now - lastVoice >= 750) this.finish(true);
      else if (now - started >= 60000) this.finish(speechStarted && voiced >= 250);
      else if (!lastVoice && now - started >= 15000) this.finish(false);
    }, 50);
  }

  finish(submit) {
    clearInterval(this.tick);
    if (this.recorder && !submit) this.recorder.submit = false;
    if (this.recorder?.state === 'recording') {
      this.recorder.submit = submit;
      this.recorder.stop();
    }
  }
}
window.HandsFreeMic = HandsFreeMic;
