/*
 * meSpeak.js (eSpeak compiled to JavaScript) in a worker: synthesis runs off the main thread
 * and returns WAV bytes, which the page decodes and plays with Web Audio.
 *
 * From the page:
 *   { type: 'load', voices: ['voices/el.json', 'voices/en-us.json'] }
 *   { type: 'speak', id, text, voice, speed, pitch, amplitude, wordgap }
 * To the page:
 *   { type: 'ready', voices: [ids] }, { type: 'audio', id, wav: ArrayBuffer, ms }, { type: 'error', id, message }
 *
 * meSpeak's npm build is CommonJS; a two-line module shim loads it here unchanged.
 */
var __mods = {};
function require(name) { return __mods[name]; }
var module = { exports: {} };
var exports = module.exports;
importScripts('ESpeak.js');
__mods['./ESpeak.js'] = module.exports;
module = { exports: {} };
exports = module.exports;
importScripts('index.js');
var meSpeak = module.exports;

function getJSON(url) {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', url, false); // synchronous XHR is allowed in workers
  xhr.send(null);
  if (xhr.status !== 200 && xhr.status !== 0) throw new Error(url + ': HTTP ' + xhr.status);
  return JSON.parse(xhr.responseText);
}

self.onmessage = function (e) {
  var d = e.data || {};
  try {
    if (d.type === 'load') {
      meSpeak.loadConfig(getJSON('mespeak_config.json'));
      var ids = [];
      (d.voices || []).forEach(function (v) {
        var data = getJSON(v);
        meSpeak.loadVoice(data);
        ids.push(data.voice_id);
      });
      self.postMessage({ type: 'ready', voices: ids });
    } else if (d.type === 'speak') {
      var t0 = Date.now();
      var bytes = meSpeak.speak(String(d.text || ''), {
        voice: d.voice, speed: d.speed || 160, pitch: d.pitch == null ? 45 : d.pitch,
        amplitude: d.amplitude || 100, wordgap: d.wordgap || 0, rawdata: 'array'
      });
      if (!bytes || !bytes.length) throw new Error('no audio produced');
      var buf = new Uint8Array(bytes).buffer;
      self.postMessage({ type: 'audio', id: d.id, wav: buf, ms: Date.now() - t0 }, [buf]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: d.id, message: String(err && err.message || err) });
  }
};
