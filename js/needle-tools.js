/*
 * The tools Zagreus exposes to the Needle 2 router, grouped in small profiles.
 *
 * Needle 2 picks well among two or three tools and poorly among eight, so the page first
 * chooses a domain with the multilingual MiniLM intent classifier and then asks Needle to
 * decide the exact call (and its arguments) inside that domain only. Each profile has its own
 * cached prompt prefix in the worker.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ZagreusNeedleTools = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var noArgs = { type: 'object', properties: {}, required: [] };
  var onOff = { type: 'object', properties: { on: { type: 'boolean' } }, required: ['on'] };

  var system = 'You control the Zagreus assistant app. Use the appropriate available tool for the user\'s request.';

  var profiles = {
    display: {
      tools: [
        { name: 'set_dark_mode', description: 'Turn dark mode on or off', parameters: onOff }
      ],
      candidates: [
        { label: 'theme_dark', json: '{"name":"set_dark_mode","arguments":{"on":true}}' },
        { label: 'theme_light', json: '{"name":"set_dark_mode","arguments":{"on":false}}' }
      ]
    },
    voice: {
      tools: [
        { name: 'set_voice', description: 'Turn the spoken voice replies on or off', parameters: onOff },
        { name: 'stop_speaking', description: 'Stop speaking right now', parameters: noArgs },
        { name: 'set_language', description: 'Switch the spoken language to Greek or English',
          parameters: { type: 'object', properties: { lang: { type: 'string', enum: ['el', 'en'] } }, required: ['lang'] } }
      ],
      candidates: [
        { label: 'voice_on', json: '{"name":"set_voice","arguments":{"on":true}}' },
        { label: 'voice_off', json: '{"name":"set_voice","arguments":{"on":false}}' },
        { label: 'stop_speaking', json: '{"name":"stop_speaking","arguments":{}}' },
        { label: 'lang_el', json: '{"name":"set_language","arguments":{"lang":"el"}}' },
        { label: 'lang_en', json: '{"name":"set_language","arguments":{"lang":"en"}}' }
      ]
    },
    call: {
      tools: [
        { name: 'start_call', description: 'Start a voice call with the assistant', parameters: noArgs },
        { name: 'end_call', description: 'End the current voice call', parameters: noArgs }
      ],
      candidates: [
        { label: 'call_start', json: '{"name":"start_call","arguments":{}}' },
        { label: 'call_end', json: '{"name":"end_call","arguments":{}}' }
      ]
    }
  };

  return { system: system, profiles: profiles };
});
