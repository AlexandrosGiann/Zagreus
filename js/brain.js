// Zagreus' brain: decides what a message is about and produces the reply.
//
// 1. Exact parsers first (arithmetic, timers, clock, notes, weather) — they need slots anyway.
// 2. Otherwise the multilingual MiniLM classifies the message against anchor phrases in Greek
//    and English (cosine over L2-normalised embeddings, scored by the WebAssembly kernel).
// 3. App commands go to Needle 2 inside a small tool profile, with the MiniLM anchor as backup.
// 4. Knowledge questions go to Wikipedia (re-ranked with MiniLM), memory questions to the
//    semantic notes store, and free conversation to SmolLM2.
// Before MiniLM has loaded, keyword rules stand in for step 2.

import { models } from './models.js';
import { loadDsp } from './dsp.js';

// ---------------------------------------------------------------- language helpers

export const isGreek = (s) => /[Ͱ-Ͽἀ-῿]/.test(s || '');
const hasLatin = (s) => /[a-z]/i.test(s || '');
export function replyLang(text, uiLang) {
  if (isGreek(text)) return 'el';
  if (hasLatin(text)) return 'en';
  return uiLang;
}
// lower-case and strip Greek accents, so "Καιρός" ~ "καιρος"
export function fold(s) {
  return String(s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/ς/g, 'σ');
}
const pick = (a) => a[Math.floor(Math.random() * a.length)];

// `\b` only knows ASCII letters, so /\bσε\b/ never matches. W() rewrites a regex's \b into a
// Unicode-aware boundary.
const UB = '(?:(?<![\\p{L}\\p{N}])(?=[\\p{L}\\p{N}])|(?<=[\\p{L}\\p{N}])(?![\\p{L}\\p{N}]))';
function W(re) {
  return new RegExp(re.source.replace(/\\b/g, UB), re.flags.indexOf('u') >= 0 ? re.flags : re.flags + 'u');
}

// ---------------------------------------------------------------- intent anchors

const ANCHORS = {
  greeting: ['γεια σου', 'καλημέρα', 'καλησπέρα Ζαγρεύ', 'γεια χαρά', 'hello', 'hi there', 'good morning', 'hey Zagreus'],
  how_are_you: ['τι κάνεις;', 'πώς είσαι;', 'όλα καλά;', 'how are you?', 'how is it going?', 'are you okay?'],
  thanks: ['ευχαριστώ', 'σε ευχαριστώ πολύ', 'thank you', 'thanks a lot', 'thanks Zagreus'],
  bye: ['αντίο', 'τα λέμε', 'καληνύχτα', 'goodbye', 'see you later', 'good night'],
  identity: ['ποιος είσαι;', 'πώς σε λένε;', 'τι είσαι;', 'who are you?', 'what is your name?', 'what are you?', 'γιατί σε λένε Ζαγρεύς;', 'why are you called Zagreus?'],
  help: ['τι μπορείς να κάνεις;', 'βοήθεια', 'πώς σε χρησιμοποιώ;', 'what can you do?', 'help', 'how do I use you?'],
  joke: ['πες μου ένα ανέκδοτο', 'κάνε με να γελάσω', 'tell me a joke', 'make me laugh', 'say something funny'],
  time: ['τι ώρα είναι;', 'πες μου την ώρα', 'what time is it?', 'tell me the time'],
  date: ['τι μέρα είναι σήμερα;', 'τι ημερομηνία έχουμε;', 'what is the date today?', 'what day is it?'],
  recall: ['τι θυμάσαι;', 'τι σου είπα για', 'θυμάσαι τι σου είπα;', 'what do you remember?', 'what did I tell you about', 'do you remember my'],
  wiki: ['τι είναι η φωτοσύνθεση;', 'ποιος ήταν ο Μέγας Αλέξανδρος;', 'πες μου για την Ακρόπολη', 'εξήγησέ μου τη βαρύτητα', 'what is photosynthesis?', 'who was Alexander the Great?', 'tell me about the Acropolis', 'explain gravity'],
  weather: ['τι καιρό κάνει;', 'θα βρέξει αύριο;', 'καιρός στην Αθήνα', 'what is the weather like?', 'will it rain today?', 'weather in London'],
  image: ['τι βλέπεις;', 'κοίτα αυτό', 'άνοιξε την κάμερα', 'τι είναι αυτό στη φωτογραφία;', 'what do you see?', 'look at this', 'open the camera', 'describe this picture'],
  cmd_display: ['βάλε σκοτεινό θέμα', 'άναψε το dark mode', 'βάλε φωτεινό θέμα', 'κάνε την οθόνη ανοιχτόχρωμη', 'turn on dark mode', 'switch to light mode', 'make the screen dark', 'use the light theme'],
  cmd_voice: ['σταμάτα να μιλάς', 'σώπα', 'κλείσε τη φωνή', 'μίλα μου δυνατά', 'μίλα ελληνικά', 'μίλα αγγλικά', 'stop talking', 'be quiet', 'mute your voice', 'read your answers out loud', 'speak Greek', 'talk in English'],
  cmd_call: ['πάρε με τηλέφωνο', 'ξεκίνα κλήση', 'κλείσε την κλήση', 'να μιλήσουμε με φωνή', 'call me', 'start a call', 'hang up', 'end the call', 'let us talk by voice'],
  cmd_clear: ['καθάρισε τη συζήτηση', 'σβήσε τα μηνύματα', 'ξεκίνα από την αρχή', 'clear the chat', 'delete the conversation', 'start over'],
  chit: ['τι πιστεύεις για τη ζωή;', 'γράψε μου ένα ποίημα', 'δώσε μου μια ιδέα', 'what do you think about music?', 'write a short poem', 'give me an idea for dinner', 'can you help me write an email?', 'what should I do this weekend?']
};

const DOMAIN = { cmd_display: 'display', cmd_voice: 'voice', cmd_call: 'call' };

// Keyword fallback for when MiniLM is not loaded (and as a cheap pre-check).
const RULES = [
  ['cmd_display', /(dark mode|light mode|σκοτειν|φωτειν|θεμα|theme)/],
  ['cmd_call', W(/(κληση|τηλεφων|\bcall\b|hang up)/)],
  ['cmd_voice', /(σταματα να μιλα|σωπα|stop talking|be quiet|mute|φωνη|voice)/],
  ['cmd_clear', /(καθαρισε|σβησε τα|clear the chat|start over)/],
  ['image', /(καμερα|φωτογραφια|τι βλεπεισ|camera|picture|photo|what do you see)/],
  ['greeting', W(/^(γεια|καλημερα|καλησπερα|hello|hi|hey)\b/)],
  ['thanks', /(ευχαριστω|thank)/],
  ['bye', /(αντιο|τα λεμε|καληνυχτα|bye|good night)/],
  ['how_are_you', /(τι κανεισ|πωσ εισαι|how are you)/],
  ['identity', /(ποιοσ εισαι|πωσ σε λενε|who are you|your name)/],
  ['help', W(/(τι μπορεισ|βοηθεια|what can you do|\bhelp\b)/)],
  ['joke', /(ανεκδοτο|αστειο|joke|funny)/],
  ['recall', /(τι θυμασαι|θυμασαι|remember\?|do you remember|what did i tell)/],
  ['wiki', W(/^(τι ειναι|ποιοσ ηταν|ποια ηταν|ποιοσ ειναι|πεσ μου για|εξηγησε|what is|what are|who was|who is|tell me about|explain)\b/)]
];

// ---------------------------------------------------------------- canned replies

const R = {
  el: {
    greeting: ['Γεια σου! Είμαι ο Ζαγρεύς. Πώς μπορώ να βοηθήσω;', 'Καλώς ήρθες! Τι θα ήθελες να κάνουμε;'],
    how_are_you: ['Είμαι μια χαρά, τρέχω ολόκληρος μέσα στον browser σου! Εσύ;', 'Πολύ καλά, ευχαριστώ! Τα μοντέλα μου είναι ζεστά και έτοιμα.'],
    thanks: ['Παρακαλώ!', 'Χαρά μου!', 'Ό,τι χρειαστείς, εδώ είμαι.'],
    bye: ['Γεια σου! Τα λέμε σύντομα.', 'Καληνύχτα και καλή ξεκούραση!'],
    identity: ['Είμαι ο Ζαγρεύς, ένας βοηθός που τρέχει εξ ολοκλήρου στη συσκευή σου: ακούω με το Whistle, καταλαβαίνω με το MiniLM, εκτελώ εντολές με το Needle 2, βλέπω με το MobileCLIP, συζητώ με το SmolLM2 και μιλάω με το meSpeak. Το όνομά μου είναι από τον Ζαγρέα, τον ορφικό θεό που ξαναγεννήθηκε.'],
    help: ['Μπορώ να απαντώ σε ερωτήσεις γνώσεων από τη Wikipedia, να σου λέω τον καιρό, την ώρα και την ημερομηνία, να κάνω πράξεις, να βάζω χρονόμετρα, να θυμάμαι πράγματα που μου λες, να αναγνωρίζω εικόνες, να αλλάζω θέμα και φωνή, και να μιλάμε σε φωνητική κλήση. Δοκίμασε «θυμήσου ότι…», «χρονόμετρο 5 λεπτά» ή πάτα το κουμπί κλήσης.'],
    joke: ['Γιατί ο υπολογιστής πήγε στον γιατρό; Γιατί είχε ιό!', 'Τι λέει ένα μηδέν σε ένα οκτώ; Ωραία ζώνη!', 'Πώς λέγεται ο προγραμματιστής που φοβάται τα bugs; Εντομοφοβικός developer!'],
    noWiki: 'Δεν βρήκα κάτι σχετικό στη Wikipedia. Δοκίμασε να το διατυπώσεις αλλιώς.',
    wikiFail: 'Δεν μπόρεσα να συνδεθώ με τη Wikipedia αυτή τη στιγμή.',
    time: (t) => `Η ώρα είναι ${t}.`,
    date: (d) => `Σήμερα είναι ${d}.`,
    math: (e, v) => `${e} = ${v}`,
    mathFail: 'Δεν μπόρεσα να υπολογίσω αυτή την παράσταση.',
    timerSet: (h) => `Εντάξει, χρονόμετρο για ${h}. Θα σε ειδοποιήσω.`,
    timerDone: (h) => `Τέλος χρόνου! Πέρασαν ${h}.`,
    remembered: (n) => `Το κράτησα: «${n}».`,
    recallNone: 'Δεν θυμάμαι κάτι σχετικό. Πες μου «θυμήσου ότι…» για να κρατήσω κάτι.',
    recallEmpty: 'Δεν μου έχεις πει ακόμα κάτι να θυμάμαι.',
    recallList: 'Να τι θυμάμαι:',
    recallTop: 'Αυτό θυμάμαι σχετικά:',
    forgot: 'Ξέχασα όλες τις σημειώσεις.',
    weather: (p, t, d, w) => `Στην πόλη ${p} τώρα: ${t}°C, ${d}, άνεμος ${w} km/h.`,
    weatherAsk: 'Για ποια πόλη θέλεις τον καιρό; Π.χ. «καιρός στη Θεσσαλονίκη».',
    weatherFail: 'Δεν μπόρεσα να βρω τον καιρό αυτή τη στιγμή.',
    weatherNoCity: (c) => `Δεν βρήκα την πόλη «${c}».`,
    imageAsk: 'Στείλε μου μια φωτογραφία ή άνοιξε την κάμερα και θα σου πω τι βλέπω.',
    imageSee: (l) => `Βλέπω κάτι που μοιάζει με «${l}».`,
    imageMaybe: (l) => `Ίσως επίσης: ${l}.`,
    cmdDone: {
      theme_dark: 'Έβαλα το σκοτεινό θέμα.', theme_light: 'Έβαλα το φωτεινό θέμα.',
      voice_on: 'Άνοιξα τη φωνή μου.', voice_off: 'Έκλεισα τη φωνή μου. Θα απαντώ μόνο γραπτά.',
      stop_speaking: 'Εντάξει.', lang_el: 'Μιλάμε ελληνικά.', lang_en: 'Switched to English.',
      call_start: 'Ξεκινάω κλήση…', call_end: 'Κλείνω την κλήση.', clear_chat: 'Καθάρισα τη συζήτηση.', camera: 'Ανοίγω την κάμερα…'
    },
    chitGreek: 'Για ελεύθερη συζήτηση χρησιμοποιώ το SmolLM2, που μιλάει μόνο αγγλικά. Στα ελληνικά μπορώ να ψάξω στη Wikipedia, τον καιρό, να θυμάμαι σημειώσεις, να κάνω πράξεις και χρονόμετρα. Ρώτα με κάτι από αυτά ή γράψε μου στα αγγλικά!',
    modelLoading: (m) => `Φορτώνω το ${m} για πρώτη φορά, μια στιγμή…`,
    smolFail: 'Δεν μπόρεσα να φτιάξω απάντηση αυτή τη στιγμή.'
  },
  en: {
    greeting: ['Hello! I am Zagreus. How can I help?', 'Hi there! What shall we do?'],
    how_are_you: ['I am great — running entirely inside your browser! And you?', 'Very well, thanks! My models are warm and ready.'],
    thanks: ['You are welcome!', 'My pleasure!', 'Any time.'],
    bye: ['Goodbye! Talk soon.', 'Good night and rest well!'],
    identity: ['I am Zagreus, an assistant that runs entirely on your device: I listen with Whistle, understand with MiniLM, run commands with Needle 2, see with MobileCLIP, chat with SmolLM2 and speak with meSpeak. My name comes from Zagreus, the Orphic god who was born again.'],
    help: ['I can answer knowledge questions from Wikipedia, tell you the weather, time and date, do arithmetic, set timers, remember things you tell me, recognise images, change the theme and voice, and talk in a voice call. Try "remember that…", "timer 5 minutes" or press the call button.'],
    joke: ['Why do programmers prefer dark mode? Because light attracts bugs!', 'I would tell you a UDP joke, but you might not get it.', 'There are 10 kinds of people: those who understand binary and those who do not.'],
    noWiki: 'I found nothing relevant on Wikipedia. Try rephrasing.',
    wikiFail: 'I could not reach Wikipedia right now.',
    time: (t) => `It is ${t}.`,
    date: (d) => `Today is ${d}.`,
    math: (e, v) => `${e} = ${v}`,
    mathFail: 'I could not evaluate that expression.',
    timerSet: (h) => `Okay, timer set for ${h}. I will let you know.`,
    timerDone: (h) => `Time is up! ${h} have passed.`,
    remembered: (n) => `Got it: "${n}".`,
    recallNone: 'I do not remember anything related. Say "remember that…" to keep a note.',
    recallEmpty: 'You have not told me anything to remember yet.',
    recallList: 'Here is what I remember:',
    recallTop: 'This is what I remember about that:',
    forgot: 'I forgot all notes.',
    weather: (p, t, d, w) => `In ${p} right now: ${t}°C, ${d}, wind ${w} km/h.`,
    weatherAsk: 'Which city? For example "weather in Thessaloniki".',
    weatherFail: 'I could not get the weather right now.',
    weatherNoCity: (c) => `I could not find the city "${c}".`,
    imageAsk: 'Send me a photo or open the camera and I will tell you what I see.',
    imageSee: (l) => `I see something that looks like ${l}.`,
    imageMaybe: (l) => `Maybe also: ${l}.`,
    cmdDone: {
      theme_dark: 'Dark theme on.', theme_light: 'Light theme on.',
      voice_on: 'Voice replies are on.', voice_off: 'Voice replies are off. I will answer in text only.',
      stop_speaking: 'Okay.', lang_el: 'Μιλάμε ελληνικά.', lang_en: 'Switched to English.',
      call_start: 'Starting a call…', call_end: 'Ending the call.', clear_chat: 'Conversation cleared.', camera: 'Opening the camera…'
    },
    chitGreek: '',
    modelLoading: (m) => `Loading ${m} for the first time, one moment…`,
    smolFail: 'I could not come up with a reply right now.'
  }
};

// ---------------------------------------------------------------- notes (semantic memory)

const NOTES_KEY = 'zagreus_notes_v1';
function loadNotes() {
  try { return JSON.parse(localStorage.getItem(NOTES_KEY) || '[]'); } catch (e) { return []; }
}
function saveNotes(n) { try { localStorage.setItem(NOTES_KEY, JSON.stringify(n)); } catch (e) {} }
const toB64 = (f32) => { const u = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength); let s = ''; for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]); return btoa(s); };
const fromB64 = (b) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };

export function listNotes() { return loadNotes().map((n) => ({ text: n.text, ts: n.ts })); }
export function clearNotes() { saveNotes([]); }

// ---------------------------------------------------------------- arithmetic

const NUMWORDS = {
  'μηδεν': 0, 'ενα': 1, 'μια': 1, 'μισο': 0.5, 'δυο': 2, 'δυο.': 2, 'τρια': 3, 'τρεισ': 3, 'τεσσερα': 4, 'τεσσερισ': 4, 'πεντε': 5, 'εξι': 6, 'επτα': 7, 'εφτα': 7,
  'οκτω': 8, 'οχτω': 8, 'εννεα': 9, 'εννια': 9, 'δεκα': 10, 'εντεκα': 11, 'δωδεκα': 12, 'δεκαπεντε': 15, 'εικοσι': 20, 'τριαντα': 30, 'σαραντα': 40, 'πενηντα': 50, 'εξηντα': 60,
  'zero': 0, 'one': 1, 'a': 1, 'an': 1, 'half': 0.5, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
  'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60
};

function mathExpression(text) {
  let s = fold(text)
    .replace(/(ποσο|ποσα|κανει|κανουν|ειναι|υπολογισε|what is|what's|how much is|calculate|compute|=|\?|;)/g, ' ')
    .replace(/(\d),(\d)/g, '$1.$2')
    .replace(W(/\b(συν|plus)\b/g), '+').replace(W(/\b(πλην|μειον|minus)\b/g), '-')
    .replace(W(/\b(επι|φορεσ|times|multiplied by)\b/g), '*').replace(/(\d)\s*[x×]\s*(?=\d)/g, '$1*')
    .replace(W(/\b(δια|divided by|over)\b/g), '/').replace(/÷/g, '/')
    .replace(W(/\b(στη|στην|to the power of|power)\b/g), '^')
    .replace(/(τετραγωνικη ριζα|square root of|sqrt)/g, 'r')
    .replace(/(\d+(?:\.\d+)?)\s*(%|τοισ εκατο|percent)\s*(του|of)?/g, '($1/100)*');
  s = s.replace(/[a-zα-ω]+/g, (w) => (w in NUMWORDS && w !== 'a' && w !== 'an' ? String(NUMWORDS[w]) : (w === 'r' ? 'r' : ' ')));
  s = s.replace(/\s+/g, '');
  if (!/\d/.test(s) || !/[-+*/^r]/.test(s) || /[^-+*/^().\dr]/.test(s)) return null;
  return s;
}

// Shunting-yard evaluator over + - * / ^ ( ) and prefix r (square root). No eval.
function evaluate(expr) {
  const toks = expr.match(/\d+(?:\.\d+)?|[-+*/^()r]/g);
  if (!toks) return null;
  const out = [], ops = [], prec = { '+': 1, '-': 1, '*': 2, '/': 2, '^': 3, 'u': 4, 'r': 4 }, right = { '^': 1, 'u': 1, 'r': 1 };
  let prev = null;
  for (const t of toks) {
    if (/\d/.test(t)) out.push(parseFloat(t));
    else if (t === '(') ops.push(t);
    else if (t === ')') {
      while (ops.length && ops[ops.length - 1] !== '(') out.push(ops.pop());
      if (!ops.length) return null;
      ops.pop();
    } else {
      let op = t;
      if (op === '-' && (prev === null || /[-+*/^(ur]/.test(prev))) op = 'u';
      while (ops.length && ops[ops.length - 1] !== '(' &&
        (prec[ops[ops.length - 1]] > prec[op] || (prec[ops[ops.length - 1]] === prec[op] && !right[op]))) out.push(ops.pop());
      ops.push(op);
    }
    prev = t === '-' && (prev === null || /[-+*/^(ur]/.test(prev)) ? 'u' : t;
  }
  while (ops.length) { const o = ops.pop(); if (o === '(') return null; out.push(o); }
  const st = [];
  for (const x of out) {
    if (typeof x === 'number') { st.push(x); continue; }
    if (x === 'u' || x === 'r') { if (!st.length) return null; const a = st.pop(); st.push(x === 'u' ? -a : Math.sqrt(a)); continue; }
    if (st.length < 2) return null;
    const b = st.pop(), a = st.pop();
    st.push(x === '+' ? a + b : x === '-' ? a - b : x === '*' ? a * b : x === '/' ? a / b : Math.pow(a, b));
  }
  return st.length === 1 && isFinite(st[0]) ? st[0] : null;
}

const fmtNum = (v, lang) => (Math.abs(v - Math.round(v)) < 1e-9 ? Math.round(v) : +v.toPrecision(10)).toLocaleString(lang === 'el' ? 'el-GR' : 'en-US', { maximumFractionDigits: 8 });

// ---------------------------------------------------------------- timers

function parseDuration(text) {
  const f = fold(text);
  if (!/(χρονομετρ|υπενθυμ|ξυπνησε|timer|remind|alarm|countdown|αντιστροφ)/.test(f)) return null;
  const unit = { sec: 1, min: 60, hour: 3600 };
  const re = /(\d+(?:[.,]\d+)?|[a-zα-ω]+)\s*(δευτερολεπτ\w*|δευτ\w*|λεπτ\w*|ωρ\w*|seconds?|secs?|minutes?|mins?|hours?|hrs?)/g;
  let total = 0, m;
  while ((m = re.exec(f))) {
    let n = /\d/.test(m[1]) ? parseFloat(m[1].replace(',', '.')) : NUMWORDS[m[1]];
    if (n === undefined) continue;
    const u = m[2];
    const k = /^(δευτ|sec)/.test(u) ? unit.sec : /^(ωρ|hour|hr)/.test(u) ? unit.hour : unit.min;
    total += n * k;
  }
  if (!total) {
    if (/(μιση ωρα|half an hour)/.test(f)) total = 1800;
    else if (/(ενα τεταρτο|quarter of an hour)/.test(f)) total = 900;
  }
  if (!total) return null;
  // the reminder text is whatever follows the last duration word: "…10 λεπτά να πάρω τηλέφωνο"
  const units = /(δευτερόλεπτ\S*|δευτ\S*|λεπτ\S*|ώρ\S*|ωρ\S*|seconds?|secs?|minutes?|mins?|hours?|hrs?)/gi;
  let last = null, u;
  while ((u = units.exec(text))) last = u;
  const tail = last ? text.slice(last.index + last[0].length) : '';
  const label = tail.replace(/^\s*(για να|να|to|that)\s+/i, '').replace(/[.!;?]+$/, '').trim();
  return { seconds: Math.round(total), label };
}

export function humanDuration(sec, lang) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60, parts = [];
  const el = lang === 'el';
  if (h) parts.push(h + (el ? (h === 1 ? ' ώρα' : ' ώρες') : (h === 1 ? ' hour' : ' hours')));
  if (m) parts.push(m + (el ? (m === 1 ? ' λεπτό' : ' λεπτά') : (m === 1 ? ' minute' : ' minutes')));
  if (s) parts.push(s + (el ? (s === 1 ? ' δευτερόλεπτο' : ' δευτερόλεπτα') : (s === 1 ? ' second' : ' seconds')));
  return parts.join(el ? ' και ' : ' and ');
}

// ---------------------------------------------------------------- weather (Open-Meteo, no key)

const WMO = {
  el: { 0: 'καθαρός ουρανός', 1: 'κυρίως αίθριος', 2: 'λίγα σύννεφα', 3: 'συννεφιά', 45: 'ομίχλη', 48: 'παγωμένη ομίχλη', 51: 'ψιλόβροχο', 53: 'ψιλόβροχο', 55: 'πυκνό ψιλόβροχο', 61: 'ασθενής βροχή', 63: 'βροχή', 65: 'δυνατή βροχή', 71: 'ασθενές χιόνι', 73: 'χιόνι', 75: 'πυκνό χιόνι', 80: 'μπόρες', 81: 'μπόρες', 82: 'ισχυρές μπόρες', 95: 'καταιγίδα', 96: 'καταιγίδα με χαλάζι', 99: 'καταιγίδα με χαλάζι' },
  en: { 0: 'clear sky', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast', 45: 'fog', 48: 'rime fog', 51: 'light drizzle', 53: 'drizzle', 55: 'dense drizzle', 61: 'light rain', 63: 'rain', 65: 'heavy rain', 71: 'light snow', 73: 'snow', 75: 'heavy snow', 80: 'showers', 81: 'showers', 82: 'violent showers', 95: 'thunderstorm', 96: 'thunderstorm with hail', 99: 'thunderstorm with hail' }
};

function weatherCity(text) {
  const m = W(/(?:\bστην?\b|\bστο\b|\bσε\b|\bγια\b|\bin\b|\bfor\b|\bat\b)\s+([^\s?.,!;][^?.,!;]*)/i).exec(text);
  if (!m) return null;
  return m[1].replace(W(/\b(σήμερα|αύριο|τώρα|today|tomorrow|now|please|παρακαλώ)\b/gi), '').trim() || null;
}

async function weather(text, lang) {
  const r = R[lang];
  let city = weatherCity(text) || (localStorage.getItem('zagreus_city') || '');
  if (!city) return { text: r.weatherAsk, source: 'Open-Meteo' };
  try {
    const g = await fetchJSON(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=${lang}&format=json`);
    const p = g && g.results && g.results[0];
    if (!p) return { text: r.weatherNoCity(city), source: 'Open-Meteo' };
    localStorage.setItem('zagreus_city', city);
    const w = await fetchJSON(`https://api.open-meteo.com/v1/forecast?latitude=${p.latitude}&longitude=${p.longitude}&current=temperature_2m,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&forecast_days=2&timezone=auto`);
    const c = w.current;
    const desc = WMO[lang][c.weather_code] || WMO[lang][Math.floor(c.weather_code / 10) * 10] || '';
    const temp = Math.round(c.temperature_2m), wind = Math.round(c.wind_speed_10m);
    const d = w.daily;
    return {
      text: r.weather(p.name, temp, desc, wind),
      cards: [{ type: 'weather', place: `${p.name}${p.country ? ', ' + p.country : ''}`, temp, desc, wind,
        days: d.time.map((t, i) => ({ date: t, max: Math.round(d.temperature_2m_max[i]), min: Math.round(d.temperature_2m_min[i]), rain: d.precipitation_probability_max[i] })) }],
      source: 'Open-Meteo'
    };
  } catch (e) {
    return { text: r.weatherFail, source: 'Open-Meteo' };
  }
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.json();
}

// ---------------------------------------------------------------- Wikipedia

const WIKI_STRIP = /^(τι είναι|τι ήταν|τι σημαίνει|ποιος ήταν|ποια ήταν|ποιο ήταν|ποιος είναι|ποια είναι|πες μου για|πες μου|μίλησέ μου για|εξήγησέ μου|εξήγησε|ψάξε|what is|what are|what was|who is|who was|who were|tell me about|explain|define|search for|look up)\s+(το |τη |την |τον |ο |η |οι |τα |τους |τις |the |a |an )?/i;

function wikiTopic(text) {
  return String(text).trim().replace(/[?;!.]+$/, '').replace(WIKI_STRIP, '').trim();
}

async function wikipedia(text, lang, embedOk) {
  const r = R[lang], topic = wikiTopic(text) || text;
  try {
    const q = await fetchJSON(`https://${lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(topic)}&srlimit=5&format=json&origin=*`);
    let hits = (q.query && q.query.search) || [];
    if (!hits.length) return { text: r.noWiki, source: 'Wikipedia' };
    // re-rank the top titles semantically against the question (titles only: each text costs a
    // MiniLM run), unless the first hit already is the topic itself
    if (embedOk && hits.length > 1 && fold(hits[0].title) !== fold(topic)) {
      try {
        hits = hits.slice(0, 3);
        const docs = hits.map((h) => h.title);
        const [qv, ...dv] = await models.minilm.embed([topic, ...docs]);
        const dsp = await loadDsp();
        const mat = new Float32Array(dv.length * qv.length);
        dv.forEach((v, i) => mat.set(v, i * qv.length));
        const sc = dsp.scores(qv, mat, dv.length, qv.length, 'wiki' + Math.random());
        // the search engine's order is a strong prior; an exact title match wins outright and
        // qualified titles like "Ακρόπολη (συνοικία)" only win on clear semantic evidence
        const ft = fold(topic);
        hits = hits.map((h, i) => {
          let s = sc[i] + 0.12 * (1 - i / hits.length);
          if (fold(h.title) === ft) s += 0.3;
          if (/\(/.test(h.title) && !/\(/.test(topic)) s -= 0.1;
          return { h, s };
        }).sort((a, b) => b.s - a.s).map((x) => x.h);
      } catch (e) { /* keep search order */ }
    }
    let s = null;
    for (const h of hits.slice(0, 3)) {
      s = await fetchJSON(`https://${lang}.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(h.title.replace(/ /g, '_'))}`);
      if (s.type !== 'disambiguation') break;
    }
    const extract = (s.extract || '').trim();
    if (!extract) return { text: r.noWiki, source: 'Wikipedia' };
    // the bubble (and the voice) gets the first two sentences, the card the rest of the summary
    const re = /[^.!;]+[.!;]+/g;
    let end = 0;
    for (let i = 0; i < 2 && re.exec(extract); i++) end = re.lastIndex;
    if (!end) end = extract.length;
    const short = extract.slice(0, end).replace(/\s+/g, ' ').trim();
    const more = extract.slice(end).trim();
    return {
      text: short,
      cards: [{ type: 'wiki', title: s.title, extract: more, url: s.content_urls && s.content_urls.desktop && s.content_urls.desktop.page, thumb: s.thumbnail && s.thumbnail.source }],
      source: 'Wikipedia'
    };
  } catch (e) {
    return { text: r.wikiFail, source: 'Wikipedia' };
  }
}

// ---------------------------------------------------------------- the brain

export class Brain {
  constructor(host) {
    // host: { uiLang(), settings(), notify(msg) }
    this.host = host;
    this.anchorVecs = null;   // Float32Array rows*dim
    this.anchorMeta = [];     // [{intent, text}]
    this.history = [];        // SmolLM conversation, [{role, content}]
    this._anchorsPromise = null;
  }

  // Embeds the anchor phrases once MiniLM is ready (cached in localStorage by content hash).
  prepareAnchors() {
    if (this._anchorsPromise) return this._anchorsPromise;
    this._anchorsPromise = (async () => {
      const meta = [];
      for (const k of Object.keys(ANCHORS)) for (const t of ANCHORS[k]) meta.push({ intent: k, text: t });
      const key = 'zagreus_anchors_' + hash(meta.map((m) => m.intent + ':' + m.text).join('|'));
      let vecs = null;
      try {
        const c = localStorage.getItem(key);
        if (c) { vecs = fromB64(c); if (vecs.length !== meta.length * 384) vecs = null; }
      } catch (e) { vecs = null; }
      if (!vecs) {
        const list = await models.minilm.embed(meta.map((m) => m.text));
        vecs = new Float32Array(list.length * 384);
        list.forEach((v, i) => vecs.set(v, i * 384));
        try {
          Object.keys(localStorage).filter((k) => k.startsWith('zagreus_anchors_')).forEach((k) => localStorage.removeItem(k));
          localStorage.setItem(key, toB64(vecs));
        } catch (e) {}
      }
      this.anchorVecs = vecs; this.anchorMeta = meta;
      return true;
    })().catch((e) => { this._anchorsPromise = null; throw e; });
    return this._anchorsPromise;
  }

  // -> { intent, score, via: 'minilm'|'rules', top: [{intent, score}] }
  async classify(text) {
    const f = fold(text);
    if (models.minilm.ready) {
      try {
        await this.prepareAnchors();
        const [q] = await models.minilm.embed([text]);
        const dsp = await loadDsp();
        const sc = dsp.scores(q, this.anchorVecs, this.anchorMeta.length, 384, 'anchors');
        const best = {};
        for (let i = 0; i < sc.length; i++) {
          const it = this.anchorMeta[i].intent;
          if (!(it in best) || sc[i] > best[it]) best[it] = sc[i];
        }
        const top = Object.keys(best).map((k) => ({ intent: k, score: best[k] })).sort((a, b) => b.score - a.score);
        this.lastQueryVec = q;
        // a matching keyword rule breaks near-ties
        const rule = RULES.find(([, re]) => re.test(f));
        if (rule && top[0].intent !== rule[0]) {
          const r = top.find((t) => t.intent === rule[0]);
          if (r && top[0].score - r.score < 0.06) return { intent: r.intent, score: r.score, via: 'minilm+rules', top };
        }
        return { intent: top[0].score >= 0.5 ? top[0].intent : 'chit', score: top[0].score, via: 'minilm', top };
      } catch (e) {
        console.warn('MiniLM classify failed', e);
      }
    }
    const rule = RULES.find(([, re]) => re.test(f));
    return { intent: rule ? rule[0] : (/[?;]\s*$/.test(text) ? 'wiki' : 'chit'), score: rule ? 1 : 0, via: 'rules', top: [] };
  }

  // Main entry. opts: { onPartial(text), onStatus(text), call: bool }
  // Resolves with { text, lang, cards?, action?, source, intent, ms, speak? }
  async reply(text, opts = {}) {
    const t0 = performance.now();
    const lang = replyLang(text, this.host.uiLang());
    const out = await this._reply(text.trim(), lang, opts);
    out.lang = out.lang || lang;
    out.ms = Math.round(performance.now() - t0);
    return out;
  }

  async _reply(text, lang, opts) {
    const r = R[lang], f = fold(text);
    const status = opts.onStatus || (() => {});

    // ---- exact parsers
    if (W(/^(θυμησου|θυμισου|να θυμασαι|κρατα σημειωση|σημειωσε|remember that|remember|note that|make a note)\b/).test(f)) {
      const note = text.replace(/^\s*(θυμήσου|θυμίσου|να θυμάσαι|κράτα σημείωση|σημείωσε|remember that|remember|note that|make a note)\s*(ότι|πως|that|:)?\s*/i, '').trim();
      if (note) return this.remember(note, lang);
    }
    if (/^(ξεχασε τα παντα|σβησε τισ σημειωσεισ|forget everything|delete all notes)/.test(f)) {
      clearNotes();
      return { text: r.forgot, intent: 'forget', source: 'notes' };
    }
    const dur = parseDuration(text);
    if (dur) {
      const h = humanDuration(dur.seconds, lang);
      return { text: r.timerSet(h), intent: 'timer', source: 'timer', action: { name: 'set_timer', args: { seconds: dur.seconds, label: dur.label, done: r.timerDone(h) } } };
    }
    const expr = mathExpression(text);
    if (expr && /\d/.test(text)) {
      const v = evaluate(expr);
      if (v !== null) return { text: r.math(expr.replace(/\*/g, '×').replace(/\//g, '÷').replace(/r/g, '√'), fmtNum(v, lang)), intent: 'math', source: 'calculator' };
    }
    if (/(καιροσ|καιρο|θερμοκρασια|βρεξει|βροχη|weather|temperature|rain|forecast)/.test(f) && !/(τι ειναι ο καιροσ|what is weather$)/.test(f)) {
      status('weather');
      return Object.assign(await weather(text, lang), { intent: 'weather' });
    }

    // ---- semantic intent
    const cls = await this.classify(text);
    let intent = cls.intent;
    // a question that closely matches a saved note is a memory question ("Πού είναι το κλειδί;")
    if (['chit', 'wiki', 'recall'].includes(intent) && this.lastQueryVec) {
      const hit = await this.noteMatch(this.lastQueryVec);
      if (hit && hit.s > 0.55) intent = 'recall';
    }
    const base = { intent, via: cls.via, score: cls.score };

    switch (intent) {
      case 'greeting': case 'how_are_you': case 'thanks': case 'bye': case 'identity': case 'help': case 'joke':
        return Object.assign(base, { text: pick(r[intent]), source: 'Zagreus' });
      case 'time': {
        const t = new Date().toLocaleTimeString(lang === 'el' ? 'el-GR' : 'en-US', { hour: '2-digit', minute: '2-digit' });
        return Object.assign(base, { text: r.time(t.replace(/\.$/, '')), source: 'clock' });
      }
      case 'date': {
        const d = new Date().toLocaleDateString(lang === 'el' ? 'el-GR' : 'en-US', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
        return Object.assign(base, { text: r.date(d), source: 'clock' });
      }
      case 'recall': return Object.assign(base, await this.recall(text, lang));
      case 'weather': status('weather'); return Object.assign(base, await weather(text, lang));
      case 'wiki':
        status('wiki');
        return Object.assign(base, await wikipedia(text, lang, models.minilm.ready));
      case 'image':
        return Object.assign(base, { text: r.cmdDone.camera, action: { name: 'open_camera' }, source: 'Zagreus' });
      case 'cmd_clear':
        return Object.assign(base, { text: r.cmdDone.clear_chat, action: { name: 'clear_chat' }, source: 'MiniLM' });
      case 'cmd_display': case 'cmd_voice': case 'cmd_call':
        return Object.assign(base, await this.command(intent, text, lang, cls, status));
      default:
        return Object.assign(base, await this.chat(text, lang, opts));
    }
  }

  // App commands: Needle 2 decides the exact call inside the domain; MiniLM's nearest anchor is
  // the fallback (and the only path for Greek, which Needle was not trained on).
  async command(intent, text, lang, cls, status) {
    const r = R[lang];
    const profile = DOMAIN[intent];
    let label = null, via = 'rules', needleMs = null, reasoning = null;
    if (!isGreek(text) && this.host.settings().useNeedle) {
      try {
        if (!models.needle.ready) status('needle');
        const o = await models.needle.route(profile, text);
        needleMs = o.ms; reasoning = o.reasoning;
        if (o.label) { label = o.label; via = 'Needle 2'; }
      } catch (e) { console.warn('Needle failed', e); }
    }
    if (!label) { label = anchorLabel(profile, text); via = cls.via === 'rules' ? 'rules' : 'MiniLM'; }
    const action = ACTIONS[label];
    return { text: r.cmdDone[label] || '', action, source: via, needleMs, reasoning, label };
  }

  async remember(note, lang) {
    const notes = loadNotes();
    let vec = null;
    if (models.minilm.ready) { try { [vec] = await models.minilm.embed([note]); } catch (e) {} }
    notes.push({ text: note, ts: Date.now(), v: vec ? toB64(vec) : null });
    saveNotes(notes);
    return { text: R[lang].remembered(note), intent: 'remember', source: 'notes' };
  }

  // Best note for an embedded query: { n, s } or null.
  async noteMatch(q) {
    const notes = loadNotes().filter((n) => n.v);
    if (!notes.length) return null;
    const mat = new Float32Array(notes.length * 384);
    notes.forEach((n, i) => mat.set(fromB64(n.v), i * 384));
    const dsp = await loadDsp();
    const sc = dsp.scores(q, mat, notes.length, 384, 'notes' + notes.length + notes[notes.length - 1].ts);
    let best = null;
    for (let i = 0; i < sc.length; i++) if (!best || sc[i] > best.s) best = { n: notes[i], s: sc[i] };
    return best;
  }

  async recall(text, lang) {
    const r = R[lang], notes = loadNotes();
    if (!notes.length) return { text: r.recallEmpty, source: 'notes' };
    const f = fold(text);
    const general = /^(τι θυμασαι|what do you remember)\s*[?;]?$/.test(f.trim());
    if (general || !models.minilm.ready) {
      const last = notes.slice(-5).map((n) => '• ' + n.text);
      return { text: r.recallList + '\n' + last.join('\n'), source: 'notes' };
    }
    // embed any notes saved before MiniLM was ready
    const missing = notes.filter((n) => !n.v);
    if (missing.length) {
      const vs = await models.minilm.embed(missing.map((n) => n.text));
      missing.forEach((n, i) => { n.v = toB64(vs[i]); });
      saveNotes(notes);
    }
    const q = this.lastQueryVec || (await models.minilm.embed([text]))[0];
    const mat = new Float32Array(notes.length * 384);
    notes.forEach((n, i) => mat.set(fromB64(n.v), i * 384));
    const dsp = await loadDsp();
    const sc = dsp.scores(q, mat, notes.length, 384, 'notes' + notes.length + notes[notes.length - 1].ts);
    const ranked = notes.map((n, i) => ({ n, s: sc[i] })).sort((a, b) => b.s - a.s).filter((x) => x.s > 0.35).slice(0, 3);
    if (!ranked.length) return { text: r.recallNone, source: 'notes' };
    return { text: r.recallTop + '\n' + ranked.map((x) => '• ' + x.n.text).join('\n'), source: 'notes · ' + dspKindLabel(dsp) };
  }

  // Free conversation with SmolLM2 (English); Greek gets Wikipedia when it looks like a question.
  async chat(text, lang, opts) {
    const r = R[lang];
    if (lang === 'el') {
      if (/[?;]\s*$/.test(text) || W(/^(τι|ποιοσ|ποια|ποιο|πωσ|γιατι|ποτε|που)\b/).test(fold(text))) {
        const w = await wikipedia(text, lang, models.minilm.ready);
        if (w.cards) return w;
      }
      return { text: r.chitGreek, source: 'Zagreus' };
    }
    if (!models.smollm.ready && opts.onStatus) opts.onStatus('smollm');
    const sys = { role: 'system', content: 'You are Zagreus, a friendly and helpful voice assistant that runs on the user\'s device. Answer briefly, in one to three sentences.' };
    this.history.push({ role: 'user', content: text });
    this.history = this.history.slice(-5);
    try {
      // in a call long answers are tiring to listen to: cap them (the last sentence may be cut)
      const res = await models.smollm.generate([sys].concat(this.history), (t) => opts.onPartial && opts.onPartial(t), opts.call ? 60 : 96);
      let reply = (res.text || '').trim() || r.smolFail;
      // drop a sentence cut off by the token limit, if there is a complete one before it
      if (res.tokens >= (opts.call ? 60 : 96) && !/[.!?]["')\]]?$/.test(reply)) {
        const cut = reply.search(/[.!?]["')\]]?\s[^.!?]*$/);
        if (cut > 20) reply = reply.slice(0, cut + 1);
      }
      this.history.push({ role: 'assistant', content: reply });
      this.history = this.history.slice(-5);
      const tps = res.tokens && res.ms > res.prefillMs ? (res.tokens / ((res.ms - res.prefillMs) / 1000)) : 0;
      return { text: reply, source: 'SmolLM2', tokens: res.tokens, tps };
    } catch (e) {
      return { text: r.smolFail, source: 'SmolLM2' };
    }
  }

  resetConversation() { this.history = []; }
}

const ACTIONS = {
  theme_dark: { name: 'set_theme', args: { mode: 'dark' } },
  theme_light: { name: 'set_theme', args: { mode: 'light' } },
  voice_on: { name: 'set_voice', args: { on: true } },
  voice_off: { name: 'set_voice', args: { on: false } },
  stop_speaking: { name: 'stop_speaking' },
  lang_el: { name: 'set_language', args: { lang: 'el' } },
  lang_en: { name: 'set_language', args: { lang: 'en' } },
  call_start: { name: 'start_call' },
  call_end: { name: 'end_call' }
};

// Rule-based argument choice inside a domain (used for Greek and as Needle's fallback).
function anchorLabel(profile, text) {
  const f = fold(text);
  if (profile === 'display') return /(φωτειν|ανοιχτ|ασπρ|λευκ|light|white|bright mode)/.test(f) && !/(πολυ φωτεινο|too bright)/.test(f) ? 'theme_light' : 'theme_dark';
  if (profile === 'call') return /(κλεισε|τερματισε|τελοσ|hang|end|stop|bye)/.test(f) ? 'call_end' : 'call_start';
  if (/(ελληνικ|greek)/.test(f)) return 'lang_el';
  if (/(αγγλικ|english)/.test(f)) return 'lang_en';
  if (/(σταματα|σωπα|σιωπη|stop|quiet|shut up|enough)/.test(f)) return 'stop_speaking';
  if (/(κλεισε|χωρισ φωνη|βουβ|mute|off|silent|no voice)/.test(f)) return 'voice_off';
  return 'voice_on';
}

function dspKindLabel(d) { return d.kind === 'wasm-simd' ? 'WASM SIMD' : d.kind === 'wasm' ? 'WASM' : 'JS'; }

function hash(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16);
}

export { R as REPLIES, evaluate, mathExpression, parseDuration, wikiTopic, anchorLabel };
