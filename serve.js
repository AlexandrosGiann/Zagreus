#!/usr/bin/env node
/*
 * Static file server for Zagreus (adapted from Cactus_Whistle-PureJS). No dependencies beyond Node itself.
 *
 *   node serve.js [port]          this computer only:  http://localhost:8780/
 *   node serve.js --lan [port]    also other devices on the network, over HTTP and HTTPS
 *
 * --lan creates a small local certificate authority in .cert/ (once) and a server
 * certificate for this machine's current addresses, so a phone can open the page
 * over https:// and use its microphone. The server only hands out files; all
 * inference still runs in the browser.
 */
'use strict';

var http = require('http'), https = require('https'), fs = require('fs'), path = require('path');
var os = require('os'), crypto = require('crypto');

var root = __dirname, certDir = path.join(root, '.cert');
var args = process.argv.slice(2), lan = args.indexOf('--lan') >= 0;
var port = +args.filter(function (a) { return /^\d+$/.test(a); })[0] || +process.env.PORT || 8780;
var httpsPort = +process.env.HTTPS_PORT || 8443;
var types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wav': 'audio/wav',
  '.cact': 'application/octet-stream', '.amr': 'audio/amr', '.md': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.txt': 'text/plain; charset=utf-8', '.bin': 'application/octet-stream', '.f32': 'application/octet-stream'
};

// ------------------------------------------------------------ certificates

// Minimal DER writer: just enough ASN.1 for an X.509 v3 certificate.
function der(tag, body) {
  var n = body.length, head;
  if (n < 128) head = [tag, n];
  else if (n < 256) head = [tag, 0x81, n];
  else head = [tag, 0x82, n >> 8, n & 255];
  return Buffer.concat([Buffer.from(head), body]);
}
function seq() { return der(0x30, Buffer.concat([].slice.call(arguments))); }
function oid(dotted) {
  var p = dotted.split('.').map(Number), out = [p[0] * 40 + p[1]];
  for (var i = 2; i < p.length; i++) {
    var v = p[i], chunk = [v & 127];
    while ((v = Math.floor(v / 128)) > 0) chunk.unshift((v & 127) | 128);
    out = out.concat(chunk);
  }
  return der(0x06, Buffer.from(out));
}
function integer(buf) {
  if (buf[0] & 0x80) buf = Buffer.concat([Buffer.from([0]), buf]);
  return der(0x02, buf);
}
function utcTime(d) {
  return der(0x17, Buffer.from(d.toISOString().replace(/[-:T]/g, '').slice(2, 14) + 'Z'));
}
function name(cn) { return seq(der(0x31, seq(oid('2.5.4.3'), der(0x0c, Buffer.from(cn, 'utf8'))))); }
function extension(id, critical, value) {
  return seq.apply(null, [oid(id)].concat(critical ? [der(0x01, Buffer.from([0xff]))] : [], [der(0x04, value)]));
}
var SHA256_RSA = seq(oid('1.2.840.113549.1.1.11'), der(0x05, Buffer.alloc(0)));

// opts: { subject, issuer, days, publicKey, signKey, ca, hosts }
function certificate(opts) {
  var now = Date.now(), exts;
  if (opts.ca) {
    exts = [
      extension('2.5.29.19', true, seq(der(0x01, Buffer.from([0xff])))),              // CA: true
      extension('2.5.29.15', true, der(0x03, Buffer.from([1, 0x06])))                 // certSign, cRLSign
    ];
  } else {
    var names = opts.hosts.map(function (h) {
      return /^\d+\.\d+\.\d+\.\d+$/.test(h) ? der(0x87, Buffer.from(h.split('.').map(Number))) : der(0x82, Buffer.from(h));
    });
    exts = [
      extension('2.5.29.19', true, seq()),
      extension('2.5.29.15', true, der(0x03, Buffer.from([5, 0xa0]))),                // signature, key encipherment
      extension('2.5.29.37', false, seq(oid('1.3.6.1.5.5.7.3.1'))),                   // serverAuth
      extension('2.5.29.17', false, seq.apply(null, names))                           // subjectAltName
    ];
  }
  var serial = crypto.randomBytes(12);
  serial[0] &= 0x7f; serial[0] |= 0x40;
  var tbs = seq(
    der(0xa0, der(0x02, Buffer.from([2]))),
    integer(serial), SHA256_RSA, name(opts.issuer),
    seq(utcTime(new Date(now - 86400000)), utcTime(new Date(now + opts.days * 86400000))),
    name(opts.subject),
    opts.publicKey.export({ type: 'spki', format: 'der' }),
    der(0xa3, seq.apply(null, exts))
  );
  var sig = crypto.sign('sha256', tbs, opts.signKey);
  return seq(tbs, SHA256_RSA, der(0x03, Buffer.concat([Buffer.from([0]), sig])));
}

function pem(derBuf) {
  return '-----BEGIN CERTIFICATE-----\n' + derBuf.toString('base64').replace(/(.{64})/g, '$1\n').trim() +
    '\n-----END CERTIFICATE-----\n';
}

function lanAddresses() {
  var out = [], ifs = os.networkInterfaces();
  Object.keys(ifs).forEach(function (k) {
    ifs[k].forEach(function (a) {
      if ((a.family === 'IPv4' || a.family === 4) && !a.internal && a.address.indexOf('169.254.') !== 0) out.push(a.address);
    });
  });
  return out;
}

// Returns { key, cert } for HTTPS, creating or refreshing the files in .cert/ as needed.
function ensureCertificates(hosts) {
  if (!fs.existsSync(certDir)) fs.mkdirSync(certDir);
  var caKeyFile = path.join(certDir, 'ca.key.pem'), caFile = path.join(certDir, 'ca.crt');
  var keyFile = path.join(certDir, 'server.key.pem'), crtFile = path.join(certDir, 'server.crt.pem');
  var hostsFile = path.join(certDir, 'server.hosts.json'), caKey, created = false;
  if (fs.existsSync(caKeyFile) && fs.existsSync(caFile)) {
    caKey = crypto.createPrivateKey(fs.readFileSync(caKeyFile));
  } else {
    var ca = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
    caKey = ca.privateKey;
    fs.writeFileSync(caKeyFile, caKey.export({ type: 'pkcs8', format: 'pem' }), { mode: 0o600 });
    fs.writeFileSync(caFile, certificate({
      subject: 'Zagreus local CA (' + os.hostname() + ')', issuer: 'Zagreus local CA (' + os.hostname() + ')',
      days: 3650, publicKey: ca.publicKey, signKey: caKey, ca: true
    }));
    created = true;
  }
  var want = JSON.stringify(hosts.slice().sort()), have = null;
  try { have = JSON.parse(fs.readFileSync(hostsFile, 'utf8')); } catch (e) {}
  var fresh = have && have.hosts === want && have.expires > Date.now() + 7 * 86400000 &&
    fs.existsSync(keyFile) && fs.existsSync(crtFile) && !created;
  if (!fresh) {
    var srv = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 }), days = 397;
    var caName = new crypto.X509Certificate(fs.readFileSync(caFile)).subject.replace(/^CN=/, '');
    fs.writeFileSync(keyFile, srv.privateKey.export({ type: 'pkcs8', format: 'pem' }), { mode: 0o600 });
    fs.writeFileSync(crtFile, pem(certificate({
      subject: 'Zagreus', issuer: caName, days: days, publicKey: srv.publicKey, signKey: caKey, hosts: hosts
    })));
    fs.writeFileSync(hostsFile, JSON.stringify({ hosts: want, expires: Date.now() + days * 86400000 }));
  }
  return { key: fs.readFileSync(keyFile), cert: fs.readFileSync(crtFile), caCreated: created };
}

// ------------------------------------------------------------------ server

function handler(req, res) {
  var rel;
  try { rel = decodeURIComponent(req.url.split('?')[0]); } catch (e) { res.writeHead(400); res.end(); return; }
  if (rel === '/') rel = '/index.html';
  if (rel === '/ca.crt' && lan) {
    // the public half of the local authority, for installing on a phone
    fs.readFile(path.join(certDir, 'ca.crt'), function (err, data) {
      if (err) { res.writeHead(404); res.end('not found'); return; }
      res.writeHead(200, { 'Content-Type': 'application/x-x509-ca-cert', 'Content-Length': data.length,
        'Content-Disposition': 'attachment; filename="zagreus-ca.crt"' });
      res.end(data);
    });
    return;
  }
  var file = path.join(root, path.normalize(rel));
  // nothing outside the folder, no dot-files or dot-folders (.cert holds private keys),
  // and none of the development folders
  var relPath = path.relative(root, file);
  if (file.indexOf(root + path.sep) !== 0 || /(^|[\\/])\./.test(relPath) || /^(_repos|wasm-src|tools)([\\/]|$)/.test(relPath)) {
    res.writeHead(403); res.end('forbidden'); return;
  }
  fs.stat(file, function (err, st) {
    if (err || !st.isFile()) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, {
      'Content-Type': types[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Content-Length': st.size, 'Cache-Control': 'no-cache'
    });
    if (req.method === 'HEAD') { res.end(); return; }
    fs.createReadStream(file).pipe(res);
  });
}

function checkFiles() {
  ['index.html', 'js/app.js', 'wasm/zagreus-simd.wasm', 'models/whistle/whistle.cact', 'models/needle/needle2.cact',
    'models/smollm/model/manifest.json', 'models/minilm/model/manifest.json', 'models/mobileclip/compiled-fp32-v1/plan.json',
    'vendor/mespeak/mespeak_config.json'].forEach(function (f) {
    if (!fs.existsSync(path.join(root, f))) console.log('WARNING: ' + f + ' is missing from ' + root);
  });
}

function fail(what) {
  return function (e) {
    console.error(what + ' could not start: ' + (e.code === 'EADDRINUSE' ? 'the port is already in use' : e.message));
    process.exit(1);
  };
}

checkFiles();
if (!lan) {
  http.createServer(handler).on('error', fail('The server')).listen(port, '127.0.0.1', function () {
    console.log('Zagreus: http://localhost:' + port + '/');
    console.log('For a phone on the same Wi-Fi, run: node serve.js --lan');
  });
} else {
  var ips = lanAddresses(), tls = ensureCertificates(['localhost', '127.0.0.1'].concat(ips));
  http.createServer(handler).on('error', fail('The HTTP server')).listen(port, '0.0.0.0', function () {
    https.createServer({ key: tls.key, cert: tls.cert }, handler).on('error', fail('The HTTPS server'))
      .listen(httpsPort, '0.0.0.0', function () {
        console.log('Zagreus, reachable from this network.\n');
        if (!ips.length) console.log('No network address found: connect this computer to the Wi-Fi and start again.\n');
        ips.forEach(function (ip) {
          console.log('  Page (microphone works):  https://' + ip + ':' + httpsPort + '/');
          console.log('  Certificate for the phone: http://' + ip + ':' + port + '/ca.crt');
          console.log('  Plain HTTP (no microphone): http://' + ip + ':' + port + '/\n');
        });
        console.log('On this computer: http://localhost:' + port + '/');
        console.log(tls.caCreated ? 'A new local certificate authority was created in .cert/ : install ca.crt on the phone (see README).'
          : 'Using the certificate authority in .cert/ (already installed phones keep working).');
        console.log('If the phone cannot connect, allow Node through this computer\'s firewall for private networks.');
      });
  });
}
