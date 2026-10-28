(function () {
"use strict";
var modules = {};
var deps = {};
modules[0] = function (module, exports, require) {
"use strict";

var FP32 = require("./mobileclip-fp32-executor-opt-v2.js");

var PLAN_URL = "compiled-fp32-v1/plan.json?v=fp32-opt-v2";
var WEIGHTS_PARTS = [
    "compiled-fp32-v1/weights.part00.bin?v=fp32-opt-v2-manual",
    "compiled-fp32-v1/weights.part01.bin?v=fp32-opt-v2-manual",
    "compiled-fp32-v1/weights.part02.bin?v=fp32-opt-v2-manual"
];
var WEIGHTS_TOTAL_BYTES = 45425796;
var LABELS_URL = "text-bank/labels.json?v=fp32-opt-v2";
var EMBEDDINGS_URL = "text-bank/embeddings.f32?v=fp32-opt-v2";

var plan = null;
var weightsBuffer = null;
var labels = null;
var textBank = null;
var textDim = 512;
var executor = null;
var ready = false;
var busy = false;

function postStatus(text) {
    self.postMessage({
        type: "status",
        text: text
    });
}

function postLoadProgress(phase, loaded, total, startPct, endPct) {
    var pct = startPct;

    if (total > 0) {
        pct = startPct +
            (endPct - startPct) * loaded / total;
    }

    if (pct < startPct) {
        pct = startPct;
    }
    if (pct > endPct) {
        pct = endPct;
    }

    self.postMessage({
        type: "load-progress",
        phase: phase,
        loaded: loaded,
        total: total,
        percent: pct
    });
}

function xhrText(url, phase, startPct, endPct, done) {
    var xhr = new XMLHttpRequest();

    xhr.open("GET", url, true);

    xhr.onprogress = function (e) {
        postLoadProgress(
            phase,
            e.loaded || 0,
            e.lengthComputable ? e.total : 0,
            startPct,
            endPct
        );
    };

    xhr.onerror = function () {
        done(new Error("XHR failed: " + url));
    };

    xhr.onload = function () {
        if (xhr.status < 200 || xhr.status >= 300) {
            done(new Error(
                "HTTP " + xhr.status + " for " + url
            ));
            return;
        }

        postLoadProgress(
            phase,
            1,
            1,
            startPct,
            endPct
        );

        done(null, xhr.responseText);
    };

    xhr.send(null);
}

function xhrArrayBuffer(url, phase, startPct, endPct, done) {
    var xhr = new XMLHttpRequest();

    xhr.open("GET", url, true);

    try {
        xhr.responseType = "arraybuffer";
    } catch (e) {
        done(new Error(
            "ArrayBuffer XHR unsupported for " + url +
            ": " + e.message
        ));
        return;
    }

    xhr.onprogress = function (e) {
        postLoadProgress(
            phase,
            e.loaded || 0,
            e.lengthComputable ? e.total : 0,
            startPct,
            endPct
        );
    };

    xhr.onerror = function () {
        done(new Error("XHR failed: " + url));
    };

    xhr.onload = function () {
        var buf;

        if (xhr.status < 200 || xhr.status >= 300) {
            done(new Error(
                "HTTP " + xhr.status + " for " + url
            ));
            return;
        }

        buf = xhr.response;

        if (!buf || typeof buf.byteLength !== "number") {
            done(new Error(
                "No ArrayBuffer response for " + url
            ));
            return;
        }

        postLoadProgress(
            phase,
            1,
            1,
            startPct,
            endPct
        );

        done(null, buf);
    };

    xhr.send(null);
}


function loadWeightParts(done) {
    var combined = new Uint8Array(WEIGHTS_TOTAL_BYTES);
    var offset = 0;
    var index = 0;

    function next() {
        var startPct;
        var endPct;

        if (index >= WEIGHTS_PARTS.length) {
            if (offset !== WEIGHTS_TOTAL_BYTES) {
                done(new Error(
                    "weight-part total mismatch: loaded=" +
                    offset + " expected=" + WEIGHTS_TOTAL_BYTES
                ));
                return;
            }

            postLoadProgress(
                "FP32 weights",
                1,
                1,
                3,
                90
            );

            done(null, combined.buffer);
            return;
        }

        startPct = 3 +
            (87 * index / WEIGHTS_PARTS.length);
        endPct = 3 +
            (87 * (index + 1) / WEIGHTS_PARTS.length);

        xhrArrayBuffer(
            WEIGHTS_PARTS[index],
            "FP32 weights part " +
                (index + 1) + "/" + WEIGHTS_PARTS.length,
            startPct,
            endPct,
            function (err, buf) {
                var part;

                if (err) {
                    done(err);
                    return;
                }

                part = new Uint8Array(buf);

                if (offset + part.length > combined.length) {
                    done(new Error(
                        "weight parts exceed expected total bytes"
                    ));
                    return;
                }

                combined.set(part, offset);
                offset += part.length;
                index += 1;

                part = null;
                buf = null;

                next();
            }
        );
    }

    next();
}

function normalizeTextBankInPlace(bank, count, dim) {
    var i;
    var j;
    var off;
    var v;
    var ss;
    var norm;

    for (i = 0; i < count; ++i) {
        off = i * dim;
        ss = 0.0;

        for (j = 0; j < dim; ++j) {
            v = bank[off + j];
            ss += v * v;
        }

        norm = Math.sqrt(ss);

        if (!(norm > 0.0) || !isFinite(norm)) {
            throw new Error(
                "invalid text embedding norm at label " + i
            );
        }

        for (j = 0; j < dim; ++j) {
            bank[off + j] /= norm;
        }
    }
}

function loadAll() {
    postStatus("Loading FP32 plan...");

    xhrText(
        PLAN_URL,
        "FP32 plan",
        0,
        3,
        function (err, text) {
            if (err) {
                fail(err);
                return;
            }

            try {
                plan = JSON.parse(text);
            } catch (e) {
                fail(new Error(
                    "plan JSON parse failed: " + e.message
                ));
                return;
            }

            postStatus("Loading FP32 weights in 3 GitHub-safe parts (~43.3 MiB)...");

            loadWeightParts(
                function (err2, buf) {
                    if (err2) {
                        fail(err2);
                        return;
                    }

                    weightsBuffer = buf;

                    postStatus("Loading text labels...");

                    xhrText(
                        LABELS_URL,
                        "text labels",
                        90,
                        92,
                        function (err3, labelText) {
                            var meta;

                            if (err3) {
                                fail(err3);
                                return;
                            }

                            try {
                                meta = JSON.parse(labelText);
                                labels = meta.labels;
                                textDim = meta.dim || 512;
                            } catch (e2) {
                                fail(new Error(
                                    "labels JSON parse failed: " +
                                    e2.message
                                ));
                                return;
                            }

                            if (!labels || !labels.length) {
                                fail(new Error(
                                    "labels array missing/empty"
                                ));
                                return;
                            }

                            postStatus(
                                "Loading text embeddings (~2.5 MiB)..."
                            );

                            xhrArrayBuffer(
                                EMBEDDINGS_URL,
                                "text embeddings",
                                92,
                                100,
                                function (err4, embBuf) {
                                    if (err4) {
                                        fail(err4);
                                        return;
                                    }

                                    try {
                                        textBank =
                                            new Float32Array(embBuf);

                                        if (
                                            textBank.length !==
                                            labels.length * textDim
                                        ) {
                                            throw new Error(
                                                "text bank size mismatch: " +
                                                textBank.length +
                                                " floats vs expected " +
                                                (labels.length * textDim)
                                            );
                                        }

                                        postStatus(
                                            "Normalizing text bank..."
                                        );

                                        normalizeTextBankInPlace(
                                            textBank,
                                            labels.length,
                                            textDim
                                        );

                                        executor =
                                            FP32.createExecutor(
                                                plan,
                                                weightsBuffer
                                            );

                                        ready = true;

                                        self.postMessage({
                                            type: "ready",
                                            nodes: plan.nodes.length,
                                            labels: labels.length,
                                            weightsBytes:
                                                weightsBuffer.byteLength,
                                            embeddingBytes:
                                                embBuf.byteLength
                                        });
                                    } catch (e3) {
                                        fail(e3);
                                    }
                                }
                            );
                        }
                    );
                }
            );
        }
    );
}

function arrayCopyF32(a) {
    var out = new Array(a.length);
    var i;
    for (i = 0; i < a.length; ++i) {
        out[i] = a[i];
    }
    return out;
}

function classify(imageEmbedding, topK) {
    var dim = textDim;
    var count = labels.length;
    var image = new Float32Array(dim);
    var ss = 0.0;
    var norm;
    var i;
    var j;
    var off;
    var v;
    var score;
    var top = [];
    var pos;

    if (imageEmbedding.length !== dim) {
        throw new Error(
            "image embedding dim mismatch: " +
            imageEmbedding.length + " vs " + dim
        );
    }

    for (j = 0; j < dim; ++j) {
        v = imageEmbedding[j];
        ss += v * v;
    }

    norm = Math.sqrt(ss);

    if (!(norm > 0.0) || !isFinite(norm)) {
        throw new Error("invalid image embedding norm");
    }

    for (j = 0; j < dim; ++j) {
        image[j] = imageEmbedding[j] / norm;
    }

    for (i = 0; i < count; ++i) {
        off = i * dim;
        score = 0.0;

        for (j = 0; j < dim; ++j) {
            score += image[j] * textBank[off + j];
        }

        pos = top.length;

        while (pos > 0 && score > top[pos - 1].score) {
            pos -= 1;
        }

        if (pos < topK) {
            top.splice(pos, 0, {
                label: labels[i],
                score: score,
                index: i
            });

            if (top.length > topK) {
                top.pop();
            }
        }
    }

    return {
        top: top,
        rawNorm: norm
    };
}

function runImage(inputBuffer) {
    var input;
    var tensor;
    var outputs;
    var embedding;
    var infStart;
    var infMs;
    var classStart;
    var classMs;
    var cls;
    var fullEmbedding;

    if (!ready) {
        throw new Error("model is not ready");
    }

    if (busy) {
        throw new Error("worker is already running");
    }

    busy = true;

    try {
        input = new Float32Array(inputBuffer);

        if (input.length !== 3 * 256 * 256) {
            throw new Error(
                "input length mismatch: " + input.length
            );
        }

        tensor = new FP32.Tensor(
            input,
            [1, 3, 256, 256],
            "f32"
        );

        infStart = Date.now();

        outputs = executor.run(tensor, {
            progressEvery: 8,
            onProgress: function (done, total) {
                self.postMessage({
                    type: "infer-progress",
                    done: done,
                    total: total,
                    percent: 100 * done / total
                });
            }
        });

        infMs = Date.now() - infStart;

        if (
            !outputs.image_embeds ||
            !outputs.image_embeds.data
        ) {
            throw new Error("image_embeds missing");
        }

        embedding = outputs.image_embeds.data;

        classStart = Date.now();
        cls = classify(embedding, 10);
        classMs = Date.now() - classStart;

        fullEmbedding = arrayCopyF32(embedding);

        self.postMessage({
            type: "result",
            inferenceMs: infMs,
            classifyMs: classMs,
            embedding: fullEmbedding,
            first16: fullEmbedding.slice(0, 16),
            rawNorm: cls.rawNorm,
            top: cls.top,
            profile: outputs.__profile
        });
    } finally {
        busy = false;
    }
}

function fail(err) {
    self.postMessage({
        type: "error",
        text: (err && err.stack) ?
            err.stack :
            String(err)
    });
}

self.onmessage = function (ev) {
    var m = ev.data || {};

    if (m.type === "run") {
        try {
            runImage(m.inputBuffer);
        } catch (e) {
            fail(e);
            busy = false;
        }
    }
};

loadAll();

};
deps[0] = {'./mobileclip-fp32-executor-opt-v2.js':1};
modules[1] = function (module, exports, require) {
"use strict";

var K = require("./mobileclip-fp32-kernels-v1.js");
var O = require("./mobileclip-fp32-kernels-opt-v2.js");

function Tensor(data, shape, dtype) {
    this.data = data;
    this.shape = shape.slice(0);
    this.dtype = dtype || "f32";
}

function product(shape, begin, end) {
    return K.product(shape, begin, end);
}

function cloneShape(shape) {
    return shape.slice(0);
}

function makeTyped(dtype, n) {
    if (dtype === "f32") {
        return new Float32Array(n);
    }
    if (dtype === "i32") {
        return new Int32Array(n);
    }
    if (dtype === "u8" || dtype === "bool") {
        return new Uint8Array(n);
    }
    throw new Error("unsupported dtype: " + dtype);
}

function unaryFloat(x, fn) {
    var out = new Float32Array(x.data.length);
    var i;
    for (i = 0; i < out.length; ++i) {
        out[i] = fn(x.data[i]);
    }
    return new Tensor(out, x.shape, "f32");
}

function opAdd(a, b) {
    var r = K.binaryF32(
        a.data, a.shape, b.data, b.shape,
        function (x, y) { return x + y; }
    );
    return new Tensor(r.data, r.shape, "f32");
}

function opMul(a, b) {
    var r = K.binaryF32(
        a.data, a.shape, b.data, b.shape,
        function (x, y) { return x * y; }
    );
    return new Tensor(r.data, r.shape, "f32");
}

function opDiv(a, b) {
    var r = K.binaryF32(
        a.data, a.shape, b.data, b.shape,
        function (x, y) { return x / y; }
    );
    return new Tensor(r.data, r.shape, "f32");
}

function opRelu(x) {
    return unaryFloat(x, function (v) {
        return v > 0.0 ? v : 0.0;
    });
}

function opErf(x) {
    return unaryFloat(x, K.erfApprox);
}


function geluChainAt(nodes, i) {
    var n0;
    var n1;
    var n2;
    var n3;
    var n4;

    if (i < 0 || i + 4 >= nodes.length) {
        return null;
    }

    n0 = nodes[i + 0];
    n1 = nodes[i + 1];
    n2 = nodes[i + 2];
    n3 = nodes[i + 3];
    n4 = nodes[i + 4];

    if (
        n0.op !== "Div" ||
        n1.op !== "Erf" ||
        n2.op !== "Add" ||
        n3.op !== "Mul" ||
        n4.op !== "Mul"
    ) {
        return null;
    }

    if (
        !n0.outputs || n0.outputs.length !== 1 ||
        !n1.outputs || n1.outputs.length !== 1 ||
        !n2.outputs || n2.outputs.length !== 1 ||
        !n3.outputs || n3.outputs.length !== 1 ||
        !n4.outputs || n4.outputs.length !== 1
    ) {
        return null;
    }

    if (
        !n1.inputs ||
        n1.inputs.indexOf(n0.outputs[0]) < 0
    ) {
        return null;
    }

    if (
        !n2.inputs ||
        n2.inputs.indexOf(n1.outputs[0]) < 0
    ) {
        return null;
    }

    if (
        !n3.inputs ||
        n3.inputs.indexOf(n2.outputs[0]) < 0 ||
        n3.inputs.indexOf(n0.inputs[0]) < 0
    ) {
        return null;
    }

    if (
        !n4.inputs ||
        n4.inputs.indexOf(n3.outputs[0]) < 0
    ) {
        return null;
    }

    return {
        n0: n0,
        n1: n1,
        n2: n2,
        n3: n3,
        n4: n4
    };
}

function otherInputName(node, producedName) {
    var i;

    for (i = 0; i < node.inputs.length; ++i) {
        if (
            node.inputs[i] &&
            node.inputs[i] !== producedName
        ) {
            return node.inputs[i];
        }
    }

    return null;
}

function opSigmoid(x) {
    return unaryFloat(x, K.sigmoid);
}

function opShape(x) {
    var out = new Int32Array(x.shape.length);
    var i;
    for (i = 0; i < x.shape.length; ++i) {
        out[i] = x.shape[i];
    }
    return new Tensor(out, [x.shape.length], "i32");
}

function opReshape(x, shapeTensor, allowzero) {
    var spec = shapeTensor.data;
    var outShape = new Array(spec.length);
    var inferAt = -1;
    var known = 1;
    var total = x.data.length;
    var i;
    var v;

    for (i = 0; i < spec.length; ++i) {
        v = spec[i];

        if (v === 0 && !allowzero) {
            if (i >= x.shape.length) {
                throw new Error("Reshape zero index exceeds input rank");
            }
            v = x.shape[i];
        }

        if (v === -1) {
            if (inferAt >= 0) {
                throw new Error("Reshape has multiple -1");
            }
            inferAt = i;
            outShape[i] = -1;
        } else {
            outShape[i] = v;
            known *= v;
        }
    }

    if (inferAt >= 0) {
        outShape[inferAt] = total / known;
    }

    if (product(outShape) !== total) {
        throw new Error(
            "Reshape size mismatch: " +
            JSON.stringify(x.shape) + " -> " +
            JSON.stringify(outShape)
        );
    }

    return new Tensor(x.data, outShape, x.dtype);
}

function opTranspose(x, perm) {
    var rank = x.shape.length;
    var p = perm || [];
    var outShape = new Array(rank);
    var inS = K.computeStrides(x.shape);
    var outS;
    var out = makeTyped(x.dtype, x.data.length);
    var i;
    var d;
    var t;
    var coord;
    var src;

    if (p.length === 0) {
        for (i = 0; i < rank; ++i) {
            p[i] = rank - 1 - i;
        }
    }

    for (i = 0; i < rank; ++i) {
        outShape[i] = x.shape[p[i]];
    }

    outS = K.computeStrides(outShape);

    for (i = 0; i < out.length; ++i) {
        t = i;
        src = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / outS[d]);
            t -= coord * outS[d];
            src += coord * inS[p[d]];
        }
        out[i] = x.data[src];
    }

    return new Tensor(out, outShape, x.dtype);
}

function opUnsqueeze(x, axes) {
    var rank = x.shape.length + axes.length;
    var marks = {};
    var outShape = [];
    var i;
    var ax;
    var src = 0;

    for (i = 0; i < axes.length; ++i) {
        ax = axes[i];
        if (ax < 0) {
            ax += rank;
        }
        marks[ax] = true;
    }

    for (i = 0; i < rank; ++i) {
        if (marks[i]) {
            outShape.push(1);
        } else {
            outShape.push(x.shape[src++]);
        }
    }

    return new Tensor(x.data, outShape, x.dtype);
}

function opSqueeze(x, axes) {
    var remove = {};
    var outShape = [];
    var i;
    var ax;

    if (!axes || axes.length === 0) {
        for (i = 0; i < x.shape.length; ++i) {
            if (x.shape[i] !== 1) {
                outShape.push(x.shape[i]);
            }
        }
    } else {
        for (i = 0; i < axes.length; ++i) {
            ax = K.normalizeAxis(axes[i], x.shape.length);
            if (x.shape[ax] !== 1) {
                throw new Error("Squeeze axis dimension != 1");
            }
            remove[ax] = true;
        }
        for (i = 0; i < x.shape.length; ++i) {
            if (!remove[i]) {
                outShape.push(x.shape[i]);
            }
        }
    }

    return new Tensor(x.data, outShape, x.dtype);
}

function opConcat(xs, axis) {
    var rank = xs[0].shape.length;
    var ax = K.normalizeAxis(axis, rank);
    var outShape = xs[0].shape.slice(0);
    var outer = product(outShape, 0, ax);
    var inner = product(outShape, ax + 1);
    var totalAxis = 0;
    var i;
    var o;
    var j;
    var srcChunk;
    var dst;
    var out;

    for (i = 0; i < xs.length; ++i) {
        totalAxis += xs[i].shape[ax];
    }
    outShape[ax] = totalAxis;
    out = makeTyped(xs[0].dtype, product(outShape));

    dst = 0;
    for (o = 0; o < outer; ++o) {
        for (i = 0; i < xs.length; ++i) {
            srcChunk = xs[i].shape[ax] * inner;
            for (j = 0; j < srcChunk; ++j) {
                out[dst++] = xs[i].data[o * srcChunk + j];
            }
        }
    }

    return new Tensor(out, outShape, xs[0].dtype);
}

function opGather(data, indices, axis) {
    var rank = data.shape.length;
    var ax = K.normalizeAxis(axis, rank);
    var before = product(data.shape, 0, ax);
    var axisDim = data.shape[ax];
    var after = product(data.shape, ax + 1);
    var outShape = data.shape.slice(0, ax)
        .concat(indices.shape)
        .concat(data.shape.slice(ax + 1));
    var out = makeTyped(data.dtype, product(outShape));
    var b;
    var j;
    var k;
    var idx;
    var dst = 0;
    var src;

    for (b = 0; b < before; ++b) {
        for (j = 0; j < indices.data.length; ++j) {
            idx = indices.data[j];
            if (idx < 0) {
                idx += axisDim;
            }
            if (idx < 0 || idx >= axisDim) {
                throw new Error("Gather index out of range");
            }
            src = (b * axisDim + idx) * after;
            for (k = 0; k < after; ++k) {
                out[dst++] = data.data[src + k];
            }
        }
    }

    return new Tensor(out, outShape, data.dtype);
}

function opSlice(data, startsT, endsT, axesT, stepsT) {
    var rank = data.shape.length;
    var starts = startsT.data;
    var ends = endsT.data;
    var axes = axesT ? axesT.data : null;
    var steps = stepsT ? stepsT.data : null;
    var begin = new Array(rank);
    var end = new Array(rank);
    var step = new Array(rank);
    var outShape = new Array(rank);
    var inS = K.computeStrides(data.shape);
    var outS;
    var i;
    var d;
    var ax;
    var s;
    var e;
    var st;
    var dim;
    var out;
    var t;
    var coord;
    var src;

    for (i = 0; i < rank; ++i) {
        begin[i] = 0;
        end[i] = data.shape[i];
        step[i] = 1;
    }

    for (i = 0; i < starts.length; ++i) {
        ax = axes ? axes[i] : i;
        ax = K.normalizeAxis(ax, rank);
        dim = data.shape[ax];
        st = steps ? steps[i] : 1;
        if (st === 0) {
            throw new Error("Slice step=0");
        }

        s = starts[i];
        e = ends[i];

        if (s < 0) {
            s += dim;
        }
        if (e < 0) {
            e += dim;
        }

        if (st > 0) {
            s = Math.max(0, Math.min(dim, s));
            e = Math.max(0, Math.min(dim, e));
        } else {
            s = Math.max(-1, Math.min(dim - 1, s));
            e = Math.max(-1, Math.min(dim - 1, e));
        }

        begin[ax] = s;
        end[ax] = e;
        step[ax] = st;
    }

    for (d = 0; d < rank; ++d) {
        if (step[d] > 0) {
            outShape[d] = Math.max(
                0,
                Math.ceil((end[d] - begin[d]) / step[d])
            );
        } else {
            outShape[d] = Math.max(
                0,
                Math.ceil((begin[d] - end[d]) / (-step[d]))
            );
        }
    }

    out = makeTyped(data.dtype, product(outShape));
    outS = K.computeStrides(outShape);

    for (i = 0; i < out.length; ++i) {
        t = i;
        src = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / outS[d]);
            t -= coord * outS[d];
            src += (begin[d] + coord * step[d]) * inS[d];
        }
        out[i] = data.data[src];
    }

    return new Tensor(out, outShape, data.dtype);
}

function opSplit(x, axis, split, outputCount) {
    var ax = K.normalizeAxis(axis, x.shape.length);
    var sizes;
    var inner = product(x.shape, ax + 1);
    var outer = product(x.shape, 0, ax);
    var axisDim = x.shape[ax];
    var outs = [];
    var i;
    var o;
    var j;
    var k;
    var offset;
    var src;
    var dst;
    var shape;
    var arr;

    if (split && split.length) {
        sizes = split.slice(0);
    } else {
        if (axisDim % outputCount !== 0) {
            throw new Error("Split not even");
        }
        sizes = [];
        for (i = 0; i < outputCount; ++i) {
            sizes.push(axisDim / outputCount);
        }
    }

    for (i = 0; i < sizes.length; ++i) {
        shape = x.shape.slice(0);
        shape[ax] = sizes[i];
        outs.push(new Tensor(
            makeTyped(x.dtype, product(shape)),
            shape,
            x.dtype
        ));
    }

    for (o = 0; o < outer; ++o) {
        offset = 0;
        for (i = 0; i < sizes.length; ++i) {
            arr = outs[i].data;
            src = (o * axisDim + offset) * inner;
            dst = o * sizes[i] * inner;
            for (j = 0; j < sizes[i] * inner; ++j) {
                arr[dst + j] = x.data[src + j];
            }
            offset += sizes[i];
        }
    }

    return outs;
}

function opReduceMean(x, axes, keepdims) {
    var rank = x.shape.length;
    var reduce = {};
    var outShape = [];
    var outShapeKeep = new Array(rank);
    var out;
    var counts;
    var inS = K.computeStrides(x.shape);
    var outS;
    var i;
    var d;
    var ax;
    var t;
    var coord;
    var oi;

    if (!axes || axes.length === 0) {
        axes = [];
        for (i = 0; i < rank; ++i) {
            axes.push(i);
        }
    }

    for (i = 0; i < axes.length; ++i) {
        ax = K.normalizeAxis(axes[i], rank);
        reduce[ax] = true;
    }

    for (d = 0; d < rank; ++d) {
        if (reduce[d]) {
            outShapeKeep[d] = 1;
            if (keepdims) {
                outShape.push(1);
            }
        } else {
            outShapeKeep[d] = x.shape[d];
            outShape.push(x.shape[d]);
        }
    }

    if (outShape.length === 0) {
        outShape = [];
    }

    out = new Float32Array(product(outShape));
    counts = new Int32Array(out.length);
    outS = K.computeStrides(outShapeKeep);

    for (i = 0; i < x.data.length; ++i) {
        t = i;
        oi = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / inS[d]);
            t -= coord * inS[d];
            if (!reduce[d]) {
                oi += coord * outS[d];
            }
        }
        out[oi] += x.data[i];
        counts[oi] += 1;
    }

    for (i = 0; i < out.length; ++i) {
        out[i] /= counts[i];
    }

    return new Tensor(out, outShape, "f32");
}

function opSoftmax(x, axis) {
    var ax = K.normalizeAxis(axis, x.shape.length);
    var outer = product(x.shape, 0, ax);
    var dim = x.shape[ax];
    var inner = product(x.shape, ax + 1);
    var out = new Float32Array(x.data.length);
    var o;
    var k;
    var j;
    var base;
    var idx;
    var maxv;
    var sum;
    var v;

    for (o = 0; o < outer; ++o) {
        for (j = 0; j < inner; ++j) {
            base = o * dim * inner + j;
            maxv = -Infinity;

            for (k = 0; k < dim; ++k) {
                v = x.data[base + k * inner];
                if (v > maxv) {
                    maxv = v;
                }
            }

            sum = 0.0;
            for (k = 0; k < dim; ++k) {
                idx = base + k * inner;
                v = Math.exp(x.data[idx] - maxv);
                out[idx] = v;
                sum += v;
            }

            for (k = 0; k < dim; ++k) {
                idx = base + k * inner;
                out[idx] /= sum;
            }
        }
    }

    return new Tensor(out, x.shape, "f32");
}

function opBatchNormalization(x, scale, bias, mean, variance, epsilon) {
    var C = x.shape[1];
    var spatial = product(x.shape, 2);
    var N = x.shape[0];
    var out = new Float32Array(x.data.length);
    var n;
    var c;
    var p;
    var base;
    var mul;
    var add;

    for (c = 0; c < C; ++c) {
        mul = scale.data[c] /
            Math.sqrt(variance.data[c] + epsilon);
        add = bias.data[c] - mean.data[c] * mul;

        for (n = 0; n < N; ++n) {
            base = (n * C + c) * spatial;
            for (p = 0; p < spatial; ++p) {
                out[base + p] = x.data[base + p] * mul + add;
            }
        }
    }

    return new Tensor(out, x.shape, "f32");
}

function opPadConstant(x, padsT, constantT) {
    var rank = x.shape.length;
    var pads = padsT.data;
    var before = new Array(rank);
    var after = new Array(rank);
    var outShape = new Array(rank);
    var inS = K.computeStrides(x.shape);
    var outS;
    var out;
    var value = constantT ? constantT.data[0] : 0.0;
    var i;
    var d;
    var t;
    var coord;
    var dst;

    if (pads.length !== rank * 2) {
        throw new Error("Pad pads length mismatch");
    }

    for (d = 0; d < rank; ++d) {
        before[d] = pads[d];
        after[d] = pads[d + rank];
        if (before[d] < 0 || after[d] < 0) {
            throw new Error("negative Pad unsupported");
        }
        outShape[d] = x.shape[d] + before[d] + after[d];
    }

    out = new Float32Array(product(outShape));
    if (value !== 0.0) {
        for (i = 0; i < out.length; ++i) {
            out[i] = value;
        }
    }
    outS = K.computeStrides(outShape);

    for (i = 0; i < x.data.length; ++i) {
        t = i;
        dst = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / inS[d]);
            t -= coord * inS[d];
            dst += (coord + before[d]) * outS[d];
        }
        out[dst] = x.data[i];
    }

    return new Tensor(out, outShape, "f32");
}

function opAveragePoolNCHW(x, attrs) {
    var N = x.shape[0];
    var C = x.shape[1];
    var H = x.shape[2];
    var W = x.shape[3];
    var kernel = attrs.kernel_shape;
    var strides = attrs.strides || kernel;
    var pads = attrs.pads || [0, 0, 0, 0];
    var KH = kernel[0];
    var KW = kernel[1];
    var sh = strides[0];
    var sw = strides[1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var OH = Math.floor((H + pt + pb - KH) / sh) + 1;
    var OW = Math.floor((W + pl + pr - KW) / sw) + 1;
    var out = new Float32Array(N * C * OH * OW);
    var n;
    var c;
    var oy;
    var ox;
    var ky;
    var kx;
    var iy;
    var ix;
    var sum;
    var count;
    var xo;
    var yo;

    if (attrs.ceil_mode) {
        throw new Error("AveragePool ceil_mode unsupported");
    }

    for (n = 0; n < N; ++n) {
        for (c = 0; c < C; ++c) {
            for (oy = 0; oy < OH; ++oy) {
                for (ox = 0; ox < OW; ++ox) {
                    sum = 0.0;
                    count = 0;

                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky;
                        if (iy < 0 || iy >= H) {
                            continue;
                        }
                        for (kx = 0; kx < KW; ++kx) {
                            ix = ox * sw - pl + kx;
                            if (ix < 0 || ix >= W) {
                                continue;
                            }
                            xo = ((n * C + c) * H + iy) * W + ix;
                            sum += x.data[xo];
                            count += 1;
                        }
                    }

                    yo = ((n * C + c) * OH + oy) * OW + ox;
                    out[yo] = sum / count;
                }
            }
        }
    }

    return new Tensor(out, [N, C, OH, OW], "f32");
}

function createExecutor(plan, weightsBuffer) {
    var weights = weightsBuffer;
    var staticCache = {};

    if (weightsBuffer instanceof Uint8Array) {
        weights = weightsBuffer.buffer.slice(
            weightsBuffer.byteOffset,
            weightsBuffer.byteOffset + weightsBuffer.byteLength
        );
    }

    if (!(weights instanceof ArrayBuffer)) {
        throw new Error("weights must be ArrayBuffer or Uint8Array");
    }

    function staticTensor(name) {
        var info;
        var data;

        if (!name) {
            return null;
        }
        if (staticCache[name]) {
            return staticCache[name];
        }

        info = plan.tensors[name];
        if (!info) {
            return null;
        }

        if (info.dtype === "f32") {
            data = new Float32Array(
                weights,
                info.offset,
                info.nbytes / 4
            );
        } else if (info.dtype === "i32") {
            data = new Int32Array(
                weights,
                info.offset,
                info.nbytes / 4
            );
        } else if (info.dtype === "u8" || info.dtype === "bool") {
            data = new Uint8Array(
                weights,
                info.offset,
                info.nbytes
            );
        } else {
            throw new Error(
                "unsupported static dtype " + info.dtype +
                " tensor=" + name
            );
        }

        staticCache[name] = new Tensor(
            data,
            info.shape,
            info.dtype
        );
        return staticCache[name];
    }

    function run(inputOrFeeds, options) {
        var env = {};
        var uses = {};
        var keep = {};
        var outputNames = {};
        var inputName = plan.inputs[0].name;
        var inputTensor;
        var ni;
        var node;
        var op;
        var attrs;
        var ins;
        var i;
        var j;
        var name;
        var t0;
        var elapsed;
        var opTimes = {};
        var opCalls = {};
        var result;
        var r;
        var outs;
        var wInfo;
        var fast;
        var geluSkip = {};
        var geluGraphCount = 0;
        var geluScan;

        options = options || {};

        if (
            inputOrFeeds &&
            inputOrFeeds.data &&
            inputOrFeeds.shape
        ) {
            inputTensor = inputOrFeeds;
        } else if (
            inputOrFeeds &&
            inputOrFeeds[inputName]
        ) {
            inputTensor = inputOrFeeds[inputName];
        } else {
            throw new Error("input tensor not supplied: " + inputName);
        }

        env[inputName] = inputTensor;

        for (ni = 0; ni < plan.nodes.length; ++ni) {
            node = plan.nodes[ni];
            for (j = 0; j < node.inputs.length; ++j) {
                name = node.inputs[j];
                if (name) {
                    uses[name] = (uses[name] || 0) + 1;
                }
            }
        }

        for (i = 0; i < plan.outputs.length; ++i) {
            keep[plan.outputs[i].name] = true;
            outputNames[plan.outputs[i].name] = true;
        }

        for (
            geluScan = 0;
            geluScan + 4 < plan.nodes.length;
            ++geluScan
        ) {
            if (geluChainAt(plan.nodes, geluScan)) {
                ++geluGraphCount;
                geluScan += 4;
            }
        }

        if (geluGraphCount !== 30) {
            throw new Error(
                "expected 30 GELU chains, found " +
                geluGraphCount
            );
        }

        function getTensor(tname) {
            var s;
            if (!tname) {
                return null;
            }
            if (env[tname]) {
                return env[tname];
            }
            s = staticTensor(tname);
            if (s) {
                return s;
            }
            throw new Error("tensor not found: " + tname);
        }

        function consume(tname) {
            if (!tname || plan.tensors[tname]) {
                return;
            }
            uses[tname] -= 1;
            if (uses[tname] <= 0 && !keep[tname]) {
                delete env[tname];
            }
        }

        for (ni = 0; ni < plan.nodes.length; ++ni) {
            node = plan.nodes[ni];
            op = node.op;
            attrs = node.attrs || {};
            ins = [];

            for (j = 0; j < node.inputs.length; ++j) {
                ins.push(
                    node.inputs[j] ? getTensor(node.inputs[j]) : null
                );
            }

            t0 = Date.now();

            try {
                if (geluSkip[ni]) {
                    /*
                     * Fused GELU continuation node.
                     * Output was already materialized at the Div node.
                     */
                }

                else if (op === "Conv") {
                    wInfo = plan.tensors[node.inputs[1]];
                    fast = node.fast || {};

                    if (!ins[0] || !ins[1] || !ins[2]) {
                        throw new Error("Conv requires x,w,b");
                    }

                    if (fast.kind === "pointwise") {
                        r = O.convPointwiseB4T8NCHW(
                            ins[0].data,
                            ins[1].data,
                            ins[2].data,
                            ins[0].shape,
                            wInfo.shape
                        );
                    } else if (fast.kind === "depthwise") {
                        r = O.convDepthwiseDW8NCHW(
                            ins[0].data,
                            ins[1].data,
                            ins[2].data,
                            ins[0].shape,
                            wInfo.shape,
                            attrs
                        );
                    } else if (fast.kind === "grouped") {
                        r = O.convGroupedM2T8NCHW(
                            ins[0].data,
                            ins[1].data,
                            ins[2].data,
                            ins[0].shape,
                            wInfo.shape,
                            attrs
                        );
                    } else {
                        r = O.convInitial3x3s2B8T4NCHW(
                            ins[0].data,
                            ins[1].data,
                            ins[2].data,
                            ins[0].shape,
                            wInfo.shape,
                            attrs
                        );
                    }

                    env[node.outputs[0]] =
                        new Tensor(r.data, r.shape, "f32");
                }

                else if (op === "MatMul") {
                    r = O.matMulBN4(
                        ins[0].data,
                        ins[0].shape,
                        ins[1].data,
                        ins[1].shape
                    );
                    env[node.outputs[0]] =
                        new Tensor(r.data, r.shape, "f32");
                }

                else if (op === "Add") {
                    env[node.outputs[0]] = opAdd(ins[0], ins[1]);
                }

                else if (op === "Mul") {
                    env[node.outputs[0]] = opMul(ins[0], ins[1]);
                }

                else if (
                    op === "Div" &&
                    geluChainAt(plan.nodes, ni)
                ) {
                    var gc = geluChainAt(
                        plan.nodes,
                        ni
                    );

                    var addOtherName = otherInputName(
                        gc.n2,
                        gc.n1.outputs[0]
                    );

                    var mulOtherName = otherInputName(
                        gc.n4,
                        gc.n3.outputs[0]
                    );

                    var addOther = addOtherName ?
                        getTensor(addOtherName) :
                        null;

                    var mulOther = mulOtherName ?
                        getTensor(mulOtherName) :
                        null;

                    var fusedData;

                    if (
                        !ins[0] ||
                        !ins[1] ||
                        ins[0].dtype !== "f32" ||
                        ins[1].data.length !== 1 ||
                        !addOther ||
                        addOther.data.length !== 1 ||
                        !mulOther ||
                        mulOther.data.length !== 1
                    ) {
                        throw new Error(
                            "unexpected GELU scalar/input contract " +
                            "at node " + ni
                        );
                    }

                    if (
                        uses[gc.n0.outputs[0]] !== 1 ||
                        uses[gc.n1.outputs[0]] !== 1 ||
                        uses[gc.n2.outputs[0]] !== 1 ||
                        uses[gc.n3.outputs[0]] !== 1
                    ) {
                        throw new Error(
                            "GELU intermediate has external consumer " +
                            "at node " + ni
                        );
                    }

                    fusedData =
                        O.geluFusedErfSat4PolyAll(
                            ins[0].data,
                            ins[1].data[0],
                            addOther.data[0],
                            mulOther.data[0]
                        );

                    env[gc.n4.outputs[0]] =
                        new Tensor(
                            fusedData,
                            cloneShape(ins[0].shape),
                            "f32"
                        );

                    env[gc.n0.outputs[0]] =
                        new Tensor(
                            ins[0].data,
                            cloneShape(ins[0].shape),
                            "f32"
                        );

                    env[gc.n1.outputs[0]] =
                        new Tensor(
                            ins[0].data,
                            cloneShape(ins[0].shape),
                            "f32"
                        );

                    env[gc.n2.outputs[0]] =
                        new Tensor(
                            ins[0].data,
                            cloneShape(ins[0].shape),
                            "f32"
                        );

                    env[gc.n3.outputs[0]] =
                        new Tensor(
                            ins[0].data,
                            cloneShape(ins[0].shape),
                            "f32"
                        );

                    geluSkip[ni + 1] = true;
                    geluSkip[ni + 2] = true;
                    geluSkip[ni + 3] = true;
                    geluSkip[ni + 4] = true;
                }

                else if (op === "Div") {
                    env[node.outputs[0]] = opDiv(ins[0], ins[1]);
                }

                else if (op === "Relu") {
                    env[node.outputs[0]] = opRelu(ins[0]);
                }

                else if (op === "Erf") {
                    env[node.outputs[0]] = opErf(ins[0]);
                }

                else if (op === "Sigmoid") {
                    env[node.outputs[0]] = opSigmoid(ins[0]);
                }

                else if (op === "Shape") {
                    env[node.outputs[0]] = opShape(ins[0]);
                }

                else if (op === "Reshape") {
                    env[node.outputs[0]] = opReshape(
                        ins[0],
                        ins[1],
                        attrs.allowzero || 0
                    );
                }

                else if (op === "Transpose") {
                    env[node.outputs[0]] =
                        opTranspose(ins[0], attrs.perm);
                }

                else if (op === "Unsqueeze") {
                    env[node.outputs[0]] =
                        opUnsqueeze(ins[0], attrs.axes || []);
                }

                else if (op === "Squeeze") {
                    env[node.outputs[0]] =
                        opSqueeze(ins[0], attrs.axes || []);
                }

                else if (op === "Concat") {
                    env[node.outputs[0]] =
                        opConcat(ins, attrs.axis);
                }

                else if (op === "Gather") {
                    env[node.outputs[0]] =
                        opGather(ins[0], ins[1], attrs.axis || 0);
                }

                else if (op === "Slice") {
                    env[node.outputs[0]] = opSlice(
                        ins[0],
                        ins[1],
                        ins[2],
                        ins[3],
                        ins[4]
                    );
                }

                else if (op === "Split") {
                    outs = opSplit(
                        ins[0],
                        attrs.axis || 0,
                        attrs.split || null,
                        node.outputs.length
                    );
                    for (i = 0; i < node.outputs.length; ++i) {
                        env[node.outputs[i]] = outs[i];
                    }
                }

                else if (op === "ReduceMean") {
                    env[node.outputs[0]] = opReduceMean(
                        ins[0],
                        attrs.axes || [],
                        attrs.keepdims === undefined ? 1 : attrs.keepdims
                    );
                }

                else if (op === "Softmax") {
                    env[node.outputs[0]] =
                        opSoftmax(
                            ins[0],
                            attrs.axis === undefined ? -1 : attrs.axis
                        );
                }

                else if (op === "BatchNormalization") {
                    env[node.outputs[0]] = opBatchNormalization(
                        ins[0],
                        ins[1],
                        ins[2],
                        ins[3],
                        ins[4],
                        attrs.epsilon === undefined ?
                            1e-5 : attrs.epsilon
                    );
                }

                else if (op === "Pad") {
                    if (
                        attrs.mode !== undefined &&
                        attrs.mode !== "constant"
                    ) {
                        throw new Error(
                            "Pad mode unsupported: " + attrs.mode
                        );
                    }
                    env[node.outputs[0]] =
                        opPadConstant(ins[0], ins[1], ins[2]);
                }

                else if (op === "AveragePool") {
                    env[node.outputs[0]] =
                        opAveragePoolNCHW(ins[0], attrs);
                }

                else {
                    throw new Error("unsupported operator: " + op);
                }
            }
            catch (e) {
                e.message =
                    "node " + ni + "/" + plan.nodes.length +
                    " op=" + op +
                    " name=" + node.name +
                    ": " + e.message;
                throw e;
            }

            elapsed = Date.now() - t0;
            opTimes[op] = (opTimes[op] || 0) + elapsed;
            opCalls[op] = (opCalls[op] || 0) + 1;

            for (j = 0; j < node.inputs.length; ++j) {
                consume(node.inputs[j]);
            }

            if (
                options.progressEvery &&
                (
                    (ni + 1) % options.progressEvery === 0 ||
                    ni + 1 === plan.nodes.length
                )
            ) {
                if (options.onProgress) {
                    options.onProgress(ni + 1, plan.nodes.length);
                } else if (
                    typeof console !== "undefined" &&
                    console.log
                ) {
                    console.log(
                        "progress:",
                        (ni + 1) + "/" + plan.nodes.length
                    );
                }
            }
        }

        result = {};
        for (name in outputNames) {
            if (outputNames.hasOwnProperty(name)) {
                result[name] = env[name];
            }
        }

        result.__profile = {
            times: opTimes,
            calls: opCalls
        };

        return result;
    }

    return {
        run: run,
        Tensor: Tensor
    };
}

module.exports = {
    createExecutor: createExecutor,
    Tensor: Tensor
};

};
deps[1] = {'./mobileclip-fp32-kernels-v1.js':2,'./mobileclip-fp32-kernels-opt-v2.js':3};
modules[2] = function (module, exports, require) {
"use strict";

/*
 * MobileCLIP-S0 FP32 pure-JavaScript reference kernels v1.
 *
 * Deliberately conservative / ES5-style:
 *   - no WASM
 *   - no WebGL/WebGPU
 *   - no BigInt
 *   - no async/await
 *   - no ES modules
 *
 * Correctness-first. Optimization comes only after the FP32 path
 * is numerically and semantically validated.
 */

function product(shape, begin, end) {
    var r = 1;
    var i;
    begin = begin === undefined ? 0 : begin;
    end = end === undefined ? shape.length : end;
    for (i = begin; i < end; ++i) {
        r *= shape[i];
    }
    return r;
}

function computeStrides(shape) {
    var s = new Array(shape.length);
    var p = 1;
    var i;
    for (i = shape.length - 1; i >= 0; --i) {
        s[i] = p;
        p *= shape[i];
    }
    return s;
}

function normalizeAxis(axis, rank) {
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw new Error("axis out of range: " + axis + " rank=" + rank);
    }
    return axis;
}

function sameShape(a, b) {
    var i;
    if (a.length !== b.length) {
        return false;
    }
    for (i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

function broadcastShape(a, b) {
    var rank = Math.max(a.length, b.length);
    var out = new Array(rank);
    var i;
    var av;
    var bv;
    for (i = 0; i < rank; ++i) {
        av = a[a.length - rank + i];
        bv = b[b.length - rank + i];
        av = av === undefined ? 1 : av;
        bv = bv === undefined ? 1 : bv;
        if (av !== bv && av !== 1 && bv !== 1) {
            throw new Error(
                "broadcast mismatch: " +
                JSON.stringify(a) + " vs " + JSON.stringify(b)
            );
        }
        out[i] = Math.max(av, bv);
    }
    return out;
}

function binaryF32(a, aShape, b, bShape, fn) {
    var outShape = broadcastShape(aShape, bShape);
    var out = new Float32Array(product(outShape));
    var rank = outShape.length;
    var aPad = new Array(rank);
    var bPad = new Array(rank);
    var outS = computeStrides(outShape);
    var aS;
    var bS;
    var i;
    var d;
    var t;
    var coord;
    var ai;
    var bi;

    for (i = 0; i < rank; ++i) {
        aPad[i] = aShape[aShape.length - rank + i];
        bPad[i] = bShape[bShape.length - rank + i];
        if (aPad[i] === undefined) {
            aPad[i] = 1;
        }
        if (bPad[i] === undefined) {
            bPad[i] = 1;
        }
    }
    aS = computeStrides(aPad);
    bS = computeStrides(bPad);

    if (sameShape(aShape, bShape)) {
        for (i = 0; i < out.length; ++i) {
            out[i] = fn(a[i], b[i]);
        }
        return {data: out, shape: outShape};
    }

    if (a.length === 1) {
        for (i = 0; i < out.length; ++i) {
            out[i] = fn(a[0], b[i]);
        }
        return {data: out, shape: outShape};
    }

    if (b.length === 1) {
        for (i = 0; i < out.length; ++i) {
            out[i] = fn(a[i], b[0]);
        }
        return {data: out, shape: outShape};
    }

    for (i = 0; i < out.length; ++i) {
        t = i;
        ai = 0;
        bi = 0;
        for (d = 0; d < rank; ++d) {
            coord = Math.floor(t / outS[d]);
            t -= coord * outS[d];
            if (aPad[d] !== 1) {
                ai += coord * aS[d];
            }
            if (bPad[d] !== 1) {
                bi += coord * bS[d];
            }
        }
        out[i] = fn(a[ai], b[bi]);
    }

    return {data: out, shape: outShape};
}

/*
 * High-accuracy erf approximation from Numerical Recipes.
 * Absolute error is ~1e-7 over the range relevant to GELU.
 */
function erfApprox(x) {
    var sign = x < 0 ? -1.0 : 1.0;
    var a = Math.abs(x);
    var t = 1.0 / (1.0 + 0.5 * a);
    var p = 0.17087277;
    p = -0.82215223 + t * p;
    p = 1.48851587 + t * p;
    p = -1.13520398 + t * p;
    p = 0.27886807 + t * p;
    p = -0.18628806 + t * p;
    p = 0.09678418 + t * p;
    p = 0.37409196 + t * p;
    p = 1.00002368 + t * p;
    var tau = t * Math.exp(-a * a - 1.26551223 + t * p);
    return sign * (1.0 - tau);
}

function sigmoid(x) {
    if (x >= 0) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    var e = Math.exp(x);
    return e / (1.0 + e);
}

function convPointwiseNCHW(x, w, bias, xShape, wShape) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var P = H * W;
    var out = new Float32Array(N * M * P);
    var n;
    var m;
    var p;
    var c;
    var m0;
    var yBase;
    var xBase;
    var w0;
    var w1;
    var w2;
    var w3;
    var o0;
    var o1;
    var o2;
    var o3;
    var a0;
    var a1;
    var a2;
    var a3;
    var xv;

    for (n = 0; n < N; ++n) {
        xBase = n * C * P;
        yBase = n * M * P;

        for (m0 = 0; m0 + 3 < M; m0 += 4) {
            w0 = m0 * C;
            w1 = (m0 + 1) * C;
            w2 = (m0 + 2) * C;
            w3 = (m0 + 3) * C;

            o0 = yBase + m0 * P;
            o1 = o0 + P;
            o2 = o1 + P;
            o3 = o2 + P;

            for (p = 0; p < P; ++p) {
                a0 = bias[m0];
                a1 = bias[m0 + 1];
                a2 = bias[m0 + 2];
                a3 = bias[m0 + 3];

                for (c = 0; c < C; ++c) {
                    xv = x[xBase + c * P + p];
                    a0 += xv * w[w0 + c];
                    a1 += xv * w[w1 + c];
                    a2 += xv * w[w2 + c];
                    a3 += xv * w[w3 + c];
                }

                out[o0 + p] = a0;
                out[o1 + p] = a1;
                out[o2 + p] = a2;
                out[o3 + p] = a3;
            }
        }

        for (m = m0; m < M; ++m) {
            o0 = yBase + m * P;
            w0 = m * C;
            for (p = 0; p < P; ++p) {
                a0 = bias[m];
                for (c = 0; c < C; ++c) {
                    a0 += x[xBase + c * P + p] * w[w0 + c];
                }
                out[o0 + p] = a0;
            }
        }
    }

    return {
        data: out,
        shape: [N, M, H, W]
    };
}

function convDepthwiseNCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var KH = wShape[2];
    var KW = wShape[3];
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var sh = strides[0];
    var sw = strides[1];
    var dh = dilations[0];
    var dw = dilations[1];
    var OH = Math.floor((H + pt + pb - dh * (KH - 1) - 1) / sh) + 1;
    var OW = Math.floor((W + pl + pr - dw * (KW - 1) - 1) / sw) + 1;
    var out = new Float32Array(N * M * OH * OW);
    var image = H * W;
    var outImage = OH * OW;
    var n;
    var c;
    var oy;
    var ox;
    var ky;
    var kx;
    var iy;
    var ix;
    var a;
    var xb;
    var yb;
    var wb;

    if (M !== C || attrs.group !== C) {
        throw new Error("depthwise contract mismatch");
    }

    for (n = 0; n < N; ++n) {
        for (c = 0; c < C; ++c) {
            xb = (n * C + c) * image;
            yb = (n * C + c) * outImage;
            wb = c * KH * KW;

            for (oy = 0; oy < OH; ++oy) {
                for (ox = 0; ox < OW; ++ox) {
                    a = bias[c];
                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky * dh;
                        if (iy < 0 || iy >= H) {
                            continue;
                        }
                        for (kx = 0; kx < KW; ++kx) {
                            ix = ox * sw - pl + kx * dw;
                            if (ix < 0 || ix >= W) {
                                continue;
                            }
                            a += x[xb + iy * W + ix] *
                                 w[wb + ky * KW + kx];
                        }
                    }
                    out[yb + oy * OW + ox] = a;
                }
            }
        }
    }

    return {
        data: out,
        shape: [N, M, OH, OW]
    };
}

function convGroupedNCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var Cg = wShape[1];
    var KH = wShape[2];
    var KW = wShape[3];
    var group = attrs.group || 1;
    var Mg;
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var sh = strides[0];
    var sw = strides[1];
    var dh = dilations[0];
    var dw = dilations[1];
    var OH = Math.floor((H + pt + pb - dh * (KH - 1) - 1) / sh) + 1;
    var OW = Math.floor((W + pl + pr - dw * (KW - 1) - 1) / sw) + 1;
    var out = new Float32Array(N * M * OH * OW);
    var n;
    var g;
    var om;
    var ic;
    var oy;
    var ox;
    var ky;
    var kx;
    var iy;
    var ix;
    var oc;
    var inC;
    var a;
    var xo;
    var wo;
    var yo;

    if (C !== Cg * group || M % group !== 0) {
        throw new Error(
            "grouped Conv contract mismatch C=" + C +
            " Cg=" + Cg + " M=" + M + " group=" + group
        );
    }

    Mg = M / group;

    for (n = 0; n < N; ++n) {
        for (g = 0; g < group; ++g) {
            for (om = 0; om < Mg; ++om) {
                oc = g * Mg + om;
                for (oy = 0; oy < OH; ++oy) {
                    for (ox = 0; ox < OW; ++ox) {
                        a = bias[oc];

                        for (ic = 0; ic < Cg; ++ic) {
                            inC = g * Cg + ic;

                            for (ky = 0; ky < KH; ++ky) {
                                iy = oy * sh - pt + ky * dh;
                                if (iy < 0 || iy >= H) {
                                    continue;
                                }

                                for (kx = 0; kx < KW; ++kx) {
                                    ix = ox * sw - pl + kx * dw;
                                    if (ix < 0 || ix >= W) {
                                        continue;
                                    }

                                    xo = ((n * C + inC) * H + iy) * W + ix;
                                    wo = (((oc * Cg + ic) * KH + ky) * KW + kx);
                                    a += x[xo] * w[wo];
                                }
                            }
                        }

                        yo = ((n * M + oc) * OH + oy) * OW + ox;
                        out[yo] = a;
                    }
                }
            }
        }
    }

    return {
        data: out,
        shape: [N, M, OH, OW]
    };
}

function matMul(a, aShape, b, bShape) {
    var ar = aShape.length;
    var br = bShape.length;
    var M;
    var K;
    var K2;
    var N;
    var aBatch;
    var bBatch;
    var batchShape;
    var batchCount;
    var aPad;
    var bPad;
    var batchRank;
    var aBatchShape;
    var bBatchShape;
    var out;
    var outShape;
    var aBatchS;
    var bBatchS;
    var outBatchS;
    var bi;
    var d;
    var t;
    var coord;
    var aBatchIndex;
    var bBatchIndex;
    var aBase;
    var bBase;
    var outBase;
    var m;
    var n;
    var k;
    var sum;

    if (ar < 2 || br < 2) {
        throw new Error("MatMul rank<2 not implemented");
    }

    M = aShape[ar - 2];
    K = aShape[ar - 1];
    K2 = bShape[br - 2];
    N = bShape[br - 1];

    if (K !== K2) {
        throw new Error("MatMul K mismatch: " + K + " vs " + K2);
    }

    aBatch = aShape.slice(0, ar - 2);
    bBatch = bShape.slice(0, br - 2);
    batchShape = broadcastShape(aBatch, bBatch);
    batchRank = batchShape.length;
    batchCount = product(batchShape);

    aBatchShape = new Array(batchRank);
    bBatchShape = new Array(batchRank);

    for (d = 0; d < batchRank; ++d) {
        aBatchShape[d] = aBatch[aBatch.length - batchRank + d];
        bBatchShape[d] = bBatch[bBatch.length - batchRank + d];
        if (aBatchShape[d] === undefined) {
            aBatchShape[d] = 1;
        }
        if (bBatchShape[d] === undefined) {
            bBatchShape[d] = 1;
        }
    }

    aBatchS = computeStrides(aBatchShape);
    bBatchS = computeStrides(bBatchShape);
    outBatchS = computeStrides(batchShape);

    outShape = batchShape.concat([M, N]);
    out = new Float32Array(batchCount * M * N);

    for (bi = 0; bi < batchCount; ++bi) {
        t = bi;
        aBatchIndex = 0;
        bBatchIndex = 0;

        for (d = 0; d < batchRank; ++d) {
            coord = Math.floor(t / outBatchS[d]);
            t -= coord * outBatchS[d];

            if (aBatchShape[d] !== 1) {
                aBatchIndex += coord * aBatchS[d];
            }
            if (bBatchShape[d] !== 1) {
                bBatchIndex += coord * bBatchS[d];
            }
        }

        aBase = aBatchIndex * M * K;
        bBase = bBatchIndex * K * N;
        outBase = bi * M * N;

        for (m = 0; m < M; ++m) {
            for (n = 0; n < N; ++n) {
                sum = 0.0;
                for (k = 0; k < K; ++k) {
                    sum += a[aBase + m * K + k] *
                           b[bBase + k * N + n];
                }
                out[outBase + m * N + n] = sum;
            }
        }
    }

    return {
        data: out,
        shape: outShape
    };
}

module.exports = {
    product: product,
    computeStrides: computeStrides,
    normalizeAxis: normalizeAxis,
    broadcastShape: broadcastShape,
    binaryF32: binaryF32,
    erfApprox: erfApprox,
    sigmoid: sigmoid,
    convPointwiseNCHW: convPointwiseNCHW,
    convDepthwiseNCHW: convDepthwiseNCHW,
    convGroupedNCHW: convGroupedNCHW,
    matMul: matMul
};

};
deps[2] = {};
modules[3] = function (module, exports, require) {
"use strict";

var Base = require("./mobileclip-fp32-kernels-v1.js");


function convPointwiseB4T8NCHW(x, w, bias, xShape, wShape) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var P = H * W;
    var out = new Float32Array(N * M * P);
    var n, m0, m, p0, p, c, xb, xBase, yBase;
    var w0;
    var o0;
    var a0_0;
    var a0_1;
    var a0_2;
    var a0_3;
    var a0_4;
    var a0_5;
    var a0_6;
    var a0_7;
    var w1;
    var o1;
    var a1_0;
    var a1_1;
    var a1_2;
    var a1_3;
    var a1_4;
    var a1_5;
    var a1_6;
    var a1_7;
    var w2;
    var o2;
    var a2_0;
    var a2_1;
    var a2_2;
    var a2_3;
    var a2_4;
    var a2_5;
    var a2_6;
    var a2_7;
    var w3;
    var o3;
    var a3_0;
    var a3_1;
    var a3_2;
    var a3_3;
    var a3_4;
    var a3_5;
    var a3_6;
    var a3_7;
    var x0;
    var x1;
    var x2;
    var x3;
    var x4;
    var x5;
    var x6;
    var x7;
    var acc;

    for (n = 0; n < N; ++n) {
        xBase = n * C * P;
        yBase = n * M * P;

        for (m0 = 0; m0 + 3 < M; m0 += 4) {
            o0 = yBase + (m0 + 0) * P;
            o1 = yBase + (m0 + 1) * P;
            o2 = yBase + (m0 + 2) * P;
            o3 = yBase + (m0 + 3) * P;

            p0 = 0;

            for (; p0 + 7 < P; p0 += 8) {
                a0_0 = bias[m0 + 0];
                a0_1 = bias[m0 + 0];
                a0_2 = bias[m0 + 0];
                a0_3 = bias[m0 + 0];
                a0_4 = bias[m0 + 0];
                a0_5 = bias[m0 + 0];
                a0_6 = bias[m0 + 0];
                a0_7 = bias[m0 + 0];
                a1_0 = bias[m0 + 1];
                a1_1 = bias[m0 + 1];
                a1_2 = bias[m0 + 1];
                a1_3 = bias[m0 + 1];
                a1_4 = bias[m0 + 1];
                a1_5 = bias[m0 + 1];
                a1_6 = bias[m0 + 1];
                a1_7 = bias[m0 + 1];
                a2_0 = bias[m0 + 2];
                a2_1 = bias[m0 + 2];
                a2_2 = bias[m0 + 2];
                a2_3 = bias[m0 + 2];
                a2_4 = bias[m0 + 2];
                a2_5 = bias[m0 + 2];
                a2_6 = bias[m0 + 2];
                a2_7 = bias[m0 + 2];
                a3_0 = bias[m0 + 3];
                a3_1 = bias[m0 + 3];
                a3_2 = bias[m0 + 3];
                a3_3 = bias[m0 + 3];
                a3_4 = bias[m0 + 3];
                a3_5 = bias[m0 + 3];
                a3_6 = bias[m0 + 3];
                a3_7 = bias[m0 + 3];
                for (c = 0; c < C; ++c) {
                    xb = xBase + c * P + p0;
                    x0 = x[xb + 0];
                    x1 = x[xb + 1];
                    x2 = x[xb + 2];
                    x3 = x[xb + 3];
                    x4 = x[xb + 4];
                    x5 = x[xb + 5];
                    x6 = x[xb + 6];
                    x7 = x[xb + 7];
                    w0 = w[(m0 + 0) * C + c];
                    a0_0 += x0 * w0;
                    a0_1 += x1 * w0;
                    a0_2 += x2 * w0;
                    a0_3 += x3 * w0;
                    a0_4 += x4 * w0;
                    a0_5 += x5 * w0;
                    a0_6 += x6 * w0;
                    a0_7 += x7 * w0;
                    w1 = w[(m0 + 1) * C + c];
                    a1_0 += x0 * w1;
                    a1_1 += x1 * w1;
                    a1_2 += x2 * w1;
                    a1_3 += x3 * w1;
                    a1_4 += x4 * w1;
                    a1_5 += x5 * w1;
                    a1_6 += x6 * w1;
                    a1_7 += x7 * w1;
                    w2 = w[(m0 + 2) * C + c];
                    a2_0 += x0 * w2;
                    a2_1 += x1 * w2;
                    a2_2 += x2 * w2;
                    a2_3 += x3 * w2;
                    a2_4 += x4 * w2;
                    a2_5 += x5 * w2;
                    a2_6 += x6 * w2;
                    a2_7 += x7 * w2;
                    w3 = w[(m0 + 3) * C + c];
                    a3_0 += x0 * w3;
                    a3_1 += x1 * w3;
                    a3_2 += x2 * w3;
                    a3_3 += x3 * w3;
                    a3_4 += x4 * w3;
                    a3_5 += x5 * w3;
                    a3_6 += x6 * w3;
                    a3_7 += x7 * w3;
                }
                out[o0 + p0 + 0] = a0_0;
                out[o0 + p0 + 1] = a0_1;
                out[o0 + p0 + 2] = a0_2;
                out[o0 + p0 + 3] = a0_3;
                out[o0 + p0 + 4] = a0_4;
                out[o0 + p0 + 5] = a0_5;
                out[o0 + p0 + 6] = a0_6;
                out[o0 + p0 + 7] = a0_7;
                out[o1 + p0 + 0] = a1_0;
                out[o1 + p0 + 1] = a1_1;
                out[o1 + p0 + 2] = a1_2;
                out[o1 + p0 + 3] = a1_3;
                out[o1 + p0 + 4] = a1_4;
                out[o1 + p0 + 5] = a1_5;
                out[o1 + p0 + 6] = a1_6;
                out[o1 + p0 + 7] = a1_7;
                out[o2 + p0 + 0] = a2_0;
                out[o2 + p0 + 1] = a2_1;
                out[o2 + p0 + 2] = a2_2;
                out[o2 + p0 + 3] = a2_3;
                out[o2 + p0 + 4] = a2_4;
                out[o2 + p0 + 5] = a2_5;
                out[o2 + p0 + 6] = a2_6;
                out[o2 + p0 + 7] = a2_7;
                out[o3 + p0 + 0] = a3_0;
                out[o3 + p0 + 1] = a3_1;
                out[o3 + p0 + 2] = a3_2;
                out[o3 + p0 + 3] = a3_3;
                out[o3 + p0 + 4] = a3_4;
                out[o3 + p0 + 5] = a3_5;
                out[o3 + p0 + 6] = a3_6;
                out[o3 + p0 + 7] = a3_7;
            }

            for (; p0 < P; ++p0) {
                a0_0 = bias[m0 + 0];
                a1_0 = bias[m0 + 1];
                a2_0 = bias[m0 + 2];
                a3_0 = bias[m0 + 3];

                for (c = 0; c < C; ++c) {
                    x0 = x[xBase + c * P + p0];
                    a0_0 += x0 * w[(m0 + 0) * C + c];
                    a1_0 += x0 * w[(m0 + 1) * C + c];
                    a2_0 += x0 * w[(m0 + 2) * C + c];
                    a3_0 += x0 * w[(m0 + 3) * C + c];
                }

                out[o0 + p0] = a0_0;
                out[o1 + p0] = a1_0;
                out[o2 + p0] = a2_0;
                out[o3 + p0] = a3_0;
            }
        }

        for (m = m0; m < M; ++m) {
            o0 = yBase + m * P;
            for (p = 0; p < P; ++p) {
                acc = bias[m];
                for (c = 0; c < C; ++c) {
                    acc += x[xBase + c * P + p] * w[m * C + c];
                }
                out[o0 + p] = acc;
            }
        }
    }

    return {
        data: out,
        shape: [N, M, H, W]
    };
}


function convDepthwiseDW8NCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var KH = wShape[2];
    var KW = wShape[3];
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var sh = strides[0];
    var sw = strides[1];
    var dh = dilations[0];
    var dw = dilations[1];
    var OH = Math.floor((H + pt + pb - dh * (KH - 1) - 1) / sh) + 1;
    var OW = Math.floor((W + pl + pr - dw * (KW - 1) - 1) / sw) + 1;
    var out = new Float32Array(N * M * OH * OW);
    var image = H * W;
    var outImage = OH * OW;
    var n, c, oy, ox, ky, kx, iy, xb, yb, wb, wv, acc, ix;
    var a0;
    var a1;
    var a2;
    var a3;
    var a4;
    var a5;
    var a6;
    var a7;
    var ix0;
    var ix1;
    var ix2;
    var ix3;
    var ix4;
    var ix5;
    var ix6;
    var ix7;

    if (M !== C || attrs.group !== C) {
        return Base.convDepthwiseNCHW(
            x, w, bias, xShape, wShape, attrs
        );
    }

    for (n = 0; n < N; ++n) {
        for (c = 0; c < C; ++c) {
            xb = (n * C + c) * image;
            yb = (n * C + c) * outImage;
            wb = c * KH * KW;

            for (oy = 0; oy < OH; ++oy) {
                ox = 0;

                for (; ox + 7 < OW; ox += 8) {
                a0 = bias[c];
                a1 = bias[c];
                a2 = bias[c];
                a3 = bias[c];
                a4 = bias[c];
                a5 = bias[c];
                a6 = bias[c];
                a7 = bias[c];
                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky * dh;
                        if (iy < 0 || iy >= H) {
                            continue;
                        }

                        for (kx = 0; kx < KW; ++kx) {
                            wv = w[wb + ky * KW + kx];
                        ix0 = (ox + 0) * sw - pl + kx * dw;
                        ix1 = (ox + 1) * sw - pl + kx * dw;
                        ix2 = (ox + 2) * sw - pl + kx * dw;
                        ix3 = (ox + 3) * sw - pl + kx * dw;
                        ix4 = (ox + 4) * sw - pl + kx * dw;
                        ix5 = (ox + 5) * sw - pl + kx * dw;
                        ix6 = (ox + 6) * sw - pl + kx * dw;
                        ix7 = (ox + 7) * sw - pl + kx * dw;
                        if (ix0 >= 0 && ix0 < W) { a0 += x[xb + iy * W + ix0] * wv; }
                        if (ix1 >= 0 && ix1 < W) { a1 += x[xb + iy * W + ix1] * wv; }
                        if (ix2 >= 0 && ix2 < W) { a2 += x[xb + iy * W + ix2] * wv; }
                        if (ix3 >= 0 && ix3 < W) { a3 += x[xb + iy * W + ix3] * wv; }
                        if (ix4 >= 0 && ix4 < W) { a4 += x[xb + iy * W + ix4] * wv; }
                        if (ix5 >= 0 && ix5 < W) { a5 += x[xb + iy * W + ix5] * wv; }
                        if (ix6 >= 0 && ix6 < W) { a6 += x[xb + iy * W + ix6] * wv; }
                        if (ix7 >= 0 && ix7 < W) { a7 += x[xb + iy * W + ix7] * wv; }
                        }
                    }
                out[yb + oy * OW + ox + 0] = a0;
                out[yb + oy * OW + ox + 1] = a1;
                out[yb + oy * OW + ox + 2] = a2;
                out[yb + oy * OW + ox + 3] = a3;
                out[yb + oy * OW + ox + 4] = a4;
                out[yb + oy * OW + ox + 5] = a5;
                out[yb + oy * OW + ox + 6] = a6;
                out[yb + oy * OW + ox + 7] = a7;
                }

                for (; ox < OW; ++ox) {
                    acc = bias[c];

                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky * dh;
                        if (iy < 0 || iy >= H) {
                            continue;
                        }

                        for (kx = 0; kx < KW; ++kx) {
                            ix = ox * sw - pl + kx * dw;
                            if (ix < 0 || ix >= W) {
                                continue;
                            }

                            acc +=
                                x[xb + iy * W + ix] *
                                w[wb + ky * KW + kx];
                        }
                    }

                    out[yb + oy * OW + ox] = acc;
                }
            }
        }
    }

    return {
        data: out,
        shape: [N, M, OH, OW]
    };
}


function convGroupedM2T8NCHW(x, w, bias, xShape, wShape, attrs) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var Cg = wShape[1];
    var KH = wShape[2];
    var KW = wShape[3];
    var group = attrs.group || 1;
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var pt = pads[0];
    var pl = pads[1];
    var pb = pads[2];
    var pr = pads[3];
    var sh = strides[0];
    var sw = strides[1];
    var dh = dilations[0];
    var dw = dilations[1];
    var OH = Math.floor((H + pt + pb - dh * (KH - 1) - 1) / sh) + 1;
    var OW = Math.floor((W + pl + pr - dw * (KW - 1) - 1) / sw) + 1;
    var out = new Float32Array(N * M * OH * OW);
    var image = H * W;
    var outImage = OH * OW;
    var n, c, oc0, oc1, oy, ox, ky, kx, iy, ix;
    var xb, yb0, yb1, wb0, wb1, w0, w1, xv;
    var acc0, acc1;
    var a0_0;
    var a0_1;
    var a0_2;
    var a0_3;
    var a0_4;
    var a0_5;
    var a0_6;
    var a0_7;
    var a1_0;
    var a1_1;
    var a1_2;
    var a1_3;
    var a1_4;
    var a1_5;
    var a1_6;
    var a1_7;
    var ix0;
    var ix1;
    var ix2;
    var ix3;
    var ix4;
    var ix5;
    var ix6;
    var ix7;

    if (
        Cg !== 1 ||
        group !== C ||
        M !== C * 2
    ) {
        return Base.convGroupedNCHW(
            x, w, bias, xShape, wShape, attrs
        );
    }

    for (n = 0; n < N; ++n) {
        for (c = 0; c < C; ++c) {
            oc0 = c * 2;
            oc1 = oc0 + 1;
            xb = (n * C + c) * image;
            yb0 = (n * M + oc0) * outImage;
            yb1 = (n * M + oc1) * outImage;
            wb0 = oc0 * KH * KW;
            wb1 = oc1 * KH * KW;

            for (oy = 0; oy < OH; ++oy) {
                ox = 0;

                for (; ox + 7 < OW; ox += 8) {
                a0_0 = bias[oc0];
                a1_0 = bias[oc1];
                a0_1 = bias[oc0];
                a1_1 = bias[oc1];
                a0_2 = bias[oc0];
                a1_2 = bias[oc1];
                a0_3 = bias[oc0];
                a1_3 = bias[oc1];
                a0_4 = bias[oc0];
                a1_4 = bias[oc1];
                a0_5 = bias[oc0];
                a1_5 = bias[oc1];
                a0_6 = bias[oc0];
                a1_6 = bias[oc1];
                a0_7 = bias[oc0];
                a1_7 = bias[oc1];

                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky * dh;

                        if (iy < 0 || iy >= H) {
                            continue;
                        }

                        for (kx = 0; kx < KW; ++kx) {
                            w0 = w[wb0 + ky * KW + kx];
                            w1 = w[wb1 + ky * KW + kx];

                        ix0 = (ox + 0) * sw - pl + kx * dw;
                        ix1 = (ox + 1) * sw - pl + kx * dw;
                        ix2 = (ox + 2) * sw - pl + kx * dw;
                        ix3 = (ox + 3) * sw - pl + kx * dw;
                        ix4 = (ox + 4) * sw - pl + kx * dw;
                        ix5 = (ox + 5) * sw - pl + kx * dw;
                        ix6 = (ox + 6) * sw - pl + kx * dw;
                        ix7 = (ox + 7) * sw - pl + kx * dw;
                        if (ix0 >= 0 && ix0 < W) { xv = x[xb + iy * W + ix0]; a0_0 += xv * w0; a1_0 += xv * w1; }
                        if (ix1 >= 0 && ix1 < W) { xv = x[xb + iy * W + ix1]; a0_1 += xv * w0; a1_1 += xv * w1; }
                        if (ix2 >= 0 && ix2 < W) { xv = x[xb + iy * W + ix2]; a0_2 += xv * w0; a1_2 += xv * w1; }
                        if (ix3 >= 0 && ix3 < W) { xv = x[xb + iy * W + ix3]; a0_3 += xv * w0; a1_3 += xv * w1; }
                        if (ix4 >= 0 && ix4 < W) { xv = x[xb + iy * W + ix4]; a0_4 += xv * w0; a1_4 += xv * w1; }
                        if (ix5 >= 0 && ix5 < W) { xv = x[xb + iy * W + ix5]; a0_5 += xv * w0; a1_5 += xv * w1; }
                        if (ix6 >= 0 && ix6 < W) { xv = x[xb + iy * W + ix6]; a0_6 += xv * w0; a1_6 += xv * w1; }
                        if (ix7 >= 0 && ix7 < W) { xv = x[xb + iy * W + ix7]; a0_7 += xv * w0; a1_7 += xv * w1; }
                        }
                    }

                out[yb0 + oy * OW + ox + 0] = a0_0;
                out[yb1 + oy * OW + ox + 0] = a1_0;
                out[yb0 + oy * OW + ox + 1] = a0_1;
                out[yb1 + oy * OW + ox + 1] = a1_1;
                out[yb0 + oy * OW + ox + 2] = a0_2;
                out[yb1 + oy * OW + ox + 2] = a1_2;
                out[yb0 + oy * OW + ox + 3] = a0_3;
                out[yb1 + oy * OW + ox + 3] = a1_3;
                out[yb0 + oy * OW + ox + 4] = a0_4;
                out[yb1 + oy * OW + ox + 4] = a1_4;
                out[yb0 + oy * OW + ox + 5] = a0_5;
                out[yb1 + oy * OW + ox + 5] = a1_5;
                out[yb0 + oy * OW + ox + 6] = a0_6;
                out[yb1 + oy * OW + ox + 6] = a1_6;
                out[yb0 + oy * OW + ox + 7] = a0_7;
                out[yb1 + oy * OW + ox + 7] = a1_7;
                }

                for (; ox < OW; ++ox) {
                    acc0 = bias[oc0];
                    acc1 = bias[oc1];

                    for (ky = 0; ky < KH; ++ky) {
                        iy = oy * sh - pt + ky * dh;

                        if (iy < 0 || iy >= H) {
                            continue;
                        }

                        for (kx = 0; kx < KW; ++kx) {
                            ix = ox * sw - pl + kx * dw;

                            if (ix < 0 || ix >= W) {
                                continue;
                            }

                            xv = x[xb + iy * W + ix];

                            acc0 +=
                                xv *
                                w[wb0 + ky * KW + kx];

                            acc1 +=
                                xv *
                                w[wb1 + ky * KW + kx];
                        }
                    }

                    out[yb0 + oy * OW + ox] = acc0;
                    out[yb1 + oy * OW + ox] = acc1;
                }
            }
        }
    }

    return {
        data: out,
        shape: [N, M, OH, OW]
    };
}


function convInitial3x3s2B8T4NCHW(
    x, w, bias, xShape, wShape, attrs
) {
    var N = xShape[0];
    var C = xShape[1];
    var H = xShape[2];
    var W = xShape[3];
    var M = wShape[0];
    var KH = wShape[2];
    var KW = wShape[3];
    var pads = attrs.pads || [0, 0, 0, 0];
    var strides = attrs.strides || [1, 1];
    var dilations = attrs.dilations || [1, 1];
    var group = attrs.group || 1;
    var OH = 128;
    var OW = 128;
    var outS = OH * OW;
    var inS = 256 * 256;
    var out = new Float32Array(64 * outS);
    var oc, oh, ow, ic, kh, kx, q;
    var ih, iw, xBase, outBase, col, row, acc;
    var w0;
    var o0;
    var a0_0;
    var a0_1;
    var a0_2;
    var a0_3;
    var w1;
    var o1;
    var a1_0;
    var a1_1;
    var a1_2;
    var a1_3;
    var w2;
    var o2;
    var a2_0;
    var a2_1;
    var a2_2;
    var a2_3;
    var w3;
    var o3;
    var a3_0;
    var a3_1;
    var a3_2;
    var a3_3;
    var w4;
    var o4;
    var a4_0;
    var a4_1;
    var a4_2;
    var a4_3;
    var w5;
    var o5;
    var a5_0;
    var a5_1;
    var a5_2;
    var a5_3;
    var w6;
    var o6;
    var a6_0;
    var a6_1;
    var a6_2;
    var a6_3;
    var w7;
    var o7;
    var a7_0;
    var a7_1;
    var a7_2;
    var a7_3;
    var x0;
    var x1;
    var x2;
    var x3;

    if (
        N !== 1 ||
        C !== 3 ||
        H !== 256 ||
        W !== 256 ||
        M !== 64 ||
        KH !== 3 ||
        KW !== 3 ||
        group !== 1 ||
        pads[0] !== 1 ||
        pads[1] !== 1 ||
        pads[2] !== 1 ||
        pads[3] !== 1 ||
        strides[0] !== 2 ||
        strides[1] !== 2 ||
        dilations[0] !== 1 ||
        dilations[1] !== 1
    ) {
        return Base.convGroupedNCHW(
            x, w, bias, xShape, wShape, attrs
        );
    }

    for (oc = 0; oc < 64; oc += 8) {
        o0 = (oc + 0) * outS;
        o1 = (oc + 1) * outS;
        o2 = (oc + 2) * outS;
        o3 = (oc + 3) * outS;
        o4 = (oc + 4) * outS;
        o5 = (oc + 5) * outS;
        o6 = (oc + 6) * outS;
        o7 = (oc + 7) * outS;

        for (oh = 0; oh < OH; ++oh) {
            ih = oh * 2 - 1;

            for (ow = 0; ow < OW; ow += 4) {
                iw = ow * 2 - 1;
                outBase = oh * OW + ow;
                a0_0 = bias[oc + 0];
                a0_1 = bias[oc + 0];
                a0_2 = bias[oc + 0];
                a0_3 = bias[oc + 0];
                a1_0 = bias[oc + 1];
                a1_1 = bias[oc + 1];
                a1_2 = bias[oc + 1];
                a1_3 = bias[oc + 1];
                a2_0 = bias[oc + 2];
                a2_1 = bias[oc + 2];
                a2_2 = bias[oc + 2];
                a2_3 = bias[oc + 2];
                a3_0 = bias[oc + 3];
                a3_1 = bias[oc + 3];
                a3_2 = bias[oc + 3];
                a3_3 = bias[oc + 3];
                a4_0 = bias[oc + 4];
                a4_1 = bias[oc + 4];
                a4_2 = bias[oc + 4];
                a4_3 = bias[oc + 4];
                a5_0 = bias[oc + 5];
                a5_1 = bias[oc + 5];
                a5_2 = bias[oc + 5];
                a5_3 = bias[oc + 5];
                a6_0 = bias[oc + 6];
                a6_1 = bias[oc + 6];
                a6_2 = bias[oc + 6];
                a6_3 = bias[oc + 6];
                a7_0 = bias[oc + 7];
                a7_1 = bias[oc + 7];
                a7_2 = bias[oc + 7];
                a7_3 = bias[oc + 7];

                q = 0;

                for (ic = 0; ic < 3; ++ic) {
                    for (kh = 0; kh < 3; ++kh) {
                        row = ih + kh;

                        if (row < 0 || row >= 256) {
                            q += 3;
                            continue;
                        }

                        xBase = ic * inS + row * 256 + iw;

                        for (kx = 0; kx < 3; ++kx) {
                            col = iw + kx;

                            w0 = w[(oc + 0) * 27 + q];
                            w1 = w[(oc + 1) * 27 + q];
                            w2 = w[(oc + 2) * 27 + q];
                            w3 = w[(oc + 3) * 27 + q];
                            w4 = w[(oc + 4) * 27 + q];
                            w5 = w[(oc + 5) * 27 + q];
                            w6 = w[(oc + 6) * 27 + q];
                            w7 = w[(oc + 7) * 27 + q];

                            if (
                                col >= 0 &&
                                (iw + 6 + kx) < 256
                            ) {
                            x0 = x[xBase + kx + 0];
                            x1 = x[xBase + kx + 2];
                            x2 = x[xBase + kx + 4];
                            x3 = x[xBase + kx + 6];
                            w0 = w[(oc + 0) * 27 + q];
                            a0_0 += x0 * w0;
                            a0_1 += x1 * w0;
                            a0_2 += x2 * w0;
                            a0_3 += x3 * w0;
                            w1 = w[(oc + 1) * 27 + q];
                            a1_0 += x0 * w1;
                            a1_1 += x1 * w1;
                            a1_2 += x2 * w1;
                            a1_3 += x3 * w1;
                            w2 = w[(oc + 2) * 27 + q];
                            a2_0 += x0 * w2;
                            a2_1 += x1 * w2;
                            a2_2 += x2 * w2;
                            a2_3 += x3 * w2;
                            w3 = w[(oc + 3) * 27 + q];
                            a3_0 += x0 * w3;
                            a3_1 += x1 * w3;
                            a3_2 += x2 * w3;
                            a3_3 += x3 * w3;
                            w4 = w[(oc + 4) * 27 + q];
                            a4_0 += x0 * w4;
                            a4_1 += x1 * w4;
                            a4_2 += x2 * w4;
                            a4_3 += x3 * w4;
                            w5 = w[(oc + 5) * 27 + q];
                            a5_0 += x0 * w5;
                            a5_1 += x1 * w5;
                            a5_2 += x2 * w5;
                            a5_3 += x3 * w5;
                            w6 = w[(oc + 6) * 27 + q];
                            a6_0 += x0 * w6;
                            a6_1 += x1 * w6;
                            a6_2 += x2 * w6;
                            a6_3 += x3 * w6;
                            w7 = w[(oc + 7) * 27 + q];
                            a7_0 += x0 * w7;
                            a7_1 += x1 * w7;
                            a7_2 += x2 * w7;
                            a7_3 += x3 * w7;
                            } else {
                                if (col >= 0 && col < 256) {
                                    x0 = x[
                                        ic * inS +
                                        row * 256 +
                                        col
                                    ];
                                    a0_0 += x0 * w0;
                                    a1_0 += x0 * w1;
                                    a2_0 += x0 * w2;
                                    a3_0 += x0 * w3;
                                    a4_0 += x0 * w4;
                                    a5_0 += x0 * w5;
                                    a6_0 += x0 * w6;
                                    a7_0 += x0 * w7;
                                }

                                col = iw + 2 + kx;
                                if (col >= 0 && col < 256) {
                                    x1 = x[
                                        ic * inS +
                                        row * 256 +
                                        col
                                    ];
                                    a0_1 += x1 * w0;
                                    a1_1 += x1 * w1;
                                    a2_1 += x1 * w2;
                                    a3_1 += x1 * w3;
                                    a4_1 += x1 * w4;
                                    a5_1 += x1 * w5;
                                    a6_1 += x1 * w6;
                                    a7_1 += x1 * w7;
                                }

                                col = iw + 4 + kx;
                                if (col >= 0 && col < 256) {
                                    x2 = x[
                                        ic * inS +
                                        row * 256 +
                                        col
                                    ];
                                    a0_2 += x2 * w0;
                                    a1_2 += x2 * w1;
                                    a2_2 += x2 * w2;
                                    a3_2 += x2 * w3;
                                    a4_2 += x2 * w4;
                                    a5_2 += x2 * w5;
                                    a6_2 += x2 * w6;
                                    a7_2 += x2 * w7;
                                }

                                col = iw + 6 + kx;
                                if (col >= 0 && col < 256) {
                                    x3 = x[
                                        ic * inS +
                                        row * 256 +
                                        col
                                    ];
                                    a0_3 += x3 * w0;
                                    a1_3 += x3 * w1;
                                    a2_3 += x3 * w2;
                                    a3_3 += x3 * w3;
                                    a4_3 += x3 * w4;
                                    a5_3 += x3 * w5;
                                    a6_3 += x3 * w6;
                                    a7_3 += x3 * w7;
                                }
                            }

                            ++q;
                        }
                    }
                }

                out[o0 + outBase + 0] = a0_0;
                out[o0 + outBase + 1] = a0_1;
                out[o0 + outBase + 2] = a0_2;
                out[o0 + outBase + 3] = a0_3;
                out[o1 + outBase + 0] = a1_0;
                out[o1 + outBase + 1] = a1_1;
                out[o1 + outBase + 2] = a1_2;
                out[o1 + outBase + 3] = a1_3;
                out[o2 + outBase + 0] = a2_0;
                out[o2 + outBase + 1] = a2_1;
                out[o2 + outBase + 2] = a2_2;
                out[o2 + outBase + 3] = a2_3;
                out[o3 + outBase + 0] = a3_0;
                out[o3 + outBase + 1] = a3_1;
                out[o3 + outBase + 2] = a3_2;
                out[o3 + outBase + 3] = a3_3;
                out[o4 + outBase + 0] = a4_0;
                out[o4 + outBase + 1] = a4_1;
                out[o4 + outBase + 2] = a4_2;
                out[o4 + outBase + 3] = a4_3;
                out[o5 + outBase + 0] = a5_0;
                out[o5 + outBase + 1] = a5_1;
                out[o5 + outBase + 2] = a5_2;
                out[o5 + outBase + 3] = a5_3;
                out[o6 + outBase + 0] = a6_0;
                out[o6 + outBase + 1] = a6_1;
                out[o6 + outBase + 2] = a6_2;
                out[o6 + outBase + 3] = a6_3;
                out[o7 + outBase + 0] = a7_0;
                out[o7 + outBase + 1] = a7_1;
                out[o7 + outBase + 2] = a7_2;
                out[o7 + outBase + 3] = a7_3;
            }
        }
    }

    return {
        data: out,
        shape: [1, 64, 128, 128]
    };
}


function matMulBN4(a, aShape, b, bShape) {
    var ar = aShape.length;
    var br = bShape.length;
    var M, K, K2, N;
    var aBatch, bBatch, batchShape;
    var batchCount, batchRank;
    var aBatchShape, bBatchShape;
    var aBatchS, bBatchS, outBatchS;
    var outShape, out;
    var bi, d, t, coord;
    var aBatchIndex, bBatchIndex;
    var aBase, bBase, outBase;
    var m, n, k;
    var av;
    var s0, s1, s2, s3;

    if (ar < 2 || br < 2) {
        return Base.matMul(a, aShape, b, bShape);
    }

    M = aShape[ar - 2];
    K = aShape[ar - 1];
    K2 = bShape[br - 2];
    N = bShape[br - 1];

    if (K !== K2) {
        throw new Error(
            "MatMul K mismatch: " + K + " vs " + K2
        );
    }

    aBatch = aShape.slice(0, ar - 2);
    bBatch = bShape.slice(0, br - 2);
    batchShape = Base.broadcastShape(aBatch, bBatch);
    batchRank = batchShape.length;
    batchCount = Base.product(batchShape);

    aBatchShape = new Array(batchRank);
    bBatchShape = new Array(batchRank);

    for (d = 0; d < batchRank; ++d) {
        aBatchShape[d] =
            aBatch[aBatch.length - batchRank + d];
        bBatchShape[d] =
            bBatch[bBatch.length - batchRank + d];

        if (aBatchShape[d] === undefined) {
            aBatchShape[d] = 1;
        }
        if (bBatchShape[d] === undefined) {
            bBatchShape[d] = 1;
        }
    }

    aBatchS = Base.computeStrides(aBatchShape);
    bBatchS = Base.computeStrides(bBatchShape);
    outBatchS = Base.computeStrides(batchShape);

    outShape = batchShape.concat([M, N]);
    out = new Float32Array(batchCount * M * N);

    for (bi = 0; bi < batchCount; ++bi) {
        t = bi;
        aBatchIndex = 0;
        bBatchIndex = 0;

        for (d = 0; d < batchRank; ++d) {
            coord = Math.floor(t / outBatchS[d]);
            t -= coord * outBatchS[d];

            if (aBatchShape[d] !== 1) {
                aBatchIndex +=
                    coord * aBatchS[d];
            }

            if (bBatchShape[d] !== 1) {
                bBatchIndex +=
                    coord * bBatchS[d];
            }
        }

        aBase = aBatchIndex * M * K;
        bBase = bBatchIndex * K * N;
        outBase = bi * M * N;

        for (m = 0; m < M; ++m) {
            n = 0;

            for (; n + 3 < N; n += 4) {
                s0 = 0.0;
                s1 = 0.0;
                s2 = 0.0;
                s3 = 0.0;

                for (k = 0; k < K; ++k) {
                    av = a[aBase + m * K + k];

                    s0 +=
                        av *
                        b[bBase + k * N + n + 0];
                    s1 +=
                        av *
                        b[bBase + k * N + n + 1];
                    s2 +=
                        av *
                        b[bBase + k * N + n + 2];
                    s3 +=
                        av *
                        b[bBase + k * N + n + 3];
                }

                out[outBase + m * N + n + 0] = s0;
                out[outBase + m * N + n + 1] = s1;
                out[outBase + m * N + n + 2] = s2;
                out[outBase + m * N + n + 3] = s3;
            }

            for (; n < N; ++n) {
                s0 = 0.0;

                for (k = 0; k < K; ++k) {
                    s0 +=
                        a[aBase + m * K + k] *
                        b[bBase + k * N + n];
                }

                out[outBase + m * N + n] = s0;
            }
        }
    }

    return {
        data: out,
        shape: outShape
    };
}


var ERF_T = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4
];

var ERF_U = [
    3.35617141647503099647E1,
    5.21357949780142679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4
];

var ERFC_P = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2
];

var ERFC_Q = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2
];

var ERFC_R = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0
];

var ERFC_S = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0
];

function polevl_ERF_T(x) {
    var ans = ERF_T[0];
    ans = ans * x + ERF_T[1];
    ans = ans * x + ERF_T[2];
    ans = ans * x + ERF_T[3];
    ans = ans * x + ERF_T[4];
    return ans;
}

function p1evl_ERF_U(x) {
    var ans = x + ERF_U[0];
    ans = ans * x + ERF_U[1];
    ans = ans * x + ERF_U[2];
    ans = ans * x + ERF_U[3];
    ans = ans * x + ERF_U[4];
    return ans;
}

function polevl_ERFC_P(x) {
    var ans = ERFC_P[0];
    ans = ans * x + ERFC_P[1];
    ans = ans * x + ERFC_P[2];
    ans = ans * x + ERFC_P[3];
    ans = ans * x + ERFC_P[4];
    ans = ans * x + ERFC_P[5];
    ans = ans * x + ERFC_P[6];
    ans = ans * x + ERFC_P[7];
    ans = ans * x + ERFC_P[8];
    return ans;
}

function p1evl_ERFC_Q(x) {
    var ans = x + ERFC_Q[0];
    ans = ans * x + ERFC_Q[1];
    ans = ans * x + ERFC_Q[2];
    ans = ans * x + ERFC_Q[3];
    ans = ans * x + ERFC_Q[4];
    ans = ans * x + ERFC_Q[5];
    ans = ans * x + ERFC_Q[6];
    ans = ans * x + ERFC_Q[7];
    return ans;
}

function polevl_ERFC_R(x) {
    var ans = ERFC_R[0];
    ans = ans * x + ERFC_R[1];
    ans = ans * x + ERFC_R[2];
    ans = ans * x + ERFC_R[3];
    ans = ans * x + ERFC_R[4];
    ans = ans * x + ERFC_R[5];
    return ans;
}

function p1evl_ERFC_S(x) {
    var ans = x + ERFC_S[0];
    ans = ans * x + ERFC_S[1];
    ans = ans * x + ERFC_S[2];
    ans = ans * x + ERFC_S[3];
    ans = ans * x + ERFC_S[4];
    ans = ans * x + ERFC_S[5];
    return ans;
}

function erfcPolyAll(a) {
    var x = a < 0 ? -a : a;
    var z;
    var p;
    var q;
    var y;

    if (x < 1.0) {
        return 1.0 - erfPolyAll(a);
    }

    z = -a * a;

    if (z < -745.0) {
        y = 0.0;
    } else {
        z = Math.exp(z);

        if (x < 8.0) {
            p = polevl_ERFC_P(x);
            q = p1evl_ERFC_Q(x);
        } else {
            p = polevl_ERFC_R(x);
            q = p1evl_ERFC_S(x);
        }

        y = z * p / q;
    }

    if (a < 0.0) {
        y = 2.0 - y;
    }

    return y;
}

function erfPolyAll(x) {
    var ax;
    var z;

    if (x !== x) {
        return x;
    }

    if (x === Infinity) {
        return 1.0;
    }

    if (x === -Infinity) {
        return -1.0;
    }

    ax = x < 0 ? -x : x;

    if (ax > 1.0) {
        return 1.0 - erfcPolyAll(x);
    }

    z = x * x;

    return x *
        polevl_ERF_T(z) /
        p1evl_ERF_U(z);
}

function geluFusedErfSat4PolyAll(
    x,
    divScalar,
    addScalar,
    mulScalar
) {
    var out = new Float32Array(x.length);
    var i;
    var xv;
    var d;
    var e;
    var a;
    var m;
    var ax;

    for (i = 0; i < out.length; ++i) {
        xv = x[i];

        d = Math.fround(
            xv / divScalar
        );

        ax = d < 0 ? -d : d;

        if (ax >= 4.0) {
            e = d < 0.0 ? -1.0 : 1.0;
        } else {
            e = Math.fround(
                erfPolyAll(d)
            );
        }

        a = Math.fround(
            e + addScalar
        );

        m = Math.fround(
            xv * a
        );

        out[i] = Math.fround(
            m * mulScalar
        );
    }

    return out;
}


module.exports = {
    convPointwiseB4T8NCHW: convPointwiseB4T8NCHW,
    convDepthwiseDW8NCHW: convDepthwiseDW8NCHW,
    convGroupedM2T8NCHW: convGroupedM2T8NCHW,
    convInitial3x3s2B8T4NCHW: convInitial3x3s2B8T4NCHW,
    matMulBN4: matMulBN4,
    geluFusedErfSat4PolyAll: geluFusedErfSat4PolyAll
};

};
deps[3] = {'./mobileclip-fp32-kernels-v1.js':2};

var cache = {};
function load(id) {
    var module;
    var localRequire;
    if (cache[id]) {
        return cache[id].exports;
    }
    module = {exports: {}};
    cache[id] = module;
    localRequire = function (name) {
        var target = deps[id][name];
        if (target === undefined) {
            throw new Error("bundle require not found: module=" + id + " name=" + name);
        }
        return load(target);
    };
    modules[id](module, module.exports, localRequire);
    return module.exports;
}
load(0);
}());
