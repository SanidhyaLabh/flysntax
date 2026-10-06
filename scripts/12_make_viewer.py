"""Turn the recorded activity into a standalone HTML viewer (works offline)."""
import argparse, base64, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="runs/viz/real.npz")
ap.add_argument("--out", default="runs/viz/real.html")
ap.add_argument("--title", default="Fly connectome reading C code")
args = ap.parse_args()

d = np.load(args.data)
meta = json.loads(str(d["meta"]))
pos = d["pos"].astype(np.float64)
c = (pos.min(0) + pos.max(0)) / 2
pos = (pos - c) / (pos.max(0) - pos.min(0)).max() * 2.0     # roughly [-1, 1]; axes: x, depth(y), down(z)
n = len(pos)

allact = np.concatenate([np.abs(d[f"av_{k}"].astype(np.float32)).ravel() for k in range(len(meta["pairs"]))] +
                        [np.abs(d[f"ab_{k}"].astype(np.float32)).ravel() for k in range(len(meta["pairs"]))])
sa = float(np.percentile(allact, 99.5)) + 1e-6
diffs = [np.abs(d[f"ab_{k}"].astype(np.float32) - d[f"av_{k}"].astype(np.float32)) for k in range(len(meta["pairs"]))]
sd = float(np.percentile(np.concatenate([x.ravel() for x in diffs]), 99.5)) + 1e-6


def q(a, s):
    return base64.b64encode((np.clip(np.abs(a) / s, 0, 1) * 255).astype(np.uint8).tobytes()).decode()


pairs = []
for k, p in enumerate(meta["pairs"]):
    av, ab = d[f"av_{k}"].astype(np.float32), d[f"ab_{k}"].astype(np.float32)
    pairs.append(dict(kind=p["kind"], tv=p["tv"], tb=p["tb"], pv=p["pv"], pb=p["pb"], steps=int(av.shape[0]),
                      av=q(av, sa), ab=q(ab, sa), ad=q(diffs[k], sd)))

data = dict(n=n, pos=[round(float(v), 4) for v in pos.ravel()], cls=[int(v) for v in d["cls"]],
            pairs=pairs, note=meta["note"], mode=meta["mode"], settle=meta["settle"])

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0b0d12;color:#dfe3ea;font:14px system-ui,Segoe UI,Arial,sans-serif}
#bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 14px;background:#12151c;border-bottom:1px solid #222}
#bar h1{font-size:16px;margin:0 8px 0 0}
select,button{background:#1b2030;color:#dfe3ea;border:1px solid #333a50;border-radius:6px;padding:5px 10px;font:inherit;cursor:pointer}
#slider{width:240px}
#boost{width:110px}
#divinfo{padding:10px 16px 0;max-width:1200px;margin:0 auto;font-size:14px;line-height:1.5}
#divinfo b{color:#ffd54a}
#wrap{display:flex;flex-wrap:wrap;gap:12px;padding:12px;justify-content:center}
.panel{background:#10131a;border:1px solid #222;border-radius:10px;padding:8px;flex:1 1 300px;max-width:420px}
.panel h3{margin:4px 6px;font-size:15px}
.panel.diffp{border-color:#6b2a24}
canvas.brain{width:100%;aspect-ratio:3/4;background:#07080c;border-radius:6px;cursor:grab;display:block}
.verdict{margin:6px;font-weight:600;min-height:20px}
.stat{margin:6px;color:#ff9a8a;font-weight:600;min-height:20px}
.tok{font:12px ui-monospace,Consolas,monospace;line-height:1.8;margin:4px 6px;word-break:break-word}
.tok span{padding:1px 2px;border-radius:3px;color:#9aa3b5}
.tok .done{color:#dfe3ea}
.tok .dv{outline:2px solid #ff5a47;color:#fff}
.tok .cur{background:#ffd54a;color:#111}
#tlwrap{padding:0 16px;max-width:1200px;margin:0 auto}
#tlwrap h2{font-size:15px;margin:14px 0 4px}
#tlhelp{color:#9aa3b5;font-size:13px;line-height:1.5;margin:0 0 8px}
#tlhelp b{color:#dfe3ea}
#tl{width:100%;height:190px;background:#10131a;border:1px solid #222;border-radius:8px;cursor:pointer;display:block}
#tlread{color:#dfe3ea;font-size:13px;line-height:1.5;margin:8px 0 0}
#info{max-width:1200px;margin:0 auto;padding:4px 16px 24px;color:#9aa3b5;font-size:13px;line-height:1.55}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px;vertical-align:middle}
</style></head><body>
<div id="bar">
  <h1>__TITLE__</h1>
  <label>Example <select id="pair"></select></label>
  <button id="jump">Jump to first difference</button>
  <button id="play">Play</button>
  <input id="slider" type="range" min="0" max="0" value="0">
  <span id="stepinfo"></span>
  <label>Speed <select id="speed"><option value="2">slow</option><option value="5" selected>normal</option><option value="12">fast</option></select></label>
  <label>Difference boost <input id="boost" type="range" min="1" max="10" value="3"></label>
</div>
<div id="divinfo"></div>
<div id="wrap">
  <div class="panel"><h3>Valid code</h3><canvas class="brain" id="cv0" width="420" height="560"></canvas><div class="verdict" id="vd0"></div><div class="tok" id="tk0"></div></div>
  <div class="panel"><h3>Broken twin</h3><canvas class="brain" id="cv1" width="420" height="560"></canvas><div class="verdict" id="vd1"></div><div class="tok" id="tk1"></div></div>
  <div class="panel diffp"><h3>Difference (valid vs broken)</h3><canvas class="brain" id="cv2" width="420" height="560"></canvas><div class="stat" id="stat"></div><div style="margin:6px;color:#9aa3b5;font-size:12px">Black = the two brains are doing exactly the same. Red to yellow = neurons that behave differently.</div></div>
</div>
<div id="tlwrap">
  <h2>How different are the two brains, step by step?</h2>
  <p id="tlhelp">Each bar is one step of the animation (one token read). <b>Bar height = how many neurons behave clearly differently</b> in the valid brain and the broken brain at that step. No bar means the two brains are doing exactly the same thing. The dashed yellow line marks the first token where the two codes differ, and the white bar is the step shown above. Click the chart to jump to a step.</p>
  <canvas id="tl" width="1200" height="190"></canvas>
  <p id="tlread"></p>
</div>
<div id="info">
  <div>Neuron types:
    <span class="sw" style="background:#4da3ff"></span>sensory
    <span class="sw" style="background:#36d1c4"></span>visual projection
    <span class="sw" style="background:#b8b8d0"></span>internal
    <span class="sw" style="background:#7bdc65"></span>ascending
    <span class="sw" style="background:#ffa347"></span>descending (read by the classifier)</div>
  <p>Each dot is one neuron at its real cell-body position (a random 6,000 of ~44,000 are drawn, plus all output neurons). Drag any brain to rotate, scroll to zoom. The network gets one token per step, then settles for a few steps with no input before the verdict.</p>
  <p id="note"></p>
  <p>This shows how <b>this model</b> responds. It is not a proof that the fly wiring detects errors: a version with randomly rewired connections learned the same task equally well.</p>
</div>
<script>
const D = __DATA__;
const N = D.n, pos = D.pos, cls = D.cls;
const TH = 40;   // a neuron counts as 'clearly different' if its difference is above this (0-255)
const COLORS = ["#4da3ff","#36d1c4","#b8b8d0","#7bdc65","#ffa347"];
const RGB = COLORS.map(h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)]);
function b64(s){ const b = atob(s), u = new Uint8Array(b.length); for (let i=0;i<b.length;i++) u[i] = b.charCodeAt(i); return u; }
const pairs = D.pairs.map(p => {
  const o = Object.assign({}, p, {av: b64(p.av), ab: b64(p.ab), ad: b64(p.ad)});
  let d = 0; while (d < p.tv.length && p.tv[d] === p.tb[d]) d++;
  o.div = d;
  const cnt = new Int32Array(p.steps); let mx = 1;
  for (let s = 0; s < p.steps; s++){ let t = 0; for (let i = 0; i < N; i++) if (o.ad[s*N + i] > TH) t++; cnt[s] = t; if (t > mx) mx = t; }
  o.cnt = cnt; o.cmax = mx;
  return o;
});
const byClass = [0,1,2,3,4].map(k => { const a = []; for (let i=0;i<N;i++) if (cls[i]===k) a.push(i); return Int32Array.from(a); });
const px = new Float32Array(N), py = new Float32Array(N), pf = new Float32Array(N);
let cur = 0, step = 0, playing = false, yaw = 0.35, zoom = 1, speed = 5, boost = 3;

const $ = id => document.getElementById(id);
const cvs = [$("cv0"), $("cv1"), $("cv2")], ctxs = cvs.map(c => c.getContext("2d"));
const tlc = $("tl"), tlx = tlc.getContext("2d");
const P = () => pairs[cur];

function draw(k, frame, hot){
  const cv = cvs[k], ctx = ctxs[k], W = cv.width, H = cv.height;
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.clearRect(0, 0, W, H);
  const c = Math.cos(yaw), s = Math.sin(yaw), sc = Math.min(W, H) * 0.46 * zoom, cx = W / 2, cy = H / 2;
  for (let i = 0; i < N; i++){
    const X = pos[3*i], Y = pos[3*i+1], Z = pos[3*i+2];
    const xr = X*c + Y*s, dr = -X*s + Y*c, f = 1 / (1 + 0.35*dr);
    px[i] = cx + xr*sc*f; py[i] = cy + Z*sc*f; pf[i] = f;
  }
  ctx.globalAlpha = hot ? 0.10 : 0.16;
  for (let k2 = 0; k2 < 5; k2++){
    ctx.fillStyle = hot ? "#8a8fa0" : COLORS[k2];
    const idx = byClass[k2];
    for (let j = 0; j < idx.length; j++){ const i = idx[j]; ctx.fillRect(px[i]-0.6, py[i]-0.6, 1.3, 1.3); }
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < N; i++){
    let a = frame[i] / 255;
    if (hot) a = Math.min(1, a * boost);
    if (a < (hot ? 0.04 : 0.08)) continue;
    let r, g, b;
    if (hot){ r = 255; g = Math.round(60 + 190 * a); b = Math.round(40 + 40 * a); }
    else {
      const base = RGB[cls[i]];
      r = Math.round(base[0] + (255 - base[0]) * a * 0.6); g = Math.round(base[1] + (255 - base[1]) * a * 0.6); b = Math.round(base[2] + (255 - base[2]) * a * 0.6);
    }
    ctx.fillStyle = "rgba(" + r + "," + g + "," + b + "," + (0.25 + 0.75*a).toFixed(2) + ")";
    const sz = (hot ? 2 + 6*a : 1.5 + 4.5*a) * pf[i];
    ctx.fillRect(px[i] - sz/2, py[i] - sz/2, sz, sz);
  }
}

function drawTimeline(){
  const cw = tlc.clientWidth || 1200;
  if (tlc.width !== cw) tlc.width = cw;
  const p = P(), W = tlc.width, H = tlc.height, T = p.tv.length, n = p.steps;
  const L = 56, R = 10, top = 28, bot = H - 30, bw = (W - L - R) / n;
  tlx.clearRect(0, 0, W, H);
  tlx.textBaseline = "alphabetic";
  tlx.fillStyle = "#171b26"; tlx.fillRect(L + T * bw, top, W - R - (L + T * bw), bot - top);
  tlx.font = "12px system-ui, sans-serif";
  [0, 0.5, 1].forEach(f => {
    const y = bot - f * (bot - top);
    tlx.strokeStyle = "#252a38"; tlx.lineWidth = 1; tlx.beginPath(); tlx.moveTo(L, y); tlx.lineTo(W - R, y); tlx.stroke();
    tlx.fillStyle = "#9aa3b5"; tlx.textAlign = "right";
    tlx.fillText(String(Math.round(f * p.cmax)), L - 6, y + 4);
  });
  tlx.save(); tlx.translate(13, (top + bot) / 2); tlx.rotate(-Math.PI / 2); tlx.textAlign = "center"; tlx.fillStyle = "#9aa3b5";
  tlx.fillText("neurons that differ", 0, 0); tlx.restore();
  for (let s = 0; s < n; s++){
    const h = (p.cnt[s] / p.cmax) * (bot - top);
    tlx.fillStyle = s === step ? "#ffffff" : "#ff6a52";
    tlx.fillRect(L + s * bw + 1, bot - h, Math.max(1, bw - 2), h);
  }
  tlx.fillStyle = "#9aa3b5"; tlx.textAlign = "center";
  for (let s = 0; s < T; s++){ if (s === 0 || (s + 1) % 10 === 0) tlx.fillText(String(s + 1), L + (s + 0.5) * bw, bot + 15); }
  tlx.textAlign = "left"; tlx.fillText("token being read \u2192", L, H - 4);
  if (p.div * bw > 110){ tlx.textAlign = "center"; tlx.fillStyle = "#6f7a90"; tlx.fillText("identical (no difference)", L + p.div * bw / 2, bot - 8); }
  const sx = L + T * bw;
  if ((n - T) * bw > 140){ tlx.textAlign = "center"; tlx.fillStyle = "#8f9ab0"; tlx.fillText("settling: no more input", sx + (W - R - sx) / 2, top + 16); }
  const dx = L + p.div * bw;
  tlx.strokeStyle = "#ffd54a"; tlx.lineWidth = 2; tlx.setLineDash([5, 4]);
  tlx.beginPath(); tlx.moveTo(dx, top - 8); tlx.lineTo(dx, bot); tlx.stroke(); tlx.setLineDash([]);
  tlx.fillStyle = "#ffd54a";
  if (dx + 170 > W){ tlx.textAlign = "right"; tlx.fillText("first different token", dx - 6, 16); }
  else { tlx.textAlign = "left"; tlx.fillText("first different token", dx + 6, 16); }
}

function readout(){
  const p = P(), T = p.tv.length;
  let pk = 0; for (let s = 0; s < p.steps; s++) if (p.cnt[s] > p.cnt[pk]) pk = s;
  const last = p.cnt[T - 1], fin = p.cnt[p.steps - 1];
  let msg = "Largest difference: " + p.cnt[pk] + " neurons at " + (pk < T ? "token " + (pk + 1) : "settling step " + (pk - T + 1)) +
    ". At the last token: " + last + " neurons. After settling (when the verdict is read): " + fin + " neurons.";
  if (p.cnt[pk] > 0 && fin < 0.25 * p.cnt[pk]) msg += " So the effect of the change fades as the network keeps reading. The verdict also uses the average and peak activity over the whole snippet, so an early change can still count.";
  $("tlread").textContent = msg;
}

let spans = [[], []];
function buildTokens(){
  const p = P();
  [p.tv, p.tb].forEach((toks, k) => {
    const el = $("tk" + k); el.innerHTML = ""; spans[k] = [];
    toks.forEach(t => { const s = document.createElement("span"); s.textContent = t + " "; el.appendChild(s); spans[k].push(s); });
  });
  [[p.pv, "vd0", "valid"], [p.pb, "vd1", "broken (" + p.kind.replace(/_/g, " ") + ")"]].forEach(([pe, id, truth]) => {
    const el = $(id), err = pe >= 0.5;
    el.textContent = "Truth: " + truth + "  |  Model says: " + (err ? "ERROR" : "VALID") + "  (P(error) = " + Math.round(pe*100) + "%)";
    el.style.color = err ? "#ff7a6b" : "#7bdc65";
  });
  $("slider").max = p.steps - 1;
  readout();
  $("divinfo").innerHTML = "The two snippets are <b>identical for the first " + p.div + " tokens</b>, so the brain activity is identical too (the difference panel stays black). " +
    "At token <b>" + (p.div + 1) + "</b> they differ (<b>" + (p.tv[p.div] || "?") + "</b> vs <b>" + (p.tb[p.div] || "?") + "</b>) and the brains start to diverge.";
}

function render(){
  const p = P(), T = p.tv.length, o = step * N, e = o + N;
  draw(0, p.av.subarray(o, e), false);
  draw(1, p.ab.subarray(o, e), false);
  draw(2, p.ad.subarray(o, e), true);
  let cnt = 0; for (let i = o; i < e; i++) if (p.ad[i] > TH) cnt++;
  $("stat").textContent = step < p.div ? "Identical so far: 0 neurons differ" : "Neurons clearly different: " + cnt + " of " + N;
  [0, 1].forEach(k => spans[k].forEach((s, i) => { s.className = i === step ? "cur" : (i === p.div ? "dv" : (i < step ? "done" : "")); }));
  $("stepinfo").textContent = step < T ? "token " + (step + 1) + " / " + T : "settling " + (step - T + 1) + " / " + (p.steps - T);
  $("slider").value = step;
  drawTimeline();
}

pairs.forEach((p, i) => { const o = document.createElement("option"); o.value = i; o.textContent = (i + 1) + ". " + p.kind.replace(/_/g, " "); $("pair").appendChild(o); });
$("note").textContent = D.note + " Model: " + D.mode + " wiring.";
$("pair").onchange = e => { cur = +e.target.value; buildTokens(); step = Math.min(P().div, P().steps - 1); render(); };
$("speed").onchange = e => { speed = +e.target.value; };
$("boost").oninput = e => { boost = +e.target.value; render(); };
$("slider").oninput = e => { step = +e.target.value; render(); };
$("jump").onclick = () => { step = Math.min(P().div, P().steps - 1); render(); };
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; };
tlc.onclick = e => { const r = tlc.getBoundingClientRect(); step = Math.max(0, Math.min(P().steps - 1, Math.floor((e.clientX - r.left) / r.width * P().steps))); render(); };
let drag = null;
cvs.forEach(cv => {
  cv.onmousedown = e => { drag = e.clientX; cv.style.cursor = "grabbing"; };
  cv.onwheel = e => { e.preventDefault(); zoom = Math.max(0.5, Math.min(4, zoom * (e.deltaY < 0 ? 1.1 : 0.9))); render(); };
});
window.onmouseup = () => { drag = null; cvs.forEach(c => c.style.cursor = "grab"); };
window.onmousemove = e => { if (drag !== null){ yaw += (e.clientX - drag) * 0.008; drag = e.clientX; render(); } };
window.onresize = () => render();
let last = 0;
function tick(ts){
  if (playing && ts - last > 1000 / speed){ last = ts; step = (step + 1) % P().steps; render(); }
  requestAnimationFrame(tick);
}
buildTokens(); step = Math.min(P().div, P().steps - 1); render(); requestAnimationFrame(tick);
</script></body></html>
"""
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(HTML.replace("__TITLE__", args.title).replace("__DATA__", json.dumps(data)), encoding="utf-8")
print(f"saved {args.out}  ({Path(args.out).stat().st_size / 1e6:.1f} MB) - open it in your browser")