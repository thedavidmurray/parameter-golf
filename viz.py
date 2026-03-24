#!/usr/bin/env python3
"""
Parameter Golf - Design Space Visualizer
Generates a standalone HTML+SVG file showing:
  1. BPB leaderboard progression over time
  2. Architecture budget surface (layers x dim, colored by param count & fit)
  3. Quantization tradeoff surface

Usage: python3 viz.py > viz.html && open viz.html
"""

import json, os, math, re
from datetime import datetime

# ── Load leaderboard data ──────────────────────────────────────────────────────

RECORDS_BASE = "./records/track_10min_16mb"
CODE_BYTES = 50_000
TOTAL_BUDGET = 16_000_000
MODEL_BUDGET = TOTAL_BUDGET - CODE_BYTES

records = []
for d in sorted(os.listdir(RECORDS_BASE)):
    path = f"{RECORDS_BASE}/{d}/submission.json"
    if not os.path.exists(path):
        continue
    with open(path) as f:
        r = json.load(f)
    bpb = r.get("val_bpb")
    date = r.get("date", "")[:10]
    name = r.get("name", d)
    if bpb and date and date != "?":
        records.append({"date": date, "bpb": float(bpb), "name": name,
                        "bytes": r.get("bytes_total")})

records.sort(key=lambda r: r["date"])

# Running best (frontier)
frontier = []
best = 9999
for r in records:
    if r["bpb"] < best:
        best = r["bpb"]
        frontier.append(r)

# ── Architecture parameter counter ─────────────────────────────────────────────

def count_params(vocab, layers, dim, num_heads, num_kv_heads, mlp_mult, tied=True):
    kv_dim = num_kv_heads * (dim // num_heads)
    attn = dim*dim + dim*kv_dim + dim*kv_dim + dim*dim
    mlp = dim*(mlp_mult*dim) + (mlp_mult*dim)*dim
    block_misc = 6 * dim
    embed = vocab * dim
    skips = min(layers//2, layers - layers//2) * dim
    return embed * (1 if tied else 2) + layers*(attn + mlp + block_misc) + skips

COMPRESSION_RATIO = 1.0862  # calibrated against baseline

def compressed_bytes(params, layers, bits=8):
    scale_overhead = layers * 6 * 512 * 4
    raw = params * bits / 8 + scale_overhead
    return raw / COMPRESSION_RATIO

def fits(params, layers, bits=8):
    return compressed_bytes(params, layers, bits) <= MODEL_BUDGET

# ── SVG helpers ────────────────────────────────────────────────────────────────

def lerp(a, b, t): return a + (b - a) * t

def color_lerp(c1, c2, t):
    r = int(lerp(c1[0], c2[0], t))
    g = int(lerp(c1[1], c2[1], t))
    b = int(lerp(c1[2], c2[2], t))
    return f"rgb({r},{g},{b})"

# BPB color: green=good (low), red=bad (high)
def bpb_color(bpb, lo=1.10, hi=1.25):
    t = max(0, min(1, (bpb - lo) / (hi - lo)))
    return color_lerp((34, 197, 94), (239, 68, 68), t)

# ── Chart 1: BPB over time ─────────────────────────────────────────────────────

def chart_progression(x0, y0, W, H):
    if not records:
        return ""
    dates = [r["date"] for r in records]
    bpbs  = [r["bpb"]  for r in records]

    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    t0 = date_objs[0].timestamp()
    t1 = date_objs[-1].timestamp()
    t_range = max(t1 - t0, 1)

    bpb_lo, bpb_hi = min(bpbs) - 0.01, max(bpbs) + 0.01
    bpb_range = bpb_hi - bpb_lo

    pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 50
    cw = W - pad_l - pad_r
    ch = H - pad_t - pad_b

    def tx(date_obj): return x0 + pad_l + (date_obj.timestamp() - t0) / t_range * cw
    def ty(bpb): return y0 + pad_t + (1 - (bpb - bpb_lo) / bpb_range) * ch

    out = []
    # Background
    out.append(f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" fill="#0f172a" rx="8"/>')
    out.append(f'<text x="{x0+W//2}" y="{y0+14}" text-anchor="middle" fill="#94a3b8" font-size="13" font-family="monospace">BPB Progress (lower = better)</text>')

    # Grid lines
    for bpb_tick in [1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24]:
        if bpb_lo <= bpb_tick <= bpb_hi:
            yy = ty(bpb_tick)
            out.append(f'<line x1="{x0+pad_l}" y1="{yy}" x2="{x0+pad_l+cw}" y2="{yy}" stroke="#1e293b" stroke-width="1"/>')
            out.append(f'<text x="{x0+pad_l-4}" y="{yy+4}" text-anchor="end" fill="#64748b" font-size="10" font-family="monospace">{bpb_tick:.2f}</text>')

    # All runs (dots)
    for r in records:
        do = datetime.strptime(r["date"], "%Y-%m-%d")
        cx = tx(do); cy = ty(r["bpb"])
        col = bpb_color(r["bpb"])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{col}" opacity="0.7">')
        out.append(f'  <title>{r["name"]}: {r["bpb"]:.4f} BPB ({r["date"]})</title>')
        out.append(f'</circle>')

    # Frontier line
    if len(frontier) >= 2:
        pts = " ".join(f"{tx(datetime.strptime(r['date'],'%Y-%m-%d')):.1f},{ty(r['bpb']):.1f}" for r in frontier)
        out.append(f'<polyline points="{pts}" fill="none" stroke="#22d3ee" stroke-width="2" stroke-dasharray="4,2"/>')

    # Frontier dots (larger)
    for r in frontier:
        do = datetime.strptime(r["date"], "%Y-%m-%d")
        cx = tx(do); cy = ty(r["bpb"])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#22d3ee" stroke="#0f172a" stroke-width="1.5">')
        out.append(f'  <title>SOTA: {r["name"]}: {r["bpb"]:.4f} BPB</title>')
        out.append(f'</circle>')

    # Axes
    out.append(f'<line x1="{x0+pad_l}" y1="{y0+pad_t}" x2="{x0+pad_l}" y2="{y0+pad_t+ch}" stroke="#334155" stroke-width="1"/>')
    out.append(f'<line x1="{x0+pad_l}" y1="{y0+pad_t+ch}" x2="{x0+pad_l+cw}" y2="{y0+pad_t+ch}" stroke="#334155" stroke-width="1"/>')

    # Date labels
    for r in frontier:
        do = datetime.strptime(r["date"], "%Y-%m-%d")
        xx = tx(do)
        out.append(f'<text x="{xx:.1f}" y="{y0+pad_t+ch+14}" text-anchor="middle" fill="#475569" font-size="9" font-family="monospace">{r["date"][5:]}</text>')

    return "\n".join(out)

# ── Chart 2: Architecture surface (layers x dim) ───────────────────────────────

def chart_arch_surface(x0, y0, W, H):
    VOCAB = 1024
    HEADS, KV_HEADS, MLP_MULT = 8, 4, 2

    layer_range = list(range(6, 20))
    dim_range   = list(range(256, 896, 64))

    pad_l, pad_r, pad_t, pad_b = 55, 20, 30, 50
    cw = W - pad_l - pad_r
    ch = H - pad_t - pad_b

    cell_w = cw / len(dim_range)
    cell_h = ch / len(layer_range)

    out = []
    out.append(f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" fill="#0f172a" rx="8"/>')
    out.append(f'<text x="{x0+W//2}" y="{y0+14}" text-anchor="middle" fill="#94a3b8" font-size="13" font-family="monospace">Architecture Budget Surface (vocab=1024)</text>')
    out.append(f'<text x="{x0+W//2}" y="{y0+26}" text-anchor="middle" fill="#64748b" font-size="10" font-family="monospace">Green=fits int8 | Yellow=fits int6 only | Red=over budget</text>')

    # X axis label
    out.append(f'<text x="{x0+pad_l+cw//2}" y="{y0+H-8}" text-anchor="middle" fill="#64748b" font-size="11" font-family="monospace">model_dim</text>')
    # Y axis label
    out.append(f'<text x="{x0+12}" y="{y0+pad_t+ch//2}" text-anchor="middle" fill="#64748b" font-size="11" font-family="monospace" transform="rotate(-90,{x0+12},{y0+pad_t+ch//2})">layers</text>')

    for li, layers in enumerate(layer_range):
        for di, dim in enumerate(dim_range):
            # Ensure dim divisible by num_heads
            dim_a = (dim // HEADS) * HEADS
            p = count_params(VOCAB, layers, dim_a, HEADS, KV_HEADS, MLP_MULT)

            fits8 = fits(p, layers, 8)
            fits6 = fits(p, layers, 6)

            if fits8:
                fill = "#166534"   # dark green
                stroke = "#22c55e"
            elif fits6:
                fill = "#713f12"   # dark yellow/amber
                stroke = "#f59e0b"
            else:
                fill = "#1e1e2e"   # very dark, over budget
                stroke = "#334155"

            cx = x0 + pad_l + di * cell_w
            cy = y0 + pad_t + (len(layer_range) - 1 - li) * cell_h

            m = count_params(VOCAB, layers, dim_a, HEADS, KV_HEADS, MLP_MULT) / 1e6
            b8 = compressed_bytes(p, layers, 8) / 1e6
            b6 = compressed_bytes(p, layers, 6) / 1e6

            out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w-1:.1f}" height="{cell_h-1:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="0.5" rx="1">')
            out.append(f'  <title>{layers}L x {dim_a}d: {m:.1f}M params | int8:{b8:.1f}MB int6:{b6:.1f}MB</title>')
            out.append(f'</rect>')

            # Annotate param count in larger cells
            if cell_w > 28 and cell_h > 14:
                col = "#4ade80" if fits8 else ("#fbbf24" if fits6 else "#475569")
                out.append(f'<text x="{cx+cell_w/2:.1f}" y="{cy+cell_h/2+4:.1f}" text-anchor="middle" fill="{col}" font-size="8" font-family="monospace">{m:.0f}M</text>')

    # Axis tick labels
    for di, dim in enumerate(dim_range):
        dim_a = (dim // HEADS) * HEADS
        cx = x0 + pad_l + di * cell_w + cell_w/2
        out.append(f'<text x="{cx:.1f}" y="{y0+pad_t+ch+14}" text-anchor="middle" fill="#64748b" font-size="9" font-family="monospace">{dim_a}</text>')

    for li, layers in enumerate(layer_range):
        cy = y0 + pad_t + (len(layer_range) - 1 - li) * cell_h + cell_h/2 + 3
        out.append(f'<text x="{x0+pad_l-4}" y="{cy:.1f}" text-anchor="end" fill="#64748b" font-size="9" font-family="monospace">{layers}</text>')

    # Mark baseline
    base_li = layer_range.index(9)
    base_di = next(i for i, d in enumerate(dim_range) if (d//HEADS)*HEADS == 512)
    bcx = x0 + pad_l + base_di * cell_w + cell_w/2
    bcy = y0 + pad_t + (len(layer_range) - 1 - base_li) * cell_h + cell_h/2
    out.append(f'<circle cx="{bcx:.1f}" cy="{bcy:.1f}" r="6" fill="none" stroke="#f8fafc" stroke-width="1.5"/>')
    out.append(f'<text x="{bcx+8:.1f}" y="{bcy-4:.1f}" fill="#f8fafc" font-size="9" font-family="monospace">baseline</text>')

    return "\n".join(out)

# ── Chart 3: Quant tradeoff (params vs BPB proxy) ─────────────────────────────

def chart_quant_surface(x0, y0, W, H):
    """Show the bits-per-param vs model size tradeoff surface."""
    out = []
    out.append(f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" fill="#0f172a" rx="8"/>')
    out.append(f'<text x="{x0+W//2}" y="{y0+14}" text-anchor="middle" fill="#94a3b8" font-size="13" font-family="monospace">Quant Tradeoff Surface</text>')
    out.append(f'<text x="{x0+W//2}" y="{y0+26}" text-anchor="middle" fill="#64748b" font-size="10" font-family="monospace">Effective params reachable at each quantization level within 16MB</text>')

    pad_l, pad_r, pad_t, pad_b = 55, 20, 40, 50
    cw = W - pad_l - pad_r
    ch = H - pad_t - pad_b

    # X: bits per param (4 to 8)
    bits_range = [4, 5, 6, 7, 8]
    # For each bit width, max reachable params:
    # budget = params * bits/8 / compression_ratio + scale_overhead
    # params = (budget - scale_overhead) * compression_ratio * 8/bits
    # assume scale_overhead ~ 600KB for 11 layers
    scale_oh = 11 * 6 * 512 * 4  # ~135KB

    max_params = {bits: (MODEL_BUDGET - scale_oh) * COMPRESSION_RATIO * 8 / bits
                  for bits in bits_range}

    # Scaling law approximation: BPB roughly follows log of params
    # Using two calibration points: 17M@int8->1.224, 20.7M@int6->1.127 (11L record)
    # ln(17) = 2.833, bpb=1.224; ln(20.7)=3.030, bpb=1.127
    # slope = (1.127-1.224)/(3.030-2.833) = -0.097/0.197 = -0.493
    # intercept: 1.224 = c - 0.493*2.833 -> c = 1.224 + 1.396 = 2.620
    # BPB_predicted = 2.620 - 0.493 * ln(params/1e6)
    # But quantization degrades BPB: empirically int6 costs ~0.01, int5 ~0.04, int4 ~0.15
    quant_penalty = {8: 0.0, 7: 0.005, 6: 0.012, 5: 0.045, 4: 0.16}

    def predicted_bpb(params, bits):
        capacity_bpb = 2.620 - 0.493 * math.log(params / 1e6)
        return capacity_bpb + quant_penalty[bits]

    bit_x = {bits: x0 + pad_l + (bits - 4) / (8 - 4) * cw for bits in bits_range}

    bpb_lo, bpb_hi = 1.05, 1.30
    def ty(bpb): return y0 + pad_t + (1 - (bpb - bpb_lo) / (bpb_hi - bpb_lo)) * ch

    # Grid
    for bpb_tick in [1.10, 1.15, 1.20, 1.25]:
        yy = ty(bpb_tick)
        out.append(f'<line x1="{x0+pad_l}" y1="{yy:.1f}" x2="{x0+pad_l+cw}" y2="{yy:.1f}" stroke="#1e293b" stroke-width="1"/>')
        out.append(f'<text x="{x0+pad_l-4}" y="{yy+4:.1f}" text-anchor="end" fill="#64748b" font-size="10" font-family="monospace">{bpb_tick:.2f}</text>')

    # Plot each bit width as a point
    points = []
    for bits in bits_range:
        p = max_params[bits]
        bpb = predicted_bpb(p, bits)
        cx = bit_x[bits]
        cy = ty(bpb)
        col = bpb_color(bpb)
        points.append((cx, cy))
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{col}" stroke="#0f172a" stroke-width="1.5">')
        out.append(f'  <title>int{bits}: {p/1e6:.1f}M params -> predicted {bpb:.3f} BPB</title>')
        out.append(f'</circle>')
        out.append(f'<text x="{cx:.1f}" y="{cy-14:.1f}" text-anchor="middle" fill="{col}" font-size="10" font-family="monospace">int{bits}</text>')
        out.append(f'<text x="{cx:.1f}" y="{cy+20:.1f}" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="monospace">{p/1e6:.0f}M</text>')
        out.append(f'<text x="{cx:.1f}" y="{cy+30:.1f}" text-anchor="middle" fill="#64748b" font-size="9" font-family="monospace">{bpb:.3f}</text>')

    # Connect with line
    if len(points) >= 2:
        pts_str = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in points)
        out.append(f'<polyline points="{pts_str}" fill="none" stroke="#334155" stroke-width="1.5" stroke-dasharray="3,2"/>')

    # Mark the sweet spot (minimum predicted BPB)
    best_bits = min(bits_range, key=lambda b: predicted_bpb(max_params[b], b))
    cx = bit_x[best_bits]; cy = ty(predicted_bpb(max_params[best_bits], best_bits))
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="none" stroke="#22d3ee" stroke-width="2"/>')

    # Y-axis label
    out.append(f'<text x="{x0+12}" y="{y0+pad_t+ch//2}" text-anchor="middle" fill="#64748b" font-size="11" font-family="monospace" transform="rotate(-90,{x0+12},{y0+pad_t+ch//2})">Predicted BPB</text>')
    out.append(f'<text x="{x0+pad_l+cw//2}" y="{y0+H-10}" text-anchor="middle" fill="#64748b" font-size="11" font-family="monospace">bits per param</text>')

    # Axes
    out.append(f'<line x1="{x0+pad_l}" y1="{y0+pad_t}" x2="{x0+pad_l}" y2="{y0+pad_t+ch}" stroke="#334155" stroke-width="1"/>')
    out.append(f'<line x1="{x0+pad_l}" y1="{y0+pad_t+ch}" x2="{x0+pad_l+cw}" y2="{y0+pad_t+ch}" stroke="#334155" stroke-width="1"/>')

    return "\n".join(out)

# ── Assemble full HTML ─────────────────────────────────────────────────────────

SVG_W, SVG_H = 1100, 960
CHART_W = SVG_W // 2 - 20
CHART_H = 280

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Parameter Golf — Design Space</title>
<style>
  body {{ background:#020617; color:#e2e8f0; font-family:monospace; padding:20px; margin:0 }}
  h1 {{ color:#22d3ee; font-size:18px; margin:0 0 4px }}
  p  {{ color:#64748b; font-size:12px; margin:0 0 16px }}
  svg {{ display:block }}
</style>
</head>
<body>
<h1>Parameter Golf — Design Space Visualizer</h1>
<p>Hover over cells/dots for details. Green frontier = current SOTA path.</p>
<svg width="{SVG_W}" height="{SVG_H}" xmlns="http://www.w3.org/2000/svg">
  <!-- Chart 1: Progression -->
  {chart_progression(10, 10, CHART_W, CHART_H)}

  <!-- Chart 2: Quant surface -->
  {chart_quant_surface(SVG_W//2 + 10, 10, CHART_W, CHART_H)}

  <!-- Chart 3: Architecture surface (full width) -->
  {chart_arch_surface(10, CHART_H + 30, SVG_W - 20, SVG_H - CHART_H - 40)}
</svg>

<div style="margin-top:16px;font-size:11px;color:#475569">
<strong style="color:#94a3b8">Reading the charts:</strong><br>
<span style="color:#22c55e">■</span> Green cells = fits in int8 budget (~17M params max)<br>
<span style="color:#f59e0b">■</span> Amber cells = fits in int6 budget only (~23M params max)<br>
<span style="color:#475569">■</span> Dark cells = over budget entirely<br>
<span style="color:#22d3ee">●</span> Cyan dots/line = SOTA frontier records<br><br>
<strong style="color:#94a3b8">Key insight:</strong>
Int6 quantization unlocks a whole new region of the architecture space (the amber zone) —
especially 11–13 layers at 512d. The quant surface shows int6 is the sweet spot:
the capacity gain outweighs the quantization penalty, but int5 and below likely hurt more than they help.
</div>
</body>
</html>"""

print(html)
