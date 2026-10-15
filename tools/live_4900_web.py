"""Live training dashboard served over HTTP (no images, browser auto-refresh).

Reads the run's anchor_metrics.jsonl + train_metrics.jsonl on every /data
poll and serves a self-contained HTML page (vanilla-JS canvas charts, no
CDN). Open http://localhost:8097 in a browser; it updates every 5 s.

Usage: python tools/live_4900_web.py [exptid] [--port 8097] [--tail 600]
"""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPT_DIR = ROOT / 'legged_gym/logs/recovery_4900'

PAGE = '''<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:monospace;background:#111;color:#ddd;margin:12px}}
h2{{font-size:16px;margin:4px 0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.card{{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:8px}}
canvas{{width:100%;height:230px}} table{{border-collapse:collapse;margin-bottom:10px}}
td,th{{border:1px solid #444;padding:3px 8px;font-size:12px}} .ok{{color:#4c4}}
</style></head><body>
<h2>{title} <span id="status" class="ok">connecting…</span></h2>
<table id="stats"></table>
<div class="grid">
<div class="card"><b>Adapter deviation</b><canvas id="c0"></canvas></div>
<div class="card"><b>Anchor pressure</b><canvas id="c1"></canvas></div>
<div class="card"><b>Transition detector inputs</b><canvas id="c2"></canvas></div>
<div class="card"><b>Exploration / LR</b><canvas id="c3"></canvas></div>
</div>
<script>
function chart(cv, series, ylog){{
 const dpr=window.devicePixelRatio||1, w=cv.clientWidth*dpr, h=cv.clientHeight*dpr;
 cv.width=w; cv.height=h; const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr);
 const W=cv.clientWidth,H=cv.clientHeight,L=46,R=52,T=8,B=18;
 let lo=Infinity,hi=-Infinity;
 for(const s of series)for(const v of s.data){{if(v==null)continue;lo=Math.min(lo,v);hi=Math.max(hi,v)}}
 if(!isFinite(lo)){{lo=0;hi=1}} if(lo===hi){{hi=lo+1}}
 const yr=v=>(ylog?Math.log10(Math.max(v,1e-12)):v);
 const a=yr(lo),b=yr(hi);
 const X=i=>L+(W-L-R)*i/Math.max(1,series[0].data.length-1);
 const Y=v=>T+(H-T-B)*(1-(yr(v)-a)/(b-a));
 ctx.font='10px monospace';ctx.strokeStyle='#333';ctx.fillStyle='#888';
 const ticks=ylog?[1e-7,1e-6,3e-6]:[lo+(hi-lo)*0,lo+(hi-lo)*.5,hi];
 ctx.textAlign='right';
 for(const t of ticks){{
   if(ylog&&t>hi)continue; if(!ylog)ctx.textAlign='right';
   const y=Y(t); if(y<T||y>H-B)continue;
   ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(W-R,y);ctx.stroke();
   ctx.fillText(ylog?t.toExponential(0):t.toFixed(3),L-4,y+3);
 }}
 for(const s of series){{
  ctx.strokeStyle=s.color;ctx.lineWidth=1.2;ctx.beginPath();
  s.data.forEach((v,i)=>{{if(v==null)return;const x=X(i),y=Y(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});
  ctx.stroke();
 }}
 ctx.textAlign='left';
 series.forEach((s,i)=>{{ctx.fillStyle=s.color;ctx.fillText(s.name,L+6+i*110,14)}});
}}
async function poll(){{
 const r=await fetch('/data'); const d=await r.json();
 document.getElementById('status').textContent=d.time+'  update '+d.latest;
 const rows=[['source_action_mae',d.series.source_action_mae.at(-1),'source_anchor_kl',d.series.source_anchor_kl.at(-1)],
  ['anchor_coef',d.series.anchor_coef_effective.at(-1),'w_mean(raw)',d.series.transition_anchor_weight_mean.at(-1)],
  ['frac w>2',d.series.transition_fraction.at(-1),'dev_p90',d.series.dev_p90.at(-1)],
  ['v_p90',d.series.v_p90.at(-1),'std_mean',d.series.std_mean.at(-1)]];
 document.getElementById('stats').innerHTML=rows.map(r=>'<tr>'+r.map((v,i)=>i%2?'<td>'+v+'</td>':'<th>'+v+'</th>').join('')+'</tr>').join('');
 chart(document.getElementById('c0'),[
  {{name:'action_mae',color:'#4cf',data:d.series.source_action_mae}},
  {{name:'rollout_kl',color:'#fc4',data:d.series.max_rollout_kl}},
  {{name:'anchor_kl',color:'#4c4',data:d.series.source_anchor_kl}}],false);
 chart(document.getElementById('c1'),[
  {{name:'anchor_coef',color:'#4cf',data:d.series.anchor_coef_effective}},
  {{name:'w_mean(raw)',color:'#fc4',data:d.series.transition_anchor_weight_mean}},
  {{name:'frac w>2',color:'#4c4',data:d.series.transition_fraction}}],false);
 chart(document.getElementById('c2'),[
  {{name:'v_p90',color:'#4cf',data:d.series.v_p90}},
  {{name:'dev_p90',color:'#fc4',data:d.series.dev_p90}},
  {{name:'turn_p90',color:'#4c4',data:d.series.turn_p90}}],false);
 chart(document.getElementById('c3'),[
  {{name:'actor_lr',color:'#f66',data:d.series.actor_lr}},
  {{name:'std_mean',color:'#4cf',data:d.series.std_mean}},
  {{name:'bank_size',color:'#888',data:d.series.bank_size}}],false);
}}
setInterval(poll,5000);poll();
</script></body></html>'''


def read_jsonl(path):
    rows = []
    try:
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def build_data(exptid, tail):
    anchor = read_jsonl(EXPT_DIR / exptid / 'anchor_metrics.jsonl')[-tail:]
    series = {}
    latest = anchor[-1]['update'] if anchor else '—'
    for key in ('source_action_mae', 'source_anchor_kl', 'max_rollout_kl',
                'anchor_coef_effective', 'transition_anchor_weight_mean',
                'transition_fraction', 'v_p90', 'dev_p90', 'turn_p90',
                'std_mean', 'actor_lr', 'bank_size'):
        series[key] = [d.get(key) for d in anchor]
    # bank_size is huge; scale to k for display
    if series['bank_size']:
        series['bank_size'] = [v / 1000 if v is not None else None for v in series['bank_size']]
    return {'exptid': exptid, 'latest': latest,
            'time': __import__('time').strftime('%H:%M:%S'), 'series': series}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exptid', nargs='?', default='transanchor_norm_seed42_20260925')
    p.add_argument('--port', type=int, default=8097)
    p.add_argument('--tail', type=int, default=600)
    a = p.parse_args()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith('/data'):
                body = json.dumps(build_data(a.exptid, a.tail)).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = PAGE.format(title=a.exptid).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    srv = ThreadingHTTPServer(('0.0.0.0', a.port), H)
    print(f'live dashboard for {a.exptid}: http://localhost:{a.port}  (Ctrl+C to stop)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
