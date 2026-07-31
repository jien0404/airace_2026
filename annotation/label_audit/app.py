# -*- coding: utf-8 -*-
"""Giao diện duyệt đề xuất sửa nhãn — hai quyết định: đồng ý sửa / giữ nguyên.

    python -m annotation.label_audit.app --audit-dir annotation/data/label_audit --port 5001

Quyết định lưu vào `<audit-dir>/decisions.json`. Không sửa nhãn gốc; muốn xuất bản đã sửa thì
chạy `python -m annotation.label_audit.run apply`.

Phím tắt: A = đồng ý sửa · G = giữ nguyên · ← → = chuyển đề xuất.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
AUDIT_DIR: Path | None = None


def _findings() -> list[dict]:
    path = AUDIT_DIR / "findings.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _decisions_path() -> Path:
    return AUDIT_DIR / "decisions.json"


def _load_decisions() -> dict:
    path = _decisions_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.route("/api/findings")
def api_findings():
    decisions = _load_decisions()
    rows = []
    for finding in _findings():
        key = f"{finding['file']}#{finding['entity_index']}"
        rows.append({
            "id": key,
            "source": finding["source"],
            "file": finding["file"],
            "kind": finding["kind"],
            "rule": finding["rule"],
            "mechanical": finding["mechanical"],
            "type": finding["entity"]["type"],
            "surface": finding["entity"]["text"],
            "current": finding["current"]["assertions"],
            "proposed": finding["proposed_assertions"],
            "current_type": finding["current"]["type"],
            "proposed_type": finding.get("proposed_type"),
            "proposed_span": finding.get("proposed_text"),
            "screen_reason": finding["reason"],
            "llm_reason": (finding.get("llm") or {}).get("reason", ""),
            "context": finding["context"],
            "decision": decisions.get(key),
        })
    return jsonify({"findings": rows, "decided": len(decisions)})


@app.route("/api/decide", methods=["POST"])
def api_decide():
    payload = request.get_json(force=True)
    key, decision = payload.get("id"), payload.get("decision")
    if decision not in {"accept", "reject", None}:
        return jsonify({"error": "decision phải là accept/reject/null"}), 400
    decisions = _load_decisions()
    if decision is None:
        decisions.pop(key, None)
    else:
        decisions[key] = decision
    _decisions_path().write_text(
        json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return jsonify({"ok": True, "decided": len(decisions)})


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Duyệt sửa nhãn</title><style>
:root{--bg:#111418;--fg:#e6e6e6;--mut:#8b949e;--acc:#2ea043;--rej:#d1584f;--card:#1b2028}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 system-ui,sans-serif;background:var(--bg);color:var(--fg);display:flex;height:100vh}
#side{width:290px;border-right:1px solid #2b313a;overflow:auto;flex:none}
#side .it{padding:7px 10px;border-bottom:1px solid #22272e;cursor:pointer;font-size:13px}
#side .it:hover{background:#20262e}#side .it.sel{background:#2b3440}
#side .it.accept{border-left:3px solid var(--acc)}#side .it.reject{border-left:3px solid var(--rej)}
#main{flex:1;overflow:auto;padding:22px 28px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;background:#2b3440;color:var(--mut);margin-right:6px}
.ctx{background:var(--card);padding:16px;border-radius:8px;white-space:pre-wrap;line-height:1.75;margin:14px 0}
mark{background:#f0c674;color:#111;padding:1px 3px;border-radius:3px;font-weight:600}
.cmp{display:flex;gap:26px;margin:14px 0;align-items:center}
.cmp div{padding:9px 14px;border-radius:7px;background:var(--card)}
.cmp .k{background:none;color:var(--mut);width:74px;padding-left:0;font-size:12px;text-transform:uppercase}
.old{text-decoration:line-through;color:var(--rej)}.new{color:var(--acc);font-weight:600}
button{font:inherit;padding:9px 18px;border:0;border-radius:7px;cursor:pointer;margin-right:9px}
.ok{background:var(--acc);color:#fff}.no{background:var(--rej);color:#fff}.un{background:#39424e;color:var(--fg)}
h2{margin:0 0 4px}.mut{color:var(--mut);font-size:13px}
#bar{position:sticky;top:0;background:var(--bg);padding-bottom:10px;border-bottom:1px solid #2b313a;margin-bottom:14px}
</style></head><body>
<div id="side"></div>
<div id="main"><div id="bar"><b id="prog"></b> <span class="mut" id="hint">A = đồng ý sửa · G = giữ nguyên · ←/→ chuyển</span></div><div id="body"></div></div>
<script>
let F=[],i=0;
async function load(){const r=await(await fetch('/api/findings')).json();F=r.findings;draw();}
function draw(){
 const s=document.getElementById('side');
 s.innerHTML=F.map((f,k)=>`<div class="it ${f.decision||''} ${k===i?'sel':''}" onclick="sel(${k})">
   <b>${f.surface.slice(0,26)}</b><br><span class="mut">${f.file} · ${f.kind}</span></div>`).join('');
 const f=F[i]; if(!f){document.getElementById('body').innerHTML='<p>Không có đề xuất nào.</p>';return;}
 const done=F.filter(x=>x.decision).length;
 document.getElementById('prog').textContent=`${i+1}/${F.length} · đã quyết ${done}`;
 const esc=t=>String(t).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
 const row=(k,a,b)=>a===b?'':`<div class="cmp"><div class="k">${k}</div>
   <div><span class="old">${esc(a)}</span></div><div>→</div>
   <div><span class="new">${esc(b)}</span></div></div>`;
 document.getElementById('body').innerHTML=`
  <h2>${esc(f.surface)} <span class="badge">${f.type}</span></h2>
  <div class="mut">${f.file} · ${f.source} · ${f.kind}${f.mechanical?' · luật cơ học':''}</div>
  ${row('assertion', JSON.stringify(f.current), JSON.stringify(f.proposed))}
  ${f.proposed_type ? row('type', f.current_type, f.proposed_type) : ''}
  ${f.proposed_span ? row('span', f.surface, f.proposed_span) : ''}
  <div class="mut"><b>Luật:</b> ${esc(f.rule)}</div>
  <div class="mut"><b>Sàng lọc:</b> ${esc(f.screen_reason)}</div>
  ${f.llm_reason?`<div class="mut"><b>LLM:</b> ${esc(f.llm_reason)}</div>`:''}
  <div class="ctx">${esc(f.context.before)}<mark>${esc(f.context.surface)}</mark>${esc(f.context.after)}</div>
  <button class="ok" onclick="dec('accept')">Đồng ý sửa (A)</button>
  <button class="no" onclick="dec('reject')">Giữ nguyên (G)</button>
  <button class="un" onclick="dec(null)">Bỏ quyết định</button>
  <div class="mut" style="margin-top:10px">Quyết định hiện tại: <b>${f.decision||'chưa'}</b></div>`;
 const el=s.children[i]; if(el) el.scrollIntoView({block:'nearest'});
}
function sel(k){i=k;draw();}
async function dec(d){const f=F[i];if(!f)return;
 await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({id:f.id,decision:d})});
 f.decision=d; if(d&&i<F.length-1)i++; draw();}
document.addEventListener('keydown',e=>{
 if(e.key==='a'||e.key==='A')dec('accept');
 else if(e.key==='g'||e.key==='G')dec('reject');
 else if(e.key==='ArrowRight'&&i<F.length-1){i++;draw();}
 else if(e.key==='ArrowLeft'&&i>0){i--;draw();}});
load();
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


def main() -> None:
    global AUDIT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    AUDIT_DIR = Path(args.audit_dir).resolve()
    if not AUDIT_DIR.exists():
        raise SystemExit(f"Không có thư mục {AUDIT_DIR}")
    print(f"Duyệt tại http://127.0.0.1:{args.port}  ({AUDIT_DIR})")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
