"""Render the static C&P->Bible apparatus HTML from the consolidated store.

    .venv/bin/python scripts/cnp_apparatus.py   # -> results/crime_and_punishment/cnp_apparatus.html
"""
import json
import html as _html  # noqa: F401  (kept for future server-side escaping needs)
from collections import OrderedDict

import ngram_baseline as nb

H = nb.HERE


# only the fields the client JS actually reads from a candidate row — embedding just
# these (not the whole row) keeps the HTML small (drops per-row duplicated seg_text etc.)
_ROW_KEEP = ("band", "tier", "verse_text", "shared_words", "methods", "scores")


def pair_by_segment(rows):
    segs = OrderedDict()
    for r in rows:
        s = segs.setdefault(r["seg_id"], {
            "seg_id": r["seg_id"], "chapter": r["chapter"],
            "paragraph_ref": r["paragraph_ref"], "seg_text": r["seg_text"],
            "_lines": OrderedDict()})
        line = s["_lines"].setdefault(r["ref"], {
            "ref": r["ref"], "is_run": r["is_run"], "nt": None, "cs": None,
            "gold": {"is_gold": False}, "best_ce": None, "best_surface": None})
        line[r["corpus"].lower()] = {k: r[k] for k in _ROW_KEEP}
        if r["gold"]["is_gold"]:
            line["gold"] = r["gold"]
        ce = r["scores"].get("ce")
        if ce is not None and (line["best_ce"] is None or ce > line["best_ce"]):
            line["best_ce"] = ce
        ss = r["scores"].get("nt_surface_idf") or r["scores"].get("cs_surface_idf")
        if ss is not None and (line["best_surface"] is None or ss > line["best_surface"]):
            line["best_surface"] = ss
    out = []
    for s in segs.values():
        s["lines"] = list(s.pop("_lines").values())
        out.append(s)
    return out


def methods_present(segments):
    s = set()
    for seg in segments:
        for ln in seg["lines"]:
            for side in ("nt", "cs"):
                if ln.get(side):
                    s.update(ln[side]["methods"])
    return sorted(s)


_CSS = """
body{font:15px/1.5 Georgia,serif;margin:0;background:#faf8f4;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:5}
header label{font:13px sans-serif;margin-right:14px}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h2.chap{margin:28px 0 6px;font:600 18px sans-serif;color:#7a1f1f;border-bottom:2px solid #e3d8c8}
.card{background:#fff;border:1px solid #e6ddcf;border-radius:6px;padding:12px 14px;margin:10px 0}
.seg{font-style:italic;margin-bottom:8px}
.line{border-top:1px dashed #eee;padding:6px 0;display:grid;grid-template-columns:130px 1fr 1fr;gap:10px}
.ref{font:600 13px sans-serif}
.badge{font:11px sans-serif;border-radius:3px;padding:1px 5px;margin-right:4px}
.b-verbatim{background:#d9483b;color:#fff}.b-close{background:#e3a33b;color:#222}.b-weak{background:#eee;color:#888}
.gold{background:#2e7d32;color:#fff}
.meth{color:#888;font:11px sans-serif}
.blank{color:#ccc}
.hl{background:#fff1a8;font-weight:600}
details.tail summary{font:12px sans-serif;color:#999;cursor:pointer}
#overlay{border-collapse:collapse}
#overlay td{font:13px sans-serif;padding:3px 8px;border-bottom:1px solid #eee;vertical-align:top}
#overlay td.gtxt{font-family:Georgia,serif;max-width:440px}
.qgap{color:#b9a98a;font-style:normal}
.st-detected{color:#2e7d32}.st-gap{color:#d9483b;font-weight:600}.st-out{color:#aaa}
table.flat{border-collapse:collapse;font:12px sans-serif;width:100%}
table.flat th,table.flat td{border:1px solid #ddd;padding:2px 6px;text-align:left;cursor:pointer}
.hidden{display:none}
.mrow{margin-top:6px;font:12px sans-serif;color:#555}
.mlab{margin-right:10px;white-space:nowrap}
"""

_JS = r"""
const DATA = window.__APP__;
const $ = s => document.querySelector(s);
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML;}
function norm(w){return w.toLowerCase().replace(/ё/g,'е');}
function tokenSet(t){const s=new Set();(String(t||'').toLowerCase().replace(/ё/g,'е').match(/[а-яa-z]{2,}/g)||[]).forEach(w=>s.add(w));return s;}
// Highlight the WHOLE words of `text` that also occur in `other` (a token set) —
// case-insensitive, ё-folded, whole-word only. Shows the actual shared phrasing
// (incl. short words like бы/ты/был), unlike the old length>=4 surface heuristic.
function hlShared(text, other){
  if(!other||!other.size) return esc(text);
  return esc(text).replace(/[А-Яа-яЁёA-Za-z]+/g, w => other.has(norm(w)) ? '<span class="hl">'+w+'</span>' : w);
}
function scoreBadges(row){
  if(!row) return '<span class="blank">—</span>';
  const s=row.scores, parts=[];
  if(s.ce!=null) parts.push('ce '+s.ce.toFixed(2));
  if(s.sw!=null) parts.push('sw '+(+s.sw).toFixed(1));
  if(s.dense_cos!=null) parts.push('cos '+s.dense_cos.toFixed(2));
  if(s.dense_cos_c!=null) parts.push('cos_c '+(+s.dense_cos_c).toFixed(2));
  if(s.longest_lemma_ngram!=null) parts.push('ngram '+s.longest_lemma_ngram);
  if(s.sum_idf!=null) parts.push('idf '+(+s.sum_idf).toFixed(0));
  if(s.cs_surface_idf!=null) parts.push('cs-idf '+(+s.cs_surface_idf).toFixed(0));
  if(s.nt_surface_idf!=null) parts.push('nt-idf '+(+s.nt_surface_idf).toFixed(0));
  return '<span class="meth">'+parts.join(' · ')+'</span>';
}
function cell(row, corpus, segSet){
  if(!row) return '<span class="blank">— ('+corpus+')</span>';
  return '<span class="badge b-'+row.band+'">'+row.band+'</span>'
    + (row.tier?'<span class="badge gold">tier '+row.tier+'</span>':'')
    + '<br>'+hlShared(row.verse_text, segSet)
    + '<br>'+scoreBadges(row)
    + '<div class="meth">'+row.methods.join(', ')+'</div>';
}
function thr(){return parseFloat($('#thr').value);}
function sthr(){return parseFloat($('#sthr').value);}
function lineHTML(seg, ln, segSet){
  const show = (ln.best_ce!=null && ln.best_ce>=thr())
            || (ln.best_surface!=null && ln.best_surface>=sthr()) || ln.gold.is_gold;
  const corpus = $('#corpus').value;
  const nt = (corpus!=='CS') ? cell(ln.nt,'NT',segSet) : '';
  const cs = (corpus!=='NT') ? cell(ln.cs,'CS',segSet) : '';
  return {show, html:'<div class="line"><div class="ref">'+esc(ln.ref)
     + (ln.gold.is_gold?' <span class="badge gold">GOLD</span>':'')
     + '</div><div>'+nt+'</div><div>'+cs+'</div></div>'};
}
function passSearch(seg){
  const q=$('#q').value.toLowerCase(); if(!q) return true;
  if(seg.seg_text.toLowerCase().includes(q)) return true;
  return seg.lines.some(l=>(l.nt&&l.nt.verse_text.toLowerCase().includes(q))
                          ||(l.cs&&l.cs.verse_text.toLowerCase().includes(q)));
}
function checkedMethods(){return new Set([...document.querySelectorAll('.mf:checked')].map(e=>e.value));}
function strongOnly(){return $('#strong').checked;}
function isStrong(l){return [l.nt,l.cs].some(r=>r&&(r.band==='verbatim'||r.band==='close'||r.tier==='A'||r.tier==='B'||r.tier==='C'));}
function render(){
  const goldOnly=$('#gold').value, cm=checkedMethods(), so=strongOnly(), root=$('#cards');
  let chap='', html='';
  DATA.segments.forEach(seg=>{
    if(!passSearch(seg)) return;
    const segLines=seg.lines.filter(l=>{
      const ms=[].concat(l.nt?l.nt.methods:[], l.cs?l.cs.methods:[]);
      if(!ms.some(m=>cm.has(m))) return false;          // per-method filter
      if(goldOnly==='gold' && !l.gold.is_gold) return false;
      if(so && !isStrong(l)) return false;              // strong-only: verbatim band or tier A/B
      return true;
    });
    if(!segLines.length) return;
    const segSet=tokenSet(seg.seg_text);
    const lines=segLines.map(l=>{const o=lineHTML(seg,l,segSet); if(goldOnly==='gold'||so)o.show=true; return o;});
    const strong=lines.filter(l=>l.show), tail=lines.filter(l=>!l.show);
    if(!strong.length && !tail.length) return;
    const vunion=new Set();          // words the novel sentence shares with any candidate verse
    segLines.forEach(l=>[l.nt,l.cs].forEach(r=>{if(r)tokenSet(r.verse_text).forEach(w=>vunion.add(w));}));
    if(seg.chapter!==chap){chap=seg.chapter;html+='<h2 class="chap">'+esc(chap)+'</h2>';}
    html+='<div class="card"><div class="seg">'+hlShared(seg.seg_text, vunion)+' <span class="meth">'+esc(seg.seg_id)+'</span></div>';
    html+=strong.map(l=>l.html).join('');
    if(tail.length) html+='<details class="tail"><summary>'+tail.length+' weaker candidate(s)</summary>'+tail.map(l=>l.html).join('')+'</details>';
    html+='</div>';
  });
  root.innerHTML=html;
}
// Novel side of a gold line: the reading sentence(s) of the quotation, words shared with the
// cited verse highlighted. A running quote split by narration is shown as separate runs — the
// dropped narration between two reading runs becomes an elision marker «[…]», so the citation is
// never silently glued across the gap. Surrounding paragraphs live in the apparatus cards below.
function overlayNovel(e){
  const para=e.paragraph||[];
  if(!para.some(s=>s.hit)) return esc(e.fragment||'');
  const vset=tokenSet(e.gold_text||'');
  const runs=[]; let cur=[];
  para.forEach(s=>{ if(s.hit) cur.push(s.t); else if(cur.length){runs.push(cur); cur=[];} });
  if(cur.length) runs.push(cur);
  return runs.map(r=>hlShared(r.join(' '), vset)).join('<span class="qgap"> […] </span>');
}
// Bible side: quotations show the full cited range (it aligns with the novel quote). For a
// thematic/meta note the verse is just context for an allusion (e.g. the «Лазаря петь» idiom →
// the whole Лк 16:19—31 parable), so cap a long citation to its first 3 sentences + «[…]».
function overlayBible(e){
  const full=e.gold_text||''; if(!full) return '—';
  if(e.type==='quotation'||e.type==='allusion') return esc(full);
  const parts=full.split(/(?<=\.)\s+/);
  return parts.length<=3 ? esc(full)
    : esc(parts.slice(0,3).join(' '))+'<span class="qgap"> […]</span>';
}
function renderOverlay(){
  const cm=checkedMethods();   // overlay is method-aware: a quotation counts as detected only
  const head='<tr><th>note</th><th>type</th><th>novel (matched phrase)</th><th>ref</th>'
    +'<th>Bible (cited verse)</th><th>status</th><th>detected by (methods)</th></tr>';
  const rows=DATA.overlay.map(e=>{
    const det=(e.methods||[]).filter(m=>cm.has(m));         // checked methods that reach this line
    const inscope=(e.type==='quotation'||e.type==='allusion');   // verbal targets: quotation OR allusion
    const reached=!inscope && det.length>0;                 // an out-of-scope note a method still hits
    const status=inscope ? (det.length?'detected':'gap') : 'out-of-scope';
    const st=(status==='out-of-scope'?'out':status);
    const lbl=(status==='gap'?'missed':status)+(reached?' · reached':'');
    return '<tr><td>'+esc(e.id)+'</td><td>'+esc(e.type)+'</td>'
      +'<td class="gtxt">'+overlayNovel(e)+'</td>'
      +'<td>'+esc(e.ref_str||'')+'</td>'
      +'<td class="gtxt">'+overlayBible(e)+'</td>'
      +'<td class="st-'+st+'">'+lbl+'</td>'
      +'<td class="meth">'+esc(det.join(', '))+'</td></tr>';
  }).join('');
  $('#overlay').innerHTML=head+rows;
}
['thr','sthr','corpus','gold','q'].forEach(id=>{const e=document.getElementById(id);e&&e.addEventListener('input',render);});
{const e=$('#strong');e&&e.addEventListener('change',render);}
document.querySelectorAll('.mf').forEach(e=>e.addEventListener('change',()=>{render();renderOverlay();}));
renderOverlay(); render();
"""


def render_html(segments, overlay):
    payload = json.dumps({"segments": segments, "overlay": overlay}, ensure_ascii=False)
    mchecks = "".join(
        "<label class='mlab'><input type='checkbox' class='mf' value='%s'> %s</label>" % (m, m)
        for m in methods_present(segments))
    return (
        "<!DOCTYPE html>\n<html lang='ru'><head><meta charset='utf-8'>"
        "<title>C&amp;P → Bible apparatus</title><style>" + _CSS + "</style></head><body>"
        "<header>"
        "<label>min ce <input id='thr' type='range' min='0' max='1' step='0.05' value='0.3'></label>"
        "<label>min surf-idf <input id='sthr' type='range' min='5' max='40' step='1' value='12'></label>"
        "<label>corpus <select id='corpus'><option>both</option><option>NT</option><option>CS</option></select></label>"
        "<label>gold <select id='gold'><option value='all'>all</option><option value='gold'>gold only</option></select></label>"
        "<label title='show only verbatim/close-band or tier A–C candidates; hide the weak ones'>"
        "<input id='strong' type='checkbox'> strong only (verbatim+close / tier A–C)</label>"
        "<label>search <input id='q' type='text' placeholder='novel or verse text'></label>"
        "<div class='mrow'>methods: " + mchecks + "</div>"
        "</header><div class='wrap'>"
        "<h2 class='chap'>Gold overlay</h2><table id='overlay'></table>"
        "<h2 class='chap'>Apparatus</h2><div id='cards'></div>"
        "</div><script>window.__APP__=" + payload + ";</script><script>" + _JS + "</script>"
        "</body></html>")


def main():
    rows = [json.loads(l) for l in open(H + "results/crime_and_punishment/cnp_connections.jsonl", encoding="utf-8")]
    overlay = json.load(open(H + "results/crime_and_punishment/cnp_gold_overlay.json", encoding="utf-8"))
    segments = pair_by_segment(rows)
    out = render_html(segments, overlay)
    with open(H + "results/crime_and_punishment/cnp_apparatus.html", "w", encoding="utf-8") as f:
        f.write(out)
    import sys
    print(f"{len(segments)} segment cards -> results/crime_and_punishment/cnp_apparatus.html", file=sys.stderr)


if __name__ == "__main__":
    main()
