# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained notebook viewer; layout geometry is computed in Python."""

import json
from html import escape

_STYLE = r"""
.kumo-graph{position:relative;display:flex;flex-direction:column;
max-width:100%;padding:8px;box-sizing:border-box;background:white;
color:#202020;font:14px Arial,Helvetica,sans-serif}
.kumo-graph [hidden]{display:none!important}
.kumo-graph:fullscreen{width:100%;height:100%;max-height:none}
.kumo-graph-toolbar{display:flex;align-items:center;justify-content:flex-end;
flex-wrap:wrap;gap:6px;flex:none;margin-bottom:8px}
.kumo-graph-layout-label{display:flex;align-items:center;gap:6px;margin-right:auto;
font:13px Arial,Helvetica,sans-serif;color:#555}
.kumo-graph-layout{height:32px;max-width:140px;font:13px Arial,Helvetica,sans-serif;
color:#333;background:#fff;border:1px solid #d1d1d1;border-radius:8px;padding:0 8px}
.kumo-graph-zoom{display:flex;align-items:center;gap:4px;flex:none}
.kumo-graph-toolbar button{box-sizing:border-box;display:inline-flex;
align-items:center;justify-content:center;flex:0 0 32px;
width:32px!important;min-width:32px!important;max-width:32px!important;
height:32px!important;min-height:32px!important;max-height:32px!important;
padding:6px;margin:0;background:#f5f5f5;color:#555;
border:1px solid #d1d1d1;border-radius:10px;corner-shape:squircle;
cursor:pointer;font:18px/18px Arial,Helvetica,sans-serif}
.kumo-graph-toolbar button:hover{background:#e8e8e8}
.kumo-graph-toolbar button:focus-visible,.kumo-graph-layout:focus-visible{
outline:2px solid #598d13;outline-offset:2px}
.kumo-graph-toolbar button>svg{display:block;flex:none;
width:18px!important;height:18px!important}
.kumo-graph-scale{min-width:44px;text-align:center;color:#666;
font:12px Arial,Helvetica,sans-serif;font-variant-numeric:tabular-nums}
.kumo-graph .kumo-graph-exit{display:none}
.kumo-graph:fullscreen .kumo-graph-enter{display:none}
.kumo-graph:fullscreen .kumo-graph-exit{display:inline-flex}
.kumo-graph-view{position:relative;overflow:auto;min-height:0;flex:none;
max-width:100%;
background:radial-gradient(circle at 1px 1px,#d9ddd5 .7px,
transparent .8px) 0 0/16px 16px}
.kumo-graph:fullscreen .kumo-graph-view{flex:1;height:auto!important}
.kumo-graph-scene{display:flex;align-items:center;justify-content:center;
min-width:100%;min-height:100%}
.kumo-graph-view svg{display:block;flex:none;max-width:none!important;
max-height:none!important}
.kumo-graph-status{font-size:13px;color:#666;padding:6px 0}
.kumo-graph-view svg>rect{fill:transparent}
.kumo-graph-tooltip{position:absolute;top:0;left:0;z-index:10;
box-sizing:border-box;
width:320px;max-width:calc(100% - 16px);border:1px solid #cbd8ba;
border-radius:18px;corner-shape:squircle;background:#fff;color:#202020;
box-shadow:0 5px 18px #24331416,0 1px 3px #2433140a;
font:16px/1.4 Arial,Helvetica,sans-serif;overflow-wrap:anywhere;
overflow:hidden;pointer-events:none;opacity:0;visibility:hidden;
transform:translateY(4px);transition:opacity 140ms ease,transform 140ms ease,
visibility 140ms ease}
.kumo-graph-tooltip[data-open]{opacity:1;visibility:visible;
transform:translateY(0)}
.kumo-tip-head{background:#f4f8ec;border-bottom:1px solid #e1e9d6;
padding:7px 10px}
.kumo-tip-context{color:#6a7a58;font-size:16px;margin-bottom:2px}
.kumo-tip-name{font-size:17px;font-weight:700;color:#386000}
.kumo-tip-body{padding:7px 10px}
.kumo-tip-badges{display:flex;flex-wrap:wrap;gap:5px}
.kumo-tip-samples{display:flex;gap:5px}
.kumo-tip-badge{background:#f2f4ef;border:1px solid #e1e6db;
border-radius:7px;corner-shape:squircle;padding:2px 6px;color:#536247}
.kumo-tip-key{background:#ecf4df;border-color:#d1e2b7;color:#487014}
.kumo-tip-type{color:#536247}
.kumo-tip-ref{padding-top:5px;color:#5b6751}
.kumo-tip-connections{list-style:none;padding:0;margin:0;display:grid;gap:4px}
.kumo-tip-connection{color:#455239;font-weight:500}
.kumo-tip-label{color:#7a8470;margin:5px 0 3px}
.kumo-tip-sample{flex:1;min-width:0;max-width:100%;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis;background:#f5f7f2;border-radius:7px;corner-shape:squircle;
padding:2px 6px;color:#455239;
font:16px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace}
.kumo-graph [data-kumo-tip]{outline:none;cursor:default}
.kumo-graph .kumo-row-light{fill:#edf6df;stroke:#9bc958;stroke-width:1;
opacity:0;pointer-events:none;transition:opacity 140ms ease}
.kumo-graph .kumo-active .kumo-row-light{opacity:1}
.kumo-graph [data-kumo-tip]:focus-visible .kumo-row-light{
stroke:#598d13;stroke-width:2}
.kumo-graph .kumo-table-veil{fill:#26341a;opacity:0;pointer-events:none;
transition:opacity 140ms ease}
.kumo-graph[data-highlight=table] .kumo-focused>.kumo-table-veil{opacity:.16}
.kumo-graph[data-highlight=table] .kumo-row-light{fill:transparent}
.kumo-graph .kumo-edge{transition:stroke 140ms ease,stroke-width 140ms ease}
.kumo-graph:fullscreen .kumo-edge{vector-effect:non-scaling-stroke}
.kumo-graph .kumo-edge-active{stroke:#6e9947;stroke-width:1.8}
@media(prefers-reduced-motion:reduce){.kumo-graph-tooltip,
.kumo-graph .kumo-row-light,.kumo-graph .kumo-table-veil,
.kumo-graph .kumo-edge{transition:none}}

.kumo-graph .kumo-edge{vector-effect:non-scaling-stroke}
"""

_HOVER = r"""
const tip=document.createElement('div');
tip.className='kumo-graph-tooltip';tip.role='tooltip';
tip.id='kumo-tip-'+Math.random().toString(36).slice(2);host.append(tip);
let active=null, card=null, timer;
const element=(cls,text)=>{const n=document.createElement('div');
n.className='kumo-tip-'+cls;n.textContent=text;return n};
const hide=()=>{clearTimeout(timer);tip.removeAttribute('data-open');
edges.forEach(edge=>edge.classList.remove('kumo-edge-active'));
if(active){active.removeAttribute('aria-describedby');
active.classList.remove('kumo-active')}
card?.classList.remove('kumo-focused');active=card=null};
const content=el=>{
const row=!!el.dataset.kumoRow, table=el.dataset.kumoTable;
const name=el.dataset.kumoColumn, full=table+(row?'.'+name:'');
const lines=el.dataset.kumoTip.slice(full.length+1).split('\n');
const head=element('head',''), body=element('body','');
head.append(element('context',row?table:'Table'),element('name',row?name:table));
const badges=element('badges','');
if(row){
for(const [cls,prefix] of [[' kumo-tip-key','Role: ']]){
const value=lines.find(x=>x.startsWith(prefix))?.slice(prefix.length);
if(value)if(value!=='Column')badges.append(element('badge'+cls,value))}
if(badges.childElementCount)body.append(badges);
for(const [prefix,label] of [['Type: ','Data type: '],
['Semantic type: ','Semantic type: ']]){
const value=lines.find(x=>x.startsWith(prefix))?.slice(prefix.length);
if(value)body.append(element('type',label+value))}
for(const ref of lines.filter(x=>x.startsWith('References: ')))
body.append(element('ref','↗ '+ref.slice(12)));
const index=lines.findIndex(x=>x.startsWith('Sample values:'));
if(index>=0){body.append(element('label','Sample values'));
const status=lines[index].slice(14).trim(), samples=element('samples','');
for(let value of status?[status]:lines.slice(index+1,index+4)){
value=value.replace(/^'|'$/g,'');
samples.append(element('sample',
value.length>90?value.slice(0,87)+'…':value))}body.append(samples)}
}else{
badges.append(element('badge',lines[0]));
body.append(badges);
const connections=JSON.parse(el.dataset.kumoConnections);
body.append(element('label',connections.length?'Connected tables':
'No connected tables'));
if(connections.length){const list=document.createElement('ul');
list.className='kumo-tip-connections';
for(const name of connections){const item=document.createElement('li');
item.className='kumo-tip-connection';item.textContent=name;list.append(item)}
body.append(list)}}
tip.replaceChildren(head,body)};
const place=()=>{
if(!active)return;
const h=host.getBoundingClientRect(), a=active.getBoundingClientRect();
const c=card.getBoundingClientRect(), v=view.getBoundingClientRect();
let left=Math.max(h.left,0), right=Math.min(h.right,innerWidth);
let top=Math.max(h.top,0), bottom=Math.min(h.bottom,innerHeight);
for(let el=host.matches(':fullscreen')?null:host.parentElement;
el;el=el.parentElement){
if(el===document.body||el===document.documentElement)break;
const style=getComputedStyle(el), r=el.getBoundingClientRect();
if(/auto|scroll|hidden|clip/.test(style.overflowX)){
left=Math.max(left,r.left);right=Math.min(right,r.right)}
if(/auto|scroll|hidden|clip/.test(style.overflowY)){
top=Math.max(top,r.top);bottom=Math.min(bottom,r.bottom)}}
if(a.right>Math.max(left,v.left))if(Math.min(right,v.right)>a.left)
if(a.bottom>Math.max(top,v.top))if(Math.min(bottom,v.bottom)>a.top){
left+=8;right-=8;top+=8;bottom-=8;
tip.style.maxWidth=Math.max(0,Math.min(320,right-left))+'px';
tip.style.maxHeight=Math.max(0,bottom-top)+'px';
const t=tip.getBoundingClientRect();
let x=c.right+12, side='right';
if(x+t.width>right){x=c.left-t.width-12;side='left'}
const raw=x; x=Math.max(left,Math.min(x,right-t.width));
const y=Math.max(top,Math.min(a.top,bottom-t.height));
tip.dataset.side=side;tip.dataset.clamped=String(x!==raw||y!==a.top);
tip.style.left=x-h.left+'px';tip.style.top=y-h.top+'px';
return}hide()};
const show=el=>{clearTimeout(timer);if(active===el)return;
hide();active=el;card=el.parentElement;
el.classList.add('kumo-active');card.classList.add('kumo-focused');
content(el);el.setAttribute('aria-describedby',tip.id);
if(el.dataset.kumoRow){
const table=el.dataset.kumoTable, column=el.dataset.kumoColumn;
edges.forEach(edge=>{const d=edge.dataset;
const src=d.srcTable===table?d.srcColumn===column:false;
const dst=d.dstTable===table?d.dstColumn===column:false;
edge.classList.toggle('kumo-edge-active',src||dst)})}
tip.setAttribute('data-open','');place()};
view.querySelectorAll('title').forEach(el=>el.remove());
view.querySelectorAll('svg>g[transform]').forEach(g=>{
const veil=g.querySelector(':scope>path').cloneNode();
veil.classList.add('kumo-table-veil');veil.removeAttribute('fill');
veil.removeAttribute('stroke');g.append(veil)});
view.querySelectorAll('[data-kumo-tip]').forEach(el=>{
const light=el.querySelector('.kumo-surface').cloneNode();
light.classList.add('kumo-row-light');light.removeAttribute('fill');
light.removeAttribute('stroke');el.insertBefore(light,el.querySelector('text'));
el.addEventListener('mouseenter',()=>show(el),{signal});
el.addEventListener('mouseleave',()=>{timer=setTimeout(()=>{
if(active!==document.activeElement)hide()},65)},{signal});
el.addEventListener('focus',()=>show(el),{signal});
el.addEventListener('blur',hide,{signal})});
view.addEventListener('scroll',()=>{
view.style.backgroundPosition=(-view.scrollLeft)+'px '+(-view.scrollTop)+'px';place()},{signal});

"""

_CAMERA = r"""
const selector=host.querySelector('.kumo-graph-layout');
const label=host.querySelector('.kumo-graph-scale');
const status=host.querySelector('[role=status]');
const enter=host.querySelector('.kumo-graph-enter');
const exit=host.querySelector('.kumo-graph-exit');
const payload=JSON.parse(host.querySelector('.kumo-graph-layout-data').textContent);
const layouts=payload.layouts;
let selected=payload.initial, scale=1, fitted=true, full=false, inlineCamera=null;
let layoutWidth=Number(svg.getAttribute('width'))||304;
let layoutHeight=Number(svg.getAttribute('height'))||112;
let resizeFrame=0, disposed=false;
const cards=new Map(Array.from(svg.querySelectorAll(':scope>[data-kumo-card]'),
el=>[el.dataset.kumoCard,el]));
const controls=host.querySelectorAll('button,select');
controls.forEach(el=>el.disabled=false);
selector.value=selected;
if(Object.keys(layouts).length<2)selector.closest('label').hidden=true;

// Only applyLayout changes graph geometry. Camera actions never call it.
const applyLayout=name=>{
const layout=layouts[name];if(!layout)return;
hide();
for(const [name,position] of Object.entries(layout.positions)){
const card=cards.get(name);
if(card)card.setAttribute('transform',`translate(${position[0]} ${position[1]})`)}
edges.forEach((edge,index)=>edge.setAttribute('d',layout.paths[index]));
svg.querySelectorAll(':scope>circle,:scope>.kumo-graph-handles').forEach(el=>el.remove());
const handles=document.createElementNS(svg.namespaceURI,'g');
handles.setAttribute('class','kumo-graph-handles');
for(const [x,y] of layout.handles){
const circle=document.createElementNS(svg.namespaceURI,'circle');
for(const [key,value] of Object.entries({cx:x,cy:y,r:4,fill:'#64748b'}))
circle.setAttribute(key,value);handles.append(circle)}
svg.append(handles);layoutWidth=layout.width;layoutHeight=layout.height;
svg.setAttribute('viewBox',`0 0 ${layoutWidth} ${layoutHeight}`);
svg.setAttribute('width',layoutWidth);svg.setAttribute('height',layoutHeight);
svg.querySelectorAll(':scope>rect').forEach(rect=>{
rect.setAttribute('width',layoutWidth);rect.setAttribute('height',layoutHeight)});
selected=name;host.dataset.layout=name;
};
const inlineHeight=()=>Math.min(720,Math.max(240,layoutHeight+32));
// Leave one extra pixel per side beyond the scene's 16px padding.
// Integer client metrics can otherwise round a fit past a fractional edge
// and repeatedly add/remove scrollbars as the observer refits the graph.
const fitScale=()=>Math.min(1,
Math.max(1,view.clientWidth-34)/layoutWidth,
Math.max(1,view.clientHeight-34)/layoutHeight);
const resizeScene=()=>{
svg.style.width=layoutWidth*scale+'px';svg.style.height=layoutHeight*scale+'px';
// CSS fills the viewport without rounding its fractional content box.
// Flooring the scene at clientWidth/clientHeight can toggle both scrollbars
// forever when a scrollbar changes the available fractional width.
scene.style.width=layoutWidth*scale+32+'px';
scene.style.height=layoutHeight*scale+32+'px';
label.textContent=Math.round(scale*100)+'%';
host.dataset.scale=String(scale);
};
const fit=()=>{
hide();fitted=true;scale=fitScale();resizeScene();
view.scrollLeft=(scene.offsetWidth-view.clientWidth)/2;
view.scrollTop=(scene.offsetHeight-view.clientHeight)/2;
};
const zoom=(next,x=view.clientWidth/2,y=view.clientHeight/2)=>{
hide();fitted=false;
const oldX=(scene.offsetWidth-layoutWidth*scale)/2;
const oldY=(scene.offsetHeight-layoutHeight*scale)/2;
const graphX=(view.scrollLeft+x-oldX)/scale;
const graphY=(view.scrollTop+y-oldY)/scale;
scale=Math.max(Math.min(.1,fitScale()),Math.min(4,next));resizeScene();
view.scrollLeft=(scene.offsetWidth-layoutWidth*scale)/2+graphX*scale-x;
view.scrollTop=(scene.offsetHeight-layoutHeight*scale)/2+graphY*scale-y;
};
selector.addEventListener('change',()=>{
applyLayout(selector.value);if(!full)view.style.height=inlineHeight()+'px';fit();
},{signal});
host.querySelector('.kumo-graph-fit').addEventListener('click',fit,{signal});
host.querySelector('.kumo-graph-zoom-in').addEventListener('click',()=>zoom(scale*1.25),{signal});
host.querySelector('.kumo-graph-zoom-out').addEventListener('click',()=>zoom(scale/1.25),{signal});
view.addEventListener('wheel',e=>{
if(!e.ctrlKey&&!e.metaKey)return;
e.preventDefault();const box=view.getBoundingClientRect();
const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?view.clientHeight:1);
zoom(scale*Math.exp(-delta*.002),e.clientX-box.left,e.clientY-box.top);
},{passive:false,signal});
enter.addEventListener('click',()=>{
status.hidden=true;
// Capture before fullscreen CSS changes viewport dimensions and clamps scrolling.
inlineCamera={selected,scale,fitted,left:view.scrollLeft,top:view.scrollTop};
const denied=()=>{status.hidden=false;inlineCamera=null};
try{Promise.resolve(host.requestFullscreen()).catch(denied)}
catch(error){denied()}
},{signal});
exit.addEventListener('click',()=>{
if(document.fullscreenElement===host)document.exitFullscreen();
},{signal});
host.addEventListener('fullscreenchange',()=>{
const entering=document.fullscreenElement===host;
if(entering===full)return;
hide();full=entering;
if(full){
if(!inlineCamera)inlineCamera={selected,scale,fitted,left:view.scrollLeft,top:view.scrollTop};
fit();exit.focus({preventScroll:true});
}else{
view.style.height=inlineHeight()+'px';
if(inlineCamera&&inlineCamera.selected===selected){
scale=inlineCamera.scale;fitted=inlineCamera.fitted;resizeScene();
view.scrollLeft=inlineCamera.left;view.scrollTop=inlineCamera.top;
}else fit();
inlineCamera=null;enter.focus({preventScroll:true});
}
},{signal});
document.addEventListener('keydown',e=>{if(e.key==='Escape')hide()},{signal});
document.addEventListener('scroll',place,{capture:true,signal});
window.addEventListener('resize',place,{signal});
const resizeObserver=new ResizeObserver(()=>{
if(disposed)return;
cancelAnimationFrame(resizeFrame);
resizeFrame=requestAnimationFrame(()=>{
if(fitted)fit();else{hide();resizeScene()}
});
});
const dispose=()=>{
if(disposed)return;disposed=true;hide();controller.abort();
cancelAnimationFrame(resizeFrame);resizeObserver.disconnect();removalObserver.disconnect();
host.dataset.disposed='true';
};
// Notebook outputs can disappear without any later scroll or keyboard event.
const removalObserver=new MutationObserver(records=>{
if(records.some(record=>record.removedNodes.length)&&!host.isConnected)dispose();
});
removalObserver.observe(host.getRootNode(),{childList:true,subtree:true});
view.style.height=inlineHeight()+'px';host.dataset.layout=selected;
fit();resizeObserver.observe(view);
"""


def render_html(svg: str, mode: str, layouts: dict, initial_layout: str) -> str:
    """Wrap one SVG with controls and serialized, precomputed geometries."""
    payload = json.dumps({'layouts': layouts, 'initial': initial_layout},
                         ensure_ascii=True, separators=(',', ':'))
    payload = payload.replace('<', r'\u003c').replace('>', r'\u003e')
    payload = payload.replace('&', r'\u0026')
    options = ''.join(
        f'<option value="{name}">{label}</option>'
        for name, label in [('original', 'Original'), ('compact', 'Compact')]
        if name in layouts)
    if not options:
        options = '<option value="original">Original</option>'
    fit_icon = ('M 3 8 V 3 H 8 M 16 3 H 21 V 8 M 21 16 V 21 H 16 '
                'M 8 21 H 3 V 16 M 7 7 H 17 V 17 H 7 Z')
    enter_icon = ('M 9 3 H 3 V 9 M 15 3 H 21 V 9 M 3 15 V 21 H 9 '
                  'M 21 15 V 21 H 15')
    exit_icon = ('M 3 9 H 9 V 3 M 15 3 V 9 H 21 M 9 21 V 15 H 3 '
                 'M 21 15 H 15 V 21')

    def button(cls: str, label: str, icon: str = '', text: str = '') -> str:
        graphic = (f'<svg width="18" height="18" viewBox="0 0 24 24" '
                   f'aria-hidden="true"><path d="{icon}" fill="none" '
                   'stroke="currentColor" stroke-width="1.5" '
                   'stroke-linecap="round" stroke-linejoin="round"/></svg>'
                   if icon else text)
        return (f'<button type="button" class="kumo-graph-{cls}" '
                f'aria-label="{label}" title="{label}" disabled="">{graphic}'
                '</button>')

    return (
        f'<div class="kumo-graph" data-highlight="{escape(mode)}">'
        f'<style>{_STYLE}</style><div class="kumo-graph-toolbar">'
        '<label class="kumo-graph-layout-label">Layout '
        f'<select class="kumo-graph-layout" aria-label="Graph layout" disabled="">'
        f'{options}</select></label>'
        '<div class="kumo-graph-zoom" role="group" aria-label="Graph zoom">'
        + button('zoom-out', 'Zoom out', text='−')
        + '<span class="kumo-graph-scale" aria-live="polite"></span>'
        + button('zoom-in', 'Zoom in', text='+')
        + button('fit', 'Fit graph', fit_icon) + '</div>'
        + button('enter', 'Enter full screen', enter_icon)
        + button('exit', 'Exit full screen', exit_icon) + '</div>'
        '<span class="kumo-graph-status" role="status" hidden="">'
        'Full screen is unavailable in this notebook. Use the graph controls '
        'here or open the exported SVG.</span>'
        '<div class="kumo-graph-view" tabindex="0" role="region" '
        'aria-label="Scrollable graph visualization">'
        f'<div class="kumo-graph-scene">{svg}</div></div>'
        f'<script type="application/json" class="kumo-graph-layout-data">'
        f'{payload}</script><script>//<![CDATA[\n(()=>{{'
        'const host=document.currentScript.parentElement;'
        'const controller=new AbortController(),signal=controller.signal;'
        'const view=host.querySelector(".kumo-graph-view");'
        'const svg=view.querySelector("svg"),scene=view.firstElementChild;'
        'const edges=Array.from(view.querySelectorAll(".kumo-edge"));'
        + _HOVER + _CAMERA + '})();\n//]]>\n</script></div>')
