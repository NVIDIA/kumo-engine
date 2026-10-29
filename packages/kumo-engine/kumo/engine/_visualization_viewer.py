# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained notebook viewer; layout geometry is computed in Python."""

from string import Template

from kumo.engine._widget import THEME, button, render_widget

_STYLE = Template(r"""
.kumo-graph{container:kumo-graph/inline-size}
.kumo-graph,.kumo-graph *{user-select:none;-webkit-user-select:none}
.kumo-graph:fullscreen{width:100%;height:100%;max-height:none}
.kumo-graph-toolbar{display:grid;
grid-template-columns:minmax(0,1fr) auto minmax(0,1fr)}
.kumo-graph-layout{display:flex;grid-column:2;justify-self:center;
border-radius:18px;background:$background;padding:2px;color:$chrome_foreground;
font:13px $font}
.kumo-graph-layout label{position:relative;cursor:pointer}
.kumo-graph-layout input{position:absolute;opacity:0;width:1px;height:1px}
.kumo-graph-layout span{display:flex;align-items:center;height:28px;
padding:0 12px;border-radius:16px}
.kumo-graph-layout input:checked+span{background:$chrome_hover;color:$foreground}
.kumo-graph-actions{display:flex;align-items:center;gap:6px;
grid-column:3;justify-self:end}
.kumo-graph-zoom{display:flex;align-items:center;gap:0;flex:none;
background:$background;border-radius:18px}
.kumo-graph-layout,.kumo-graph-zoom,.kumo-graph-actions>button{
box-shadow:$chrome_shadow}
@container kumo-graph (max-width:560px){
.kumo-graph-toolbar{grid-template-columns:minmax(0,1fr)}
.kumo-graph-layout,.kumo-graph-actions{grid-column:1}}
.kumo-graph-layout input:focus-visible+span{
outline:2px solid $focus;outline-offset:2px}
.kumo-graph-scale{min-width:44px;text-align:center;color:#666;
font:12px $font;font-variant-numeric:tabular-nums}
.kumo-graph .kumo-graph-exit{display:none}
.kumo-graph:fullscreen .kumo-graph-enter{display:none}
.kumo-graph:fullscreen .kumo-graph-exit{display:inline-flex}
.kumo-graph-view{position:relative;overflow:clip;min-height:0;flex:none;
max-width:100%;cursor:grab;
background:radial-gradient(circle at 1px 1px,$grid .7px,
transparent .8px) 0 0/16px 16px}
.kumo-graph:fullscreen .kumo-graph-view{flex:1;height:auto!important}
.kumo-graph-scene{position:absolute;left:0;top:0;transform-origin:0 0}
.kumo-graph .kumo-graph-view [data-kumo-tip]{cursor:inherit}
.kumo-graph-view[data-dragging],.kumo-graph-view[data-dragging] *{cursor:grabbing}
.kumo-graph-view svg{display:block;flex:none;max-width:none!important;
max-height:none!important}
.kumo-graph-status{font-size:13px;color:#666;padding:6px 0}
.kumo-graph-view svg>rect{fill:transparent}
.kumo-graph-tooltip{position:absolute;top:0;left:0;z-index:10;
box-sizing:border-box;
width:320px;max-width:calc(100% - 16px);border:1px solid #cbd8ba;
border-radius:18px;corner-shape:squircle;background:$background;color:$foreground;
box-shadow:0 5px 18px #24331416,0 1px 3px #2433140a;
font:16px/1.4 $font;overflow-wrap:anywhere;
overflow:hidden;pointer-events:none;opacity:0;visibility:hidden;
transform:translateY(4px);transition:opacity 140ms ease,transform 140ms ease,
visibility 140ms ease}
.kumo-graph-tooltip[data-open]{opacity:1;visibility:visible;
transform:translateY(0)}
.kumo-graph .kumo-tip-head{background:$header_background;border-bottom:1px solid $tooltip_header_border;
padding:7px 10px}
.kumo-graph .kumo-tip-context{color:#6a7a58;font-size:16px;margin-bottom:2px}
.kumo-graph .kumo-tip-name{font-size:17px;font-weight:700;color:$accent}
.kumo-graph .kumo-tip-body{padding:7px 10px}
.kumo-graph .kumo-tip-badges{display:flex;flex-wrap:wrap;gap:5px}
.kumo-graph .kumo-tip-samples{display:flex;gap:5px}
.kumo-graph .kumo-tip-badge{background:#f2f4ef;border:1px solid #e1e6db;
border-radius:7px;corner-shape:squircle;padding:2px 6px;color:#536247}
.kumo-graph .kumo-tip-key{background:#ecf4df;border-color:#d1e2b7;color:#487014}
.kumo-graph .kumo-tip-type{color:#536247}
.kumo-graph .kumo-tip-ref{padding-top:5px;color:#5b6751}
.kumo-graph .kumo-tip-connections{list-style:none;padding:0;margin:0;display:grid;gap:4px 10px;
grid-auto-flow:column;grid-template-columns:minmax(0,1fr)}
.kumo-graph .kumo-tip-connections[data-columns="2"]{grid-template-columns:repeat(2,minmax(0,1fr))}
.kumo-graph .kumo-tip-connection{color:#455239;font-weight:500;min-width:0;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kumo-graph .kumo-tip-label{color:#7a8470;margin:5px 0 3px}
.kumo-graph .kumo-tip-sample{flex:1;min-width:0;max-width:100%;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis;background:#f5f7f2;border-radius:7px;corner-shape:squircle;
padding:2px 6px;color:#455239;
font:16px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace}
.kumo-graph [data-kumo-tip]{outline:none;cursor:default}
.kumo-graph .kumo-row-light{fill:#edf6df;stroke:#9bc958;stroke-width:1;
opacity:0;pointer-events:none;transition:opacity 140ms ease}
.kumo-graph .kumo-active .kumo-row-light{opacity:1}
.kumo-graph [data-kumo-tip]:focus-visible .kumo-row-light{
stroke:$focus;stroke-width:2}
.kumo-graph [data-kumo-row]:not([data-kumo-row=""]) .kumo-row-light{
stroke:none;filter:drop-shadow(0 1px 2px #00000026)}
.kumo-graph [data-kumo-row]:not([data-kumo-row=""]):focus-visible .kumo-row-light{
filter:drop-shadow(0 0 2px $focus)}
.kumo-graph .kumo-edge{transition:stroke 140ms ease,stroke-width 140ms ease,
opacity 140ms ease}
.kumo-graph.kumo-links-focused .kumo-edge:not(.kumo-edge-active){opacity:.35}
.kumo-graph .kumo-edge-active{stroke:#6e9947;stroke-width:1.8}
@media(prefers-reduced-motion:reduce){.kumo-graph-tooltip,
.kumo-graph .kumo-row-light,
.kumo-graph .kumo-edge{transition:none}}

.kumo-graph .kumo-edge{vector-effect:non-scaling-stroke}
""").substitute(THEME)

_HOVER = r"""
const tip=document.createElement('div');
tip.className='kumo-graph-tooltip';tip.role='tooltip';
tip.id='kumo-tip-'+Math.random().toString(36).slice(2);host.append(tip);
let active=null, card=null, timer, placement=null, hoverLock=null, pointer=null;
const firstCard=svg.querySelector(':scope>[data-kumo-card]');
const element=(cls,text)=>{const n=document.createElement('div');
n.className='kumo-tip-'+cls;n.textContent=text;return n};
const cancelHide=()=>{clearTimeout(timer);timer=null};
const hide=()=>{cancelHide();tip.removeAttribute('data-open');
placement=hoverLock=null;
if(host.classList.contains('kumo-links-focused'))
edges.forEach(edge=>svg.insertBefore(edge.parentElement,firstCard));
host.classList.remove('kumo-links-focused');
edges.forEach(edge=>edge.classList.remove('kumo-edge-active'));
if(active){active.removeAttribute('aria-describedby');
active.classList.remove('kumo-active')}
active=card=null};
const content=el=>{
const row=!!el.dataset.kumoRow, table=el.dataset.kumoTable;
const name=el.dataset.kumoColumn, full=table+(row?'.'+name:'');
const lines=el.dataset.kumoTip.slice(full.length+1).split('\n');
const head=element('head',''), body=element('body','');
head.append(element('context',row?table:'Table'),element('name',row?name:table));
const badges=element('badges','');
if(row){
const role=lines.find(x=>x.startsWith('Role: '))?.slice(6);
if(role&&role!=='Column')badges.append(element('badge kumo-tip-key',role));
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
item.className='kumo-tip-connection';item.textContent=name;item.title=name;list.append(item)}
body.append(list)}}
tip.replaceChildren(head,body)};
const layoutConnections=sx=>{
const list=tip.querySelector('.kumo-tip-connections');if(!list)return;
const items=Array.from(list.children),short=[],long=[],range=document.createRange();
const half=(list.getBoundingClientRect().width-parseFloat(getComputedStyle(list).columnGap)*sx)/2;
for(const item of items){range.selectNodeContents(item);
(range.getBoundingClientRect().width<=half?short:long).push(item)}
const fallback=long.length>items.length/2||Math.ceil(short.length/2)+long.length>15;
const paired=items.length===1?[]:fallback?items:short;
const full=items.length===1?items:fallback?[]:long;
const rows=Math.min(15,Math.ceil(paired.length/2));
list.dataset.columns=paired.length?'2':'1';
list.style.gridTemplateRows=`repeat(${rows+full.length},auto)`;
paired.forEach((item,i)=>{
item.style.gridColumn=String(Math.floor(i/rows)+1);
item.style.gridRow=String(i%rows+1)});
full.forEach((item,i)=>{item.style.gridColumn='1 / -1';
item.style.gridRow=String(rows+i+1)})};
const viewport=()=>{
const h=host.getBoundingClientRect(), v=view.getBoundingClientRect();
const sx=h.width/host.offsetWidth||1, sy=h.height/host.offsetHeight||1;
let left=Math.max(h.left,v.left,0), right=Math.min(h.right,v.right,innerWidth);
let top=Math.max(h.top,v.top,0), bottom=Math.min(h.bottom,v.bottom,innerHeight);
for(let el=host.matches(':fullscreen')?null:host.parentElement;
el;el=el.parentElement){
if(el===document.body||el===document.documentElement)break;
const style=getComputedStyle(el), r=el.getBoundingClientRect();
if(/auto|scroll|hidden|clip/.test(style.overflowX)){
left=Math.max(left,r.left);right=Math.min(right,r.right)}
if(/auto|scroll|hidden|clip/.test(style.overflowY)){
top=Math.max(top,r.top);bottom=Math.min(bottom,r.bottom)}}
return{h,sx,sy,left:left+8*sx,right:right-8*sx,
top:top+8*sy,bottom:bottom-8*sy}};
const intersects=(r,b)=>r.right>b.left&&r.left<b.right&&r.bottom>b.top&&r.top<b.bottom;
const edgeCurves=new WeakMap();
const edgeExtent=(edge,b,spans)=>{
const r=edge.getBoundingClientRect();if(!intersects(r,b))return null;
if(!spans&&r.left>=b.left&&r.right<=b.right&&r.top>=b.top&&r.bottom<=b.bottom)
return[r.left,r.right];
const d=edge.getAttribute('d');let cached=edgeCurves.get(edge);
if(cached?.d!==d){
// Generated edges contain absolute M/L/Q/C segments; retain their control points.
const tokens=d.match(/[MLQC]|[-+]?(?:\d*\.)?\d+(?:e[-+]?\d+)?/gi);
const curves=[];let point;
for(let i=0;i<tokens.length;){const command=tokens[i++],points=[];
for(let n=0;n<({M:1,L:1,Q:2,C:3})[command];n++)
points.push([Number(tokens[i++]),Number(tokens[i++])]);
if(command!=='M')curves.push([point,...points]);point=points.at(-1)}
cached={d,curves};edgeCurves.set(edge,cached)}
const matrix=edge.getScreenCTM(),epsilon=.25;
let left=Infinity,right=-Infinity;
const visit=points=>{
const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
if(Math.max(...xs)<b.left||Math.min(...xs)>b.right||
Math.max(...ys)<b.top||Math.min(...ys)>b.bottom)return;
const before=points[0],after=points.at(-1),dx=after.x-before.x,dy=after.y-before.y;
// The Bezier convex hull bounds deviation from the parameterized chord.
if(points.some((p,i)=>Math.abs(p.x-before.x-dx*i/(points.length-1))>epsilon||
Math.abs(p.y-before.y-dy*i/(points.length-1))>epsilon)){
const a=[before],z=[after];let level=points;
while(level.length>1){level=level.slice(1).map((p,i)=>({
x:(p.x+level[i].x)/2,y:(p.y+level[i].y)/2}));
a.push(level[0]);z.unshift(level.at(-1))}
visit(a);visit(z);return}
let lo=0,hi=1;
for(const [p,q] of [[-dx,before.x-b.left+epsilon],[dx,b.right-before.x+epsilon],
[-dy,before.y-b.top+epsilon],[dy,b.bottom-before.y+epsilon]]){
if(!p){if(q<0){hi=-1;break}}else if(p<0)lo=Math.max(lo,q/p);else hi=Math.min(hi,q/p)}
if(lo<=hi){const a=before.x+lo*dx,z=before.x+hi*dx;
left=Math.min(left,a,z);right=Math.max(right,a,z);
if(spans)spans.push([Math.min(a,z)-2,Math.max(a,z)+2])}};
for(const curve of cached.curves)visit(curve.map(([x,y])=>({
x:matrix.a*x+matrix.c*y+matrix.e,y:matrix.b*x+matrix.d*y+matrix.f})));
return left===Infinity?null:[left-2,right+2]};
const classify=()=>{
const b=viewport(),c=card.getBoundingClientRect(),table=active.dataset.kumoTable;
const visible=Array.from(svg.querySelectorAll(':scope>[data-kumo-card]'),
x=>x.getBoundingClientRect()).filter(r=>intersects(r,b));
const leftmost=c.left<=Math.min(...visible.map(r=>r.left))+1;
const rightmost=c.right>=Math.max(...visible.map(r=>r.right))-1;
const role=active.dataset.kumoTip.split('\n').find(x=>x.startsWith('Role: '))||'';
const pk=/\bPK\b/.test(role),fk=/\bFK\b/.test(role);
const incident=edges.filter(e=>e.dataset.srcTable===table||e.dataset.dstTable===table);
const selected=incident.filter(e=>e.classList.contains('kumo-edge-active'));
const relevant=active.dataset.kumoRow?selected:incident;
const extents=new Map(),extent=e=>{if(!extents.has(e))extents.set(e,edgeExtent(e,b));return extents.get(e)};
let side;
if(leftmost!==rightmost)side=leftmost?'left':'right';
else if(pk!==fk)side=pk?'left':'right';
else{
let left=0,right=0;for(const e of relevant){const r=extent(e);if(!r)continue;
if(r[0]<c.left-1)left++;if(r[1]>c.right+1)right++}
side=left===right?(c.left-b.left>=b.right-c.right?'left':'right'):
(left<right?'left':'right')}
if(active.dataset.kumoRow)return{side,offset:0};
let boundary=side==='left'?c.left:c.right;
for(const e of leftmost!==rightmost?edges:relevant){const r=extent(e);if(!r)continue;
boundary=side==='left'?Math.min(boundary,r[0]):Math.max(boundary,r[1])}
return{side,offset:(boundary-(side==='left'?c.left:c.right))/b.sx}};
const place=(allowPan=false)=>{
if(!active||!placement)return;
const b=viewport(),a=active.getBoundingClientRect(),c=card.getBoundingClientRect();
if(b.right<=b.left||b.bottom<=b.top){hide();return}
tip.style.maxWidth=Math.max(0,Math.min(320,(b.right-b.left)/b.sx))+'px';
tip.style.maxHeight=Math.max(0,(b.bottom-b.top)/b.sy)+'px';
layoutConnections(b.sx);
const t=tip.getBoundingClientRect(),left=placement.side==='left';
let x=(left?c.left-t.width-12*b.sx:c.right+12*b.sx)+placement.offset*b.sx;
const y=Math.max(b.top,Math.min(a.top,b.bottom-t.height));
if(allowPan===true){
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(v,hi));
const dy=clamp(y-a.top,b.top+Math.min(8*b.sy,a.height/2)-a.bottom,
b.bottom-Math.min(8*b.sy,a.height/2)-a.top);
if(active.dataset.kumoRow){
// Compare the final popup with only route portions visible before the one pan.
const top=Math.max(b.top,Math.min(a.top+dy,b.bottom-t.height))-dy;
const band={...b,top:Math.max(b.top,top-2),bottom:Math.min(b.bottom,top+t.height+2)};
const spans=[];
if(band.top<band.bottom)for(const edge of edges)
if(edge.classList.contains('kumo-edge-active'))edgeExtent(edge,band,spans);
spans.sort((a,z)=>left?z[1]-a[1]:a[0]-z[0]);
const adjacent=x;
for(const span of spans)if(x<span[1]&&x+t.width>span[0])
x=left?span[0]-t.width-12*b.sx:span[1]+12*b.sx;
placement.offset+=(x-adjacent)/b.sx}
const dx=clamp(clamp(x<b.left?b.left-x:x+t.width>b.right?b.right-x-t.width:0,
-(b.right-b.left),b.right-b.left),b.left+Math.min(24*b.sx,a.width/2)-a.right,
b.right-Math.min(24*b.sx,a.width/2)-a.left);
if(Math.abs(dx)>.5||Math.abs(dy)>.5){hoverLock=pointer?{...pointer}:{};
panBy(dx/b.sx,dy/b.sy);place();return}}
if(!intersects(a,b)){hide();return}
const raw=x;x=Math.max(b.left,Math.min(x,b.right-t.width));
tip.dataset.side=placement.side;tip.dataset.clamped=String(x!==raw||y!==a.top);
tip.style.left=(x-b.h.left)/b.sx+'px';tip.style.top=(y-b.h.top)/b.sy+'px'};
const show=(el,event)=>{if(isDragging||hoverLock)return;
cancelHide();if(active===el)return;
if(event&&Number.isFinite(event.clientX))pointer={x:event.clientX,y:event.clientY};
hide();active=el;card=el.parentElement;
el.classList.add('kumo-active');
content(el);el.setAttribute('aria-describedby',tip.id);
if(el.dataset.kumoRow){
const table=el.dataset.kumoTable, column=el.dataset.kumoColumn;
edges.forEach(edge=>{const d=edge.dataset;
const src=d.srcTable===table?d.srcColumn===column:false;
const dst=d.dstTable===table?d.dstColumn===column:false;
edge.classList.toggle('kumo-edge-active',src||dst);
if(src||dst){svg.insertBefore(edge.parentElement,firstCard);
host.classList.add('kumo-links-focused')}})}
placement=classify();tip.setAttribute('data-open','');place(true)};
const scheduleHide=()=>{if(timer)return;timer=setTimeout(()=>{timer=null;
if(!hoverLock&&active!==document.activeElement)hide()},150)};
view.querySelectorAll('title').forEach(el=>el.remove());
view.querySelectorAll('[data-kumo-tip]').forEach(el=>{
const light=el.querySelector('.kumo-surface').cloneNode();
light.classList.add('kumo-row-light');light.removeAttribute('fill');
light.removeAttribute('stroke');el.insertBefore(light,el.querySelector('text'));
el.addEventListener('pointerenter',e=>{if(e.pointerType!=='touch')show(el,e)},{signal});
el.addEventListener('pointerleave',scheduleHide,{signal});
el.addEventListener('focus',()=>{hoverLock=null;show(el)},{signal});
el.addEventListener('blur',scheduleHide,{signal})});
host.addEventListener('pointerleave',()=>{hoverLock=null;scheduleHide()},{signal});
document.addEventListener('pointerout',e=>{
if(!e.relatedTarget){hoverLock=null;scheduleHide()}
},{signal});
document.addEventListener('pointermove',e=>{
pointer={x:e.clientX,y:e.clientY};if(isDragging||!active)return;
if(hoverLock&&Math.hypot(pointer.x-hoverLock.x,pointer.y-hoverLock.y)<=1)return;
hoverLock=null;const hit=document.elementFromPoint(e.clientX,e.clientY);
const next=hit?.closest('[data-kumo-tip]');
if(next&&view.contains(next)){show(next,e);return}
scheduleHide();
},{capture:true,signal});

"""

_CAMERA = r"""
const selector=host.querySelector('.kumo-graph-layout');
const label=host.querySelector('.kumo-graph-scale');
const status=host.querySelector('[role=status]');
const enter=host.querySelector('.kumo-graph-enter');
const exit=host.querySelector('.kumo-graph-exit');
const payload=data;
const layouts=payload.layouts;
let selected=payload.initial, scale=1, fitted=true, full=false, inlineCamera=null;
let layoutWidth=Number(svg.getAttribute('width'))||304;
let layoutHeight=Number(svg.getAttribute('height'))||112;
let resizeFrame=0, panX=0, panY=0, isDragging=false, drag=null;
let viewWidth=0, viewHeight=0;
const cards=new Map(Array.from(svg.querySelectorAll(':scope>[data-kumo-card]'),
el=>[el.dataset.kumoCard,el]));
const controls=host.querySelectorAll('button,input');
selector.querySelectorAll('input').forEach(el=>{
el.name=tip.id+'-layout';el.checked=el.value===selected});
if(Object.keys(layouts).length<2)selector.hidden=true;

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
for(const [key,value] of Object.entries({cx:x,cy:y,r:4,fill:'__MUTED_COLOR__'}))
circle.setAttribute(key,value);handles.append(circle)}
svg.append(handles);layoutWidth=layout.width;layoutHeight=layout.height;
svg.setAttribute('viewBox',`0 0 ${layoutWidth} ${layoutHeight}`);
svg.setAttribute('width',layoutWidth);svg.setAttribute('height',layoutHeight);
svg.querySelectorAll(':scope>rect').forEach(rect=>{
rect.setAttribute('width',layoutWidth);rect.setAttribute('height',layoutHeight)});
selected=name;host.dataset.layout=name;
};
const inlineHeight=()=>Math.min(720,Math.max(240,layoutHeight+32));
const fitScale=()=>{
const side=348;
const padding=view.clientWidth>=side*2+320?side:16;
return Math.min(1,Math.max(1,view.clientWidth-padding*2)/layoutWidth,
Math.max(1,view.clientHeight-32)/layoutHeight)};
const resizeScene=()=>{
viewWidth=view.clientWidth;viewHeight=view.clientHeight;
svg.style.width=layoutWidth*scale+'px';svg.style.height=layoutHeight*scale+'px';
scene.style.transform=`translate(${panX}px,${panY}px)`;
view.style.backgroundSize=`${16*scale}px ${16*scale}px`;
view.style.backgroundPosition=`${panX}px ${panY}px`;
view.style.backgroundImage=`radial-gradient(circle at ${scale}px ${scale}px,
__GRID_COLOR__ ${.7*scale}px,transparent ${.8*scale}px)`;
label.textContent=Math.round(scale*100)+'%';host.dataset.scale=String(scale);
};
const panBy=(dx,dy)=>{fitted=false;panX+=dx;panY+=dy;resizeScene()};
const fit=()=>{
hide();fitted=true;scale=fitScale();
panX=(view.clientWidth-layoutWidth*scale)/2;
panY=(view.clientHeight-layoutHeight*scale)/2;resizeScene();
};
const zoom=(next,x=view.clientWidth/2,y=view.clientHeight/2)=>{
hide();fitted=false;
const graphX=(x-panX)/scale, graphY=(y-panY)/scale;
scale=Math.max(Math.min(.1,fitScale()),Math.min(4,next));
panX=x-graphX*scale;panY=y-graphY*scale;resizeScene();
};
view.addEventListener('pointerdown',e=>{
if(e.button!==0||e.pointerType==='touch')return;
const rect=view.getBoundingClientRect();
drag={id:e.pointerId,x:e.clientX,y:e.clientY,panX,panY,
sx:rect.width/view.clientWidth,sy:rect.height/view.clientHeight};
view.setPointerCapture(e.pointerId);
},{capture:true,signal});
view.addEventListener('pointermove',e=>{
if(!drag||e.pointerId!==drag.id)return;
const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
if(!isDragging&&Math.hypot(dx,dy)<4)return;
if(!isDragging){isDragging=true;hide();view.dataset.dragging='true'}
e.preventDefault();panBy(drag.panX+dx/drag.sx-panX,drag.panY+dy/drag.sy-panY);
},{capture:true,signal});
const stopDrag=e=>{
if(e&&drag&&e.pointerId!==drag.id)return;
const id=drag?.id;drag=null;isDragging=false;delete view.dataset.dragging;
if(id!==undefined&&view.hasPointerCapture(id))view.releasePointerCapture(id);
};
for(const event of ['pointerup','pointercancel','lostpointercapture'])
view.addEventListener(event,stopDrag,{capture:true,signal});
view.addEventListener('keydown',e=>{
if(e.target!==view)return;
const delta={ArrowLeft:[40,0],ArrowRight:[-40,0],
ArrowUp:[0,40],ArrowDown:[0,-40]}[e.key];
if(delta){e.preventDefault();hide();panBy(...delta)}
},{signal});
selector.addEventListener('change',event=>{
applyLayout(event.target.value);if(!full)view.style.height=inlineHeight()+'px';fit();
},{signal});
host.querySelector('.kumo-graph-fit').addEventListener('click',fit,{signal});
host.querySelector('.kumo-graph-zoom-in').addEventListener('click',()=>zoom(scale*1.25),{signal});
host.querySelector('.kumo-graph-zoom-out').addEventListener('click',()=>zoom(scale/1.25),{signal});
view.addEventListener('wheel',e=>{
if(!e.ctrlKey&&!e.metaKey)return;
e.preventDefault();const box=view.getBoundingClientRect();
const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?view.clientHeight:1);
zoom(scale*Math.exp(-delta*.004),
(e.clientX-box.left)*view.clientWidth/box.width,
(e.clientY-box.top)*view.clientHeight/box.height);
},{passive:false,signal});
enter.addEventListener('click',()=>{
status.hidden=true;
// Capture the inline camera before fullscreen changes viewport dimensions.
inlineCamera={selected,scale,fitted,panX,panY};
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
if(!inlineCamera)inlineCamera={selected,scale,fitted,panX,panY};
fit();exit.focus({preventScroll:true});
}else{
view.style.height=inlineHeight()+'px';
if(inlineCamera&&inlineCamera.selected===selected){
scale=inlineCamera.scale;fitted=inlineCamera.fitted;
panX=inlineCamera.panX;panY=inlineCamera.panY;resizeScene();
}else fit();
inlineCamera=null;enter.focus({preventScroll:true});
}
},{signal});
document.addEventListener('keydown',e=>{if(e.key==='Escape')hide()},{signal});
document.addEventListener('scroll',place,{capture:true,signal});
window.addEventListener('resize',place,{signal});
const resizeObserver=new ResizeObserver(()=>{
if(signal.aborted)return;
cancelAnimationFrame(resizeFrame);
resizeFrame=requestAnimationFrame(()=>{
if(view.clientWidth===viewWidth&&view.clientHeight===viewHeight)return;
if(fitted)fit();else{hide();resizeScene()}
});
});
onDispose(()=>{
stopDrag();hide();cancelAnimationFrame(resizeFrame);resizeObserver.disconnect();
});
view.style.height=inlineHeight()+'px';host.dataset.layout=selected;
fit();resizeObserver.observe(view);controls.forEach(el=>el.disabled=false);
""".replace('__GRID_COLOR__', THEME['grid']).replace(
    '__MUTED_COLOR__', THEME['muted'])


def render_html(svg: str, layouts: dict, initial_layout: str) -> str:
    """Wrap one SVG with controls and serialized, precomputed geometries."""
    options = ''.join(
        '<label><input type="radio" '
        f'value="{name}" disabled=""/><span>{label}</span></label>'
        for name, label in [('standard', 'Standard'), ('compact', 'Compact')]
        if name in layouts)
    if not options:
        options = ('<label><input type="radio" value="standard" disabled=""/>'
                   '<span>Standard</span></label>')
    fit_icon = ('M 3 8 V 3 H 8 M 16 3 H 21 V 8 M 21 16 V 21 H 16 '
                'M 8 21 H 3 V 16 M 7 7 H 17 V 17 H 7 Z')
    enter_icon = ('M 9 3 H 3 V 9 M 15 3 H 21 V 9 M 3 15 V 21 H 9 '
                  'M 21 15 V 21 H 15')
    exit_icon = ('M 3 9 H 9 V 3 M 15 3 V 9 H 21 M 9 21 V 15 H 3 '
                 'M 21 15 H 15 V 21')

    body = (
        '<div class="kumo-widget-toolbar kumo-graph-toolbar">'
        '<div class="kumo-graph-layout" role="radiogroup" '
        f'aria-label="Graph layout">{options}</div>'
        '<div class="kumo-graph-actions">'
        '<div class="kumo-graph-zoom" role="group" aria-label="Graph zoom">'
        '<span class="kumo-graph-scale" aria-live="polite"></span>'
        + button('kumo-graph-zoom-out', 'Zoom out', text='−')
        + button('kumo-graph-zoom-in', 'Zoom in', text='+')
        + '</div>' + button('kumo-graph-fit', 'Fit graph', fit_icon)
        + button('kumo-graph-enter', 'Enter full screen', enter_icon)
        + button('kumo-graph-exit', 'Exit full screen', exit_icon) + '</div></div>'
        '<span class="kumo-graph-status" role="status" hidden="">'
        'Full screen is unavailable in this notebook. Use the graph controls '
        'here or open the exported SVG.</span>'
        '<div class="kumo-graph-view" tabindex="0" role="region" '
        'aria-label="Pannable graph visualization">'
        f'<div class="kumo-graph-scene">{svg}</div></div>')
    script = (
        'const view=host.querySelector(".kumo-graph-view");'
        'const svg=view.querySelector("svg"),scene=view.firstElementChild;'
        'const edges=Array.from(view.querySelectorAll(".kumo-edge"));'
        + _HOVER + _CAMERA)
    return render_widget(body, style=_STYLE, script=script,
                         data={'layouts': layouts, 'initial': initial_layout},
                         class_name='kumo-graph')
