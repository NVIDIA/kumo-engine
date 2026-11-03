# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training diagnostics composed with the shared notebook shell."""

from html import escape
from math import isfinite
from string import Template

from kumo.engine._widget import CSS_THEME


def static_summary(payload: dict) -> str:
    """Return escaped context, metrics and availability without example data."""
    context = payload.get('context', {})
    provenance = payload.get('provenance', {})
    metrics = payload.get('metrics', {}).get('rows', [])
    parts = [
        '<div class="kumo-training-static"><h2>Training results</h2>',
        '<p>Recorded TEST diagnostics. Selected trial: ' +
        escape(str(provenance.get('selected_trial') or 'Unavailable')) +
        '. Explanation fit: ' +
        escape(str(provenance.get('explanation_trial') or 'Unavailable')) +
        '.</p>'
    ]
    identities = [('Run', context.get('run')),
                  ('Task', context.get('task_type')
                   or provenance.get('task_type')),
                  ('Predictive query', context.get('query'))]
    if context.get('target'):
        identities.append(('Target', context['target']))
    for label, value in identities:
        value = str(value or 'Unavailable')
        content = ('<p class="kt-note" style="white-space:pre-wrap;'
                   'overflow-wrap:anywhere">' + escape(label + ': ' + value) +
                   '</p>')
        if len(value) > 120 or '\n' in value or '\r' in value:
            content = ('<details><summary>Explore ' + label.lower() +
                       '</summary><div class="kt-sample-values" tabindex="0" '
                       'role="region" aria-label="' + label + '" '
                       'style="max-height:400px;overflow:auto">' + content +
                       '</div></details>')
        parts.append(content)
    if len(metrics) > 100:
        parts.append(f'<p>Showing 100 of {len(metrics)} loaded metrics. '
                     'Rerun this cell in a trusted notebook to inspect all '
                     'loaded metrics.</p>')
    parts.append('<ul>')
    if provenance.get('generation_status') in ('disabled', 'failed'):
        parts.append('<li>Explanation generation: ' +
                     escape(str(provenance['generation_status'])) + '</li>')
    if (provenance.get('generation_failure')
            and provenance.get('generation_status') != 'failed'):
        parts.append('<li>' + escape(str(provenance['generation_failure'])) +
                     '</li>')
    if provenance.get('selection_verified') is False:
        parts.append('<li>Selection unverified: ' + escape(str(
            provenance.get('selection_reason') or
            'Selection metadata is missing or invalid')) + '</li>')
    for row in metrics[:100]:
        value = row.get('value')
        number = (str(value) if isinstance(value, (int, float))
                  and not isinstance(value, bool) and isfinite(value)
                  else 'Unavailable')
        label = ' / '.join(str(row.get(key) or 'Unavailable')
                           for key in ('fit', 'split', 'name'))
        label += ' / ' + str(row.get('origin') or 'Unknown metric origin')
        parts.append('<li>' + escape(label) + ': ' + escape(number) + '</li>')
    parts.append('</ul><h2>Recorded patterns</h2><ul>')
    for key in ('gradient', 'cohorts', 'samples', 'history', 'trials'):
        section = payload.get(key, {})
        label = key + ': ' + str(section.get('status', 'unavailable'))
        if section.get('reason'):
            label += '. ' + str(section['reason'])
        if section.get('producer_note'):
            prefix = ('Producer scope'
                      if key == 'gradient' else 'Producer eligibility')
            label += f'. {prefix}: ' + str(section['producer_note'])
        parts.append('<li>' + escape(label) + '</li>')
    parts.append('</ul><h2>Selected examples</h2><p>Example values are excluded '
                 'from this summary. Rerun this cell in a trusted notebook '
                 'to activate diagnostics.</p></div>')
    return ''.join(parts)


_STYLE = Template(r"""
.kumo-training{container:kumo-training/inline-size;padding:16px;line-height:1.45;
overflow-wrap:anywhere;text-align:left}
.kumo-training *{box-sizing:border-box;min-width:0}
.kumo-training .kt-content{max-width:1040px;width:100%;margin:0 auto}
.kumo-training{border:1px solid $surface_border;border-radius:12px;padding:24px}
.kumo-training .kt-header{padding-bottom:16px}
.kumo-training .kt-header h2{font-size:24px;color:$foreground;margin-bottom:8px}
.kumo-training .kt-header>p{font-size:15px;margin-bottom:8px}
.kumo-training .kt-navigation{display:flex;align-items:center;flex-wrap:wrap;gap:8px;
padding:12px 0;margin-bottom:24px;border-block:1px solid $surface_border}
.kumo-training .kt-navigation>.kt-controls{margin:0 0 0 auto}
.kumo-training .kt-navigation>.kt-action{font-weight:600;padding:8px 14px}
.kumo-training .kt-navigation>.kt-action[aria-pressed="true"]{color:$accent;
box-shadow:inset 0 -2px $accent}
.kumo-training .kt-content>section{margin:0;padding:0;border:0}
.kumo-training .kt-header details .kt-note{margin:8px 0}
.kumo-training .kt-provenance{display:flex;flex-wrap:wrap;gap:8px 24px}
.kumo-training .kt-provenance>.kt-note{flex:1 1 200px;margin:4px 0}
@container kumo-training (max-width:520px){
.kumo-training .kt-navigation>.kt-controls{margin:0;width:100%}}

.kumo-training h2{font:700 20px/1.3 $font;color:$heading;margin:0 0 12px}
.kumo-training h3{font:600 16px/1.4 $font;color:$foreground;margin:20px 0 8px}
.kumo-training h4{font:600 14px/1.4 $font;margin:12px 0 6px}
.kumo-training p{margin:6px 0 12px}
.kumo-training section+section{margin-top:30px;padding-top:24px;
border-top:1px solid $surface_border}
.kumo-training .kt-note{color:$muted;font-size:13px;white-space:pre-wrap}
.kumo-training .kt-table{border-collapse:collapse;width:100%;margin:8px 0 16px;
font:14px/1.45 $font;color:$foreground;table-layout:fixed}
.kumo-training .kt-table th,.kumo-training .kt-table td{
padding:8px;text-align:left;vertical-align:top;border-bottom:1px solid $surface_border;
white-space:pre-wrap;overflow-wrap:anywhere}
.kumo-training .kt-table th{font-weight:600;color:$muted;font-size:12px}
.kumo-training .kt-table th:first-child,.kumo-training .kt-table td:first-child{padding-left:0}
.kumo-training .kt-table .kt-number{text-align:right;font-variant-numeric:tabular-nums}
.kumo-training .kt-trials .kt-table{table-layout:auto}
.kumo-training .kt-trials td.kt-number{white-space:nowrap;overflow-wrap:normal}
.kumo-training .kt-trials td.kt-number:before{white-space:normal;overflow-wrap:anywhere}
.kumo-training .kt-table tr[aria-selected="true"]{background:$row_highlight}
.kumo-training .kt-action,.kumo-training select{font:13px/1.4 $font;
background:$background;color:$foreground;border:1px solid $surface_border;
border-radius:7px;padding:6px 10px;max-width:100%}
.kumo-training .kt-action{cursor:pointer;white-space:normal;text-align:left}
.kumo-training .kt-action:hover{background:$chrome_hover}
.kumo-training summary{cursor:pointer;color:$muted;font-size:13px;padding:6px 0}
.kumo-training :is(button,select,summary,h2,h3,h4,.kt-history-values,.kt-sample-values):focus-visible{outline:2px solid $focus;outline-offset:2px}
.kumo-training .kt-action[aria-pressed="true"]{background:$row_highlight;
border-color:$focus}
.kumo-training .kt-detail{padding:12px;border:1px solid $surface_border;
border-radius:10px;margin:12px 0 20px}
.kumo-training .kt-detail h3{margin-top:0}
.kumo-training .kt-controls{display:flex;align-items:center;flex-wrap:wrap;
gap:8px;margin:8px 0}
.kumo-training .kt-controls label{font-size:13px;color:$muted}
.kumo-training .kt-score{display:grid;grid-template-columns:minmax(0,1fr) 84px;
align-items:center;gap:8px}
.kumo-training .kt-track{height:7px;background:$surface;border-radius:4px;overflow:hidden}
.kumo-training .kt-bar{height:100%;background:$accent;border-radius:4px}
.kumo-training .kt-value{text-align:right;font-variant-numeric:tabular-nums}
.kumo-training .kt-feature-list,.kumo-training .kt-feature-cells{list-style:none;padding:0;margin:8px 0 16px}
.kumo-training .kt-feature-list>li+li{margin-top:6px}
.kumo-training .kt-feature-choice{display:grid;grid-template-columns:minmax(0,1fr) minmax(120px,38%);
align-items:center;gap:12px;width:100%;padding:10px}
.kumo-training .kt-feature-choice .kt-note{display:block;margin:2px 0 0;font-size:12px}
.kumo-training .kt-feature-cells>li{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 180px;
align-items:start;gap:8px 16px;padding:10px 0;border-bottom:1px solid $surface_border}
.kumo-training .kt-feature-cells .kt-note{margin:2px 0;font-size:12px}
.kumo-training .kt-feature-context>h3{margin-top:0}
.kumo-training .kt-history-chart{display:block;width:100%;height:auto;max-width:600px;margin:12px 0}
.kumo-training :is(.kt-history-values,.kt-sample-values){max-height:400px;overflow:auto;padding:4px}
.kumo-training .kt-list{padding-left:20px;margin:8px 0 12px}
.kumo-training .kt-result{border-left:3px solid $accent;padding-left:16px;margin:16px 0 24px}
.kumo-training .kt-result h3{margin:0 0 4px}
.kumo-training .kt-result-value{font-size:32px;font-weight:600;font-variant-numeric:tabular-nums}
.kumo-training .kt-ranking{margin:20px 0 24px}
.kumo-training .kt-ranking h3{margin-top:0}
.kumo-training .kt-ranked-list{list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(94px,1fr));
gap:8px;padding:0;margin:8px 0 16px}
.kumo-training .kt-ranked-list>li{padding:10px;background:$surface;border:1px solid $surface_border;border-radius:6px}
.kumo-training .kt-ranked-list>li.kt-match{border-color:$accent;background:$row_highlight}
.kumo-training .kt-ranked-list strong{display:block;font-size:17px;margin:4px 0;overflow-wrap:anywhere}
.kumo-training .kt-ranked-list .kt-note{font-size:12px;margin:4px 0}
.kumo-training .kt-observed-list{list-style:none;padding:0;margin:8px 0 16px}
.kumo-training .kt-observed-list>li{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 16px;
padding:8px 0;border-bottom:1px solid $surface_border}
.kumo-training .kt-observed-list .kt-note{margin:0}
.kumo-training .kt-observed-list details{flex-basis:100%}
.kumo-training .kt-neighborhood{margin:20px 0}
.kumo-training .kt-neighborhood-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);
gap:20px;align-items:start}
.kumo-training .kt-neighborhood h4{margin-top:0}
.kumo-training .kt-selected-node{border-left:3px solid $accent;padding-left:12px}
.kumo-training .kt-selected-node>h4{color:$accent}
.kumo-training .kt-connections{list-style:none;margin:8px 0 12px;padding:0}
.kumo-training .kt-connections>li{padding:10px 0;border-bottom:1px solid $surface_border}
.kumo-training .kt-connection{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr);
gap:8px;align-items:center}
.kumo-training .kt-connection>.kt-action{width:100%}
.kumo-training .kt-connection-direction{color:$muted;font-size:12px}
.kumo-training .kt-node-features{list-style:none;padding:0;margin:8px 0}
.kumo-training .kt-node-features>li{padding:10px 0;border-bottom:1px solid $surface_border}
.kumo-training .kt-node-features>li.kt-match{background:$row_highlight;border-left:3px solid $accent;padding-left:8px}
.kumo-training .kt-node-features .kt-note{margin:4px 0}
@container kumo-training (max-width:700px){
.kumo-training .kt-neighborhood-layout{grid-template-columns:minmax(0,1fr)}}
@container kumo-training (max-width:520px){
.kumo-training .kt-table,.kumo-training .kt-table tbody,
.kumo-training .kt-table tr,.kumo-training .kt-table td{display:block;width:100%}
.kumo-training .kt-table thead{display:none}
.kumo-training .kt-table tr{padding:8px 0;border-bottom:1px solid $surface_border}
.kumo-training .kt-table td{padding:3px 0;border:0;white-space:normal}
.kumo-training .kt-table td:before{content:attr(data-label);font-size:12px;
color:$muted;display:block;text-align:left}
.kumo-training .kt-table .kt-number{text-align:left}
.kumo-training .kt-value{text-align:right}
.kumo-training .kt-detail{padding:10px}
.kumo-training .kt-feature-choice,.kumo-training .kt-feature-cells>li{grid-template-columns:minmax(0,1fr)}
}
""").substitute(CSS_THEME)

_SCRIPT = r"""
const root=host.querySelector('.kt-content'),fallback=host.querySelector('.kumo-training-static');
const list=v=>Array.isArray(v)?v:[], obj=v=>v&&typeof v==='object'&&!Array.isArray(v)?v:{};
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const text=v=>v===null||v===undefined?'Unavailable':typeof v==='object'?JSON.stringify(v):String(v);
const number=v=>finite(v)?(v===0?'0':String(Number(v.toPrecision(6)))):'Unavailable';
const el=(tag,value,cls)=>{const node=document.createElement(tag);if(value!==undefined)node.textContent=text(value);
if(cls)node.className=cls;return node};
const note=(parent,value)=>parent.append(el('p',value,'kt-note'));
const section=title=>{const node=el('section');node.append(el('h2',title));root.append(node);return node};
const state=(parent,source,label,producerLabel='Producer eligibility')=>{source=obj(source);if(source.status!=='available')
note(parent,label+': '+text(source.status||'unavailable')+(source.reason?'. '+text(source.reason):''));
if(source.producer_note)note(parent,producerLabel+': '+text(source.producer_note));};
const table=(parent,headers,rows,numeric=[],exact=false)=>{const node=el('table',undefined,'kt-table');
const head=el('thead'),tr=el('tr');for(const [i,h]of headers.entries()){
const th=el('th',h,numeric.includes(i)?'kt-number':'');th.scope='col';tr.append(th)}head.append(tr);node.append(head);
const body=el('tbody');for(const values of rows){const row=el('tr');values.forEach((value,i)=>{
const td=el('td',undefined,numeric.includes(i)?'kt-number':'');td.dataset.label=headers[i];
if(value instanceof Node){if(exact){if(value.classList.contains('kt-class-label'))td.append(value.cloneNode(true));
else td.textContent=value.textContent;}else td.append(value)}
else td.textContent=finite(value)?(exact?String(value):number(value)):text(value);row.append(td)});body.append(row)}
node.append(body);parent.append(node);
if(!exact&&numeric.length&&rows.some(row=>numeric.some(i=>finite(row[i])&&String(row[i])!==number(row[i])))){
const details=el('details');details.append(el('summary','Exact recorded values'));
table(details,headers,rows,numeric,true);parent.append(details)}return node};
const action=(label,callback)=>{const button=el('button',label,'kt-action');button.type='button';button.disabled=true;
button.addEventListener('click',callback,{signal});return button};
const control=(parent,label,values,callback)=>{const wrap=el('div',undefined,'kt-controls'),select=el('select');
select.id=widgetId+'-control-'+(++controlId);select.disabled=true;const caption=el('label',label);caption.htmlFor=select.id;
for(const [value,name]of values){const option=el('option',name);option.value=value;select.append(option)}
select.addEventListener('change',()=>callback(select.value),{signal});wrap.append(caption,select);parent.append(wrap);return select};
const enable=()=>root.querySelectorAll('button,select').forEach(node=>node.disabled=false);
let controlId=0;
const identityNote=(parent,title,value,tag='p',limit=120)=>{
if(value.length<=limit&&!/[\r\n]/.test(value)){parent.append(el(tag,value,'kt-note'));return;}
const details=el('details');details.append(el('summary','Explore '+title.toLowerCase()));
const values=el('div',undefined,'kt-sample-values');values.tabIndex=0;
values.setAttribute('role','region');values.setAttribute('aria-label',title);
note(values,value);details.append(values);parent.append(details);};
const provenance=obj(data.provenance),context=obj(data.context);
const linkTask=['static_link_prediction','temporal_link_prediction'].includes(context.task_type||provenance.task_type);
const baseMetric=row=>text(row.name).replace(/^(test_refit_|train_|val_|test_)/,'');
const results=section('Training results');
const queryNode=el('div');
if(context.query){identityNote(queryNode,'Predictive query',text(context.query));results.append(queryNode);}
note(results,[(context.task_type||provenance.task_type)&&('Task: '+text(context.task_type||provenance.task_type)),context.target&&('Target: '+text(context.target)),
context.run&&('Run: '+context.run),'Selected trial: '+text(provenance.selected_trial),
'Explanation fit: '+text(provenance.explanation_trial),'Explanation split: TEST',
'Refit on train + validation: '+(provenance.refitted_trainval===true?'yes':provenance.refitted_trainval===false?'no':'unknown')].filter(Boolean).join(' · '));
if(provenance.selection_verified===false)note(results,'Selection unverified: '+text(provenance.selection_reason));
if(['disabled','failed'].includes(provenance.generation_status))note(results,'Explanation generation: '+text(provenance.generation_status));
if(provenance.generation_failure&&provenance.generation_status!=='failed')note(results,text(provenance.generation_failure));
const metricRows=list(obj(data.metrics).rows);state(results,data.metrics,'Metrics');
let rankingEntry;
if(linkTask)results.append(el('p',context.query?'This task ranks linked IDs for each entity in the predictive query above.':
'This task ranks linked IDs for each entity. The predictive query was not recorded.'));
const metricDetail=linkTask?el('details'):results;
if(linkTask){
const selectedName=typeof provenance.selection_metric==='string'?provenance.selection_metric.replace(/^val_/,''):null;
const fit=provenance.refitted_trainval===true?'refit_trainval':provenance.refitted_trainval===false?'original':null;
const selected=provenance.selection_verified===true&&fit&&selectedName?metricRows.find(row=>
row.name===selectedName&&row.split==='test'&&row.fit===fit&&finite(row.value)&&row.origin==='Selected-checkpoint evaluation'):null;
const headline=el('div',undefined,'kt-result');results.append(headline);
if(selected){
const map=/^map@([1-9][0-9]*)$/.exec(selected.name);
headline.append(el('h3',map?'TEST MAP@'+map[1]:'TEST '+text(selected.name)),el('div',number(selected.value),'kt-result-value'));
identityNote(headline,'Result scope',(fit==='original'?'Original fit':'Refit on train + validation')+' · TEST · '+selected.origin);
if(map)note(headline,'Mean average precision over the first '+map[1]+' ranked IDs: observed IDs placed earlier raise the score. Higher is better. This is not a probability or an accuracy percentage.');
else note(headline,'Recorded selection metric on TEST'+(provenance.selection_direction==='min'?'; lower is better.':provenance.selection_direction==='max'?'; higher is better.':'. Metric direction is not recorded.'));
note(headline,'No baseline or quality target is recorded here. This score alone does not establish model quality.');
const exact=el('details');exact.append(el('summary','Exact result and provenance'));
table(exact,['Recorded metric','Exact value','Fit / split / origin'],[[selected.recorded_name,selected.value,
text(selected.fit)+' / TEST / '+selected.origin]],[1],true);headline.append(exact);
}else{headline.append(el('h3','Selected TEST result unavailable'));
note(headline,'A verified selection, matching TEST fit, selected-checkpoint origin and numeric selection metric are required. Inspect the recorded metrics below.');}
rankingEntry=el('div',undefined,'kt-ranking');results.append(rankingEntry);
metricDetail.append(el('summary','Explore all recorded metrics'));results.append(metricDetail);
}
const fits=[...new Set(metricRows.map(row=>text(row.fit)))];
for(const fit of fits){metricDetail.append(el('h3','Measured results · '+fit));
const rows=metricRows.filter(row=>text(row.fit)===fit),names=[...new Set(rows.map(baseMetric))];
const splits=['train','val','test'].filter(split=>rows.some(row=>row.split===split));
for(const split of rows.map(row=>text(row.split)))if(!splits.includes(split))splits.push(split);
const originNotes=el('div',undefined,'kt-provenance');metricDetail.append(originNotes);
for(const split of splits){const group=rows.filter(row=>row.split===split);
const origins=[...new Set(group.map(row=>row.origin||'Unknown metric origin'))];
for(const origin of origins)note(originNotes,split+': '+origin+(origins.length>1?' · '+group
.filter(row=>(row.origin||'Unknown metric origin')===origin).map(baseMetric).join(', '):''));}
table(metricDetail,['Metric',...splits],names.map(name=>[name,...splits.map(split=>{
const row=rows.find(item=>baseMetric(item)===name&&text(item.split)===split);return row?row.value:'Unavailable'})]),splits.map((_,i)=>i+1));}
if(provenance.selection_metric||obj(data.metrics).selection_metric)
note(metricDetail,'Selection metric: '+text(provenance.selection_metric||data.metrics.selection_metric)+
(provenance.selection_direction==='min'?' · Recorded optimization: minimize':provenance.selection_direction==='max'?' · Recorded optimization: maximize':''));
const histories=list(obj(data.history).rows);state(results,data.history,'Learning history');
if(histories.length){results.append(el('h3','Learning history'));
note(results,'Recorded epoch keys. Missing values remain unavailable; fits are shown separately.');
const jumps=el('nav',undefined,'kt-controls');jumps.setAttribute('aria-label','After learning history');
for(const [index,label]of [[1,'recorded patterns'],[2,'selected examples']])jumps.append(action('Go to '+label,()=>{
showPanel(index,true);}));
results.append(jumps);
const historyDetails=el('details');historyDetails.append(el('summary','Explore exact epoch values · '+histories.length+' recorded epochs'));
const metricName=name=>name.replace(/^(train_|val_|validation_)/,'');
const historyValue=(values,metric)=>{const entry=Object.entries(obj(values)).find(([name])=>metricName(name)===metric);return entry?.[1]};
const metrics=[...new Set(histories.flatMap(row=>[...Object.keys(obj(row.train_metrics)),...Object.keys(obj(row.validation_metrics))]).map(metricName))];
const historyView=el('div'),historyValues=el('div',undefined,'kt-history-values');historyValues.tabIndex=0;
historyValues.setAttribute('role','region');historyValues.setAttribute('aria-label','Exact learning history epoch values');historyDetails.append(historyValues);
let historySeries='both';
const showHistorySeries=()=>historyView.querySelectorAll('[data-history-series]').forEach(node=>
node.toggleAttribute('hidden',historySeries!=='both'&&node.dataset.historySeries!==historySeries));
const chart=(parent,rows,metric)=>{const ordered=rows.filter(row=>Number.isFinite(Number(row.epoch)))
.slice().sort((a,b)=>Number(a.epoch)-Number(b.epoch));
const values=ordered.flatMap(row=>[historyValue(row.train_metrics,metric),historyValue(row.validation_metrics,metric)]).filter(finite);
if(!values.length||!ordered.length)return;
const low=Math.min(...values),high=Math.max(...values),first=Number(ordered[0].epoch),last=Number(ordered.at(-1).epoch);
const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
svg.setAttribute('viewBox','0 0 600 160');svg.setAttribute('role','img');svg.classList.add('kt-history-chart');
svg.setAttribute('aria-label','Learning history for '+metric+'. History series selects visible curves on a fixed scale. Train: circles, validation: squares. Expand Explore exact epoch values for recorded values.');
const x=epoch=>20+(Number(epoch)-first)/(last-first||1)*560,y=value=>140-(value-low)/(high-low||1)*120;
parent.append(svg);
for(const [key,color,shape]of [['train_metrics','var(--kumo-accent)','circle'],['validation_metrics','var(--kumo-foreground)','rect']]){
const group=document.createElementNS(ns,'g'),line=document.createElementNS(ns,'path');group.dataset.historySeries=key;svg.append(group);group.append(line);
let previous=null,path='';for(const [index,row]of ordered.entries()){const value=historyValue(row[key],metric),epoch=Number(row.epoch);
if(!finite(value)){previous=null;continue;}
const adjacent=previous&&epoch-previous.epoch===1,next=ordered[index+1],offset=shape==='rect'?16:0;
path+=(adjacent?'L':'M')+x(epoch)+' '+y(value)+' ';
if(!adjacent||!next||Number(next.epoch)-epoch!==1||!finite(historyValue(next[key],metric))||
Math.floor((x(epoch)-20+offset)/32)!==Math.floor((x(previous.epoch)-20+offset)/32)){
const point=document.createElementNS(ns,shape);for(const [name,attribute]of Object.entries(shape==='circle'?{cx:x(epoch),cy:y(value),r:3}:{x:x(epoch)-4,y:y(value)-4,width:8,height:8}))point.setAttribute(name,attribute);
point.setAttribute('fill',shape==='circle'?color:'none');point.setAttribute('stroke',color);point.setAttribute('stroke-width','1.5');group.append(point);}previous={epoch};}
for(const [name,attribute]of Object.entries({d:path,fill:'none',stroke:color,'stroke-width':2}))line.setAttribute(name,attribute);
if(shape==='rect')line.setAttribute('stroke-dasharray','5 4');
if(!path){const missing=el('p',(shape==='circle'?'Train':'Validation')+': no recorded values for this metric.','kt-note');missing.dataset.historySeries=key;parent.append(missing);}}
note(parent,'Train: circles and solid lines. Validation: hollow squares and dashed lines. Markers are spaced; choose History series to inspect overlaps on the same scale. Epoch '+first+' to '+last+
'; metric range '+number(low)+' to '+number(high)+'. Missing values and unrecorded epochs break the lines.');};
const drawHistory=metric=>{historyView.replaceChildren();historyValues.replaceChildren();
for(const identity of [...new Set(histories.map(row=>text(row.trial)+' / '+text(row.fit)))]){
const rows=histories.filter(row=>text(row.trial)+' / '+text(row.fit)===identity);
historyView.append(el('h4',identity));chart(historyView,rows,metric);
historyValues.append(el('h4',identity));table(historyValues,['Trial / fit','Epoch','Train '+metric,'Validation '+metric],rows.map(row=>[
text(row.trial)+' / '+text(row.fit),text(row.epoch),historyValue(row.train_metrics,metric),historyValue(row.validation_metrics,metric)]),[1,2,3],true);}};
if(metrics.length){control(results,'History metric',metrics.map(name=>[name,name]),metric=>{drawHistory(metric);showHistorySeries()});
control(results,'History series',[['both','Both'],['train_metrics','Train'],['validation_metrics','Validation']],series=>{historySeries=series;showHistorySeries()});
drawHistory(metrics[0]);results.append(historyView,historyDetails)}
else note(results,'No recorded epoch metric values.');}
const trials=list(obj(data.trials).rows);state(results,data.trials,'Trial comparison');
if(trials.length){results.append(el('h3','Recorded trials'));
const names=[...new Set(trials.flatMap(row=>Object.keys(obj(row.comparison_metrics??row.final_validation_metrics))))];
if(obj(data.trials).total!==undefined)note(results,'Recorded completed trials: '+text(data.trials.completed)+' / '+text(data.trials.total));
const trialView=el('div',undefined,'kt-trials');
const drawTrials=metric=>{trialView.replaceChildren();
table(trialView,['Trial / fit','Selection / progress',...(metric?[metric]:[])],trials.map(row=>[
text(row.trial)+' / '+text(row.fit),(provenance.selection_verified!==true?'Selection unverified':row.selected?'Selected by recorded winner':'Not selected')+' · '+
(row.status||row.reason||('Epochs: '+text(row.num_epochs_completed)+' · Device: '+text(row.accelerator)))+
(row.comparison_source==='last_recorded_epoch'?' · Validation from last recorded epoch '+text(row.comparison_epoch):row.comparison_source==='final_validation_metrics'?' · Selected-checkpoint validation':' · Validation origin unverified'),
...(metric?[obj(row.comparison_metrics??row.final_validation_metrics)[metric]]:[])]),metric?[2]:[]);};
if(names.length){const metric=names.includes(provenance.selection_metric)?provenance.selection_metric:names[0];
const select=control(results,'Trial metric',names.map(name=>[name,name]),drawTrials);select.value=metric;drawTrials(metric)}
else drawTrials(null);
results.append(trialView);
note(results,'Missing progress is not evidence that a trial failed.');}
const patterns=section('Recorded patterns');patterns.append(el('h3','Global input sensitivity'));
note(patterns,'Unsigned recorded gradient magnitude, normalized across all columns when the total is positive. '
+'These scores describe sensitivity, not causal effects or prediction probabilities.');
state(patterns,data.gradient,'Global sensitivity','Producer scope');
const gradients=list(obj(data.gradient).rows).map(obj), valid=gradients.filter(row=>finite(row.score)&&row.score>=0)
.sort((a,b)=>b.score-a.score||text(a.table).localeCompare(text(b.table))||text(a.column).localeCompare(text(b.column)));
if(valid.length!==gradients.length)note(patterns,'Invalid or negative gradient scores are excluded from the unsigned ranking.');
note(patterns,'Highest recorded gradient scores; showing '+Math.min(20,valid.length)+' of '+text(obj(data.gradient).total??gradients.length)+' columns. The visible subset is not renormalized.');
const cohorts=list(obj(data.cohorts).rows);
let cohortView,cohortButtons=[],selectedFeature=valid[0],selectedExample;
const featureButtons=[],featureContext=el('div',undefined,'kt-detail kt-feature-context');
const chooseFeature=row=>{selectedFeature=row;drawFeatureContext(featureContext);
featureButtons.forEach(({button,feature})=>{button.setAttribute('aria-pressed',String(feature===row));
button.querySelector('.kt-note').textContent=feature===row?'Selected feature':'Inspect loaded context';});
if(!sampleView.hidden&&selectedExample)inspectSample(selectedExample.ref,selectedExample.index);enable();};
const inspectCohort=index=>{cohortView.replaceChildren();const c=cohorts[index];
cohortView.append(el('h3',text(c.table_name)+'.'+text(c.col_name)+' · hop '+text(c.hop)));
note(cohortView,'Prediction variation rank score: '+number(c.prediction_score)+'. The ranking averages recorded class dimensions.');
const bins=el('div'),classes=Array.isArray(c.predictions)?[]:Object.keys(obj(c.predictions));
const draw=className=>{bins.replaceChildren();if(classes.length&&!className){note(bins,'Choose a recorded class to compare cohort means.');return;}
const predictions=classes.length?obj(c.predictions)[className]:c.predictions;
const targets=Array.isArray(c.targets)?c.targets:obj(c.targets)[className];
let predictionLabel='Mean prediction',targetLabel='Mean observed target';
if(text(obj(data.cohorts).task_type).includes('binary')){predictionLabel='Positive-class probability';targetLabel='Observed positive-class share'}
else if(classes.length){predictionLabel='Mean class probability';targetLabel='Observed class share'}
table(bins,['Cohort',predictionLabel,targetLabel,'Population share (%)'],list(c.cohorts).slice(0,100).map((label,i)=>[
label,list(predictions)[i],list(targets)[i],finite(list(c.populations)[i])?c.populations[i]*100:'Unavailable']),[1,2,3]);
if(list(c.cohorts).length>100)note(bins,'Showing the first 100 recorded cohorts.');};
if(classes.length)control(cohortView,'Recorded class',[['','Choose a class'],...classes.map(name=>[name,name])],draw);
else note(cohortView,text(obj(data.cohorts).task_type).includes('regression')?'Mean values use recorded target units.':'Recorded prediction and target means.');
cohortView.append(bins);draw('');cohortButtons.forEach((button,i)=>{button.setAttribute('aria-pressed',String(i===index));
button.closest('tr').setAttribute('aria-selected',String(i===index));});enable();};
if(valid.length){const rows=el('ul',undefined,'kt-feature-list');rows.setAttribute('aria-label','Global input sensitivity');
for(const row of valid.slice(0,20)){
const score=el('div',undefined,'kt-score'),track=el('div',undefined,'kt-track'),bar=el('div',undefined,'kt-bar');
bar.style.width=Math.min(100,row.score*100)+'%';track.setAttribute('aria-hidden','true');track.append(bar);
score.append(track,el('span',number(row.score),'kt-value'));
const button=action('',()=>chooseFeature(row));button.classList.add('kt-feature-choice');
const label=el('span',text(row.table)+'.'+text(row.column));label.append(el('span','Inspect loaded context','kt-note'));
button.append(label,score);featureButtons.push({button,feature:row});const item=el('li');item.append(button);rows.append(item);}
patterns.append(rows,featureContext);
const exact=el('details');exact.append(el('summary','Exact global sensitivity values'));
table(exact,['Input column','Global score'],valid.slice(0,20).map(row=>[text(row.table)+'.'+text(row.column),row.score]),[1],true);patterns.append(exact);}
patterns.append(el('h3','Prediction variation across cohorts'));state(patterns,data.cohorts,'Cohorts');
note(patterns,'Ranked separately from gradient sensitivity. Table, hop and column identify each cohort record.');
const ranked=cohorts.map((row,index)=>({row,index})).sort((a,b)=>(finite(b.row.prediction_score)?b.row.prediction_score:-Infinity)-(finite(a.row.prediction_score)?a.row.prediction_score:-Infinity)||a.index-b.index);
if(ranked.length){cohortButtons=Array(cohorts.length);table(patterns,['Table / column / hop','Variation score','Detail'],ranked.slice(0,100).map(({row,index})=>{
const button=action('Inspect cohorts',()=>inspectCohort(index));cohortButtons[index]=button;return[
text(row.table_name)+'.'+text(row.col_name)+' · hop '+text(row.hop),row.prediction_score,button];}),[1]);
if(ranked.length>100)note(patterns,'Showing the first 100 cohort records by recorded variation.');}
cohortView=el('div',undefined,'kt-detail');patterns.append(cohortView);
if(ranked.length)inspectCohort(ranked[0].index);else{cohortView.hidden=true;}
const examples=section('Selected examples');
note(examples,'Selected TEST examples with sampled relational context. Recorded selection strategies can overlap and do not define thresholded prediction outcomes. '
+'Collector IDs identify artifacts, not original entities.');state(examples,data.samples,'Examples');
const samples=obj(data.samples),coverage=obj(samples.coverage),references=list(samples.references),bodies=obj(samples.bodies);
note(examples,'Coverage: '+text(coverage.references_listed??references.length)+' listed of '+text(coverage.references_total)+' recorded references; '
+text(coverage.available)+' available graph files; '+text(coverage.loaded??Object.keys(bodies).length)+' loaded; '
+text(coverage.missing)+' missing; '+text(coverage.malformed)+' malformed; '+text(coverage.excluded)+' excluded by load limits.');
if(references.length)note(examples,'Preloaded example identities and values may be saved in notebook HTML or widget state. '
+'Inspect other retained graphs in Python with explanations.sample(collector_id).');
const sampleView=el('div',undefined,'kt-detail'),sampleButtons=[];
const nodeIdentity=(store,id)=>{const node=obj(obj(store.nodes)[id]);
const tables=Object.entries(obj(store.tables)).filter(([,ids])=>list(ids).some(value=>String(value)===String(id))).map(([name])=>name);
return node.count!==undefined?'Aggregate of '+text(node.count)+' recorded nodes'+(tables.length?' · '+tables.join(', '):''):
Object.hasOwn(node,'pkey')&&node.pkey!==null?(tables.length?tables.join(', ')+'.':'')+text(node.pkey_name)+': '+JSON.stringify(node.pkey):'Original entity unavailable';};
const sampleIdentity=ref=>{const store=obj(obj(bodies[ref.artifact_id]).subgraph_store);return nodeIdentity(store,store.root_id);};
const featureCells=body=>Object.entries(obj(obj(body.feature_store).features)).flatMap(([id,columns])=>
Object.entries(obj(columns)).map(([name,value])=>({id,name,feature:obj(value)})));
const localAxis=body=>featureCells(body).reduce((max,{feature})=>
finite(feature.score)&&feature.score>=0?Math.max(max,feature.score):max,1);
const localScore=(parent,feature,axis)=>{
if(finite(feature.score)&&feature.score>=0){const mark=el('div',undefined,'kt-score'),track=el('div',undefined,'kt-track'),bar=el('div',undefined,'kt-bar');
bar.style.width=(feature.score/axis*100)+'%';track.setAttribute('aria-hidden','true');track.append(bar);mark.append(track,el('span',number(feature.score),'kt-value'));parent.append(mark);}
else note(parent,feature.score==null?'Unavailable':'Invalid recorded score');};
const matchingCells=ref=>featureCells(obj(bodies[ref.artifact_id])).filter(({feature})=>
typeof selectedFeature.table==='string'&&typeof selectedFeature.column==='string'&&
feature.table_name===selectedFeature.table&&feature.name===selectedFeature.column);
const drawFeatureContext=(parent,entry)=>{parent.replaceChildren();if(!selectedFeature)return;
const featureName=text(selectedFeature.table)+'.'+text(selectedFeature.column);
parent.append(el('h3','Inspect '+featureName));
note(parent,'Global sensitivity: '+number(selectedFeature.score)+' · across recorded input columns.');
const matches=loaded.filter(({ref})=>matchingCells(ref).length);
if(!entry){if(!selectedExample)selectedExample=matches[0];entry=selectedExample;
if(matches.length){const choices=matches.map(item=>[String(item.index),sampleIdentity(item.ref)+' · collector '+text(item.ref.artifact_id)]);
if(entry&&!matches.some(item=>item.index===entry.index))choices.unshift([String(entry.index),sampleIdentity(entry.ref)+' · feature not recorded']);
const select=control(parent,'Matching loaded example',choices,value=>{
selectedExample=matches.find(item=>String(item.index)===value)||entry;
if(!sampleView.hidden)inspectSample(selectedExample.ref,selectedExample.index);else drawFeatureContext(parent);enable();});select.value=String(entry.index);}}
if(!entry){note(parent,'No loaded example contains this exact recorded table and feature name. Missing, malformed and unloaded bodies cannot establish a match.');}
else{const body=obj(bodies[entry.ref.artifact_id]),store=obj(body.subgraph_store),cells=matchingCells(entry.ref);
identityNote(parent,'Selected example',sampleIdentity(entry.ref)+' · Collector '+text(entry.ref.artifact_id));
if(!cells.length)note(parent,'This loaded example has no matching recorded feature.');
else{const axis=localAxis(body);
note(parent,'Local scores use this example’s own scale (0 to '+number(axis)+'). They cannot be compared numerically with global scores or other examples. Unavailable means unrecorded, not zero.');
if(cells.some(({id})=>obj(obj(store.nodes)[id]).count!==undefined))note(parent,'Aggregate scores sum normalized source cells and can exceed 1; aggregate values are not retained.');
const rows=el('ul',undefined,'kt-feature-cells');rows.setAttribute('aria-label','Matching recorded feature values and local sensitivity');
for(const {id,feature}of cells.slice(0,80)){const row=el('li'),identity=el('div'),value=el('div'),score=el('div');
identity.append(el('strong',nodeIdentity(store,id)));note(identity,'Recorded node: '+id);
value.append(el('span','Recorded value','kt-note'),el('div',feature.value==null&&obj(obj(store.nodes)[id]).count!==undefined?'Aggregate value not retained':text(feature.value)));
score.append(el('span','Local sensitivity','kt-note'));
localScore(score,feature,axis);
row.append(identity,value,score);rows.append(row);}parent.append(rows);
if(cells.length>80)note(parent,'Showing 80 of '+cells.length+' matching recorded cells; complete loaded body remains available in Python.');
const exact=el('details');exact.append(el('summary','Exact matching feature values and local scores'));
note(exact,'Local sensitivity is divided by this example’s maximum source-cell sensitivity. A source-cell score of 1 marks that maximum. Aggregate scores sum these normalized cells. Numeric scores are unchanged when the bar axis extends above 1.');
table(exact,['Recorded node','Value','Local score'],cells.slice(0,80).map(({id,feature})=>[id,text(feature.value),feature.score]),[2],true);parent.append(exact);}
if(parent===featureContext)parent.append(action('Inspect this example’s context',()=>{inspectSample(entry.ref,entry.index);showPanel(2);
const heading=sampleView.querySelector('h3');heading.tabIndex=-1;heading.focus();}));}
if(parent===featureContext){note(parent,matches.length+' of '+loaded.length+' loaded examples contain this exact feature. These are recorded contexts, not cohort memberships or a population sample.');
const cohortMatches=cohorts.map((c,i)=>c.table_name===selectedFeature.table&&c.col_name===selectedFeature.column?i:-1).filter(i=>i>=0);
for(const index of cohortMatches)parent.append(action('Inspect cohorts · hop '+text(cohorts[index].hop),()=>{inspectCohort(index);const heading=cohortView.querySelector('h3');heading.tabIndex=-1;heading.focus();}));
if(!cohortMatches.length)note(parent,'No matching cohort record.');}};
const rankedOutput=(parent,output,artifact)=>{
state(parent,output,'Output context');
if(!['available','partial'].includes(output.status))return;
const predictions=output.prediction,targets=output.target;
// Only complete typed identities support matches; unknown namespaces cannot prove absence.
const key=node=>{node=obj(node);return (typeof node.id==='string'||finite(node.id))&&typeof node.id_name==='string'&&node.id_name.length?
JSON.stringify([typeof node.id,node.id_name,node.id]):null;};
const first=new Map(),observed=new Set();
list(predictions).forEach((node,index)=>{const identity=key(node);if(identity!==null&&!first.has(identity))first.set(identity,index)});
list(targets).forEach(node=>{const identity=key(node);if(identity!==null)observed.add(identity)});
const appendID=(parent,node)=>{node=obj(node);const value=JSON.stringify(node.id)??'Unavailable',name=text(node.id_name);
parent.append(el('strong',value.length>48||/[\r\n]/.test(value)?value.replace(/\s+/g,' ').slice(0,48)+'…':value));
if(value.length>48||/[\r\n]/.test(value))identityNote(parent,'Full recorded ID',value,'div',0);
if(name.length>48||/[\r\n]/.test(name)){note(parent,'ID column: '+name.replace(/\s+/g,' ').slice(0,48)+'…');identityNote(parent,'Full ID column',name,'div',0);}
else note(parent,'ID column: '+(node.id_name==null||name===''?'not recorded':name));};
parent.append(el('h4','Predicted IDs · '+(Array.isArray(predictions)?'recorded top '+predictions.length:'ranking unavailable')));
if(predictions==null)note(parent,'Prediction list not recorded.');
else if(!Array.isArray(predictions))note(parent,'Prediction list is invalid.');
else if(!predictions.length)note(parent,'Prediction list recorded empty (0 IDs).');
else{const ranks=el('ol',undefined,'kt-ranked-list');ranks.setAttribute('aria-label','Predicted IDs in recorded rank order');
for(const [index,node]of predictions.slice(0,100).entries()){
const identity=key(node),duplicate=identity!==null&&first.get(identity)!==index;
const match=identity!==null&&observed.has(identity)&&!duplicate;
const item=el('li',undefined,match?'kt-match':'');item.append(el('span','Rank '+(index+1),'kt-note'));appendID(item,node);
if(match)note(item,'Observed ID');else if(duplicate)note(item,'Repeated ID');ranks.append(item);}parent.append(ranks);}
parent.append(el('h4','Observed TEST IDs'));
if(targets==null)note(parent,'Observed target list not recorded; matches cannot be assessed.');
else if(!Array.isArray(targets))note(parent,'Observed target list is invalid; matches cannot be assessed.');
else if(!targets.length)note(parent,'Observed target list recorded empty (0 IDs).');
else{const values=el('ul',undefined,'kt-observed-list'),seen=new Set();values.setAttribute('aria-label','Observed TEST IDs and their predicted ranks');
for(const node of targets.slice(0,100)){
const identity=key(node),rank=identity===null?undefined:first.get(identity),item=el('li');appendID(item,node);
let comparison=predictions==null?'Ranking not recorded':!Array.isArray(predictions)?'Ranking invalid':
rank!==undefined?'Rank '+(rank+1)+' of '+predictions.length:
identity===null||predictions.some(pred=>key(pred)===null||obj(pred).id_name!==obj(node).id_name)?
'Comparison unavailable: ID columns differ or are missing':'Absent from '+predictions.length+' retained predictions';
if(identity!==null&&seen.has(identity))comparison='Repeated observed ID · '+comparison;
if(identity!==null)seen.add(identity);note(item,comparison);values.append(item);}parent.append(values);}
if(list(predictions).length>100||list(targets).length>100)note(parent,'Showing the first 100 IDs of each list; matched ranks and absence compare the complete retained prediction list. '
+'Inspect complete output in Python from the matching TEST fit’s subgraph_collector/test/output/'+text(artifact)+'.json.');
note(parent,'Observed IDs are TEST targets, not a complete link history. ID values and recorded ID columns must agree for comparison. '
+'No retrieval scores, probabilities or output-ID links to sampled graph nodes are retained.');
};
const sampleDetailRow=el('tr'),sampleDetailCell=el('td');sampleDetailCell.colSpan=3;
sampleDetailCell.append(sampleView);sampleDetailRow.append(sampleDetailCell);
const sampleTable=(title,headers,rows,total,numeric=[])=>{
const details=el('details');details.append(el('summary','Explore '+title.toLowerCase()+' · '+rows.length+' of '+total+' shown'));
const values=el('div',undefined,'kt-sample-values');values.tabIndex=0;
values.setAttribute('role','region');values.setAttribute('aria-label',title);
table(values,headers,rows,numeric);details.append(values);sampleView.append(details);};
const selectedNodes=new Map();
const drawRelationships=(parent,body,ref)=>{
const store=obj(body.subgraph_store),nodes=obj(store.nodes),cells=featureCells(body),axis=localAxis(body),relations=[];
let invalidRelations=0;
for(const [from,kinds]of Object.entries(obj(store.edges))){
if(!kinds||typeof kinds!=='object'||Array.isArray(kinds)){invalidRelations++;continue;}
for(const [kind,tos]of Object.entries(kinds)){
if(!Array.isArray(tos)){invalidRelations++;continue;}
for(const to of tos){if(typeof to==='string'||finite(to))relations.push([from,kind,String(to)]);else invalidRelations++;}}}
const nodeIds=Object.keys(nodes),featureIds=[...new Set(cells.map(cell=>cell.id))];
const highlighted=new Set(selectedFeature?matchingCells(ref).map(({feature})=>feature):[]);
const allIds=[...new Set([...nodeIds,...featureIds])],rootId=store.root_id==null?null:String(store.root_id);
const adjacent=new Map();for(const edge of relations)for(const id of new Set([edge[0],edge[2]])){
if(!adjacent.has(id))adjacent.set(id,[]);adjacent.get(id).push(edge);}
const reached=new Set(),queue=Object.hasOwn(nodes,rootId)?[rootId]:[];
for(let i=0;i<queue.length;i++){const id=queue[i];if(reached.has(id))continue;reached.add(id);
for(const [from,,to]of adjacent.get(id)||[])if(Object.hasOwn(nodes,from)&&Object.hasOwn(nodes,to)){
const next=from===id?to:from;if(!reached.has(next))queue.push(next);}}
const groups={connected:nodeIds.filter(id=>reached.has(id)),disconnected:nodeIds.filter(id=>!reached.has(id)),
features:featureIds.filter(id=>!Object.hasOwn(nodes,id))};
let selected=selectedNodes.get(ref.artifact_id);
if(!allIds.includes(selected))selected=allIds.includes(rootId)?rootId:allIds[0];
let group=groups.connected.length?'connected':groups.disconnected.length?'disconnected':'features',recordPage=0,edgePage=0,featurePage=0;
const view=el('div',undefined,'kt-neighborhood');parent.append(view);
const label=id=>Object.hasOwn(nodes,id)?nodeIdentity(store,id):'Node '+id+' · structure unavailable';
const compact=value=>value.length>80||/[\r\n]/.test(value)?value.replace(/\s+/g,' ').slice(0,80)+'…':value;
const draw=(focus=false)=>{const browseOpen=view.querySelector('.kt-record-browser')?.open||false;
view.replaceChildren();view.append(el('h3','Explore recorded relationships'));
if(!body.subgraph_store)note(view,'Relational structure unavailable. Recorded feature values remain inspectable below.');
else if(!nodeIds.length)note(view,'No recorded nodes in this loaded graph.');
else if(!Object.hasOwn(nodes,rootId))note(view,'Root record unavailable; a complete connection to the example cannot be established.');
note(view,nodeIds.length+' structure records · '+relations.length+' recorded relations · '+groups.disconnected.length+' records without a recorded path to the root'
+(groups.features.length?' · '+groups.features.length+' feature-only records':''));
if(invalidRelations)note(view,invalidRelations+' malformed relation entries cannot be drawn. Inspect the retained body in Python.');
const dangling=relations.filter(([from,,to])=>!Object.hasOwn(nodes,from)||!Object.hasOwn(nodes,to)).length;
if(dangling)note(view,dangling+' recorded relations have missing endpoints. Missing node structure is labeled explicitly.');
const controls=el('div',undefined,'kt-controls');
if(Object.hasOwn(nodes,rootId)){const button=action('Return to root',()=>choose(rootId));button.setAttribute('aria-pressed',String(selected===rootId));controls.append(button);}
view.append(controls);
if(!allIds.length){note(view,'No recorded feature values. Inspect another loaded example or explanations.sample(collector_id).');enable();return;}
const layout=el('div',undefined,'kt-neighborhood-layout'),connections=el('div'),detail=el('div',undefined,'kt-selected-node');
const selectedLabel=label(selected),heading=el('h4',compact(selectedLabel));heading.tabIndex=-1;heading.id=widgetId+'-selected-node';detail.setAttribute('aria-labelledby',heading.id);
detail.append(el('span','Selected record'+(selected===rootId?' · Root':''),'kt-note'),heading);
if(compact(selectedLabel)!==selectedLabel)identityNote(detail,'Full selected record identity',selectedLabel,'div',0);
const node=obj(nodes[selected]);note(detail,'Recorded node: '+compact(selected));
if(compact(selected)!==selected)identityNote(detail,'Full recorded node ID',selected,'div',0);
if(!Object.hasOwn(nodes,selected))note(detail,'Feature values only; this record has no loaded node structure.');
else if(node.count!==undefined){note(detail,'Aggregate of '+text(node.count)+' recorded nodes. Constituent identities and individual values are not retained.');
if(node.start_time!==undefined||node.end_time!==undefined)note(detail,'Recorded interval: '+text(node.start_time)+' to '+text(node.end_time));}
else if(!reached.has(selected))note(detail,'No recorded path to the root. This does not establish absence of a relationship outside this retained graph.');
const local=cells.filter(cell=>cell.id===selected),matched=local.some(({feature})=>highlighted.has(feature));
if(selectedFeature)note(detail,(matched?'Highlighted global feature: ':'Selected global feature not recorded on this node: ')+text(selectedFeature.table)+'.'+text(selectedFeature.column));
if(!local.length)note(detail,'No recorded feature values or local scores for this record.');
else{note(detail,'Local sensitivity · same example axis: 0 to '+number(axis)+'. Unavailable is unrecorded, not zero.');
const rows=el('ul',undefined,'kt-node-features');rows.setAttribute('aria-label','Selected record feature values and local sensitivity');
for(const {name,feature}of local.slice(featurePage*80,(featurePage+1)*80)){
const match=highlighted.has(feature);
const item=el('li',undefined,match?'kt-match':'');item.append(el('strong',text(feature.table_name)+'.'+text(feature.name??name)));
if(match)note(item,'Selected global feature');
note(item,'Recorded value');item.append(el('div',feature.value==null&&node.count!==undefined?'Aggregate value not retained':text(feature.value)));
note(item,'Local sensitivity');localScore(item,feature,axis);rows.append(item);}detail.append(rows);
pages(detail,'feature cells',local.length,80,featurePage,value=>{featurePage=value;draw(true);});
const exact=el('details');exact.append(el('summary','Exact selected record values and scores'));
table(exact,['Feature','Value','Local score'],local.slice(featurePage*80,(featurePage+1)*80).map(({name,feature})=>[
text(feature.table_name)+'.'+text(feature.name??name),text(feature.value),feature.score]),[2],true);detail.append(exact);}
connections.append(el('h4','Connections to selected record'));
note(connections,'Arrows show recorded traversal from → to, not foreign-key or causal direction. Select a connected record to follow its context.');
const edges=adjacent.get(selected)||[],rows=el('ul',undefined,'kt-connections');rows.setAttribute('aria-label','Recorded connections to selected record');
for(const [from,kind,to]of edges.slice(edgePage*79,(edgePage+1)*79)){
const row=el('li'),line=el('div',undefined,'kt-connection');row.append(line);
for(const [i,id]of [from,to].entries()){
if(i)line.append(el('span','→ '+compact(kind)+' →','kt-connection-direction'));
const full=id===selected||allIds.includes(id)?label(id):'Missing node '+id;
const endpoint=id===selected?el('strong',compact(full)):allIds.includes(id)?action(compact(full),()=>choose(id)):el('span',compact(full));
line.append(endpoint);if(compact(full)!==full)identityNote(row,'Full '+(i?'to':'from')+' record identity',full,'div',0);}
if(compact(kind)!==kind)identityNote(row,'Full relation label',kind,'div',0);rows.append(row);}
connections.append(rows);if(!edges.length)note(connections,'No recorded relations for this record.');
pages(connections,'relations',edges.length,79,edgePage,value=>{edgePage=value;draw(true);});
layout.append(detail,connections);view.append(layout);
note(view,'Local sensitivity is divided by this example’s maximum source-cell sensitivity. Aggregate scores sum normalized cells and can exceed 1. Scores have no sign and are not percentages or relationship strength. They cannot be compared numerically with global scores or other examples.');
const browse=el('details',undefined,'kt-record-browser');browse.open=browseOpen;
browse.append(el('summary','Browse all loaded records · including disconnected context'));
const available=Object.entries(groups).filter(([,ids])=>ids.length),names={connected:'Root-connected records',disconnected:'No recorded path to root',features:'Feature values without structure'};
const scope=control(browse,'Record group',available.map(([key,ids])=>[key,names[key]+' ('+ids.length+')']),value=>{group=value;recordPage=0;draw();view.querySelector('.kt-record-browser').open=true;
view.querySelector('.kt-record-browser select').focus();});scope.value=group;
const ids=groups[group],choices=ids.slice(recordPage*80,(recordPage+1)*80);
const picker=control(browse,'Record',[['','Choose a record'],...choices.map(id=>[id,compact(label(id))])],value=>{if(value!=='')choose(value);});picker.className='kt-record-picker';picker.value=choices.includes(selected)?selected:'';
pages(browse,'records',ids.length,80,recordPage,value=>{recordPage=value;draw();view.querySelector('.kt-record-browser').open=true;
view.querySelector('.kt-record-picker').focus();});
view.append(browse);enable();if(focus)heading.focus();};
const choose=id=>{selected=id;selectedNodes.set(ref.artifact_id,id);edgePage=0;featurePage=0;draw(true);};
const pages=(parent,label,total,size,page,change)=>{if(total<=size)return;
const controls=el('div',undefined,'kt-controls');note(controls,'Showing '+(page*size+1)+'–'+Math.min(total,(page+1)*size)+' of '+total+' '+label+'.');
if(page>0)controls.append(action('Previous '+label,()=>change(page-1)));
if((page+1)*size<total)controls.append(action('Next '+label,()=>change(page+1)));parent.append(controls);};
if(selected!==undefined)selectedNodes.set(ref.artifact_id,selected);draw();};
const inspectSample=(ref,index)=>{selectedExample={ref,index};sampleView.replaceChildren();sampleView.hidden=false;
if(selectedFeature)drawFeatureContext(featureContext);
sampleButtons.find(item=>item.rowIndex===index)?.button.closest('tr').after(sampleDetailRow);
sampleView.append(el('h3',sampleIdentity(ref).length>100?sampleIdentity(ref).slice(0,100)+'…':sampleIdentity(ref)));
identityNote(sampleView,'Example identity',sampleIdentity(ref)+' · Collector '+text(ref.artifact_id));
identityNote(sampleView,'Recorded original IDs','Recorded original IDs: '+text(Object.keys(obj(ref.original_ids)).length?ref.original_ids:'Unavailable')
+' · Strategies: '+(list(ref.categories).join(', ')||'Unavailable'));
const output=obj(obj(samples.outputs)[ref.artifact_id]);
sampleView.append(el('h4','Prediction and observed target'));
if(!['static_link_prediction','temporal_link_prediction'].includes(output.task_type)||!['available','partial'].includes(output.status))state(sampleView,output,'Output context');
if(output.status==='available'||output.status==='partial'){
const classes=output.task_type==='multiclass_classification'||output.task_type==='multilabel_classification';
if(output.task_type==='static_link_prediction'||output.task_type==='temporal_link_prediction'){
rankedOutput(sampleView,output,ref.artifact_id);
note(sampleView,'The retained retrieval list can differ from query TOP K. Selected examples do not decompose the aggregate TEST metric.');
}else if(classes){
const labels=Object.keys(obj(output.prediction??output.target)),rows=labels.slice(0,100).map(label=>{
let name=label;
if(label.length>120||/[\r\n]/.test(label)){
name=el('div',undefined,'kt-class-label');name.append(el('div',label.replace(/\s+/g,' ').slice(0,80)+'…'));
identityNote(name,'Full recorded label',label,'div');}
return[name,output.prediction?.[label],output.target?.[label]];});
note(sampleView,output.task_type==='multilabel_classification'?
'Independent label probabilities; observed membership: 0 = absent, 1 = present. Multiple labels may be present; values need not sum to one.':
'Mutually exclusive class probabilities; observed membership: 0 = absent, 1 = present. Recorded values are not renormalized.');
note(sampleView,'Showing '+rows.length+' of '+labels.length+' retained labels. Total class coverage is not recorded.');
const headers=['Recorded label','Class probability (0 to 1)','Observed membership (0 or 1)'];
sampleTable('Recorded class outputs',headers,rows,labels.length,[1,2]);
if(labels.length>100)note(sampleView,'Inspect complete retained output in Python from the matching TEST fit’s subgraph_collector/test/output/'+text(ref.artifact_id)+'.json.');
}else{
const binary=output.task_type==='binary_classification';
note(sampleView,binary?'Probability of the positive class; observed target: 0 = negative class, 1 = positive class.':'Values use recorded target units.');
table(sampleView,[binary?'Positive-class probability (0 to 1)':'Prediction','Observed target'],[[output.prediction,output.target]],[0,1]);}}
const body=obj(bodies[ref.artifact_id]),store=obj(body.subgraph_store),nodes=obj(store.nodes);
const rootNode=obj(nodes[store.root_id]);
identityNote(sampleView,'Root identity','Root node: '+text(store.root_id)+' · Original entity: '+text(rootNode.pkey)
+' · Seed time: '+text(store.seed_time));
if(!body.subgraph_store)note(sampleView,'Relational structure unavailable.');
if(store.task_type==='static_link_prediction'||store.task_type==='temporal_link_prediction')
note(sampleView,'Link-prediction records can include retrieved items outside the sampled graph, with feature values but no local gradient scores. '
+'Relations and local scores cover only recorded context.');
drawRelationships(sampleView,body,ref);
if(selectedFeature){const selectedContext=el('details',undefined,'kt-feature-context');
selectedContext.append(el('summary','Selected global feature across this example'));
const values=el('div');selectedContext.append(values);sampleView.append(selectedContext);drawFeatureContext(values,{ref,index});}
const nodeRows=Object.entries(nodes);if(nodeRows.length){
sampleTable('Recorded nodes',['Node','Identity / aggregate','Table membership'],nodeRows.slice(0,80).map(([id,node])=>[
id,node.count!==undefined?'Aggregate: '+text(node.count)+' nodes':text(node.pkey_name)+': '+text(node.pkey),
Object.entries(obj(store.tables)).filter(([,ids])=>list(ids).some(value=>String(value)===id)).map(([name])=>name).join(', ')||'Unavailable']),nodeRows.length);
if(nodeRows.length>80)note(sampleView,'Showing 80 of '+nodeRows.length+' recorded nodes; complete loaded body is retained.');}
const relations=[];for(const [from,kinds]of Object.entries(obj(store.edges)))for(const [kind,tos]of Object.entries(obj(kinds)))
for(const to of list(tos))relations.push([from,kind,text(to)]);
if(relations.length){sampleTable('Recorded relations',['From node','Relation','To node'],relations.slice(0,80),relations.length);
if(relations.length>80)note(sampleView,'Showing 80 of '+relations.length+' recorded relations.');}
else note(sampleView,'No recorded relations in this loaded example.');

const cells=featureCells(body).map(({id,name,feature})=>[id,text(feature.table_name)+'.'+text(feature.name??name),text(feature.value),feature.score]);
if(cells.length)sampleTable('Node features',['Node','Feature','Recorded value','Local score'],cells.slice(0,200),cells.length,[3]);
else note(sampleView,'No recorded feature values.');
if(cells.length>200)note(sampleView,'Showing 200 of '+cells.length+' recorded feature cells.');
sampleButtons.forEach(({button,rowIndex})=>{button.setAttribute('aria-pressed',String(rowIndex===index));
button.closest('tr').setAttribute('aria-selected',String(rowIndex===index));});};
const catalogRows=references.slice(0,100).map((ref,index)=>({ref,index}));
const referenceTable=(parent,entries)=>table(parent,['Collector / original IDs','Recorded strategies','Availability / detail'],entries.map(({ref,index})=>{
const identity=el('div'),original=[...new Set(Object.values(obj(ref.original_ids)).map(text))];
identity.append(el('div','Collector '+text(ref.artifact_id)));
identityNote(identity,'Original IDs',original.length?'Original ID: '+original.join(', '):'Original IDs unavailable','div');
let detail;if(Object.hasOwn(bodies,ref.artifact_id)){detail=action('Inspect example',()=>inspectSample(ref,index));sampleButtons.push({button:detail,rowIndex:index});}
else detail=el('span',ref.reason||ref.status||(ref.available?'Not loaded':'Graph file missing'));
return[identity,list(ref.categories).join(', ')||'Unavailable',detail];}));

const loaded=catalogRows.filter(({ref})=>Object.hasOwn(bodies,ref.artifact_id));
const remaining=catalogRows.filter(({ref})=>!Object.hasOwn(bodies,ref.artifact_id));
if(loaded.length){examples.append(el('h3','Loaded examples'));referenceTable(examples,loaded)}
if(remaining.length){const details=el('details');details.append(el('summary','Other recorded references ('+remaining.length+')'));
referenceTable(details,remaining);examples.append(details)}
if(references.length>100)note(examples,'Showing the first 100 catalog references.');
if(rankingEntry){
rankingEntry.append(el('h3','Where did observed IDs appear?'));
note(rankingEntry,'Compare a retained TEST ranking with its observed IDs, then inspect the same example’s recorded context. '
+'These selected examples do not decompose the aggregate metric or represent all evaluation entities. The query’s TOP K and each retained list width are separate.');
const rankingView=el('div');
if(loaded.length){
let selected=loaded[0];
const drawRanking=()=>{rankingView.replaceChildren();const identity=sampleIdentity(selected.ref);
rankingView.append(el('h4',identity.length>100?identity.slice(0,100)+'…':identity));
identityNote(rankingView,'Selected entity identity',identity+' · Collector '+text(selected.ref.artifact_id));
identityNote(rankingView,'Recorded selection strategies','Retained strategies: '+(list(selected.ref.categories).join(', ')||'Unavailable'));
rankedOutput(rankingView,obj(obj(samples.outputs)[selected.ref.artifact_id]),selected.ref.artifact_id);
rankingView.append(action('Inspect this example’s context',()=>{inspectSample(selected.ref,selected.index);showPanel(2);
const heading=sampleView.querySelector('h3');heading.tabIndex=-1;heading.focus();}));enable();};
control(rankingEntry,'Loaded original entity',loaded.map(({ref,index})=>{
const identity=sampleIdentity(ref);return[String(index),(identity.length>100?identity.slice(0,100)+'…':identity)+' · collector '+text(ref.artifact_id)];}),value=>{
selected=loaded.find(entry=>String(entry.index)===value);drawRanking();});
rankingEntry.append(rankingView);drawRanking();
note(rankingEntry,'Loaded '+loaded.length+' of '+text(coverage.references_total??references.length)+' recorded references; selection strategies may overlap.');
}else note(rankingEntry,'No loaded examples are available for ranked-output comparison. Inspect recorded reference availability in Examples.');
}
sampleView.hidden=true;
const panels=[results,patterns,examples],navigation=el('nav',undefined,'kt-navigation');
navigation.setAttribute('aria-label','Training explanation sections');
const showPanel=(index,focus=false)=>{panels.forEach((panel,i)=>{panel.hidden=i!==index;
navigation.querySelectorAll(':scope>button')[i].setAttribute('aria-pressed',String(i===index));});
if(focus){const heading=panels[index].querySelector('h2');heading.tabIndex=-1;heading.focus();}};
for(const [index,label]of ['Results','Patterns','Examples'].entries()){
const button=action(label,()=>showPanel(index));panels[index].id=widgetId+'-panel-'+index;
button.setAttribute('aria-controls',panels[index].id);navigation.append(button);}
const theme=control(navigation,'Theme',[['auto','Automatic'],['light','Light'],['dark','Dark']],value=>{
if(value==='auto')delete host.dataset.kumoTheme;else host.dataset.kumoTheme=value;});
theme.value=host.dataset.kumoTheme||'auto';
const header=el('header',undefined,'kt-header');header.append(el('h2','Training explainability'));
if(context.query)header.append(queryNode);
const metadata=results.querySelector(':scope>.kt-note'),details=el('details');
details.append(el('summary','Run details'));if(metadata)details.append(metadata);
note(header,'Task: '+text(context.task_type||provenance.task_type)+' · Explanation split: TEST');
header.append(details);root.prepend(header,navigation);showPanel(0);
if(selectedFeature)chooseFeature(selectedFeature);
root.hidden=false;fallback.hidden=true;enable();
onDispose(()=>{root.querySelectorAll('button,select').forEach(node=>node.disabled=true);
root.hidden=true;fallback.hidden=false;});
"""


def viewer_spec(payload: dict) -> dict:
    """Define one view for HTML and optional widget transports."""
    return dict(class_name='kumo-training', style=_STYLE, script=_SCRIPT,
                body=static_summary(payload) +
                '<div class="kt-content" hidden=""></div>', data=payload)
