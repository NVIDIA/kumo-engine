# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training diagnostics composed with the shared notebook shell."""

from html import escape
from math import isfinite
from string import Template

from kumo.engine._widget import CSS_THEME


def static_summary(payload: dict) -> str:
    """Return escaped metrics and availability without selected-example data."""
    provenance = payload.get('provenance', {})
    parts = ['<div class="kumo-training-static"><h2>Training results</h2>',
             '<p>Recorded TEST diagnostics. Selected trial: ' +
             escape(str(provenance.get('selected_trial') or 'Unavailable')) +
             '. Explanation fit: ' +
             escape(str(provenance.get('explanation_trial') or 'Unavailable')) +
             '.</p><ul>']
    if provenance.get('selection_verified') is False:
        parts.append('<li>Selection unverified: ' + escape(str(
            provenance.get('selection_reason') or
            'Selection metadata is missing or invalid')) + '</li>')
    for row in payload.get('metrics', {}).get('rows', [])[:100]:
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
        parts.append('<li>' + escape(label) + '</li>')
    parts.append('</ul><h2>Selected examples</h2><p>Example values are excluded '
                 'from this summary. Rerun this cell in a trusted notebook '
                 'to activate diagnostics.</p></div>')
    return ''.join(parts)


_STYLE = Template(r"""
.kumo-training{container:kumo-training/inline-size;padding:16px;line-height:1.45;
overflow-wrap:anywhere;text-align:left}
.kumo-training *{box-sizing:border-box;min-width:0}
.kumo-training .kt-content{max-width:960px;width:100%}
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
.kumo-training .kt-table tr[aria-selected="true"]{background:$row_highlight}
.kumo-training .kt-action,.kumo-training select{font:13px/1.4 $font;
background:$background;color:$foreground;border:1px solid $surface_border;
border-radius:7px;padding:6px 10px;max-width:100%}
.kumo-training .kt-action{cursor:pointer;white-space:normal;text-align:left}
.kumo-training .kt-action:hover{background:$chrome_hover}
.kumo-training summary{cursor:pointer;color:$muted;font-size:13px;padding:6px 0}
.kumo-training :is(button,select,summary):focus-visible{outline:2px solid $focus;outline-offset:2px}
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
.kumo-training .kt-history-chart{display:block;width:100%;height:auto;max-width:600px;margin:12px 0}
.kumo-training .kt-list{padding-left:20px;margin:8px 0 12px}
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
const state=(parent,source,label)=>{source=obj(source);if(source.status!=='available')
note(parent,label+': '+text(source.status||'unavailable')+(source.reason?'. '+text(source.reason):''));};
const table=(parent,headers,rows,numeric=[],exact=false)=>{const node=el('table',undefined,'kt-table');
const head=el('thead'),tr=el('tr');for(const [i,h]of headers.entries()){
const th=el('th',h,numeric.includes(i)?'kt-number':'');th.scope='col';tr.append(th)}head.append(tr);node.append(head);
const body=el('tbody');for(const values of rows){const row=el('tr');values.forEach((value,i)=>{
const td=el('td',undefined,numeric.includes(i)?'kt-number':'');td.dataset.label=headers[i];
if(value instanceof Node){if(exact)td.textContent=value.textContent;else td.append(value)}
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
const provenance=obj(data.provenance),context=obj(data.context);
const baseMetric=row=>text(row.name).replace(/^(test_refit_|train_|val_|test_)/,'');
const results=section('Training results');
if(context.query)results.append(el('p',context.query));
note(results,[(context.task_type||provenance.task_type)&&('Task: '+text(context.task_type||provenance.task_type)),context.target&&('Target: '+text(context.target)),
context.run&&('Run: '+context.run),'Selected trial: '+text(provenance.selected_trial),
'Explanation fit: '+text(provenance.explanation_trial),'Explanation split: TEST',
'Refit on train + validation: '+(provenance.refitted_trainval===true?'yes':provenance.refitted_trainval===false?'no':'unknown')].filter(Boolean).join(' · '));
if(provenance.selection_verified===false)note(results,'Selection unverified: '+text(provenance.selection_reason));
const metricRows=list(obj(data.metrics).rows);state(results,data.metrics,'Metrics');
const fits=[...new Set(metricRows.map(row=>text(row.fit)))];
for(const fit of fits){results.append(el('h3','Measured results · '+fit));
const rows=metricRows.filter(row=>text(row.fit)===fit),names=[...new Set(rows.map(baseMetric))];
const splits=['train','val','test'].filter(split=>rows.some(row=>row.split===split));
for(const split of rows.map(row=>text(row.split)))if(!splits.includes(split))splits.push(split);
for(const split of splits){const group=rows.filter(row=>row.split===split);
const origins=[...new Set(group.map(row=>row.origin||'Unknown metric origin'))];
for(const origin of origins)note(results,split+': '+origin+(origins.length>1?' · '+group
.filter(row=>(row.origin||'Unknown metric origin')===origin).map(baseMetric).join(', '):''));}
table(results,['Metric',...splits],names.map(name=>[name,...splits.map(split=>{
const row=rows.find(item=>baseMetric(item)===name&&text(item.split)===split);return row?row.value:'Unavailable'})]),splits.map((_,i)=>i+1));}
if(provenance.selection_metric||obj(data.metrics).selection_metric)
note(results,'Selection metric: '+text(provenance.selection_metric||data.metrics.selection_metric)+
(provenance.selection_direction==='min'?' · Recorded optimization: minimize':provenance.selection_direction==='max'?' · Recorded optimization: maximize':''));
const histories=list(obj(data.history).rows);state(results,data.history,'Learning history');
if(histories.length){results.append(el('h3','Learning history'));
note(results,'Recorded epoch keys. Missing values remain unavailable; fits are shown separately.');
const metricName=name=>name.replace(/^(train_|val_|validation_)/,'');
const historyValue=(values,metric)=>{const entry=Object.entries(obj(values)).find(([name])=>metricName(name)===metric);return entry?.[1]};
const metrics=[...new Set(histories.flatMap(row=>[...Object.keys(obj(row.train_metrics)),...Object.keys(obj(row.validation_metrics))]).map(metricName))];
const historyView=el('div');
const chart=(parent,rows,metric)=>{const ordered=rows.filter(row=>Number.isFinite(Number(row.epoch)))
.slice().sort((a,b)=>Number(a.epoch)-Number(b.epoch));
const values=ordered.flatMap(row=>[historyValue(row.train_metrics,metric),historyValue(row.validation_metrics,metric)]).filter(finite);
if(!values.length||!ordered.length)return;
const low=Math.min(...values),high=Math.max(...values),first=Number(ordered[0].epoch),last=Number(ordered.at(-1).epoch);
const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
svg.setAttribute('viewBox','0 0 600 160');svg.setAttribute('role','img');svg.classList.add('kt-history-chart');
svg.setAttribute('aria-label','Learning history for '+metric+'. Train: circles, validation: squares. Exact values follow in the table.');
const x=epoch=>20+(Number(epoch)-first)/(last-first||1)*560,y=value=>140-(value-low)/(high-low||1)*120;
for(const [key,color,shape]of [['train_metrics','var(--kumo-accent)','circle'],['validation_metrics','var(--kumo-foreground)','rect']]){
let previous=null;for(const row of ordered){const value=historyValue(row[key],metric),epoch=Number(row.epoch);
if(!finite(value)){previous=null;continue;}
if(previous&&epoch-previous.epoch===1){const line=document.createElementNS(ns,'line');
for(const [name,attribute]of Object.entries({x1:x(previous.epoch),y1:y(previous.value),x2:x(epoch),y2:y(value),stroke:color,'stroke-width':2}))line.setAttribute(name,attribute);
if(shape==='rect')line.setAttribute('stroke-dasharray','5 4');svg.append(line)}
const point=document.createElementNS(ns,shape);for(const [name,attribute]of Object.entries(shape==='circle'?{cx:x(epoch),cy:y(value),r:4}:{x:x(epoch)-4,y:y(value)-4,width:8,height:8}))point.setAttribute(name,attribute);
point.setAttribute('fill',color);svg.append(point);previous={epoch,value};}}
parent.append(svg);note(parent,'Train: circles and solid lines. Validation: squares and dashed lines. Epoch '+first+' to '+last+
'; metric range '+number(low)+' to '+number(high)+'. Missing values and unrecorded epochs break the lines.');};
const drawHistory=metric=>{historyView.replaceChildren();
for(const identity of [...new Set(histories.map(row=>text(row.trial)+' / '+text(row.fit)))]){
historyView.append(el('h4',identity));chart(historyView,histories.filter(row=>text(row.trial)+' / '+text(row.fit)===identity),metric)}
table(historyView,['Trial / fit','Epoch','Train '+metric,'Validation '+metric],histories.map(row=>[
text(row.trial)+' / '+text(row.fit),text(row.epoch),historyValue(row.train_metrics,metric),historyValue(row.validation_metrics,metric)]),[1,2,3]);};
if(metrics.length){control(results,'History metric',metrics.map(name=>[name,name]),drawHistory);drawHistory(metrics[0]);results.append(historyView)}
else note(results,'No recorded epoch metric values.');}
const trials=list(obj(data.trials).rows);state(results,data.trials,'Trial comparison');
if(trials.length){results.append(el('h3','Recorded trials'));
const names=[...new Set(trials.flatMap(row=>Object.keys(obj(row.comparison_metrics??row.final_validation_metrics))))].slice(0,10);
if(obj(data.trials).total!==undefined)note(results,'Recorded completed trials: '+text(data.trials.completed)+' / '+text(data.trials.total));
table(results,['Trial / fit','Selection / progress',...names],trials.map(row=>[
text(row.trial)+' / '+text(row.fit),(provenance.selection_verified!==true?'Selection unverified':row.selected?'Selected by recorded winner':'Not selected')+' · '+
(row.status||row.reason||('Epochs: '+text(row.num_epochs_completed)+' · Device: '+text(row.accelerator)))+
(row.comparison_source==='last_recorded_epoch'?' · Validation from last recorded epoch '+text(row.comparison_epoch):row.comparison_source==='final_validation_metrics'?' · Selected-checkpoint validation':' · Validation origin unverified'),
...names.map(name=>obj(row.comparison_metrics??row.final_validation_metrics)[name])]),names.map((_,i)=>i+2));
note(results,'Missing progress is not evidence that a trial failed.');}
const patterns=section('Recorded patterns');patterns.append(el('h3','Global input sensitivity'));
note(patterns,'Unsigned recorded gradient magnitude, normalized across all columns when the total is positive. '
+'These scores describe sensitivity, not causal effects or prediction probabilities.');
state(patterns,data.gradient,'Global sensitivity');
const gradients=list(obj(data.gradient).rows), valid=gradients.filter(row=>finite(row.score)&&row.score>=0)
.sort((a,b)=>b.score-a.score||text(a.table).localeCompare(text(b.table))||text(a.column).localeCompare(text(b.column)));
if(valid.length!==gradients.length)note(patterns,'Invalid or negative gradient scores are excluded from the unsigned ranking.');
note(patterns,'Highest recorded gradient scores; showing '+Math.min(20,valid.length)+' of '+text(obj(data.gradient).total??gradients.length)+' columns. The visible subset is not renormalized.');
const cohorts=list(obj(data.cohorts).rows);
let cohortView,cohortButtons=[];
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
if(valid.length)table(patterns,['Input column','Recorded score','Cohort detail'],valid.slice(0,20).map(row=>{
const score=el('div',undefined,'kt-score'),track=el('div',undefined,'kt-track'),bar=el('div',undefined,'kt-bar');
bar.style.width=Math.min(100,row.score*100)+'%';track.setAttribute('aria-hidden','true');track.append(bar);
score.append(track,el('span',number(row.score),'kt-value'));
const matches=cohorts.map((c,i)=>c.table_name===row.table&&c.col_name===row.column?i:-1).filter(i=>i>=0);
const choices=el('div');for(const index of matches)choices.append(action('Inspect cohorts · hop '+text(cohorts[index].hop),()=>inspectCohort(index)));
if(!matches.length)choices.textContent='No matching cohort';return[text(row.table)+'.'+text(row.column),score,choices];}));
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
note(examples,'Preloaded example identities and values may be saved in notebook HTML or widget state. '
+'Inspect other retained graphs in Python with explanations.sample(collector_id).');
const sampleView=el('div',undefined,'kt-detail'),sampleButtons=[];
const sampleDetailRow=el('tr'),sampleDetailCell=el('td');sampleDetailCell.colSpan=3;
sampleDetailCell.append(sampleView);sampleDetailRow.append(sampleDetailCell);
const inspectSample=(ref,index)=>{sampleView.replaceChildren();sampleView.hidden=false;
sampleButtons.find(item=>item.rowIndex===index)?.button.closest('tr').after(sampleDetailRow);
sampleView.append(el('h3','Collector '+text(ref.artifact_id)));
note(sampleView,'Recorded original IDs: '+text(Object.keys(obj(ref.original_ids)).length?ref.original_ids:'Unavailable')
+' · Strategies: '+(list(ref.categories).join(', ')||'Unavailable'));
const output=obj(obj(samples.outputs)[ref.artifact_id]);
sampleView.append(el('h4','Prediction and observed target'));
state(sampleView,output,'Output context');
if(output.status==='available'||output.status==='partial'){
const binary=output.task_type==='binary_classification';
note(sampleView,binary?'Probability of the positive class; observed target: 0 = negative class, 1 = positive class.':'Values use recorded target units.');
table(sampleView,[binary?'Positive-class probability (0 to 1)':'Prediction','Observed target'],[[output.prediction,output.target]],[0,1]);}
const body=obj(bodies[ref.artifact_id]),store=obj(body.subgraph_store),features=obj(obj(body.feature_store).features),nodes=obj(store.nodes);
const rootNode=obj(nodes[store.root_id]);
note(sampleView,'Root node: '+text(store.root_id)+' · Original entity: '+text(rootNode.pkey)
+' · Seed time: '+text(store.seed_time));
if(!body.subgraph_store)note(sampleView,'Relational structure unavailable.');
const nodeRows=Object.entries(nodes);if(nodeRows.length){sampleView.append(el('h4','Recorded nodes'));
table(sampleView,['Node','Identity / aggregate','Table membership'],nodeRows.slice(0,80).map(([id,node])=>[
id,node.count!==undefined?'Aggregate: '+text(node.count)+' nodes':text(node.pkey_name)+': '+text(node.pkey),
Object.entries(obj(store.tables)).filter(([,ids])=>list(ids).some(value=>String(value)===id)).map(([name])=>name).join(', ')||'Unavailable']));
if(nodeRows.length>80)note(sampleView,'Showing 80 of '+nodeRows.length+' recorded nodes; complete loaded body is retained.');}
const relations=[];for(const [from,kinds]of Object.entries(obj(store.edges)))for(const [kind,tos]of Object.entries(obj(kinds)))
for(const to of list(tos))relations.push([from,kind,text(to)]);
if(relations.length){sampleView.append(el('h4','Recorded relations'));table(sampleView,['From node','Relation','To node'],relations.slice(0,80));
if(relations.length>80)note(sampleView,'Showing 80 of '+relations.length+' recorded relations.');}
else note(sampleView,'No recorded relations in this loaded example.');
sampleView.append(el('h4','Node features'));
note(sampleView,'Local scores are per-example normalized sensitivity. Aggregate scores may exceed one; '
+'unavailable scores differ from zero and are separate from global scores.');
const cells=[];for(const [id,columns]of Object.entries(features))for(const [name,feature]of Object.entries(obj(columns)))
cells.push([id,text(feature.table_name)+'.'+text(feature.name??name),text(feature.value),feature.score]);
if(cells.length)table(sampleView,['Node','Feature','Recorded value','Local score'],cells.slice(0,200),[3]);
else note(sampleView,'No recorded feature values.');if(cells.length>200)note(sampleView,'Showing 200 of '+cells.length+' recorded feature cells.');
sampleButtons.forEach(({button,rowIndex})=>{button.setAttribute('aria-pressed',String(rowIndex===index));
button.closest('tr').setAttribute('aria-selected',String(rowIndex===index));});};
const catalogRows=references.slice(0,100).map((ref,index)=>({ref,index}));
const referenceTable=(parent,entries)=>table(parent,['Collector / original IDs','Recorded strategies','Availability / detail'],entries.map(({ref,index})=>{
const identity=el('div'),original=[...new Set(Object.values(obj(ref.original_ids)).map(text))];
identity.append(el('div','Collector '+text(ref.artifact_id)),el('div',original.length?'Original ID: '+original.join(', '):'Original IDs unavailable','kt-note'));
let detail;if(Object.hasOwn(bodies,ref.artifact_id)){detail=action('Inspect example',()=>inspectSample(ref,index));sampleButtons.push({button:detail,rowIndex:index});}
else detail=el('span',ref.reason||ref.status||(ref.available?'Not loaded':'Graph file missing'));
return[identity,list(ref.categories).join(', ')||'Unavailable',detail];}));

const loaded=catalogRows.filter(({ref})=>Object.hasOwn(bodies,ref.artifact_id));
const remaining=catalogRows.filter(({ref})=>!Object.hasOwn(bodies,ref.artifact_id));
if(loaded.length){examples.append(el('h3','Loaded examples'));referenceTable(examples,loaded)}
if(remaining.length){const details=el('details');details.append(el('summary','Other recorded references ('+remaining.length+')'));
referenceTable(details,remaining);examples.append(details)}
if(references.length>100)note(examples,'Showing the first 100 catalog references.');
sampleView.hidden=true;
root.hidden=false;fallback.hidden=true;enable();
onDispose(()=>{root.querySelectorAll('button,select').forEach(node=>node.disabled=true);
root.hidden=true;fallback.hidden=false;});
"""


def viewer_spec(payload: dict) -> dict:
    """Define one view for HTML and optional widget transports."""
    return dict(class_name='kumo-training', style=_STYLE, script=_SCRIPT,
                body=static_summary(payload) +
                '<div class="kt-content" hidden=""></div>', data=payload)
