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
        '<p>Static summary. Run this cell in a trusted notebook for the '
        'interactive view.</p>',
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
    notices = []
    if provenance.get('generation_status') in ('disabled', 'failed'):
        notices.append('Explanation generation: ' +
                       escape(str(provenance['generation_status'])))
    if (provenance.get('generation_failure')
            and provenance.get('generation_status') != 'failed'):
        notices.append(escape(str(provenance['generation_failure'])))
    if provenance.get('selection_verified') is False:
        reason = (provenance.get('selection_reason')
                  or 'Selection metadata is missing or invalid')
        notices.append('Selection unverified: ' + escape(str(reason)))
    if notices:
        items = ''.join(f'<li>{item}</li>' for item in notices)
        parts.append(f'<ul>{items}</ul>')
    parts.append(f'<details><summary>Recorded metrics ({len(metrics)})'
                 '</summary><ul>')
    for row in metrics[:100]:
        value = row.get('value')
        number = (str(value) if isinstance(value, (int, float))
                  and not isinstance(value, bool) and isfinite(value)
                  else 'Unavailable')
        label = ' / '.join(str(row.get(key) or 'Unavailable')
                           for key in ('fit', 'split', 'name'))
        label += ' / ' + str(row.get('origin') or 'Unknown metric origin')
        parts.append('<li>' + escape(label) + ': ' + escape(number) + '</li>')
    parts.append('</ul></details><h2>Recorded patterns</h2><ul>')
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
.kumo-training{container:kumo-training/inline-size;padding:20px 24px 24px;
border:1px solid $surface_border;border-radius:12px;line-height:1.45;text-align:left}
.kumo-training *{box-sizing:border-box;min-width:0}
.kumo-training .kt-content{width:100%;max-width:1120px;margin:0 auto;overflow-wrap:anywhere}
.kumo-training :is(.kt-h2,.kt-h3,.kt-h4,.kt-p,.kt-table,.kt-dl){margin:0;padding:0;
font-style:normal;letter-spacing:normal;text-align:left;text-transform:none}
.kumo-training .kt-h2{font:700 20px/1.25 $font;color:$foreground}
.kumo-training .kt-h3{font:600 15px/1.35 $font;color:$foreground}
.kumo-training .kt-h4{font:600 13px/1.35 $font;color:$foreground}
.kumo-training .kt-p{font-size:13px;line-height:1.5}
.kumo-training .kt-muted{color:$muted}
.kumo-training .kt-small{font-size:12px}
.kumo-training .kt-num{font-variant-numeric:tabular-nums;white-space:nowrap}
.kumo-training .kt-sr{position:absolute;width:1px;height:1px;overflow:hidden;
clip-path:inset(50%);white-space:nowrap}
.kumo-training .kt-missing{color:$muted}
.kumo-training .kt-stack{display:grid;gap:16px;align-content:start}
.kumo-training .kt-row{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px}
.kumo-training .kt-spread{display:flex;flex-wrap:wrap;align-items:baseline;
justify-content:space-between;gap:4px 16px}

.kumo-training .kt-head{display:flex;flex-wrap:wrap;align-items:flex-start;
justify-content:space-between;gap:12px 24px;margin:0 0 12px}
.kumo-training .kt-head-main{flex:1 1 420px;display:grid;gap:4px}
.kumo-training .kt-tools{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;padding-top:2px}
.kumo-training .kt-query{margin:6px 0 0;padding:6px 10px;border:1px solid $surface_border;
border-radius:6px;background:$surface;color:$foreground;
font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
.kumo-training .kt-query-text{display:block}
.kumo-training .kt-query-text.kt-clamped{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.kumo-training .kt-notice{display:flex;flex-wrap:wrap;gap:2px 8px;align-items:baseline;margin:0 0 12px;padding:8px 12px;
border:1px solid $surface_border;border-left:3px solid $foreground;border-radius:6px;
background:$surface;font-size:13px}
.kumo-training .kt-notice>strong{flex:none}

.kumo-training :is(button,select,input){font:inherit;color:inherit;margin:0}
.kumo-training .kt-field{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:$muted}
.kumo-training .kt-select{height:30px;max-width:100%;padding:0 6px;font-size:13px;color:$foreground;
background:$background;border:1px solid $card_border;border-radius:6px}
.kumo-training .kt-check{display:inline-flex;align-items:center;gap:6px;font-size:13px;
color:$foreground;cursor:pointer}
.kumo-training .kt-check input{width:15px;height:15px;accent-color:$accent;cursor:pointer}
.kumo-training .kt-btn{display:inline-flex;align-items:center;gap:6px;min-height:30px;max-width:100%;padding:4px 10px;
font-size:13px;line-height:1.3;color:$foreground;background:$background;border:1px solid $card_border;
border-radius:6px;cursor:pointer;text-align:left}
.kumo-training .kt-btn:hover:not(:disabled){background:$chrome_hover}
.kumo-training .kt-btn:disabled{opacity:.45;cursor:default}
.kumo-training .kt-link{padding:2px 0;font-size:13px;font-weight:600;color:$accent;background:none;
border:0;cursor:pointer;text-align:left}
.kumo-training .kt-link:hover{text-decoration:underline}
.kumo-training :is(button,select,input,summary,[tabindex]):focus-visible{
outline:2px solid $focus;outline-offset:2px}

.kumo-training .kt-tabs{display:flex;gap:2px;margin:0 0 20px;border-bottom:1px solid $surface_border;
overflow-x:auto;scrollbar-width:none}
.kumo-training .kt-tab{flex:none;margin:0 0 -1px;padding:10px 14px;font-size:14px;font-weight:600;
color:$muted;background:none;border:0;border-bottom:2px solid transparent;
border-radius:6px 6px 0 0;cursor:pointer}
.kumo-training .kt-tab:hover{color:$foreground;background:$chrome_hover}
.kumo-training .kt-tab[aria-selected="true"]{color:$accent;border-bottom-color:$accent}
.kumo-training .kt-tab:focus-visible{outline-offset:-2px}

.kumo-training .kt-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.kumo-training .kt-card{display:flex;flex-direction:column;gap:12px;padding:16px;
border:1px solid $surface_border;border-radius:10px}
.kumo-training .kt-card>.kt-link{align-self:flex-start}
.kumo-training .kt-metric{display:grid;gap:4px}
.kumo-training .kt-metric-name{font-size:13px;font-weight:600;color:$muted}
.kumo-training .kt-metric-value{font:600 34px/1.15 $font;font-variant-numeric:tabular-nums;color:$foreground}
.kumo-training .kt-facts{display:grid;gap:8px;margin:0;padding:0;list-style:none}
.kumo-training .kt-facts>li{display:grid;gap:2px;margin:0;padding:0}
.kumo-training .kt-fact-value{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}

.kumo-training .kt-bar{position:relative;display:block;height:8px;border-radius:4px;
background:$surface;overflow:hidden}
.kumo-training .kt-bar>span{position:absolute;inset:0 auto 0 0;border-radius:4px;background:$accent}
.kumo-training .kt-bar.kt-thin{height:4px;border-radius:2px}

.kumo-training .kt-split{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);
gap:24px;align-items:start}
.kumo-training .kt-detail{position:sticky;top:8px;display:grid;gap:20px;padding:16px 18px;
border:1px solid $surface_border;border-radius:10px}
.kumo-training .kt-section{display:grid;gap:10px}
.kumo-training .kt-section-head{display:flex;flex-wrap:wrap;align-items:center;
justify-content:space-between;gap:8px 16px}

.kumo-training .kt-list{display:grid;gap:2px;margin:0;padding:0}
.kumo-training .kt-group-label{padding:12px 10px 4px;font-size:12px;font-weight:600;color:$muted}
.kumo-training .kt-option{display:grid;align-items:center;gap:4px 12px;padding:8px 10px;
border-radius:6px;color:$foreground;cursor:pointer;text-align:left}
.kumo-training .kt-option:hover{background:$chrome_hover}
.kumo-training .kt-option[aria-selected="true"]{background:$row_highlight;box-shadow:inset 3px 0 $accent}
.kumo-training .kt-option:focus-visible{outline-offset:-2px}
.kumo-training .kt-feature{grid-template-columns:minmax(0,1fr) minmax(56px,34%) 48px}
.kumo-training .kt-feature .kt-num{text-align:right}
.kumo-training .kt-label{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kumo-training .kt-label>.kt-muted{font-weight:400}
.kumo-training .kt-example{grid-template-columns:minmax(0,1fr) auto}
.kumo-training .kt-example>.kt-outcome{text-align:right}
.kumo-training .kt-list-head{display:grid;gap:4px;margin-bottom:8px}
.kumo-training .kt-col-head{display:grid;gap:12px;padding:0 10px 4px;font-size:12px;color:$muted}
.kumo-training .kt-col-head>:last-child{text-align:right}
.kumo-training .kt-hit{color:$accent;font-weight:600}

.kumo-training .kt-table{width:auto;max-width:100%;border-collapse:collapse;font-size:13px;table-layout:auto;
color:$foreground;background:none;border:0}
.kumo-training .kt-table :is(th,td){padding:6px 16px 6px 10px;border:0;border-bottom:1px solid $surface_border;
text-align:left;vertical-align:baseline;line-height:1.4;white-space:normal}
.kumo-training .kt-table th{font-size:12px;font-weight:600;color:$muted}
.kumo-training .kt-table :is(th,td):first-child{padding-left:0}
.kumo-training .kt-table :is(th,td):last-child{padding-right:0}
.kumo-training .kt-table :is(th,td).kt-num{text-align:right}
.kumo-training .kt-table tbody tr:nth-child(n){background:transparent}
.kumo-training .kt-table tr.kt-strong td{font-weight:600}
.kumo-training .kt-table tr[aria-selected="true"] td{background:$row_highlight}
.kumo-training .kt-table .kt-bar{width:72px;display:inline-block;vertical-align:middle;margin-right:8px}
.kumo-training .kt-scroll{overflow-x:auto}
.kumo-training .kt-mark{margin-left:2px;color:$muted}

.kumo-training details{margin:0}
.kumo-training details>summary{width:fit-content;padding:2px 0;font-size:13px;font-weight:600;
color:$foreground;cursor:pointer}
.kumo-training details[open]>summary{margin-bottom:8px}
.kumo-training .kt-dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:6px 16px;font-size:13px}

.kumo-training .kt-axis{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:$muted;
font-variant-numeric:tabular-nums}
.kumo-training .kt-track{position:relative;height:24px}
.kumo-training .kt-track:before{content:"";position:absolute;left:0;right:0;top:50%;
border-top:1px solid $surface_border}
.kumo-training .kt-gap{position:absolute;top:50%;height:2px;margin-top:-1px;background:$edge}
.kumo-training .kt-dot,.kumo-training .kt-diamond{position:absolute;top:50%;width:10px;height:10px;
margin:-5px 0 0 -5px}
.kumo-training .kt-paired>.kt-dot{margin-top:-10px}
.kumo-training .kt-paired>.kt-diamond{margin-top:0}
.kumo-training .kt-dot{border-radius:50%;background:$accent;box-shadow:0 0 0 2px $background}
.kumo-training .kt-diamond{background:$foreground;transform:rotate(45deg) scale(.86);
box-shadow:0 0 0 2px $background}
.kumo-training .kt-legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:$muted}
.kumo-training .kt-legend .kt-key{color:$accent}
.kumo-training .kt-legend .kt-key-observed{color:$foreground}

.kumo-training .kt-cohort{display:grid;gap:0}
.kumo-training .kt-cohort-row{display:grid;grid-template-columns:minmax(72px,26%) minmax(0,1fr) 58px 58px 64px;
align-items:center;gap:8px 12px;padding:5px 0;border-bottom:1px solid $surface_border;font-size:13px}
.kumo-training .kt-cohort-row.kt-cohort-head{border-bottom:0;padding:0 0 4px;font-size:12px;color:$muted}
.kumo-training .kt-cohort-row>.kt-num{text-align:right}
.kumo-training .kt-share{display:grid;gap:3px;text-align:right}

.kumo-training .kt-ranks{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:6px;
margin:0;padding:0;list-style:none}
.kumo-training .kt-rank{display:grid;gap:2px;margin:0;padding:4px 6px;border-radius:4px;font-size:12px;color:$muted}
.kumo-training .kt-rank>strong{font-size:14px;color:$foreground;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kumo-training .kt-rank.kt-match{background:$row_highlight}
.kumo-training .kt-rank.kt-match>span{color:$accent;font-weight:600}
.kumo-training .kt-compare{display:grid;grid-template-columns:minmax(0,max-content) minmax(0,max-content) minmax(0,1fr);
align-items:center;gap:8px 14px;font-size:13px}
.kumo-training .kt-compare .kt-num{text-align:right;font-weight:600}
.kumo-training .kt-classes{display:grid;grid-template-columns:minmax(0,max-content) minmax(80px,1fr) 52px auto;
align-items:center;gap:6px 12px;font-size:13px}
.kumo-training .kt-classes .kt-num{text-align:right}

.kumo-training .kt-chart-box{position:relative;width:100%}
.kumo-training .kt-chart{display:block;width:100%;overflow:visible}
.kumo-training .kt-chart text{font:11px $font;fill:$muted}
.kumo-training .kt-chart .kt-gridline{stroke:$surface_border;stroke-width:1}
.kumo-training .kt-chart .kt-cursor{stroke:$muted;stroke-dasharray:2 3}
.kumo-training .kt-chart-frame{display:grid;gap:6px}
.kumo-training .kt-readout{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;min-height:19px}
.kumo-training .kt-readout b{font-weight:600}
.kumo-training .kt-plot{border-radius:6px}

.kumo-training .kt-records{display:grid;gap:12px}
.kumo-training .kt-record-group{display:grid;gap:6px;padding-left:12px;border-left:2px solid $surface_border}
.kumo-training .kt-record-cell{display:grid;gap:3px}
.kumo-training .kt-record-cell .kt-bar{width:100%;max-width:96px}

@container kumo-training (max-width:760px){
.kumo-training .kt-split{grid-template-columns:minmax(0,1fr);gap:16px}
.kumo-training .kt-detail{position:static}
.kumo-training .kt-grid{grid-template-columns:minmax(0,1fr)}
.kumo-training .kt-ranks{grid-template-columns:repeat(5,minmax(0,1fr))}}
@container kumo-training (max-width:480px){
.kumo-training{padding:14px 12px 16px}
.kumo-training .kt-detail{padding:12px}
.kumo-training .kt-tab{padding:10px 10px}
.kumo-training .kt-cohort-row{grid-template-columns:minmax(0,1fr) 52px 52px;
grid-template-areas:"label label share" "track pred obs"}
.kumo-training .kt-cohort-row>:nth-child(1){grid-area:label}
.kumo-training .kt-cohort-row>:nth-child(2){grid-area:track}
.kumo-training .kt-cohort-row>:nth-child(3){grid-area:pred}
.kumo-training .kt-cohort-row>:nth-child(4){grid-area:obs}
.kumo-training .kt-cohort-row>:nth-child(5){grid-area:share}
.kumo-training .kt-cohort-head>:nth-child(2){visibility:hidden}
.kumo-training .kt-ranks{grid-template-columns:repeat(4,minmax(0,1fr))}
.kumo-training .kt-classes{grid-template-columns:minmax(0,1fr) 52px auto}
.kumo-training .kt-classes .kt-bar{grid-column:1/-1;grid-row:span 1}
.kumo-training .kt-dl{grid-template-columns:minmax(0,1fr)}
.kumo-training .kt-dl>:nth-child(2n){margin-bottom:6px}}
.kumo-training .kt-related{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0;
padding-top:12px;border-top:1px solid $surface_border}
.kumo-training .kt-related-value{font-size:15px;font-weight:600}
.kumo-training .kt-strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px;margin:0;padding:0;list-style:none}
.kumo-training .kt-strip>li{margin:0;padding:0}
.kumo-training .kt-strip-cell{display:grid;gap:1px;width:100%;padding:5px 4px;border:1px solid $surface_border;
border-radius:6px;background:none;color:$foreground;text-align:center;cursor:pointer}
.kumo-training .kt-strip-cell:hover{background:$chrome_hover}
.kumo-training .kt-strip-cell>strong{font-weight:600}
.kumo-training .kt-strip-cell.kt-match>strong{color:$accent}
.kumo-training .kt-card.kt-wide{grid-column:1/-1}
.kumo-training .kt-gaps{display:grid;gap:4px;margin:16px 0 0;padding:0;list-style:none;font-size:13px;color:$muted}
.kumo-training .kt-gaps b{color:$foreground;font-weight:600}
.kumo-training .kt-code{padding:1px 4px;border-radius:4px;background:$surface;color:$foreground;
font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}
.kumo-training .kt-dotrow{display:grid;grid-template-columns:132px minmax(0,1fr);align-items:center;gap:10px}
.kumo-training button.kt-option{width:100%;margin:0;font:inherit;background:none;border:0}
.kumo-training .kt-btn,.kumo-training .kt-link{justify-self:start;align-self:flex-start}
.kumo-training .kt-settings .kt-table,.kumo-training .kt-detail .kt-table{max-width:640px}
.kumo-training .kt-settings .kt-table td.kt-num{white-space:normal;overflow-wrap:anywhere}
.kumo-training .kt-stack>.kt-fit{max-width:860px}
@container kumo-training (max-width:480px){
.kumo-training .kt-strip{grid-template-columns:repeat(4,minmax(0,1fr))}
.kumo-training .kt-dotrow{grid-template-columns:minmax(0,1fr)}
.kumo-training .kt-related{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kumo-training :is(.kt-section,.kt-stack,.kt-card,.kt-list-head){align-content:start}
.kumo-training .kt-narrow{display:grid;gap:2px;max-width:640px}
.kumo-training .kt-scores{display:grid;max-width:640px}
.kumo-training .kt-score-row{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr) auto;gap:2px 16px;
align-items:start;padding:7px 0;border-bottom:1px solid $surface_border;font-size:13px}
.kumo-training .kt-score-head{padding-top:0;font-size:12px;font-weight:600;color:$muted}
.kumo-training .kt-score-row>:last-child{text-align:right}
.kumo-training .kt-score{display:flex;align-items:center;justify-content:flex-end;gap:8px;min-width:112px}
.kumo-training .kt-score .kt-bar{flex:none;width:64px}
.kumo-training .kt-strip-cell>strong{white-space:nowrap}
@container kumo-training (max-width:480px){
.kumo-training .kt-score-row{grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"name name" "value score"}
.kumo-training .kt-score-row>:nth-child(1){grid-area:name}
.kumo-training .kt-score-row>:nth-child(2){grid-area:value}
.kumo-training .kt-score-row>:nth-child(3){grid-area:score}
.kumo-training .kt-score-head>:nth-child(1){display:none}
.kumo-training .kt-tabs{flex-wrap:wrap;overflow:visible}
.kumo-training .kt-tab{padding:8px 4px;font-size:12.5px}
.kumo-training .kt-tab .kt-count{display:none}
.kumo-training .kt-label{white-space:normal}
.kumo-training .kt-feature{grid-template-columns:minmax(0,1fr) 48px;grid-template-areas:"name name" "bar value";row-gap:6px}
.kumo-training .kt-feature>:nth-child(1){grid-area:name}
.kumo-training .kt-feature>:nth-child(2){grid-area:bar;align-self:center}
.kumo-training .kt-feature>:nth-child(3){grid-area:value}}
.kumo-training .kt-fname{display:flex;align-items:baseline;min-width:0}
.kumo-training .kt-ftable{flex:0 4 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kumo-training .kt-fcol{flex:0 1 auto;min-width:0;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
overflow:hidden;overflow-wrap:anywhere}
.kumo-training .kt-ranks.kt-ranks-long{grid-template-columns:repeat(2,minmax(0,1fr))}
.kumo-training .kt-ranks-long>.kt-rank{grid-template-columns:3.2em minmax(0,1fr);align-items:baseline}
@container kumo-training (max-width:480px){.kumo-training .kt-ranks.kt-ranks-long{grid-template-columns:minmax(0,1fr)}}
.kumo-training .kt-rank>span{white-space:nowrap}
.kumo-training .kumo-training-static{display:grid;gap:8px;font-size:13px}
.kumo-training .kumo-training-static :is(h2,p,ul){margin:0}
.kumo-training .kumo-training-static h2{font:600 15px/1.35 $font}
.kumo-training .kumo-training-static ul{padding-left:18px}
.kumo-training .kt-axis>span{white-space:nowrap}
.kumo-training .kt-cohort-row>.kt-label{white-space:normal;overflow:visible;text-overflow:clip}
.kumo-training.kumo-widget .kt-content :is(ul,ol){margin:0;padding:0;list-style:none}
.kumo-training.kumo-widget .kt-content :is(ul,ol)>li{margin:0;padding:0}
.kumo-training.kumo-widget .kt-table tbody tr:hover{background:transparent}
.kumo-training .kt-related-value .kt-num,.kumo-training .kt-metric-value .kt-num{white-space:normal;overflow-wrap:anywhere}
.kumo-training .kt-tab:hover .kt-count{color:$foreground}
.kumo-training .kt-content .kt-gaps{margin-top:16px}
.kumo-training .kt-scores-num .kt-score-row>:nth-child(2){text-align:right}
.kumo-training .kt-score-row{align-items:baseline}
.kumo-training .kt-score-row .kt-score{align-self:center}
.kumo-training .kt-link.kt-inline{padding:0;font-weight:500;color:$foreground;text-decoration:underline;
text-decoration-color:$surface_border;text-underline-offset:3px}
.kumo-training .kt-link.kt-inline:hover{text-decoration-color:currentColor}
.kumo-training .kt-kv{display:grid;grid-template-columns:max-content max-content;gap:0;font-size:13px;max-width:100%}
.kumo-training .kt-kv.kt-kv-3{grid-template-columns:max-content max-content max-content}
.kumo-training .kt-kv-row{display:contents}
.kumo-training .kt-kv-row>span{padding:6px 32px 6px 0;border-bottom:1px solid $surface_border;overflow-wrap:anywhere}
.kumo-training .kt-kv-row>span:last-child{padding-right:0}
.kumo-training .kt-kv-head>span{font-size:12px;font-weight:600;color:$muted}
.kumo-training .kt-kv-row.kt-strong>span{font-weight:600}
.kumo-training .kt-training{max-width:860px}
.kumo-training .kt-exact .kt-num,.kumo-training .kt-content.kt-exact .kt-num{white-space:normal;overflow-wrap:anywhere}
.kumo-training .kt-content.kt-exact .kt-label{white-space:normal;overflow:visible;text-overflow:clip}
.kumo-training .kt-content.kt-exact .kt-cohort-row{grid-template-columns:minmax(72px,22%) minmax(0,1fr) minmax(0,96px) minmax(0,96px) 64px}
.kumo-training .kt-content.kt-exact .kt-feature{grid-template-columns:minmax(0,1fr) minmax(48px,24%) minmax(48px,auto)}
.kumo-training .kt-content.kt-exact .kt-example>.kt-outcome{max-width:50%}
@media (min-height:480px){@container kumo-training (min-width:761px){
.kumo-training .kt-content:not(.kt-framed) .kt-detail{max-height:calc(100vh - 16px);overflow:auto}}}
@container kumo-training (max-width:480px){
.kumo-training .kt-kv,.kumo-training .kt-kv.kt-kv-3{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
.kumo-training .kt-kv.kt-kv-3{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr) minmax(0,1fr)}
.kumo-training .kt-content.kt-exact .kt-cohort-row{grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr)}}
""").substitute(CSS_THEME)

_SCRIPT = r"""
const content=host.querySelector('.kt-content'),fallback=host.querySelector('.kumo-training-static');
const NS='http://www.w3.org/2000/svg';
const list=value=>Array.isArray(value)?value:[];
const obj=value=>value&&typeof value==='object'&&!Array.isArray(value)?value:{};
const has=(value,key)=>Object.prototype.hasOwnProperty.call(value,key);
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const str=value=>value===null||value===undefined?'':typeof value==='object'?JSON.stringify(value):String(value);
const plural=(count,one,many)=>count+' '+(count===1?one:many||one+'s');
let uid=0;
const nextId=name=>widgetId+'-'+name+'-'+(++uid);
const append=(node,...children)=>{
  for(const child of children.flat(Infinity))
    if(child!==null&&child!==undefined&&child!==false&&child!=='')node.append(child instanceof Node?child:String(child));
  return node;
};
const h=(tag,props,...children)=>{
  const node=document.createElement(tag);
  for(const [key,value] of Object.entries(props||{})){
    if(value===null||value===undefined||value===false)continue;
    if(key==='class')node.className=value;
    else if(key==='text')node.textContent=value;
    else if(key==='style')for(const [name,item] of Object.entries(value))node.style.setProperty(name,item);
    else if(key.startsWith('on'))node.addEventListener(key.slice(2),value,{signal});
    else node.setAttribute(key,value===true?'':String(value));
  }
  return append(node,children);
};
const sv=(tag,attributes)=>{
  const node=document.createElementNS(NS,tag);
  for(const [key,value] of Object.entries(attributes||{}))
    if(value!==null&&value!==undefined)node.setAttribute(key,String(value));
  return node;
};
const svText=(attributes,text)=>{const node=sv('text',attributes);node.textContent=text;return node};

// View state survives tab changes; each view keeps its own copy.
const S={tab:0,exact:false,feature:null,allFeatures:false,hop:new Map(),cls:new Map(),
  example:0,allLocal:false,metric:null,epochs:new Map(),trial:null};

const fmt=(value,places)=>{
  if(!finite(value))return null;
  if(S.exact)return String(value);
  if(value===0)return '0';
  const size=Math.abs(value);
  if(size>=1e6)return value.toExponential(2).replace('e+','e');
  if(size>=1e3)return Math.round(value).toLocaleString('en-US');
  const digits=places??(size>=100?1:size>=10?2:size>=1?3:4);
  if(size<0.5*10**-digits)return value.toExponential(2).replace('e+','e');
  return value.toFixed(digits);
};
// Recorded data values drop padding zeros: 70, 72.5, -0.5508.
const dataValue=value=>{if(S.exact||Number.isInteger(value)&&Math.abs(value)<1e15)return String(value);
  const text=fmt(value);return /e/.test(text)?text:text.replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'')};
// Shared decimal places keep a column of comparable numbers aligned.
const placesFor=(values,significant=4)=>{const size=Math.max(0,...values.filter(finite).map(Math.abs));
  return size>0&&size<1e3?Math.max(0,Math.min(6,significant-1-Math.floor(Math.log10(size)))):undefined};
const prob=value=>{
  if(!(finite(value)&&value>=0&&value<=1))return fmt(value,3);
  if(S.exact)return String(value);
  if(value===0||value===1)return value.toFixed(3);
  return value<0.001?'<0.001':value>0.999?'>0.999':value.toFixed(3)};
const numProb=(value,label)=>{const text=prob(value);return text===null?missing(label):h('span',{class:'kt-num',text})};
const missing=(label='not recorded')=>h('span',{class:'kt-missing'},
  h('span',{'aria-hidden':'true',text:'—'}),h('span',{class:'kt-sr',text:label}));
const num=(value,places,label)=>{const text=fmt(value,places);
  return text===null?missing(label):h('span',{class:'kt-num',text})};
const signed=(value,places)=>{const text=fmt(value,places);return text===null?null:value>0?'+'+text:text};
const axisLabel=value=>value===0?'0':Math.abs(value)>=1e4||Math.abs(value)<1e-3?value.toExponential(1).replace('e+','e'):
  String(Number(value.toPrecision(3)));
const compareValues=(a,b)=>finite(a)&&finite(b)?a-b:finite(a)?-1:finite(b)?1:str(a).localeCompare(str(b),undefined,{numeric:true});
const rangeLabel=label=>str(label).replace(/ - /g,' to ');
const percent=value=>finite(value)?(S.exact?String(value*100):(value*100).toFixed(1))+'%':null;
const bar=(value,scale,cls='')=>{
  const share=finite(value)&&finite(scale)&&scale>0?Math.max(0,Math.min(1,value/scale)):0;
  return h('span',{class:'kt-bar '+cls,'aria-hidden':'true'},h('span',{style:{width:(share*100).toFixed(2)+'%'}}));
};
const p=(content,cls='')=>Array.isArray(content)||content instanceof Node?h('p',{class:'kt-p '+cls},content):
  h('p',{class:'kt-p '+cls,text:content});
const note=text=>p(text,'kt-muted kt-small');
const heading=(level,text,props)=>h('h'+level,{class:'kt-h'+level,text,...props});
const code=text=>h('span',{class:'kt-code',text});

// ---------------------------------------------------------------- payload
const provenance=obj(data.provenance),context=obj(data.context);
const task=str(context.task_type||provenance.task_type);
const isLink=/^(static|temporal)_link_prediction$/.test(task),isRegression=task==='regression',
  isBinary=task==='binary_classification',isMulticlass=task==='multiclass_classification',
  isMultilabel=task==='multilabel_classification',isClass=isBinary||isMulticlass||isMultilabel;
const TASKS={static_link_prediction:'Link prediction',temporal_link_prediction:'Temporal link prediction',
  regression:'Regression',binary_classification:'Binary classification',
  multiclass_classification:'Multiclass classification',multilabel_classification:'Multilabel classification',
  forecasting:'Forecasting'};
const verified=provenance.selection_verified===true;
const explainedFit=verified&&provenance.refitted_trainval===true?'refit_trainval':
  verified&&provenance.refitted_trainval===false?'original':null;
const fitName=fit=>({original:'Original fit',refit_trainval:'Refit on train + validation',
  refit_full:'Refit on all data'}[fit]||'Unverified fit');
const section=name=>obj(data[name]);

const metricBase=name=>str(name).replace(/^(test_refit_|train_|val_|validation_|test_)/,'');
const FAMILIES={map:'MAP',precision:'Precision',recall:'Recall',f1:'F1',ndcg:'NDCG',hit_ratio:'Hit rate',mrr:'MRR'};
const NAMES={auroc:'AUROC',auprc:'AUPRC',ap:'Average precision',acc:'Accuracy',accuracy:'Accuracy',
  mae:'MAE',rmse:'RMSE',mse:'MSE',loss:'Loss',r2:'R²',mape:'MAPE',smape:'SMAPE'};
const metricLabel=name=>{
  const base=metricBase(name),ranked=/^([a-z0-9_]+)@([0-9]+)$/.exec(base);
  if(ranked&&FAMILIES[ranked[1]])return FAMILIES[ranked[1]]+'@'+ranked[2];
  const averaged=/^(ap|auroc|auprc|f1|precision|recall)_(macro|micro)$/.exec(base);
  if(averaged)return (NAMES[averaged[1]]||averaged[1])+' ('+averaged[2]+')';
  const label=/^(ap|auroc|auprc)_per_label_(.+)$/.exec(base);
  if(label)return (NAMES[label[1]]||label[1])+' · '+label[2];
  return NAMES[base]||base;
};
const metricPhrase=name=>{const label=metricLabel(name);return /^[A-Z][a-z]/.test(label)?label[0].toLowerCase()+label.slice(1):label};
const selectionBase=typeof provenance.selection_metric==='string'?metricBase(provenance.selection_metric):null;
const LOWER=/^(loss|mae|mse|rmse|mape|smape)$/;
const HIGHER=/^((map|precision|recall|f1|ndcg|hit_ratio|mrr)@[0-9]+|auroc|auprc|ap|acc|accuracy|r2|(ap|auroc|auprc|f1|precision|recall)_(macro|micro)|(ap|auroc|auprc)_per_label_.+)$/;
const direction=name=>{
  const base=metricBase(name);
  if(base===selectionBase&&['min','max'].includes(provenance.selection_direction))
    return provenance.selection_direction==='max'?'higher':'lower';
  return LOWER.test(base)?'lower':HIGHER.test(base)?'higher':null;
};
const metricMeaning=name=>{
  const base=metricBase(name),ranked=/^([a-z0-9_]+)@([0-9]+)$/.exec(base);
  let text=null;
  if(ranked){const k=ranked[2];text={
    map:'Mean average precision of each entity’s top '+k+' ranked IDs; observed IDs ranked higher score more.',
    precision:'Share of each entity’s top '+k+' ranked IDs that were observed.',
    recall:'Share of each entity’s observed IDs found in its top '+k+'.',
    f1:'Harmonic mean of precision@'+k+' and recall@'+k+'.',
    ndcg:'Ranking quality of the top '+k+', weighting higher positions more.',
    hit_ratio:'Share of entities with at least one observed ID in the top '+k+'.',
    mrr:'Average reciprocal rank of the first observed ID within the top '+k+'.'}[ranked[1]]||null;}
  else text={auroc:'Chance that a positive example is scored above a negative one; 0.5 is random ordering.',
    auprc:'Area under the precision-recall curve.',ap:'Average precision across score thresholds.',
    acc:'Share of examples whose most probable class is the observed class.',
    accuracy:'Share of examples whose most probable class is the observed class.',
    mae:'Average absolute difference between prediction and observed value, in target units.',
    rmse:'Square root of the mean squared error, in target units; large errors weigh more.',
    mse:'Mean squared error, in squared target units.',loss:'Training objective measured on this split.',
    ap_macro:'Average precision computed for each label, then averaged across labels.',
    auroc_macro:'AUROC computed for each label, then averaged across labels.'}[base]||null;
  const way=direction(name);
  return [text,way==='higher'?'Higher is better.':way==='lower'?'Lower is better.':''].filter(Boolean).join(' ');
};
const ORDER=['map','ndcg','mrr','hit_ratio','precision','recall','f1','auroc','auprc','ap','acc','accuracy','mae','rmse','mse','loss'];
const metricOrder=(a,b)=>{
  const key=name=>{const base=metricBase(name),ranked=/^([a-z0-9_]+)@([0-9]+)$/.exec(base);
    const family=ORDER.indexOf(ranked?ranked[1]:base.split('_')[0]);
    return [base===selectionBase?0:1,family<0?ORDER.length:family,ranked?Number(ranked[2]):Infinity,base]};
  const x=key(a),y=key(b);
  for(let i=0;i<4;i++)if(x[i]!==y[i])return i===3?String(x[i]).localeCompare(String(y[i])):x[i]<y[i]?-1:1;
  return 0;
};

const REASONS={disabled:'Explanations were turned off for this training run.',
  failed:'Explanation generation failed during training.',
  model_only:'Only a published model was opened; it does not include training diagnostics.'};
const NOTES=[
  ['Gradient explanations exclude this module','Two-tower link-prediction models do not produce feature sensitivity.'],
  ['Column cohorts require a node-prediction module','Cohort analysis is produced for entity-level predictions, not for link prediction.'],
  ['Selected-example collection excludes this module','Two-tower link-prediction models do not save examples.'],
  ['Column cohorts exclude forecasting','Cohort analysis is not produced for forecasting models.'],
  ['Selected-example collection excludes forecasting','Examples are not saved for forecasting models.']];
const producerNote=name=>{const text=str(section(name).producer_note),found=NOTES.find(([marker])=>text.includes(marker));
  return found?found[1]:null};
const unavailable=(name,what)=>{
  const part=section(name),status=str(part.status);
  if(status==='unavailable')return REASONS[str(part.reason)]||what+' is not available for this run.';
  if(status==='invalid')return what+' could not be read because the saved file is invalid.';
  if(status==='limited')return what+' is too large to display in a notebook.';
  return producerNote(name)||what+' was not saved with this training run.';
};
const degraded=name=>str(section(name).reason)==='Explanation generation failed'?
  note('This may be incomplete because explanation generation failed.'):null;
const scopeNote=()=>{const text=str(section('gradient').producer_note);
  if(text.includes('shallow-RHS'))return 'For this ranking model, sensitivity is how the ranking scores respond to features in each entity’s sampled graph, not whether the ranked IDs change; the ranked IDs themselves have no scores.';
  if(text.startsWith('Recorded forecasting'))return 'For forecasting, sensitivity sums over all forecast horizons; dedicated lag and year-over-year inputs are not scored.';
  return null};

const metricRows=list(section('metrics').rows).map(obj).filter(row=>typeof row.name==='string');
const headline=verified&&selectionBase&&explainedFit?metricRows.find(row=>metricBase(row.name)===selectionBase&&
  row.split==='test'&&row.fit===explainedFit&&finite(row.value)&&row.origin==='Selected-checkpoint evaluation'&&
  !(isRegression&&['mae','rmse','mse'].includes(row.name)&&row.value<0))||null:null;

const gradientRows=list(section('gradient').rows).map(obj);
const gradient=gradientRows.filter(row=>finite(row.score)&&row.score>=0&&typeof row.table==='string'&&typeof row.column==='string')
  .sort((a,b)=>b.score-a.score||a.table.localeCompare(b.table)||a.column.localeCompare(b.column));
const cohorts=list(section('cohorts').rows).map(obj);
// The recorded query names the ranked column and its TOP k; nothing else is inferred from it.
const query=str(context.query);
const rankedColumn=isLink?(/LIST_DISTINCT\(\s*([A-Za-z_][\w.]*)/i.exec(query)||[])[1]||null:null;
const idName=rankedColumn?rankedColumn.split('.').pop():'ID';
const queryTopK=isLink?Number((/\bRANK\s+TOP\s+([0-9]+)/i.exec(query)||[])[1])||null:null;
const featureKey=(table,column)=>JSON.stringify([str(table),str(column)]);
const features=[],byKey=new Map(),cohortOnly=[];
gradient.forEach((row,rank)=>{const feature={key:featureKey(row.table,row.column),table:row.table,
  column:row.column,score:row.score,rank,cohorts:[]};features.push(feature);byKey.set(feature.key,feature)});
cohorts.forEach((row,index)=>{const key=featureKey(row.table_name,row.col_name);let feature=byKey.get(key);
  if(!feature){feature={key,table:str(row.table_name),column:str(row.col_name),score:null,rank:null,cohorts:[]};
    byKey.set(key,feature);cohortOnly.push(feature)}
  feature.cohorts.push(index)});
cohortOnly.sort((a,b)=>a.table.localeCompare(b.table)||a.column.localeCompare(b.column));
const featureList=[...features,...cohortOnly];
S.feature=featureList[0]?.key??null;
const featureName=(table,column)=>h('span',{class:'kt-fname'},h('span',{class:'kt-ftable kt-muted',text:str(table)+'.'}),
  h('span',{class:'kt-fcol',text:str(column)}));

// ---------------------------------------------------------------- examples
const samples=section('samples'),coverage=obj(samples.coverage),bodies=obj(samples.bodies),outputs=obj(samples.outputs);
const references=list(samples.references).map(obj);
const nodeTables=store=>{const map=new Map();
  for(const [table,ids] of Object.entries(obj(store.tables)))for(const id of list(ids)){
    const key=String(id);if(!map.has(key))map.set(key,[]);map.get(key).push(table)}
  return map};
const shortText=(value,limit=60)=>{const text=str(value).replace(/\s+/g,' ');
  return text.length>limit?text.slice(0,limit-1)+'…':text};
// Long identifiers keep their distinguishing end: acct-xxx…xx15.
const midText=(value,limit=28)=>{const text=str(value).replace(/\s+/g,' ');
  if(text.length<=limit)return text;const head=Math.ceil((limit-1)/2);return text.slice(0,head)+'…'+text.slice(text.length-(limit-1-head))};
const recordName=(store,tables,id)=>{
  const node=obj(obj(store.nodes)[id]),table=(tables.get(String(id))||[])[0];
  if(node.count!==undefined)return (table?table+' · ':'')+'aggregate of '+str(node.count)+' records';
  if(has(node,'pkey')&&node.pkey!==null&&node.pkey!==undefined)
    return (table?table+'.':'')+(str(node.pkey_name)||'id')+' '+str(node.pkey);
  return (table?table+' ':'')+'record '+id;
};
const recordList=(store,tables,ids)=>{
  const names=ids.map(id=>recordName(store,tables,id)),prefix=/^(\S+ )/.exec(names[0]||'');
  if(prefix&&names.every(name=>name.startsWith(prefix[1])&&!name.includes('aggregate')))
    return prefix[1]+names.map(name=>shortText(name.slice(prefix[1].length),24)).join(', ');
  return names.map(name=>shortText(name,32)).join(', ')};
const cellsOf=body=>Object.entries(obj(obj(body.feature_store).features)).flatMap(([id,columns])=>
  Object.entries(obj(columns)).map(([name,cell])=>{cell=obj(cell);
    const valid=finite(cell.score)&&cell.score>=0;
    return {id:String(id),table:str(cell.table_name),column:str(cell.name??name),value:cell.value,
      score:valid?cell.score:null,invalid:cell.score!==null&&cell.score!==undefined&&!valid}}));
const examples=references.filter(ref=>typeof ref.artifact_id==='string'&&has(bodies,ref.artifact_id)).map((ref,index)=>{
  const body=obj(bodies[ref.artifact_id]),store=obj(body.subgraph_store),tables=nodeTables(store);
  const root=str(store.root_id),node=obj(obj(store.nodes)[root]);
  const originals=[...new Set(Object.values(obj(ref.original_ids)).map(str))];
  const name=has(node,'pkey')&&node.pkey!==null?recordName(store,tables,root):
    originals.length===1?'ID '+originals[0]:'Example '+ref.artifact_id;
  return {index,ref,body,store,tables,root,name,table:(tables.get(root)||[])[0]||null,
    output:obj(outputs[ref.artifact_id]),cells:cellsOf(body).map(cell=>Object.assign(cell,{store}))}});
const entityTables=[...new Set(examples.map(example=>example.table).filter(Boolean))];
const entityNoun=entityTables.length===1?entityTables[0]:'examples';
const outputReady=output=>['available','partial'].includes(output.status)&&output.task_type===task;
const idKey=node=>{node=obj(node);
  return (typeof node.id==='string'||finite(node.id))&&typeof node.id_name==='string'&&node.id_name?
    JSON.stringify([typeof node.id,node.id_name,node.id]):null};
const ranking=output=>{
  const predictions=Array.isArray(output.prediction)?output.prediction.map(obj):null;
  const targets=Array.isArray(output.target)?output.target.map(obj):null,first=new Map();
  (predictions||[]).forEach((node,i)=>{const key=idKey(node);if(key!==null&&!first.has(key))first.set(key,i)});
  const observed=(targets||[]).map(node=>{const key=idKey(node);
    const comparable=key!==null&&predictions!==null&&predictions.every(item=>idKey(item)!==null&&item.id_name===node.id_name);
    return {node,comparable,rank:comparable?first.get(key):undefined}});
  const ranks=observed.map(item=>item.rank).filter(rank=>rank!==undefined);
  return {predictions,targets,observed,best:ranks.length?Math.min(...ranks):null,hits:new Set(ranks)}};
const probability=value=>finite(value)&&value>=0&&value<=1;
const membership=value=>value===0||value===1;
const classRows=output=>{
  const predictions=isBinary?{'Positive class':output.prediction}:obj(output.prediction);
  const targets=isBinary?{'Positive class':output.target}:obj(output.target);
  return [...new Set([...Object.keys(predictions),...Object.keys(targets)])].map(name=>({name,
    probability:probability(predictions[name])?predictions[name]:null,
    observed:membership(targets[name])?targets[name]:null}))
    .sort((a,b)=>(b.probability??-1)-(a.probability??-1)||a.name.localeCompare(b.name))};
const scalarRange=(()=>{const values=examples.flatMap(example=>outputReady(example.output)&&isRegression?
  [example.output.prediction,example.output.target].filter(finite):[]);
  return values.length?[Math.min(0,...values),Math.max(0,...values)]:null})();
const regressionPlaces=placesFor(examples.flatMap(example=>isRegression&&outputReady(example.output)?
  [example.output.prediction,example.output.target,finite(example.output.prediction)&&finite(example.output.target)?
    example.output.prediction-example.output.target:null]:[]));
const outcome=example=>{
  const output=example.output;
  if(!outputReady(output))return {ready:false};
  if(isLink){const result=ranking(output);return {ready:true,...result}}
  if(isRegression){const prediction=finite(output.prediction)?output.prediction:null,target=finite(output.target)?output.target:null;
    const error=prediction!==null&&target!==null&&finite(prediction-target)?prediction-target:null;
    return {ready:true,prediction,target,error}}
  if(isClass){const rows=classRows(output),top=rows[0]&&rows[0].probability!==null?rows[0]:null;
    const observed=rows.filter(row=>row.observed===1);
    return {ready:true,rows,top,observed,match:isMulticlass&&top&&observed.length===1?observed[0].name===top.name:null}}
  return {ready:false}};

// ---------------------------------------------------------------- components
const listbox=({label,items,selected,onSelect,render,cls,groups})=>{
  const box=h('div',{class:'kt-list',role:'listbox','aria-label':label}),options=[];
  let how='key';
  const choose=(index,focus)=>{
    options.forEach((option,i)=>{option.setAttribute('aria-selected',String(i===index));option.tabIndex=i===index?0:-1});
    if(focus)options[index].focus();
    onSelect(index,how)};
  items.forEach((item,index)=>{
    const group=(groups||[]).find(([,start])=>start===index);
    if(group)append(box,h('div',{class:'kt-group-label','aria-hidden':'true',text:group[0]}));
    const rendered=render(item,index);
    const option=h('div',{class:'kt-option '+cls,role:'option',id:nextId('option'),
      'aria-selected':String(index===selected),tabindex:index===selected||(selected<0&&index===0)?'0':'-1',
      'aria-label':rendered.label,onclick:()=>{how='pointer';choose(index,true);how='key'}},rendered.content);
    options.push(option);append(box,option)});
  box.addEventListener('keydown',event=>{
    const current=options.indexOf(event.target.closest('[role=option]'));
    if(current<0)return;
    if(event.key===' '||event.key==='Enter'){event.preventDefault();how='pointer';choose(current,true);how='key';return}
    let next={ArrowDown:current+1,ArrowUp:current-1,Home:0,End:options.length-1}[event.key];
    if(next===undefined)return;
    event.preventDefault();choose(Math.max(0,Math.min(options.length-1,next)),true)},{signal});
  return box};
// When the detail is stacked below its list, an explicit pick scrolls the detail into view; arrow keys do not.
const revealDetail=(detail,how)=>{
  if(how!=='pointer'||!detail||getComputedStyle(detail).position==='sticky')return;
  const box=detail.getBoundingClientRect();
  if(box.top<0||box.top>innerHeight*0.6)detail.scrollIntoView({block:'start'})};
const selectField=(label,choices,value,onChange)=>{
  const id=nextId('select'),select=h('select',{class:'kt-select',id,onchange:()=>onChange(select.value)},
    choices.map(([key,text])=>h('option',{value:key,text})));
  select.value=value;
  return h('label',{class:'kt-field',for:id},label,select)};
const link=(text,onClick,props)=>h('button',{type:'button',class:'kt-link',onclick:onClick,...props},text);

// Charts are drawn at their measured width so text and markers stay crisp.
const charts=new Map();
const resizer=new ResizeObserver(entries=>{for(const entry of entries){
  const draw=charts.get(entry.target);if(draw)draw(Math.floor(entry.contentRect.width))}});
onDispose(()=>resizer.disconnect());
const forgetCharts=node=>{for(const box of node.querySelectorAll('.kt-chart-box')){resizer.unobserve(box);charts.delete(box)}};
const chartBox=(draw,props)=>{
  const box=h('div',{class:'kt-chart-box',...props});let drawn=0;
  charts.set(box,width=>{if(width>0&&width!==drawn){drawn=width;draw(box,width)}});
  resizer.observe(box);return box};
const niceTicks=(low,high,count)=>{
  if(!(high>low)){const pad=Math.abs(low)*0.1||1;low-=pad;high+=pad}
  const raw=(high-low)/count,magnitude=10**Math.floor(Math.log10(raw)),scaled=raw/magnitude;
  const step=(scaled<1.5?1:scaled<3?2:scaled<7?5:10)*magnitude;
  const start=Math.floor(low/step+1e-9)*step,end=Math.ceil(high/step-1e-9)*step,values=[];
  for(let i=0;start+i*step<=end+step*1e-6&&i<50;i++)values.push(start+i*step);
  const places=Math.max(0,Math.min(10,-Math.floor(Math.log10(step)+1e-9)));
  return {values,low:start,high:end,label:value=>Math.abs(value)<step*1e-6?'0':value.toFixed(places)}};

// One learning-curve chart: a slider over recorded epochs with a readout.
const learningChart=({series,title,height=190,stateKey})=>{
  const epochs=[...new Set(series.flatMap(item=>item.points.map(point=>point.x)))].sort((a,b)=>a-b);
  const values=series.flatMap(item=>item.points.map(point=>point.y)).filter(finite);
  const frame=h('div',{class:'kt-chart-frame'});
  if(!epochs.length||!values.length){append(frame,note('No numeric values were recorded for this metric.'));return frame}
  let cursor=Math.min(S.epochs.get(stateKey)??epochs.length-1,epochs.length-1),geometry=null;
  const readout=h('div',{class:'kt-readout','aria-hidden':'true'});
  const valueAt=(item,epoch)=>{const point=item.points.find(point=>point.x===epoch);return point?point.y:null};
  const describe=()=>'Epoch '+epochs[cursor]+': '+series.map(item=>item.name+' '+(fmt(valueAt(item,epochs[cursor]))??'not recorded')).join(', ');
  const box=chartBox((box,width)=>{
    const ticks=niceTicks(Math.min(...values),Math.max(...values),4);
    const left=Math.max(...ticks.values.map(value=>ticks.label(value).length))*6.4+14,right=12,top=8,bottom=24;
    const inner=Math.max(20,width-left-right),tall=height-top-bottom;
    const x=epoch=>left+(epochs.length===1?inner/2:(epoch-epochs[0])/(epochs.at(-1)-epochs[0])*inner);
    const y=value=>top+tall-(value-ticks.low)/(ticks.high-ticks.low)*tall;
    const svg=sv('svg',{class:'kt-chart',width,height,viewBox:'0 0 '+width+' '+height,'aria-hidden':'true'});
    for(const value of ticks.values){
      append(svg,sv('line',{class:'kt-gridline',x1:left,x2:width-right,y1:y(value),y2:y(value)}));
      append(svg,svText({x:left-8,y:y(value)+4,'text-anchor':'end'},ticks.label(value)))}
    const every=Math.max(1,Math.ceil(epochs.length/Math.max(2,Math.floor(inner/44))));
    epochs.forEach((epoch,i)=>{if(i%every===0||i===epochs.length-1&&(epochs.length-1)%every>every/2)
      append(svg,svText({x:x(epoch),y:height-6,'text-anchor':'middle'},String(epoch)))});
    const markers=epochs.length<=inner/8;
    for(const item of series){
      const color=item.secondary?'var(--kumo-foreground)':'var(--kumo-accent)';let path='',previous=null;
      for(const point of item.points.slice().sort((a,b)=>a.x-b.x)){
        if(!finite(point.y)){previous=null;continue}
        path+=(previous!==null&&point.x-previous===1?'L':'M')+x(point.x).toFixed(1)+' '+y(point.y).toFixed(1)+' ';
        previous=point.x}
      append(svg,sv('path',{d:path,fill:'none',stroke:color,'stroke-width':2,'stroke-linejoin':'round',
        'stroke-dasharray':item.secondary?'5 4':null}));
      if(markers)for(const point of item.points)if(finite(point.y))append(svg,item.secondary?
        sv('rect',{x:x(point.x)-3.5,y:y(point.y)-3.5,width:7,height:7,fill:'var(--kumo-background)',stroke:color,'stroke-width':1.5}):
        sv('circle',{cx:x(point.x),cy:y(point.y),r:3.2,fill:color}))}
    const line=sv('line',{class:'kt-cursor',y1:top,y2:top+tall});append(svg,line);
    geometry={x,line,left,inner};box.replaceChildren(svg);update()});
  const update=()=>{
    if(geometry){const at=geometry.x(epochs[cursor]);geometry.line.setAttribute('x1',at);geometry.line.setAttribute('x2',at)}
    readout.replaceChildren(h('b',{text:'Epoch '+epochs[cursor]+(cursor===epochs.length-1&&epochs.length>1?' (last)':'')}),...series.map(item=>
      h('span',{},h('span',{class:item.secondary?'kt-key-observed':'kt-key','aria-hidden':'true',text:item.secondary?'□ ':'● '}),item.name+' ',num(valueAt(item,epochs[cursor])))));
    box.setAttribute('aria-valuenow',String(epochs[cursor]));box.setAttribute('aria-valuetext',describe())};
  let committed=cursor;
  const move=(index,commit=true)=>{cursor=Math.max(0,Math.min(epochs.length-1,index));if(commit){committed=cursor;S.epochs.set(stateKey,cursor)}update()};
  Object.entries({tabindex:'0',role:'slider','aria-label':title+' by epoch',
    'aria-valuemin':String(epochs[0]),'aria-valuemax':String(epochs.at(-1))}).forEach(([key,value])=>box.setAttribute(key,value));
  box.classList.add('kt-plot');
  box.addEventListener('keydown',event=>{
    const next={ArrowRight:cursor+1,ArrowUp:cursor+1,ArrowLeft:cursor-1,ArrowDown:cursor-1,Home:0,End:epochs.length-1,
      PageUp:cursor+10,PageDown:cursor-10}[event.key];
    if(next!==undefined){event.preventDefault();move(next)}},{signal});
  box.addEventListener('pointerleave',()=>{if(cursor!==committed)move(committed,false)},{signal});
  box.addEventListener('click',()=>move(cursor),{signal});
  box.addEventListener('pointermove',event=>{if(!geometry)return;
    const rect=box.getBoundingClientRect(),position=event.clientX-rect.left;let best=0;
    epochs.forEach((epoch,i)=>{if(Math.abs(geometry.x(epoch)-position)<Math.abs(geometry.x(epochs[best])-position))best=i});
    if(best!==cursor)move(best,false)},{signal});
  update();
  append(frame,readout,box);
  return frame};

// Name, value and local-sensitivity rows that reflow into two lines at narrow widths.
const scoreTable=(headers,rows,numeric)=>h('div',{class:'kt-scores'+(numeric?' kt-scores-num':''),role:'table'},
  h('div',{class:'kt-score-row kt-score-head',role:'row'},headers.map(text=>h('span',{role:'columnheader',text}))),
  rows.map(([name,value,score])=>h('div',{class:'kt-score-row',role:'row'},h('span',{role:'cell'},name),
    h('span',{role:'cell'},value),h('span',{role:'cell',class:'kt-score'},score))));
const scoreCell=(cell,scale)=>cell.score!==null?[bar(cell.score,scale),num(cell.score,2)]:
  [missing(cell.invalid?'invalid score':'not scored')];
const legend=items=>h('div',{class:'kt-legend'},items.map(([symbol,cls,text])=>
  h('span',{},h('span',{class:cls,'aria-hidden':'true',text:symbol+' '}),text)));
const axisRange=(values,isProbability)=>{
  if(isProbability)return [0,1];
  const finiteValues=values.filter(finite);
  if(!finiteValues.length)return null;
  let low=Math.min(0,...finiteValues),high=Math.max(0,...finiteValues);
  if(low===high){low-=1;high+=1}
  return [low,high]};
const place=(value,range)=>((value-range[0])/(range[1]-range[0])*100).toFixed(2)+'%';
const track=(prediction,observed,range,label)=>{
  // A paired prediction sits above its observation so equal values never hide each other.
  const node=h('div',{class:'kt-track'+(finite(prediction)&&finite(observed)?' kt-paired':''),role:label?'img':null,
    'aria-label':label,'aria-hidden':label?null:'true'});
  if(!range)return node;
  if(finite(prediction)&&finite(observed)){const a=Math.min(prediction,observed),b=Math.max(prediction,observed);
    append(node,h('span',{class:'kt-gap',style:{left:place(a,range),width:((b-a)/(range[1]-range[0])*100).toFixed(2)+'%'}}))}
  if(finite(observed))append(node,h('span',{class:'kt-diamond',style:{left:place(observed,range)}}));
  if(finite(prediction))append(node,h('span',{class:'kt-dot',style:{left:place(prediction,range)}}));
  return node};

// ---------------------------------------------------------------- navigation
const panels=[],tabs=[];
const TAB_NAMES=['Overview','Features','Examples','Training'];
const renderers=[];
const showTab=(index,focusTarget)=>{
  S.tab=index;
  tabs.forEach((tab,i)=>{tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1});
  panels.forEach((panel,i)=>{panel.hidden=i!==index});
  forgetCharts(panels[index]);panels[index].replaceChildren();renderers[index](panels[index]);
  if(focusTarget){const target=panels[index].querySelector('[role=option][aria-selected=true]')||panels[index].querySelector('.kt-focus');
    if(target){if(!target.hasAttribute('tabindex'))target.tabIndex=-1;target.focus({preventScroll:true})}
    tablist.scrollIntoView({block:'nearest'})}};
const openFeature=key=>{S.feature=key;showTab(1,true)};
const openExample=index=>{S.example=index;S.allLocal=false;showTab(2,true)};

// ---------------------------------------------------------------- overview
const card=(title,...children)=>{const id=nextId('card');
  return h('section',{class:'kt-card','aria-labelledby':id},heading(3,title,{id}),children)};
const topKNote=()=>{const k=Number((/@([0-9]+)$/.exec(metricBase(headline?.name||''))||[])[1]);
  return queryTopK&&k&&k!==queryTopK?note('The query asks for the top '+queryTopK+'; evaluation and saved examples use ranked lists of '+k+'.'):null};
const performanceCard=()=>{
  if(!headline){
    const tests=metricRows.filter(row=>row.split==='test'&&finite(row.value)).sort((a,b)=>metricOrder(a.name,b.name)).slice(0,6);
    return card('Test performance',p(verified?'The selection metric has no verified test value for the explained model.':
      'Model selection could not be verified, so no headline score is shown.','kt-muted'),
      tests.length?h('div',{class:'kt-related'},tests.map(row=>h('div',{},h('div',{class:'kt-small kt-muted',
        text:metricLabel(row.name)+(row.fit==='refit_trainval'?' · refit':'')}),h('div',{class:'kt-related-value'},num(row.value))))):
        note(unavailable('metrics','The metrics file')))}
  const k=(/@([0-9]+)$/.exec(metricBase(headline.name))||[])[1];
  const pool=metricRows.filter(row=>row.split==='test'&&row.fit===headline.fit&&row.name!==headline.name&&
    metricBase(row.name)!=='loss'&&!/_per_label_/.test(row.name)).sort((a,b)=>metricOrder(a.name,b.name));
  const related=[...pool.filter(row=>k&&row.name.endsWith('@'+k)),...pool.filter(row=>!(k&&row.name.endsWith('@'+k)))].slice(0,3);
  return card('Test performance',h('div',{class:'kt-metric'},
      h('div',{class:'kt-metric-name',text:metricLabel(headline.name)}),
      h('div',{class:'kt-metric-value'},num(headline.value)),
      p(metricMeaning(headline.name))),
    note([fitName(headline.fit),'test split'].join(' · ')),
    selectedValidation()!==null?note('Validation '+metricLabel(selectionBase)+' at the selected checkpoint: '+fmt(selectedValidation())+
      (trialRows.length>1?'; it chose this model among '+trialRows.length+' candidates.':'.')):null,
    topKNote(),
    related.length?h('div',{class:'kt-related'},related.map(row=>h('div',{},h('div',{class:'kt-small kt-muted',text:metricLabel(row.name)}),
      h('div',{class:'kt-related-value'},num(row.value))))):null)};
const featuresCard=()=>{
  if(!features.length)return card('Strongest features',p(gradientRows.length?'No valid sensitivity scores were recorded.':unavailable('gradient','Feature sensitivity'),'kt-muted'));
  const top=features.slice(0,5),scale=top[0].score;
  return card('Strongest features',note('Global sensitivity: how strongly test predictions respond to each feature.'+
      (isLink&&scopeNote()?' For ranking, this is score sensitivity, not whether the ranked IDs change.':'')),
    h('div',{class:'kt-list'},top.map(feature=>h('button',{type:'button',class:'kt-option kt-feature',
      'aria-label':feature.table+'.'+feature.column+', sensitivity '+fmt(feature.score,3)+'. Open feature details',
      onclick:()=>openFeature(feature.key)},
      h('span',{title:feature.table+'.'+feature.column},featureName(feature.table,feature.column)),bar(feature.score,scale),num(feature.score,3)))),
    link(features.length>5?'All '+features.length+' features →':'Feature details →',()=>openFeature(S.feature)))};
const exampleFacts=()=>{
  const ready=examples.map(example=>({example,result:outcome(example)})).filter(item=>item.result.ready);
  if(!ready.length)return null;
  const n=ready.length,facts=[];
  const fact=(value,text)=>h('li',{},h('span',{class:'kt-fact-value'},value),p(text,'kt-muted'));
  const jump=(item,text)=>link(item.example.name+' · '+text+' →',()=>openExample(item.example.index));
  if(isLink){
    const hits=ready.filter(item=>item.result.best!==null),sizes=new Set(ready.map(item=>(item.result.predictions||[]).length));
    const top=sizes.size===1?'top '+[...sizes][0]:'ranked list';
    facts.push(fact(hits.length+' of '+n,entityNoun+' have an observed ID in their '+top));
    const best=hits.slice().sort((a,b)=>a.result.best-b.result.best)[0];
    if(best)facts.push(jump(best,'observed '+idName+' '+shortText(str(best.result.predictions[best.result.best].id),24)+' at rank '+(best.result.best+1)));
    const slots=ready.flatMap(item=>(item.result.predictions||[]).map(idKey).filter(key=>key!==null)),counts=new Map();
    for(const key of slots)counts.set(key,(counts.get(key)||0)+1);
    const common=[...counts.entries()].sort((a,b)=>b[1]-a[1])[0];
    if(slots.length&&common)facts.push(fact(plural(counts.size,'distinct '+idName),'fill the '+slots.length+' ranked slots of the '+n+' saved '+
      (entityNoun==='examples'?'examples':entityNoun)+'; '+idName+' '+shortText(str(JSON.parse(common[0])[2]),16)+' is in '+
      (common[1]===n?'all '+n:common[1]+' of '+n)+' lists'))}
  else if(isRegression){
    const errors=ready.filter(item=>item.result.error!==null).map(item=>({item,size:Math.abs(item.result.error)}));
    if(errors.length){const sorted=errors.map(e=>e.size).sort((a,b)=>a-b),middle=sorted.length>>1;
      const median=sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;
      facts.push(fact(num(median),'median absolute error across '+(errors.length<ready.length?errors.length+' of '+ready.length+' examples with both values':
        plural(errors.length,'example'))));
      const worst=errors.slice().sort((a,b)=>b.size-a.size)[0];
      facts.push(jump(worst.item,'largest error '+signed(worst.item.result.error)))}}
  else if(isBinary){
    const groups=[1,0].map(label=>ready.filter(item=>item.result.observed.length===(label?1:0)&&item.result.top&&
      item.result.rows[0].observed===label));
    const mean=items=>items.length?items.reduce((sum,item)=>sum+item.result.top.probability,0)/items.length:null;
    facts.push(h('li',{},h('span',{class:'kt-fact-value'},numProb(mean(groups[0])),h('span',{class:'kt-muted',text:' vs '}),numProb(mean(groups[1]))),
      p('mean positive-class probability for observed positives ('+groups[0].length+') vs negatives ('+groups[1].length+')','kt-muted')));
    const doubtful=groups[1].slice().sort((a,b)=>b.result.top.probability-a.result.top.probability)[0];
    if(doubtful)facts.push(jump(doubtful,'highest-probability negative ('+prob(doubtful.result.top.probability)+')'))}
  else if(isMulticlass){
    const judged=ready.filter(item=>item.result.match!==null),matches=judged.filter(item=>item.result.match);
    facts.push(fact(matches.length+' of '+judged.length,'examples where the most probable class is the observed class'));
    const miss=judged.find(item=>!item.result.match);
    if(miss)facts.push(jump(miss,'predicted '+shortText(miss.result.top.name,20)+', observed '+shortText(miss.result.observed[0].name,20)))}
  else if(isMultilabel){
    const cells=ready.flatMap(item=>item.result.rows.filter(row=>row.probability!==null&&row.observed!==null).map(row=>({item,row})));
    const mean=rows=>rows.length?rows.reduce((sum,cell)=>sum+cell.row.probability,0)/rows.length:null;
    const present=cells.filter(cell=>cell.row.observed===1),absent=cells.filter(cell=>cell.row.observed===0);
    facts.push(h('li',{},h('span',{class:'kt-fact-value'},numProb(mean(present)),h('span',{class:'kt-muted',text:' vs '}),numProb(mean(absent))),
      p('mean probability for observed labels vs absent labels','kt-muted')));
    const weakest=present.slice().sort((a,b)=>a.row.probability-b.row.probability)[0];
    if(weakest)facts.push(jump(weakest.item,'lowest-probability observed label '+shortText(weakest.row.name,20)))}
  return facts.length?facts:null};
const exampleKey=example=>{const node=obj(obj(example.store.nodes)[example.root]);
  return has(node,'pkey')&&node.pkey!==null?midText(node.pkey,10):midText(example.name,10)};
const outcomeStrip=ready=>{
  if(isRegression){
    const points=ready.filter(item=>item.result.prediction!==null&&item.result.target!==null);
    if(points.length<2)return null;
    const values=points.flatMap(item=>[item.result.prediction,item.result.target]);
    let low=Math.min(...values),high=Math.max(...values);const pad=(high-low)*0.08||1;low-=pad;high+=pad;
    return h('div',{class:'kt-section'},chartBox((box,width)=>{
      const height=170,left=46,bottom=22,top=6,right=8,inner=width-left-right,tall=height-top-bottom;
      const x=value=>left+(value-low)/(high-low)*inner,y=value=>top+tall-(value-low)/(high-low)*tall;
      const svg=sv('svg',{class:'kt-chart',width,height,viewBox:'0 0 '+width+' '+height,role:'img',
        'aria-label':'Predicted versus observed values for '+plural(points.length,'saved example')+'; the dashed line marks exact predictions.'});
      append(svg,sv('rect',{x:left,y:top,width:inner,height:tall,fill:'none',stroke:'var(--kumo-surface-border)'}),
        sv('line',{x1:x(low),y1:y(low),x2:x(high),y2:y(high),stroke:'var(--kumo-muted)','stroke-dasharray':'4 4'}),
        svText({x:left+inner/2,y:height-4,'text-anchor':'middle'},'Observed'),
        svText({x:left,y:height-4,'text-anchor':'start'},axisLabel(low)),svText({x:left+inner,y:height-4,'text-anchor':'end'},axisLabel(high)),
        svText({x:left-5,y:top+tall,'text-anchor':'end'},axisLabel(low)),svText({x:left-5,y:top+9,'text-anchor':'end'},axisLabel(high)),
        svText({x:12,y:top+tall/2,'text-anchor':'middle',transform:'rotate(-90 12 '+(top+tall/2)+')'},'Predicted'),
        points.map(item=>sv('circle',{cx:x(item.result.target),cy:y(item.result.prediction),r:4,fill:'var(--kumo-accent)',
          stroke:'var(--kumo-background)','stroke-width':1.5})));
      box.replaceChildren(svg)}),note('Each dot is a saved example; dots on the dashed line are exact.'))}
  if(isBinary){
    const rows=[[1,'Observed positive'],[0,'Observed negative']].map(([label,name])=>[name,ready.filter(item=>item.result.top&&
      item.result.rows[0].observed===label)]).filter(([,items])=>items.length);
    if(!rows.length)return null;
    return h('div',{class:'kt-section',role:'img','aria-label':rows.map(([name,items])=>name+': probabilities '+
        items.map(item=>fmt(item.result.top.probability,2)).join(', ')).join('; ')},
      rows.map(([name,items])=>h('div',{class:'kt-dotrow','aria-hidden':'true'},h('span',{class:'kt-small kt-muted',text:name+' ('+items.length+')'}),
        h('div',{class:'kt-track'},items.map(item=>h('span',{class:'kt-dot',style:{left:place(item.result.top.probability,[0,1])}}))))),
      h('div',{class:'kt-dotrow','aria-hidden':'true'},h('span'),h('div',{class:'kt-axis'},h('span',{text:'0'}),
        h('span',{text:'positive-class probability'}),h('span',{text:'1'}))))}
  const cells=ready.map(({example,result})=>{
    const good=isLink?result.best!==null:isMulticlass?result.match===true:isMultilabel?!!(result.top&&result.observed.some(row=>row.name===result.top.name)):false;
    const value=isLink?(result.best!==null?'✓ #'+(result.best+1):'—'):good?'✓':result.top?'✗':'—';
    const label=example.name+': '+(isLink?(result.best!==null?'observed ID at rank '+(result.best+1):'no observed ID ranked'):
      good?'most probable '+(isMultilabel?'label':'class')+' observed':'most probable '+(isMultilabel?'label':'class')+' not observed');
    return h('li',{},h('button',{type:'button',class:'kt-strip-cell'+(good?' kt-match':''),'aria-label':label+'. Open example',
      onclick:()=>openExample(example.index)},h('span',{class:'kt-small kt-muted kt-label',text:exampleKey(example)}),h('strong',{text:value})))});
  return h('div',{class:'kt-section'},h('ul',{class:'kt-strip','aria-label':isLink?'Best observed rank per saved example':
    'Whether the most probable '+(isMultilabel?'label':'class')+' was observed, per saved example'},cells),
    note(isLink?'Best rank of an observed ID for each saved '+(entityNoun==='examples'?'example':entityNoun.replace(/s$/,''))+'; — means none was ranked.':
      '✓ the most probable '+(isMultilabel?'label':'class')+' was observed · ✗ it was not.'))};
const examplesCard=()=>{
  if(!examples.length)return card('Examples',p(examplesUnavailable(),'kt-muted'));
  const ready=examples.map(example=>({example,result:outcome(example)})).filter(item=>item.result.ready);
  const facts=exampleFacts();
  return card('Examples',facts?h('ul',{class:'kt-facts'},facts.map(item=>item.tagName==='LI'?item:h('li',{},item))):
      p('Predictions were not saved for these examples.','kt-muted'),ready.length?outcomeStrip(ready):null,
    note(loadedNote()),
    link('Open examples →',()=>openExample(S.example)))};

// ---------------------------------------------------------------- learning history
const historyRows=list(section('history').rows).map(obj);
const historyBase=name=>str(name).replace(/^(train_|val_|validation_)/,'');
const historyMetrics=[...new Set(historyRows.flatMap(row=>[...Object.keys(obj(row.train_metrics)),
  ...Object.keys(obj(row.validation_metrics))]).map(historyBase))].sort(metricOrder);
const fits=[...new Map(historyRows.map(row=>[JSON.stringify([row.trial,row.fit]),{trial:str(row.trial),fit:str(row.fit)}])).values()]
  .sort((a,b)=>(a.fit==='original'?0:1)-(b.fit==='original'?0:1));
S.metric=historyMetrics.includes(selectionBase)?selectionBase:historyMetrics[0]??null;
const epochOf=row=>typeof row.epoch==='string'&&/^(0|[1-9][0-9]{0,8})$/.test(row.epoch)?Number(row.epoch):null;
const historyValue=(values,metric)=>{const entry=Object.entries(obj(values)).find(([name])=>historyBase(name)===metric);
  return entry&&finite(entry[1])?entry[1]:null};
const fitSeries=(fit,metric)=>{
  const rows=historyRows.filter(row=>str(row.trial)===fit.trial&&str(row.fit)===fit.fit&&epochOf(row)!==null);
  return [['Train','train_metrics',false],['Validation','validation_metrics',true]].map(([name,key,secondary])=>
    ({name,secondary,points:rows.map(row=>({x:epochOf(row),y:historyValue(row[key],metric)}))}))
    .filter(item=>item.points.some(point=>point.y!==null))};
const learningCard=()=>{
  const fit=fits[0];
  if(!fit||!S.metric)return card('Learning curve',p(historyRows.length?'No numeric epoch values were recorded.':
    unavailable('history','Learning history'),'kt-muted'));
  const series=fitSeries(fit,S.metric),epochs=new Set(historyRows.filter(row=>str(row.trial)===fit.trial&&str(row.fit)===fit.fit).map(row=>row.epoch));
  return card('Learning curve',
    p(metricLabel(S.metric)+' by epoch · '+fitName(fit.fit)+' · '+series.map(item=>item.secondary?'validation dashed':'train solid').join(', '),'kt-muted kt-small'),
    learningChart({series,title:metricLabel(S.metric)+' learning curve',height:150,stateKey:'overview'}),
    note(plural(epochs.size,'recorded epoch')+' · '+plural(trialRows.length,'candidate')+' trained'),
    link('Training details →',()=>showTab(3,true)))};

// ---------------------------------------------------------------- metrics table
const metricsTable=()=>{
  const columns=[],key=row=>str(row.fit)+'|'+str(row.split);
  const order=['original|train','original|val','original|test','refit_trainval|test','refit_full|test'];
  for(const name of order)if(metricRows.some(row=>key(row)===name)){columns.push(name);
    if(name==='original|val'&&selectedTrialRow()?.comparison_source==='final_validation_metrics')columns.push('selected|val')}
  for(const row of metricRows)if(!columns.includes(key(row)))columns.push(key(row));
  const refit=metricRows.some(row=>row.fit!=='original');
  const label=name=>({'original|train':'Train (last epoch)','original|val':'Validation (last epoch)',
    'selected|val':'Validation (selected checkpoint)','original|test':refit?'Test · original fit':'Test','refit_trainval|test':'Test · refit',
    'refit_full|test':'Test · full refit'}[name]||name.replace('|',' · '));
  const names=[...new Set(metricRows.map(row=>row.name))].sort(metricOrder);
  let unverified=false;
  const table=h('table',{class:'kt-table'},
    h('thead',{},h('tr',{},h('th',{scope:'col',text:'Metric'}),columns.map(name=>h('th',{scope:'col',class:'kt-num',text:label(name)})))),
    h('tbody',{},names.map(name=>{
      return h('tr',{class:headline&&name===headline.name?'kt-strong':''},
      h('td',{},metricLabel(name),metricBase(name)===selectionBase?h('span',{class:'kt-muted',text:' · used for selection'}):null),
      columns.map(column=>{const row=metricRows.find(item=>item.name===name&&key(item)===column);
        if(column==='selected|val'){const value=obj(selectedTrialRow().comparison_metrics)['val_'+metricBase(name)];
          return h('td',{class:'kt-num'},finite(value)?num(value):missing())}
        const cell=h('td',{class:'kt-num'},row?num(row.value):missing());
        if(row&&finite(row.value)&&row.origin==='Unknown metric origin'){unverified=true;
          append(cell,h('span',{class:'kt-mark','aria-hidden':'true',text:'*'}),h('span',{class:'kt-sr',text:' origin not verified'}))}
        return cell}))})));
  const origins=[...new Set(metricRows.map(row=>row.origin))];
  const notes=[origins.includes('Last recorded training metrics (epoch unverified)')?'Train and validation values are the last values recorded during training.':null,
    origins.includes('Selected-checkpoint evaluation')?'Test values evaluate the selected checkpoint.':null,
    unverified?'* The source of this value could not be confirmed.':null].filter(Boolean);
  return [h('div',{class:'kt-scroll'},table),notes.length?note(notes.join(' ')):null]};

// ---------------------------------------------------------------- overview panel
renderers[0]=panel=>{
  const metricsCount=new Set(metricRows.map(row=>row.name)).size;
  const cards=[performanceCard()],gaps=[];
  if(features.length)cards.push(featuresCard());
  else gaps.push(['Features',gradientRows.length?'No valid sensitivity scores were recorded.':unavailable('gradient','Feature sensitivity')]);
  if(examples.length)cards.push(examplesCard());else gaps.push(['Examples',examplesUnavailable()]);
  if(fits.length&&S.metric)cards.push(learningCard());
  else gaps.push(['Learning curve',historyRows.length?'No numeric epoch values were recorded.':unavailable('history','Learning history')]);
  if(cards.length%2)cards.at(-1).classList.add('kt-wide');
  append(panel,h('h3',{class:'kt-sr kt-focus',text:'Overview'}),h('div',{class:'kt-grid'},cards),
    gaps.length?h('ul',{class:'kt-gaps','aria-label':'Not available for this run'},gaps.map(([name,text])=>
      h('li',{},h('b',{text:name+': '}),text))):null);
  if(metricsCount){const details=h('details',{class:'kt-section'},h('summary',{text:'All recorded metrics ('+metricsCount+')'}),metricsTable());
    details.style.marginTop='16px';append(panel,details)}
  else append(panel,note(unavailable('metrics','The metrics file')))};

// ---------------------------------------------------------------- features panel
const cohortSection=feature=>{
  const wrap=h('div',{class:'kt-section'});
  const title=heading(4,'Predictions across '+feature.table+'.'+feature.column);
  if(!feature.cohorts.length){
    if(!cohorts.length&&producerNote('cohorts'))return null;
    const reason=cohorts.length?'No cohort analysis was saved for this feature.':unavailable('cohorts','Cohort analysis');
    return append(wrap,title,note(reason))}
  const chosen=feature.cohorts.includes(S.hop.get(feature.key))?S.hop.get(feature.key):feature.cohorts[0];
  const cohort=cohorts[chosen];
  const controls=h('div',{class:'kt-row'});
  if(feature.cohorts.length>1)append(controls,selectField('Hop',feature.cohorts.map(index=>[String(index),
    'Hop '+str(cohorts[index].hop)]),String(chosen),value=>{S.hop.set(feature.key,Number(value));redrawFeature()}));
  const classNames=Array.isArray(cohort.predictions)?[]:[...new Set([...Object.keys(obj(cohort.predictions)),...Object.keys(obj(cohort.targets))])];
  let className=null;
  if(classNames.length){className=classNames.includes(S.cls.get(feature.key))?S.cls.get(feature.key):classNames[0];
    append(controls,selectField(isMultilabel?'Label':'Class',classNames.map(name=>[name,shortText(name,40)]),className,
      value=>{S.cls.set(feature.key,value);redrawFeature()}))}
  const pick=value=>Array.isArray(value)?value:className!==null&&Array.isArray(obj(value)[className])?obj(value)[className]:[];
  const predictions=pick(cohort.predictions),targets=pick(cohort.targets),shares=list(cohort.populations);
  const labels=list(cohort.cohorts).slice(0,100);
  const isProbability=isClass||str(section('cohorts').task_type).includes('classification');
  const range=axisRange(labels.flatMap((_,i)=>[predictions[i],targets[i]]),isProbability&&
    labels.every((_,i)=>[predictions[i],targets[i]].every(value=>!finite(value)||value>=0&&value<=1)));
  const maxShare=Math.max(0,...shares.filter(finite));
  const places=placesFor(labels.flatMap((_,i)=>[predictions[i],targets[i]]));
  const predictionName=isBinary?'Mean positive-class probability':isClass?'Mean probability':'Mean prediction';
  const observedName=isBinary?'Observed positive share':isClass?'Observed share':'Mean observed value';
  const rows=h('div',{class:'kt-cohort',role:'table','aria-label':'Predictions across '+feature.table+'.'+feature.column},
    h('div',{class:'kt-cohort-row kt-cohort-head',role:'row'},h('span',{role:'columnheader',text:'Group'}),
      h('span',{role:'columnheader'},range?h('span',{class:'kt-axis'},h('span',{text:axisLabel(range[0])}),h('span',{text:axisLabel(range[1])})):null),
      h('span',{role:'columnheader',class:'kt-num',title:predictionName},h('span',{'aria-hidden':'true',class:'kt-key',text:'● '}),'Pred.'),
      h('span',{role:'columnheader',class:'kt-num',title:observedName},h('span',{'aria-hidden':'true',text:'◆ '}),'Obs.'),
      h('span',{role:'columnheader',class:'kt-num',text:'Share'})),
    labels.map((label,i)=>h('div',{class:'kt-cohort-row',role:'row'},
      h('span',{role:'cell',class:'kt-label',title:str(label),text:shortText(rangeLabel(label),48)}),
      h('span',{role:'cell'},track(predictions[i],targets[i],range)),
      h('span',{role:'cell',class:'kt-num'},isProbability?numProb(predictions[i]):num(predictions[i],places)),
      h('span',{role:'cell',class:'kt-num'},isProbability?numProb(targets[i]):num(targets[i],places)),
      h('span',{role:'cell',class:'kt-share'},percent(shares[i])??missing(),bar(shares[i],maxShare,'kt-thin')))));
  append(wrap,h('div',{class:'kt-section-head'},title,controls),
    legend([['●','kt-key',predictionName],['◆','kt-key-observed',observedName],['','','Share: fraction of the analyzed test population; counts are not recorded']]),rows);
  if(list(cohort.cohorts).length>100)append(wrap,note('Showing the first 100 of '+list(cohort.cohorts).length+' groups.'));
  if(!isProbability)append(wrap,note('Target units are not recorded.'));
  return wrap};
const exampleValues=feature=>{
  const wrap=h('div',{class:'kt-section'},heading(4,feature.table+'.'+feature.column+' in the saved examples'));
  if(!examples.length)return append(wrap,note(examplesUnavailable()));
  const rows=examples.map(example=>{const cells=example.cells.filter(cell=>cell.table===feature.table&&cell.column===feature.column);
    const scored=cells.filter(cell=>cell.score!==null).sort((a,b)=>b.score-a.score);
    return {example,cells,top:scored[0]||cells[0]||null}}).filter(row=>row.cells.length)
    .sort((a,b)=>compareValues(a.top.value,b.top.value)||a.example.index-b.example.index);
  if(!rows.length)return append(wrap,note('No saved example records this feature.'));
  const maxScore=Math.max(1,...rows.map(row=>row.top.score).filter(finite));
  const trivial=features.length===1&&rows.every(row=>row.top.score===1);
  append(wrap,scoreTable(['Example','Value',trivial?'':'Local sensitivity'],rows.map(({example,cells,top})=>[
      link(midText(example.name,32),()=>openExample(example.index),{class:'kt-link kt-inline',title:example.name,'aria-label':'Open example '+example.name}),
      [valueText(top),cells.length>1?h('span',{class:'kt-muted',text:' +'+(cells.length-1)+' more'}):null],
      trivial?null:scoreCell(top,maxScore)]),rows.every(row=>finite(row.top.value))),
    note(trivial?'This is the only scored feature, so its local sensitivity is 1 in every example.':
      'Sorted by value. Local sensitivity compares features within one example (its strongest is 1); it is not comparable across examples.'));
  if(rows.length<examples.length)append(wrap,note(plural(examples.length-rows.length,'saved example')+' do not record this feature.'));
  return wrap};
const valueText=cell=>{
  const node=obj(obj(cellStore(cell).nodes)[cell.id]);
  if((cell.value===null||cell.value===undefined)&&node.count!==undefined)return h('span',{class:'kt-muted',text:'aggregate (value not saved)'});
  if(cell.value===null||cell.value===undefined)return missing();
  if(finite(cell.value))return h('span',{class:'kt-num',title:String(cell.value),text:dataValue(cell.value)});
  return h('span',{title:str(cell.value),text:shortText(cell.value,40)})};
const cellStore=cell=>cell.store||{};

let redrawFeature=()=>{};
renderers[1]=panel=>{
  if(!featureList.length){
    append(panel,h('div',{class:'kt-section'},heading(3,'Features',{class:'kt-h3 kt-focus'}),p(gradientRows.length?'No valid sensitivity scores were recorded.':
      unavailable('gradient','Feature sensitivity'),'kt-muted'),producerNote('cohorts')&&!cohorts.length?note(producerNote('cohorts')):null));
    return}
  const shown=S.allFeatures?featureList:[...features.slice(0,20),...cohortOnly.slice(0,10)];
  if(!shown.some(feature=>feature.key===S.feature))S.feature=shown[0].key;
  const scale=features.length?features[0].score:1;
  const detail=h('div',{class:'kt-detail'});
  redrawFeature=()=>{forgetCharts(detail);detail.replaceChildren(featureDetail(byKey.get(S.feature)))};
  const firstCohortOnly=shown.findIndex(feature=>feature.score===null);
  const groups=firstCohortOnly>0?[['Cohort analysis only (no sensitivity score)',firstCohortOnly]]:[];
  const box=listbox({label:'Features',items:shown,selected:shown.findIndex(feature=>feature.key===S.feature),cls:'kt-feature',groups,
    onSelect:(index,how)=>{S.feature=shown[index].key;redrawFeature();revealDetail(detail,how)},
    render:feature=>({label:feature.table+'.'+feature.column+(feature.score===null?', no sensitivity score':', sensitivity '+fmt(feature.score,3)),
      content:[h('span',{title:feature.table+'.'+feature.column},featureName(feature.table,feature.column)),
        feature.score===null?h('span'):bar(feature.score,scale),feature.score===null?missing('no sensitivity score'):num(feature.score,3)]})});
  const master=h('div',{class:'kt-stack'},h('div',{class:'kt-list-head'},heading(3,'Global sensitivity'),
      note('How strongly test predictions respond to each feature. Unsigned: not a direction or a cause. Scores sum to 1 across features.'),
      degraded('gradient')),
    features.length?h('div',{class:'kt-col-head',style:{'grid-template-columns':'minmax(0,1fr) auto'}},h('span',{text:'Feature'}),h('span',{text:'Sensitivity'})):null,
    box,
    features.length>20||cohortOnly.length>10?h('button',{type:'button',class:'kt-btn kt-more','aria-expanded':String(S.allFeatures),
      onclick:()=>{S.allFeatures=!S.allFeatures;showTab(1,false);panels[1].querySelector('.kt-more')?.focus()}},
      S.allFeatures?'Show fewer features':'Show all '+featureList.length+' features'):null,
    !features.length?note(unavailable('gradient','Feature sensitivity')):null,
    scopeNote()?note(scopeNote()):null,
    gradientRows.length>gradient.length?note(plural(gradientRows.length-gradient.length,'invalid or negative score')+' not shown.'):null);
  append(panel,h('div',{class:'kt-split'},master,detail));
  redrawFeature()};
const featureDetail=feature=>{
  const total=features.length;
  return h('div',{class:'kt-stack'},
    h('div',{class:'kt-section'},heading(3,feature.table+'.'+feature.column),
      feature.score===null?note('No global sensitivity score was recorded for this feature.'):
        h('p',{class:'kt-p'},'Global sensitivity ',h('b',{},num(feature.score,3)),h('span',{class:'kt-muted',text:' · rank '+(feature.rank+1)+' of '+total}))),
    cohortSection(feature),
    exampleValues(feature))};

// ---------------------------------------------------------------- examples panel
const examplesUnavailable=()=>{
  const part=section('samples'),loaded=finite(coverage.loaded)?coverage.loaded:0;
  if(['available','limited','partial'].includes(part.status)&&references.length&&!loaded)
    return ['No examples are loaded in this view. Show up to 12 by displaying ',code('explanations.visualize(max_samples=12)'),'.'];
  return unavailable('samples','Example collection')};
const exampleSummary=example=>{
  const result=outcome(example);
  if(!result.ready)return {line:'Prediction not saved',value:missing('no prediction'),label:'prediction not saved'};
  if(isLink){
    const targets=result.targets;
    if(targets===null)return {line:'Observed IDs not recorded',value:missing(),label:'observed IDs not recorded'};
    if(!targets.length)return {line:'No observed IDs',value:missing('no observed IDs'),label:'no observed IDs'};
    const observed='Observed '+idName+' '+targets.slice(0,3).map(node=>shortText(str(node.id),16)).join(', ')+(targets.length>3?' +'+(targets.length-3):'');
    if(result.best!==null)return {line:observed,value:h('span',{},h('span',{class:'kt-hit','aria-hidden':'true',text:'✓ '}),'rank '+(result.best+1)),
      label:observed+', best rank '+(result.best+1)};
    const size=(result.predictions||[]).length,comparable=result.observed.some(item=>item.comparable);
    return {line:observed,value:h('span',{class:'kt-muted',text:comparable?'not in top '+size:'not comparable'}),
      label:observed+(comparable?', not in top '+size:', not comparable')}}
  if(isRegression){
    const places=regressionPlaces;
    return {line:'Predicted '+(fmt(result.prediction,places)??'—')+' · observed '+(fmt(result.target,places)??'—'),
      value:result.error===null?missing('no error'):h('span',{class:'kt-num',text:signed(result.error,places)}),
      label:'predicted '+(fmt(result.prediction,places)??'not recorded')+', observed '+(fmt(result.target,places)??'not recorded')+(result.error===null?'':', error '+signed(result.error,places))}}
  if(isBinary){
    const row=result.rows[0]||{},observed=row.observed===1?'positive':row.observed===0?'negative':'not recorded';
    return {line:'Observed '+observed,value:numProb(row.probability),
      label:'positive-class probability '+(prob(row.probability)??'not recorded')+', observed '+observed}}
  const observed=result.observed.map(row=>shortText(row.name,14));
  const line='Observed '+(observed.length?observed.slice(0,2).join(', ')+(observed.length>2?' +'+(observed.length-2):''):'none');
  if(!result.top)return {line,value:missing(),label:line};
  return {line,value:h('span',{},result.match?h('span',{class:'kt-hit','aria-hidden':'true',text:'✓ '}):null,shortText(result.top.name,14)+' '+prob(result.top.probability)),
    label:line+', most probable '+result.top.name+' '+prob(result.top.probability)+(result.match?', matches observed':'')}};
renderers[2]=panel=>{
  if(!examples.length){append(panel,h('div',{class:'kt-section'},heading(3,'Examples',{class:'kt-h3 kt-focus'}),p(examplesUnavailable(),'kt-muted'),coverageNote()));return}
  S.example=Math.min(S.example,examples.length-1);
  const detail=h('div',{class:'kt-detail'});
  const box=listbox({label:'Saved test examples',items:examples,selected:S.example,cls:'kt-example',
    onSelect:(index,how)=>{S.example=index;S.allLocal=false;drawExample(detail,false);revealDetail(detail,how)},
    render:example=>{const summary=exampleSummary(example);
      return {label:example.name+', '+summary.label,content:[h('span',{class:'kt-name'},
        h('span',{class:'kt-label',title:example.name,text:midText(example.name,34)}),h('span',{class:'kt-label kt-muted kt-small',text:summary.line})),
        h('span',{class:'kt-outcome'},summary.value)]}}});
  const column=isLink?'Best observed rank':isRegression?'Error (pred. − obs.)':isBinary?'Positive probability':'Most probable';
  const master=h('div',{class:'kt-stack'},h('div',{class:'kt-list-head'},heading(3,'Saved test examples'),
      note(loadedNote()),
      sharedStrategies?note('Selected by: '+sharedStrategies.join(', ')):null,degraded('samples')),
    h('div',{class:'kt-col-head',style:{'grid-template-columns':'minmax(0,1fr) auto'}},h('span',{text:'Example'}),h('span',{text:column})),
    box,coverageNote());
  append(panel,h('div',{class:'kt-split'},master,detail));
  drawExample(detail,false)};
const loadedNote=()=>(references.length>examples.length?examples.length+' of '+references.length:String(examples.length))+
  ' saved test examples loaded; they do not represent the whole test set.';
const coverageNote=()=>{
  const extra=references.length-examples.length;
  if(extra<=0)return null;
  const ids=references.filter(ref=>!has(bodies,ref.artifact_id)).map(ref=>str(ref.artifact_id)).slice(0,20);
  return h('details',{},h('summary',{class:'kt-small',text:examples.length?plural(extra,'more saved example')+' not loaded':
      plural(extra,'saved example')+(extra===1?' is':' are')+' not loaded'}),
    h('p',{class:'kt-p kt-muted kt-small'},'Load one in Python with ',code('explanations.sample("<collector ID>")'),'; it returns the sampled graph without the prediction.'),
    note('Collector IDs (artifact names, not entity IDs): '+ids.join(', ')+(extra>ids.length?' …':'')))};
const strategiesOf=example=>list(example.ref.categories).map(name=>str(name).replace(/_/g,' '));
const sharedStrategies=examples.length&&examples.every(example=>strategiesOf(example).join('|')===strategiesOf(examples[0]).join('|'))&&
  strategiesOf(examples[0]).length?strategiesOf(examples[0]):null;

const drawExample=(detail,focus)=>{
  forgetCharts(detail);
  const example=examples[S.example];
  const strategies=list(example.ref.categories).map(name=>str(name).replace(/_/g,' '));
  const nav=h('div',{class:'kt-row'},
    h('button',{type:'button',class:'kt-btn',disabled:S.example===0,'aria-label':'Previous example',
      onclick:()=>selectExample(S.example-1)},'‹ Previous'),
    h('span',{class:'kt-muted kt-small',text:(S.example+1)+' of '+examples.length}),
    h('button',{type:'button',class:'kt-btn',disabled:S.example===examples.length-1,'aria-label':'Next example',
      onclick:()=>selectExample(S.example+1)},'Next ›'));
  const title=heading(3,example.name,{class:'kt-h3 kt-focus',tabindex:'-1'});
  detail.replaceChildren(h('div',{class:'kt-stack'},
    h('div',{class:'kt-section'},h('div',{class:'kt-section-head'},title,nav),
      strategies.length&&!sharedStrategies?note('Selected by: '+strategies.join(', ')):null,
      example.store.seed_time!==null&&example.store.seed_time!==undefined?note('Prediction time: '+str(example.store.seed_time)):null),
    predictionSection(example),localSection(example),recordsSection(example)));
  if(focus)title.focus()};
const selectExample=index=>{
  S.example=Math.max(0,Math.min(examples.length-1,index));S.allLocal=false;
  const option=panels[2].querySelectorAll('[role=option]')[S.example];
  if(option){panels[2].querySelectorAll('[role=option]').forEach(item=>{item.setAttribute('aria-selected',String(item===option));item.tabIndex=item===option?0:-1})}
  drawExample(panels[2].querySelector('.kt-detail'),false)};
const predictionSection=example=>{
  const wrap=h('div',{class:'kt-section'},heading(4,'Prediction'));
  const output=example.output,result=outcome(example);
  if(!result.ready){
    const status=str(output.status);
    return append(wrap,note(status==='unsupported'?'Prediction display is not supported for this task.':
      status==='invalid'?'The saved prediction for this example is invalid.':
      status==='limited'?'The saved prediction is too large to display.':'The prediction for this example was not saved.'))}
  if(str(output.status)==='partial')append(wrap,note(/target/.test(str(output.reason))?'The observed value was not saved for this example.':
    /prediction/.test(str(output.reason))?'The prediction was not saved for this example.':str(output.reason)));
  if(isLink){
    if(result.predictions===null)append(wrap,note('Ranked IDs were not saved.'));
    else if(!result.predictions.length)append(wrap,note('The ranked list is empty.'));
    else{
      const shown=result.predictions.slice(0,100);
      const long=shown.some(node=>str(node.id).length>6);
      append(wrap,note('Ranked '+idName+' values, best first · observed values are marked ✓'+
        (queryTopK&&queryTopK!==result.predictions.length?' · the query asks for the top '+queryTopK:'')),h('ol',{class:'kt-ranks'+(long?' kt-ranks-long':''),'aria-label':'Ranked '+idName+' values'},shown.map((node,i)=>{
        const match=result.hits.has(i),id=str(node.id);
        return h('li',{class:'kt-rank'+(match?' kt-match':'')},h('span',{text:'#'+(i+1)+(match?' ✓':'')}),
          h('strong',{title:id,text:shortText(id,long?48:24)}),match?h('span',{class:'kt-sr',text:'observed'}):null)})));
      if(result.predictions.length>100)append(wrap,note('Showing the first 100 of '+result.predictions.length+' ranked IDs.'))}
    if(result.targets===null)append(wrap,note('Observed IDs were not saved, so matches cannot be checked.'));
    else if(!result.targets.length)append(wrap,note('No observed IDs were recorded for this example.'));
    else append(wrap,h('div',{class:'kt-section'},h('div',{class:'kt-small kt-muted',text:'Observed '+idName+' values (test)'}),
      h('div',{class:'kt-row'},result.observed.slice(0,30).map(item=>h('span',{},h('b',{title:str(item.node.id),text:shortText(str(item.node.id),24)}),
        h('span',{class:item.rank!==undefined?'kt-hit':'kt-muted',text:item.rank!==undefined?' · rank '+(item.rank+1):
          item.comparable?' · not ranked':' · not comparable'})))),
      result.observed.length>30?note('+'+(result.observed.length-30)+' more observed IDs.'):null));
    return wrap}
  if(isRegression){
    const range=scalarRange;
    append(wrap,h('div',{class:'kt-compare'},
      h('span',{},h('span',{class:'kt-key','aria-hidden':'true',text:'● '}),'Predicted'),num(result.prediction,regressionPlaces),h('span'),
      h('span',{},h('span',{'aria-hidden':'true',text:'◆ '}),'Observed'),num(result.target,regressionPlaces),h('span'),
      h('span',{class:'kt-muted',text:'Error'}),result.error===null?missing():h('span',{class:'kt-num',text:signed(result.error,regressionPlaces)}),
      h('span',{class:'kt-muted kt-small',text:'predicted − observed'})),
      range?h('div',{},track(result.prediction,result.target,range,'Predicted '+(fmt(result.prediction)??'not recorded')+', observed '+(fmt(result.target)??'not recorded')),
        h('div',{class:'kt-axis'},h('span',{text:axisLabel(range[0])}),h('span',{text:axisLabel(range[1])}))):null,
      range?note('Shared scale across the saved examples.'):null,
      note('Target units are not recorded.'));
    return wrap}
  const rows=result.rows.slice(0,100);
  if(isBinary){const row=rows[0]||{};
    append(wrap,h('div',{class:'kt-compare'},
      h('span',{},h('span',{class:'kt-key','aria-hidden':'true',text:'● '}),'Positive-class probability'),numProb(row.probability),
      track(row.probability,null,[0,1],'Positive-class probability '+(prob(row.probability)??'not recorded')),
      h('span',{},h('span',{'aria-hidden':'true',text:'◆ '}),'Observed'),
      h('span',{class:'kt-num',text:row.observed===1?'1':row.observed===0?'0':'—'}),
      h('span',{class:'kt-muted',text:row.observed===1?'positive':row.observed===0?'negative':'not recorded'})),
      note('No decision threshold is recorded, so the probability is not labeled as a predicted class.'));
    return wrap}
  append(wrap,note(isMultilabel?'Each label has its own probability; several can be present.':'Class probabilities, most probable first.'),
    h('div',{class:'kt-classes',role:'table','aria-label':'Class probabilities'},
      h('span',{role:'columnheader',class:'kt-small kt-muted',text:isMultilabel?'Label':'Class'}),h('span',{role:'columnheader','aria-hidden':'true'}),
      h('span',{role:'columnheader',class:'kt-small kt-muted kt-num',text:'Prob.'}),h('span',{role:'columnheader',class:'kt-small kt-muted',text:'Observed'}),
      rows.map(row=>[h('span',{role:'cell',class:'kt-label',title:row.name,text:shortText(row.name,32)}),
        h('span',{role:'cell'},bar(row.probability,1)),h('span',{role:'cell',class:'kt-num'},numProb(row.probability)),
        h('span',{role:'cell',class:row.observed===1?'kt-hit':'kt-muted',text:row.observed===1?'✓ observed':row.observed===0?'absent':'—'})])));
  if(result.rows.length>100)append(wrap,note('Showing 100 of '+result.rows.length+' classes.'));
  return wrap};
const localSection=example=>{
  const wrap=h('div',{class:'kt-section'},heading(4,'What this prediction responded to'));
  const cells=example.cells;
  if(!cells.length)return append(wrap,note('No feature values were saved for this example.'));
  const sorted=cells.slice().sort((a,b)=>(b.score??-1)-(a.score??-1)||(a.id===example.root?-1:b.id===example.root?1:0));
  const scored=sorted.filter(cell=>cell.score!==null);
  const shown=S.allLocal?scored:scored.slice(0,8);
  const scale=Math.max(1,...scored.map(cell=>cell.score));
  append(wrap,note('Local sensitivity: relative to this example’s strongest feature (1). Not comparable across examples.'),
    isLink&&scopeNote()?note(scopeNote()):null);
  if(!scored.length)append(wrap,note('No local sensitivity scores were saved; values are listed without scores.'));
  const rows=(shown.length?shown:sorted.slice(0,8)).map(cell=>{
    const feature=byKey.get(featureKey(cell.table,cell.column));
    const name=feature?link([h('span',{class:'kt-muted',text:cell.table+'.'}),cell.column],()=>openFeature(feature.key),
      {class:'kt-link kt-inline',title:'Open '+cell.table+'.'+cell.column+' in Features'}):h('span',{},featureName(cell.table,cell.column));
    const where=recordName(example.store,example.tables,cell.id);
    return [[name,h('div',{class:'kt-muted kt-small',text:shortText(where,48),title:where})],valueText(cell),scoreCell(cell,scale)]});
  append(wrap,scoreTable(['Feature','Value','Local sensitivity'],rows));
  const hidden=scored.length-rows.length;
  if(hidden>0||S.allLocal)append(wrap,h('button',{type:'button',class:'kt-btn','aria-expanded':String(S.allLocal),
    onclick:()=>{S.allLocal=!S.allLocal;drawExample(panels[2].querySelector('.kt-detail'),false);
      panels[2].querySelector('.kt-local-toggle')?.focus()}},S.allLocal?'Show strongest only':'Show all '+scored.length+' scored values'));
  if(scored.length&&cells.length>scored.length)append(wrap,note(plural(cells.length-scored.length,'unscored value')+' appear under Related records.'));
  wrap.querySelector('.kt-btn')?.classList.add('kt-local-toggle');
  return wrap};
const recordsSection=example=>{
  const wrap=h('div',{class:'kt-section'},heading(4,'Related records'));
  const store=example.store,nodes=obj(store.nodes),root=example.root;
  if(!Object.keys(nodes).length)return append(wrap,note(example.body.subgraph_store?'No related records were saved.':
    'The sampled graph structure was not saved for this example.'));
  const children=new Map();let malformed=0;
  for(const [from,kinds] of Object.entries(obj(store.edges))){
    if(!kinds||typeof kinds!=='object'||Array.isArray(kinds)){malformed++;continue}
    for(const [relation,targets] of Object.entries(kinds)){
      if(!Array.isArray(targets)){malformed++;continue}
      for(const target of targets){if(typeof target!=='string'&&!finite(target)){malformed++;continue}
        if(!children.has(from))children.set(from,[]);children.get(from).push([relation,String(target)])}}}
  const depth=new Map(),via=new Map();
  if(has(nodes,root)){depth.set(root,0);const queue=[root];
    for(let i=0;i<queue.length;i++)for(const [relation,target] of children.get(queue[i])||[])
      if(!depth.has(target)&&has(nodes,target)){depth.set(target,depth.get(queue[i])+1);via.set(target,relation);queue.push(target)}}
  const groups=new Map(),cellsById=new Map();
  for(const cell of example.cells){if(!cellsById.has(cell.id))cellsById.set(cell.id,[]);cellsById.get(cell.id).push(cell)}
  for(const id of new Set([...Object.keys(nodes),...cellsById.keys()])){
    if(id===root)continue;
    const table=(example.tables.get(id)||[])[0]||'unknown table';
    const key=depth.has(id)?JSON.stringify([depth.get(id),table,via.get(id)]):JSON.stringify([Infinity,table,null]);
    if(!groups.has(key))groups.set(key,{hop:depth.get(id)??null,table,relation:via.get(id)??null,ids:[]});
    groups.get(key).ids.push(id)}
  const scale=Math.max(1,...example.cells.map(cell=>cell.score).filter(finite));
  const sorted=[...groups.values()].sort((a,b)=>(a.hop??Infinity)-(b.hop??Infinity)||a.table.localeCompare(b.table));
  append(wrap,note(sorted.length?'Records sampled around '+example.name+' when this prediction was explained.':
    'Only the example’s own record was saved.'));
  const container=h('div',{class:'kt-records'});
  for(const group of sorted){
    const columns=[...new Set(group.ids.flatMap(id=>(cellsById.get(id)||[]).map(cell=>cell.column)))].slice(0,6);
    const outside=group.hop===null;
    const rankOf=new Map();if(outside&&isLink)(ranking(example.output).predictions||[]).forEach((node,i)=>{
      const key=str(node.id_name)+'|'+str(node.id);if(!rankOf.has(key))rankOf.set(key,i)});
    const rankFor=id=>{const node=obj(nodes[id]);return rankOf.get(str(node.pkey_name)+'|'+str(node.pkey))};
    if(rankOf.size)group.ids.sort((a,b)=>(rankFor(a)??Infinity)-(rankFor(b)??Infinity));
    const title=outside?'Outside the sampled graph · '+group.table:
      'Hop '+group.hop+' · '+group.table+(group.relation?' via '+group.relation:'');
    const box=h('div',{class:'kt-record-group'},h('div',{class:'kt-spread'},h('span',{class:'kt-h4',text:title}),
      h('span',{class:'kt-muted kt-small',text:plural(group.ids.length,'record')})));
    const draw=all=>{
      box.querySelector('.kt-record-body')?.remove();
      const ids=all||group.ids.length<=8?group.ids:group.ids.slice(0,6),body=h('div',{class:'kt-record-body kt-section'});
      if(columns.length)append(body,h('div',{class:'kt-scroll'},h('table',{class:'kt-table'},
        h('thead',{},h('tr',{},h('th',{scope:'col',text:'Record'}),rankOf.size?h('th',{scope:'col',class:'kt-num',text:'Ranked'}):null,
          columns.map(column=>h('th',{scope:'col',text:column})))),
        h('tbody',{},ids.map(id=>h('tr',{},h('td',{text:midText(recordName(store,example.tables,id),36),title:recordName(store,example.tables,id)}),
          rankOf.size?h('td',{class:'kt-num'},rankFor(id)!==undefined?'#'+(rankFor(id)+1):missing('not in the ranked list')):null,
          columns.map(column=>{const cell=(cellsById.get(id)||[]).find(item=>item.column===column);
            if(!cell)return h('td',{},missing());
            return h('td',{},h('div',{class:'kt-record-cell'},valueText(cell),
              cell.score!==null?bar(cell.score,scale,'kt-thin'):null,
              h('span',{class:'kt-sr',text:cell.score!==null?'local sensitivity '+fmt(cell.score,2):'not scored'})))})))))));
      else append(body,p(recordList(store,example.tables,ids),'kt-small'));
      if(group.ids.length>8)append(body,h('button',{type:'button',class:'kt-btn','aria-expanded':String(all),
        onclick:event=>{draw(!all);box.querySelector('.kt-record-body .kt-btn')?.focus()}},all?'Show fewer':'Show all '+group.ids.length+' records'));
      append(box,body)};
    if(outside)append(box,note(isLink?'No recorded path connects these records to '+midText(example.name,32)+'. Link-prediction samples can include retrieved candidates here; they carry values but no scores.'+
      (rankOf.size?' Ranked matches a record to the ranked list by ID value.':''):'No recorded path connects these records to '+midText(example.name,32)+'.'));
    draw(false);append(container,box)}
  if(sorted.length)append(wrap,container);
  if(sorted.some(group=>group.ids.some(id=>(cellsById.get(id)||[]).some(cell=>cell.score!==null))))
    append(wrap,note('Bars under values show local sensitivity on the same scale as the list above.'));
  if(malformed)append(wrap,note(plural(malformed,'malformed relation')+' could not be shown.'));
  return wrap};

// ---------------------------------------------------------------- training panel
const trials=section('trials'),trialRows=list(trials.rows).map(obj);
const trialMetrics=row=>obj(row.comparison_metrics??row.final_validation_metrics);
const trialMetricNames=[...new Set(trialRows.flatMap(row=>Object.keys(trialMetrics(row))))].sort(metricOrder);
const trialOrigin=row=>row.comparison_source==='final_validation_metrics'?'final validation':
  row.comparison_source==='last_recorded_epoch'?'last recorded epoch '+str(row.comparison_epoch):'origin unverified';
const selectedTrial=trialRows.findIndex(row=>row.selected===true&&verified);
S.trial=selectedTrial>=0?selectedTrial:0;
const settingsOf=row=>{const part=obj(row.settings);
  return new Map(list(part.rows).map(obj).filter(item=>typeof item.key==='string'&&has(item,'value')&&
    (item.value===null||typeof item.value==='string'||typeof item.value==='boolean'||finite(item.value))).map(item=>[item.key,item]))};
const settingText=item=>item===undefined?null:item.value===null?'none':str(item.value);
const settingsView=()=>{
  const wrap=h('div',{class:'kt-section'});
  const row=trialRows[S.trial],current=settingsOf(row),reference=selectedTrial>=0&&selectedTrial!==S.trial?trialRows[selectedTrial]:null;
  append(wrap,heading(4,'Settings · '+str(row.trial)));
  const status=str(obj(row.settings).status);
  if(!current.size){append(wrap,note(row.settings?'Settings for this candidate are '+(status==='available'?'empty.':'not available ('+(status||'missing')+').'):
    'Settings were not saved with this training run.'));return wrap}
  const other=reference?settingsOf(reference):null;
  const keys=[...current.keys(),...(other?[...other.keys()].filter(key=>!current.has(key)):[])];
  const differs=key=>other&&settingText(current.get(key))!==settingText(other.get(key));
  if(other)append(wrap,note(keys.filter(differs).length?plural(keys.filter(differs).length,'setting')+' differ from the selected candidate '+str(reference.trial)+'.':
    'Recorded settings match the selected candidate '+str(reference.trial)+'.'));
  const orderKeys=other?[...keys.filter(differs),...keys.filter(key=>!differs(key))]:keys;
  append(wrap,h('div',{class:'kt-kv'+(other?' kt-kv-3':''),role:'table','aria-label':'Recorded settings'},
    h('div',{class:'kt-kv-row kt-kv-head',role:'row'},h('span',{role:'columnheader',text:'Setting'}),
      h('span',{role:'columnheader',text:str(row.trial)}),other?h('span',{role:'columnheader',text:str(reference.trial)+' (selected)'}):null),
    orderKeys.map(key=>{const item=current.get(key)||other.get(key);
      return h('div',{class:'kt-kv-row'+(differs(key)?' kt-strong':''),role:'row'},
        h('span',{role:'cell'},item.label||key,differs(key)?h('span',{class:'kt-muted',text:' · differs'}):null),
        h('span',{role:'cell'},settingText(current.get(key))??missing()),other?h('span',{role:'cell'},settingText(other.get(key))??missing()):null)})));
  return wrap};
const selectedTrialRow=()=>selectedTrial>=0?trialRows[selectedTrial]:null;
const selectedValidation=()=>{const row=selectedTrialRow();if(!row||!selectionBase||row.comparison_source!=='final_validation_metrics')return null;
  const value=obj(row.comparison_metrics)['val_'+selectionBase];return finite(value)?value:null};

renderers[3]=panel=>{
  const learning=h('section',{class:'kt-section'});
  const metricControl=historyMetrics.length>1?selectField('Metric',historyMetrics.map(name=>[name,metricLabel(name)]),S.metric,
    value=>{S.metric=value;showTab(3,false);panels[3].querySelector('select')?.focus()}):null;
  append(learning,h('div',{class:'kt-section-head'},heading(3,'Learning curves',{class:'kt-h3 kt-focus'}),metricControl),
    degraded('history'));
  if(!fits.length||!S.metric)append(learning,p(historyRows.length?'No numeric epoch values were recorded.':unavailable('history','Learning history'),'kt-muted'));
  else{
    const present=new Set(fits.flatMap(fit=>fitSeries(fit,S.metric).map(item=>item.secondary?'validation':'train')));
    append(learning,p(metricMeaning(S.metric)||metricLabel(S.metric),'kt-muted kt-small'),
      legend([present.has('train')?['●','kt-key','Train (solid line)']:null,present.has('validation')?['□','kt-key-observed','Validation (dashed line)']:null].filter(Boolean)),
      S.metric===selectionBase&&selectedValidation()!==null?note('The selected checkpoint’s epoch is not recorded; its validation '+metricLabel(S.metric)+' was '+fmt(selectedValidation())+'.'):null,
      verified&&fits[0]&&fits[0].trial.replace(/_refit_trainval$/,'')!==str(provenance.selected_trial)?
        note('This history is recorded for '+fits[0].trial+', not the selected candidate '+str(provenance.selected_trial)+'.'):null);
    const grid=h('div',{class:fits.length>1?'kt-grid':'kt-stack'});
    for(const fit of fits){
      const series=fitSeries(fit,S.metric);
      append(grid,h('div',{class:'kt-section kt-fit'},
        h('div',{class:'kt-spread'},heading(4,fitName(fit.fit)),h('span',{class:'kt-muted kt-small',text:fit.trial})),
        series.length?learningChart({series,title:metricLabel(S.metric)+', '+fitName(fit.fit),stateKey:fit.trial+'|'+fit.fit}):
          note('No values were recorded for this metric.'),
        fit.fit!=='original'&&!series.some(item=>item.secondary)?note('No validation curve: validation data was used to train this fit.'):null))}
    append(learning,grid);
    if(str(section('history').status)==='partial')append(learning,note('One fit’s history could not be read.'))}
  const candidates=h('section',{class:'kt-section'});
  append(candidates,h('div',{class:'kt-section-head'},heading(3,'Candidates'),
    finite(trials.total)?h('span',{class:'kt-muted kt-small',text:(finite(trials.completed)?trials.completed:'?')+' of '+plural(trials.total,'planned trial')+' completed'}):null));
  if(!trialRows.length)append(candidates,p(unavailable('trials','Candidate history'),'kt-muted'));
  else{
    const metric=trialMetricNames.includes('val_'+selectionBase)?'val_'+selectionBase:trialMetricNames[0]??null;
    const origins=[...new Set(trialRows.map(trialOrigin))];
    if(trialRows.length===1){
      const row=trialRows[0];
      append(candidates,p('One candidate was trained'+(row.selected===true&&verified?' and selected':'')+': '+str(row.trial)+
        (metric&&finite(trialMetrics(row)[metric])?' · validation '+metricLabel(metric)+' '+fmt(trialMetrics(row)[metric]):'')+
        (finite(row.num_epochs_completed)?' · '+plural(row.num_epochs_completed,'epoch'):'')+'.','kt-p'))}
    else{
      const box=listbox({label:'Candidates',items:trialRows,selected:S.trial,cls:'kt-feature kt-trial',
        onSelect:index=>{S.trial=index;const target=panels[3].querySelector('.kt-settings');
          target.replaceChildren(settingsView())},
        render:row=>{const value=metric?trialMetrics(row)[metric]:undefined,chosen=row.selected===true&&verified;
          return {label:str(row.trial)+(chosen?', selected':'')+', validation '+(metric?metricLabel(metric):'metric')+' '+(fmt(value)??'not recorded'),
            content:[h('span',{class:'kt-name'},h('span',{class:'kt-label',text:str(row.trial)}),
              h('span',{class:'kt-label kt-muted kt-small',text:[chosen?'Selected':null,finite(row.num_epochs_completed)?plural(row.num_epochs_completed,'epoch'):null,
                finite(row.elapsed_time)?fmt(row.elapsed_time,1)+' s':null].filter(Boolean).join(' · ')||' '})),
              h('span'),num(value)]}}});
      append(candidates,h('div',{class:'kt-narrow'},h('div',{class:'kt-col-head',style:{'grid-template-columns':'minmax(0,1fr) auto'}},h('span',{text:'Candidate'}),
        h('span',{text:'Validation '+(metric?metricLabel(metric):'metric')})),box));
      const unusual=trialRows.filter(row=>trialOrigin(row)!=='final validation');
      if(unusual.length)append(candidates,note(unusual.slice(0,8).map(row=>str(row.trial)+': '+(row.comparison_source==='last_recorded_epoch'?
        'value from '+trialOrigin(row):'no final validation value')).join('; ')+(unusual.length>8?'; …':'')+'. Other values are final validation scores.'))}
    append(candidates,h('div',{class:'kt-settings'},settingsView()))}
  const details=h('section',{class:'kt-section'},heading(3,'Run details'));
  const rows=[['Run',context.run?midText(context.run,40):null],
    ['Selected candidate',verified?str(provenance.selected_trial):'Not verified'+(provenance.selection_reason?': '+str(provenance.selection_reason):'')],
    ['Explained model',explainedFit?fitName(explainedFit)+' · '+str(provenance.explanation_trial):str(provenance.explanation_trial)||null],
    ['Selection metric',selectionBase?'Validation '+metricLabel(selectionBase)+(provenance.selection_direction==='max'?', higher is better':
      provenance.selection_direction==='min'?', lower is better':''):null],
    ['Explanations',['disabled','failed'].includes(provenance.generation_status)?str(provenance.generation_status):null],
    ['Task',TASKS[task]||task||null]].filter(([,value])=>value);
  append(details,h('div',{class:'kt-dl'},rows.map(([term,value])=>[h('span',{class:'kt-muted',text:term}),h('span',{text:value})])));
  append(panel,h('div',{class:'kt-stack kt-training',style:{gap:'28px'}},learning,candidates,details))};

// ---------------------------------------------------------------- shell
const header=h('header',{class:'kt-head'});
const queryText=query?h('span',{class:'kt-query-text'+(query.length>160||/\n/.test(query)?' kt-clamped':''),text:query}):null;
const queryBox=queryText?h('div',{class:'kt-query'},queryText):null;
const queryToggle=queryText&&queryText.classList.contains('kt-clamped')?h('button',{type:'button',class:'kt-link','aria-expanded':'false',
  onclick:()=>{const open=queryText.classList.toggle('kt-clamped');queryToggle.setAttribute('aria-expanded',String(!open));
    queryToggle.textContent=open?'Show full query':'Show less'}},'Show full query'):null;
const exact=h('input',{type:'checkbox',onchange:()=>{S.exact=exact.checked;content.classList.toggle('kt-exact',S.exact);showTab(S.tab,false)}});
const theme=selectField('Theme',[['auto','Automatic'],['light','Light'],['dark','Dark']],host.dataset.kumoTheme||'auto',value=>{
  if(value==='auto')delete host.dataset.kumoTheme;else host.dataset.kumoTheme=value});
append(header,h('div',{class:'kt-head-main'},heading(2,'Training explainability'),
    p([TASKS[task]||(task?task.replace(/_/g,' '):'Task not recorded'),context.target?'target '+str(context.target):null,
      'test split'].filter(Boolean).join(' · '),'kt-muted'),
    queryBox||note('Predictive query not recorded.'),queryToggle),
  h('div',{class:'kt-tools'},h('label',{class:'kt-check'},exact,'Exact values'),theme));
const notices=[];
if(!verified)notices.push(['Selection unverified.',str(provenance.selection_reason||'The selected candidate and fit could not be confirmed from the saved records.')]);
if(provenance.generation_status==='disabled')notices.push(['Explanations off.','This run did not generate explanations; training results are still shown.']);
if(provenance.generation_status==='failed')notices.push(['Explanations failed.','Generation failed during training; anything shown may be incomplete.']);
else if(provenance.generation_failure)notices.push(['Explanations failed.',str(provenance.generation_failure)]);
const counts=[null,features.length||null,examples.length||null,null];
const tablist=h('div',{class:'kt-tabs',role:'tablist','aria-label':'Training explanation views'});
TAB_NAMES.forEach((name,index)=>{
  const panel=h('div',{class:'kt-panel',role:'tabpanel',id:nextId('panel'),hidden:index!==0});
  const tab=h('button',{type:'button',class:'kt-tab',role:'tab',id:nextId('tab'),'aria-controls':panel.id,
    'aria-selected':String(index===0),tabindex:index===0?'0':'-1',onclick:()=>showTab(index,false)},name,
    counts[index]?h('span',{class:'kt-muted kt-count',style:{'font-weight':'400'},text:' '+counts[index]}):null);
  panel.setAttribute('aria-labelledby',tab.id);
  tabs.push(tab);panels.push(panel);append(tablist,tab)});
tablist.addEventListener('keydown',event=>{
  const current=tabs.indexOf(event.target.closest('[role=tab]'));if(current<0)return;
  const next={ArrowRight:current+1,ArrowLeft:current-1,Home:0,End:tabs.length-1}[event.key];
  if(next===undefined)return;
  event.preventDefault();const index=(next+tabs.length)%tabs.length;showTab(index,false);tabs[index].focus()},{signal});
append(content,header,...notices.map(([title,text])=>h('div',{class:'kt-notice',role:'note'},h('strong',{text:title}),h('span',{text}))),
  tablist,...panels);
showTab(0,false);
// Hosts that size an output frame to its content cannot keep a viewport-height pane in view.
try{if(window.self!==window.top)content.classList.add('kt-framed')}catch(error){content.classList.add('kt-framed')}
content.hidden=false;fallback.hidden=true;
onDispose(()=>{content.querySelectorAll('button,select,input').forEach(node=>node.disabled=true);
  content.hidden=true;fallback.hidden=false});
"""


def viewer_spec(payload: dict) -> dict:
    """Define one view for HTML and optional widget transports."""
    return dict(class_name='kumo-training', style=_STYLE, script=_SCRIPT,
                body=static_summary(payload) +
                '<div class="kt-content" hidden=""></div>', data=payload)
