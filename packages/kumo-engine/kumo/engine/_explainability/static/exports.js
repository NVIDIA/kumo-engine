/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Feature-local exports reuse the live SVGs and the same widget initializer. */
function widgetExports(root, {host, data, signal, onDispose, css, source, render}) {
  const head = root.querySelector('.kx-head') || root;
  const title = head.querySelector('.kx-query');
  const headline = H('div', 'kx-headline', head);
  head.prepend(headline);
  if (title) headline.append(title);
  const actions = H('div', 'kx-exports', headline);
  const pdf = H('button', null, actions, 'PDF');
  const html = H('button', null, actions, 'HTML');
  for (const [button, label] of [[pdf, 'Export PDF'], [html, 'Export HTML']]) {
    button.setAttribute('aria-label', label);
    const icon = S('svg', {viewBox: '0 0 20 20', 'aria-hidden': 'true'}, button);
    S('path', {d: 'M10 2v10m-4-4 4 4 4-4M3 13v4h14v-4'}, icon);
    button.prepend(icon);
  }
  pdf.type = html.type = 'button';
  pdf.title = 'Print all chart views on the current page or save them as PDF';
  html.title = 'Download the full interactive widget, including its saved data';
  const status = H('span', 'kx-export-status', actions);
  status.setAttribute('role', 'status');
  root.addEventListener('keydown', event => { if (event.key === 'Escape') status.replaceChildren(); }, {signal});
  root.addEventListener('click', event => { if (!actions.contains(event.target)) status.replaceChildren(); }, {signal});
  const urls = new Set();
  onDispose(() => { for (const url of urls) URL.revokeObjectURL(url); });
  // The UI state, copied so an export or print copy never shares it.
  const capture = () => JSON.parse(JSON.stringify(root.state.values));
  function documentBase(title, print = false) {
    const doc = document.implementation.createHTMLDocument(title);
    const charset = doc.createElement('meta');
    charset.setAttribute('charset', 'utf-8'); doc.head.append(charset);
    const meta = doc.createElement('meta');
    meta.name = 'viewport'; meta.content = 'width=device-width, initial-scale=1';
    doc.head.append(meta);
    const style = doc.createElement('style');
    style.textContent = 'html{color-scheme:' + (print ? 'light' : 'light dark') +
      '}body{margin:clamp(8px,2vw,24px);background:light-dark(white,#1a1c1f)}' +
      '.kumo-widget{max-width:1200px;margin:auto}[hidden]{display:none!important}' + css;
    doc.head.append(style);
    return doc;
  }
  function download(doc, name) {
    const blob = new Blob(['<!doctype html>\n', doc.documentElement.outerHTML], {type: 'text/html;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    urls.add(url);
    const a = H('a', null, actions);
    a.href = url; a.download = name; a.click(); a.remove();
    const timer = setTimeout(() => { URL.revokeObjectURL(url); urls.delete(url); }, 60000);
    onDispose(() => clearTimeout(timer));
  }
  html.addEventListener('click', () => {
    try {
      const doc = documentBase('Kumo explanations');
      const target = doc.createElement('div');
      target.id = 'kumo-export'; target.className = host.className;
      const fallback = host.querySelector(':scope > .kx-static').cloneNode(true);
      fallback.hidden = false; target.append(fallback); doc.body.append(target);
      const payload = doc.createElement('script');
      payload.type = 'application/json'; payload.id = 'kumo-export-data';
      payload.textContent = JSON.stringify({data, saved: capture()}).replaceAll('<', '\\u003c');
      doc.body.append(payload);
      const script = doc.createElement('script');
      script.textContent = 'const host=document.getElementById("kumo-export");' +
        'const {data,saved}=JSON.parse(document.getElementById("kumo-export-data").textContent);' +
        'const controller=new AbortController(), cleanups=[];' +
        'addEventListener("pagehide",event=>{if(!event.persisted){controller.abort();cleanups.forEach(fn=>fn())}});' +
        '(' + source + ')({host,widgetId:host.id,data,signal:controller.signal,onDispose:fn=>cleanups.push(fn)},saved);';
      doc.body.append(script);
      download(doc, 'kumo-' + (data.kind || 'training') + '-explanations.html');
      status.textContent = 'HTML downloaded with the widget’s saved data.';
    } catch (error) { status.textContent = 'HTML export failed. Please retry.'; console.error(error); }
  }, {signal});
  function printCopy() {
    const staging = H('div', host.className, host.parentNode);
    staging.id = host.id + '-print';
    staging.style.cssText = 'position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none;width:' + root.getBoundingClientRect().width + 'px';
    H('div', 'kx-static', staging);
    const controller = new AbortController(), cleanups = [];
    const clone = element => {
      element.querySelectorAll('.kx-chart').forEach(chart => chart.redraw?.());
      const copy = element.cloneNode(true);
      copy.querySelectorAll('select').forEach((el, i) => {
        const selected = element.querySelectorAll('select')[i].selectedIndex;
        [...el.options].forEach((option, index) => option.toggleAttribute('selected', index === selected));
      });
      copy.querySelectorAll('input').forEach((el, i) => {
        const original = element.querySelectorAll('input')[i];
        el.setAttribute('value', original.value); el.toggleAttribute('checked', original.checked);
      });
      return copy;
    };
    try {
      render({host: staging, widgetId: staging.id, data, signal: controller.signal,
        onDispose: fn => cleanups.push(fn)}, capture());
      const rendered = staging.querySelector('.kx'), copy = clone(rendered);
      const panel = rendered.querySelector('.kx-panel:not([hidden])');
      const pages = [];
      for (const card of panel.querySelectorAll('[data-card]')) {
        const choices = [...card.querySelectorAll('.kx-switch button[data-view]')];
        const selected = choices.find(button => button.getAttribute('aria-pressed') === 'true');
        const current = clone(card);
        for (const choice of choices.length ? choices : [null]) {
          if (choice && choice !== selected) choice.click();
          const page = choice === selected || !choice ? current : clone(card);
          if (choice) {
            page.querySelector('.kx-title').append(' · ' + choice.textContent);
            page.querySelectorAll('.kx-switch, .kx-switch-row').forEach(el => el.remove());
          }
          pages.push(page);
        }
      }
      copy.querySelector('.kx-panel:not([hidden])').replaceChildren(...pages);
      return copy;
    } finally {
      controller.abort(); cleanups.forEach(fn => fn()); staging.remove();
    }
  }
  pdf.addEventListener('click', () => {
    try {
      const name = data.tabs[root.state.values.tab]?.label || 'Explanations';
      const doc = documentBase('Kumo · ' + name, true);
      const wrapper = doc.createElement('div'); wrapper.className = host.className;
      const copy = printCopy();
      copy.querySelector('.kx-exports')?.remove();
      copy.querySelectorAll('.kx-panel[hidden]').forEach(el => el.remove());
      copy.querySelectorAll('.kx-tab[aria-selected="false"]').forEach(el => el.remove());
      // SVGs keep their measured coordinate system but fit the printed page.
      copy.querySelectorAll('.kx-chart svg').forEach(svg => {
        if (!svg.hasAttribute('viewBox')) svg.setAttribute('viewBox', '0 0 ' + svg.getAttribute('width') + ' ' + svg.getAttribute('height'));
        svg.removeAttribute('width'); svg.removeAttribute('height');
        svg.style.width = '100%'; svg.style.height = 'auto';
      });
      wrapper.append(copy); doc.body.append(wrapper);
      const style = doc.createElement('style');
      style.textContent = '@page{size:landscape;margin:12mm}' +
        '*{animation:none!important;transition:none!important;print-color-adjust:exact;-webkit-print-color-adjust:exact}' +
        '.kx{container-type:normal}.kx-chart,.kx-sg-canvas,.kx-sg-detail{break-inside:avoid}.kx-card-top{break-after:avoid}.kx-table-wrap{overflow:visible;max-height:none}' +
        '.kx-sg-main{display:block}.kx-sg-detail{margin-top:12px;max-height:none;overflow:visible}' +
        '.kx-sg-canvas{max-height:none!important}' +
        '.kx-sg-filters{display:none}.kx-print-action{margin:0 0 16px;font:16px system-ui}' +
        '@media print{.kx-print-action{display:none}body{margin:0}}';
      doc.head.append(style);
      const button = doc.createElement('button');
      button.className = 'kx-print-action'; button.textContent = 'Print / Save as PDF';
      doc.body.prepend(button);
      const script = doc.createElement('script');
      script.textContent = 'document.querySelector(".kx-print-action").addEventListener("click",()=>window.print());' +
        'addEventListener("load",()=>{document.fonts.ready.then(()=>window.print())},{once:true});';
      doc.body.append(script);
      let popup;
      try {
        popup = window.open('', '_blank');
        if (!popup) throw new Error('Pop-up blocked');
        popup.opener = null;
        popup.document.open(); popup.document.write('<!doctype html>\n' + doc.documentElement.outerHTML); popup.document.close();
        status.textContent = 'Choose Save as PDF in the print dialog. ';
        const retry = H('button', null, status, 'Download print page');
        retry.type = 'button';
        retry.addEventListener('click', () => download(doc, 'kumo-current-page-print.html'), {signal});
      } catch (_) {
        popup?.close();
        download(doc, 'kumo-current-page-print.html');
        status.textContent = 'Open the downloaded print page, then choose Save as PDF.';
      }
    } catch (error) { status.textContent = 'PDF export could not open. Allow pop-ups and retry.'; console.error(error); }
  }, {signal});
}
