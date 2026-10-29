# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared notebook theme, controls, and output cleanup."""

import json
from html import escape
from string import Template
from types import MappingProxyType
from uuid import uuid4

THEME = MappingProxyType({
    'font': 'Arial, Helvetica, sans-serif',
    'foreground': '#202020', 'muted': '#687078', 'accent': '#386000',
    'heading': '#386000', 'header_background': '#f4f8ec',
    'tooltip_header_border': '#e3e7df', 'grid': '#d9ddd5',
    'edge': '#a1a1a1', 'edge_active': '#6e9947', 'focus': '#598d13',
    'card_border': '#c5c9cd', 'card_background': 'white',
    'role_background': '#f7f7f7', 'surface': '#f5f6f4',
    'surface_border': '#e1e4df', 'row_highlight': '#edf6df',
    'chrome_hover': '#e8e8e8', 'chrome_foreground': '#555',
    'chrome_shadow': '0 1px 4px #0000001a', 'background': 'white',
    'tooltip_background': 'white', 'tooltip_border': '#d1d6cb',
    'tooltip_shadow': '0 8px 28px #00000018,0 1px 4px #0000000a',
})
_DARK = {
    'foreground': '#e2e5e8', 'muted': '#a0a8b0', 'accent': '#a3d85b',
    'heading': '#f0f2f3', 'header_background': '#292e33',
    'tooltip_header_border': '#3a4148', 'grid': '#292e33',
    'edge': '#69737d', 'edge_active': '#a3d85b', 'focus': '#a3d85b',
    'card_border': '#3f474f', 'card_background': '#202428',
    'role_background': '#272d32', 'surface': '#2a3036',
    'surface_border': '#3b434b', 'row_highlight': '#36452c',
    'chrome_hover': '#343c43', 'chrome_foreground': '#bdc5cd',
    'background': '#15181b', 'tooltip_background': '#242a2f',
    'tooltip_border': '#4c565f',
}
CSS_THEME = {key: f'var(--kumo-{key.replace("_", "-")})'
             for key in THEME if key != 'font'} | {'font': THEME['font']}


_STYLE = Template(r"""
.kumo-widget{position:relative;display:flex;flex-direction:column;
max-width:100%;padding:8px;box-sizing:border-box;background:$background;
color:$foreground;font:14px $font}
.kumo-widget [hidden]{display:none!important}
:where(.kumo-widget-toolbar){display:flex;align-items:center;gap:6px;
flex:none;margin-bottom:16px}
:where(.kumo-widget) .kumo-widget-button{box-sizing:border-box;display:inline-flex;
align-items:center;justify-content:center;flex:0 0 32px;
width:32px!important;min-width:32px!important;max-width:32px!important;
height:32px!important;min-height:32px!important;max-height:32px!important;
padding:6px;margin:0;background:$background;color:$chrome_foreground;
border:0;border-radius:16px;cursor:pointer;font:18px/18px $font}
.kumo-widget-toolbar>.kumo-widget-button{box-shadow:$chrome_shadow}
:where(.kumo-widget) .kumo-widget-button:hover{background:$chrome_hover}
:where(.kumo-widget) .kumo-widget-button:focus-visible{
outline:2px solid $focus;outline-offset:2px}
:where(.kumo-widget) .kumo-widget-button>svg{display:block;flex:none;
width:18px!important;height:18px!important}
""").substitute(CSS_THEME)
_STYLE = (
    '.kumo-widget{color-scheme:light dark;' + ''.join(
        f'--kumo-{key.replace("_", "-")}:light-dark({THEME[key]},{dark});'
        for key, dark in _DARK.items()) +
    '--kumo-chrome-shadow:0 1px 4px light-dark(#0000001a,#00000040);'
    '--kumo-tooltip-shadow:0 8px 28px light-dark(#00000018,#00000050),'
    '0 1px 4px light-dark(#0000000a,#00000030);}'
    '.kumo-widget[data-kumo-theme="light"]{color-scheme:light}'
    '.kumo-widget[data-kumo-theme="dark"]{color-scheme:dark}'
    + _STYLE)


def button(cls: str, label: str, icon: str = '', text: str = '') -> str:
    """Build a labeled button, disabled until initialization."""
    graphic = (f'<svg width="18" height="18" viewBox="0 0 24 24" '
               f'aria-hidden="true"><path d="{escape(icon)}" fill="none" '
               'stroke="currentColor" stroke-width="1.5" '
               'stroke-linecap="round" stroke-linejoin="round"/></svg>'
               if icon else escape(text))
    label = escape(label)
    return (f'<button type="button" class="kumo-widget-button {escape(cls)}" '
            f'aria-label="{label}" title="{label}" disabled="">{graphic}'
            '</button>')


def render_widget(body: str, *, style: str = '', script: str = '',
                  data: dict | None = None, class_name: str = '',
                  attributes: dict[str, str] | None = None) -> str:
    """Compose trusted feature markup/code with escaped data and local cleanup.

    Feature scripts use host, widgetId, data, signal and onDispose(callback).
    Root attributes are data-* or aria-* names; controls are enabled by features.
    """
    attributes = attributes or {}
    if any(not name.startswith(('data-', 'aria-')) or not all(
            c.isascii() and (c.isalnum() or c == '-') for c in name)
           for name in attributes):
        raise ValueError('Widget attributes must be data-* or aria-* names.')
    attrs = ''.join(f' {key}="{escape(value)}"'
                    for key, value in attributes.items())
    payload = json.dumps(data or {}, ensure_ascii=True, separators=(',', ':'))
    for char, encoded in [('<', r'\u003c'), ('>', r'\u003e'), ('&', r'\u0026')]:
        payload = payload.replace(char, encoded)
    return (
        f'<div id="kumo-{uuid4().hex}" '
        f'class="kumo-widget {escape(class_name)}"{attrs}>'
        f'<style>{_STYLE}{style}</style>{body}'
        f'<script type="application/json" class="kumo-widget-data">'
        f'{payload}</script><script>//<![CDATA[\n(()=>{{'
        'const host=document.currentScript.parentElement,widgetId=host.id;'
        'const data=JSON.parse(host.querySelector(":scope>.kumo-widget-data").textContent);'
        'const controller=new AbortController(),signal=controller.signal;'
        'const cleanups=[],onDispose=callback=>cleanups.push(callback);'
        'let disposed=false;'
        'const dispose=()=>{if(disposed)return;disposed=true;'
        'controller.abort();removalObserver.disconnect();'
        'host.dataset.disposed="true";'
        'for(const cleanup of cleanups.splice(0)){try{cleanup()}catch(error){console.error(error)}}};'
        'const removalObserver=new MutationObserver(records=>{'
        'if(records.some(record=>record.removedNodes.length)&&!host.isConnected)dispose();});'
        'removalObserver.observe(host.getRootNode(),{childList:true,subtree:true});'
        'try{' + script + '}catch(error){dispose();throw error}'
        '})();\n//]]>\n</script></div>')
