# Widget validation

## Training widget

Validated on Linux on 2026-11-19 using retained classification, regression, forecasting and link-prediction artifacts plus synthetic edge cases.
No training or prediction jobs ran.

| Check | Result |
| --- | --- |
| `python -m pytest packages/kumo-engine/test/test_explainability.py -q` | 128 passed |
| Changed Python files: flake8 | Passed |
| Frontend JavaScript: `node --check` | Passed |
| Chromium, Firefox and WebKit export checks | Passed in all three browsers |

The browser check verifies two plotted epoch SVGs when history is selected and when it is not, preserving the selected metric and experiments across view switches and matching plotted values to the live chart.
It also checks all active-page views, expanded importance and linked-column HTML restoration, unchanged notebook scroll position, the single-tab Evaluation title, all-negative/mixed forecast values inside the SVG, hidden heatmap outlier settings, and sparse distribution geometry spanning 500,051 bins.
Separated-range and constant-axis heatmaps conserve their counts within a 10,000-cell allocation bound.
The original PDF regression reproduced as two live epoch plots and zero printed plots; the corrected export contains both.

Run the browser checks against HTML from retained binary training, forecasting and regression views:

```sh
NODE_PATH=<directory-containing-playwright> node packages/kumo-engine/test/widget_exports.cjs training.html forecasting.html regression.html
```

Save UTF-8 HTML documents with a `<meta charset="utf-8">` declaration, wrapping the fragment returned by this checkout’s `kumo.engine._explainability.open_run(retained_run).html()`; the classification input must include epoch history and more than ten importance entries.
The script launches installed Chromium, Firefox and WebKit browsers and fails on browser errors or missing/invalid plot geometry.

Frontend source fingerprint (SHA-256 over sorted static filenames, a null byte and each file’s contents): `bc1ab908778b4a5db6e6bd65bae117e50269cadfe2f0ee6e5bc76b2baadfa0a0`.
These checks cover local browser behavior, including offline HTML and the printable document; they do not prove native notebook print dialogs, comment-overlay placement or every host/OS combination.
