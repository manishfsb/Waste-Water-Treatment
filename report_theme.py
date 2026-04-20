"""
Shared dark-mode CSS and JS for all HTML reports.

Usage in any report generator:
    from report_theme import DARK_MODE_HEAD, dark_mode_css

    # In your <head> tag, replace <style>...</style> with:
    #   {DARK_MODE_HEAD}
    #
    # In your CSS string, call dark_mode_css(base_css) to wrap it with
    # automatic light/dark variable switching.

The resulting page:
  - Follows the OS / browser colour-scheme preference automatically
    (prefers-color-scheme media query).
  - Shows a small sun/moon toggle button (top-right corner) to override.
  - Remembers the override in localStorage so it persists across page loads.
"""

# ── CSS custom-property palette ───────────────────────────────────────────────
# All colour tokens are defined on :root (light) and overridden on [data-theme="dark"].
# Any rule in the existing CSS that references bg/text/card colours should use
# these variables — or simply call dark_mode_css() which rewrites the common ones.

_VARIABLES = """
:root {
  --bg:           #F5F5F5;
  --bg-page:      #F0F2F5;
  --card:         #FFFFFF;
  --card-shadow:  rgba(0,0,0,0.10);
  --text:         #222222;
  --text-muted:   #555555;
  --text-meta:    #888888;
  --border:       #DDDDDD;
  --border-light: #E8E8E8;
  --table-even:   #F9F9F9;
  --obs-bg:       #EBF3FB;
  --interp-bg:    #FFFBF0;
  --interp-bdr:   #F0D080;
  --toc-bg:       #E8F0FE;
  --details-bg:   #FAFBFD;
  --details-bdr:  #D0D7E3;
  --summary-hover:#EEF3FA;
  --input-bg:     #FFFFFF;
  color-scheme: light;
}

[data-theme="dark"] {
  --bg:           #1A1D23;
  --bg-page:      #13151A;
  --card:         #23272F;
  --card-shadow:  rgba(0,0,0,0.40);
  --text:         #E2E4E9;
  --text-muted:   #A0A6B2;
  --text-meta:    #6B7280;
  --border:       #383C46;
  --border-light: #2E3240;
  --table-even:   #1E2128;
  --obs-bg:       #1A2535;
  --interp-bg:    #1F1E14;
  --interp-bdr:   #5A4800;
  --toc-bg:       #161C2D;
  --details-bg:   #1C2030;
  --details-bdr:  #2E3450;
  --summary-hover:#1A2238;
  --input-bg:     #23272F;
  color-scheme: dark;
}
"""

# ── Toggle button ─────────────────────────────────────────────────────────────
_TOGGLE_CSS = """
#theme-toggle {
  position: fixed;
  top: 16px;
  right: 20px;
  z-index: 9999;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 5px 12px;
  cursor: pointer;
  font-size: 16px;
  box-shadow: 0 2px 8px var(--card-shadow);
  color: var(--text);
  transition: background 0.2s, border-color 0.2s;
  user-select: none;
  line-height: 1;
}
#theme-toggle:hover { border-color: var(--text-muted); }
"""

# ── JS — respects OS preference, allows manual override stored in localStorage ─
_TOGGLE_JS = """
<script>
(function() {
  var stored = localStorage.getItem('theme');
  var osDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  var active = stored ? stored : (osDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', active);

  // React to OS changes if user has not manually overridden
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
    if (!localStorage.getItem('theme')) {
      document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      updateIcon();
    }
  });

  function updateIcon() {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.textContent = isDark ? '☀' : '☾';
    btn.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
  }

  document.addEventListener('DOMContentLoaded', function() {
    var btn = document.createElement('button');
    btn.id = 'theme-toggle';
    btn.setAttribute('aria-label', 'Toggle dark mode');
    document.body.appendChild(btn);
    updateIcon();

    btn.addEventListener('click', function() {
      var current = document.documentElement.getAttribute('data-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateIcon();
    });
  });
})();
</script>
"""

# ── Common structural CSS that every report shares ────────────────────────────
# These rules reference the CSS variables defined above. Each report appends
# its own accent colours (h1/h2 colours, table header colours, etc.) on top.
COMMON_CSS = """
  body {
    font-family: Calibri, Arial, sans-serif;
    margin: 0;
    background: var(--bg-page);
    color: var(--text);
    transition: background 0.25s, color 0.25s;
  }
  .card {
    background: var(--card);
    box-shadow: 0 2px 8px var(--card-shadow);
    border-radius: 8px;
  }
  .summary-table th, .summary-table td { border-color: var(--border); }
  .summary-table tbody tr:nth-child(even) { background: var(--table-even); }
  .table-note  { color: var(--text-muted); }
  .meta        { color: var(--text-meta); font-size: 13px; }
  .obs-card    { background: var(--obs-bg); }
  .interp      { background: var(--interp-bg); border-color: var(--interp-bdr); }
  .toc         { background: var(--toc-bg); }
  details      { background: var(--details-bg); border-color: var(--details-bdr); }
  details > summary:hover { background: var(--summary-hover); }
  img          { background: var(--card); }
"""


def dark_mode_css(report_css: str) -> str:
    """
    Prepend the variable palette + common structural overrides to an existing
    CSS string.  The report's own rules (accent colours, layout) follow and
    take precedence where needed.

    Example:
        CSS = dark_mode_css(\"\"\"
          body { ... }
          .card { background: white; ... }   # will be overridden by var(--card)
        \"\"\")
    """
    return _VARIABLES + _TOGGLE_CSS + COMMON_CSS + report_css


# ── Convenience: the full <head> block including the JS ───────────────────────
# Insert this between <head> and </head> (after your own <style> block).
DARK_MODE_JS = _TOGGLE_JS
