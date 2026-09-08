import argparse
import json
import re
import html as htmllib
from pathlib import Path
from datetime import datetime

"""
Combine all pytest-html reports in a directory into a single HTML report.

Usage:
    python combine_html_reports.py <input_directory> [-o <output>]

Output behavior:
    - If -o is omitted, the report is saved as combined.html in the input directory.
    - If <output> is only a filename, the report is saved with that name in the input directory.
    - If <output> is a directory, the report is saved there as combined.html.
    - If <output> is a file path, the report is saved using the specified path and filename.

Examples:
    python combine_html_reports.py ./reports
    python combine_html_reports.py ./reports -o test_summary.html
    python combine_html_reports.py ./reports -o ./combined_reports
    python combine_html_reports.py ./reports -o ./combined_reports/test_summary.html
"""

ALL_RESULTS = ["failed", "error", "xpassed", "skipped", "xfailed", "rerun", "retried", "passed"]


def extract_css(html):
    match = re.search(r"<head[^>]*>(.*?)</head>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    head = match.group(1)
    styles = re.findall(r"<style[^>]*>.*?</style>", head, flags=re.IGNORECASE | re.DOTALL)
    return "\n".join(styles)


def extract_json_blob(html):
    match = re.search(r'data-jsonblob="([^"]+)"', html)
    if not match:
        return None
    return json.loads(htmllib.unescape(match.group(1)))


def render_extras(extras):
    parts = []
    for extra in extras:
        fmt = extra.get("format_type", "")
        content = extra.get("content", "")
        name = extra.get("name") or ""
        label = f'<div class="extra-name">{htmllib.escape(name)}</div>' if name else ""
        if fmt == "html":
            inner = re.sub(r"</?tr[^>]*>|</?td[^>]*>", "", content)
            parts.append(f'<div class="extra-item">{label}{inner}</div>')
        elif fmt in ("image", "png", "jpg", "jpeg", "gif", "svg"):
            parts.append(f'<div class="extra-item">{label}<img src="{content}" /></div>')
        elif fmt == "url":
            link_label = name or content
            parts.append(f'<div class="extra-item"><a href="{content}" target="_blank">{htmllib.escape(link_label)}</a></div>')
        elif content:
            parts.append(f'<div class="extra-item">{label}<pre>{htmllib.escape(str(content))}</pre></div>')
    return "\n".join(parts)


def render_test_tbody(test, row_index):
    result = test.get("result", "unknown").lower()
    test_id = test.get("testId", "")
    duration = test.get("duration", "")
    log = test.get("log", "")
    extras = test.get("extras", [])

    extras_html = render_extras(extras)

    log_html = ""
    if log:
        log_html = f"""
        <div class="logwrapper">
          <div class="logexpander"></div>
          <div class="log">{log}</div>
        </div>"""

    has_details = bool(log or extras_html)
    details_hidden = not has_details or result in ("passed", "skipped", "xfailed")
    collapsed_class = "collapsed" if details_hidden else ""

    tbody_id = f"tbody_{row_index}"
    extra_row_id = f"{tbody_id}-extra"

    return f"""
    <tbody class="results-table-row {result}" id="{tbody_id}">
      <tr class="collapsible" data-target="{extra_row_id}">
        <td class="col-result {result} {collapsed_class}">{test.get('result', '')}</td>
        <td class="col-testId">{htmllib.escape(test_id)}</td>
        <td class="col-duration">{htmllib.escape(duration)}</td>
      </tr>
      <tr class="extras-row {"hidden" if details_hidden else ""}" id="{extra_row_id}">
        <td class="extra" colspan="3">
          {extras_html}
          {log_html}
        </td>
      </tr>
    </tbody>"""


def render_environment(env):
    if not env:
        return ""
    rows = []
    for key, value in env.items():
        if isinstance(value, dict):
            cell = "<ul>" + "".join(
                f"<li>{htmllib.escape(str(k))}: {htmllib.escape(str(v))}</li>"
                for k, v in value.items()
            ) + "</ul>"
        else:
            cell = htmllib.escape(str(value))
        rows.append(f"<tr><td>{htmllib.escape(str(key))}</td><td>{cell}</td></tr>")
    return "\n".join(rows)


def combine_html_files(input_directory, output=None):
    input_directory = Path(input_directory)
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")

    if output is None:
        output_file = input_directory / "combined.html"
    else:
        output = Path(output)

        if output.is_dir() or (not output.exists() and not output.suffix):
            output.mkdir(parents=True, exist_ok=True)
            output_file = output / "combined.html"
        else:
            if output.parent == Path("."):
                output_file = input_directory / output.name
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                output_file = output

    html_files = sorted(
        f for f in input_directory.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".html"
        and f.resolve() != output_file.resolve()
    )

    if not html_files:
        raise ValueError(f"No HTML files found in: {input_directory}")

    shared_css = ""
    environment = {}
    counts = {r: 0 for r in ALL_RESULTS}
    all_rows = []
    row_index = 0
    num_reports = 0

    for html_file in html_files:
        print(f"Processing: {html_file.name}")
        with open(html_file, "r", encoding="utf-8") as f:
            raw = f.read()

        if not shared_css:
            shared_css = extract_css(raw)

        data = extract_json_blob(raw)
        if not data:
            print(f"  WARNING: no JSON blob found, skipping {html_file.name}")
            continue

        if not environment and data.get("environment"):
            environment = data["environment"]

        file_had_tests = False
        for test_cases in data.get("tests", {}).values():
            for test in test_cases:
                result_key = test.get("result", "").lower()
                if result_key in counts:
                    counts[result_key] += 1
                all_rows.append(render_test_tbody(test, row_index))
                row_index += 1
                file_had_tests = True

        if file_had_tests:
            num_reports += 1

    total = sum(counts.values())

    # --- Filter checkboxes ---
    def filter_checkbox(result, label):
        disabled = " disabled" if counts[result] == 0 else ""
        checked = " checked" if counts[result] > 0 else ""
        return (
            f'<input{checked} class="filter" name="filter_checkbox" type="checkbox" '
            f'data-test-result="{result}"{disabled}>'
            f'<span class="{result}">{counts[result]} {label}</span>'
        )

    filters_html = ", ".join([
        filter_checkbox("failed", "Failed"),
        filter_checkbox("passed", "Passed"),
        filter_checkbox("skipped", "Skipped"),
        filter_checkbox("xfailed", "Expected failures"),
        filter_checkbox("xpassed", "Unexpected passes"),
        filter_checkbox("error", "Errors"),
        filter_checkbox("rerun", "Reruns"),
        filter_checkbox("retried", "Retried"),
    ])

    # --- Environment table ---
    env_rows_html = render_environment(environment)
    env_section = f"""
  <div id="environment-header">
    <h2>Environment</h2>
  </div>
  <table id="environment">
    {env_rows_html}
  </table>""" if env_rows_html else ""

    all_rows_html = "\n".join(all_rows)
    total = sum(counts.values())
    generated_at = datetime.now().strftime("%d-%b-%Y at %H:%M:%S")

    combined_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Combined Test Report</title>
  {shared_css}
  <style>
    .extra-item {{ margin: 6px 0; }}
    .extra-item img {{ max-width: 100%; display: block; }}
    .extra-name {{ font-weight: bold; font-size: 11px; color: #555; margin-bottom: 3px; }}

    .hidden {{ display: none !important; }}

    tr.extras-row td {{ padding: 8px; background: #f9f9f9; }}

    #environment {{ margin-bottom: 20px; }}
    #environment td {{
      padding: 5px;
      border: 1px solid #e6e6e6;
      vertical-align: top;
    }}
    #environment tr:nth-child(odd) {{ background-color: #f6f6f6; }}
    #environment ul {{ margin: 0; padding: 0 20px; }}

    .summary {{ margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>Combined Test Report</h1>
  <p>Report generated on {generated_at} &mdash; {total} test(s) from {num_reports} report(s)
     by <a href="https://pypi.python.org/pypi/pytest-html">pytest-html</a>.</p>

  {env_section}

  <div class="summary">
    <div class="summary__data">
      <p class="filter">(Un)check the boxes to filter the results.</p>
      <div class="controls">
        <div class="filters">
          {filters_html}
        </div>
        <div class="collapse">
          <button id="show_all_details">Show all details</button>&nbsp;/&nbsp;<button id="hide_all_details">Hide all details</button>
        </div>
      </div>
    </div>
  </div>

  <table id="results-table">
    <thead id="results-table-head">
      <tr>
        <th class="sortable" data-column-type="result">Result</th>
        <th class="sortable" data-column-type="testId">Test</th>
        <th class="sortable" data-column-type="duration">Duration</th>
      </tr>
    </thead>
    {all_rows_html}
    <tbody id="not-found-row" class="hidden">
      <tr>
        <td colspan="3" id="not-found-message">No results found. Check the filters.</td>
      </tr>
    </tbody>
  </table>

  <script>
    (function () {{
      const ALL_RESULTS = {json.dumps(ALL_RESULTS)};
      const checkboxes = document.querySelectorAll('input.filter[data-test-result]');
      const notFoundRow = document.getElementById('not-found-row');

      function applyFilters() {{
        const active = new Set(
          [...checkboxes].filter(cb => cb.checked).map(cb => cb.dataset.testResult)
        );

        let totalVisible = 0;

        // Show/hide test rows
        document.querySelectorAll('tbody.results-table-row').forEach(tbody => {{
          const result = ALL_RESULTS.find(r => tbody.classList.contains(r));
          const show = result ? active.has(result) : true;
          tbody.classList.toggle('hidden', !show);
          if (show) totalVisible++;
        }});

        notFoundRow.classList.toggle('hidden', totalVisible > 0);
      }}

      checkboxes.forEach(cb => cb.addEventListener('change', applyFilters));
      applyFilters();

      // Expand/collapse individual test rows
      document.querySelectorAll('tr.collapsible').forEach(row => {{
        row.addEventListener('click', function (e) {{
          if (e.target.tagName === 'A') return;
          const targetId = this.dataset.target;
          if (!targetId) return;
          const extraRow = document.getElementById(targetId);
          if (!extraRow) return;
          extraRow.classList.toggle('hidden');
          const resultCell = this.querySelector('.col-result');
          if (resultCell) resultCell.classList.toggle('collapsed');
        }});
      }});

      // Show/hide all details
      document.getElementById('show_all_details').addEventListener('click', function () {{
        document.querySelectorAll('tr.extras-row').forEach(r => r.classList.remove('hidden'));
        document.querySelectorAll('.col-result').forEach(c => c.classList.remove('collapsed'));
      }});
      document.getElementById('hide_all_details').addEventListener('click', function () {{
        document.querySelectorAll('tr.extras-row').forEach(r => r.classList.add('hidden'));
        document.querySelectorAll('.col-result').forEach(c => c.classList.add('collapsed'));
      }});

      // Expand/collapse log wrappers
      document.querySelectorAll('.logwrapper').forEach(wrapper => {{
        const expander = wrapper.querySelector('.logexpander');
        if (expander) {{
          expander.addEventListener('click', function (e) {{
            e.stopPropagation();
            wrapper.classList.toggle('expanded');
          }});
        }}
      }});

      // Environment section toggle (matches pytest-html style)
      const envHeader = document.getElementById('environment-header');
      if (envHeader) {{
        envHeader.addEventListener('click', function () {{
          this.classList.toggle('collapsed');
          const table = document.getElementById('environment');
          if (table) table.classList.toggle('hidden');
        }});
      }}
    }})();
  </script>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_html)

    print(f"\nCombined {num_reports} reports ({row_index} test entries).")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all pytest-html reports in a directory into one HTML file."
    )
    parser.add_argument("input_directory", help="Directory containing the HTML files.")
    parser.add_argument(
        "-o", "--output", default=None,
        help=(
            "Optional output directory, filename, or file path. A filename alone is "
            "saved in the input directory; directories use combined.html; file paths "
            "use the supplied path and name. Defaults to combined.html in the input directory."
        ),
    )
    args = parser.parse_args()
    combine_html_files(args.input_directory, args.output)
