"""Write each experiment's results as one Markdown table file and one JSON file."""

import json

from active_oracle_evaluation.paths import output_dir


def markdown_table(header, rows):
    """Format a header and rows of strings as a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def write_results(experiment, tables, numbers, out_dir=None):
    """Write <experiment>.md (titled tables) and <experiment>.json (numbers); return both paths.

    ``tables`` is a list of (title, header, rows). ``numbers`` is a JSON-serialisable dict.
    """
    out_dir = output_dir(experiment) if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"# {experiment}"]
    for title, header, rows in tables:
        parts += [f"## {title}", markdown_table(header, rows)]
    md_path = out_dir / f"{experiment}.md"
    md_path.write_text("\n\n".join(parts) + "\n")
    json_path = out_dir / f"{experiment}.json"
    json_path.write_text(json.dumps(numbers, indent=1) + "\n")
    return md_path, json_path


def add_run(experiment, key, numbers, rows_of, title, header):
    """Add or replace one run in <experiment>.json and rewrite <experiment>.md from every run.

    ``rows_of(numbers)`` returns the table rows of one run. Runs are kept in key order, so an
    experiment that is called once per agent or suite keeps every result, not only the last.
    """
    json_path = output_dir(experiment) / f"{experiment}.json"
    runs = json.loads(json_path.read_text())["runs"] if json_path.exists() else {}
    runs[key] = numbers
    rows = [row for k in sorted(runs) for row in rows_of(runs[k])]
    md_path, _ = write_results(experiment, [(title, header, rows)], {"runs": runs})
    return md_path, json_path
