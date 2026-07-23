"""Human-inspectable transport-map report with no external assets."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

from .models import ContextualObservation, TransportQuery


STATE_CLASS = {
    "exact": "exact",
    "compatible": "compatible",
    "unknown": "unknown",
    "mismatch": "mismatch",
    "not_requested": "not-requested",
}


def write_transport_html(
    path: str | Path,
    observations: Iterable[ContextualObservation],
    query: TransportQuery,
    scores: list[dict[str, Any]],
) -> None:
    observation_map = {item.id: item for item in observations}
    dimensions = [item["dimension"] for item in scores[0]["components"]] if scores else []
    header = "".join(f"<th scope=\"col\">{_pretty(name)}</th>" for name in dimensions)
    body_rows = []
    for result in sorted(scores, key=lambda row: (-row["score"], row["observation_id"])):
        observation = observation_map[result["observation_id"]]
        cells = []
        for component in result["components"]:
            state = component["state"]
            value = "—" if component["score"] is None else f"{component['score']:.2f}"
            details = json.dumps(
                {
                    "evidence": component["evidence_values"],
                    "query": component["query_values"],
                    "matched_pair": component["matched_pair"],
                },
                ensure_ascii=False,
            )
            cells.append(
                f"<td class=\"{STATE_CLASS[state]}\" aria-label=\"{html.escape(state)}: "
                f"{html.escape(details)}\"><span>{html.escape(state)}</span><small>{value}</small></td>"
            )
        body_rows.append(
            "<tr>"
            f"<th scope=\"row\"><code>{html.escape(observation.id)}</code>"
            f"<small>{html.escape(observation.provenance.source_id)}</small></th>"
            f"<td class=\"score\"><strong>{result['score']:.3f}</strong>"
            f"<small>{html.escape(result['classification'])}</small></td>"
            + "".join(cells)
            + "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evidence transport map — {html.escape(query.query_id)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f7f5; --fg:#1c2128; --muted:#59636e;
  --line:#d0d7de; --exact:#c7ead3; --compatible:#dce8fb; --unknown:#eee8d8;
  --mismatch:#f4cece; --not:#e8eaed; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#11161c; --fg:#e6edf3; --muted:#9da7b1;
  --line:#39424c; --exact:#173d2a; --compatible:#193657; --unknown:#443c23;
  --mismatch:#552526; --not:#262d35; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
  font:14px/1.45 system-ui,sans-serif; }} main {{ max-width:1400px; margin:auto; }}
h1 {{ font-size:1.35rem; margin:0 0 4px; }} p {{ margin:0 0 18px; color:var(--muted); }}
.table-wrap {{ overflow:auto; }} table {{ border-collapse:collapse; width:100%; min-width:1050px; }}
th,td {{ border-bottom:1px solid var(--line); padding:9px; text-align:center; }} thead th {{ position:sticky;
  top:0; background:var(--bg); }} tbody th {{ text-align:left; min-width:210px; }} small {{ display:block;
  color:var(--muted); font-weight:400; }} td span {{ display:block; }} td.exact {{ background:var(--exact); }}
td.compatible {{ background:var(--compatible); }} td.unknown {{ background:var(--unknown); }}
td.mismatch {{ background:var(--mismatch); }} td.not-requested {{ background:var(--not); }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin:14px 0; color:var(--muted); }}
.legend i {{ display:inline-block; width:12px; height:12px; margin-right:5px; border:1px solid var(--line); }}
</style>
</head>
<body><main>
<h1>Evidence transport map</h1>
<p><code>{html.escape(query.query_id)}</code> — {html.escape(query.description or 'query context')}</p>
<div class="legend" aria-label="Compatibility legend">
<span><i style="background:var(--exact)"></i>exact</span>
<span><i style="background:var(--compatible)"></i>configured compatibility</span>
<span><i style="background:var(--unknown)"></i>context not reported</span>
<span><i style="background:var(--mismatch)"></i>mismatch</span>
<span><i style="background:var(--not)"></i>not requested</span>
</div>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Observation</th><th scope="col">Transport score</th>{header}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table></div>
</main></body></html>
"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")


def _pretty(value: str) -> str:
    return value.replace("_", " ").title()

