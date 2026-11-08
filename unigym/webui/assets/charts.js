/* charts.js -- vanilla SVG sparkline (zero dependencies).
 *
 * sparkline(values, opts) -> an <svg> element.  Values may contain
 * null/undefined gaps (collect runs have no rewards); gaps are skipped,
 * an all-empty series renders the empty placeholder instead.
 */

const NS = "http://www.w3.org/2000/svg";

export function sparkline(values, {width = 260, height = 44,
                                   stroke = "var(--accent)"} = {}) {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", height);
  svg.classList.add("sparkline");
  const pts = values.filter((v) => v !== null && v !== undefined
                                    && Number.isFinite(Number(v)))
      .map(Number);
  if (pts.length < 2) {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", width / 2);
    t.setAttribute("y", height / 2 + 4);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "var(--muted)");
    t.setAttribute("font-size", "11");
    t.textContent = "no reward history yet";
    svg.append(t);
    return svg;
  }
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const span = hi > lo ? hi - lo : 1.0;
  const x = (i) => (i / (pts.length - 1)) * (width - 2) + 1;
  const y = (v) => height - 3 - ((v - lo) / span) * (height - 6);
  const line = document.createElementNS(NS, "polyline");
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", stroke);
  line.setAttribute("stroke-width", "1.5");
  line.setAttribute("points",
      pts.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" "));
  svg.append(line);
  // last-point marker
  const dot = document.createElementNS(NS, "circle");
  dot.setAttribute("cx", x(pts.length - 1));
  dot.setAttribute("cy", y(pts[pts.length - 1]));
  dot.setAttribute("r", "2.5");
  dot.setAttribute("fill", stroke);
  svg.append(dot);
  return svg;
}
