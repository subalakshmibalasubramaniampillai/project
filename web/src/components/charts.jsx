import React, { useLayoutEffect, useRef, useState } from "react";

// Measure the rendered width so charts draw at real pixel size.
function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(640);
  useLayoutEffect(() => {
    if (!ref.current) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(280, Math.floor(entry.contentRect.width)));
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

function scale([d0, d1], [r0, r1]) {
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

function Tip({ tip }) {
  if (!tip) return null;
  return (
    <div className="chart-tip" style={{ left: tip.x, top: tip.y }}>
      {tip.content}
    </div>
  );
}

function tipPosition(x, y, width) {
  return { x: Math.min(Math.max(8, x + 14), width - 220), y: Math.max(0, y - 10) };
}

// ── forest chart: one row per item, dot = mean, whisker = spread ──────────
export function ForestChart({
  rows, domain, ticks, format = (v) => v.toFixed(3), tickFormat = (v) => v.toFixed(2), ariaLabel,
}) {
  const [ref, width] = useWidth();
  const [tip, setTip] = useState(null);
  const labelW = Math.min(230, Math.max(130, width * 0.34));
  const valueW = 70;
  const rowH = 32;
  const headerH = 30;
  const axisH = 30;
  const x = scale(domain, [labelW + 12, width - valueW - 12]);
  let y = 6;
  const laid = rows.map((r) => {
    const top = y;
    y += r.header ? headerH : rowH;
    return { ...r, top };
  });
  const height = y + axisH;

  return (
    <div className="chart" ref={ref} onMouseLeave={() => setTip(null)}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={x(t)} x2={x(t)} y1={6} y2={height - axisH} />
            <text className="axis-text" x={x(t)} y={height - axisH + 18} textAnchor="middle">
              {tickFormat(t)}
            </text>
          </g>
        ))}
        <line className="baseline" x1={labelW + 12} x2={width - valueW - 12}
          y1={height - axisH} y2={height - axisH} />
        <text className="axis-text" x={width} y={14} textAnchor="end">mean</text>
        {laid.map((r) => {
          if (r.header) {
            return (
              <text key={r.header} x={0} y={r.top + 21} style={{ fontSize: 12, fill: "var(--muted)",
                letterSpacing: "0.06em", textTransform: "uppercase" }}>
                {r.header}
              </text>
            );
          }
          const cy = r.top + rowH / 2;
          return (
            <g key={r.key}
              onMouseMove={(e) => {
                const box = ref.current.getBoundingClientRect();
                setTip({
                  ...tipPosition(e.clientX - box.left, e.clientY - box.top, width),
                  content: (
                    <>
                      <strong>{r.label}</strong>
                      <div>Mean {format(r.value)}</div>
                      {r.lo !== undefined && (
                        <div className="muted">Range shown {format(r.lo)} to {format(r.hi)}</div>
                      )}
                    </>
                  ),
                });
              }}>
              <rect x={0} y={r.top} width={width} height={rowH} fill="transparent" />
              <text x={0} y={cy + 5} style={{ fontSize: 14, fontWeight: r.bold ? 600 : 400,
                fill: r.bold ? "var(--ink)" : "var(--ink-2)" }}>
                {r.label}
              </text>
              {r.lo !== undefined && (
                <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={r.color}
                  strokeWidth={2} strokeLinecap="round" />
              )}
              <circle cx={x(r.value)} cy={cy} r={r.bold ? 6 : 5} fill={r.color}
                stroke="var(--surface)" strokeWidth={2} />
              <text x={width} y={cy + 5} textAnchor="end"
                style={{ fontSize: 14, fontVariantNumeric: "tabular-nums",
                  fontWeight: r.bold ? 600 : 400, fill: "var(--ink)" }}>
                {format(r.value)}
              </text>
            </g>
          );
        })}
      </svg>
      <Tip tip={tip} />
    </div>
  );
}

// ── line chart with hover read-out ─────────────────────────────────────
export function LineChart({
  series, xDomain, yDomain, xTicks, yTicks, xFormat = String, yFormat = String,
  xLabel, yLabel, diagonal = false, height = 300, logX = false, ariaLabel,
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const pad = { l: 52, r: 16, t: 12, b: 44 };
  const tx = logX ? Math.log10 : (v) => v;
  const x = scale(xDomain.map(tx), [pad.l, width - pad.r]);
  const y = scale(yDomain, [height - pad.b, pad.t]);
  const X = (v) => x(tx(v));

  function onMove(e) {
    const box = ref.current.getBoundingClientRect();
    const mx = e.clientX - box.left;
    const rows = series.map((s) => {
      let best = s.points[0];
      s.points.forEach((p) => {
        if (Math.abs(X(p.x) - mx) < Math.abs(X(best.x) - mx)) best = p;
      });
      return { s, p: best };
    });
    const anchor = rows[0].p;
    setHover({ mx: X(anchor.x), rows, my: e.clientY - box.top });
  }

  return (
    <div className="chart" ref={ref} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid" x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} />
            <text className="axis-text" x={pad.l - 8} y={y(t) + 4} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="axis-text" x={X(t)} y={height - pad.b + 18} textAnchor="middle">
            {xFormat(t)}
          </text>
        ))}
        <line className="baseline" x1={pad.l} x2={width - pad.r} y1={height - pad.b} y2={height - pad.b} />
        {xLabel && (
          <text className="axis-text" x={(pad.l + width - pad.r) / 2} y={height - 6} textAnchor="middle">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text className="axis-text" transform={`translate(12 ${(pad.t + height - pad.b) / 2}) rotate(-90)`}
            textAnchor="middle">
            {yLabel}
          </text>
        )}
        {diagonal && (
          <line x1={X(xDomain[0])} y1={y(yDomain[0])} x2={X(Math.min(xDomain[1], yDomain[1]))}
            y2={y(Math.min(xDomain[1], yDomain[1]))} stroke="var(--rule-strong)" strokeWidth={1} />
        )}
        {series.map((s) => (
          <g key={s.key}>
            <path
              d={s.points.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${y(p.y)}`).join(" ")}
              fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
            />
            {s.points.map((p) => (
              <circle key={`${s.key}-${p.x}`} cx={X(p.x)} cy={y(p.y)} r={4} fill={s.color}
                stroke="var(--surface)" strokeWidth={2} />
            ))}
          </g>
        ))}
        {hover && (
          <line x1={hover.mx} x2={hover.mx} y1={pad.t} y2={height - pad.b} stroke="var(--rule-strong)" />
        )}
      </svg>
      {hover && (
        <Tip tip={{
          ...tipPosition(hover.mx, hover.my, width),
          content: (
            <>
              {hover.rows.map(({ s, p }) => (
                <div key={s.key}>
                  <span className="legend-key" style={{ background: s.color }} />
                  {s.label}: {p.tip || `${xFormat(p.x)} → ${yFormat(p.y)}`}
                </div>
              ))}
            </>
          ),
        }} />
      )}
    </div>
  );
}

// ── horizontal bars from a zero baseline (handles negative values) ────────
// Each label sits on its own line above its bar, so long labels never collide.
export function BarList({ rows, format = (v) => v.toFixed(3), color = "var(--c-ours)", ariaLabel }) {
  const [ref, width] = useWidth();
  const [tip, setTip] = useState(null);
  const valueW = 56;
  const rowH = 46;
  const lo = Math.min(0, ...rows.map((r) => r.value));
  const hi = Math.max(0, ...rows.map((r) => r.value));
  const x = scale([lo, hi || 1], [lo < 0 ? valueW : 0, width - valueW]);
  const height = rows.length * rowH;
  return (
    <div className="chart" ref={ref} onMouseLeave={() => setTip(null)}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        {rows.map((r, i) => {
          const top = i * rowH;
          const x0 = x(Math.min(0, r.value));
          const w = Math.max(2, Math.abs(x(r.value) - x(0)));
          const end = r.value >= 0 ? x(r.value) + 6 : x(r.value) - 6;
          return (
            <g key={r.label}
              onMouseMove={(e) => {
                const box = ref.current.getBoundingClientRect();
                setTip({ ...tipPosition(e.clientX - box.left, e.clientY - box.top, width),
                  content: <><strong>{r.label}</strong><div>{r.tip || format(r.value)}</div></> });
              }}>
              <rect x={0} y={top} width={width} height={rowH} fill="transparent" />
              <text x={0} y={top + 16} style={{ fontSize: 14 }}>{r.label}</text>
              <rect x={x0} y={top + 23} width={w} height={14} rx={3} fill={r.color || color} />
              <text x={end} y={top + 34} textAnchor={r.value >= 0 ? "start" : "end"}
                style={{ fontSize: 13, fontVariantNumeric: "tabular-nums", fill: "var(--ink)" }}>
                {format(r.value)}
              </text>
            </g>
          );
        })}
        {lo < 0 && <line className="baseline" x1={x(0)} x2={x(0)} y1={20} y2={height - 6} />}
      </svg>
      <Tip tip={tip} />
    </div>
  );
}

// ── predicted risk at each stay of one patient ───────────────────────────
export function RiskTimeline({ points, eventLabel, height = 260 }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const pad = { l: 48, r: 16, t: 12, b: 64 };
  const n = points.length;
  const maxRisk = Math.min(1, Math.max(0.3, ...points.map((p) => p.risk || 0)) * 1.15);
  const x = scale([1, Math.max(2, n)], [pad.l + 10, width - pad.r - 10]);
  const y = scale([0, maxRisk], [height - pad.b, pad.t]);
  const stripY = height - 22;
  const yTicks = [0, maxRisk / 2, maxRisk].map((v) => Math.round(v * 20) / 20);
  const line = points.filter((p) => p.risk !== null && p.risk !== undefined);
  const every = n > 14 ? Math.ceil(n / 12) : 1;

  return (
    <div className="chart" ref={ref} onMouseLeave={() => setHover(null)}
      onMouseMove={(e) => {
        const box = ref.current.getBoundingClientRect();
        const mx = e.clientX - box.left;
        let best = points[0];
        points.forEach((p) => { if (Math.abs(x(p.stay) - mx) < Math.abs(x(best.stay) - mx)) best = p; });
        setHover({ p: best, my: e.clientY - box.top });
      }}>
      <svg width={width} height={height} role="img" aria-label="Predicted risk at each stay">
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid" x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} />
            <text className="axis-text" x={pad.l - 8} y={y(t) + 4} textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        <line className="baseline" x1={pad.l} x2={width - pad.r} y1={height - pad.b} y2={height - pad.b} />
        {points.map((p) => (p.stay - 1) % every === 0 && (
          <text key={`t${p.stay}`} className="axis-text" x={x(p.stay)} y={height - pad.b + 16} textAnchor="middle">
            {p.stay}
          </text>
        ))}
        <path d={line.map((p, i) => `${i ? "L" : "M"}${x(p.stay)},${y(p.risk)}`).join(" ")}
          fill="none" stroke="var(--c-ours)" strokeWidth={2} strokeLinejoin="round" />
        {line.map((p) => (
          <circle key={p.stay} cx={x(p.stay)} cy={y(p.risk)} r={4} fill="var(--c-ours)"
            stroke="var(--surface)" strokeWidth={2} />
        ))}
        <text className="axis-text" x={pad.l - 8} y={stripY + 4} textAnchor="end">Outcome</text>
        {points.map((p) => (
          <circle key={`o${p.stay}`} cx={x(p.stay)} cy={stripY} r={5}
            fill={p.outcome ? "var(--ink)" : "var(--surface)"} stroke="var(--ink)" strokeWidth={1.5} />
        ))}
        {hover && <line x1={x(hover.p.stay)} x2={x(hover.p.stay)} y1={pad.t} y2={stripY + 8}
          stroke="var(--rule-strong)" />}
      </svg>
      {hover && (
        <Tip tip={{
          ...tipPosition(x(hover.p.stay), hover.my, width),
          content: (
            <>
              <strong>Stay {hover.p.stay}</strong>
              <div>Predicted risk {hover.p.risk === null || hover.p.risk === undefined
                ? "not scored" : `${(hover.p.risk * 100).toFixed(1)}%`}</div>
              <div className="muted">{hover.p.outcome ? `Outcome: ${eventLabel}` : "Outcome: no"}</div>
            </>
          ),
        }} />
      )}
    </div>
  );
}
