"use client";

import { useEffect, useMemo, useState } from "react";
import type { Activity } from "../api";

export type GridMode = "idle" | "loading" | "result";

const PAST_WEEKS = 26; // plus the current week and one week ahead
const COLS = PAST_WEEKS + 2;
const MOBILE_COLS = 16; // narrow screens show the most recent 16 weeks
const LIT_DAYS_AHEAD = 3;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

type Cell = { key: string; col: number; level: number; future: boolean; lit: boolean };

function iso(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function level(count: number): number {
  return count >= 4 ? 3 : count >= 2 ? 2 : count >= 1 ? 1 : 0;
}

function buildCells(activity: Activity | undefined, today: Date): { cells: Cell[]; months: { col: number; label: string }[] } {
  const counts = new Map((activity || []).map((a) => [a.date, a.count]));
  const start = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()));
  start.setUTCDate(start.getUTCDate() - start.getUTCDay() - PAST_WEEKS * 7); // a Sunday
  const lit = new Date(today);
  lit.setUTCDate(lit.getUTCDate() + LIT_DAYS_AHEAD);
  const todayKey = iso(today);
  const litKey = iso(lit);
  const cells: Cell[] = [];
  const months: { col: number; label: string }[] = [];
  for (let col = 0; col < COLS; col++) {
    for (let row = 0; row < 7; row++) {
      const d = new Date(start);
      d.setUTCDate(start.getUTCDate() + col * 7 + row);
      const key = iso(d);
      if (row === 0 && (col === 0 || d.getUTCDate() <= 7)) months.push({ col, label: MONTHS[d.getUTCMonth()] });
      cells.push({ key, col, level: level(counts.get(key) || 0), future: key > todayKey, lit: key === litKey });
    }
  }
  return { cells, months };
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(m.matches);
    const on = () => setReduced(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return reduced;
}

export function ContributionGrid({ mode, activity, label }: { mode: GridMode; activity?: Activity; label: string }) {
  const [today] = useState(() => new Date());
  const { cells, months } = useMemo(() => buildCells(mode === "result" ? activity : undefined, today), [activity, mode, today]);
  const reduced = usePrefersReducedMotion();
  const pastOrder = useMemo(() => cells.map((c, i) => ({ c, i })).filter(({ c }) => !c.future).map(({ i }) => i), [cells]);
  const [filled, setFilled] = useState(0);

  useEffect(() => {
    if (mode !== "loading") {
      setFilled(0);
      return;
    }
    if (reduced) {
      setFilled(Math.floor(pastOrder.length * 0.3));
      return;
    }
    // one square every 300 ms: the whole half-year takes about a minute, a slow cold start
    setFilled(0);
    const t = setInterval(() => setFilled((n) => Math.min(n + 1, pastOrder.length)), 300);
    return () => clearInterval(t);
  }, [mode, reduced, pastOrder.length]);

  const progress = useMemo(() => new Set(pastOrder.slice(0, filled)), [pastOrder, filled]);
  const oldCol = COLS - MOBILE_COLS;

  return (
    <figure className={`grid-figure mode-${mode}`} aria-label={label} role="img">
      <div className="grid-months wide" aria-hidden="true">
        {months.map((m) => (
          <span key={m.col} className="month" style={{ gridColumn: m.col + 1 }}>
            {m.label}
          </span>
        ))}
      </div>
      <div className="grid-months narrow" aria-hidden="true">
        {months
          .filter((m) => m.col >= oldCol)
          .map((m) => (
            <span key={m.col} className="month" style={{ gridColumn: m.col - oldCol + 1 }}>
              {m.label}
            </span>
          ))}
      </div>
      <div className="grid-cells" aria-hidden="true">
        {cells.map((c, i) => (
          <span
            key={c.key}
            className={[
              "cell",
              c.col < oldCol ? "old" : "",
              c.future ? "future" : "",
              c.lit ? "lit" : "",
              progress.has(i) ? "progress" : "",
            ].join(" ")}
            data-level={c.level}
            style={{ ["--col" as string]: c.col }}
          />
        ))}
      </div>
    </figure>
  );
}
