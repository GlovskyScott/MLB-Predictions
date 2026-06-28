# Dashboard UI Enhancements — Design

**Date:** 2026-06-28
**Branch:** `jack/ui-enhancements`
**Goal:** Turn the dashboard from a static, fully-expanded report into a scannable, decision-first dashboard for finding model-vs-market edges across the day's slate.

## Context

The dashboard (`templates/index.html`, ~644 lines) renders the day's games in the left column with each game **fully expanded** — win bar, betting board, pitchers/venue, inning table, a tall score-distribution chart, and streamed AI analysis. With ~15 games this is a ~15-screen scroll. On load, **all** charts render and **all** AI analyses stream (`autoExplainAll`), which is heavy. There is no sort/filter, the edges are buried per-card, and responsive support is minimal (2 media queries in `index.html`; none in `archive.html`, `archive_version.html`, `game.html`).

Stack stays as-is: Jinja templates + vanilla JS + Chart.js, no build step. No model/prediction/store changes — this is presentation only.

## Scope

Four enhancements (all approved), plus a print requirement:

1. Collapsible game cards with lazy chart/analysis loading
2. "Today's Best Edges" board
3. Sort & filter toolbar
4. Responsive / mobile layout
5. Print: render as if every card is fully expanded

Out of scope: live scores/status, loading skeletons (future polish).

## 1. Collapsible cards (+ lazy load)

Each game renders as a **collapsed summary row** by default:

```
▸ [away logo] WSN @ BAL [home logo]    BAL 61%    ⌁ WSN ML +3%    7:05 PM ET
```

- Summary row fields: chevron, matchup (logos + abbrs), favored side + calibrated win%, best-edge badge (or "—"), first-pitch time/status.
- Clicking the row toggles the existing full detail block (`.card-detail`).
- **Collapsed by default.** An **Expand all / Collapse all** control in the toolbar. The all/each state persists in `localStorage` (`ui.expandAll`, and a set of expanded game IDs).
- **Lazy render on first expand:**
  - Score chart: Chart.js needs a visible canvas to size correctly, so the chart renders the first time its card expands (data already embedded as `window.cdata_<id>`). A `rendered` flag prevents re-render.
  - AI analysis: `streamExplanation(gameId)` fires on first expand instead of `autoExplainAll` on load. A `loaded` flag prevents re-streaming.
- This removes the all-15-charts + all-15-streams cost from initial load.

**Data:** no new server data needed beyond what each card already has; collapse is pure markup/JS/CSS. Each `.game-card` gets `data-edge`, `data-firstpitch`, `data-confidence`, `data-teams` attributes (see §3).

## 2. "Today's Best Edges" board

A compact, ranked panel at the top of the games column.

- Lists the top **N=6** model-vs-market edges across the whole slate, each: colored side + market + edge%, with the matchup, e.g. `WSN ML +3% · WSN @ BAL`.
- Clicking a row scrolls to and expands that game's card.
- Hidden entirely when there is no market data (no ESPN odds) or no positive edges.

**Server-side:** new helper `_top_edges(games) -> list[dict]` in `src/app.py`. For each game with a `market` block, emit candidate edges from `edge_ml_*`, `edge_total_*`, `edge_rl_*` as `{game_id, matchup, market, side, pct}`, drop zero/None, sort by `pct` desc, return top N. Passed to the template as `top_edges`.

## 3. Sort & filter toolbar

A slim toolbar above the game list:

- **Sort** (select): Best edge (default) · First pitch · Win confidence · Matchup A–Z.
- **Filter** (select): All (default) · Has edge.
- **Team search** (text input): substring match on either team name/abbr.
- **Expand all / Collapse all** toggle (from §1).

Implemented client-side: cards already exist in the DOM, so JS reorders them (by reading `data-*`) and toggles `display` for filtering/search. Selections persist in `localStorage` (`ui.sort`, `ui.filter`, `ui.search`) and reapply on load.

**Server-side:** each `.game-card` emits sortable attributes:
- `data-edge` = best edge % for the game (0 if none) — from `_best_edge`.
- `data-firstpitch` = ISO/UTC game datetime (string sort works).
- `data-confidence` = max(home_win_pct, away_win_pct).
- `data-teams` = lowercased "away away_abbr home home_abbr" for search.

New helper `_best_edge(market) -> dict | None` in `src/app.py`: the single biggest of the game's ML/total/run-line edges, `{pct, side, market}`. Attached to each game as `g['best_edge']` (used by the summary row, the edges board ranking, and `data-edge`).

## 4. Responsive / mobile

- `@media (max-width: 900px)`: `.dashboard` grid → single column (today, then historical below).
- `@media (max-width: 600px)`: reduce paddings; `.odds-table` and `.inn-table` wrap in horizontally-scrollable containers; summary rows wrap gracefully; toolbar controls stack full-width; interactive targets ≥40px.
- Add breakpoints to `archive.html`, `archive_version.html`, `game.html` (their tables get horizontal scroll; headers stack).

## 5. Print

Printing must look as if **every card is fully expanded**.

- `@media print` stylesheet:
  - Force `.card-detail { display: block !important; }` regardless of collapsed state.
  - Hide interactive chrome: the sort/filter toolbar, Expand/Collapse + Refresh/Retrain buttons, the chat FAB/panel, and the edges board's "click to jump" affordance (the board itself may print as a summary).
  - Avoid page breaks inside a card (`break-inside: avoid`).
- `window.onbeforeprint` handler: expand all cards and synchronously render any not-yet-rendered score charts (data is embedded, so this is immediate and reliable). AI analyses that have not streamed yet will show their placeholder — acceptable; charts and all model/market data print fully.

## Testing

- **Python (pytest, `tests/test_app.py`):**
  - `test_best_edge_picks_largest` — `_best_edge` returns the biggest of ML/total/run-line, with side/market; `None` when no market.
  - `test_top_edges_ranks_and_caps` — `_top_edges` flattens, sorts desc, caps at N, skips games without market.
- **In-browser verification:** desktop screenshots (collapsed scan, one expanded, edges board, sort/filter, expand-all); mobile-viewport screenshot (single column, scrollable tables); print-preview check (all cards expanded, charts present, chrome hidden).

## File map

| File | Change |
|---|---|
| `src/app.py` | add `_best_edge`, `_top_edges`; attach `g['best_edge']`; pass `top_edges` to render |
| `templates/index.html` | summary-row markup + collapsible detail; edges board; sort/filter toolbar; lazy chart/analysis JS; `data-*` attrs; responsive + print CSS |
| `templates/archive.html`, `archive_version.html`, `game.html` | responsive + print CSS |
| `tests/test_app.py` | tests for `_best_edge`, `_top_edges` |
