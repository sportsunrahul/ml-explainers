<style>
#viz-scoreboard .score-scroll { overflow-x:auto; }
#viz-scoreboard table.scoreboard { min-width:640px; margin:0; }
#viz-scoreboard table.scoreboard th { font-weight:600; vertical-align:bottom; }
#viz-scoreboard table.scoreboard th[scope=colgroup] { text-align:center; border-bottom:1px solid var(--rule); }
#viz-scoreboard table.scoreboard th button { font:inherit; color:inherit; background:none; border:0; padding:0; cursor:pointer; text-align:inherit; }
#viz-scoreboard table.scoreboard th button:hover { color:var(--blue); }
#viz-scoreboard table.scoreboard th button:focus-visible { outline:2px solid var(--yellow); outline-offset:2px; }
#viz-scoreboard table.scoreboard th[aria-sort] button::after { content:" \2193"; color:var(--muted); }
#viz-scoreboard table.scoreboard th[aria-sort=ascending] button::after { content:" \2191"; }
#viz-scoreboard table.scoreboard tbody th { text-align:left; font-weight:600; white-space:nowrap; }
#viz-scoreboard table.scoreboard tbody th a { color:var(--fg); text-decoration:none; }
#viz-scoreboard table.scoreboard tbody th a:hover { color:var(--blue); }
#viz-scoreboard table.scoreboard td:nth-child(3) { text-align:left; color:var(--muted); }
#viz-scoreboard table.scoreboard td.empty { color:var(--muted); font-size:.78rem; white-space:nowrap; }
#viz-scoreboard table.scoreboard thead tr:first-child th:nth-child(3) { text-align:left; }
#viz-scoreboard table.scoreboard td.rerun { color:var(--muted); font-style:italic; }
#viz-scoreboard table.scoreboard td.rerun sup { font-style:normal; margin-left:.1em; }
#viz-scoreboard table.scoreboard tr.moved td, #viz-scoreboard table.scoreboard tr.moved th { transition:background-color .6s; }
#viz-scoreboard table.scoreboard tr.flash td, #viz-scoreboard table.scoreboard tr.flash th { background:var(--panel2); }
@media (prefers-reduced-motion: reduce) { #viz-scoreboard table.scoreboard tr.moved td, #viz-scoreboard table.scoreboard tr.moved th { transition:none; } }
.guide li { margin:.5rem 0; }
</style>
<script>
(() => {
  // ch-summary: sortable scoreboard. Cells carry data-sort (numbers from notes/claims via build/f7/gen.py); empty cells sort last.
  // Sort buttons SET an order (descending for numbers, ascending for names); pressing twice keeps it (cue-safe).
  const fig = document.getElementById('viz-scoreboard'), tbl = document.getElementById('tbl-scoreboard'), tb = tbl.tBodies[0];
  const out = fig.querySelector('.readout');
  const original = [...tb.rows];
  const label = col => fig.querySelector(`#btn-score-sort-${col}`).textContent + (col >= 6 ? ' on VisA' : col >= 3 ? ' on MVTec AD' : '');
  function key(row, col) {
    const c = row.cells[col]; // th is cells[0]
    if (col === 0) return row.cells[0].textContent.trim().toLowerCase();
    const s = c.dataset.sort; return s === '' || s === undefined ? null : +s;
  }
  function apply(rows) {
    rows.forEach(r => tb.appendChild(r));
    if (!PE.instant()) { rows.forEach(r => { r.classList.add('moved', 'flash'); }); requestAnimationFrame(() => setTimeout(() => rows.forEach(r => r.classList.remove('flash')), 30)); }
  }
  function sortBy(col) {
    const rows = original.slice().sort((a, b) => {
      const x = key(a, col), y = key(b, col);
      if (col === 0) return x < y ? -1 : x > y ? 1 : 0;
      if (x === null && y === null) return 0; if (x === null) return 1; if (y === null) return -1;
      return col === 1 ? x - y : y - x;
    });
    apply(rows);
    tbl.querySelectorAll('th').forEach(th => th.removeAttribute('aria-sort'));
    fig.querySelector(`#btn-score-sort-${col}`).closest('th').setAttribute('aria-sort', col <= 1 ? 'ascending' : 'descending');
    const top = rows[0], v = col > 1 ? top.cells[col].textContent.replace(/r$/, '') : '';
    out.textContent = col <= 1 ? `Sorted by ${label(col).toLowerCase()}.` : `Sorted by ${label(col)}, highest first; methods with no number sit at the bottom. Top: ${top.cells[0].textContent.trim()} (${v}). Rows mix settings and papers, so compare within a setting.`;
  }
  fig.querySelectorAll('button[data-col]').forEach(b => b.addEventListener('click', () => sortBy(+b.dataset.col)));
  document.getElementById('btn-score-reset').addEventListener('click', () => {
    apply(original); tbl.querySelectorAll('th').forEach(th => th.removeAttribute('aria-sort'));
    out.textContent = 'Sorted in reading order. Hover or focus a cell for its table and row.';
  });
  tb.addEventListener('mouseover', e => { const td = e.target.closest('td[title]'); if (td) out.textContent = `${td.parentElement.cells[0].textContent.trim()}: ${td.textContent.replace(/r$/, '')}. Source: ${td.title}.`; });
})();
</script>
