// Rows = attending token, cols = attended token. 8 queries (drawn subset of 32) + 6 text tokens.
const NQ = 8, NT = 6, N = NQ + NT;
const allow = (m, i, j) => {
  const qi = i < NQ, qj = j < NQ;
  if (qi && qj) return true;                       // queries always see queries
  if (m === 'itm') return true;                    // bidirectional: everyone sees everyone
  if (qi && !qj) return false;                     // ITC and ITG: queries never see text
  if (!qi && qj) return m === 'itg';               // ITG: text sees all queries; ITC: not
  return m === 'itc' ? true : j <= i;              // text->text: bidirectional (ITC) or causal (ITG)
};
for (const m of ['itm', 'itg', 'itc']) {
  console.log(m);
  let open = 0;
  for (let i = 0; i < N; i++) { let s = ''; for (let j = 0; j < N; j++) { const a = allow(m, i, j); open += a; s += (j === NQ ? ' ' : '') + (a ? '.' : '#'); } console.log((i === NQ ? '\n' : '') + s); }
  console.log('open cells', open, 'of', N * N);
}
