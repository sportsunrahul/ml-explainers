// Toy CapFilt data (hand-set ITM match probabilities); mirrors parts/viz-capfilt.js.
const D = [
  { web: ['IMG_4032.JPG', .04], beam: ['a dog on the beach', .81], nuc: ['a brown dog leaps for a frisbee over the sand', .88] },
  { web: ['Best sofa deals, free shipping!', .31], beam: ['a cat sitting on a couch', .86], nuc: ['a sleepy cat curled up on a striped sofa', .83] },
  { web: ['homemade margherita pizza with fresh basil', .91], beam: ['a pizza on a table', .79], nuc: ['a pizza topped with basil leaves on a wooden board', .87] },
  { web: ['vintage car parked by the road', .89], beam: ['a car parked on the street', .84], nuc: ['an old car towing a camper through the snow', .22] },
  { web: ['Our trip, summer 2019, day 3 :)', .12], beam: ['a lake with mountains in the background', .80], nuc: ['two snowy peaks reflected in a still lake under the sun', .85] },
];
const stop = new Set('a an the on in of with for by over under up through and to'.split(' '));
const words = s => s.toLowerCase().replace(/[^a-z ]/g, ' ').split(/\s+/).filter(w => w && !stop.has(w));
for (const mode of ['beam', 'nuc']) {
  const kept = D.filter(d => d[mode][1] >= .5), webKept = D.filter(d => d.web[1] >= .5).length;
  const vocab = new Set(kept.flatMap(d => words(d[mode][0])));
  const newW = new Set(kept.flatMap(d => words(d[mode][0]).filter(w => !words(d.web[0]).includes(w))));
  console.log(mode, 'synthetic kept', kept.length, '/5; web kept', webKept, '/5; texts kept', kept.length + webKept, '/10; distinct content words', vocab.size, '; words not in own web text', newW.size, 'avg caption words', (kept.reduce((s, d) => s + words(d[mode][0]).length, 0) / kept.length).toFixed(1));
}
