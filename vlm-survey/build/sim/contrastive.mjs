// 2D toy: N image angles + N caption angles on the unit circle, symmetric InfoNCE on cos sims / tau
function loss(th, ph, tau, kind='softmax', b=0) {
  const N = th.length; let L = 0; const gth = Array(N).fill(0), gph = Array(N).fill(0);
  const s = (i,j) => Math.cos(th[i]-ph[j]);
  const ds = (i,j) => -Math.sin(th[i]-ph[j]); // d s / d th_i ; d s / d ph_j = +sin
  if (kind==='softmax') {
    for (const dir of [0,1]) for (let i=0;i<N;i++) {
      const lg = [...Array(N)].map((_,j)=> (dir? s(j,i):s(i,j))/tau);
      const m = Math.max(...lg); const e = lg.map(x=>Math.exp(x-m)); const Z = e.reduce((a,c)=>a+c);
      L += -(lg[i]-m-Math.log(Z))/(2*N);
      for (let j=0;j<N;j++){ const g=((e[j]/Z)-(i===j?1:0))/tau/(2*N);
        const [a,c]= dir? [j,i]:[i,j]; gth[a]+=g*ds(a,c); gph[c]+=-g*ds(a,c); }
    }
  } else {
    for (let i=0;i<N;i++) for (let j=0;j<N;j++){ const z=(i===j?1:-1); const x=z*(s(i,j)/tau+b);
      L += Math.log1p(Math.exp(-x))/N; const g = -z/(1+Math.exp(x))/tau/N; gth[i]+=g*ds(i,j); gph[j]+=-g*ds(i,j); }
  }
  return {L,gth,gph};
}
function train(tau, kind, steps=300, lr=0.05, b=0, seed=1){
  let r=seed; const rnd=()=>{r=(r*16807)%2147483647; return r/2147483647;};
  const N=5; let th=[...Array(N)].map(()=>rnd()*6.283), ph=[...Array(N)].map(()=>rnd()*6.283);
  const hist=[];
  for(let t=0;t<=steps;t++){ const {L,gth,gph}=loss(th,ph,tau,kind,b); if(t%50===0)hist.push(L.toFixed(3));
    th=th.map((v,i)=>v-lr*gth[i]); ph=ph.map((v,i)=>v-lr*gph[i]); }
  const acc = th.map((_,i)=>{let best=0;for(let j=1;j<5;j++) if(Math.cos(th[i]-ph[j])>Math.cos(th[i]-ph[best]))best=j;return best===i;}).filter(Boolean).length;
  const gap = th.map((v,i)=>((Math.abs(v-ph[i])%6.283)*57.3).toFixed(0));
  const deg = th.map(v=>(((v%6.283)+6.283)%6.283*57.3).toFixed(0));
  return {kind,tau,hist,acc,gap,deg};
}
for (const tau of [1,0.3,0.1,0.07]) console.log(JSON.stringify(train(tau,'softmax')));
for (const tau of [1,0.3,0.1]) console.log(JSON.stringify(train(tau,'sigmoid',300,0.05,-1)));
