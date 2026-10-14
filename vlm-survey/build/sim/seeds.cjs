require('./pe.cjs');
function train(th, ph, tau, steps=400, lr=.05){ for(let k=0;k<steps;k++){const g=PE.contrast(th,ph,tau); th=th.map((v,i)=>v-lr*g.gth[i]); ph=ph.map((v,i)=>v-lr*g.gph[i]);} return [th,ph]; }
const acc=(th,ph)=>{const S=PE.contrast(th,ph,.07).S; return S.filter((r,i)=>r.indexOf(Math.max(...r))===i).length;};
const out=[];
for(let seed=1;seed<400;seed++){
  const r=PE.rng(seed), th=[...Array(5)].map(()=>r()*2*Math.PI), ph=[...Array(5)].map(()=>r()*2*Math.PI);
  const all=[...th,...ph].map(a=>((a%(2*Math.PI))+2*Math.PI)%(2*Math.PI)).sort((a,b)=>a-b);
  let minsep=2*Math.PI; for(let i=0;i<all.length;i++){const d=(all[(i+1)%all.length]-all[i]+2*Math.PI)%(2*Math.PI); minsep=Math.min(minsep,d);}
  const a0=acc(th,ph);
  const [c1,c2]=train(th.slice(),ph.slice(),.07), [h1,h2]=train(th.slice(),ph.slice(),1);
  const Lc=PE.contrast(c1,c2,.07).L, Lh=PE.contrast(h1,h2,1).L;
  // cold final: min angular gap between different pairs
  if(acc(c1,c2)===5 && acc(h1,h2)===5 && a0<=2 && minsep>.12 && Lc<.2) out.push([seed,minsep.toFixed(2),a0,Lc.toFixed(3),Lh.toFixed(3)]);
}
console.log(out.length, JSON.stringify(out.slice(0,12)));
let cnt={c:0,h:0,both:0};
for(let seed=1;seed<400;seed++){const r=PE.rng(seed), th=[...Array(5)].map(()=>r()*2*Math.PI), ph=[...Array(5)].map(()=>r()*2*Math.PI);
 const [c1,c2]=train(th.slice(),ph.slice(),.07,1500), [h1,h2]=train(th.slice(),ph.slice(),1,1500); const a=acc(c1,c2)===5,b=acc(h1,h2)===5; cnt.c+=a;cnt.h+=b;cnt.both+=a&&b;}
console.log('1500 steps', cnt);
