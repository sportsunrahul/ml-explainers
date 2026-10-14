require('./pe.cjs');
const N=5;
function start(seed){const r=PE.rng(seed);return {th:[...Array(N)].map(()=>r()*2*Math.PI),ph:[...Array(N)].map(()=>r()*2*Math.PI)};}
function train(seed,b0,steps=400,lr=.05,tau=.1,learnB=true){let {th,ph}=start(seed),b=b0;
 for(let k=0;k<steps;k++){const g=PE.contrast(th,ph,tau,'sigmoid',b);th=th.map((v,i)=>v-lr*g.gth[i]);ph=ph.map((v,i)=>v-lr*g.gph[i]);if(learnB)b-=lr*g.gb;}
 const g=PE.contrast(th,ph,tau,'sigmoid',b);const acc=g.S.filter((r,i)=>r.indexOf(Math.max(...r))===i).length;return {acc,b,L:g.L};}
const steps=+process.argv[2]||400;
for(const b0 of [-10,0]){
 const seeds=[65,1,2,3,4,5,6,7,8,9,10];
 const R=seeds.map(s=>train(s,b0,steps));
 console.log('b0',b0,'steps',steps,R.map(r=>r.acc).join(''),'mean(10 seeds 1-10)',(R.slice(1).reduce((a,r)=>a+r.acc,0)/10).toFixed(1),'seed65 b',R[0].b.toFixed(2),'L',R[0].L.toFixed(3));
 const all=[];for(let s=1;s<=200;s++)all.push(train(s,b0,steps).acc);console.log('  200 seeds mean',(all.reduce((a,c)=>a+c)/200).toFixed(2),'5/5 frac',all.filter(a=>a===5).length/200);
}
