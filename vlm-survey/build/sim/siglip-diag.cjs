require('./pe.cjs');
const N=5,names=['dog','cat','car','pizza','tree'];
const r=PE.rng(65),th0=names.map(()=>r()*2*Math.PI),ph0=names.map(()=>r()*2*Math.PI);
for(const b0 of [-10,0]){let th=th0.slice(),ph=ph0.slice(),b=b0;
 const g0=PE.contrast(th,ph,.1,'sigmoid',b);
 const gn=Math.hypot(...g0.gth,...g0.gph);
 // split gradient contribution positives vs negatives at step 0
 console.log('b0',b0,'L0',g0.L.toFixed(2),'|grad|0',gn.toFixed(2),'gb0',g0.gb.toFixed(2));
 for(let k=0;k<400;k++){const g=PE.contrast(th,ph,.1,'sigmoid',b);th=th.map((v,i)=>v-.05*g.gth[i]);ph=ph.map((v,i)=>v-.05*g.gph[i]);b-=.05*g.gb;
  if([0,5,20,100,399].includes(k)){const S=g.S;console.log(' step',k,'b',b.toFixed(2),'diag',S.map((r,i)=>r[i].toFixed(2)).join(' '),'maxneg',Math.max(...S.flatMap((r,i)=>r.filter((_,j)=>j!==i))).toFixed(2),'L',g.L.toFixed(2));}}
 const S=PE.contrast(th,ph,.1,'sigmoid',b).S;console.log(' final best',S.map(r=>names[r.indexOf(Math.max(...r))]).join(','));
 // p(match) = sigmoid(t*s+b) for diag and prob matrix
 console.log(' diag prob',S.map((r,i)=>(1/(1+Math.exp(-(r[i]/.1+b)))).toFixed(2)).join(' '));
 console.log(' th',th.map(a=>(((a%6.283)+6.283)%6.283).toFixed(2)).join(' '),' ph',ph.map(a=>(((a%6.283)+6.283)%6.283).toFixed(2)).join(' '));
}
for(const b0 of [-10,0]){const S=PE.contrast(th0,ph0,.1,'sigmoid',b0).S;let yes=0,wrong=0;S.forEach((r,i)=>r.forEach((s,j)=>{if(s/.1+b0>0){yes++;if(i!==j)wrong++;}}));console.log('b0',b0,'yes cells',yes,'wrong yes',wrong);}
