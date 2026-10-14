function run(kind, tau, b0, learnB, seed, steps=600, lr=0.05){
  let r=seed; const rnd=()=>{r=(r*16807)%2147483647; return r/2147483647;};
  const N=5; let th=[...Array(N)].map(()=>rnd()*6.283), ph=[...Array(N)].map(()=>rnd()*6.283), b=b0;
  let L=0;
  for(let t=0;t<=steps;t++){
    const gth=Array(N).fill(0), gph=Array(N).fill(0); let gb=0; L=0;
    for(let i=0;i<N;i++)for(let j=0;j<N;j++){
      const s=Math.cos(th[i]-ph[j]), d=-Math.sin(th[i]-ph[j]);
      if(kind==='sig'){ const z=i===j?1:-1, x=z*(s/tau+b); L+=Math.log1p(Math.exp(-x))/N;
        const g=-z/(1+Math.exp(x))/N; gth[i]+=g/tau*d; gph[j]-=g/tau*d; gb+=g; }
    }
    if(kind==='soft'){ for(const dir of[0,1])for(let i=0;i<N;i++){const lg=[...Array(N)].map((_,j)=>(dir?Math.cos(th[j]-ph[i]):Math.cos(th[i]-ph[j]))/tau);
      const m=Math.max(...lg),e=lg.map(x=>Math.exp(x-m)),Z=e.reduce((a,c)=>a+c); L+=-(lg[i]-m-Math.log(Z))/(2*N);
      for(let j=0;j<N;j++){const g=(e[j]/Z-(i===j?1:0))/tau/(2*N);const[a,c]=dir?[j,i]:[i,j];const d=-Math.sin(th[a]-ph[c]);gth[a]+=g*d;gph[c]-=g*d;}}}
    th=th.map((v,i)=>v-lr*gth[i]); ph=ph.map((v,i)=>v-lr*gph[i]); if(learnB) b-=lr*gb;
  }
  const acc=th.map((_,i)=>{let best=0;for(let j=1;j<N;j++)if(Math.cos(th[i]-ph[j])>Math.cos(th[i]-ph[best]))best=j;return best===i}).filter(Boolean).length;
  return acc;
}
for (const cfg of [['soft',0.1,0,false],['sig',0.1,-10,false],['sig',0.1,-10,true],['sig',0.1,0,true],['sig',0.2,-5,true],['sig',0.3,-3,true]]) {
  const accs=[1,2,3,4,5,6,7,8,9,10].map(s=>run(...cfg,s*7919));
  console.log(cfg.join(' '), 'acc per seed', accs.join(''), 'mean', accs.reduce((a,c)=>a+c)/10);
}
