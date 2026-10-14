const W=640,H=400;
const q=(v,S,B)=>Math.min(B-1,Math.floor(v/S*B));
const dec=(k,S,B)=>(k+.5)*S/B;
const boxes={dog:[88,205,272,342],sign:[296,64,404,132]};
for(const [n,b] of Object.entries(boxes)){
 for(const B of [4,8,16,32,100,1000]){
  const ks=[q(b[0],W,B),q(b[1],H,B),q(b[2],W,B),q(b[3],H,B)];
  const d=[dec(ks[0],W,B),dec(ks[1],H,B),dec(ks[2],W,B),dec(ks[3],H,B)];
  const err=Math.max(...d.map((v,i)=>Math.abs(v-b[i])));
  console.log(n,B,ks.join(','),'maxerr',err.toFixed(2),'bound x',(W/2/B).toFixed(2),'y',(H/2/B).toFixed(2));
 }
 console.log('qwen',b.map((v,i)=>Math.floor(v/(i%2?H:W)*1000)).join(','));
}
// molmo point one decimal on 0-100
const p=[331,318]; console.log('molmo',(p[0]/W*100).toFixed(1),(p[1]/H*100).toFixed(1));
for (const S of [640,1344,3840]) console.log('1000 bins on',S,'px: bin',S/1000,'max err',S/2000);
