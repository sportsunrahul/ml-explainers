import sys
sys.path.insert(0, '/Users/rahulmishra/paper-explainers/shared')
import common
from common import once

D = '/Users/rahulmishra/paper-explainers/world-models/'
s = open(D + 'original.html').read()

s = common.apply(s, hl='var(--acc)', extra_css='''
header.hero .lede{margin-bottom:26px}
#f-hero .canvas{height:clamp(260px,34vw,380px)}
''')
# The page has no math, so it should not load the KaTeX stylesheet.
s = once(s, common.KATEX_LINK + '\n', '')

# ---- hero: a rollout, imagined frames continuing from real ones ----
s = once(s, 'simulators.</p>\n</div></header>', '''simulators.</p>
 <figure class="fig" id="f-hero" style="margin:0"><div class="canvas" role="img" aria-label="A rollout: real frames of a small platform game sit in a row, outlined in green. For each next step an orange action cube drops into a purple world-model block, which predicts the next frame, outlined in pink, and that frame feeds the next step. The character walks right, jumps onto a ledge and walks on."><span class="hint">Drag to rotate</span></div>
  <figcaption><div class="controls"><button class="btn" type="button" id="hero-replay">Replay</button>
   <span class="legend"><i style="background:var(--real)"></i>real frames</span><span class="legend"><i style="background:var(--act)"></i>agent's actions</span><span class="legend"><i style="background:var(--time)"></i>world model</span><span class="legend"><i style="background:var(--imag)"></i>imagined frames</span></div>
   <p>A rollout. At each step the world model reads the latest frame and the agent's next action, and predicts the frame that follows. That prediction is the input to the next step, so imagined frames carry on from the real ones. The game and its frames are drawn by hand for illustration.</p></figcaption></figure>
</div></header>''')

HERO = r'''
/* ---------- hero: a world model rolls out imagined frames (illustrative) ---------- */
// The tilt brings the right end closer, so the camera aims a little right of center.
function buildHero(s){const B=base([.2,.85,9.4],[.2,.55,0],53);s.tRotY=-.12;s.tRotX=.05;s.vmin=30;
 // Character position per frame in the f-loop game's pixels: walk right, jump onto the ledge, walk on.
 const P=[[10,0],[16,0],[22,0],[28,0],[33,12],[40,14],[46,14]],ACT=['right','jump','right','right'],N=P.length,NR=N-ACT.length,S=64;
 function draw(c,[x,y]){const g=c.getContext('2d');g.fillStyle='#CFE3F3';g.fillRect(0,0,S,S);g.fillStyle='#7FAE6B';g.fillRect(0,50,S,14);g.fillStyle='#A5795A';g.fillRect(38,36,14,4);
  g.fillStyle='#F4C65A';g.beginPath();g.arc(52,10,5,0,7);g.fill();g.fillStyle='#5F3DC4';g.fillRect(x-3,41-y,6,9);g.fillStyle='#F2D3B3';g.fillRect(x-2,38-y,4,3);}
 let F,sp,MY,AY,narrow=null;const X=i=>(i-(N-1)/2)*sp;
 const pg=new THREE.PlaneGeometry(1,1),bx=new THREE.BoxGeometry(1.09,1.09,.06);
 const frames=P.map((p,i)=>{const c=document.createElement('canvas');c.width=c.height=S;draw(c,p);const t=new THREE.CanvasTexture(c);t.magFilter=THREE.NearestFilter;
  const g=new THREE.Group(),im=new THREE.Mesh(pg,new THREE.MeshBasicMaterial({map:t}));im.position.z=.035;g.add(new THREE.Mesh(bx,std(i<NR?C.real:C.imag)),im);B.root.add(g);return g;});
 const mg=new THREE.BoxGeometry(.5,.5,.5),ag=new THREE.BoxGeometry(.3,.3,.3),cg=new THREE.CylinderGeometry(.02,.02,1,8),hg=new THREE.ConeGeometry(.07,.16,12);cg.translate(0,-.5,0);hg.rotateX(Math.PI);
 const tube=c=>{const m=new THREE.Mesh(new THREE.BufferGeometry(),std(c));B.root.add(m);return m;};
 const steps=ACT.map(a=>{const m=new THREE.Mesh(mg,std(C.time,{transparent:true,opacity:.9}));m.material.emissive.copy(C.time);
  const q=new THREE.Mesh(ag,std(C.act,{transparent:true,opacity:0})),k=new THREE.Mesh(cg,std(C.act)),h=new THREE.Mesh(hg,std(C.imag)),l=label(a,.26,hex('act'));
  B.root.add(m,q,k,h,l);return {m,q,k,h,l,tin:tube(C.muted),tout:tube(C.imag),x:0};});
 const lR=label('real frames',.26,hex('real')),lI=label('imagined frames',.26,hex('imag')),lT=label('time →',.26,hex('time')),lA=label("agent's actions",.24,hex('act')),lM=label('world model',.24,hex('time'));
 const barG=new THREE.BoxGeometry(1,.035,.035),bR=new THREE.Mesh(barG,std(C.real,{transparent:true})),bI=new THREE.Mesh(barG,std(C.imag,{transparent:true}));B.root.add(lR,lI,lT,lA,lM,bR,bI);
 const labs=[lR,lI,lT,lA,lM,...steps.map(st=>st.l)];labs.forEach(l=>l.userData.s0=l.scale.clone());
 // Narrow canvases squeeze the row, so the labels grow to stay legible.
 function layout(nw){narrow=nw;F=nw?.92:1;sp=nw?1.1:1.25;MY=F/2+.78;AY=MY+.86;const lo=-F/2-.2,top=F*.545+.02,ls=nw?1.45:1;
  labs.forEach(l=>l.scale.set(l.userData.s0.x*ls,l.userData.s0.y*ls,1));
  const ar=(x0,y0,x1,y1,cx,cy)=>new THREE.QuadraticBezierCurve3(V(x0,y0,0),V(cx,cy,0),V(x1,y1,0));
  steps.forEach((st,k)=>{const i=NR+k,x=st.x=(X(i-1)+X(i))/2;st.m.position.set(x,MY,0);st.k.position.set(x,AY-.15,0);st.l.position.set(x,AY+.25+.13*ls,0);st.h.position.set(X(i),top+.08,0);
   [[st.tin,ar(X(i-1),top,x-.25,MY,X(i-1),MY)],[st.tout,ar(x+.25,MY,X(i),top+.14,X(i),MY)]].forEach(([m,cv])=>{m.geometry.dispose();m.geometry=new THREE.TubeGeometry(cv,24,.022,6,false);});});
  const span=(b,a,z)=>{b.position.set((X(a)+X(z))/2,lo,0);b.scale.x=X(z)-X(a)+1.09*F;};span(bR,0,NR-1);span(bI,NR,N-1);
  lR.position.set((X(0)+X(NR-1))/2,lo-.17-.13*ls,0);lI.position.set((X(NR)+X(N-1))/2,lo-.17-.13*ls,0);lT.position.set(0,lo-.37-.39*ls,0);
  const rx=steps[0].x-.45;lA.position.set(rx-lA.scale.x/2,AY,0);lM.position.set(rx-lM.scale.x/2,MY,0);
  B.hfov=s.hfov=nw?48:53;s.w=0;}
 const T_END=6;let t0=performance.now()/1000;document.getElementById('hero-replay').onclick=()=>{t0=performance.now()/1000;};
 const fade=(o,u)=>{o.material.opacity=u;o.visible=u>.002;};
 const grow=(m,u)=>{m.geometry.setDrawRange(0,Math.floor(u*24)*36);m.visible=u>0;};
 function set(e){const sg=(a,b)=>smooth(a,b,e);
  for(let i=0;i<NR;i++){const u=sg(.1+i*.22,.5+i*.22),g=frames[i];g.position.set(X(i),0,0);g.scale.setScalar(F*Math.max(.001,u));g.visible=u>0;}
  fade(lT,sg(.4,.8));fade(lR,sg(.8,1.2));fade(bR,sg(.8,1.2));fade(lM,sg(1.3,1.7));fade(lA,sg(1.4,1.8));fade(lI,sg(5.1,5.5));fade(bI,sg(5.1,5.5));
  steps.forEach((st,k)=>{const t=1.15+k*1.1,i=NR+k,u1=sg(t,t+.3),ua=sg(t+.1,t+.45),uc=sg(t+.35,t+.5),u2=sg(t+.5,t+.95),g=frames[i];
   st.m.scale.setScalar(Math.max(.001,u1));st.m.visible=u1>0;st.m.material.emissiveIntensity=.4*sg(t+.3,t+.5)*(1-sg(t+.8,t+1.1));grow(st.tin,u1);
   st.q.position.set(st.x,AY+1.1*(1-ua),0);fade(st.q,ua);fade(st.l,ua);st.k.scale.y=Math.max(.001,(AY-MY-.4)*uc);st.k.visible=uc>0;
   grow(st.tout,u2);st.h.visible=u2>=1;
   g.position.set(st.x+(X(i)-st.x)*u2,MY*(1-u2),0);g.scale.setScalar(F*(.3+.7*u2));g.visible=u2>0;});}
 layout(false);set(reduced?T_END:0);
 return Object.assign(B,{update(t){const nw=s.wrap.clientWidth/Math.max(1,s.wrap.clientHeight)<1.6;if(nw!==narrow)layout(nw);set(reduced?T_END:Math.min(T_END,t-t0));}});}
'''
s = once(s, "/* ---------- planning with a learned model ---------- */", HERO + "\n/* ---------- planning with a learned model ---------- */")
s = once(s, "stage('f-plan',buildPlan);", "stage('f-hero',buildHero);stage('f-plan',buildPlan);")

open(D + 'page.src.html', 'w').write(s)
print('ok', len(s))
