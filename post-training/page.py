import sys
sys.path.insert(0, '/Users/rahulmishra/paper-explainers/shared')
import common
from common import once

D = '/Users/rahulmishra/paper-explainers/post-training/'
s = open(D + 'original.html').read()

s = common.apply(s, hl='var(--acc)', extra_css='''
.t-pol{color:var(--pol)}.t-ref{color:var(--ref)}.t-rew{color:var(--rew)}.t-good{color:var(--good)}.t-bad{color:var(--bad)}
header.hero .lede{margin-bottom:26px}
#f-hero .canvas{height:clamp(320px,44vw,420px)}
@media (max-width:600px){#f-hero .canvas{height:380px}}
''')

# ---- hero: preference updates pull the policy while the frozen reference holds it back ----
s = once(s, 'reasoning-focused reinforcement learning.</p>\n</div></header>',
'''reasoning-focused reinforcement learning.</p>
 <figure class="fig" id="f-hero" style="margin:0"><div class="canvas" role="img" aria-label="Candidate replies to the prompt What is the capital of France?, each with a blue probability bar inside a grey outline of the frozen reference and a green or red reward dot. In a few pulls the bars of the Paris answers grow and the bars of the other question and of Lyon shrink, while grey springs stretch back toward the reference."><span class="hint">Drag to rotate</span></div>
  <figcaption><div class="controls"><button class="btn" type="button" id="hero-replay">Replay</button>
   <span class="legend"><i style="background:var(--pol)"></i>policy</span><span class="legend"><i style="background:var(--ref)"></i>frozen reference</span><span class="legend"><i style="background:var(--good)"></i>preferred</span><span class="legend"><i style="background:var(--bad)"></i>rejected</span></div>
   <p>Post-training in one picture. The policy starts as the pretrained model, which most often answers a question with another question. Each update moves probability toward the preferred responses and away from the rejected ones. The frozen reference (grey outlines) is where the policy started; the KL penalty pulls back toward it like a spring, so the policy shifts without putting all its probability on a single answer. The responses, rewards and probabilities in this picture are made up for illustration.</p></figcaption></figure>
</div></header>''')

# ---- equations: Unicode text -> TeX, colors from the page tokens; originals kept in comments ----
EQ = [
 ('''<div class="eq">P( y<sub>w</sub> ≻ y<sub>l</sub> | x ) = σ( r(x, y<sub>w</sub>) − r(x, y<sub>l</sub>) ), &nbsp;&nbsp; L<sub>RM</sub> = − log σ( r(x, y<sub>w</sub>) − r(x, y<sub>l</sub>) )
  <small>''',
  r'''<div class="eq" id="eq-bt" data-viz="f-rm"><div class="tex tex-display">P\big(\htmlData{link=chosen, label=the chosen responses}{\htmlClass{t-good}{y_w}} \succ \htmlData{link=rejected, label=the rejected responses}{\htmlClass{t-bad}{y_l}} \mid x\big) = \sigma\big(\htmlData{link=reward, label=the learned reward}{\htmlClass{t-rew}{r}(x, \htmlClass{t-good}{y_w}) - \htmlClass{t-rew}{r}(x, \htmlClass{t-bad}{y_l})}\big), \qquad \htmlData{link=loss, label=the Bradley-Terry loss}{L_{\mathrm{RM}}} = -\log \sigma\big(\htmlClass{t-rew}{r}(x, \htmlClass{t-good}{y_w}) - \htmlClass{t-rew}{r}(x, \htmlClass{t-bad}{y_l})\big)</div><!-- original: P( y<sub>w</sub> ≻ y<sub>l</sub> | x ) = σ( r(x, y<sub>w</sub>) − r(x, y<sub>l</sub>) ), &nbsp;&nbsp; L<sub>RM</sub> = − log σ( r(x, y<sub>w</sub>) − r(x, y<sub>l</sub>) ) -->
  <small>'''),
 ('''<div class="eq">max<sub>π</sub> E<sub>x, y ~ π</sub> [ r(x, y) ] − β · KL( π(·|x) ‖ π<sub>ref</sub>(·|x) )
  <small>''',
  r'''<div class="eq"><div class="tex tex-display">\max_{\htmlClass{t-pol}{\pi}}\; \mathbb{E}_{x,\, y \sim \htmlClass{t-pol}{\pi}}\big[\, \htmlClass{t-rew}{r}(x, y) \,\big] - \beta \cdot \mathrm{KL}\big(\htmlClass{t-pol}{\pi}(\cdot \mid x) \,\big\|\, \htmlClass{t-ref}{\pi_{\mathrm{ref}}}(\cdot \mid x)\big)</div><!-- original: max<sub>π</sub> E<sub>x, y ~ π</sub> [ r(x, y) ] − β · KL( π(·|x) ‖ π<sub>ref</sub>(·|x) ) -->
  <small>'''),
 ('''<div class="eq">L<sub>DPO</sub> = − log σ( β [ log π(y<sub>w</sub>|x)/π<sub>ref</sub>(y<sub>w</sub>|x) − log π(y<sub>l</sub>|x)/π<sub>ref</sub>(y<sub>l</sub>|x) ] )
  <small>''',
  r'''<div class="eq" id="eq-dpo" data-viz="f-dpo"><div class="tex tex-display">\htmlData{link=loss, label=the DPO loss}{L_{\mathrm{DPO}}} = -\log \sigma\left(\htmlData{link=beta, label=the beta control}{\beta} \left[\htmlData{link=chosen, label=the chosen log-ratio}{\log \frac{\htmlClass{t-pol}{\pi}(\htmlClass{t-good}{y_w} \mid x)}{\htmlClass{t-ref}{\pi_{\mathrm{ref}}}(\htmlClass{t-good}{y_w} \mid x)}} - \htmlData{link=rejected, label=the rejected log-ratio}{\log \frac{\htmlClass{t-pol}{\pi}(\htmlClass{t-bad}{y_l} \mid x)}{\htmlClass{t-ref}{\pi_{\mathrm{ref}}}(\htmlClass{t-bad}{y_l} \mid x)}}\right]\right)</div><!-- original: L<sub>DPO</sub> = − log σ( β [ log π(y<sub>w</sub>|x)/π<sub>ref</sub>(y<sub>w</sub>|x) − log π(y<sub>l</sub>|x)/π<sub>ref</sub>(y<sub>l</sub>|x) ] ) -->
  <small>'''),
 ('''<div class="eq">A<sub>i</sub> = ( r<sub>i</sub> − mean(r<sub>1..G</sub>) ) / std(r<sub>1..G</sub>)
  <small>''',
  r'''<div class="eq" id="eq-grpo" data-viz="f-grpo"><div class="tex tex-display">\htmlData{link=adv, label=the advantages}{A_i} = \frac{\htmlData{link=reward, label=the rewards}{r_i} - \htmlData{link=group, label=the group mean}{\mathrm{mean}(r_{1\ldots G})}}{\htmlData{link=group, label=the group spread}{\mathrm{std}(r_{1\ldots G})}}</div><!-- original: A<sub>i</sub> = ( r<sub>i</sub> − mean(r<sub>1..G</sub>) ) / std(r<sub>1..G</sub>) -->
  <small>'''),
]
for a, b in EQ:
    s = once(s, a, b)
s = once(s, "Only differences in reward matter, so the reward's absolute level is arbitrary.</small></div>",
         "Only differences in reward matter, so the reward's absolute level is arbitrary.</small><span class=\"eq-hint\">Hover or tap a term to see it in the figure below.</span></div>")

# ---- figure elements the equation terms point at ----
s = once(s, '<div class="status" id="rm-read" aria-live="polite">', '<div class="status" id="rm-read" aria-live="polite" data-link="loss">')
s = once(s, '<label class="ctl">β <input type="range" id="dpo-b"', '<label class="ctl" data-link="beta">β <input type="range" id="dpo-b"')
s = once(s, '<div class="status" id="dpo-read" aria-live="polite">', '<div class="status" id="dpo-read" aria-live="polite" data-link="loss">')
s = once(s, '<label class="ctl">Group size G <input', '<label class="ctl" data-link="group">Group size G <input')
# The SFT figure had no id; QA shoots figures by id.
s = once(s, '<figure class="fig"><div class="pad chat" id="sft-chat"', '<figure class="fig" id="f-sft"><div class="pad chat" id="sft-chat"')

# ---- reward model: spheres grow for chosen/rejected, the surface glows for the reward ----
s = once(s, "B.root.add(new THREE.Mesh(geo,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,transparent:true,opacity:.85,roughness:.7})));",
            "const surf=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,transparent:true,opacity:.85,roughness:.7}));B.root.add(surf);")
s = once(s, "let pairs=[],w=[0,0,0,0,0],grp=new THREE.Group(),running=false;",
            "let pairs=[],w=[0,0,0,0,0],grp=new THREE.Group(),running=false,wS=[],lS=[];")
# ArrowHelper has no geometry of its own, so the original per-child dispose threw on "New comparisons".
s = once(s, "w=[0,0,0,0,0];while(grp.children.length){const c=grp.children.pop();c.geometry.dispose();c.material.dispose();}",
            "w=[0,0,0,0,0];while(grp.children.length)disposeTree(grp.children.pop());wS=[];lS=[];")
s = once(s, "m2.position.set(ls[0],.05,-ls[1]);grp.add(m2);});}",
            "m2.position.set(ls[0],.05,-ls[1]);grp.add(m2);wS.push(m1);lS.push(m2);});}")
s = once(s, "return Object.assign(B,{update(){if(running){for(let k=0;k<3;k++)step();draw();}}});}",
            "return Object.assign(B,{update(){if(running){for(let k=0;k<3;k++)step();draw();}\n"
            "  const gw=s.g('chosen'),gl=s.g('rejected');wS.forEach(m=>m.scale.setScalar(1+1.6*gw));lS.forEach(m=>m.scale.setScalar(1+1.6*gl));\n"
            "  surf.material.emissive.copy(C.acc);surf.material.emissiveIntensity=.7*s.g('reward');}});}")

# ---- DPO: axis labels grow and the axes light for the log-ratios, the surface glows for the loss ----
s = once(s, "ll.position.set(.3,.1,-R+.3);B.root.add(ll);",
            "ll.position.set(.3,.1,-R+.3);B.root.add(ll);\n"
            " const axW=new THREE.Mesh(new THREE.BoxGeometry(2*R,.07,.07),new THREE.MeshBasicMaterial({color:C.good,transparent:true,opacity:0,depthTest:false}));\n"
            " const axL=new THREE.Mesh(new THREE.BoxGeometry(.07,.07,2*R),new THREE.MeshBasicMaterial({color:C.bad,transparent:true,opacity:0,depthTest:false}));\n"
            " [axW,axL].forEach(m=>{m.renderOrder=9;m.visible=false;B.root.add(m);});[lw,ll].forEach(l=>l.userData.s0=l.scale.clone());\n"
            " const grow=(l,g)=>l.scale.set(l.userData.s0.x*(1+.4*g),l.userData.s0.y*(1+.4*g),1);")
s = once(s, "return Object.assign(B,{update(){if(running){step();place();}}});}",
            "return Object.assign(B,{update(){if(running){step();place();}\n"
            "  const gw=s.g('chosen'),gl=s.g('rejected');grow(lw,gw);grow(ll,gl);axW.material.opacity=.9*gw;axW.visible=gw>0;axL.material.opacity=.9*gl;axL.visible=gl>0;\n"
            "  surf.material.emissive.copy(C.acc);surf.material.emissiveIntensity=.4*s.g('loss');}});}")

# ---- GRPO: sticks for the advantages, spheres and the "% right" labels for the rewards, the group label ----
s = once(s, "const ref=[2,1.4,1,.3,-.4,-1.2];let th=ref.slice();", "const ref=[2,1.4,1,.3,-.4,-1.2];let th=ref.slice();const rL=[];")
s = once(s, "l2.position.set((i-(S-1)/2)*1.1,-.55,.7);B.root.add(l2);return m;});",
            "l2.position.set((i-(S-1)/2)*1.1,-.55,.7);B.root.add(l2);rL.push(l2);return m;});")
s = once(s, "const grp=new THREE.Group();B.root.add(grp);let running=false,it=0,acc=0;",
            "const grp=new THREE.Group();B.root.add(grp);const sph=[],stk=[];let running=false,it=0,acc=0;")
s = once(s, "function draw(){while(grp.children.length){const c=grp.children.pop();c.geometry.dispose();c.material.dispose();}const p=pi();",
            "function draw(){while(grp.children.length){const c=grp.children.pop();c.geometry.dispose();c.material.dispose();}sph.length=0;stk.length=0;const p=pi();")
s = once(s, "m.position.set(x,3.9,-1.6);grp.add(m);", "m.position.set(x,3.9,-1.6);grp.add(m);sph.push(m);")
s = once(s, "st.position.set(x,3.55+(a>=0?-Math.abs(a)*.25:Math.abs(a)*.25)-.3,-1.6);grp.add(st);",
            "st.position.set(x,3.55+(a>=0?-Math.abs(a)*.25:Math.abs(a)*.25)-.3,-1.6);grp.add(st);stk.push(st);")
s = once(s, "lg.position.set(0,4.4,-1.6);B.root.add(lg);\n return Object.assign(B,{update(t,dt){if(running){acc+=dt;if(acc>(reduced?.6:.25)){acc=0;step();}}bars.forEach(m=>m.scale.y+=(m.userData.h-m.scale.y)*.15);}});}",
            "lg.position.set(0,4.4,-1.6);B.root.add(lg);\n"
            " [...rL,lg].forEach(l=>l.userData.s0=l.scale.clone());const grow=(ls,g)=>ls.forEach(l=>l.scale.set(l.userData.s0.x*(1+.4*g),l.userData.s0.y*(1+.4*g),1));\n"
            " const glow=(m,g)=>{m.material.emissive.copy(m.material.color);m.material.emissiveIntensity=.7*g;};\n"
            " return Object.assign(B,{update(t,dt){if(running){acc+=dt;if(acc>(reduced?.6:.25)){acc=0;step();}}bars.forEach(m=>m.scale.y+=(m.userData.h-m.scale.y)*.15);\n"
            "  const ga=s.g('adv'),gr=s.g('reward');stk.forEach(m=>{m.scale.set(1+2*ga,1,1+2*ga);glow(m,ga);});sph.forEach(m=>{m.scale.setScalar(1+.6*gr);glow(m,gr);});\n"
            "  grow(rL,gr);grow([lg],Math.max(ga,s.g('group')));}});}")

# ---- hero builder ----
HERO = r'''
/* ---------- hero: preference updates pull the policy, the frozen reference holds it back (illustrative numbers) ---------- */
function buildHero(s){const B=base([0,.45,9.4],[0,0,0],56);s.tRotY=-.14;s.tRotX=.06;s.vmin=27;
 const RESP=['What is the capital of Germany?','The capital of France is Paris.','Paris.','Lyon.'];
 // Made-up reference probabilities and rewards. Each pull raises u; the policy is pi_ref * exp(u*K*r), normalised,
 // so it never leaves the reference's support. U grows by less each time, so the pulls shrink.
 const P0=[.44,.22,.18,.16],RW=[-1.2,1,.8,-1.6],U=[0,.45,.72,.88,1],K=1.25,SX=6,DX=4;
 const pol=u=>{const w=P0.map((p,i)=>p*Math.exp(u*K*RW[i]));const z=w.reduce((a,b)=>a+b,0);return w.map(v=>v/z);};
 const G=new THREE.Group();B.root.add(G);
 const bg=new THREE.BoxGeometry(1,.3,.3);bg.translate(.5,0,0);const gg=new THREE.BoxGeometry(1,.44,.44);gg.translate(.5,0,0);const eg=new THREE.EdgesGeometry(gg);
 const zz=[];for(let j=0;j<=14;j++)zz.push(V(j/14,j%14?(j%2?.06:-.06):0,0));const cg=new THREE.ConeGeometry(.09,.24,14);
 const tag=o=>{o.userData.a=o.scale.x/o.scale.y;G.add(o);return o;};
 const rows=RESP.map((t,i)=>{const good=RW[i]>0,col=good?C.good:C.bad;
  const bar=new THREE.Mesh(bg,std(C.pol)),gh=new THREE.Mesh(gg,std(C.ref,{transparent:true,opacity:0,depthWrite:false}));
  const ed=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:C.ref,transparent:true,opacity:0}));
  const sp=line(zz,C.ref,{opacity:0}),dot=new THREE.Mesh(new THREE.SphereGeometry(.14,18,12),std(col));
  const cone=new THREE.Mesh(cg,std(col,{transparent:true,opacity:0}));cone.rotation.z=good?-Math.PI/2:Math.PI/2;
  [bar,gh,ed,sp,dot,cone].forEach(o=>G.add(o));return {bar,gh,ed,sp,dot,cone,l:tag(label(t,.24,hex('ink'))),y:0};});
 const pr=tag(label('What is the capital of France?',.3,hex('ink'))),hP=tag(label('probability of each response',.2,hex('muted'))),hR=tag(label('reward',.2,hex('rew')));
 // align: -1 puts the text's left edge at x, 1 its right edge, 0 centers it; .138 is the label canvas padding as a fraction of its height.
 const lab=(o,z,x,y,al)=>{const w=o.userData.a*z;o.scale.set(w,z,1);o.position.set(al<0?x+w/2-.138*z:al>0?x-w/2+.138*z:x,y,0);};
 // Wide canvases put each response left of its bar; narrow ones put it above, so the scene is about as tall as it is wide.
 let narrow=null;
 function layout(nw){narrow=nw;const gap=nw?1:.75,y0=nw?1:.95,z=nw?.27:.24;
  rows.forEach((r,i)=>{const y=r.y=y0-i*gap;[r.bar,r.gh,r.ed].forEach(o=>o.position.set(0,y,0));r.dot.position.set(DX,y,0);if(nw)lab(r.l,z,0,y+.42,-1);else lab(r.l,z,-.3,y,1);});
  if(nw){lab(pr,.28,0,2.45,-1);lab(hP,.2,0,1.85,-1);lab(hR,.2,DX,1.85,0);G.position.set(-2.05,-.36,0);}
  else{lab(pr,.3,.42,2.3,0);lab(hP,.2,0,1.55,-1);lab(hR,.2,DX,1.55,0);G.position.set(-.42,-.47,0);}
  B.hfov=s.hfov=nw?30:56;s.vmin=nw?32:27;s.w=0;}
 const T_END=6,TS=[2.7,3.45,4.2,4.95],MV=.55;let t0=performance.now()/1000;document.getElementById('hero-replay').onclick=()=>{t0=performance.now()/1000;};
 const fade=(o,u)=>{o.material.opacity=u;o.visible=u>.002;};
 function set(e){const sg=(a,b)=>smooth(a,b,e);let u=0,pulse=0;
  TS.forEach((ts,j)=>{if(e>=ts){u=U[j]+(U[j+1]-U[j])*sg(ts,ts+MV);pulse=e<ts+MV?Math.sin(Math.PI*(e-ts)/MV):0;}});
  const p=pol(u),gr=sg(1.7,2.2);fade(pr,sg(0,.5));fade(hP,sg(.5,1));fade(hR,sg(2,2.4));
  rows.forEach((r,i)=>{fade(r.l,sg(.15+i*.12,.6+i*.12));const L0=P0[i]*SX,L=p[i]*SX*sg(.5+i*.1,1.4+i*.1),d=L-L0;
   r.bar.scale.x=Math.max(.001,L);r.bar.visible=L>.01;r.gh.scale.x=r.ed.scale.x=L0;r.gh.material.opacity=.16*gr;r.gh.visible=gr>.002;fade(r.ed,.85*gr);
   r.sp.position.set(L0,r.y-.3,0);r.sp.scale.x=Math.abs(d)>.001?d:.001;fade(r.sp,Math.abs(d)>.04?.9*gr:0);
   const pop=sg(2+i*.1,2.4+i*.1);r.dot.scale.setScalar(Math.max(.001,pop));r.dot.visible=pop>0;r.dot.material.emissive.copy(r.dot.material.color);r.dot.material.emissiveIntensity=.5*pulse;
   r.cone.position.set(L+.22,r.y,0);fade(r.cone,.95*pulse);});}
 layout(false);set(reduced?T_END:0);
 return Object.assign(B,{update(t){const nw=s.wrap.clientWidth/Math.max(1,s.wrap.clientHeight)<1.6;if(nw!==narrow)layout(nw);set(reduced?T_END:Math.min(T_END,t-t0));}});}
'''
s = once(s, "/* ---------- pipeline ---------- */", HERO + "\n/* ---------- pipeline ---------- */")
s = once(s, "stage('f-pipe',buildPipe);", "stage('f-hero',buildHero);stage('f-pipe',buildPipe);")

open(D + 'page.src.html', 'w').write(s)
print('ok', len(s))
