import sys
sys.path.insert(0, '/Users/rahulmishra/paper-explainers/shared')
import common
from common import once

D = '/Users/rahulmishra/paper-explainers/transformers/'
s = open(D + 'original.html').read()

s = common.apply(s, hl='var(--acc)', extra_css='''
.t-q{color:var(--q)}.t-k{color:var(--k)}.t-v{color:var(--v)}.t-acc{color:var(--acc)}
header.hero .lede{margin-bottom:26px}
#f-hero .canvas{height:clamp(260px,34vw,380px)}
''')

# ---- hero: one prediction step replaces the static token chips ----
s = once(s, s[s.index(' <div class="toks" aria-hidden="true">'):s.index('</div>\n</div></header>') + len('</div>\n')],
''' <figure class="fig" id="f-hero" style="margin:0"><div class="canvas" role="img" aria-label="One prediction step: six orange token blocks read The cat sat on the mat; blue arcs of different thickness run from an empty seventh slot back to each token; green values flow back along the arcs; three purple probability bars rise and the tallest, because, fills the slot."><span class="hint">Drag to rotate</span></div>
  <figcaption><div class="controls"><button class="btn" type="button" id="hero-replay">Replay</button>
   <span class="legend"><i style="background:var(--k)"></i>keys of earlier tokens</span><span class="legend"><i style="background:var(--q)"></i>the new query</span><span class="legend"><i style="background:var(--v)"></i>values</span><span class="legend"><i style="background:var(--acc)"></i>next token</span></div>
   <p>One step of generation. The empty slot's query compares itself with every earlier token's key; arc thickness is the attention weight. The values flow back and mix into the slot, and the model turns the result into a probability for every possible next token. The weights and probabilities in this picture are made up for illustration.</p></figcaption></figure>
''')

# ---- equations: Unicode text -> TeX, colors from the page tokens; originals kept in comments ----
EQ = [
 ('''<div class="eq">Attention(Q, K, V) = softmax( Q Kᵀ / √d<sub>k</sub> + M ) V
   <small>''',
  '''<div class="eq" id="eq-attn" data-viz="f-attn"><div class="tex tex-display">\\mathrm{Attention}(\\htmlClass{t-q}{Q}, \\htmlClass{t-k}{K}, \\htmlClass{t-v}{V}) = \\htmlData{link=softmax, label=the attention weights}{\\mathrm{softmax}}\\!\\left(\\frac{\\htmlData{link=q, label=the queries}{\\htmlClass{t-q}{Q}}\\,\\htmlData{link=k, label=the keys}{\\htmlClass{t-k}{K}^{\\top}}}{\\htmlData{link=scale, label=the score scale}{\\sqrt{d_k}}} + \\htmlData{link=mask, label=the causal mask}{M}\\right)\\htmlClass{t-v}{V}</div><!-- original: Attention(Q, K, V) = softmax( Q Kᵀ / √d<sub>k</sub> + M ) V -->
   <small>'''),
 ('''<div class="eq">h = x + Attention( Norm(x) ) &nbsp;&nbsp; → &nbsp;&nbsp; out = h + MLP( Norm(h) )
  <small>''',
  '''<div class="eq"><div class="tex tex-display">h = x + \\mathrm{Attention}\\big(\\mathrm{Norm}(x)\\big) \\quad\\longrightarrow\\quad \\mathrm{out} = h + \\mathrm{MLP}\\big(\\mathrm{Norm}(h)\\big)</div><!-- original: h = x + Attention( Norm(x) ) &nbsp;&nbsp; → &nbsp;&nbsp; out = h + MLP( Norm(h) ) -->
  <small>'''),
 ('''<div class="eq">⟨ R(mθ) q, R(nθ) k ⟩ = ⟨ q, R((n − m)θ) k ⟩
   <small>''',
  '''<div class="eq" id="eq-rope" data-viz="f-rope"><div class="tex tex-display">\\big\\langle\\, \\htmlData{link=q, label=the rotated query}{R(m\\theta)\\,\\htmlClass{t-q}{q}},\\; \\htmlData{link=k, label=the rotated key}{R(n\\theta)\\,\\htmlClass{t-k}{k}} \\,\\big\\rangle = \\big\\langle\\, \\htmlClass{t-q}{q},\\; \\htmlData{link=rel, label=the relative rotation}{R\\big((n-m)\\theta\\big)}\\,\\htmlClass{t-k}{k} \\,\\big\\rangle</div><!-- original: ⟨ R(mθ) q, R(nθ) k ⟩ = ⟨ q, R((n − m)θ) k ⟩ -->
   <small>'''),
 ('''<div class="eq">L = − (1/T) Σ<sub>t</sub> log p<sub>θ</sub>( x<sub>t</sub> | x<sub>&lt;t</sub> )</div>''',
  '''<div class="eq"><div class="tex tex-display">L = -\\frac{1}{T}\\sum_{t} \\log p_{\\theta}\\big(x_t \\mid x_{&lt;t}\\big)</div><!-- original: L = − (1/T) Σ<sub>t</sub> log p<sub>θ</sub>( x<sub>t</sub> | x<sub>&lt;t</sub> ) --></div>'''),
 ('''<div class="eq">KV memory = 2 × layers × KV heads × head dim × tokens × bytes per value
   <small>''',
  '''<div class="eq" id="eq-kv" data-viz="f-kv"><div class="tex tex-display">\\text{KV memory} = \\htmlData{link=kv, label=keys and values}{2} \\times \\htmlData{link=layers, label=the layers}{\\text{layers}} \\times \\htmlData{link=heads, label=the KV heads}{\\text{KV heads}} \\times \\text{head dim} \\times \\htmlData{link=tokens, label=the tokens}{\\text{tokens}} \\times \\text{bytes per value}</div><!-- original: KV memory = 2 × layers × KV heads × head dim × tokens × bytes per value -->
   <small>'''),
 ('''<div class="eq">L(N, D) = E + A / N<sup>α</sup> + B / D<sup>β</sup> &nbsp;&nbsp; with E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28
   <small>''',
  '''<div class="eq" id="eq-chin" data-viz="f-scale"><div class="tex tex-display">L(N, D) = \\htmlData{link=floor, label=the irreducible loss}{E} + \\htmlData{link=params, label=the model-size term}{\\frac{A}{N^{\\alpha}}} + \\htmlData{link=data, label=the data term}{\\frac{B}{D^{\\beta}}} \\qquad \\text{with } E = 1.69,\\; A = 406.4,\\; B = 410.7,\\; \\alpha = 0.34,\\; \\beta = 0.28</div><!-- original: L(N, D) = E + A / N<sup>α</sup> + B / D<sup>β</sup> &nbsp;&nbsp; with E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28 -->
   <small>'''),
]
for a, b in EQ:
    s = once(s, a, b)
# The equations that point at a figure say so once, under the first one.
s = once(s, 'Without it (as in BERT-style encoders), attention is bidirectional.</small></div>',
         'Without it (as in BERT-style encoders), attention is bidirectional.</small><span class="eq-hint">Hover or tap a term to see it in the figure below.</span></div>')

# ---- figure controls the equation terms point at ----
s = once(s, '<label class="ctl">Score scale <input', '<label class="ctl" data-link="scale">Score scale <input')
s = once(s, '<label class="ctl">Query position m <input', '<label class="ctl" data-link="q">Query position m <input')
s = once(s, '<label class="ctl">Key position n <input', '<label class="ctl" data-link="k">Key position n <input')
s = once(s, '<div class="status" id="rp-read" aria-live="polite">', '<div class="status" id="rp-read" aria-live="polite" data-link="rel">')
s = once(s, '<div class="seg" role="group" aria-label="Attention type" id="kv-type">', '<div class="seg" role="group" aria-label="Attention type" id="kv-type" data-link="heads">')
s = once(s, '<label class="ctl">Context length <input', '<label class="ctl" data-link="tokens">Context length <input')

# ---- 3D: equation glows ----
# attention: q/k labels grow, masked cells and the softmax bars glow
s = once(s, "const l=label(TOK[i],.24,hex('q'));l.position.set(-(N+1)/2*sp-.35,.1,(i-(N-1)/2)*sp);B.root.add(l);const l2=label(TOK[i],.24,hex('k'));l2.position.set((i-(N-1)/2)*sp,.1,-(N+1)/2*sp-.25);B.root.add(l2);}",
            "const l=label(TOK[i],.24,hex('q'));l.position.set(-(N+1)/2*sp-.35,.1,(i-(N-1)/2)*sp);B.root.add(l);const l2=label(TOK[i],.24,hex('k'));l2.position.set((i-(N-1)/2)*sp,.1,-(N+1)/2*sp-.25);B.root.add(l2);qL.push(l);kL.push(l2);}")
s = once(s, "const g=new THREE.BoxGeometry(.32,1,.32);g.translate(0,.5,0);const bars=[];",
            "const g=new THREE.BoxGeometry(.32,1,.32);g.translate(0,.5,0);const bars=[],qL=[],kL=[];")
s = once(s, "return Object.assign(B,{update(){for(const row of bars)for(const m of row)m.scale.y+=(m.userData.h-m.scale.y)*.15;}});}",
            "[...qL,...kL].forEach(l=>l.userData.s0=l.scale.clone());\n"
            " const grow=(ls,g)=>ls.forEach(l=>l.scale.set(l.userData.s0.x*(1+.4*g),l.userData.s0.y*(1+.4*g),1));\n"
            " return Object.assign(B,{update(){const gm=s.g('mask'),gs=s.g('softmax');grow(qL,s.g('q'));grow(kL,s.g('k'));\n"
            "  for(let i=0;i<N;i++)for(let j=0;j<N;j++){const m=bars[i][j];m.scale.y+=(m.userData.h-m.scale.y)*.15;const e=j>i?gm:0;\n"
            "   m.material.emissive.copy(e>0?C.con:C.acc);m.material.emissiveIntensity=Math.max(e*.8,gs*.55);}}});}")
# RoPE: the rotated arrows lengthen
s = once(s, "discs[i].ka.setDirection(V(b[0]/lb,0,-b[1]/lb));discs[i].ka.setLength(Math.min(1.1,.45+lb*.3),.2,.12);}}});}",
            "discs[i].ka.setDirection(V(b[0]/lb,0,-b[1]/lb));discs[i].ka.setLength(Math.min(1.1,.45+lb*.3),.2,.12);\n"
            "  const gq=Math.max(s.g('q'),s.g('rel')),gk=Math.max(s.g('k'),s.g('rel'));\n"
            "  if(gq)discs[i].qa.setLength(Math.min(1.1,.45+la*.3)*(1+.3*gq),.2+.12*gq,.12+.08*gq);if(gk)discs[i].ka.setLength(Math.min(1.1,.45+lb*.3)*(1+.3*gk),.2+.12*gk,.12+.08*gk);}}});}")
# KV cache: cubes glow for the 2 (keys and values), layer labels and the token label grow
s = once(s, "const lb=label('layer '+(l+1),.26,hex('muted'));lb.position.set(-(T+2)/2*.42-.3,l*.8+.3,0);B.root.add(lb);}",
            "const lb=label('layer '+(l+1),.26,hex('muted'));lb.position.set(-(T+2)/2*.42-.3,l*.8+.3,0);B.root.add(lb);layL.push(lb);}")
s = once(s, "const L=4,T=14;const cubes=[];", "const L=4,T=14;const cubes=[],layL=[];")
s = once(s, "const lt=label('tokens generated →',.28,hex('muted'));lt.position.set(0,-.2,.6);B.root.add(lt);",
            "const lt=label('tokens generated →',.28,hex('muted'));lt.position.set(0,-.2,.6);B.root.add(lt);\n"
            " [...layL,lt].forEach(l=>l.userData.s0=l.scale.clone());const grow=(ls,g)=>ls.forEach(l=>l.scale.set(l.userData.s0.x*(1+.4*g),l.userData.s0.y*(1+.4*g),1));")
s = once(s, "cubes.forEach(r=>r.forEach(c=>{c[2]=Math.max(0,c[2]-dt*2.2);c[0].material.emissiveIntensity=c[1].material.emissiveIntensity=c[2]*.9;",
            "const gkv=s.g('kv');grow(layL,s.g('layers'));grow([lt],s.g('tokens'));\n"
            "  cubes.forEach(r=>r.forEach(c=>{c[2]=Math.max(0,c[2]-dt*2.2);c[0].material.emissiveIntensity=c[1].material.emissiveIntensity=Math.max(c[2]*.9,gkv*.8);")
# scaling: a floor plane for E, the axis labels for the N and D terms
s = once(s, "[[7,'10M'],[9,'1B'],[11,'100B'],[12,'1T']].forEach(([n,t])=>{const l=label(t+' params',.22,hex('muted'));l.position.set(X(n),0,W/2+.35);B.root.add(l);});",
            "const nL=[],dL=[];[[7,'10M'],[9,'1B'],[11,'100B'],[12,'1T']].forEach(([n,t])=>{const l=label(t+' params',.22,hex('muted'));l.position.set(X(n),0,W/2+.35);B.root.add(l);nL.push(l);});")
s = once(s, "[[9,'1B'],[11,'100B'],[13,'10T']].forEach(([d,t])=>{const l=label(t+' tokens',.22,hex('muted'));l.position.set(-W/2-.6,0,Z(d));B.root.add(l);});",
            "[[9,'1B'],[11,'100B'],[13,'10T']].forEach(([d,t])=>{const l=label(t+' tokens',.22,hex('muted'));l.position.set(-W/2-.6,0,Z(d));B.root.add(l);dL.push(l);});\n"
            " const floor=new THREE.Mesh(new THREE.PlaneGeometry(W,W),new THREE.MeshBasicMaterial({color:C.acc,transparent:true,opacity:0,side:THREE.DoubleSide,depthWrite:false}));floor.rotation.x=-Math.PI/2;floor.position.y=.005;B.root.add(floor);\n"
            " [...nL,...dL].forEach(l=>l.userData.s0=l.scale.clone());const grow=(ls,g)=>ls.forEach(l=>l.scale.set(l.userData.s0.x*(1+.45*g),l.userData.s0.y*(1+.45*g),1));")
s = once(s, "cs.oninput=compute;compute();return B;}",
            "cs.oninput=compute;compute();return Object.assign(B,{update(){floor.material.opacity=.5*s.g('floor');grow(nL,s.g('params'));grow(dL,s.g('data'));}});}")

# ---- hero builder ----
HERO = r'''
/* ---------- hero: one prediction step (illustrative weights) ---------- */
function buildHero(s){const B=base([1.05,1.45,9.4],[1.05,.8,0],56);s.tRotY=-.12;s.tRotX=.05;s.vmin=30;
 const W=['The','cat','sat','on','the','mat'],A=[.05,.22,.16,.08,.14,.35],N=W.length,ink=hex('ink');
 // Wide canvases put the probability bars right of the slot; narrow ones put them under the row so the row can be larger.
 let narrow=null,sp=1.05;const X=i=>(i-N/2)*sp;
 const tg=new THREE.BoxGeometry(.8,.46,.46);tg.translate(0,.23,0);
 const toks=W.map(w=>{const m=new THREE.Mesh(tg,std(C.k.clone().lerp(C.panel,.25)));m.scale.y=.001;B.root.add(m);const l=label(w,.3,ink);B.root.add(l);return {m,l};});
 const slot=new THREE.Mesh(tg,std(C.acc.clone(),{transparent:true,opacity:0}));B.root.add(slot);
 const edge=new THREE.LineSegments(new THREE.EdgesGeometry(tg),new THREE.LineDashedMaterial({color:C.q,dashSize:.07,gapSize:.05,transparent:true,opacity:0}));edge.computeLineDistances();B.root.add(edge);
 const qm=label('?',.44,hex('q'));B.root.add(qm);
 const arcs=A.map(w=>{const m=new THREE.Mesh(new THREE.BufferGeometry(),std(C.q.clone(),{transparent:true,opacity:.85}));B.root.add(m);
  const dot=new THREE.Mesh(new THREE.SphereGeometry(.05+.13*w,14,10),std(C.v));dot.visible=false;B.root.add(dot);return {m,dot,w,cv:null};});
 const bg=new THREE.BoxGeometry(.42,1,.42);bg.translate(0,.5,0);
 const bars=[['because',.95],['.',.42],['and',.3]].map(([w,h],i)=>{const m=new THREE.Mesh(bg,std(i?C.acc.clone().lerp(C.panel,.55):C.acc.clone()));m.scale.y=.001;B.root.add(m);
  const l=label(w,.24,i?hex('muted'):hex('acc'));B.root.add(l);return {m,l,h};});
 const win=label('because',.3,hex('acc'));B.root.add(win);
 const nA=label('attention weights',.24,hex('q'));B.root.add(nA);const nP=label('next-token probabilities',.22,hex('muted'));B.root.add(nP);
 function layout(nw){narrow=nw;sp=nw?.92:1.05;const by=nw?-1.75:0,bx=i=>nw?X(N)-1.24+i*.62:X(N)+1.05+i*.62;
  toks.forEach(({m,l},i)=>{m.position.x=X(i);l.position.set(X(i),-.32,0);});slot.position.x=edge.position.x=X(N);qm.position.set(X(N),.25,.3);
  arcs.forEach((a,j)=>{a.cv=new THREE.QuadraticBezierCurve3(V(X(N),.5,0),V((X(N)+X(j))/2,.8+.4*(N-j),0),V(X(j),.5,0));a.m.geometry.dispose();a.m.geometry=new THREE.TubeGeometry(a.cv,48,.012+.07*a.w,6,false);});
  bars.forEach((b,i)=>{b.m.position.set(bx(i),by,0);b.l.position.set(bx(i),by-.32,0);});
  nA.position.set(0,2.2,0);nP.position.set(bx(1),by-.74,0);
  B.hfov=s.hfov=nw?42:56;B.camera.position.set(nw?X(N/2):1.05,nw?.9:1.45,9.4);B.camera.lookAt(nw?X(N/2):1.05,nw?.25:.8,0);s.w=0;}
 const T_END=6.3;let t0=performance.now()/1000;document.getElementById('hero-replay').onclick=()=>{t0=performance.now()/1000;};
 const fade=(o,u)=>{o.material.opacity=u;o.visible=u>.002;};
 function set(e){const sg=(a,b)=>smooth(a,b,e);
  toks.forEach(({m,l},i)=>{const u=sg(.1+i*.14,.55+i*.14);m.scale.y=Math.max(.001,u);fade(l,u);});
  const sl=sg(1.05,1.45);edge.material.opacity=sl*(1-sg(5.4,6));qm.material.opacity=sl*(1-sg(5.1,5.5));qm.visible=qm.material.opacity>.002;
  arcs.forEach((a,j)=>{const u=sg(1.5+j*.1,2.5+j*.1);a.m.geometry.setDrawRange(0,Math.floor(u*48)*36);a.m.material.opacity=.85-.35*sg(5.4,6.2);
   const v=sg(3+(N-1-j)*.08,4.1+(N-1-j)*.08);a.dot.visible=v>0&&v<1;if(a.dot.visible)a.dot.position.copy(a.cv.getPoint(1-v));});
  fade(nA,sg(2.5,3.1));slot.material.opacity=.18*sg(3.4,4.3)+.82*sg(5.3,6);
  bars.forEach((b,i)=>{const u=sg(4.2+i*.1,4.9+i*.1);b.m.scale.y=Math.max(.001,b.h*u);b.m.visible=u>0;fade(b.l,u);});
  fade(nP,sg(4.6,5.1));const w=sg(5.2,6),bp=bars[0].m.position;win.position.set(bp.x+(X(N)-bp.x)*w,bars[0].l.position.y+(-.32-bars[0].l.position.y)*w,0);fade(win,sg(5.2,5.4));}
 layout(false);set(reduced?T_END:0);
 return Object.assign(B,{update(t){const nw=s.wrap.clientWidth/Math.max(1,s.wrap.clientHeight)<1.6;if(nw!==narrow)layout(nw);set(reduced?T_END:Math.min(T_END,t-t0));}});}
'''
s = once(s, "/* ---------- attention matrix ---------- */", HERO + "\n/* ---------- attention matrix ---------- */")
s = once(s, "stage('f-attn',buildAttn);", "stage('f-hero',buildHero);stage('f-attn',buildAttn);")

open(D + 'page.src.html', 'w').write(s)
print('ok', len(s))
