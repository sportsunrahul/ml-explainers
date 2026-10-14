import sys
sys.path.insert(0, '/Users/rahulmishra/paper-explainers/shared')
import common
from common import once

D = '/Users/rahulmishra/paper-explainers/seeing-tokens/'
s = open(D + 'original.html').read()

s = common.apply(s, hl='var(--acc)', extra_css='''
.t-enc{color:var(--enc)}.t-code{color:var(--code)}
header.hero .lede{margin-bottom:26px}
#f-hero .canvas{height:clamp(260px,34vw,380px)}
''')

# ---- hero: encode, snap to the codebook, decode ----
s = once(s, 'This guide bridges diffusion models and LLMs.</p>\n</div></header>',
'''This guide bridges diffusion models and LLMs.</p>
 <figure class="fig" id="f-hero" style="margin:0"><div class="canvas" role="img" aria-label="A toy image tokenizer. A picture passes through a teal encoder into a grid of patch vectors. Each vector snaps to its nearest entry in an orange codebook and takes that entry's colour, so the grid becomes discrete tokens. A purple decoder rebuilds the picture from the tokens alone."><span class="hint">Drag to rotate</span></div>
  <figcaption><div class="controls"><button class="btn" type="button" id="hero-replay">Replay</button>
   <span class="legend"><i style="background:var(--enc)"></i>encoder and its patch vectors</span><span class="legend"><i style="background:var(--code)"></i>codebook</span><span class="legend"><i style="background:var(--dec)"></i>decoder</span></div>
   <p>The encoder turns the picture into a grid of patch vectors. Each vector snaps to its nearest codebook entry and takes that entry's colour, so the picture becomes a grid of discrete tokens. The decoder rebuilds the picture from the tokens alone. This picture is an illustration: its toy tokenizer gets the codebook from k-means on patch colours, and its decoder only blends the code colours.</p></figcaption></figure>
</div></header>''')


# ---- equations: Unicode text -> TeX; the original formula is kept in a comment ----
def eq_original(s, start):
    """The original formula text of the .eq block that starts with `start`, up to its <small>."""
    a = s.index(start)
    b = s.index('\n  <small>', a)
    return s[a:b], s[a + len('<div class="eq">'):b]


old, orig = eq_original(s, '<div class="eq">L = E<sub>q(z|x)</sub>')
s = once(s, old, '<div class="eq" id="eq-vae" data-viz="f-vae"><div class="tex tex-display">'
    r'L = \mathbb{E}_{q(z \mid x)}\big[ \htmlData{link=recon, label=the reconstruction term}{-\log p(x \mid z)} \big]'
    r' + \htmlData{link=beta, label=the KL weight}{\beta} \cdot \htmlData{link=kl, label=the KL to the prior}{\mathrm{KL}}'
    r'\big( \htmlData{link=q, label=the encoder Gaussians}{q(z \mid x)} \,\|\, \htmlData{link=prior, label=the prior}{\mathcal{N}(0, I)} \big),'
    r' \qquad z = \mu + \htmlData{link=sigma, label=the posterior width}{\sigma} \odot \varepsilon'
    '</div><!-- original: ' + orig + ' -->')

old, orig = eq_original(s, '<div class="eq">z<sub>q</sub> = e<sub>k</sub>')
ze, ek = r'\htmlClass{t-enc}{z_e}', r'\htmlClass{t-code}{e_k}'
s = once(s, old, '<div class="eq" id="eq-vq" data-viz="f-vq"><div class="tex tex-display">'
    r'\htmlClass{t-code}{z_q} = \htmlData{link=code, label=the chosen code}{' + ek + r'}, \quad '
    r'\htmlData{link=assign, label=the nearest-code assignment}{k = \operatorname{arg\,min}_{j}}\, \| '
    r'\htmlData{link=ze, label=the encoder outputs}{' + ze + r'} - '
    r'\htmlData{link=code, label=the codebook entries}{\htmlClass{t-code}{e_j}} \|'
    '</div><div class="tex tex-display">'
    r'L = \| x - D(\htmlClass{t-code}{z_q}) \|^2 + '
    r'\htmlData{link=cbloss, label=the codebook loss}{\| \mathrm{sg}[' + ze + r'] - ' + ek + r' \|^2} + '
    r'\beta \| ' + ze + r' - \mathrm{sg}[' + ek + r'] \|^2'
    '</div><!-- original: ' + orig + ' -->')

s = once(s, 'at the cost of blurrier reconstructions.</small></div>',
         'at the cost of blurrier reconstructions.</small><span class="eq-hint">Hover or tap a term to see it in the figure below.</span></div>')

# ---- figure DOM the equation terms point at ----
s = once(s, '<label class="ctl">KL weight \u03b2 <input', '<label class="ctl" data-link="beta">KL weight \u03b2 <input')
s = once(s, '<div class="status" id="va-read" aria-live="polite">', '<div class="status" id="va-read" aria-live="polite" data-link="kl sigma">')
s = once(s, '<button class="btn" type="button" id="va-s">', '<button class="btn" type="button" id="va-s" data-link="prior">')
s = once(s, '<div class="status" id="vq-read" aria-live="polite">', '<div class="status" id="vq-read" aria-live="polite" data-link="assign">')
s = once(s, '<button class="btn" type="button" id="vq-run">', '<button class="btn" type="button" id="vq-run" data-link="cbloss">')

# ---- 3D: equation glows ----
# VAE: discs brighten (q) or widen (sigma), thumbnails grow (recon), a unit ring marks the prior's one-sigma contour.
s = once(s, 'const prior=new THREE.Group();B.root.add(prior);',
            'const ring=new THREE.Mesh(new THREE.RingGeometry(.96,1.04,72),new THREE.MeshBasicMaterial({color:C.ink,transparent:true,opacity:0,side:THREE.DoubleSide,depthWrite:false}));ring.rotation.x=-Math.PI/2;ring.position.y=.04;ring.visible=false;B.root.add(ring);\n'
            ' const prior=new THREE.Group();B.root.add(prior);')
s = once(s, 'compute();};compute();return B;}',
            'compute();};compute();\n'
            ' return Object.assign(B,{update(){const gq=s.g(\'q\'),gs=s.g(\'sigma\'),gr=s.g(\'recon\');\n'
            '  discs.forEach((m,i)=>{m.material.opacity=.28+.4*gq;m.scale.setScalar(2*P[i].s*(1+.35*gs));});\n'
            '  thumbs.forEach(th=>{const v=.34*(1+.5*gr);th.sp.scale.set(v,v,1);});\n'
            '  ring.material.opacity=.8*s.g(\'prior\');ring.visible=ring.material.opacity>.002;}});}')
# VQ: encoder dots glow (ze), assignment lines brighten (assign), code spheres grow (code, cbloss).
# Growing keeps the used/dead colours; an emissive glow would tint dead codes orange.
s = once(s, 'B.root.add(new THREE.LineSegments(lg,new THREE.LineBasicMaterial({color:C.muted,transparent:true,opacity:.35})));',
            'const segs=new THREE.LineSegments(lg,new THREE.LineBasicMaterial({color:C.muted,transparent:true,opacity:.35}));B.root.add(segs);pts.material.emissive.copy(C.enc);')
s = once(s, 'let K=24,codes=[],age=[],running=false,codeMesh=null;', 'let K=24,codes=[],age=[],running=false,codeMesh=null,cntK=[],cg=0,zg=0;')
s = once(s, 'for(let k=0;k<K;k++){d.position.copy(codes[k]);d.scale.setScalar(cnt[k]?1:.7);',
            'cntK=cnt;for(let k=0;k<K;k++){d.position.copy(codes[k]);d.scale.setScalar((cnt[k]?1:.7)*(1+.6*cg));')
s = once(s, 'let acc=0;return Object.assign(B,{update(t,dt){if(running){acc+=dt;if(acc>(reduced?.3:.06)){acc=0;step();draw();}}}});}',
            'let acc=0;return Object.assign(B,{update(t,dt){if(running){acc+=dt;if(acc>(reduced?.3:.06)){acc=0;step();draw();}}\n'
            '  const gz=s.g(\'ze\');pts.material.emissiveIntensity=.6*gz;if(gz||zg){zg=gz;X.forEach((p,i)=>{d.position.copy(p);d.scale.setScalar(1+gz);d.updateMatrix();pts.setMatrixAt(i,d.matrix);});pts.instanceMatrix.needsUpdate=true;}\n'
            '  segs.material.opacity=.35+.55*s.g(\'assign\');\n'
            '  const gc=Math.max(s.g(\'code\'),s.g(\'cbloss\'));if(gc||cg){cg=gc;for(let k=0;k<K;k++){d.position.copy(codes[k]);d.scale.setScalar((cntK[k]?1:.7)*(1+.6*cg));d.updateMatrix();codeMesh.setMatrixAt(k,d.matrix);}codeMesh.instanceMatrix.needsUpdate=true;}}});}')

# ---- hero builder ----
HERO = r'''
/* ---------- hero: encode, snap to the codebook, decode (toy tokenizer, illustrative) ---------- */
function buildHero(s){const B=base([0,1.15,10.5],[0,.25,0],60);s.tRotY=-.1;s.tRotX=.06;s.vmin=20;
 const n=8,K=8,muted=hex('muted');
 // Same toy tokenizer as the tokenizer figure: average colour per patch, k-means codebook, codes ordered dark to light.
 const px=PIC.getContext('2d').getImageData(0,0,256,256).data,P=256/n,cols=[];
 for(let j=0;j<n;j++)for(let i=0;i<n;i++){let r_=0,g_=0,b_=0,c=0;for(let y=j*P;y<(j+1)*P;y+=2)for(let x=i*P;x<(i+1)*P;x+=2){const o=(y*256+x)*4;r_+=px[o];g_+=px[o+1];b_+=px[o+2];c++;}cols.push([r_/c,g_/c,b_/c]);}
 const d2=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2,r=rng(5);let cen=[cols[Math.floor(r()*cols.length)].slice()];
 while(cen.length<K){const dd=cols.map(c=>Math.min(...cen.map(e=>d2(c,e))));let u=r()*dd.reduce((a,b)=>a+b,0),k=0;while(k<dd.length-1&&u>dd[k]){u-=dd[k];k++;}cen.push(cols[k].slice());}
 const as=new Int32Array(cols.length);
 for(let it=0;it<12;it++){cols.forEach((c,i)=>{let b=1e18;cen.forEach((e,k)=>{const v=d2(c,e);if(v<b){b=v;as[i]=k;}});});
  const S=cen.map(()=>[0,0,0,0]);cols.forEach((c,i)=>{const q=S[as[i]];q[0]+=c[0];q[1]+=c[1];q[2]+=c[2];q[3]++;});S.forEach((q,k)=>{if(q[3])cen[k]=[q[0]/q[3],q[1]/q[3],q[2]/q[3]];});}
 const lum=e=>(.3*e[0]+.59*e[1]+.11*e[2])/255,ord=cen.map((e,k)=>k).sort((a,b)=>lum(cen[a])-lum(cen[b])),rank=[];ord.forEach((k,i)=>{rank[k]=i;});
 const codeCol=ord.map(k=>new THREE.Color(cen[k][0]/255,cen[k][1]/255,cen[k][2]/255)),tok=[...as].map(k=>rank[k]);
 const img=picPlane(1.6);img.material.transparent=true;B.root.add(img);
 // The decoder output is the code colours alone, blended by linear filtering.
 const rc=document.createElement('canvas');rc.width=rc.height=n;const rg=rc.getContext('2d'),rid=rg.createImageData(n,n);
 tok.forEach((t,i)=>{const c=codeCol[t];rid.data.set([c.r*255,c.g*255,c.b*255,255],i*4);});rg.putImageData(rid,0,0);
 const rt=new THREE.CanvasTexture(rc);rt.minFilter=THREE.LinearFilter;
 const reb=new THREE.Mesh(new THREE.PlaneGeometry(1.6,1.6),new THREE.MeshBasicMaterial({map:rt,side:THREE.DoubleSide,transparent:true,opacity:0}));B.root.add(reb);
 const slab=(col,hs)=>{const gr=new THREE.Group();hs.forEach((h,k)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(.09,h,.8),std(col,{transparent:true,opacity:.9}));m.position.x=(k-(hs.length-1)/2)*.19;gr.add(m);});B.root.add(gr);return gr;};
 const enc=slab(C.enc,[1.5,1.25,1,.8]),dec=slab(C.dec,[.8,1,1.25,1.5]);
 const tg=new THREE.BoxGeometry(1,1,1);
 const tiles=tok.map((t,i)=>{const c0=C.enc.clone().multiplyScalar(.65+.45*lum(cols[i]));const m=new THREE.Mesh(tg,std(c0.clone()));m.userData={c0,c1:codeCol[t]};B.root.add(m);return m;});
 const swg=new THREE.BoxGeometry(.17,.17,.17),sw=codeCol.map(c=>{const m=new THREE.Mesh(swg,std(c));B.root.add(m);return m;});
 const tray=new THREE.Mesh(new THREE.BoxGeometry(1,.04,.26),std(C.code));B.root.add(tray);
 const lp=new Float32Array(n*n*6),lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.BufferAttribute(lp,3));
 // Only the row being looked up shows its lines, drawn over the tiles so the lookup reads as a scan.
 const links=new THREE.LineSegments(lg,new THREE.LineBasicMaterial({color:C.ink,transparent:true,opacity:0,depthTest:false}));links.renderOrder=5;B.root.add(links);
 const L=(t,c,z)=>{const l=label(t,z||.26,c);B.root.add(l);return l;};
 const lImg=L('image',muted),lEnc=L('encoder',hex('enc')),lPv=L('patch vectors',hex('enc')),lTok=L('discrete tokens',hex('code')),
  lCb=L('codebook',hex('code')),lDec=L('decoder',hex('dec')),lReb=L('rebuilt image',muted);
 const arr=[0,1,2,3].map(()=>L('\u2192',muted,.34));
 let narrow=null;
 // Wide canvases run the pipeline in one row; narrow ones fold it into a "<": encode along the top, decode back along the bottom.
 function layout(nw){narrow=nw;const G=nw?1.75:1.9,cell=G/n,gx=nw?1.5:0,gy=nw?-.05:0,sp=nw?.21:.235,iw=nw?1.3:1.6;
  const pI=nw?[-1.95,.95]:[-4.6,0],pE=nw?[-.35,.95]:[-2.6,0],pD=nw?[-.35,-.95]:[2.6,0],pR=nw?[-1.95,-.95]:[4.6,0];
  img.scale.setScalar(iw/1.6);img.position.set(pI[0],pI[1],0);reb.scale.setScalar(iw/1.6);reb.position.set(pR[0],pR[1],0);
  enc.position.set(pE[0],pE[1],0);dec.position.set(pD[0],pD[1],0);enc.scale.setScalar(nw?.75:1);dec.scale.setScalar(nw?.75:1);
  const cy=gy+G/2+(nw?.3:.38);
  sw.forEach((m,k)=>m.position.set(gx+(k-(K-1)/2)*sp,cy,0));tray.scale.x=K*sp+.12;tray.position.set(gx,cy-.13,0);
  tiles.forEach((m,i)=>{const x=i%n,y=Math.floor(i/n);m.userData.p=V(gx-G/2+(x+.5)*cell,gy+G/2-(y+.5)*cell,0);m.userData.w=cell*.86;m.position.copy(m.userData.p);
   const q=sw[tok[i]].position;lp.set([m.userData.p.x,m.userData.p.y,.06,q.x,q.y,q.z],i*6);});lg.attributes.position.needsUpdate=true;
  const by=nw?0:-1.2;
  lImg.position.set(pI[0],nw?pI[1]-iw/2-.17:by,0);lReb.position.set(pR[0],nw?pR[1]-iw/2-.17:by,0);
  lEnc.position.set(pE[0],nw?pE[1]+.77:by,0);lDec.position.set(pD[0],nw?pD[1]-.77:by,0);
  lPv.position.set(gx,gy-G/2-.22,0);lTok.position.copy(lPv.position);lCb.position.set(gx,cy+.3,0);
  const ap=nw?[[-1.15,.95,0],[.25,.55,-.46],[.25,-.6,Math.PI+.46],[-1.15,-.95,Math.PI]]:[[-3.37,0,0],[-1.61,0,0],[1.61,0,0],[3.37,0,0]];
  arr.forEach((a,k)=>{a.position.set(ap[k][0],ap[k][1],0);a.material.rotation=ap[k][2];});
  B.hfov=s.hfov=nw?30:60;B.camera.position.set(nw?-.1:0,nw?.85:1.15,10.5);B.camera.lookAt(nw?-.1:0,nw?-.05:.25,0);s.w=0;}
 const T_END=6.2;let t0=performance.now()/1000;document.getElementById('hero-replay').onclick=()=>{t0=performance.now()/1000;};
 const fade=(o,u)=>{o.material.opacity=u;o.visible=u>.002;};
 const grow=(o,u)=>{o.scale.y=Math.max(.001,u);o.visible=u>0;};
 function set(e){const st=(a,b)=>smooth(a,b,e);
  fade(img,st(0,.5));fade(lImg,st(.2,.6));fade(arr[0],st(.5,.8));
  enc.children.forEach((m,k)=>grow(m,st(.6+k*.1,1+k*.1)));fade(lEnc,st(.8,1.2));fade(arr[1],st(1.1,1.4));
  const snapped=Math.floor(clamp((e-2.4)/1.4,0,1)*n*n),from=Math.max(0,snapped-n);
  tiles.forEach((m,i)=>{const x=i%n,y=Math.floor(i/n),u=st(1.2+x*.07+y*.02,1.6+x*.07+y*.02),w=m.userData.w*u;m.scale.set(Math.max(.001,w),Math.max(.001,w),.12);m.visible=u>0;
   const ts=2.4+i/(n*n)*1.4,v=st(ts,ts+.2);m.material.color.copy(m.userData.c0).lerp(m.userData.c1,v);m.position.z=.15*Math.sin(Math.PI*v);});
  fade(lPv,st(1.4,1.8)*(1-st(3.55,3.8)));
  sw.forEach((m,k)=>{const u=st(1.9+k*.05,2.2+k*.05);m.scale.setScalar(Math.max(.001,u));m.visible=u>0;});
  const tu=st(1.9,2.3);tray.visible=tu>0;tray.scale.y=Math.max(.001,tu);fade(lCb,tu);
  lg.setDrawRange(from*2,(snapped-from)*2);links.material.opacity=.7*(1-st(3.9,4.3));links.visible=snapped>0&&links.material.opacity>.002;
  fade(lTok,st(3.85,4.15));fade(arr[2],st(4.1,4.4));
  dec.children.forEach((m,k)=>grow(m,st(4.2+k*.1,4.6+k*.1)));fade(lDec,st(4.4,4.8));fade(arr[3],st(4.7,5));
  fade(reb,st(4.9,5.7));fade(lReb,st(5.2,5.6));}
 layout(false);set(reduced?T_END:0);
 return Object.assign(B,{update(t){const nw=s.wrap.clientWidth/Math.max(1,s.wrap.clientHeight)<1.9;if(nw!==narrow)layout(nw);set(reduced?T_END:Math.min(T_END,t-t0));}});}
'''
s = once(s, "/* ---------- VAE latent map ---------- */", HERO + "\n/* ---------- VAE latent map ---------- */")
s = once(s, "stage('f-vae',buildVAE);", "stage('f-hero',buildHero);stage('f-vae',buildVAE);")

open(D + 'page.src.html', 'w').write(s)
print('ok', len(s))
