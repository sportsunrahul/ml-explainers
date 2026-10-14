import sys
sys.path.insert(0, '/Users/rahulmishra/paper-explainers/shared')
import common
from common import once

D = '/Users/rahulmishra/paper-explainers/clip/'
s = open(D + 'original.html').read()

s = common.apply(s, hl='var(--img)', extra_css='''
.t-img{color:var(--img)}.t-txt{color:var(--txt)}
/* A pressed seg button has panel-colored text, which vanishes on the link highlight. */
.seg button[aria-pressed="true"].pe-lit{color:var(--ink)}
''')

# ---- hero: play the pairing once and hold the final layout (it used to restart every 13 s) ----
s = once(s, "update(t){if(reduced)return;let ph=(t-t0)/7;if(ph>1.9){t0=t;ph=0;}set(ph<1?ph*ph*(3-2*ph):1);}",
            "update(t){if(reduced)return;const ph=Math.min(1,(t-t0)/7);set(ph*ph*(3-2*ph));}")
s = once(s, 'and related pairs end up near each other.</p></figcaption>',
            'and related pairs end up near each other. The animation is an illustration with toy embeddings, not a real training run.</p></figcaption>')

# ---- equation: two Unicode text lines -> two TeX lines; originals kept in comments ----
# The original lines are sliced from the source so this script stays ASCII.
assert s.count('<div class="eq">') == 1
a = s.index('<div class="eq">s<sub>ij</sub>')
b = s.index('<small>', a)
old = s[a:b]
l1, l2 = [x.strip() for x in old[len('<div class="eq">'):].split('<br>')]
T1 = (r'\htmlData{link=cos, label=the similarity scores}{s_{ij}} = \langle \htmlData{link=img, label=the image embeddings}{\htmlClass{t-img}{I_i}},'
      r' \htmlData{link=txt, label=the caption embeddings}{\htmlClass{t-txt}{T_j}} \rangle / \htmlData{link=tau, label=the temperature}{\tau}'
      r' \qquad (\htmlClass{t-img}{I},\ \htmlClass{t-txt}{T}\ \text{unit vectors})')
T2 = (r'\htmlData{link=loss, label=the loss}{L} = \frac{1}{2}\left[ -\frac{1}{N}\sum_{i} \log'
      r' \htmlData{link=i2t, label=the softmax over captions}{\mathrm{softmax}_{j}}\htmlData{link=diag, label=the true pairs on the diagonal}{(s_{ij})_{j=i}}'
      r' \;-\; \frac{1}{N}\sum_{j} \log'
      r' \htmlData{link=t2i, label=the softmax over images}{\mathrm{softmax}_{i}}\htmlData{link=diag, label=the true pairs on the diagonal}{(s_{ij})_{i=j}} \right]')
s = once(s, old, '<div class="eq" id="eq-loss" data-viz="f-mat">'
         f'<div class="tex tex-display">{T1}</div><!-- original: {l1} -->\n'
         f'      <div class="tex tex-display">{T2}</div><!-- original: {l2} -->\n      ')
s = once(s, 'per direction.</small></div>',
         'per direction.</small><span class="eq-hint">Hover or tap a term to see it in the figure below.</span></div>')

# ---- figure controls the equation terms point at ----
s = once(s, '<button type="button" data-m="cos" aria-pressed="true">', '<button type="button" data-m="cos" aria-pressed="true" data-link="cos">')
s = once(s, '<button type="button" data-m="i2t">', '<button type="button" data-m="i2t" data-link="i2t">')
s = once(s, '<button type="button" data-m="t2i">', '<button type="button" data-m="t2i" data-link="t2i">')
s = once(s, '<label class="ctl">Temperature', '<label class="ctl" data-link="tau">Temperature')
s = once(s, '<div class="status" id="m-read" aria-live="polite">', '<div class="status" id="m-read" aria-live="polite" data-link="loss">')

# ---- 3D: equation glows in the similarity matrix ----
s = once(s, "const g=new THREE.BoxGeometry(.38,1,.38);g.translate(0,.5,0);const bars=[];",
            "const g=new THREE.BoxGeometry(.38,1,.38);g.translate(0,.5,0);const bars=[],emo=[],capL=[];")
s = once(s, "const e=emojiSprite(EMO[i],.42);e.position.set(-(N+1)/2*sp-.1,.2,(i-(N-1)/2)*sp);B.root.add(e);}",
            "const e=emojiSprite(EMO[i],.42);e.position.set(-(N+1)/2*sp-.1,.2,(i-(N-1)/2)*sp);B.root.add(e);emo.push(e);}")
s = once(s, "const l=label(CAP[j],.22,hex('txt'));l.position.set((j-(N-1)/2)*sp,.25,-(N+1)/2*sp-.1);B.root.add(l);}",
            "const l=label(CAP[j],.22,hex('txt'));l.position.set((j-(N-1)/2)*sp,.25,-(N+1)/2*sp-.1);B.root.add(l);capL.push(l);}")
s = once(s, "la.position.set(1.4,.05,(N+1)/2*sp+.2);B.root.add(la);",
            "la.position.set(1.4,.05,(N+1)/2*sp+.2);B.root.add(la);capL.push(la);\n"
            " [...emo,...capL].forEach(l=>l.userData.s0=l.scale.clone());const grow=(ls,g)=>ls.forEach(l=>l.scale.set(l.userData.s0.x*(1+.4*g),l.userData.s0.y*(1+.4*g),1));")
# cos: every bar; i2t: the front row (one image's captions; back rows hide behind the diagonal); t2i: the first column; diag: the true pairs, in their own green
s = once(s, "return Object.assign(B,{update(){for(const row of bars)for(const m of row)m.scale.y+=(m.userData.h-m.scale.y)*.15;}});}",
            "return Object.assign(B,{update(){const gc=s.g('cos'),gr=s.g('i2t'),gk=s.g('t2i'),gd=s.g('diag');grow(emo,s.g('img'));grow(capL,s.g('txt'));\n"
            "  for(let i=0;i<N;i++)for(let j=0;j<N;j++){const m=bars[i][j];m.scale.y+=(m.userData.h-m.scale.y)*.15;\n"
            "   const e=i===j?gd:0,o=Math.max(gc*.6,i===N-1?gr:0,j===0?gk:0);m.material.emissive.copy(e>o?C.pos:C.img);m.material.emissiveIntensity=Math.max(e,o)*.8;}}});}")

open(D + 'page.src.html', 'w').write(s)
print('ok', len(s))
