import sys
sys.path.insert(0, '/Users/rahulmishra/paper-explainers/shared')
import common
from common import once

D = '/Users/rahulmishra/paper-explainers/dino/'
s = open(D + 'original.html').read()

s = common.apply(s, hl='var(--g1)', extra_css='''
.t-teacher{color:var(--teacher)}.t-student{color:var(--student)}.t-plum{color:var(--plum)}
.nav-panel a .dot{min-width:0;flex:none;align-self:center;margin-right:0}
#g-heat.pe-lit{box-shadow:0 0 0 3px var(--hl)}
/* Tabular figures in Schibsted Grotesk also widen '.' ',' ':' to digit width; the sentence readouts drop them. The slider values keep them so the next control does not shift while dragging. */
#k-read,#g-read,#cur-read{font-variant-numeric:normal}
''')

# ---- hero: play the clustering once and stay clustered (it used to restart every 15 s) ----
s = once(s, 'if(ph>1.9){t0=t;ph=0;}', '')


# ---- equations: Unicode text -> TeX; the original formula goes into a comment ----
def eq(s, start, attrs, tex):
    assert s.count(start) == 1, start
    a = s.index(start)
    b = s.index('<small>', a)
    orig = ' '.join(s[a + len('<div class="eq">'):b].split())
    return s[:a] + '<div class="eq"' + attrs + '><div class="tex tex-display">' + tex + \
        '</div><!-- original: ' + orig + ' -->\n      ' + s[b:]


s = eq(s, '<div class="eq">P<sub>t</sub>(x) = softmax(', '', r'''\begin{aligned}
&amp;\htmlClass{t-teacher}{P_t}(x) = \mathrm{softmax}\!\left(\frac{g_{\theta_t}(x) - c}{\tau_t}\right) \qquad \htmlClass{t-student}{P_s}(x') = \mathrm{softmax}\!\left(\frac{g_{\theta_s}(x')}{\tau_s}\right) \\
&amp;\htmlClass{t-plum}{L} = \sum_{x \in \{\text{global}\}} \; \sum_{x' \neq x} -\,\htmlClass{t-teacher}{P_t}(x) \cdot \log \htmlClass{t-student}{P_s}(x')
\end{aligned}''')

s = eq(s, '<div class="eq">\u03b8<sub>t</sub>', '', r'''\htmlClass{t-teacher}{\theta_t} \leftarrow \lambda\,\htmlClass{t-teacher}{\theta_t} + (1 - \lambda)\,\htmlClass{t-student}{\theta_s} \qquad \lambda \text{ goes from } 0.996 \text{ to } 1 \text{ on a cosine schedule}''')

s = eq(s, '<div class="eq">L<sub>KoLeo</sub>', ' id="eq-koleo" data-viz="f-koleo"',
       r'''\htmlData{link=koleo, label=the KoLeo loss}{L_{\text{KoLeo}}} = -\frac{1}{n}\sum_{i} \log \htmlData{link=nn, label=the nearest-neighbour distances}{d_i}, \qquad d_i = \min_{j \neq i} \lVert \htmlData{link=pts, label=the features}{x_i} - x_j \rVert''')

s = eq(s, '<div class="eq">L<sub>Gram</sub>', ' id="eq-gram" data-viz="f-gram"',
       r'''\htmlData{link=gram, label=the Gram anchoring loss}{L_{\text{Gram}}} = \left\lVert \htmlData{link=gs, label=the student patch similarities}{\htmlClass{t-student}{X_S X_S^{\top}}} - \htmlClass{t-teacher}{X_G X_G^{\top}} \right\rVert_F^{2}''')

# The equations that point at a figure say so once, under the first one.
s = once(s, 'Evenly spread features make nearest-neighbour retrieval much better.</small></div>',
         'Evenly spread features make nearest-neighbour retrieval much better.</small><span class="eq-hint">Hover or tap a term to see it in the figure below.</span></div>')

# ---- figure elements the equation terms point at ----
s = once(s, '<label class="ctl"><input type="checkbox" id="k-on"> Apply KoLeo</label>',
         '<label class="ctl" data-link="koleo"><input type="checkbox" id="k-on"> Apply KoLeo</label>')
s = once(s, '<span class="readout" id="k-read"></span>', '<span class="readout" id="k-read" data-link="koleo nn"></span>')
s = once(s, '<label class="ctl"><input type="checkbox" id="g-on"> Gram anchoring from 1M iterations</label>',
         '<label class="ctl" data-link="gram"><input type="checkbox" id="g-on"> Gram anchoring from 1M iterations</label>')
s = once(s, '<canvas id="g-heat" width="144" height="144" aria-label="Gram matrix heatmap"></canvas>',
         '<canvas id="g-heat" width="144" height="144" aria-label="Gram matrix heatmap" data-link="gs"></canvas>')

# ---- 3D: equation glows ----
# KoLeo: the points grow and glow for x_i; for d_i the points shrink so the nearest-neighbour lines show, and the lines brighten
s = once(s, "B.root.add(new THREE.LineSegments(lg,new THREE.LineBasicMaterial({color:C.plum,transparent:true,opacity:.7})));",
         "const lm=new THREE.LineBasicMaterial({color:C.plum,transparent:true,opacity:.7});B.root.add(new THREE.LineSegments(lg,lm));")
s = once(s, "return Object.assign(B,{update(){neighbours();",
         "return Object.assign(B,{update(){neighbours();\n"
         "  const gp=s.g('pts'),gn=s.g('nn');dummy.scale.setScalar((1+.7*gp)*(1-.6*gn));inst.material.emissive.copy(C.g2);inst.material.emissiveIntensity=.8*gp;\n"
         "  lm.color.copy(C.plum).lerp(C.ink,gn);lm.opacity=.7+.3*gn;")
# Gram: the similarity surface (one row of the student's Gram matrix) glows
s = once(s, "return Object.assign(B,{update(){const pos=geo.attributes.position;",
         "return Object.assign(B,{update(){const pos=geo.attributes.position;surf.material.emissive.copy(C.plum);surf.material.emissiveIntensity=.55*s.g('gs');")

open(D + 'page.src.html', 'w').write(s)
print('ok', len(s))
