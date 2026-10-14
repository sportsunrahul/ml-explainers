(function(){"use strict";
// Section bar: current-section highlight, reading progress, and a Sections menu whenever the links do not fit.
const nav=document.querySelector('nav.top');
if(nav){
 const wrap=nav.querySelector('.wrap'),links=[...wrap.querySelectorAll(':scope > a[href^="#"]')];
 const secs=links.map(a=>document.getElementById(a.getAttribute('href').slice(1)));
 const cur=document.createElement('span');cur.className='nav-cur';
 const btn=document.createElement('button');btn.type='button';btn.className='nav-menu';btn.textContent='Sections';
 btn.setAttribute('aria-expanded','false');btn.setAttribute('aria-controls','nav-panel');
 const brand=wrap.querySelector('.brand');(brand||wrap.firstChild).after(cur);wrap.appendChild(btn);
 const panel=document.createElement('div');panel.className='nav-panel';panel.id='nav-panel';
 panel.innerHTML='<ol>'+links.map((a,i)=>'<li><a href="'+a.getAttribute('href')+'"><span>'+(i+1)+'</span>'+a.innerHTML+'</a></li>').join('')+'</ol>';
 const prog=document.createElement('span');prog.className='nav-prog';nav.append(panel,prog);
 const plinks=[...panel.querySelectorAll('a')];
 const setOpen=o=>{nav.classList.toggle('nav-open',o);btn.setAttribute('aria-expanded',String(o));};
 btn.addEventListener('click',()=>setOpen(!nav.classList.contains('nav-open')));
 panel.addEventListener('click',e=>{if(e.target.closest('a'))setOpen(false);});
 addEventListener('keydown',e=>{if(e.key==='Escape')setOpen(false);});
 function fit(){nav.classList.remove('nav-collapsed');if(wrap.scrollWidth>wrap.clientWidth+1)nav.classList.add('nav-collapsed');else setOpen(false);}
 let queued=false;
 function spy(){queued=false;const line=innerHeight*.3;let k=-1;
  secs.forEach((s,i)=>{if(s&&s.getBoundingClientRect().top<=line)k=i;});
  [links,plinks].forEach(set=>set.forEach((a,i)=>{if(i===k)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');}));
  cur.innerHTML=k>=0?'<b>'+(k+1)+'</b>'+links[k].textContent:'';
  const max=document.documentElement.scrollHeight-innerHeight;nav.style.setProperty('--p',max>0?(scrollY/max).toFixed(4):0);}
 addEventListener('scroll',()=>{if(!queued){queued=true;requestAnimationFrame(spy);}},{passive:true});
 addEventListener('resize',()=>{fit();spy();});
 if(document.fonts&&document.fonts.ready)document.fonts.ready.then(fit);
 fit();spy();
}
// Equation terms: hover, focus or tap lights the term, the figure elements with the same data-link, and pulses the 3D stage.
document.querySelectorAll('.eq[data-viz]').forEach(eq=>{
 const fig=document.getElementById(eq.dataset.viz);if(!fig)return;
 eq.querySelectorAll('[data-link]').forEach(t=>{
  t.tabIndex=0;t.setAttribute('role','button');t.setAttribute('aria-label','Show '+(t.dataset.label||t.dataset.link)+' in the figure');
  let last=0;
  const fire=()=>{const now=performance.now();if(now-last<600)return;last=now;const lit=[t];
   t.dataset.link.split(/\s+/).forEach(l=>{fig.querySelectorAll('[data-link~="'+l+'"]').forEach(x=>lit.push(x));fig.dispatchEvent(new CustomEvent('pe-link',{detail:l}));});
   lit.forEach(x=>x.classList.add('pe-lit'));setTimeout(()=>lit.forEach(x=>x.classList.remove('pe-lit')),1100);};
  t.addEventListener('mouseenter',fire);t.addEventListener('focus',fire);t.addEventListener('click',fire);
  t.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();last=0;fire();}});
 });
});
})();
