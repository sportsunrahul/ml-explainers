<script>
(() => {
  // Hero: the KV cache grows a column per token; the SSM state only changes shade. Logic lives in CoreHero.
  const el = document.getElementById('hero-canvas'), btn = document.getElementById('btn-hero-replay');
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .95 : .44, fov: 34, dist: 6.2, yaw: .38, pitch: .34 });
  // pe3's field of view is vertical, so a portrait canvas needs the camera farther back to fit the width.
  let narrow = false;
  new ResizeObserver(() => { const v = CoreHero.view(el.clientWidth, el.clientHeight); narrow = v.narrow; s3.setView(v, 0); frame(u); }).observe(el);
  let u = 0, cancel = () => {};
  const frame = v => { u = v; s3.items = CoreHero.items(v, narrow); s3.redraw(); };
  function play() { cancel(); if (PE.instant()) { frame(1); return; } cancel = s3.tween(6200, frame); }
  btn.addEventListener('click', play);
  PE.onTheme(() => frame(u));
  frame(PE.instant() ? 1 : 0); play();
})();
</script>
