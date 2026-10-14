<script>
(() => {
  const el = document.getElementById('hero-canvas'), sc = SceneHero(el);
  document.getElementById('btn-hero-replay').addEventListener('click', sc.play);
  // Play once when the hero first scrolls into view (it is at the top, so usually on load).
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); sc.play(); } });
  io.observe(el);
})();
</script>
