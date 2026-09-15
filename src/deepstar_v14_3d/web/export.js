// Export actions stay in a small companion module so the viewer remains usable
// when the server is upgraded independently of the rendering code.
(() => {
  const $ = (selector) => document.querySelector(selector);
  const ply = $('#download');
  const packageLink = $('#package');
  const snapshot = $('#snapshot');
  const canvas = $('#scene');
  if (!ply || !packageLink || !snapshot || !canvas) return;

  const syncActions = () => {
    if (!ply.href || !ply.href.includes('/artifacts/')) return;
    const match = ply.href.match(/\/artifacts\/([A-Za-z0-9_-]+)\/outputs\/scene\.ply(?:$|[?#])/);
    if (!match) return;
    packageLink.href = `/api/jobs/${match[1]}/export`;
    packageLink.classList.remove('hidden');
    snapshot.classList.remove('hidden');
  };
  new MutationObserver(syncActions).observe(ply, { attributes: true });
  syncActions();

  snapshot.addEventListener('click', () => {
    if (!canvas.width || !canvas.height) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'deepstar-scene-snapshot.png';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }, 'image/png');
  });
})();
