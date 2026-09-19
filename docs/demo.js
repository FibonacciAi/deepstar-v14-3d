(() => {
  const canvas = document.querySelector('#scene');
  const stage = document.querySelector('#stage');
  const loading = document.querySelector('#loading');
  const status = document.querySelector('#status');
  const resetButton = document.querySelector('#reset');
  const fullscreenButton = document.querySelector('#fullscreen');

  if (!window.pc) {
    loading.classList.add('error');
    loading.querySelector('b').textContent = 'Gaussian renderer unavailable';
    loading.querySelector('small').textContent = 'Reload the page in a WebGL-capable browser.';
    return;
  }

  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: {
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    }
  });
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.scene.toneMapping = pc.TONEMAP_NONE;
  app.scene.exposure = 1;
  if (app.scene.gsplat) app.scene.gsplat.useTonemap = false;

  const camera = new pc.Entity('Research camera');
  camera.addComponent('camera', {
    clearColor: new pc.Color(.02, .025, .04),
    fov: 61,
    nearClip: .01,
    farClip: 1000
  });
  app.root.addChild(camera);
  app.start();

  const view = {
    target: new pc.Vec3(),
    baseDistance: 1,
    yaw: 0,
    pitch: 0,
    zoom: 1
  };

  const resize = () => app.resizeCanvas();
  const applyCamera = () => {
    const distance = view.baseDistance * view.zoom;
    const pitchCosine = Math.cos(view.pitch);
    camera.setPosition(
      view.target.x + Math.sin(view.yaw) * pitchCosine * distance,
      view.target.y + Math.sin(view.pitch) * distance,
      view.target.z + Math.cos(view.yaw) * pitchCosine * distance
    );
    camera.lookAt(view.target, pc.Vec3.UP);
  };
  const reset = () => {
    view.yaw = 0;
    view.pitch = 0;
    view.zoom = 1;
    applyCamera();
  };

  let dragging = false;
  let pointerX = 0;
  let pointerY = 0;
  canvas.addEventListener('pointerdown', event => {
    dragging = true;
    pointerX = event.clientX;
    pointerY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    view.yaw = Math.max(-.38, Math.min(.38, view.yaw - (event.clientX - pointerX) * .004));
    view.pitch = Math.max(-.25, Math.min(.25, view.pitch - (event.clientY - pointerY) * .004));
    pointerX = event.clientX;
    pointerY = event.clientY;
    applyCamera();
  });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('pointercancel', () => { dragging = false; });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    view.zoom = Math.max(.58, Math.min(2.2, view.zoom * Math.exp(event.deltaY * .001)));
    applyCamera();
  }, { passive: false });
  canvas.addEventListener('dblclick', reset);
  resetButton.addEventListener('click', reset);
  fullscreenButton.addEventListener('click', () => stage.requestFullscreen?.());
  window.addEventListener('resize', resize);
  resize();

  const asset = new pc.Asset('Dimitri SHARP study', 'gsplat', { url: './assets/dimitri-sharp.ply' });
  asset.once('load', () => {
    const data = asset.resource?.gsplatData;
    if (!data) throw new Error('The published PLY is not a Gaussian scene.');

    const entity = new pc.Entity('Dimitri Gaussians');
    entity.setEulerAngles(180, 0, 0);
    entity.addComponent('gsplat', { asset });
    app.root.addChild(entity);

    const opacity = data.getProp('opacity');
    const focus = new pc.Vec3();
    data.calcFocalPoint(focus, index => !opacity || opacity[index] > -2.2);
    const pivotDepth = Math.max(.25, Number.isFinite(focus.z) ? focus.z : 1);
    view.baseDistance = pivotDepth;
    view.target.set(0, 0, -pivotDepth);

    const intrinsic = data.getElement('intrinsic')?.properties?.[0]?.storage;
    const imageSize = data.getElement('image_size')?.properties?.[0]?.storage;
    if (intrinsic?.[0] > 0 && imageSize?.[1] > 0) {
      camera.camera.fov = 2 * Math.atan(imageSize[1] / (2 * intrinsic[0])) * 180 / Math.PI;
    }
    camera.camera.nearClip = Math.max(.002, pivotDepth / 200);
    camera.camera.farClip = Math.max(50, pivotDepth * 20);
    reset();

    status.textContent = `${data.numSplats.toLocaleString()} Gaussian splats · ready`;
    loading.classList.add('done');
  });
  asset.once('error', error => {
    loading.classList.add('error');
    loading.querySelector('b').textContent = 'Could not load the public scene';
    loading.querySelector('small').textContent = String(error);
    status.textContent = 'Scene unavailable';
  });
  app.assets.add(asset);
  app.assets.load(asset);
})();
