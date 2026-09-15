# 2D to 3D pipeline

```text
explicit image ─┐
                ├─> isolated job/source ─> 3D backend ─> scene.ply
image generator ┘          │                    │
                           └──────────────> manifest.json
```

## Stage 1: image source

An imported image is copied into the new job. A generated image comes from a
configured command executed as an argument array, never through a shell. The
receipt stores the image hash and a prompt hash, not the prompt text. Rights
confirmation is required for both routes.

## Stage 2: representation

- `depth-card` (Apple Vision spatial portrait): uses the operating system's
  person segmentation and face detection, removes the background, builds a
  face-aware front surface, and adds a back shell plus silhouette rim. It needs
  no downloaded weights and remains deterministic geometry rather than a
  learned monocular-depth claim.
- `apple-sharp`: calls the official SHARP predictor and returns its metric 3D
  Gaussian `.ply`. The adapter uses a dedicated Torch/cache root and defaults
  to MPS on Apple silicon. The Apple checkpoint is research-only.
- `vision-sharp-hybrid`: runs the same gated SHARP predictor once on the
  original job image, then asks Apple Vision for a local person mask and face
  metadata. It records both as sidecars and in the receipt. SHARP remains the
  geometry authority: the hybrid never combines point clouds or rewrites its
  `.ply`, so there is no fabricated geometric correction or doubled surface.

The backend interface is deliberately provider-neutral so a future
Deepstar-owned, commercially cleared single-image-to-Gaussian model can replace
the research adapter without changing job, provenance, or UI contracts.

## Stage 3: preview and export

The local viewer reads ASCII or binary little-endian PLY vertices. For SHARP it
shows sampled Gaussian centers as a responsive geometry preview; it is not a
full anisotropic Gaussian renderer. The downloaded `.ply` remains compatible
with dedicated 3DGS tooling.

## v14 lessons retained

- model implementations live behind a narrow rendering seam;
- temporary identity/image state is job-scoped;
- caches are independently versioned and disposable;
- consent and provenance fail closed;
- working plumbing is reported separately from model quality and licensing.

No v14 weights, datasets, private media, caches, bundle identifiers, or app
state are reused.
