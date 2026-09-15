# Deepstar v14 3D

Deepstar v14 3D is a separate, local-first 2D-to-3D studio. It turns either an
explicitly selected image or the output of a user-configured image generator
into a portable `.ply` scene and a provenance receipt.

It is **not** build 15 of Deepstar, does not replace the installed build 14,
and does not read or write Deepstar's projects, models, caches, training data,
or application container.

## What works now

- Import a PNG, JPEG, HEIC, or other image macOS can decode.
- Feed a prompt through a locally configured image-generation command.
- Create an immediate Apple Vision spatial portrait: person masking, face-aware
  volume, a back shell, and a silhouette rim with no downloaded model weights.
- Use Apple's official SHARP CLI as a separately installed, opt-in research
  backend on CPU or Apple-silicon MPS.
- Use the opt-in Vision-guided SHARP research backend to retain SHARP's original
  Gaussian `.ply` and write Apple Vision person-mask/face-analysis sidecars.
- Inspect the result in a local browser viewer, download the `.ply`, and retain
  a privacy-minimized provenance receipt.
- Export a portable `scene-package.zip` containing the scene, receipt, metadata,
  and local-use instructions. The package intentionally excludes source pixels,
  prompt text, and model weights; the viewer can also save its current orbit as
  a PNG snapshot.
- Pause 3D inference when another Deepstar training process is detected.

The built-in spatial portrait is deterministic geometry informed by Apple's
on-device Vision framework, not learned monocular depth reconstruction.
Apple SHARP produces the high-quality Gaussian representation, but Apple's
released checkpoint is licensed only for non-commercial scientific research
and academic development. The app therefore never downloads, bundles, or
silently enables those weights.

## Run the isolated studio

```sh
./scripts/deepstar3d doctor
./scripts/deepstar3d serve --open
```

Or double-click `Deepstar v14 3D.command` in Finder.

The browser server binds only to `127.0.0.1:47143`. Generated work lives in:

- `~/Library/Application Support/Deepstar v14 3D`
- `~/Library/Caches/Deepstar v14 3D`

Override both for a disposable run:

```sh
DEEPSTAR3D_HOME=/tmp/deepstar3d-work \
DEEPSTAR3D_CACHE=/tmp/deepstar3d-cache \
  ./scripts/deepstar3d create --image photo.png --backend depth-card \
  --confirm-rights
```

## Image-generation to 3D

Set a command whose arguments contain `{prompt}` and `{output}`. Deepstar runs
it without a shell and requires the command to create the requested output
file. This works with a local generator, a private worker client, or any image
API wrapper without putting credentials in this repository.

```sh
export DEEPSTAR3D_IMAGE_COMMAND='my-imagegen --prompt {prompt} --output {output}'
./scripts/deepstar3d create --prompt 'a glass observatory at blue hour' \
  --backend depth-card --confirm-rights
```

The compatible CLI name remains `depth-card`; after the separate research
installation, use `apple-sharp` for a pure learned scene or
`vision-sharp-hybrid` for the same SHARP scene plus local Vision sidecars. The
hybrid never merges the two point clouds or edits SHARP's `.ply`.

## Apple SHARP research backend

Read [THIRD_PARTY.md](THIRD_PARTY.md), then install the official repository in
this repo's ignored `vendor/` directory:

```sh
./scripts/install_apple_sharp_research.sh --accept-research-license
export DEEPSTAR3D_ACCEPT_APPLE_RESEARCH_LICENSE=1
./scripts/deepstar3d create --image photo.jpg --backend apple-sharp \
  --confirm-rights
```

For the hybrid, replace `apple-sharp` with `vision-sharp-hybrid`. It has the
same installation, runtime-license, and training guard. SHARP receives the
original job image; Apple Vision then writes `subject-mask.pgm` and
`vision-analysis.json` beside the unmodified `scene.ply`. These sidecars are
identified in the job receipt and are local-only artifacts, not model input or
a claim of SHARP geometry correction.

The installer pins the inspected Apple source revision. Runtime caches and the
downloaded checkpoint stay inside Deepstar v14 3D's cache root. On macOS, the
adapter uses MPS for prediction when available; Apple's trajectory renderer is
CUDA-only, so this app uses its own lightweight local geometry preview.

## Verify

```sh
./scripts/test
```

See [ISOLATION.md](ISOLATION.md) and [docs/PIPELINE.md](docs/PIPELINE.md) for
the hard boundaries and pipeline contract.

## Export and use a scene

Every completed job creates `outputs/scene-package.zip`. In the browser, choose
**Export scene package** to download it, or choose **Download .ply** when a raw
scene is all you need. **Save snapshot** downloads the current local viewer
canvas as `deepstar-scene-snapshot.png`; it is a presentation image, not a
replacement for the 3D scene.

Extract the package and open `scene.ply` in MeshLab, Blender with a PLY
importer, or another PLY/3D Gaussian-compatible tool. Keep `manifest.json` next
to the scene to preserve the source and scene fingerprints. Packages are
assembled locally and do not upload anything. If the backend is Apple SHARP,
the included README and receipt retain its research-only licensing warning.
Apple Vision mask/face sidecars, when available, travel under `guidance/` so a
downstream tool can distinguish the subject without altering the original PLY.
