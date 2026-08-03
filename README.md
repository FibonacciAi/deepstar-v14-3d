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
- Inspect the result in a local browser viewer, download the `.ply`, and retain
  a privacy-minimized provenance receipt.
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

The compatible CLI name remains `depth-card`; swap it for `apple-sharp` after
the separate research installation when genuine learned scene reconstruction
is required.

## Apple SHARP research backend

Read [THIRD_PARTY.md](THIRD_PARTY.md), then install the official repository in
this repo's ignored `vendor/` directory:

```sh
./scripts/install_apple_sharp_research.sh --accept-research-license
export DEEPSTAR3D_ACCEPT_APPLE_RESEARCH_LICENSE=1
./scripts/deepstar3d create --image photo.jpg --backend apple-sharp \
  --confirm-rights
```

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
