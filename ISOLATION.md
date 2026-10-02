# Isolation contract

Deepstar v14 3D has a hard no-impact boundary around the existing products and
training work.

## Protected, read-only origins

- Existing application source checkouts and model-training workspaces.
- Installed application bundles.
- Existing application support and training-intake storage.

The new code contains no writer, migration, symlink, model loader, or cleanup
routine targeting those locations. It does not import Deepstar v14 model artifacts.
General engineering lessons are reimplemented behind new interfaces.

## Unique runtime identity

- Repository: `deepstar-v14-3d`
- Python package: `deepstar_v14_3d`
- Local port: `47143`
- Workspace: `~/Library/Application Support/Deepstar v14 3D`
- Cache: `~/Library/Caches/Deepstar v14 3D`
- Apple adapter environment: repo-local `.venv-sharp`
- Apple source checkout: repo-local, ignored `vendor/apple-ml-sharp`

No global Python environment, shared Torch cache, installed Deepstar bundle, or
existing app support directory is mutated.

## Resource courtesy

Before launching a 3D model, the runner checks for active Deepstar training
commands. It stops with a clear `training-active` result unless the operator
explicitly uses `--allow-concurrent-training`. The subprocess is launched with
lower scheduling priority. Tests never load neural weights or MPS.

