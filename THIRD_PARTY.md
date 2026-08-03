# Third-party and model licensing

## Apple ml-sharp

The optional `apple-sharp` adapter targets Apple's official `ml-sharp`
command-line interface at revision:

`1eaa046834b81852261262b41b0919f5c1efdd2e`

Deepstar v14 3D does not copy Apple source or weights into this repository.
The optional installer clones Apple's repository into ignored local storage.

Apple publishes the software under the terms in its `LICENSE`, which permit
use, modification, and redistribution subject to those terms. The released
model is governed separately by `LICENSE_MODEL`. That model license limits the
model, derivatives, and use to non-commercial scientific research and academic
development and excludes commercial exploitation, product development, and
commercial products or services.

Consequences enforced here:

1. No Apple model is bundled or downloaded during normal setup or tests.
2. Installation requires an explicit research-license acceptance flag.
3. Runtime requires a second explicit acceptance setting.
4. The receipt identifies `apple-ml-sharp-research` and never calls the result
   commercially approved.
5. Deepstar-owned 2D v14 weights are never fine-tuned from, distilled from, or
   combined into Apple SHARP weights.

Review the authoritative current license before every redistribution or use:
https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL

