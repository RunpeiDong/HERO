# Third-party software and assets

The repository's MIT license applies to HERO-authored code. Third-party software,
robot models, textures, and datasets retain their own terms.

| Component | Source and terms | Included material |
| --- | --- | --- |
| HoloSoma | [amazon-far/holosoma](https://github.com/amazon-far/holosoma), Apache-2.0 | Patched Python source in `third_party/holosoma`; its `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES`, `VENDOR.md`, and `PATCHES.md` travel with this source. |
| Unitree G1 / Dex3 | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros), BSD-3-Clause | Adapted robot descriptions and their mesh dependencies under `assets/robots/g1_modified`; see `LICENSE.unitree`, `SOURCES.md`, and `SOURCE_MANIFEST.json`. |
| MuJoCo | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco), Apache-2.0 | Installed from Python/npm; the browser distribution preserves the engine license in its third-party notices. |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime), MIT | Installed from Python/npm; browser packaging retains the dependency license. |
| Three.js | [mrdoob/three.js](https://github.com/mrdoob/three.js), MIT | Installed through npm; browser packaging retains the dependency license. |
| GR00T-WholeBodyControl | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), Apache-2.0 | G1 foot collision geometry and deployment physics parameters transcribed from its robot description (`sim2sim/physics_profiles.py`, `third_party/holosoma/holosoma/simulator/isaacsim/foot_collision.py`); no files from that repository are redistributed. |
| YCB 003 cracker box | Calli et al., *The YCB Object and Model Set*, ICAR 2015 | The demo asset-fetch step downloads the textured scan. The scan and texture are not committed or included in Python packages. Attribution is in `sim2sim/interactive_client/licenses/ycb-003_cracker_box.txt`. |

Isaac Sim and Isaac Lab are installed separately. Their packages and additional
NVIDIA materials are not redistributed here. See the NVIDIA terms referenced by
HoloSoma's `THIRD_PARTY_LICENSES` and the installed simulator release.

AMASS data and SMPL/SMPL-X models are not included. Obtain them from their
respective providers under the applicable terms. A dataset provenance label
written by a conversion script does not grant redistribution rights.

The `checkpoints/example` directory contains the example HERO ONNX policy and its
observation metadata.
