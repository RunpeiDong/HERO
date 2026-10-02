# HERO browser demo

HERO tabletop grasping with MuJoCo, ONNX inference, and IK running in the browser. The example ONNX model uses delta anchor to further improve end-effector tracking.

See the [README](../../README.md) for setup, training, and model export. After exporting policy and scene assets to `build/demo/`, run from this directory:

```sh
npm ci
python prepare_assets.py --delivery ../../build/demo
npm test
npm run build
python package_standalone.py --output ../../build/demo
```

`npm run dev` serves the app at http://127.0.0.1:8768. `dist/` can be hosted as a static website; `build/demo/Tabletop_Lab.html` can be opened locally after packaging. No Python server is needed by the browser demo.

Run the Python exporters from the repository root:

```sh
python -m sim2sim.interactive_client.export_policy_assets --hero checkpoints/example/model.onnx --parity-fixture
python -m sim2sim.interactive_client.export_scene --hero checkpoints/example/model.onnx
```

For your own model, add `--allow-custom-policy` to the policy export command and use the same checkpoint for scene export. Export metadata defines the model's inputs.

After export and `npm ci`, verify native versus WebAssembly policy observations/actions with:

```sh
node policy_parity.mjs ../../build/demo/policies
```

The demo uses the Dex3 hand. Dependencies and asset credits are in `THIRD_PARTY_NOTICES.txt`; licenses are in `licenses/`.

## Demo behaviour

- **Rendering.** Both views target up to 90 Hz (60 Hz on a 60 Hz display) using buffered position/quaternion interpolation, with about 60 ms of display delay. Physics and policy control stay at 50 Hz. Reset clears the display buffer; if IK pauses, the last known pose is held without extrapolation. Actual FPS depends on the display, GPU and browser throttling.
- **Grasp attempts.** When the hand cannot settle inside the 15 mm / 15° grasp gate, the controller makes one alignment retry: it first holds and refines the accepted pose, then backs the open palm out (up to 8 cm) and re-enters for a real second approach. If the gates are still unmet it grasps from the closest reachable pose (final attempt, palm within 6 cm of the target) and lets the lift check decide; only a palm farther than 6 cm opens and returns. The retry, the single supported-stance replan and the final attempt announce themselves in the status line, and an unsuccessful attempt names an IK or reach cause there when the controller can identify one.
- **Release and return.** Once the fingers are measured open and the released object rests quietly on the tray (or after a 7 s opening deadline), the object is no longer treated as an obstacle: the hand retreats 5–10 cm toward the robot, rises clear of the tray wall and returns to the initial posture. Clearance shortfalls after release are recorded and answered with a bounded withdrawal rather than stopping the robot; the whole post-release motion is limited to 60 s, after which the episode ends with "Placed · Return incomplete" when the placement itself succeeded. Robot body or hand contact with the table edge or tray stops the robot only when it stays above 10 N for 0.1 s.
- **Cheez-It carton.** Both hands grasp the carton with its narrow end face seated in the thumb-finger web (hand turned 35° from the end-face frame). The carton's yaw is kept inside the hand-facing band sign(y) × [40°, 90°] on every path: adds and drags draw sign(y) × (60° ± 20°), the rotation slider only offers that band for the side the carton sits on, and saved layouts are clamped into it. The carton is added at |y| = 0.37 m, nearer the table's side edge than the other objects (0.32 m). The grasp variant can be pinned for comparison with the URL parameter `?carton_grasp=auto|end_face|crotch|spine|spine90` (default `auto`, which selects `spine`; `spine45` is accepted as an alias of `spine`). The worker validates the value and keeps it across Reset.
- **Placement area.** The placeable area shown on the page ends 2 cm before the exported scene's far edge (x ≤ 0.55 m instead of 0.57 m); the scene's own placement validation is unchanged.
- **Status bar.** The status reads "Live MuJoCo simulation …" so the physics engine running in the browser is visible, and the page carries a "Controls & tips" section (including the note that browser IK runs on the CPU without cuRobo).
