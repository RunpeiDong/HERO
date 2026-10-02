# G1 / Dex3 asset sources

The G1 and Dex3 meshes originate from Unitree Robotics' `unitree_ros` repository:

https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description

`LICENSE.unitree` is the BSD-3-Clause license from that exact revision. Every
included mesh was matched byte for byte against the upstream Git blob SHA.
`SOURCE_MANIFEST.json` records local SHA-256 hashes, matching upstream paths,
and the revision used for verification.

The URDF and MJCF files are local adaptations. Their purposes are:

- `g1_29dof_dex3fixed_hero.urdf`: training robot with fixed finger joints, preserved
  palm bodies, and HoloSoma foot-contact frames.
- `g1_29dof_with_hand_rev_1_0.urdf`: articulated Dex3 hands for the interactive demo.
- `g1_29dof_with_hand_rev_1_0.xml`: forward kinematics for conversion of G1 motions.
- `scene_g1_29dof_freebase_fixed_dex3.xml` and its included
  `g1_29dof_old_freebase_fixed_dex3.xml`: model used by the IK reach generator.

The two MJCF variants have different waist-frame geometry. The converter uses
the revision-1.0 model matching the training URDF. The IK generator retains its
established model and validates the generated trajectories separately.

The training URDF's two foot-contact fixed links derive from HoloSoma, whose
Apache-2.0 license and notices are retained in `third_party/holosoma`. HERO-authored
adaptations are covered by the root MIT license in addition to upstream terms.
