"""Build local static ONNX assets and optional native parity fixtures.

Run from the repository root: python -m sim2sim.interactive_client.export_policy_assets --hero PATH
Python is needed only during this export/verification step, never by the app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEMO_SHA256 = '521402c04092df1f441a5a5e4f09a1ed17480ffa27e7220f5e1ee664ccbb2c33'


def clean(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,Path): return str(x)
    return x


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),separators=(",",":"),allow_nan=False)+"\n")


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def wxyz(q): return np.asarray(q)[..., [3,0,1,2]]


def metadata_for(policy,kind):
    common=dict(kind=kind,dofNames=policy.dof_names,kp=policy.kp,kd=policy.kd,
        effortLimit=policy.effort_limit,defaultDofPos=policy.default_dof_pos,
        actionScale=policy.action_scale,actionClip=policy.action_clip,
        controlDt=.02,quaternionConvention="WXYZ",policyObservationOdometry="truth")
    if kind=='hero_plus':
        from sim2sim.policy_hero_export import HERO_TERM_SCALES
        common.update(terms=policy.layout.terms,termDimensions=policy.layout.term_dims,
            termScales=HERO_TERM_SCALES,historyLength=policy.layout.history_length,
            historyLayout=policy.layout.history_layout,observationClip=policy.obs_clip,
            inputNames=policy.input_names,outputName=policy.output_name,
            residualIndices=policy.residual_dof_idx,refAttr=policy.ref_attr,
            lookaheadFrames=policy.cmd.ref_lookahead_frames,futureSteps=policy.h20_future_steps,
            heightFromClip=policy.cmd.h_offset_from_clip,heightMin=policy.cmd.h_cmd_min,
            heightDefault=policy.cmd.h_cmd_default,objectRule=policy.cmd.object_rule(),
            standFlagOverride=0.,sourceSidecar=policy.sidecar_path.name if policy.sidecar_path else None,
            observationDim=policy.layout.total_dim,
            variant='hero_anchor' if policy.layout.has_hero_plus_terms else
                ('hero_no_anchor' if policy.layout.has_delta_ee_terms else 'hero_no_anchor_no_delta_ee'))
    return common


def make_fixture(hero):
    import mujoco
    from scipy.spatial.transform import Rotation
    from sim2sim.interactive_scene import DemoScene
    from sim2sim.reference import ClipReference
    from sim2sim.state import read_state
    from hero_isaacsim.constants import HOLOSOMA_BODY_NAMES_32

    scene=DemoScene(policy=hero); plant=scene.plant;m,d=scene.model,scene.data
    template=d.qpos.copy();root=template[:3].copy();body_pos=[];body_quat=[];joint=[]
    for index in range(65):
        phase=index*.047
        d.qpos[:]=template
        d.qpos[:3]=root+[.025*np.sin(phase),.015*np.cos(phase),.008*np.sin(phase*.7)]
        d.qpos[3:7]=Rotation.from_euler('xyz',[.02*np.sin(phase),.035*np.cos(phase),.3+.08*np.sin(phase*.8)]).as_quat()[[3,0,1,2]]
        d.qpos[plant.dof_qadr]=hero.default_dof_pos+.04*np.sin(np.arange(29)*.37+phase)
        mujoco.mj_forward(m,d);bp,bq=plant.canonical_body_poses()
        body_pos.append(bp.copy());body_quat.append(wxyz(bq));joint.append(np.r_[d.qpos[:7],plant.dof_pos])
    ref=ClipReference.from_arrays(fps=50,joint_pos=np.array(joint),body_pos_w=np.array(body_pos),body_quat_w=np.array(body_quat),name='browser_policy_parity')
    frames=[]
    for i in range(ref.T):
        pp,pq=ref.palm_pose_w(i)
        frames.append(dict(jointPos=ref.joint_pos[i],jointVel=ref.joint_vel[i],bodyPosW=ref.body_pos_w[i],bodyQuatW=wxyz(ref.body_quat_w[i]),
            bodyLinVelW=ref.body_lin_vel_w[i],bodyAngVelW=ref.body_ang_vel_w[i],rootPosW=ref.root_pos_w[i],rootQuatW=wxyz(ref.root_quat_w[i]),
            palmPosW=pp,palmQuatW=wxyz(pq)))
    rng=np.random.default_rng(27000);cases=[]
    indices=[0,1,2,3,5,7,11,20,30,48,63,64,0,3,17,40,64]
    for k,index in enumerate(indices):
        reset=k in (0,12)
        if reset:
            hero.reset()
        d.qpos[:]=template
        d.qpos[:3]=ref.root_pos_w[index]+rng.normal(0,.015,3)
        dq=Rotation.from_rotvec(rng.normal(0,.04,3))
        d.qpos[3:7]=(dq*Rotation.from_quat(ref.root_quat_w[index])).as_quat()[[3,0,1,2]]
        d.qpos[plant.dof_qadr]=ref.joint_pos[index]+rng.normal(0,.018,29)
        d.qvel[:]=0;d.qvel[:6]=rng.normal(0,.15,6);d.qvel[plant.dof_vadr]=rng.normal(0,.2,29)
        mujoco.mj_forward(m,d);state=read_state(plant)
        js=dict(rootPosW=state.root_pos,rootQuatW=wxyz(state.root_quat),rootAngVelB=state.root_ang_vel_b,
            rootLinVelW=state.root_lin_vel_w,dofPos=state.dof_pos,dofVel=state.dof_vel,palmPosW=state.palm_pos_w,palmQuatW=wxyz(state.palm_quat_w))
        ht=hero.control(ref,index,state)
        case=dict(index=index,reset=reset,state=js,
            hero=dict(observation=hero.last_obs,rawAction=hero.last_action,target=ht))
        cases.append(case)
    return dict(schema='browser_policy_parity_v1',seed=27000,physicsSteps=0,
        reference=dict(fps=50,frameCount=ref.T,bodyNames=HOLOSOMA_BODY_NAMES_32,hasObject=False,frames=frames),cases=cases)


def main():
    from sim2sim.policy_hero_export import HeroExportPolicy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'build/demo/policies')
    parser.add_argument('--hero', type=Path, required=True,
        help='HERO ONNX or export directory; matching sidecar or embedded metadata required')
    parser.add_argument('--parity-fixture', action='store_true',
        help='Generate native observation/action cases; requires robot and scene assets')
    parser.add_argument('--allow-custom-policy', action='store_true',
        help='Explicitly allow a separately trained checkpoint instead of the provided example ONNX model')
    args = parser.parse_args()
    hero = HeroExportPolicy(args.hero, stand_flag=0.)
    model_hash = digest(hero.onnx_path)
    if model_hash != DEMO_SHA256 and not args.allow_custom_policy:
        parser.error('The default demo requires the example ONNX model. Use --allow-custom-policy for your own model.')
    args.output.mkdir(parents=True, exist_ok=True)
    model_name = 'hero.onnx'
    shutil.copy2(hero.onnx_path, args.output/model_name)
    metadata = metadata_for(hero, 'hero_plus')
    write(args.output/'hero_plus.json', metadata)
    manifest = dict(schema='browser_policy_assets_v1', onnxRuntimeWeb='1.23.2',
        allowCustomPolicy=args.allow_custom_policy, policies={
        'hero_plus': dict(label='HERO', checkpoint=hero.onnx_path.stem,
            variant=metadata['variant'], metadata='hero_plus.json', files={'model': model_name},
            provenance=dict(files={'model': dict(sha256=model_hash, bytes=hero.onnx_path.stat().st_size)},
                source='HERO training export', nativePolicyModule='sim2sim.policy_hero_export'))})
    write(args.output/'manifest.json', manifest)
    if args.parity_fixture:
        write(args.output/'parity_fixture.json', make_fixture(hero))
    print(json.dumps({'output': str(args.output), 'policies': ['hero_plus'],
        'variant': metadata['variant'], 'observationDim': metadata['observationDim'],
        'fixture': 'parity_fixture.json' if args.parity_fixture else None}, indent=2))


if __name__ == '__main__':
    main()
