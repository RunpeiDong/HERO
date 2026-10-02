import assert from 'node:assert/strict';
import test from 'node:test';
import {HeroPolicy} from '../policy_runtime.mjs';

const zeros=n=>Array(n).fill(0), identity=[1,0,0,0];
const metadata={kind:'hero_plus',kp:zeros(29),kd:zeros(29),effortLimit:zeros(29),
  defaultDofPos:zeros(29),actionScale:zeros(29),historyLength:5,
  historyLayout:'frame_major_hero_v1',termDimensions:[8,12,2,3],
  terms:['h20_ref_root_pose_b','h21_ref_root_rot_b','h22_ref_root_height_b','h23_base_lin_vel_odom'],termScales:{},
  observationClip:100,lookaheadFrames:0,futureSteps:[0,1],heightFromClip:true,heightMin:.15,
  objectRule:{},refAttr:'joint_pos'};
const bodyNames=['left_ankle_roll_link','right_ankle_roll_link','left_knee_link','right_knee_link'];
function fixture(){
  const frames=[0,1].map(i=>({rootPosW:[3+i,5+i,1.2+i*.1],rootQuatW:identity,
    palmPosW:[[3,5,1],[3,4,1]],palmQuatW:[identity,identity],jointPos:zeros(29),bodyPosW:bodyNames.map(()=>[0,0,0])}));
  const reference={frameCount:2,bodyNames,hasObject:false,sample:i=>frames[i]};
  const state={rootPosW:[-4,7,.9],rootQuatW:identity,rootAngVelB:[0,0,0],rootLinVelW:[9,8,7],
    dofPos:zeros(29),dofVel:zeros(29),palmPosW:[[-3.8,7.2,.9],[-3.8,6.8,.9]],palmQuatW:[identity,identity],
    odom:{source:'leg',posW:[1,2,.7],quatW:[Math.SQRT1_2,0,0,Math.SQRT1_2],linVelW:[.2,.4,.1]}};
  return {reference,state};
}
const near=(actual,expected)=>{assert.equal(actual.length,expected.length);expected.forEach((v,i)=>assert.ok(Math.abs(actual[i]-v)<1e-12,`${i}: ${actual[i]} != ${v}`));};

test('HERO requires leg odometry by default; missing estimates never silently use truth',()=>{
  const policy=new HeroPolicy({},metadata,{}),{reference,state}=fixture();
  assert.equal(policy.odomSource,'leg');
  assert.throws(()=>policy.frameTerms(reference,0,{...state,odom:null}),/require leg odometry/);
  assert.throws(()=>policy.frameTerms(reference,0,{...state,odom:{...state.odom,source:'truth'}}),/require leg odometry/);
  assert.throws(()=>policy.frameTerms(reference,0,{...state,odom:{...state.odom,posW:[NaN,0,0]}}),/finite/);
});

test('anchor translation, orientation, height and optional velocity share the estimated frame',()=>{
  const policy=new HeroPolicy({},metadata,{}),{reference,state}=fixture();
  const {terms}=policy.frameTerms(reference,0,state);
  near(terms.h20_ref_root_pose_b,[3,-2,-1,0,4,-3,-1,0]);
  near(terms.h21_ref_root_rot_b,[0,-1,0,1,0,0,0,-1,0,1,0,0]);
  near(terms.h22_ref_root_height_b,[.5,.6]);
  near(terms.h23_base_lin_vel_odom,[.4,-.2,.1]);
  assert.equal(terms.odom_source,'leg');
});

test('changing simulator world pose and velocity cannot change leg-based anchor observations',()=>{
  const policy=new HeroPolicy({},metadata,{}),{reference,state}=fixture();
  const before=policy.frameTerms(reference,0,state).terms;
  const changed={...state,rootPosW:[200,-300,-9],rootQuatW:[0,0,0,1],rootLinVelW:[100,200,300]};
  const after=policy.frameTerms(reference,0,changed).terms;
  for(const key of ['h20_ref_root_pose_b','h21_ref_root_rot_b','h22_ref_root_height_b','h23_base_lin_vel_odom'])
    assert.deepEqual(after[key],before[key],key);
});

test('leg odometry changes only the anchor group, preserving other observations and residual reference',()=>{
  const {reference,state}=fixture(),leg=new HeroPolicy({},metadata,{}),truth=new HeroPolicy({},metadata,{},{odomSource:'truth'});
  const a=leg.frameTerms(reference,0,state),b=truth.frameTerms(reference,0,state);
  for(const key of Object.keys(a.terms))if(!/^h2[0-3]_/.test(key)&&key!=='odom_source')
    assert.deepEqual(a.terms[key],b.terms[key],key);
  assert.deepEqual(a.residualReference,b.residualReference);
  assert.notDeepEqual(a.terms.h20_ref_root_pose_b,b.terms.h20_ref_root_pose_b);
  assert.notDeepEqual(a.terms.h22_ref_root_height_b,b.terms.h22_ref_root_height_b);
  const observations=leg.observe(a.terms);
  assert.ok(Array.from(observations.slice(0,-25)).every(v=>v===0),'History starts with zero padding.');
  leg.reset();assert.deepEqual(leg.observe(a.terms),observations,'Reset restarts the existing history contract.');
});


test('a separately trained no-anchor contract needs no odometry or future root samples',()=>{
  const base={...metadata,terms:['h00_actions','h09_dof_pos'],termDimensions:[29,29],futureSteps:undefined};
  const policy=new HeroPolicy({},base,{}),{reference,state}=fixture();
  delete state.odom;
  const result=policy.frameTerms(reference,0,state);
  assert.equal(result.terms.odom_source,'unused');
  assert.equal(Object.hasOwn(result.terms,'h20_ref_root_pose_b'),false);
  assert.equal(Object.hasOwn(result.terms,'h23_base_lin_vel_odom'),false);
  assert.equal(policy.observe(result.terms).length,290);
});

test('anchor2 without h23 does not read odometry velocity',()=>{
  const base={...metadata,terms:metadata.terms.slice(0,3),termDimensions:[8,12,2]};
  const policy=new HeroPolicy({},base,{}),{reference,state}=fixture();
  delete state.odom.linVelW;
  const result=policy.frameTerms(reference,0,state);
  assert.equal(Object.hasOwn(result.terms,'h23_base_lin_vel_odom'),false);
  assert.equal(policy.observe(result.terms).length,110);
});

test('paper and optional anchor layouts produce 675 and 950 inputs without legacy object channels',()=>{
  const baseTerms=['h00_actions','h01_base_ang_vel','h02_command_ang_vel','h03_command_base_height',
    'h04_command_lin_vel','h05_command_stand','h06_command_waist_dofs','h07_dif_local_rigid_body_pos_ee',
    'h08_dif_local_rigid_body_rot_ee','h09_dof_pos','h10_dof_vel','h11_projected_gravity',
    'h12_ref_upper_dof_pos','h13_roll_and_pitch'];
  const baseDimensions=[29,3,1,1,2,1,3,6,12,29,29,3,14,2];
  for(const anchor of [false,true]) {
    const terms=anchor?[...baseTerms,...metadata.terms.slice(0,3)]:baseTerms;
    const termDimensions=anchor?[...baseDimensions,20,30,5]:baseDimensions;
    const policy=new HeroPolicy({}, {...metadata,terms,termDimensions,futureSteps:[0,5,10,15,20]},{});
    const {reference,state}=fixture();if(!anchor)delete state.odom;
    const observation=policy.observe(policy.frameTerms(reference,0,state).terms);
    assert.equal(observation.length,anchor?950:675);
    assert.ok(Array.from(observation).every(Number.isFinite));
  }
});

test('the without_delta_ee ablation layout (no h07/h08) produces a 585 input from the same frame terms',()=>{
  const terms=['h00_actions','h01_base_ang_vel','h02_command_ang_vel','h03_command_base_height',
    'h04_command_lin_vel','h05_command_stand','h06_command_waist_dofs','h09_dof_pos','h10_dof_vel',
    'h11_projected_gravity','h12_ref_upper_dof_pos','h13_roll_and_pitch'];
  const termDimensions=[29,3,1,1,2,1,3,29,29,3,14,2];
  const policy=new HeroPolicy({}, {...metadata,terms,termDimensions,futureSteps:undefined},{});
  const {reference,state}=fixture();delete state.odom;
  const result=policy.frameTerms(reference,0,state);
  const observation=policy.observe(result.terms);
  assert.equal(observation.length,585);
  assert.ok(Array.from(observation).every(Number.isFinite));
  assert.equal(Object.hasOwn(result.terms,'h20_ref_root_pose_b'),false);
});
