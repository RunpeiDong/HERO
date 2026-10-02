// Exercises the production modules without changing live physics or model data.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createSimulation} from '../simulation.js';
import {InteractiveController} from '../controller.js';
import {RollingReference,ReferenceClip} from '../reference.js';
import {IKError,bodyPose,geomBounds} from '../ik.js';
import {norm,sub,quatToMat} from '../numerics.js';

const assets=new URL('../../../build/demo/sceneassets/',import.meta.url);
const clone=structuredClone;
const maximumDelta=(a,b)=>Math.max(...Array.from(a,(v,i)=>Math.abs(v-b[i])));
const anchors=ik=>clone({feet:ik.feet,support:ik.supportPolygon,position:ik.anchorPosition,rotation:ik.anchorRotation});
const state=scene=>({qpos:Array.from(scene.data.qpos),qvel:Array.from(scene.data.qvel),ctrl:Array.from(scene.data.ctrl),xpos:Array.from(scene.data.xpos),xmat:Array.from(scene.data.xmat),time:scene.time});

async function fixture(t){
  const scene=await createSimulation({mode:'hero_plus',tableKind:'workbench',tableHeight:.74,placements:{},readAsset:p=>readFile(new URL(p,assets))});
  scene.placeObject('uiuc_i',.47,.31,-Math.PI/18);
  const reference=new RollingReference(scene,{bestEffort:true}),homeAuditData=scene.scratchData();
  t.after(()=>{reference.dispose();homeAuditData.delete();scene.dispose();});
  // The nominal reference is the reset posture. Measured root tracking lags
  // by 12 cm in Y; placing the fixed object at the nominal wrist creates a
  // collision which exists only in the nominal reconstruction.
  const ik=reference.ik,q=ik.defaultQ.slice(),root=ik.anchorPosition.slice(),nominal={rootPosition:root,rootHeight:root[2],rootPitch:0};
  ik.setPose(q,nominal);
  const wristGeom=ik.audit.robotGeoms.find(g=>scene.bodyName(scene.model.geom_bodyid[g])==='left_wrist_pitch_link'),object=scene.objects.uiuc_i,objectGeom=ik.audit.physicalObjectGeoms(object).find(g=>scene.geomName(g)==='demo_uiuc_i_geom_2');
  assert.ok(Number.isInteger(wristGeom)&&Number.isInteger(objectGeom));scene.data.qpos.set(ik.data.qpos);scene.forward(scene.data);
  // Align native shape centres, not link-frame origins or free-joint bases.
  const wristBox=geomBounds(scene,ik.data,[wristGeom]),objectBox=geomBounds(scene,scene.data,[objectGeom]);
  for(let k=0;k<3;k++)scene.data.qpos[object.qadr+k]+=(wristBox.lower[k]+wristBox.upper[k]-objectBox.lower[k]-objectBox.upper[k])/2;
  scene.data.qpos[1]-=.12;scene.data.qvel.fill(0);scene.forward(scene.data);
  const frame=reference.fkFrame(q,nominal);
  Object.assign(reference,{qCurrent:q.slice(),rootPosition:root.slice(),rootHeight:root[2],rootPitch:0,currentIndex:0,windowTime:scene.time,clip:new ReferenceClip(Array.from({length:51},()=>clone(frame)))});
  const c=Object.assign(Object.create(InteractiveController.prototype),{scene,reference,audit:ik.audit,homeAuditData,hand:'left',objectId:'uiuc_i',phase:'transfer',restQ:q.slice(),rootPosition:root.slice(),rootHeight:root[2],rootPitch:0,dt:.02,syntheticPair:{a:wristGeom,b:objectGeom}});
  // Catch forwarding live data even when a caller restores qpos afterwards.
  for(const key of ['forward','ikForward'])if(typeof scene[key]==='function'){
    const native=scene[key].bind(scene);scene[key]=(data,...args)=>{assert.notEqual(data,scene.data,`${key} must only receive independent scratch data.`);return native(data,...args);};
  }
  return c;
}

test('actual opening captures the issued frame and native measured geometry removes the phantom wrist/object overlap',async t=>{
  const c=await fixture(t),{scene,reference}=c,ik=reference.ik,issued=clone(reference.clip.sample(reference.currentIndex+1)),before=state(scene),support=anchors(ik),oldClip=reference.clip,oldQ=reference.qCurrent.slice();
  c.beginRelease();const target=c.releaseTarget(),frame=target.releaseOpeningCollisionFrame,q=issued.jointPos;
  assert.deepEqual(c.releaseHoldQ,q);assert.deepEqual(frame.referenceUpperQ,q.slice(12));assert.deepEqual(frame.referenceLowerQ,q.slice(0,12));
  assert.deepEqual(c.releasePalms,Object.fromEntries(['left','right'].map((side,i)=>[side,{position:issued.palmPosW[i],rotation:quatToMat(issued.palmQuatW[i])}])));
  assert.ok(norm(sub(frame.measuredRootPosition,frame.referenceRootPosition))>.11,'Synthetic fixture must contain a substantial root tracking bias.');
  assert.ok(Object.isFrozen(frame)&&Object.isFrozen(frame.measuredLowerQ)&&Object.isFrozen(frame.measuredUpperQ));
  const nominal={...target};delete nominal.releaseOpeningCollisionFrame;
  const {a,b}=c.syntheticPair;ik.setPose(q,nominal);const residual=ik.residuals(nominal),distance=ik.audit.nativeDistance(ik.data,a,b,.1),nominalGeometry=ik.audit.check(ik.data,ik.collisionOptions(nominal));
  assert.match(scene.bodyName(scene.model.geom_bodyid[a]),/^left_wrist_/);assert.ok(distance<-.01,`Issued command must reproduce a phantom wrist overlap exceeding 1 cm; distance ${distance}.`);assert.equal(nominalGeometry.passed,false);
  ik.setPose(q,target);const geometry=ik.audit.check(ik.data,ik.collisionOptions(target)),measuredDistance=scene.geomDistance(c.releaseMeasuredData(),a,b,.1);
  assert.ok(measuredDistance>=.005,`Physical wrist must clear the unchanged 5 mm margin; distance ${measuredDistance}.`);
  t.diagnostic(`Issued-frame phantom wrist distance ${distance} m; measured wrist distance ${measuredDistance} m.`);
  assert.ok(Math.abs(ik.audit.nativeDistance(ik.releaseData,a,b,.1)-measuredDistance)<1e-12);
  assert.equal(geometry.passed,true,JSON.stringify(geometry.limitingPair));assert.equal(geometry.margin,.005);
  assert.ok(maximumDelta(ik.releaseData.qpos,before.qpos)<1e-12,'Zero reference increment reproduces all measured root/joint/finger/object qpos.');
  assert.deepEqual(ik.residuals(target),residual,'Nominal foot/palm/COM tasks must remain independent of the collision scratch.');
  assert.deepEqual(anchors(ik),support);assert.equal(reference.clip,oldClip);assert.deepEqual(reference.qCurrent,oldQ);assert.deepEqual(state(scene),before);
  const options=ik.collisionOptions(target),pairs=ik.audit.pairs(options),pairKeys=new Set(pairs.map(pair=>pair.join(':'))),objectGeoms=ik.audit.physicalObjectGeoms(scene.objects.uiuc_i);
  assert.ok(pairKeys.has(`${a}:${b}`),'Selected-hand contact permission must preserve wrist/object checks.');
  for(const handGeom of ik.audit.hands.left)for(const objectGeom of objectGeoms)assert.equal(pairKeys.has(`${handGeom}:${objectGeom}`),false);
  assert.ok(ik.audit.selfPairs.every(pair=>pairKeys.has(pair.join(':'))));
  // A positive native 3 mm wrist gap must still fail the original 5 mm floor.
  // Translate only scratch object geometry, leaving the captured live pose fixed.
  const witness=scene.geomDistanceInfo(ik.releaseData,a,b,.2);assert.ok(witness.hasWitness!==false&&witness.fromto?.length===6);
  const direction=sub(Array.from(witness.fromto.slice(0,3)),Array.from(witness.fromto.slice(3,6))),length=norm(direction),object=scene.objects.uiuc_i;
  for(let k=0;k<3;k++)ik.releaseData.qpos[object.qadr+k]+=direction[k]/length*(witness.distance-.003);
  if(scene.ikForward)scene.ikForward(ik.releaseData);else scene.forward(ik.releaseData);
  const physicalGap=scene.geomDistance(ik.releaseData,a,b,.02),unsafe=ik.audit.check(ik.data,options),row=unsafe.near.find(pair=>pair.a===a&&pair.b===b);
  assert.ok(physicalGap>0&&physicalGap<.005);assert.equal(row.requiredMargin,.005);assert.equal(row.poseFrame,'measured_release_pose');assert.equal(unsafe.passed,false);assert.deepEqual(state(scene),before);
});

test('actual bounded IK and rolling opening window preserve nominal root/legs/waist and measured support without touching live physics',async t=>{
  const c=await fixture(t),{scene,reference}=c,ik=reference.ik,before=state(scene),support=anchors(ik);c.beginRelease();const target=c.releaseTarget(),q=c.releaseHoldQ.slice(),native=scene.geomDistanceInfo.bind(scene),seen=[];
  scene.geomDistanceInfo=(data,...args)=>{seen.push(data);return native(data,...args);};
  const result=ik.solve(q,target,{iterations:40,maxStep:.035});assert.equal(result.passed,true);assert.deepEqual(result.q.slice(0,15),q.slice(0,15));assert.ok(maximumDelta(result.q,q)<1e-8,'A feasible issued opening command must not gain a phantom-collision arm correction.');assert.ok(seen.every(data=>data===ik.releaseData),'Native avoidance witnesses must use measured collision scratch.');
  // Start at the issued command. A same-time force preserves the old/current
  // frame too; the QA trial begins from the API's captured issued anchor instead.
  const trial=Object.assign(Object.create(Object.getPrototypeOf(reference)),reference,{clip:null,qCurrent:q.slice(),rootPosition:target.rootPosition.slice(),rootHeight:target.rootHeight,rootPitch:target.rootPitch,windowTime:-Infinity,currentIndex:0,lastAudit:null});
  const clip=trial.window(()=>c.releaseTarget(),scene.time,{force:true,bestEffort:false});
  assert.equal(trial.lastAudit.passed,true,JSON.stringify(trial.lastAudit));assert.ok(trial.lastAudit.samples>=51);assert.ok(trial.lastAudit.maximumFootDrift<1e-12);
  const measuredFeet=Object.fromEntries(['left','right'].map(side=>[side,bodyPose(c.releaseMeasuredData(),scene.ankleBodyIds[side])]));
  for(const frame of clip.frames){
    assert.deepEqual(frame.jointPos.slice(0,15),q.slice(0,15));assert.deepEqual(frame.rootPosW,target.rootPosition);assert.deepEqual(frame.rootQuatW,clip.frames[0].rootQuatW);
    ik.setPose(frame.jointPos,target);for(const side of ['left','right']){const foot=bodyPose(ik.releaseData,scene.ankleBodyIds[side]);assert.ok(norm(sub(foot.position,measuredFeet[side].position))<1e-12);assert.ok(maximumDelta(foot.rotation,measuredFeet[side].rotation)<1e-12);}
    assert.equal(ik.audit.check(ik.data,ik.collisionOptions(target)).passed,true);
  }
  assert.deepEqual(anchors(ik),support);assert.deepEqual(state(scene),before);
});

test('opening API rejects detached/opening mixing, ignored objects, payload/separation, unlocks and malformed anchors',async t=>{
  const c=await fixture(t),{scene,reference}=c,ik=reference.ik,before=state(scene);c.beginRelease();const target=c.releaseTarget(),q=c.releaseHoldQ,frame=target.releaseOpeningCollisionFrame;
  const badTargets=[
    ['simultaneous detached frame',{releaseCollisionFrame:frame}],
    ['selected object ignored',{ignoreObjectId:'uiuc_i'}],
    ['different object ignored',{ignoreObjectId:'cracker_box'}],
    ['empty ignore ID',{ignoreObjectId:''}],
    ['contact permission disabled',{allowHandObject:false}],
    ['waist unlocked',{lockWaist:false}],
    ['lower body unlocked',{lockLowerBody:false}],
    ['detached separation supplied',{releaseSeparation:{hand:'left',objectId:'uiuc_i',pairs:[]}}],
    ['carried payload',{carriedPayload:{hand:'left',objectId:'uiuc_i',positionInPalm:[0,0,0],rotationInPalm:[1,0,0,0,1,0,0,0,1]}}],
    ['wrong frame hand',{releaseOpeningCollisionFrame:{...frame,hand:'right'}}],
    ['wrong frame object',{releaseOpeningCollisionFrame:{...frame,objectId:'cracker_box'}}],
    ['nonfinite measured root',{releaseOpeningCollisionFrame:{...frame,measuredRootPosition:[NaN,0,0]}}],
    ['missing lower joints',{releaseOpeningCollisionFrame:{...frame,measuredLowerQ:[]}}],
    ['nonfinite upper joints',{releaseOpeningCollisionFrame:{...frame,referenceUpperQ:Array(17).fill(NaN)}}],
    ['root translation',{rootPosition:frame.referenceRootPosition.map((v,i)=>v+(i===0?.001:0))}],
    ['root rotation',{rootPitch:.001}],
  ];
  for(const [label,change] of badTargets){assert.throws(()=>ik.setPose(q,{...target,...change}),IKError,label);assert.deepEqual(state(scene),before,`${label} must not mutate live physics.`);}
  const changed=q.slice();changed[0]+=.001;assert.throws(()=>ik.setPose(changed,target),IKError,'Lower-body nominal movement must be rejected.');
  const shiftedUpper=frame.measuredUpperQ.slice(),outside=q.slice();shiftedUpper[3]+=.04;outside[15]=ik.upper[15];assert.throws(()=>ik.setPose(outside,{...target,releaseOpeningCollisionFrame:{...frame,measuredUpperQ:shiftedUpper}}),IKError,'Mapped physical safety limits must remain active during opening.');
  assert.throws(()=>ik.solve(q,{...target,releaseJointBounds:{lower:Array(29).fill(-10),upper:Array(29).fill(10)}}),IKError,'Detached-only bounds cannot be injected into opening.');assert.deepEqual(state(scene),before);
});
