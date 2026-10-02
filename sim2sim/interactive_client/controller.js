import {add,sub,scale,dot,norm,lerp,clamp,smooth,eye3,matVec,matMul,transpose3,rotateY,rotateZ,quatToMat,matToQuat,rotationBlend,rotationError,polygonMargin} from './numerics.js';
import {IKError,bodyPosition,bodyRotation,bodyPose,geomBounds,geomCorners} from './ik.js';
import {RollingReference,payloadTargetSampler} from './reference.js';
import {planReachStance} from './stance.js';
import {BOTTLE_SIDE_GRASP_LOWER_M,objectTrayOverlap} from './object_geometry.js';

const SIDES=['left','right'],copy=v=>Array.from(v),otherHand=hand=>hand==='right'?'left':'right';
const PHASES=['idle','planning','crouch','stance_recovery','approach','close','lift','hold','stand','transfer','release','settle','return_home','succeeded','failed'];
const poseOf=(scene,id)=>{
  const value=scene.objectPose(id);
  if(Array.isArray(value))return {position:copy(value[0]),quaternion:copy(value[1])};
  return {position:copy(value.positionW||value.position_w||value.position),quaternion:copy(value.quaternionW||value.quaternion_wxyz||value.quaternion)};
};
const tableZ=scene=>scene.tableTopZ;
const nowOf=scene=>Number(scene.data.time);
const configValue=(config,camel,snake,fallback)=>config[camel]??config[snake]??fallback;
export const GRASP_TOLERANCE_M=.015;
export const GRASP_ANGLE_TOLERANCE_RAD=Math.PI/12;
const NEAR_GOAL_FORCE_MAX_M=.03;
export const MEASURED_SELF_OVERLAP_TOLERANCE_M=.010;
// Post-release budgets: the whole exit + return is bounded in simulation seconds from the open command.
export const POST_RELEASE_DEADLINE_S=60;
export const RELEASE_OPENING_DEADLINE_S=7;
export const RELEASE_STAGE_SLACK_S=3;
export const RELEASE_RETREAT_NOMINAL_M=.10,RELEASE_RETREAT_MIN_M=.05,RELEASE_RETREAT_OUTBOARD_M=.02,RELEASE_RETREAT_WALL_MARGIN_M=.01;
export const HOME_CLEARANCE_RECOVERY_CAP=3;
export const POST_RELEASE_HAZARD_LOG_MAX=32;
// Robot body / hand-fixture contact (table edge, tray) is acceptable up to 10 N; only a sustained stronger push
// stops the robot. Peak forces are collected over 1 ms substeps, so the limit must hold for
// BODY_FIXTURE_FORCE_DWELL_S of control time before it is terminal.
export const BODY_FIXTURE_FORCE_LIMIT_N=10,BODY_FIXTURE_FORCE_DWELL_S=.1;
// After the single alignment retry the attempt is never abandoned without closing. If the hand still cannot come
// back within the 15 mm gate, it grasps from the closest reachable pose as long as the palm is within
// FINAL_ATTEMPT_MAX_M of the target (beyond that the fingers cannot touch the object); the wrist tolerance doubles for it.
export const FINAL_ATTEMPT_MAX_M=.06;
export const TOP_DOWN_TABLE_RELEASE_MIN_CLEARANCE_M=.010;
export const TOP_DOWN_TABLE_RELEASE_MAX_M=.04;
export const HAND_CLOSURE_GATES={dex3:{normal:GRASP_TOLERANCE_M,forced:NEAR_GOAL_FORCE_MAX_M},inspire:{normal:.010,forced:NEAR_GOAL_FORCE_MAX_M}};
// Hands with a long wrist-to-grasp offset need a separate calibrated-center
// check. Dex3 uses palm position for grasp admission.
export const INSPIRE_CENTER_CLOSURE_GATES={normal:.010,forced:.015};
export const APPLE_SIDE_CENTER_ABOVE_TABLE_M=.065;
export const OBJECT_FORCED_CLOSURE_M={apple:.04,can:.04,mug:.04,cracker_box:.04,bottle:.04};
export const SIDE_APPROACH_BIAS_M=0;
export const HAND_FOLD_ELBOW_RAD={dex3:-.8,inspire:-1.2};
export const HAND_FOLD_DURATION_SCALE={dex3:1,inspire:1.6};
// YCB cracker box (Cheez-It): a Side grasp puts the palm on the end face nearest the hand, with the thumb and
// fingers pinching the two printed faces across the 72 mm thickness. The wrap centre sits 33 mm inside the end
// face (a Can's radius), so the box centre is offset from it by the half-length minus that depth.
export const CRACKER_BOX_GRASP_INSET_M=.082-.033;
// The 164 mm carton must overlap the tray by at least half its footprint before it is lowered.
export const CRACKER_BOX_TRAY_OVERLAP_MIN_M2=.5*.0718*.164;
export const CRACKER_BOX_UPRIGHT_MIN_COS=Math.cos(15*Math.PI/180);
/** Fold the carton's yaw (2-fold symmetric) so the grasped end face faces the hand, and follow the carton with wrist yaw 0.
 * `toward` (world, horizontal) points from the carton centre to the palm's start. The two symmetric branches put the
 * palm at point-symmetric spots about the carton centre, so the branch whose palm is nearer the palm start is kept:
 * comparing palms (not end faces) stays well conditioned for a square carton, whose two ends are equally near, and
 * flips only when the world-y default would plan the far end (carton turned across or away from the reach). */
// Carton grasp variants. Each turns the hand by `deltaSign*sign*delta` about the vertical relative to the carton-aligned
// end-face plan and puts a hand-frame anchor point at an anchor on the grasped end of the carton:
// anchor = carton centre - sign*edgeInsetM*along + edgeX*x_c (x_c = carton thickness axis; along = long axis; the grasped end is at -sign*along).
//   crotch  : hand -sign*45 deg, anchor = vertical edge E2 (-x_c side) at the thumb-index web (a pure V contact; slips)
//   spine   : hand +sign*CARTON_SPINE_TURN_RAD, anchor = end-face centre (narrow face in the thumb-finger web; thumb on the -x_c face,
//             fingers on the +x_c face) -- the thumb-web grasp used for the carton at every yaw. At a 45 deg turn the fingers
//             point too far along the carton and poke it (with the end face at the Dex3 web point (55, 20.3) mm the near outer
//             edge sat at (80.5, -5.2) mm: at the palm's edge, inside the finger-base links), so the turn is 35 deg.
//   spine90 : hand +sign*90 deg, anchor = vertical edge E1 (+x_c side; palm against the +x_c face, fingers along the carton)
// `web` holds the thumb-index web point per hand model; a hand without one falls back to the end-face pinch.
export const CRACKER_BOX_CROTCH={deltaRad:Math.PI/4,edgeInsetM:.082,edgeHalfThicknessM:.0359,web:{dex3:[.055,.0203,0]}};
export const CARTON_SPINE_TURN_RAD=35*Math.PI/180;
export const CARTON_GRASP_VARIANTS={crotch:{deltaSign:-1,delta:Math.PI/4,edgeX:-CRACKER_BOX_CROTCH.edgeHalfThicknessM},spine:{deltaSign:1,delta:CARTON_SPINE_TURN_RAD,edgeX:0},spine90:{deltaSign:1,delta:Math.PI/2,edgeX:CRACKER_BOX_CROTCH.edgeHalfThicknessM}};
export const CARTON_GRASP_ALIASES={spine45:'spine'};// the 45 deg turn's former name
export const CARTON_GRASP_SETTINGS=['auto','end_face',...Object.keys(CARTON_GRASP_VARIANTS)];
/** Canonical `carton_grasp` setting: resolves the legacy alias; non-strings pass through for the caller's validation. */
export function cartonGraspSettingOf(value){return typeof value==='string'?CARTON_GRASP_ALIASES[value]??value:value;}
// Dex3 spine anchor (palm frame, m): the end-face centre at (50.5, 40.7, 0) mm. With the 35 deg turn the carton axis in the palm
// frame is (sin 35, cos 35) and its thickness axis (cos 35, -sin 35), so the near outer edge E1 = (80, 20) sits 7 mm above the
// index/middle proximal links (top y 13) at the palm's edge, the near inner edge E2 = (21, 61) is 23 mm above the thumb_0 body and
// 33 mm on the +x side of the open thumb, and the end face stands ~20 mm in front of the palm surface, so the forward insertion
// slides the carton in without sweeping it. Closing: the thumb (pivot (23, 19), straightening from -40 deg to 0, then the distal
// curling +x) meets the inner face at its corner and its distal pad lands on the face ~15-30 mm deep; the finger proximal links
// rotating about (78, -2) reach the outer face ~26 mm deep (tip (97, 40) at ~65 deg) -- both jaws at a similar depth.
export const CARTON_SPINE_ANCHOR_DEX3=[.0505,.0407,0];
/** Spine geometry: the hand turn about the vertical and the hand-frame point placed at the end-face centre (the Dex3 seat). */
export function cartonSpineGeometry(){return {deltaRad:CARTON_SPINE_TURN_RAD,anchor:CARTON_SPINE_ANCHOR_DEX3};}
/** Hand-frame point placed at the variant's carton anchor: the spine seat for `spine`, the thumb-index web point for the
 * crotch / spine90 variants; null when the hand has no such pose (falls back to end_face). */
export function cartonVariantAnchor(variant,handModel){
  if(variant==='spine')return cartonSpineGeometry(handModel).anchor;
  return CRACKER_BOX_CROTCH.web[handModel]??null;
}
// 'auto' selects the thumb-web `spine` grasp for every carton orientation (CARTON_SPINE_AUTO_MAX_RAD = pi keeps the cone test
// trivially true): the end-face push pose is not used at any carton yaw. The yaw band (worker.js, sign(y) x [40, 90] deg) bounds
// the placements the hand faces.
export const CARTON_SPINE_AUTO_MAX_RAD=Math.PI;
// Hands whose 'auto' setting may pick `spine`.
export const CARTON_SPINE_AUTO_HANDS=['dex3'];
export function cartonGraspChoice(setting,handModel,objectYaw,robotHeading){
  if(setting!=='auto')return setting;
  if(!CARTON_SPINE_AUTO_HANDS.includes(handModel))return 'end_face';
  const along=[-Math.sin(objectYaw),Math.cos(objectYaw)],heading=[Math.cos(robotHeading),Math.sin(robotHeading)];
  return Math.abs(along[0]*heading[0]+along[1]*heading[1])>=Math.cos(CARTON_SPINE_AUTO_MAX_RAD)-1e-12?'spine':'end_face';
}
export function crackerBoxSidePlan(objectYaw,sign,toward=null,offset=[.077,.055,0],{variant='end_face',web=null,deltaRad=null}={}){
  let yaw=objectYaw;while(yaw>Math.PI/2)yaw-=Math.PI;while(yaw<=-Math.PI/2)yaw+=Math.PI;
  let along=[-Math.sin(yaw),Math.cos(yaw),0];// carton long axis (local +y) in the world; the grasped end is at -sign*along
  if(toward){
    const shift=scale(along,-sign*CRACKER_BOX_GRASP_INSET_M),palmRel=sub(shift,matVec(rotateZ(yaw),[offset[0],sign*offset[1],0]));
    if(palmRel[0]*toward[0]+palmRel[1]*toward[1]<0){yaw+=yaw>0?-Math.PI:Math.PI;along=scale(along,-1);}
  }
  const base={rotation:rotateZ(yaw),yawDeg:0,foldedYaw:yaw,centerShift:scale(along,-sign*CRACKER_BOX_GRASP_INSET_M)};
  const spec=CARTON_GRASP_VARIANTS[variant];
  if(!spec||!web)return base;
  const xc=[Math.cos(yaw),Math.sin(yaw),0];// carton local +x (thickness axis) in the world
  const rotation=rotateZ(yaw+spec.deltaSign*sign*(deltaRad??spec.delta));
  const edgeShift=add(scale(along,-sign*CRACKER_BOX_CROTCH.edgeInsetM),scale(xc,spec.edgeX));
  return {...base,rotation,variant,edgeShift,graspOffset:[web[0],sign*web[1],web[2]]};
}
export const INSPIRE_SIDE_ROLL_RAD=Number(globalThis.process?.env?.TABLETOP_INSPIRE_ROLL_DEG??0)*Math.PI/180;
export const INSPIRE_FINGER_DROP_M=.064,INSPIRE_FINGER_TABLE_MARGIN_M=.006;
function rollAboutX(t){const c=Math.cos(t),s=Math.sin(t);return [1,0,0,0,c,-s,0,s,c];}
export const HAND_GRASP_GEOMETRY={dex3:{offset:[.077,.055,0],topDownFloor:.105,topDownAboveObject:.015},inspire:{offset:[.077,.073,-.026],topDownFloor:.145,topDownAboveObject:.05,...(INSPIRE_SIDE_ROLL_RAD>0?{sideRoll:INSPIRE_SIDE_ROLL_RAD,sideCenterMinAboveTable:INSPIRE_FINGER_DROP_M*Math.cos(INSPIRE_SIDE_ROLL_RAD)+INSPIRE_FINGER_TABLE_MARGIN_M-.026}:{})}};
export const handModelOf=(scene,config={})=>config.hand_model??scene?.handModel??'dex3';
const GRASP_ALIGNMENT_RETRY_S=3.2,GRASP_FINAL_APPROACH_BUDGET_S=10;
// The single alignment retry has two phases. Phase 1 holds the accepted pose for GRASP_ALIGNMENT_RETRY_S while the
// fingers open and the bounded refinement runs (a hand that is still converging closes here under the base gates).
// Only when the hold deadline passes with the gates still unmet does phase 2 start: a real second approach that backs
// the open palm out along the entry direction, re-enters and settles again with the ordinary GA loop rebuilt from the
// nominal grasp; a second approach that still fails (settle timeout, lost closure, GRASP_SECOND_APPROACH_S budget) ends
// in the final attempt. Nothing ever closes on a timer while the hand is still moving.
const GRASP_SECOND_APPROACH_S=10,RETRY_BACKOFF_M=.08,RETRY_BACKOFF_S=1.2,RETRY_REENTER_S=1.7;
const NEAR_GOAL_MIN_DWELL_S=2,NEAR_GOAL_STAGNATION_S=1,NEAR_GOAL_MAX_DWELL_S=4,NEAR_GOAL_PROGRESS_MAX_DWELL_S=6,NEAR_GOAL_PROGRESS_M=.001;
// Keep the recovery guard independent of finger-closing precision so
// small grasp corrections do not trigger a full withdrawal.
const NEAR_GRASP_REFINEMENT_M=.05;
const GRASP_SETTLE_TIMEOUT_S=3.4;
const minimumJerk=t=>{const u=clamp(t,0,1);return u*u*u*(10+u*(-15+6*u));};
const stanceSummary=stance=>stance?Object.fromEntries(['id','feasible','reason','rootPosition','rootHeight','rootPitch','waistPitchTarget','score','jointMargin','waistMargin','comMargin','evaluated','selectionReason'].filter(key=>stance[key]!==undefined).map(key=>[key,stance[key]])):null;
export function policyFeedback(mode){return {replan:mode==='hero_plus',goal_adjust:mode==='hero_plus'};}

/** Bounded acquisition correction, based on measured live EE error in meters. */
export function acquisitionStepLimit(distance,maxStep=.01){
  if(!Number.isFinite(distance)||distance<0||!Number.isFinite(maxStep)||maxStep<0)throw new RangeError('Acquisition distance and step limit must be finite and nonnegative.');
  const u=clamp((distance-.015)/.045,0,1);
  const scheduled=u===0?.001:u===1?.01:.001+.009*u*u;
  return Math.min(maxStep,scheduled);
}

/** Evidence from consumed collision-checked references, never physical control. */
export class ApproachObstructionMonitor {
  constructor({dwell=1,code='approach_obstructed'}={}){this.dwell=dwell;this.code=code;this.reset();}
  reset(){this.tracks=new Map();this.confirmed=null;}
  update({time,eligible,referenceDistance,requestedError,constraints=[]}){
    if(!eligible||!Number.isFinite(time)||!(referenceDistance>.02)||!(requestedError>.02)||!constraints.length){this.reset();return null;}
    const present=new Set(constraints.map(row=>row.object_id));
    for(const id of this.tracks.keys())if(!present.has(id))this.tracks.delete(id);
    for(const row of constraints){
      let track=this.tracks.get(row.object_id);
      if(!track){track={since:time,bestDistance:referenceDistance};this.tracks.set(row.object_id,track);}
      // Accumulate sub-millimeter improvements; a useful two-millimeter gain
      // restarts the observation window instead of labelling passing contact.
      if(referenceDistance<track.bestDistance-.002){track.bestDistance=referenceDistance;track.since=time;}
      track.evidence={...row,observed_s:time-track.since,reference_distance_m:referenceDistance,requested_position_error_m:requestedError};
    }
    const blocked=[...this.tracks.values()].filter(track=>time-track.since>=this.dwell-1e-9).map(track=>track.evidence);
    this.confirmed=blocked.length?{code:this.code,objects:blocked}:null;
    return this.confirmed;
  }
}

/** Exact UI grasp calibration: wrist yaw changes orientation, never the approach ray. */
export function computeGraspPreview(scene,config={}){
  const objectId=config.objectId||config.object_id||scene.selectedObjectId||Object.keys(scene.objects||{}).find(id=>scene.objects[id].active);
  if(!objectId||!scene.objects[objectId]?.active)return null;
  const {position,quaternion}=poseOf(scene,objectId),state=scene.readState(objectId);
  const local=matVec(transpose3(quatToMat(state.rootQuatW)),sub(position,state.rootPosW));
  const hand=config.hand&&config.hand!=='auto'?config.hand:local[1]>0?'left':'right';
  if(!SIDES.includes(hand))throw new Error('Select the left or right hand.');
  const mode=config.mode||'hero_plus',approach=configValue(config,'approach','grasp_style','side'),yawDeg=Number(configValue(config,'yawDeg','ee_yaw_deg',0));
  if(!['side','top_down'].includes(approach)||![0,30,45,60].includes(yawDeg))throw new Error('Use Side or Top Down with outward wrist yaw 0°, 30°, 45°, or 60°.');
  const sign=hand==='right'?1:-1,[w,x,y,z]=quaternion,objectYaw=Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));
  // Block I may turn during placement, but its side grasp keeps the robot's
  // forward frame. The independent wrist-yaw setting still applies.
  const blockForward=objectId==='uiuc_i'&&approach==='side';
  const rootForward=matVec(quatToMat(state.rootQuatW),[1,0,0]),robotHeading=Math.atan2(rootForward[1],rootForward[0]);
  const handModel=handModelOf(scene,config),geometry=HAND_GRASP_GEOMETRY[handModel]??HAND_GRASP_GEOMETRY.dex3;
  const palmStart=bodyPosition(scene.data,scene.palmBodyIds[hand]);
  if(objectId==='cracker_box'&&quatToMat(quaternion)[8]<CRACKER_BOX_UPRIGHT_MIN_COS)throw new Error('Stand the carton upright before picking it.');
  // Carton grasp setting (`carton_grasp`: auto | end_face | crotch | spine | spine90; default auto).
  const cartonGraspSetting=cartonGraspSettingOf(configValue(config,'cartonGrasp','carton_grasp','auto'));
  if(!CARTON_GRASP_SETTINGS.includes(cartonGraspSetting))throw new Error('Use the auto, end_face, crotch, spine or spine90 carton grasp.');
  const cartonGrasp=cartonGraspChoice(cartonGraspSetting,handModel,objectYaw,robotHeading);
  const cartonOffset=geometry.offset;
  const box=objectId==='cracker_box'&&approach==='side'?crackerBoxSidePlan(objectYaw,sign,[palmStart[0]-position[0],palmStart[1]-position[1],0],cartonOffset,{variant:cartonGrasp,web:cartonVariantAnchor(cartonGrasp,handModel),...(cartonGrasp==='spine'?{deltaRad:cartonSpineGeometry(handModel).deltaRad}:{})}):null,planYawDeg=box?box.yawDeg:yawDeg;
  const rotationYaw=box?box.rotation:rotateZ((blockForward?robotHeading:objectYaw)-sign*yawDeg*Math.PI/180),offset=box?[cartonOffset[0],sign*cartonOffset[1],cartonOffset[2]]:[geometry.offset[0],sign*geometry.offset[1],geometry.offset[2]];
  let rotation,grasp,pregrasp,retract=null,side=null;
  if(approach==='top_down'){
    // Open Dex3 fingers extend along local +X. Fan this horizontal axis from
    // the selected palm toward the object, independently of the object's yaw.
    // The optional outward wrist rotation never tilts the vertical approach.
    const dx=position[0]-palmStart[0],dy=position[1]-palmStart[1];
    const forward=matVec(quatToMat(state.rootQuatW),[1,0,0]);
    const heading=Math.hypot(dx,dy)>1e-8?Math.atan2(dy,dx):Math.atan2(forward[1],forward[0]);
    rotation=matMul(rotateZ(heading-sign*yawDeg*Math.PI/180),[1,0,0,0,0,sign,0,-sign,0]);grasp=sub(position,matVec(rotation,offset));grasp[2]=Math.max(grasp[2],tableZ(scene)+geometry.topDownFloor);
    if(['uiuc_i','bottle','cracker_box'].includes(objectId))grasp[2]=Math.max(grasp[2],position[2]+scene.objects[objectId].height*.5+geometry.topDownAboveObject);// tall objects: the palm stays above the top
    if(objectId==='can'&&tableZ(scene)<=.74&&!(mode==='hero_plus'&&hand==='right'))grasp[2]+=.02;
    pregrasp=add(grasp,[0,0,.12]);
  }else{
    // Place the Apple pinch around the fruit, below its narrow crown and stem.
    rotation=geometry.sideRoll&&!box?matMul(rotationYaw,rollAboutX(-sign*geometry.sideRoll)):rotationYaw;const center=box?add(position,box.edgeShift??box.centerShift):position.slice();
    if(geometry.sideCenterMinAboveTable!==undefined){if(!box)center[2]=Math.max(position[2],tableZ(scene)+geometry.sideCenterMinAboveTable);}
    else if(objectId==='can')center[2]=tableZ(scene)+.065;else if(objectId==='apple')center[2]=tableZ(scene)+APPLE_SIDE_CENTER_ABOVE_TABLE_M;else if(objectId==='mug')center[2]=tableZ(scene)+.065;
    grasp=sub(center,matVec(rotation,box?.graspOffset??offset));if(objectId==='bottle')grasp=add(grasp,[.015*Math.cos(objectYaw),.015*Math.sin(objectYaw),-BOTTLE_SIDE_GRASP_LOWER_M]);
    let direction=blockForward?[Math.cos(robotHeading),Math.sin(robotHeading),0]:[position[0]-palmStart[0],position[1]-palmStart[1],0];if(norm(direction)<1e-8)direction=[1,0,0];direction=scale(direction,1/norm(direction));
    if(box){// never back the carton pregrasp off across its end-face plane: the fingers would sweep the carton on the way in
      const n=matVec(rotation,[0,sign,0]),dn=direction[0]*n[0]+direction[1]*n[1];
      if(dn<0){const adjusted=[direction[0]-dn*n[0],direction[1]-dn*n[1],0];direction=norm(adjusted)>1e-6?scale(adjusted,1/norm(adjusted)):[n[0],n[1],0];}
    }
    if(SIDE_APPROACH_BIAS_M)grasp=add(grasp,scale(direction,SIDE_APPROACH_BIAS_M));
    pregrasp=sub(grasp,scale(direction,.12));retract=palmStart.slice();retract[0]=Math.min(palmStart[0],pregrasp[0]-.08);retract[2]=grasp[2];side=pregrasp.slice();side[0]=retract[0];
  }
  const lift=add(grasp,[-.015,0,.13]);
  return {objectId,hand,approach,yawDeg:planYawDeg,signedYawDeg:-sign*planYawDeg||0,...(box?{cartonFoldedYaw:box.foldedYaw,cartonGrasp:box.variant??'end_face',cartonGraspSetting}:{}),position_w:grasp,rotation_w:rotation,quaternion_wxyz:matToQuat(rotation),approach_start_w:pregrasp,lift_position_w:lift,retract_position_w:retract,side_position_w:side,label:approach==='side'?'Side · horizontal':'Top Down · vertical',object_position_w:position,palm_start_w:palmStart};
}

/** Persistent HERO goal adjustment, with the original measured goal kept immutable. */
export class GoalAdjuster {
  constructor(goal,{enabled=true,gain=.6,maxStep=.01,gate=.15,stay=.0175,skip=.02,maxReplans=20}={}){this.original=copy(goal);this.goal=copy(goal);Object.assign(this,{enabled,gain,maxStep,gate,stay,skip,maxReplans});this.replans=0;this.stayed=false;}
  update(actual){const error=sub(actual,this.original),distance=norm(error);if(this.stayed||distance<=this.stay){this.stayed=true;return {action:'stay',shift:[0,0,0]};}if(distance<this.skip||this.replans>=this.maxReplans)return {action:'skip',shift:[0,0,0]};let shift=[0,0,0];if(this.enabled&&(distance<this.gate||this.replans>2)){shift=scale(error,-Math.min(this.gain,this.maxStep/distance));this.goal=add(this.goal,shift);}this.replans++;return {action:'replan',shift};}
}

/** Physical pick/place state machine. Only policy output drives the live body PD. */
export class InteractiveController {
  // Closure gates follow the hand model; objects built without the constructor (tests) get Dex3 gates.
  // After the single alignment retry or the single stance replan the object-specific 40 mm fallback no longer
  // applies: the second approach must come back within the base gates (15 mm normal / 30 mm forced).
  get retriedAttempt(){return (this.alignmentRetryCount??0)>0||(this.stanceReplanCount??0)>0;}
  get gates(){const base=HAND_CLOSURE_GATES[this.handModel]??HAND_CLOSURE_GATES.dex3;if(this.finalAttempt)return {normal:base.normal,forced:Math.max(base.forced,FINAL_ATTEMPT_MAX_M)};const forced=base===HAND_CLOSURE_GATES.dex3&&!this.retriedAttempt?OBJECT_FORCED_CLOSURE_M[this.objectId]:undefined;return forced>base.forced?{normal:base.normal,forced}:base;}
  get closureAngleLimit(){return this.finalAttempt?2*GRASP_ANGLE_TOLERANCE_RAD:GRASP_ANGLE_TOLERANCE_RAD;}
  constructor(scene,policy,config={}){
    this.handModel=handModelOf(scene,config);
    if(!policy?.control)throw new Error('Load a HERO ONNX policy before starting the controller.');
    this.scene=scene;this.policy=policy;this.config=config;this.mode=config.mode||'hero_plus';const feedback=policyFeedback(this.mode);this.replanEnabled=feedback.replan;this.goalAdjustEnabled=feedback.goal_adjust;this.depositEnabled=config.deposit!==false;
    this.reference=new RollingReference(scene,{...config.referenceOptions,bestEffort:true});this.audit=this.reference.ik.audit;this.homeAuditData=scene.scratchData();this.dt=.02;this.reset({resetScene:false,resetPolicy:false});
  }
  reset({resetScene=false,resetPolicy=true}={}){
    if(resetScene)this.scene.reset();if(resetPolicy)this.policy.reset();this.reference.reset();
    this.phase='idle';this.stage=null;this.objectId=null;this.hand='right';this.message='Place an object in the highlighted area, then click Pick & place.';this.startTime=nowOf(this.scene);this.phaseTime=this.startTime;this.segmentTime=this.startTime;
    this.success=null;this.graspSuccess=null;this.depositSuccess=null;this.returnSuccess=null;this.inTray=false;this.stableLift=0;this.stableDeposit=0;this.maxLift=0;this.lift=0;this.contacts=0;this.handContacts=0;this.trayContacts=0;this.recentContacts=[];this.graspSupport=null;
    this.everPlaced=false;this.trayLanding=null;this.scene.lastStepTrayLandings=[];
    this.lowerContactHold=0;this.lowerContactGrounded=false;this.lowerSupportContactTime=null;this.lowerSupport=null;this.homeRetractHeadCorridor=null;
    this.restQ=this.reference.ik.defaultQ.slice();this.homeQ=this.restQ.slice();this.homePalms=structuredClone(this.reference.ik.initialPalms);const rootR=this.reference.ik.anchorRotation,root=this.reference.ik.anchorPosition;
    this.homeLocal={};for(const side of SIDES)this.homeLocal[side]={position:matVec(transpose3(rootR),sub(this.homePalms[side].position,root)),rotation:matMul(transpose3(rootR),this.homePalms[side].rotation)};
    this.rootPosition=root.slice();this.standingHeight=root[2];this.rootHeight=this.standingHeight;this.rootPitch=0;this.crouched=false;this.stancePlan=null;this.crouchOffsets={left:[0,.05,0],right:[0,-.05,0]};
    this.plan=null;this.coordinatedReach=null;this.restPalmCache=null;this.homeStage=null;this.homePlanChecks=[];this.homeEndpointAudit=null;this.homeRootRecovery=null;this.homeFrom=null;this.homeRetracted=null;this.homeMeasuredAuditTime=-Infinity;this.homeVerifyTime=null;this.homeHandOpening=null;this.releasePalms=null;this.clearTrayFrom=null;this.homeClearance=null;this.homeJointError=null;this.homePalmError=null;this.homeAngleError=null;this.actualSelfClearance=null;this.trayOverlap=0;this.trayBottomContacts=0;this.bodyFixtureForce=0;this.bodyFixturePenetration=0;this.handTableForce=0;this.handTablePenetration=0;
    this.replans=0;this.gaUpdates=0;this.goalAdjustment=[0,0,0];this.nextReplan=0;this.goalAdjuster=null;this.gaGraspTransition=null;this.nearGraspRefinement=null;this.liftFrom=null;this.liftPosture=null;this.liftWaistPostureWeights=null;this.objectLocked=false;this.reseed=false;this.forceReference=true;this.lastError=null;this.homeTerminal=null;
    this.graspGoal=null;this.grasp=null;this.closeTrigger=null;this.forcedClose=false;this.forcedCloseTrigger=null;this.forcedCloseTarget=null;this.completionFallback=false;this.completionReasons=[];this.completionTransfer=false;this.motionCompleted=false;this.closeDuration=1.8;this.failurePhase=null;
    this.closeProgress=0;this.graspReadyTime=0;this.graspWaitBest=Infinity;this.graspProgressTime=0;this.nearGoalAttempt=null;this.fingerContactCount=0;this.gripContactTime=0;this.homeClearanceRetries=0;this.homeClearanceRecovery=null;this.transferGoalTransitions=[];this.transferPostureFrom=null;this.transferWaistWeightsFrom=null;this.transferContinuityFrom=null;this.transferBodyRelativeHands=null;this.transferPalmOffsetsTorso=null;this.standPosture=null;this.standWaistPostureWeights=null;
    this.stanceHistory=[];this.stanceReplanCount=0;this.gaFrozenReason=null;this.goalHistory=[];this.reachTracking=null;this.topDownTableRelease=null;this.homeRetractOutboardFirst=null;this.failureMessage=null;this.recoveryReturn=false;this.pendingReturn=false;this.fistReturn=false;this.approachRate=1;this.recoveryPlan=null;this.stanceRecoveryReason=null;this.recoveryPosture=null;this.recoveryWaistPostureWeights=null;
    this.activeHandContacts=0;this.payloadPredictionEnabled=false;this.payloadContactSince=null;this.payloadPredictionTransitions=[];
    this.releaseExit=null;this.releaseHandOpening=null;this.homeRetractHeadCorridor=null;this.postReleaseDeadline=null;this.postReleaseHazards=[];this.homeStageVisit=0;this.homeRecoveryAttemptVisit=null;this.homeClearanceViolations=0;this.bodyFixtureForceSince=null;this.attemptNotice=null;this.finalAttempt=false;
    this.approachObstruction=new ApproachObstructionMonitor();this.forcedCloseObstruction=null;this.graspObstructionCandidate=null;
    this.transferObstruction=new ApproachObstructionMonitor({dwell:3,code:'transfer_obstructed'});
    this.failureReason=null;this.transferBudgetStarted=null;this.transferStageStarted=null;
    this.alignmentRetry=null;this.alignmentRetryCount=0;this.finalApproachDeadline=null;this.closeDeadline=null;this.graspAbort=null;this.closureRejectedTrigger=null;this.liftRotation=null;this.lastClosureGate=null;this.closureCommitted=null;
    for(const side of SIDES)this.scene.setHandClosure(side,0);return this.snapshot();
  }
  dispose(){this.reference.dispose();this.homeAuditData.delete?.();}
  cancel(){if(this.busy){this.observeTrayLanding();this.depositSuccess=this.everPlaced===true;this.success=this.depositSuccess;this.returnSuccess=false;this.phase='failed';this.message='Motion stopped. Reset the scene before starting another grasp.';}return this.snapshot();}
  get busy(){return !['idle','succeeded','failed'].includes(this.phase);}
  get ignoredObjectId(){return this.releaseExit?.objectIgnored?this.objectId:null;}
  get postRelease(){return !!this.releaseExit&&!this.graspAbort&&['release','settle','return_home'].includes(this.phase);}
  preview(options={}){return computeGraspPreview(this.scene,{...this.config,...options});}
  /** Preview for state reporting: a scene that cannot be planned (e.g. a toppled carton) yields no ghost instead of an error. */
  safePreview(options={}){try{return this.preview(options);}catch{return null;}}
  enter(phase,message){
    if(!PHASES.includes(phase))throw new Error(`Unknown controller phase: ${phase}`);
    if(['lift','release','return_home','succeeded','failed'].includes(phase))this.attemptNotice=null;
    if(phase==='lift'){
      this.liftRotation=(this.forcedCloseTarget?this.forcedCloseTarget.palms[this.hand].rotation:this.alignmentRetry?this.approachTarget().palms[this.hand].rotation:this.graspRotation).slice();
      // Keep the coordinated waist objective through loaded lift and hold.
      // Dropping it at closure redistributes the same palm pose abruptly
      // between torso and arm, disturbing the newly formed finger contact.
      const approach=this.coordinatedReach?this.approachTarget():null;
      this.liftPosture=approach?.posture?.slice()??null;this.liftWaistPostureWeights=approach?.waistPostureWeights?.slice()??null;
      // A secured grip can start lifting before the latest GA ramp finishes.
      // Continue from the current command instead of jumping to its endpoint.
      this.liftFrom=this.forcedCloseTarget?this.forcedCloseTarget.palms[this.hand].position.slice():this.gaGraspTransition?this.approachTarget().palms[this.hand].position.slice():this.grasp.slice();
      this.gaGraspTransition=null;
    }
    if(phase==='stance_recovery'){this.gaGraspTransition=null;this.liftFrom=null;}
    if(phase!=='approach'){this.nearGoalAttempt=null;this.approachObstruction?.reset();}
    this.phase=phase;this.phaseTime=nowOf(this.scene);this.message=message;this.forceReference=true;
  }
  fail(message,error=null,{terminal=false}={}){
    // The demo counts a released object that has physically reached the tray,
    // even if it later bounces out or arm restoration is interrupted.
    this.observeTrayLanding();this.depositSuccess=this.depositEnabled!==false&&this.everPlaced===true;
    this.lastError=error?{message:error.message,details:error.details||null}:null;
    const hardStop=terminal||['idle','planning','return_home','failed','succeeded'].includes(this.phase)||this.recoveryReturn;
    if(!hardStop){
      this.failurePhase??=this.phase;this.failureMessage??=message;
      this.failureReason??={code:'motion_stopped',reason:message,phase:this.phase,stage:this.stage,target_object_id:this.objectId};
      this.success=this.depositSuccess;if(this.depositSuccess)this.returnSuccess=false;this.recoveryReturn=true;this.pendingReturn=true;this.message=`${message} Returning the hand to the initial posture.`;return;
    }
    if(this.phase==='return_home'||this.depositSuccess)this.returnSuccess=false;
    this.failurePhase=this.recoveryReturn?(this.failurePhase??this.phase):this.phase;this.success=this.depositSuccess;
    const final=this.recoveryReturn&&this.phase==='return_home'?`${this.failureMessage??message} No collision-free return path was found; motion stopped.`:message;
    if(this.recoveryReturn||this.fistReturn)for(const side of SIDES)this.scene.setHandClosure?.(side,0);
    this.enter('failed',final);
  }
  failObstructed(evidence){
    this.failureReason={...structuredClone(evidence),phase:evidence.phase??this.phase,stage:evidence.stage??this.stage,reset_required:true};
    this.fail('Path blocked by another object. Could not find a collision-free IK solution. Please reset the scene.');
  }
  rememberGraspObstruction(){
    if(this.phase!=='approach')return null;
    const evidence=this.approachObstruction?.confirmed;
    const objects=evidence?.code==='approach_obstructed'?evidence.objects?.filter(row=>row.object_id&&row.object_id!==this.objectId):null;
    if(!objects?.length)return null;
    // Closing clears the live approach monitor. Retain its confirmed evidence
    // only as a candidate until physical lift verifies whether the grasp worked.
    this.graspObstructionCandidate={...structuredClone(evidence),objects:structuredClone(objects),
      phase:this.phase,stage:this.stage,target_object_id:this.objectId,observed_time_s:nowOf(this.scene)};
    return this.graspObstructionCandidate;
  }
  failUnsecuredObstructedGrasp(){
    if(this.graspSuccess===true||this.stableLift>=.8||this.stableDeposit>=.8){this.graspObstructionCandidate=null;return false;}
    const candidate=this.graspObstructionCandidate;
    if(!candidate||candidate.target_object_id!==this.objectId)return false;
    this.graspSuccess=false;
    this.failObstructed({...candidate,grasp_validation:'not_secured',failure_phase:this.phase,failure_stage:this.stage});
    return true;
  }
  palm(side=this.hand,data=this.scene.data){return bodyPose(data,this.scene.palmBodyIds[side]);}
  plannedPalms(){const f=this.reference.currentFrame();return Object.fromEntries(SIDES.map((s,i)=>[s,{position:copy(f.palmPosW[i]),rotation:quatToMat(f.palmQuatW[i])}])) ;}
  executionIK(q,target,options={}){
    const result=this.reference.ik.solve(q,target,{...options,bestEffort:true});
    if(result.passed===false)this.markCompletionFallback('approximate_ik');
    return result;
  }
  carriedPayloadPrediction(){
    if(!['lift','hold','stand','transfer'].includes(this.phase)||!(this.activeHandContacts>0)||!this.scene.objects[this.objectId]?.active)return null;
    // The live body and object remain exclusively physics-driven. Only the IK
    // scratch scene predicts where the measured grasp would carry the object.
    const palm=this.palm(),object=poseOf(this.scene,this.objectId),inverse=transpose3(palm.rotation);
    return {objectId:this.objectId,hand:this.hand,
      positionInPalm:matVec(inverse,sub(object.position,palm.position)),
      rotationInPalm:matMul(inverse,quatToMat(object.quaternion)),
      measuredPositionW:copy(object.position),
      measuredRotationW:quatToMat(object.quaternion),
      allowTableContact:this.phase==='lift'&&this.lift<.025,
      allowTraySupport:this.phase==='transfer'&&this.transferSegments?.[this.transferIndex]?.name==='lower_load'};
  }
  updatePayloadPredictionMode(){
    const contact=!!this.carriedPayloadPrediction(),now=nowOf(this.scene);
    if(contact)this.payloadContactSince??=now;else this.payloadContactSince=null;
    // An isolated fingertip touch must not repeatedly replace the IK collision
    // model. Require 60 ms of contact (or an already measured opposing grip)
    // before enabling it. Contact loss still disables prediction immediately:
    // never keep carrying a detached object in the scratch scene.
    const enabled=contact&&(!!this.payloadPredictionEnabled||this.gripContactTime>=.06||now-this.payloadContactSince>=.06-1e-9);
    if(enabled!==!!this.payloadPredictionEnabled){
      (this.payloadPredictionTransitions??=[]).push({time_s:nowOf(this.scene),phase:this.phase,enabled,active_hand_contacts:this.activeHandContacts});
      this.payloadPredictionEnabled=enabled;this.forceReference=true;
    }
  }
  baseTarget(palms,{posture=this.restQ,rootHeight=this.rootHeight,rootPitch=this.rootPitch,rootPosition=this.rootPosition,allowHandObject=false,bodyRelativeHands=[],...extra}={}){return {hand:this.hand,objectId:this.objectId,palms,posture:copy(posture),rootHeight,rootPitch,rootPosition:copy(rootPosition),allowHandObject,bodyRelativeHands,...(this.releaseExit?.collisionFrame&&['release','settle'].includes(this.phase)?{releaseCollisionFrame:this.releaseExit.collisionFrame,lockLowerBody:true}:{}),...(this.ignoredObjectId&&['release','settle','return_home'].includes(this.phase)?{ignoreObjectId:this.objectId}:{}),carriedPayload:this.payloadPredictionEnabled?this.carriedPayloadPrediction():null,...extra};}
  oneHandTarget(position,rotation,options={}){
    const other=otherHand(this.hand),posture=options.posture||this.restQ,height=options.rootHeight??this.rootHeight,pitch=options.rootPitch??this.rootPitch,rootPosition=options.rootPosition??this.rootPosition;
    let inactive=this.homePalms[other];
    if(!this.crouched){
      // Native bilateral IK derives the inactive goal from the requested rest
      // posture. Holding the reset world palm instead would undo elbow tucking
      // and let accumulated physical root drift carry its fingers into the table.
      const key=[...rootPosition,height,pitch,...posture].join(',');
      if(this.restPalmCache?.key!==key){
        const frame=this.reference.fkFrame(posture,{rootPosition,rootHeight:height,rootPitch:pitch}),torsoRotation=quatToMat(frame.bodyQuatW[17]);
        // Retain the same modest outboard allowance used by the relaxed-arm
        // crouch. It leaves room for native tracking error around the hip shells.
        this.restPalmCache={key,palms:Object.fromEntries(SIDES.map((s,i)=>[s,{position:add(copy(frame.palmPosW[i]),matVec(torsoRotation,[0,s==='left'?.05:-.05,0])),rotation:quatToMat(frame.palmQuatW[i])}]))};
      }
      inactive=this.restPalmCache.palms[other];
    }
    const palms={[this.hand]:{position,rotation},[other]:inactive};
    return this.baseTarget(palms,{bodyRelativeHands:this.crouched?[other]:[],palmOffsetsTorso:this.crouchOffsets,...options});
  }
  async prepare(options={}){
    if(this.phase!=='idle')throw new Error('Reset the scene before starting another grasp.');
    const active=Object.entries(this.scene.objects).filter(([,o])=>o.active);if(!active.length)throw new Error('Place an object before starting a grasp.');
    const objectId=options.objectId||options.object_id||this.scene.selectedObjectId||this.config.objectId||this.config.object_id||(active.length===1?active[0][0]:null);
    if(!objectId)throw new Error('Select which object to pick.');
    const pose=this.preview({...options,objectId});if(!pose)throw new Error('Place the selected object on the table first.');
    this.objectId=pose.objectId;this.hand=pose.hand;this.plan=pose;this.initialObjectZ=pose.object_position_w[2];this.objectAnchor=pose.object_position_w.slice();this.graspGoal=pose.position_w.slice();this.grasp=pose.position_w.slice();this.pregrasp=pose.approach_start_w.slice();this.liftTarget=pose.lift_position_w.slice();this.graspRotation=pose.rotation_w.slice();
    this.startTime=nowOf(this.scene);this.enter('planning','Checking planted-foot reach and collision clearance.');
    const inactive=this.hand==='right'?15:22;this.restQ.splice(inactive,7,.2,this.hand==='right'?.2:-.2,0,this.objectId==='bottle'?.6:.9,0,0,0);
    try{
      this.tableFront=geomBounds(this.scene,this.scene.data,this.scene.tableGeomIds).lower[0];
      for(const side of SIDES){const bound=this.audit.handBounds(this.scene.data,side);this.crouchOffsets[side][0]=-Math.max(0,bound.upper[0]-(this.tableFront-.10));}
      let stance=await planReachStance(this.reference.ik,this.reachStanceTarget(),{onProgress:(a,b)=>this.message=`Checking reach posture ${a}/${b}.`});
      if(!stance.feasible){
        // A reachability estimate is guidance for the demo, not permission to
        // execute. Continue with the closest candidate and record its residual.
        const cost=c=>{const r=c.residual||{},p=r.palms?.[this.hand];return (p?.positionError??1)+.1*(p?.rotationError??1)+10*(r.footPositionError??1)+10*Math.max(0,.02-(r.comMargin??0));};
        const candidate=stance.candidates?.filter(c=>!c.excluded).sort((a,b)=>cost(a)-cost(b))[0];
        stance={...candidate,feasible:false,bestEffort:true,rootPosition:candidate?.rootPosition?.slice()||this.rootPosition.slice(),rootHeight:candidate?.rootHeight??this.rootHeight,rootPitch:candidate?.rootPitch??0,waistPitchTarget:candidate?.waistPitchTarget??0,evaluated:stance.evaluated};
        this.markCompletionFallback('reach_ik_best_effort');
      }
      this.stanceHistory.push({reason:'initial',...stance});
      await this.beginStanceTransition(stance);
    }catch(error){if(!(error instanceof IKError))throw error;this.markCompletionFallback('reach_plan_retry');this.coordinatedReach=null;this.buildApproach();}
    return this.snapshot();
  }
  reachStanceTarget(){
    return this.oneHandTarget(this.graspGoal,this.graspRotation,{posture:this.homeQ,
      bodyRelativeHands:[otherHand(this.hand)],palmOffsetsTorso:this.crouchOffsets,allowHandObject:true,allowHandTableContact:this.plan?.approach==='side'});
  }
  reachRestPosture(stance){
    const posture=this.restQ.slice(),waist=stance.feasible&&stance.posture?.slice(12,15);
    // The seed is only a soft prior during stance search. A feasible neutral
    // seed can require a substantial solved bend; execute that accepted waist
    // pose instead of applying the stronger reach prior to the original seed.
    if(waist?.length===3&&waist.every(Number.isFinite))posture.splice(12,3,...waist);
    else posture[14]=stance.waistPitchTarget;
    return posture;
  }
  async beginStanceTransition(stance){
    if(!this.stanceReplanCount){await this.beginCoordinatedReach(stance);return;}
    this.coordinatedReach=null;
    const frame=this.reference.currentFrame();
    this.stancePlan=stance;this.stanceFromRoot=frame.rootPosW.slice();this.stanceFromPitch=frame.plannedRootPitch;
    this.crouchHeight=stance.rootHeight;this.crouchPitch=stance.rootPitch;this.stanceTargetRoot=stance.rootPosition.slice();
    // Carry the solved waist pose through the relaxed transition, keeping the
    // reaching arm free and preserving the inactive-arm tuck. The waist stays
    // free in IK; this is the prior for the accepted stance, not a joint lock.
    this.stanceRestPosture=this.reachRestPosture(stance);
    this.crouchPosture=this.homeQ.slice();this.crouchPosture.splice(12,3,...this.stanceRestPosture.slice(12,15));
    this.crouchPostureFrom=this.homeQ.slice();this.crouchPostureFrom.splice(12,3,...frame.jointPos.slice(12,15));
    this.crouchFrom=this.plannedPalms();
    const waistChange=Math.max(...this.crouchPosture.slice(12,15).map((q,i)=>Math.abs(q-this.crouchPostureFrom[12+i])));
    const changed=norm(sub(this.stanceTargetRoot,this.stanceFromRoot))>.002||Math.abs(this.crouchPitch-this.stanceFromPitch)>.015||waistChange>.015;
    if(!changed){this.rootPosition=stance.rootPosition.slice();this.rootHeight=stance.rootHeight;this.rootPitch=stance.rootPitch;this.restQ=this.stanceRestPosture.slice();this.buildApproach();return;}
    const f=this.reference.fkFrame(this.crouchPostureFrom,{rootPosition:this.stanceFromRoot,rootHeight:this.stanceFromRoot[2],rootPitch:this.stanceFromPitch});
    const torso=quatToMat(f.bodyQuatW[17]);
    this.relaxedStartGoals=Object.fromEntries(SIDES.map((side,i)=>[side,{position:add(f.palmPosW[i],matVec(torso,this.crouchOffsets[side])),rotation:quatToMat(f.palmQuatW[i])}]));
    this.stanceDuration=Math.max(1.8,norm(sub(this.stanceTargetRoot,this.stanceFromRoot))/.035,Math.abs(this.crouchPitch-this.stanceFromPitch)/.15,waistChange/.25);
    // Audit entry while both palms are still under Cartesian control, then
    // let them follow the torso naturally as the supported pelvis moves.
    let q=this.reference.qCurrent.slice();const duration=1.6+this.stanceDuration,steps=Math.ceil(duration/this.dt);
    for(let i=1;i<=steps;i++){
      const t=i*this.dt,preparing=t<1.6,target=this.relaxedTarget(preparing?0:smooth((t-1.6)/this.stanceDuration),preparing?smooth(t/1.6):1);
      q=this.executionIK(q,target,{iterations:16,maxStep:.035,strictEndpoint:i===steps}).q;
      if(i%15===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    this.stage='prepare_crouch';this.stableBody=0;this.enter('crouch','Adjusting the body posture for a farther reach before approaching again (one retry).');
  }
  relaxedTarget(alpha,offsetAlpha=1){
    const root=lerp(this.stanceFromRoot,this.stanceTargetRoot,alpha),pitch=this.stanceFromPitch+alpha*(this.crouchPitch-this.stanceFromPitch);
    const palms=Object.fromEntries(SIDES.map(s=>[s,{position:lerp(this.crouchFrom[s].position,this.relaxedStartGoals[s].position,offsetAlpha),rotation:rotationBlend(this.crouchFrom[s].rotation,this.relaxedStartGoals[s].rotation,offsetAlpha)}]));
    return this.baseTarget(palms,{posture:lerp(this.crouchPostureFrom,this.crouchPosture,alpha),rootPosition:root,rootHeight:root[2],rootPitch:pitch,
      bodyRelativeHands:offsetAlpha>=1?SIDES:[],palmOffsetsTorso:this.crouchOffsets});
  }
  async beginCoordinatedReach(stance){
    this.finalApproachDeadline=null;this.nearGoalAttempt=null;
    const frame=this.reference.currentFrame(),arm=this.hand==='left'?15:22;
    this.stancePlan=stance;this.stanceRestPosture=this.reachRestPosture(stance);
    this.rootPosition=stance.rootPosition.slice();this.rootHeight=stance.rootHeight;this.rootPitch=stance.rootPitch;
    this.crouchHeight=stance.rootHeight;this.crouchPitch=stance.rootPitch;
    this.crouched=norm(sub(this.rootPosition,this.reference.ik.anchorPosition))>.002||Math.abs(this.rootPitch)>.015||this.stanceRestPosture.slice(12,15).some((q,i)=>Math.abs(q-this.homeQ[12+i])>.015);
    this.restQ=this.stanceRestPosture.slice();
    this.coordinatedReach={fromRoot:frame.rootPosW.slice(),fromPitch:frame.plannedRootPitch,fromQ:frame.jointPos.slice(),
      arm,foldQ:this.homeQ.slice(),foldDuration:(this.tallHandFold()?HAND_FOLD_DURATION_SCALE[this.handModel]??1:1)*Math.max(2.8,norm(sub(this.rootPosition,frame.rootPosW))/.04,Math.abs(this.rootPitch-frame.plannedRootPitch)/.10)};
    const motion=this.coordinatedReach;
    motion.foldQ[14]=Math.min(stance.waistPitchTarget,8*Math.PI/180);
    motion.foldQ.splice(arm,7,.4,this.hand==='left'?.4:-.4,0,this.tallHandFold()?HAND_FOLD_ELBOW_RAD[this.handModel]:HAND_FOLD_ELBOW_RAD.dex3,0,0,0);
    let q=frame.jointPos.slice();const steps=Math.ceil(motion.foldDuration/this.dt);
    for(let i=1;i<=steps;i++){
      q=this.executionIK(q,this.coordinatedFoldTarget(smooth(i/steps)),{iterations:16,maxStep:.035,strictEndpoint:i===steps}).q;
      if(i%15===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    const fold=this.coordinatedFoldTarget(1).palms[this.hand],start=this.plannedPalms()[this.hand],actual=this.palm(),local=[];
    for(const g of this.audit.hands[this.hand])for(const point of geomCorners(this.scene,this.scene.data,g))local.push(matVec(transpose3(actual.rotation),sub(point,actual.position)));
    let lowest=Infinity;
    for(let i=0;i<=20;i++){const rotation=rotationBlend(fold.rotation,this.graspRotation,i/20);for(const point of local)lowest=Math.min(lowest,matVec(rotation,point)[2]);}
    this.safePalmZ=Math.max(fold.position[2],this.pregrasp[2],tableZ(this.scene)+.05-lowest);
    const high=this.pregrasp.slice();high[2]=this.safePalmZ;
    this.segments=[{name:'raise',position:fold.position.slice(),rotation:fold.rotation.slice(),duration:motion.foldDuration,gated:true}];
    if(this.safePalmZ>fold.position[2]+.005){this.segments[0].gated=false;this.segments.push({name:'clearance',position:[fold.position[0],fold.position[1],this.safePalmZ],rotation:fold.rotation.slice(),duration:Math.max(.8,(this.safePalmZ-fold.position[2])/.10),gated:true});}
    this.segments.push({name:'reach',position:high,rotation:this.graspRotation,duration:2.8});
    if(this.plan.approach==='side')this.segments.push({name:'descend',position:this.pregrasp.slice(),rotation:this.graspRotation,duration:1},
      {name:'enter_side',position:this.grasp.slice(),rotation:this.graspRotation,duration:1.7});
    else this.segments.push({name:'descend',position:this.grasp.slice(),rotation:this.graspRotation,duration:2});
    this.segments.push({name:'settle',position:this.grasp.slice(),rotation:this.graspRotation,duration:.4});
    this.approachRate=1;this.graspReadyTime=0;this.graspWaitBest=Infinity;this.graspProgressTime=nowOf(this.scene);
    this.segmentIndex=0;this.segmentFrom=start;this.segmentTime=nowOf(this.scene);this.stage='raise';
    this.enter('approach','Reaching with a coordinated waist and arm motion.');
  }
  coordinatedFoldTarget(alpha){
    const motion=this.coordinatedReach,beta=alpha**3,root=lerp(motion.fromRoot,this.rootPosition,beta),pitch=motion.fromPitch+beta*(this.rootPitch-motion.fromPitch);
    const posture=lerp(motion.fromQ,motion.foldQ,alpha);
    for(let i=12;i<15;i++)posture[i]=motion.fromQ[i]+beta*(motion.foldQ[i]-motion.fromQ[i]);
    posture[motion.arm]+=.3*Math.sin(Math.PI*alpha);
    const frame=this.reference.fkFrame(posture,{rootPosition:root,rootHeight:root[2],rootPitch:pitch}),index=this.hand==='left'?0:1;
    return this.oneHandTarget(frame.palmPosW[index].slice(),quatToMat(frame.palmQuatW[index]),{posture,rootPosition:root,rootHeight:root[2],rootPitch:pitch,
      bodyRelativeHands:[otherHand(this.hand)],palmOffsetsTorso:Object.fromEntries(SIDES.map(side=>[side,scale(this.crouchOffsets[side],alpha)])),
      waistPostureWeights:[1,1,.7]});
  }
  buildApproach(){
    // A fresh approach after a supported-stance recovery gets its own final
    // alignment window. An earlier attempt's clock must not expire in transit.
    this.finalApproachDeadline=null;this.nearGoalAttempt=null;
    this.approachRate=1;
    this.graspReadyTime=0;this.graspWaitBest=Infinity;this.graspProgressTime=nowOf(this.scene);
    const start=this.plannedPalms()[this.hand],actual=this.palm();let local=[];
    for(const g of this.audit.hands[this.hand])for(const p of geomCorners(this.scene,this.scene.data,g))local.push(matVec(transpose3(actual.rotation),sub(p,actual.position)));
    let lowest=Infinity;for(let i=0;i<=20;i++){const r=rotationBlend(start.rotation,this.graspRotation,i/20);for(const p of local)lowest=Math.min(lowest,matVec(r,p)[2]);}
    this.safePalmZ=Math.max(start.position[2]+.035,this.pregrasp[2],tableZ(this.scene)+.05-lowest);const raised=start.position.slice();raised[2]=this.safePalmZ;
    this.segments=[{name:'raise',position:raised.slice(),rotation:start.rotation,duration:1.6,gated:true},{name:'orient',position:raised.slice(),rotation:this.graspRotation,duration:1,gated:true}];
    if(this.plan.approach==='side'){
      for(const [name,point,duration]of [['retract',this.plan.retract_position_w,1.2],['side',this.plan.side_position_w,1.3],['reach',this.pregrasp,1.7]]){const high=point.slice();high[2]=this.safePalmZ;this.segments.push({name,position:high,rotation:this.graspRotation,duration});}
      this.segments.push({name:'descend',position:this.pregrasp.slice(),rotation:this.graspRotation,duration:1.2},{name:'enter_side',position:this.grasp.slice(),rotation:this.graspRotation,duration:1.7});
    }else{const high=this.pregrasp.slice();high[2]=this.safePalmZ;this.segments.push({name:'reach',position:high,rotation:this.graspRotation,duration:2},{name:'descend',position:this.grasp.slice(),rotation:this.graspRotation,duration:2});}
    this.segments.push({name:'settle',position:this.grasp.slice(),rotation:this.graspRotation,duration:.4});this.segmentIndex=0;this.segmentFrom=start;this.segmentTime=nowOf(this.scene);this.stage='raise';this.enter('approach','Lifting the open hand clear of the tabletop.');
  }
  graspCommandOffset(offset=0){
    const transition=this.gaGraspTransition;if(!transition)return [0,0,0];
    const t=clamp((nowOf(this.scene)+Math.max(0,offset)-transition.startTime)/transition.duration,0,1);
    const blend=t*t*t*(10+t*(-15+6*t));
    return scale(transition.fromOffset,1-blend);
  }
  graspCommandRotation(offset=0){
    const retry=this.alignmentRetry,transition=retry?.rotationTransition;
    if(!transition)return retry?.commandRotation??this.graspRotation;
    return rotationBlend(transition.from,transition.to,minimumJerk((nowOf(this.scene)+Math.max(0,offset)-transition.time)/.4));
  }
  approachTarget(offset=0){
    if(this.alignmentRetry&&!this.alignmentRetry.prepared&&this.phase==='approach')return structuredClone(this.alignmentRetry.holdTarget);
    if(this.forcedClose&&this.phase==='close')return structuredClone(this.forcedCloseTarget);
    let elapsed=nowOf(this.scene)-this.segmentTime+Math.max(0,offset)*(this.approachRate??1),origin=this.segmentFrom.position,orientation=this.segmentFrom.rotation,result;
    for(let i=this.segmentIndex;i<this.segments.length;i++){
      // Keep the prepared waist steady while lifting and orienting the open
      // hand. Otherwise IK can replace arm lift with extreme waist roll that
      // the policy cannot track. Restore full-body freedom for forward reach.
      const seg=this.segments[i],alpha=smooth(elapsed/seg.duration);
      if(this.coordinatedReach&&seg.name==='raise')result=this.coordinatedFoldTarget(alpha);
      else{
        const options={allowHandObject:['descend','enter_side','settle','back_off'].includes(seg.name),lockWaist:['raise','orient','clearance'].includes(seg.name)};
        options.allowHandTableContact=this.plan?.approach==='side'&&options.allowHandObject;
        if(this.plan?.approach==='top_down'&&['descend','settle','back_off'].includes(seg.name))options.handTableAllowance=this.topDownTableAllowance();
        if(this.coordinatedReach){const from=this.coordinatedReach.reachWaistPostureFrom??this.coordinatedReach.foldQ.slice(12,15);options.posture=this.restQ.slice();options.posture.splice(12,3,...lerp(from,this.restQ.slice(12,15),seg.name==='reach'?alpha:seg.name==='clearance'?0:1));options.waistPostureWeights=[.6,.8,.4];}
        const position=seg.name==='settle'&&this.gaGraspTransition?add(seg.position,this.graspCommandOffset(offset)):seg.position;
        const rotation=seg.name==='settle'&&this.alignmentRetry?.prepared?this.graspCommandRotation(offset):seg.rotation;
        result=this.oneHandTarget(lerp(origin,position,alpha),rotationBlend(orientation,rotation,alpha),options);
      }
      if(elapsed<=seg.duration||seg.gated&&this.audit.handTableClearance(this.scene.data,this.hand)<.025)break;
      elapsed-=seg.duration;origin=seg.position;orientation=seg.rotation;
    }return result;
  }
  graspDistance(){return this.graspAlignment()?.distance_m??null;}
  graspFeedback(alignment=this.graspAlignment()){
    return this.handModel==='inspire'?{distance_m:alignment.center_distance_m,error_w:alignment.center_error_w}:
      {distance_m:alignment.distance_m,error_w:alignment.position_error_w};
  }
  closureCenterLimit(palmLimit=this.gates.forced){
    if(this.handModel!=='inspire'||palmLimit===null)return null;
    // The normal palm gate may include its existing 2.5 mm aperture drift
    // allowance; that never widens the normal center gate.
    return palmLimit<this.gates.forced?INSPIRE_CENTER_CLOSURE_GATES.normal:INSPIRE_CENTER_CLOSURE_GATES.forced;
  }
  graspAlignment(){
    if(!this.graspGoal)return null;
    const palm=this.palm(),target=poseOf(this.scene,this.objectId).position;
    // The policy tracks the palm-body origin, ~9 cm behind the grasp center.
    // Evaluate that calibrated center with the *measured* wrist rotation:
    // retain it separately from the displayed EE pose tolerance. Hand-specific
    // calibration may also use it for admission and translation feedback.
    const local=matVec(transpose3(this.graspRotation),sub(this.objectAnchor,this.graspGoal));
    const center=add(palm.position,matVec(palm.rotation,local)),error=sub(center,target);
    // Translate the calibrated EE target with the measured object, never with
    // GA compensation. This retains the requested palm tolerance if contact
    // moves the object, without double-counting wrist rotation through a
    // virtual wrist-to-COM lever arm. Dex3 keeps the COM estimate diagnostic.
    const palmTarget=add(this.graspGoal,sub(target,this.objectAnchor)),palmError=sub(palm.position,palmTarget);
    const distance=norm(palmError),centerDistance=norm(error),angle=norm(rotationError(this.graspRotation,palm.rotation));
    const alignment={distance_m:distance,position_w:palm.position.slice(),target_w:palmTarget,position_error_w:palmError,fixed_target_w:this.graspGoal.slice(),fixed_distance_m:norm(sub(palm.position,this.graspGoal)),
      center_position_w:center,center_target_w:target,center_error_w:error,center_distance_m:centerDistance,angle_error_rad:angle};
    return {...alignment,ready:this.closureAllowed(alignment,this.gates.normal)};
  }
  closureAllowed(alignment=this.graspAlignment(),limit=this.gates.forced){
    const centerLimit=this.closureCenterLimit(limit);
    return !!alignment&&[alignment.distance_m,alignment.center_distance_m,alignment.angle_error_rad].every(Number.isFinite)&&
      alignment.distance_m<=limit+1e-12&&(centerLimit===null||alignment.center_distance_m<=centerLimit+1e-12)&&alignment.angle_error_rad<=this.closureAngleLimit+1e-12;
  }
  recordClosureGate(allowed,kind,alignment,limit,reason=null){
    const centerLimit=this.closureCenterLimit(limit);
    this.lastClosureGate={allowed,kind,palm_limit_m:limit,center_limit_m:centerLimit,center_diagnostic_only:centerLimit===null,angle_limit_rad:this.closureAngleLimit,final_attempt:this.finalAttempt||false,reason,time_s:nowOf(this.scene),
      palm_distance_m:alignment?.distance_m??null,center_distance_m:alignment?.center_distance_m??null,angle_error_rad:alignment?.angle_error_rad??null};
  }
  beginClose(alignment=this.graspAlignment()){
    alignment=this.graspAlignment();const allowed=this.closureAllowed(alignment,this.gates.normal);
    this.recordClosureGate(allowed,'normal',alignment,this.gates.normal,'admission');if(!allowed)return false;
    this.rememberGraspObstruction();
    this.closeTrigger={...alignment,gate:'normal_pose',stage:this.stage,time_s:nowOf(this.scene)-this.startTime};
    this.closeProgress=0;this.gripContactTime=0;
    // Preserve the continuous approach; actual alignment and finger contacts
    // govern closure and lift, rather than predicting lift on a fixed timer.
    const remaining=this.segments.slice(this.segmentIndex).reduce((sum,seg)=>sum+seg.duration,0)-(nowOf(this.scene)-this.segmentTime);
    this.closeDuration=Math.max(1.8,remaining);
    this.closeDeadline=nowOf(this.scene)+this.closeDuration+6;
    this.enter('close','Grasp pose aligned. Closing the fingers while completing the approach.');
    return true;
  }
  markCompletionFallback(reason){
    this.completionFallback=true;this.completionReasons??=[];
    if(!this.completionReasons.includes(reason))this.completionReasons.push(reason);
  }
  recordPostReleaseHazard(reason,details={}){
    // After the hand opens, clearance shortfalls, unplanned exits and stage timeouts are recorded and answered by
    // the bounded exit/return machinery; they never stop the episode. Only the physical guards in tick() do.
    this.markCompletionFallback(reason);
    const pair=details.geometry?.limitingPair??details.limitingPair??null,log=this.postReleaseHazards??=[],last=log.at(-1);
    const row={reason,phase:this.phase,stage:this.stage,time_s:nowOf(this.scene)-(this.startTime??0),count:1,limiting_pair:pair?{a:pair.a??null,b:pair.b??null,distance_m:pair.distance??null,required_margin_m:pair.requiredMargin??null}:null,...(details.note?{note:String(details.note)}:{})};
    if(last&&last.reason===reason&&last.phase===row.phase&&last.stage===row.stage){last.count++;last.last_time_s=row.time_s;last.limiting_pair=row.limiting_pair??last.limiting_pair;}
    else if(log.length<POST_RELEASE_HAZARD_LOG_MAX)log.push(row);
    if(this.releaseExit)this.releaseExit.hazardCount=(this.releaseExit.hazardCount??0)+1;
  }
  beginForcedClose(reason,issuedTarget=null){
    const alignment=this.graspAlignment();
    this.recordClosureGate(this.closureAllowed(alignment),'forced',alignment,this.gates.forced,reason);
    const obstruction=this.rememberGraspObstruction();
    // A measured obstacle that still blocks the exhausted approach ends
    // the attempt before grasping or transfer.
    // A blocked GA compensation target can coexist with a physically close
    // hand. Preserve the existing near-object grasp attempt in that case.
    if(['alignment_timeout','near_goal_stagnation'].includes(reason)&&alignment.distance_m>NEAR_GRASP_REFINEMENT_M&&obstruction){
      this.failObstructed(obstruction);return;
    }
    // An admitted finger ramp is physically committed: finishing it (grip_timeout) is never re-gated on alignment;
    // only a fallen object still aborts it.
    if(!this.closureAllowed(alignment)&&(reason==='object_fell'||!this.closureCommitted)){
      this.closureRejectedTrigger={...alignment,reason,time_s:nowOf(this.scene)-this.startTime};
      if(reason==='object_fell')this.beginGraspAbort(reason);else this.beginAlignmentRetry(reason);
      return;
    }
    if(this.forcedClose)return;
    const target=issuedTarget??structuredClone(this.approachTarget());
    const gate=reason==='near_goal_stagnation'?'near_goal_stagnation':'completion_fallback',nearAttempt=gate==='near_goal_stagnation'?this.nearGoalAttemptStatus():null;
    this.markCompletionFallback(reason);this.forcedClose=true;this.forcedCloseTarget=target;
    this.forcedCloseTime=nowOf(this.scene);this.forcedCloseTrigger={...alignment,reason,gate,near_goal_attempt:nearAttempt,time_s:nowOf(this.scene)-this.startTime};
    this.closeTrigger??={...alignment,gate,stage:this.stage,time_s:nowOf(this.scene)-this.startTime,forced:true};
    this.closeProgress??=0;this.gaGraspTransition=null;this.gaFrozenReason='attempt_budget';
    this.liftTarget=add(target.palms[this.hand].position,[-.015,0,.13]);
    this.enter('close','Completing the grasp attempt before lifting and placing.');
  }
  beginAlignmentRetry(reason){
    const now=nowOf(this.scene);
    if(this.alignmentRetry&&this.phase==='approach'){
      const retry=this.alignmentRetry;
      // Phase 1 (hold) spent with the gates still unmet: start the second approach. Phase 2 run its course (its settle
      // timed out or lost the closure gate, or its budget is spent): no third approach -- grasp from the closest
      // reachable pose, or abandon beyond 60 mm.
      if(retry.phase!=='approach'){if(now>=retry.deadline)this.beginSecondApproach(reason);return;}
      if(now>=retry.deadline||this.stage==='settle')this.beginFinalAttempt(reason);
      return;
    }
    if((this.alignmentRetryCount??0)>=1){this.beginFinalAttempt(reason);return;}
    const holdTarget=structuredClone(this.approachTarget()),closure=smooth(this.closeProgress??0);
    this.alignmentRetryCount=(this.alignmentRetryCount??0)+1;
    this.alignmentRetry={reason,phase:'hold',startedAt:now,openingUntil:now+.8,deadline:now+GRASP_ALIGNMENT_RETRY_S,closure,holdTarget,prepared:false,translationM:0,rotationRad:0,corrections:[],commandRotation:holdTarget.palms[this.hand].rotation.slice()};
    this.forcedClose=false;this.forcedCloseTarget=null;this.closureCommitted=null;this.closeProgress=0;this.closeTrigger=null;this.closeDeadline=null;
    this.graspReadyTime=0;this.nearGoalAttempt=null;this.gripContactTime=0;
    this.attemptNotice='Re-aligning the open hand before a second approach (one retry).';
    this.enter('approach',this.attemptNotice);this.stage='settle';
  }
  beginFinalAttempt(reason){
    // The exhausted retry does not give up. Grasp from the closest reachable pose while the palm is within
    // FINAL_ATTEMPT_MAX_M; only a hand so far away that the fingers cannot touch the object abandons the attempt.
    if(this.finalAttempt){if(!this.forcedClose&&!this.closureCommitted)this.beginGraspAbort('alignment_retry_exhausted');return;}
    const alignment=this.graspAlignment();
    if(!(alignment.distance_m<=FINAL_ATTEMPT_MAX_M+1e-12)){this.beginGraspAbort('alignment_retry_exhausted');return;}
    // Snapshot the currently issued command (with the retry's accumulated wrist correction) before dropping the retry,
    // so the held forced-close target does not jump back to the nominal grasp rotation.
    const issued=this.alignmentRetry?structuredClone(this.approachTarget()):null;
    this.finalAttempt=true;this.alignmentRetry=null;this.markCompletionFallback('final_attempt_close');
    this.finalAttemptTrigger={...alignment,reason,time_s:nowOf(this.scene)-(this.startTime??0)};
    this.attemptNotice='Grasping from the closest reachable pose (final attempt).';
    this.beginForcedClose('final_attempt',issued);
  }
  prepareAlignmentRetry(){
    const retry=this.alignmentRetry,now=nowOf(this.scene);if(!retry||retry.prepared||now<retry.openingUntil)return;
    // Retarget only after opening. A moved object is never teleported back to
    // the old anchor, and a large displacement requires a fresh user attempt.
    const position=poseOf(this.scene,this.objectId).position,delta=sub(position,this.objectAnchor);
    if(norm(delta)>.05||position[2]<tableZ(this.scene)-.10){this.beginGraspAbort('object_moved_out_of_grasp');return;}
    this.graspGoal=add(this.graspGoal,delta);this.grasp=add(this.grasp,delta);this.liftTarget=add(this.liftTarget,delta);this.pregrasp=this.pregrasp?add(this.pregrasp,delta):this.pregrasp;this.objectAnchor=position.slice();
    // Phase 1 (hold): keep the accepted, GA-compensated command and let the bounded refinement work for the hold budget.
    this.segments=[{name:'settle',position:this.grasp.slice(),rotation:this.graspRotation.slice(),duration:.6}];
    this.segmentIndex=0;this.segmentFrom=structuredClone(retry.holdTarget.palms[this.hand]);this.segmentTime=now;
    this.goalAdjuster=new GoalAdjuster(this.graspGoal,{enabled:this.goalAdjustEnabled,stay:this.gates.normal*.8,skip:this.gates.normal,maxStep:.002,maxReplans:6});
    this.goalHistory=[];this.gaFrozenReason=null;this.gaGraspTransition=null;this.nearGraspRefinement=null;this.nextReplan=0;
    this.graspWaitBest=Infinity;this.graspProgressTime=now;retry.prepared=true;retry.refineUntil=now+2;retry.nextRefine=now;this.forceReference=true;this.reseed=false;
  }
  beginSecondApproach(reason){
    // Phase 2 of the single retry: the hold ended with the gates still unmet, so the hand visibly tries again -- back the
    // open palm out along the entry direction (toward the pre-grasp point, at most RETRY_BACKOFF_M), re-enter with the
    // same segment kind as the first approach and settle again under the base gates.
    const retry=this.alignmentRetry,now=nowOf(this.scene);if(!retry||retry.phase==='approach'||this.phase!=='approach')return;
    const from=structuredClone(this.approachTarget().palms[this.hand]);
    // Rebuild from the nominal grasp (as the supported-stance recovery does): the first attempt's GA compensation -- a
    // command up to 10 cm beyond an unreachable target -- is dropped so the back-off is a real retreat, and the ordinary
    // GA loop recompensates from scratch during the second settle (closedLoopUpdate).
    const liftOffset=sub(this.liftTarget,this.grasp),preOffset=sub(this.pregrasp??add(this.grasp,[-.12,0,0]),this.graspGoal);
    this.grasp=this.graspGoal.slice();this.liftTarget=add(this.grasp,liftOffset);this.pregrasp=add(this.grasp,preOffset);
    this.goalAdjustment=[0,0,0];this.goalAdjuster=null;this.gaGraspTransition=null;this.goalHistory=[];this.gaFrozenReason=null;this.nextReplan=0;this.nearGraspRefinement=null;
    const entry=sub(this.pregrasp,this.grasp),entryLength=norm(entry);
    const backoff=entryLength>1e-6?scale(entry,Math.min(RETRY_BACKOFF_M,entryLength)/entryLength):[-RETRY_BACKOFF_M,0,0];
    const reenter=this.plan?.approach==='top_down'?'descend':'enter_side';
    this.segments=[{name:'back_off',position:add(this.grasp,backoff),rotation:this.graspRotation.slice(),duration:RETRY_BACKOFF_S},
      {name:reenter,position:this.grasp.slice(),rotation:this.graspRotation.slice(),duration:RETRY_REENTER_S},
      {name:'settle',position:this.grasp.slice(),rotation:this.graspRotation.slice(),duration:.4}];
    this.segmentIndex=0;this.segmentFrom=from;this.segmentTime=now;this.stage='back_off';this.message='Backing the open hand away for a second approach.';
    // The second approach gets its own budget, final-alignment window, near-goal history and refinement window.
    retry.phase='approach';retry.secondReason=reason;retry.secondStartedAt=now;retry.deadline=now+GRASP_SECOND_APPROACH_S;retry.refineUntil=retry.deadline;retry.nextRefine=now;retry.backoff=backoff.slice();retry.reenter=reenter;
    this.finalApproachDeadline=null;this.nearGoalAttempt=null;this.graspReadyTime=0;this.graspWaitBest=Infinity;this.graspProgressTime=now;
    this.forcedClose=false;this.forcedCloseTarget=null;this.closeProgress=0;this.closeTrigger=null;this.closeDeadline=null;this.forceReference=true;this.reseed=false;
  }
  refineAlignmentRetry(){
    const retry=this.alignmentRetry,now=nowOf(this.scene);
    if(!this.replanEnabled||!retry?.prepared||this.phase!=='approach'||retry.phase==='approach'&&this.stage!=='settle'||now>=retry.refineUntil||now>=retry.deadline||now<retry.nextRefine)return;
    retry.nextRefine=now+.4;
    const alignment=this.graspAlignment(),palm=this.palm();
    if(alignment.ready)return;
    const remaining=Math.max(0,.008-retry.translationM),feedback=this.graspFeedback(alignment),errorVector=feedback.error_w;
    let shift=feedback.distance_m>this.gates.normal&&remaining>0?scale(errorVector,-Math.min(.5,.002/Math.max(norm(errorVector),1e-12),remaining/Math.max(norm(errorVector),1e-12))):[0,0,0];
    const error=rotationError(this.graspRotation,palm.rotation),angle=norm(error),rotationStep=angle>4*Math.PI/180?Math.min(angle*.5,Math.PI/180,Math.max(0,4*Math.PI/180-retry.rotationRad)):0;
    // Project the translation without discarding a valid wrist correction at
    // the position budget boundary. Retry step/cumulative budgets remain intact.
    const proposed=add(this.goalAdjustment,shift),length=norm(proposed);
    if(length>.10)shift=sub(scale(proposed,.10/length),this.goalAdjustment);
    if(norm(shift)<1e-12&&rotationStep<1e-12)return;
    const commandRotation=this.graspCommandRotation(),axis=scale(error,Math.sin(rotationStep/2)/Math.max(angle,1e-12)),rotation=matMul(quatToMat([Math.cos(rotationStep/2),...axis]),retry.commandRotation);
    const current=this.approachTarget(),target=this.oneHandTarget(add(this.grasp,shift),rotation,{posture:current.posture,waistPostureWeights:current.waistPostureWeights,allowHandObject:true,allowHandTableContact:this.plan?.approach==='side',handTableAllowance:this.topDownTableAllowance()});
    try{
      const checked=this.reference.ik.solve(this.reference.qCurrent,target,{iterations:60,maxStep:3,strictEndpoint:false,bestEffort:false});
      if(checked.passed===false){retry.rejected='reference_infeasible';return;}
    }catch(error){if(!(error instanceof IKError))throw error;retry.rejected='reference_infeasible';return;}
    this.gaGraspTransition={startTime:now,duration:.4,fromOffset:sub(this.graspCommandOffset(),shift)};
    this.grasp=add(this.grasp,shift);this.liftTarget=add(this.liftTarget,shift);this.segments.at(-1).position=this.grasp.slice();
    this.goalAdjustment=add(this.goalAdjustment,shift);retry.translationM+=norm(shift);retry.rotationRad+=rotationStep;
    retry.commandRotation=rotation;retry.rotationTransition={time:now,from:commandRotation,to:rotation};
    retry.corrections.push({time_s:now,translation_m:shift.slice(),rotation_rad:rotationStep});
    this.replans++;this.gaUpdates++;this.forceReference=true;this.reseed=false;
  }
  beginGraspAbort(reason){
    if(this.graspAbort)return;
    if(this.failUnsecuredObstructedGrasp())return;
    const closure=this.scene.handClosure?.[this.hand]??(this.alignmentRetry?this.alignmentRetry.closure*(1-smooth((nowOf(this.scene)-this.alignmentRetry.startedAt)/.6)):['lift','hold','stand','transfer'].includes(this.phase)?1:smooth(this.closeProgress??0));
    const palms=this.plannedPalms(),bounds=this.audit.handBounds(this.scene.data,this.hand);
    this.graspAbort={reason,phase:this.phase,stage:this.stage,startedAt:nowOf(this.scene),closure,palms,posture:this.reference.qCurrent.slice(),
      retreat:add(palms[this.hand].position,[0,0,Math.max(.025,Math.min(.06,tableZ(this.scene)+.015-bounds.lower[2]))])};
    this.graspSuccess=this.graspSuccess===true?true:false;this.success=null;
    this.failureReason??={code:'grasp_aborted',reason,phase:this.phase,stage:this.stage,target_object_id:this.objectId};
    this.forcedClose=false;this.forcedCloseTarget=null;this.alignmentRetry=null;
    this.markCompletionFallback(reason);this.enter('release','Opening the hand before returning.');this.stage='grasp_abort';
  }
  async advanceGraspAbort(){
    const elapsed=nowOf(this.scene)-this.graspAbort.startedAt;if(elapsed<.8)return;
    const geometry=this.audit.check(this.scene.data,{handsOnly:true,includeSelf:false,margin:.005});
    if(!(this.activeHandContacts>0)&&geometry.passed){await this.beginReturn();return;}
    if(elapsed>=3.2){this.failureMessage??='The open hand could not clear the object safely.';this.lastError={message:'The open hand could not clear the object safely.',details:{geometry,activeHandContacts:this.activeHandContacts}};this.markCompletionFallback('abort_clearance_timeout');await this.beginReturn();}
  }
  advanceClose(){
    const alignment=this.graspAlignment(),elapsed=nowOf(this.scene)-this.phaseTime;
    // Before the first actual positive command, enforce the admission gate.
    // Once admitted, complete one smooth finger ramp. Contact can move the
    // object and wrist, so reopening on that motion can interrupt acquisition.
    // Physical force/balance guards and measured lift verification stay active.
    if(!this.closureCommitted&&!(this.finalAttempt&&this.forcedClose)&&!this.closureAllowed(alignment)){this.recordClosureGate(false,this.forcedClose?'forced':'normal',alignment,this.gates.forced,'hard_alignment_lost');this.beginAlignmentRetry('closure_alignment_lost');return;}
    // First positive normal closure requires a 15 mm measured EE error.
    // Only an already-moving aperture gets the existing 2.5 mm drift allowance.
    // Finger contacts confirm lift; they never authorize closing out of bounds.
    const normalLimit=this.gates.normal+(this.closeProgress>0?.0025:0);
    const retainedGrip=this.closureCommitted||this.closeTrigger?.ready&&this.closureAllowed(alignment,normalLimit);
    const limit=this.forcedClose||retainedGrip?1:0;
    if(this.closureCommitted)this.recordClosureGate(true,'committed_close',alignment,null,'continuous_admitted_ramp');else this.recordClosureGate(!!limit,this.forcedClose?'forced':'normal',alignment,this.forcedClose?this.gates.forced:normalLimit,limit?'closing':'alignment_lost');
    this.closeProgress=clamp(this.closeProgress+clamp(limit-this.closeProgress,-this.dt/.6,this.dt/1.4),0,1);
    if(this.forcedClose){if(this.closeProgress>=1-1e-9&&nowOf(this.scene)-this.forcedCloseTime>=1.6)this.enter('lift','Lifting after the completed grasp attempt.');return;}
    if(elapsed>=this.closeDuration&&this.closeProgress>=1-1e-9&&this.gripContactTime>=.12){this.enter('lift','Lifting through physical finger contact.');return;}
    this.closeDeadline??=this.phaseTime+this.closeDuration+6;
    if(nowOf(this.scene)>this.closeDeadline)this.beginForcedClose('grip_timeout');
  }
  advanceApproach(){
    if(this.alignmentRetry){
      if(nowOf(this.scene)>=this.alignmentRetry.deadline){this.beginForcedClose('alignment_retry_timeout');return;}
      this.prepareAlignmentRetry();if(this.phase!=='approach'||!this.alignmentRetry.prepared)return;
    }
    const seg=this.segments[this.segmentIndex],elapsed=nowOf(this.scene)-this.segmentTime;this.stage=seg.name;
    const messages={raise:'Lifting the open hand clear of the tabletop.',clearance:'Keeping the open fingers clear through the wrist rotation.',orient:'Orienting the wrist at a safe height.',retract:'Retracting with the fingers above the table.',side:'Moving beside the object.',reach:'Reaching with the fingers clear of the tabletop.',descend:this.plan.approach==='side'?'Lowering beside the object.':'Descending vertically toward the object.',enter_side:'Approaching horizontally from the side.',back_off:'Backing the open hand away for a second approach.',settle:'Steadying the palm before closing the fingers.'};this.message=messages[seg.name];
    if(['retract','side','reach'].includes(seg.name)&&this.audit.handTableClearance(this.scene.data,this.hand)<.005){this.fail('The fingers lost tabletop clearance. Motion stopped.');return;}
    const finalApproach=seg.name==='settle'||seg.name===(this.plan.approach==='side'?'enter_side':'descend');
    const alignment=this.graspAlignment();
    if(finalApproach&&!this.alignmentRetry){
      this.finalApproachDeadline??=nowOf(this.scene)+GRASP_FINAL_APPROACH_BUDGET_S;
      if(nowOf(this.scene)>=this.finalApproachDeadline&&!alignment.ready){this.beginForcedClose('alignment_timeout');return;}
    }
    this.graspReadyTime=finalApproach&&alignment.ready?(this.graspReadyTime||0)+this.dt:0;
    const nearAttempt=this.updateNearGoalAttempt(alignment,finalApproach);
    if(this.graspReadyTime>=.08){this.beginClose(alignment);return;}
    // A normal alignment gets its uninterrupted 80 ms even if another
    // approach deadline expired on this tick. It always wins over fallback.
    if(finalApproach&&alignment.ready)return;
    if(nearAttempt?.criterion){this.beginForcedClose('near_goal_stagnation');return;}
    if(elapsed<seg.duration)return;
    if(seg.gated&&this.audit.handTableClearance(this.scene.data,this.hand)<.025){if(elapsed>seg.duration+3)this.fail('The hand could not reach safe clearance. Move the object closer.');return;}
    if(this.segmentIndex===this.segments.length-1){
      const feedbackDistance=this.graspFeedback(alignment).distance_m,error=Math.max(feedbackDistance,.08*alignment.angle_error_rad);
      if(error<(this.graspWaitBest??Infinity)-.002){this.graspWaitBest=error;this.graspProgressTime=nowOf(this.scene);}
      const progressing=nowOf(this.scene)-this.graspProgressTime<1.2;
      const refining=this.nearGraspRefinement&&nowOf(this.scene)-this.nearGraspRefinement.startTime<2&&feedbackDistance<=NEAR_GRASP_REFINEMENT_M&&alignment.angle_error_rad<=GRASP_ANGLE_TOLERANCE_RAD;
      if(nearAttempt||elapsed<GRASP_SETTLE_TIMEOUT_S||progressing&&elapsed<7.4||refining){this.message='Waiting for the measured hand position and wrist to align.';return;}
      this.beginForcedClose('alignment_timeout');return;
    }
    this.segmentFrom={position:seg.position.slice(),rotation:seg.rotation.slice()};this.segmentIndex++;this.segmentTime=nowOf(this.scene);this.forceReference=true;
  }
  updateNearGoalAttempt(alignment,finalApproach){
    const distance=this.graspFeedback(alignment).distance_m,needsAlignment=this.handModel==='inspire'?!alignment.ready:distance>this.gates.normal,
      eligible=this.phase==='approach'&&finalApproach&&needsAlignment&&this.closureAllowed(alignment);
    if(!eligible){this.nearGoalAttempt=null;return null;}
    const now=nowOf(this.scene);
    this.nearGoalAttempt??={startTime:now,lastProgressTime:now,progressDistanceM:distance,bestDistanceM:distance};
    const attempt=this.nearGoalAttempt;
    attempt.bestDistanceM=Math.min(attempt.bestDistanceM,distance);
    // Compare against the last meaningful improvement so smaller changes
    // accumulate toward one millimeter instead of perpetually resetting it.
    if(distance<=attempt.progressDistanceM-NEAR_GOAL_PROGRESS_M+1e-12){attempt.progressDistanceM=distance;attempt.lastProgressTime=now;}
    return this.nearGoalAttemptStatus();
  }
  nearGoalAttemptStatus(){
    const attempt=this.nearGoalAttempt;if(!attempt)return null;
    const now=nowOf(this.scene),dwell=now-attempt.startTime,stagnation=now-attempt.lastProgressTime;
    // The 4 s fallback must not freeze a still-converging GA ramp just outside
    // the normal gate. Give measured millimeter-scale progress at most 2 s
    // more; a real plateau, frozen feedback and the final-approach deadline
    // retain their existing stops. Merely waiting with GA enabled is not progress.
    const progressing=this.replanEnabled&&this.goalAdjustEnabled&&this.gaUpdates>0&&!this.gaFrozenReason&&attempt.lastProgressTime>attempt.startTime&&stagnation+1e-12<NEAR_GOAL_STAGNATION_S;
    const maximumDwell=progressing?NEAR_GOAL_PROGRESS_MAX_DWELL_S:NEAR_GOAL_MAX_DWELL_S;
    const criterion=dwell+1e-12>=maximumDwell?'dwell_cap':dwell+1e-12>=NEAR_GOAL_MIN_DWELL_S&&stagnation+1e-12>=NEAR_GOAL_STAGNATION_S?'stagnation':null;
    return {dwell_s:dwell,stagnation_s:stagnation,best_distance_m:attempt.bestDistanceM,criterion,
      minimum_distance_m:this.gates.normal,maximum_distance_m:this.closureCenterLimit()??this.gates.forced,
      ...(this.handModel==='inspire'?{distance_kind:'calibrated_center'}:{}),
      minimum_dwell_s:NEAR_GOAL_MIN_DWELL_S,stagnation_limit_s:NEAR_GOAL_STAGNATION_S,maximum_dwell_s:maximumDwell,progress_m:NEAR_GOAL_PROGRESS_M};
  }
  refineNearGrasp(now){
    // A soft tracking/stagnation warning must not turn the final millimeters
    // into a full-body withdrawal. Permit one short, bounded correction window
    // through the same collision/support checks and smooth GA schedule.
    if(!['body_tracking','goal_stagnation'].includes(this.gaFrozenReason)||this.phase!=='approach'||this.stage!=='settle')return false;
    const a=this.graspAlignment(),tracking=this.reachTracking,distance=this.graspFeedback(a).distance_m;
    if(distance<=this.gates.normal||distance>NEAR_GRASP_REFINEMENT_M||a.angle_error_rad>GRASP_ANGLE_TOLERANCE_RAD||tracking?.root_xy_error_m>.10||tracking?.root_angle_error_rad>.25)return false;
    this.nearGraspRefinement??={startTime:now,correctionM:0};
    return now-this.nearGraspRefinement.startTime<2&&this.nearGraspRefinement.correctionM<.008-1e-10;
  }
  closedLoopUpdate(){
    if(this.alignmentRetry&&this.phase==='approach'){
      const retry=this.alignmentRetry;if(!retry.prepared)return;
      // Hold phase: only the bounded refinement. Second approach: nothing closes the loop while the open hand backs off
      // and re-enters; in its settle the refinement runs alongside the ordinary GA loop below, rebuilt from scratch.
      if(retry.phase!=='approach'){this.refineAlignmentRetry();return;}
      if(this.stage!=='settle')return;
      this.refineAlignmentRetry();
    }
    if(!this.replanEnabled||!['approach','close'].includes(this.phase)||['raise','orient','clearance'].includes(this.stage))return;const now=nowOf(this.scene),atGrasp=this.stage==='settle'||this.phase==='close',firstGA=atGrasp&&!this.goalAdjuster;if(now<this.nextReplan&&!firstGA)return;
    // Continue the accepted joint trajectory during precision approach. A
    // measured pose includes policy lag; reusing it as the IK seed every
    // 0.4 s can abruptly redistribute waist/shoulder motion at a fixed EE goal.
    // Replanning and measured-error GA remain active with the continuous seed.
    const precision=this.phase==='close'||['enter_side','settle'].includes(this.stage)||this.stage==='descend'&&this.plan.approach==='top_down';
    this.nextReplan=now+.4;this.replans++;this.reseed=!precision;this.forceReference=true;this.objectLocked||=this.contacts>0;
    if(!this.objectLocked&&!this.forcedClose){
      // Rebasing the nominal anchor leaves the measured live EE target
      // unchanged. Keep its admission history; measured gate exits still
      // reset that history in advanceApproach/updateNearGoalAttempt.
      const position=poseOf(this.scene,this.objectId).position,delta=sub(position,this.objectAnchor);
      if(norm(delta)>.005){const target=this.approachTarget(),current=target.palms[this.hand];if(this.coordinatedReach&&this.segments[this.segmentIndex].name==='reach')this.coordinatedReach.reachWaistPostureFrom=target.posture.slice(12,15);this.graspGoal=add(this.graspGoal,delta);this.grasp=add(this.grasp,delta);this.pregrasp=add(this.pregrasp,delta);this.liftTarget=add(this.liftTarget,delta);this.objectAnchor=position;for(let i=this.segmentIndex;i<this.segments.length;i++){const seg=this.segments[i];if(!['raise','orient','retract','clearance'].includes(seg.name)){seg.position=add(seg.position,delta);if(['side','reach'].includes(seg.name))seg.position[2]=Math.max(seg.position[2],this.safePalmZ);}}this.segmentFrom=current;this.segmentTime=now;this.goalAdjuster=null;if(this.phase==='close')this.closeDuration=Math.max(this.closeDuration,now-this.phaseTime+this.segments.slice(this.segmentIndex).reduce((sum,seg)=>sum+seg.duration,0));}
    }
    // A single finger brushing the object locks its reference anchor, but
    // must not permanently stop convergence before an opposing grip forms.
    if(!this.closureCommitted&&atGrasp&&(firstGA||!this.objectLocked||this.fingerContactCount<2||!this.graspAlignment().ready)){
      let refining=this.refineNearGrasp(now);
      if(this.gaFrozenReason&&!refining)return;
      if(this.stancePlan){
        // Give delayed policy tracking the normal settling window. A short
        // transient plateau, especially near a low table, can precede a useful
        // GA update. Moderate body lag slows approach without cancelling it.
        const distance=this.handModel==='inspire'?this.graspFeedback().distance_m:this.graspDistance();this.goalHistory.push(distance);if(this.goalHistory.length>8)this.goalHistory.shift();
        const stagnant=this.goalHistory.length===8&&this.goalHistory[0]-Math.min(...this.goalHistory.slice(1))<.004;
        const drift=this.reachTracking?.root_xy_error_m>.08||this.reachTracking?.root_angle_error_rad>.20;
        if(stagnant||drift||norm(this.goalAdjustment)>.10){this.gaFrozenReason=norm(this.goalAdjustment)>.10?'compensation_limit':drift?'body_tracking':'goal_stagnation';refining=this.refineNearGrasp(now);if(!refining)return;}
      }
      // Keep both GA deadbands inside the 15 mm admission tolerance so
      // correction remains active until the hand enters the grasp gate.
      this.goalAdjuster??=new GoalAdjuster(this.graspGoal,{enabled:this.goalAdjustEnabled,stay:this.gates.normal*.8,skip:this.gates.normal});
      // Admission follows the measured object, including tabletop settling and
      // a finger brush. Compensate against that same live EE target while the
      // grasp is uncommitted; the nominal anchor and issued command stay intact.
      const alignment=this.graspAlignment(),feedback=this.graspFeedback(alignment);
      // Calibrated-center feedback uses the measured residual, expressed as
      // a live palm target. Only the compensated command moves; nominal palm
      // calibration and the measured admission target remain unchanged.
      this.goalAdjuster.original=this.handModel==='inspire'?sub(alignment.position_w,feedback.error_w):alignment.target_w.slice();
      if(this.goalAdjuster.stayed&&feedback.distance_m>this.gates.normal+1e-12)this.goalAdjuster.stayed=false;
      const maxStep=this.goalAdjuster.maxStep;
      // Reduce acquisition corrections continuously as the measured live EE
      // error shrinks; the reference still follows the same 400 ms blend.
      this.goalAdjuster.maxStep=acquisitionStepLimit(feedback.distance_m,maxStep);
      if(refining)this.goalAdjuster.maxStep=Math.min(this.goalAdjuster.maxStep,.002,.008-this.nearGraspRefinement.correctionM);
      let result;try{result=this.goalAdjuster.update(this.palm().position);}finally{this.goalAdjuster.maxStep=maxStep;}
      if(result.action==='replan'){
        // Bound the proposed command, not only the previous accumulated
        // offset. Projection also permits small corrections along the budget
        // boundary instead of overshooting it and disabling every later retry.
        const proposed=add(this.goalAdjustment,result.shift),length=norm(proposed);
        if(length>.10){const bounded=sub(scale(proposed,.10/length),this.goalAdjustment);this.goalAdjuster.goal=add(this.goalAdjuster.goal,sub(bounded,result.shift));result.shift=bounded;}
        if(this.stancePlan&&norm(result.shift)>0){
          // GA compensates measured tracking error. Its rolling reference may
          // still approach the command, so only geometry and support are hard
          // gates here; the measured progress history above detects stagnation.
          // Stance and withdrawal endpoints retain their strict palm checks.
          // Validate the command with the same waist, root and inactive-hand
          // objectives execution will use; default IK priors can admit a
          // different waist/arm solution that this reach cannot follow.
          const current=this.approachTarget(),target={...current,palms:{...current.palms,[this.hand]:{position:add(this.grasp,result.shift),rotation:this.graspRotation}}};
          try{this.reference.ik.solve(this.reference.qCurrent,target,{iterations:60,maxStep:3,strictEndpoint:false});}
          catch(error){if(!(error instanceof IKError))throw error;this.gaFrozenReason='reference_infeasible';return;}
        }
        if(norm(result.shift)>0){
          // GA's validated endpoint and cumulative correction remain exact.
          // Only the settling command transitions over the next update period;
          // future policy samples see the same continuous temporal schedule.
          this.gaGraspTransition={startTime:now,duration:.4,fromOffset:sub(this.graspCommandOffset(),result.shift)};
          this.gaUpdates++;
          if(refining)this.nearGraspRefinement.correctionM+=norm(result.shift);
        }
        this.goalAdjustment=add(this.goalAdjustment,result.shift);this.grasp=add(this.grasp,result.shift);this.liftTarget=add(this.liftTarget,result.shift);this.segments.at(-1).position=this.grasp.slice();
      }
    }
  }
  topDownTableAllowance(){
    if(this.plan?.approach!=='top_down'||!this.stancePlan||!['approach','close'].includes(this.phase))return 0;
    const now=nowOf(this.scene);if(this.topDownTableRelease?.time_s===now)return this.topDownTableRelease.allowance_m;
    const index=this.hand==='left'?0:1,frame=this.reference.currentFrame();
    // Measured clearance = native mesh distance of the actual hand to the tabletop (the bounding-box
    // clearance underestimates it by ~8 mm and would leave nothing to release).
    const bias=this.palm().position[2]-frame.palmPosW[index][2],clearance=this.audit.handTableDistance(this.scene.data,this.hand);
    const allowance=Math.max(0,Math.min(bias,clearance-TOP_DOWN_TABLE_RELEASE_MIN_CLEARANCE_M,TOP_DOWN_TABLE_RELEASE_MAX_M));
    this.topDownTableRelease={time_s:now,bias_m:bias,clearance_m:clearance,allowance_m:allowance};
    return allowance;
  }
  // A tall hand at a table >= 0.80 m needs a deeper, slower fold
  // to clear the tabletop edge during the lift.
  tallHandFold(){return HAND_FOLD_ELBOW_RAD[this.handModel]!==undefined&&this.handModel!=='dex3'&&tableZ(this.scene)>=.80;}
  // Diagnostic only: the palm position the approach is commanding right now (null outside the approach).
  approachTargetPalmDiagnostic(){
    if(this.phase!=='approach'||!this.segments||!this.plan)return null;
    try{return copy(this.approachTarget().palms[this.hand].position);}catch{return null;}
  }
  updateReachTracking(){
    if(!this.stancePlan)return;
    const state=this.scene.readState(this.objectId),frame=this.reference.currentFrame(),index=this.hand==='left'?0:1;
    this.reachTracking={root_xy_error_m:Math.hypot(...sub(state.rootPosW,frame.rootPosW).slice(0,2)),
      root_angle_error_rad:norm(rotationError(quatToMat(frame.rootQuatW),quatToMat(state.rootQuatW))),
      palm_tracking_error_m:norm(sub(state.palmPosW[index],frame.palmPosW[index])),
      reference_goal_error_m:norm(sub(frame.palmPosW[index],this.grasp)),pace:1};
    const slow=this.replanEnabled&&['descend','enter_side'].includes(this.stage)&&
      (this.reachTracking.root_xy_error_m>.04||this.reachTracking.root_angle_error_rad>.10||this.reachTracking.palm_tracking_error_m>.08);
    this.approachRate=slow?.5:1;this.reachTracking.pace=this.approachRate;
    if(slow)this.segmentTime+=(1-this.approachRate)*this.dt;
  }
  observeApproachObstruction(){
    if(!this.approachObstruction)return;
    const frame=this.reference.currentFrame(),diagnostic=frame.approachCollision,index=this.hand==='left'?0:1;
    this.approachObstruction?.update({time:nowOf(this.scene),
      eligible:this.phase==='approach'&&['descend','enter_side','settle'].includes(this.stage),
      referenceDistance:this.graspGoal?norm(sub(frame.palmPosW[index],this.graspGoal)):0,
      requestedError:diagnostic?.position_error_m??0,constraints:diagnostic?.constraints??[]});
  }
  observeTransferObstruction(){
    if(!this.transferObstruction)return null;
    const seg=this.transferSegments?.[this.transferIndex],now=nowOf(this.scene);
    const diagnostic=this.reference.currentFrame().approachCollision;
    // Compare the consumed IK frame to its own requested pose. Distance to
    // the original grasp is irrelevant once the hand is carrying to the tray.
    return this.transferObstruction?.update({time:now,
      eligible:this.phase==='transfer'&&!!seg&&now-this.transferTime>=seg.duration&&
        (!this.clearanceRecovery||now-this.clearanceRecovery.time>=.8),
      referenceDistance:diagnostic?.position_error_m??0,
      requestedError:diagnostic?.position_error_m??0,constraints:diagnostic?.constraints??[]});
  }
  startTransferStage(){
    this.transferStageStarted=nowOf(this.scene);this.transferBudgetStarted??=this.transferStageStarted;
    this.transferObstruction?.reset();
  }
  async maybeReplanStance(){
    if(this.alignmentRetry||!this.stancePlan||!this.replanEnabled||this.stage!=='settle'||!this.gaFrozenReason||this.stanceReplanCount>=1||this.objectLocked||this.handContacts>0||this.contacts>0)return false;
    // A tracking warning may freeze further GA while the accepted reference
    // is still bringing the palm into its grasp gate. Near the object, let
    // that approach finish within its existing timeout instead of replacing
    // it with a full lift/withdrawal that can sweep the open fingers into it.
    const alignment=this.graspAlignment();
    if(alignment.distance_m<=NEAR_GRASP_REFINEMENT_M&&alignment.angle_error_rad<=GRASP_ANGLE_TOLERANCE_RAD)return false;
    const approach=this.approachTarget();
    const stance=await planReachStance(this.reference.ik,this.reachStanceTarget(),{previous:this.stancePlan});
    this.stanceReplanCount++;
    this.stanceHistory.push({reason:this.gaFrozenReason,...stance});
    if(!stance.feasible)return false;
    const from=this.plannedPalms(),bounds=this.audit.handBounds(this.scene.data,this.hand);
    const high=add(from[this.hand].position,[0,0,Math.max(.03,this.safePalmZ-this.palm().position[2],tableZ(this.scene)+.06-bounds.lower[2])]);
    const back=high.slice();back[0]-=Math.max(0,bounds.upper[0]-(this.tableFront-.10));
    this.recoveryPlan=stance;this.stanceRecoveryReason=this.gaFrozenReason;this.recoveryFrom=from;
    // Keep the waist objective from the approach until the open hand clears
    // the table. Dropping these weights here lets IK abruptly unbend the
    // torso and counter-rotate the arm even when the palm target is unchanged.
    this.recoveryPosture=copy(approach.posture);this.recoveryWaistPostureWeights=approach.waistPostureWeights?.slice()??null;
    this.recoverySegments=[{name:'raise',position:high,duration:1.5},{name:'retract',position:back,duration:2}];
    this.recoveryIndex=0;this.recoveryOrigin=from[this.hand].position;this.recoveryTime=nowOf(this.scene);
    // Do not relax or reposition the body while the open fingers are over the
    // table. Audit the lift/retraction first and gate it against actual hands.
    let accepted=false;this.recoveryPathChecks=[];
    const homeY=this.homePalms[this.hand].position[1],sign=this.hand==='left'?1:-1;
    for(const y of [back[1],homeY+sign*.04,homeY+sign*.08]){
      this.recoverySegments[1].position[1]=y;
      try{const audit=await this.preflightHomePath(t=>this.recoveryTarget(t,{preflight:true}),3.5);this.recoveryPathChecks.push({y,passed:true,...audit});accepted=true;break;}
      catch(error){if(!(error instanceof IKError))throw error;this.recoveryPathChecks.push({y,passed:false,reason:error.message,details:error.details});}
    }
    if(!accepted){this.recoveryPlan=null;this.gaFrozenReason='recovery_path_infeasible';return false;}
    this.attemptNotice='Re-planning the body posture for a farther reach (one retry): lifting and withdrawing the open hand first.';this.enter('stance_recovery',this.attemptNotice);this.stage='raise';this.approachRate=1;return true;
  }
  recoveryTarget(offset=0,{preflight=false}={}){
    let elapsed=nowOf(this.scene)-this.recoveryTime+Math.max(0,offset),origin=this.recoveryOrigin,position=origin;
    for(let i=this.recoveryIndex;i<this.recoverySegments.length;i++){
      const segment=this.recoverySegments[i];position=lerp(origin,segment.position,smooth(elapsed/segment.duration));
      // Only an explicit full-path audit crosses stages without a measured
      // clearance gate. Execution and its future window hold this endpoint
      // until advanceStanceRecovery authorizes the next stage.
      if(!preflight||elapsed<=segment.duration)break;elapsed-=segment.duration;origin=segment.position;
    }
    return this.oneHandTarget(position,this.recoveryFrom[this.hand].rotation,{allowHandObject:true,posture:this.recoveryPosture??this.restQ,waistPostureWeights:this.recoveryWaistPostureWeights??undefined});
  }
  async advanceStanceRecovery(){
    const segment=this.recoverySegments[this.recoveryIndex],elapsed=nowOf(this.scene)-this.recoveryTime;this.stage=segment.name;
    const bounds=this.audit.handBounds(this.scene.data,this.hand),clear=segment.name==='raise'?bounds.lower[2]>=tableZ(this.scene)+.025:bounds.upper[0]<=this.tableFront-.025;
    if(elapsed<segment.duration||!clear){if(elapsed>segment.duration+4){this.failureMessage??='The open hand could not clear the table for another posture.';this.failureReason??={code:'grasp_aborted',reason:'stance_recovery_clearance_timeout',phase:this.phase,stage:this.stage,target_object_id:this.objectId};this.graspAbort??={reason:'stance_recovery_clearance_timeout',phase:this.phase,stage:this.stage,startedAt:nowOf(this.scene)};this.markCompletionFallback('stance_recovery_clearance_timeout');await this.beginReturn();}return;}
    if(this.recoveryIndex===0){this.recoveryIndex=1;this.recoveryOrigin=segment.position.slice();this.recoveryTime=nowOf(this.scene);this.forceReference=true;return;}
    const liftOffset=sub(this.liftTarget,this.grasp),preOffset=sub(this.pregrasp,this.graspGoal);
    this.grasp=this.graspGoal.slice();this.liftTarget=add(this.grasp,liftOffset);this.pregrasp=add(this.grasp,preOffset);
    this.goalAdjustment=[0,0,0];this.goalAdjuster=null;this.gaGraspTransition=null;this.liftFrom=null;this.goalHistory=[];this.gaFrozenReason=null;this.nextReplan=0;
    try{await this.beginStanceTransition(this.recoveryPlan);}
    catch(error){if(!(error instanceof IKError))throw error;this.markCompletionFallback('stance_transition_retry');this.coordinatedReach=null;this.buildApproach();}
  }
  rootHomeMetrics(){
    const state=this.scene.readState(this.objectId),anchor=this.reference.ik.anchorPosition;
    const xy=Math.hypot(...sub(state.rootPosW,anchor).slice(0,2)),height=Math.abs(state.rootPosW[2]-anchor[2]);
    const angle=norm(rotationError(this.reference.ik.anchorRotation,quatToMat(state.rootQuatW))),speed=norm(state.rootLinVelW),angularSpeed=norm(state.rootAngVelB);
    return {xy_m:xy,height_m:height,angle_rad:angle,speed_m_s:speed,angular_speed_rad_s:angularSpeed,reached:xy<.04&&height<.03&&angle<.12,quiet:speed<.08&&angularSpeed<.3};
  }
  async beginStand(){
    const frame=this.reference.currentFrame();this.standFrom=this.plannedPalms();this.standStartRoot=frame.rootPosW.slice();this.standStartPitch=frame.plannedRootPitch;
    // A loaded base recovery continues the lift's waist objective. Replacing
    // its strong weights with the default .06 at this boundary lets the arm
    // and waist trade motion abruptly even though the palm barely moves.
    this.standPosture=(this.liftPosture??this.restQ).slice();
    this.standWaistPostureWeights=(this.liftWaistPostureWeights??[.06,.06,.06]).slice();
    // Bring a distant load closer while recovering the supported base. Holding
    // the arm fully extended can leave the policy in a quiet but rear-shifted
    // equilibrium, just outside the unchanged home-position tolerance.
    const loadReach=this.standFrom[this.hand].position[0]-this.reference.ik.anchorPosition[0];
    this.standLoadRetraction=[-clamp(loadReach-.46,0,.06),0,0];
    this.standDuration=Math.max(2,norm(sub(this.standStartRoot,this.reference.ik.anchorPosition))/.035,Math.abs(this.standStartPitch)/.15);this.stableBody=0;
    try{await this.preflightHomePath(t=>this.standTargetAt(smooth(t/this.standDuration)),this.standDuration);}
    catch(error){if(!(error instanceof IKError))throw error;await this.preflightHomePath(t=>this.standTargetAt(smooth(t/this.standDuration)),this.standDuration,{strictEndpoint:false,bestEffort:true});this.markCompletionFallback('standing_pose_retry');}
    this.enter('stand','Keeping the grasp while returning to the initial supported stance.');
  }
  standTargetAt(alpha){
    const root=lerp(this.standStartRoot,this.reference.ik.anchorPosition,alpha),palms=structuredClone(this.standFrom);
    palms[this.hand].position=add(palms[this.hand].position,add(sub(root,this.standStartRoot),scale(this.standLoadRetraction??[0,0,0],alpha)));
    return this.baseTarget(palms,{rootPosition:root,rootHeight:root[2],rootPitch:(1-alpha)*this.standStartPitch,bodyRelativeHands:[otherHand(this.hand)],palmOffsetsTorso:this.crouchOffsets,allowHandObject:true,
      posture:this.standPosture??this.liftPosture??this.restQ,waistPostureWeights:this.standWaistPostureWeights??this.liftWaistPostureWeights??undefined,jointContinuityWeight:2*alpha});
  }
  standTarget(offset=0){return this.standTargetAt(smooth((nowOf(this.scene)-this.phaseTime+offset)/this.standDuration));}
  beginCompletionTransfer(reason){
    if(this.failUnsecuredObstructedGrasp())return;
    this.markCompletionFallback(reason);this.completionTransfer=true;
    this.beginTransferPosture();
    const tray=this.scene.tray,actual=this.palm(),reference=this.plannedPalms()[this.hand],handBounds=this.audit.handBounds(this.scene.data,this.hand);
    // A missed object stays where physics leaves it. Use the calibrated grasp
    // center for the empty-hand gesture, never its distant live object offset.
    const held=this.handContacts>0,bounds=held?this.objectBounds():handBounds;
    const local=matVec(transpose3(this.graspRotation),sub(this.objectAnchor,this.graspGoal));
    const offset=held?sub(poseOf(this.scene,this.objectId).position,actual.position):matVec(actual.rotation,local);
    const bias=sub(reference.position,actual.position),floor=Math.max(tableZ(this.scene),tray.wallTopZ);
    const rise=Math.max(0,floor+.07-Math.min(handBounds.lower[2],bounds.lower[2]));
    const raised=add(reference.position,[0,0,rise]),over=[tray.center[0]-offset[0]+bias[0],tray.center[1]-offset[1]+bias[1],raised[2]];
    this.depositRotation=reference.rotation;this.unwindRotation=null;this.clearanceRecovery=null;
    this.completionClearanceTop=floor;this.transferSegments=[
      {name:'raise_load',position:raised,duration:1.6},
      {name:'over_tray',position:over,duration:2.8},
      {name:'lower_load',position:add(over,[0,0,-.02]),duration:1.2}];
    this.transferIndex=0;this.transferFrom=reference.position.slice();this.transferTime=nowOf(this.scene);
    this.startTransferStage();
    this.enter('transfer','Completing the transfer and release over the tray.');
  }
  advanceCompletionTransfer(){
    const seg=this.transferSegments[this.transferIndex],elapsed=nowOf(this.scene)-this.transferTime;this.stage=seg.name;
    if(elapsed<seg.duration)return;
    if(seg.name==='raise_load'){
      const lower=this.audit.handBounds(this.scene.data,this.hand).lower[2],loadLower=this.handContacts>0?this.objectBounds().lower[2]:Infinity;
      if(Math.min(lower,loadLower)<this.completionClearanceTop+.025){
        if(elapsed<seg.duration+3){this.applyTransferShift([0,0,.0008],{propagate:false});return;}
        // Do not send an uncleared hand sideways through the rim. Open and
        // withdraw from the achieved pose if vertical tracking cannot clear it.
        this.markCompletionFallback('transfer_clearance_timeout');this.beginRelease();return;
      }
      const dz=seg.position[2]-this.transferSegments[1].position[2];
      for(const next of this.transferSegments.slice(1))next.position[2]+=dz;
    }
    if(seg.name==='over_tray'&&norm(sub(this.palm().position,seg.position))>.04&&elapsed<seg.duration+3)return;
    if(this.transferIndex===this.transferSegments.length-1){this.beginRelease();return;}
    this.transferFrom=seg.position.slice();this.transferIndex++;this.transferTime=nowOf(this.scene);this.forceReference=true;
    this.startTransferStage();
  }
  beginDeposit(previousTarget=null){
    this.beginTransferPosture(previousTarget);
    const tray=this.scene.tray,position=poseOf(this.scene,this.objectId).position,bounds=this.objectBounds(),actual=this.palm(),reference=this.plannedPalms()[this.hand],bias=sub(reference.position,actual.position);this.objectHandOffset=sub(position,actual.position);
    this.depositRotation=reference.rotation;this.unwindRotation=this.objectId==='uiuc_i'&&this.plan.approach==='side'&&this.plan.yawDeg!==0?reference.rotation:null;
    if(this.objectId==='uiuc_i')this.restQ[12]=(this.hand==='right'?1:-1)*.25;
    let rise=Math.max(0,tray.wallTopZ+.05-bounds.lower[2],tray.wallTopZ+.04-this.audit.handBounds(this.scene.data,this.hand).lower[2]);this.highRelease=this.objectId==='uiuc_i'&&this.hand==='right';if(this.highRelease)rise+=.06;
    const raised=add(reference.position,[0,0,rise]),over=raised.slice();over[0]=tray.center[0]-this.objectHandOffset[0]+bias[0];over[1]=tray.center[1]-this.objectHandOffset[1]+bias[1];
    const lower=over.slice(),drop=this.highRelease?.068:this.objectId==='uiuc_i'?.008:.02;this.dropClearance=drop;lower[2]=tray.bottomTopZ+drop+(position[2]-bounds.lower[2])-this.objectHandOffset[2]+bias[2];
    this.transferSegments=[{name:'raise_load',position:raised,duration:1.2},{name:'over_tray',position:over,duration:2.6}];if(this.objectId==='uiuc_i')this.transferSegments.push({name:'align_upright',position:over.slice(),duration:this.unwindRotation?2:1});this.transferSegments.push({name:'lower_load',position:lower,duration:1.8});
    this.transferIndex=0;this.transferFrom=reference.position.slice();this.transferTime=nowOf(this.scene);this.transferAdjustment=[0,0,0];this.nextTransferAdjust=0;this.transferAdjuster=null;this.transferAdjustStage=null;this.clearanceRecovery=null;this.lowerContactHold=0;this.lowerContactGrounded=false;this.lowerSupportContactTime=null;this.lowerSupport=null;this.enter('transfer','Grasp secured. Raising the object above the tray rim.');
    this.startTransferStage();
  }
  beginTransferPosture(previousTarget=null){
    const previous=previousTarget??(this.phase==='stand'?this.standTarget():this.phase==='transfer'&&this.transferSegments?.length?this.transferMotionOptions():
      ['lift','hold'].includes(this.phase)?{posture:this.liftPosture??this.restQ,waistPostureWeights:this.liftWaistPostureWeights}:null);
    // Both a completed stand and its bounded fallback continue the reached
    // root request; a timeout must not jump back to the old crouched root.
    if(this.phase==='stand'){
      this.rootPosition=previous.rootPosition.slice();this.rootHeight=previous.rootHeight;this.rootPitch=previous.rootPitch;
    }
    this.transferPostureFrom=(previous?.posture??this.restQ).slice();
    this.transferWaistWeightsFrom=(previous?.waistPostureWeights??[.06,.06,.06]).slice();
    this.transferContinuityFrom=previous?.jointContinuityWeight??2;
    // Preserve the inactive arm's torso-relative IK meaning throughout this
    // carry. Release later captures the planned palms before holding them in
    // world space, instead of switching to an unrelated home-arm FK target.
    this.transferBodyRelativeHands=previous?.bodyRelativeHands?.slice()??null;
    this.transferPalmOffsetsTorso=previous?.palmOffsetsTorso?structuredClone(previous.palmOffsetsTorso):null;
    this.transferPostureTime=nowOf(this.scene);this.transferGoalTransitions=[];
  }
  transferMotionOptions(offset=0){
    const blend=this.transferPostureFrom?minimumJerk((nowOf(this.scene)+Math.max(0,offset)-this.transferPostureTime)/1.2):1;
    const posture=this.transferPostureFrom?lerp(this.transferPostureFrom,this.restQ,blend):this.restQ.slice();
    // Release the lift's waist objective continuously and retain joint
    // continuity at every table height, including a fully standing transfer.
    const lowering=this.transferSegments?.[this.transferIndex]?.name==='lower_load';
    const jointContinuityWeight=lowering?2-minimumJerk((nowOf(this.scene)+Math.max(0,offset)-this.transferTime)/.4):(this.transferContinuityFrom??2)+(2-(this.transferContinuityFrom??2))*blend;
    const waistPostureWeights=lerp(this.transferWaistWeightsFrom??[.06,.06,.06],[.06,.06,.06],blend);
    return {allowHandObject:true,posture,waistPostureWeights,jointContinuityWeight,
      ...(this.transferBodyRelativeHands?{bodyRelativeHands:this.transferBodyRelativeHands.slice(),palmOffsetsTorso:structuredClone(this.transferPalmOffsetsTorso)}:{})};
  }
  transferCorrectionAt(offset=0){
    const time=nowOf(this.scene)+Math.max(0,offset);
    return (this.transferGoalTransitions||[]).reduce((sum,r)=>add(sum,scale(r.shift,minimumJerk((time-r.time)/r.duration)-1)),[0,0,0]);
  }
  applyTransferShift(shift,{propagate=true,duration=.4,forceReference=true}={}){
    if(norm(shift)===0)return;
    const now=nowOf(this.scene);
    // Keep completed waypoints committed, and smoothly introduce each new
    // correction. Summed ramps preserve velocity even for overlapping updates.
    this.transferGoalTransitions=(this.transferGoalTransitions||[]).filter(r=>now<r.time+r.duration);
    this.transferGoalTransitions.push({time:now,duration,shift:shift.slice()});
    const last=propagate?this.transferSegments.length:this.transferIndex+1;
    for(let i=this.transferIndex;i<last;i++)this.transferSegments[i].position=add(this.transferSegments[i].position,shift);
    if(this.clearanceRecovery)this.clearanceRecovery.goal=add(this.clearanceRecovery.goal,shift);
    if(forceReference)this.forceReference=true;
  }
  transferTarget(offset=0){
    const current=this.transferSegments[this.transferIndex],now=nowOf(this.scene),rotation=current.name==='align_upright'&&this.unwindRotation?rotationBlend(this.unwindRotation,eye3(),smooth((now-this.transferTime+offset)/current.duration)):this.depositRotation;
    const correction=this.transferCorrectionAt(offset),options=this.transferMotionOptions(offset);
    if(this.clearanceRecovery){const r=this.clearanceRecovery;return this.oneHandTarget(add(lerp(r.start,r.goal,smooth((now-r.time+offset)/.8)),correction),rotation,options);}
    // Every transfer stage waits for measured clearance/support. Future policy
    // frames hold this stage's endpoint until its actual gate permits the next
    // movement; position and orientation therefore describe the same stage.
    const elapsed=now-this.transferTime+Math.max(0,offset),position=lerp(this.transferFrom,current.position,smooth(elapsed/current.duration));
    return this.oneHandTarget(add(position,correction),rotation,options);
  }
  beginRelease(){
    // Hold the already-issued command while the fingers open. Collision
    // queries use measured geometry plus command deltas throughout this hold.
    const issued=this.reference.clip?.sample(this.reference.currentIndex+1)??this.reference.currentFrame();
    this.releasePalms=Object.fromEntries(SIDES.map((side,i)=>[side,{position:copy(issued.palmPosW[i]),rotation:quatToMat(issued.palmQuatW[i])}]));this.releaseFrom=this.releasePalms[this.hand];this.releaseHoldQ=copy(issued.jointPos);
    const bounds=this.audit.handBounds(this.scene.data,this.hand),top=Math.max(this.scene.tray.wallTopZ,this.objectBounds().upper[2]);
    this.releaseRetreat=add(this.releaseFrom.position,[0,0,Math.max(.08,top+.035-bounds.lower[2])]);
    const now=nowOf(this.scene);
    this.releaseHandOpening=null;this.releaseExit={stage:'opening',checks:[],startedAt:now,segment:null,stable:0,objectIgnored:false,ignoredAt:null,ignoreGate:null,margin:.005,retreat:null,hazardCount:0,supportStable:0,clearTop:null,plannedTier:null,lastMeasured:null,openingCollisionFrame:this.captureReleaseCollisionFrame(issued),openingRootPitch:issued.plannedRootPitch};this.postReleaseDeadline=now+POST_RELEASE_DEADLINE_S;
    this.enter('release','Opening the fingers before lifting vertically clear of the object.');
    this.stage='opening';
  }
  releaseMovingGeoms(){
    return this.audit.robotGeoms.filter(g=>{const name=this.scene.bodyName(this.scene.model.geom_bodyid[g]);return name.startsWith(`${this.hand}_hand_`)||name.startsWith(`${this.hand}_wrist_`);});
  }
  releaseSweepAudit(delta,{separation=null}={}){
    // This independent scratch is used only for native shape queries. A
    // rigid sweep of the measured open hand filters routes before articulated
    // IK. Never serialize it as FK or use its shifted geoms for Jacobians.
    const scene=this.scene,data=this.homeAuditData,moving=new Set(this.releaseMovingGeoms()),margin=this.releaseExit?.margin??.005;
    if(data===scene.data||!moving.size||delta.length!==3||!delta.every(Number.isFinite))throw new IKError('Release sweep lacks independent native geometry.');
    data.qpos.set(scene.data.qpos);scene.forward(data);
    const original=new Map([...moving].map(g=>[g,copy(data.geom_xpos.slice(3*g,3*g+3))]));
    // A rigid translation of the hand/wrist cannot model the same arm's elbow and shoulder following it; those self pairs
    // are audited on the articulated joint path instead (releaseArticulatedAudit), not in this pre-filter.
    const ownArm=g=>{const name=scene.bodyName(scene.model.geom_bodyid[g]);return name.startsWith(`${this.hand}_shoulder_`)||name.startsWith(`${this.hand}_elbow`);};
    const pairs=this.audit.pairs({hand:this.hand,objectId:this.objectId,allowHandObject:false,includeSelf:true,ignoreObjectId:this.ignoredObjectId}).filter(([a,b])=>moving.has(a)!==moving.has(b)&&!ownArm(moving.has(a)?b:a));
    const floors=new Map((separation?.pairs??[]).map(p=>[`${Math.min(p.a,p.b)}:${Math.max(p.a,p.b)}`,p.minimumDistance])),distanceToTravel=norm(delta),attempts=[];
    let spacing=Math.min(.002,...[...floors.values()].filter(v=>v>0)),totalSamples=0;
    try{
      while(true){
        const steps=Math.max(1,Math.ceil(distanceToTravel/spacing)),samplingAllowance=distanceToTravel/(2*steps),previous=new Map(floors);
        if(!Number.isFinite(steps)||totalSamples+steps+1>512)return {passed:false,reason:'separation_sampling_budget_exhausted',requiredSamples:Number.isFinite(steps)?steps+1:null,maximumSamples:512,totalSamples,attempts};
        const attempt={spacing,plannedSamples:steps+1,samplingAllowance,samples:0,passed:false};attempts.push(attempt);let minimum=Infinity,limitingPair=null,refine=false;
        sweep:for(let i=0;i<=steps;i++){
          attempt.samples++;totalSamples++;
          for(const [g,p]of original)for(let k=0;k<3;k++)data.geom_xpos[3*g+k]=p[k]+delta[k]*i/steps;
          const boxes=new Map(),box=g=>{if(!boxes.has(g))boxes.set(g,geomBounds(scene,data,[g]));return boxes.get(g);};
          for(const pair of pairs){
            const [a,b]=pair,key=this.audit.marginKey(pair,a,b),required=Math.min(margin,this.audit.pairMarginOverrides.get(key)??margin),floor=floors.get(key),hardThreshold=floor!==undefined&&i<steps?floor:required,threshold=hardThreshold+(floor!==undefined&&i<steps?0:samplingAllowance);
            const aa=box(a),bb=box(b),gap=Math.hypot(...[0,1,2].map(k=>Math.max(0,aa.lower[k]-bb.upper[k],bb.lower[k]-aa.upper[k])));
            if(!Number.isFinite(gap))throw new IKError('Release sweep bounds are missing or non-finite.',{a,b,step:i});
            if(floor===undefined&&scene.model.geom_type[b]!==0&&gap>Math.max(0,threshold)+.001)continue;
            const distance=this.audit.nativeDistance(data,a,b,Math.max(.01,threshold+.001)),surplus=distance-threshold;
            if(!Number.isFinite(distance))throw new IKError('Release sweep distance is missing or non-finite.',{a,b,step:i});
            if(floor!==undefined&&(distance+1e-9<previous.get(key)||floor>=0&&distance<samplingAllowance)){
              Object.assign(attempt,{reason:'separation_distance_decreased',minimumClearanceSurplus:distance-previous.get(key),limitingPair:{a,b,distance,minimumDistance:previous.get(key),step:i}});
              return {...attempt,totalSamples,attempts};
            }
            if(floor!==undefined)previous.set(key,Math.min(margin,Math.max(previous.get(key),distance)));
            if(surplus<minimum){minimum=surplus;limitingPair={a,b,distance,requiredMargin:required,samplingAllowance,step:i};}
            if(surplus< -1e-9){
              // Refine only the conservative half-spacing allowance, along
              // this exact route. Hard margins and captured monotonic floors
              // still fail immediately; all attempts share the 512 cap.
              refine=distance>=hardThreshold;Object.assign(attempt,{reason:refine?'sampling_allowance':'hard_clearance_margin',minimumClearanceSurplus:minimum,limitingPair});
              if(!refine)return {...attempt,totalSamples,attempts};
              break sweep;
            }
          }
        }
        if(!refine){Object.assign(attempt,{passed:true,minimumClearanceSurplus:Number.isFinite(minimum)?minimum:null,limitingPair});return {...attempt,totalSamples,attempts};}
        spacing/=2;
      }
    }finally{scene.forward(data);}
  }
  releasePredictedQ(q,collisionFrame){
    if(!collisionFrame)return copy(q);
    if(collisionFrame.hand!==this.hand||collisionFrame.objectId!==this.objectId||collisionFrame.measuredUpperQ?.length!==17||collisionFrame.referenceUpperQ?.length!==17||![...collisionFrame.measuredUpperQ,...collisionFrame.referenceUpperQ].every(Number.isFinite))throw new IKError('Missing measured release upper-joint anchor.');
    return q.map((v,i)=>i<12?v:collisionFrame.measuredUpperQ[i-12]+v-collisionFrame.referenceUpperQ[i-12]);
  }
  releaseActualPoseAudit(q,separation=null,previous=null,collisionFrame=null){
    q=this.releasePredictedQ(q,collisionFrame);
    const data=this.homeAuditData,scene=this.scene;data.qpos.set(scene.data.qpos);
    this.reference.ik.qadr.slice(12).forEach((address,i)=>data.qpos[address]=q[12+i]);
    // Keep the actual lower body, root, fingers and detached object. Unlike
    // the home-posture check, include all wrists, body and self pairs.
    scene.forward(data);const geometry=this.audit.check(data,{hand:this.hand,objectId:this.objectId,allowHandObject:false,includeSelf:true,margin:this.releaseExit?.margin??.005,releaseSeparation:separation,ignoreObjectId:this.ignoredObjectId});
    if(geometry.passed&&separation)for(const pair of separation.pairs){
      const key=`${pair.a}:${pair.b}`,distance=this.audit.nativeDistance(data,pair.a,pair.b,.01),minimum=previous?.get(key)??pair.minimumDistance;
      if(!Number.isFinite(distance)||!Number.isFinite(minimum)||distance+1e-9<minimum)return {...geometry,passed:false,separationViolation:{...pair,distance,minimum}};
      previous?.set(key,Math.min(.005,Math.max(minimum,distance)));
    }
    return geometry;
  }
  releasePathTarget(from,goal,alpha,separation=null,collisionFrame=this.releaseExit?.collisionFrame){
    const palms=structuredClone(from);palms[this.hand].position=lerp(from[this.hand].position,goal,alpha);
    return this.baseTarget(palms,{posture:this.releaseHoldQ,allowHandObject:false,lockWaist:true,jointContinuityWeight:separation?0:1,...(collisionFrame?{releaseCollisionFrame:collisionFrame,lockLowerBody:true}:{}),...(separation&&alpha<1?{releaseSeparation:separation}:{}),...(this.releaseExit?.margin<.005?{collisionMargin:this.releaseExit.margin}:{})});
  }
  releasePathProgress(elapsed,duration,separation=null){
    if(!Number.isFinite(elapsed)||!Number.isFinite(duration)||duration<=0)throw new IKError('Invalid checked release timing.');
    const alpha=clamp(elapsed/duration,0,1);
    // A finite, modest outward rate opposes measured drift immediately while
    // close-pair floors are active. Withdrawal and rise retain eased timing.
    return separation?alpha:smooth(alpha);
  }
  async planReleaseSegment(delta,stage,separation=null){
    // Every detached-object stage needs the same measured full-pose anchor
    // and transactional reference splice, including a clear-start rise.
    return this.planReleaseReconciliation(delta,separation,{stage,duration:Math.max(1.2,norm(delta)/.08)});
  }
  async planReleaseStage(stage,candidates){
    // Strict 5 mm fixture/self margin first. Once the released object is ignored, one relaxed tier bounded by the
    // measured start clearance (never closer than the hand already is minus 0.5 mm, never below 1 mm).
    const exit=this.releaseExit,tiers=[.005],previous=exit.plannedTier;exit.plannedTier=null;
    if(exit.objectIgnored){const relaxed=this.releaseExitMargin();if(relaxed<.005-1e-12)tiers.push(relaxed);}
    for(const margin of tiers)for(const delta of candidates){
      exit.margin=margin;
      if(await this.planReleaseSegment(delta,stage)){exit.plannedTier={stage,margin,delta:delta.slice()};if(margin<.005)this.recordPostReleaseHazard(`release_${stage}_relaxed_margin`,{note:`margin ${margin}`});return true;}
    }
    // Nothing planned: keep the bookkeeping of the segment that is still executing (if any).
    exit.plannedTier=previous;exit.margin=exit.segment?previous?.margin??.005:.005;return false;
  }
  releaseExitMargin(){
    const moving=new Set(this.releaseMovingGeoms()),geometry=this.audit.check(this.releaseMeasuredData(),{hand:this.hand,objectId:this.objectId,allowHandObject:false,includeSelf:true,margin:.005,ignoreObjectId:this.objectId});
    // Relaxed margin = the measured start distance of the tightest fixture/self pair of the moving hand minus 0.5 mm,
    // clamped to [1 mm, 5 mm]; a start at or above 5.5 mm keeps the ordinary margin (no relaxed tier).
    const arm=g=>(this.scene?.bodyName?.(this.scene?.model?.geom_bodyid?.[g])??'').startsWith(`${this.hand}_`);
    const surplus=Math.min(...(geometry.near||[]).filter(r=>moving.has(r.a)||moving.has(r.b)||arm(r.a)||arm(r.b)).map(r=>r.distance-r.requiredMargin));
    return Number.isFinite(surplus)?clamp(surplus+.0045,.001,.005):.005;
  }
  releaseRetreatCandidates(){
    // After opening, first retreat 5-10 cm toward the robot (table -X) with a small outboard bias (left hand +Y,
    // right hand -Y), clipped to the tray interior from the measured hand/wrist parts below the wall top.
    const exit=this.releaseExit,tray=this.scene.tray,inner=tray.innerBounds,dirY=this.hand==='left'?1:-1,data=this.releaseMeasuredData();
    let back=RELEASE_RETREAT_NOMINAL_M,out=RELEASE_RETREAT_OUTBOARD_M,low=0;
    for(const g of this.releaseMovingGeoms()){
      const b=geomBounds(this.scene,data,[g]);if(!(b.lower[2]<tray.wallTopZ+RELEASE_RETREAT_WALL_MARGIN_M))continue;low++;
      back=Math.min(back,b.lower[0]-(inner.x_min+RELEASE_RETREAT_WALL_MARGIN_M));
      out=Math.min(out,dirY>0?inner.y_max-b.upper[1]-RELEASE_RETREAT_WALL_MARGIN_M:b.lower[1]-inner.y_min-RELEASE_RETREAT_WALL_MARGIN_M);
    }
    out=clamp(Number.isFinite(out)?out:0,0,RELEASE_RETREAT_OUTBOARD_M);
    let candidates=[];
    if(Number.isFinite(back)&&back>=RELEASE_RETREAT_MIN_M-1e-9){
      const raw=[[-back,dirY*out,0],[-back,0,0],[-Math.max(RELEASE_RETREAT_MIN_M,.7*back),dirY*.5*out,0],[-RELEASE_RETREAT_MIN_M,0,0]];
      candidates=raw.filter(d=>d.every(Number.isFinite)&&-d[0]>=RELEASE_RETREAT_MIN_M-1e-9).map(d=>d.map(v=>Math.round(v*1e4)/1e4)).filter((d,i,a)=>a.findIndex(v=>norm(sub(v,d))<1e-9)===i);
    }
    exit.retreat={back_m:back,outboard_m:out,low_geoms:low,candidates,selected:null,reached_m:null};return candidates;
  }
  async planReleaseRetreat(){
    const exit=this.releaseExit,candidates=this.releaseRetreatCandidates();
    if(!candidates.length){this.recordPostReleaseHazard('release_retreat_skipped');return false;}
    if(await this.planReleaseStage('retreat',candidates)){exit.retreat.selected=exit.plannedTier.delta.slice();return true;}
    this.recordPostReleaseHazard('release_retreat_unplanned');return false;
  }
  async planReleaseRise(){
    if(await this.planReleaseStage('rise',[this.releaseRiseDelta()]))return true;
    // No checked rise: fall forward to the un-audited best-effort vertical lift, then the return.
    this.recordPostReleaseHazard('release_rise_unplanned');this.beginClearTray();return false;
  }
  releaseClearTop(){
    // Height the hand/wrist must clear before the return: the tray wall top; the object top is added only while the
    // hand's XY footprint is within 2 cm of the object's (after the retreat it is usually behind the object).
    const wall=this.scene.tray.wallTopZ;
    if(!this.releaseExit?.objectIgnored)return Math.max(wall,this.objectBounds().upper[2]);
    const data=this.releaseMeasuredData(),hand=geomBounds(this.scene,data,this.releaseMovingGeoms()),object=geomBounds(this.scene,data,this.audit.physicalObjectGeoms(this.scene.objects[this.objectId]));
    const xyGap=Math.max(...[0,1].flatMap(k=>[hand.lower[k]-object.upper[k],object.lower[k]-hand.upper[k]]));
    this.releaseExit.clearTop={wall_top_z:wall,object_top_z:object.upper[2],hand_object_xy_gap_m:xyGap,object_top_required:!(xyGap>=.02)};
    return xyGap>=.02?wall:Math.max(wall,object.upper[2]);
  }
  releaseRiseDelta(){
    const bounds=geomBounds(this.scene,this.scene.data,this.releaseMovingGeoms()),top=this.releaseClearTop();
    return [0,0,Math.max(.08,top+.035-bounds.lower[2])];
  }
  releaseMeasuredData(){
    // Native live derived geometry may precede the latest integrated qpos.
    // Capture and compare every separation distance in the same coherent FK
    // frame, without forwarding or otherwise writing the live scene.
    const data=this.homeAuditData;data.qpos.set(this.scene.data.qpos);this.scene.forward(data);return data;
  }
  releaseSupportAudit(){
    const scene=this.scene,object=this.audit.physicalObjectGeoms(scene.objects[this.objectId]),own=new Set(object),data=this.releaseMeasuredData(),bounds=geomBounds(scene,data,object),bottom=bounds.lower[2]-scene.tray.bottomTopZ;
    const contacts=scene.contacts().filter(c=>(own.has(c.geom1)&&c.geom2===scene.trayBottomGeomId||own.has(c.geom2)&&c.geom1===scene.trayBottomGeomId)&&Number.isFinite(c.normalForce)&&c.normalForce>.02&&Number.isFinite(c.dist)&&c.dist>=-.003);
    const v=copy(scene.data.qvel.slice(scene.objects[this.objectId].vadr,scene.objects[this.objectId].vadr+6)),linear=norm(v.slice(0,3)),angular=norm(v.slice(3)),overlap=objectTrayOverlap(scene,this.objectId,data);
    const supported=contacts.length>0&&Number.isFinite(bottom)&&bottom>=-.003&&bottom<=.0005&&Number.isFinite(overlap)&&overlap>1e-8;
    return {supported,quiet:v.length===6&&v.every(Number.isFinite)&&linear<.04&&angular<.25,bottom_clearance_m:bottom,bottom_contacts:contacts.length,overlap_m2:overlap,linear_speed_m_s:linear,angular_speed_rad_s:angular};
  }
  releaseContactNormal(data,a,b){
    if(!Number.isInteger(data.ncon)||data.ncon<=0)return null;
    const contacts=data.contact;
    try{for(let i=0;i<Math.min(data.ncon,256);i++){const contact=contacts.get(i);try{
      if(!(contact.geom1===a&&contact.geom2===b||contact.geom1===b&&contact.geom2===a))continue;
      const normal=copy(contact.frame?.slice(0,3)??[]);if(normal.length===3&&normal.every(Number.isFinite)&&norm(normal)>.5)return scale(normal,(contact.geom1===a?-1:1)/norm(normal));
    }finally{contact.delete?.();}}}finally{contacts.delete?.();}return null;
  }
  releaseSeparationContract({allowInitialContact=false}={}){
    const data=this.releaseMeasuredData(),pairs=[],directions=[],object=this.audit.physicalObjectGeoms(this.scene.objects[this.objectId]),contacts=this.scene.contacts?.()??[];
    const contactKeys=new Set(contacts.filter(c=>c.dist<=.0002).map(c=>`${Math.min(c.geom1,c.geom2)}:${Math.max(c.geom1,c.geom2)}`));
    for(const a of this.releaseMovingGeoms())for(const b of object){
      const info=this.scene.geomDistanceInfo(data,a,b,.01),distance=info?.distance;
      if(!Number.isFinite(distance))throw new IKError('The open hand distance is missing or non-finite.',{a,b,distance});
      const existingContact=allowInitialContact&&contactKeys.has(`${Math.min(a,b)}:${Math.max(a,b)}`)&&distance<=.0002;
      if(distance<(existingContact?-.0002:0))throw new IKError('The open hand is outside the bounded existing-contact relief band.',{a,b,distance,code:allowInitialContact?'wait_for_contact_relief_band':null});
      if(distance>=.005)continue;
      if(!info.hasWitness||info.fromto?.length!==6||!info.fromto.every(Number.isFinite))throw new IKError('A close release pair lacks native separating witnesses.',{a,b,distance});
      let away=scale(sub(info.fromto.slice(0,3),info.fromto.slice(3)),distance<0?-1:1);
      if(norm(away)<1e-12)away=this.releaseContactNormal(data,a,b)??[0,0,0];const length=norm(away);
      if(!(length>0))throw new IKError('A close release pair has no separating direction.',{a,b,distance});
      pairs.push(Object.freeze({a,b,minimumDistance:distance,...(existingContact?{existingContact:true}:{})}));directions.push(scale(away,(.007-distance)/length));
    }
    return {contract:pairs.length?Object.freeze({hand:this.hand,objectId:this.objectId,pairs:Object.freeze(pairs),...(pairs.some(p=>p.existingContact)?{initialContactRelief:true}:{})}):null,directions};
  }
  captureReleaseCollisionFrame(referenceFrame=this.reference.currentFrame()){
    const data=this.releaseMeasuredData(),freeze=v=>Object.freeze(copy(v));
    return Object.freeze({hand:this.hand,objectId:this.objectId,measuredRootPosition:freeze(data.qpos.slice(0,3)),measuredRootRotation:freeze(quatToMat(copy(data.qpos.slice(3,7)))),referenceRootPosition:freeze(referenceFrame.rootPosW),referenceRootRotation:freeze(quatToMat(referenceFrame.rootQuatW)),measuredLowerQ:freeze(this.reference.ik.qadr.slice(0,12).map(a=>data.qpos[a])),referenceLowerQ:freeze(referenceFrame.jointPos.slice(0,12)),measuredUpperQ:freeze(this.reference.ik.qadr.slice(12).map(a=>data.qpos[a])),referenceUpperQ:freeze(referenceFrame.jointPos.slice(12))});
  }
  releaseArticulatedAudit(goal,separation,{collisionFrame=null,frames=null}={}){
    if(frames&&!collisionFrame)throw new IKError('An articulated release window requires its explicit candidate frame.');
    const ik=this.reference.ik,data=this.releaseMeasuredData(),actual=ik.qadr.map(a=>data.qpos[a]),goals=(frames?frames.map(f=>f.jointPos):[goal]).map(q=>this.releasePredictedQ(q,collisionFrame));
    if(collisionFrame){
      if(norm(sub(copy(data.qpos.slice(0,3)),collisionFrame.measuredRootPosition))>1e-9||norm(rotationError(quatToMat(copy(data.qpos.slice(3,7))),collisionFrame.measuredRootRotation))>1e-9||actual.slice(0,12).some((v,i)=>Math.abs(v-collisionFrame.measuredLowerQ[i])>1e-9)||actual.slice(12).some((v,i)=>Math.abs(v-collisionFrame.measuredUpperQ[i])>1e-9))return {passed:false,reason:'stale_measured_release_frame'};
      if(frames?.some(f=>f.rootPosW?.length!==3||f.rootQuatW?.length!==4||f.jointPos?.length!==29||![...f.rootPosW,...f.rootQuatW,...f.jointPos].every(Number.isFinite)||norm(sub(f.rootPosW,collisionFrame.referenceRootPosition))>1e-9||norm(rotationError(quatToMat(f.rootQuatW),collisionFrame.referenceRootRotation))>1e-9||f.jointPos.slice(0,12).some((v,i)=>Math.abs(v-collisionFrame.referenceLowerQ[i])>1e-9)))return {passed:false,reason:'release_nominal_root_or_lower_body_changed'};
    }
    const feet=SIDES.map(side=>bodyPose(data,this.scene.ankleBodyIds[side])),previous=new Map();let minimumComMargin=Infinity,maximumFootDrift=0,samples=0,start=actual;
    for(let frame=0;frame<goals.length;frame++){
      const end=goals[frame],steps=Math.max(1,Math.ceil(Math.max(...end.slice(12).map((v,i)=>Math.abs(v-start[12+i])))/.01));
      if(!Number.isFinite(steps)||samples+steps+1>512)return {passed:false,reason:'articulated_sampling_budget_exhausted',maximumSamples:512};
      for(let i=0;i<=steps;i++){
        const q=lerp(start,end,i/steps),endpoint=frame===goals.length-1&&i===steps;
        if(q.slice(12).some((v,j)=>!Number.isFinite(v)||v<ik.lower[12+j]-1e-9||v>ik.upper[12+j]+1e-9))return {passed:false,reason:'actual_upper_joint_limit',frame,step:i};
        const geometry=this.releaseActualPoseAudit(q,endpoint?null:separation,endpoint?null:previous);samples++;
        if(!geometry.passed)return {passed:false,frame,step:i,geometry};
        const com=polygonMargin(copy(data.subtree_com.slice(ik.rootBodyId*3,ik.rootBodyId*3+3)),ik.supportPolygon);minimumComMargin=Math.min(minimumComMargin,com);
        for(let j=0;j<SIDES.length;j++){const foot=bodyPose(data,this.scene.ankleBodyIds[SIDES[j]]);maximumFootDrift=Math.max(maximumFootDrift,norm(sub(foot.position,feet[j].position)));if(norm(rotationError(foot.rotation,feet[j].rotation))>1e-8)return {passed:false,reason:'actual_foot_rotation_changed',frame,step:i};}
        if(!Number.isFinite(com)||com<.02||maximumFootDrift>1e-8)return {passed:false,reason:'actual_support_changed',frame,step:i,com,maximumFootDrift};
      }
      start=end;
    }
    return {passed:true,samples,maximumJointStepRad:.01,minimumComMargin,maximumFootDrift,endpointPalm:this.palm(this.hand,data)};
  }
  planReleaseReconciliation(delta,separation,{stage='separate',duration=1.2}={}){
    if(!['separate','withdraw','rise','retreat'].includes(stage)||!Number.isFinite(duration)||duration<1.2)throw new IKError('Invalid checked release stage or duration.');
    const reference=this.reference,current=reference.currentFrame(),issued=reference.clip?.sample(reference.currentIndex+1)??current,collisionFrame=this.captureReleaseCollisionFrame(issued);
    const from=Object.fromEntries(SIDES.map((side,i)=>[side,{position:copy(issued.palmPosW[i]),rotation:quatToMat(issued.palmQuatW[i])}])),goal=add(from[this.hand].position,delta);
    const check={stage,planner:'measured_reference_reconciliation',delta:delta.slice(),margin:this.releaseExit?.margin??.005,ignoreObjectId:this.ignoredObjectId,sweep:this.releaseSweepAudit(delta,{separation})};this.releaseExit.checks.push(check);if(!check.sweep.passed)return false;
    const initialBounds=Object.freeze({lower:Object.freeze(current.jointPos.map((v,i)=>Math.max(v,issued.jointPos[i])-.035)),upper:Object.freeze(current.jointPos.map((v,i)=>Math.min(v,issued.jointPos[i])+.035))});
    const targetAt=t=>({...this.releasePathTarget(from,goal,this.releasePathProgress(t,duration,separation),separation,collisionFrame),posture:copy(issued.jointPos),...(t<=this.dt+1e-9?{releaseJointBounds:initialBounds}:{})});
    // The trial wrapper shares only independent IK scratch. Failed planning
    // cannot replace committed frames, qCurrent, anchors, or live physics.
    // The measured full-pose anchor makes the already-issued frame a valid
    // start under the captured pair floors. Preserve the rolling reference's
    // committed-frame logic rather than re-solving that command at the splice.
    const trial=Object.assign(Object.create(Object.getPrototypeOf(reference)),reference,{horizon:Math.max(reference.horizon,Math.ceil(duration/this.dt)),qCurrent:copy(issued.jointPos),rootPosition:copy(issued.rootPosW),rootHeight:issued.rootPosW[2],rootPitch:issued.plannedRootPitch,lastAudit:null});
    try{
      const clip=trial.window(targetAt,nowOf(this.scene),{force:true,bestEffort:false,initialFrameMaxStep:.0175});
      const first=clip.sample(0),next=clip.sample(1),max=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
      // Frame zero preserves the already-issued command, including its legs.
      // Compare immobility to that same captured anchor, while separately
      // recording any motion already scheduled from the prior current frame.
      check.reference={audit:trial.lastAudit,maximumCurrentFrameChange:max(first.jointPos,current.jointPos),maximumIssuedCommandChange:max(next.jointPos,issued.jointPos),lowerBodyChange:max(first.jointPos.slice(0,12),issued.jointPos.slice(0,12)),maximumCurrentLowerBodyChange:max(first.jointPos.slice(0,12),current.jointPos.slice(0,12)),lowerBodyAnchor:'already_issued_reference',previousCurrentQ:copy(current.jointPos),previousIssuedQ:copy(issued.jointPos),newCurrentQ:copy(first.jointPos),newIssuedQ:copy(next.jointPos)};
      if(check.reference.maximumCurrentFrameChange>.035000001||check.reference.maximumIssuedCommandChange>.035000001||check.reference.lowerBodyChange>1e-10)return false;
      // Full clearance belongs to the planned segment endpoint. Audit every
      // intervening physical increment with the candidate's explicit frame.
      check.actualPath=this.releaseArticulatedAudit(clip.frames.at(-1).jointPos,separation,{collisionFrame,frames:clip.frames});if(!check.actualPath.passed)return false;
      const actualStart=this.palm().position.slice();
      this.reference=trial;this.releaseHoldQ=copy(issued.jointPos);this.releaseExit.collisionFrame=collisionFrame;
      this.releaseExit.segment={from,goal,delta:sub(check.actualPath.endpointPalm.position,actualStart),actualStart,startedAt:nowOf(this.scene),duration,separation,reconciled:true};
      this.releaseExit.stage=stage;this.releaseExit.stable=0;this.releaseExit.separationProgress=Object.fromEntries((separation?.pairs??[]).map(p=>[`${p.a}:${p.b}`,p.minimumDistance]));this.stage=stage;this.forceReference=false;return true;
    }catch(error){if(!(error instanceof IKError))throw error;check.rejected={message:error.message,details:error.details};return false;}
  }
  async planReleaseSeparation({allowInitialContact=false}={}){
    let captured;try{captured=this.releaseSeparationContract({allowInitialContact});}catch(error){if(!(error instanceof IKError))throw error;this.releaseExit.separationError={message:error.message,details:error.details};return error.details?.code==='wait_for_contact_relief_band'?null:false;}
    if(!captured.contract)return this.objectId==='uiuc_i'?this.planReleaseSegment([0,0,0],'separate'):this.planReleaseExit();
    // A measured near-contact state is not yet a full-clearance state. Restore
    // the ordinary margin along native outward normals before planning exit;
    // no new overlap is allowed. The native planned path is monotonic from
    // each captured floor; measured contact compliance is guarded separately.
    const combined=captured.directions.reduce((sum,d)=>add(sum,d),[0,0,0]);
    const candidates=[combined,...captured.directions].filter((d,i,a)=>norm(d)>0&&norm(d)<=.03&&a.findIndex(v=>norm(sub(v,d))<1e-9)===i).slice(0,4);
    for(const delta of candidates)if(await this.planReleaseSegment(delta,'separate',captured.contract)){this.releaseExit.separationProgress=Object.fromEntries(captured.contract.pairs.map(p=>[`${p.a}:${p.b}`,p.minimumDistance]));return true;}
    return false;
  }
  releaseMeasuredClearance(){
    const separation=this.releaseExit?.stage==='separate'?this.releaseExit.segment.separation:null;
    if(!separation)return this.releaseSweepAudit([0,0,0]);
    const data=this.releaseMeasuredData(),runtimePairs=[],measurements=[];
    const tracking=this.releaseExit.separationTracking??={semantics:'bounded_measured_tracking',measurementIntervalS:this.dt,contactReliefBandM:.0002,positiveGapRollbackM:.0002,fullClearanceQuietDwellS:.2,maximumDurationS:this.releaseExit.segment.duration+3,pairs:{}};
    for(const pair of separation.pairs){
      const key=`${pair.a}:${pair.b}`,distance=this.audit.nativeDistance(data,pair.a,pair.b,.01),earned=this.releaseExit.separationProgress[key],inherited=separation.initialContactRelief===true&&pair.existingContact===true;
      if(!Number.isFinite(distance)||!Number.isFinite(earned)||!Number.isFinite(pair.minimumDistance)||pair.minimumDistance<(inherited?-.0002:0)||pair.minimumDistance>=.005||earned<pair.minimumDistance||earned>.005)throw new IKError('Invalid measured release tracking state.',{pair,distance,earned});
      // Planned floors stay immutable and monotonic. Live positive gaps may
      // roll back by the existing 0.2 mm contact/compliance scale, never into
      // penetration. Witnessed nonpositive contacts retain their relief band.
      // Once 5 mm is earned it stays required.
      const floor=earned>=.005?.005:inherited&&pair.minimumDistance<=0?-.0002:Math.max(0,pair.minimumDistance-.0002);
      const record=tracking.pairs[key]??={capturedDistanceM:pair.minimumDistance,minimumDistanceM:pair.minimumDistance,maximumDistanceM:pair.minimumDistance,maximumRegressionM:0,maximumCapturedFloorRegressionM:0,maximumPenetrationM:Math.max(0,-pair.minimumDistance)};
      record.maximumRegressionM=Math.max(record.maximumRegressionM,record.maximumDistanceM-distance);record.minimumDistanceM=Math.min(record.minimumDistanceM,distance);record.maximumDistanceM=Math.max(record.maximumDistanceM,distance);record.maximumCapturedFloorRegressionM=Math.max(record.maximumCapturedFloorRegressionM,pair.minimumDistance-distance);record.maximumPenetrationM=Math.max(record.maximumPenetrationM,-distance);record.runtimeFloorM=floor;record.fullClearanceEarned=earned>=.005;
      if(pair.minimumDistance>0&&distance<0)return {passed:false,reason:'positive_release_gap_penetrated',limitingPair:{...pair,distance,requiredMargin:floor}};
      if(floor<.005)runtimePairs.push({...pair,minimumDistance:floor});
      measurements.push({key,distance,earned,record});
    }
    const runtimeSeparation=runtimePairs.length?{...separation,pairs:runtimePairs}:null;
    const geometry=this.audit.check(data,{hand:this.hand,objectId:this.objectId,allowHandObject:false,includeSelf:true,margin:.005,releaseSeparation:runtimeSeparation,ignoreObjectId:this.ignoredObjectId});
    if(!geometry.passed)return geometry;
    for(const {key,distance,earned,record}of measurements){this.releaseExit.separationProgress[key]=Math.min(.005,Math.max(earned,distance));record.fullClearanceEarned=this.releaseExit.separationProgress[key]>=.005;}
    return geometry;
  }
  releaseContactViolation(contacts=this.scene.contacts?.()??[]){
    // Once the released object is ignored this is evidence only (recorded as 'release_object_contact'); it never gates motion.
    const moving=new Set(this.releaseMovingGeoms()),object=new Set(this.audit.physicalObjectGeoms(this.scene.objects[this.objectId])),separation=this.releaseExit?.stage==='separate'?this.releaseExit.segment?.separation:null;
    for(const contact of contacts){
      const a=moving.has(contact.geom1)&&object.has(contact.geom2)?contact.geom1:moving.has(contact.geom2)&&object.has(contact.geom1)?contact.geom2:null;if(a===null)continue;
      if(!Number.isFinite(contact.dist))return {a,distance:contact.dist,reason:'nonfinite_contact_distance'};
      if(!(contact.dist<=.0002||contact.normalForce>.02))continue;const b=contact.geom1===a?contact.geom2:contact.geom1;
      const inherited=separation?.initialContactRelief&&separation.pairs.find(p=>p.a===a&&p.b===b&&p.existingContact);
      const floor=inherited?.minimumDistance>0?inherited.minimumDistance:-.0002;
      if(!inherited||!(this.releaseExit.separationProgress[`${a}:${b}`]<.005)||!Number.isFinite(contact.dist)||contact.dist<floor)return {a,b,distance:contact.dist,normalForce:contact.normalForce};
    }
    return null;
  }
  releaseRiseClearance(){
    const hand=geomBounds(this.scene,this.releaseMeasuredData(),this.releaseMovingGeoms()),clearance=hand.lower[2]-this.releaseClearTop();
    return {passed:Number.isFinite(clearance)&&clearance>=.025,clearance_m:clearance,required_clearance_m:.025};
  }
  async planReleaseExit(){
    if(await this.planReleaseSegment(this.releaseRiseDelta(),'rise'))return true;
    if(this.objectId!=='uiuc_i')return false;
    const actual=this.palm().position,object=poseOf(this.scene,this.objectId).position,away=sub(actual,object);away[2]=0;
    const directions=[...(norm(away)>.001?[scale(away,1/norm(away))]:[]),[-1,0,0],[0,this.hand==='left'?1:-1,0]];
    const handPoints=this.releaseMovingGeoms().flatMap(g=>geomCorners(this.scene,this.scene.data,g));
    const objectPoints=this.audit.physicalObjectGeoms(this.scene.objects[this.objectId]).flatMap(g=>geomCorners(this.scene,this.scene.data,g));
    for(const direction of directions){
      // Project the complete shapes: move the nearest part of the hand beyond
      // the farthest object part, instead of guessing a flange escape length.
      const distance=Math.max(...objectPoints.map(p=>dot(p,direction)))-Math.min(...handPoints.map(p=>dot(p,direction)))+.01;
      if(!(distance>0&&distance<=.25))continue;
      if(await this.planReleaseSegment(scale(direction,distance),'withdraw'))return true;
    }
    return false;
  }
  async advanceRelease(){
    const now=nowOf(this.scene),exit=this.releaseExit;
    if(this.checkPostReleaseDeadline())return;
    this.releaseHandOpening=this.homeHandOpeningAudit(this.hand);
    if(exit.stage==='opening'){
      exit.supportAudit=this.releaseSupportAudit();exit.supportStable=exit.supportAudit.supported&&exit.supportAudit.quiet?(exit.supportStable??0)+this.dt:0;
      const open=this.releaseHandOpening.opened&&this.releaseHandOpening.quiet,elapsed=now-this.phaseTime;
      const ready=elapsed>=.8&&open&&exit.supportStable>=.1,timedOut=elapsed>=RELEASE_OPENING_DEADLINE_S;
      if(!ready&&!timedOut)return;
      exit.ignoreGate={time_s:now,ready,timedOut,open,supportStable:exit.supportStable,support:exit.supportAudit,handContacts:this.handContacts,opening:this.releaseHandOpening};
      // Once the fingers are commanded open the released object is no longer an obstacle. At the opening deadline this
      // holds even if the fingers are not verified open: a pinched object may be dragged, and the placement is judged
      // from the object's final state by the unchanged measurement.
      exit.objectIgnored=true;exit.ignoredAt=now;
      if(!ready)this.recordPostReleaseHazard(open?'release_support_timeout':'release_opening_timeout');
      if(await this.planReleaseRetreat())return;
      await this.planReleaseRise();return;
    }
    const geometry=this.releaseMeasuredClearance(),contactViolation=this.releaseContactViolation();
    if(!geometry.passed)this.recordPostReleaseHazard('release_exit_clearance',{geometry});
    if(contactViolation)this.recordPostReleaseHazard('release_object_contact',{limitingPair:contactViolation});
    if(!this.releaseHandOpening.opened)this.recordPostReleaseHazard('release_hand_reclosed');
    exit.lastMeasured={time_s:now,geometry,contactViolation,opening:this.releaseHandOpening};
    const segment=exit.segment,elapsed=now-segment.startedAt,reached=norm(sub(this.palm().position,add(segment.actualStart,segment.delta)))<.04;
    const separated=exit.stage!=='separate'||this.releaseSweepAudit([0,0,0]).passed;
    if(exit.stage==='rise')exit.verticalClearance=this.releaseRiseClearance();
    const stageReached=exit.stage==='separate'?true:exit.stage==='rise'?exit.verticalClearance.passed:reached;
    exit.stable=stageReached&&separated&&this.releaseHandOpening.quiet?exit.stable+this.dt:0;
    const backward=()=>{if(exit.retreat&&exit.stage==='retreat')exit.retreat.reached_m=segment.actualStart[0]-this.palm().position[0];};
    if(elapsed>=segment.duration&&exit.stable>=.2){
      if(exit.stage==='separate'){exit.segment=null;exit.separationProgress=null;if(!await this.planReleaseExit()){this.recordPostReleaseHazard('release_exit_unplanned');this.beginClearTray();}}
      else if(['retreat','withdraw'].includes(exit.stage)){backward();await this.planReleaseRise();}
      else this.beginClearTray();
    }else if(elapsed>=segment.duration+RELEASE_STAGE_SLACK_S){
      // Every checked stage hands over within a fixed budget: a stalled rise goes to the un-audited clear-tray lift,
      // a stalled retreat/withdraw/separation to the rise plan. Nothing here stops the episode.
      this.recordPostReleaseHazard(`release_${exit.stage}_timeout`,{geometry});
      if(exit.stage==='rise')this.beginClearTray();
      else{backward();if(exit.stage==='separate'){exit.segment=null;exit.separationProgress=null;}await this.planReleaseRise();}
    }
  }
  releaseTarget(offset=0){
    if(this.graspAbort){
      const a=this.graspAbort,palms=structuredClone(a.palms),elapsed=nowOf(this.scene)-a.startedAt+offset;
      palms[this.hand].position=lerp(a.palms[this.hand].position,a.retreat,smooth((elapsed-.8)/1.2));
      return this.baseTarget(palms,{posture:a.posture,allowHandObject:true,lockWaist:true,jointContinuityWeight:2});
    }
    const segment=this.releaseExit?.segment;
    if(!segment){
      const frame=this.releaseExit?.openingCollisionFrame;
      return this.baseTarget(structuredClone(this.releasePalms),{posture:this.releaseHoldQ,allowHandObject:true,lockWaist:true,jointContinuityWeight:1,...(frame?{releaseOpeningCollisionFrame:frame,lockLowerBody:true,rootPosition:frame.referenceRootPosition,rootHeight:frame.referenceRootPosition[2],rootPitch:this.releaseExit.openingRootPitch}:{})});
    }
    return this.releasePathTarget(segment.from,segment.goal,this.releasePathProgress(nowOf(this.scene)-segment.startedAt+offset,segment.duration,segment.separation),segment.separation);
  }
  beginClearTray(){
    if(this.releaseExit)this.releaseExit.segment=null;
    this.clearTrayFrom=this.plannedPalms();this.clearTrayGoals=structuredClone(this.clearTrayFrom);this.clearTrayHoldQ=this.reference.qCurrent.slice();this.clearTrayStable=0;
    this.tableFront=geomBounds(this.scene,this.scene.data,this.scene.tableGeomIds).lower[0];
    this.clearTrayTop=Math.max(tableZ(this.scene),this.releaseClearTop());
    for(const side of SIDES){
      const b=this.audit.handBounds(this.scene.data,side);
      // Keep a safely stowed inactive arm still. The loaded hand clears the
      // complete object/rim envelope at fixed XY before any return motion.
      const obstacles=[...this.audit.fixtureGeoms,...this.audit.objectGeoms(this.ignoredObjectId)],unsafe=this.audit.hands[side].some(a=>obstacles.some(b=>this.audit.nativeDistance(this.scene.data,a,b,.006)<.005));
      if(side===this.hand||unsafe)
        this.clearTrayGoals[side].position[2]+=Math.max(0,this.clearTrayTop+.035-b.lower[2]);
      this.scene.setHandClosure(side,0);
    }
    this.enter('settle','Lifting the open hand vertically clear of the object and tray.');this.stage='clear_tray';
  }
  clearTrayTarget(offset=0){
    const alpha=smooth((nowOf(this.scene)-this.phaseTime+offset)/1.2),palms={};
    for(const side of SIDES)palms[side]={position:lerp(this.clearTrayFrom[side].position,this.clearTrayGoals[side].position,alpha),rotation:this.clearTrayFrom[side].rotation};
    return this.baseTarget(palms,{posture:this.clearTrayHoldQ,allowHandObject:!this.releaseExit,jointContinuityWeight:1});
  }
  async advanceClearTray(){
    if(this.checkPostReleaseDeadline())return;
    const elapsed=nowOf(this.scene)-this.phaseTime,b=this.audit.handBounds(this.scene.data,this.hand),geometry=this.audit.check(this.scene.data,{handsOnly:true,includeSelf:false,margin:.005,ignoreObjectId:this.ignoredObjectId});
    const touching=this.handContacts!==0&&!this.releaseExit?.objectIgnored,clear=b.lower[2]>=this.clearTrayTop+.025&&geometry.passed&&!touching;
    this.clearTrayStable=clear?this.clearTrayStable+this.dt:0;
    if(elapsed>=1.2&&this.clearTrayStable>=.2){await this.beginReturn();return;}
    if(elapsed>=5){
      // Clearing above the complete object bounding box is sufficient, but
      // not necessary. A clearance shortfall here is recorded and answered by
      // the return's own audited withdrawal; it never stops the episode.
      if(geometry.passed&&!touching)this.markCompletionFallback('clear_tray_path_retry');else this.recordPostReleaseHazard('clear_tray_clearance_timeout',{geometry});
      await this.beginReturn();
    }
  }
  measureLowerSupport(bounds){
    const now=nowOf(this.scene),clearance=bounds.lower[2]-this.scene.tray.bottomTopZ,overlap=this.objectTrayOverlap();
    const o=this.scene.objects[this.objectId],v=copy(this.scene.data.qvel.slice(o.vadr,o.vadr+6));
    const finite=v.length===6&&v.every(Number.isFinite)&&Number.isFinite(clearance)&&Number.isFinite(overlap);
    const atBottom=finite&&clearance>=-.0005&&clearance<=.0005&&overlap>1e-8;
    const grounded=atBottom&&this.trayBottomContacts>0;
    if(grounded)this.lowerSupportContactTime=now;
    const age=this.lowerSupportContactTime==null?Infinity:now-this.lowerSupportContactTime;
    // Native contacts can disappear for a few substeps while the held object
    // stays on the tray plane. Require fresh real contact plus measured geometry,
    // instead of continually pushing down and resetting on every contact edge.
    const supported=atBottom&&age>=0&&age<=.1+1e-9;
    if(!supported)this.lowerSupportContactTime=null;
    const linear=finite?norm(v.slice(0,3)):Infinity,angular=finite?norm(v.slice(3)):Infinity;
    this.lowerSupport={bottom_clearance_m:Number.isFinite(clearance)?clearance:null,overlap_m2:Number.isFinite(overlap)?overlap:null,
      current_bottom_contact:grounded,recent_contact_age_s:Number.isFinite(age)?age:null,supported,
      linear_speed_m_s:Number.isFinite(linear)?linear:null,angular_speed_rad_s:Number.isFinite(angular)?angular:null};
    return {...this.lowerSupport,quiet:grounded&&linear<.04&&angular<.25,steady:supported&&linear<.04};
  }
  advanceTransfer(){
    const seg=this.transferSegments[this.transferIndex],now=nowOf(this.scene),elapsed=now-this.transferTime;this.stage=seg.name;
    if(!this.completionTransfer&&this.stableDeposit>=.8){this.beginRelease();return;}
    this.transferStageStarted??=this.transferTime;this.transferBudgetStarted??=this.transferStageStarted;
    // A measured deposit disproves a blocked placement. Keep the existing
    // completion release/return sequence instead of shortening its trajectory.
    if(this.stableDeposit>=.8)this.transferObstruction?.reset();
    const obstruction=this.stableDeposit>=.8?null:this.observeTransferObstruction();
    if(obstruction){this.failObstructed(obstruction);return;}
    // These clocks belong to the attempt, not its interpolation. Clearing the
    // rim may restart interpolation but must not grant another unlimited wait.
    if(now-this.transferBudgetStarted>=45){
      this.failureReason={code:'transfer_stalled',phase:this.phase,stage:this.stage,reset_required:true};
      this.fail('Motion could not complete after repeated recovery attempts. Please reset the scene.');return;
    }
    if(this.completionTransfer){this.advanceCompletionTransfer();return;}
    if(now-this.transferStageStarted>seg.duration+(seg.name==='raise_load'?3:12)){
      this.beginCompletionTransfer('transfer_stage_timeout');return;
    }
    if(this.stableLift===0&&this.lift<.025&&seg.name!=='lower_load'){this.beginCompletionTransfer('transport_slip');return;}
    const bounds=this.objectBounds(),tray=this.scene.tray;
    let lowerSupport=null;
    if(seg.name==='lower_load'&&this.objectId==='uiuc_i'&&!this.highRelease){
      lowerSupport=this.measureLowerSupport(bounds);
      // Entering or leaving measured support changes the descent request.
      // A single missing native contact does not restart a whole IK horizon.
      if(lowerSupport.supported!==(this.lowerContactGrounded??false))this.forceReference=true;
      this.lowerContactGrounded=lowerSupport.supported;
    }
    if(seg.name==='over_tray'){
      if(!this.clearanceRecovery&&bounds.lower[2]<tray.wallTopZ+.02){const start=sub(this.transferTarget().palms[this.hand].position,this.transferCorrectionAt());this.clearanceRecovery={start,goal:add(start,[0,0,tray.wallTopZ+.05-bounds.lower[2]]),time:now};this.forceReference=true;}
      if(this.clearanceRecovery){
        const r=this.clearanceRecovery;if(bounds.lower[2]>=tray.wallTopZ+.04){const current=sub(this.transferTarget().palms[this.hand].position,this.transferCorrectionAt());seg.position[2]=Math.max(seg.position[2],current[2]);this.transferFrom=current;this.transferTime=now;this.clearanceRecovery=null;this.forceReference=true;return;}
        else{if(now-r.time>.8)this.applyTransferShift([0,0,.0008],{propagate:false});this.message='Pausing sideways motion and lifting the object clear of the tray rim.';if(now-r.time>=3)this.beginCompletionTransfer('tray_rim_timeout');return;}
      }
    }
    if(this.replanEnabled&&['over_tray','align_upright'].includes(seg.name)&&now>=this.nextTransferAdjust&&elapsed>=seg.duration){
      this.nextTransferAdjust=now+.5;const position=poseOf(this.scene,this.objectId).position,palm=this.palm().position,offset=sub(position,palm),trueGoal=[tray.center[0]-offset[0],tray.center[1]-offset[1],palm[2]];
      if(!this.transferAdjuster||this.transferAdjustStage!==seg.name||Math.hypot(...sub(trueGoal,this.transferAdjuster.original).slice(0,2))>.003){this.transferAdjuster=new GoalAdjuster(trueGoal,{enabled:this.goalAdjustEnabled,stay:.004,skip:.006,maxStep:.005});this.transferAdjustStage=seg.name;}
      const measured=palm.slice();measured[2]=this.transferAdjuster.original[2];const update=this.transferAdjuster.update(measured);
      if(update.action==='replan'){this.replans++;this.applyTransferShift(update.shift);this.goalAdjustment=add(this.goalAdjustment,update.shift);this.gaUpdates+=Number(norm(update.shift)>0);this.forceReference=true;}
    }
    if(seg.name==='lower_load'&&this.highRelease){const height=bounds.lower[2]-tray.bottomTopZ;
      if(height>=.045&&height<=this.dropClearance&&this.handContacts>0&&this.trayContacts===0&&this.objectTrayOverlap()>1e-8){this.beginRelease();return;}
      if(elapsed>=seg.duration&&height<.045){this.applyTransferShift([0,0,.0008],{propagate:false});this.message='Regaining an airborne release height before opening the fingers.';if(elapsed>=seg.duration+3)this.beginCompletionTransfer('release_height_timeout');return;}
      // A nominal waypoint duration cannot replace the measured airborne,
      // held, over-tray gate. Keep the same bounded recovery for tracking loss.
      if(elapsed>=seg.duration+3)this.beginCompletionTransfer('release_pose_timeout');
      return;
    }
    if(elapsed<seg.duration)return;
    let gate=true;if(seg.name==='raise_load')gate=bounds.lower[2]>=tray.wallTopZ+.02;
    if(['over_tray','align_upright'].includes(seg.name))gate=this.objectTrayOverlap()>(this.objectId==='cracker_box'?CRACKER_BOX_TRAY_OVERLAP_MIN_M2:1e-8);
    if(!gate){if(elapsed>seg.duration+(seg.name==='raise_load'?3:12))this.beginCompletionTransfer('tray_position_timeout');return;}
    if(seg.name==='align_upright'){
      const position=poseOf(this.scene,this.objectId).position,palm=this.palm().position,offset=sub(position,palm),bias=sub(seg.position,palm),lower=this.transferSegments.at(-1).position;
      lower[0]=tray.center[0]-offset[0]+bias[0];lower[1]=tray.center[1]-offset[1]+bias[1];lower[2]=tray.bottomTopZ+this.dropClearance+position[2]-bounds.lower[2]-offset[2]+bias[2];
    }
    if(seg.name==='lower_load'&&this.objectId==='uiuc_i'&&!this.highRelease){
      // Keep the same continuous micro-adjustments, consumed by the normal
      // five-tick reference refresh to avoid solving the full horizon
      // again on every 20 ms control tick.
      if(!lowerSupport.supported)this.applyTransferShift([0,0,-.0008],{propagate:false,forceReference:false});
      this.lowerContactHold=lowerSupport.steady?this.lowerContactHold+this.dt:0;
      // A held object may rock against the fingers before release. Validate
      // sustained physical support, then open only on a current quiet contact.
      // Released-object settling is measured separately after the hand opens.
      if(this.lowerContactHold<.3||!lowerSupport.quiet){if(elapsed>seg.duration+5)this.beginCompletionTransfer('tray_support_timeout');return;}
    }
    if(this.transferIndex===this.transferSegments.length-1){this.beginRelease();return;}
    if(seg.name==='align_upright'&&this.unwindRotation)this.depositRotation=eye3();this.transferFrom=seg.position.slice();this.transferIndex++;this.transferTime=now;this.transferAdjustment=[0,0,0];this.forceReference=true;
    this.startTransferStage();
  }
  actualRootPostureAudit(posture=this.homeQ){
    // An independent FK query asks whether the requested upper posture is safe
    // at the *measured* root. It never moves the live robot or the object.
    const data=this.homeAuditData;data.qpos.set(this.scene.data.qpos);
    this.reference.ik.qadr.slice(12).forEach((address,i)=>data.qpos[address]=posture[12+i]);
    for(const address of this.scene.fingerQposAddresses14||[])data.qpos[address]=this.reference.ik.template[address];
    this.scene.forward(data);const geometry=this.audit.check(data,{handsOnly:true,includeSelf:false,margin:.005,ignoreObjectId:this.ignoredObjectId});
    return {passed:geometry.passed,minimumClearance:geometry.minimumClearance,limitingPair:geometry.limitingPair,root:copy(data.qpos.slice(0,7)),palms:Object.fromEntries(SIDES.map(s=>[s,this.palm(s,data)])),bounds:Object.fromEntries(SIDES.map(s=>[s,this.audit.handBounds(data,s)]))};
  }
  async preflightHomePath(targetAt,duration,{checkActualRoot=false,strictEndpoint=true,bestEffort=false,measuredPoseAudit=null}={}){
    const ik=this.reference.ik,start=this.reference.qCurrent.slice();
    const measuredAudit=(q,context)=>measuredPoseAudit?measuredPoseAudit(q,context):this.actualRootPostureAudit(q);
    const sample=payloadTargetSampler(targetAt,()=>this.reference.currentFrame());
    sample(0); // Freeze the measured load at the start, before testing the endpoint.
    const endpointTarget=sample(duration);
    // A one-shot endpoint solve must assess the complete displacement. The
    // continuity prior applies to the bounded path steps below.
    const endpoint=ik.solve(start,{...endpointTarget,jointContinuityWeight:0},{iterations:80,maxStep:3,strictEndpoint,bestEffort,seed:endpointTarget.trackPosture?endpointTarget.posture:null});
    const violations=[];if(endpoint.passed===false)violations.push({stage:'endpoint',unmetConstraints:endpoint.unmetConstraints});
    if(checkActualRoot){const measured=measuredAudit(endpoint.q,{endpoint:true,target:endpointTarget});if(!measured.passed){if(!bestEffort)throw new IKError('The return endpoint is unsafe at the measured root.',{measuredRootEndpoint:measured});violations.push({stage:'measured_root_endpoint',geometry:measured});}}
    let q=start,minimumClearance=Infinity,footError=0,comMargin=Infinity;const steps=Math.ceil(duration/this.dt);
    for(let i=1;i<=steps;i++){
      const target=sample(i*this.dt),result=ik.solve(q,target,{iterations:12,maxStep:.035,strictEndpoint:strictEndpoint&&i===steps,bestEffort});q=result.q;
      if(result.passed===false)violations.push({step:i,unmetConstraints:result.unmetConstraints});
      minimumClearance=Math.min(minimumClearance,result.geometry.minimumClearanceSurplus);footError=Math.max(footError,result.residual.footPositionError);comMargin=Math.min(comMargin,result.residual.comMargin);
      if(checkActualRoot){const measured=measuredAudit(q,{endpoint:i===steps,target,step:i});if(!measured.passed){if(!bestEffort)throw new IKError('The return path is unsafe at the measured root.',{step:i,measuredRootPath:measured});violations.push({step:i,stage:'measured_root_path',geometry:measured});}}
      if(i%10===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    return {passed:violations.length===0,bestEffort,violations,duration,samples:steps,minimumClearanceSurplus:minimumClearance,maximumFootError:footError,minimumComMargin:comMargin,endpointResidual:endpoint.residual,endpointQ:q.slice()};
  }
  homeCartesianTarget(alpha){
    const palms={};for(const side of SIDES)palms[side]={position:lerp(this.homeFrom[side].position,this.homeRetracted[side],alpha),rotation:this.homeFrom[side].rotation};
    // Withdraw the open arm before restoring the body. A crouched reference
    // must not replace arm retraction with extreme backward waist bending
    // that brings the upper-arm shell into the torso under policy lag.
    return this.baseTarget(palms,{posture:this.homeHoldQ,lockWaist:true,jointContinuityWeight:1});
  }
  homeRetractCorridorAudit(){
    const scene=this.scene,name=g=>scene.bodyName?.(scene.model?.geom_bodyid?.[g])??'';
    const head=(this.audit.robotGeoms??[]).filter(g=>name(g)==='head_link');
    const hand=(this.audit.robotGeoms??[]).filter(g=>name(g).startsWith(`${this.hand}_wrist_`)||name(g).startsWith(`${this.hand}_hand_`));
    if(!head.length||!hand.length||!this.homeAuditData||!scene.jointQposAddresses29)return {available:false,intersects:false};
    const shift=sub(this.homeRetracted[this.hand],this.homeFrom[this.hand].position),margin=.03,states=[];
    const inspect=(data,kind)=>{
      const h=geomBounds(scene,data,head),w=geomBounds(scene,data,hand);
      const sweep={lower:w.lower.map((v,i)=>Math.min(v,v+shift[i])),upper:w.upper.map((v,i)=>Math.max(v,v+shift[i]))};
      const intersects=[0,1,2].every(i=>sweep.lower[i]<=h.upper[i]+margin&&sweep.upper[i]>=h.lower[i]-margin);
      states.push({kind,head:h,swept_hand_and_wrist:sweep,intersects});
    };
    inspect(scene.data,'actual');
    // The full hand/wrist envelope can be at head height even on a 74 cm
    // table after clearing a tall object. Audit the planned start as well as
    // the measured start; table height alone misses this return corridor.
    const frame=this.reference.currentFrame(),data=this.homeAuditData;data.qpos.set(scene.data.qpos);
    data.qpos.set(frame.rootPosW,0);data.qpos.set(frame.rootQuatW,3);
    scene.jointQposAddresses29.forEach((a,i)=>data.qpos[a]=this.reference.qCurrent[i]);scene.forward(data);
    inspect(data,'reference');
    return {available:true,intersects:states.some(s=>s.intersects),margin_m:margin,states};
  }
  async beginReturn(){
    // Every re-entry must earn its own safe fist clearance. A withdrawal
    // recovery can have moved the hand back near the released object.
    // A clearance-recovery re-entry that discards an already earned fist may not be able to re-form it below
    // tableZ+2 cm, so the retract plan (fist geometry) would execute with open fingers toward the table edge.
    // Keep an earned fist across re-entries; the 4 cm object band below still opens the hand near any object.
    const hadFist=this.fistReturn===true;
    this.fistReturn=false;this.homeVerifyTime=null;this.homeHandOpening=null;
    this.postReleaseDeadline??=nowOf(this.scene)+POST_RELEASE_DEADLINE_S;
    this.depositSuccess=this.everPlaced===true;this.success=null;this.returnSuccess=null;this.homeFrom=this.plannedPalms();this.homeHoldQ=this.reference.qCurrent.slice();this.homeRetracted={};this.homeMovingSides=[];this.homeRotationForward={};this.homePlanChecks=[];
    this.tableFront=geomBounds(this.scene,this.scene.data,this.scene.tableGeomIds).lower[0];this.homeEndpointAudit=this.actualRootPostureAudit();
    this.enterHomeStage('retract');this.homeDuration=2.2;this.homeStable=0;this.homeTime=nowOf(this.scene);
    this.enter('return_home','Checking the shortest clear return path behind the table.');this.stage='retract';
    if(this.crouched&&tableZ(this.scene)<.60){
      // Once the low-table object is released, raise the body with the arms
      // still forward. Retracting them toward the hips while deeply crouched
      // can sweep a forearm into the torso even with a clear tabletop.
      for(const side of SIDES)this.scene.setHandClosure(side,0);
      await this.beginHomeRestore();return;
    }
    for(const side of SIDES){
      const actual=this.palm(side),bounds=this.audit.handBounds(this.scene.data,side),local=[];
      for(const gid of this.audit.hands[side])for(const point of geomCorners(this.scene,this.scene.data,gid))local.push(matVec(transpose3(actual.rotation),sub(point,actual.position)));
      let forward=-Infinity;for(let i=0;i<=30;i++){const rotation=rotationBlend(actual.rotation,this.homePalms[side].rotation,i/30);for(const point of local)forward=Math.max(forward,matVec(rotation,point)[0]);}
      this.homeRotationForward[side]=forward;
      const retreat=Math.max(0,Math.max(bounds.upper[0],actual.position[0]+forward)-(this.tableFront-.035));
      this.homeRetracted[side]=add(this.homeFrom[side].position,[-retreat,0,0]);
      if(retreat>.001)this.homeMovingSides.push(side);
      const objectGeoms=this.audit.objectGeoms?.()??null,measurable=!!objectGeoms&&typeof this.audit.nativeDistance==='function'&&Array.isArray(this.audit.hands?.[side]);
      const nearObject=!measurable||(side===this.hand&&this.audit.hands[side].some(a=>objectGeoms.some(b=>this.audit.nativeDistance(this.scene.data,a,b,.045)<.04)));
      const fist=side===this.hand&&!nearObject&&(hadFist||Number.isFinite(bounds?.lower?.[2])&&bounds.lower[2]>=tableZ(this.scene)+.02);
      if(fist)this.fistReturn=true;this.scene.setHandClosure(side,fist?1:0);
    }
    // Prefer no lateral motion. Small outboard alternatives are considered only
    // when the complete planted-foot IK path cannot realize that shortest route.
    const original=structuredClone(this.homeRetracted),homeY=this.homePalms[this.hand].position[1],direction=this.hand==='left'?1:-1,softCandidates=[];let accepted=false,lastError;
    this.homeRetractHeadCorridor=this.homeRetractCorridorAudit();
    const outboardFirst=tableZ(this.scene)>=.80||this.homeRetractHeadCorridor.intersects;
    const ys=[original[this.hand][1],... [.04,.08,.12,.16].map(d=>original[this.hand][1]+direction*d),homeY,homeY+direction*.04,homeY+direction*.08].filter((y,i,a)=>a.findIndex(v=>Math.abs(v-y)<.005)===i)
      .sort((a,b)=>outboardFirst?Math.abs(a-homeY)-Math.abs(b-homeY):Math.abs(a-original[this.hand][1])-Math.abs(b-original[this.hand][1]));
    this.homeRetractOutboardFirst=outboardFirst;
    for(const y of ys){
      const outboard=Math.abs(y-original[this.hand][1]);this.homeRetracted=structuredClone(original);if(this.homeMovingSides.includes(this.hand))this.homeRetracted[this.hand][1]=y;
      try{const result=await this.preflightHomePath(t=>this.homeCartesianTarget(smooth(t/this.homeDuration)),this.homeDuration);this.homePlanChecks.push({stage:'retract',outboard,passed:true,...result});accepted=true;break;}
      catch(error){if(!(error instanceof IKError))throw error;lastError=error;this.homePlanChecks.push({stage:'retract',outboard,passed:false,reason:error.message,details:error.details});
        if(error.details?.geometry?.passed&&error.details.footPositionError<.003&&error.details.footRotationError<.05&&error.details.comMargin>=.02&&error.details.endpoint===false)softCandidates.push({y,outboard});}
    }
    // A high palm may have no reachable level withdrawal with the waist
    // held. Audit small descending routes only after all level routes reject.
    if(!accepted&&outboardFirst&&this.homeMovingSides.includes(this.hand)){
      for(const drop of [.04,.06,.08]){
        for(const y of ys){
          this.homeRetracted=structuredClone(original);this.homeRetracted[this.hand][1]=y;this.homeRetracted[this.hand][2]-=drop;
          const outboard=Math.abs(y-original[this.hand][1]);
          try{
            // A one-shot endpoint can converge to a different IK branch.
            // Keep every native path/root check, then require the bounded
            // path's actual final frame to meet the original pose thresholds.
            const result=await this.preflightHomePath(t=>this.homeCartesianTarget(smooth(t/this.homeDuration)),this.homeDuration,{checkActualRoot:true,strictEndpoint:false});
            const endpoint=this.reference.ik.solve(result.endpointQ,this.homeCartesianTarget(1),{iterations:0,maxStep:0,strictEndpoint:true});
            result.endpointResidual=endpoint.residual;
            this.homePlanChecks.push({stage:'retract_descending',outboard,vertical_drop_m:drop,passed:true,...result});accepted=true;break;
          }catch(error){if(!(error instanceof IKError))throw error;lastError=error;this.homePlanChecks.push({stage:'retract_descending',outboard,vertical_drop_m:drop,passed:false,reason:error.message,details:error.details});}
        }
        if(accepted)break;
      }
    }
    if(!accepted)for(const {y,outboard}of softCandidates){
      this.homeRetracted=structuredClone(original);if(this.homeMovingSides.includes(this.hand))this.homeRetracted[this.hand][1]=y;
      try{const result=await this.preflightHomePath(t=>this.homeCartesianTarget(smooth(t/this.homeDuration)),this.homeDuration,{checkActualRoot:true,strictEndpoint:false});this.homePlanChecks.push({stage:'retract_best_effort',outboard,passed:true,...result});this.markCompletionFallback('return_pose_retry');accepted=true;break;}
      catch(error){if(!(error instanceof IKError))throw error;lastError=error;this.homePlanChecks.push({stage:'retract_best_effort',outboard,passed:false,reason:error.message,details:error.details});}
    }
    if(!accepted){
      this.homeRetracted=structuredClone(original);
      const result=await this.preflightHomePath(t=>this.homeCartesianTarget(smooth(t/this.homeDuration)),this.homeDuration,{checkActualRoot:true,strictEndpoint:false,bestEffort:true});
      this.homePlanChecks.push({stage:'retract_best_effort',...result,previousRejection:lastError?.message});this.markCompletionFallback('return_ik_best_effort');
    }
    this.homeTime=nowOf(this.scene);this.forceReference=true;this.message='Retracting the open hand just behind the table edge.';
  }
  async beginHomeRestore({bodyRecovered=false}={}){
    this.homeRootError=this.rootHomeMetrics();
    if(!bodyRecovered&&(norm(sub(this.rootPosition,this.reference.ik.anchorPosition))>.002||Math.abs(this.rootPitch)>.015||!this.homeRootError.reached)){
      const frame=this.reference.currentFrame();this.homeRootRecovery={from:frame.rootPosW.slice(),pitch:frame.plannedRootPitch,posture:this.reference.qCurrent.slice()};
      this.homeDuration=Math.max(2.2,norm(sub(frame.rootPosW,this.reference.ik.anchorPosition))/.035,Math.abs(frame.plannedRootPitch)/.15);
      try{const check=await this.preflightHomePath(t=>this.homeRootTarget(smooth(t/this.homeDuration)),this.homeDuration);this.homePlanChecks.push({stage:'root_recovery',passed:true,...check});}
      catch(error){if(!(error instanceof IKError))throw error;const check=await this.preflightHomePath(t=>this.homeRootTarget(smooth(t/this.homeDuration)),this.homeDuration,{strictEndpoint:false,bestEffort:true});this.homePlanChecks.push({stage:'root_recovery_best_effort',...check});this.markCompletionFallback('root_ik_best_effort');}
      this.enterHomeStage('root_recovery');this.homeTime=nowOf(this.scene);this.homeStable=0;this.forceReference=true;return;
    }
    this.homeRestoreFrom=this.reference.qCurrent.slice();this.homeDuration=3;this.homeEndpointAudit=this.actualRootPostureAudit();
    try{
      const result=await this.preflightHomePath(t=>this.baseTarget(this.homePalms,{posture:lerp(this.homeRestoreFrom,this.homeQ,smooth(t/this.homeDuration)),trackPosture:true}),this.homeDuration,{checkActualRoot:true});
      this.homePlanChecks.push({stage:'restore',passed:true,...result});
    }catch(error){
      if(!(error instanceof IKError))throw error;this.homePlanChecks.push({stage:'restore',passed:false,reason:error.message,details:error.details});
      // A soft rest-posture residual need not abort the gesture. Retry without
      // that endpoint-accuracy gate, keeping every geometry/support check and
      // the independent measured-root audit. Execution still targets homeQ.
      if(error.details?.geometry?.passed&&error.details.footPositionError<.003&&error.details.comMargin>=.02&&error.details.endpoint===false){
        try{const result=await this.preflightHomePath(t=>this.baseTarget(this.homePalms,{posture:lerp(this.homeRestoreFrom,this.homeQ,smooth(t/this.homeDuration)),trackPosture:true}),this.homeDuration,{checkActualRoot:true,strictEndpoint:false});this.homePlanChecks.push({stage:'restore_best_effort',passed:true,...result});this.markCompletionFallback('rest_posture_retry');}
        catch(retry){if(!(retry instanceof IKError))throw retry;await this.planApproximateHomeRestore();}
      }else await this.planApproximateHomeRestore();
    }
    this.enterHomeStage('restore');this.homeTime=nowOf(this.scene);this.homeStable=0;this.forceReference=true;this.message='Relaxing both arms back into their initial posture.';
  }
  async planApproximateHomeRestore(){
    const result=await this.preflightHomePath(t=>this.baseTarget(this.homePalms,{posture:lerp(this.homeRestoreFrom,this.homeQ,smooth(t/this.homeDuration)),trackPosture:true}),this.homeDuration,{checkActualRoot:true,strictEndpoint:false,bestEffort:true});
    this.homePlanChecks.push({stage:'restore_approximate',...result});this.markCompletionFallback('rest_ik_best_effort');
  }
  homeRootTarget(alpha){
    const r=this.homeRootRecovery,root=lerp(r.from,this.reference.ik.anchorPosition,alpha);
    return this.baseTarget(this.homePalms,{posture:r.posture,trackPosture:true,trackUpperPostureOnly:true,rootPosition:root,rootHeight:root[2],rootPitch:r.pitch*(1-alpha)});
  }
  enterHomeStage(stage){this.homeStage=stage;this.stage=stage;this.homeStageVisit=(this.homeStageVisit||0)+1;}
  homeTarget(offset=0){
    const alpha=smooth((nowOf(this.scene)-this.homeTime+offset)/this.homeDuration);
    if(this.homeStage==='clearance_recovery')return this.homeClearanceTarget(alpha);
    if(this.homeStage==='root_recovery')return this.homeRootTarget(alpha);
    if(['restore','verify'].includes(this.homeStage))return this.baseTarget(this.homePalms,{posture:lerp(this.homeRestoreFrom,this.homeQ,alpha),trackPosture:true});
    return this.homeCartesianTarget(alpha);
  }
  homeClearanceTarget(alpha){
    const r=this.homeClearanceRecovery,palms=structuredClone(r.from);
    for(const side of r.sides)palms[side].position=lerp(r.from[side].position,r.goals[side],alpha);
    return this.baseTarget(palms,{posture:r.posture,rootPosition:r.root,rootHeight:r.root[2],rootPitch:r.pitch,jointContinuityWeight:1});
  }
  async beginHomeClearanceRecovery(geometry){
    const visit=this.homeStageVisit||0;
    // One recovery attempt (both candidate shifts) per home-stage visit, never from inside a recovery, at most HOME_CLEARANCE_RECOVERY_CAP accepted per episode.
    if((this.homeClearanceRetries||0)>=HOME_CLEARANCE_RECOVERY_CAP||this.homeStage==='clearance_recovery'||this.homeRecoveryAttemptVisit===visit)return false;
    this.homeRecoveryAttemptVisit=visit;
    const pair=geometry.limitingPair,side=SIDES.find(s=>this.audit.hands[s].includes(pair?.a)||this.audit.hands[s].includes(pair?.b))||this.hand;
    const frame=this.reference.currentFrame(),from=this.plannedPalms();
    this.homeClearanceRecovery={from,goals:{},sides:[side],posture:this.reference.qCurrent.slice(),root:frame.rootPosW.slice(),pitch:frame.plannedRootPitch};
    const duration=1.2,direction=side==='left'?1:-1;
    for(const shift of [[-.035,0,.05],[-.05,direction*.035,.05]]){
      this.homeClearanceRecovery.goals[side]=add(from[side].position,shift);
      try{
        const result=await this.preflightHomePath(t=>this.homeClearanceTarget(smooth(t/duration)),duration,{checkActualRoot:true});
        this.homePlanChecks.push({stage:'clearance_recovery',passed:true,shift,...result});
        this.homeClearanceRetries=(this.homeClearanceRetries||0)+1;this.markCompletionFallback('return_clearance_recovery');
        this.enterHomeStage('clearance_recovery');this.homeDuration=duration;this.homeTime=nowOf(this.scene);this.homeStable=0;this.forceReference=true;
        this.message='Lifting and withdrawing the open hand before continuing the return.';return true;
      }catch(error){if(!(error instanceof IKError))throw error;this.homePlanChecks.push({stage:'clearance_recovery',passed:false,shift,reason:error.message,details:error.details});}
    }
    this.homeClearanceRecovery=null;return false;
  }
  async advanceReturn(){
    const now=nowOf(this.scene),elapsed=now-this.homeTime,state=this.scene.readState(this.objectId),rootRotation=quatToMat(state.rootQuatW);this.stage=this.homeStage;
    if(this.checkPostReleaseDeadline())return;
    this.homeJointError=Math.max(...state.dofPos.slice(12).map((q,i)=>Math.abs(q-this.homeQ[12+i])));this.homePalmError=0;this.homeAngleError=0;
    for(const side of SIDES){const actual=this.palm(side),local=matVec(transpose3(rootRotation),sub(actual.position,state.rootPosW));this.homePalmError=Math.max(this.homePalmError,norm(sub(local,this.homeLocal[side].position)));this.homeAngleError=Math.max(this.homeAngleError,norm(rotationError(matMul(rootRotation,this.homeLocal[side].rotation),actual.rotation)));}
    const previousClearance=this.homeClearance,geometry=this.audit.check(this.scene.data,{handsOnly:true,includeSelf:false,margin:.005,ignoreObjectId:this.ignoredObjectId});this.homeClearance=geometry.minimumClearance;
    // Act while there is still stopping distance. A 5 mm guard alone freezes
    // the robot only after tracking lag has consumed the planned clearance.
    const closingSpeed=Number.isFinite(previousClearance)&&Number.isFinite(geometry.minimumClearance)?Math.max(0,(previousClearance-geometry.minimumClearance)/this.dt):0;
    // Raising a crouched root also sweeps the held arm pose past a released
    // object. Give that stage the same early, audited withdrawal as restore.
    if(geometry.passed&&['root_recovery','restore','verify'].includes(this.homeStage)&&geometry.minimumClearance<.03&&(geometry.minimumClearance<.012||geometry.minimumClearance-.35*closingSpeed<.005)){
      if(await this.beginHomeClearanceRecovery(geometry))return;
    }
    if(!geometry.passed){
      // A measured clearance shortfall during the return is answered with one audited withdrawal per stage visit,
      // otherwise recorded while the current audited/best-effort plan keeps executing. It never stops the episode.
      if(this.homeStage!=='clearance_recovery'&&await this.beginHomeClearanceRecovery(geometry))return;
      this.homeClearanceViolations=(this.homeClearanceViolations||0)+1;this.recordPostReleaseHazard('return_clearance_violation',{geometry});
    }
    if(this.homeStage==='clearance_recovery'){
      this.homeStable=geometry.minimumClearance>=.025?this.homeStable+this.dt:0;
      if(elapsed>=this.homeDuration&&this.homeStable>=.2){await this.beginReturn();return;}
      if(elapsed>this.homeDuration+3){this.recordPostReleaseHazard('return_clearance_recovery_timeout',{geometry});await this.beginReturn();}return;
    }
    if(now-this.homeMeasuredAuditTime>=.2){this.homeEndpointAudit=this.actualRootPostureAudit();this.homeMeasuredAuditTime=now;}
    this.homeRootError=this.rootHomeMetrics();
    if(this.homeStage==='root_recovery'){
      const frame=this.reference.currentFrame(),referenceHome=norm(sub(frame.rootPosW,this.reference.ik.anchorPosition))<.002&&Math.abs(frame.plannedRootPitch)<.015;
      // Holding the withdrawn arms can itself bias the physical equilibrium.
      // Once the body reference is home and measured motion has settled, audit
      // arm restoration at the actual root instead of waiting indefinitely in
      // that biased pose. Final success still requires the original root gate.
      this.homeStable=referenceHome&&this.homeRootError.quiet?this.homeStable+this.dt:0;
      if(elapsed>=this.homeDuration&&this.homeStable>=.3){this.rootPosition=this.reference.ik.anchorPosition.slice();this.rootHeight=this.rootPosition[2];this.rootPitch=0;this.crouched=false;await this.beginHomeRestore({bodyRecovered:true});return;}
      if(elapsed>this.homeDuration+6){this.recordPostReleaseHazard('root_recovery_timeout');this.rootPosition=this.reference.ik.anchorPosition.slice();this.rootHeight=this.rootPosition[2];this.rootPitch=0;this.crouched=false;await this.beginHomeRestore({bodyRecovered:true});}return;
    }
    if(this.homeStage==='retract'){
      const behind=SIDES.every(side=>{const b=this.audit.handBounds(this.scene.data,side);return Math.max(b.upper[0],this.palm(side).position[0]+this.homeRotationForward[side])<=this.tableFront-.025;});
      this.homeStable=behind?this.homeStable+this.dt:0;
      if(elapsed>=this.homeDuration&&this.homeStable>=.2){await this.beginHomeRestore();return;}
      if(elapsed>this.homeDuration+4){
        // The swept AABB is conservative. If the rest pose is clear at the
        // measured root, try the fully audited return rather than waiting on
        // that bounding-box condition indefinitely.
        this.markCompletionFallback('return_clearance_retry');await this.beginHomeRestore();
      }return;
    }
    // Keep commanding the original rest posture throughout verification. A
    // single near-home sample must never freeze a biased intermediate target.
    if(elapsed>=this.homeDuration&&this.homeStage!=='verify'){
      // Opening is a physical movement, not a terminal command. Discard the
      // closed-fist dwell and execute at least one open-hand physics step.
      this.fistReturn=false;this.enterHomeStage('verify');this.homeVerifyTime=now;this.homeStable=0;
      for(const side of SIDES)this.scene.setHandClosure(side,0);
      return;
    }
    this.homeHandOpening=this.homeHandOpeningAudit();
    const reached=this.homeJointError<.20&&this.homePalmError<.06&&this.homeAngleError<.30;
    const quiet=reached&&this.homeRootError.reached&&this.homeRootError.quiet&&Math.max(...state.dofVel.slice(12).map(Math.abs))<.35&&this.homeHandOpening.opened&&this.homeHandOpening.quiet;
    this.homeStable=quiet?this.homeStable+this.dt:0;
    if(this.homeStage==='verify'){if(this.homeStable>=.5){this.finishCycle();}else if(now-(this.homeVerifyTime??this.homeTime+this.homeDuration)>7)this.finishCycle({returned:false});}
  }
  homeHandOpeningAudit(hand=null){
    const scene=this.scene,qadr=scene.fingerQposAddresses14,vadr=scene.fingerDofAddresses14,meta=scene.metadata;
    const open=SIDES.flatMap(side=>meta?.handOpen?.[side]??[]),positionTolerance=.15,speedTolerance=.5;
    let complete=open.length>0&&open.length%2===0&&qadr?.length===open.length&&vadr?.length===open.length&&meta?.fingerLimits?.length===open.length&&meta?.handOpen?.left?.length===open.length/2&&meta?.handOpen?.right?.length===open.length/2&&(hand===null||SIDES.includes(hand));
    let positionError=0,speed=0;
    const start=hand===null?0:SIDES.indexOf(hand)*open.length/2,end=hand===null?open.length:start+open.length/2;
    if(complete)for(let i=start;i<end;i++){
      const limit=meta.fingerLimits[i],q=scene.data.qpos[qadr[i]],v=scene.data.qvel[vadr[i]];
      if(!Number.isInteger(qadr[i])||qadr[i]<0||!Number.isInteger(vadr[i])||vadr[i]<0||limit?.length!==2||![open[i],q,v,...limit].every(Number.isFinite)||limit[0]>limit[1]){complete=false;break;}
      positionError=Math.max(positionError,Math.abs(q-clamp(open[i],...limit)));speed=Math.max(speed,Math.abs(v));
    }
    return {metadata_complete:!!complete,maximum_position_error_rad:complete?positionError:null,maximum_speed_rad_s:complete?speed:null,
      position_tolerance_rad:positionTolerance,speed_tolerance_rad_s:speedTolerance,opened:!!complete&&positionError<positionTolerance,quiet:!!complete&&speed<speedTolerance};
  }
  checkPostReleaseDeadline(){
    if(this.postReleaseDeadline==null||nowOf(this.scene)<this.postReleaseDeadline)return false;
    this.recordPostReleaseHazard('return_deadline');this.finishCycle({returned:false});return true;
  }
  finishCycle({returned=true}={}){
    this.observeTrayLanding();this.motionCompleted=true;this.returnSuccess=returned;this.depositSuccess=this.everPlaced===true;this.success=this.depositSuccess;this.restQ=this.homeQ.slice();
    this.fistReturn=false;for(const side of SIDES)this.scene.setHandClosure?.(side,0);
    if(!this.success){
      this.failurePhase=this.graspAbort?.phase??this.failurePhase??this.phase;
      const base=this.failureMessage??(this.graspAbort?'Attempt unsuccessful.':'The object was not placed in the tray.');
      this.enter('failed',returned?`${base} The hand has returned to the initial posture.`:`${base} No collision-free return path was found; the return is incomplete.`);return;
    }
    this.enter('succeeded',returned?'Placement successful. Both hands are retracted and the initial posture is restored.':'Placement successful. The return to the initial posture is incomplete.');
  }
  returnDiagnostics(){
    if(!this.homeFrom&&!this.clearTrayFrom&&!this.releasePalms)return null;
    return {phase:this.phase,stage:this.stage,planned:this.plannedPalms(),actual:Object.fromEntries(SIDES.map(side=>[side,this.palm(side)])),target:this.busy?this.targetAt().palms:null,releaseFrom:this.releaseFrom,releaseRetreat:this.releaseRetreat,clearTrayGoals:this.clearTrayGoals,homeFrom:this.homeFrom,homeRetracted:this.homeRetracted,homeQ:this.homeQ.slice(),restQ:this.restQ.slice(),referenceQ:this.reference.qCurrent.slice(),rootPosition:this.rootPosition.slice(),actualRootHome:this.actualRootPostureAudit(),planChecks:this.homePlanChecks};
  }
  objectBounds(){
    // Decorative skin, stems and leaves must not enlarge the carried collider.
    const m=this.scene.model,geoms=this.scene.objects[this.objectId].geomIds.filter(g=>m.geom_contype[g]||m.geom_conaffinity[g]);
    return geomBounds(this.scene,this.scene.data,geoms);
  }
  measureGraspSupport(contacts,recent){
    const scene=this.scene,m=scene.model,obj=scene.objects[this.objectId],objGeoms=new Set(obj.geomIds),clearanceThreshold=.0002;
    // A raised COM can still leave a tilted object resting on a fixture. Use
    // the measured physical collider bottom, never nominal height or a preview.
    const physicalMetadata=obj.geomIds.length>0&&obj.geomIds.every(g=>Number.isInteger(g)&&g>=0&&Number.isFinite(m.geom_contype?.[g])&&Number.isFinite(m.geom_conaffinity?.[g]));
    let clearance=null;
    if(physicalMetadata&&Number.isFinite(scene.tableTopZ)){
      try{
        const bounds=this.objectBounds();
        if(bounds.lower?.length===3&&bounds.upper?.length===3&&bounds.lower.every((v,i)=>Number.isFinite(v)&&Number.isFinite(bounds.upper[i])&&v<=bounds.upper[i]))clearance=bounds.lower[2]-scene.tableTopZ;
      }catch{/* Missing native bounds cannot certify a grasp. */}
    }
    const tableIds=scene.tableGeomIds,handIds=this.audit.hands?.[this.hand];
    const tableMetadata=Array.isArray(tableIds)&&tableIds.length>0&&tableIds.every(g=>Number.isInteger(g)&&g>=0);
    const tableGeoms=new Set(tableMetadata?tableIds:[]),selectedHandGeoms=new Set();
    let handMetadata=Array.isArray(handIds)&&handIds.length>0&&this.audit.fingers instanceof Map&&typeof scene.bodyName==='function';
    if(handMetadata)for(const g of handIds){
      const body=m.geom_bodyid?.[g],name=Number.isInteger(body)?scene.bodyName(body):null;
      if(typeof name!=='string'||!name.length){handMetadata=false;continue;}
      // Wrist/arm contact is external support, even for the selected hand.
      if(name.startsWith(`${this.hand}_hand_`))selectedHandGeoms.add(g);
    }
    handMetadata&&=selectedHandGeoms.size>0;
    let tableContacts=0,tableForce=0,externalContacts=0,handContacts=0,contactMetadata=true;
    for(const c of contacts){
      const other=objGeoms.has(c.geom1)?c.geom2:objGeoms.has(c.geom2)?c.geom1:null;if(other==null)continue;
      if(!Number.isInteger(other)||other<0||!Number.isFinite(c.normalForce)||!Number.isFinite(c.dist)){contactMetadata=false;continue;}
      if(!(c.normalForce>.02&&c.dist<=.0002))continue;
      if(tableGeoms.has(other)){tableContacts++;tableForce+=c.normalForce;}
      if(selectedHandGeoms.has(other))handContacts++;else externalContacts++;
    }
    const metadataComplete=physicalMetadata&&clearance!==null&&tableMetadata&&handMetadata&&contactMetadata;
    const clearOfTable=clearance!==null&&clearance>clearanceThreshold&&tableMetadata&&contactMetadata&&tableContacts===0;
    return {object_bottom_clearance_m:clearance,clearance_threshold_m:clearanceThreshold,table_contact_count:tableContacts,table_normal_force_n:tableForce,external_contact_count:externalContacts,hand_contact_count:handContacts,recent_finger_group_count:recent.size,metadata_complete:metadataComplete,clear_of_table:clearOfTable,held_away_from_support:metadataComplete&&clearOfTable&&externalContacts===0&&handContacts>0};
  }
  objectTrayOverlap(){return objectTrayOverlap(this.scene,this.objectId);}
  observeTrayLanding(){
    if(this.everPlaced||this.objectId==null)return;
    const landed=this.scene.lastStepTrayLandings?.find(event=>event.object_id===this.objectId&&event.time_s>=this.startTime-1e-9);
    if(landed){this.everPlaced=true;this.trayLanding=structuredClone(landed);this.depositSuccess=true;}
  }
  measure(){
    this.observeTrayLanding();
    const scene=this.scene,obj=scene.objects[this.objectId],position=poseOf(scene,this.objectId).position,contacts=scene.contacts(),m=scene.model,groups=new Set(),objGeoms=new Set(obj.geomIds),trayGeoms=new Set(scene.trayGeomIds);this.contacts=0;this.handContacts=0;this.activeHandContacts=0;this.trayContacts=0;this.trayBottomContacts=0;this.bodyFixtureForce=0;this.bodyFixturePenetration=0;this.handTableForce=0;this.handTablePenetration=0;
    this.lift=position[2]-this.initialObjectZ;this.maxLift=Math.max(this.maxLift,this.lift);this.trayOverlap=this.objectTrayOverlap();this.inTray=this.trayOverlap>1e-8;
    for(const c of contacts){const a=c.geom1,b=c.geom2,other=objGeoms.has(a)?b:objGeoms.has(b)?a:null;
      if(other!=null){const finger=this.audit.fingers?.get(other);if(finger?.side===this.hand&&c.normalForce>.02){groups.add(finger.finger);this.contacts++;}const name=scene.bodyName?.(m.geom_bodyid?.[other])??'';if((name.includes('_hand_')||name.includes('_wrist_'))&&c.dist<=.0002){this.handContacts++;if(name.startsWith(`${this.hand}_`)&&c.normalForce>.02)this.activeHandContacts++;}if(trayGeoms.has(other)&&c.normalForce>.02){this.trayContacts++;if(other===scene.trayBottomGeomId)this.trayBottomContacts++;}if(other===scene.trayBottomGeomId&&c.dist<-.003)this.inTray=false;}
    }
    const forceContacts=scene.lastStepContacts||contacts;
    for(const c of forceContacts){const a=c.geom1,b=c.geom2,robot=this.audit.fixtureGeoms.includes(a)?b:this.audit.fixtureGeoms.includes(b)?a:null;if(robot==null||!this.audit.robotGeoms.includes(robot))continue;
      // Low-force tabletop contact is legal, but a peak between control ticks
      // must not disappear before the impact guard. Native contact forces and
      // deepest distances are collected across all 1 ms physics substeps.
      const fixture=a===robot?b:a;
      if(fixture===this.audit.tableTopGeomId&&this.audit.handSet.has(robot)){
        this.handTableForce=Math.max(this.handTableForce,c.normalForce||0);
        this.handTablePenetration=Math.max(this.handTablePenetration,Math.max(0,-c.dist));
      }
      const name=scene.bodyName(m.geom_bodyid[robot]),lower=name==='pelvis'||/hip|knee|ankle/.test(name);
      const stanceMotion=['crouch','stance_recovery'].includes(this.phase),nonHand=!this.audit.handSet.has(robot);
      if(lower||this.stancePlan&&nonHand||stanceMotion||(this.phase==='return_home'||this.postRelease)&&this.audit.handSet.has(robot)){this.bodyFixtureForce=Math.max(this.bodyFixtureForce,c.normalForce||0);this.bodyFixturePenetration=Math.max(this.bodyFixturePenetration,Math.max(0,-c.dist));}
    }
    this.fingerContactCount=groups.size;this.gripContactTime=groups.size>=2?this.gripContactTime+this.dt:0;
    this.recentContacts.push({time:nowOf(scene),groups});while(this.recentContacts.length&&this.recentContacts[0].time<nowOf(scene)-.15)this.recentContacts.shift();const recent=new Set(this.recentContacts.flatMap(c=>[...c.groups]));
    this.graspSupport=this.measureGraspSupport(contacts,recent);this.stableLift=this.graspSupport.held_away_from_support&&recent.size>=2?this.stableLift+this.dt:0;this.objectLocked||=this.contacts>0;
    if(this.stableLift>=.8)this.graspObstructionCandidate=null;
    const velocity=copy(scene.data.qvel.slice(obj.vadr,obj.vadr+6)),stable=this.inTray&&this.trayContacts>0&&this.handContacts===0&&norm(velocity.slice(0,3))<.035&&norm(velocity.slice(3,6))<.35;this.stableDeposit=stable?this.stableDeposit+this.dt:0;
  }
  targetAt(offset=0){
    const elapsed=nowOf(this.scene)-this.phaseTime+offset;
    if(this.phase==='crouch')return this.relaxedTarget(this.stage==='prepare_crouch'?0:smooth(elapsed/this.stanceDuration),this.stage==='prepare_crouch'?smooth(elapsed/1.6):1);
    if(this.phase==='stance_recovery')return this.recoveryTarget(offset);
    if(this.phase==='approach')return this.approachTarget(offset);
    if(this.phase==='stand')return this.standTarget(offset);
    if(this.phase==='transfer')return this.transferTarget(offset);
    if(this.phase==='return_home')return this.homeTarget(offset);
    if(this.phase==='release')return this.releaseTarget(offset);
    if(this.phase==='settle')return this.clearTrayTarget(offset);
    if(this.phase==='close')return this.approachTarget(offset);
    const position=this.phase==='lift'?lerp(this.liftFrom??this.grasp,this.liftTarget,smooth(elapsed/2)):this.phase==='hold'?this.liftTarget:this.grasp;
    return this.oneHandTarget(position,this.liftRotation??this.graspRotation,{allowHandObject:true,allowHandTableContact:this.plan?.approach==='side',posture:this.liftPosture??this.restQ,waistPostureWeights:this.liftWaistPostureWeights??undefined});
  }
  async tick(){
    if(!this.busy||this.phase==='planning')return this.snapshot();const elapsed=nowOf(this.scene)-this.phaseTime,closureBefore=this.closeProgress??0;
    if(this.phase==='crouch'){
      if(this.stage==='prepare_crouch'){
        const clear=SIDES.every(s=>this.audit.handBounds(this.scene.data,s).upper[0]<=this.tableFront-.025);
        if(elapsed>=1.6&&clear){this.stage='crouch';this.phaseTime=nowOf(this.scene);this.forceReference=true;this.message='Adjusting the supported pelvis while the relaxed hands follow the torso.';}else if(elapsed>5.6)this.fail('Both hands could not retract before the posture adjustment.');
      }else{
        if(SIDES.some(s=>this.audit.handBounds(this.scene.data,s).upper[0]>this.tableFront-.005))this.fail('A hand approached the table edge while crouching.');
        const state=this.scene.readState(),target=this.stanceTargetRoot,angle=norm(rotationError(matMul(this.reference.ik.anchorRotation,rotateY(this.crouchPitch)),quatToMat(state.rootQuatW)));
        const quiet=Math.hypot(...sub(state.rootPosW,target).slice(0,2))<.035&&Math.abs(state.rootPosW[2]-target[2])<.035&&angle<.12&&norm(state.rootLinVelW)<.08;
        this.stableBody=quiet?this.stableBody+this.dt:0;
        if(elapsed>=this.stanceDuration&&this.stableBody>=.25){this.rootPosition=target.slice();this.rootHeight=this.crouchHeight;this.rootPitch=this.crouchPitch;this.restQ=this.stanceRestPosture.slice();this.crouched=true;this.buildApproach();}else if(elapsed>this.stanceDuration+5)this.fail('The robot could not stabilize at the planned supported posture.');
      }
    }
    if(this.pendingReturn){this.pendingReturn=false;
      try{await this.beginReturn();}
      catch(error){if(!(error instanceof IKError)||!this.postRelease)throw error;this.lastError={message:error.message,details:error.details??null};this.recordPostReleaseHazard('return_plan_error');this.finishCycle({returned:false});return this.snapshot();}
    }
    if(this.phase==='approach'){
      this.updateReachTracking();
      this.observeApproachObstruction();
      if(this.graspDistance()<=this.gates.normal)this.advanceApproach();
      else if(this.gaFrozenReason&&await this.maybeReplanStance()){}else this.advanceApproach();
    }
    else if(this.phase==='stance_recovery')await this.advanceStanceRecovery();
    else if(this.phase==='close')this.advanceClose();
    else if(this.phase==='lift'){if(elapsed>=2.1)this.enter('hold','Checking that the object remains lifted.');}
    else if(this.phase==='hold'){
      if(elapsed>=1.2){this.graspSuccess=this.stableLift>=.8;if(!this.graspSuccess)this.beginGraspAbort('grasp_not_secured');
        // The tray is on the same low tabletop. Place from the reached stance
        // and restore the body after releasing the load, avoiding a needless
        // loaded stand that sweeps the extended hand across the torso.
        else if(this.crouched&&this.depositEnabled&&tableZ(this.scene)<.60)this.beginDeposit();
        else if(this.crouched)await this.beginStand();else if(this.depositEnabled)this.beginDeposit();else{this.success=true;this.enter('succeeded','Grasp successful: sustained lift with multiple fingers.');}}
    }else if(this.phase==='stand'){
      const root=this.rootHomeMetrics();this.stableBody=root.reached&&root.quiet?this.stableBody+this.dt:0;
      if(elapsed>=this.standDuration&&this.stableBody>=.25&&this.stableLift>=.8){const previous=this.standTarget();this.rootPosition=this.reference.ik.anchorPosition.slice();this.rootHeight=this.standingHeight;this.rootPitch=0;this.crouched=false;this.restQ.splice(12,3,...this.homeQ.slice(12,15));if(this.depositEnabled)this.beginDeposit(previous);else{this.success=true;this.enter('succeeded','Grasp secured at standing height.');}}else if(elapsed>this.standDuration+5)this.beginCompletionTransfer(this.stableLift>=.8?'body_recovery_timeout':'standing_grip_lost');
    }else if(this.phase==='transfer'){this.advanceTransfer();}
    else if(['release','settle','return_home'].includes(this.phase)){
      try{
        if(this.phase==='release'){if(this.graspAbort)await this.advanceGraspAbort();else await this.advanceRelease();}
        else if(this.phase==='settle')await this.advanceClearTray();
        else await this.advanceReturn();
      }catch(error){
        // A planner error after the hand opened ends the episode with the honest 'return incomplete' outcome instead of an unhandled worker error.
        if(!(error instanceof IKError)||!this.postRelease)throw error;
        this.lastError={message:error.message,details:error.details??null};this.recordPostReleaseHazard('return_plan_error');this.finishCycle({returned:false});return this.snapshot();
      }
    }
    if(!this.busy)return this.snapshot();this.closedLoopUpdate();
    // Retargeting may change admission on this same tick. Recheck before the
    // first actual positive command; an admitted ramp is not a new grasp.
    if(this.phase==='close'&&!this.closureCommitted&&!(this.finalAttempt&&this.forcedClose)){
      const alignment=this.graspAlignment(),limit=this.forcedClose?this.gates.forced:this.gates.normal+(closureBefore>0?.0025:0);
      const allowed=!!(this.forcedClose||this.closeTrigger?.ready)&&this.closureAllowed(alignment,limit);
      this.recordClosureGate(allowed,this.forcedClose?'forced':'normal',alignment,limit,allowed?'closing':'alignment_lost');
      if(!allowed)this.closeProgress=Math.min(this.closeProgress,Math.max(0,closureBefore-this.dt/.6));
      if(!this.closureAllowed(alignment))this.beginAlignmentRetry('closure_alignment_lost');
    }
    this.updatePayloadPredictionMode();
    try{
      const strict=!!(this.releaseExit?.segment&&['release','settle'].includes(this.phase)&&!this.releaseExit.segment.degraded);
      let ref;
      try{ref=this.reference.window(offset=>this.targetAt(offset),nowOf(this.scene),{reseed:this.reseed,force:this.forceReference,...(strict?{bestEffort:false}:{})});}
      catch(error){
        if(!strict||!(error instanceof IKError))throw error;
        // Once per checked segment: the strict measured-frame window is downgraded to the ordinary best-effort window instead of stopping.
        this.releaseExit.segment.degraded=true;this.releaseExit.collisionFrame=null;this.recordPostReleaseHazard('release_reference_best_effort',{note:error.message});
        ref=this.reference.window(offset=>this.targetAt(offset),nowOf(this.scene),{reseed:this.reseed,force:true});
      }
      if(this.reference.lastAudit?.constraintPassed===false)this.markCompletionFallback('approximate_ik');
      this.reseed=false;this.forceReference=false;const state=this.scene.readState(this.objectId),target=await this.policy.control(ref,this.reference.currentIndex,state);
      if(target.length!==29||!Array.from(target).every(Number.isFinite))throw new Error('Policy returned invalid body joint targets.');
      const finalElapsed=nowOf(this.scene)-this.phaseTime;let closure=this.phase==='approach'&&this.alignmentRetry?this.alignmentRetry.closure*(1-smooth((nowOf(this.scene)-this.alignmentRetry.startedAt)/.6)):
        this.phase==='close'?smooth(this.closeProgress):['lift','hold','stand','transfer'].includes(this.phase)?1:this.phase==='release'?
        this.graspAbort?this.graspAbort.closure*(1-smooth(finalElapsed/.6)):1-smooth(finalElapsed/.8):this.phase==='return_home'&&this.fistReturn?1:0;
      if(this.phase==='close'&&closure>0&&!this.closureCommitted){
        if(this.lastClosureGate?.allowed===true)this.closureCommitted={time_s:nowOf(this.scene),hand:this.hand,object_id:this.objectId,forced:this.forcedClose,alignment:structuredClone(this.graspAlignment()),gate:structuredClone(this.lastClosureGate)};
        else{closure=0;this.closeProgress=0;}
      }
      if(['lift','hold','stand','transfer'].includes(this.phase))this.recordClosureGate(true,'carry',null,null,'authorized_lift_chain');
      else if(this.phase!=='close')this.recordClosureGate(false,this.phase==='approach'&&this.alignmentRetry?'retry_opening':this.phase==='release'?'release':'open',null,null,this.graspAbort?.reason??null);
      this.scene.setHandClosure(this.hand,closure,this.objectId);this.scene.setHandClosure(otherHand(this.hand),0);
      this.scene.step(target);this.measure();
      if(this.releaseExit?.segment&&['release','settle'].includes(this.phase)){
        // Substep contacts and measured clearance are recorded against the
        // progress earned BEFORE this physics step; they never stop the episode.
        const peak=this.releaseContactViolation(this.scene.lastStepContacts??[]),geometry=this.releaseMeasuredClearance();
        if(!geometry.passed)this.recordPostReleaseHazard('release_exit_clearance',{geometry,note:'substep'});
        if(peak)this.recordPostReleaseHazard('release_object_contact',{limitingPair:peak,note:'substep'});
      }
      const actualSelf=this.audit.check(this.scene.data,{selfOnly:true,margin:-MEASURED_SELF_OVERLAP_TOLERANCE_M,detect:.001});this.actualSelfClearance=actualSelf.minimumClearance;this.actualSelfPair=actualSelf.limitingPair?['a','b'].map(k=>this.scene.metadata?.bodyNames?.[this.scene.model.geom_bodyid[actualSelf.limitingPair[k]]]??actualSelf.limitingPair[k]):null;
      if(!actualSelf.passed){if(this.postRelease)this.recordPostReleaseHazard('self_clearance_violation',{geometry:actualSelf});else{this.fail('The robot links lost measured self-clearance. Motion stopped.',new IKError('Measured robot self-intersection.',{geometry:actualSelf}));return this.snapshot();}}
    }catch(error){this.fail(error instanceof IKError?'The reference contains invalid numerical state. Motion stopped.':'Policy execution failed. Motion stopped.',error,{terminal:true});return this.snapshot();}
    const state=this.scene.readState(this.objectId),forceLimit=this.phase==='return_home'?5:15,penetrationLimit=this.phase==='return_home'?.002:.003;
    // Body / hand-fixture contact: the dwell timer runs outside the guard chain so the later balance and object checks still execute.
    if(this.bodyFixtureForce>BODY_FIXTURE_FORCE_LIMIT_N)this.bodyFixtureForceSince??=nowOf(this.scene);else this.bodyFixtureForceSince=null;
    const bodyPush=this.bodyFixtureForceSince!=null&&nowOf(this.scene)-this.bodyFixtureForceSince>=BODY_FIXTURE_FORCE_DWELL_S-1e-9;
    if(this.handTableForce>forceLimit||this.handTablePenetration>penetrationLimit)this.fail('The hand contacted the tabletop too strongly. Motion stopped.');
    else if(bodyPush)this.fail(this.phase==='return_home'?'The robot contacted a fixture during return.':'The robot body contacted the table or tray. Motion stopped.');
    else if(!Array.from(this.scene.data.qpos).every(Number.isFinite)||state.rootPosW[2]<.48)this.fail('The robot lost balance. Reset and try another object position.',null,{terminal:true});
    else if(poseOf(this.scene,this.objectId).position[2]<tableZ(this.scene)-.10){
      this.markCompletionFallback('object_fell');
      if(['approach','close'].includes(this.phase))this.beginForcedClose('object_fell');
      else if(['lift','hold'].includes(this.phase)&&this.graspSuccess!==true)this.beginGraspAbort('object_fell');
      else if(['hold','stand','transfer'].includes(this.phase)&&!this.completionTransfer)this.beginCompletionTransfer('object_fell');
    }
    return this.snapshot();
  }
  snapshot(){
    const root=this.scene.readState?.(this.objectId),frame=this.reference.currentFrame();
    return {attempt_notice:this.attemptNotice??null,final_attempt:this.finalAttempt||false,final_attempt_trigger:this.finalAttemptTrigger??null,body_fixture_force_n:this.bodyFixtureForce??null,release_hand_opening:this.releaseHandOpening??null,release_exit:this.releaseExit??null,post_release_hazards:this.postReleaseHazards?.slice()||[],post_release_deadline_s:this.postReleaseDeadline??null,home_stage_visit:this.homeStageVisit||0,home_clearance_retries:this.homeClearanceRetries||0,home_clearance_violations:this.homeClearanceViolations||0,home_retract_head_corridor:this.homeRetractHeadCorridor??null,lower_support:this.lowerSupport?{...this.lowerSupport,steady_s:this.lowerContactHold}:null,home_hand_opening:this.homeHandOpening??null,ever_placed:this.everPlaced===true,tray_landing:this.trayLanding??null,closure_committed:this.closureCommitted??null,closure_rejected_trigger:this.closureRejectedTrigger??null,alignment_retry:this.alignmentRetry?{count:this.alignmentRetryCount,started_at_s:this.alignmentRetry.startedAt,deadline_s:this.alignmentRetry.deadline,prepared:this.alignmentRetry.prepared,translation_m:this.alignmentRetry.translationM,rotation_rad:this.alignmentRetry.rotationRad,rejected:this.alignmentRetry.rejected??null}:null,grasp_abort:this.graspAbort?{reason:this.graspAbort.reason,started_at_s:this.graspAbort.startedAt}:null,phase:this.phase,stage:this.stage,busy:this.busy,message:this.message,failure_reason:(this.phase==='failed'||this.recoveryReturn)?(this.failureReason??(this.success===false&&this.graspSuccess!==true?this.forcedCloseObstruction??null:null)):null,approach_obstruction:this.approachObstruction?.confirmed??null,forced_close_obstruction:this.forcedCloseObstruction??null,grasp_obstruction_candidate:this.graspObstructionCandidate??null,motion_completed:!!this.motionCompleted,completion_fallback:!!this.completionFallback,completion_reasons:this.completionReasons?.slice()||[],forced_close_trigger:this.forcedCloseTrigger??null,failure_phase:this.failurePhase,grasp_distance_m:this.graspDistance(),grasp_tolerance_m:this.gates.normal,grasp_angle_tolerance_rad:GRASP_ANGLE_TOLERANCE_RAD,grasp_alignment:this.graspAlignment(),near_goal_attempt:this.nearGoalAttemptStatus(),finger_contact_count:this.fingerContactCount,grip_contact_s:this.gripContactTime,payload_prediction:{active:!!this.payloadPredictionEnabled,transitions:this.payloadPredictionTransitions?.slice()||[]},close_trigger:this.closeTrigger,object_id:this.objectId,hand:this.hand,mode:this.mode,success:this.success,grasp_success:this.graspSuccess,deposit_success:this.depositSuccess,return_success:this.returnSuccess,in_tray:this.inTray,lift_m:this.lift,max_lift_m:this.maxLift,contacts:this.contacts,hand_contacts:this.handContacts,tray_contacts:this.trayContacts,tray_overlap_area:this.trayOverlap||0,stable_lift_s:this.stableLift,grasp_support:this.graspSupport?{...this.graspSupport}:null,deposit_stable_s:this.stableDeposit,replan_enabled:this.replanEnabled,goal_adjust_enabled:this.goalAdjustEnabled,replan_count:this.replans,goal_adjust_count:this.gaUpdates,goal_adjustment:this.goalAdjustment.slice(),elapsed_s:nowOf(this.scene)-this.startTime,crouch_enabled:!!this.stancePlan,crouch_height:this.crouchHeight??null,stance_plan:stanceSummary(this.stancePlan),stance_history:(this.stanceHistory||[]).map(stanceSummary),stance_replan_count:this.stanceReplanCount||0,ga_frozen_reason:this.gaFrozenReason||null,reach_tracking:this.reachTracking,planned_root_position_w:frame.rootPosW.slice(),planned_root_pitch_rad:frame.plannedRootPitch,actual_root_quaternion_wxyz:root?.rootQuatW?Array.from(root.rootQuatW):null,actual_root_velocity_w:root?.rootLinVelW?Array.from(root.rootLinVelW):null,home_stage:this.homeStage??null,home_joint_error:this.homeJointError??null,home_palm_error:this.homePalmError??null,home_angle_error:this.homeAngleError??null,home_obstacle_clearance:this.homeClearance??null,reference_audit:this.reference.lastAudit,self_collision_pairs:this.audit.selfPairs.length,actual_self_clearance:this.actualSelfClearance,actual_self_clearance_pair:this.actualSelfPair??null,top_down_table_release:this.topDownTableRelease??null,approach_target_palm_w:this.approachTargetPalmDiagnostic(),failure_message:this.failureMessage??null,recovery_return:!!this.recoveryReturn,fist_return:!!this.fistReturn,home_retract_outboard_first:this.homeRetractOutboardFirst??null,home_retracted_w:this.homeRetracted&&this.hand&&this.homeRetracted[this.hand]?copy(this.homeRetracted[this.hand]):null,body_fixture_force_n:this.bodyFixtureForce,body_fixture_penetration_m:this.bodyFixturePenetration,hand_table_force_n:this.handTableForce,hand_table_penetration_m:this.handTablePenetration,rejected_measured_ik_seeds:this.reference.ik.rejectedMeasuredSeeds||0,error:this.lastError,target_ghost:this.plan||this.safePreview(),backend:'mujoco_wasm_onnx',ik_backend:'browser_bounded_dls',implementation_note:'Browser DLS uses native geometry, feet and COM checks; numerical iterates are not identical to native Mink.'};
  }
}

export {InteractiveController as InteractiveGraspController};
