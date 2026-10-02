import {add,sub,scale,dot,norm,lerp,cross,clamp,matVec,matMul,transpose3,quatToMat,matToQuat,rotateY,rotationError,boundedDLS,convexHull,polygonMargin} from './numerics.js';

// Squared hinge tasks for the inset support polygon. Their correction tends
// continuously to zero at each soft boundary; the 2 cm hard gate is separate.
export function continuousComTasks(point,polygon,margin=.035){
  return polygon.map((a,i)=>{const b=polygon[(i+1)%polygon.length],dx=b[0]-a[0],dy=b[1]-a[1],length=Math.max(Math.hypot(dx,dy),1e-12),normal=[-dy/length,dx/length],distance=dot(normal,sub(point.slice(0,2),a));return {normal,error:Math.max(0,margin-distance)};}).filter(task=>task.error>0);
}

const SIDES=['left','right'];
const corners=[];for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])corners.push([x,y,z]);
const copy=v=>Array.from(v);
export class IKError extends Error { constructor(message,details={}){super(message);this.name='IKError';this.details=details;} }
export function unmetIKConstraints(residual,geometry,{strictEndpoint=false,trackPosture=false}={}){
  const unmet=[];
  if(!geometry.passed)unmet.push({constraint:'clearance',actual:geometry.minimumClearanceSurplus,limit:0});
  if(residual.footPositionError>=.003)unmet.push({constraint:'foot_position',actual:residual.footPositionError,limit:.003});
  if(residual.footRotationError>=.05)unmet.push({constraint:'foot_rotation',actual:residual.footRotationError,limit:.05});
  if(residual.comMargin<.02)unmet.push({constraint:'com_support',actual:residual.comMargin,limit:.02});
  if(strictEndpoint){
    if(trackPosture){if(residual.upperPostureError>=.05)unmet.push({constraint:'posture',actual:residual.upperPostureError,limit:.05});}
    else for(const [side,palm]of Object.entries(residual.palms)){
      if(palm.positionError>=.012)unmet.push({constraint:'palm_position',side,actual:palm.positionError,limit:.012});
      if(palm.rotationError>=.12)unmet.push({constraint:'palm_rotation',side,actual:palm.rotationError,limit:.12});
    }
  }
  return unmet;
}
export const bodyPosition=(data,id)=>copy(data.xpos.slice(id*3,id*3+3));
export const bodyRotation=(data,id)=>copy(data.xmat.slice(id*9,id*9+9));
export const bodyPose=(data,id)=>({position:bodyPosition(data,id),rotation:bodyRotation(data,id)});

export function geomCorners(scene,data,gid) {
  const a=scene.model.geom_aabb?.slice(gid*6,gid*6+6),size=scene.model.geom_size.slice(gid*3,gid*3+3);
  if(!a||a.length!==6)throw new IKError('The native runtime must expose compiled geometry bounds.');
  const position=copy(data.geom_xpos.slice(gid*3,gid*3+3)),rotation=copy(data.geom_xmat.slice(gid*9,gid*9+9));
  return corners.map(sign=>add(position,matVec(rotation,[0,1,2].map(k=>a[k]+sign[k]*a[k+3]))));
}
export function geomBounds(scene,data,ids) {
  const lower=[Infinity,Infinity,Infinity],upper=[-Infinity,-Infinity,-Infinity];
  for(const gid of ids){if(scene.geomBounds){const b=scene.geomBounds(data,gid);for(let k=0;k<3;k++){lower[k]=Math.min(lower[k],b.lower[k]);upper[k]=Math.max(upper[k],b.upper[k]);}}else for(const p of geomCorners(scene,data,gid))for(let k=0;k<3;k++){lower[k]=Math.min(lower[k],p[k]);upper[k]=Math.max(upper[k],p[k]);}}
  return {lower,upper};
}
const boxGap=(a,b)=>Math.hypot(
  Math.max(0,a.lower[0]-b.upper[0],b.lower[0]-a.upper[0]),
  Math.max(0,a.lower[1]-b.upper[1],b.lower[1]-a.upper[1]),
  Math.max(0,a.lower[2]-b.upper[2],b.lower[2]-a.upper[2]));
const getName=(scene,id)=>scene.bodyName?scene.bodyName(id):(scene.bodyNames?.[id]||'');

/** Native collision shapes are queried without changing any collision mask. */
/** Fitted shoulder-yaw / trunk shells may interpenetrate by this much before the audit fails. */
export const UPPER_ARM_TRUNK_SHELL_OVERLAP_M=.003;
export class CollisionAudit {
  constructor(scene) {
    this.scene=scene;const m=scene.model;this.robotGeoms=[];this.hands={left:[],right:[]};this.fingers=new Map();
    this.fixtureGeoms=[...(scene.tableGeomIds||[]),...(scene.trayGeomIds||[])];this.floorGeoms=[];
    let pelvis=scene.rootBodyId??scene.bodyReferenceIds32?.[0];if(pelvis==null)throw new IKError('Missing pelvis body mapping.');
    const robotBody=b=>{while(b>0&&b!==pelvis)b=m.body_parentid[b];return b===pelvis;};
    for(let g=0;g<m.ngeom;g++){
      if(!(m.geom_contype[g]||m.geom_conaffinity[g]))continue;const b=m.geom_bodyid[g],name=getName(scene,b);
      if(m.geom_type[g]===0)this.floorGeoms.push(g);
      if(!robotBody(b))continue;this.robotGeoms.push(g);
      for(const side of SIDES)if(name.startsWith(`${side}_hand_`)||name===`${side}_wrist_yaw_link`){this.hands[side].push(g);for(const finger of ['thumb','index','middle','ring','pinky'])if(name.startsWith(`${side}_hand_${finger}_`))this.fingers.set(g,{side,finger});}
    }
    this.handSet=new Set([...this.hands.left,...this.hands.right]);
    // Round tables use a dedicated solid top; rectangular tops keep the
    // original name. Never treat a visual mesh, table leg or tray as a top.
    const tableTops=(scene.tableGeomIds||[]).filter(g=>m.geom_contype[g]||m.geom_conaffinity[g]);
    this.tableTopGeomId=tableTops.find(g=>scene.geomName?.(g)==='demo_table_top_collision')??tableTops.find(g=>scene.geomName?.(g)==='demo_table_0')??-1;
    this.supportGeoms=new Set(this.robotGeoms.filter(g=>SIDES.some(s=>getName(scene,m.geom_bodyid[g])===`${s}_ankle_roll_link`)));
    this.selfPairs=[];this.pairCache=new Map();this.pairMarginOverrides=new Map();this.pairMarginKeys=new WeakMap();const excluded=new Set(copy(m.exclude_signature||[]));
    // Native physics disables robot self-contact. Planning still checks the
    // relevant arm/body pairs geometrically, independently of those masks.
    // As in hero_reach_generator, joined links (up to two joints), siblings,
    // welded links and the internals of one Dex3 hand are intentional overlaps.
    const weld=b=>m.body_weldid?m.body_weldid[b]:b;
    const parent=b=>weld(m.body_parentid[weld(b)]);
    const ancestors=b=>new Set([weld(b),parent(b),parent(parent(b))]);
    const isArm=name=>/shoulder|elbow|wrist|_hand_/.test(name);
    const handSide=name=>(name.includes('_hand_')||name.includes('_wrist_'))?(name.startsWith('left_')?'left':name.startsWith('right_')?'right':null):null;
    for(let i=0;i<this.robotGeoms.length;i++)for(let j=i+1;j<this.robotGeoms.length;j++){
      const a=this.robotGeoms[i],b=this.robotGeoms[j],ba=m.geom_bodyid[a],bb=m.geom_bodyid[b];
      const na=getName(scene,ba),nb=getName(scene,bb);if(!isArm(na)&&!isArm(nb))continue;
      if(ancestors(ba).has(weld(bb))||ancestors(bb).has(weld(ba))||parent(ba)===parent(bb))continue;
      if(handSide(na)&&handSide(na)===handSide(nb))continue;
      if(excluded.has((Math.min(ba,bb)<<16)+Math.max(ba,bb)))continue;
      this.selfPairs.push([a,b]);
      const isUpperArm=n=>n==='left_shoulder_yaw_link'||n==='right_shoulder_yaw_link';
      const isTrunk=body=>['pelvis','torso_link','waist_yaw_link','waist_roll_link'].includes(getName(scene,weld(body)));
      if(isUpperArm(na)&&isTrunk(bb)||isUpperArm(nb)&&isTrunk(ba))this.pairMarginOverrides.set(`${a}:${b}`,-UPPER_ARM_TRUNK_SHELL_OVERLAP_M);
    }
  }
  physicalObjectGeoms(object) {
    const m=this.scene.model,metadata=this.scene.metadata;
    const contype=metadata?.originalGeomContype??m.geom_contype,conaffinity=metadata?.originalGeomConaffinity??m.geom_conaffinity;
    return (object?.geomIds||[]).filter(g=>contype[g]||conaffinity[g]);
  }
  objectGeoms(exclude=null) {return Object.entries(this.scene.objects||{}).filter(([id,o])=>o.active&&id!==exclude).flatMap(([,o])=>this.physicalObjectGeoms(o));}
  nativeDistance(data,a,b,maxDistance=.05){const v=this.scene.geomDistance(data,a,b,maxDistance);const d=typeof v==='number'?v:v?.distance;if(!Number.isFinite(d))throw new IKError('Native geometry distance returned a non-finite value.',{a,b});return d;}
  pairs({hand='right',objectId=null,allowHandObject=false,includeSelf=true,handsOnly=false,selfOnly=false,carriedPayload=null,ignoreObjectId=null}={}){
    if(selfOnly)return this.selfPairs;
    const key=[hand,objectId,allowHandObject,includeSelf,handsOnly,carriedPayload?.objectId??'',ignoreObjectId??'',...Object.keys(this.scene.objects||{}).filter(id=>this.scene.objects[id].active)].join('|');if(this.pairCache.has(key))return this.pairCache.get(key);
    const selected=new Set(this.hands[hand]),obj=this.scene.objects?.[objectId],activeObject=new Set(this.physicalObjectGeoms(obj));
    const robot=handsOnly?[...this.hands.left,...this.hands.right]:this.robotGeoms,obstacles=[...this.fixtureGeoms,...this.objectGeoms(ignoreObjectId)];
    const pairs=[];for(const a of robot)for(const b of obstacles){if(allowHandObject&&selected.has(a)&&activeObject.has(b))continue;pairs.push([a,b]);}
    if(!handsOnly)for(const a of robot)if(!this.supportGeoms.has(a))for(const b of this.floorGeoms)pairs.push([a,b]);
    // A carried object's planned pose must clear fixtures and other objects as
    // well as the robot. Wrist/payload pairs above retain their usual checks.
    const payload=this.scene.objects?.[carriedPayload?.objectId];
    if(payload?.active&&!handsOnly)for(const a of this.physicalObjectGeoms(payload))for(const b of [...this.fixtureGeoms,...this.floorGeoms,...this.objectGeoms(carriedPayload.objectId)])pairs.push([a,b]);
    if(includeSelf&&!handsOnly)pairs.push(...this.selfPairs);this.pairCache.set(key,pairs);return pairs;
  }
  marginKey(pair,a,b){
    let cached=this.pairMarginKeys.get(pair);
    // Cache only the canonical key, never the current override value. Keep
    // externally updated overrides and edited pair arrays visible immediately.
    if(!cached||cached.a!==a||cached.b!==b){cached={a,b,key:`${Math.min(a,b)}:${Math.max(a,b)}`};this.pairMarginKeys.set(pair,cached);}
    return cached.key;
  }
  check(data,options={}){
    if(options.releaseCollisionData)data=options.releaseCollisionData;
    const margin=options.margin??.005,detect=Math.max(options.detect??.025,margin),bounds=[],rows=[];let minimum=Infinity,nearest=null,minimumSurplus=Infinity,minimumBufferedSurplus=Infinity,limitingPair=null;
    let geomTypes=this.scene.model.geom_type;
    const auditPairs=this.pairs(options),separationFloors=new Map(),separation=options.releaseSeparation;
    if(separation){
      const objectGeoms=new Set(this.physicalObjectGeoms(this.scene.objects?.[options.objectId]));
      if(options.ignoreObjectId!=null&&options.ignoreObjectId===options.objectId)throw new IKError('Release separation cannot constrain an ignored object.');
      if(!this.scene.objects?.[options.objectId]?.active||!objectGeoms.size||separation.hand!==(options.hand||'right')||separation.objectId!==options.objectId||options.allowHandObject||options.carriedPayload||options.selfOnly||options.handsOnly||options.includeSelf===false||!Array.isArray(separation.pairs)||!separation.pairs.length)throw new IKError('Invalid measured release-separation contract.');
      const presentPairs=new Set(auditPairs.map(pair=>this.marginKey(pair,...pair)));
      for(const pair of separation.pairs){
        if(!pair||typeof pair!=='object')throw new IKError('Release separation requires explicit pair records.');
        const name=getName(this.scene,this.scene.model.geom_bodyid[pair.a]),key=`${Math.min(pair.a,pair.b)}:${Math.max(pair.a,pair.b)}`;
        if(!Number.isInteger(pair.a)||!Number.isInteger(pair.b)||!this.robotGeoms.includes(pair.a)||!objectGeoms.has(pair.b)||!(name.startsWith(`${separation.hand}_hand_`)||name.startsWith(`${separation.hand}_wrist_`))||!Number.isFinite(pair.minimumDistance)||pair.minimumDistance<(separation.initialContactRelief===true&&pair.existingContact===true?-.0002:0)||pair.minimumDistance>=margin||separationFloors.has(key)||!presentPairs.has(key))throw new IKError('Release separation can only constrain measured close hand/object pairs within the explicit contact-relief band.');
        separationFloors.set(key,pair.minimumDistance);
      }
    }
    const boundsQuery=this.scene.geomBoundsQuery?.(data),payload=options.carriedPayload,payloadGeoms=new Set(this.physicalObjectGeoms(this.scene.objects?.[payload?.objectId]));
    const box=g=>bounds[g]??(bounds[g]=boundsQuery?boundsQuery(g):geomBounds(this.scene,data,[g]));
    const graspData=options.payloadGraspData,graspBounds=[],graspQuery=graspData&&this.scene.geomBoundsQuery?.(graspData);
    const graspBox=g=>graspBounds[g]??(graspBounds[g]=graspQuery?graspQuery(g):geomBounds(this.scene,graspData,[g]));
    for(const pair of auditPairs){
      const [a,b]=pair;
      if(!geomTypes.length)geomTypes=this.scene.model.geom_type;
      // Robot/load pairs describe the measured grasp configuration. Load/
      // environment pairs describe its world sweep. Both keep native shapes
      // and margins; the reference's tracking lag must not deform the grasp.
      const graspFrame=!!graspData&&payloadGeoms.has(b),pairData=graspFrame?graspData:data;
      const floor=geomTypes[b]===0;let d=floor?0:boxGap(box(a),graspFrame?graspBox(b):box(b));
      if(!Number.isFinite(d))throw new IKError('Native geometry bounds returned a non-finite gap.',{a,b});
      if(d<=detect||separationFloors.has(this.marginKey(pair,a,b)))d=this.nativeDistance(pairData,a,b,detect+.001);
      const handTable=b===this.tableTopGeomId&&this.hands[options.hand||'right'].includes(a),tableTouch=options.allowHandTableContact&&handTable;
      // Top Down floor release: the caller may trade MEASURED hand/tabletop clearance for a planned
      // interpenetration of the active hand with the tabletop (see controller topDownTableAllowance).
      const allowance=handTable&&!tableTouch?Math.max(0,options.handTableAllowance??0):0;
      const payloadGeom=payloadGeoms.has(a);
      const payloadSupport=payloadGeom&&(payload.allowTableContact&&b===this.tableTopGeomId||payload.allowTraySupport&&b===this.scene.trayBottomGeomId);
      const normalMargin=tableTouch||payloadSupport?0:Math.min(margin,this.pairMarginOverrides.get(this.marginKey(pair,a,b))??margin)-allowance;
      const requiredMargin=separationFloors.get(this.marginKey(pair,a,b))??normalMargin,surplus=d-requiredMargin,buffer=requiredMargin>0?(options.optimizationBuffer??0):0;
      const row=surplus<minimumSurplus||d<detect?{a,b,distance:d,requiredMargin,...(options.releaseCollisionData?{poseFrame:'measured_release_pose'}:graspFrame?{poseFrame:'measured_grasp'}:{})}:null;
      if(d<minimum){minimum=d;nearest=[a,b];}if(surplus<minimumSurplus){minimumSurplus=surplus;limitingPair=row;}minimumBufferedSurplus=Math.min(minimumBufferedSurplus,surplus-buffer);if(d<detect)rows.push(row);
    }
    rows.sort((a,b)=>(a.distance-a.requiredMargin)-(b.distance-b.requiredMargin));return {passed:minimumSurplus>=-1e-8,minimumClearance:minimum,minimumClearanceSurplus:minimumSurplus,minimumBufferedSurplus,limitingPair,nearest,near:rows,margin,...(options.handTableAllowance>0?{handTableAllowance:options.handTableAllowance}:{}),selfPairCount:options.includeSelf===false||options.handsOnly?0:this.selfPairs.length,distanceSemantics:'Native shape distance within the detection band; conservative AABB/cutoff lower bounds farther away. Named upperarm/trunk pairs retain the native zero-margin joint-shell rule.'};
  }
  handBounds(data,side){return geomBounds(this.scene,data,this.hands[side]);}
  handTableClearance(data,side){const b=this.handBounds(data,side);return b.lower[2]-this.scene.tableTopZ;}
  // Native (mesh) distance between the hand and the tabletop; the bounding-box clearance above is a
  // lower bound that sits about 8 mm under the true mesh distance for the open Dex3 fingers.
  handTableDistance(data,side,cutoff=.06){
    let distance=Infinity;
    for(const g of this.hands[side]){const gap=geomBounds(this.scene,data,[g]).lower[2]-this.scene.tableTopZ;if(gap>cutoff){distance=Math.min(distance,gap);continue;}distance=Math.min(distance,this.nativeDistance(data,g,this.tableTopGeomId,cutoff+.001));}
    return distance;
  }
}

/** Constrained browser DLS: explicit planned root, planted feet, both palms and COM.
 * This is a numerical JS port, not a claim of identical Mink QP iterates.
 * Only independent MjData is written; live execution is owned by the policy.
 */
export class WholeBodyIK {
  constructor(scene,{jointMargin=.02,maxJointSpeed=1.75,collisionMargin=.005,optimizationBuffer=.002}={}){
    this.scene=scene;this.model=scene.model;this.data=scene.scratchData();this.graspData=scene.scratchData();this.releaseData=scene.scratchData();this.audit=new CollisionAudit(scene);
    this.qadr=copy(scene.jointQposAddresses29);this.vadr=copy(scene.jointDofAddresses29);this.jointMargin=jointMargin;this.maxJointSpeed=maxJointSpeed;this.collisionMargin=collisionMargin;this.optimizationBuffer=Math.max(0,optimizationBuffer);
    if(this.qadr.length!==29||this.vadr.length!==29)throw new IKError('IK requires the explicit 29-joint native address maps.');
    // The official WASM 3.13 bool-array getter is unregistered in Embind.
    // Read the exact native flags exported with the scene, never infer limits.
    const limitedJoints=scene.jointLimited??this.model.jnt_limited;
    if(limitedJoints.length!==this.model.njnt)throw new IKError('Missing native joint-limit flags.');
    this.lower=[];this.upper=[];for(const a of this.qadr){let j=-1;for(let i=0;i<this.model.njnt;i++)if(this.model.jnt_qposadr[i]===a){j=i;break;}if(j<0)throw new IKError('Missing native body joint.');const limited=limitedJoints[j];this.lower.push(limited?this.model.jnt_range[2*j]+jointMargin:-10);this.upper.push(limited?this.model.jnt_range[2*j+1]-jointMargin:10);}
    this.reset();
  }
  reset(){
    this.template=copy(this.scene.data.qpos);this.data.qpos.set(this.template);this.forward();
    this.anchorPosition=this.template.slice(0,3);this.anchorRotation=quatToMat(this.template.slice(3,7));this.defaultQ=this.qadr.map(a=>this.template[a]);
    this.feet={};for(const side of SIDES)this.feet[side]=bodyPose(this.data,this.scene.ankleBodyIds[side]);
    const points=[];for(const g of this.audit.supportGeoms)points.push(...geomCorners(this.scene,this.data,g));
    this.supportPolygon=convexHull(points);if(this.supportPolygon.length<3)throw new IKError('Missing native foot support geometry.');
    this.initialPalms={};for(const side of SIDES)this.initialPalms[side]=bodyPose(this.data,this.scene.palmBodyIds[side]);
    this.torsoBodyId=this.scene.torsoBodyId??[...Array(this.model.nbody).keys()].find(i=>getName(this.scene,i)==='torso_link');
    this.rootBodyId=this.scene.rootBodyId??this.scene.bodyReferenceIds32[0];
    const torso=bodyPose(this.data,this.torsoBodyId);this.relativePalms={};
    for(const side of SIDES)this.relativePalms[side]={position:matVec(transpose3(torso.rotation),sub(this.initialPalms[side].position,torso.position)),rotation:matMul(transpose3(torso.rotation),this.initialPalms[side].rotation)};
  }
  dispose(){this.data.delete?.();this.graspData.delete?.();this.releaseData.delete?.();}
  forward(){if(this.scene.ikForward)this.scene.ikForward(this.data);else this.scene.forward(this.data);}
  rootPositionFor(target={}){
    const position=copy(target.rootPosition??this.anchorPosition);
    if(position.length!==3||!position.every(Number.isFinite))throw new IKError('Root position must contain three finite world coordinates.');
    position[2]=target.rootHeight??position[2];
    if(!Number.isFinite(position[2]))throw new IKError('Root height must be finite.');
    return position;
  }
  setPose(q,target={}){
    this.data.qpos.set(this.template);this.data.qpos.set(this.rootPositionFor(target),0);
    this.data.qpos.set(matToQuat(matMul(this.anchorRotation,rotateY(target.rootPitch??0))),3);
    this.qadr.forEach((a,i)=>this.data.qpos[a]=q[i]);
    // Object state is measured on every solve; no trajectory is assigned to it.
    for(const obj of Object.values(this.scene.objects||{}))if(obj.active&&obj.qadr!=null)this.data.qpos.set(this.scene.data.qpos.slice(obj.qadr,obj.qadr+7),obj.qadr);
    if(this.scene.fingerQposAddresses14)for(const a of this.scene.fingerQposAddresses14)this.data.qpos[a]=this.scene.data.qpos[a];
    this.forward();
    const payload=target.carriedPayload;
    if(payload){
      const object=this.scene.objects?.[payload.objectId];
      if(!object?.active||!SIDES.includes(payload.hand)||payload.objectId!==target.objectId||payload.hand!==(target.hand||'right')||payload.positionInPalm?.length!==3||payload.rotationInPalm?.length!==9||![...payload.positionInPalm,...payload.rotationInPalm].every(Number.isFinite))throw new IKError('Carried payload prediction needs a finite selected-hand transform.');
      const palm=bodyPose(this.data,this.scene.palmBodyIds[payload.hand]);
      if(payload.graspPositionInPalm!==undefined||payload.graspRotationInPalm!==undefined){
        if(payload.graspPositionInPalm?.length!==3||payload.graspRotationInPalm?.length!==9||![...payload.graspPositionInPalm,...payload.graspRotationInPalm].every(Number.isFinite))throw new IKError('Carried payload internal clearance needs a finite measured grasp transform.');
        this.graspData.qpos.set(this.data.qpos);
        this.graspData.qpos.set(add(palm.position,matVec(palm.rotation,payload.graspPositionInPalm)),object.qadr);
        this.graspData.qpos.set(matToQuat(matMul(palm.rotation,payload.graspRotationInPalm)),object.qadr+3);
        if(this.scene.ikForward)this.scene.ikForward(this.graspData);else this.scene.forward(this.graspData);
      }
      this.data.qpos.set(add(palm.position,matVec(palm.rotation,payload.positionInPalm)),object.qadr);
      this.data.qpos.set(matToQuat(matMul(palm.rotation,payload.rotationInPalm)),object.qadr+3);
      // FK only in independent MjData. No live qpos, qvel or weld is changed.
      this.forward();
    }
    this.setReleaseCollisionPose(q,target);
  }
  setReleaseCollisionPose(q,target){
    const opening=target.releaseOpeningCollisionFrame,frame=target.releaseCollisionFrame??opening;
    if(!frame)return;
    // Opening keeps the grasp's finger contact permission. Detached segments
    // retain their separate, stricter separation contract.
    if(opening&&(target.releaseCollisionFrame||target.allowHandObject!==true||target.lockWaist!==true||target.ignoreObjectId!=null||target.releaseSeparation))throw new IKError('Invalid opening collision frame.');
    const vectors=[frame.measuredRootPosition,frame.referenceRootPosition,frame.measuredRootRotation,frame.referenceRootRotation,frame.measuredLowerQ,frame.referenceLowerQ,frame.measuredUpperQ,frame.referenceUpperQ],lengths=[3,3,9,9,12,12,17,17];
    if(frame.hand!==(target.hand||'right')||frame.objectId!==target.objectId||!this.scene.objects?.[target.objectId]?.active||(!opening&&target.allowHandObject)||target.carriedPayload||!target.lockLowerBody||vectors.some((v,i)=>v?.length!==lengths[i]||!Array.from(v).every(Number.isFinite)))throw new IKError('Invalid release collision frame.');
    const data=this.releaseData;data.qpos.set(this.scene.data.qpos);
    const nominalPosition=copy(this.data.qpos.slice(0,3)),nominalRotation=quatToMat(copy(this.data.qpos.slice(3,7)));
    if(norm(sub(nominalPosition,frame.referenceRootPosition))>1e-9||norm(rotationError(nominalRotation,frame.referenceRootRotation))>1e-9||q.slice(0,12).some((v,i)=>Math.abs(v-frame.referenceLowerQ[i])>1e-9))throw new IKError('Checked release requires its fixed nominal root and lower-body frame.');
    data.qpos.set(add(frame.measuredRootPosition,sub(nominalPosition,frame.referenceRootPosition)),0);
    data.qpos.set(matToQuat(matMul(matMul(nominalRotation,transpose3(frame.referenceRootRotation)),frame.measuredRootRotation)),3);
    this.qadr.forEach((a,i)=>{const predicted=i<12?frame.measuredLowerQ[i]+q[i]-frame.referenceLowerQ[i]:frame.measuredUpperQ[i-12]+q[i]-frame.referenceUpperQ[i-12];
      if(i>=12&&(!Number.isFinite(predicted)||predicted<this.lower[i]-1e-9||predicted>this.upper[i]+1e-9))throw new IKError('Predicted release upper joints violate their safety margin.',{joint:i,predicted,lower:this.lower[i],upper:this.upper[i]});data.qpos[a]=predicted;});
    // The detached object and physical fingers retain their exact measured
    // state. Robot commands move the measured full pose by reference deltas;
    // upper-body tracking bias is not an instantaneous physical motion.
    if(this.scene.ikForward)this.scene.ikForward(data);else this.scene.forward(data);
  }
  jacPoint(data,point,body){
    if(this.scene.jacPoint){const result=this.scene.jacPoint(data,point,body);return result.position||result;}
    const j=this.scene.jacBody(data,body),arm=sub(point,bodyPosition(data,body)),nv=this.model.nv,out=new Float64Array(j.position);
    for(let c=0;c<nv;c++){const v=cross([j.rotation[c],j.rotation[nv+c],j.rotation[2*nv+c]],arm);for(let r=0;r<3;r++)out[r*nv+c]+=v[r];}return out;
  }
  collisionPointJacobian(data,point,geom,target){
    const payload=target.carriedPayload,object=this.scene.objects?.[payload?.objectId];
    // Scratch payload witnesses move with the predicted palm, rather than the
    // object's unrelated free-joint columns (which the body IK does not own).
    const body=object?.geomIds.includes(geom)?this.scene.palmBodyIds[payload.hand]:this.model.geom_bodyid[geom];
    return this.jacPoint(data,point,body);
  }
  goals(target){
    const result={...(target.palms||{})};const relative=target.bodyRelativeHands||[],torso=bodyPose(this.data,this.torsoBodyId);
    for(const side of relative){const rest=this.relativePalms[side],offset=target.palmOffsetsTorso?.[side]||[0,0,0];result[side]={position:add(torso.position,matVec(torso.rotation,add(rest.position,offset))),rotation:matMul(torso.rotation,rest.rotation),relative:true};}
    return result;
  }
  residuals(target){
    const goals=this.goals(target),palms={};let footPositionError=0,footRotationError=0;
    for(const side of SIDES){const foot=bodyPose(this.data,this.scene.ankleBodyIds[side]);footPositionError=Math.max(footPositionError,norm(sub(foot.position,this.feet[side].position)));footRotationError=Math.max(footRotationError,norm(rotationError(this.feet[side].rotation,foot.rotation)));if(goals[side]){const pose=bodyPose(this.data,this.scene.palmBodyIds[side]);palms[side]={positionError:norm(sub(goals[side].position,pose.position)),rotationError:norm(rotationError(goals[side].rotation,pose.rotation))};}}
    const com=copy(this.data.subtree_com.slice(this.rootBodyId*3,this.rootBodyId*3+3));
    const upperPostureError=target.posture?Math.max(...this.qadr.slice(12).map((a,i)=>Math.abs(this.data.qpos[a]-target.posture[12+i]))):null;
    return {palms,upperPostureError,footPositionError,footRotationError,comMargin:polygonMargin(com,this.supportPolygon),com};
  }
  collisionOptions(target){return {hand:target.hand||'right',objectId:target.objectId,allowHandObject:!!target.allowHandObject,ignoreObjectId:target.ignoreObjectId??null,allowHandTableContact:!!target.allowHandTableContact,handTableAllowance:Math.max(0,Number(target.handTableAllowance)||0),carriedPayload:target.carriedPayload??null,payloadGraspData:target.carriedPayload?.graspPositionInPalm!==undefined?this.graspData:null,releaseSeparation:target.releaseSeparation??null,releaseCollisionData:target.releaseCollisionFrame||target.releaseOpeningCollisionFrame?this.releaseData:null,margin:target.collisionMargin??this.collisionMargin,includeSelf:true,optimizationBuffer:this.optimizationBuffer};}
  bestEffortCost(q,target,residual,geometry,qStart){
    // Keep the original task priorities, with extra penalties for crossing a
    // hard support/clearance margin. This ranks imperfect numerical candidates;
    // it never changes the collision geometry or the physical simulation.
    let cost=1e4*residual.footPositionError**2+400*residual.footRotationError**2
      +400*Math.max(0,.035-residual.comMargin)**2+4e4*Math.max(0,.02-residual.comMargin)**2
      +4e4*Math.max(0,-(geometry.minimumClearanceSurplus??0))**2;
    for(const pair of geometry.near||[])cost+=4e4*Math.max(0,pair.requiredMargin-pair.distance)**2;
    if(!target.trackPosture)for(const palm of Object.values(residual.palms))cost+=144*palm.positionError**2+(12*.16*palm.rotationError)**2;
    for(let i=0;i<29;i++){const weight=target.trackPosture?(target.trackUpperPostureOnly&&i<12?.06:3):i>=12&&i<15?(target.waistPostureWeights?.[i-12]??.06):.06;cost+=(weight*(target.posture[i]-q[i]))**2;}
    if(target.jointContinuityWeight>0)for(let i=12;i<29;i++)cost+=(target.jointContinuityWeight*(qStart[i]-q[i]))**2;
    if(!Number.isFinite(cost))throw new IKError('IK produced non-finite task residuals.');
    return cost;
  }
  solve(qStart,target,options={}){
    if(!options.bestEffort)return this.solveBounded(qStart,target,options);
    try{
      // Best effort changes rejection handling, not a working reference. Keep
      // the original solve bit-for-bit whenever its requested checks pass.
      const result=this.solveBounded(qStart,target,{...options,bestEffort:false});
      return {...result,bestEffort:true,bestEffortUsed:false};
    }catch(error){
      const infeasible=error instanceof IKError&&['The constrained IK target is infeasible.','No collision-free whole-body step exists.'].includes(error.message);
      if(!infeasible)throw error;
      const result=this.solveBounded(qStart,target,options);
      return {...result,bestEffortUsed:true,strictFailure:{message:error.message,details:error.details}};
    }
  }
  solveBounded(qStart,target,{iterations=12,maxStep=.035,strictEndpoint=false,seed=null,continuousCom=false,bestEffort=false}={}){
    if(!target||!Array.isArray(target.posture)&&!ArrayBuffer.isView(target.posture))throw new IKError('Missing IK rest posture.');
    if(qStart.length!==29||!Array.from(qStart).every(Number.isFinite)||target.posture.length!==29||!Array.from(target.posture).every(Number.isFinite)||seed&&(seed.length!==29||!Array.from(seed).every(Number.isFinite)))throw new IKError('IK joint positions and posture must contain 29 finite values.');
    if(!Number.isFinite(maxStep)||maxStep<0||!Number.isFinite(target.rootPitch??0))throw new IKError('IK step size and root pitch must be finite.');
    if(target.waistPostureWeights&&!(target.waistPostureWeights.length===3&&Array.from(target.waistPostureWeights).every(v=>Number.isFinite(v)&&v>0)))throw new IKError('Waist posture weights must be three positive finite values.');
    const jointContinuityWeight=target.jointContinuityWeight??0;
    if(!Number.isFinite(jointContinuityWeight)||jointContinuityWeight<0)throw new IKError('Joint continuity weight must be a nonnegative finite value.');
    const lower=this.lower.map((v,i)=>Math.max(v,qStart[i]-maxStep)),upper=this.upper.map((v,i)=>Math.min(v,qStart[i]+maxStep));
    const releaseFrame=target.releaseCollisionFrame??target.releaseOpeningCollisionFrame;
    if(releaseFrame){
      if(releaseFrame.measuredUpperQ?.length!==17||releaseFrame.referenceUpperQ?.length!==17||![...releaseFrame.measuredUpperQ,...releaseFrame.referenceUpperQ].every(Number.isFinite))throw new IKError('Invalid measured release upper-joint anchor.');
      for(let i=12;i<29;i++){const offset=releaseFrame.measuredUpperQ[i-12]-releaseFrame.referenceUpperQ[i-12];lower[i]=Math.max(lower[i],this.lower[i]-offset);upper[i]=Math.min(upper[i],this.upper[i]-offset);}
    }
    const releaseBounds=target.releaseJointBounds;
    if(releaseBounds){
      if(!target.releaseCollisionFrame||!target.lockLowerBody||releaseBounds.lower?.length!==29||releaseBounds.upper?.length!==29||![...releaseBounds.lower,...releaseBounds.upper].every(Number.isFinite)||releaseBounds.lower.some((v,i)=>v>releaseBounds.upper[i]))throw new IKError('Invalid initial release reference bounds.');
      for(let i=0;i<29;i++){lower[i]=Math.max(lower[i],releaseBounds.lower[i]);upper[i]=Math.min(upper[i],releaseBounds.upper[i]);}
    }
    if(lower.some((v,i)=>v>upper[i]+1e-8))throw new IKError('The requested reference starts outside the allowed joint margin.');
    let q=copy(seed||qStart).map((v,i)=>clamp(v,lower[i],upper[i]));const indices=target.lockLowerBody?(target.lockWaist?Array.from({length:14},(_,i)=>15+i):Array.from({length:17},(_,i)=>12+i)):target.lockWaist?[...Array(12).keys(),...Array.from({length:14},(_,i)=>15+i)]:[...Array(29).keys()];
    if(target.lockLowerBody)for(let i=0;i<12;i++)q[i]=qStart[i];
    if(target.lockLowerBody&&target.lockWaist)for(let i=12;i<15;i++)q[i]=qStart[i];
    const nv=this.model.nv,collisionOptions=this.collisionOptions(target);
    // This cache exists only during one synchronous solve. Targets, measured
    // objects/fingers and model geometry cannot change while it is active.
    // Accepted trials leave native scratch data at the accepted configuration;
    // restore scratch after a rejected trial, but reuse exact prior audits and
    // residuals when their pose and complete collision options are unchanged.
    const evaluated=new WeakMap();let currentPose=null;
    const poseAt=value=>{
      let state=evaluated.get(value);
      if(!state||state.q.length!==value.length||state.q.some((v,i)=>!Object.is(v,value[i]))){
        state={q:copy(value),audits:new Map(),residual:null};evaluated.set(value,state);
      }
      if(currentPose!==state){this.setPose(value,target);currentPose=state;}
      return state;
    };
    const residualAtPose=()=>currentPose.residual??=(this.residuals(target));
    const auditAtPose=(detect=null)=>{
      if(!currentPose.audits.has(detect))currentPose.audits.set(detect,this.audit.check(this.data,
        detect===null?collisionOptions:{...collisionOptions,detect}));
      return currentPose.audits.get(detect);
    };
    if(seed){
      poseAt(q);
      // A measured seed is a numerical suggestion, not a command. Native
      // tracking error may put that suggestion inside the planning margin;
      // retain the last accepted command as the feasible solver start instead.
      if(bestEffort){
        const seedCost=this.bestEffortCost(q,target,residualAtPose(),auditAtPose(),qStart),baseline=copy(qStart).map((v,i)=>clamp(v,lower[i],upper[i]));
        poseAt(baseline);const baselineCost=this.bestEffortCost(baseline,target,residualAtPose(),auditAtPose(),qStart);
        if(baselineCost<seedCost){q=baseline;this.rejectedMeasuredSeeds=(this.rejectedMeasuredSeeds||0)+1;}
      }else if(!auditAtPose().passed){q=copy(qStart).map((v,i)=>clamp(v,lower[i],upper[i]));this.rejectedMeasuredSeeds=(this.rejectedMeasuredSeeds||0)+1;}
    }
    const addRows=(rows,errors,jac,error,weight)=>{for(let r=0;r<error.length;r++){rows.push(indices.map(i=>jac[r*nv+this.vadr[i]]*weight));errors.push(error[r]*weight);}};
    for(let iteration=0;iteration<iterations;iteration++){
      poseAt(q);const rows=[],errors=[],goals=this.goals(target);
      for(const side of SIDES){const body=this.scene.ankleBodyIds[side],pose=bodyPose(this.data,body),jac=this.scene.jacBody(this.data,body);addRows(rows,errors,jac.position,sub(this.feet[side].position,pose.position),100);addRows(rows,errors,jac.rotation,rotationError(this.feet[side].rotation,pose.rotation),20);}
      for(const side of SIDES)if(goals[side]&&!target.trackPosture){
        const goal=goals[side],body=this.scene.palmBodyIds[side],pose=bodyPose(this.data,body),jac=this.scene.jacBody(this.data,body),weight=12;
        let jp=jac.position,jr=jac.rotation;
        if(goal.relative){const torso=this.scene.jacBody(this.data,this.torsoBodyId),point=this.jacPoint(this.data,goal.position,this.torsoBodyId);jp=Float64Array.from(jp,(v,i)=>v-point[i]);jr=Float64Array.from(jr,(v,i)=>v-torso.rotation[i]);}
        addRows(rows,errors,jp,sub(goal.position,pose.position),weight);addRows(rows,errors,jr,rotationError(goal.rotation,pose.rotation),weight*.16);
      }
      for(let k=0;k<indices.length;k++){const i=indices[k],r=Array(indices.length).fill(0);r[k]=target.trackPosture?(target.trackUpperPostureOnly&&i<12?.06:3):i>=12&&i<15?(target.waistPostureWeights?.[i-12]??.06):.06;rows.push(r);errors.push((target.posture[i]-q[i])*r[k]);}
      // Temporal posture is anchored to the last input reference, not to the
      // current DLS iterate or the generic rest pose. Each subsequent solve
      // can still make progress, without exchanging waist/arm nullspace motion
      // gratuitously inside one rolling step. Zero preserves the original rows.
      if(jointContinuityWeight>0)for(let k=0;k<indices.length;k++)if(indices[k]>=12){const r=Array(indices.length).fill(0);r[k]=jointContinuityWeight;rows.push(r);errors.push((qStart[indices[k]]-q[indices[k]])*jointContinuityWeight);}
      const geometry=auditAtPose(.03);
      // Actual nearest-point normals provide collision-avoidance Jacobian rows.
      if(this.scene.geomDistanceInfo)for(const pair of geometry.near.slice(0,24)){
        if(pair.distance>=.025)continue;const pairData=pair.poseFrame==='measured_release_pose'?this.releaseData:pair.poseFrame==='measured_grasp'?this.graspData:this.data;
        const info=this.scene.geomDistanceInfo(pairData,pair.a,pair.b,.04),segment=info.fromto;if(info.hasWitness===false||!segment||segment.length!==6)continue;
        // For penetration the surface witnesses cross. Reverse their segment
        // so the Jacobian still points toward increasing signed clearance.
        const n=scale(unitSafe(sub(copy(segment.slice(0,3)),copy(segment.slice(3,6)))),info.distance<0?-1:1);if(norm(n)<.5)continue;
        const ja=this.collisionPointJacobian(pairData,copy(segment.slice(0,3)),pair.a,target),jb=this.collisionPointJacobian(pairData,copy(segment.slice(3,6)),pair.b,target);
        // Explicit contact/shell allowances retain the zero-clearance soft
        // objective. A negative hard margin must not introduce 25 mm of
        // repulsion when a formerly zero-margin shell gains overlap tolerance.
        // Opening newly enables every selected-hand/object pair. Their soft
        // 25 mm repulsion must not splice a sudden arm correction into a
        // gradual, checked separation. Keep each hard floor and every other
        // pair's objective unchanged; only these release rows seek the floor.
        const releaseName=target.releaseCollisionFrame?getName(this.scene,this.model.geom_bodyid[pair.a]):'',releaseObject=this.scene.objects?.[target.objectId];
        const releasePair=!!target.releaseCollisionFrame&&releaseObject?.geomIds.includes(pair.b)&&(releaseName.startsWith(`${target.hand}_hand_`)||releaseName.startsWith(`${target.hand}_wrist_`));
        const weight=pair.distance<pair.requiredMargin+this.optimizationBuffer+.002?30:8,desiredDistance=releasePair?pair.requiredMargin:pair.requiredMargin>0?.025:0;
        rows.push(indices.map(i=>[0,1,2].reduce((s,r)=>s+n[r]*(ja[r*nv+this.vadr[i]]-jb[r*nv+this.vadr[i]]),0)*weight));errors.push(Math.max(0,desiredDistance-pair.distance)*weight);
      }
      // COM correction is an analytic mass-weighted body Jacobian near the edge.
      const residual=residualAtPose(),supportRows=[];if(residual.comMargin<.035){
        const jcom=new Float64Array(3*nv);let mass=0;
        for(let b=1;b<this.model.nbody;b++){if(!this.model.body_mass[b])continue;let ancestor=b;while(ancestor>0&&ancestor!==this.rootBodyId)ancestor=this.model.body_parentid[ancestor];if(ancestor!==this.rootBodyId)continue;const m=this.model.body_mass[b],point=copy(this.data.xipos.slice(3*b,3*b+3)),j=this.jacPoint(this.data,point,b);mass+=m;for(let c=0;c<j.length;c++)jcom[c]+=m*j[c];}
        for(let i=0;i<jcom.length;i++)jcom[i]/=mass;
        if(continuousCom){
          for(const task of continuousComTasks(residual.com,this.supportPolygon)){
            const row=indices.map(i=>task.normal[0]*jcom[this.vadr[i]]+task.normal[1]*jcom[nv+this.vadr[i]]);
            rows.push(scale(row,20));errors.push(task.error*20);
            supportRows.push({row,error:task.error-.0145});
          }
        }else{
          const center=scale(this.supportPolygon.reduce((a,p)=>add(a,[...p,0]),[0,0,0]),1/this.supportPolygon.length);addRows(rows,errors,jcom,sub(center,residual.com).slice(0,2),20);
        }
      }
      const stepLower=indices.map(i=>Math.max(lower[i]-q[i],-.12)),stepUpper=indices.map(i=>Math.min(upper[i]-q[i],.12));
      let delta=boundedDLS(rows,errors,stepLower,stepUpper);
      // A weighted COM objective can point outside the planted-foot inset.
      // Re-solve only that blocked direction with its active support planes;
      // the native line search still enforces the exact 20 mm support margin.
      const activeSupport=[];
      for(let pass=0;pass<supportRows.length;pass++){
        const blocked=supportRows.filter(task=>!activeSupport.includes(task)&&dot(task.row,delta)<task.error);
        if(!blocked.length)break;activeSupport.push(...blocked);
        delta=boundedDLS([...rows,...activeSupport.map(task=>scale(task.row,2000))],[...errors,...activeSupport.map(task=>task.error*2000)],stepLower,stepUpper);
      }
      if(!Array.from(delta).every(Number.isFinite))throw new IKError('IK produced a non-finite joint step.');
      if(norm(delta)<1e-7)break;let accepted=false;
      const currentCost=bestEffort?this.bestEffortCost(q,target,residual,geometry,qStart):null;
      for(const alpha of [1,.5,.25,.125,.0625]){
        const trial=q.slice();indices.forEach((i,k)=>trial[i]+=alpha*delta[k]);poseAt(trial);const next=auditAtPose();
        if(bestEffort){
          const nextCost=this.bestEffortCost(trial,target,residualAtPose(),next,qStart);
          if(nextCost<currentCost-1e-12){q=trial;accepted=true;break;}
          continue;
        }
        // The soft COM objective can lose to another task even when the last
        // command is supported. Keep that hard margin through every accepted
        // rolling step. A changed root can start outside it; permit improving
        // iterates there, while retaining the same hard endpoint validation.
        const trialCom=continuousCom?polygonMargin(this.data.subtree_com.slice(this.rootBodyId*3,this.rootBodyId*3+3),this.supportPolygon):null;
        const supported=!continuousCom||(residual.comMargin>=.02?trialCom>=.02:trialCom>residual.comMargin);
        // Starting in an unsafe scene is never made eligible by taking a smaller step.
        const bufferedFloor=Math.min(0,geometry.minimumBufferedSurplus);
        if(supported&&(next.passed&&next.minimumBufferedSurplus>=bufferedFloor-1e-8||(strictEndpoint&&!geometry.passed&&next.minimumClearanceSurplus>geometry.minimumClearanceSurplus+1e-7))){q=trial;accepted=true;break;}
      }
      if(!accepted){poseAt(q);if(!bestEffort&&!geometry.passed)throw new IKError('No collision-free whole-body step exists.',{geometry});break;}
    }
    poseAt(q);const residual=residualAtPose(),geometry=auditAtPose();
    const fixedFeet=residual.footPositionError<.003&&residual.footRotationError<.05,balanced=residual.comMargin>=.02;
    // Posture tracking intentionally releases Cartesian palm goals while legs
    // compensate for the requested root position with the original feet fixed.
    const endpoint=target.trackPosture?residual.upperPostureError<.05:Object.values(residual.palms).every(p=>p.positionError<.012&&p.rotationError<.12);
    if(!q.every((value,i)=>Number.isFinite(value)&&value>=lower[i]-1e-10&&value<=upper[i]+1e-10))throw new IKError('IK output violates finite joint or per-step bounds.');
    if(!bestEffort&&(!geometry.passed||!fixedFeet||!balanced||(strictEndpoint&&!endpoint)))throw new IKError('The constrained IK target is infeasible.',{...residual,geometry,endpoint});
    const rootPosition=copy(this.data.qpos.slice(0,3));
    const unmetConstraints=unmetIKConstraints(residual,geometry,{strictEndpoint,trackPosture:target.trackPosture});
    return {q,rootPosition,rootHeight:rootPosition[2],rootPitch:target.rootPitch??0,residual,geometry,bestEffort,bestEffortUsed:bestEffort,passed:unmetConstraints.length===0,endpointReached:endpoint,unmetConstraints};
  }
  async planStance(target,{tableHeight=this.scene.tableTopZ,onProgress=null}={}){
    const heights=tableHeight<.60?[.08,.10,.12,.14].map(drop=>this.anchorPosition[2]-drop):[this.anchorPosition[2],.70,.68,.66,.64].filter(h=>h<=this.anchorPosition[2]);
    const candidates=[];
    for(const lean of [0,10,20,30,40,50])for(const height of heights){
      const posture=copy(target.posture);posture[14]=Math.min(lean*.6*Math.PI/180,.50);const candidate={...target,posture,rootHeight:height,rootPitch:lean*.4*Math.PI/180};
      let row={rootHeight:height,rootPitch:candidate.rootPitch,leanDegrees:lean,feasible:false};
      try {const solution=this.solve(this.defaultQ,candidate,{iterations:100,maxStep:3,strictEndpoint:true});row={...row,...solution.residual,posture:solution.q,feasible:true};candidates.push(row);return {...row,candidates};}
      catch(error){if(!(error instanceof IKError))throw error;row.reason=error.message;row.residual=error.details;}
      candidates.push(row);onProgress?.(candidates.length,heights.length*6);await new Promise(resolve=>setTimeout(resolve,0));
    }return {feasible:false,candidates,message:'No safe planted-foot stance reaches this object.'};
  }
}
const unitSafe=v=>norm(v)>1e-10?scale(v,1/norm(v)):[0,0,0];
