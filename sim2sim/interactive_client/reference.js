import {add,sub,scale,lerp,clamp,matVec,matMul,transpose3,matToQuat,quatToMat,angularVelocity} from './numerics.js';
import {WholeBodyIK,IKError,bodyPosition,bodyRotation,unmetIKConstraints} from './ik.js';

export const BODY_NAMES_32=['pelvis','left_hip_pitch_link','left_hip_roll_link','left_hip_yaw_link','left_knee_link','left_ankle_pitch_link','left_ankle_roll_link','left_foot_contact_point','right_hip_pitch_link','right_hip_roll_link','right_hip_yaw_link','right_knee_link','right_ankle_pitch_link','right_ankle_roll_link','right_foot_contact_point','waist_yaw_link','waist_roll_link','torso_link','left_shoulder_pitch_link','left_shoulder_roll_link','left_shoulder_yaw_link','left_elbow_link','left_wrist_roll_link','left_wrist_pitch_link','left_wrist_yaw_link','right_shoulder_pitch_link','right_shoulder_roll_link','right_shoulder_yaw_link','right_elbow_link','right_wrist_roll_link','right_wrist_pitch_link','right_wrist_yaw_link'];

/** Freeze measured grasp and world-sweep transforms for one synchronous planning batch. */
export function payloadTargetSampler(targetAt,frameAtStart){
  const payloadAnchors=new Map();let payloadFrame=null;
  return offset=>{
      const target={...targetAt(offset)},payload=target.carriedPayload;
      if(!payload||payload.measuredPositionW===undefined&&payload.measuredRotationW===undefined)return target;
      const side=['left','right'].indexOf(payload.hand);
      if(side<0||payload.measuredPositionW?.length!==3||payload.measuredRotationW?.length!==9||![...payload.measuredPositionW,...payload.measuredRotationW].every(Number.isFinite))throw new IKError('Carried payload window needs a finite measured world pose.');
      const graspPosition=payload.graspPositionInPalm??payload.positionInPalm,graspRotation=payload.graspRotationInPalm??payload.rotationInPalm;
      if(graspPosition?.length!==3||graspRotation?.length!==9||![...graspPosition,...graspRotation].every(Number.isFinite))throw new IKError('Carried payload window needs a finite measured grasp transform.');
      const key=`${payload.objectId}:${payload.hand}`;
      if(!payloadAnchors.has(key)){
        payloadFrame??=frameAtStart();
        const inverse=transpose3(quatToMat(payloadFrame.palmQuatW[side]));
        payloadAnchors.set(key,{
          positionInPalm:matVec(inverse,sub(payload.measuredPositionW,payloadFrame.palmPosW[side])),
          rotationInPalm:matMul(inverse,payload.measuredRotationW),
          // The world anchor predicts the load's sweep from its measured
          // position. Internal robot/load clearance must retain the actual
          // grasp relation, which a lagging reference palm must not distort.
          graspPositionInPalm:Array.from(graspPosition),
          graspRotationInPalm:Array.from(graspRotation),
          measuredPositionW:Array.from(payload.measuredPositionW),measuredRotationW:Array.from(payload.measuredRotationW),
        });
      }
      target.carriedPayload={...payload,...payloadAnchors.get(key)};return target;
    };
}

/** The 32-body, 50 Hz reference consumed by the exported policy. */
export class ReferenceClip {
  constructor(frames){if(frames.length<47)throw new Error('A policy reference needs its complete 45-frame future horizon.');this.frames=frames;this.fps=50;this.frameCount=frames.length;this.bodyNames=BODY_NAMES_32;this.hasObject=false;}
  sample(index){return this.frames[clamp(Math.floor(index),0,this.frameCount-1)];}
}

export class RollingReference {
  constructor(scene,{horizon=50,stride=5,refreshSteps=5,ikIterations=12,bestEffort=false,...ikOptions}={}){
    this.scene=scene;this.ik=new WholeBodyIK(scene,ikOptions);this.horizon=Math.max(50,horizon);this.stride=stride;this.refreshSteps=Math.min(5,Math.max(1,refreshSteps));this.ikIterations=ikIterations;this.bestEffort=bestEffort;this.dt=.02;this.reset();
  }
  reset(){this.ik.reset();this.qCurrent=this.ik.defaultQ.slice();this.rootPosition=this.ik.anchorPosition.slice();this.rootHeight=this.rootPosition[2];this.rootPitch=0;this.clip=null;this.windowTime=-Infinity;this.windowBestEffort=null;this.currentIndex=0;this.lastAudit=null;}
  dispose(){this.ik.dispose();}
  _consumeFrame(frame){this.qCurrent=frame.jointPos.slice();this.rootPosition=frame.rootPosW.slice();this.rootHeight=this.rootPosition[2];this.rootPitch=frame.plannedRootPitch;}
  fkFrame(q,target){
    this.ik.setPose(q,target);return this.frameFromCurrentPose(q,target);
  }
  // Serialize only after the chosen frame's FK is already in the IK scratch.
  // Callers must not change that scratch between solving and reading it.
  frameFromCurrentPose(q,target){
    const data=this.ik.data,bodyPosW=[],bodyQuatW=[];
    for(let i=0;i<32;i++){
      let id=this.scene.bodyReferenceIds32[i];const virtual=i===7||i===14;
      if(id<0||id==null){if(!virtual)throw new IKError(`Missing canonical body ${BODY_NAMES_32[i]}.`);id=this.scene.ankleBodyIds[i===7?'left':'right'];}
      let position=bodyPosition(data,id);const rotation=bodyRotation(data,id);
      if(virtual&&(!this.scene.bodyName||this.scene.bodyName(id)!==BODY_NAMES_32[i]))position=add(position,matVec(rotation,[0,0,-.037]));
      bodyPosW.push(position);bodyQuatW.push(matToQuat(rotation));
    }
    const palmPosW=['left','right'].map(side=>bodyPosition(data,this.scene.palmBodyIds[side]));
    const palmQuatW=['left','right'].map(side=>matToQuat(bodyRotation(data,this.scene.palmBodyIds[side])));
    return {jointPos:Array.from(q),jointVel:Array(29).fill(0),bodyPosW,bodyQuatW,bodyLinVelW:Array.from({length:32},()=>[0,0,0]),bodyAngVelW:Array.from({length:32},()=>[0,0,0]),rootPosW:Array.from(data.qpos.slice(0,3)),rootQuatW:Array.from(data.qpos.slice(3,7)),palmPosW,palmQuatW,plannedRootPitch:target.rootPitch??0};
  }
  /** Every interpolated frame is FK-checked; best-effort windows retain failed checks as diagnostics. */
  window(targetAt,now,{reseed=false,force=false,bestEffort=this.bestEffort??false,initialFrameMaxStep=.035}={}){
    if(!Number.isFinite(initialFrameMaxStep)||initialFrameMaxStep<=0||initialFrameMaxStep>.035)throw new IKError('Invalid initial reference joint bound.');
    const age=Math.round((now-this.windowTime)/this.dt);
    if(this.clip&&!force&&!reseed&&age>=0&&age<this.refreshSteps&&this.windowBestEffort===bestEffort){this.currentIndex=age;this._consumeFrame(this.clip.sample(age));return this.clip;}
    let committed=[];
    if(this.clip&&Number.isFinite(age)){
      // HERO reads the current lower-body frame and a one-frame lookahead for
      // waist, arms and EE goals. On the next control tick that already-issued
      // lookahead becomes frame0: keep it, then solve only the next frame1.
      // Re-solving an issued frame can reverse the upper-body command
      // during a rolling-window refresh.
      const advanced=age>this.currentIndex,start=this.currentIndex+Number(advanced);
      committed=advanced?[this.clip.sample(start)]:[this.clip.sample(start),this.clip.sample(start+1)];
      // A same-time forced rebuild must preserve both already-issued frames.
      // All committed poses still receive the fresh FK and constraint audits
      // below; committing a pose never suppresses invalid-state diagnostics.
      this._consumeFrame(this.clip.sample(this.currentIndex));
      // Root orientation remains tied to the immutable reset heading.
    }
    // A measured grasp transform belongs to the actual palm. Applying it to
    // a lagging reference palm can put an airborne payload inside the table.
    // Freeze one measured-world/reference-palm anchor for the entire window;
    // future nodes and line-search candidates move it only by reference FK.
    // Callers may also supply an explicit palm transform.
    const windowTargetAt=payloadTargetSampler(targetAt,()=>committed[0]??this.fkFrame(this.qCurrent,{rootPosition:this.rootPosition,rootHeight:this.rootHeight,rootPitch:this.rootPitch}));
    const nodes=[0,1];for(let i=this.stride;i<=this.horizon;i+=this.stride)nodes.push(i);if(nodes.at(-1)!==this.horizon)nodes.push(this.horizon);
    let q=this.qCurrent.slice(),position=(this.rootPosition??this.ik.anchorPosition).slice(),height=this.rootHeight,pitch=this.rootPitch,previousStep=-1,reseedPending=reseed;const solved=[];
    for(const step of nodes){
      const target=windowTargetAt(step*this.dt),n=Math.max(1,step-previousStep),seconds=n*this.dt;
      const desired=target.rootPosition??this.ik.anchorPosition;
      if(desired.length!==3||!Array.from(desired).every(Number.isFinite))throw new IKError('Root position must contain three finite world coordinates.');
      if(!Number.isFinite(target.rootHeight??desired[2])||!Number.isFinite(target.rootPitch??0))throw new IKError('Root height and pitch must be finite.');
      if(committed[step]){
        const frame=committed[step];q=frame.jointPos.slice();position=frame.rootPosW.slice();height=position[2];pitch=frame.plannedRootPitch;
        Object.assign(target,{rootPosition:position,rootHeight:height,rootPitch:pitch});
        solved.push({step,q,target,bestEffortUsed:false});previousStep=step;continue;
      }
      const dx=desired[0]-position[0],dy=desired[1]-position[1],distance=Math.hypot(dx,dy),fraction=distance>0?Math.min(1,.06*seconds/distance):0;
      target.rootHeight=height+clamp((target.rootHeight??desired[2])-height,-.12*seconds,.12*seconds);
      target.rootPosition=[position[0]+dx*fraction,position[1]+dy*fraction,target.rootHeight];
      target.rootPitch=pitch+clamp((target.rootPitch??0)-pitch,-.25*seconds,.25*seconds);
      const result=this.ik.solve(q,target,{iterations:this.ikIterations,maxStep:step<=1?Math.min(initialFrameMaxStep,1.75*seconds):1.75*seconds,seed:reseedPending?this.scene.readState().dofPos:null,continuousCom:true,bestEffort});
      reseedPending=false;
      solved.push({step,q:result.q,target,residual:result.residual,bestEffortUsed:!!result.bestEffortUsed});q=result.q;position=target.rootPosition;height=target.rootHeight;pitch=target.rootPitch;previousStep=step;
    }
    // Diagnostic metadata only: reuse the native audits already performed for
    // each consumed frame. Visual-only object accents cannot block a route.
    const obstacleGeoms=new Map();
    for(const [id,obj]of Object.entries(this.scene.objects||{}))if(obj.active)
      for(const geom of obj.geomIds||[])if(this.scene.model.geom_contype[geom]||this.scene.model.geom_conaffinity[geom])obstacleGeoms.set(geom,id);
    const frames=[],audits=[];let segment=0,bestEffortUsed=solved.some(node=>node.bestEffortUsed);
    for(let step=0;step<=this.horizon;step++){
      while(segment+1<solved.length-1&&step>solved[segment+1].step)segment++;
      const a=solved[segment],b=solved[Math.min(segment+1,solved.length-1)],alpha=a===b?0:(step-a.step)/(b.step-a.step);
      const rootPosition=lerp(a.target.rootPosition,b.target.rootPosition,alpha);
      const target={...windowTargetAt(step*this.dt),rootPosition,rootHeight:rootPosition[2],rootPitch:a.target.rootPitch+alpha*(b.target.rootPitch-a.target.rootPitch)};
      let jointPos=lerp(a.q,b.q,alpha);this.ik.setPose(jointPos,target);
      let geometry=this.ik.audit.check(this.ik.data,this.ik.collisionOptions(target)),residual=this.ik.residuals(target);
      const previous=frames.at(-1)?.jointPos,rateViolation=previous&&jointPos.some((v,i)=>Math.abs(v-previous[i])>.035000001);
      // Refine an imperfect interpolated point with the same IK mode. The
      // finite joint-rate limit remains mandatory in either mode.
      if(previous&&step>=committed.length&&(!geometry.passed||residual.footPositionError>.003||residual.footRotationError>.05||residual.comMargin<.02||rateViolation)){
        const refined=this.ik.solve(previous,target,{iterations:20,maxStep:step<=1?initialFrameMaxStep:.035,seed:jointPos,continuousCom:true,bestEffort});jointPos=refined.q;geometry=refined.geometry;residual=refined.residual;
        bestEffortUsed||=!!refined.bestEffortUsed;
      }
      const frame=this.frameFromCurrentPose(jointPos,target);
      if(obstacleGeoms.size&&[...obstacleGeoms.values()].some(id=>id!==target.objectId)){
        const constraints=new Map();
        // Match solve()'s active soft-avoidance rows, not just its hard margin:
        // the 25 mm repulsion task can hold a palm away without any contact.
        for(const pair of (geometry.near||[]).slice(0,24)){
          const id=obstacleGeoms.get(pair.b),surplus=pair.distance-pair.requiredMargin;
          if(!id||id===target.objectId||!(pair.requiredMargin>0)||!(pair.distance<.025))continue;
          const prior=constraints.get(id);
          if(!prior||surplus<prior.clearance_m-prior.required_clearance_m)constraints.set(id,{object_id:id,robot_geom:pair.a,object_geom:pair.b,clearance_m:pair.distance,required_clearance_m:pair.requiredMargin,avoidance_distance_m:.025});
        }
        frame.approachCollision={position_error_m:residual.palms[target.hand]?.positionError??0,constraints:[...constraints.values()]};
      }
      const predecessor=previous??this.qCurrent;
      if(!jointPos.every(Number.isFinite)||jointPos.some((value,i)=>Math.abs(value-predecessor[i])>.035000001||value<this.ik.lower[i]-1e-10||value>this.ik.upper[i]+1e-10))throw new IKError('An interpolated reference frame violates finite joint or per-step bounds.',{step});
      if(![...frame.rootPosW,...frame.rootQuatW,...frame.bodyPosW.flat(),...frame.bodyQuatW.flat()].every(Number.isFinite))throw new IKError('Reference FK produced a non-finite body pose.',{step});
      const unmetConstraints=unmetIKConstraints(residual,geometry);
      if(!bestEffort&&unmetConstraints.length)throw new IKError('An interpolated reference frame violates clearance or planted-foot support.',{step,geometry,residual});
      frames.push(frame);audits.push({step,clearance:geometry.minimumClearance,footError:residual.footPositionError,comMargin:residual.comMargin,unmetConstraints});
    }
    for(let i=0;i<frames.length;i++){
      const a=frames[Math.max(0,i-1)],b=frames[Math.min(frames.length-1,i+1)],dt=(i===0||i===frames.length-1)?this.dt:2*this.dt;
      frames[i].jointVel=scale(sub(b.jointPos,a.jointPos),1/dt);
      frames[i].bodyLinVelW=b.bodyPosW.map((p,k)=>scale(sub(p,a.bodyPosW[k]),1/dt));
      frames[i].bodyAngVelW=b.bodyQuatW.map((q,k)=>angularVelocity(a.bodyQuatW[k],q,dt));
    }
    this.clip=new ReferenceClip(frames);this.currentIndex=0;this.windowTime=now;this.windowBestEffort=bestEffort;this._consumeFrame(frames[0]);
    const constraints=new Map();for(const audit of audits)for(const violation of audit.unmetConstraints){
      const amount=Math.abs(violation.actual-violation.limit),previous=constraints.get(violation.constraint);
      if(!previous)constraints.set(violation.constraint,{...violation,count:1,maximumViolation:amount,worstStep:audit.step});
      else{previous.count++;if(amount>previous.maximumViolation)Object.assign(previous,{...violation,maximumViolation:amount,worstStep:audit.step});}
    }
    const unmetConstraints=[...constraints.values()];
    bestEffortUsed||=bestEffort&&unmetConstraints.length>0;
    this.lastAudit={bestEffort,bestEffortUsed,passed:unmetConstraints.length===0,constraintPassed:unmetConstraints.length===0,unmetConstraints,violationFrames:audits.filter(a=>a.unmetConstraints.length).map(a=>a.step),minimumClearance:Math.min(...audits.map(a=>a.clearance)),maximumFootDrift:Math.max(...audits.map(a=>a.footError)),minimumComMargin:Math.min(...audits.map(a=>a.comMargin)),maximumRootXYSpeed:Math.max(...frames.slice(1).map((f,i)=>Math.hypot(f.rootPosW[0]-frames[i].rootPosW[0],f.rootPosW[1]-frames[i].rootPosW[1])/this.dt)),samples:audits.length};
    return this.clip;
  }
  currentFrame(){return this.clip?.sample(this.currentIndex)||this.fkFrame(this.qCurrent,{rootPosition:this.rootPosition,rootHeight:this.rootHeight,rootPitch:this.rootPitch});}
}
