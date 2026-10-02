// Faithful inference contracts from policy_hero_export.py.
// Physics, joint PD, reference generation, and rendering belong to the caller.
import {sub,clip,conjugate,multiply,rotateInverse,yaw,heading,rollPitch,rot6Row,rot6Column,finiteVector} from './policy_math.mjs';

const H16 = ['left_ankle_roll_link','right_ankle_roll_link','left_knee_link','right_knee_link'];
const DOWN = [0,0,-1];
const zeros = n => new Float64Array(n);
const sample = (ref, index) => ref.sample(Math.max(0, Math.min(ref.frameCount-1, index)));
const copy = values => Float64Array.from(values);

function validateState(state) {
  for (const [key,size] of [['rootPosW',3],['rootQuatW',4],['rootAngVelB',3],['dofPos',29],['dofVel',29]])
    finiteVector(state[key],size,key);
}

class History {
  constructor(length, dimensions, layout, padding) {
    this.length=length; this.dimensions=dimensions; this.width=dimensions.reduce((a,b)=>a+b,0);
    this.layout=layout; this.padding=padding; this.reset();
  }
  reset() { this.frames=[]; }
  push(frame) {
    finiteVector(frame,this.width,'policy history frame');
    if (!this.frames.length && this.padding==='repeat_first_sample')
      this.frames=Array.from({length:this.length},()=>copy(frame));
    else { this.frames.push(copy(frame)); if(this.frames.length>this.length)this.frames.shift(); }
  }
  flat() {
    const out=new Float32Array(this.length*this.width), pad=this.length-this.frames.length;
    if(this.layout==='frame_major_hero_v1') this.frames.forEach((f,i)=>out.set(f,(i+pad)*this.width));
    else {
      let src=0,dst=0;
      for(const size of this.dimensions) {
        this.frames.forEach((f,i)=>out.set(f.subarray(src,src+size),dst+(i+pad)*size));
        src+=size; dst+=size*this.length;
      }
    }
    return out;
  }
}

class BasePolicy {
  constructor(ort, metadata) {
    this.ort=ort; this.metadata=metadata;
    for(const name of ['kp','kd','effortLimit','defaultDofPos','actionScale'])
      this[name]=copy(finiteVector(metadata[name],29,name));
    this.dofNames=metadata.dofNames; this.kind=metadata.kind; this.controlDt=.02;
    this.steps=0; this.inferenceMilliseconds=0; this.busy=false;
  }
  async infer(session, inputNames, outputName, values) {
    const tensor=new this.ort.Tensor('float32',values,[1,values.length]);
    const feeds=Object.fromEntries(inputNames.map(name=>[name,tensor]));
    const start=performance.now(), outputs=await session.run(feeds);
    this.inferenceMilliseconds=performance.now()-start;
    const output=Float32Array.from(outputs[outputName].data);
    tensor.dispose?.(); for(const value of Object.values(outputs))value.dispose?.();
    finiteVector(output,output.length,'ONNX output'); return output;
  }
  async exclusive(fn) {
    if(this.busy)throw new Error('Policy inference steps must be awaited sequentially.');
    this.busy=true;try{return await fn();}finally{this.busy=false;}
  }
  async dispose() { for(const session of this.sessions??[])await session.release(); }
}

export class HeroPolicy extends BasePolicy {
  constructor(ort, metadata, session, {odomSource='leg'}={}) {
    super(ort,metadata); this.session=session; this.sessions=[session];
    if (!['leg','truth'].includes(odomSource)) throw new Error('Unsupported HERO odometry source.');
    // Ground truth is available for native parity checks.
    // The interactive runtime always uses the default leg estimate.
    this.odomSource=odomSource;
    this.history=new History(metadata.historyLength,metadata.termDimensions,metadata.historyLayout,'zeros_after_reset');
    this.reset();
  }
  reset() { this.history.reset();this.lastAction=zeros(29);this.lastObs=null;this.lastTerms=null;this.steps=0; }
  resetTarget(reference) {
    return this.qTarget(zeros(29), sample(reference,this.metadata.refAttr==='joint_pos'?0:this.metadata.lookaheadFrames).jointPos);
  }
  objectEffective(reference) {
    if(!reference.hasObject)return false;
    const first=sample(reference,0).object;
    if(!first?.positionW)return false;
    const rule=this.metadata.objectRule, missing=!reference.objectSize;
    const size=reference.objectSize??[.3,.3,.3];
    if(rule.object_box_side!=null && (Math.abs(Math.max(...size)-rule.object_box_side)>rule.object_box_tol+1e-6 || (missing&&!rule.object_box_size_missing_ok)))return false;
    return rule.max_start_bottom_z_m==null || first.positionW[2]-size[2]/2<=rule.max_start_bottom_z_m;
  }
  frameTerms(reference, index, state) {
    validateState(state);
    const m=this.metadata, now=sample(reference,index), command=sample(reference,index+m.lookaheadFrames);
    const iq=conjugate(state.rootQuatW), refIq=conjugate(command.rootQuatW);
    const dp=[],dr=[];
    for(let hand=0;hand<2;hand++) {
      const curP=rotateInverse(state.rootQuatW,sub(state.palmPosW[hand],state.rootPosW));
      const refP=rotateInverse(command.rootQuatW,sub(command.palmPosW[hand],command.rootPosW));
      const curQ=multiply(iq,state.palmQuatW[hand]), refQ=multiply(refIq,command.palmQuatW[hand]);
      dp.push(...sub(curP,refP)); dr.push(...rot6Row(multiply(conjugate(curQ),refQ)));
    }
    const h16=[];
    for(const name of H16) {
      const slot=reference.bodyNames.indexOf(name);
      if(slot<0)throw new Error(`Reference is missing ${name}`);
      h16.push(...rotateInverse(now.rootQuatW,sub(now.bodyPosW[slot],now.rootPosW)));
    }
    const anchorTerms={};
    const requested=new Set(m.terms), anchorNames=['h20_ref_root_pose_b','h21_ref_root_rot_b','h22_ref_root_height_b','h23_base_lin_vel_odom'];
    if (anchorNames.some(name=>requested.has(name))) {
      const odom=this.odomSource==='leg'?state.odom:
        {posW:state.rootPosW,quatW:state.rootQuatW,linVelW:state.rootLinVelW??[0,0,0]};
      if (!odom || (this.odomSource==='leg' && odom.source!=='leg'))
        throw new Error('HERO anchor observations require leg odometry.');
      finiteVector(odom.posW,3,'odometry position'); finiteVector(odom.quatW,4,'odometry orientation');
      const robotHeading=heading(odom.quatW),anchorIq=conjugate(odom.quatW);
      for(const name of anchorNames) if(requested.has(name)) anchorTerms[name]=[];
      for(const offset of m.futureSteps??[]) {
        const future=sample(reference,index+offset);
        if(requested.has(anchorNames[0])) {
          const delta=rotateInverse(robotHeading,sub(future.rootPosW,odom.posW)),dyaw=yaw(future.rootQuatW)-yaw(odom.quatW);
          anchorTerms[anchorNames[0]].push(delta[0],delta[1],Math.sin(dyaw),Math.cos(dyaw));
        }
        if(requested.has(anchorNames[1]))anchorTerms[anchorNames[1]].push(...rot6Column(multiply(anchorIq,future.rootQuatW)));
        if(requested.has(anchorNames[2]))anchorTerms[anchorNames[2]].push(future.rootPosW[2]-odom.posW[2]);
      }
      if(requested.has(anchorNames[3]))anchorTerms[anchorNames[3]]=rotateInverse(robotHeading,finiteVector(odom.linVelW,3,'odometry velocity'));
    }
    let objP=[0,0,0],objR=[0,0,0,0,0,0],objF=[0];
    if(state.object?.hasObject && this.objectEffective(reference)) {
      // Native object_state_10 explicitly rounds this group to float32 first.
      objP=rotateInverse(state.rootQuatW,sub(state.object.positionW,state.rootPosW)).map(Math.fround);
      objR=rot6Row(multiply(iq,state.object.quaternionW)).map(Math.fround);objF=[1];
    }
    const terms={
      h00_actions:this.lastAction,
      h01_base_ang_vel:state.rootAngVelB,
      h02_command_ang_vel:[0],
      h03_command_base_height:[m.heightFromClip?Math.max(command.hRef??command.rootPosW[2],m.heightMin):m.heightDefault],
      h04_command_lin_vel:[0,0],h05_command_stand:[0],
      h06_command_waist_dofs:command.jointPos.slice(12,15),
      h07_dif_local_rigid_body_pos_ee:dp,h08_dif_local_rigid_body_rot_ee:dr,
      h09_dof_pos:sub(Array.from(state.dofPos),this.defaultDofPos),h10_dof_vel:state.dofVel,
      h11_projected_gravity:rotateInverse(state.rootQuatW,DOWN),
      h12_ref_upper_dof_pos:command.jointPos.slice(15,29),h13_roll_and_pitch:rollPitch(state.rootQuatW),
      h14_ref_lower_dof_pos:now.jointPos.slice(0,12),h15_ref_root_pitch_roll:rollPitch(now.rootQuatW),h16_ref_body_pos_b:h16,
      h17_obj_pos_b:objP,h18_obj_ori_b:objR,h19_has_object_flag:objF,
      ...anchorTerms,
      odom_source:anchorNames.some(name=>requested.has(name))?this.odomSource:'unused',
    };
    return {terms,residualReference:m.refAttr==='joint_pos'?now.jointPos:command.jointPos};
  }
  observe(terms) {
    const frame=new Float64Array(this.history.width);let offset=0;
    this.metadata.terms.forEach((name,i)=>{
      const values=finiteVector(terms[name],this.metadata.termDimensions[i],name),scale=this.metadata.termScales[name]??1;
      for(const v of values)frame[offset++]=clip(v*scale,this.metadata.observationClip);
    });
    this.history.push(frame);return this.history.flat();
  }
  qTarget(action, reference) {
    const target=new Float64Array(29),m=this.metadata,residual=new Set(m.residualIndices);
    for(let j=0;j<29;j++)target[j]=(residual.has(j)?reference[j]:this.defaultDofPos[j])+this.actionScale[j]*clip(action[j],m.actionClip);
    return target;
  }
  async control(reference,index,state) {
    return this.exclusive(async()=>{
      const {terms,residualReference}=this.frameTerms(reference,index,state),obs=this.observe(terms);
      const raw=await this.infer(this.session,this.metadata.inputNames,this.metadata.outputName,obs);
      finiteVector(raw,29,'HERO action'); this.lastAction=copy(raw);this.lastObs=obs;this.lastTerms=terms;this.steps++;
      return this.qTarget(raw,residualReference);
    });
  }
}

export async function createPolicy(ort, metadata, modelSources, {odomSource='leg'}={}) {
  const options={executionProviders:['wasm'],graphOptimizationLevel:'all'};
  if(metadata.kind==='hero_plus')return new HeroPolicy(ort,metadata,await ort.InferenceSession.create(modelSources.model,options),{odomSource});
  throw new Error(`Unsupported browser policy ${metadata.kind}`);
}

export async function loadPolicy(ort, manifestURL, kind='hero_plus') {
  const url=new URL(manifestURL,globalThis.location?.href),response=await fetch(url);
  if(!response.ok)throw new Error(`Policy manifest ${response.status}: ${url}`);
  const manifest=await response.json(),entry=manifest.policies[kind];
  if(!entry)throw new Error(`Missing static model ${kind}`);
  const metaResponse=await fetch(new URL(entry.metadata,url));
  if(!metaResponse.ok)throw new Error(`Policy metadata ${metaResponse.status}`);
  const metadata=await metaResponse.json(),sources={};
  for(const [key,file] of Object.entries(entry.files))sources[key]=new URL(file,url).href;
  return createPolicy(ort,metadata,sources);
}

// assetResolver returns local/embedded bytes. No network or Python service is
// required by this entry point, including for standalone HTML delivery.
export async function loadResolvedPolicy(kind,{manifest,assetResolver,ort,odomSource='leg'}) {
  const entry=manifest.policies[kind];
  if(!entry)throw new Error(`Missing static policy ${kind}`);
  const rawMetadata=await assetResolver(entry.metadata);
  const metadata=typeof rawMetadata==='string'?JSON.parse(rawMetadata):
    (rawMetadata instanceof ArrayBuffer || ArrayBuffer.isView(rawMetadata))?
      JSON.parse(new TextDecoder().decode(rawMetadata)):rawMetadata;
  const sources={};
  for(const [key,path] of Object.entries(entry.files))sources[key]=await assetResolver(path);
  const policy=await createPolicy(ort,metadata,sources,{odomSource});
  policy.label=entry.label;policy.checkpoint=entry.checkpoint;policy.provenance=entry.provenance;
  return policy;
}
