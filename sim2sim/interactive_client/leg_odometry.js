/** Contact-aided leg odometry with WXYZ quaternions.
 * Only reset accepts a world position/translation velocity. Subsequent estimates
 * use encoder FK, foot contact forces and IMU roll/pitch + integrated gyro yaw.
 */
import {add,sub,scale,dot,norm,cross,unit,matVec,matMul,transpose3,quatMul,quatConj,quatToMat,rotateZ} from './numerics.js';

const FOOT_NAMES=['left_ankle_roll_link','right_ankle_roll_link'];
const IDENTITY=[1,0,0,0];
const finiteVector=(value,n,name)=>{
  if(value?.length!==n||!Array.from(value).every(Number.isFinite))throw new Error(`${name} must contain ${n} finite values.`);
  return Array.from(value);
};
const quaternion=(value,name)=>{
  const q=finiteVector(value,4,name);if(!(norm(q)>1e-12))throw new Error(`${name} must be a nonzero quaternion.`);return unit(q);
};
const slice=(array,index,size)=>Array.from(array.subarray?array.subarray(index*size,(index+1)*size):array.slice(index*size,(index+1)*size));
const euler=q=>{const [w,x,y,z]=q;return [Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x)))),Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))];};
const fromEuler=(r,p,y)=>{
  const cr=Math.cos(r/2),sr=Math.sin(r/2),cp=Math.cos(p/2),sp=Math.sin(p/2),cy=Math.cos(y/2),sy=Math.sin(y/2);
  return [cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy];
};
const fromRotvec=v=>{
  const a=norm(v);return a<1e-12?scale([1,...scale(v,.5)],1/Math.sqrt(1+.25*a*a)):[Math.cos(a/2),...scale(v,Math.sin(a/2)/a)];
};
const toRotvec=value=>{
  let q=unit(value);if(q[0]<0)q=scale(q,-1);const s=norm(q.slice(1));
  return scale(q.slice(1),s<1e-12?2:2*Math.atan2(s,q[0])/s);
};
const mean=rows=>scale(rows.reduce((sum,row)=>add(sum,row),[0,0,0]),1/rows.length);

/** Collision geoms on each ankle-roll body and its descendants, in native order. */
export function footGeomIds(scene){
  const {model:m,mj}=scene;
  return FOOT_NAMES.map(name=>{
    const id=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY.value,name);
    if(id<0)throw new Error(`Leg odometry foot body ${name} is missing.`);
    const bodies=new Set([id]);let changed=true;
    while(changed){changed=false;for(let b=0;b<m.nbody;b++)if(!bodies.has(b)&&bodies.has(m.body_parentid[b])){bodies.add(b);changed=true;}}
    const geoms=[];for(let g=0;g<m.ngeom;g++)if(bodies.has(m.geom_bodyid[g])&&(m.geom_contype[g]||m.geom_conaffinity[g]))geoms.push(g);
    if(!geoms.length)throw new Error(`Leg odometry foot body ${name} has no collision geometry.`);
    return geoms;
  });
}

function geomPointsLocal(model,g){
  const size=slice(model.geom_size,g,3),type=model.geom_type[g];let points=[];
  if(type===7){const mesh=model.geom_dataid[g],start=model.mesh_vertadr[mesh],count=model.mesh_vertnum[mesh];for(let i=start;i<start+count;i++)points.push(slice(model.mesh_vert,i,3));}
  else if(type===6){for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])points.push([x*size[0],y*size[1],z*size[2]]);}
  else if(type===5){for(const z of [-size[1],size[1]])for(let i=0;i<16;i++)points.push([size[0]*Math.cos(i*2*Math.PI/16),size[0]*Math.sin(i*2*Math.PI/16),z]);}
  else if([2,3,4].includes(type)){
    const dirs=[];for(const x of [-1,0,1])for(const y of [-1,0,1])for(const z of [-1,0,1])if(x||y||z)dirs.push(unit([x,y,z]));
    if(type===3)for(const z of [-size[1],size[1]])for(const d of dirs)points.push(add([0,0,z],scale(d,size[0])));
    else points=dirs.map(d=>d.map((v,i)=>v*(type===4?size[i]:size[0])));
  }
  const R=quatToMat(slice(model.geom_quat,g,4)),p=slice(model.geom_pos,g,3);
  return points.map(v=>add(p,matVec(R,v)));
}

/** Encoder-only scratch FK. No live MjData is copied or used for foot poses. */
export class MujocoLegFK{
  constructor(scene){
    this.scene=scene;this.model=scene.model;this.mj=scene.mj;this.nFeet=2;
    this.dofQadr=Array.from(scene.jointQposAddresses29??scene.metadata?.jointQposAddresses29??[]);
    if(this.dofQadr.length!==29)throw new Error('Leg odometry requires the 29 encoder qpos addresses.');
    this.q0=Array.from(this.model.qpos0);this.footGeomIds=footGeomIds(scene);
    this.footBodyIds=FOOT_NAMES.map(name=>this.mj.mj_name2id(this.model,this.mj.mjtObj.mjOBJ_BODY.value,name));
    this.data=new this.mj.MjData(this.model);this.forceBuffer=new this.mj.DoubleBuffer(6);
    try{
      const nominal=this.footPosesB(this.dofQadr.map(i=>this.q0[i]));
      this.solePointsLocal=[];this.soleCentroidLocal=[];
      for(let k=0;k<2;k++){
        const foot=this.footBodyIds[k],R0=nominal.rotations[k],inv=transpose3(R0),p0=nominal.positions[k],points=[];
        for(const g of this.footGeomIds[k]){
          const body=this.model.geom_bodyid[g];let local=geomPointsLocal(this.model,g);
          if(body!==foot){const p=slice(this.data.xpos,body,3),R=slice(this.data.xmat,body,9),delta=matVec(inv,sub(p,p0)),relative=matMul(inv,R);local=local.map(v=>add(delta,matVec(relative,v)));}
          points.push(...local);
        }
        if(!points.length)throw new Error('Leg odometry could not derive a collision sole.');
        const down=matVec(inv,[0,0,-1]),heights=points.map(p=>dot(p,down)),lowest=Math.max(...heights),band=points.filter((p,i)=>heights[i]>=lowest-.003);
        const e1=unit(cross(down,Math.abs(down[1])<.9?[0,1,0]:[1,0,0])),e2=cross(down,e1),indices=new Set();
        for(let d=0;d<16;d++){
          const a=d*2*Math.PI/16;let best=-Infinity,index=0;
          for(let i=0;i<band.length;i++){const value=dot(band[i],e1)*Math.cos(a)+dot(band[i],e2)*Math.sin(a);if(value>best){best=value;index=i;}}
          indices.add(index);
        }
        const sole=[...indices].sort((a,b)=>a-b).map(i=>band[i]),bandHeights=band.map(p=>dot(p,down)),maxBand=Math.max(...bandHeights);
        if(Math.max(...sole.map(p=>dot(p,down)))<maxBand-1e-9)sole.push(band[bandHeights.indexOf(maxBand)]);
        this.solePointsLocal.push(sole);this.soleCentroidLocal.push(mean(band));
      }
      const max=Math.max(...this.solePointsLocal.map(s=>s.length));for(const sole of this.solePointsLocal)while(sole.length<max)sole.push(sole.at(-1).slice());
    }catch(error){this.dispose();throw error;}
  }
  footPosesB(dofPos){
    const q=finiteVector(dofPos,29,'Encoder positions');if(!this.data)throw new Error('Leg FK has been disposed.');
    this.data.qpos.set(this.q0);this.data.qpos.set([0,0,0,...IDENTITY],0);this.dofQadr.forEach((address,i)=>{this.data.qpos[address]=q[i];});
    this.mj.mj_kinematics(this.model,this.data);
    return {positions:this.footBodyIds.map(i=>slice(this.data.xpos,i,3)),rotations:this.footBodyIds.map(i=>slice(this.data.xmat,i,9))};
  }
  readContactForces(){
    const live=this.scene.data,out=[0,0];if(!live.ncon)return out;
    const geoms=this.footGeomIds.map(ids=>new Set(ids)),contacts=live.contact;
    try{for(let i=0;i<live.ncon;i++){
      const c=contacts.get(i);
      try{
        const hit=geoms.map(set=>set.has(c.geom1)||set.has(c.geom2));if(!hit.some(Boolean))continue;
        this.mj.mj_contactForce(this.model,live,i,this.forceBuffer);const force=Math.abs(this.forceBuffer.GetView()[0]);
        for(let k=0;k<2;k++)if(hit[k])out[k]+=force;
      }finally{c.delete();}
    }}finally{contacts.delete();}
    return out;
  }
  dispose(){this.data?.delete();this.forceBuffer?.delete();this.data=null;this.forceBuffer=null;}
}

export class LegOdometry{
  constructor(scene,{dt=scene?.controlDt??.02,fk=null}={}){
    if(!Number.isFinite(dt)||dt<=0)throw new Error('Leg odometry dt must be positive.');
    this.dt=dt;this.ownsFK=!fk;this.fk=fk??new MujocoLegFK(scene);this.state=null;this.disposed=false;
  }
  reset({rootPosW,rootQuatW,rootLinVelW=[0,0,0],dofPos=null,rootAngVelB=null,imuStepCount=null}){
    if(this.disposed)throw new Error('Leg odometry has been disposed.');
    this.pos=finiteVector(rootPosW,3,'Initial root position');const q=quaternion(rootQuatW,'Initial root orientation'),[roll,pitch,yaw]=euler(q);
    this.quat=fromEuler(roll,pitch,yaw);const velocity=finiteVector(rootLinVelW,3,'Initial root velocity');this.velocityB=matVec(transpose3(quatToMat(this.quat)),velocity);
    this.previousGyro=rootAngVelB===null?null:finiteVector(rootAngVelB,3,'Initial gyro');this.previousFeet=dofPos===null?null:this.soles(this.fk.footPosesB(dofPos)).centroids;
    this.previousStance=null;this.previousImuCount=imuStepCount;this.stepCount=0;
    return this.publish({roll,pitch,yaw,velocity,stance:[false,false],forces:[NaN,NaN],heightSource:'init',feet:this.previousFeet??[[0,0,0],[0,0,0]],use:[false,false],yawSource:'init'});
  }
  soles({positions,rotations}){
    return {centroids:positions.map((p,k)=>add(p,matVec(rotations[k],this.fk.soleCentroidLocal[k]))),points:positions.map((p,k)=>this.fk.solePointsLocal[k].map(v=>add(p,matVec(rotations[k],v))))};
  }
  stanceFor(forces,soleWorld,centroids,previousR,currentR,heldWorld){
    if(forces!==null)return {stance:forces.map(f=>f>=20),forces};
    const low=soleWorld.map(points=>Math.min(...points.map(p=>p[2]))),lowest=Math.min(...low);let stance=low.map(z=>z<=lowest+.02);
    if(stance.every(Boolean)&&this.previousFeet){
      const currentRelative=matVec(currentR,sub(centroids[0],centroids[1])),previousRelative=matVec(previousR,sub(this.previousFeet[0],this.previousFeet[1]));
      if(norm(sub(currentRelative,previousRelative))/this.dt>.25){
        const previous=this.previousStance??[false,false],hysteresis=stance.map((value,i)=>value&&previous[i]);let keep;
        if(hysteresis.filter(Boolean).length===1)keep=hysteresis.indexOf(true);
        else{const deviations=centroids.map((p,i)=>norm(sub(scale(sub(matVec(previousR,this.previousFeet[i]),matVec(currentR,p)),1/this.dt),heldWorld)));keep=deviations[0]<=deviations[1]?0:1;}
        stance=[keep===0,keep===1];
      }
    }
    return {stance,forces:[NaN,NaN]};
  }
  update({dofPos,rootQuatW,rootAngVelB,imuDeltaQuatB=null,imuStepCount=null,contactForces=undefined}){
    if(this.disposed||!this.state)throw new Error('Reset leg odometry before updating.');
    const gyro=finiteVector(rootAngVelB,3,'IMU gyro'),imu=quaternion(rootQuatW,'IMU orientation'),[roll,pitch]=euler(imu);
    let increment,yawSource;
    if(imuDeltaQuatB!==null){
      increment=quaternion(imuDeltaQuatB,'IMU increment');
      if(imuStepCount!==null&&this.previousImuCount!==null&&imuStepCount===this.previousImuCount)increment=IDENTITY;
      yawSource='imu_increment';
    }else{increment=fromRotvec(scale(add(this.previousGyro??gyro,gyro),.5*this.dt));yawSource='gyro_trapezoid';}
    const previousQ=this.quat,propagated=unit(quatMul(previousQ,increment)),yaw=euler(propagated)[2],q=fromEuler(roll,pitch,yaw);
    const midpoint=unit(quatMul(previousQ,fromRotvec(scale(toRotvec(quatMul(quatConj(previousQ),q)),.5))));
    const previousR=quatToMat(previousQ),currentR=quatToMat(q),midR=quatToMat(midpoint),{centroids,points}=this.soles(this.fk.footPosesB(dofPos));
    const soleWorld=points.map(foot=>foot.map(p=>matVec(currentR,p))),heldWorld=matVec(midR,this.velocityB);
    const measuredForces=contactForces===undefined?(this.fk.readContactForces?.()??null):contactForces;
    const forces=measuredForces===null?null:finiteVector(measuredForces,2,'Foot contact forces');
    const contact=this.stanceFor(forces,soleWorld,centroids,previousR,currentR,heldWorld),stance=contact.stance;
    const use=stance.map((loaded,i)=>!!this.previousFeet&&loaded&&(this.previousStance===null||this.previousStance[i]));
    let velocity=heldWorld;
    if(use.some(Boolean)){
      const displacement=mean(centroids.flatMap((p,i)=>use[i]?[sub(matVec(previousR,this.previousFeet[i]),matVec(currentR,p))]:[]));
      velocity=scale(displacement,1/this.dt);this.velocityB=matVec(transpose3(midR),velocity);
    }
    this.pos=add(this.pos,scale(velocity,this.dt));let heightSource='integrated';
    if(stance.some(Boolean)){this.pos[2]=-Math.min(...soleWorld.flatMap((foot,i)=>stance[i]?foot.map(p=>p[2]):[]));heightSource='stance_sole';}
    this.quat=q;this.previousGyro=gyro;this.previousFeet=centroids;this.previousStance=stance.slice();if(imuStepCount!==null)this.previousImuCount=imuStepCount;this.stepCount++;
    return this.publish({roll,pitch,yaw,velocity,stance,forces:contact.forces,heightSource,feet:centroids,use,yawSource});
  }
  publish({roll,pitch,yaw,velocity,stance,forces,heightSource,feet,use,yawSource}){
    this.state={source:'leg',quatConvention:'wxyz',step:this.stepCount,posW:this.pos.slice(),quatW:this.quat.slice(),linVelW:velocity.slice(),linVelB:this.velocityB.slice(),linVelHeading:matVec(transpose3(rotateZ(yaw)),velocity),roll,pitch,yaw,
      stance:stance.slice(),contactForces:forces.slice(),heightSource,footPosB:feet.map(p=>p.slice()),velocityFeet:use.filter(Boolean).length,velocityStance:use.slice(),velocityWeights:use.map(Number),gatedOut:[false,false],yawSource};
    return this.state;
  }
  dispose(){if(this.disposed)return;this.disposed=true;if(this.ownsFK)this.fk.dispose();}
}
