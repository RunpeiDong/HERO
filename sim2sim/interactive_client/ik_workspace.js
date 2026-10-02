import {WholeBodyIK,IKError,geomBounds} from './ik.js';
import {computeGraspPreview} from './controller.js';
import {reachStanceCandidates} from './stance.js';

export function workspaceKey(scene,config,yaw=0) {
  const id=scene.selectedObjectId;
  return JSON.stringify([config.mode,scene.tableKind,scene.tableTopZ,id,config.grasp_style,config.ee_yaw_deg,yaw,
    Array.from(scene.data.qpos.slice(0,7)),scene.jointQposAddresses29.map(a=>scene.data.qpos[a]),
    Object.entries(scene.objects).filter(([,o])=>o.active).map(([other,o])=>[other,
      other===id?null:Array.from(scene.data.qpos.slice(o.qadr,o.qadr+7))])]);
}

/** An incremental static IK query. Its MjData and placement scratch are private;
 * it never steps physics or changes the live robot/object poses. Each step
 * performs at most one constrained solve, so queued UI work can take priority.
 * A colored cell means a solution was found at its center, not that every point
 * in the cell, the approach trajectory, or policy execution has been certified.
 */
export class IKWorkspaceScan {
  constructor(scene,config,{yaw=0,resolution=.05,maxCandidates=24,iterations=60,samplePoints=null}={}) {
    if(!scene.objects[scene.selectedObjectId]?.active)throw new Error('Select an object for IK reach.');
    if(!(resolution>=.03&&resolution<=.1)||!Number.isInteger(iterations)||iterations<1||iterations>100)throw new TypeError('Invalid IK map sampling budget.');
    this.config={...config};this.yaw=yaw;this.id=scene.selectedObjectId;this.maxCandidates=maxCandidates;this.iterations=iterations;
    // Methods bind to this proxy's data; the immutable compiled model and native
    // query buffers can be shared because scan steps and simulation never overlap.
    this.scene=Object.create(scene);this.scene.data=scene.scratchData();
    this.scene._placementScratch=scene.scratchData();this.scene.objects=structuredClone(scene.objects);
    this.scene.catalog=structuredClone(scene.catalog);
    const table=scene.metadata.table,half=table.shape==='circle'?[table.radius,table.radius]:table.half_size;
    const bounds={x_min:table.center[0]-half[0],x_max:table.center[0]+half[0],y_min:table.center[1]-half[1],y_max:table.center[1]+half[1]};
    this.scene.catalog.placement_region=bounds; // Diagnostic domain only; live placement limits stay unchanged.
    this.ik=new WholeBodyIK(this.scene);
    const front=geomBounds(this.scene,this.scene.data,scene.tableGeomIds).lower[0];
    this.offsets=Object.fromEntries(['left','right'].map(side=>[side,[-Math.max(0,this.ik.audit.handBounds(this.scene.data,side).upper[0]-(front-.10)),side==='left'?.05:-.05,0]]));
    this.points=samplePoints?.map(p=>p.slice())??[];
    if(!samplePoints)for(let x=bounds.x_min+resolution/2;x<bounds.x_max;x+=resolution)
      for(let y=bounds.y_min+resolution/2;y<bounds.y_max;y+=resolution)this.points.push([x,y]);
    const preferred=scene.data.qpos.slice(scene.objects[this.id].qadr,scene.objects[this.id].qadr+2);
    this.points.sort((a,b)=>Math.hypot(a[0]-preferred[0],a[1]-preferred[1])-Math.hypot(b[0]-preferred[0],b[1]-preferred[1]));
    this.state={status:'computing',object_id:this.id,grasp_style:config.grasp_style,ee_yaw_deg:config.ee_yaw_deg,
      yaw_deg:yaw*180/Math.PI,resolution_m:resolution,bounds,checked:0,total:this.points.length,cells:[],ik_solves:0,
      maximum_palm_error_m:0,maximum_foot_error_m:0,minimum_com_margin_m:null,
      scope:'Sampled static endpoint IK with fixed foot references, collision and balance checks; execution may differ.'};
    this.pending=null;this.warm={left:null,right:null};
  }
  dispose(){this.ik.dispose();this.scene._placementScratch.delete();this.scene.data.delete();}
  finishPoint(){this.state.checked++;this.pending=null;if(this.state.checked>=this.points.length)this.state.status='ready';}
  step(){
    if(this.state.status!=='computing')return;
    if(!this.pending){
      const point=this.points[this.state.checked];if(!point){this.state.status='ready';return;}
      const [x,y]=point;
      // The carton's setup yaw follows the side it is dropped on (worker cartonSetupYaw), so sample each half with the
      // yaw a carton there would actually have.
      const yaw=this.id==='cracker_box'?(y>=0?1:-1)*Math.abs(this.yaw):this.yaw;
      try{this.scene.validatePlacement(this.id,x,y,yaw);}catch{this.finishPoint();return;}
      const object=this.scene.objects[this.id];
      this.scene.data.qpos.set([x,y,object.restZ,Math.cos(yaw/2),0,0,Math.sin(yaw/2)],object.qadr);
      this.scene.forward();
      const preview=computeGraspPreview(this.scene,{...this.config,objectId:this.id,hand:'auto'}),hand=preview.hand,other=hand==='left'?'right':'left';
      const target={hand,objectId:this.id,posture:this.ik.defaultQ.slice(),allowHandObject:true,
        allowHandTableContact:preview.approach==='side',
        palms:{[hand]:{position:preview.position_w,rotation:preview.rotation_w},[other]:this.ik.initialPalms[other]},
        bodyRelativeHands:[other],palmOffsetsTorso:this.offsets};
      const candidates=reachStanceCandidates(this.ik,target,{maxCandidates:this.maxCandidates});
      const warm=this.warm[hand];if(warm)candidates.sort((a,b)=>Number(b.id===warm.id)-Number(a.id===warm.id));
      this.pending={point,target,candidates,index:0};
    }
    const p=this.pending,candidate=p.candidates[p.index++],hand=p.target.hand;
    if(!candidate){this.finishPoint();return;}
    const target={...p.target,posture:candidate.posture,rootPosition:candidate.rootPosition,rootHeight:candidate.rootHeight,rootPitch:candidate.rootPitch};
    const start=this.warm[hand]?.id===candidate.id?this.warm[hand].q:this.ik.defaultQ;
    this.state.ik_solves++;
    try{
      const result=this.ik.solve(start,target,{iterations:this.iterations,maxStep:3,strictEndpoint:true});
      this.state.cells.push([...p.point,hand]);this.warm[hand]={id:candidate.id,q:result.q};
      this.state.maximum_palm_error_m=Math.max(this.state.maximum_palm_error_m,...Object.values(result.residual.palms).map(p=>p.positionError));
      this.state.maximum_foot_error_m=Math.max(this.state.maximum_foot_error_m,result.residual.footPositionError);
      this.state.minimum_com_margin_m=Math.min(this.state.minimum_com_margin_m??Infinity,result.residual.comMargin);
      this.finishPoint();
    }catch(error){if(!(error instanceof IKError))throw error;if(p.index>=p.candidates.length)this.finishPoint();}
  }
}
