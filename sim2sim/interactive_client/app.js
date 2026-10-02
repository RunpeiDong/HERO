import { ClientBridge } from "./client_bridge.js";
import { currentIKWorkspace, drawIKWorkspace, ikWorkspaceStatus } from "./ik_workspace_overlay.js";
import { clampPlacement, placementAxisRange, placementPolygon } from "./placement_workspace.js";
const $ = (id) => document.getElementById(id);
let bridge;
const ui = {catalog: null, state: null, selected: "uiuc_i", pending: false, resetting:false, commandEpoch:0, online:false, loading:true, loadFailed:false, drag: null, camera: null, transform: null};
const canvas = $("placement-map"), ctx = canvas.getContext("2d");
const names = {apple:"Apple",can:"Can",bottle:"Bottle",mug:"Mug",uiuc_i:"UIUC I",cracker_box:"Cheez-It",workbench:"Workbench",round:"Round table",pedestal:"Pedestal"};
const CHEEZ_IT_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" focusable="false" aria-hidden="true">
  <g transform="rotate(-7 16 16)">
    <path d="M6 3h19l3 3v23l-3 2H6z" fill="#a7222b"/>
    <path d="M6 3h19v28H6a1.5 1.5 0 0 1-1.5-1.5v-25A1.5 1.5 0 0 1 6 3Z" fill="#dc373b"/>
    <path d="M6 3h19l3 3H4.5V4.5A1.5 1.5 0 0 1 6 3Z" fill="#f56858"/>
    <path d="M7 16c6-3 12-2 18 1v12H7Z" fill="#c92530"/>
    <text x="15" y="12.5" transform="translate(4.8 0) scale(.68 1)" text-anchor="middle" fill="#fff4d6" font-family="Arial,sans-serif" font-size="6.3" font-weight="900" font-style="italic">Cheez-It</text>
    <path d="m8 15 12-1" stroke="#ffb95c" stroke-width="1" stroke-linecap="round"/>
  </g>
  <g transform="translate(11 24) rotate(-18)">
    <rect x="-4.5" y="-4.5" width="9" height="9" rx="1.3" fill="#f4b643" stroke="#d78722" stroke-width=".7"/>
    <circle r=".8" fill="#c17a20"/>
  </g>
  <g transform="translate(22 23) rotate(15)">
    <path d="m-6-5 1-1 2 .4L0-6l2 .4L5-6l1 1-.4 2L6 0l-.4 2L6 5l-1 1-2-.4L0 6l-2-.4L-5 6l-1-1 .4-2L-6 0l.4-2Z" fill="#ffcf62" stroke="#d88c28" stroke-width=".7" stroke-linejoin="round"/>
    <rect x="-4" y="-4" width="8" height="8" rx="1" fill="none" stroke="#ffe4a0" stroke-width=".7"/>
    <g fill="#c88729"><circle r=".85"/><circle cx="-2.8" cy="-2.8" r=".45"/><circle cx="2.8" cy="-2.8" r=".45"/><circle cx="-2.8" cy="2.8" r=".45"/><circle cx="2.8" cy="2.8" r=".45"/></g>
  </g>
  <path d="m29 7 .6 1.6L31 9l-1.4.5L29 11l-.5-1.5L27 9l1.5-.4Z" fill="#efb440"/>
  <path d="m2 17 1.2 1.1M1 21l1.6-.3" fill="none" stroke="#edb13d" stroke-width="1.1" stroke-linecap="round"/>
</svg>`;
let toastTimer;
let ikShowTimer;
function updateSimulationStatus(){
  // IK activity is shown by its own centred badge; it never replaces or relabels the live indicator.
  const state=ui.resetting?"resetting":ui.loadFailed?"stopped":ui.loading||!ui.online?"loading":
    ui.state?.controller?.busy?"running":"ready";
  const label={loading:"Live MuJoCo simulation · Loading",resetting:"Live MuJoCo simulation · Resetting scene",
    stopped:"Live MuJoCo simulation · Stopped",
    running:"Live MuJoCo simulation with HERO policy running",ready:"Live MuJoCo simulation · HERO policy ready"}[state];
  if($("simulation-status").textContent!==label)$("simulation-status").textContent=label;
  $("simulation-indicator").dataset.state=state;
}
function showIKActivity(active){
  const badge=$("ik-activity");
  if(active){
    if(badge.hidden&&ikShowTimer===undefined)ikShowTimer=setTimeout(()=>{badge.hidden=false;ikShowTimer=undefined;updateSimulationStatus();},60);
  }else{
    clearTimeout(ikShowTimer);ikShowTimer=undefined;
    badge.hidden=true;updateSimulationStatus();
  }
}
function toast(message) { $("toast").textContent=message; $("toast").hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$("toast").hidden=true,4500); }
function objects() { return ui.state?.objects || []; }
function orderedObjectChoices(items) { return [...items].sort((a,b)=>Number(a.id==="uiuc_i")-Number(b.id==="uiuc_i")); }
function spec(id) { return ui.catalog.objects.find(o=>o.id===id); }
// Why an attempt ended unsuccessfully, for the status line: an IK or reach failure says so instead of a bare
// "Attempt unsuccessful". Reads only the controller snapshot; the controller's own stop messages are shown for
// physical stops. Returns null when there is nothing more specific to say.
function failureHint(control){
  if(!control||control.success===true) return null;
  const reason=control.failure_reason,reasons=Array.isArray(control.completion_reasons)?control.completion_reasons:[];
  const cm=v=>Number.isFinite(v)?`${(v*100).toFixed(1)} cm`:null;
  const closest=cm(control.final_attempt_trigger?.distance_m??control.closure_rejected_trigger?.distance_m??control.grasp_alignment?.distance_m);
  const reachLimited=control.stance_plan?.feasible===false||reasons.includes("reach_ik_best_effort")||["compensation_limit","reference_infeasible"].includes(control.ga_frozen_reason);
  const moveCloser="Move the object closer to the robot or toward the grasping hand's side.";
  if(reason?.reason==="alignment_retry_exhausted") return `IK could not bring the hand within reach of the grasp pose${closest?` (closest ${closest})`:""}. ${moveCloser}`;
  if(reasons.includes("final_attempt_close")&&control.grasp_success!==true) return `IK could not reach the grasp pose${reachLimited?" (arm at its reach limit)":""}: the fingers closed ${closest?`${closest} `:""}short of the target and caught nothing. ${moveCloser}`;
  if(typeof reason?.reason==="string"&&/safe clearance/i.test(reason.reason)) return `IK found no approach with safe tabletop clearance for this placement. ${moveCloser}`;
  if(reason?.reason==="object_fell") return "The object fell or was knocked over during the grasp. Stand it up and try again.";
  if(reason?.reason==="object_moved_out_of_grasp") return "The object moved too far during the attempt. Reposition it and try again.";
  if(reason?.reason==="grasp_not_secured") return reachLimited?`IK could only approximate the reach to this object, and the fingers did not secure it. ${moveCloser}`:"The fingers closed on the object but could not hold it. Try a slightly different position or rotation.";
  if(reason?.code==="motion_stopped"&&typeof control.message==="string"&&control.message) return control.message.replace(/\s*(The hand has returned to the initial posture\.?|No collision-free return path was found; the return is incomplete\.?)\s*$/,"").trim()||null;
  if(reachLimited) return `IK could only approximate the reach to this object (no fully feasible planted-foot stance). ${moveCloser}`;
  return null;
}
function selectedObject() { return objects().find(o=>o.id===ui.selected); }
function ikWorkspace() { return currentIKWorkspace(ui.state,ui.selected); }
function syncSelection() {
  const id=ui.state?.selectedObjectId??ui.state?.activeObjectId;
  ui.selected=objects().some(o=>o.id===id)?id:objects()[0]?.id??null;
}
function busy() { return !ui.online || ui.pending || Boolean(ui.state?.controller?.busy); }
async function request(path, body) {
  return bridge.request(path, body);
}
async function command(path, body) {
  if(ui.pending) return;
  const epoch=++ui.commandEpoch;
  ui.pending=true; updateControls();
  try { const state=await request(path,body); if(epoch!==ui.commandEpoch)return; if(state.catalog) ui.catalog=state.catalog; ui.state=state; syncSelection(); updateControls(); drawMap(); }
  catch(error) { if(epoch===ui.commandEpoch&&error.code!=="SIMULATION_RESET"){syncSelection(); toast(error.message);} }
  finally { if(epoch===ui.commandEpoch){ui.pending=false; updateControls();} }
}
async function resetScene() {
  if(!bridge||ui.resetting)return;
  const epoch=++ui.commandEpoch;
  ui.resetting=true;ui.pending=true;ui.drag=null;
  showIKActivity(false);clearTimeout(toastTimer);$("toast").hidden=true;
  $("loading-card").classList.remove("error");$("loading-title").textContent="Resetting your tabletop";
  $("phase-pill").hidden=true;$("status-message").hidden=true;
  showProgress("Restoring the scene…");updateControls();
  try { receiveState(await bridge.reset()); }
  catch(error) {
    if(epoch!==ui.commandEpoch)return;
    ui.online=false;ui.loading=false;ui.loadFailed=true;$("loading-card").classList.add("error");$("loading-card").hidden=false;
    $("loading-title").textContent="Could not reset the scene";$("loading-detail").textContent="Please click Reset to try again.";
    toast(error.message);
  }
  finally { if(epoch===ui.commandEpoch){ui.resetting=false;ui.pending=false;updateControls();} }
}
function createOptions() {
  $("table-options").replaceChildren(...ui.catalog.tables.map(table=>{
    const button=document.createElement("button");button.textContent=names[table.id]||table.label;button.dataset.id=table.id;
    button.onclick=()=>command("/api/config",{table_kind:table.id});return button;
  }));
  const hands=ui.state?.hand_models||[{id:"dex3",label:"Dex3"}];
  $("hand-options").replaceChildren(...hands.map(hand=>{
    const button=document.createElement("button");button.textContent=hand.label;button.dataset.id=hand.id;
    button.setAttribute("aria-label",`${hand.label} hand model`);button.onclick=()=>command("/api/config",{hand_model:hand.id});return button;
  }));
  $("hand-model-label").hidden=$("hand-options").hidden=hands.length<2;
  $("height-options").replaceChildren(...ui.catalog.heights.map((height,index)=>{
    const label=["Low","Mid","High"][index]||"";
    const button=document.createElement("button");button.textContent=`${label} · ${Math.round(height*100)}`;button.dataset.height=height;
    button.setAttribute("aria-label",`${label} table, ${Math.round(height*100)} centimeters`);button.onclick=()=>command("/api/config",{table_height:height});return button;
  }));
  $("object-palette").replaceChildren(...orderedObjectChoices(ui.catalog.objects).map(object=>{
    const button=document.createElement("button");button.className="object-tile";button.dataset.id=object.id;
    const icon=document.createElement("span");icon.className=`object-icon ${object.id}`;icon.setAttribute("aria-hidden","true");
    if(object.id==="cracker_box")icon.innerHTML=CHEEZ_IT_ICON;
    if(object.id!=="uiuc_i"&&Array.isArray(object.color))icon.style.setProperty("--object-color",`rgb(${object.color.slice(0,3).map(v=>Math.round(v*255)).join(",")})`);
    const label=document.createElement("span");label.textContent=names[object.id]||object.label;button.append(icon,label);
    button.onclick=()=>addOrSelect(object.id);return button;
  }));
}
function region() { return ui.state?.placement_region || ui.catalog?.placement_region; }
function clampPosition(id, x, y, axis) {
  const position=clampPlacement(region(),x,y,axis?{axis}:{});
  return {x:position[0],y:position[1]};
}
async function addOrSelect(id) {
  if(busy()) return;
  await command(objects().some(o=>o.id===id)?"/api/select":"/api/add",objects().some(o=>o.id===id)?{object_id:id}:{kind:id});
}
function updateControls() {
  updateSimulationStatus();
  if(!ui.catalog || !ui.state) return;
  const state=ui.state, control=state.controller || {}, item=selectedObject(), locked=busy();
  document.querySelectorAll("#table-options button").forEach(b=>{b.classList.toggle("selected",b.dataset.id===state.table.kind);b.setAttribute("aria-pressed",String(b.dataset.id===state.table.kind));b.disabled=locked;});
  document.querySelectorAll("#height-options button").forEach(b=>{const selected=Math.abs(Number(b.dataset.height)-state.table.height)<.001;b.classList.toggle("selected",selected);b.setAttribute("aria-pressed",String(selected));b.disabled=locked;});
  document.querySelectorAll("#hand-options button").forEach(b=>{const selected=b.dataset.id===(state.hand_model||"dex3");b.classList.toggle("selected",selected);b.setAttribute("aria-pressed",String(selected));b.disabled=locked;});
  document.querySelectorAll(".object-tile").forEach(b=>{const selected=b.dataset.id===ui.selected;b.classList.toggle("selected",selected);b.classList.toggle("present",objects().some(o=>o.id===b.dataset.id));b.setAttribute("aria-pressed",String(selected));b.disabled=locked;});
  $("height-value").textContent=`${Math.round(state.table.height*100)} cm`;
  $("selected-name").textContent=item?names[item.id]||spec(item.id).label:"No object";
  const reach=ikWorkspace(),reachStatus=ikWorkspaceStatus(reach,Boolean(item));
  $("ik-workspace-status").textContent=reachStatus;$("ik-workspace-status").hidden=!reachStatus;
  $("ik-workspace-legend").classList.toggle("inactive",!reach||!['ready','computing'].includes(reach.status));
  const needsReset=control.phase && control.phase!=="idle" && !control.busy;
  $("grasp").disabled=locked || !item || needsReset;
  $("grasp").firstChild.textContent=control.busy?"Running ":needsReset?"Select next target ":"Pick & place ";
  const choices=orderedObjectChoices(objects()),targetSelect=$("pick-object"),choiceKey=choices.map(object=>object.id).join(",");
  if(targetSelect.dataset.choices!==choiceKey){
    const placeholder=document.createElement("option");placeholder.value="";placeholder.textContent="Choose an object";placeholder.disabled=true;
    targetSelect.replaceChildren(placeholder,...choices.map(object=>{
      const option=document.createElement("option");option.value=object.id;option.textContent=names[object.id]||object.label||object.id;return option;
    }));
    targetSelect.dataset.choices=choiceKey;
  }
  targetSelect.value=needsReset?"":ui.selected||"";
  targetSelect.disabled=locked||!objects().length;
  // Reset replaces the worker; it must never wait behind a pending IK request.
  $("reset").disabled=ui.resetting || !bridge;
  $("clear-objects").disabled=locked || !objects().length;
  $("remove-object").disabled=locked || !item;
  const graspStyle=state.grasp_style||control.grasp_style||"side";
  const policy=state.policy||{};
  $("mode-note").textContent=`${policy.label||"Loading policy"} · ${policy.checkpoint||""} · Dex3 finger PD`;
  $("active-hand").textContent=item?(control.hand==="left"?"Left hand":"Right hand"):"Awaiting placement";
  const preview=state.grasp_preview;
  const approach=preview?.approach_family||(control.busy?control.grasp_approach_family:null)||(graspStyle==="side"?"side":"top");
  const tilt=preview?.palm_tilt_deg??control.grasp_palm_tilt_deg;
  $("grasp-approach").textContent=approach==="side"?"Horizontal line":"Vertical line";
  $("grasp-pose").textContent=Number.isFinite(tilt)?(tilt<.1?"Level":tilt>89.9?"Palm down":`${tilt.toFixed(0)}° tilt`):"—";
  $("target-legend").hidden=!preview?.visible;
  if(preview?.visible) $("target-label").textContent=`Target grasp · ${names[preview.objectId]||preview.objectId} · ${preview.hand==="left"?"Left":"Right"} hand`;
  const pelvisActual=control.pelvis_height_actual_m??state.robot?.position?.[2];
  const pelvisTarget=control.pelvis_height_target_m;
  $("pelvis-height").textContent=Number.isFinite(pelvisActual)?`${(100*pelvisActual).toFixed(1)}${Number.isFinite(pelvisTarget)?` → ${(100*pelvisTarget).toFixed(1)}`:""} cm`:"—";
  $("scenario-note").hidden=false;
  $("scenario-note").textContent="Success: part of the released object lands in the tray, even briefly. It still counts if the object later falls out.";
  const r=region();
  const editablePosition=item?clampPlacement(r,...item.position.slice(0,2)):null;
  for(const axis of ["x","y"]) {
    const slider=$("position-"+axis),range=editablePosition?placementAxisRange(r,axis,editablePosition[axis==="x"?1:0]):[r[axis+"_min"],r[axis+"_max"]];
    [slider.min,slider.max]=range;slider.disabled=locked||!item;
    if(item && !ui.drag && document.activeElement!==slider) slider.value=editablePosition[axis==="x"?0:1];
  }
  $("position-yaw").disabled=locked||!item;
  // The carton's rotation is confined to the hand-facing band of its side, sign(y) * [40, 90] deg; the worker clamps
  // every placement into it, the slider only shows that range.
  // The band follows the side the carton sits on now (a slider commit re-places it at its live position, where the
  // worker clamps into that side's band); a carton the robot carried or pushed across the midline keeps a recorded yaw
  // of the other side, so its magnitude is shown on the live side -- resetYaw's own rule.
  const carton=item?.id==="cracker_box",cartonSide=carton&&(item.position?.[1]??0)>0?1:-1;
  [$("position-yaw").min,$("position-yaw").max]=carton?(cartonSide>0?[40,90]:[-90,-40]):[-90,90];
  $("position-yaw").title=carton?"The carton keeps its near end toward the hand on its side: rotation limited to "+(cartonSide>0?"40° to 90°":"−90° to −40°")+". Both hands grasp it with the narrow end face in the thumb web; at ±90° its long axis points at the robot.":"";
  if(item && !ui.drag && document.activeElement!==$("position-yaw")){const yawDeg=(item.placement_yaw??0)*180/Math.PI;$("position-yaw").value=carton?cartonSide*Math.abs(yawDeg):yawDeg;}
  const phase=control.phase || "idle";
  const terminal=!ui.resetting&&(phase==="succeeded"||phase==="failed"),result=terminal?(control.success===true?(control.return_success===false?"Placed · Return incomplete":"Attempt successful"):"Attempt unsuccessful"):"";
  $("phase-pill").hidden=!terminal;$("phase-pill").textContent=result;
  $("phase-pill").className="phase-pill"+(control.success===true?" success":control.success===false?" failed":"");
  const reason=control.failure_reason,blocked=["approach_obstructed","transfer_obstructed"].includes(reason?.code);
  const notice=!terminal&&control.busy&&typeof control.attempt_notice==="string"?control.attempt_notice:"";
  const hint=terminal&&control.success!==true&&!reason?.reset_required&&!blocked?failureHint(control):null;
  $("status-message").hidden=!terminal&&!notice;$("status-message").textContent=notice?notice:terminal&&control.success!==true&&reason?.reset_required
    ?blocked?"Path blocked by another object. Could not find a collision-free IK solution. Please reset the scene.":"Motion could not complete after repeated recovery attempts. Please reset the scene."
    :terminal&&control.success===false&&blocked?"Path blocked by another object. Could not find a collision-free IK solution. Please reset the scene.":hint?`${result} · ${hint}`:result;
  $("status-message").className=notice?"notice":control.success===true?"success":"failed";
  const value=(v,suffix,scale=1)=>v!=null&&Number.isFinite(v)?`${(v*scale).toFixed(1)} ${suffix}`:"—";
  const homeMetric=phase==="return_home"||control.return_success!=null;
  $("distance-label").textContent=homeMetric?"Home pose error":"Grasp distance";
  $("distance-value").title=homeMetric?"Measured palm error relative to the initial body posture":"Measured palm distance to the object grasp target";
  $("lift-value").textContent=value(control.lift_m,"cm",100);$("distance-value").textContent=value(homeMetric?control.home_palm_error:control.palm_error_m,"cm",100);
  $("deposit-value").textContent=control.deposit_success===true?"Placed":terminal&&(control.success===false||control.deposit_success===false)?"Unsuccessful":control.in_tray?"Settling":control.grasp_success?"Placing":"Ready";
  const contact=control.contacts;$("contact-value").textContent=typeof contact==="number"?String(contact):contact&&typeof contact==="object"?String(Object.values(contact).reduce((a,b)=>a+(typeof b==="number"?b:0),0)):"—";
  const stages=["approach","close","lift","transfer","release","return_home"],stage=stages.indexOf(control.stand_with_object?"lift":({crouch:"approach",stance_recovery:"approach",rise:"lift",stand:"lift",hold:"lift",settle:"release",closing:"close",lifting:"lift"})[phase]||phase);
  document.querySelectorAll("[data-phase]").forEach((el,i)=>{el.classList.toggle("active",i===stage);el.classList.toggle("done",el.dataset.phase==="return_home"?control.return_success===true:i<stage || control.motion_completed===true || control.success===true);});
}
function mapTransform() {
  const rect=canvas.getBoundingClientRect(),table=ui.state.table;
  const center=table.center || [.55,-.2,table.height],half=table.half_size || [.35,.4,.02];
  let xmin=-.14,xmax=Math.max(.95,center[0]+(table.radius||half[0])+.08),ymin=Math.min(-.65,center[1]-(table.radius||half[1])-.05),ymax=Math.max(.28,center[1]+(table.radius||half[1])+.05);
  const reach=ikWorkspace(),bounds=reach?.bounds;
  if(bounds && ["ready","computing"].includes(reach.status) && [bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max].every(Number.isFinite)) {
    const padding=(reach.resolution_m||0)/2;
    xmin=Math.min(xmin,bounds.x_min-padding);xmax=Math.max(xmax,bounds.x_max+padding);
    ymin=Math.min(ymin,bounds.y_min-padding);ymax=Math.max(ymax,bounds.y_max+padding);
  }
  const scale=Math.min((rect.width-28)/(ymax-ymin),(rect.height-38)/(xmax-xmin));
  const cx=(ymax+ymin)/2,cy=(xmax+xmin)/2;
  ui.transform={scale,width:rect.width,height:rect.height,cx,cy};
  return (x,y)=>[rect.width/2-(y-cx)*scale,rect.height/2-(x-cy)*scale+5];
}
function fromPointer(event) {
  const rect=canvas.getBoundingClientRect(),t=ui.transform;
  return {x:t.cy-(event.clientY-rect.top-t.height/2-5)/t.scale,y:t.cx-(event.clientX-rect.left-t.width/2)/t.scale};
}
function drawMap() {
  if(!ui.catalog || !ui.state || !canvas.clientWidth) return;
  const dpr=window.devicePixelRatio||1,width=canvas.clientWidth,height=canvas.clientHeight;
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)) {canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
  const point=mapTransform(),s=ui.transform.scale,table=ui.state.table,r=region();
  ctx.strokeStyle="#e7ecdf";ctx.lineWidth=.6;
  for(let x=-.1;x<1.2;x+=.1){const a=point(x,-1),b=point(x,1);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
  for(let y=-.9;y<.8;y+=.1){const a=point(-.2,y),b=point(1.2,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
  const center=table.center||[.55,-.2],half=table.half_size||[.35,.4];
  ctx.fillStyle="#ffffff";ctx.strokeStyle="#c7d1c0";ctx.lineWidth=1.2;ctx.beginPath();
  const cp=point(...center);
  if(table.shape==="round" || table.shape==="circle" || table.radius) ctx.arc(...cp,(table.radius||half[0])*s,0,Math.PI*2);
  else ctx.roundRect(cp[0]-half[1]*s,cp[1]-half[0]*s,half[1]*2*s,half[0]*2*s,3);
  ctx.fill();ctx.stroke();
  const a=point(r.x_max,r.y_max),b=point(r.x_min,r.y_min);
  ctx.save();ctx.clip();
  drawIKWorkspace(ctx,point,s,ikWorkspace());
  ctx.beginPath();placementPolygon(r).forEach(([x,y],i)=>ctx[i?"lineTo":"moveTo"](...point(x,y)));ctx.closePath();
  ctx.fillStyle="#dcebc3aa";ctx.fill();ctx.strokeStyle="#97b276";ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);
  ctx.restore();
  const mid=point(r.x_max,0),bottom=point(r.x_min,0);
  ctx.strokeStyle="#b0bba7";ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(...mid);ctx.lineTo(...bottom);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#64805a";ctx.font="9px sans-serif";ctx.textAlign="center";
  ctx.fillText("Left hand",(a[0]+mid[0])/2,b[1]+12);ctx.fillText("Right hand",(b[0]+mid[0])/2,b[1]+12);
  const tray=ui.state.tray||ui.catalog.tray;
  if(tray){
    const tp=point(...tray.center),tw=tray.outer_size[1]*s,th=tray.outer_size[0]*s;
    ctx.fillStyle="#e3eaf0";ctx.strokeStyle="#879daa";ctx.lineWidth=3;ctx.beginPath();
    ctx.roundRect(tp[0]-tw/2,tp[1]-th/2,tw,th,4);ctx.fill();ctx.stroke();
    ctx.fillStyle="#617b8d";ctx.font="8px sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText("TRAY",tp[0],tp[1]);ctx.textBaseline="alphabetic";
  }
  const robotPosition=ui.state.robot?.position||[-(ui.catalog.robot_setback_m||0),0];
  const robot=point(robotPosition[0],robotPosition[1]);ctx.fillStyle="#6b7d62";ctx.beginPath();ctx.roundRect(robot[0]-12,robot[1]-5,24,10,3);ctx.fill();ctx.beginPath();ctx.moveTo(robot[0],robot[1]-15);ctx.lineTo(robot[0]-4,robot[1]-9);ctx.lineTo(robot[0]+4,robot[1]-9);ctx.fill();ctx.fillStyle="#93a086";ctx.font="7px sans-serif";ctx.textAlign="center";ctx.fillText("ROBOT",robot[0],robot[1]+16);
  for(const object of objects()) {
    const def=spec(object.id),selected=object.id===ui.selected,p=ui.drag?.id===object.id?ui.drag: {x:object.position[0],y:object.position[1]},xy=point(p.x,p.y),rad=Math.max((def.footprint_radius||.035)*s,5);
    ctx.save();ctx.translate(...xy);ctx.rotate(-(object.yaw||0)-Math.PI/2);
    if(selected) {ctx.strokeStyle="#527a35";ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(0,0,rad+4,0,Math.PI*2);ctx.stroke();}
    const color=def.color||object.color||"#c79161";ctx.fillStyle=Array.isArray(color)?`rgb(${color.slice(0,3).map(v=>Math.round(v*255)).join(",")})`:color;
    ctx.strokeStyle="#53614966";ctx.lineWidth=.8;ctx.beginPath();
    if(object.id==="apple") {
      ctx.moveTo(0,-rad*.63);
      ctx.bezierCurveTo(-rad*.85,-rad,-rad*1.1,rad*.18,-rad*.49,rad*.78);
      ctx.bezierCurveTo(-rad*.24,rad*.98,-rad*.10,rad*.73,0,rad*.77);
      ctx.bezierCurveTo(rad*.10,rad*.73,rad*.24,rad*.98,rad*.49,rad*.78);
      ctx.bezierCurveTo(rad*1.1,rad*.18,rad*.85,-rad,0,-rad*.63);
      ctx.closePath();
    }
    else if(object.id==="uiuc_i"||object.id==="cracker_box") ctx.roundRect(-def.size[0]*s/2,-def.size[1]*s/2,def.size[0]*s,def.size[1]*s,1);
    else ctx.arc(0,0,rad*.83,0,Math.PI*2);
    ctx.fill();ctx.stroke();
    if(object.id==="apple") {
      ctx.strokeStyle="#77502f";ctx.lineWidth=Math.max(1,rad*.15);ctx.beginPath();
      ctx.moveTo(0,-rad*.60);ctx.quadraticCurveTo(-rad*.08,-rad*.88,rad*.07,-rad*1.06);ctx.stroke();
      ctx.fillStyle="#57804a";ctx.beginPath();ctx.ellipse(rad*.28,-rad*.89,rad*.32,rad*.15,-.5,0,Math.PI*2);ctx.fill();
    }
    if(object.id==="mug") {ctx.strokeStyle="#936d60";ctx.lineWidth=2;ctx.beginPath();ctx.arc(rad*.8,0,rad*.4,-Math.PI/2,Math.PI/2);ctx.stroke();ctx.fillStyle="#eee7d9";ctx.beginPath();ctx.arc(0,0,rad*.5,0,Math.PI*2);ctx.fill();}
    if(object.id==="bottle") {ctx.fillStyle="#d8b679";ctx.beginPath();ctx.arc(0,0,rad*.4,0,Math.PI*2);ctx.fill();}
    if(object.id==="uiuc_i") {ctx.fillStyle="#13294b";ctx.font="bold 10px Georgia";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("I",0,0);}
    ctx.restore();
  }
}
canvas.addEventListener("pointerdown",event=>{
  if(busy()||!ui.transform) return;
  const p=fromPointer(event),hit=[...objects()].reverse().find(o=>Math.hypot(o.position[0]-p.x,o.position[1]-p.y)<Math.max(spec(o.id).footprint_radius||.04,12/ui.transform.scale));
  if(!hit) return;ui.selected=hit.id;ui.drag={id:hit.id,x:hit.position[0],y:hit.position[1],startX:hit.position[0],startY:hit.position[1],moved:false,dx:hit.position[0]-p.x,dy:hit.position[1]-p.y};canvas.setPointerCapture(event.pointerId);updateControls();drawMap();
});
canvas.addEventListener("pointermove",event=>{if(!ui.drag)return;const p=fromPointer(event);Object.assign(ui.drag,clampPosition(ui.drag.id,p.x+ui.drag.dx,p.y+ui.drag.dy));ui.drag.moved||=Math.hypot(ui.drag.x-ui.drag.startX,ui.drag.y-ui.drag.startY)>.002;drawMap();});
canvas.addEventListener("pointerup",async()=>{if(!ui.drag)return;const {id,x,y,moved}=ui.drag;ui.drag=null;await command(moved?"/api/place":"/api/select",moved?{kind:id,x,y}:{object_id:id});drawMap();});
canvas.addEventListener("pointercancel",()=>{ui.drag=null;syncSelection();updateControls();drawMap();});
canvas.addEventListener("keydown",event=>{const item=selectedObject();if(!item||busy())return;const d={ArrowUp:[.005,0],ArrowDown:[-.005,0],ArrowLeft:[0,.005],ArrowRight:[0,-.005]}[event.key];if(!d)return;event.preventDefault();const p=clampPlacement(region(),...item.position.slice(0,2));command("/api/place",{kind:item.id,...clampPosition(item.id,p[0]+d[0],p[1]+d[1],d[0]?"x":"y")});});
for(const axis of ["x","y","yaw"]) $("position-"+axis).addEventListener("change",()=>{const item=selectedObject();if(item){const base=clampPlacement(region(),...item.position.slice(0,2));if(axis!=="yaw")base[axis==="x"?0:1]=Number($("position-"+axis).value);const p=clampPosition(item.id,...base,axis==="yaw"?undefined:axis);command("/api/place",{kind:item.id,...p,...(axis==="yaw"?{yaw:Number($("position-yaw").value)*Math.PI/180,rotation_edit:true}:{})});}});
$("pick-object").addEventListener("change",event=>{if(!busy()&&event.target.value)command("/api/select",{object_id:event.target.value});});
$("grasp").onclick=()=>command("/api/grasp",{object_id:ui.selected});
$("reset").onclick=resetScene;
$("clear-objects").onclick=()=>command("/api/clear",{});
$("remove-object").onclick=()=>command("/api/remove",{kind:ui.selected});
let viewDrag=null;
function cameraCurrent() {return ui.camera||{azimuth:62,elevation:-26,distance:3.1};}
function cameraSend(update) {ui.camera={...cameraCurrent(),...update};bridge?.setCamera(ui.camera);}
$("view-home").onclick=()=>cameraSend({lookat:[.38,0,.66],distance:3.1,azimuth:62,elevation:-26,preset:"home"});
$("view-side").onclick=()=>{const left=ui.state?.controller?.hand==="left";cameraSend({lookat:[.35,left ? .15 : -.15,.70],distance:2.7,azimuth:left ? -90 : 90,elevation:-22,preset:"side"});};
$("viewport").addEventListener("pointerdown",event=>{if(event.target.closest(".ego-card"))return;viewDrag={x:event.clientX,y:event.clientY,camera:{...cameraCurrent()}};$("viewport").setPointerCapture(event.pointerId);});
$("viewport").addEventListener("pointermove",event=>{if(viewDrag)cameraSend({azimuth:viewDrag.camera.azimuth-(event.clientX-viewDrag.x)*.4,elevation:Math.max(-75,Math.min(-12,viewDrag.camera.elevation-(event.clientY-viewDrag.y)*.25))});});
$("viewport").addEventListener("pointerup",()=>{viewDrag=null;});$("viewport").addEventListener("pointercancel",()=>{viewDrag=null;});
$("viewport").addEventListener("wheel",event=>{if(event.target.closest(".ego-card"))return;event.preventDefault();cameraSend({distance:Math.max(1.4,Math.min(5,cameraCurrent().distance+event.deltaY*.002))});},{passive:false});
$("ego-snapshot").onclick=event=>{event.preventDefault();bridge?.saveEgoImage();};
new ResizeObserver(drawMap).observe(canvas);
function receiveState(state) {
  if(!state)return;
  const first=!ui.catalog;
  ui.state=state;ui.catalog=state.catalog||ui.catalog;ui.online=true;ui.loading=false;ui.loadFailed=false;
  if(!ui.drag)syncSelection();
  if(first&&ui.catalog)createOptions();
  updateControls();drawMap();
  $("connection").className="connection online";
  $("connection").innerHTML="<i></i> Running on your device";
  $("loading-card").hidden=true;
  $("simulation-rate").textContent=state.performance?.simulation_rate!=null?`Simulation ${state.performance.simulation_rate.toFixed(2)}×`:"";
}
function showProgress(message) {
  ui.loading=true;ui.loadFailed=false;updateSimulationStatus();
  $("loading-card").hidden=false;
  $("loading-detail").textContent=message;
  $("connection").className="connection loading";
  $("connection").innerHTML="<i></i> Preparing local simulation";
}
async function init() {
  bridge=new ClientBridge({mainCanvas:$("simulation"),egoCanvas:$("ego-simulation"),onState:receiveState,onProgress:showProgress,onError:message=>{
    ui.online=false;ui.loading=false;ui.loadFailed=true;showIKActivity(false);updateControls();toast(message);
  },onIKActivity:showIKActivity});
  try {await bridge.start();}
  catch(error){
    if(error.code==="SIMULATION_RESET")return;
    ui.online=false;ui.loading=false;ui.loadFailed=true;$("loading-card").classList.add("error");$("loading-card").hidden=false;
    $("loading-title").textContent="The simulation could not start";
    $("loading-detail").textContent=error.message;
    $("status-message").textContent=error.message;
    $("connection").className="connection offline";$("connection").innerHTML="<i></i> Could not start";
    updateSimulationStatus();
  }
}
init();
