// The map is a static, sampled reach estimate for the current grasp settings.
// Placement limits remain independent of this visualization.
export function currentIKWorkspace(state, selectedId) {
  const workspace=state?.ik_workspace;
  const object=state?.objects?.find(item=>item.id===selectedId);
  if(!workspace || !object || workspace.object_id!==selectedId) return null;
  const control=state.controller||{};
  const graspStyle=state.grasp_style||control.grasp_style||"side";
  const eeYaw=state.ee_yaw_deg??control.ee_yaw_deg??45;
  const objectYaw=(object.placement_yaw??0)*180/Math.PI;
  const angleMatches=(a,b)=>Number.isFinite(a)&&Math.abs(((a-b+180)%360+360)%360-180)<.001;
  if(workspace.grasp_style!==graspStyle || !angleMatches(workspace.ee_yaw_deg,eeYaw)
    || !angleMatches(workspace.yaw_deg,objectYaw)) return null;
  return workspace;
}

export function ikWorkspaceStatus(workspace, hasSelection) {
  if(!hasSelection) return "Select an object for IK reach";
  if(!workspace) return "Updating IK reach";
  if(workspace.status==="paused") return "IK scan paused";
  if(workspace.status==="unavailable") return "IK reach unavailable";
  if(workspace.status==="ready") return workspace.cells?.length?"":"No reachable IK samples";
  const progress=workspace.total>0 && Number.isFinite(workspace.checked)
    ?` · ${Math.max(0,Math.min(100,Math.floor(100*workspace.checked/workspace.total)))}%`:"";
  return `Calculating IK reach${progress}`;
}

export function drawIKWorkspace(ctx, point, scale, workspace) {
  if(!workspace || !["computing","ready"].includes(workspace.status)
    || !Number.isFinite(workspace.resolution_m) || workspace.resolution_m<=0) return;
  const half=workspace.resolution_m/2,side=workspace.resolution_m*scale;
  ctx.save();ctx.fillStyle="rgba(103, 116, 212, 0.30)";ctx.beginPath();
  for(const [x,y] of workspace.cells||[]) {
    if(!Number.isFinite(x)||!Number.isFinite(y)) continue;
    const corner=point(x+half,y+half);
    ctx.rect(corner[0],corner[1],side,side);
  }
  ctx.fill();ctx.restore();
}
