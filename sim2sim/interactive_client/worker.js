import * as ort from 'onnxruntime-web/wasm';
import { createSimulation, loadRuntime } from './simulation.js';
import { loadResolvedPolicy } from './policy_runtime.mjs';
import { InteractiveController, policyFeedback, CARTON_GRASP_SETTINGS, cartonGraspSettingOf } from './controller.js';
import { resolveObjectCommand } from './object_selection.js';
import { IKWorkspaceScan, workspaceKey } from './ik_workspace.js';

const HIDDEN_OBJECTS = new Set(['apple']);
// Default object placements use the right half of the table. Keep side
// grasps outboard of the tray to leave room between the arm and torso; the
// carton starts nearer the table's side edge so the turned hand has room.
const defaultAddSpot = kind => kind === 'uiuc_i' ? [.41, -.23, 0] : kind === 'cracker_box' ? [.41, -.37, 0] : [.41, -.32, 0];
// The carton is grasped by its near end, so its setup yaw turns that end toward the hand on its side of the table: a base yaw plus a
// fresh random variation on every add, drag and reset; the grasp plan follows the carton's yaw. The carton's yaw is confined to a
// hand-facing band on every path (slider rotations are clamped into it, drag/reset draws already fall inside it, the no-fit
// fallbacks use the band's edges instead of 0); the carton has 2-fold symmetry, so a slider value is first folded into (-90, 90]
// and the nearest in-band yaw over its half-turn copies is used. Band sign(y) * [40, 90] deg: at 90 deg the carton points its
// long axis at the robot for the thumb-web grasp; a carton lying nearly across the table (yaw <= 25 deg) is pushed away by the
// open hand, so such placements are not offered. Draws: sign(y) * (60 +- 20) deg = the band's lower 40-80 deg.
const CARTON_BASE_YAW_RAD = 60 * Math.PI / 180, CARTON_YAW_JITTER_RAD = 20 * Math.PI / 180;
const CARTON_YAW_BAND_RAD = [40 * Math.PI / 180, 90 * Math.PI / 180];
const cartonSetupYaw = (kind, y, random = Math.random) =>
  kind === 'cracker_box' ? (y > 0 ? 1 : -1) * (CARTON_BASE_YAW_RAD + (2 * random() - 1) * CARTON_YAW_JITTER_RAD) : null;
// [low, high] yaw band (radians, ascending) for a carton at lateral position y (y > 0 is the left hand's side, as in
// the controller's hand choice).
const cartonYawBand = y => y > 0 ? [CARTON_YAW_BAND_RAD[0], CARTON_YAW_BAND_RAD[1]] : [-CARTON_YAW_BAND_RAD[1], -CARTON_YAW_BAND_RAD[0]];
// Nearest in-band yaw to a requested carton yaw (any angle), honouring the carton's half-turn symmetry.
function clampCartonYaw(yaw, y) {
  const [low, high] = cartonYawBand(y);
  if (!Number.isFinite(yaw)) return (low + high) / 2;
  const folded = yaw - Math.PI * Math.round(yaw / Math.PI); // into [-90, 90] deg
  let best = null, bestDistance = Infinity;
  for (const k of [0, -1, 1]) { // k = 0 first: ties resolve to the folded value itself, mirror-symmetrically
    const copy = folded + k * Math.PI, clamped = Math.min(high, Math.max(low, copy)), distance = Math.abs(copy - clamped);
    if (distance < bestDistance - 1e-12) { best = clamped; bestDistance = distance; }
  }
  return best;
}
// No-fit fallbacks for a carton whose drawn yaw does not fit the cell: the exact base yaw, then the band's edges.
const cartonFallbackYaws = y => [(y > 0 ? 1 : -1) * CARTON_BASE_YAW_RAD, ...cartonYawBand(y)];
// The page's placeable area ends 2 cm before the scene's far edge (x_max 0.57 -> 0.55 m). The drag bounds, the placement
// search, the saved-layout clamp and the highlighted workspace all use this trimmed region; the scene's own validation
// (TabletopSimulation.validatePlacement) keeps the exported region, so rollouts that place objects directly are unchanged.
const PAGE_PLACEMENT_X_MAX_TRIM_M = .02;
const pagePlacementRegion = region => ({...region, x_max: Math.round((region.x_max - PAGE_PLACEMENT_X_MAX_TRIM_M) * 1e6) / 1e6});
// Block I has 180 deg symmetry, so one half-turn covers every distinct setup orientation.
// Draw only while placing/resetting; manual rotations and running physics keep their orientation.
const blockISetupYaw = (random = Math.random) => (random() - .5) * Math.PI;
// Repositioning a manipulated object restores its upright setup yaw: robot-facing 0, or for the carton a hand-facing
// yaw drawn once per reset (its magnitude is kept until that reset completes; the sign follows the object's side).
const resetYaws = new Map();
// A carton knocked over during an attempt must be re-placed (its setup yaw is restored by resetYaw); R[2][2] of the
// pose quaternion [w, x, y, z] is 1 - 2 (x^2 + y^2).
const CARTON_UPRIGHT_MIN_COS = Math.cos(15 * Math.PI / 180);
function cartonUpright(kind) {
  if (kind !== 'cracker_box' || !scene?.objects?.[kind]?.active) return true;
  const [, x, y] = scene.objectPose(kind).quaternionW;
  return 1 - 2 * (x * x + y * y) >= CARTON_UPRIGHT_MIN_COS;
}
function resetYaw(kind, y) {
  if (kind !== 'cracker_box') return 0;
  if (!resetYaws.has(kind)) resetYaws.set(kind, Math.abs(cartonSetupYaw(kind, 1)));
  return (y > 0 ? 1 : -1) * resetYaws.get(kind);
}
const publicCatalog = catalog => ({...catalog, objects: catalog.objects.filter(o => !HIDDEN_OBJECTS.has(o.id))});
const HAND_MODELS = {dex3: {label: 'Dex3', prefix: ''}};
let availableHands = ['dex3'];
const config = {table_kind: 'workbench', table_height: .74, mode: 'hero_plus', hand_model: 'dex3',
  grasp_style: 'side', ee_yaw_deg: 45, ...policyFeedback('hero_plus')};
const assets = new Map(), assetWaiters = new Map(), queue = [];
const orientationResetIds = new Set();
let baseURL, embedded = false, assetId = 0, scene, policy, controller, policyManifest;
let pumping = false, timer, nextTick = 0, lastPublished = -Infinity, frameNumber = 0;
let measurementStart = 0, measurementSimStart = 0, simulationRate = null;
let workspaceScan=null,workspace=null,workspaceIdentity=null,lastWorkspacePublished=-Infinity;
const workspaceCache=new Map();
let ikComputing=false;
function showIKComputation(active){
  if(ikComputing===active)return;
  ikComputing=active;self.postMessage({type:'ik_activity',active});
}
function observeIKWork(value){
  const reference=value.reference,ik=reference.ik,solve=ik.solve.bind(ik);
  let planningDepth=0;
  const finishPlanning=()=>{if(--planningDepth===0)showIKComputation(false);};
  // A real solve starts the indicator. Group the solves belonging to one
  // planning operation, and end it before policy inference or physics starts.
  // A cached reference window never starts an activity interval.
  ik.solve=(...args)=>{
    showIKComputation(true);
    try{return solve(...args);}finally{if(!planningDepth)showIKComputation(false);}
  };
  const window=reference.window.bind(reference);
  reference.window=(...args)=>{
    planningDepth++;
    try{return window(...args);}finally{finishPlanning();}
  };
  for(const method of ['prepare','preflightHomePath']){
    const plan=value[method].bind(value);
    value[method]=async(...args)=>{
      planningDepth++;
      try{return await plan(...args);}finally{finishPlanning();}
    };
  }
  return value;
}

function disposeWorkspaceScan(){workspaceScan?.dispose();workspaceScan=null;}
function refreshWorkspace(){
  if(!scene||!controller)return;
  if(controller.phase!=='idle')return;
  const id=scene.selectedObjectId;
  if(!id){disposeWorkspaceScan();workspaceIdentity=null;workspace=null;return;}
  const yaw=orientationResetIds.has(id)?resetYaw(id,scene._placements[id]?.[1]??0):scene._placements[id]?.[2]??0,key=workspaceKey(scene,config,yaw);
  if(key===workspaceIdentity)return;
  disposeWorkspaceScan();workspaceIdentity=key;
  const cached=workspaceCache.get(key);
  if(cached){workspace=structuredClone(cached);return;}
  try{workspaceScan=new IKWorkspaceScan(scene,config,{yaw});workspace=workspaceScan.state;}
  catch(error){workspace={status:'unavailable',object_id:id,grasp_style:config.grasp_style,ee_yaw_deg:config.ee_yaw_deg,yaw_deg:yaw*180/Math.PI,cells:[]};console.warn('IK workspace unavailable:',error.message);}
}
function advanceWorkspace(){
  if(!workspaceScan||controller.phase!=='idle')return;
  try{workspaceScan.step();}
  catch(error){workspace.status='unavailable';console.warn('IK workspace query stopped:',error.message);}
  const done=workspace.status!=='computing';
  if(done){if(workspace.status==='ready'){workspaceCache.set(workspaceIdentity,structuredClone(workspace));if(workspaceCache.size>12)workspaceCache.delete(workspaceCache.keys().next().value);}disposeWorkspaceScan();}
  if(done||performance.now()-lastWorkspacePublished>120){lastWorkspacePublished=performance.now();publish(true,{stateOnly:true});}
}

const progress = message => self.postMessage({type: 'progress', message});
async function disposeRetired(resource) {
  try { await resource?.dispose?.(); }
  catch (error) { console.warn('Could not release an inactive simulation resource.', error); }
}
async function readAsset(path) {
  path = path.replace(/^\.\//, '');
  if (!assets.has(path)) assets.set(path, (async () => {
    if (embedded) {
      const id = ++assetId;
      const compressed = await new Promise((resolve, reject) => {
        assetWaiters.set(id, {resolve, reject}); self.postMessage({type: 'asset', id, path});
      });
      return new Uint8Array(await new Response(new Blob([compressed]).stream()
        .pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    }
    const response = await fetch(new URL(path, baseURL));
    if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  })());
  return assets.get(path);
}
const readJSON = async path => JSON.parse(new TextDecoder().decode(await readAsset(path)));

let previewError = null;
function previewState() {
  let preview;
  try { preview = controller?.preview(); previewError = null; }
  catch (error) { previewError = error.message; return null; } // e.g. a toppled carton: the pick request repeats the message
  if (!preview) return null;
  return {...preview, visible: controller.phase === 'idle',
    palm_tilt_deg: config.grasp_style === 'side' ? 0 : 90,
    approach_family: config.grasp_style === 'side' ? 'side' : 'top'};
}
function currentState() {
  if (!scene || !controller) return null;
  const control = controller.snapshot(), preview = previewState();
  const hand = control.phase === 'idle' ? preview?.hand || 'right' : control.hand;
  const state = scene.readState(), index = hand === 'left' ? 0 : 1;
  const palm = Array.isArray(state.palmPosW) ? state.palmPosW[index] : state.palmPosW[hand];
  const target = controller.graspGoal || preview?.position_w;
  const sceneState = scene.uiState();
  sceneState.objects = sceneState.objects.map(object => ({...object,
    placement_yaw: orientationResetIds.has(object.id) ? resetYaw(object.id, scene._placements[object.id]?.[1] ?? 0) : scene._placements[object.id]?.[2] ?? 0,
    needs_orientation_reset: orientationResetIds.has(object.id)}));
  const pageRegion = pagePlacementRegion(scene.catalog.placement_region), pageCatalog = publicCatalog(scene.catalog);
  if (pageCatalog?.placement_region) pageCatalog.placement_region = pageRegion;
  const handWorkspaces = sceneState.hand_workspaces && Object.fromEntries(Object.entries(sceneState.hand_workspaces).map(([side, w]) => [side, {...w, x_max: Math.min(w.x_max, pageRegion.x_max)}]));
  return {app: 'tabletop_lab_browser_v1', ...sceneState, placement_region: pageRegion, ...(handWorkspaces ? {hand_workspaces: handWorkspaces} : {}), catalog: pageCatalog, preview_error: previewError,
    reset_snapshot: resetSnapshot(),
    ik_workspace:workspace?{...workspace,status:controller.phase==='idle'?workspace.status:'paused'}:null,
    grasp_style: config.grasp_style, ee_yaw_deg: config.ee_yaw_deg, grasp_preview: preview,
    hand_model: config.hand_model, hand_models: availableHands.map(id => ({id, label: HAND_MODELS[id].label})),
    controller_choices: ['hero_plus'], policy: policyManifest.policies[config.mode],
    controller: {...control, hand, approach_stage: control.stage,
      goal_adjustment_m: control.goal_adjustment, ga_update_count: control.goal_adjust_count,
      pelvis_height_actual_m: state.rootPosW[2], pelvis_height_target_m: controller.rootHeight,
      palm_error_m: control.grasp_distance_m ?? (palm && target ? Math.hypot(...Array.from(palm, (v, i) => v - target[i])) : null),
      grasp_palm_tilt_deg: config.grasp_style === 'side' ? 0 : 90},
    performance: {simulation_rate: simulationRate},
    render: {frame_number: frameNumber, simulation_time: scene.time}};
}
function publish(force = false, {stateOnly=false}={}) {
  if (!scene || !controller) return;
  const now = performance.now();
  if (!force && now - lastPublished < 30) return;
  lastPublished = now;if(!stateOnly)frameNumber++;
  const state = currentState();
  if(!stateOnly)self.postMessage({type: 'frame', frame: scene.renderSnapshot(), preview: state.grasp_preview});
  self.postMessage({type: 'state', state});
}

async function loadPolicy({commit = true} = {}) {
  progress('Loading HERO on your device…');
  policyManifest ??= await readJSON('policies/manifest.json');
  const next = await loadResolvedPolicy(config.mode, {manifest: policyManifest,
    assetResolver: path => readAsset(`policies/${path}`), ort});
  if (commit) { await policy?.dispose?.(); policy = next; }
  return next;
}
function rebuildController() {
  // Construct before disposing: a constructor failure must never leave a disposed controller as the live one.
  const next = observeIKWork(new InteractiveController(scene, policy, {...config, objectId: scene.selectedObjectId}));
  controller?.dispose?.();
  policy.reset(); controller = next;
  measurementStart = performance.now(); measurementSimStart = scene.time; simulationRate = null;
}
function choosePlacement(kind, preferred = [.41, -.23, 0], {spacing = 0, resetRobot = false, targetScene = scene, autoYaw = false} = {}) {
  const r = pagePlacementRegion(targetScene.catalog.placement_region);
  const candidate = [Math.max(r.x_min, Math.min(r.x_max, preferred[0])),
    Math.max(r.y_min, Math.min(r.y_max, preferred[1])), preferred[2] || 0];
  const choices = [candidate];
  // Integer indices avoid accumulating past an exact boundary (e.g. .44).
  // Include the boundary even when the range is not a multiple of the grid.
  for (let i = 0; i <= Math.ceil((r.x_max - r.x_min) / .02); i++)
    for (let j = 0; j <= Math.ceil((r.y_max - r.y_min) / .02); j++)
      choices.push([Math.min(r.x_max, r.x_min + i * .02),
        Math.min(r.y_max, r.y_min + j * .02), candidate[2]]);
  choices.sort((a, b) => Math.hypot(a[0] - candidate[0], a[1] - candidate[1])
    - Math.hypot(b[0] - candidate[0], b[1] - candidate[1]));
  const radius = targetScene.catalog.objects.find(o => o.id === kind)?.footprint_radius ?? .04;
  const placed = targetScene.uiState().objects;
  const blockIYaw = autoYaw && kind === 'uiuc_i' ? blockISetupYaw() : null;
  for (const gap of spacing > 0 ? [spacing, 0] : [0]) for (const pose of choices) {
    if (gap && placed.some(other => other.id !== kind &&
      Math.hypot(pose[0] - other.position[0], pose[1] - other.position[1]) < radius + other.footprint_radius + gap)) continue;
    // The carton's setup yaw depends on the side of the cell it finally lands in, not on the requested spot; a drawn
    // yaw that does not fit the cell falls back to the exact base yaw and then to the band's edges before the cell is given up.
    const yaws = autoYaw && kind === 'cracker_box'
      ? [cartonSetupYaw(kind, pose[1]), ...cartonFallbackYaws(pose[1])]
      : kind === 'cracker_box' ? [clampCartonYaw(pose[2], pose[1]), ...cartonFallbackYaws(pose[1])]
      : blockIYaw !== null ? [blockIYaw, 0] : [pose[2]];
    for (const yaw of yaws) {
      try { targetScene.placeObject(kind, pose[0], pose[1], yaw, {resetRobot}); return; } catch { /* Try the next yaw, then the nearest available valid placement. */ }
    }
  }
  throw new Error('This object has no valid placement on the selected table.');
}
async function rebuildScene(placements, nextPolicy = policy, {selectedObjectId = scene?.selectedObjectId, randomizeBlockIYaw = false} = {}) {
  const selected = selectedObjectId;
  progress('Preparing the robot, table, and physical contacts…');
  const handPrefix = HAND_MODELS[config.hand_model]?.prefix ?? '';
  const next = await createSimulation({mode: config.mode, tableKind: config.table_kind,
    tableHeight: config.table_height, readAsset: path => readAsset(`sceneassets/${handPrefix}${path}`),
    placements: {}});
  let nextController;
  try {
    next.setPolicyParameters(nextPolicy);
    for (const [kind, pose] of Object.entries(placements ?? {uiuc_i: [.41, -.23, 0]})) {
      if (!next.objects[kind]) throw new Error('The saved scene contains an unknown object.');
      choosePlacement(kind, pose, {targetScene: next, autoYaw: randomizeBlockIYaw && kind === 'uiuc_i'});
    }
    if (selected && next.objects[selected]?.active) next.selectObject(selected);
    nextController = observeIKWork(new InteractiveController(next, nextPolicy, {...config, objectId: next.selectedObjectId}));
    const description = next.renderDescription();
    nextPolicy.reset();
    self.postMessage({type: 'model', description});
  } catch (error) {
    await disposeRetired(nextController); await disposeRetired(next); throw error;
  }
  // A smaller table may have no room for the saved layout. Commit only once
  // every object fits, so a rejected change keeps the original scene usable.
  const previousController = controller, previousScene = scene;
  disposeWorkspaceScan();workspaceIdentity=null;workspace=null;
  scene = next; policy = nextPolicy; controller = nextController;
  measurementStart = performance.now(); measurementSimStart = scene.time; simulationRate = null;
  orientationResetIds.clear();
  // Cleanup follows the commit and must never roll back the live configuration.
  await disposeRetired(previousController); await disposeRetired(previousScene);
}
function savedPlacements() {
  return Object.fromEntries(Object.entries(scene?._placements ?? {}).map(([id, pose]) =>
    [id, [pose[0], pose[1], orientationResetIds.has(id) ? resetYaw(id, pose[1]) : pose[2]]]));
}
function validatedConfig(body = {}, previous = config) {
  const mode = body.mode ?? previous.mode, kind = body.table_kind ?? previous.table_kind;
  const height = Number(body.table_height ?? previous.table_height);
  const style = body.grasp_style ?? previous.grasp_style ?? 'side', handModel = body.hand_model ?? previous.hand_model ?? 'dex3';
  if (mode !== 'hero_plus') throw new Error('This demo uses HERO.');
  if (!HAND_MODELS[handModel] || !availableHands.includes(handModel)) throw new Error('Unsupported hand model.');
  if (!['workbench', 'round', 'pedestal'].includes(kind) || ![.5, .74, .88].includes(height))
    throw new Error('Unsupported controller or table.');
  if (style !== 'side') throw new Error('This demo grasps from the side.');
  // Carton grasp selection: 'auto' is the default; the page may pin a variant through the ?carton_grasp= URL
  // parameter (passed with /init) so the poses can be compared on the page. Saved snapshots keep it.
  const cartonGrasp = cartonGraspSettingOf(body.carton_grasp ?? previous.carton_grasp ?? 'auto');
  if (typeof cartonGrasp !== 'string' || !CARTON_GRASP_SETTINGS.includes(cartonGrasp))
    throw new Error('Use the auto, end_face, crotch, spine or spine90 carton grasp.');
  // Side grasps use a fixed wrist angle, including after a scene restore.
  return {mode, table_kind: kind, table_height: height, hand_model: handModel, grasp_style: 'side', ee_yaw_deg: 45, carton_grasp: cartonGrasp, ...policyFeedback(mode)};
}
function resetSnapshot() {
  return {version: 1, config: validatedConfig(config), placements: savedPlacements(),
    selectedObjectId: scene?.selectedObjectId ?? null};
}
function validateResetSnapshot(snapshot) {
  if (snapshot === undefined) return null;
  if (!snapshot || snapshot.version !== 1 || !snapshot.config || typeof snapshot.config !== 'object'
      || !snapshot.placements || typeof snapshot.placements !== 'object' || Array.isArray(snapshot.placements))
    throw new Error('The saved scene setup is invalid.');
  const placements = Object.fromEntries(Object.entries(snapshot.placements).map(([id, pose]) => {
    if (HIDDEN_OBJECTS.has(id)) throw new Error('The saved scene contains an object that is not in this demo.');
    if (!Array.isArray(pose) || pose.length !== 3 || !pose.every(Number.isFinite))
      throw new Error('The saved object placement is invalid.');
    return [id, pose.slice()];
  }));
  const selectedObjectId = snapshot.selectedObjectId ?? null;
  if (selectedObjectId !== null && (typeof selectedObjectId !== 'string' || !Object.hasOwn(placements, selectedObjectId)))
    throw new Error('The saved target is not in the scene.');
  return {config: validatedConfig(snapshot.config), placements, selectedObjectId};
}
function requireEditable() {
  if (controller?.busy) throw new Error('Reset the current attempt before editing the scene.');
}
function prepareEditing() {
  if (controller.phase !== 'idle') scene.resetRobot();
}

async function dispatch({path, body = {}}) {
  if (path === '/init') {
    baseURL = body.assetsBase; embedded = body.embedded;
    // Probe optional hand-model asset trees BEFORE validating the restored snapshot: a Reset
    // replaces the worker and restores hand_model, which must already count as available.
    for (const [id, hand] of Object.entries(HAND_MODELS)) {
      if (id === 'dex3' || availableHands.includes(id)) continue;
      try { await readJSON(`sceneassets/${hand.prefix}manifest.json`); availableHands.push(id); } catch { /* not shipped in this build */ }
    }
    const restore = validateResetSnapshot(body.restore);
    Object.assign(config, restore?.config ?? validatedConfig());
    if (body.carton_grasp !== undefined) Object.assign(config, validatedConfig({carton_grasp: body.carton_grasp}));
    progress('Loading local physics and inference…');
    const physicsWasm = await readAsset('runtime/mujoco.wasm');
    const physicsURL = URL.createObjectURL(new Blob([physicsWasm], {type: 'application/wasm'}));
    try {
      await loadRuntime({wasmBinary: physicsWasm, locateFile: () => physicsURL});
    } finally { URL.revokeObjectURL(physicsURL); }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    // Emscripten still resolves its binary URL even when wasmBinary is set.
    // An explicit local blob URL also works for a classic worker opened from file://.
    const wasm = await readAsset('runtime/ort-wasm-simd-threaded.wasm');
    ort.env.wasm.wasmPaths = {wasm: URL.createObjectURL(new Blob([wasm], {type: 'application/wasm'}))};
    await loadPolicy();
    // UI Reset replaces the worker and restores the layout through /init. A fresh Block I yaw
    // belongs to this setup action; configuration rebuilds preserve the saved orientation.
    await rebuildScene(restore?.placements, policy, {selectedObjectId: restore?.selectedObjectId, randomizeBlockIYaw: true});
    return currentState();
  }
  if (!scene) throw new Error('Wait for the local simulation to finish loading.');
  if (path === '/api/catalog') return publicCatalog(scene.catalog);
  if (path === '/api/state') return currentState();
  if (path === '/api/reset') {
    const placements = savedPlacements(), selected = scene.selectedObjectId;
    controller.cancel(); scene.reset({});
    // A newly placed object may occupy a deposited object's former setup spot.
    // Restore the layout through the same collision checks used when adding.
    // Every reset re-draws the carton's hand-facing yaw and the Block I's half-turn setup yaw.
    for (const [id, pose] of Object.entries(placements)) choosePlacement(id, pose, {autoYaw: id === 'cracker_box' || id === 'uiuc_i'});
    if (selected && scene.objects[selected]?.active) scene.selectObject(selected);
    orientationResetIds.clear(); resetYaws.clear(); rebuildController(); return currentState();
  }
  requireEditable();
  if (path === '/api/config') {
    const nextConfig = validatedConfig(body), {mode, table_kind: kind, table_height: height, hand_model: handModel} = nextConfig;
    const policyChanged = mode !== config.mode;
    const sceneChanged = policyChanged || kind !== config.table_kind || height !== config.table_height || handModel !== config.hand_model;
    const placements = savedPlacements(), previousConfig = {...config};
    const previousPolicy = policy;
    let nextPolicy = policy;
    Object.assign(config, nextConfig);
    try {
      if (policyChanged) nextPolicy = await loadPolicy({commit: false});
      if (sceneChanged) await rebuildScene(placements, nextPolicy); else { prepareEditing(); rebuildController(); }
    } catch (error) {
      Object.assign(config, previousConfig);
      if (nextPolicy !== previousPolicy) await disposeRetired(nextPolicy);
      throw error;
    }
    if (policyChanged) await disposeRetired(previousPolicy);
  } else if (path === '/api/add') {
    if (HIDDEN_OBJECTS.has(body.kind) || !scene.objects[body.kind]) throw new Error('Choose an available object.');
    if (scene.objects[body.kind].active) { scene.selectObject(body.kind); prepareEditing(); }
    else choosePlacement(body.kind, defaultAddSpot(body.kind), {spacing: .06, resetRobot: controller.phase !== 'idle', autoYaw: true});
    rebuildController();
  } else if (path === '/api/select') {
    const robot = scene.readState();
    const id = body.text !== undefined ? resolveObjectCommand(body.text, scene.uiState().objects,
      {position: Array.from(robot.rootPosW), quaternion: Array.from(robot.rootQuatW)}) : body.object_id;
    if (HIDDEN_OBJECTS.has(id)) throw new Error('Choose an available object.');
    scene.selectObject(id); prepareEditing(); rebuildController();
  } else if (path === '/api/place') {
    // Setup yaw is independent of the object's simulated landing orientation.
    // Repositioning a manipulated object restores its upright setup yaw (see resetYaw).
    if (HIDDEN_OBJECTS.has(body.kind)) throw new Error('Choose an available object.');
    const x = Number(body.x), y = Number(body.y), previous = scene._placements[body.kind]?.[2] ?? 0;
    let candidates;
    if (body.kind === 'cracker_box' && !body.rotation_edit)
      // Every drag or reset re-draws the carton's hand-facing yaw; fall back to the exact base yaw, the previous yaw folded
      // into the band and finally the band's edges when the drawn footprint does not fit where the user dropped it.
      candidates = [cartonSetupYaw(body.kind, y), (y > 0 ? 1 : -1) * CARTON_BASE_YAW_RAD, clampCartonYaw(previous, y), ...cartonYawBand(y)];
    else if (body.kind === 'cracker_box')
      // A slider rotation is kept only inside the hand-facing band of the carton's side; like any explicit
      // rotation it is refused, not replaced, when that footprint does not fit.
      candidates = [clampCartonYaw(Number(body.yaw ?? previous), y)];
    else if (body.kind === 'uiuc_i' && !body.rotation_edit)
      candidates = [blockISetupYaw(), 0];
    else candidates = [orientationResetIds.has(body.kind) && !body.rotation_edit ? 0 : Number(body.yaw ?? previous)];
    let failure = null;
    for (const yaw of candidates) {
      try { scene.placeObject(body.kind, x, y, yaw, {resetRobot: controller.phase !== 'idle'}); failure = null; break; }
      catch (error) { failure ??= error; }
    }
    if (failure) throw failure;
    orientationResetIds.delete(body.kind); resetYaws.delete(body.kind); rebuildController();
  } else if (path === '/api/clear') {
    scene.clearObjects(); orientationResetIds.clear(); resetYaws.clear(); prepareEditing(); rebuildController();
  } else if (path === '/api/remove') {
    scene.removeObject(body.kind); orientationResetIds.delete(body.kind); resetYaws.delete(body.kind); prepareEditing(); rebuildController();
  } else if (path === '/api/grasp') {
    if (controller.phase !== 'idle') throw new Error('Reposition an object or select the next target before starting another attempt.');
    measurementStart = performance.now(); measurementSimStart = scene.time;
    const graspConfig = validatedConfig();
    await controller.prepare({objectId: body.object_id || scene.activeObjectId, hand: 'auto',
      approach: graspConfig.grasp_style, yawDeg: graspConfig.ee_yaw_deg});
    nextTick = performance.now();
  } else throw new Error('Unsupported local command.');
  return currentState();
}

function schedule(delay = 0) {
  if (timer !== undefined) return;
  timer = setTimeout(() => { timer = undefined; pump(); }, delay);
}
function advanceSimulationDeadline(previousDeadline, now) {
  // Keep up to 250 ms of missed wall time after a costly reference rebuild.
  // Every catch-up tick still advances exactly one 20 ms control/physics step
  // and yields through setTimeout; a suspended tab cannot accumulate unlimited debt.
  return Math.max(previousDeadline + 20, now - 250);
}
async function pump() {
  if (pumping) return;
  pumping = true;
  try {
    while (queue.length) {
      const request = queue.shift();
      try {
        const result = await dispatch(request);
        refreshWorkspace();publish(true); self.postMessage({type: 'response', id: request.id, result:request.path==='/api/catalog'?result:currentState()});
      } catch (error) {
        publish(true); self.postMessage({type: 'response', id: request.id, error: error.message});
      }
    }
    if (controller?.busy) {
      await controller.tick();
      if (controller.objectId && (controller.maxLift > .04 || controller.graspSuccess || !cartonUpright(controller.objectId)))
        orientationResetIds.add(controller.objectId);
      const now = performance.now(), elapsed = (now - measurementStart) / 1000;
      if (elapsed > .1) simulationRate = (scene.time - measurementSimStart) / elapsed;
      nextTick = advanceSimulationDeadline(nextTick, now);
      // Show completed catch-up poses even when they arrive inside the normal
      // publication interval; the renderer can coalesce them on its next frame.
      publish(!controller.busy || nextTick <= now);
    } else advanceWorkspace();
  } catch (error) {
    controller?.cancel(); self.postMessage({type: 'error', message: error.message}); publish(true);
  } finally {
    showIKComputation(false);
    pumping = false;
    if (queue.length) schedule(0);
    else if (controller?.busy) schedule(Math.max(0, nextTick - performance.now()));
    else if (workspaceScan&&controller?.phase==='idle') schedule(0);
  }
}
self.onmessage = ({data}) => {
  if (data.type === 'asset') {
    const waiter = assetWaiters.get(data.id); assetWaiters.delete(data.id);
    if (data.error) waiter?.reject(new Error(data.error)); else waiter?.resolve(data.bytes);
  } else if (data.type === 'request') {
    queue.push(data); schedule();
  }
};
