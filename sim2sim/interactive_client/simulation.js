/** Native MuJoCo WASM scene adapter. All physics stays in the caller's worker. */
import loadMujoco from '@mujoco/mujoco';
import {prepareBrowserObjectGeometry,preserveCompiledBottleInertia,objectTrayOverlap} from './object_geometry.js';
import {containsPlacement,clampPlacement} from './placement_workspace.js';
import {LegOdometry} from './leg_odometry.js';

let runtimePromise;
export function loadRuntime(options = {}) { return runtimePromise ??= loadMujoco(options); }
const copy = value => new Float64Array(value);
const pick = (values, addresses) => Float64Array.from(addresses, i => values[i]);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const GRIP_KP_SCALE = 3, GRIP_SQUEEZE_FRACTION = 0.6, GRIP_BLOCKED_ERROR_RAD = 0.05;
const isFlexionFingerJoint = name => !/thumb_0|thumb_1|_yaw/.test(name);
const finite = (values, length) => values?.length === length && Array.from(values).every(Number.isFinite);
const asBytes = value => typeof value === 'string' ? new TextEncoder().encode(value)
  : value instanceof Uint8Array ? value : new Uint8Array(value);

/** readAsset receives a path relative to sceneassets and returns bytes or text. */
export async function createSimulation(options = {}) {
  const mj = options.mj ?? await loadRuntime();
  const baseURL = options.assetBaseURL ?? '/sceneassets/';
  const read = options.readAsset ?? options.loadAsset ?? (async path => {
    const response = await fetch(new URL(path, new URL(baseURL, globalThis.location?.href ?? 'http://localhost/')));
    if (!response.ok) throw new Error(`Scene asset ${path}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  });
  const json = async path => JSON.parse(new TextDecoder().decode(asBytes(await read(path))));
  const manifest = options.manifest ?? await json('manifest.json');
  const mode = options.mode ?? 'hero_plus';
  const tableKind = options.tableKind ?? options.table_kind ?? 'workbench';
  const height = options.tableHeight ?? options.table_height ?? .74;
  const key = `${mode}_${tableKind}_${Math.round(height * 100)}`;
  const entry = manifest.scenes[key];
  if (!entry) throw new Error(`Unsupported scene: ${key}`);
  const root = '/tabletop_client';
  for (const path of [root, `${root}/meshes`, `${root}/textures`]) {
    try { mj.FS.mkdir(path); } catch (error) { if (!mj.FS.analyzePath(path).exists) throw error; }
  }
  await Promise.all(Object.keys(manifest.assets).map(async path => {
    if (!mj.FS.analyzePath(`${root}/${path}`).exists) mj.FS.writeFile(`${root}/${path}`, asBytes(await read(path)));
  }));
  const [metadata, xml] = await Promise.all([json(entry.metadata), read(entry.xml)]);
  const browserXML=prepareBrowserObjectGeometry(metadata,new TextDecoder().decode(asBytes(xml)));
  mj.FS.writeFile(`${root}/${entry.xml}`, asBytes(browserXML));
  const model = mj.MjModel.from_xml_path(`${root}/${entry.xml}`);
  preserveCompiledBottleInertia(model,metadata);
  return new TabletopSimulation(mj, model, metadata, options);
}
export const loadScene = createSimulation;

export class TabletopSimulation {
  constructor(mj, model, metadata, options = {}) {
    this.mj = mj; this.model = model; this.metadata = metadata;
    this.data = new mj.MjData(model);
    for (const [name, values] of Object.entries(metadata.modelOverrides ?? {})) {
      const target = model[name];
      if (!target || target.length !== values.length) throw new Error(`Physical model field ${name} has incompatible dimensions.`);
      target.set(values);
    }
    mj.mj_setConst(model, this.data);
    for (const key of ['mode', 'tableKind', 'tableTopZ', 'physicsDt', 'controlDt', 'jointNames29',
      'jointQposAddresses29', 'jointDofAddresses29', 'jointLimited', 'fingerJointNames14', 'fingerQposAddresses14',
      'fingerDofAddresses14', 'defaultBodyQ', 'bodyReferenceIds32', 'referenceBodyNames32',
      'virtualReferenceBodies', 'palmBodyIds', 'ankleBodyIds', 'tableGeomIds', 'trayGeomIds',
      'trayBottomGeomId', 'floorGeomId', 'egoCameraId', 'catalog']) this[key] = structuredClone(metadata[key]);
    this.objects = structuredClone(metadata.objects);
    // The exported assets contain one free body for each catalog object. All
    // can be present together; selection identifies the next grasp target.
    this.catalog.max_objects = Object.keys(this.objects).length;
    this.catalog.placement_mode = 'multiple';
    // The exported editor polygon governs validation, automatic placement
    // and every UI control. Static IK reach is an informational
    // overlay; table, tray, object and robot collisions still constrain edits.
    this.catalog.hand_workspaces = {
      left: {...structuredClone(this.catalog.placement_region), y_min: 0},
      right: {...structuredClone(this.catalog.placement_region), y_max: 0},
    };
    this.objectBodyIds = Object.fromEntries(Object.entries(this.objects).map(([id, obj]) => [id, obj.bodyId]));
    this.tray = {...structuredClone(metadata.tray), innerBounds: metadata.tray.inner_bounds,
      outerBounds: metadata.tray.outer_bounds, bottomTopZ: metadata.tray.bottom_top_z, wallTopZ: metadata.tray.wall_top_z};
    this.kp = copy(metadata.kp); this.kd = copy(metadata.kd); this.effortLimit = copy(metadata.effortLimit);
    this.handClosure = {left: 0, right: 0};
    this.handGripKpScale = {left: 1, right: 1};
    // Finger joint count comes from the exported scene (Dex3: 7 per hand; other hands may differ). Left hand first.
    this.fingerCount = metadata.fingerJointNames14.length; this.fingersPerHand = this.fingerCount / 2;
    this.handModel = metadata.handModel ?? (this.fingersPerHand === 7 ? 'dex3' : this.fingersPerHand === 12 ? 'inspire' : 'unknown');
    if (!Number.isInteger(this.fingersPerHand) || this.fingersPerHand < 1) throw new Error('Expected an even number of finger joints, left hand first.');
    this.fingerGoal = new Float64Array(this.fingerCount); this.fingerTarget = new Float64Array(this.fingerCount);
    this.flexionFinger = metadata.fingerJointNames14.map(isFlexionFingerJoint);
    this.lastTorque = new Float64Array(29); this.lastFingerTorque = new Float64Array(this.fingerCount);
    this._jacPosition = new mj.DoubleBuffer(3 * model.nv);
    this._jacRotation = new mj.DoubleBuffer(3 * model.nv);
    this._distancePoints = new mj.DoubleBuffer(6); this._contactForce = new mj.DoubleBuffer(6);
    this._imuRotation = new mj.DoubleBuffer(4);
    this.imuDeltaQuatB = new Float64Array([1,0,0,0]); this.imuStepCount = 0;
    this.legOdometry = null;
    this._placements = {}; this._selectedObjectId = null; this.lastStepContacts = [];
    this.lastStepTrayLandings = [];
    this._cacheTrayLandingParticipants();
    this._placementScratch = new mj.MjData(model);
    this._renderDescription = null;
    const initial = metadata.defaultPlacements?.uiuc_i ?? [.41, -.23, 0];
    const [initialX,initialY] = clampPlacement(this.catalog.placement_region,initial[0],initial[1]);
    this.reset(options.placements ?? {[options.objectId ?? 'uiuc_i']: [
      initialX, initialY, initial[2],
    ]});
    if (this.mode === 'hero_plus') {
      this.legOdometry = new LegOdometry(this);
      this._resetLegOdometry();
    }
  }

  get time() { return this.data.time; }
  get selectedObjectId() { return this._selectedObjectId; }
  get activeObjectId() { return this.selectedObjectId; }
  bodyName(id) { return this.metadata.bodyNames[id] ?? ''; }
  geomName(id) { return this.metadata.geomNames[id] ?? ''; }
  bodyId(name) { return this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_BODY.value, name); }
  geomId(name) { return this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_GEOM.value, name); }

  scratchData() { return new this.mj.MjData(this.model, this.data); }
  forward(data = this.data) { this.mj.mj_forward(this.model, data); return data; }
  kinematics(data = this.data) { this.mj.mj_kinematics(this.model, data); return data; }
  ikForward(data) {
    if (data === this.data) throw new Error('IK kinematics requires independent scratch data.');
    this.mj.mj_kinematics(this.model, data);
    this.mj.mj_comPos(this.model, data);
    return data;
  }
  jacBody(data, id) {
    this.mj.mj_jacBody(this.model, data, this._jacPosition, this._jacRotation, id);
    return {position: copy(this._jacPosition.GetView()), rotation: copy(this._jacRotation.GetView())};
  }
  jacPoint(data, pointW, id) {
    this.mj.mj_jac(this.model, data, this._jacPosition, this._jacRotation, Array.from(pointW), id);
    return {position: copy(this._jacPosition.GetView()), rotation: copy(this._jacRotation.GetView())};
  }
  geomDistanceInfo(data, a, b, maxDistance = 1) {
    // Canonicalize the call, including same-type ties, so reverse queries use
    // the same closest-point solution even when a face has multiple witnesses.
    // Return fromto[0:3] on a and fromto[3:6] on b in the caller's order.
    const types = this.model.geom_type;
    const flip = types[a] > types[b] || types[a] === types[b] && a > b;
    this._distancePoints.GetView().fill(0);
    const distance = this.mj.mj_geomDistance(this.model, data, flip ? b : a, flip ? a : b, maxDistance, this._distancePoints);
    const fromto = copy(this._distancePoints.GetView());
    const hasWitness = Number.isFinite(distance) && distance !== 0 && distance < maxDistance
      && fromto.every(Number.isFinite) && Math.hypot(fromto[0] - fromto[3], fromto[1] - fromto[4], fromto[2] - fromto[5]) > 1e-12;
    if (!hasWitness) fromto.fill(0);
    else if (flip) {
      const first = fromto.slice(0, 3); fromto.copyWithin(0, 3, 6); fromto.set(first, 3);
    }
    return {distance, fromto, hasWitness};
  }
  geomDistance(data, a, b, maxDistance = 1) {
    // Audits consume only the scalar signed distance. Keep exactly the same
    // canonical native query without copying and normalizing unused witnesses.
    const types = this.model.geom_type;
    const flip = types[a] > types[b] || types[a] === types[b] && a > b;
    this._distancePoints.GetView().fill(0);
    return this.mj.mj_geomDistance(this.model, data, flip ? b : a, flip ? a : b, maxDistance, this._distancePoints);
  }

  geomBounds(data, id, views = null) {
    const {box, matrix, pos} = views ?? {box: this.model.geom_aabb, matrix: data.geom_xmat, pos: data.geom_xpos};
    const lo = new Float64Array(3), hi = new Float64Array(3);
    for (let axis = 0; axis < 3; axis++) {
      let center = pos[3 * id + axis], radius = 0;
      for (let k = 0; k < 3; k++) {
        center += matrix[9 * id + 3 * axis + k] * box[6 * id + k];
        radius += Math.abs(matrix[9 * id + 3 * axis + k]) * box[6 * id + 3 + k];
      }
      lo[axis] = center - radius; hi[axis] = center + radius;
    }
    return {lower: lo, upper: hi};
  }

  geomBoundsQuery(data) {
    // Scope the WASM views to one synchronous audit, not to a physics frame.
    // A growing non-shared WASM heap detaches its previous views; reacquire
    // them before reading if a native query has grown that heap in between.
    const read = () => ({box: this.model.geom_aabb, matrix: data.geom_xmat, pos: data.geom_xpos});
    let views = read();
    return id => {
      if (!views.box.byteLength || !views.matrix.byteLength || !views.pos.byteLength) views = read();
      return this.geomBounds(data, id, views);
    };
  }

  contacts(data = this.data, {excludeFloor = false} = {}) {
    const rows = [];
    // Embind returns owning handles for both the vector and each copied contact.
    // Releasing only MjData leaves these allocations alive at every physics tick.
    if (!data.ncon) return rows;
    const vector = data.contact;
    try {
      for (let i = 0; i < data.ncon; i++) {
        const c = vector.get(i);
        try {
          if (excludeFloor && (c.geom1 === this.floorGeomId || c.geom2 === this.floorGeomId)) continue;
          this.mj.mj_contactForce(this.model, data, i, this._contactForce);
          rows.push({geom1: c.geom1, geom2: c.geom2, dist: c.dist,
            normalForce: this._contactForce.GetView()[0], position: Array.from(c.pos)});
        } finally { c.delete(); }
      }
    } finally { vector.delete(); }
    return rows;
  }

  _cacheTrayLandingParticipants() {
    // Cache original physical geometry before editor visibility can disable
    // object masks. This observer never changes collision participation.
    const m=this.model,meta=this.metadata;
    const physical=g=>Boolean((meta.originalGeomContype?.[g]??m.geom_contype[g])
      ||(meta.originalGeomConaffinity?.[g]??m.geom_conaffinity[g]));
    this._trayLandingObjectByGeom=new Map(Object.entries(this.objects).flatMap(([id,obj])=>
      obj.geomIds.filter(physical).map(g=>[g,id])));
    this._trayLandingTrayGeoms=new Set(this.trayGeomIds.filter(physical));
    this._trayLandingHandGeoms=new Set();
    const palms=new Set(Object.values(this.palmBodyIds));
    for(let g=0;g<m.ngeom;g++)if(physical(g)){
      for(let body=m.geom_bodyid[g];body>0;body=m.body_parentid[body]){
        if(palms.has(body)||/^(left|right)_(hand_|wrist_)/.test(this.bodyName(body))){
          this._trayLandingHandGeoms.add(g);break;
        }
      }
    }
  }

  _observeTrayLandings(contacts,time,recorded) {
    const objects=this._trayLandingObjectByGeom,hands=this._trayLandingHandGeoms,tray=this._trayLandingTrayGeoms;
    const excluded=new Set(),overlaps=new Map();
    for(const c of contacts){
      if(c.dist<=.0002){
        if(objects.has(c.geom1)&&hands.has(c.geom2))excluded.add(objects.get(c.geom1));
        if(objects.has(c.geom2)&&hands.has(c.geom1))excluded.add(objects.get(c.geom2));
      }
      // A wall contact cannot legitimize an object penetrating the bottom.
      if(c.dist<-.003){
        if(objects.has(c.geom1)&&c.geom2===this.trayBottomGeomId)excluded.add(objects.get(c.geom1));
        if(objects.has(c.geom2)&&c.geom1===this.trayBottomGeomId)excluded.add(objects.get(c.geom2));
      }
    }
    for(const c of contacts){
      const objectGeom=objects.has(c.geom1)&&tray.has(c.geom2)?c.geom1:
        objects.has(c.geom2)&&tray.has(c.geom1)?c.geom2:null;
      if(objectGeom===null)continue;
      const id=objects.get(objectGeom),p=c.position;
      if(recorded.has(id)||excluded.has(id)||!this.objects[id].active||!(c.normalForce>.02)||!(c.dist>=-.003)
        ||!finite(p,3)
        ||p[2]<this.tray.bottomTopZ-.003||p[2]>this.tray.wallTopZ+.002)continue;
      if(!overlaps.has(id))overlaps.set(id,objectTrayOverlap(this,id));
      const overlap=overlaps.get(id);if(!(overlap>1e-8))continue;
      // Contact can disappear again before the 20 ms control step ends.
      // Preserve the first interior, released contact for the controller.
      recorded.add(id);this.lastStepTrayLandings.push({object_id:id,time_s:time,
        contact_position_w:Array.from(p),object_geom:objectGeom,tray_geom:objectGeom===c.geom1?c.geom2:c.geom1,overlap_m2:overlap});
    }
  }

  setPolicyParameters(policy) {
    for (const [target, source] of [['kp', 'kp'], ['kd', 'kd'], ['effortLimit', 'effortLimit']]) {
      const value = policy[source] ?? policy[source === 'effortLimit' ? 'effort_limit' : source];
      if (!finite(value, 29)) throw new Error(`Policy ${source} must contain 29 finite values.`);
      this[target].set(value);
    }
  }

  setHandClosure(side, fraction, objectId = this.selectedObjectId) {
    if (!['left', 'right'].includes(side) || !Number.isFinite(fraction) || fraction < 0 || fraction > 1)
      throw new Error('Hand closure must be between 0 and 1.');
    this.handClosure[side] = fraction;
    // Dex3 uses firmer position gains while closing, within the exported
    // motor torque limits. Open fingers and other hands use the exported gains.
    this.handGripKpScale[side] = fraction > 0 && this.handModel === 'dex3' ? GRIP_KP_SCALE : 1;
    const start = side === 'left' ? 0 : this.fingersPerHand;
    for (let i = 0; i < this.fingersPerHand; i++) {
      const target = this.metadata.handOpen[side][i] * (1 - fraction) + this.metadata.handClosed[side][i] * fraction;
      this.fingerGoal[start + i] = clamp(target, ...this.metadata.fingerLimits[start + i]);
    }
  }

  step(bodyTarget29, fingerTargets14 = null) {
    if (!finite(bodyTarget29, 29)) throw new Error('Expected 29 finite body joint targets.');
    if (fingerTargets14 !== null) {
      if (!finite(fingerTargets14, this.fingerCount)) throw new Error(`Expected ${this.fingerCount} finite finger targets.`);
      for (let i = 0; i < this.fingerCount; i++) this.fingerGoal[i] = clamp(fingerTargets14[i], ...this.metadata.fingerLimits[i]);
    }
    const d = this.data, meta = this.metadata, collected = new Map(), landed = new Set();
    this.lastStepTrayLandings = [];
    this._imuRotation.GetView().set([1,0,0,0]);
    let previousGyro = Array.from(d.qvel.subarray(3,6));
    for (let substep = 0; substep < Math.round(this.controlDt / this.physicsDt); substep++) {
      // Refresh WASM-backed views on every substep; heap growth can invalidate a cached view.
      const q = d.qpos, v = d.qvel, forces = d.qfrc_applied;
      forces.fill(0);
      for (let i = 0; i < 29; i++) {
        const torque = clamp(this.kp[i] * (bodyTarget29[i] - q[this.jointQposAddresses29[i]])
          - this.kd[i] * v[this.jointDofAddresses29[i]], -this.effortLimit[i], this.effortLimit[i]);
        forces[this.jointDofAddresses29[i]] = this.lastTorque[i] = torque;
      }
      for (let i = 0; i < this.fingerCount; i++) {
        this.fingerTarget[i] += clamp(this.fingerGoal[i] - this.fingerTarget[i], -2 * this.physicsDt, 2 * this.physicsDt);
        const side = i < this.fingersPerHand ? 'left' : 'right', j = i % this.fingersPerHand, cap = meta.fingerEffortLimit[i];
        const gripKp = meta.fingerKp[i] * this.handGripKpScale[side], error = this.fingerTarget[i] - q[this.fingerQposAddresses14[i]];
        let torque = clamp(gripKp * error - meta.fingerKd[i] * v[this.fingerDofAddresses14[i]], -cap, cap);
        if (this.handModel === 'dex3' && this.handClosure[side] >= 1 && this.flexionFinger[i]) {
          // Guaranteed squeeze on a BLOCKED flexion joint only; a joint that reached its target keeps pure PD.
          const dir = Math.sign(meta.handClosed[side][j] - meta.handOpen[side][j]);
          if (dir !== 0 && error * dir > GRIP_BLOCKED_ERROR_RAD && torque * dir < GRIP_SQUEEZE_FRACTION * cap) torque = dir * GRIP_SQUEEZE_FRACTION * cap;
        }
        forces[this.fingerDofAddresses14[i]] = this.lastFingerTorque[i] = torque;
      }
      this.mj.mj_step(this.model, d);
      const gyro = Array.from(d.qvel.subarray(3,6));
      // Same trapezoidal 1 kHz gyro increment as the native leg-odometry
      // adapter. Integrating only the 50 Hz endpoints aliases contact sway.
      this.mj.mju_quatIntegrate(this._imuRotation, gyro.map((v,i)=>(previousGyro[i]+v)/2), this.physicsDt);
      previousGyro = gyro;
      const presentPairs = new Set(), contacts = this.contacts(d, {excludeFloor: true});
      this._observeTrayLandings(contacts,d.time,landed);
      for (const contact of contacts) {
        const key = `${contact.geom1}:${contact.geom2}`, old = collected.get(key);
        if (!old) collected.set(key, {...contact, duration: this.physicsDt});
        else {
          old.normalForce = Math.max(old.normalForce, contact.normalForce); old.dist = Math.min(old.dist, contact.dist);
          if (!presentPairs.has(key)) old.duration += this.physicsDt;
        }
        presentPairs.add(key);
      }
    }
    this.lastStepContacts = [...collected.values()];
    if (!Array.from(d.qpos).every(Number.isFinite)) throw new Error('Non-finite MuJoCo state; execution stopped.');
    this.imuDeltaQuatB.set(this._imuRotation.GetView()); this.imuStepCount++;
    this._updateLegOdometry();
    return this.lastTorque;
  }

  _updateLegOdometry() {
    if (!this.legOdometry) return;
    const d=this.data;
    this.legOdometry.update({dofPos:pick(d.qpos,this.jointQposAddresses29),
      rootQuatW:copy(d.qpos.subarray(3,7)),rootAngVelB:copy(d.qvel.subarray(3,6)),
      imuDeltaQuatB:this.imuDeltaQuatB,imuStepCount:this.imuStepCount});
  }

  _resetLegOdometry() {
    if (!this.legOdometry) return;
    const d=this.data;
    // Only episode alignment uses the initial world pose. Every later
    // estimate uses encoders, IMU and foot contacts, never truth translation.
    this.legOdometry.reset({rootPosW:copy(d.qpos.subarray(0,3)),rootQuatW:copy(d.qpos.subarray(3,7)),
      rootLinVelW:copy(d.qvel.subarray(0,3)),dofPos:pick(d.qpos,this.jointQposAddresses29),
      rootAngVelB:copy(d.qvel.subarray(3,6)),imuStepCount:this.imuStepCount});
  }

  objectPose(id = this.activeObjectId, data = this.data) {
    const object = this.objects[id];
    if (!object) return {positionW: new Float64Array(3), quaternionW: new Float64Array([1, 0, 0, 0]), hasObject: false};
    return {positionW: copy(data.xpos.subarray(3 * object.bodyId, 3 * object.bodyId + 3)),
      quaternionW: copy(data.xquat.subarray(4 * object.bodyId, 4 * object.bodyId + 4)), hasObject: object.active};
  }

  readState(objectId = this.activeObjectId) {
    const d = this.data, palmPosW = [], palmQuatW = [];
    for (const side of ['left', 'right']) {
      const id = this.palmBodyIds[side];
      palmPosW.push(copy(d.xpos.subarray(3 * id, 3 * id + 3)));
      palmQuatW.push(copy(d.xquat.subarray(4 * id, 4 * id + 4)));
    }
    return {time: d.time, rootPosW: copy(d.qpos.subarray(0, 3)), rootQuatW: copy(d.qpos.subarray(3, 7)),
      rootLinVelW: copy(d.qvel.subarray(0, 3)), rootAngVelB: copy(d.qvel.subarray(3, 6)),
      dofPos: pick(d.qpos, this.jointQposAddresses29), dofVel: pick(d.qvel, this.jointDofAddresses29),
      palmPosW, palmQuatW, object: this.objectPose(objectId),
      odom:this.legOdometry?structuredClone(this.legOdometry.state):null};
  }

  _setObject(id, pose) {
    const obj = this.objects[id], m = this.model, d = this.data, active = pose !== null;
    obj.active = active;
    for (const g of obj.geomIds) {
      m.geom_contype[g] = active ? this.metadata.originalGeomContype[g] : 0;
      m.geom_conaffinity[g] = active ? this.metadata.originalGeomConaffinity[g] : 0;
      for (let k = 0; k < 4; k++) m.geom_rgba[g * 4 + k] = k === 3 && !active ? 0 : this.metadata.originalGeomRGBA[g * 4 + k];
    }
    m.body_gravcomp[obj.bodyId] = active ? 0 : 1;
    const [x, y, yaw] = pose ?? [0, 0, 0];
    d.qpos.set([x, y, active ? obj.restZ : -5, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], obj.qadr);
    d.qvel.fill(0, obj.vadr, obj.vadr + 6);
  }

  validatePlacement(id, x, y, yaw = 0, {resetRobot = false} = {}) {
    const info = this.catalog.objects.find(o => o.id === id);
    if (!info || ![x, y, yaw].every(Number.isFinite)) throw new Error('Invalid object or placement.');
    const region = this.catalog.placement_region;
    if (!containsPlacement(region,x,y))
      throw new Error('Place the object inside the highlighted workspace.');
    let local;
    if (['apple', 'can', 'bottle'].includes(id)) local = Array.from({length: 32}, (_, i) => [info.footprint_radius * Math.cos(i * Math.PI / 16), info.footprint_radius * Math.sin(i * Math.PI / 16)]);
    else if (id === 'mug') local = [[-.038, -.038], [.072, -.038], [.072, .038], [-.038, .038]];
    else { const [sx, sy] = info.size; local = [[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]]; }
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const points = local.map(([a, b]) => [x + a * c - b * s, y + a * s + b * c]);
    const table = this.metadata.table;
    for (const p of points) {
      const dx = p[0] - table.center[0], dy = p[1] - table.center[1];
      if (table.shape === 'circle' ? Math.hypot(dx, dy) > table.radius - .006
        : Math.abs(dx) > table.half_size[0] - .006 || Math.abs(dy) > table.half_size[1] - .006)
        throw new Error('The entire object must fit on the tabletop.');
    }
    const b = this.tray.outerBounds;
    const tray = [[b.x_min, b.y_min], [b.x_max, b.y_min], [b.x_max, b.y_max], [b.x_min, b.y_max]];
    let separated = false;
    for (const polygon of [points, tray]) for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i], next = polygon[(i + 1) % polygon.length], length = Math.hypot(next[0] - a[0], next[1] - a[1]);
      const axis = [-(next[1] - a[1]) / length, (next[0] - a[0]) / length];
      const pa = points.map(p => p[0] * axis[0] + p[1] * axis[1]), pb = tray.map(p => p[0] * axis[0] + p[1] * axis[1]);
      if (Math.max(...pa) + .003 < Math.min(...pb) || Math.max(...pb) + .003 < Math.min(...pa)) separated = true;
    }
    if (!separated) throw new Error('Keep the object outside the drop tray.');
    const scratch = this._placementScratch, obj = this.objects[id];
    scratch.qpos.set(resetRobot ? this.metadata.initialQpos : this.data.qpos);
    scratch.qvel.set(resetRobot ? this.metadata.initialQvel : this.data.qvel);
    if (resetRobot) {
      // Validate against the robot posture that will exist after the edit,
      // while every other object remains at its current physical pose.
      for (const object of Object.values(this.objects)) {
        scratch.qpos.set(this.data.qpos.subarray(object.qadr, object.qadr + 7), object.qadr);
        scratch.qvel.set(this.data.qvel.subarray(object.vadr, object.vadr + 6), object.vadr);
      }
      const open = [...this.metadata.handOpen.left, ...this.metadata.handOpen.right];
      this.fingerQposAddresses14.forEach((address, i) => {
        scratch.qpos[address] = clamp(open[i], ...this.metadata.fingerLimits[i]);
      });
    }
    scratch.qpos.set([x, y, obj.restZ, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], obj.qadr);
    const old = obj.geomIds.map(g => [this.model.geom_contype[g], this.model.geom_conaffinity[g]]);
    try {
      obj.geomIds.forEach(g => { this.model.geom_contype[g] = this.metadata.originalGeomContype[g]; this.model.geom_conaffinity[g] = this.metadata.originalGeomConaffinity[g]; });
      this.forward(scratch);
      // Use the other objects' measured 3D poses, including objects that have
      // fallen over or moved into the tray since their initial placement.
      const physicalGeoms = object => object.geomIds.filter(g =>
        this.metadata.originalGeomContype[g] || this.metadata.originalGeomConaffinity[g]);
      for (const [otherId, other] of Object.entries(this.objects)) {
        if (otherId === id || !other.active) continue;
        for (const a of physicalGeoms(obj)) for (const b of physicalGeoms(other))
          if (this.geomDistance(scratch, a, b, .01) < 0)
            throw new Error('The object overlaps another object. Choose a clear spot on the table.');
      }
      for (const contact of this.contacts(scratch)) {
        if (contact.dist >= -.0005) continue;
        const other = obj.geomIds.includes(contact.geom1) ? contact.geom2 : obj.geomIds.includes(contact.geom2) ? contact.geom1 : null;
        if (other === null) continue;
        const body = this.bodyName(this.model.geom_bodyid[other]);
        if (body !== 'world' && !body.startsWith('demo_')) throw new Error('The object intersects the robot.');
      }
    } finally { obj.geomIds.forEach((g, i) => { this.model.geom_contype[g] = old[i][0]; this.model.geom_conaffinity[g] = old[i][1]; }); }
    return true;
  }

  placeObject(id, x, y, yaw = 0, {resetRobot = false} = {}) {
    this.validatePlacement(id, x, y, yaw, {resetRobot});
    if (resetRobot) this.resetRobot();
    this._setObject(id, [x, y, yaw]);
    this._placements[id] = [x, y, yaw]; this._selectedObjectId = id;
    this.forward(); return this.uiState();
  }
  selectObject(id) {
    if (!this.objects[id]?.active) throw new Error('Select an object that is already in the scene.');
    this._selectedObjectId = id; return this.uiState();
  }
  removeObject(id) {
    if (!this.objects[id]) throw new Error('Unknown object.');
    delete this._placements[id]; this._setObject(id, null);
    if (this.selectedObjectId === id) this._selectedObjectId = Object.keys(this._placements)[0] ?? null;
    this.forward(); return this.uiState();
  }
  clearObjects() {
    for (const id of Object.keys(this.objects)) this._setObject(id, null);
    this._placements = {}; this._selectedObjectId = null; this.forward(); return this.uiState();
  }
  _resetRobotData() {
    this.mj.mj_resetData(this.model, this.data);
    this.data.qpos.set(this.metadata.initialQpos); this.data.qvel.set(this.metadata.initialQvel);
    for (const side of ['left', 'right']) this.setHandClosure(side, 0);
    this.fingerTarget.set(this.fingerGoal);
    this.fingerQposAddresses14.forEach((address, i) => { this.data.qpos[address] = this.fingerGoal[i]; });
    this.lastTorque.fill(0); this.lastFingerTorque.fill(0); this.lastStepContacts = []; this.lastStepTrayLandings = [];
    this.imuDeltaQuatB.set([1,0,0,0]); this.imuStepCount=0;
  }
  reset(placements = this._placements) {
    this._resetRobotData();
    this._placements = structuredClone(placements);
    for (const id of Object.keys(this.objects)) this._setObject(id, this._placements[id] ?? null);
    if (!this.objects[this.selectedObjectId]?.active) this._selectedObjectId = Object.keys(this._placements)[0] ?? null;
    this.forward(); this._resetLegOdometry(); return this.uiState();
  }
  resetRobot() {
    // Starting a new attempt must not return deposited objects to the table.
    // Save all free-object state before resetting robot dynamics and controls;
    // leave placements, selection, visibility and collision masks untouched.
    const states = Object.values(this.objects).map(obj => ({obj,
      qpos: copy(this.data.qpos.subarray(obj.qadr, obj.qadr + 7)),
      qvel: copy(this.data.qvel.subarray(obj.vadr, obj.vadr + 6))}));
    this._resetRobotData();
    for (const {obj, qpos, qvel} of states) {
      this.data.qpos.set(qpos, obj.qadr); this.data.qvel.set(qvel, obj.vadr);
    }
    this.forward(); this._resetLegOdometry(); return this.uiState();
  }

  uiState() {
    const objects = Object.entries(this.objects).filter(([, o]) => o.active).map(([id]) => {
      const pose = this.objectPose(id), [w, x, y, z] = pose.quaternionW;
      return {...this.catalog.objects.find(o => o.id === id), active: true, position: Array.from(pose.positionW),
        quaternion: [x, y, z, w], yaw: Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))};
    });
    return {table: this.metadata.table, tray: this.metadata.tray, objects,
      selectedObjectId: this.selectedObjectId, activeObjectId: this.activeObjectId,
      robot: {position: Array.from(this.data.qpos.subarray(0, 3)), joint_positions: Array.from(pick(this.data.qpos, this.jointQposAddresses29)), setback_m: this.catalog.robot_setback_m},
      time: this.data.time, hand_closure: {...this.handClosure}, placement_region: this.catalog.placement_region,
      hand_workspaces: this.catalog.hand_workspaces, ego_camera: this.metadata.egoCamera};
  }

  renderDescription() {
    if (this._renderDescription) return this._renderDescription;
    const m = this.model, visualBodies = new Set();
    for (let g = 0; g < m.ngeom; g++) if (!this.metadata.originalGeomContype[g] && !this.metadata.originalGeomConaffinity[g]) visualBodies.add(m.geom_bodyid[g]);
    const geoms = Array.from({length: m.ngeom}, (_, id) => {
      const bodyId = m.geom_bodyid[id], name = this.bodyName(bodyId);
      const collision = this.metadata.originalGeomContype[id] || this.metadata.originalGeomConaffinity[id];
      return {id, bodyId, name: this.geomName(id), bodyName: name, type: m.geom_type[id],
        size: Array.from(m.geom_size.subarray(id * 3, id * 3 + 3)), meshId: m.geom_dataid[id],
        materialId: m.geom_matid[id], rgba: this.metadata.originalGeomRGBA.slice(id * 4, id * 4 + 4),
        visible: !(bodyId !== 0 && !name.startsWith('demo_') && collision && visualBodies.has(bodyId))};
    });
    const meshes = Array.from({length: m.nmesh}, (_, id) => {
      const mesh = {
        positions: new Float32Array(m.mesh_vert.subarray(3 * m.mesh_vertadr[id], 3 * (m.mesh_vertadr[id] + m.mesh_vertnum[id]))),
        indices: new Uint32Array(m.mesh_face.subarray(3 * m.mesh_faceadr[id], 3 * (m.mesh_faceadr[id] + m.mesh_facenum[id]))),
      };
      if (m.mesh_texcoordnum?.[id] > 0) {
        // MuJoCo indexes texture coordinates separately from vertices, so unroll the faces for a per-corner UV layout.
        const corners = 3 * m.mesh_facenum[id], faceAdr = 3 * m.mesh_faceadr[id], vertAdr = m.mesh_vertadr[id], texAdr = m.mesh_texcoordadr[id];
        const positions = new Float32Array(3 * corners), uv = new Float32Array(2 * corners);
        for (let k = 0; k < corners; k++) {
          const v = 3 * (vertAdr + m.mesh_face[faceAdr + k]), t = 2 * (texAdr + m.mesh_facetexcoord[faceAdr + k]);
          positions[3 * k] = m.mesh_vert[v]; positions[3 * k + 1] = m.mesh_vert[v + 1]; positions[3 * k + 2] = m.mesh_vert[v + 2];
          uv[2 * k] = m.mesh_texcoord[t]; uv[2 * k + 1] = m.mesh_texcoord[t + 1];
        }
        mesh.unrolledPositions = positions; mesh.uv = uv;
      }
      return mesh;
    });
    const textureStride = m.mat_texid.length / m.nmat;
    const materials = Array.from({length: m.nmat}, (_, id) => ({rgba: Array.from(m.mat_rgba.subarray(id * 4, id * 4 + 4)),
      specular: m.mat_specular[id], shininess: m.mat_shininess[id], reflectance: m.mat_reflectance[id],
      textureId: m.mat_texid[id * textureStride + 1], textureRepeat: Array.from(m.mat_texrepeat.subarray(id * 2, id * 2 + 2))}));
    const textures = Array.from({length: m.ntex}, (_, id) => {
      const offset = Number(m.tex_adr[id]); // Official bindings expose texture addresses as int64.
      return {type: m.tex_type[id], width: m.tex_width[id], height: m.tex_height[id], channels: m.tex_nchannel[id],
        data: new Uint8Array(m.tex_data.subarray(offset, offset + m.tex_width[id] * m.tex_height[id] * m.tex_nchannel[id]))};
    });
    this._renderDescription = {format: 'tabletop_render_v1', geoms, meshes, materials, textures,
      palmBodyIds: this.palmBodyIds, bodyNames: this.metadata.bodyNames, egoCameraId: this.egoCameraId,
      egoFovy: this.metadata.egoCamera.fovy, table: this.metadata.table,
      homeCamera: {lookat: [.38, 0, .66], distance: 3.1, azimuth: 62, elevation: -26, fovy: 45}};
    return this._renderDescription;
  }
  exportRenderModel() { return this.renderDescription(); }
  renderSnapshot() {
    const d = this.data;
    return {time: d.time, geomPositions: new Float32Array(d.geom_xpos), geomMatrices: new Float32Array(d.geom_xmat),
      geomRGBA: new Float32Array(this.model.geom_rgba), bodyPositions: new Float32Array(d.xpos),
      bodyQuaternions: new Float32Array(d.xquat), cameraPositions: new Float32Array(d.cam_xpos),
      cameraMatrices: new Float32Array(d.cam_xmat), activeObjectId: this.activeObjectId};
  }
  dispose() {
    this.legOdometry?.dispose(); this._imuRotation.delete();
    for (const buffer of [this._jacPosition, this._jacRotation, this._distancePoints, this._contactForce]) buffer.delete();
    this._placementScratch.delete(); this.data.delete(); this.model.delete();
  }
}
