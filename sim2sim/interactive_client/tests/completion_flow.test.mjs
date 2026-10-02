import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {runInNewContext} from 'node:vm';
import {InteractiveController, ApproachObstructionMonitor} from '../controller.js';
import {clampPlacement, placementAxisRange, placementPolygon} from '../placement_workspace.js';
import {IKError, WholeBodyIK} from '../ik.js';
import {eye3, norm, sub} from '../numerics.js';

const near = (actual, expected, tolerance = 1e-10) =>
  assert.ok(norm(sub(actual, expected)) <= tolerance, `${actual} != ${expected}`);

function fixture({phase = 'approach', distance = .065, elapsed = 4} = {}) {
  const palm = {position: [.4, -.2, .86], rotation: eye3()};
  const goal = [palm.position[0] + distance, -.2, .86];
  const current = {position: [.43, -.2, .86], rotation: eye3()};
  const state = {rootPosW: [0, 0, .76], rootQuatW: [1, 0, 0, 0],
    rootLinVelW: [0, 0, 0], rootAngVelB: [0, 0, 0], dofPos: Array(29).fill(0), dofVel: Array(29).fill(0)};
  const object = {positionW: [.50, -.2, .81], quaternionW: [1, 0, 0, 0]};
  const closures = {};
  const scene = {
    data: {time: elapsed, qpos: new Float64Array([0, 0, .76, 1, 0, 0, 0]), qvel: new Float64Array(6),
      geom_xpos: new Float64Array([.65, 0, .69]), geom_xmat: new Float64Array(eye3())},
    model: {geom_type: [6], geom_size: [.35, .6, .05], geom_aabb: [0, 0, 0, .35, .6, .05]},
    objects: {apple: {active: true, geomIds: [], vadr: 0}}, tableGeomIds: [0], tableTopZ: .74,
    tray: {center: [.47, .08], wallTopZ: .83, bottomTopZ: .755,
      innerBounds: {x_min: .38, x_max: .56, y_min: -.02, y_max: .18}},
    objectPose: () => object, readState: () => state,
    setHandClosure: (side, value) => { closures[side] = value; }, step: () => {},
  };
  const c = Object.assign(Object.create(InteractiveController.prototype), {
    scene, dt: .02, phase, stage: 'settle', startTime: 0, phaseTime: 0,
    hand: 'right', objectId: 'apple', mode: 'hero_plus', plan: {approach: 'side', yawDeg: 45},
    segments: [{name: 'settle', position: current.position.slice(), rotation: eye3(), duration: .4}],
    segmentIndex: 0, segmentTime: 0, segmentFrom: structuredClone(current),
    graspGoal: goal, grasp: current.position.slice(), graspRotation: eye3(),
    pregrasp: [.31, -.2, .86], liftTarget: [.415, -.2, .99], objectAnchor: object.positionW.slice(),
    closeProgress: 0, closeDuration: 1.8, graspReadyTime: 0, graspWaitBest: distance,
    graspProgressTime: 0, fingerContactCount: 0, gripContactTime: 0,
    contacts: 0, handContacts: 0, trayContacts: 0, stableLift: 0, stableDeposit: 0,
    lift: 0, maxLift: 0, initialObjectZ: .81, inTray: false, trayOverlap: 0,
    success: null, graspSuccess: null, depositSuccess: null, returnSuccess: null, everPlaced: false, trayLanding: null,
    motionCompleted: false, completionFallback: false, forcedClose: false,
    forcedCloseTrigger: null, forcedCloseTarget: null, completionTransfer: false,
    approachObstruction: new ApproachObstructionMonitor(), graspObstructionCandidate: null,
    objectLocked: true, goalAdjustment: [0, 0, 0], goalAdjuster: null, gaGraspTransition: null,
    replanEnabled: true, goalAdjustEnabled: true, nextReplan: 0, replans: 0, gaUpdates: 0,
    restQ: Array(29).fill(0), homeQ: Array(29).fill(0), rootPosition: [0, 0, .76],
    rootHeight: .76, rootPitch: 0, crouched: false, depositEnabled: true,
    homePalms: {left: {position: [.1, .2, .9], rotation: eye3()}, right: structuredClone(palm)},
    bodyFixtureForce: 0, bodyFixturePenetration: 0,
    audit: {selfPairs: [], handTableClearance: () => .05,
      handBounds: () => ({lower: [.35, -.25, .85], upper: [.48, -.15, .95]}),
      check: () => ({passed: true, minimumClearance: .05}), hands: {left: [], right: []}},
    reference: {qCurrent: Array(29).fill(0), currentIndex: 0, lastAudit: null,
      currentFrame: () => ({rootPosW: [0, 0, .76], plannedRootPitch: 0}),
      window: () => ({}), ik: {anchorPosition: [0, 0, .76], anchorRotation: eye3()}},
    palm: () => structuredClone(palm),
    plannedPalms() { return {left: structuredClone(this.homePalms.left), right: structuredClone(current)}; },
    oneHandTarget(position, rotation, options = {}) { return {palms: {right: {position, rotation}}, ...options}; },
    baseTarget: (palms, options) => ({palms, ...options}),
    objectBounds: () => ({lower: [.47, -.23, .75], upper: [.53, -.17, .87]}),
    objectTrayOverlap: () => 0, measure: () => {},
    policy: {control: async () => Array(29).fill(0)},
  });
  return {c, palm, current, object, state, closures};
}

test('unconverged precision approach receives its normal retry interval before forced closure', () => {
  const {c} = fixture({elapsed: 1});
  c.advanceApproach();
  assert.equal(c.phase, 'approach');
  assert.equal(c.forcedClose, false);
  assert.equal(c.closeProgress, 0);
  assert.equal(c.success, null);
});

test('exhausted far approach holds its current reference with open fingers for one bounded retry', () => {
  const {c} = fixture();
  const held = structuredClone(c.approachTarget());
  assert.equal(c.graspAlignment().ready, false);
  c.advanceApproach();
  assert.equal(c.phase, 'approach');
  assert.equal(c.forcedClose, false);
  assert.ok(c.closureRejectedTrigger.distance_m > .03);
  assert.equal(c.alignmentRetryCount, 1);
  assert.equal(c.closeProgress, 0);
  for (const offset of [0, .4, 10]) near(c.targetAt(offset).palms.right.position, held.palms.right.position);
  c.grasp[0] += .15;
  c.segments[0].position[2] += .1;
  c.closedLoopUpdate();
  near(c.targetAt(10).palms.right.position, held.palms.right.position);
  assert.equal(c.graspSuccess, null, 'Trying the closure must not certify physical grasp success.');
  const status = c.snapshot();
  assert.equal(status.alignment_retry.count, 1);
  assert.equal(status.motion_completed, false);
  assert.deepEqual(status.closure_rejected_trigger, c.closureRejectedTrigger);
});

test('an authorized near forced closure attempts lift without inventing physical contact', () => {
  const {c, object} = fixture({distance: .025});
  const originalObjectPose = structuredClone(object);
  c.beginForcedClose('test tracking timeout');
  const held = c.targetAt().palms.right.position.slice();
  for (let i = 0; i < 180 && c.phase === 'close'; i++) {
    c.scene.data.time += c.dt;
    c.advanceClose();
  }
  assert.equal(c.phase, 'lift');
  near([c.closeProgress], [1]);
  near(c.targetAt().palms.right.position, held);
  assert.equal(c.fingerContactCount, 0);
  assert.equal(c.graspSuccess, null);
  assert.equal(c.success, null);
  assert.deepEqual(object, originalObjectPose, 'The fallback must leave the object under physical simulation.');
});

test('timed-out aligned close uses forced closure instead of declaring a grip that does not exist', () => {
  const {c} = fixture({distance: .01, elapsed: 0});
  c.beginClose(c.graspAlignment());
  c.closeProgress = 1;
  c.scene.data.time = c.phaseTime + c.closeDuration + 6.02;
  c.advanceClose();
  assert.equal(c.phase, 'close');
  assert.equal(c.forcedClose, true);
  assert.equal(c.graspSuccess, null);
  c.scene.data.time += 2;
  c.advanceClose();
  assert.equal(c.phase, 'lift');
  assert.equal(c.graspSuccess, null);
});

test('empty completion transport reaches release without waiting for the object to leave the table', () => {
  const {c, palm, object} = fixture({phase: 'hold'});
  c.graspSuccess = false;
  c.audit.handBounds = () => ({lower: [palm.position[0] - .05, palm.position[1] - .05, palm.position[2] - .01],
    upper: [palm.position[0] + .08, palm.position[1] + .05, palm.position[2] + .05]});
  const originalObjectPose = structuredClone(object);
  c.beginCompletionTransfer('empty grasp');
  assert.equal(c.phase, 'transfer');
  assert.equal(c.completionTransfer, true);
  assert.deepEqual(c.transferSegments.map(segment => segment.name), ['raise_load', 'over_tray', 'lower_load']);
  const visited = [];
  c.beginRelease = function () { this.enter('release', 'Opening the fingers.'); };
  for (let i = 0; i < 600 && c.phase === 'transfer'; i++) {
    visited.push(c.transferSegments[c.transferIndex].name);
    c.scene.data.time += c.dt;
    palm.position = c.transferTarget().palms.right.position.slice();
    c.advanceTransfer();
  }
  assert.equal(c.phase, 'release');
  assert.deepEqual([...new Set(visited)], ['raise_load', 'over_tray', 'lower_load']);
  assert.equal(c.graspSuccess, false);
  assert.notEqual(c.depositSuccess, true);
  assert.deepEqual(object, originalObjectPose);
});

function confirmNearObstruction(c) {
  for (const time of [0, .2, .4, .6, .8, 1]) c.approachObstruction.update({time, eligible: true,
    referenceDistance: .06, requestedError: .05,
    constraints: [{object_id: 'uiuc_i', clearance_m: .007, required_clearance_m: .005}]});
  assert.equal(c.approachObstruction.confirmed?.code, 'approach_obstructed');
}

test('an obstructed near closure attempts the lift, then fails before any empty transfer or policy step', async () => {
  for (const normal of [false, true]) {
    const {c, object} = fixture({distance: normal ? .01 : .025, elapsed: 0});
    const originalObject = structuredClone(object);
    confirmNearObstruction(c);
    if (normal) c.beginClose(c.graspAlignment());
    else c.beginForcedClose('alignment_timeout');
    assert.equal(c.phase, 'close'); assert.equal(c.success, null);
    assert.equal(c.graspObstructionCandidate.phase, 'approach');
    if (normal) {
      c.scene.data.time = c.phaseTime + c.closeDuration + 6.02;
      c.advanceClose(); assert.equal(c.forcedClose, true);
    }
    for (let i = 0; i < 180 && c.phase === 'close'; i++) {
      c.scene.data.time += c.dt; c.advanceClose();
    }
    assert.equal(c.phase, 'lift', 'A near hand must receive its real physical grasp/lift attempt.');
    assert.equal(c.graspSuccess, null);
    c.enter('hold', 'Checking that the object remains lifted.');
    c.scene.data.time = c.phaseTime + 1.21;
    c.policy.control = () => { throw new Error('Blocked unsuccessful lift must not step the policy again.'); };
    c.beginDeposit = () => { throw new Error('Blocked unsuccessful lift must not create transfer segments.'); };
    await c.tick();
    assert.equal(c.phase, 'failed'); assert.equal(c.busy, false);
    assert.equal(c.graspSuccess, false); assert.equal(c.motionCompleted, false);
    assert.equal(c.completionTransfer, false); assert.equal(c.transferSegments, undefined);
    const reason = c.snapshot().failure_reason;
    assert.equal(reason.code, 'approach_obstructed'); assert.equal(reason.reset_required, true);
    assert.equal(reason.phase, 'approach'); assert.equal(reason.stage, 'settle');
    assert.equal(reason.failure_phase, 'hold'); assert.equal(reason.grasp_validation, 'not_secured');
    assert.deepEqual(object, originalObject, 'Failure reporting cannot attach or move the object.');
  }
});

test('real sustained lift measurements clear approach blame permanently before later contact loss', () => {
  const {c, object} = fixture({distance: .025, elapsed: 0});
  confirmNearObstruction(c); c.beginForcedClose('alignment_timeout');
  c.enter('lift', 'Lifting.');
  let contacts = [], bottom = .74;
  Object.assign(c, {recentContacts: [], objectTrayOverlap: () => 0, measure: InteractiveController.prototype.measure,
    objectBounds: InteractiveController.prototype.objectBounds});
  Object.assign(c.scene, {objects: {apple: {geomIds: [10], vadr: 0}}, contacts: () => contacts,
    model: {geom_bodyid: Array.from({length: 12}, (_, i) => i), geom_contype: Array(12).fill(1), geom_conaffinity: Array(12).fill(1)},
    bodyName: id => ({1: 'right_hand_index_1_link', 2: 'right_hand_middle_1_link', 10: 'demo_apple'}[id] ?? 'world'),
    geomBounds: () => ({lower: [.47, -.23, bottom], upper: [.53, -.17, object.positionW[2] + .06]}),
    trayGeomIds: [], trayBottomGeomId: 11});
  Object.assign(c.audit, {fingers: new Map([[1, {side: 'right', finger: 'index'}], [2, {side: 'right', finger: 'middle'}]]),
    fixtureGeoms: [], robotGeoms: [1, 2], hands: {left: [], right: [1, 2]}, handSet: new Set([1, 2])});
  const touch = geom => ({geom1: 10, geom2: geom, normalForce: .1, dist: 0});
  const step = () => {c.scene.data.time += c.dt; c.measure();};
  contacts = [touch(1), touch(2), {...touch(0), normalForce: 2}]; object.positionW[2] += .03;
  for (let i = 0; i < 50; i++) step();
  assert.equal(c.stableLift, 0); assert.ok(c.graspObstructionCandidate,
    'Finger contact and a raised COM cannot certify a tilted object still supported by the table.');
  contacts = [touch(1), touch(2)]; bottom = c.scene.tableTopZ + .00065;
  for (let i = 0; i < 39; i++) step();
  assert.ok(c.stableLift < .8); assert.ok(c.graspObstructionCandidate);
  step(); assert.ok(c.stableLift >= .8); assert.equal(c.graspObstructionCandidate, null);
  near([c.lift], [.03]);
  assert.equal(c.snapshot().grasp_support.held_away_from_support, true);
  contacts = []; for (let i = 0; i < 10; i++) step();
  assert.equal(c.stableLift, 0); assert.equal(c.graspSuccess, null);
  assert.equal(c.failUnsecuredObstructedGrasp(), false);
  assert.equal(c.snapshot().failure_reason, null, 'A later slip must not revive obsolete approach blockage.');
});

test('grasp evidence from another selected object cannot terminate the current completion', () => {
  const {c} = fixture({phase: 'hold'});
  c.graspObstructionCandidate = {code: 'approach_obstructed', target_object_id: 'mug', objects: [{object_id: 'uiuc_i'}]};
  c.beginCompletionTransfer('grasp_not_secured');
  assert.equal(c.phase, 'transfer'); assert.equal(c.snapshot().failure_reason, null);
});

test('completion transport opens instead of moving sideways when the hand cannot clear the rim', () => {
  const {c} = fixture({phase: 'hold'});
  c.audit.handBounds = () => ({lower: [.35, -.25, .82], upper: [.48, -.15, .9]});
  c.beginCompletionTransfer('empty grasp');
  c.beginRelease = function () { this.enter('release', 'Opening the fingers.'); };
  c.scene.data.time = c.transferTime + c.transferSegments[0].duration + 3.02;
  c.advanceTransfer();
  assert.equal(c.phase, 'release');
  assert.equal(c.transferIndex, 0, 'Insufficient vertical clearance must never trigger sideways transport.');
});

test('repeated successful rim recoveries cannot reset the transfer stage budget forever', () => {
  const {c} = fixture({phase: 'hold', elapsed: 0});
  c.stableLift = 1; c.lift = .1; c.beginDeposit();
  c.transferIndex = 1; c.transferTime = 0; c.startTransferStage();
  // Reproduce repeated dip, successful vertical recovery, renewed sideways
  // interpolation, dip again. Each individual recovery clears under 3 s.
  for (let i = 0; i < 800 && !c.completionTransfer; i++) {
    const bottom = c.clearanceRecovery ? .88 : .84;
    c.objectBounds = () => ({lower: [.47, -.23, bottom], upper: [.53, -.17, bottom + .12]});
    c.advanceTransfer(); c.scene.data.time += .02;
  }
  assert.equal(c.completionTransfer, true);
  assert.ok(c.completionReasons.includes('transfer_stage_timeout'));
  assert.ok(c.scene.data.time < 14.7);
  assert.equal(c.transferBudgetStarted, 0, 'The single fallback retains the attempt budget.');
  assert.equal(c.snapshot().failure_reason, null, 'Rim recovery alone is not proof of another object blocking IK.');
});

test('persistent transfer IK obstruction ends with reset guidance without empty release or a policy step', async () => {
  const {c, object} = fixture({phase: 'hold', elapsed: 0});
  c.beginDeposit(); c.transferIndex = 1; c.transferTime = 0; c.startTransferStage();
  c.stableLift = 1; c.lift = .1;
  c.objectBounds = () => ({lower: [.47, -.23, .9], upper: [.53, -.17, 1.02]});
  c.transferObstruction = new ApproachObstructionMonitor({dwell: 3, code: 'transfer_obstructed'});
  const frame = c.reference.currentFrame();
  frame.approachCollision = {position_error_m: .08, constraints: [{object_id: 'uiuc_i', clearance_m: .006, required_clearance_m: .005}]};
  c.reference.currentFrame = () => frame;
  const before = structuredClone(object);
  for (const time of [2.6, 3.6, 4.6, 5.59]) {
    c.scene.data.time = time; c.advanceTransfer(); assert.equal(c.phase, 'transfer');
  }
  c.scene.data.time = 5.61;
  c.policy.control = () => { throw new Error('A final obstruction must not run policy again.'); };
  await c.tick();
  assert.equal(c.phase, 'failed'); assert.equal(c.busy, false);
  assert.equal(c.snapshot().failure_reason.code, 'transfer_obstructed');
  assert.equal(c.snapshot().failure_reason.reset_required, true);
  assert.equal(c.motionCompleted, false); assert.equal(c.success, false);
  assert.deepEqual(object, before);
});

test('transient transfer proximity and policy lag cannot claim collision-free IK is blocked', () => {
  const {c} = fixture({phase: 'hold', elapsed: 0});
  c.beginDeposit(); c.transferIndex = 1; c.transferTime = 0; c.startTransferStage();
  c.stableLift = 1; c.lift = .1;
  c.objectBounds = () => ({lower: [.47, -.23, .9], upper: [.53, -.17, 1.02]});
  c.transferObstruction = new ApproachObstructionMonitor({dwell: 3, code: 'transfer_obstructed'});
  const frame = c.reference.currentFrame();
  const diagnostic = {position_error_m: .08, constraints: [{object_id: 'uiuc_i'}]};
  frame.approachCollision = diagnostic; c.reference.currentFrame = () => frame;
  for (const time of [2.6, 3.6, 4.6]) {c.scene.data.time = time; c.advanceTransfer();}
  diagnostic.position_error_m = .009;
  for (const time of [5.6, 6.6, 7.6]) {c.scene.data.time = time; c.advanceTransfer();}
  assert.equal(c.phase, 'transfer'); assert.equal(c.snapshot().failure_reason, null);
  diagnostic.position_error_m = .08; diagnostic.constraints = [];
  for (const time of [8.6, 9.6, 10.6, 11.6]) {c.scene.data.time = time; c.advanceTransfer();}
  assert.equal(c.phase, 'transfer'); assert.equal(c.snapshot().failure_reason, null);
});

test('a bounded transfer stall without collision evidence requests reset without inventing an obstacle', () => {
  const {c} = fixture({phase: 'hold', elapsed: 0});
  c.beginCompletionTransfer('grasp_not_secured');
  c.scene.data.time = 45;
  c.advanceTransfer();
  assert.equal(c.pendingReturn, true); assert.equal(c.busy, true);
  assert.equal(c.snapshot().failure_reason.code, 'transfer_stalled');
  assert.equal(c.snapshot().failure_reason.reset_required, true);
  assert.doesNotMatch(c.message, /collision-free|another object/);
});

test('a verified deposit preserves the completion release trajectory and suppresses obsolete obstruction evidence', () => {
  const {c} = fixture({phase: 'hold', elapsed: 0});
  c.beginCompletionTransfer('transport_slip');
  c.transferIndex = 1; c.transferTime = 0; c.startTransferStage();
  c.scene.data.time = 4; c.stableDeposit = .8; c.everPlaced = true;
  c.transferObstruction = new ApproachObstructionMonitor({dwell: 3, code: 'transfer_obstructed'});
  c.transferObstruction.confirmed = {code: 'transfer_obstructed', objects: [{object_id: 'uiuc_i'}]};
  c.advanceTransfer();
  assert.equal(c.phase, 'transfer'); assert.equal(c.transferIndex, 1);
  assert.equal(c.transferObstruction.confirmed, null);
  assert.equal(c.snapshot().failure_reason, null);
  assert.equal(c.motionCompleted, false);
});

test('detached clear hand proceeds to return even when the object missed the tray', async () => {
  const {c} = fixture({phase: 'settle', elapsed: 1.5});
  c.clearTrayTop = .8;
  c.clearTrayStable = 0;
  c.beginReturn = async function () { this.enter('return_home', 'Returning home.'); };
  for (let i = 0; i < 15 && c.phase === 'settle'; i++) {
    await c.advanceClearTray();
    c.scene.data.time += c.dt;
  }
  assert.equal(c.phase, 'return_home');
  assert.notEqual(c.depositSuccess, true);
});

test('hand contact or unsafe clearance still prevents the return shortcut', async () => {
  for (const obstruction of ['contact', 'clearance']) {
    const {c} = fixture({phase: 'settle', elapsed: 1.5});
    c.clearTrayTop = .8;
    c.clearTrayStable = 0;
    c.stableDeposit = 1; c.everPlaced = true;
    if (obstruction === 'contact') c.handContacts = 1;
    else c.audit.check = () => ({passed: false, minimumClearance: -.001});
    c.beginReturn = async function () { throw new Error('Unsafe hand must not return.'); };
    for (let i = 0; i < 20; i++) {
      await c.advanceClearTray();
      c.scene.data.time += c.dt;
    }
    assert.equal(c.phase, 'settle', obstruction);
  }
});

test('beginning a safe return does not promote an unsuccessful deposit to success', async () => {
  const {c} = fixture({phase: 'settle'});
  c.depositSuccess = false;
  c.actualRootPostureAudit = () => ({passed: true, minimumClearance: .05});
  c.preflightHomePath = async () => ({passed: true, minimumClearanceSurplus: .04});
  await c.beginReturn();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.depositSuccess, false);
  assert.equal(c.success, null);
  assert.equal(c.returnSuccess, null);
});

test('finishing the robot motion records actual task failure when no object was deposited', () => {
  const {c} = fixture({phase: 'return_home'});
  c.graspSuccess = false;
  c.depositSuccess = false;
  c.finishCycle();
  assert.equal(c.motionCompleted, true);
  assert.equal(c.returnSuccess, true);
  assert.equal(c.success, false);
  assert.equal(c.phase, 'failed');
  const status = c.snapshot();
  assert.equal(status.motion_completed, true);
  assert.equal(status.return_success, true);
  assert.equal(status.success, false);
});

test('a recorded released landing certifies success after fallback motion', () => {
  const {c} = fixture({phase: 'return_home'});
  c.graspSuccess = true;
  c.depositSuccess = true;
  c.stableDeposit = 1; c.everPlaced = true;
  c.inTray = true;
  c.completionFallback = true;
  c.finishCycle();
  assert.equal(c.motionCompleted, true);
  assert.equal(c.returnSuccess, true);
  assert.equal(c.success, true);
  assert.equal(c.phase, 'succeeded');
});

test('recorded landing overrides a stale result flag without rewriting grasp history', () => {
  for (const supported of [false, true]) {
    const {c} = fixture({phase: 'return_home'});
    c.graspSuccess = false;
    c.depositSuccess = !supported;
    c.stableDeposit = supported ? .8 : 0; c.everPlaced = supported;
    c.inTray = supported;
    c.finishCycle();
    assert.equal(c.success, supported);
    assert.equal(c.depositSuccess, supported);
    assert.equal(c.graspSuccess, false);
    assert.equal(c.motionCompleted, true);
    assert.equal(c.returnSuccess, true);
  }
});

test('fallback transport continues when the ungrasped object is below the table', async () => {
  const {c, object} = fixture({phase: 'transfer'});
  c.completionFallback = true;
  c.completionTransfer = true;
  object.positionW[2] = .4;
  c.advanceTransfer = () => {};
  await c.tick();
  assert.equal(c.phase, 'transfer');
  assert.notEqual(c.success, true);
});

test('fallback retains measured collision, invalid policy, and lost-balance hard stops', async () => {
  for (const hazard of ['self collision', 'fixture contact', 'invalid policy', 'lost balance']) {
    const {c, state} = fixture({phase: 'transfer'});
    c.completionFallback = true;
    c.completionTransfer = true;
    c.advanceTransfer = () => {};
    if (hazard === 'self collision') c.audit.check = () => ({passed: false, minimumClearance: -.001});
    if (hazard === 'fixture contact') c.bodyFixtureForce = 20;
    if (hazard === 'invalid policy') c.policy.control = async () => [NaN];
    if (hazard === 'lost balance') state.rootPosW[2] = .4;
    await c.tick();
    // Body/fixture force is terminal only when it stays above 10 N for 0.1 s: hold it for six ticks.
    if (hazard === 'fixture contact') for (let i = 0; i < 6 && !c.pendingReturn; i++) { c.bodyFixtureForce = 20; c.scene.data.time += .02; await c.tick(); }
    // Internal errors and lost balance are terminal; collision and fixture contact first withdraw the hand.
    if (['invalid policy', 'lost balance'].includes(hazard)) assert.equal(c.phase, 'failed', hazard); else { assert.equal(c.pendingReturn, true, hazard); assert.ok(c.failureMessage, hazard); }
    assert.equal(c.success, false, hazard);
    assert.equal(c.motionCompleted, false, hazard);
  }
});

function returnFixture({stage = 'retract', elapsed = 6.3} = {}) {
  const result = fixture({phase: 'return_home', elapsed});
  const {c, state} = result, solves = [], audits = [];
  c.scene.data.qpos=new Float64Array([...c.scene.data.qpos,...Array(14).fill(0)]);
  c.scene.data.qvel=new Float64Array(20);
  c.scene.fingerQposAddresses14=Array.from({length:14},(_,i)=>7+i);
  c.scene.fingerDofAddresses14=Array.from({length:14},(_,i)=>6+i);
  c.scene.metadata={handOpen:{left:Array(7).fill(0),right:Array(7).fill(0)},fingerLimits:Array.from({length:14},()=>[-1,2])};
  Object.assign(c, {
    homeStage: stage, homeTime: 0, homeDuration: stage === 'retract' ? 2.2 : 3,
    homeStable: 0, homeMeasuredAuditTime: -Infinity, homeEndpointAudit: null,
    homePlanChecks: [], tableFront: .3, homeRotationForward: {left: .12, right: .12},
    homeRestoreFrom: Array(29).fill(.1),
    homeLocal: Object.fromEntries(['left', 'right'].map(side => [side, {
      position: sub(c.palm(side).position, state.rootPosW), rotation: eye3(),
    }])),
    rootHomeMetrics: () => ({reached: true, quiet: true}),
    actualRootPostureAudit(q = c.homeQ) {
      audits.push(q.slice());
      return {passed: true, minimumClearance: .04};
    },
  });
  c.reference.ik.solve = (q, target, options) => {
    solves.push({q: q.slice(), target: structuredClone(target), options: {...options}});
    return {q: target.posture.slice(), geometry: {passed: true, minimumClearanceSurplus: .035},
      residual: {footPositionError: .001, comMargin: .04}};
  };
  return {...result, solves, audits};
}

test('a settled closed-fist return executes opening before measured fingers can certify completion',async()=>{
  const {c,closures}=returnFixture({stage:'restore',elapsed:3.02});
  c.fistReturn=true;c.homeStable=.5;c.everPlaced=true;
  for(const address of c.scene.fingerQposAddresses14)c.scene.data.qpos[address]=1;
  let steps=0;c.scene.step=()=>{steps++;c.scene.data.time+=.02;};
  await c.tick();assert.equal(c.phase,'return_home');assert.equal(c.homeStage,'verify');
  assert.equal(c.fistReturn,false);assert.equal(c.homeStable,0);assert.equal(closures[c.hand],0);assert.equal(steps,1);
  for(let i=0;i<30;i++)await c.tick();
  assert.equal(c.returnSuccess,null);assert.equal(c.homeStable,0);assert.equal(c.homeHandOpening.opened,false,
    'An open command cannot substitute for measured finger movement.');
  for(const address of c.scene.fingerQposAddresses14)c.scene.data.qpos[address]=0;
  for(let i=0;i<24;i++)await c.tick();assert.equal(c.phase,'return_home');
  await c.tick();assert.equal(c.phase,'succeeded');assert.equal(c.returnSuccess,true);assert.equal(c.homeHandOpening.opened,true);
});

test('blocked, moving, or unobservable fingers cannot certify return and the opening verification remains bounded',async()=>{
  for(const fault of ['closed','moving','missing','nonfinite']){
    const {c}=returnFixture({stage:'verify',elapsed:3.1});c.homeVerifyTime=3;c.everPlaced=true;
    if(fault==='closed')c.scene.data.qpos[c.scene.fingerQposAddresses14[0]]=.3;
    if(fault==='moving')c.scene.data.qvel[c.scene.fingerDofAddresses14[0]]=.51;
    if(fault==='missing')delete c.scene.fingerQposAddresses14;
    if(fault==='nonfinite')c.scene.data.qpos[c.scene.fingerQposAddresses14[0]]=NaN;
    for(let i=0;i<30;i++){await c.advanceReturn();c.scene.data.time+=.02;}
    assert.equal(c.phase,'return_home',fault);assert.equal(c.homeStable,0,fault);
    c.scene.data.time=10.02;await c.advanceReturn();
    assert.equal(c.returnSuccess,false,fault);assert.equal(c.depositSuccess,true,fault);
  }
});

test('finger return evidence covers both supported hand joint counts and the clamped open pose',()=>{
  for(const count of [14,24]){
    const {c}=returnFixture();
    c.scene.data.qpos=new Float64Array(7+count);c.scene.data.qvel=new Float64Array(6+count);
    c.scene.fingerQposAddresses14=Array.from({length:count},(_,i)=>7+i);
    c.scene.fingerDofAddresses14=Array.from({length:count},(_,i)=>6+i);
    c.scene.metadata={handOpen:{left:Array(count/2).fill(-.2),right:Array(count/2).fill(-.2)},fingerLimits:Array.from({length:count},()=>[0,2])};
    assert.equal(c.homeHandOpeningAudit().opened,true);
    c.scene.data.qpos[c.scene.fingerQposAddresses14.at(-1)]=.151;
    assert.equal(c.homeHandOpeningAudit().opened,false,'The final joint of the other hand is also measured.');
  }
});

// The palm can move backward through either waist pitch or shoulder motion.
// Run the actual DLS, bounded steps, residual checks, and controller preflight;
// linear FK/Jacobians isolate the unwanted waist/arm exchange from full physics.
function redundantRetractionIK(start, leftPalm) {
  const nv = 29, qadr = Array.from({length: nv}, (_, i) => i + 7);
  const data = {qpos: new Float64Array(36), xpos: new Float64Array(9),
    xmat: new Float64Array([...eye3(), ...eye3(), ...eye3()]),
    subtree_com: new Float64Array([0, 0, .8, 0, 0, .8])};
  const palmJac = new Float64Array(3 * nv); palmJac[14] = 1; palmJac[22] = 1;
  const geometry = {passed: true, minimumClearance: .04, minimumClearanceSurplus: .035,
    minimumBufferedSurplus: .032, near: []};
  return Object.assign(Object.create(WholeBodyIK.prototype), {
    data, qadr, vadr: Array.from({length: nv}, (_, i) => i), model: {nv}, rootBodyId: 0,
    lower: Array(nv).fill(-2), upper: Array(nv).fill(2), defaultQ: start.slice(),
    anchorPosition: [0, 0, .76], anchorRotation: eye3(),
    supportPolygon: [[-.2, -.1], [.2, -.1], [.2, .1], [-.2, .1]],
    feet: {left: {position: [0, 0, 0], rotation: eye3()}, right: {position: [0, 0, 0], rotation: eye3()}},
    scene: {ankleBodyIds: {left: 0, right: 0}, palmBodyIds: {left: 2, right: 1},
      jacBody: (data, body) => ({position: body === 1 ? palmJac : new Float64Array(3 * nv),
        rotation: new Float64Array(3 * nv)})},
    goals: target => target.palms || {}, collisionOptions: () => ({}),
    audit: {check: () => structuredClone(geometry)},
    setPose(q) {
      data.qpos.set([0, 0, .76, 1, 0, 0, 0]);
      qadr.forEach((address, i) => { data.qpos[address] = q[i]; });
      data.xpos.set([q[22] + q[14] - start[14], -.008, .86], 3);
      data.xpos.set(leftPalm.position, 6);
    },
  });
}

test('midline Cartesian return moves the arm while retaining the waist in crouched and standing preflight and runtime horizons', async () => {
  for (const crouched of [true, false]) {
    const {c, palm, current, object, closures} = returnFixture({elapsed: 0});
    const start = Array(29).fill(0); start[12] = .08; start[13] = -.03; start[14] = .007; start[22] = .43;
    c.crouched = crouched; c.rootPosition = [0, 0, crouched ? .72 : .76];
    c.rootHeight = c.rootPosition[2]; c.rootPitch = crouched ? .08 : 0;
    c.baseTarget = InteractiveController.prototype.baseTarget;
    current.position[1] = palm.position[1] = -.008;
    c.palm = (side = 'right') => structuredClone(side === 'right' ? palm : c.homePalms.left);
    c.audit.handBounds = (data, side = 'right') => side === 'left'
      ? {lower: [.05, .15, .84], upper: [.15, .25, .96]}
      : {lower: [.35, -.06, .80], upper: [.48, .04, .92]};
    c.reference.qCurrent = start.slice();
    c.reference.ik = redundantRetractionIK(start, c.homePalms.left);
    const solved = [], solve = c.reference.ik.solve;
    c.reference.ik.solve = function (q, target, options) {
      const result = solve.call(this, q, target, options);
      solved.push({q: result.q.slice(), target: structuredClone(target), options: {...options}});
      return result;
    };
    c.graspSuccess = false; c.stableDeposit = 0;
    const live = Array.from(c.scene.data.qpos), originalObject = structuredClone(object);
    await c.beginReturn();
    assert.equal(c.phase, 'return_home'); assert.equal(c.homeStage, 'retract');
    assert.equal(solved.length, 111, 'Endpoint IK and every 20 ms withdrawal sample must execute.');
    assert.equal(solved[0].options.strictEndpoint, true); assert.equal(solved.at(-1).options.strictEndpoint, true);
    assert.ok(solved.every(row => row.options.bestEffort === false));
    for (const row of solved) near(row.q.slice(12, 15), start.slice(12, 15));
    assert.ok(solved.at(-1).q[22] < start[22] - .15, 'The retract must actually progress through arm motion.');
    for (const offset of [0, .5, 1.1, 2.2, 4]) {
      const target = c.targetAt(offset);
      const result = solve.call(c.reference.ik, start, target, {iterations: 20, maxStep: 3, strictEndpoint: true});
      near(result.q.slice(12, 15), start.slice(12, 15));
      assert.deepEqual(target.rootPosition, c.rootPosition); assert.equal(target.rootPitch, c.rootPitch);
      near([result.q[22] + result.q[14] - start[14]], [target.palms.right.position[0]], .012);
    }
    const endpoint = c.homeTarget(10);
    const unlocked = solve.call(c.reference.ik, start, {...endpoint, lockWaist: false},
      {iterations: 20, maxStep: 3, strictEndpoint: true});
    assert.ok(unlocked.q[14] < start[14] - .05,
      'The unlocked control must reproduce the backward-waist substitution this regression prevents.');
    assert.deepEqual(Array.from(c.scene.data.qpos), live); assert.deepEqual(object, originalObject);
    assert.deepEqual(closures, {left: 0, right: 0});
    assert.equal(c.graspSuccess, false); assert.equal(c.depositSuccess, false);
    assert.equal(c.success, null); assert.equal(c.returnSuccess, null);
  }
});

test('waist-held retraction still stops at measured self collision and never promotes grasp or placement results', async () => {
  for (const supported of [false, true]) {
    const {c, object} = returnFixture({elapsed: 0});
    c.crouched = true; c.graspSuccess = false; c.stableDeposit = supported ? .8 : 0; c.everPlaced = supported;
    await c.beginReturn();
    c.audit.check = (data, options = {}) => options.selfOnly
      ? {passed: false, minimumClearance: -.001, limitingPair: {a: 43, b: 83}}
      : {passed: true, minimumClearance: .04};
    const originalObject = structuredClone(object);
    await c.tick();
    assert.equal(c.phase, 'failed'); assert.equal(c.failurePhase, 'return_home');
    assert.equal(c.returnSuccess, false); assert.equal(c.motionCompleted, false);
    assert.equal(c.graspSuccess, false);
    assert.equal(c.depositSuccess, supported); assert.equal(c.success, supported);
    assert.deepEqual(object, originalObject);
    assert.match(c.lastError.message, /self-intersection/);
  }
});

test('a measured self-collision after the released object is ignored is recorded and the return continues', async () => {
  for (const supported of [false, true]) {
    const {c, object} = returnFixture({elapsed: 0});
    c.crouched = true; c.graspSuccess = false; c.stableDeposit = supported ? .8 : 0; c.everPlaced = supported;
    c.releaseExit = {objectIgnored: true, stage: 'rise', checks: [], segment: null};
    await c.beginReturn();
    c.lastError = null;
    c.audit.check = (data, options = {}) => options.selfOnly
      ? {passed: false, minimumClearance: -.001, limitingPair: {a: 43, b: 83}}
      : {passed: true, minimumClearance: .04};
    const originalObject = structuredClone(object);
    await c.tick();
    assert.equal(c.phase, 'return_home'); assert.equal(c.homeStage, 'retract');
    assert.ok(c.completionReasons.includes('self_clearance_violation'));
    assert.equal(c.postReleaseHazards.at(-1).reason, 'self_clearance_violation');
    assert.deepEqual(c.postReleaseHazards.at(-1).limiting_pair, {a: 43, b: 83, distance_m: null, required_margin_m: null});
    assert.equal(c.returnSuccess, null); assert.equal(c.motionCompleted, false);
    assert.equal(c.graspSuccess, false);
    assert.equal(c.depositSuccess, supported); assert.equal(c.success, null);
    assert.deepEqual(object, originalObject);
    assert.equal(c.lastError, null); assert.equal(c.failureMessage, undefined); assert.ok(!c.pendingReturn);
  }
});

function softEndpointFailure(extra = {}) {
  return new IKError('The rest posture is outside endpoint accuracy.', {
    geometry: {passed: true}, footPositionError: .001, comMargin: .04, endpoint: false, ...extra,
  });
}

function approximateResult(result, details = {}) {
  return {...result, passed: false, bestEffort: true, unmetConstraints: Object.keys(details),
    geometry: {...result.geometry, ...details.geometry},
    residual: {...result.residual, ...Object.fromEntries(Object.entries(details).filter(([key]) => key !== 'geometry'))}};
}

test('conservative retract timeout invokes the audited restore pipeline when the measured-root endpoint is safe', async () => {
  const {c, solves, audits} = returnFixture();
  assert.ok(c.audit.handBounds().upper[0] > c.tableFront - .025);
  await c.advanceReturn();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.homeStage, 'restore');
  assert.ok(c.completionReasons.includes('return_clearance_retry'));
  assert.equal(solves.length, 151, 'Restoration must run endpoint IK and every 20 ms trajectory sample.');
  assert.equal(solves[0].options.strictEndpoint, true);
  assert.equal(solves.at(-1).options.strictEndpoint, true);
  assert.ok(audits.length >= 152, 'The endpoint and each frame must be checked at the measured root.');
  assert.deepEqual(c.homeTarget(10).posture, c.homeQ);
  assert.equal(c.returnSuccess, null);
});

test('conservative retract timeout continues approximately when only the predicted home endpoint is rejected', async () => {
  const {c, solves} = returnFixture();
  c.actualRootPostureAudit = () => ({passed: false, minimumClearance: -.001});
  await c.advanceReturn();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.homeStage, 'restore');
  assert.equal(c.motionCompleted, false);
  assert.equal(solves.length, 152, 'A rejected predicted endpoint must still produce its final approximate path.');
  assert.ok(solves.slice(1).every(call => call.options.bestEffort === true));
  const report = c.homePlanChecks.find(check => check.stage === 'restore_approximate');
  assert.equal(report.passed, false);
  assert.equal(report.violations.length, 151);
  assert.equal(c.returnSuccess, null);
});

test('soft endpoint accuracy first retries precision while auditing every measured-root frame', async () => {
  const {c, solves, audits} = returnFixture();
  const solve = c.reference.ik.solve;
  c.reference.ik.solve = (q, target, options) => {
    const answer = solve(q, target, options);
    if (solves.length === 1) throw softEndpointFailure();
    return answer;
  };
  await c.beginHomeRestore();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.homeStage, 'restore');
  assert.equal(solves.length, 152);
  assert.equal(solves[0].options.strictEndpoint, true);
  assert.ok(solves.slice(1).every(call => call.options.strictEndpoint === false));
  assert.equal(audits.length, 152, 'Relaxed endpoint accuracy must retain initial, endpoint, and all 150 frame audits.');
  assert.ok(c.homePlanChecks.some(check => check.stage === 'restore_best_effort' && check.passed));
  assert.ok(c.completionReasons.includes('rest_posture_retry'));
  assert.deepEqual(c.homeTarget(10).posture, c.homeQ, 'Runtime still commands the original home posture.');
  assert.equal(c.returnSuccess, null, 'A feasible relaxed reference does not certify actual return.');
});

test('rejected geometry, foot support, and COM preflights continue with explicitly approximate results', async () => {
  const failures = [
    {geometry: {passed: false}}, {footPositionError: .003}, {comMargin: .01999}, {endpoint: true},
  ];
  for (const details of failures) {
    const {c, solves} = returnFixture();
    const solve = c.reference.ik.solve;
    c.reference.ik.solve = (q, target, options) => {
      const result = solve(q, target, options);
      if (!options.bestEffort) throw softEndpointFailure(details);
      return approximateResult(result, details);
    };
    await c.beginHomeRestore();
    assert.equal(c.phase, 'return_home', JSON.stringify(details));
    assert.equal(c.homeStage, 'restore');
    assert.equal(solves.length, 152, 'Approximate restoration still computes every trajectory sample.');
    assert.ok(solves.slice(1).every(call => call.options.bestEffort === true));
    const report = c.homePlanChecks.find(check => check.stage === 'restore_approximate');
    assert.equal(report.passed, false, 'Continuing the motion must not mark rejected constraints as valid.');
    assert.equal(report.bestEffort, true);
    assert.equal(report.violations.length, 151);
    assert.deepEqual(report.violations[0].unmetConstraints, Object.keys(details));
    assert.equal(c.motionCompleted, false);
    assert.equal(c.returnSuccess, null);
  }
});

test('rejected measured-root endpoint and trajectory predictions remain recorded during approximate restoration', async () => {
  for (const blockedAudit of [2, 4]) {
    const {c, solves, audits} = returnFixture();
    const solve = c.reference.ik.solve, audit = c.actualRootPostureAudit;
    c.reference.ik.solve = (q, target, options) => {
      const answer = solve(q, target, options);
      if (solves.length === 1) throw softEndpointFailure();
      return answer;
    };
    c.actualRootPostureAudit = q => {
      const answer = audit(q);
      return audits.length >= blockedAudit ? {passed: false, minimumClearance: -.002} : answer;
    };
    await c.beginHomeRestore();
    assert.equal(c.phase, 'return_home');
    assert.equal(c.homeStage, 'restore');
    assert.equal(c.returnSuccess, null);
    assert.equal(c.motionCompleted, false);
    assert.equal(audits.length, blockedAudit + 151);
    const report = c.homePlanChecks.find(check => check.stage === 'restore_approximate');
    assert.equal(report.passed, false);
    assert.equal(report.bestEffort, true);
    assert.ok(report.violations.some(row => row.stage === 'measured_root_endpoint'));
    assert.equal(report.violations.filter(row => row.stage === 'measured_root_path').length, 150);
  }
});

test('trajectory IK rejection produces a final approximate path with truthful constraint diagnostics', async () => {
  for (const hazard of ['geometry', 'COM']) {
    const {c, solves} = returnFixture();
    const solve = c.reference.ik.solve;
    c.reference.ik.solve = (q, target, options) => {
      const answer = solve(q, target, options);
      if (solves.length === 1) throw softEndpointFailure();
      if (solves.length === 4) throw new IKError(`Unsafe ${hazard} at a trajectory sample.`, {
        geometry: {passed: hazard !== 'geometry'}, footPositionError: .001, comMargin: hazard === 'COM' ? .01 : .04,
      });
      return options.bestEffort ? approximateResult(answer, {geometry: {passed: hazard !== 'geometry'}, comMargin: hazard === 'COM' ? .01 : .04}) : answer;
    };
    await c.beginHomeRestore();
    assert.equal(c.phase, 'return_home', hazard);
    assert.equal(c.homeStage, 'restore');
    assert.equal(solves.length, 155);
    assert.equal(c.motionCompleted, false);
    const report = c.homePlanChecks.find(check => check.stage === 'restore_approximate');
    assert.equal(report.passed, false);
    assert.equal(report.violations.length, 151);
    assert.ok(solves.slice(4).every(call => call.options.bestEffort === true));
  }
});

test('rejected root recovery computes a full approximate trajectory and retains its violations', async () => {
  const {c, solves} = returnFixture();
  c.rootPosition = [.03, 0, .72];
  const solve = c.reference.ik.solve;
  c.reference.ik.solve = (q, target, options) => {
    const result = solve(q, target, options);
    if (!options.bestEffort) throw softEndpointFailure({comMargin: .015});
    return approximateResult(result, {comMargin: .015});
  };
  await c.beginHomeRestore();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.homeStage, 'root_recovery');
  assert.equal(solves.length, 112);
  assert.ok(solves.slice(1).every(call => call.options.bestEffort === true));
  const report = c.homePlanChecks.find(check => check.stage === 'root_recovery_best_effort');
  assert.equal(report.passed, false);
  assert.equal(report.bestEffort, true);
  assert.equal(report.violations.length, 111);
  assert.ok(c.completionReasons.includes('root_ik_best_effort'));
  assert.equal(c.returnSuccess, null);
});

test('exhausted return preflight candidates still produce an approximate retract without marking it valid', async () => {
  const {c, solves} = returnFixture();
  const solve = c.reference.ik.solve;
  c.reference.ik.solve = (q, target, options) => {
    const result = solve(q, target, options);
    if (!options.bestEffort) throw softEndpointFailure({geometry: {passed: false}});
    return approximateResult(result, {geometry: {passed: false}});
  };
  await c.beginReturn();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.homeStage, 'retract');
  const strict = solves.filter(call => !call.options.bestEffort), approximate = solves.filter(call => call.options.bestEffort);
  assert.ok(strict.length > 0 && strict.length <= 8, 'The finite candidate search must precede the last approximate path.');
  assert.equal(approximate.length, 111);
  const report = c.homePlanChecks.at(-1);
  assert.equal(report.stage, 'retract_best_effort');
  assert.equal(report.passed, false);
  assert.equal(report.bestEffort, true);
  assert.equal(report.violations.length, 111);
  assert.ok(c.completionReasons.includes('return_ik_best_effort'));
  assert.equal(c.returnSuccess, null);
});

test('direct execution IK requests best effort and preserves the approximate solver result', () => {
  const {c, solves} = returnFixture();
  const solve = c.reference.ik.solve;
  c.reference.ik.solve = (q, target, options) => approximateResult(solve(q, target, options), {comMargin: .015});
  const result = c.executionIK(c.reference.qCurrent, {posture: c.restQ}, {iterations: 16, maxStep: .035});
  assert.equal(solves.length, 1);
  assert.equal(solves[0].options.bestEffort, true);
  assert.equal(solves[0].options.maxStep, .035);
  assert.equal(result.passed, false);
  assert.deepEqual(result.unmetConstraints, ['comMargin']);
  assert.equal(result.residual.comMargin, .015);
  assert.ok(c.completionReasons.includes('approximate_ik'));
});

test('final return timeout preserves physical placement success without falsely certifying return', async () => {
  const {c, state} = returnFixture({stage: 'verify', elapsed: 10.02});
  state.dofPos.fill(.3, 12);
  c.stableDeposit = 1; c.everPlaced = true;
  await c.advanceReturn();
  assert.equal(c.phase, 'succeeded');
  assert.equal(c.motionCompleted, true);
  assert.equal(c.returnSuccess, false);
  assert.equal(c.depositSuccess, true);
  assert.equal(c.success, true);
  assert.equal(c.homeJointError, .3, 'The original actual-pose acceptance threshold remains in force.');
});

test('a rejected predicted home posture cannot block physically settled root recovery or final return', async () => {
  for (const stage of ['root_recovery', 'verify']) {
    const {c, state} = returnFixture({stage, elapsed: 3.2});
    state.dofPos.fill(.1, 12);
    c.reference.qCurrent = state.dofPos.slice();
    c.homeStable = stage === 'root_recovery' ? .3 : .5;
    c.stableDeposit = 1; c.everPlaced = true;
    c.actualRootPostureAudit = () => ({passed: false, minimumClearance: -.002});
    assert.equal(c.audit.check().passed, true, 'The live measured pose remains physically clear.');
    await c.advanceReturn();
    assert.equal(c.homeEndpointAudit.passed, false, 'The rejected predicted posture must remain visible in diagnostics.');
    if (stage === 'root_recovery') {
      assert.equal(c.phase, 'return_home');
      assert.equal(c.homeStage, 'restore');
      assert.equal(c.crouched, false);
      assert.ok(c.homePlanChecks.some(check => check.stage === 'restore_approximate' && check.passed === false));
      assert.equal(c.returnSuccess, null);
    } else {
      assert.equal(c.phase, 'succeeded');
      assert.equal(c.motionCompleted, true);
      assert.equal(c.returnSuccess, true);
      assert.equal(c.success, true);
      assert.equal(c.homeJointError, .1, 'Final acceptance still uses the actual joint error.');
    }
  }
});

test('a measured return clearance shortfall starts one audited withdrawal per stage visit and the 60 s wall ends the episode honestly', async () => {
  const {c, solves} = returnFixture({stage: 'restore'});
  c.stableDeposit = 23.36; c.everPlaced = true;
  c.depositSuccess = true;
  c.inTray = true;
  c.fail = () => assert.fail('A measured clearance shortfall during the return never stops the controller.');
  c.audit.check = () => ({passed: false, minimumClearance: -.001});
  await c.advanceReturn();
  assert.equal(c.phase, 'return_home'); assert.equal(c.busy, true);
  assert.equal(c.homeStage, 'clearance_recovery'); assert.equal(c.homeClearanceRetries, 1);
  assert.ok(c.completionReasons.includes('return_clearance_recovery'));
  assert.equal(solves.length, 61, 'The withdrawal is fully audited before it executes.');
  assert.equal(c.depositSuccess, true); assert.equal(c.motionCompleted, false); assert.equal(c.returnSuccess, null);

  // The same stage visit never starts a second recovery search: the shortfall is recorded and the plan keeps executing.
  const second = returnFixture({stage: 'restore', elapsed: .1});
  second.c.stableDeposit = 23.36; second.c.everPlaced = true; second.c.depositSuccess = true; second.c.inTray = true;
  second.c.fail = () => assert.fail('never');
  second.c.homeRecoveryAttemptVisit = second.c.homeStageVisit || 0;
  second.c.audit.check = () => ({passed: false, minimumClearance: -.001});
  await second.c.advanceReturn();
  assert.equal(second.c.phase, 'return_home'); assert.equal(second.c.homeStage, 'restore');
  assert.ok(second.c.completionReasons.includes('return_clearance_violation'));
  assert.equal(second.c.homeClearanceViolations, 1); assert.equal(second.solves.length, 0);
  assert.equal(second.c.motionCompleted, false); assert.equal(second.c.returnSuccess, null);

  // The global post-release wall ends the episode with the existing honest outcome.
  second.c.postReleaseDeadline = second.c.scene.data.time;
  await second.c.advanceReturn();
  assert.equal(second.c.phase, 'succeeded'); assert.equal(second.c.busy, false);
  assert.equal(second.c.message, 'Placement successful. The return to the initial posture is incomplete.');
  assert.equal(second.c.returnSuccess, false); assert.equal(second.c.depositSuccess, true);
  assert.equal(second.c.success, true, 'The bounded return must not erase the measured physical placement.');
  assert.equal(second.c.motionCompleted, true);
  assert.ok(second.c.completionReasons.includes('return_deadline'));
  assert.equal(second.c.postReleaseHazards.at(-1).reason, 'return_deadline');
});

test('a bounded return preserves recorded landing evidence instead of a stale result flag', async () => {
  for (const supported of [false, true]) {
    const {c} = returnFixture({stage: 'restore'});
    c.stableDeposit = supported ? .8 : 0; c.everPlaced = supported;
    c.depositSuccess = !supported;
    c.inTray = supported;
    c.fail = () => assert.fail('A measured clearance shortfall during the return never stops the controller.');
    c.audit.check = () => ({passed: false, minimumClearance: -.001});
    await c.advanceReturn();
    assert.equal(c.phase, 'return_home');
    assert.equal(c.homeStage, 'clearance_recovery');
    assert.ok(c.completionReasons.includes('return_clearance_recovery'));
    assert.equal(c.returnSuccess, null);
    assert.equal(c.motionCompleted, false);
    c.postReleaseDeadline = c.scene.data.time;
    await c.advanceReturn();
    assert.equal(c.phase, supported ? 'succeeded' : 'failed');
    assert.equal(c.returnSuccess, false);
    assert.equal(c.motionCompleted, true);
    assert.equal(c.depositSuccess, supported);
    assert.equal(c.success, supported);
    assert.ok(c.completionReasons.includes('return_deadline'));
  }
});

test('a settled placement detected at an earlier safety stop reports return as incomplete', () => {
  const {c} = fixture({phase: 'release'});
  c.stableDeposit = .8; c.everPlaced = true;
  c.depositSuccess = null;
  c.fail('A genuine robot collision stopped the release.');
  assert.equal(c.pendingReturn, true);
  assert.equal(c.failurePhase, 'release');
  assert.equal(c.motionCompleted, false);
  assert.equal(c.success, true);
  assert.equal(c.depositSuccess, true);
  assert.equal(c.returnSuccess, false);
});

test('return completion preserves a released landing after the object rolls out', () => {
  const {c} = fixture({phase: 'return_home'});
  c.everPlaced = true;
  c.depositSuccess = null;
  c.stableDeposit = 0;
  c.inTray = false;
  c.finishCycle({returned: false});
  assert.equal(c.phase, 'succeeded');
  assert.equal(c.motionCompleted, true);
  assert.equal(c.returnSuccess, false);
  assert.equal(c.depositSuccess, true);
  assert.equal(c.success, true);
});

function landingFixture() {
  const {c} = fixture({phase: 'release'});
  c.scene.contacts = () => [];
  c.scene.trayGeomIds = [];
  Object.assign(c.scene.model, {geom_bodyid: [0, 1, 2], geom_contype: [1, 1, 1], geom_conaffinity: [1, 1, 1]});
  c.scene.objects.apple.geomIds = [1];
  c.scene.bodyName = id => ['world', 'demo_apple', 'right_hand_palm_link'][id];
  c.scene.geomBounds = () => ({lower: [.47, -.23, .75], upper: [.53, -.17, .87]});
  c.objectBounds = InteractiveController.prototype.objectBounds;
  Object.assign(c.audit, {fingers: new Map(), fixtureGeoms: [], robotGeoms: [2], hands: {left: [], right: [2]}, handSet: new Set([2])});
  c.recentContacts = [];
  c.scene.lastStepTrayLandings = [];
  c.measure = InteractiveController.prototype.measure;
  return c;
}

test('a brief released landing latches without a dwell, quiet velocity, current overlap, or verified grasp', () => {
  const c = landingFixture();
  c.scene.data.qvel.fill(2);
  const landing = {object_id: 'apple', time_s: 3.981,
    contact_position_w: [.47, .08, .755], object_geom: 1, tray_geom: 2, overlap_m2: .0002};
  c.scene.lastStepTrayLandings = [landing];
  c.measure();
  assert.equal(c.stableDeposit, 0); assert.equal(c.inTray, false);
  assert.equal(c.everPlaced, true); assert.deepEqual(c.trayLanding, landing);
  assert.notEqual(c.graspSuccess, true);
  // Evidence must be owned by the attempt, not the next simulation-step array.
  landing.contact_position_w[0] = 99;
  c.scene.lastStepTrayLandings = [];
  for (let i = 0; i < 5; i++) { c.scene.data.time += .02; c.measure(); }
  assert.equal(c.trayLanding.contact_position_w[0], .47);
  assert.equal(c.snapshot().ever_placed, true);
  assert.equal(c.snapshot().tray_landing.time_s, 3.981);
  c.finishCycle({returned: false});
  assert.equal(c.success, true); assert.equal(c.returnSuccess, false);
});

test('another object, a previous attempt, and absent contact evidence cannot certify placement', () => {
  for (const events of [[], [{object_id: 'bottle', time_s: 3.9}], [{object_id: 'apple', time_s: 1.9}]]) {
    const c = landingFixture(); c.startTime = 2;
    c.scene.lastStepTrayLandings = events;
    c.measure();
    assert.equal(c.everPlaced, false); assert.equal(c.trayLanding, null);
    c.finishCycle();
    assert.equal(c.success, false); assert.equal(c.depositSuccess, false);
  }
});

test('cancel preserves an actual landing but cannot turn a stale result flag into success', () => {
  for (const landed of [false, true]) {
    const c = landingFixture();
    c.everPlaced = landed; c.depositSuccess = !landed;
    const status = c.cancel();
    assert.equal(status.success, landed); assert.equal(status.deposit_success, landed);
    assert.equal(status.busy, false); assert.equal(status.return_success, false);
    assert.equal(status.motion_completed, false);
  }
});

test('a physics error after a valid substep landing preserves it even before measure runs', async () => {
  const {c} = fixture({phase: 'lift', elapsed: .1});
  c.scene.step = () => {
    c.scene.lastStepTrayLandings = [{object_id: 'apple', time_s: .101,
      contact_position_w: [.47, .08, .755], object_geom: 1, tray_geom: 2, overlap_m2: .0002}];
    throw new Error('Non-finite MuJoCo state; execution stopped.');
  };
  const status = await c.tick();
  assert.equal(status.phase, 'failed'); assert.equal(status.busy, false);
  assert.equal(status.ever_placed, true); assert.equal(status.success, true);
  assert.equal(status.deposit_success, true); assert.equal(status.return_success, false);
  assert.equal(status.motion_completed, false);
  assert.match(status.error.message, /Non-finite MuJoCo state/);
});

test('a fallen object during an unverified lift opens the hand instead of completing an empty tray gesture', async () => {
  const {c, object} = fixture({phase: 'lift', elapsed: .2});
  object.positionW[2] = .4;
  await c.tick();
  assert.equal(c.phase, 'release');
  assert.equal(c.graspAbort.reason, 'object_fell');
  assert.equal(c.graspSuccess, false);
  assert.equal(c.completionTransfer, false);
});

test('a secured low-table mug places from its reached stance while higher tables retain loaded standing', async () => {
  for (const mode of ['hero_plus', 'other_policy']) for (const height of [.5, .74, .88]) {
    const {c, object, palm, current, state, closures} = returnFixture();
    const low = height < .60;
    Object.assign(c, {mode, phase: 'hold', phaseTime: 0, objectId: 'mug', crouched: true,
      stableLift: .8, handContacts: 2, rootPosition: [.02, 0, .70], rootHeight: .70, rootPitch: .08});
    c.scene.objects.mug = c.scene.objects.apple; delete c.scene.objects.apple;
    c.scene.data.time = 1.21;
    c.scene.data.qpos[2] = .70;
    c.scene.tableTopZ = height;
    c.scene.tray.bottomTopZ = height + .015;
    c.scene.tray.wallTopZ = height + .09;
    state.rootPosW = c.rootPosition.slice();
    object.positionW[2] = height + .13;
    palm.position[2] = current.position[2] = height + .16;
    c.objectBounds = () => ({lower: [.47, -.23, height + .09], upper: [.53, -.17, height + .17]});
    c.audit.handBounds = () => ({lower: [.35, -.25, height + .1], upper: [.48, -.15, height + .2]});
    c.reference.currentFrame = () => ({rootPosW: [.02, 0, .70], plannedRootPitch: .08});
    c.baseTarget = (palms, options) => ({palms, posture: c.restQ.slice(), ...options});
    Object.assign(closures, {right: 1, left: 0});
    const initialPose = Array.from(c.scene.data.qpos), initialObject = structuredClone(object), initialPalm = structuredClone(palm);
    let stepped = false, targetAtTransition;
    c.reference.window = targetAt => { targetAtTransition = targetAt(0); return {}; };
    c.scene.step = () => {
      stepped = true;
      assert.deepEqual(Array.from(c.scene.data.qpos), initialPose, 'The phase transition must not reposition the live robot.');
      assert.deepEqual(object, initialObject, 'The phase transition must not reposition the held object.');
      assert.deepEqual(c.palm(), initialPalm);
      assert.deepEqual(closures, {right: 1, left: 0}, 'The active fingers must remain closed across the transition.');
    };
    await c.tick();
    assert.equal(stepped, true);
    assert.equal(c.phase, low ? 'transfer' : 'stand', `${mode} at ${height} m`);
    assert.equal(c.graspSuccess, true);
    assert.equal(c.completionFallback, false);
    assert.equal(c.crouched, true, 'The reached stance is retained until the later recovery stage.');
    near(targetAtTransition.palms.right.position, current.position);
    if (low) {
      assert.deepEqual(c.transferSegments.map(segment => segment.name), ['raise_load', 'over_tray', 'lower_load']);
      assert.equal(c.standStartRoot, undefined, 'Low-table placement must bypass the loaded standing transition.');
    } else assert.deepEqual(c.standStartRoot, [.02, 0, .70]);
  }
});

test('a released low-table object raises the crouched body before restoring the forward arms', async () => {
  for (const mode of ['hero_plus', 'other_policy']) for (const height of [.5, .74]) {
    const {c, solves, state, closures} = returnFixture({stage: 'clear_tray', elapsed: 1.5});
    const low = height < .60, forwardPosture = Array(29).fill(.2);
    Object.assign(c, {mode, phase: 'settle', phaseTime: 0, crouched: true,
      rootPosition: [.02, 0, .64], rootHeight: .64, rootPitch: .1,
      stableDeposit: 1, inTray: true, handContacts: 0, clearTrayTop: .8, clearTrayStable: .2});
    c.scene.tableTopZ = height;
    c.reference.qCurrent = forwardPosture.slice();
    c.reference.currentFrame = () => ({rootPosW: [.02, 0, .64], plannedRootPitch: .1});
    const livePose = Array.from(c.scene.data.qpos);
    await c.advanceClearTray();
    assert.equal(c.phase, 'return_home');
    assert.equal(c.homeStage, low ? 'root_recovery' : 'retract', `${mode} at ${height} m`);
    assert.deepEqual(closures, {right: 0, left: 0});
    assert.deepEqual(Array.from(c.scene.data.qpos), livePose, 'Planning the return must not teleport the live body.');
    assert.equal(c.returnSuccess, null);
    if (!low) {
      assert.ok(c.homePlanChecks.some(check => check.stage === 'retract'));
      assert.ok(solves.every(call => call.target.lockWaist === true), 'The crouched high-table arm withdrawal retains its waist.');
      continue;
    }
    assert.deepEqual(c.homeRetracted, {}, 'Low-table recovery must not first pull the arms into the crouched hips.');
    assert.ok(solves.length > 1);
    assert.ok(solves.every(call => call.target.trackPosture === true));
    assert.ok(solves.every(call => call.target.lockWaist !== true),
      'Low-table root restoration must remain distinct from the locked-waist Cartesian retract.');
    assert.ok(solves.every(call => call.target.posture.every((value, i) => value === forwardPosture[i])),
      'The root recovery reference must retain the reached forward-arm joint posture.');
    const from = c.homeTarget(0), upright = c.homeTarget(c.homeDuration);
    near(from.rootPosition, [.02, 0, .64]);
    near(upright.rootPosition, c.reference.ik.anchorPosition);
    near(upright.posture, forwardPosture);
    assert.notEqual(from.lockWaist, true); assert.notEqual(upright.lockWaist, true);
    const rootRecoverySolves = solves.length;

    // Once the body reference is upright and measured root motion is quiet,
    // continue through the real audited arm-restoration path.
    c.reference.currentFrame = () => ({rootPosW: c.reference.ik.anchorPosition.slice(), plannedRootPitch: 0});
    state.rootPosW = c.reference.ik.anchorPosition.slice();
    c.homeStable = .3;
    c.scene.data.time = c.homeTime + c.homeDuration + .1;
    await c.advanceReturn();
    assert.equal(c.homeStage, 'restore');
    assert.equal(c.crouched, false);
    assert.deepEqual(c.homeRestoreFrom, forwardPosture);
    near(c.homeTarget(10).posture, c.homeQ);
    assert.notEqual(c.homeTarget(10).lockWaist, true);
    assert.ok(solves.slice(rootRecoverySolves).every(call => call.target.trackPosture && call.target.lockWaist !== true));
    assert.notEqual(forwardPosture[14], c.homeTarget(10).posture[14], 'The later restore must still return the waist to its original posture.');
    assert.deepEqual(c.homePlanChecks.map(check => check.stage), ['root_recovery', 'restore']);
    assert.equal(c.returnSuccess, null);
  }
});

test('shrinking return clearance starts a fully audited withdrawal before the 5 mm hard guard', async () => {
  const {c, solves, audits} = returnFixture({stage: 'restore', elapsed: .1});
  c.homeClearance = .030;
  c.audit.hands = {left: [11], right: [22]};
  c.audit.check = () => ({passed: true, minimumClearance: .028, limitingPair: {a: 11, b: 0}});
  const from = c.plannedPalms();
  await c.advanceReturn();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.homeStage, 'clearance_recovery');
  assert.equal(c.homeClearanceRetries, 1);
  assert.ok(c.homeClearance > .005);
  assert.equal(solves.length, 61, 'Withdrawal must run endpoint IK and all 60 trajectory samples.');
  assert.equal(audits.length, 61, 'Every accepted pose must also be safe at the measured root.');
  assert.equal(solves[0].options.strictEndpoint, true);
  assert.equal(solves.at(-1).options.strictEndpoint, true);
  const target = c.homeTarget(10);
  near(target.palms.left.position, [from.left.position[0] - .035, from.left.position[1], from.left.position[2] + .05]);
  near(target.palms.right.position, from.right.position);
  near(target.rootPosition, c.rootPosition);
  assert.equal(c.returnSuccess, null);
});

test('a safely increasing return gap does not repeatedly trigger withdrawal', async () => {
  const {c, solves} = returnFixture({stage: 'restore', elapsed: .1});
  c.homeClearance = .019;
  c.audit.check = () => ({passed: true, minimumClearance: .02});
  await c.advanceReturn();
  assert.equal(c.homeStage, 'restore');
  assert.equal(solves.length, 0);
});

test('a slow hand near its natural rest clearance does not trigger spurious withdrawal', async () => {
  const {c, solves} = returnFixture({stage: 'restore', elapsed: .1});
  c.homeClearance = .0288;
  let clearance = .02877;
  c.audit.check = () => ({passed: true, minimumClearance: clearance});
  for (let i = 0; i < 20; i++) {
    clearance = .02877 - .00001 * Math.sin(i / 4);
    await c.advanceReturn();
    c.scene.data.time += c.dt;
  }
  assert.equal(c.homeStage, 'restore');
  assert.equal(solves.length, 0);
  assert.equal(c.completionFallback, false);
});

test('missing or nonfinite clearance history cannot invent an approaching speed', async () => {
  for (const previous of [undefined, null, Infinity, -Infinity, NaN]) {
    const {c, solves} = returnFixture({stage: 'restore', elapsed: .1});
    c.homeClearance = previous;
    c.audit.check = () => ({passed: true, minimumClearance: .02877});
    await c.advanceReturn();
    assert.equal(c.homeStage, 'restore', `Invalid previous clearance ${previous} must not trigger recovery.`);
    assert.equal(solves.length, 0);
  }
});

test('refused withdrawal IK does not start a second recovery search in the same stage visit', async () => {
  const {c, solves} = returnFixture({stage: 'restore', elapsed: .1});
  let clearance = .011;
  c.audit.check = () => ({passed: clearance >= .005, minimumClearance: clearance});
  const solve = c.reference.ik.solve;
  c.reference.ik.solve = (q, target, options) => {
    solve(q, target, options);
    throw new IKError('No supported withdrawal path.', {geometry: {passed: false}, comMargin: .01});
  };
  await c.advanceReturn();
  assert.equal(c.homeStage, 'restore');
  assert.equal(c.homeClearanceRecovery, null);
  assert.equal(solves.length, 2, 'Only the two bounded withdrawal candidates may be attempted.');
  assert.equal(c.homeRecoveryAttemptVisit, c.homeStageVisit || 0, 'The rejected early recovery consumed this stage visit.');
  clearance = .0046;
  c.scene.data.time += c.dt;
  await c.advanceReturn();
  assert.equal(c.phase, 'return_home');
  assert.equal(c.homeStage, 'restore');
  assert.equal(c.motionCompleted, false);
  assert.equal(solves.length, 2, 'The measured shortfall must not start another recovery IK search in the same stage visit.');
  assert.ok(c.completionReasons.includes('return_clearance_violation'));
  assert.equal(c.homeClearanceViolations, 1);
  assert.equal(c.returnSuccess, null);
});

test('withdrawal recovery resumes the audited return only after sustained measured clearance', async () => {
  const {c, solves} = returnFixture({stage: 'clearance_recovery', elapsed: 1.3});
  c.homeDuration = 1.2;
  c.audit.check = () => ({passed: true, minimumClearance: .02877});
  for (let i = 0; i < 9; i++) {
    await c.advanceReturn();
    c.scene.data.time += c.dt;
  }
  assert.equal(c.homeStage, 'clearance_recovery');
  assert.equal(solves.length, 0);
  for (let i = 0; i < 3 && c.homeStage === 'clearance_recovery'; i++) {
    await c.advanceReturn();
    c.scene.data.time += c.dt;
  }
  assert.equal(c.homeStage, 'retract');
  assert.equal(c.phase, 'return_home');
  assert.equal(solves.length, 111, 'Resuming return must audit its endpoint and full 2.2 second trajectory.');
  assert.equal(c.returnSuccess, null);
});

test('withdrawal recovery has a bounded retry count and measured-clearance timeout', async () => {
  const capped = returnFixture({stage: 'restore', elapsed: .1});
  capped.c.homeClearanceRetries = 3;
  assert.equal(await capped.c.beginHomeClearanceRecovery({passed: true, minimumClearance: .011}), false);
  assert.equal(capped.solves.length, 0);
  const {c, solves} = returnFixture({stage: 'clearance_recovery', elapsed: 4.22});
  c.homeDuration = 1.2;
  c.stableDeposit = 1; c.everPlaced = true; c.lastError = null;
  c.fail = () => assert.fail('A withdrawal timeout never stops the controller.');
  c.audit.check = () => ({passed: true, minimumClearance: .02});
  await c.advanceReturn();
  assert.ok(c.completionReasons.includes('return_clearance_recovery_timeout'));
  assert.equal(c.homeStage, 'retract', 'The timed-out withdrawal re-enters the audited return.');
  assert.equal(c.phase, 'return_home');
  assert.ok(solves.length > 0, 'Re-entering the return preflights its retract path again.');
  assert.equal(c.motionCompleted, false);
  assert.equal(c.depositSuccess, true); assert.equal(c.everPlaced, true);
  assert.equal(c.returnSuccess, null); assert.equal(c.success, null, 'beginReturn resets the result flags until the return completes.');
  assert.equal(c.lastError, null);
});

test('standing recovery continues a rejected endpoint through the explicit best-effort path', async () => {
  const {c, solves} = returnFixture();
  c.baseTarget = (palms, options) => ({palms, posture: c.restQ.slice(), ...options});
  const solve = c.reference.ik.solve;
  c.reference.ik.solve = (q, target, options) => {
    const result = solve(q, target, options);
    if (solves.length === 1) throw softEndpointFailure({footRotationError: .01});
    return result;
  };
  await c.beginStand();
  assert.equal(c.phase, 'stand');
  assert.equal(solves.length, 102, 'The retry must run its endpoint and each 20 ms standing sample.');
  assert.ok(solves.slice(1).every(call => call.options.strictEndpoint === false));
  assert.ok(c.completionReasons.includes('standing_pose_retry'));
  assert.equal(c.completionTransfer, false);
});

test('rejected standing support continues approximately while real measured contact still stops execution', async () => {
  for (const details of [{geometry: {passed: false}}, {comMargin: .019}, {footRotationError: .05}]) {
    const {c, solves} = returnFixture();
    c.baseTarget = (palms, options) => ({palms, posture: c.restQ.slice(), ...options});
    const solve = c.reference.ik.solve;
    c.reference.ik.solve = (q, target, options) => {
      const result = solve(q, target, options);
      if (!options.bestEffort) throw softEndpointFailure({footRotationError: .01, ...details});
      return approximateResult(result, details);
    };
    const preflight = c.preflightHomePath, reports = [];
    c.preflightHomePath = async (...args) => { const report = await preflight.apply(c, args); reports.push(report); return report; };
    await c.beginStand();
    assert.equal(c.phase, 'stand');
    assert.equal(c.completionTransfer, false);
    assert.equal(solves.length, 102);
    assert.ok(solves.slice(1).every(call => call.options.bestEffort === true));
    assert.equal(reports[0].passed, false);
    assert.equal(reports[0].violations.length, 101);
    let checked = false;
    c.reference.window = targetAt => {
      checked = true;
      assert.ok(targetAt(0).palms.right);
      return {};
    };
    await c.tick();
    assert.equal(checked, true);
    assert.equal(c.phase, 'stand');
    for (let i = 0; i < 7 && !c.pendingReturn; i++) { c.bodyFixtureForce = 20; c.scene.data.time += .02; await c.tick(); }
    assert.equal(c.pendingReturn, true);
    assert.equal(c.motionCompleted, false);
  }
});

async function appFixture() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {textContent: '', hidden: false, dataset: {}, firstChild: {textContent: ''},
        classList: {toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); },
          contains: name => classes.has(name), add: name => classes.add(name), remove: name => classes.delete(name)},
        addEventListener(type, listener) { (this.listeners ??= {})[type] = listener; }, setAttribute() {}, replaceChildren(...children) { this.children = children; }, getContext: () => ({}), clientWidth: 0});
    }
    return elements.get(id);
  };
  const stages = ['approach', 'close', 'lift', 'transfer', 'release', 'return_home'].map(phase => {
    const el = element(`phase-${phase}`); el.dataset.phase = phase; return el;
  });
  element('ik-activity').hidden = true;
  const sandbox = {
    document: {getElementById: element, createElement: tag => element(Symbol(tag)),
      querySelectorAll: selector => selector === '[data-phase]' ? stages : []},
    ClientBridge: class { async start() {} }, ResizeObserver: class { observe() {} },
    currentIKWorkspace: () => null, drawIKWorkspace() {}, ikWorkspaceStatus: () => '',
    clampPlacement, placementAxisRange, placementPolygon,
    setTimeout, clearTimeout,
  };
  const source = (await readFile(new URL('../app.js', import.meta.url), 'utf8')).replace(/^import .*;\n/gm, '');
  runInNewContext(`${source}\nglobalThis.renderControl = (control, items = []) => {
    ui.online = true; ui.loading = false; ui.loadFailed = false;
    ui.catalog = {objects: items, placement_region: {x_min: .3, x_max: .53, y_min: -.36, y_max: .36}};
    ui.state = {controller: control, table: {kind: 'workbench', height: .74}, objects: items};
    syncSelection();
    updateControls();
  };
  globalThis.appUI = ui; globalThis.setBridge = value => bridge = value;
  globalThis.issueCommand = command; globalThis.resetScene = resetScene;
  globalThis.captureSelection = () => {
    globalThis.selectionCommands = [];
    command = (path, body) => selectionCommands.push({path, body});
  };`, sandbox);
  return {sandbox, element, stages, elements};
}

test('the UI exposes neither a grasp style nor a wrist-angle override and sends no style commands', async () => {
  const {sandbox, elements} = await appFixture();
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.doesNotMatch(html, /id="ee-yaw"|id="ee-rotation-direction"|Wrist rotation/);
  assert.doesNotMatch(html, /id="grasp-style"|data-style=|Grasp style|Top Down/, 'The demo uses the Side grasp style.');
  sandbox.renderControl({phase: 'idle', busy: false}, [{id: 'apple', position: [.41, -.23, .8]}]);
  sandbox.captureSelection();
  assert.equal(sandbox.selectionCommands.length, 0, 'Rendering sends no configuration command.');
  assert.equal(elements.has('ee-yaw'), false, 'Rendering must not access a removed control.');
  assert.equal(elements.has('ee-rotation-direction'), false);
  assert.equal(elements.has('grasp-style'), false);
  assert.equal((await readFile(new URL('../app.js', import.meta.url), 'utf8')).includes('#grasp-style'), false, 'app.js no longer references the removed selector.');
});

test('the compact guide is visible by default and the header reports actual policy execution', async () => {
  const {sandbox, element} = await appFixture();
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const guide = html.match(/<div class="quick-guide"[\s\S]*?<\/div>/)?.[0];
  assert.ok(guide); assert.doesNotMatch(guide, /\bhidden\b|<details/);
  assert.equal((guide.match(/<li>/g) || []).length, 3);
  assert.match(guide, /icons to add or select objects/); assert.match(guide, /Drag objects in the top view/);
  assert.match(guide, /Object to pick/); assert.match(guide, /Pick &amp; place/); assert.match(guide, /Reset/);
  assert.match(html, /id="simulation-status" role="status" aria-live="polite" aria-atomic="true"/);
  assert.match(html, /id="clear-objects" title="Remove all objects from the scene"/);
  assert.match(html, /id="remove-object" class="text-button" title="Remove the selected object"/);
  const items = [{id: 'apple', position: [.41, -.23, .8]}];
  for (const phase of ['approach', 'close', 'lift', 'transfer', 'release', 'return_home']) {
    sandbox.renderControl({phase, busy: true}, items);
    assert.equal(element('simulation-status').textContent, 'Live MuJoCo simulation with HERO policy running');
  }
  for (const phase of ['idle', 'failed', 'succeeded']) {
    sandbox.renderControl({phase, busy: false}, items);
    assert.equal(element('simulation-status').textContent, 'Live MuJoCo simulation · HERO policy ready');
  }
  sandbox.renderControl({phase: 'idle', busy: false}, items); sandbox.captureSelection();
  element('remove-object').onclick();
  assert.equal(sandbox.selectionCommands.at(-1).path, '/api/remove');
  assert.equal(sandbox.selectionCommands.at(-1).body.kind, 'apple');
  element('clear-objects').onclick();
  assert.equal(sandbox.selectionCommands.at(-1).path, '/api/clear');
});

test('Reset bypasses a pending command and its stale rejection cannot unlock or overwrite the new scene', async () => {
  const {sandbox, element} = await appFixture();
  sandbox.renderControl({phase: 'approach', busy: true}, [{id: 'apple', position: [.41, -.23, .8]}]);
  let rejectCommand, resolveReset, resetCalls = 0;
  sandbox.setBridge({
    request: () => new Promise((resolve, reject) => {rejectCommand = reject;}),
    reset: () => {
      resetCalls++;
      rejectCommand(Object.assign(new Error('Canceled by Reset'), {code: 'SIMULATION_RESET'}));
      return new Promise(resolve => {resolveReset = resolve;});
    },
  });
  const command = sandbox.issueCommand('/api/grasp', {object_id: 'apple'});
  assert.equal(sandbox.appUI.pending, true); assert.equal(element('reset').disabled, false);
  const reset = element('reset').onclick();
  assert.equal(resetCalls, 1); assert.equal(element('reset').disabled, true);
  assert.equal(element('simulation-status').textContent, 'Live MuJoCo simulation · Resetting scene');
  await command;
  assert.equal(sandbox.appUI.pending, true, 'Old finally cannot unlock the resetting scene.');
  assert.equal(sandbox.appUI.resetting, true); assert.equal(element('toast').hidden, true);
  const fresh = {controller: {phase: 'idle', busy: false}, table: {kind: 'workbench', height: .74},
    objects: [{id: 'apple', position: [.41, -.23, .8]}], selectedObjectId: 'apple'};
  resolveReset(fresh); await reset;
  assert.equal(sandbox.appUI.state, fresh); assert.equal(sandbox.appUI.pending, false);
  assert.equal(sandbox.appUI.resetting, false); assert.equal(element('reset').disabled, false);
  assert.equal(element('grasp').disabled, false); assert.equal(element('pick-object').disabled, false);
  assert.equal(element('simulation-status').textContent, 'Live MuJoCo simulation · HERO policy ready');
  assert.equal(element('status-message').hidden, true); assert.equal(element('ik-activity').hidden, true);
});

test('a stale successful command cannot restore the old running state after Reset', async () => {
  const {sandbox, element} = await appFixture();
  sandbox.renderControl({phase: 'idle', busy: false}, [{id: 'apple', position: [.41, -.23, .8]}]);
  let resolveCommand;
  const fresh = {controller: {phase: 'idle', busy: false}, table: {kind: 'workbench', height: .74},
    objects: [{id: 'apple', position: [.41, -.23, .8]}], selectedObjectId: 'apple'};
  sandbox.setBridge({request: () => new Promise(resolve => {resolveCommand = resolve;}), reset: async () => fresh});
  const command = sandbox.issueCommand('/api/grasp', {object_id: 'apple'});
  await element('reset').onclick();
  resolveCommand({...fresh, controller: {phase: 'transfer', busy: true}}); await command;
  assert.equal(sandbox.appUI.state, fresh);
  assert.equal(element('grasp').disabled, false); assert.equal(element('toast').hidden, true);
});

test('the actual UI distinguishes placed-without-return and never marks an incomplete return done', async () => {
  const {sandbox, element, stages} = await appFixture();
  sandbox.captureSelection();
  const cases = [
    {control: {phase: 'failed', success: true, deposit_success: true, return_success: false, motion_completed: false},
      message: 'Placed · Return incomplete', terminal: true, returned: false},
    {control: {phase: 'succeeded', success: true, deposit_success: true, return_success: false, motion_completed: true},
      message: 'Placed · Return incomplete', terminal: true, returned: false},
    {control: {phase: 'succeeded', success: true, deposit_success: true, return_success: true, motion_completed: true},
      message: 'Attempt successful', terminal: true, returned: true},
    {control: {phase: 'failed', success: false, deposit_success: false, return_success: true, motion_completed: true},
      message: 'Attempt unsuccessful', terminal: true, returned: true},
    {control: {phase: 'return_home', busy: true, success: null, deposit_success: true, return_success: null},
      message: '', terminal: false, returned: false},
    {control: {phase: 'failed', success: false, deposit_success: false, return_success: true, motion_completed: true,
      failure_reason: {code: 'approach_obstructed'}}, message: 'Attempt unsuccessful',
      detail: 'Path blocked by another object. Could not find a collision-free IK solution. Please reset the scene.', terminal: true, returned: true},
    {control: {phase: 'return_home', busy: true, success: null, return_success: null,
      failure_reason: {code: 'approach_obstructed'}}, message: '', terminal: false, returned: false},
    {control: {phase: 'succeeded', success: true, deposit_success: true, return_success: true,
      failure_reason: {code: 'approach_obstructed'}}, message: 'Attempt successful', terminal: true, returned: true},
    {control: {phase: 'failed', success: false, return_success: true,
      failure_reason: {code: 'tracking_error'}}, message: 'Attempt unsuccessful', terminal: true, returned: true},
  ];
  for (const code of ['approach_obstructed', 'transfer_obstructed']) cases.push({
    control: {phase: 'failed', success: false, motion_completed: false, failure_reason: {code, reset_required: true}},
    message: 'Attempt unsuccessful', detail: 'Path blocked by another object. Could not find a collision-free IK solution. Please reset the scene.', terminal: true, returned: false,
  });
  cases.push({control: {phase: 'failed', success: true, ever_placed: true, in_tray: false,
    deposit_success: true, return_success: false, failure_reason: {code: 'transfer_obstructed', reset_required: true}},
    message: 'Placed · Return incomplete', terminal: true, returned: false});
  for (const {control, message, detail=message, terminal, returned} of cases) {
    sandbox.renderControl(control);
    assert.equal(element('status-message').textContent, detail);
    assert.equal(element('phase-pill').textContent, message);
    assert.equal(element('status-message').hidden, !terminal);
    assert.equal(element('phase-pill').hidden, !terminal);
    assert.equal(stages.at(-1).classList.contains('done'), returned);
    if (control.deposit_success) assert.equal(element('deposit-value').textContent, 'Placed');
  }
  const items = ['apple', 'mug'].map(id => ({id, label: id, position: [.41, -.23, .8]}));
  sandbox.renderControl({phase: 'idle', busy: false}, items);
  const choice = element('pick-object');
  assert.deepEqual(choice.children.map(option => option.value), ['', 'apple', 'mug']);
  assert.equal(choice.value, 'apple');
  assert.equal(choice.disabled, false);
  choice.listeners.change({target: {value: 'mug'}});
  assert.equal(sandbox.selectionCommands.at(-1).path, '/api/select');
  assert.equal(sandbox.selectionCommands.at(-1).body.object_id, 'mug');
  assert.equal(sandbox.selectionCommands.at(-1).body.text, undefined);
  sandbox.renderControl({phase: 'failed', busy: false, success: false}, items);
  assert.equal(choice.value, '', 'A finished attempt can select the same object again.');
  sandbox.renderControl({phase: 'approach', busy: true}, items);
  assert.equal(choice.disabled, true);
  const count = sandbox.selectionCommands.length;
  choice.listeners.change({target: {value: 'mug'}});
  assert.equal(sandbox.selectionCommands.length, count, 'Selection cannot interrupt a running grasp.');
  sandbox.renderControl({phase: 'idle'}, items.slice(1));
  assert.deepEqual(choice.children.map(option => option.value), ['', 'mug']);
  sandbox.renderControl({phase: 'idle'});
  assert.equal(choice.disabled, true);
  assert.deepEqual(choice.children.map(option => option.value), ['']);
});

test('the yaw slider offers only the carton band of the side it sits on, shows the recorded yaw on that side, and the full range for other objects', async () => {
  // The carton's yaw stays inside sign(y) * [40, 90] deg; a slider commit re-places the carton at its live position,
  // so the range follows the live side and a carried carton's recorded yaw is mirrored onto it.
  const {sandbox, element} = await appFixture();
  const carton = (y, yawDeg) => ({id: 'cracker_box', position: [.41, y, .85], placement_yaw: yawDeg * Math.PI / 180, footprint_radius: .0895});
  sandbox.renderControl({phase: 'idle', busy: false}, [carton(-.30, -55)]);
  assert.deepEqual([element('position-yaw').min, element('position-yaw').max, Number(element('position-yaw').value)], [-90, -40, -55]);
  assert.match(element('position-yaw').title, /−90° to −40°/);
  sandbox.renderControl({phase: 'idle', busy: false}, [carton(.03, -55)]); // deposited across the midline by the robot
  assert.deepEqual([element('position-yaw').min, element('position-yaw').max, Number(element('position-yaw').value)], [40, 90, 55]);
  assert.match(element('position-yaw').title, /40° to 90°/);
  sandbox.renderControl({phase: 'idle', busy: false}, [carton(.25, 40)]);
  assert.deepEqual([element('position-yaw').min, element('position-yaw').max, Number(element('position-yaw').value)], [40, 90, 40]);
  sandbox.renderControl({phase: 'idle', busy: false}, [{id: 'can', position: [.4, .2, .8], placement_yaw: -1.3, footprint_radius: .04}]);
  assert.deepEqual([element('position-yaw').min, element('position-yaw').max], [-90, 90]);
  assert.ok(Math.abs(Number(element('position-yaw').value) - (-1.3 * 180 / Math.PI)) < 1e-9); assert.equal(element('position-yaw').title, '');
});

test('an unsuccessful attempt names an IK or reach cause in the status line when the snapshot shows one', async () => {
  // A bare "Attempt unsuccessful" is not enough when IK / reach is the cause.
  const {sandbox, element} = await appFixture();
  const render = control => { sandbox.renderControl({phase: 'failed', success: false, return_success: true, motion_completed: true, message: 'Attempt unsuccessful. The hand has returned to the initial posture.', completion_reasons: [], ...control}); return element('status-message').textContent; };
  const beyond = render({failure_reason: {code: 'grasp_aborted', reason: 'alignment_retry_exhausted'}, closure_rejected_trigger: {distance_m: .0713}});
  assert.match(beyond, /^Attempt unsuccessful · IK could not bring the hand within reach of the grasp pose \(closest 7\.1 cm\)\. Move the object closer/);
  const shortClose = render({failure_reason: {code: 'grasp_aborted', reason: 'grasp_not_secured'}, grasp_success: false, completion_reasons: ['final_attempt_close', 'final_attempt'], final_attempt_trigger: {distance_m: .0514}, ga_frozen_reason: 'compensation_limit'});
  assert.match(shortClose, /IK could not reach the grasp pose \(arm at its reach limit\): the fingers closed 5\.1 cm short of the target and caught nothing\. Move the object closer/);
  const clearance = render({failure_reason: {code: 'motion_stopped', reason: 'The hand could not reach safe clearance. Move the object closer.'}, message: 'The hand could not reach safe clearance. Move the object closer. The hand has returned to the initial posture.'});
  assert.match(clearance, /IK found no approach with safe tabletop clearance for this placement\. Move the object closer/);
  const stopped = render({failure_reason: {code: 'motion_stopped', reason: 'The robot body contacted the table or tray. Motion stopped.'}, message: 'The robot body contacted the table or tray. Motion stopped. The hand has returned to the initial posture.'});
  assert.equal(stopped, 'Attempt unsuccessful · The robot body contacted the table or tray. Motion stopped.');
  const fell = render({failure_reason: {code: 'grasp_aborted', reason: 'object_fell'}});
  assert.match(fell, /The object fell or was knocked over during the grasp/);
  const notSecured = render({failure_reason: {code: 'grasp_aborted', reason: 'grasp_not_secured'}, grasp_success: false});
  assert.equal(notSecured, 'Attempt unsuccessful · The fingers closed on the object but could not hold it. Try a slightly different position or rotation.');
  const bestEffortStance = render({failure_reason: {code: 'grasp_aborted', reason: 'something_else'}, completion_reasons: ['reach_ik_best_effort'], stance_plan: {feasible: false}});
  assert.match(bestEffortStance, /IK could only approximate the reach to this object \(no fully feasible planted-foot stance\)/);
  // Nothing specific: the bare result stays; successes and reset-required stops keep their existing texts.
  assert.equal(render({failure_reason: {code: 'tracking_error'}}), 'Attempt unsuccessful');
  sandbox.renderControl({phase: 'succeeded', success: true, return_success: true, completion_reasons: ['final_attempt_close'], final_attempt_trigger: {distance_m: .04}});
  assert.equal(element('status-message').textContent, 'Attempt successful');
  sandbox.renderControl({phase: 'failed', success: false, motion_completed: false, failure_reason: {code: 'approach_obstructed', reset_required: true}, completion_reasons: ['final_attempt_close']});
  assert.equal(element('status-message').textContent, 'Path blocked by another object. Could not find a collision-free IK solution. Please reset the scene.');
  // During the attempt the notice still wins and no hint is shown.
  sandbox.renderControl({phase: 'approach', busy: true, success: null, attempt_notice: 'Re-aligning the open hand before a second approach (one retry).', completion_reasons: ['final_attempt_close']});
  assert.equal(element('status-message').textContent, 'Re-aligning the open hand before a second approach (one retry).');
});

test('the standalone page carries the homepage "Controls & tips" (six steps and both notes) so a new tab keeps them', async () => {
  // The tips must not disappear when the demo is opened in its own tab.
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const tips = html.match(/<section class="control-section tips-section"[\s\S]*?<\/section>/)?.[0];
  assert.ok(tips, 'tips section present'); assert.doesNotMatch(tips, /\bhidden\b|<details/);
  assert.equal((tips.match(/<li>/g) || []).length, 6);
  for (const text of ['Add objects', 'Arrange the tabletop', 'Select a target', 'Run the attempt', 'Reset or edit', 'Inspect the motion', 'Browser IK', 'Keep the tray clear', 'Pick &amp; place', 'Object to pick'])
    assert.match(tips, new RegExp(text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.doesNotMatch(tips, /Top Down|Grasp style|Wrist rotation|Apple/);
  const css = await readFile(new URL('../style.css', import.meta.url), 'utf8');
  assert.match(css, /\.tips-section\{/); assert.match(css, /\.tips-note\{/);
});
