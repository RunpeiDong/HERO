import {IKError, bodyPosition} from './ik.js';
import {matVec} from './numerics.js';

const radians = degrees => degrees * Math.PI / 180;
const finiteVector = (value, length) => value?.length === length && Array.from(value).every(Number.isFinite);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const distance = (a, b) => Math.hypot(...a.map((value, index) => value - b[index]));
const yieldBrowser = () => new Promise(resolve => setTimeout(resolve, 0));

// A small deterministic catalogue, not a Cartesian product. All offsets use
// the immutable reset heading and feet. Low targets retain the original
// 8–14 cm crouch range; ordinary targets first try small standing changes.
function stanceSeeds(low, side) {
  const neutral = [0, 0, 0, 0]; // forward, lateral, height drop, total lean degrees
  if (low) return [neutral,
    ...[.08, .10, .12, .14].map(drop => [0, 0, drop, 0]),
    ...[.08, .12].map(drop => [0, 0, drop, 10]),
    ...[.08, .10, .12, .14].map(drop => [0, 0, drop, 20]),
    ...[.08, .12].map(drop => [0, 0, drop, 30]),
    ...[.10, .14].map(drop => [0, 0, drop, 40]),
    ...[.12, .14].map(drop => [0, 0, drop, 50]),
    [.02, 0, .08, 10], [.04, 0, .08, 10], [.02, 0, .10, 20], [.04, 0, .10, 20],
    [.02, side * .02, .10, 20], [.04, side * .02, .12, 20], [0, side * .02, .10, 20],
    [.02, 0, .12, 30], [.04, 0, .12, 30], [.02, side * .02, .12, 30],
    [.04, side * .02, .14, 30], [0, side * .02, .14, 40],
    [.02, 0, .14, 40], [.04, 0, .14, 40], [.02, side * .02, .14, 50]];
  // Root and waist pitch need not increase together. Moving the hips slightly
  // back leaves COM room for a waist bend, while a small root pitch preserves
  // leg tracking. Keep these within the ordinary 24-candidate budget.
  const waistBend = (forward, drop, rootPitchDegrees, lateral = 0) =>
    ({forward, lateral, drop, rootPitchDegrees, waistPitchDegrees: 25});
  return [neutral,
    [.02, 0, 0, 0], [.04, 0, 0, 0], [0, side * .02, 0, 0], [.02, side * .015, 0, 0],
    [0, 0, .02, 10], [.02, 0, .02, 10], [.04, 0, .02, 10],
    [.02, side * .02, .02, 10], [.04, side * .02, .02, 10],
    [0, 0, .04, 20], [.02, 0, .04, 20], [.04, 0, .04, 20],
    [0, side * .02, .04, 20], [.02, side * .02, .04, 20], [.04, side * .02, .04, 20],
    waistBend(-.02, .02, 0), waistBend(-.02, .02, 8),
    waistBend(-.02, .04, 4), waistBend(-.02, .06, 0),
    waistBend(0, .02, 0), waistBend(0, .02, 8),
    waistBend(-.02, .02, 8, side * .015), waistBend(-.02, .04, 4, side * .015),
    [0, 0, .08, 40], [.02, 0, .08, 40], [.04, 0, .08, 40],
    [0, side * .02, .08, 40], [.02, side * .02, .08, 40],
    [0, 0, .10, 50], [.02, 0, .10, 50], [.04, 0, .10, 50]];
}

function makeCandidate(ik, target, seed, neutral) {
  const explicit = !Array.isArray(seed);
  const [forward, lateral, drop, lean] = explicit
    ? [seed.forward, seed.lateral, seed.drop, seed.rootPitchDegrees + seed.waistPitchDegrees]
    : seed;
  const delta = matVec(ik.anchorRotation, [forward, lateral, 0]);
  const rootPosition = [ik.anchorPosition[0] + delta[0], ik.anchorPosition[1] + delta[1], ik.anchorPosition[2] - drop];
  const rootPitch = radians(explicit ? seed.rootPitchDegrees : lean * .4), posture = Array.from(target.posture);
  // This is a soft rest-posture preference. The original native joint bounds
  // and IK joint margin remain the only hard waist limits.
  const waistPitchTarget = neutral ? posture[14] : clamp(Math.min(radians(explicit ? seed.waistPitchDegrees : lean * .6), .50), ik.lower[14], ik.upper[14]);
  posture[14] = waistPitchTarget;
  const id = `stance:${[forward, lateral, drop, rootPitch, waistPitchTarget].map(value => Math.round(value * 1e6)).join(':')}`;
  return {id, neutral, rootPosition, rootHeight: rootPosition[2], rootPitch,
    waistPitchTarget, forward, lateral, drop, leanDegrees: lean, waistDominant: explicit, posture};
}

// Reuse the controller's exact posture catalogue for static reach sampling.
// Feasibility-only queries may stop after the first witnessed solution.
export function reachStanceCandidates(ik, target, {maxCandidates=24,tableHeight=ik.scene.tableTopZ}={}) {
  if(!Number.isInteger(maxCandidates)||maxCandidates<1||maxCandidates>32)throw new TypeError('Use 1–32 stance candidates.');
  const low=tableHeight<.60||(target.palms[target.hand]?.position[2]??Infinity)<.78;
  return stanceSeeds(low,target.hand==='left'?1:-1).slice(0,maxCandidates).map((seed,index)=>makeCandidate(ik,target,seed,index===0));
}

function isSamePose(candidate, previous, solved = false) {
  if (!previous) return false;
  if (typeof previous === 'string') return candidate.id === previous;
  if (candidate.id === previous.id) return true;
  if (!finiteVector(previous.rootPosition, 3) || !Number.isFinite(previous.rootPitch)) return false;
  if (distance(candidate.rootPosition, previous.rootPosition) >= (solved ? .0075 : .001)) return false;
  if (Math.abs(candidate.rootPitch - previous.rootPitch) >= (solved ? .025 : .001)) return false;
  if (solved && finiteVector(previous.posture, 29))
    return Math.max(...candidate.posture.slice(12, 15).map((value, index) => Math.abs(value - previous.posture[index + 12]))) < .04;
  return Number.isFinite(previous.waistPitchTarget) && Math.abs(candidate.waistPitchTarget - previous.waistPitchTarget) < .001;
}

function armReachMetrics(ik, hand) {
  const scene = ik.scene;
  if (!ik.data?.xpos || !scene.model?.nbody || !scene.bodyName || !scene.palmBodyIds) return null;
  const find = name => Array.from({length: scene.model.nbody}, (_, index) => index)
    .find(index => scene.bodyName(index) === name);
  const shoulderId = find(`${hand}_shoulder_yaw_link`), elbowId = find(`${hand}_elbow_link`);
  if (shoulderId == null || elbowId == null || scene.palmBodyIds[hand] == null) return null;
  const shoulder = bodyPosition(ik.data, shoulderId), elbow = bodyPosition(ik.data, elbowId);
  const palm = bodyPosition(ik.data, scene.palmBodyIds[hand]);
  const available = distance(shoulder, elbow) + distance(elbow, palm), used = distance(shoulder, palm);
  if (!(available > .1)) return null;
  return {armExtensionRatio: used / available, armReachReserveM: Math.max(0, available - used)};
}

function scoreStance(ik, candidate, solution, previous, hand, headroomCosts) {
  const q = Array.from(solution.q), residual = solution.residual;
  const margins = q.map((value, index) => Math.min(value - ik.lower[index], ik.upper[index] - value));
  const normalizedMargins = margins.map((margin, index) => margin / Math.max(.1, ik.upper[index] - ik.lower[index]));
  const jointMargin = Math.min(...margins), waistMargin = Math.min(...margins.slice(12, 15));
  const nearLimit = normalizedMargins.map(margin => (Math.max(0, .15 - margin) / .15) ** 2);
  const palms = Object.values(residual.palms);
  const palmPositionError = Math.max(...palms.map(palm => palm.positionError));
  const palmRotationError = Math.max(...palms.map(palm => palm.rotationError));
  const armReach = armReachMetrics(ik, hand);
  const scoreComponents = {
    jointLimits: 3 * Math.max(...nearLimit) + nearLimit.reduce((sum, value) => sum + value, 0) / q.length,
    waistLimits: 2 * Math.max(...nearLimit.slice(12, 15)),
    // Body motion has a real tracking cost even when static IK allows it.
    bodyMotion: .9 * ((candidate.forward / .04) ** 2 + (candidate.lateral / .02) ** 2)
      + .8 * (candidate.drop / .14) ** 2 + .9 * (candidate.rootPitch / radians(20)) ** 2,
    waistMotion: .5 * q.slice(12, 15).reduce((sum, value, index) => sum + ((value - ik.defaultQ[index + 12]) / .5) ** 2, 0),
    // Palm error alone can hide a nearly straight arm or a large trunk twist.
    // Low tables retain their validated upright-crouch cost and preference.
    armExtension: headroomCosts && armReach ? .8 * (Math.max(0, armReach.armExtensionRatio - .93) / .05) ** 2 : 0,
    waistTwist: headroomCosts ? .6 * (Math.max(0, Math.abs(q[12] - ik.defaultQ[12]) - radians(20)) / .5) ** 2
      + (Math.max(0, Math.abs(q[13] - ik.defaultQ[13]) - radians(10)) / .3) ** 2 : 0,
    balance: .8 * (Math.max(0, .05 - residual.comMargin) / .03) ** 2,
    endpoint: .3 * (palmPositionError / .012) ** 2 + .1 * (palmRotationError / .12) ** 2,
    previousMotion: previous && finiteVector(previous.rootPosition, 3)
      ? .15 * (distance(candidate.rootPosition, previous.rootPosition) / .05) ** 2 : 0,
  };
  return {posture: q, score: Object.values(scoreComponents).reduce((sum, value) => sum + value, 0),
    scoreComponents, jointMargin, waistMargin, minimumNormalizedJointMargin: Math.min(...normalizedMargins),
    comMargin: residual.comMargin, palmPositionError, palmRotationError, ...armReach,
    residual, geometry: solution.geometry};
}

/**
 * Find a conservative planted-foot reach stance using native geometry IK.
 * Lower score is better; scores estimate posture/constraint headroom, not
 * dynamic policy success. The caller must audit and execute the complete
 * transition, then verify actual support and tracking before reaching.
 *
 * `previous` excludes both the previous seed and essentially unchanged solved
 * poses. `exclude` accepts returned candidates or candidate IDs. Search runs
 * once per request, never per physics frame. Only IK scratch data is written.
 */
export async function planReachStance(ik, target, {
  previous = null, exclude = [], maxCandidates = 24, iterations = 80,
  onProgress = null, yieldEvery = 1, tableHeight = ik.scene.tableTopZ,
  neutralPreference = .15,
} = {}) {
  if (!finiteVector(ik.anchorPosition, 3) || !finiteVector(ik.anchorRotation, 9)
      || !finiteVector(ik.defaultQ, 29) || !finiteVector(ik.lower, 29) || !finiteVector(ik.upper, 29)
      || !finiteVector(target?.posture, 29) || !Object.keys(target?.palms ?? {}).length)
    throw new TypeError('Reach stance planning requires native IK anchors, limits, posture, and palm targets.');
  if (!Number.isInteger(maxCandidates) || maxCandidates < 1 || maxCandidates > 32
      || !Number.isInteger(iterations) || iterations < 1 || iterations > 150
      || !Number.isInteger(yieldEvery) || yieldEvery < 1 || yieldEvery > 32
      || !Array.isArray(exclude) || !Number.isFinite(neutralPreference) || neutralPreference < 0)
    throw new TypeError('Use 1–32 stance candidates, 1–150 IK iterations, and a positive browser yield interval.');
  const hand = target.hand ?? 'right', palmHeight = target.palms[hand]?.position?.[2] ?? Infinity;
  const low = tableHeight < .60 || palmHeight < .78;
  const seeds = stanceSeeds(low, hand === 'left' ? 1 : -1), candidates = [];
  const collisionMargin = Math.max(.005, ik.collisionMargin ?? .005, target.collisionMargin ?? .005);
  const excluded = previous ? [...exclude, previous] : exclude;
  let best = null, neutral = null, upright = null, evaluated = 0, solveAttempts = 0;
  for (let index = 0; index < seeds.length && evaluated < maxCandidates; index++) {
    const candidate = makeCandidate(ik, target, seeds[index], index === 0);
    if (excluded.some(item => isSamePose(candidate, item))) {
      candidates.push({...candidate, feasible: false, excluded: true, reason: 'Previously attempted stance.'});
      continue;
    }
    evaluated++;
    let row;
    try {
      const solveTarget = {...target, posture: candidate.posture,
        rootPosition: candidate.rootPosition, rootHeight: candidate.rootHeight, rootPitch: candidate.rootPitch,
        collisionMargin, trackPosture: false, lockWaist: false};
      const solve = start => { solveAttempts++; return ik.solve(start, solveTarget,
        {iterations, maxStep: 3, strictEndpoint: true}); };
      let solution, solverStart = 'default';
      try { solution = solve(ik.defaultQ); }
      catch (error) {
        if (!(error instanceof IKError)) throw error;
        // A changed root with reset leg angles can begin with a shin inside
        // the floor. One already accepted endpoint is a bounded numerical
        // alternative; the same geometry, feet and endpoint gates still apply.
        const seed = candidate.waistDominant && candidates.filter(item => item.feasible)
          .sort((a, b) => distance(a.rootPosition, candidate.rootPosition) + .1 * Math.abs(a.rootPitch - candidate.rootPitch)
            - distance(b.rootPosition, candidate.rootPosition) - .1 * Math.abs(b.rootPitch - candidate.rootPitch))[0];
        if (!seed) throw error;
        solution = solve(seed.posture); solverStart = seed.id;
      }
      const residual = solution.residual, palms = Object.values(residual?.palms ?? {});
      if (!finiteVector(solution.q, 29) || !solution.geometry?.passed || !palms.length
          || !(residual.footPositionError < .003) || !(residual.footRotationError < .05)
          || !(residual.comMargin >= .02) || !palms.every(palm => palm.positionError < .012 && palm.rotationError < .12))
        throw new IKError('The reach stance did not pass endpoint, geometry, feet, and COM checks.', residual);
      row = {...candidate, ...scoreStance(ik, candidate, solution, previous, hand, !low), solverStart, feasible: true};
      if (previous && isSamePose(row, previous, true)) {
        row.excluded = true; row.reason = 'The solved posture is unchanged from the previous attempt.';
      }
      if (row.neutral) neutral = row;
      if (!row.excluded && row.forward === 0 && row.lateral === 0 && row.rootPitch === 0
          && (!upright || row.score < upright.score)) upright = row;
      if (!row.excluded && (!best || row.score < best.score)) best = row;
    } catch (error) {
      if (!(error instanceof IKError)) throw error;
      row = {...candidate, feasible: false, reason: error.message, residual: error.details ?? null};
      if (row.neutral) neutral = row;
    }
    candidates.push(row);
    onProgress?.(evaluated, maxCandidates, row);
    if (evaluated % yieldEvery === 0) await yieldBrowser();
  }
  // A small static score advantage does not justify adding lean to an already
  // feasible upright crouch. Prefer height-only motion on similarly scored
  // candidates, then preserve the original standing posture when feasible.
  // Keeping an almost extended arm has less value when feedback may need a
  // few extra centimetres of reach. Scale the preference by actual geometry,
  // preserving its original value for low tables and >=4 cm reach reserve.
  const preferenceFor = row => !low && Number.isFinite(row.armReachReserveM)
    ? neutralPreference * clamp(row.armReachReserveM / .04, 0, 1) : neutralPreference;
  if (upright && best && upright.score <= best.score + preferenceFor(upright)) best = upright;
  if (neutral?.feasible && !neutral.excluded && best && neutral.score <= best.score + preferenceFor(neutral)) best = neutral;
  const common = {candidates, neutral, evaluated, solveAttempts, scope: 'Static planted-foot feasibility; dynamic policy execution must be verified separately.'};
  if (!best) return {feasible: false, ...common, message: 'No alternative safe planted-foot stance reaches this object.'};
  return {...best, ...common, selectionReason: best.neutral ? 'neutral_preserved' : best === upright ? 'upright_preserved' : 'lower_constraint_and_motion_cost'};
}
