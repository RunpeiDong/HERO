import assert from 'node:assert/strict';
import test from 'node:test';
import {WholeBodyIK, IKError} from '../ik.js';
import {eye3} from '../numerics.js';

// One upper-body joint moves a palm and its collision surface toward a fixed
// plane. Run the real bounded DLS and geometry line search; linear FK isolates
// the signed-margin regression from policy tracking and native mesh details.
function fixture(requiredMargin, start = .002) {
  const nv = 29, joint = 22, qadr = Array.from({length: nv}, (_, i) => i + 7);
  const q = Array(nv).fill(0); q[joint] = start;
  const data = {qpos: new Float64Array(36), xpos: new Float64Array(6),
    xmat: new Float64Array([...eye3(), ...eye3()]), subtree_com: new Float64Array([0, 0, .8])};
  const jac = new Float64Array(3 * nv); jac[joint] = 1;
  const distance = () => data.qpos[qadr[joint]];
  const ik = Object.assign(Object.create(WholeBodyIK.prototype), {
    data, qadr, vadr: Array.from({length: nv}, (_, i) => i), model: {nv}, rootBodyId: 0,
    lower: Array(nv).fill(-1), upper: Array(nv).fill(1), optimizationBuffer: .002,
    supportPolygon: [[-.2, -.1], [.2, -.1], [.2, .1], [-.2, .1]],
    feet: {left: {position: [0, 0, 0], rotation: eye3()}, right: {position: [0, 0, 0], rotation: eye3()}},
    scene: {ankleBodyIds: {left: 0, right: 0}, palmBodyIds: {left: 1, right: 1},
      jacBody: (data, body) => ({position: body === 1 ? jac : new Float64Array(3 * nv), rotation: new Float64Array(3 * nv)}),
      geomDistanceInfo: () => ({distance: distance(), fromto: [distance(), 0, 0, 0, 0, 0]})},
    goals: target => target.palms, collisionOptions: () => ({}),
    collisionPointJacobian: (data, point, geom) => geom === 0 ? jac : new Float64Array(3 * nv),
    audit: {check: () => {
      const d = distance(), surplus = d - requiredMargin;
      return {passed: surplus >= -1e-8, minimumClearance: d, minimumClearanceSurplus: surplus,
        minimumBufferedSurplus: surplus - (requiredMargin > 0 ? .002 : 0),
        near: [{a: 0, b: 1, distance: d, requiredMargin}]};
    }},
    setPose(value) {
      qadr.forEach((address, i) => { data.qpos[address] = value[i]; });
      data.xpos.set([value[joint], 0, 1], 3);
    },
  });
  return {ik, q, joint, target: {posture: q.slice(), palms: {right: {position: [start, 0, 1], rotation: eye3()}}}};
}

const options = {iterations: 12, maxStep: .035, continuousCom: true, strictEndpoint: true};

test('permitting shell overlap does not add a 25 mm avoidance target to an already settled palm', () => {
  for (const margin of [0, -.003, -.012]) {
    const {ik, q, target} = fixture(margin);
    const result = ik.solve(q, target, options);
    assert.deepEqual(result.q, q);
    assert.equal(result.residual.palms.right.positionError, 0);
    assert.equal(result.geometry.passed, true);
  }
});

test('a tolerated shell overlap still prefers separation without pushing to the general avoidance radius', () => {
  const {ik, q, joint, target} = fixture(-.003, -.0015);
  const result = ik.solve(q, target, options);
  assert.ok(result.q[joint] > q[joint], 'The zero-clearance objective should relieve the overlap.');
  assert.ok(result.q[joint] <= 0, 'The soft target should not push past the plane.');
  assert.ok(result.residual.palms.right.positionError < .0015);
  assert.equal(result.geometry.passed, true);
});

test('ordinary positive clearances retain the existing 25 mm soft avoidance objective', () => {
  const {ik, q, joint, target} = fixture(.005, .007);
  const result = ik.solve(q, target, options);
  assert.ok(result.q[joint] > .012, 'Ordinary obstacle avoidance must still move the palm away.');
  assert.ok(result.q[joint] < .025);
  assert.equal(result.geometry.passed, true);
});

test('negative soft-margin handling does not admit a pose beyond its hard overlap tolerance', () => {
  const {ik, q, target} = fixture(-.003, -.004);
  assert.throws(() => ik.solve(q, target, {...options, maxStep: 0}),
    error => error instanceof IKError && error.details.geometry?.passed === false);
});
