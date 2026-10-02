import assert from 'node:assert/strict';
import test from 'node:test';
import {PRESENTATION_HZ, PresentationClock, PresentationBuffer} from '../presentation.js';

function snapshot(time, x = 0, activeObjectId = 'uiuc_i') {
  return {time, activeObjectId,
    geomPositions: new Float32Array([x, 0, 1]),
    geomMatrices: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    geomRGBA: new Float32Array([1, 1, 1, 1]),
    bodyPositions: new Float32Array([x, 0, 1]),
    bodyQuaternions: new Float32Array([1, 0, 0, 0])};
}

function renderedCallbacks(callbackHz, durationMs, presentationHz = PRESENTATION_HZ) {
  const clock = new PresentationClock({hz: presentationHz}), draws = [];
  for (let i = 0; i * 1000 / callbackHz < durationMs; i++) {
    const timestamp = i * 1000 / callbackHz;
    if (clock.shouldRender(timestamp)) draws.push(timestamp);
  }
  return draws;
}

test('a 90 Hz presentation clock caps fast displays and respects a slower RAF', () => {
  assert.equal(PRESENTATION_HZ, 90);
  for (const hz of [30, 60, 90, 120, 144, 165]) {
    const draws = renderedCallbacks(hz, 10_000);
    assert.equal(draws.length, Math.min(hz, 90) * 10, `${hz} Hz callbacks`);
    assert.equal(new Set(draws).size, draws.length);
  }
});

test('an explicit 60 Hz clock retains its cap on every tested display rate', () => {
  for (const hz of [30, 60, 90, 120, 144, 165])
    assert.equal(renderedCallbacks(hz, 10_000, 60).length, Math.min(hz, 60) * 10, `${hz} Hz callbacks`);
});

test('near-60/90 Hz callback drift and small jitter do not halve presentation rate', () => {
  assert.equal(renderedCallbacks(59.94, 10_000).length, 600);
  assert.equal(renderedCallbacks(89.91, 10_000).length, 900);
  const clock = new PresentationClock();
  for (let i = 0; i < 270; i++) {
    const timestamp = i * 1000 / 90 + (i % 3 - 1) * .25;
    assert.equal(clock.shouldRender(timestamp), true, `callback ${i}`);
  }
});

test('a missed deadline draws once and discards GPU catch-up work', () => {
  const clock = new PresentationClock();
  assert.equal(clock.shouldRender(0), true);
  assert.equal(clock.shouldRender(7), false);
  assert.equal(clock.shouldRender(1000), true);
  for (const timestamp of [1000, 1000.1, 1001, 1008]) assert.equal(clock.shouldRender(timestamp), false);
  assert.equal(clock.shouldRender(1000 + 1000 / PRESENTATION_HZ), true);
  assert.equal(clock.shouldRender(1040), true);
  assert.equal(clock.shouldRender(1040.1), false);
});

test('clock resets recover from lifecycle restarts and invalid callback times', () => {
  const clock = new PresentationClock();
  assert.equal(clock.shouldRender(NaN), false);
  assert.equal(clock.shouldRender(100), true);
  assert.equal(clock.shouldRender(Infinity), false);
  assert.equal(clock.shouldRender(101), false);
  assert.equal(clock.shouldRender(90), true);
  assert.equal(clock.shouldRender(90), false);
  clock.reset();
  assert.equal(clock.shouldRender(90), true);
  assert.throws(() => new PresentationClock({hz: 0}), RangeError);
});

test('the delayed buffer interpolates observed arrivals and holds both endpoints', () => {
  const buffer = new PresentationBuffer({delayMs: 60}), a = snapshot(0), b = snapshot(.04, .04), c = snapshot(.08, .08);
  assert.equal(buffer.sample(0), null);
  assert.equal(buffer.push(a, 100), true);
  assert.equal(buffer.push(b, 140), false);
  assert.equal(buffer.push(c, 180), false);
  assert.deepEqual(buffer.sample(100), {from: a, to: a, alpha: 1});
  assert.deepEqual(buffer.sample(180), {from: a, to: b, alpha: .5});
  assert.deepEqual(buffer.sample(220), {from: b, to: c, alpha: .5});
  assert.deepEqual(buffer.sample(1000), {from: c, to: c, alpha: 1});
});

test('variable arrival gaps are sampled by wall time without extrapolating motion', () => {
  const buffer = new PresentationBuffer({delayMs: 40}), a = snapshot(0), b = snapshot(.04, .04), c = snapshot(.06, .06);
  buffer.push(a, 0); buffer.push(b, 80); buffer.push(c, 100);
  assert.deepEqual(buffer.sample(80), {from: a, to: b, alpha: .5});
  assert.deepEqual(buffer.sample(130), {from: b, to: c, alpha: .5});
  assert.deepEqual(buffer.sample(500), {from: c, to: c, alpha: 1});
});

test('equal arrival timestamps retain only the newest observation at that instant', () => {
  const buffer = new PresentationBuffer({delayMs: 0}), a = snapshot(0), b = snapshot(.02, .02), c = snapshot(.04, .04);
  buffer.push(a, 0); buffer.push(b, 20);
  assert.equal(buffer.push(c, 20), false);
  assert.deepEqual(buffer.sample(10), {from: a, to: c, alpha: .5});
  assert.deepEqual(buffer.sample(20), {from: c, to: c, alpha: 1});
  assert.equal(buffer.samples.length, 2);
});

test('history stays bounded while retaining the required pair and latest endpoint', () => {
  const buffer = new PresentationBuffer({delayMs: 60, maxSamples: 3});
  const a = snapshot(0), b = snapshot(.02, .02);
  buffer.push(a, 0); buffer.push(b, 10);
  let latest;
  for (let i = 2; i <= 20; i++) { latest = snapshot(i * .02, i * .01); buffer.push(latest, 10 + i); }
  assert.equal(buffer.samples.length, 3);
  assert.deepEqual(buffer.sample(65), {from: a, to: b, alpha: .5});
  assert.deepEqual(buffer.sample(1000), {from: latest, to: latest, alpha: 1});
  assert.throws(() => new PresentationBuffer({maxSamples: 2}), RangeError);
});

test('a receipt newer than the RAF timestamp cannot trim its needed display pair', () => {
  const buffer = new PresentationBuffer({delayMs: 60});
  const a = snapshot(0), b = snapshot(.04, .04), c = snapshot(.08, .08), d = snapshot(.1, .1);
  buffer.push(a, 0); buffer.push(b, 40); buffer.push(c, 80);
  assert.deepEqual(buffer.sample(90), {from: a, to: b, alpha: .75});
  buffer.push(d, 100);
  assert.deepEqual(buffer.sample(95), {from: a, to: b, alpha: .875});
  assert.deepEqual(buffer.sample(115), {from: b, to: c, alpha: .375});
});

test('simulation resets, duplicate times, and invalid time metadata snap immediately', () => {
  for (const time of [-1, 0, NaN, Infinity, undefined]) {
    const buffer = new PresentationBuffer(), a = snapshot(0), b = snapshot(time, .1);
    buffer.push(a, 0);
    assert.equal(buffer.push(b, 20), true, `time ${time}`);
    assert.deepEqual(buffer.sample(20), {from: b, to: b, alpha: 1});
  }
});

test('long IK pauses and a backwards wall clock never animate an old interval', () => {
  for (const timestamp of [-1, 181, NaN]) {
    const buffer = new PresentationBuffer(), a = snapshot(0), b = snapshot(.02, .02);
    buffer.push(a, 0);
    assert.equal(buffer.push(b, timestamp), true);
    assert.deepEqual(buffer.sample(500), {from: b, to: b, alpha: 1});
  }
  const buffer = new PresentationBuffer();
  buffer.push(snapshot(0), NaN);
  const valid = snapshot(.02, .02);
  assert.equal(buffer.push(valid, 20), true);
  assert.deepEqual(buffer.sample(20), {from: valid, to: valid, alpha: 1});
  const later = snapshot(.04, .04);
  buffer.push(later, 40); buffer.sample(90);
  assert.deepEqual(buffer.sample(80), {from: later, to: later, alpha: 1}, 'A backwards RAF clock snaps instead of replaying history backwards.');
});

test('model shape, active object, and visibility changes are discrete snapshots', () => {
  const mutations = [
    b => { b.geomPositions = new Float32Array(6); },
    b => { b.bodyQuaternions = new Float32Array(8); },
    b => { b.geomQuaternions = new Float32Array(4); },
    b => { b.activeObjectId = 'can'; },
    b => { b.geomRGBA[3] = 0; },
  ];
  for (const mutate of mutations) {
    const buffer = new PresentationBuffer(), a = snapshot(0), b = snapshot(.02, .02);
    buffer.push(a, 0); mutate(b);
    assert.equal(buffer.push(b, 20), true);
    assert.deepEqual(buffer.sample(20), {from: b, to: b, alpha: 1});
  }
});

test('teleports use complete translation distance in every geom and body', () => {
  for (const key of ['geomPositions', 'bodyPositions']) {
    const buffer = new PresentationBuffer(), a = snapshot(0), b = snapshot(.02);
    b[key].set([.4, .4, 1]);
    buffer.push(a, 0);
    assert.equal(buffer.push(b, 20), true, key);
    assert.deepEqual(buffer.sample(20), {from: b, to: b, alpha: 1});
  }
  const buffer = new PresentationBuffer(), a = snapshot(0), boundary = snapshot(.02, .5);
  buffer.push(a, 0);
  assert.equal(buffer.push(boundary, 20), false, 'Exactly half a meter remains inside the stated bound.');
});

test('sampling never mutates snapshots or substitutes synthetic snapshot data', () => {
  const a = snapshot(0), b = snapshot(.04, .04), savedA = structuredClone(a), savedB = structuredClone(b);
  Object.freeze(a); Object.freeze(b);
  const buffer = new PresentationBuffer(); buffer.push(a, 100); buffer.push(b, 140);
  for (const timestamp of [160, 170, 180, 190, 200]) {
    const result = buffer.sample(timestamp);
    assert.ok([a, b].includes(result.from)); assert.ok([a, b].includes(result.to));
    assert.ok(Number.isFinite(result.alpha) && result.alpha >= 0 && result.alpha <= 1);
  }
  assert.deepEqual(a, savedA); assert.deepEqual(b, savedB);
  buffer.clear(); assert.equal(buffer.sample(1000), null);
  assert.equal(buffer.push(b, 1000), true);
});
