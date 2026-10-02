/** Presentation clocks never advance the simulation or predict a future pose. */
export const PRESENTATION_HZ = 90;

export class PresentationClock {
  constructor({hz = PRESENTATION_HZ} = {}) {
    if (!Number.isFinite(hz) || hz <= 0) throw new RangeError('Presentation frequency must be positive and finite.');
    this.intervalMs = 1000 / hz;
    this.toleranceMs = Math.min(.75, this.intervalMs / 10);
    this.reset();
  }

  reset() {
    this.nextTimestamp = null;
    this.lastTimestamp = null;
  }

  shouldRender(timestamp) {
    if (!Number.isFinite(timestamp)) return false;
    if (this.nextTimestamp === null || timestamp < this.lastTimestamp) {
      this.lastTimestamp = timestamp;
      this.nextTimestamp = timestamp + this.intervalMs;
      return true;
    }
    this.lastTimestamp = timestamp;
    if (timestamp + this.toleranceMs < this.nextTimestamp) return false;
    // Keep the phase, but discard missed deadlines. One RAF can draw once;
    // a delayed callback must not cause a burst of catch-up GPU submissions.
    const skipped = Math.floor(Math.max(0, timestamp + this.toleranceMs - this.nextTimestamp) / this.intervalMs);
    this.nextTimestamp += (skipped + 1) * this.intervalMs;
    return true;
  }
}

const dimensionKeys = ['geomPositions', 'geomMatrices', 'geomQuaternions', 'geomRGBA', 'bodyPositions', 'bodyQuaternions'];
const translationKeys = ['geomPositions', 'bodyPositions'];
const lengthOf = value => value == null ? -1 : value.length;

function discontinuousPose(previous, snapshot) {
  if (!Number.isFinite(previous.time) || !Number.isFinite(snapshot.time) || snapshot.time <= previous.time
      || previous.activeObjectId !== snapshot.activeObjectId) return true;
  for (const key of dimensionKeys) if (lengthOf(previous[key]) !== lengthOf(snapshot[key])) return true;
  const before = previous.geomRGBA, after = snapshot.geomRGBA;
  if (before && after) for (let i = 3; i < after.length; i += 4) if (before[i] !== after[i]) return true;
  for (const key of translationKeys) {
    const a = previous[key], b = snapshot[key];
    if (!a || !b) continue;
    for (let i = 0; i < b.length; i += 3) {
      const dx = b[i] - a[i], dy = b[i + 1] - a[i + 1], dz = b[i + 2] - a[i + 2];
      if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz) || dx * dx + dy * dy + dz * dz > .25) return true;
    }
  }
  return false;
}

export class PresentationBuffer {
  constructor({delayMs = 60, maxGapMs = 180, maxSamples = 12} = {}) {
    if (!Number.isFinite(delayMs) || delayMs < 0) throw new RangeError('Presentation delay must be nonnegative and finite.');
    if (!Number.isFinite(maxGapMs) || maxGapMs <= 0) throw new RangeError('Presentation gap must be positive and finite.');
    if (!Number.isInteger(maxSamples) || maxSamples < 3) throw new RangeError('Presentation history needs at least three samples.');
    Object.assign(this, {delayMs, maxGapMs, maxSamples});
    this.samples = [];
    this.lastSampleTarget = null;
  }

  clear() { this.samples.length = 0; this.lastSampleTarget = null; }

  push(snapshot, timestamp = performance.now()) {
    if (!snapshot || typeof snapshot !== 'object') { this.clear(); return true; }
    const previous = this.samples.at(-1), validTimestamp = Number.isFinite(timestamp);
    const snap = !previous || !validTimestamp || !previous.validTimestamp
      || timestamp < previous.timestamp || timestamp - previous.timestamp > this.maxGapMs
      || discontinuousPose(previous.snapshot, snapshot);
    if (snap) this.clear();
    const sample = {snapshot, timestamp: validTimestamp ? timestamp : 0, validTimestamp};
    // Forced worker publications can arrive in one timer quantum. Retain the
    // newest observation at that instant without a zero-length segment.
    if (!snap && timestamp === previous.timestamp) this.samples[this.samples.length - 1] = sample;
    else this.samples.push(sample);
    // RAF's timestamp can precede a just-received worker message. Trim using
    // the last displayed time, so receipt cannot discard its still-needed pair.
    this._trim(this.lastSampleTarget);
    return snap;
  }

  _trim(target) {
    // Keep the last observation before the display time and its successor.
    while (Number.isFinite(target) && this.samples.length > 2 && this.samples[1].timestamp <= target) this.samples.shift();
    // A dense catch-up burst can exceed the cap before display time advances.
    // Preserve that pair and the latest endpoint; discard middle future rows.
    while (this.samples.length > this.maxSamples) this.samples.splice(2, 1);
  }

  sample(timestamp) {
    if (!this.samples.length) return null;
    const latest = this.samples.at(-1);
    if (!Number.isFinite(timestamp) || !latest.validTimestamp) return {from: latest.snapshot, to: latest.snapshot, alpha: 1};
    const target = timestamp - this.delayMs;
    if (this.lastSampleTarget !== null && target < this.lastSampleTarget) {
      this.clear(); this.samples.push(latest);
      this.lastSampleTarget = target;
      return {from: latest.snapshot, to: latest.snapshot, alpha: 1};
    }
    this.lastSampleTarget = target;
    this._trim(target);
    const first = this.samples[0];
    if (target <= first.timestamp) return {from: first.snapshot, to: first.snapshot, alpha: 1};
    if (target >= latest.timestamp) return {from: latest.snapshot, to: latest.snapshot, alpha: 1};
    for (let i = 1; i < this.samples.length; i++) {
      const right = this.samples[i];
      if (target <= right.timestamp) {
        const left = this.samples[i - 1];
        return {from: left.snapshot, to: right.snapshot, alpha: (target - left.timestamp) / (right.timestamp - left.timestamp)};
      }
    }
    return {from: latest.snapshot, to: latest.snapshot, alpha: 1};
  }
}
