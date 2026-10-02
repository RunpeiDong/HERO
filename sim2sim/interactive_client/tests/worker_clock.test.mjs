import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

// Exercise the actual pure helper without starting worker imports, physics or
// timers. The worker entry itself cannot be imported in a Node test process.
const worker=await readFile(new URL('../worker.js',import.meta.url),'utf8');
const declaration=worker.match(/function advanceSimulationDeadline\([^)]*\)\s*\{[^}]*\}/)?.[0];
assert.ok(declaration,'The worker must retain its independently testable clock helper.');
const advance=vm.runInNewContext(`(${declaration})`);

test('on-time control maintains its 20 ms cadence',()=>{
  let deadline=1000;
  for(let step=0;step<20;step++){
    const finished=1000+20*step+2;
    deadline=advance(deadline,finished);
    assert.equal(deadline,1000+20*(step+1));
    assert.equal(deadline-finished,18);
  }
});

test('a 97 ms IK rebuild keeps the missed ticks instead of adding premature idle waits',()=>{
  let deadline=0,finished=97;
  const delays=[];
  for(let tick=0;tick<6;tick++){
    deadline=advance(deadline,finished);delays.push(Math.max(0,deadline-finished));finished+=2;
  }
  assert.deepEqual(delays,[0,0,0,0,0,13]);
  assert.equal(deadline,120,'Six actual control ticks advance the deadline by six 20 ms periods.');
});

test('long suspension limits catch-up debt to 250 ms and returns to paced execution',()=>{
  let now=5000,deadline=advance(100,now);
  assert.equal(now-deadline,250);
  let ticks=0;
  while(deadline<=now&&ticks<100){
    now+=2;deadline=advance(deadline,now);ticks++;
    assert.ok(now-deadline<=250);
  }
  assert.equal(ticks,14);
  assert.ok(deadline>now,'A bounded catch-up period must eventually wait again.');
});

test('a new grasp starts from its current wall clock without inheriting old catch-up debt',()=>{
  assert.match(worker,/nextTick = performance\.now\(\);/,'The existing grasp-start deadline reset remains present.');
  const restart=20000;
  assert.equal(advance(restart,restart+2),restart+20);
  assert.equal(advance(restart+20,restart+22),restart+40);
});
