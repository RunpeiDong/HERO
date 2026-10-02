import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const worker = await readFile(new URL('../worker.js', import.meta.url), 'utf8');
const slice = (from, to) => worker.slice(worker.indexOf(from), worker.indexOf(to));
const dispatchSource = slice('async function dispatch(', '\nfunction schedule(');
const configSource = slice('function validatedConfig(', '\nfunction resetSnapshot(');
const yawSource = slice('const CARTON_BASE_YAW_RAD', '\nconst publicCatalog');
const spotSource = slice('const defaultAddSpot', '\nconst CARTON_BASE_YAW_RAD');
const chooseSource = slice('function choosePlacement(', '\nasync function rebuildScene(');
const editableSource = slice('function requireEditable(', '\nfunction prepareEditing(');
assert.ok(yawSource.includes('function resetYaw') && chooseSource.startsWith('function choosePlacement('));

function fixture({random = () => .75} = {}) { // random .75 -> jitter +10 deg
  const config = {mode: 'hero_plus', table_kind: 'workbench', table_height: .74, grasp_style: 'side', ee_yaw_deg: 45, replan: true, goal_adjust: true};
  const placed = [], calls = [];
  const scene = {time: 0, selectedObjectId: 'cracker_box', _placements: {cracker_box: [.41, -.23, -.7], can: [.4, .2, .3], uiuc_i: [.41, -.23, -.4]},
    objects: {cracker_box: {active: true}, can: {active: true}, uiuc_i: {active: true}}, reject: null,
    placeObject(kind, x, y, yaw) { if (scene.reject?.(kind, x, y, yaw)) throw new Error('The entire object must fit on the tabletop.'); placed.push({kind, x, y, yaw}); scene._placements[kind] = [x, y, yaw]; },
    reset() { calls.push('reset'); },
    selectObject() {}, readState: () => ({rootPosW: [0, 0, .76], rootQuatW: [1, 0, 0, 0]})};
  const context = {config, scene, __random: random, structuredClone, HIDDEN_OBJECTS: new Set(['apple']), HAND_MODELS: {},
    controller: {phase: 'idle', busy: false, cancel() { calls.push('cancel'); }}, policy: {}, performance: {now: () => 0}, orientationResetIds: new Set(),
    choosePlacement(kind, preferred, options) { calls.push({choose: kind, preferred, options}); },
    currentState: () => ({}), prepareEditing() {}, rebuildController() { calls.push('controller'); }, async rebuildScene() {}, savedPlacements: () => structuredClone(scene._placements)};
  vm.createContext(context);
  vm.runInContext(`Math.random = __random;\n${spotSource}\n${yawSource}\n${configSource}\n${editableSource}\n${dispatchSource}`, context);
  const dispatch = vm.runInContext('dispatch', context);
  return {config, scene, placed, calls, context, dispatch, resetIds: context.orientationResetIds,
    cartonSetupYaw: vm.runInContext('cartonSetupYaw', context), blockISetupYaw: vm.runInContext('blockISetupYaw', context), resetYaw: vm.runInContext('resetYaw', context)};
}
const deg = r => r * 180 / Math.PI, plain = value => JSON.parse(JSON.stringify(value)); // sandbox objects live in another realm

test('setup yaw: 60 deg toward the side of the carton with a variation inside +-20 deg (band 40-90); other objects get null', () => {
  const f = fixture();
  for (const [y, expected] of [[-.2, -70], [.2, 70]]) assert.ok(Math.abs(deg(f.cartonSetupYaw('cracker_box', y)) - expected) < 1e-9);
  assert.equal(f.cartonSetupYaw('can', -.2), null);
  const draws = new Set();
  for (let i = 0; i < 400; i++) { const yaw = deg(fixture({random: Math.random}).cartonSetupYaw('cracker_box', -.3)); assert.ok(yaw <= -40 - 1e-9 && yaw >= -80 + 1e-9, yaw); draws.add(Math.round(yaw)); }
  assert.ok(draws.size > 10, 'the variation is random, not fixed');
  // the reset yaw keeps one magnitude per reset and follows the object's side
  assert.ok(Math.abs(deg(f.resetYaw('cracker_box', -.1)) + 70) < 1e-9 && Math.abs(deg(f.resetYaw('cracker_box', .4)) - 70) < 1e-9);
  assert.equal(f.resetYaw('can', -.1), 0);
});

test('/api/add lets the placement search choose the yaw from the cell the carton lands in', async () => {
  const f = fixture(); f.scene.objects.cracker_box.active = false;
  await f.dispatch({path: '/api/add', body: {kind: 'cracker_box'}});
  // Non-Block objects start outboard of the tray (the carton nearer the side edge); the Block I starts at (.41, -.23).
  assert.deepEqual(plain(f.calls[0]), {choose: 'cracker_box', preferred: [.41, -.37, 0], options: {spacing: .06, resetRobot: false, autoYaw: true}});
  const f2 = fixture(); f2.scene.objects.can.active = false;
  await f2.dispatch({path: '/api/add', body: {kind: 'can'}});
  assert.deepEqual(plain(f2.calls[0]), {choose: 'can', preferred: [.41, -.32, 0], options: {spacing: .06, resetRobot: false, autoYaw: true}});
  const f3 = fixture(); f3.scene.objects.uiuc_i.active = false;
  await f3.dispatch({path: '/api/add', body: {kind: 'uiuc_i'}});
  assert.deepEqual(plain(f3.calls[0]), {choose: 'uiuc_i', preferred: [.41, -.23, 0], options: {spacing: .06, resetRobot: false, autoYaw: true}});
});

test('Block I setup yaw spans its half-turn symmetry interval and accepts a seeded RNG', () => {
  const f = fixture();
  for (const [draw, expected] of [[0, -90], [.25, -45], [.5, 0], [.75, 45], [1, 90]])
    assert.ok(Math.abs(deg(f.blockISetupYaw(() => draw)) - expected) < 1e-9);
  const seeded = seed => () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 2 ** 32);
  const first = seeded(329), second = seeded(329), draws = [];
  for (let i = 0; i < 20; i++) {
    const yaw = f.blockISetupYaw(first); draws.push(yaw);
    assert.equal(yaw, f.blockISetupYaw(second)); assert.ok(yaw >= -Math.PI / 2 && yaw <= Math.PI / 2);
  }
  assert.ok(new Set(draws).size > 1);
});

test('Block I placement uses one draw per add, then 0 before searching another cell', () => {
  const attempted = []; let draws = 0;
  // the page trims the region's far edge by 2 cm (PAGE_PLACEMENT_X_MAX_TRIM_M), so x_max .44 leaves the .40 / .42 search cells
  const targetScene = {catalog: {placement_region: {x_min: .4, x_max: .44, y_min: -.3, y_max: -.28}, objects: [{id: 'uiuc_i', footprint_radius: .065}]},
    uiState: () => ({objects: []}),
    placeObject(kind, x, y, yaw) { attempted.push({x, y, yaw}); if (x === .4 || yaw !== 0) throw new Error('Does not fit.'); }};
  const context = vm.createContext({__random: () => { draws++; return .75; }});
  vm.runInContext(`Math.random = __random;\n${yawSource}\n${chooseSource}`, context);
  vm.runInContext('choosePlacement', context)('uiuc_i', [.4, -.3, -.4], {targetScene, autoYaw: true});
  assert.equal(draws, 1);
  assert.deepEqual(attempted.slice(0, 2), [{x: .4, y: -.3, yaw: Math.PI / 4}, {x: .4, y: -.3, yaw: 0}]);
  assert.equal(attempted.at(-1).yaw, 0); assert.ok(attempted.at(-1).x > .4);
});

test('Block I drags draw fresh yaws, manual rotation consumes no draw, and rejected footprints fall back to 0', async () => {
  const values = [.75, .25, .9]; let draws = 0;
  const f = fixture({random: () => values[draws++]});
  await f.dispatch({path: '/api/place', body: {kind: 'uiuc_i', x: .42, y: -.24, yaw: -.4}});
  assert.equal(f.placed.at(-1).yaw, Math.PI / 4);
  f.resetIds.add('uiuc_i');
  await f.dispatch({path: '/api/place', body: {kind: 'uiuc_i', x: .42, y: .24}});
  assert.equal(f.placed.at(-1).yaw, -Math.PI / 4); assert.equal(f.resetIds.has('uiuc_i'), false);
  await f.dispatch({path: '/api/place', body: {kind: 'uiuc_i', x: .42, y: .24, yaw: .3, rotation_edit: true}});
  assert.equal(f.placed.at(-1).yaw, .3); assert.equal(draws, 2);
  const attempted = [];
  f.scene.reject = (_kind, _x, _y, yaw) => { attempted.push(yaw); return yaw !== 0; };
  await f.dispatch({path: '/api/place', body: {kind: 'uiuc_i', x: .42, y: .24}});
  assert.deepEqual(attempted, [(.9 - .5) * Math.PI, 0]); assert.equal(f.placed.at(-1).yaw, 0);
});

test('direct reset redraws carton and Block I setup yaw; active attempts reject all placement edits', async () => {
  let draws = 0; const f = fixture({random: () => { draws++; return .75; }});
  await f.dispatch({path: '/api/reset'});
  assert.deepEqual(plain(f.calls.filter(call => call.choose).map(call => [call.choose, call.options.autoYaw])),
    [['cracker_box', true], ['can', false], ['uiuc_i', true]]);
  f.calls.length = 0; f.context.controller.busy = true;
  const before = plain(f.scene._placements);
  for (const request of [{path: '/api/add', body: {kind: 'uiuc_i'}},
    {path: '/api/place', body: {kind: 'uiuc_i', x: .42, y: .24}},
    {path: '/api/place', body: {kind: 'uiuc_i', x: .42, y: .24, yaw: .3, rotation_edit: true}}])
    await assert.rejects(f.dispatch(request), /Reset the current attempt/);
  assert.equal(draws, 0); assert.equal(f.placed.length, 0); assert.equal(f.calls.length, 0);
  assert.deepEqual(plain(f.scene._placements), before);
});

test('choosePlacement draws the carton yaw per candidate cell so a relocation across the midline still faces the hand', () => {
  const placed = [];
  const targetScene = {catalog: {placement_region: {x_min: .30, x_max: .57, y_min: -.40, y_max: .40}, objects: [{id: 'cracker_box', footprint_radius: .0895}]},
    uiState: () => ({objects: [{id: 'uiuc_i', position: [.41, -.23], footprint_radius: .065}]}),
    placeObject(kind, x, y, yaw) { if (y < .05) throw new Error('crowded: the right half and the midline are taken'); placed.push({x, y, yaw}); }};
  const context = {}; vm.createContext(context);
  vm.runInContext(`Math.random = () => .5;\n${yawSource}\n${chooseSource}`, context);
  vm.runInContext('choosePlacement', context)('cracker_box', [.41, -.23, 0], {spacing: .06, targetScene, autoYaw: true});
  assert.equal(placed.length, 1); assert.ok(placed[0].y > 0); assert.ok(Math.abs(deg(placed[0].yaw) - 60) < 1e-9, 'yaw follows the final (left) cell');
  placed.length = 0;
  vm.runInContext('choosePlacement', context)('can', [.41, -.23, 0], {spacing: .06, targetScene});
  assert.equal(placed[0].yaw, 0, 'other objects keep the requested yaw');
});

test('/api/place: drags re-draw the hand-facing yaw for the destination side; slider rotations are clamped into the band', async () => {
  const f = fixture();
  await f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .42, y: -.24}});
  assert.ok(Math.abs(deg(f.placed.at(-1).yaw) + 70) < 1e-9, 'same-side drag: fresh hand-facing yaw');
  await f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .42, y: .24, yaw: -.7}});
  assert.ok(Math.abs(deg(f.placed.at(-1).yaw) - 70) < 1e-9, 'crossing the midline without rotation_edit ignores the stale yaw');
  // A slider rotation stays inside sign(y) * [40, 90] deg -- the nearest in-band yaw over the carton's half-turn copies is
  // placed instead of the raw value.
  for (const [y, yaw, expected] of [[.24, -.3, 40], [.24, 1.4, deg(1.4)], [.24, 2.0, 90], [.24, .8, deg(.8)], [-.24, .3, -40], [-.24, -1.6, -90], [-.24, Math.PI, -40], [-.24, -.9, deg(-.9)], [.24, .5, 40]]) {
    await f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .42, y, yaw, rotation_edit: true}});
    assert.ok(Math.abs(deg(f.placed.at(-1).yaw) - expected) < 1e-9, `yaw ${yaw} at y ${y} -> ${deg(f.placed.at(-1).yaw)} (expected ${expected})`);
  }
  await f.dispatch({path: '/api/place', body: {kind: 'can', x: .4, y: .2}});
  assert.equal(f.placed.at(-1).yaw, .3, 'other objects keep their previous yaw on a drag');
  await f.dispatch({path: '/api/place', body: {kind: 'can', x: .4, y: .2, yaw: -1.3, rotation_edit: true}});
  assert.equal(f.placed.at(-1).yaw, -1.3, 'other objects keep an explicit rotation as set');
});

test('clampCartonYaw folds any angle into the hand-facing band of the carton side', () => {
  const f = fixture(), clamp = f.context.clampCartonYaw ?? vm.runInContext('clampCartonYaw', f.context);
  for (const [yaw, y, expected] of [[0, .1, 40], [0, -.1, -40], [Math.PI / 4, .1, 45], [-Math.PI / 4, .1, 90], [Math.PI / 2, .1, 90], [-Math.PI / 2, .1, 90], [Math.PI / 2, -.1, -90],
      [100 * Math.PI / 180, .1, 90], [160 * Math.PI / 180, .1, 40], [-160 * Math.PI / 180, .1, 40], [NaN, .1, 65], [NaN, -.1, -65],
      [10, .1, 40], [5 * Math.PI, .1, 40], [5 * Math.PI, -.1, -40], [-7 * Math.PI / 2, .1, 90], [0, 0, -40], [35 * Math.PI / 180, .1, 40], [-35 * Math.PI / 180, -.1, -40]])
    assert.ok(Math.abs(deg(clamp(yaw, y)) - expected) < 1e-9, `${yaw} @ ${y} -> ${deg(clamp(yaw, y))} != ${expected}`);
  const band = vm.runInContext('cartonYawBand', f.context);
  assert.deepEqual(plain(band(.3)).map(deg).map(Math.round), [40, 90]); assert.deepEqual(plain(band(-.3)).map(deg).map(Math.round), [-90, -40]);
});

test('/api/place falls back to the 60 deg base yaw, the band edges and the folded previous yaw -- never to 0 -- when the drawn carton footprint does not fit', async () => {
  const f = fixture(); const exact = -60 * Math.PI / 180;
  f.scene.reject = (kind, x, y, yaw) => Math.abs(yaw - exact) > 1e-9; // only the exact 60 deg base yaw fits
  await f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .34, y: -.30}});
  assert.ok(Math.abs(f.placed.at(-1).yaw - exact) < 1e-9);
  f.scene.reject = (kind, x, y, yaw) => Math.abs(deg(yaw) + 50) > 1e-9; // only the previous (in-band, -50 deg) yaw fits: it comes before the band edges
  f.scene._placements.cracker_box = [.34, -.30, -50 * Math.PI / 180];
  await f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .34, y: -.30}});
  assert.ok(Math.abs(deg(f.placed.at(-1).yaw) + 50) < 1e-9, 'the previous in-band yaw is kept before snapping to an edge');
  f.scene.reject = (kind, x, y, yaw) => Math.abs(deg(yaw) + 90) > 1e-9; // only the steep band edge fits
  await f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .34, y: -.30}});
  assert.ok(Math.abs(deg(f.placed.at(-1).yaw) + 90) < 1e-9);
  // A slider rotation that does not fit is refused, not replaced (as any explicit rotation was before).
  f.scene.reject = (kind, x, y, yaw) => Math.abs(deg(yaw) + 60) < 1e-9;
  await assert.rejects(f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .34, y: -.30, yaw: -60 * Math.PI / 180, rotation_edit: true}}), /fit on the tabletop/);
  assert.ok(Math.abs(deg(f.placed.at(-1).yaw) + 90) < 1e-9, 'nothing else was placed');
  f.scene.reject = (kind, x, y, yaw) => yaw !== 0; // only square would fit: square is outside the band, so the drop is refused
  await assert.rejects(f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .34, y: -.30}}), /fit on the tabletop/);
  assert.ok(f.placed.every(row => row.kind !== 'cracker_box' || (Math.abs(deg(row.yaw)) >= 40 - 1e-9 && Math.abs(deg(row.yaw)) <= 90 + 1e-9)), 'every carton placement stays inside the band');
  f.scene.reject = () => true;
  await assert.rejects(f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .34, y: -.30}}), /fit on the tabletop/);
});

test('a manipulated carton is reset to a hand-facing yaw (not 0) and the reset bookkeeping is cleared', async () => {
  const f = fixture(); f.resetIds.add('cracker_box'); f.resetIds.add('can');
  await f.dispatch({path: '/api/place', body: {kind: 'cracker_box', x: .41, y: -.23}});
  assert.ok(Math.abs(deg(f.placed.at(-1).yaw) + 70) < 1e-9); assert.equal(f.resetIds.has('cracker_box'), false);
  await f.dispatch({path: '/api/place', body: {kind: 'can', x: .4, y: .2, yaw: .9}});
  assert.equal(f.placed.at(-1).yaw, 0, 'other manipulated objects still reset to robot-facing 0 unless rotated explicitly');
});

test('choosePlacement carton fallbacks: a drawn yaw that does not fit tries the base yaw and both band edges, never 0; a restored layout is clamped into the band', () => {
  const placed = [], accept = {yaw: null};
  const targetScene = {catalog: {placement_region: {x_min: .30, x_max: .57, y_min: -.40, y_max: .40}, objects: [{id: 'cracker_box', footprint_radius: .0895}]},
    uiState: () => ({objects: []}),
    placeObject(kind, x, y, yaw) { if (accept.yaw !== null && Math.abs(yaw - accept.yaw) > 1e-9) throw new Error('does not fit'); placed.push({x, y, yaw}); }};
  const context = {}; vm.createContext(context);
  vm.runInContext(`Math.random = () => .5;\n${yawSource}\n${chooseSource}`, context);
  const choose = vm.runInContext('choosePlacement', context);
  // autoYaw (add / worker reset): only the steep edge fits -> -90 deg on the right, never 0
  accept.yaw = -90 * Math.PI / 180; choose('cracker_box', [.41, -.23, 0], {spacing: .06, targetScene, autoYaw: true});
  assert.equal(placed.length, 1); assert.ok(Math.abs(deg(placed.at(-1).yaw) + 90) < 1e-9);
  accept.yaw = 0; placed.length = 0;
  assert.throws(() => choose('cracker_box', [.41, -.23, 0], {spacing: .06, targetScene, autoYaw: true}), /no valid placement/, 'square is never used for the carton: a cell that only fits yaw 0 is given up');
  assert.equal(placed.length, 0);
  // restore (page Reset / table or hand change): the saved yaw is kept when inside the band, clamped when outside
  accept.yaw = null; placed.length = 0;
  choose('cracker_box', [.41, -.23, -40 * Math.PI / 180], {spacing: .06, targetScene});
  assert.ok(Math.abs(deg(placed.at(-1).yaw) + 40) < 1e-9, 'an in-band saved yaw is restored as saved');
  choose('cracker_box', [.41, -.23, 0], {spacing: .06, targetScene});
  assert.ok(Math.abs(deg(placed.at(-1).yaw) + 40) < 1e-9, 'a saved square carton (older layout) is clamped to the band edge of its side');
  choose('cracker_box', [.41, .23, -1.3], {spacing: .06, targetScene}); // -74.5 deg: its half-turn copy 105.5 deg is nearest the left band -> 90 deg
  assert.ok(Math.abs(deg(placed.at(-1).yaw) - 90) < 1e-9, 'a saved yaw of the other side is folded into this side\'s band');
});

test('the page trims the placeable area 2 cm before the scene far edge: search and state use x_max - 0.02, the scene region is untouched', () => {
  const placed = [];
  const targetScene = {catalog: {placement_region: {x_min: .30, x_max: .57, y_min: -.40, y_max: .40, corner_cut: {x_start: .45, y_start: .24}}, objects: [{id: 'can', footprint_radius: .04}]},
    uiState: () => ({objects: []}), placeObject(kind, x, y, yaw) { placed.push({x, y, yaw}); }};
  const context = vm.createContext({__random: () => .5});
  vm.runInContext(`Math.random = __random;\n${yawSource}\n${chooseSource}`, context);
  const choose = vm.runInContext('choosePlacement', context);
  choose('can', [.60, 0, 0], {targetScene});
  assert.ok(Math.abs(placed.at(-1).x - .55) < 1e-9, `a far request is clamped to .55, got ${placed.at(-1).x}`);
  assert.equal(targetScene.catalog.placement_region.x_max, .57, 'the scene region itself is not modified');
  const region = vm.runInContext('pagePlacementRegion', context)(targetScene.catalog.placement_region);
  assert.deepEqual(plain(region), {x_min: .30, x_max: .55, y_min: -.40, y_max: .40, corner_cut: {x_start: .45, y_start: .24}}); // plain(): vm-context objects differ by prototype
  assert.equal(vm.runInContext('PAGE_PLACEMENT_X_MAX_TRIM_M', context), .02);
});
