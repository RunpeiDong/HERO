import assert from 'node:assert/strict';
import {access, readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import {policyFeedback} from '../controller.js';
import {createSimulation} from '../simulation.js';

const bridgeSource = (await readFile(new URL('../client_bridge.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\n/gm, '').replace('export class ClientBridge', 'class ClientBridge');
const workerSource = (await readFile(new URL('../worker.js', import.meta.url), 'utf8')).replace(/^import .*;\n/gm, '');
const plain = value => JSON.parse(JSON.stringify(value));
const setup = () => ({version: 1,
  config: {mode: 'hero_plus', table_kind: 'round', table_height: .5, grasp_style: 'side', ee_yaw_deg: 30,
    replan: true, goal_adjust: true},
  placements: {uiuc_i: [.41, -.23, 0], can: [.4, .23, .3]}, selectedObjectId: 'can'});

function bridgeHarness({embedded = false} = {}) {
  const workers = [], events = {states: [], errors: [], progress: [], activity: []}, frames = new Map();
  let raf = 0;
  class Worker {
    constructor() { this.sent = []; this.terminated = false; workers.push(this); }
    postMessage(message, transfer = []) {
      if (this.postFailure) throw this.postFailure;
      this.sent.push(structuredClone(message, {transfer}));
    }
    terminate() { this.terminated = true; }
    emit(data) { this.onmessage?.({data}); }
  }
  class Renderer {
    constructor() { this.builds = []; this.updates = []; this.renderCount = 0; this.presentationResets = 0; this.cameraOptions = {distance: 2.7}; }
    build(description) { this.builds.push(description); this.cameraOptions = {distance: 3.1}; }
    update(...args) { this.updates.push(args); }
    enqueueSnapshot(...args) { this.updates.push(args); }
    resetPresentation() { this.presentationResets++; }
    setCamera(options) { this.cameraOptions = options; }
    render(timestamp) { this.renderCount++; this.renderTimestamp = timestamp; }
    dispose() { this.disposed = true; }
  }
  const context = vm.createContext({SimulationWorker: Worker, TabletopRenderer: Renderer, URL, structuredClone,
    Uint8Array, atob, document: {baseURI: 'https://example.test/demo/index.html'},
    requestAnimationFrame: callback => {frames.set(++raf, callback); return raf;}, cancelAnimationFrame: id => frames.delete(id),
    ...(embedded ? {__TABLETOP_ASSETS__: {'asset.gz': btoa('compressed bytes')}} : {})});
  const Bridge = vm.runInContext(`${bridgeSource}\nClientBridge;`, context);
  const bridge = new Bridge({mainCanvas: {}, onState: value => events.states.push(value),
    onError: value => events.errors.push(value), onProgress: value => events.progress.push(value),
    onIKActivity: value => events.activity.push(value)});
  const response = (worker, request, result) => worker.emit({type: 'response', id: request.id, result});
  const initialize = async (snapshot = setup()) => {
    const promise = bridge.start(), worker = bridge.worker, request = worker.sent.at(-1);
    const state = {controller: {phase: 'idle'}, reset_snapshot: snapshot};
    worker.emit({type: 'state', state}); response(worker, request, state); await promise;
  };
  return {bridge, workers, events, frames, response, initialize};
}

test('Reset terminates a worker with unresolved planning requests and restores its last confirmed setup', async t => {
  const h = bridgeHarness(); t.after(() => h.bridge.dispose()); await h.initialize();
  const old = h.bridge.worker, callback = old.onmessage, errorCallback = old.onerror;
  const pending = h.bridge.request('/api/grasp', {object_id: 'can'});
  const cancelled = assert.rejects(pending, error => error.name === 'AbortError' && error.code === 'SIMULATION_RESET');
  old.emit({type: 'ik_activity', active: true});
  const reset = h.bridge.request('/api/reset', {}), current = h.bridge.worker, init = current.sent.at(-1);
  assert.equal(old.terminated, true, 'Termination happens in the same call, before waiting for a response.');
  assert.equal(h.bridge.renderer.presentationResets, 1, 'Reset clears queued display poses immediately.');
  assert.notEqual(current, old); assert.equal(init.path, '/init');
  assert.equal(old.sent.filter(message => message.path === '/api/reset').length, 0, 'Reset never queues behind the blocked request.');
  assert.equal(h.events.activity.at(-1), false);
  assert.deepEqual(plain(init.body.restore), setup());
  assert.equal(init.body.assetsBase, 'https://example.test/demo/'); assert.equal(init.body.embedded, false);
  await cancelled;

  let completed = false; reset.then(() => { completed = true; });
  const before = {states: h.events.states.length, errors: h.events.errors.length, updates: h.bridge.renderer.updates.length};
  // Saved native callbacks can already be in the event queue at termination.
  // Even an old response using the new request's ID must be ignored.
  for (const data of [{type: 'response', id: init.id, result: {stale: true}}, {type: 'state', state: {stale: true}},
    {type: 'frame', frame: {stale: true}}, {type: 'model', description: {stale: true}},
    {type: 'progress', message: 'Old progress'}, {type: 'ik_activity', active: true},
    {type: 'asset', id: 99, path: 'missing.gz'}]) callback({data});
  errorCallback({message: 'Late old worker error'});
  await Promise.resolve();
  assert.equal(completed, false); assert.equal(current.sent.length, 1);
  assert.equal(h.events.states.length, before.states); assert.equal(h.events.errors.length, before.errors);
  assert.equal(h.bridge.renderer.updates.length, before.updates); assert.equal(h.bridge.renderer.builds.length, 0);
  assert.equal(h.events.activity.at(-1), false); assert.notEqual(h.events.progress.at(-1), 'Old progress');
  current.emit({type: 'model', description: {homeCamera: {distance: 3.1}}});
  current.emit({type: 'frame', frame: {fresh: true}, preview: null});
  h.response(current, init, {controller: {phase: 'idle'}, reset_snapshot: setup()});
  assert.equal((await reset).controller.phase, 'idle'); assert.equal(h.bridge.pending.size, 0);
  assert.equal(h.bridge.renderer.builds.length, 1); assert.equal(h.bridge.renderer.updates.length, 1);
  assert.equal(h.bridge.renderer.cameraOptions.distance, 2.7, 'Worker replacement preserves the user camera.');
  const render = h.frames.values().next().value; render(1234);
  assert.equal(h.bridge.renderer.renderCount, 1, 'The existing render loop continues after replacement.');
  assert.equal(h.bridge.renderer.renderTimestamp, 1234, 'RAF time reaches the presentation clock unchanged.');
});

test('repeated Reset cancels initialization cleanly, including before the first state arrives', async t => {
  const h = bridgeHarness(); t.after(() => h.bridge.dispose());
  const start = h.bridge.start(); const startCancelled = assert.rejects(start, {code: 'SIMULATION_RESET'});
  const first = h.bridge.reset(); const firstCancelled = assert.rejects(first, {code: 'SIMULATION_RESET'});
  const middle = h.bridge.worker, stale = middle.onmessage;
  const second = h.bridge.reset(), latest = h.bridge.worker, init = latest.sent.at(-1);
  assert.equal(h.workers.length, 3); assert.ok(h.workers.slice(0, 2).every(worker => worker.terminated));
  assert.equal(init.body.restore, undefined, 'No unconfirmed or synthetic layout is restored.');
  stale({data: {type: 'response', id: init.id, result: {stale: true}}});
  h.response(latest, init, {controller: {phase: 'idle'}});
  assert.equal((await second).controller.phase, 'idle'); await startCancelled; await firstCancelled;
});

test('worker failures reject later requests promptly but Reset can recover from the failure', async t => {
  const h = bridgeHarness(); t.after(() => h.bridge.dispose()); await h.initialize();
  const pending = h.bridge.request('/api/grasp', {}), failed = assert.rejects(pending, /Worker crashed/);
  const old = h.bridge.worker; old.onerror({message: 'Worker crashed'}); await failed;
  const count = old.sent.length; await assert.rejects(h.bridge.request('/api/state', {}), /Worker crashed/);
  assert.equal(old.sent.length, count, 'A request must not be left waiting on a known failed worker.');
  const reset = h.bridge.reset(), current = h.bridge.worker;
  h.response(current, current.sent.at(-1), {controller: {phase: 'idle'}});
  await reset; assert.equal(h.bridge.workerFailure, null); assert.equal(h.events.errors.length, 1);
});

test('confirmed snapshots are copied, and embedded asset requests still work after worker replacement', async t => {
  const h = bridgeHarness({embedded: true}); t.after(() => h.bridge.dispose()); const original = setup();
  await h.initialize(original); original.placements.can[0] = 99;
  const old = h.bridge.worker; await h.bridge.receive({type: 'asset', id: 3, path: 'asset.gz'}, old);
  assert.equal(new TextDecoder().decode(old.sent.at(-1).bytes), 'compressed bytes');
  const reset = h.bridge.reset(), current = h.bridge.worker, init = current.sent.at(-1);
  assert.equal(init.body.embedded, true); assert.equal(init.body.restore.placements.can[0], .4);
  await h.bridge.receive({type: 'asset', id: 4, path: 'asset.gz'}, current);
  assert.equal(new TextDecoder().decode(current.sent.at(-1).bytes), 'compressed bytes');
  h.response(current, init, {controller: {phase: 'idle'}}); await reset;
});

test('closing the bridge rejects pending work and prevents stale callbacks or replacement workers', async () => {
  const h = bridgeHarness(); const promise = h.bridge.start(), cancelled = assert.rejects(promise, {code: 'DEMO_CLOSED'});
  const old = h.bridge.worker, callback = old.onmessage; h.bridge.dispose(); h.bridge.dispose();
  await cancelled; callback({data: {type: 'frame', frame: {stale: true}}});
  await assert.rejects(h.bridge.reset(), {code: 'DEMO_CLOSED'});
  assert.equal(h.workers.length, 1); assert.equal(old.terminated, true); assert.equal(h.bridge.renderer.disposed, true);
  assert.equal(h.bridge.renderer.updates.length, 0); assert.equal(h.frames.size, 0);
});

const assets = new URL('../public/', import.meta.url);
const nativeAvailable = await access(new URL('sceneassets/manifest.json', assets)).then(() => true, () => false);
async function workerHarness({native = false, random = () => .5} = {}) {
  const sent = [], requests = [], scenes = [];
  const metadata = native ? JSON.parse(await readFile(new URL('policies/hero_plus.json', assets), 'utf8')) : {};
  const policy = {kp: metadata.kp, kd: metadata.kd, effortLimit: metadata.effortLimit, reset() {}, dispose() {}};
  class Controller {
    constructor(scene, _policy, config) {
      this.phase = 'idle'; this.scene = scene; this.config = config; this.objectId = config.objectId;
      this.reference = {ik: {solve() {}}, window() {}};
    }
    prepare() {} preflightHomePath() {} preview() { return null; } dispose() {} cancel() { this.phase = 'idle'; }
    snapshot() { return {phase: this.phase, busy: false}; }
  }
  const context = vm.createContext({console, URL, Blob, Response, TextDecoder, TextEncoder, Uint8Array,
    CARTON_GRASP_SETTINGS: ['auto', 'end_face', 'crotch', 'spine', 'spine90'], cartonGraspSettingOf: value => typeof value === 'string' ? ({spine45: 'spine'}[value] ?? value) : value,
    DecompressionStream, structuredClone, performance, setTimeout, clearTimeout, policyFeedback, __random: random,
    ort: {env: {wasm: {}}}, self: {postMessage: message => sent.push(message)},
    loadRuntime: async () => {},
    loadResolvedPolicy: async (mode, {manifest, assetResolver}) => {
      assert.equal(mode, 'hero_plus'); assert.equal(manifest.policies.hero_plus.files.model, 'hero.onnx');
      await assetResolver(manifest.policies.hero_plus.metadata);
      await assetResolver(manifest.policies.hero_plus.files.model); return policy;
    },
    InteractiveController: Controller,
    createSimulation: async options => {
      assert.equal(native, true, 'This fixture only instantiates real MuJoCo scenes.');
      const scene = await createSimulation(options); scenes.push(scene); return scene;
    },
    fetch: async url => { const path = new URL(url).pathname.replace('/demo/', ''); requests.push(path);
      return new Response(await readFile(new URL(path, assets))); },
  });
  const api = vm.runInContext(`Math.random = __random;\n${workerSource}\n({dispatch, validateResetSnapshot, resetSnapshot, savedPlacements,
    scene:()=>scene, config:()=>({...config}), orientationResetIds});`, context);
  // rebuildScene disposes retired native scenes itself; only the current scene remains owned by this fixture.
  return {api, sent, requests, scenes, dispose: () => api.scene()?.dispose()};
}

test('worker rejects malformed restore data before loading assets and restores authoritative HERO feedback', async () => {
  const h = await workerHarness();
  for (const restore of [{...setup(), version: 2}, {...setup(), placements: {can: [.4, NaN, 0]}},
    {...setup(), selectedObjectId: 'missing'}, {...setup(), config: {...setup().config, mode: 'other_policy'}},
    {...setup(), config: {...setup().config, table_height: .63}}, {...setup(), config: {...setup().config, grasp_style: 'diagonal'}},
    {...setup(), config: {...setup().config, grasp_style: 'top_down', ee_yaw_deg: 0}},
    {...setup(), placements: {...setup().placements, apple: [.41, .25, 0]}}]) {
    await assert.rejects(h.api.dispatch({path: '/init', body: {restore}}));
  }
  assert.equal(h.requests.length, 0);
  const restore = h.api.validateResetSnapshot({...setup(), config: {...setup().config, replan: false, goal_adjust: false}});
  assert.equal(restore.config.replan, true); assert.equal(restore.config.goal_adjust, true);
  assert.equal(restore.config.ee_yaw_deg, 45, 'Restoring a scene discards a legacy wrist-angle override.');
  const side = h.api.validateResetSnapshot({...setup(), config: {...setup().config, grasp_style: 'side', ee_yaw_deg: 0}});
  assert.equal(side.config.ee_yaw_deg, 45, 'Restoring Side uses its fixed outward rotation.');
  assert.deepEqual(plain(h.api.validateResetSnapshot({...setup(), placements: {}, selectedObjectId: null}).placements), {});
});

test('fresh worker init restores real scene setup, validates placements, and preserves orientation-reset semantics', {skip: !nativeAvailable}, async t => {
  let draws = 0;
  const h = await workerHarness({native: true, random: () => { draws++; return .75; }}); t.after(h.dispose);
  const restore = setup();
  restore.placements.uiuc_i = [.42, -.2, 0]; // Valid Block I footprint on the low round table.
  const state = await h.api.dispatch({path: '/init', body: {assetsBase: 'https://example.test/demo/', embedded: false, restore}});
  assert.equal(state.table.kind, 'round'); assert.equal(state.table.height, .5);
  assert.equal(state.selectedObjectId, 'can'); assert.equal(state.grasp_style, 'side'); assert.equal(state.ee_yaw_deg, 45);
  assert.deepEqual(plain(state.reset_snapshot.placements.can), restore.placements.can);
  assert.deepEqual(plain(state.reset_snapshot.placements.uiuc_i.slice(0, 2)), restore.placements.uiuc_i.slice(0, 2));
  assert.equal(draws, 1, 'Worker replacement redraws Block I setup yaw even when restoring a snapshot.');
  let expectedYaw = Math.PI / 4;
  try { h.api.scene().validatePlacement('uiuc_i', ...restore.placements.uiuc_i.slice(0, 2), expectedYaw); }
  catch { expectedYaw = 0; }
  assert.equal(state.reset_snapshot.placements.uiuc_i[2], expectedYaw, 'A reset uses the drawn yaw, or 0 if the footprint does not fit.');
  assert.equal(state.controller.phase, 'idle'); assert.equal(state.time, 0);
  assert.equal(state.reset_snapshot.config.replan, true); assert.equal(state.reset_snapshot.config.goal_adjust, true);
  assert.equal(h.sent.filter(message => message.type === 'model').length, 1);
  assert.ok(h.requests.includes('policies/hero.onnx'));
  const scene = h.api.scene();
  assert.equal(scene.validatePlacement('uiuc_i', ...scene._placements.uiuc_i), true);
  assert.equal(scene.validatePlacement('can', ...scene._placements.can), true);
  await h.api.dispatch({path: '/api/place', body: {kind: 'uiuc_i', x: .42, y: -.2, yaw: 0, rotation_edit: true}});
  assert.equal(draws, 1, 'A manual rotation does not consume a random draw.');
  const configured = await h.api.dispatch({path: '/api/config', body: {table_height: .74}});
  assert.equal(configured.reset_snapshot.placements.uiuc_i[2], 0, 'A table configuration rebuild preserves the manual rotation.');
  assert.equal(draws, 1);
  const reset = await h.api.dispatch({path: '/api/reset'});
  assert.equal(draws, 2, 'Direct reset also draws a fresh Block I yaw.');
  assert.equal(h.api.scene().validatePlacement('uiuc_i', ...reset.reset_snapshot.placements.uiuc_i), true);
  // The configuration rebuild retired the initial scene. Inspect the current scene's snapshot bookkeeping.
  const current = h.api.scene(), currentMasks = Array.from(current.model.geom_contype), currentAffinity = Array.from(current.model.geom_conaffinity);
  h.api.orientationResetIds.add('can');
  current.data.qpos[current.objects.can.qadr] = .52; // A later simulated object location is not its saved setup.
  const snapshot = h.api.resetSnapshot();
  assert.deepEqual(plain(snapshot.placements.can), [.4, .23, 0]);
  assert.deepEqual(Array.from(current.model.geom_contype), currentMasks); assert.deepEqual(Array.from(current.model.geom_conaffinity), currentAffinity);
});

test('ClientBridge.initOptions forwards ?carton_grasp= from the page URL to /init and omits it otherwise', async () => {
  const context = vm.createContext({URL, document: {baseURI: 'https://example.test/demo/index.html?carton_grasp=spine45&x=1'}});
  const Bridge = vm.runInContext(`${bridgeSource}\nClientBridge;`, context);
  assert.deepEqual(plain(Bridge.initOptions()), {assetsBase: 'https://example.test/demo/', embedded: false, carton_grasp: 'spine45'});
  assert.deepEqual(plain(Bridge.initOptions('https://example.test/demo/index.html')), {assetsBase: 'https://example.test/demo/', embedded: false});
  assert.deepEqual(plain(Bridge.initOptions('file:///Users/x/Tabletop_Lab.html?carton_grasp=end_face')), {assetsBase: 'file:///Users/x/', embedded: false, carton_grasp: 'end_face'});
  // start() sends exactly these options: the harness document has no query string, so the body carries no carton_grasp
  const h = bridgeHarness(); const started = h.bridge.start();
  assert.deepEqual(plain(h.bridge.worker.sent.at(-1).body), {assetsBase: 'https://example.test/demo/', embedded: false});
  const request = h.bridge.worker.sent.at(-1); h.bridge.worker.emit({type: 'state', state: {controller: {phase: 'idle'}, reset_snapshot: setup()}});
  h.bridge.worker.emit({type: 'response', id: request.id, result: {controller: {phase: 'idle'}}}); await started; h.bridge.dispose();
});
