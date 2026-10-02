import assert from 'node:assert/strict';
import {cp, mkdtemp, mkdir, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import test from 'node:test';
import vm from 'node:vm';
import {heroBuildAssetsPlugin, retainHeroBuildAssets} from '../build_hero_assets.mjs';

const worker = await readFile(new URL('../worker.js', import.meta.url), 'utf8');
const dispatchSource = worker.slice(worker.indexOf('async function dispatch('), worker.indexOf('\nfunction schedule('));
const configSource = worker.slice(worker.indexOf('function validatedConfig('), worker.indexOf('\nfunction resetSnapshot('));
assert.ok(dispatchSource.startsWith('async function dispatch('));
assert.ok(configSource.startsWith('function validatedConfig('));

function dispatchFixture() {
  const config = {mode: 'hero_plus', table_kind: 'workbench', table_height: .74, grasp_style: 'side', ee_yaw_deg: 45,
    replan: true, goal_adjust: true}, calls = [];
  const context = {config, scene: {time: 0, activeObjectId: 'can', objects: {can: {active: false}}, readState: () => ({rootPosW: [0, 0, .76], rootQuatW: [1, 0, 0, 0]})}, HAND_MODELS: {dex3: {label: 'Dex3', prefix: ''}}, availableHands: ['dex3'], CARTON_GRASP_SETTINGS: ['auto', 'end_face', 'crotch', 'spine', 'spine90'], cartonGraspSettingOf: value => typeof value === 'string' ? ({spine45: 'spine'}[value] ?? value) : value, HIDDEN_OBJECTS: new Set(['apple']),
    controller: {phase: 'idle', async prepare(options) {calls.push({prepare: structuredClone(options)});}},
    policy: {id: 'same policy'}, performance: {now: () => 0},
    requireEditable() {}, savedPlacements() {calls.push('placements'); return {can: [.4, -.2, 0]};},
    currentState: () => structuredClone(config), policyFeedback: () => ({replan: true, goal_adjust: true}),
    prepareEditing() {calls.push('editing');}, rebuildController() {calls.push('controller');},
    async rebuildScene() {calls.push('scene');},
    async loadPolicy() {throw new Error('A HERO table edit must retain the current policy.');},
    async disposeRetired() {throw new Error('A HERO table edit must not retire its policy.');}};
  return {config, calls, dispatch: vm.runInNewContext(`${configSource}\n${dispatchSource}\ndispatch`, context)};
}

test('worker rejects OTHER and unknown controller changes before config, scene, or policy mutation', async () => {
  for (const mode of ['other_policy', 'other', '']) {
    const {config, calls, dispatch} = dispatchFixture(), before = structuredClone(config);
    await assert.rejects(dispatch({path: '/api/config', body: {mode, table_height: .5, ee_yaw_deg: 30}}), /This demo uses HERO/);
    assert.deepEqual(config, before); assert.deepEqual(calls, []);
  }
});

test('ordinary HERO configuration preserves existing scene rebuild and feedback behavior', async () => {
  const f = dispatchFixture();
  await f.dispatch({path: '/api/config', body: {mode: 'hero_plus', table_height: .5, replan: false, goal_adjust: false}});
  assert.equal(f.config.table_height, .5); assert.equal(f.config.mode, 'hero_plus');
  assert.equal(f.config.replan, true); assert.equal(f.config.goal_adjust, true);
  assert.deepEqual(f.calls, ['placements', 'scene']);
  f.calls.length = 0;
  await f.dispatch({path: '/api/config', body: {ee_yaw_deg: 30}});
  assert.equal(f.config.ee_yaw_deg, 45);
  assert.deepEqual(f.calls, ['placements', 'editing', 'controller']);
});

test('worker grasps from the side only, at 45 degrees, for requests and legacy configs', async () => {
  const f = dispatchFixture();
  for (const style of [undefined, 'side']) for (const requested of [undefined, 0, 30, 45, 90, NaN, 'invalid']) {
    // A prior saved angle must not survive an ordinary table edit.
    f.config.ee_yaw_deg = 30;
    await f.dispatch({path: '/api/config', body: {grasp_style: style, ee_yaw_deg: requested,
      replan: false, goal_adjust: false}});
    assert.equal(f.config.grasp_style, 'side');
    assert.equal(f.config.ee_yaw_deg, 45);
    assert.equal(f.config.replan, true); assert.equal(f.config.goal_adjust, true);
    assert.equal(f.config.table_kind, 'workbench'); assert.equal(f.config.table_height, .74);
  }
  for (const style of ['top_down', 'diagonal']) {
    await assert.rejects(f.dispatch({path: '/api/config', body: {grasp_style: style}}), /grasps from the side/);
    assert.equal(f.config.grasp_style, 'side'); assert.equal(f.config.ee_yaw_deg, 45);
  }
  f.config.ee_yaw_deg = 30;
  f.calls.length = 0;
  await f.dispatch({path: '/api/grasp', body: {object_id: 'mug', ee_yaw_deg: 90, yawDeg: 90, approach: 'other'}});
  assert.deepEqual(f.calls, [{prepare: {objectId: 'mug', hand: 'auto', approach: 'side', yawDeg: 45}}]);
});

async function fingerprint(directory, prefix = '') {
  const result = {};
  for (const entry of await readdir(join(directory, prefix), {withFileTypes: true})) {
    const path = `${prefix}${entry.name}`;
    if (entry.isDirectory()) Object.assign(result, await fingerprint(directory, `${path}/`));
    else result[path] = createHash('sha256').update(await readFile(join(directory, path))).digest('hex');
  }
  return result;
}

async function buildFixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'tabletop-hero-build-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const source = join(root, 'public'), output = join(root, 'dist');
  async function put(path, data) {await mkdir(join(source, path, '..'), {recursive: true}); await writeFile(join(source, path), data);}
  const policies = {
    hero_plus: {label: 'HERO', checkpoint: 'example_model', metadata: 'hero_plus.json', files: {model: 'hero_plus_example.onnx'}},
    other_policy: {label: 'OTHER', metadata: 'other_policy.json', files: {encoder: 'other_encoder.onnx', decoder: 'other_decoder.onnx'}},
  };
  for (const [id, entry] of Object.entries(policies)) {
    await put(`policies/${entry.metadata}`, JSON.stringify({kind: id, observationIdentity: 'preserved', terms:['h00_actions'], termDimensions:[29], historyLength:5, historyLayout:'frame_major_hero_v1', observationDim:145, inputNames:['actor_obs_lower_body','actor_obs_upper_body'], outputName:'action', dofNames:Array.from({length:29},(_,i)=>`joint_${i}`)}));
    for (const path of Object.values(entry.files)) await put(`policies/${path}`, Buffer.from(`unchanged binary ${path}\0`));
  }
  policies.hero_plus.provenance = {files:{model:{sha256:createHash('sha256').update(Buffer.from('unchanged binary hero_plus_example.onnx\0')).digest('hex')}}};
  await put('policies/manifest.json', JSON.stringify({schema: 'fixture', allowCustomPolicy:true, policies}));
  const scenes = {};
  for (const mode of ['hero_plus', 'other_policy']) for (const tableKind of ['workbench', 'round', 'pedestal']) for (const height of [50, 74, 88]) {
    const key = `${mode}_${tableKind}_${height}`, xml = `${key}.xml`, metadata = `${key}.json`;
    scenes[key] = {mode, tableKind, tableHeight: height / 100, xml, metadata};
    await put(`sceneassets/${xml}`, '<mujoco>unchanged physical XML</mujoco>');
    await put(`sceneassets/${metadata}`, JSON.stringify({mode, tableKind, tableHeight: height / 100}));
    await put(`sceneassets/${key}.native_trace.json`, JSON.stringify({source: key}));
  }
  await put('sceneassets/meshes/shared.STL', Buffer.from('shared mesh\0bytes'));
  await put('sceneassets/textures/shared.png', Buffer.from('shared texture\0bytes'));
  await put('sceneassets/manifest.json', JSON.stringify({scenes, assets: {'meshes/shared.STL': {}, 'textures/shared.png': {}},
    compatibilityEdits: ['unchanged'], sourceSha256: {native: 'preserved'}}));
  await cp(source, output, {recursive: true});
  return {root, source, output};
}

test('demo build retains one verified HERO export and nine scenes without altering source assets or retained binaries', async t => {
  const f = await buildFixture(t), before = await fingerprint(f.source);
  const plugin = heroBuildAssetsPlugin();
  plugin.configResolved({root: f.root, publicDir: f.source, build: {outDir: 'dist'}});
  await plugin.closeBundle();
  assert.deepEqual(await fingerprint(f.source), before, 'Source fixtures must remain untouched.');
  const after = await fingerprint(f.output);
  for (const [path, hash] of Object.entries(after)) if (!path.endsWith('/manifest.json'))
    assert.equal(hash, before[path === 'policies/hero.onnx' ? 'policies/hero_plus_example.onnx' : path], path);
  const policies = JSON.parse(await readFile(join(f.output, 'policies/manifest.json')));
  const scenes = JSON.parse(await readFile(join(f.output, 'sceneassets/manifest.json')));
  assert.deepEqual(Object.keys(policies.policies), ['hero_plus']);
  assert.equal(policies.policies.hero_plus.label, 'HERO'); assert.equal(policies.policies.hero_plus.checkpoint, 'example_model');
  assert.equal(policies.policies.hero_plus.files.model, 'hero.onnx');
  assert.ok(!Object.hasOwn(after, 'policies/hero_plus_example.onnx'));
  assert.equal(Object.keys(scenes.scenes).length, 9);
  assert.ok(Object.values(scenes.scenes).every(entry => entry.mode === 'hero_plus'));
  assert.deepEqual(scenes.compatibilityEdits, ['unchanged']); assert.deepEqual(scenes.sourceSha256, {native: 'preserved'});
  assert.ok(!Object.keys(after).some(path => path.includes('other')));
  const once = await fingerprint(f.output); await retainHeroBuildAssets(f.output);
  assert.deepEqual(await fingerprint(f.output), once, 'Repeated filtering is stable.');
});

test('unapproved custom checkpoint or missing scene fails before pruning any build files', async t => {
  const f = await buildFixture(t), path = join(f.output, 'policies/manifest.json');
  const original = JSON.parse(await readFile(path));
  const wrong = structuredClone(original); wrong.allowCustomPolicy = false;
  await writeFile(path, JSON.stringify(wrong)); const wrongBefore = await fingerprint(f.output);
  await assert.rejects(retainHeroBuildAssets(f.output), /verified example ONNX model/);
  assert.deepEqual(await fingerprint(f.output), wrongBefore);
  await writeFile(path, JSON.stringify(original));
  await rm(join(f.output, 'sceneassets/hero_plus_round_74.xml'));
  const missingBefore = await fingerprint(f.output);
  await assert.rejects(retainHeroBuildAssets(f.output), /ENOENT/);
  assert.deepEqual(await fingerprint(f.output), missingBefore);
});

test('build plugin rejects output overlapping the public/ assets directory', () => {
  for (const outDir of ['public', 'public/build', '.']) assert.throws(() => heroBuildAssetsPlugin().configResolved({
    root: '/fixture/client', publicDir: '/fixture/client/public', build: {outDir}}), /separate from the public\/ assets directory/);
});

test('the demo rejects adding, placing or selecting an unavailable apple', async () => {
  const f = dispatchFixture();
  for (const [path, body] of [['/api/add', {kind: 'apple'}], ['/api/place', {kind: 'apple', x: .4, y: -.2}], ['/api/select', {object_id: 'apple'}]]) {
    f.calls.length = 0;
    await assert.rejects(f.dispatch({path, body}), /available object/, path);
    assert.deepEqual(f.calls, [], `${path} must not touch the scene or controller`);
  }
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.doesNotMatch(html, /data-id="apple"|>Apple</, 'The page carries no static Apple control.');
});

test('a changed ONNX binary is rejected even when custom policy export was explicitly allowed',async t=>{
  const f=await buildFixture(t);
  await writeFile(join(f.output,'policies/hero_plus_example.onnx'),'modified model bytes');
  const before=await fingerprint(f.output);
  await assert.rejects(retainHeroBuildAssets(f.output),/model hash differs/);
  assert.deepEqual(await fingerprint(f.output),before);
});

test('inconsistent observation width and mismatched scene checkpoint fail before pruning',async t=>{
  const f=await buildFixture(t),path=join(f.output,'policies/hero_plus.json');
  const metadata=JSON.parse(await readFile(path));
  await writeFile(path,JSON.stringify({...metadata,observationDim:675}));
  await assert.rejects(retainHeroBuildAssets(f.output),/Invalid HERO observation/);
  await writeFile(path,JSON.stringify(metadata));
  const scenePath=join(f.output,'sceneassets/manifest.json'),scene=JSON.parse(await readFile(scenePath));
  await writeFile(scenePath,JSON.stringify({...scene,policySha256:{hero_plus:'different-checkpoint'}}));
  const before=await fingerprint(f.output);
  await assert.rejects(retainHeroBuildAssets(f.output),/different checkpoints/);
  assert.deepEqual(await fingerprint(f.output),before);
});

async function addHandBuildFixture(f) {
  const staging = join(f.root, 'hand-source');
  const hand = join(f.output, 'sceneassets/inspire30');
  await cp(join(f.output, 'sceneassets'), staging, {recursive: true});
  await cp(staging, hand, {recursive: true});
  return hand;
}

test('optional hand scenes retain their complete HERO runtime assets and prune other policies', async t => {
  const f = await buildFixture(t), hand = await addHandBuildFixture(f);
  const result = await retainHeroBuildAssets(f.output);
  assert.deepEqual(result.hands, ['dex3', 'inspire30']);
  const manifest = JSON.parse(await readFile(join(hand, 'manifest.json')));
  assert.equal(Object.keys(manifest.scenes).length, 9);
  assert.ok(Object.keys(await fingerprint(hand)).every(path => !path.includes('other_policy')));
  for (const path of [...Object.keys(manifest.assets), ...Object.values(manifest.scenes).flatMap(scene => [scene.xml, scene.metadata])])
    assert.ok((await readFile(join(hand, path))).length, path);
});

test('an optional hand missing a required scene asset fails before renaming or pruning output files', async t => {
  const f = await buildFixture(t), hand = await addHandBuildFixture(f);
  await rm(join(hand, 'hero_plus_round_74.json'));
  const before = await fingerprint(f.output);
  await assert.rejects(retainHeroBuildAssets(f.output), /ENOENT/);
  assert.deepEqual(await fingerprint(f.output), before);
});

test('an optional hand exported for a different checkpoint fails before modifying output files', async t => {
  const f = await buildFixture(t), hand = await addHandBuildFixture(f), path = join(hand, 'manifest.json');
  const manifest = JSON.parse(await readFile(path));
  await writeFile(path, JSON.stringify({...manifest, policySha256: {hero_plus: 'different-checkpoint'}}));
  const before = await fingerprint(f.output);
  await assert.rejects(retainHeroBuildAssets(f.output), /different checkpoints/);
  assert.deepEqual(await fingerprint(f.output), before);
});

test('a malformed optional hand manifest cannot silently remove the hand from a successful build', async t => {
  const f = await buildFixture(t), hand = await addHandBuildFixture(f);
  await writeFile(join(hand, 'manifest.json'), '{malformed json');
  const before = await fingerprint(f.output);
  await assert.rejects(retainHeroBuildAssets(f.output), SyntaxError);
  assert.deepEqual(await fingerprint(f.output), before);
});

test('carton_grasp config: auto by default, pinned variants survive table edits, invalid values are refused', async () => {
  const f = dispatchFixture();
  await f.dispatch({path: '/api/config', body: {table_height: .5}});
  assert.equal(f.config.carton_grasp, 'auto');
  f.calls.length = 0;
  await f.dispatch({path: '/api/config', body: {carton_grasp: 'spine'}});
  assert.equal(f.config.carton_grasp, 'spine');
  assert.deepEqual(f.calls, ['placements', 'editing', 'controller']); // a controller rebuild, no scene rebuild
  await f.dispatch({path: '/api/config', body: {table_kind: 'round'}});
  assert.equal(f.config.carton_grasp, 'spine'); // kept across an ordinary table edit
  await f.dispatch({path: '/api/config', body: {carton_grasp: null}}); assert.equal(f.config.carton_grasp, 'spine'); // null/undefined keep the current value
  await f.dispatch({path: '/api/config', body: {carton_grasp: 'spine45'}}); assert.equal(f.config.carton_grasp, 'spine'); // legacy alias (the 45 deg name) resolves to the shipped variant
  for (const bad of ['nonsense', 42, '', 'SPINE45', ['spine45'], ['spine']]) {
    const before = structuredClone(f.config);
    await assert.rejects(f.dispatch({path: '/api/config', body: {carton_grasp: bad}}), /auto, end_face, crotch, spine or spine90/);
    assert.deepEqual(f.config, before);
  }
  for (const ok of ['end_face', 'crotch', 'spine90', 'auto']) { await f.dispatch({path: '/api/config', body: {carton_grasp: ok}}); assert.equal(f.config.carton_grasp, ok); }
});
