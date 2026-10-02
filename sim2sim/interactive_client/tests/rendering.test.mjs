import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {TabletopRenderer, D435I_RGB_CAMERA, configureEgoProjection} from '../rendering.js';

// Use the real scene/camera/update paths with a counted GPU boundary. These
// checks need no display server and catch redraws that would repeat shadow passes.
function harness() {
  const view = Object.create(TabletopRenderer.prototype);
  const gpu = () => ({
    draws: 0, size: new THREE.Vector2(800, 600), bufferReady: false,
    shadowDraws: 0, shadowMap: {autoUpdate: false, needsUpdate: true},
    getSize(out) { return out.copy(this.size); },
    setSize(w, h) { this.size.set(w, h); },
    render() {
      this.draws++; this.bufferReady = true;
      if (this.shadowMap.needsUpdate) this.shadowDraws++;
      this.shadowMap.needsUpdate = false;
    },
  });
  Object.assign(view, {
    mainCanvas: {clientWidth: 800, clientHeight: 600},
    renderer: gpu(), egoRenderer: gpu(), _canvasSize: new THREE.Vector2(),
    mainDirty: true, egoDirty: true, scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(45, 4 / 3, .01, 40),
    egoCamera: new THREE.PerspectiveCamera(42, 16 / 9, .012, 30),
    controls: {target: new THREE.Vector3(), changed: false,
      update() { const changed = this.changed; this.changed = false; return changed; }},
    modelGroup: new THREE.Group(), previewGroup: new THREE.Group(),
    meshes: [], geometries: new Set(), materials: new Set(), textures: new Set(),
    _ghostLocals: new Map(), _ghostMeshes: [], _previewHelpers: [],
  });
  view.camera.up.set(0, 0, 1);
  view.egoCanvas = {toDataURL(type) {
    assert.equal(type, 'image/png'); assert.equal(view.egoRenderer.bufferReady, true);
    return 'data:image/png;base64,current-frame';
  }};
  return view;
}
const description = {geoms: [], materials: [], textures: [], meshes: [],
  bodyNames: ['d435_link'],
  palmBodyIds: {left: 0, right: 0}, egoCameraId: 0, egoFovy: 86,
  homeCamera: {lookat: [.38, 0, .66], distance: 3.1, azimuth: 62, elevation: -26, fovy: 45}};
const snapshot = () => ({geomPositions: new Float32Array(), geomMatrices: new Float32Array(),
  geomRGBA: new Float32Array(), bodyPositions: new Float32Array(3), bodyQuaternions: new Float32Array([1, 0, 0, 0]),
  cameraPositions: new Float32Array([.1, .2, 1.2]), cameraMatrices: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1])});
const counts = view => [view.renderer.draws, view.egoRenderer.draws];

test('idle frames do not repeat either draw; each scene update refreshes both views', () => {
  const view = harness();
  view.build(description); view.update(snapshot()); view.render();
  assert.deepEqual(counts(view), [1, 1]);
  for (let i = 0; i < 300; i++) view.render();
  assert.deepEqual(counts(view), [1, 1]);
  view.update(snapshot()); view.render();
  assert.deepEqual(counts(view), [2, 2]);
  view.build(description); view.render();
  assert.deepEqual(counts(view), [3, 3]);
});

test('camera input and control motion redraw the main view on the next animation frame', () => {
  const view = harness(); view.build(description); view.render();
  view.setCamera({azimuth: 90, elevation: -30, distance: 3, lookat: [.4, 0, .7]});
  assert.ok(Math.abs(view.camera.position.x - .4) < 1e-12);
  assert.ok(Math.abs(view.camera.position.y + 3 * Math.cos(Math.PI / 6)) < 1e-12);
  assert.ok(Math.abs(view.camera.position.z - 2.2) < 1e-12);
  view.render(); assert.deepEqual(counts(view), [2, 1]);
  view.controls.changed = true;
  view.render(); assert.deepEqual(counts(view), [3, 1]);
  view.render(); assert.deepEqual(counts(view), [3, 1]);
});

test('canvas resize redraws the main view and keeps the ego camera unchanged', () => {
  const view = harness(); view.build(description); view.render();
  view.mainCanvas.clientWidth = 1200;
  view.render();
  assert.deepEqual(counts(view), [2, 1]);
  assert.equal(view.camera.aspect, 2);
  assert.equal(view.camera.fov, 45);
  assert.equal(view.egoCamera.aspect, 16 / 9);
  assert.equal(view.egoCamera.fov, 42);
  view.render(); assert.deepEqual(counts(view), [2, 1]);
});

test('ego capture forces an immediate draw after the presentation buffer is discarded', () => {
  const view = harness(); view.build(description); view.render();
  view.egoRenderer.bufferReady = false;
  assert.equal(view.egoImage(), 'data:image/png;base64,current-frame');
  assert.deepEqual(counts(view), [1, 2]);
  view.render(); assert.deepEqual(counts(view), [1, 2]);
  view.update(snapshot()); view.egoImage(); view.render();
  assert.deepEqual(counts(view), [2, 3]);
});

test('D435i RGB frustum preserves nominal horizontal and vertical FoV at 16:9', () => {
  const camera = new THREE.PerspectiveCamera(86, 4 / 3, .012, 30);
  configureEgoProjection(camera);
  const ray = (x, y) => new THREE.Vector3(x, y, 0).unproject(camera).normalize();
  const horizontal = THREE.MathUtils.radToDeg(ray(-1, 0).angleTo(ray(1, 0)));
  const vertical = THREE.MathUtils.radToDeg(ray(0, -1).angleTo(ray(0, 1)));
  assert.ok(Math.abs(horizontal - 69) < 1e-10);
  assert.ok(Math.abs(vertical - 42) < 1e-10);
  assert.equal(D435I_RGB_CAMERA.width / D435I_RGB_CAMERA.height, 16 / 9);
  assert.equal(D435I_RGB_CAMERA.sensorWidth / D435I_RGB_CAMERA.width, 3);
  assert.equal(D435I_RGB_CAMERA.sensorHeight / D435I_RGB_CAMERA.height, 3);
  assert.ok(ray(0, 0).distanceTo(new THREE.Vector3(0, 0, -1)) < 1e-12);
});

test('first-person pose follows the URDF mount and its forward/left/up axes', () => {
  const view = harness(); view.build(description);
  const frame = snapshot(), mount = new THREE.Quaternion().setFromEuler(new THREE.Euler(.1, .8307767239493009, .3));
  frame.bodyPositions.set([.0576235, .01753, 1.2]);
  frame.bodyQuaternions.set([mount.w, mount.x, mount.y, mount.z]);
  view.update(frame);
  assert.ok(view.egoCamera.position.distanceTo(new THREE.Vector3(...frame.bodyPositions)) < 1e-12);
  const actual = new THREE.Vector3(0, 0, -1).applyQuaternion(view.egoCamera.quaternion);
  const expected = new THREE.Vector3(1, 0, 0).applyQuaternion(mount);
  assert.ok(actual.distanceTo(expected) < 1e-7);
  assert.ok(new THREE.Vector3(1, 0, 0).applyQuaternion(view.egoCamera.quaternion)
    .distanceTo(new THREE.Vector3(0, -1, 0).applyQuaternion(mount)) < 1e-7);
  // The obsolete virtual camera arrays deliberately disagree with the mount.
  assert.notDeepEqual(view.egoCamera.position.toArray(), Array.from(frame.cameraPositions));
});

test('UV-mapped meshes render unrolled with per-corner texture coordinates and an sRGB atlas applied once', () => {
  const view = harness(); view.renderer.capabilities = {getMaxAnisotropy: () => 4};
  const mesh = {positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]), indices: new Uint32Array([0, 1, 2, 2, 1, 3]),
    unrolledPositions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]), uv: new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1])};
  const geom = {id: 0, type: 7, meshId: 0, size: [1, 1, 1], rgba: [1, 1, 1, 1], materialId: 0, bodyName: 'demo_cracker_box', name: 'demo_cracker_box_geom_1', visible: true};
  const texture = {type: 0, width: 2, height: 2, channels: 3, data: new Uint8Array(12).fill(200)};
  const geometry = view._geometry(geom, {meshes: [mesh]}, new Map());
  assert.equal(geometry.getIndex(), null, 'unrolled: no index buffer');
  assert.equal(geometry.getAttribute('position').count, 6); assert.equal(geometry.getAttribute('uv').count, 6);
  assert.deepEqual(Array.from(geometry.getAttribute('uv').array.slice(0, 4)), [0, 0, 1, 0]);
  const map = view._texture(texture, [1.5, 1.5], geom);
  assert.equal(map.flipY, false, 'MuJoCo stores 1 - v for OBJ texture coordinates and top-down rows: no second flip');
  assert.equal(map.colorSpace, THREE.SRGBColorSpace); assert.equal(map.wrapS, THREE.ClampToEdgeWrapping);
  assert.equal(map.minFilter, THREE.LinearMipmapLinearFilter); assert.equal(map.magFilter, THREE.LinearFilter); assert.equal(map.generateMipmaps, true);
  assert.deepEqual([map.repeat.x, map.repeat.y], [1, 1], 'a photo atlas is never tiled');
  const box = {...geom, type: 6, size: [.2, .3, .1]};
  const tiled = view._texture(texture, [1.5, 1.5], box);
  assert.equal(tiled.flipY, false); assert.equal(tiled.colorSpace, THREE.LinearSRGBColorSpace); assert.ok(tiled.repeat.x > .7, 'procedural tiling unchanged for primitives');
  // an indexed mesh without uv keeps the original path
  const plain = view._geometry({...geom, meshId: 1}, {meshes: [mesh, {positions: mesh.positions, indices: mesh.indices}]}, new Map());
  assert.equal(plain.getIndex().count, 6); assert.equal(plain.getAttribute('uv'), undefined);
});

function movingView() {
  const view = harness();
  view.build({...description, bodyNames: ['head_link', 'd435_link'], geoms: [{id: 0, type: 6,
    size: [.05, .05, .05], rgba: [1, 1, 1, 1], materialId: 0,
    bodyName: 'head_link', bodyId: 0, name: 'test-head', visible: true}]});
  return view;
}
function movingFrame(time, x, angle) {
  const c = Math.cos(angle), s = Math.sin(angle), w = Math.cos(angle / 2), z = Math.sin(angle / 2);
  return {...snapshot(), time, activeObjectId: 'uiuc_i',
    geomPositions: new Float32Array([x, 0, 0]),
    geomMatrices: new Float32Array([c, -s, 0, s, c, 0, 0, 0, 1]),
    geomRGBA: new Float32Array([1, 1, 1, 1]),
    bodyPositions: new Float32Array([x, 0, 0, x, 0, 0]),
    bodyQuaternions: new Float32Array([w, 0, 0, z, w, 0, 0, z])};
}

test('high refresh presentation interpolates geometry, head accessories and ego camera together without mutating physics', () => {
  const view = movingView(), a = movingFrame(0, 0, 0), b = movingFrame(.04, .2, Math.PI / 2);
  const originals = structuredClone([a, b]);
  view.enqueueSnapshot(a, null, 0); view.render(0);
  view.enqueueSnapshot(b, null, 40); view.render(80); // 60 ms display delay -> midpoint at 20 ms.
  const expected = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 4);
  for (const object of [view.meshes[0], view.headAccessoryMount]) {
    assert.ok(Math.abs(object.position.x - .1) < 1e-6);
    assert.ok(object.quaternion.angleTo(expected) < 1e-6);
    assert.ok(Math.abs(object.quaternion.length() - 1) < 1e-7);
  }
  assert.ok(Math.abs(view.egoCamera.position.x - .1) < 1e-6);
  assert.ok(new THREE.Vector3(0, 0, -1).applyQuaternion(view.egoCamera.quaternion)
    .distanceTo(new THREE.Vector3(1, 0, 0).applyQuaternion(expected)) < 1e-6);
  assert.deepEqual([a, b], originals);
  assert.deepEqual(counts(view), [2, 2]);
  // Both views keep presenting at the target cadence while holding the actual endpoint.
  view.render(120); view.render(140); view.render(5000);
  assert.ok(Math.abs(view.meshes[0].position.x - .2) < 1e-6);
  assert.deepEqual(counts(view), [5, 5]);
});

test('presentation takes the shortest quaternion arc across 179 degrees and handles antipodal body quaternions', () => {
  const view = movingView(), a = movingFrame(0, 0, THREE.MathUtils.degToRad(179));
  const b = movingFrame(.04, .1, THREE.MathUtils.degToRad(-179));
  // q and -q describe the same orientation; SLERP must remain finite and normalized.
  for (let i = 0; i < b.bodyQuaternions.length; i++) b.bodyQuaternions[i] *= -1;
  view.enqueueSnapshot(a, null, 0); view.enqueueSnapshot(b, null, 40); view.render(80);
  const halfTurn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI);
  assert.ok(view.meshes[0].quaternion.angleTo(halfTurn) < 1e-6);
  assert.ok(view.headAccessoryMount.quaternion.angleTo(halfTurn) < 1e-6);
});

test('Reset and model rebuild discard interpolated history; long worker stalls resume at the real pose', () => {
  const view = movingView();
  view.enqueueSnapshot(movingFrame(0, 0, 0), null, 0);
  view.enqueueSnapshot(movingFrame(.04, .2, 1), null, 40); view.render(80);
  view.resetPresentation(); view.render(100);
  assert.ok(Math.abs(view.meshes[0].position.x - .1) < 1e-6, 'Reset freezes the visible pose until its new model arrives.');
  view.build({...description, bodyNames: ['d435_link']});
  const fresh = snapshot(); fresh.time = 0; fresh.bodyPositions[0] = .35;
  view.enqueueSnapshot(fresh, null, 110); view.render(120);
  assert.ok(Math.abs(view.egoCamera.position.x - .35) < 1e-6);
  const resumed = {...fresh, time: .02, bodyPositions: new Float32Array([.6, 0, 0])};
  view.enqueueSnapshot(resumed, null, 1000); view.render(1000);
  assert.ok(Math.abs(view.egoCamera.position.x - .6) < 1e-6, 'Do not turn a long IK pause into slow motion.');
});

test('held-pose display frames reuse shadows; motion and model changes refresh both shadow maps', () => {
  const view = movingView();
  view.enqueueSnapshot(movingFrame(0, 0, 0), null, 0); view.render(0);
  for (let i = 1; i <= 120; i++) view.render(i * 1000 / 60);
  assert.deepEqual(counts(view), [121, 121]);
  assert.deepEqual([view.renderer.shadowDraws, view.egoRenderer.shadowDraws], [1, 1]);
  view.enqueueSnapshot(movingFrame(.04, .2, 1), null, 2020); view.render(2040);
  assert.deepEqual([view.renderer.shadowDraws, view.egoRenderer.shadowDraws], [2, 2]);
  view.build(description); view.render(2060);
  assert.deepEqual([view.renderer.shadowDraws, view.egoRenderer.shadowDraws], [3, 3]);
});
