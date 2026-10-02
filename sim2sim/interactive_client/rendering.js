/** Three.js presentation of worker-produced MuJoCo geometry and poses. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createCowboyHat} from './cowboy_hat.js';
import {createG1HeadLights} from './g1_head_lights.js';
import {PresentationBuffer, PresentationClock} from './presentation.js';

// Official D435i RGB specification, not its wider stereo-depth field of view.
// https://www.realsenseai.com/products/depth-camera-d435i/ (Tech Specs)
// The 1920x1080 RGB mode is rendered at one-third resolution without cropping.
// No device-specific CameraInfo or RGB-to-mount calibration is supplied here.
export const D435I_RGB_CAMERA = Object.freeze({
  model: 'RealSense D435i', stream: 'RGB', width: 640, height: 360,
  sensorWidth: 1920, sensorHeight: 1080, horizontalFovDeg: 69, verticalFovDeg: 42,
  mountBody: 'd435_link', calibration: 'manufacturer_nominal',
});
export function configureEgoProjection(camera) {
  const {width, height, horizontalFovDeg, verticalFovDeg} = D435I_RGB_CAMERA;
  const fx = width / (2 * Math.tan(THREE.MathUtils.degToRad(horizontalFovDeg / 2)));
  const fy = height / (2 * Math.tan(THREE.MathUtils.degToRad(verticalFovDeg / 2)));
  const cx = width / 2, cy = height / 2, near = camera.near;
  camera.fov = verticalFovDeg; camera.aspect = width / height;
  // Preserve both nominal FoV axes and the RGB aspect ratio.
  // Three.js expects vertical FoV.
  camera.projectionMatrix.makePerspective(-near * cx / fx, near * (width - cx) / fx,
    near * cy / fy, -near * (height - cy) / fy, near, camera.far);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  camera.userData.rgbIntrinsics = {width, height, fx, fy, cx, cy,
    source: 'D435i nominal RGB FoV; centered undistorted pinhole approximation'};
}
// URDF body axes are forward/left/up. Three.js camera axes are right/up/back.
const cameraFromMount = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(
  0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1));

// Dex3 shells share the arm's pale source material in the exported URDF.
// Apply their black finish only to palm/finger bodies, retaining arm/wrist colors.
const dex3Finishes = {
  palm: {color: '#202225', roughness: .46, metalness: .12, map: null},
  finger: {color: '#151719', roughness: .68, metalness: .04, map: null},
};

function addAppleSkinColors(geometry) {
  if (geometry.hasAttribute('color')) return;
  const positions = geometry.getAttribute('position');
  geometry.computeBoundingBox();
  const center = geometry.boundingBox.getCenter(new THREE.Vector3());
  const size = geometry.boundingBox.getSize(new THREE.Vector3());
  const colors = new Float32Array(positions.count * 3);
  // Soft, seamless skin variation in mesh space. Build this once; the native
  // color remains the base, and all linear RGB multipliers stay within 4%.
  for (let i = 0; i < positions.count; i++) {
    const x = 2 * (positions.getX(i) - center.x) / Math.max(size.x, 1e-6);
    const y = 2 * (positions.getY(i) - center.y) / Math.max(size.y, 1e-6);
    const z = 2 * (positions.getZ(i) - center.z) / Math.max(size.z, 1e-6);
    const broad = Math.sin(1.65 * x + 1.1 * y - .8 * z + .45) * Math.cos(.65 * x - 1.7 * y + 1.2 * z);
    const soft = Math.sin(3.2 * x + 2.6 * y + 1.3 * z + .2) * Math.cos(1.8 * x - .7 * y - 3.4 * z + .9);
    colors[i * 3] = 1 + .026 * broad + .012 * soft;
    colors[i * 3 + 1] = 1 + .016 * broad - .020 * soft;
    colors[i * 3 + 2] = 1 + .012 * broad - .018 * soft;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

const matrix4 = new THREE.Matrix4();
function rotationMatrix(values, offset = 0, out = new THREE.Matrix4()) {
  return out.set(values[offset], values[offset + 1], values[offset + 2], 0,
    values[offset + 3], values[offset + 4], values[offset + 5], 0,
    values[offset + 6], values[offset + 7], values[offset + 8], 0, 0, 0, 0, 1);
}
const position = new THREE.Vector3(), nextPosition = new THREE.Vector3();
const quaternion = new THREE.Quaternion(), nextQuaternion = new THREE.Quaternion();
function applyPose(object) {
  const changed = !object.position.equals(position) || !object.quaternion.equals(quaternion);
  object.position.copy(position); object.quaternion.copy(quaternion);
  return changed;
}
function bodyPose(object, from, to, alpha, id) {
  position.fromArray(from.bodyPositions, id * 3);
  const a = from.bodyQuaternions, b = to.bodyQuaternions, offset = id * 4;
  quaternion.set(a[offset + 1], a[offset + 2], a[offset + 3], a[offset]).normalize();
  if (from !== to) {
    position.lerp(nextPosition.fromArray(to.bodyPositions, id * 3), alpha);
    quaternion.slerp(nextQuaternion.set(b[offset + 1], b[offset + 2], b[offset + 3], b[offset]).normalize(), alpha);
  }
  return applyPose(object);
}

export class TabletopRenderer {
  constructor({mainCanvas, egoCanvas}) {
    this.mainCanvas = mainCanvas; this.egoCanvas = egoCanvas;
    this.resetPresentation();
    this.mainDirty = true; this.egoDirty = true;
    this.renderer = this._renderer(mainCanvas);
    this.egoRenderer = egoCanvas ? this._renderer(egoCanvas) : null;
    if (this.egoRenderer) {
      this.egoRenderer.setPixelRatio(1); this.egoRenderer.setSize(D435I_RGB_CAMERA.width, D435I_RGB_CAMERA.height, false);
    }
    this._onMainContextRestored = () => {
      this.mainDirty = true; this.renderer.shadowMap.needsUpdate = true;
    };
    this._onEgoContextRestored = () => {
      this.egoDirty = true; this.egoRenderer.shadowMap.needsUpdate = true;
    };
    mainCanvas.addEventListener('webglcontextrestored', this._onMainContextRestored);
    egoCanvas?.addEventListener('webglcontextrestored', this._onEgoContextRestored);
    this._canvasSize = new THREE.Vector2();
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#e8e7e3');
    this.scene.fog = new THREE.Fog('#e8e7e3', 8, 24);
    this.camera = new THREE.PerspectiveCamera(45, 4 / 3, .01, 40);
    this.camera.up.set(0, 0, 1);
    this.egoCamera = new THREE.PerspectiveCamera(D435I_RGB_CAMERA.verticalFovDeg, 16 / 9, .012, 30);
    configureEgoProjection(this.egoCamera);
    this.egoCamera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, mainCanvas);
    // The application owns pointer gestures so canvas and overlay drags agree.
    this.controls.enabled = false;
    this.controls.enableDamping = true; this.controls.dampingFactor = .13;
    this.controls.minDistance = 1.25; this.controls.maxDistance = 6;
    this.controls.maxPolarAngle = Math.PI * .49;
    this.controls.target.set(.38, 0, .66);
    this.meshes = []; this.geometries = new Set(); this.materials = new Set(); this.textures = new Set();
    this.modelGroup = new THREE.Group(); this.scene.add(this.modelGroup);
    this.headAccessoryMount = null; this.cowboyHat = null; this.headLights = null;
    this.previewGroup = new THREE.Group(); this.scene.add(this.previewGroup);
    this._ghostLocals = new Map(); this._ghostMeshes = []; this._previewHelpers = [];
    this._addLights();
    this.setCamera({lookat: [.38, 0, .66], distance: 3.1, azimuth: 62, elevation: -26});
  }

  _renderer(canvas) {
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false, powerPreference: 'high-performance'});
    renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 1.5));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = .85;
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    // Re-presenting a held pose needs no new shadow pass in either context.
    renderer.shadowMap.autoUpdate = false; renderer.shadowMap.needsUpdate = true;
    return renderer;
  }

  _addLights() {
    this.scene.add(new THREE.HemisphereLight('#e8f1ff', '#9b8872', .8));
    const key = new THREE.DirectionalLight('#fff2dd', 2.7);
    key.position.set(-2.5, -3.5, 6); key.target.position.set(.35, 0, .65);
    key.castShadow = true; key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, {left: -2.0, right: 2.0, top: 2.0, bottom: -2.0, near: .2, far: 12});
    key.shadow.normalBias = .008; key.shadow.bias = -.00015;
    this.scene.add(key, key.target);
    const fill = new THREE.DirectionalLight('#d4e4ff', .65); fill.position.set(4, 3, 3.5); this.scene.add(fill);
    const rim = new THREE.DirectionalLight('#fffaf0', 1.2); rim.position.set(-2, 3, 4); this.scene.add(rim);
  }

  _geometry(geom, description, meshCache) {
    const [a, b, c] = geom.size;
    let geometry;
    if (geom.type === 7) {
      if (meshCache.has(geom.meshId)) return meshCache.get(geom.meshId);
      const mesh = description.meshes[geom.meshId];
      geometry = new THREE.BufferGeometry();
      if (mesh.uv) {
        // Textured mesh: per-corner positions and UVs (see the worker's unrolled layout); flat normals per face.
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(mesh.unrolledPositions), 3));
        geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(mesh.uv), 2));
      } else {
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(mesh.positions), 3));
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(mesh.indices), 1));
      }
      geometry.computeVertexNormals(); meshCache.set(geom.meshId, geometry);
    } else if (geom.type === 0) geometry = new THREE.PlaneGeometry(30, 30);
    else if (geom.type === 2) geometry = new THREE.SphereGeometry(a, 24, 16);
    else if (geom.type === 3) { geometry = new THREE.CapsuleGeometry(a, 2 * b, 8, 20); geometry.rotateX(Math.PI / 2); }
    else if (geom.type === 4) { geometry = new THREE.SphereGeometry(1, 24, 16); geometry.scale(a, b, c); }
    else if (geom.type === 5) { geometry = new THREE.CylinderGeometry(a, a, 2 * b, 48); geometry.rotateX(Math.PI / 2); }
    else if (geom.type === 6) geometry = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    else return null;
    this.geometries.add(geometry); return geometry;
  }

  _texture(texture, repeat, geom) {
    if (!texture || texture.type !== 0) return null;
    const pixels = texture.width * texture.height, rgba = new Uint8Array(pixels * 4);
    for (let i = 0; i < pixels; i++) {
      for (let k = 0; k < 3; k++) rgba[4 * i + k] = texture.data[i * texture.channels + Math.min(k, texture.channels - 1)];
      rgba[4 * i + 3] = texture.channels === 4 ? texture.data[4 * i + 3] : 255;
    }
    const map = new THREE.DataTexture(rgba, texture.width, texture.height, THREE.RGBAFormat);
    if (geom.type === 7) {
      // UV-mapped mesh (e.g. the YCB carton): the image is a photo atlas in sRGB, applied once without tiling.
      // MuJoCo already stores OBJ texture coordinates as 1 - v and keeps the PNG rows top-down, so the data must
      // not be flipped again; the 1024 px print is minified on screen, so filter it with mipmaps.
      map.wrapS = map.wrapT = THREE.ClampToEdgeWrapping; map.flipY = false; map.colorSpace = THREE.SRGBColorSpace;
      map.minFilter = THREE.LinearMipmapLinearFilter; map.magFilter = THREE.LinearFilter; map.generateMipmaps = true;
    } else {
      map.wrapS = map.wrapT = THREE.RepeatWrapping;
      const scale = geom.type === 0 ? [30, 30] : [Math.max(.5, geom.size[0] * 2), Math.max(.5, geom.size[1] * 2)];
      map.repeat.set((repeat?.[0] ?? 1) * scale[0], (repeat?.[1] ?? 1) * scale[1]);
      map.colorSpace = THREE.LinearSRGBColorSpace;
    }
    map.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    map.needsUpdate = true; this.textures.add(map); return map;
  }

  build(description) {
    this._clearModel(); this.description = description;
    const cache = new Map();
    for (const geom of description.geoms) {
      if (!geom.visible) continue;
      const geometry = this._geometry(geom, description, cache);
      if (!geometry) continue;
      const source = description.materials[geom.materialId] ?? {};
      const rgba = geom.rgba;
      const dex3Part = /^(?:left|right)_hand_/.test(geom.bodyName) ? (/^(?:left|right)_hand_(?:palm|base)/.test(geom.bodyName) ? 'palm' : 'finger') : undefined;
      const appleSkin = geom.name === 'demo_apple_geom_1';
      const appleStem = geom.name === 'demo_apple_geom_2', appleLeaf = geom.name === 'demo_apple_geom_3';
      if (appleSkin) addAppleSkinColors(geometry);
      const Material = appleSkin ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;
      const material = new Material({color: new THREE.Color().setRGB(...rgba.slice(0, 3), THREE.SRGBColorSpace),
        roughness: Math.max(.24, .77 - (source.shininess ?? .3) * .48),
        metalness: geom.bodyName.includes('wrist') || geom.bodyName.includes('ankle') ? .22 : .08,
        map: this._texture(description.textures?.[source.textureId], source.textureRepeat, geom),
        opacity: rgba[3], transparent: rgba[3] < 1,
        ...(dex3Part ? dex3Finishes[dex3Part === 'palm' ? 'palm' : 'finger'] : {}),
        ...(appleSkin ? {metalness: 0, roughness: .70, clearcoat: 0, vertexColors: true} : {}),
        ...(appleStem ? {metalness: 0, roughness: .82} : {}),
        ...(appleLeaf ? {metalness: 0, roughness: .65, side: THREE.DoubleSide} : {})});
      this.materials.add(material);
      const mesh = new THREE.Mesh(geometry, material); mesh.name = geom.name || `${geom.bodyName}:${geom.id}`;
      mesh.castShadow = geom.type !== 0; mesh.receiveShadow = true; mesh.userData.geom = geom;
      this.meshes[geom.id] = mesh; this.modelGroup.add(mesh);
    }
    this.headBodyId = description.bodyNames?.indexOf('head_link') ?? -1;
    if (this.headBodyId >= 0) {
      this.headAccessoryMount = new THREE.Group(); this.headAccessoryMount.name = 'G1 head accessories';
      this.cowboyHat = createCowboyHat(); this.headLights = createG1HeadLights();
      this.headAccessoryMount.add(this.cowboyHat, this.headLights);
      this.headAccessoryMount.traverse(object => {
        if (object.geometry) this.geometries.add(object.geometry);
        for (const material of Array.isArray(object.material) ? object.material : [object.material])
          if (material) this.materials.add(material);
      });
      this.headAccessoryMount.visible = false;
      this.modelGroup.add(this.headAccessoryMount);
    }
    this.egoMountBodyId = description.bodyNames?.indexOf(D435I_RGB_CAMERA.mountBody) ?? -1;
    if (this.egoMountBodyId < 0) throw new Error('The scene is missing its URDF D435 camera mount.');
    configureEgoProjection(this.egoCamera);
    this.setCamera(description.homeCamera);
    this._firstSnapshot = null;
    if (this.renderer.shadowMap) this.renderer.shadowMap.needsUpdate = true;
    if (this.egoRenderer?.shadowMap) this.egoRenderer.shadowMap.needsUpdate = true;
    this.mainDirty = this.egoDirty = true;
    return this;
  }

  setCamera(options = {}) {
    const preset = options.preset === 'home' ? this.description?.homeCamera
      : options.preset === 'side' ? {lookat: [.35, -.15, .70], distance: 2.7, azimuth: 90, elevation: -22} : null;
    this.cameraOptions = {...this.cameraOptions, ...preset, ...options};
    const {lookat = [.38, 0, .66], distance = 3.1, azimuth = 62, elevation = -26, fovy = 45} = this.cameraOptions;
    const az = azimuth * Math.PI / 180, el = elevation * Math.PI / 180;
    this.controls.target.fromArray(lookat);
    // MuJoCo's azimuth points from the camera toward the look-at position.
    this.camera.position.set(lookat[0] - distance * Math.cos(az) * Math.cos(el),
      lookat[1] - distance * Math.sin(az) * Math.cos(el), lookat[2] - distance * Math.sin(el));
    this.camera.fov = fovy; this.camera.updateProjectionMatrix(); this.camera.lookAt(this.controls.target); this.controls.update();
    this.mainDirty = true;
  }

  _rememberOpenHands(snapshot) {
    for (const side of ['left', 'right']) {
      const palmId = this.description.palmBodyIds[side];
      const q = snapshot.bodyQuaternions.subarray(palmId * 4, palmId * 4 + 4);
      const palm = new THREE.Matrix4().compose(new THREE.Vector3().fromArray(snapshot.bodyPositions, palmId * 3),
        new THREE.Quaternion(q[1], q[2], q[3], q[0]), new THREE.Vector3(1, 1, 1));
      const inverse = palm.clone().invert();
      const rows = [];
      for (const mesh of this.meshes) {
        if (!mesh) continue;
        const {bodyName} = mesh.userData.geom;
        if (!bodyName.startsWith(`${side}_hand_`) && bodyName !== `${side}_wrist_yaw_link`) continue;
        mesh.updateMatrix(); rows.push({geometry: mesh.geometry, matrix: inverse.clone().multiply(mesh.matrix)});
      }
      this._ghostLocals.set(side, rows);
    }
  }

  _preview(preview) {
    const signature = preview ? JSON.stringify([preview.visible, preview.hand, preview.position_w, preview.quaternion_wxyz, preview.approach_start_w]) : '';
    if (signature === this._previewSignature) return;
    this._previewSignature = signature;
    this._clearPreview();
    if (!preview || preview.visible === false) return;
    const root = new THREE.Group();
    root.position.fromArray(preview.position_w);
    const q = preview.quaternion_wxyz; root.quaternion.set(q[1], q[2], q[3], q[0]);
    for (const row of this._ghostLocals.get(preview.hand) ?? []) {
      const material = new THREE.MeshStandardMaterial({color: '#6daeb4', opacity: .30, transparent: true,
        depthWrite: false, roughness: .55, metalness: .05});
      this.materials.add(material);
      const mesh = new THREE.Mesh(row.geometry, material); mesh.matrix.copy(row.matrix); mesh.matrixAutoUpdate = false;
      mesh.renderOrder = 3; root.add(mesh); this._ghostMeshes.push(mesh);
    }
    const axes = new THREE.AxesHelper(.10); axes.renderOrder = 5; root.add(axes); this._previewHelpers.push(axes);
    this.previewGroup.add(root);
    const start = new THREE.Vector3().fromArray(preview.approach_start_w);
    const direction = new THREE.Vector3().fromArray(preview.position_w).sub(start), length = direction.length();
    if (length > 1e-6) {
      const arrow = new THREE.ArrowHelper(direction.normalize(), start, length, '#df9a31', .032, .018);
      arrow.renderOrder = 5; this.previewGroup.add(arrow); this._previewHelpers.push(arrow);
    }
  }

  update(snapshot, graspPreview = null) {
    if (!this.description || !snapshot) return;
    // Immediate update remains available to callers that do not use the RAF buffer.
    this.resetPresentation();
    this.snapshot = snapshot;
    this._applyPosePair(snapshot, snapshot, 0);
    this._preview(graspPreview);
  }

  resetPresentation() {
    this.presentation ??= new PresentationBuffer();
    this.presentationClock ??= new PresentationClock();
    this.presentation.clear(); this.presentationClock.reset();
    this._poseQuaternions = new WeakMap();
    this.snapshot = null;
  }

  _geomQuaternions(snapshot) {
    let values = this._poseQuaternions.get(snapshot);
    if (values) return values;
    values = new Float64Array(snapshot.geomMatrices.length / 9 * 4);
    for (let id = 0; id < values.length / 4; id++)
      quaternion.setFromRotationMatrix(rotationMatrix(snapshot.geomMatrices, id * 9, matrix4)).normalize().toArray(values, id * 4);
    this._poseQuaternions.set(snapshot, values);
    return values;
  }

  enqueueSnapshot(snapshot, graspPreview = null, timestamp = performance.now()) {
    if (!this.description || !snapshot) return;
    this._geomQuaternions(snapshot); // Convert matrices only on receipt, not on every display frame.
    const snap = this.presentation.push(snapshot, timestamp);
    this.snapshot = snapshot;
    if (snap) this._applyPosePair(snapshot, snapshot, 0);
    this._preview(graspPreview);
  }

  _applyPosePair(from, to, alpha) {
    const a = this._geomQuaternions(from), b = this._geomQuaternions(to);
    let changed = false;
    for (let id = 0; id < this.meshes.length; id++) {
      const mesh = this.meshes[id]; if (!mesh) continue;
      position.fromArray(from.geomPositions, id * 3); quaternion.fromArray(a, id * 4);
      if (from !== to) {
        position.lerp(nextPosition.fromArray(to.geomPositions, id * 3), alpha);
        quaternion.slerp(nextQuaternion.fromArray(b, id * 4), alpha);
      }
      changed = applyPose(mesh) || changed;
      const visible = to.geomRGBA[id * 4 + 3] > .001;
      changed = mesh.visible !== visible || changed; mesh.visible = visible;
    }
    if (this.headAccessoryMount) {
      const head = this.headBodyId;
      changed = bodyPose(this.headAccessoryMount, from, to, alpha, head) || changed;
      if (!this.headAccessoryMount.visible) {
        // MuJoCo centers mesh vertices in their principal axes. Recover the
        // visible head in body coordinates once. Seat the brim around the
        // upper head, letting the helmet enter the hat rather than balancing
        // the brim on its highest vertex.
        this.headAccessoryMount.updateMatrix();
        const inverseHead = this.headAccessoryMount.matrix.clone().invert();
        const bounds = new THREE.Box3(), point = new THREE.Vector3();
        for (const mesh of this.meshes) {
          if (!mesh || mesh.userData.geom.bodyId !== head) continue;
          mesh.updateMatrix();
          const local = inverseHead.clone().multiply(mesh.matrix), positions = mesh.geometry.getAttribute('position');
          for (let i = 0; i < positions.count; i++)
            bounds.expandByPoint(point.fromBufferAttribute(positions, i).applyMatrix4(local));
        }
        if (!bounds.isEmpty()) {
          this.cowboyHat.position.set((bounds.min.x + bounds.max.x) / 2, (bounds.min.y + bounds.max.y) / 2, bounds.max.z - .040);
          this.headAccessoryMount.visible = true;
          changed = true;
        }
      }
    }
    // Follow the URDF camera mount through waist and body motion.
    bodyPose(this.egoCamera, from, to, alpha, this.egoMountBodyId);
    this.egoCamera.quaternion.multiply(cameraFromMount);
    if (!this._firstSnapshot) { this._rememberOpenHands(from); this._firstSnapshot = true; }
    if (changed) {
      if (this.renderer.shadowMap) this.renderer.shadowMap.needsUpdate = true;
      if (this.egoRenderer?.shadowMap) this.egoRenderer.shadowMap.needsUpdate = true;
    }
    this.mainDirty = this.egoDirty = true;
  }

  render(timestamp) {
    const timed = Number.isFinite(timestamp);
    if (timed && !this.presentationClock.shouldRender(timestamp)) return;
    if (timed) {
      const pose = this.presentation.sample(timestamp);
      if (pose) this._applyPosePair(pose.from, pose.to, pose.alpha);
      // Submit real WebGL frames at the target cadence, even when IK holds a pose.
      if (this.snapshot) this.mainDirty = this.egoDirty = true;
    }
    const width = Math.max(1, Math.round(this.mainCanvas.clientWidth || this.mainCanvas.width || 1024));
    const height = Math.max(1, Math.round(this.mainCanvas.clientHeight || this.mainCanvas.height || 768));
    const size = this.renderer.getSize(this._canvasSize);
    if (size.x !== width || size.y !== height) {
      this.renderer.setSize(width, height, false); this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
      this.mainDirty = true;
    }
    if (this.controls.update()) this.mainDirty = true;
    if (this.mainDirty) {
      this.renderer.render(this.scene, this.camera); this.mainDirty = false;
    }
    if (this.egoRenderer && this.egoDirty) {
      this.egoRenderer.render(this.scene, this.egoCamera);
      this.egoDirty = false;
    }
  }
  egoImage() {
    if (!this.egoRenderer) return null;
    // The default WebGL drawing buffer may be discarded after presentation.
    // Draw immediately before the synchronous capture, even when the pose is unchanged.
    this.egoRenderer.render(this.scene, this.egoCamera);
    this.egoDirty = false;
    return this.egoCanvas.toDataURL('image/png');
  }

  _clearModel() {
    this.resetPresentation();
    this.modelGroup.clear(); this._clearPreview();
    for (const geometry of this.geometries) geometry.dispose();
    for (const material of this.materials) material.dispose();
    for (const texture of this.textures) texture.dispose();
    this.geometries.clear(); this.materials.clear(); this.textures.clear(); this.meshes = [];
    this.headAccessoryMount = null; this.cowboyHat = null; this.headLights = null; this.headBodyId = -1;
    this._ghostLocals.clear(); this._ghostMeshes = []; this._previewSignature = null;
  }
  _clearPreview() {
    for (const mesh of this._ghostMeshes) { mesh.material.dispose(); this.materials.delete(mesh.material); }
    for (const helper of this._previewHelpers) helper.dispose();
    this._ghostMeshes = []; this._previewHelpers = []; this.previewGroup.clear();
  }
  dispose() {
    this.mainCanvas.removeEventListener('webglcontextrestored', this._onMainContextRestored);
    this.egoCanvas?.removeEventListener('webglcontextrestored', this._onEgoContextRestored);
    this._clearModel(); this.controls.dispose();
    this.scene.traverse(object => { if (object.isLight) object.dispose(); });
    this.renderer.dispose(); this.egoRenderer?.dispose();
  }
}
