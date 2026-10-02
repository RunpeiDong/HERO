/** Presentation-only hat: local X forward, Y left, Z up; brim underside starts at Z = 0. */
import * as THREE from 'three';

const SEGMENTS = 96;
const BRIM_THICKNESS = .0032, EDGE_RADIUS = .0008;
const BRIM_X = .16 - EDGE_RADIUS, BRIM_Y = .155 - EDGE_RADIUS;
const OPENING_X = .086, OPENING_Y = .064;
const CROWN_X = .097, CROWN_Y = .075;

function finishGeometry(positions, indices, seamRows = [], seamOffset = 0) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  // The UV-style angular seam has duplicate positions. Join their normals too.
  const normals = geometry.getAttribute('normal');
  for (const row of seamRows) {
    const a = seamOffset + row * (SEGMENTS + 1), b = a + SEGMENTS;
    const normal = new THREE.Vector3(normals.getX(a) + normals.getX(b),
      normals.getY(a) + normals.getY(b), normals.getZ(a) + normals.getZ(b)).normalize();
    normals.setXYZ(a, normal.x, normal.y, normal.z);
    normals.setXYZ(b, normal.x, normal.y, normal.z);
  }
  geometry.computeBoundingBox(); geometry.computeBoundingSphere();
  return geometry;
}

function wrappedSurface(point, layers, reverse = false) {
  const positions = [], indices = [], stride = SEGMENTS + 1;
  for (let row = 0; row <= layers; row++) for (let i = 0; i <= SEGMENTS; i++)
    positions.push(...point(i / SEGMENTS * Math.PI * 2, row / layers));
  for (let row = 0; row < layers; row++) for (let i = 0; i < SEGMENTS; i++) {
    const a = row * stride + i, b = a + 1, d = a + stride, c = d + 1;
    indices.push(...(reverse ? [a, d, b, b, d, c] : [a, b, d, b, c, d]));
  }
  return finishGeometry(positions, indices, Array.from({length: layers + 1}, (_, i) => i));
}

function brimPoint(theta, radius, top = false) {
  const x = THREE.MathUtils.lerp(OPENING_X, BRIM_X, radius) * Math.cos(theta);
  const y = THREE.MathUtils.lerp(OPENING_Y, BRIM_Y, radius) * Math.sin(theta);
  const eased = radius * radius * (3 - 2 * radius);
  // Broad, quiet side rolls; the front and rear tips rise only slightly.
  const roll = .018 * Math.sin(theta) ** 6 + .0025 * Math.cos(theta) ** 6;
  return [x, y, roll * eased * eased + (top ? BRIM_THICKNESS : 0)];
}

function crownRadii(height) {
  const fullness = .014 * Math.sin(Math.PI * height);
  return [CROWN_X * (1 - .115 * height + fullness), CROWN_Y * (1 - .15 * height + fullness)];
}

function crownTopPoint(theta, radius) {
  const [rx, ry] = crownRadii(1), cosine = Math.cos(theta);
  const x = radius * rx * cosine;
  const y = radius * ry * Math.sin(theta) * (1 - .22 * cosine ** 4);
  // A longitudinal cattleman crease sits between two softly rounded ridges.
  // Let it reach the front/rear shoulders so the crown never reads as a flat cylinder.
  const dome = .013 * (1 - radius * radius);
  const crease = .011 * Math.exp(-((y / .016) ** 2)) * (.7 + .3 * (1 - radius * radius));
  return [x, y, .078 + dome - crease];
}

function crownWallPoint(theta, height, offset = 0) {
  const [rx, ry] = crownRadii(height), cosine = Math.cos(theta);
  return [(rx + offset) * cosine,
    (ry + offset) * Math.sin(theta) * (1 - .22 * height * height * cosine ** 4),
    THREE.MathUtils.lerp(BRIM_THICKNESS, crownTopPoint(theta, 1)[2], height)];
}

function crownCap() {
  const rings = 14, stride = SEGMENTS + 1;
  const positions = crownTopPoint(0, 0), indices = [];
  for (let row = 1; row <= rings; row++) for (let i = 0; i <= SEGMENTS; i++)
    positions.push(...crownTopPoint(i / SEGMENTS * Math.PI * 2, row / rings));
  for (let i = 0; i < SEGMENTS; i++) indices.push(0, i + 1, i + 2);
  for (let row = 0; row < rings - 1; row++) for (let i = 0; i < SEGMENTS; i++) {
    const a = 1 + row * stride + i, b = a + 1, d = a + stride, c = d + 1;
    indices.push(a, d, b, b, d, c);
  }
  return finishGeometry(positions, indices, Array.from({length: rings}, (_, i) => i), 1);
}

function bandPoint(theta, z, offset = .00065) {
  const height = (z - BRIM_THICKNESS) / (crownTopPoint(theta, 1)[2] - BRIM_THICKNESS);
  const point = crownWallPoint(theta, height, offset);
  point[2] = z;
  return point;
}

function piping(point, radius) {
  const points = Array.from({length: SEGMENTS}, (_, i) =>
    new THREE.Vector3(...point(i / SEGMENTS * Math.PI * 2)));
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points, true, 'centripetal'), SEGMENTS, radius, 6, true);
}

function roundedRectangle(width, height, radius, ShapeClass = THREE.Shape) {
  const shape = new ShapeClass(), x = -width / 2, y = -height / 2;
  shape.moveTo(x + radius, y);
  shape.lineTo(x + width - radius, y); shape.quadraticCurveTo(x + width, y, x + width, y + radius);
  shape.lineTo(x + width, y + height - radius); shape.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  shape.lineTo(x + radius, y + height); shape.quadraticCurveTo(x, y + height, x, y + height - radius);
  shape.lineTo(x, y + radius); shape.quadraticCurveTo(x, y, x + radius, y);
  return shape;
}

/** Create an unanimated, self-contained 0.32 m long × 0.31 m wide brown cowboy hat. */
export function createCowboyHat() {
  const group = new THREE.Group();
  group.name = 'G1 cowboy hat';
  group.userData.kind = 'cowboy_hat';
  const felt = new THREE.MeshStandardMaterial({color: '#795033', roughness: .94, metalness: 0});
  const underside = new THREE.MeshStandardMaterial({color: '#604027', roughness: .97, metalness: 0});
  const leather = new THREE.MeshStandardMaterial({color: '#35251d', roughness: .74, metalness: .02});
  const binding = new THREE.MeshStandardMaterial({color: '#4d3424', roughness: .89, metalness: 0});
  const gold = new THREE.MeshStandardMaterial({color: '#b39355', roughness: .44, metalness: .72});
  function add(name, geometry, material) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name; mesh.castShadow = true; mesh.receiveShadow = true;
    group.add(mesh); return mesh;
  }

  add('Rolled felt brim', wrappedSurface((theta, radius) => brimPoint(theta, radius, true), 12, true), felt);
  add('Brim underside', wrappedSurface((theta, radius) => brimPoint(theta, radius), 12), underside);
  add('Brim outer edge', wrappedSurface((theta, height) => {
    const point = brimPoint(theta, 1); point[2] += height * BRIM_THICKNESS; return point;
  }, 1), felt);
  add('Brim opening edge', wrappedSurface((theta, height) =>
    [OPENING_X * Math.cos(theta), OPENING_Y * Math.sin(theta), height * BRIM_THICKNESS], 1, true), underside);
  add('Brim bound rim', piping(theta => {
    const point = brimPoint(theta, 1); point[2] += BRIM_THICKNESS / 2; return point;
  }, EDGE_RADIUS), binding);
  add('Tapered crown', wrappedSurface((theta, height) => crownWallPoint(theta, height), 12), felt);
  add('Creased crown top', crownCap(), felt);
  add('Dark leather hatband', wrappedSurface((theta, height) =>
    bandPoint(theta, THREE.MathUtils.lerp(.010, .024, height)), 3), leather);
  add('Hatband upper welt', piping(theta => bandPoint(theta, .024, .00085), .00045), binding);

  const buckleShape = roundedRectangle(.017, .0095, .0018);
  buckleShape.holes.push(roundedRectangle(.012, .0054, .001, THREE.Path));
  const buckle = new THREE.ExtrudeGeometry(buckleShape, {depth: .0011, steps: 1,
    bevelEnabled: true, bevelSize: .00015, bevelThickness: .00015, bevelSegments: 2, curveSegments: 5});
  buckle.rotateX(-Math.PI / 2);
  const bucklePosition = bandPoint(Math.PI / 2, .017);
  buckle.translate(...bucklePosition);
  add('Small brass buckle', buckle, gold);
  const tongue = new THREE.BoxGeometry(.0115, .0013, .0008);
  tongue.translate(bucklePosition[0], bucklePosition[1] + .0008, bucklePosition[2]);
  add('Buckle tongue', tongue, gold);
  return group;
}
