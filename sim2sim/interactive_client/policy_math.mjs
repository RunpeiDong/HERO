// Native sim2sim policy quaternion conventions, exposed as WXYZ to MuJoCo Web.
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const clip = (v, bound) => Math.max(-bound, Math.min(bound, v));
export const conjugate = q => [q[0], -q[1], -q[2], -q[3]];
export function multiply(a, b) {
  const [w,x,y,z] = a, [v,i,j,k] = b;
  return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j, w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v];
}
export function rotate(q, v) {
  const [w,x,y,z] = q, [a,b,c] = v;
  const tx=2*(y*c-z*b), ty=2*(z*a-x*c), tz=2*(x*b-y*a);
  return [a+w*tx+y*tz-z*ty, b+w*ty+z*tx-x*tz, c+w*tz+x*ty-y*tx];
}
export const rotateInverse = (q, v) => rotate(conjugate(q), v);
export function yaw(q) {
  const [w,x,y,z] = q;
  return Math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z));
}
export function heading(q) {
  const a = yaw(q)/2;
  return [Math.cos(a), 0, 0, Math.sin(a)];
}
export function rollPitch(q) {
  const [w,x,y,z] = q;
  return [Math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)), Math.asin(clip(2*(w*y-z*x), 1))];
}
export function matrix(q) {
  const [w,x,y,z]=q, s=2/Math.max(w*w+x*x+y*y+z*z,1e-12);
  return [1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w),
    s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w),
    s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)];
}
export function rot6Row(q) { const m=matrix(q); return [m[0],m[1],m[3],m[4],m[6],m[7]]; }
export function rot6Column(q) { const m=matrix(q); return [m[0],m[3],m[6],m[1],m[4],m[7]]; }
export function finiteVector(value, length, name) {
  if (!value || value.length !== length || Array.from(value).some(v => !Number.isFinite(v)))
    throw new Error(`${name} must contain ${length} finite values`);
  return value;
}
