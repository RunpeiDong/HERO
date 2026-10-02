/** Small deterministic numerical routines. Quaternions are WXYZ throughout. */
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const smooth = t => { const u = clamp(t, 0, 1); return u * u * (3 - 2 * u); };
export const add = (a, b) => a.map((x, i) => x + b[i]);
export const sub = (a, b) => a.map((x, i) => x - b[i]);
export const scale = (a, s) => a.map(x => x * s);
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = a => Math.sqrt(dot(a, a));
export const unit = a => scale(a, 1 / Math.max(norm(a), 1e-15));
export const lerp = (a, b, t) => a.map((x, i) => x + t * (b[i] - x));
export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const eye3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];
export const transpose3 = a => [a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]];
export const matVec = (a, v) => [dot(a.slice(0,3),v),dot(a.slice(3,6),v),dot(a.slice(6,9),v)];
export function matMul(a,b) { const t=transpose3(b); return a.flatMap ? [0,1,2].flatMap(i=>[0,1,2].map(j=>dot(a.slice(3*i,3*i+3),t.slice(3*j,3*j+3)))) : matMul(Array.from(a),Array.from(b)); }
export const rotateY = a => [Math.cos(a),0,Math.sin(a),0,1,0,-Math.sin(a),0,Math.cos(a)];
export const rotateZ = a => [Math.cos(a),-Math.sin(a),0,Math.sin(a),Math.cos(a),0,0,0,1];
export const quatConj = q => [q[0],-q[1],-q[2],-q[3]];
export function quatMul(a,b) { return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2], a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]; }
export function quatToMat(q) { const [w,x,y,z]=unit(Array.from(q)); return [1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]; }
export function matToQuat(m) {
  const t=m[0]+m[4]+m[8]; let q,s;
  if(t>0){s=2*Math.sqrt(1+t);q=[s/4,(m[7]-m[5])/s,(m[2]-m[6])/s,(m[3]-m[1])/s];}
  else if(m[0]>m[4]&&m[0]>m[8]){s=2*Math.sqrt(1+m[0]-m[4]-m[8]);q=[(m[7]-m[5])/s,s/4,(m[1]+m[3])/s,(m[2]+m[6])/s];}
  else if(m[4]>m[8]){s=2*Math.sqrt(1+m[4]-m[0]-m[8]);q=[(m[2]-m[6])/s,(m[1]+m[3])/s,s/4,(m[5]+m[7])/s];}
  else{s=2*Math.sqrt(1+m[8]-m[0]-m[4]);q=[(m[3]-m[1])/s,(m[2]+m[6])/s,(m[5]+m[7])/s,s/4];}
  return unit(q);
}
export function quatBlend(a,b,t) { let end=Array.from(b);if(dot(a,end)<0)end=scale(end,-1);return unit(lerp(Array.from(a),end,t)); }
export const rotationBlend = (a,b,t) => quatToMat(quatBlend(matToQuat(a),matToQuat(b),t));
export function rotationError(target,current) {
  let q=quatMul(matToQuat(target),quatConj(matToQuat(current)));if(q[0]<0)q=scale(q,-1);
  const n=norm(q.slice(1));return n<1e-10?scale(q.slice(1),2):scale(q.slice(1),2*Math.atan2(n,q[0])/n);
}
export function angularVelocity(a,b,dt) { return scale(rotationError(quatToMat(b),quatToMat(a)),1/dt); }
export function solveLinear(matrix,rhs) {
  const n=rhs.length,a=matrix.map((r,i)=>[...r,rhs[i]]);
  for(let k=0;k<n;k++){
    let p=k;for(let i=k+1;i<n;i++)if(Math.abs(a[i][k])>Math.abs(a[p][k]))p=i;
    if(Math.abs(a[p][k])<1e-14)throw new Error('Singular IK normal equation');
    [a[p],a[k]]=[a[k],a[p]];
    for(let i=k+1;i<n;i++){const f=a[i][k]/a[k][k];for(let j=k+1;j<=n;j++)a[i][j]-=f*a[k][j];a[i][k]=0;}
  }
  const x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let s=a[i][n];for(let j=i+1;j<n;j++)s-=a[i][j]*x[j];x[i]=s/a[i][i];}return x;
}
// Optional read-only diagnostics for cross-runtime numerical regression tests.
// No trace allocations occur during ordinary controller execution.
let dlsTraceSink=null;
export function setDlsTraceSink(sink=null){
  if(sink!==null&&typeof sink!=='function')throw new TypeError('The DLS trace sink must be a function or null.');
  const previous=dlsTraceSink;dlsTraceSink=sink;return previous;
}
/** Active-set bound-constrained DLS, mirroring the native seven-arm solver. */
export function boundedDLS(rows,error,lower,upper,damping=0.00012) {
  const n=lower.length,h=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?damping:0)),r=Array(n).fill(0);
  for(let k=0;k<rows.length;k++)for(let i=0;i<n;i++){r[i]+=rows[k][i]*error[k];for(let j=0;j<n;j++)h[i][j]+=rows[k][i]*rows[k][j];}
  const trace=dlsTraceSink;trace?.({type:'start',rows,error,lower,upper,damping,hessian:h,rhs:r});
  let x=lower.map((l,i)=>clamp(0,l,upper[i]));const active=Array(n).fill(0);
  for(let iteration=0;iteration<80;iteration++){
    const free=[],fixed=[];active.forEach((s,i)=>(s?fixed:free).push(i));const trial=x.slice();
    if(free.length){const answer=solveLinear(free.map(i=>free.map(j=>h[i][j])),free.map(i=>r[i]-fixed.reduce((s,j)=>s+h[i][j]*x[j],0)));free.forEach((i,k)=>trial[i]=answer[k]);}
    let alpha=1,blocking=-1,side=0;
    for(const i of free){const dir=trial[i]-x[i];let t=1,s=0;if(trial[i]<lower[i]-1e-10){t=(lower[i]-x[i])/dir;s=-1;}else if(trial[i]>upper[i]+1e-10){t=(upper[i]-x[i])/dir;s=1;}if(t<alpha){alpha=Math.max(0,t);blocking=i;side=s;}}
    if(blocking>=0){trace?.({type:'block',iteration,active,x,trial,alpha,blocking,side});x=lerp(x,trial,alpha);x[blocking]=side<0?lower[blocking]:upper[blocking];active[blocking]=side;continue;}
    x=trial;let worst=-1,violation=1e-10;
    for(const i of fixed){const g=dot(h[i],x)-r[i];if(((active[i]<0&&g<0)||(active[i]>0&&g>0))&&Math.abs(g)>violation){worst=i;violation=Math.abs(g);}}
    trace?.({type:worst<0?'optimal':'release',iteration,active,x,worst,violation});
    if(worst<0)break;active[worst]=0;
  }
  const result=x.map((v,i)=>clamp(v,lower[i],upper[i]));trace?.({type:'finish',result});return result;
}
export function convexHull(points) {
  const p=[...new Map(points.map(v=>[`${v[0]},${v[1]}`,[v[0],v[1]]])).values()].sort((a,b)=>a[0]-b[0]||a[1]-b[1]);if(p.length<3)return p;
  const turn=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const half=arr=>{const h=[];for(const v of arr){while(h.length>=2&&turn(h.at(-2),h.at(-1),v)<=0)h.pop();h.push(v);}h.pop();return h;};return [...half(p),...half([...p].reverse())];
}
export function polygonMargin(point,polygon) {let d=Infinity;for(let i=0;i<polygon.length;i++){const a=polygon[i],b=polygon[(i+1)%polygon.length],dx=b[0]-a[0],dy=b[1]-a[1];d=Math.min(d,(dx*(point[1]-a[1])-dy*(point[0]-a[0]))/Math.max(Math.hypot(dx,dy),1e-12));}return d;}
export function footprintOverlap(points,bounds) {
  let p=convexHull(points);for(const [axis,edge,sign] of [[0,bounds.x_min,1],[0,bounds.x_max,-1],[1,bounds.y_min,1],[1,bounds.y_max,-1]]){
    const out=[];if(!p.length)return 0;let previous=p.at(-1);
    for(const current of p){const a=sign*(previous[axis]-edge),b=sign*(current[axis]-edge);if((a>=0)!==(b>=0))out.push(lerp(previous,current,a/(a-b)));if(b>=0)out.push(current);previous=current;}p=out;
  }return p.length<3?0:Math.abs(p.reduce((s,v,i)=>s+v[0]*p[(i+1)%p.length][1]-v[1]*p[(i+1)%p.length][0],0))/2;
}
