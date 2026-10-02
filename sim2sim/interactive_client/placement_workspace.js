/** Convex editor workspace geometry. Physical support and collisions are checked separately. */
export const DEFAULT_PLACEMENT_REGION=Object.freeze({x_min:.30,x_max:.57,y_min:-.40,y_max:.40,
  corner_cut:Object.freeze({x_start:.45,y_start:.24})});
const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));

function planes(region){
  if(!region||!['x_min','x_max','y_min','y_max'].every(key=>Number.isFinite(region[key]))||
      region.x_min>=region.x_max||region.y_min>=region.y_max)throw new TypeError('Invalid placement region bounds.');
  const cut=region.corner_cut;
  if(cut==null)return [];
  const lateral=Math.max(Math.abs(region.y_min),Math.abs(region.y_max));
  if(!Number.isFinite(cut.x_start)||!Number.isFinite(cut.y_start)||cut.x_start<region.x_min||
      cut.x_start>region.x_max||cut.y_start<0||cut.y_start>lateral)throw new TypeError('Invalid placement corner cut.');
  const dx=region.x_max-cut.x_start,dy=lateral-cut.y_start,c=dy*cut.x_start+dx*lateral;
  // A symmetric |Y| cap also describes either hand's half of the full region.
  return [[dy,dx,c],[dy,-dx,c]];
}

/** Counterclockwise vertices, without a repeated closing point. No corner_cut means a rectangle. */
export function placementPolygon(region){
  const clipping=planes(region);
  let polygon=[[region.x_min,region.y_min],[region.x_max,region.y_min],
    [region.x_max,region.y_max],[region.x_min,region.y_max]];
  for(const [a,b,c] of clipping){
    const next=[];
    for(let i=0;i<polygon.length;i++){
      const p=polygon[i],q=polygon[(i+1)%polygon.length],dp=a*p[0]+b*p[1]-c,dq=a*q[0]+b*q[1]-c;
      if(dp<=0)next.push(p);
      if((dp<=0)!==(dq<=0)){
        const t=dp/(dp-dq);next.push([p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])]);
      }
    }
    polygon=next.filter((p,i)=>!i||Math.hypot(p[0]-next[i-1][0],p[1]-next[i-1][1])>1e-12);
    if(polygon.length>1&&Math.hypot(polygon[0][0]-polygon.at(-1)[0],polygon[0][1]-polygon.at(-1)[1])<1e-12)polygon.pop();
  }
  return polygon;
}

/** Boundary-inclusive point containment; epsilon is a distance in world meters. */
export function containsPlacement(region,x,y,epsilon=1e-9){
  if(!Number.isFinite(epsilon)||epsilon<0)throw new TypeError('Placement epsilon must be nonnegative and finite.');
  const polygon=placementPolygon(region);
  if(!Number.isFinite(x)||!Number.isFinite(y))return false;
  return polygon.every((p,i)=>{
    const q=polygon[(i+1)%polygon.length],dx=q[0]-p[0],dy=q[1]-p[1];
    return dx*(y-p[1])-dy*(x-p[0])>=-epsilon*Math.hypot(dx,dy);
  });
}

/** Intersect one editor axis with the polygon. Null means the fixed other coordinate has no valid slice. */
export function placementAxisRange(region,axis,otherCoordinate){
  if(!['x','y'].includes(axis)||!Number.isFinite(otherCoordinate))throw new TypeError('Use a finite coordinate and placement axis x or y.');
  const polygon=placementPolygon(region),index=axis==='x'?0:1,other=1-index,values=[];
  for(let i=0;i<polygon.length;i++){
    const p=polygon[i],q=polygon[(i+1)%polygon.length],span=q[other]-p[other];
    if(Math.abs(span)<1e-12){if(Math.abs(otherCoordinate-p[other])<1e-12)values.push(p[index],q[index]);continue;}
    const t=(otherCoordinate-p[other])/span;
    if(t>=-1e-12&&t<=1+1e-12)values.push(p[index]+clamp(t,0,1)*(q[index]-p[index]));
  }
  return values.length?[Math.min(...values),Math.max(...values)]:null;
}

/** Free dragging projects to the nearest polygon point; axis edits preserve the other coordinate. */
export function clampPlacement(region,x,y,{axis}={}){
  if(!Number.isFinite(x)||!Number.isFinite(y))throw new TypeError('Placement coordinates must be finite.');
  if(axis!==undefined){
    const range=placementAxisRange(region,axis,axis==='x'?y:x);
    if(!range)throw new RangeError('The fixed placement coordinate is outside the workspace.');
    return axis==='x'?[clamp(x,...range),y]:[x,clamp(y,...range)];
  }
  if(containsPlacement(region,x,y,0))return [x,y];
  const polygon=placementPolygon(region);let nearest=null,best=Infinity;
  for(let i=0;i<polygon.length;i++){
    const p=polygon[i],q=polygon[(i+1)%polygon.length],dx=q[0]-p[0],dy=q[1]-p[1];
    const t=clamp(((x-p[0])*dx+(y-p[1])*dy)/(dx*dx+dy*dy),0,1),point=[p[0]+t*dx,p[1]+t*dy];
    const distance=(x-point[0])**2+(y-point[1])**2;
    if(distance<best){best=distance;nearest=point;}
  }
  return nearest;
}
