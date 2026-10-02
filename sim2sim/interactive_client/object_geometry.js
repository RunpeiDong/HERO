import {add,scale,matVec,footprintOverlap} from './numerics.js';
import {geomBounds,geomCorners} from './ik.js';

export const BOTTLE_DIAMETER_M=.062;
export const BOTTLE_SIDE_GRASP_LOWER_M=.003;

// Shared instantaneous footprint diagnostic using per-shape sampling,
// physical collision masks, and an above-floor test.
// This overlap alone is not proof of a landing; native contact supplies that.
export function objectTrayOverlap(scene,objectId,data=scene.data){
  const m=scene.model,d=data,b=scene.tray.innerBounds,floor=scene.tray.bottomTopZ;
  const copy=value=>Array.from(value);let total=0;
  for(const g of scene.objects[objectId].geomIds){
    if(!(m.geom_contype[g]||m.geom_conaffinity[g]))continue;
    if(geomBounds(scene,d,[g]).upper[2]<=floor)continue;
    const center=copy(d.geom_xpos.slice(3*g,3*g+3)),r=copy(d.geom_xmat.slice(9*g,9*g+9)),
      size=copy(m.geom_size.slice(3*g,3*g+3)),type=m.geom_type[g];let points=[];
    if(type===2||type===3||type===5){
      for(let i=0;i<32;i++){
        const a=2*Math.PI*i/32,circle=[size[0]*Math.cos(a),size[0]*Math.sin(a),0];
        if(type===2)points.push(add(center,circle));
        else for(const s of [-1,1])points.push(type===3
          ?add(add(center,scale([r[2],r[5],r[8]],s*size[1])),circle)
          :add(center,matVec(r,[circle[0],circle[1],s*size[1]])));
      }
    }else points=geomCorners(scene,d,g);
    total+=footprintOverlap(points,b);
  }
  return total;
}

// Apply a browser-specific radial resize before native compilation, so shape
// bounds, broad-phase collision data and render geometry all agree. The native
// source assets remain reproducible; this transform is part of the build hash.
export function prepareBrowserObjectGeometry(metadata,xml) {
  const object=metadata.objects.bottle,info=metadata.catalog.objects.find(o=>o.id==='bottle');
  const scale=BOTTLE_DIAMETER_M/info.size[0],overrides=metadata.modelOverrides;
  const parts=object.geomIds.map((g,index)=>{
    const name=metadata.geomNames[g],match=xml.match(new RegExp(`<geom\\b[^>]*\\bname="${name}"[^>]*>`));
    if(!match)throw new Error(`Missing bottle geometry ${name}.`);
    const tag=match[0],type=tag.match(/\btype="([^"]+)"/)?.[1]??'sphere',old=overrides.geom_size.slice(g*3,g*3+3);
    if(!['sphere','cylinder'].includes(type))throw new Error('Unsupported source bottle shape.');
    const size=type==='sphere'?[old[0]*scale,old[0]*scale,old[0]]:[old[0]*scale,old[1],0];
    // A broad neck leaves a usable pinch surface near the shoulder as well as
    // around the main body without falling between the Dex3 fingers.
    if(index===2)size[0]=.022;
    if(index===3)size[0]=.024;
    const volume=type==='sphere'?4/3*Math.PI*size[0]*size[1]*size[2]:Math.PI*size[0]**2*2*size[1];
    return {g,tag,type,size,volume};
  });
  const volume=parts.reduce((sum,p)=>sum+p.volume,0);
  const attribute=(tag,name,value)=>new RegExp(`\\b${name}="[^"]*"`).test(tag)
    ?tag.replace(new RegExp(`\\b${name}="[^"]*"`),`${name}="${value}"`):tag.replace(/\s*\/?>(\s*)$/,` ${name}="${value}" />$1`);
  for(const part of parts){
    let tag=attribute(part.tag,'size',(part.type==='sphere'?part.size:part.size.slice(0,2)).join(' '));
    if(part.type==='sphere')tag=attribute(tag,'type','ellipsoid');
    tag=attribute(tag,'mass',String(object.mass*part.volume/volume));xml=xml.replace(part.tag,tag);
    overrides.geom_size.splice(part.g*3,3,...part.size);
  }
  object.size=[BOTTLE_DIAMETER_M,BOTTLE_DIAMETER_M,object.height];info.size=object.size.slice();info.footprint_radius=BOTTLE_DIAMETER_M/2;
  metadata.browserObjectOverrides={bottle:{diameterM:BOTTLE_DIAMETER_M,neckDiameterM:.044,capDiameterM:.048,radialScale:scale,heightM:object.height,massKg:object.mass}};
  return xml;
}

export function preserveCompiledBottleInertia(model,metadata){
  const body=metadata.objects.bottle.bodyId;
  for(const [field,width]of [['body_ipos',3],['body_iquat',4],['body_mass',1],['body_inertia',3]])
    metadata.modelOverrides[field].splice(body*width,width,...model[field].slice(body*width,(body+1)*width));
}
