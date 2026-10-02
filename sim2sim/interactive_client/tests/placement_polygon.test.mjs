import assert from 'node:assert/strict';
import test from 'node:test';
import {DEFAULT_PLACEMENT_REGION as region,placementPolygon,containsPlacement,clampPlacement,placementAxisRange} from '../placement_workspace.js';
const close=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-12,`${actual} != ${expected}`);
const point=(actual,expected)=>{assert.equal(actual.length,2);actual.forEach((v,i)=>close(v,expected[i]));};

test('clipped corners form a symmetric convex hexagon with the expected area',()=>{
  const polygon=placementPolygon(region);
  assert.equal(polygon.length,6);
  [[.30,-.40],[.45,-.40],[.57,-.24],[.57,.24],[.45,.40],[.30,.40]].forEach((p,i)=>point(polygon[i],p));
  const area=polygon.reduce((sum,p,i)=>{const q=polygon[(i+1)%polygon.length];return sum+p[0]*q[1]-p[1]*q[0];},0)/2;
  close(area,.1968);// 0.27 x 0.80 rectangle minus two 0.12 x 0.16 corner triangles.
  for(const p of polygon){assert.ok(containsPlacement(region,...p));assert.ok(containsPlacement(region,p[0],-p[1]));}
  for(const p of [[.55,.38],[.55,-.38],[.57,.40],[.57,-.40],[.510001,.32],[.29,0],[.570001,0]])assert.equal(containsPlacement(region,...p),false);
  for(const p of [[.57,.23],[.57,-.23],[.45,.40],[.45,-.40],[.51,.32],[.51,-.32],[.30,0]])assert.ok(containsPlacement(region,...p));
});

test('free dragging finds the nearest diagonal point instead of clamping each rectangle axis',()=>{
  point(clampPlacement(region,.57,.40),[.4932,.3424]);
  point(clampPlacement(region,.57,-.40),[.4932,-.3424]);
  point(clampPlacement(region,.7,0),[.57,0]);
  point(clampPlacement(region,.2,.5),[.30,.40]);
  // This convex projection inequality certifies the nearest point independently
  // of which segment the implementation selects, including vertices and ties.
  const vertices=placementPolygon(region);
  for(let x=.20;x<.71;x+=.037)for(let y=-.51;y<.52;y+=.043){
    const q=clampPlacement(region,x,y);assert.ok(containsPlacement(region,...q));
    point(clampPlacement(region,...q),q);
    for(const v of vertices)assert.ok((x-q[0])*(v[0]-q[0])+(y-q[1])*(v[1]-q[1])<1e-12);
  }
});

test('axis ranges follow the diagonal and sliders preserve the other coordinate',()=>{
  point(placementAxisRange(region,'x',.32),[.30,.51]);
  point(placementAxisRange(region,'x',-.40),[.30,.45]);
  point(placementAxisRange(region,'y',.57),[-.24,.24]);
  point(placementAxisRange(region,'y',.53),[-.2933333333333333,.2933333333333333]);
  assert.equal(placementAxisRange(region,'y',.58),null);
  point(clampPlacement(region,.57,.32,{axis:'x'}),[.51,.32]);
  point(clampPlacement(region,.53,.4,{axis:'y'}),[.53,.2933333333333333]);
  assert.equal(placementAxisRange(region,'x',.5),null);
  assert.throws(()=>clampPlacement(region,.50,.5,{axis:'x'}),RangeError);
  for(const axis of ['x','y'])for(const p of [[.7,.32],[.53,-.6]]){
    const other=axis==='x'?p[1]:p[0];if(placementAxisRange(region,axis,other)===null)continue;
    const q=clampPlacement(region,...p,{axis});assert.equal(q[axis==='x'?1:0],other);assert.ok(containsPlacement(region,...q));
  }
});

test('either hand half keeps its same diagonal and full-table rectangles remove the cut completely',()=>{
  for(const sign of [-1,1]){
    const half={...region,...(sign<0?{y_max:0}:{y_min:0})};
    assert.ok(containsPlacement(half,.57,sign*.23));assert.equal(containsPlacement(half,.55,sign*.38),false);
    assert.equal(containsPlacement(half,.4,-sign*.1),false);
    point(placementAxisRange(half,'x',sign*.32),[.30,.51]);
  }
  const rectangle={x_min:.28,x_max:.94,y_min:-.5,y_max:.5};
  assert.equal(placementPolygon(rectangle).length,4);assert.ok(containsPlacement(rectangle,.55,-.38));
  point(clampPlacement(rectangle,1,-.6),[.94,-.5]);
  point(placementAxisRange(rectangle,'y',.55),[-.5,.5]);
});

test('geometry helpers do not mutate inputs and reject invalid coordinates or region schema',()=>{
  const mutable=structuredClone(region),before=structuredClone(mutable);
  placementPolygon(mutable);containsPlacement(mutable,.5,.2);clampPlacement(mutable,1,1);placementAxisRange(mutable,'x',0);
  assert.deepEqual(mutable,before);
  assert.equal(containsPlacement(region,NaN,0),false);assert.throws(()=>clampPlacement(region,Infinity,0),TypeError);
  assert.throws(()=>placementAxisRange(region,'z',0),TypeError);
  assert.throws(()=>placementPolygon({...region,corner_cut:{x_start:.1,y_start:.24}}),TypeError);
  assert.throws(()=>placementPolygon({...region,x_min:1}),TypeError);
  assert.throws(()=>containsPlacement(region,.5,0,-1),TypeError);
});
