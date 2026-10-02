// Node verification of the SAME ORT Web WASM backend used by the browser worker.
// Run: node policy_parity.mjs ../../build/demo/policies
import fs from 'node:fs/promises';
import path from 'node:path';
import * as ort from 'onnxruntime-web/wasm';
import {loadResolvedPolicy} from './policy_runtime.mjs';

ort.env.wasm.numThreads=1;
ort.env.wasm.proxy=false;
const base=path.resolve(process.argv[2]??'../../build/demo/policies');
const read=async p=>fs.readFile(path.join(base,p));
const manifest=JSON.parse(await read('manifest.json'));
const fixture=JSON.parse(await read('parity_fixture.json'));
const reference={...fixture.reference,sample(i){return this.frames[Math.max(0,Math.min(this.frameCount-1,i))];}};
const maxima={};
function compare(key,actual,expected,tolerance,step) {
  if(actual.length!==expected.length)throw new Error(`${key} length mismatch`);
  let max=0,idx=0;
  for(let i=0;i<actual.length;i++){const error=Math.abs(actual[i]-expected[i]);if(error>max){max=error;idx=i;}}
  if(!maxima[key]||max>maxima[key].maxAbs)maxima[key]={maxAbs:max,index:idx,step,tolerance};
  if(!Number.isFinite(max)||max>tolerance)throw new Error(`${key} step ${step}: max ${max} > ${tolerance}, index ${idx}; actual ${actual[idx]}, expected ${expected[idx]}`);
}
const timings={};
for(const kind of Object.keys(manifest.policies)) {
  const start=performance.now();
  // This recorded fixture explicitly uses native odom_source="truth".
  const policy=await loadResolvedPolicy(kind,{ort,manifest,assetResolver:read,odomSource:'truth'});
  timings[kind]={loadMilliseconds:performance.now()-start,inferenceMilliseconds:[]};
  for(let step=0;step<fixture.cases.length;step++) {
    const test=fixture.cases[step];if(test.reset)policy.reset();
    const tick=performance.now(),target=await policy.control(reference,test.index,test.state);
    timings[kind].inferenceMilliseconds.push(performance.now()-tick);
    if(kind==='hero_plus') {
      compare('hero.observation',policy.lastObs,test.hero.observation,3e-5,step);
      compare('hero.rawAction',policy.lastAction,test.hero.rawAction,2e-4,step);
      compare('hero.target',target,test.hero.target,1e-4,step);

    }
  }
  // Check saturation and the distinct residual contracts without model inference.
  const raw=Array.from({length:29},(_,i)=>i%2?150:-150),ref=reference.sample(1).jointPos;
  const saturated=kind==='hero_plus'?policy.qTarget(raw,ref):policy.qTarget(raw);
  const expected=Array.from({length:29},(_,j)=>{
    const index=kind==='hero_plus'?j:policy.metadata.isaaclabToMujoco[j];
    const value=Math.max(-policy.metadata.actionClip,Math.min(policy.metadata.actionClip,raw[index]));
    return (kind==='hero_plus'&&policy.metadata.residualIndices.includes(j)?ref[j]:policy.defaultDofPos[j])+policy.actionScale[j]*value;
  });
  compare(kind+'.saturatedActionContract',saturated,expected,1e-12,'saturation');
  await policy.dispose();
}
for(const result of Object.values(timings)) {
  const sorted=result.inferenceMilliseconds.slice(1).sort((a,b)=>a-b);
  result.warmMedianMilliseconds=sorted[Math.floor(sorted.length/2)];
  result.warmMaxMilliseconds=Math.max(...sorted);
}
const report={schema:'browser_policy_parity_result_v1',passed:true,backend:'onnxruntime-web/wasm',
  ortVersion:manifest.onnxRuntimeWeb,threads:1,cases:fixture.cases.length,resets:fixture.cases.filter(x=>x.reset).length,
  physicsSteps:0,maxima,timings,modelProvenance:Object.fromEntries(Object.entries(manifest.policies).map(([k,v])=>[k,v.provenance]))};
await fs.writeFile(path.join(base,'parity_result.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
