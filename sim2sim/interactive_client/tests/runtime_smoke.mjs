// Exercise the shipped ONNX, reference planner, leg odometry, and real MuJoCo WASM physics together.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import * as ort from 'onnxruntime-web/wasm';
import {createSimulation} from '../simulation.js';
import {loadResolvedPolicy} from '../policy_runtime.mjs';
import {InteractiveController} from '../controller.js';

ort.env.wasm.numThreads=1;
ort.env.wasm.proxy=false;
const publicRoot=new URL('../public/',import.meta.url);
const read=path=>readFile(new URL(path,publicRoot));
const manifest=JSON.parse(await read('policies/manifest.json'));
const policy=await loadResolvedPolicy('hero_plus',{ort,manifest,assetResolver:path=>read(`policies/${path}`)});
const scene=await createSimulation({mode:'hero_plus',tableKind:'workbench',tableHeight:.74,
  placements:{bottle:[.41,-.32,0]},readAsset:path=>read(`sceneassets/${path}`)});
scene.setPolicyParameters(policy);
const controller=new InteractiveController(scene,policy,{mode:'hero_plus',grasp_style:'side',ee_yaw_deg:45});
try {
  const initial=Array.from(scene.data.qpos);
  await controller.prepare({objectId:'bottle',hand:'auto',approach:'side',yawDeg:45});
  assert.deepEqual(Array.from(scene.data.qpos),initial,'Planning must preserve the live state.');
  for(let i=0;i<25&&controller.busy;i++)await controller.tick();
  assert.ok(policy.steps>=5,'The actual ONNX must control several physics steps.');
  assert.ok(scene.time>0);
  assert.equal(scene.readState().odom.source,'leg');
  assert.equal(policy.lastObs.length,policy.metadata.observationDim);
  assert.ok(Array.from(scene.data.qpos).every(Number.isFinite));
  assert.ok(Array.from(policy.lastAction).every(Number.isFinite));
  console.log(JSON.stringify({passed:true,policySteps:policy.steps,simulationSeconds:scene.time,
    phase:controller.phase,observationDim:policy.lastObs.length,odometry:'leg'}));
} finally {
  controller.dispose();scene.dispose();await policy.dispose();
}
