import {access, copyFile, readFile, readdir, unlink, writeFile} from 'node:fs/promises';
import {resolve, relative, isAbsolute, sep} from 'node:path';
import {createHash} from 'node:crypto';

const TABLES = ['workbench', 'round', 'pedestal'], HEIGHTS = [50, 74, 88];
const EXPECTED_SCENES = TABLES.flatMap(table => HEIGHTS.map(height => `hero_plus_${table}_${height}`));
const DEMO_SHA256 = '521402c04092df1f441a5a5e4f09a1ed17480ffa27e7220f5e1ee664ccbb2c33';

function assetPath(directory, path) {
  if (typeof path !== 'string' || !path || isAbsolute(path)) throw new Error('Build asset paths must be relative.');
  const target = resolve(directory, path), local = relative(directory, target);
  if (!local || local === '..' || local.startsWith(`..${sep}`)) throw new Error(`Build asset escapes its directory: ${path}`);
  return target;
}

async function filesWithin(directory, prefix = '') {
  const files = [];
  for (const entry of await readdir(resolve(directory, prefix), {withFileTypes: true})) {
    const path = `${prefix}${entry.name}`;
    if (entry.isDirectory()) files.push(...await filesWithin(directory, `${path}/`));
    else if (entry.isFile()) files.push(path);
  }
  return files;
}

/** Keep HERO assets in the generated output without changing public/. */
export async function retainHeroBuildAssets(outputDirectory) {
  const root = resolve(outputDirectory), policies = resolve(root, 'policies'), scenes = resolve(root, 'sceneassets');
  const policyManifest = JSON.parse(await readFile(resolve(policies, 'manifest.json'), 'utf8'));
  const sceneManifest = JSON.parse(await readFile(resolve(scenes, 'manifest.json'), 'utf8'));
  const hero = policyManifest.policies?.hero_plus;
  if (!hero?.metadata || !hero.files?.model?.endsWith('.onnx'))
    throw new Error('The demo requires a HERO ONNX export and its observation/action metadata.');
  const metadata = JSON.parse(await readFile(assetPath(policies, hero.metadata), 'utf8'));
  const {terms, termDimensions, historyLength, observationDim} = metadata;
  if (metadata.kind !== 'hero_plus' || !Array.isArray(terms) || !terms.length ||
      new Set(terms).size !== terms.length || !Array.isArray(termDimensions) || terms.length !== termDimensions.length ||
      !termDimensions.every(size => Number.isInteger(size) && size > 0) || !Number.isInteger(historyLength) || historyLength < 1 ||
      !['frame_major_hero_v1','term_major_holosoma_v1'].includes(metadata.historyLayout) ||
      (observationDim != null && observationDim !== termDimensions.reduce((a,b)=>a+b,0)*historyLength) ||
      !Array.isArray(metadata.inputNames) || !metadata.inputNames.length || typeof metadata.outputName !== 'string' ||
      !Array.isArray(metadata.dofNames) || metadata.dofNames.length !== 29)
    throw new Error('Invalid HERO observation/action contract. Export metadata from the matching checkpoint.');
  const modelBytes = await readFile(assetPath(policies, hero.files.model));
  const modelHash = createHash('sha256').update(modelBytes).digest('hex');
  if (hero.provenance?.files?.model?.sha256 !== modelHash)
    throw new Error('HERO model hash differs from its export manifest. Re-export the policy assets.');
  if (modelHash !== DEMO_SHA256 && policyManifest.allowCustomPolicy !== true)
    throw new Error('The default demo requires the verified example ONNX model. Export with --allow-custom-policy for another checkpoint.');
  if (sceneManifest.policySha256?.hero_plus && sceneManifest.policySha256.hero_plus !== modelHash)
    throw new Error('Scene and policy exports use different checkpoints. Re-export scenes with the same --hero path.');
  const heroScenes = Object.fromEntries(EXPECTED_SCENES.map(key => {
    const entry = sceneManifest.scenes?.[key];
    if (!entry || entry.mode !== 'hero_plus') throw new Error(`The demo is missing scene ${key}.`);
    return [key, entry];
  }));
  const policyFiles = new Set(['manifest.json', hero.metadata, ...Object.values(hero.files)]);
  const sceneFiles = new Set(['manifest.json', ...Object.keys(sceneManifest.assets ?? {}),
    ...Object.values(heroScenes).flatMap(entry => [entry.xml, entry.metadata])]);
  // Validate the complete retained runtime set before changing any output files.
  await Promise.all([...policyFiles].map(path => access(assetPath(policies, path))));
  await Promise.all([...sceneFiles].map(path => access(assetPath(scenes, path))));
  for (const entry of Object.values(heroScenes)) sceneFiles.add(entry.xml.replace(/\.xml$/, '.native_trace.json'));
  // Optional hand-model subtrees (sceneassets/<hand>/ with their own manifest) are retained with
  // the same HERO-only filtering; the worker offers a hand switch only when such a tree exists.
  const handTrees = [], handManifests = [];
  for (const entry of await readdir(scenes, {withFileTypes: true})) {
    if (!entry.isDirectory()) continue;
    const handDirectory = assetPath(scenes, entry.name);
    let manifestText;
    try { manifestText = await readFile(assetPath(handDirectory, 'manifest.json'), 'utf8'); }
    catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    const handManifest = JSON.parse(manifestText);
    if (handManifest.policySha256?.hero_plus && handManifest.policySha256.hero_plus !== modelHash)
      throw new Error(`Hand model ${entry.name} and policy exports use different checkpoints. Re-export scenes with the same --hero path.`);
    const handScenes = Object.fromEntries(EXPECTED_SCENES.map(key => {
      const scene = handManifest.scenes?.[key];
      if (!scene || scene.mode !== 'hero_plus') throw new Error(`Hand model ${entry.name} is missing scene ${key}.`);
      return [key, scene];
    }));
    const prefix = `${entry.name}/`;
    const handFiles = new Set(['manifest.json', ...Object.keys(handManifest.assets ?? {}),
      ...Object.values(handScenes).flatMap(scene => [scene.xml, scene.metadata])]);
    await Promise.all([...handFiles].map(path => access(assetPath(handDirectory, path))));
    for (const scene of Object.values(handScenes)) handFiles.add(scene.xml.replace(/\.xml$/, '.native_trace.json'));
    for (const path of handFiles) sceneFiles.add(prefix + path);
    handManifests.push([`${prefix}manifest.json`, {...handManifest, scenes: handScenes}]);
    handTrees.push(entry.name);
  }
  if (hero.files.model !== 'hero.onnx') {
    await copyFile(assetPath(policies, hero.files.model), assetPath(policies, 'hero.onnx'));
    policyFiles.delete(hero.files.model); policyFiles.add('hero.onnx');
  }
  const removed = [];
  for (const [directory, retained, prefix] of [[policies, policyFiles, 'policies'], [scenes, sceneFiles, 'sceneassets']]) {
    for (const path of await filesWithin(directory)) if (!retained.has(path)) {
      await unlink(assetPath(directory, path)); removed.push(`${prefix}/${path}`);
    }
  }
  await writeFile(resolve(policies, 'manifest.json'), JSON.stringify({...policyManifest,
    policies: {hero_plus: {...hero, label: 'HERO', files: {...hero.files, model: 'hero.onnx'}}}}, null, 2) + '\n');
  await writeFile(resolve(scenes, 'manifest.json'), JSON.stringify({...sceneManifest, scenes: heroScenes}, null, 2) + '\n');
  for (const [path, manifest] of handManifests)
    await writeFile(assetPath(scenes, path), JSON.stringify(manifest, null, 2) + '\n');
  return {policies: ['hero_plus'], scenes: Object.keys(heroScenes), hands: ['dex3', ...handTrees], removed};
}

export function heroBuildAssetsPlugin() {
  let outputDirectory;
  return {
    name: 'tabletop-hero-public-assets', apply: 'build',
    configResolved(config) {
      outputDirectory = resolve(config.root, config.build.outDir);
      const publicDirectory = config.publicDir && resolve(config.root, config.publicDir);
      if (publicDirectory && (outputDirectory === publicDirectory ||
          outputDirectory.startsWith(`${publicDirectory}${sep}`) || publicDirectory.startsWith(`${outputDirectory}${sep}`)))
        throw new Error('The generated demo output must be separate from the public/ assets directory.');
    },
    async closeBundle() {await retainHeroBuildAssets(outputDirectory);},
  };
}
