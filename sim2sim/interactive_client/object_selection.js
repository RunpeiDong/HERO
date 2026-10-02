/** Small, deterministic English command grammar for the tabletop objects.
 * This selects an object; it does not execute arbitrary language or robot actions.
 * Colors follow interactive_scene.py and the exported geometry, including both
 * the orange faces and navy/blue body of the UIUC Block I.
 */
const CATALOG = {
  apple: {label: 'Apple', colors: ['red']},
  can: {label: 'Can', colors: ['blue']},
  bottle: {label: 'Bottle', colors: ['green']},
  mug: {label: 'Mug', colors: ['tan', 'gold', 'brown']},
  uiuc_i: {label: 'UIUC Block I', colors: ['orange', 'navy', 'blue']},
  cracker_box: {label: 'Cheez-It box', colors: ['red']},
};
const NAMES = [
  ['uiuc block i', 'uiuc_i'], ['uiuc i', 'uiuc_i'], ['block i', 'uiuc_i'],
  ['letter i', 'uiuc_i'], ['i shaped block', 'uiuc_i'], ['i shaped', 'uiuc_i'], ['i block', 'uiuc_i'],
  ['water bottle', 'bottle'], ['soda can', 'can'], ['tin can', 'can'],
  ['coffee mug', 'mug'], ['coffee cup', 'mug'],
  ['apple', 'apple'], ['fruit', 'apple'], ['can', 'can'],
  ['bottle', 'bottle'], ['mug', 'mug'], ['cup', 'mug'], ['i', 'uiuc_i'],
  ['cheez it box', 'cracker_box'], ['cracker box', 'cracker_box'], ['box of crackers', 'cracker_box'], ['cereal box', 'cracker_box'],
  ['cheez it', 'cracker_box'], ['crackers', 'cracker_box'], ['carton', 'cracker_box'], ['box', 'cracker_box'],
].map(([text, id]) => ({words: text.split(' '), id})).sort((a, b) => b.words.length - a.words.length);
const COLORS = new Set(Object.values(CATALOG).flatMap(object => object.colors));
const RANKS = {leftmost: 'left', left: 'left', rightmost: 'right', right: 'right',
  nearest: 'nearest', closest: 'nearest', farthest: 'farthest', furthest: 'farthest'};
const GENERIC = new Set(['object', 'item', 'thing', 'one', 'it']);
const FILLERS = new Set(['the', 'a', 'an', 'that', 'this', 'please', 'kindly', 'on', 'at', 'to', 'from',
  'of', 'for', 'me', 'you', 'your', 'side', 'robot', "robot's", 'colored', 'coloured', 'color', 'colour']);
const TIE_EPSILON_M = 1e-6;
const example = 'Try “pick the bottle” or “pick the leftmost object”.';

function selectorWords(text) {
  if (typeof text !== 'string' || !text.trim()) throw new Error(`Enter a command. ${example}`);
  if (text.length > 512) throw new Error(`Use a short command for one object. ${example}`);
  let value = text.normalize('NFKC').toLowerCase().replace(/[’‘]/g, "'").trim();
  if (/\b(?:not|no|never|don't|dont|cannot|can't|avoid|except|without|unless|instead)\b/.test(value))
    throw new Error('Negation and exceptions are not supported. Ask to pick one object directly.');
  if (/[;\n\r]/.test(value) || /\b(?:both|all|each|every|objects|items|things|them|apples|fruits|bottles|mugs|cans|or|then)\b/.test(value))
    throw new Error('Choose one object in one command; multiple targets and alternatives are not supported.');
  value = value.replace(/[.!?]+$/, '').replace(/,/g, ' ').replace(/[_-]/g, ' ').replace(/\s+/g, ' ').trim();
  if (/[^a-z0-9' ]/.test(value)) throw new Error(`Use a plain English object command. ${example}`);
  value = value.replace(/^(?:please|kindly) /, '')
    .replace(/^(?:(?:can|could|would|will) you|i (?:want|would like)(?: you)? to) /, '')
    .replace(/^(?:please|kindly) /, '');
  const action = /^(pick(?: up)?|grab|grasp)(?:\s+|$)(.*)$/.exec(value);
  // A noun description is sufficient for the Select target UI. Unknown verbs
  // remain unrecognized selector tokens, so dropping the action prefix never
  // makes commands such as 'throw the apple' eligible.
  let selector = (action ? action[2] : value).replace(/(?: please| thanks| thank you| for me)+$/, '').trim();
  if (action && action[1].startsWith('pick')) selector = selector.replace(/ up$/, '');
  if (!selector) throw new Error(`Name the object to pick. ${example}`);
  return selector.replace(/\bleft most\b/g, 'leftmost').replace(/\bright most\b/g, 'rightmost')
    .replace(/\bmost distant\b/g, 'farthest').split(' ');
}

function parseSelector(words) {
  const tokens = [];
  for (let index = 0; index < words.length;) {
    const name = NAMES.find(alias => alias.words.every((word, offset) => words[index + offset] === word));
    if (name) { tokens.push({type: 'name', value: name.id}); index += name.words.length; continue; }
    const word = words[index++];
    if (COLORS.has(word)) tokens.push({type: 'color', value: word});
    else if (Object.hasOwn(RANKS, word)) tokens.push({type: 'rank', value: RANKS[word]});
    else if (GENERIC.has(word)) tokens.push({type: 'generic', value: word});
    else if (word === 'and') tokens.push({type: 'and'});
    else if (FILLERS.has(word)) tokens.push({type: 'filler'});
    else throw new Error(`I don't recognize “${word}” in an object description. ${example}`);
  }
  // Two aliases of the same object in one command ('the box of crackers') still name one object.
  const names = [...new Set(tokens.filter(token => token.type === 'name').map(token => token.value))];
  const generics = tokens.filter(token => token.type === 'generic');
  if (names.length > 1 || generics.length > 1) throw new Error('Choose one object per command.');
  for (let i = 0; i < tokens.length; i++) if (tokens[i].type === 'and'
      && !(tokens[i - 1]?.type === 'color' && tokens[i + 1]?.type === 'color'))
    throw new Error('Choose one object per command. “And” can only join its colors.');
  const colors = [...new Set(tokens.filter(token => token.type === 'color').map(token => token.value))];
  const ranks = [...new Set(tokens.filter(token => token.type === 'rank').map(token => token.value))];
  if (ranks.length > 1) throw new Error('Use one position description: leftmost, rightmost, nearest, or farthest.');
  if (!names.length && !colors.length && !ranks.length && !generics.length)
    throw new Error(`Name the object to pick. ${example}`);
  return {name: names[0], colors, rank: ranks[0]};
}

function coordinates(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length
      || !Array.from(value).every(Number.isFinite)) throw new Error(`${label} must contain ${length} finite coordinates.`);
  return Array.from(value);
}
function rankedCandidates(candidates, rank, robot) {
  const origin = coordinates(robot?.position ?? [0, 0, .75], 3, 'Robot position');
  let leftAxis;
  if (rank === 'left' || rank === 'right') {
    let q = coordinates(robot?.quaternion ?? [1, 0, 0, 0], 4, 'Robot quaternion (w, x, y, z)');
    const length = Math.hypot(...q);
    if (!Number.isFinite(length) || length < 1e-12) throw new Error('Robot quaternion must have a finite nonzero length.');
    const [w, x, y, z] = q.map(value => value / length);
    // World-space local +Y axis. Positive Y is the robot's left, independently
    // of the camera. Projecting onto it is the inverse rotation's Y component.
    leftAxis = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)];
  }
  const scored = candidates.map(object => {
    const point = coordinates(object.position, 3, `${CATALOG[object.id].label} position`);
    const delta = point.map((value, index) => value - origin[index]);
    const score = leftAxis ? delta.reduce((sum, value, index) => sum + value * leftAxis[index], 0) : Math.hypot(...delta);
    return {object, score};
  });
  const best = (rank === 'left' || rank === 'farthest' ? Math.max : Math.min)(...scored.map(row => row.score));
  return scored.filter(row => Math.abs(row.score - best) <= TIE_EPSILON_M).map(row => row.object);
}

/** Resolve an English object description or supported pick command to one placed object's ID.
 * objects: [{id, position: [worldX, worldY, worldZ], active?: boolean}].
 * robot.quaternion uses WXYZ. Distances use 3D object centers relative to the
 * robot position. Names/colors filter first; position ranks those candidates.
 * Missing, ambiguous, unsupported, and tied selections throw an English Error.
 * Inputs are never modified. No network, model, or DOM is involved.
 */
export function resolveObjectCommand(text, objects, robot = {position: [0, 0, .75], quaternion: [1, 0, 0, 0]}) {
  const selector = parseSelector(selectorWords(text));
  if (!Array.isArray(objects)) throw new Error('Object selection requires the placed object list.');
  const active = objects.filter(object => object?.active !== false);
  const ids = new Set();
  for (const object of active) {
    if (!object || !Object.hasOwn(CATALOG, object.id)) throw new Error('The placed object list contains an unsupported object.');
    if (ids.has(object.id)) throw new Error(`More than one ${CATALOG[object.id].label} has the same ID.`);
    ids.add(object.id);
  }
  if (!active.length) throw new Error('Place an object on the table before choosing one.');
  let candidates = selector.name ? active.filter(object => object.id === selector.name) : active.slice();
  if (!candidates.length) throw new Error(`The ${CATALOG[selector.name].label} is not on the table. Place it first.`);
  candidates = candidates.filter(object => selector.colors.every(color => CATALOG[object.id].colors.includes(color)));
  if (!candidates.length) throw new Error('No placed object matches that name and color. Check the object description.');
  if (selector.rank) candidates = rankedCandidates(candidates, selector.rank, robot);
  if (candidates.length !== 1) {
    const labels = candidates.map(object => CATALOG[object.id].label).join(', ');
    throw new Error(`That description is ambiguous (${labels}). Name one object or choose a different position description.`);
  }
  return candidates[0].id;
}
