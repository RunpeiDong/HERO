import assert from 'node:assert/strict';
import test from 'node:test';
import {resolveObjectCommand} from '../object_selection.js';

const objects = [
  {id: 'apple', label: 'Apple', active: true, position: [.4, .2, .8]},
  {id: 'can', label: 'Can', active: true, position: [.45, -.1, .8]},
  {id: 'bottle', label: 'Bottle', active: true, position: [.5, .1, .85]},
  {id: 'mug', label: 'Mug', active: true, position: [.4, .3, .82]},
  {id: 'uiuc_i', label: 'UIUC Block I', active: true, position: [.5, -.3, .87]},
  {id: 'cracker_box', label: 'Cheez-It box', active: true, position: [.42, -.24, .85]},
];
const withoutCarton = objects.filter(o => o.id !== 'cracker_box');

test('natural pick wording and complete catalog labels resolve an explicit single object', () => {
  for (const [command, id] of [
    ['pick the apple', 'apple'], ['grab the red fruit', 'apple'], ['Can you please grab the bottle?', 'bottle'],
    ['Please pick up the coffee cup.', 'mug'], ['pick the tin can up for me', 'can'],
    ['I would like you to grasp the UIUC Block I', 'uiuc_i'], ['pick the letter I', 'uiuc_i'],
    ['pick the I-shaped object', 'uiuc_i'], ['pick uiuc_i', 'uiuc_i'],
    ['pick the box', 'cracker_box'], ['grab the carton', 'cracker_box'], ['pick the Cheez-It', 'cracker_box'], ['pick up the cheez it box', 'cracker_box'],
    ['pick the box of crackers', 'cracker_box'], ['grab the cracker box', 'cracker_box'], ['pick the red box', 'cracker_box'],
  ]) assert.equal(resolveObjectCommand(command, objects), id, command);
});

test('real shared colors remain ambiguous while names and multiple colors disambiguate', () => {
  assert.equal(resolveObjectCommand('pick the orange object', objects), 'uiuc_i');
  assert.throws(() => resolveObjectCommand('grab the blue one', objects), /ambiguous.*Can.*UIUC/);
  assert.equal(resolveObjectCommand('pick the orange I', objects), 'uiuc_i');
  assert.equal(resolveObjectCommand('pick the red apple', objects), 'apple');
  assert.equal(resolveObjectCommand('pick the orange and navy object', objects), 'uiuc_i');
  assert.equal(resolveObjectCommand('pick the navy blue I', objects), 'uiuc_i');
  assert.equal(resolveObjectCommand('grab the tan mug', objects), 'mug');
  assert.equal(resolveObjectCommand('pick the red object', withoutCarton), 'apple');
  assert.throws(() => resolveObjectCommand('pick the red object', objects), /ambiguous.*Apple.*Cheez-It|ambiguous.*Cheez-It.*Apple/, 'the carton print is red too');
  assert.equal(resolveObjectCommand('pick the red fruit', objects), 'apple');
  assert.throws(() => resolveObjectCommand('pick the green apple', objects), /No placed object matches/);
  assert.throws(() => resolveObjectCommand('pick the pink mug', objects), /don't recognize/);
});

test('left and right use robot-local Y even when its heading reverses the world ordering', () => {
  const layout = [{id: 'apple', position: [-.4, 0, .8]}, {id: 'bottle', position: [.4, 0, .8]}];
  const robot = {position: [0, 0, .75], quaternion: [Math.SQRT1_2, 0, 0, Math.SQRT1_2]};
  assert.equal(resolveObjectCommand('pick the leftmost object', layout, robot), 'apple');
  assert.equal(resolveObjectCommand('pick the rightmost object', layout, robot), 'bottle');
  assert.equal(resolveObjectCommand('pick the object on your left side', layout, robot), 'apple');
  const reversed = {...robot, quaternion: [Math.SQRT1_2, 0, 0, -Math.SQRT1_2]};
  assert.equal(resolveObjectCommand('grab the object on the left', layout, reversed), 'bottle');
  assert.equal(resolveObjectCommand('pick the left-most object', layout.slice().reverse(), robot), 'apple');
});

test('position descriptions rank the name/color candidates rather than another object type', () => {
  assert.equal(resolveObjectCommand('pick the leftmost object', objects), 'mug');
  assert.equal(resolveObjectCommand('pick the leftmost blue object', objects), 'can');
  assert.equal(resolveObjectCommand('pick the rightmost blue object', objects), 'uiuc_i');
  assert.equal(resolveObjectCommand('pick the nearest bottle', objects), 'bottle');
  assert.equal(resolveObjectCommand('pick the blue can on the left', objects), 'can');
});

test('nearest and farthest measure 3D distance from the supplied robot position', () => {
  const layout = [{id: 'apple', position: [2, 0, .75]}, {id: 'bottle', position: [0, 0, .75]},
    {id: 'mug', position: [1.5, 0, 1.5]}];
  const robot = {position: [1.5, 0, .75]};
  assert.equal(resolveObjectCommand('grab the closest object to me', layout, robot), 'apple');
  assert.equal(resolveObjectCommand('pick the farthest object', layout, robot), 'bottle');
  assert.equal(resolveObjectCommand('pick the nearest object', layout), 'bottle');
});

test('tied extrema and unqualified pronouns never fall back to array order', () => {
  const tied = [{id: 'apple', position: [.4, .1, .8]}, {id: 'can', position: [-.4, .1, .8]}];
  for (const command of ['pick the leftmost object', 'pick the nearest object', 'pick it'])
    for (const order of [tied, tied.slice().reverse()]) assert.throws(() => resolveObjectCommand(command, order), /ambiguous/);
  assert.equal(resolveObjectCommand('pick it', [objects[2]]), 'bottle');
  assert.equal(resolveObjectCommand('pick the object', [objects[2]]), 'bottle');
});

test('inactive and absent named objects cannot redirect a command to another object', () => {
  const layout = [objects[0], {...objects[4], active: false}];
  assert.equal(resolveObjectCommand('pick the red object', layout), 'apple');
  assert.throws(() => resolveObjectCommand('pick the orange I', layout), /UIUC Block I is not on the table/);
  assert.throws(() => resolveObjectCommand('grab the bottle', layout), /Bottle is not on the table/);
  assert.throws(() => resolveObjectCommand('pick it', []), /Place an object/);
});

test('negation, multiple targets, unknown descriptions and unsupported actions are rejected', () => {
  for (const command of ["don't pick the apple", 'do not pick the apple', 'never grab the mug',
    'pick the apple without the can', 'pick apple and mug', 'pick apple or mug',
    'pick orange object and blue object', 'pick both objects', 'pick all objects',
    'pick apple; grab mug', 'pick apple then wait', 'throw the apple', 'place the bottle',
    'move the apple and pick the mug', 'pick the pear', 'pick the cube', 'pick the crate', 'pick the block', 'pick the apples', 'pick the middle object',
    'pick the nearest rightmost object', 'pick the apple with the left hand', 'pick', '', 'pick the'])
    assert.throws(() => resolveObjectCommand(command, objects), Error, command);
});

test('spatial selection validates geometry and cannot quietly choose from invalid positions', () => {
  assert.equal(resolveObjectCommand('pick the apple', [{id: 'apple'}]), 'apple');
  assert.throws(() => resolveObjectCommand('pick the nearest object', [{id: 'apple'}]), /position.*finite/);
  assert.throws(() => resolveObjectCommand('pick the leftmost object', objects, {quaternion: [0, 0, 0, 0]}), /nonzero/);
  assert.throws(() => resolveObjectCommand('pick it', [{id: 'apple'}, {id: 'apple'}]), /same ID/);
});

test('selection is read-only and normalizes the robot quaternion without mutating it', () => {
  const layout = structuredClone(objects), robot = {position: [0, 0, .75], quaternion: [0, 0, 0, 4]};
  const before = structuredClone({layout, robot});
  assert.equal(resolveObjectCommand('pick the leftmost object', layout, robot), 'uiuc_i');
  assert.throws(() => resolveObjectCommand('pick the blue object', layout, robot), /ambiguous/);
  assert.deepEqual({layout, robot}, before);
});


test('Select target accepts noun descriptions without permitting unsupported actions', () => {
  assert.equal(resolveObjectCommand('apple', objects), 'apple');
  assert.equal(resolveObjectCommand('the green bottle', objects), 'bottle');
  assert.equal(resolveObjectCommand('leftmost object', objects), 'mug');
  assert.equal(resolveObjectCommand('orange I', objects), 'uiuc_i');
  assert.throws(() => resolveObjectCommand('blue object', objects), /ambiguous/);
  for (const description of ['throw apple', 'move the green bottle', 'not the apple', 'apple and mug'])
    assert.throws(() => resolveObjectCommand(description, objects), Error, description);
});
