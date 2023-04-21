import functools
from collections import namedtuple
import os
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import cairo

latent_t = namedtuple('latent_t', 'agx agy bgx bgy rho_o rat_a rat_b')
state_t = namedtuple('state_t', 'ax ay bx by rx ry')

from enum import IntEnum
class Actions(IntEnum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    STAY = 4
num_actions = len(Actions)

weights_pfx = 'weights_2x3'

if weights_pfx == 'weights_2x3':
    maze = '''
...#...#...
...........
...#...#...
#.#######.#
...#...#...
...........
...#...#...
'''[1:-1].split('\n')

elif weights_pfx == 'weights_2x2':
    maze = '''
...#...
.......
...#...
#.###.#
...#...
.......
...#...
'''[1:-1].split('\n')
else:
    raise ValueError(weights_pfx)

maze_h = len(maze)
maze_w = len(maze[0])

state_max = state_t(* (maze_w, maze_h) * 3)
state_max_vec = tuple(state_max)

def state_valid(s):
    if maze[s.ay][s.ax] != '.': return False
    if maze[s.by][s.bx] != '.': return False
    if (s.ax, s.ay) == (s.bx, s.by): return False
    if (s.ax, s.ay) == (s.rx, s.ry): return False
    if (s.rx, s.ry) == (s.bx, s.by): return False
    return True

def find(s, x, y):
    if not (0 <= x < maze_w): return '#'
    if not (0 <= y < maze_h): return '#'
    if x == s.rx and y == s.ry: return 'r'
    if x == s.ax and y == s.ay: return 'a'
    if x == s.bx and y == s.by: return 'b'
    return maze[y][x]

ds = [(+1, 0), (-1, 0), (0, +1), (0, -1)]

def move_a(s, a):
    d = ds[a]
    dx, dy = s.ax + d[0], s.ay + d[1]
    if find(s, dx, dy) == '.':
        return s._replace(ax=dx, ay=dy)
    return None

def move_b(s, a):
    d = ds[a]
    dx, dy = s.bx + d[0], s.by + d[1]
    if find(s, dx, dy) == '.':
        return s._replace(bx=dx, by=dy)
    return None

def move_r(s, a):
    d = ds[a]
    dx, dy = s.rx + d[0], s.ry + d[1]
    if find(s, dx, dy) == '.':
        return s._replace(rx=dx, ry=dy)
    return None

cost_stay = -0.1
cost_move = -1.0
beta = 2.0
gamma = 0.99

def step_a(s, a, L):
    r = 1.0 if s.ax == L.agx and s.ay == L.agy else 0.0
    r = r + (cost_stay if a == 4 else cost_move)

    if a == 4:
        return r, [(s, 1.0)]
    s_ = move_a(s, a)
    if s_ is not None:
        return r, [(s_, 0.6), (s, 0.4)]
    else:
        return r, [(s, 1.0)]

def step_b(s, a, L):
    r = 1.0 if s.bx == L.bgx and s.by == L.bgy else 0.0
    r = r + (cost_stay if a == 4 else cost_move)

    if a == 4:
        return r, [(s, 1.0)]
    d = ds[a]
    dx, dy = s.bx + d[0], s.by + d[1]
    o = find(s, dx, dy)
    if o == 'a':
        s_ = move_a(s, a) or s
    elif o == 'r':
        s_ = move_r(s, a) or s
    else:
        s_ = s

    s_ = move_b(s_, a)
    if s_ is not None:
        return r, [(s_, 1.0)]
    else:
        return r, [(s, 1.0)]

def step_complete(trs, step_o, Q_o):
    for s_, p in trs:
        qps = [Q_o[s_][a_] for a_ in range(num_actions)]
        qps = [math.exp(beta * q) for q in qps]
        qps = [q / sum(qps) for q in qps]
        for a_ in Actions:
            r_, trs_ = step_o(s_, a_)
            for s__, p_ in trs_:
                yield s__, p * p_ * qps[a_], r_


import math
import itertools
S = list(itertools.product(*[range(n) for n in state_max_vec]))

V_null = np.zeros(np.concatenate([state_max_vec]))
Q_null = np.zeros(np.concatenate([state_max_vec, np.array([num_actions])]))
Q_null[..., 0:4] = cost_move
Q_null[..., 4] = cost_stay

def value_iteration(step, step_o, Q_o, rho_o):
    V = np.copy(V_null)
    Q = np.copy(Q_null)
    bellres = []
    for vi in tqdm(range(200 + 1)):
        V0 = np.copy(V)
        for s in S:
            sx = state_t(*s)
            if not state_valid(sx):
                continue
            best = 0
            for a in Actions:
                z = 0.
                r, trs = step(sx, a)
                for s_, p, r_o in step_complete(trs, step_o, Q_o):
                    z += p * (V0[s_] + r_o * rho_o)
                best = max(best, r + gamma * z)
                Q[s][a] = r + gamma * z
            V[s] = best
        if vi % 10 == 0:
            bellres.append(np.max(np.abs(V0 - V)))
    return V, Q, bellres


def state_to_string(s):
    grid = [[' '] * maze_h for _ in range(maze_w)]
    for x in range(maze_w):
        for y in range(maze_h):
            grid[x][y] = find(s, x, y)
    return '\n'.join(' '.join(line) for line in grid)


def load_vq(fname):
    x = np.load(fname)
    return x['V'], x['Q']


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        L = latent_t(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), True, True)

        if sys.argv[1] == 'a':
            V_a, Q_a, bellres = value_iteration(functools.partial(step_a, L=L), functools.partial(step_b, L=L), Q_null, 0.0)
            np.savez(f'{weights_pfx}/a_{L.agx}_{L.agy}.npz', V=V_a, Q=Q_a)

        elif sys.argv[1] == 'b':
            V_a, Q_a = load_vq(f'{weights_pfx}/a_{L.agx}_{L.agy}.npz')
            V_b, Q_b, bellres = value_iteration(functools.partial(step_b, L=L), functools.partial(step_a, L=L), Q_a, L.rho_o)
            np.savez(f'{weights_pfx}/b_{L.agx}_{L.agy}_{L.bgx}_{L.bgy}_{L.rho_o:+}.npz', V=V_b, Q=Q_b)

        else:
            print("Bad command...")
            exit()

        with open('log.txt', 'a') as f:
            print(L, bellres, file=f)
