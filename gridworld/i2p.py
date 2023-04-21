import hoh
from hoh import *
from draw_video import *

def load_params(L):
    V_a, Q_a = load_vq(f'{weights_pfx}/a_{L.agx}_{L.agy}.npz')
    V_b, Q_b = load_vq(f'{weights_pfx}/b_{L.agx}_{L.agy}_{L.bgx}_{L.bgy}_{L.rho_o:+d}.npz')
    return V_a, Q_a, V_b, Q_b

hypothesis_space = []
for rho_o in [-3, -1, 0, +1, +3]:
    for ag in [(1, 1), (1, 5)]:
        for bg in [(1, 1), (1, 5), (3, 3)]:
            # (not/or) for at most one goal;  (not/^) for exactly one; (and) for at least one
            # if not ((rho_o == 0) ^ (bg == (3, 3))):
            if rho_o == 0 and bg == (3, 3):
                continue
            hypothesis_space.append( latent_t(* ag + bg, rho_o, True, True) )
for rho_o in [0]:
    for bg in [(1, 1), (1, 5), (3, 3)]:
        hypothesis_space.append( latent_t(* (1, 5) + bg, rho_o, False, True) )
for ag in [(1, 1), (1, 5)]:
    hypothesis_space.append( latent_t(* ag + (3, 3), 0, True, False))
hypothesis_space.append(latent_t(1, 5, 3, 3, 0, False, False))

vqs = {}
# vq = None
for L in hypothesis_space:
    vq = load_params(L)
    if L.rat_a and L.rat_b:
        vqs[L] = vq
    elif L.rat_a:
        vqs[L] = vq[0], vq[1], np.zeros_like(vq[2]), np.zeros_like(vq[3])
    elif L.rat_b:
        vqs[L] = np.zeros_like(vq[0]), np.zeros_like(vq[1]), vq[2], vq[3]
    else:
        vqs[L] = tuple(np.zeros_like(x) for x in vq)
# vqs = {L: load_params(L) for L in hypothesis_space if L.rat_a and L.rat_b}
# vqs[hypothesis_space[-1]] = tuple(np.zeros_like(x) for x in vqs[hypothesis_space[0]])
init_particles = {L: 1. / len(hypothesis_space) for L in hypothesis_space}

v_avg_a = {H: np.mean(vqs[H][0]) for H in hypothesis_space}
v_avg_b = {H: np.mean(vqs[H][2]) for H in hypothesis_space}

def advance_transition(particles, transition):
    epsilon = 1e-5
    particles = particles.copy()
    s, a, who, _ = transition
    
    if who == 'r' or who is None:
        return particles

    for H in hypothesis_space:
        if who == 'a':
            q = vqs[H][1]
        elif who == 'b':
            q = vqs[H][3]
        else:
            raise ValueError(who)

        q = q[s] - np.min(q[s])
        p = math.exp(beta * q[a]) / sum(math.exp(beta * q[a_]) for a_ in Actions)
        # p = 0.99 if a == max(Actions, key=lambda a: q[s][a]) else 0.0025
        # if who == 'a' and not H.rat_a: p = 0.2
        # if who == 'b' and not H.rat_b: p = 0.2

        particles[H] *= p
    norm = sum(particles.values())
    particles = {k:v / norm for k, v in particles.items()}
    particles = {k: v * (1 - epsilon) + (1 - p) * epsilon / len(particles) for k, v in particles.items()}
    return particles

def Pr(belief, predicate, condition=None):
    if condition is None:
        return sum(p for H, p in belief.items() if predicate(H))
    
    joint = sum(p for H, p in belief.items() if predicate(H) and condition(H))
    marginal = sum(p for H, p in belief.items() if condition(H))
    return joint / marginal

def EE(belief, fn, condition=None):
    if condition is None:
        return sum(p * fn(H) for H, p in belief.items())
    
    joint = sum(p * fn(H) for H, p in belief.items() if condition(H))
    marginal = sum(p for H, p in belief.items() if condition(H))
    return joint / marginal


def graph_history(transitions, fname=None):
    particles = init_particles
    history = [particles]
    states = [transitions[0][0]]

    for tr in transitions:
        particles = advance_transition(particles, tr)
        states.append(tr[3])
        history.append(particles)
    
    plt.figure(figsize=(12, 2.5))

    plt.subplot(1, 5, 2)
    plt.plot([Pr(b, lambda H: H.bgy == 1, lambda H: H.rat_b) for b in history], c='magenta')
    plt.plot([Pr(b, lambda H: H.bgy == 5, lambda H: H.rat_b) for b in history], c='limegreen')
    plt.plot([Pr(b, lambda H: H.bgy == 3, lambda H: H.rat_b) for b in history], c='silver', ls='--', label='None')
    plt.title('Robot goal')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.yticks([0, 1])
    plt.legend()

    plt.subplot(1, 5, 3)
    plt.plot([Pr(b, lambda H: H.rho_o < 0, lambda H: H.rat_b) for b in history], c='red', label='Hindering')
    plt.plot([Pr(b, lambda H: H.rho_o == 0, lambda H: H.rat_b) for b in history], c='silver', ls='--', label='Ambivalent')
    plt.plot([Pr(b, lambda H: H.rho_o > 0, lambda H: H.rat_b) for b in history], c='blue', label='Helping')
    plt.title('Robot alignment')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.yticks([0, 1])
    plt.legend()

    plt.subplot(1, 5, 1)
    plt.plot([Pr(b, lambda H: H.agy == 1, lambda H: H.rat_a) for b in history], c='magenta', label='Pink')
    plt.plot([Pr(b, lambda H: H.agy == 5, lambda H: H.rat_a) for b in history], c='limegreen', label='Green')
    plt.title('Cheese goal')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.yticks([0, 1])
    plt.legend()
    
    plt.subplot(1, 5, 4)
    plt.plot([Pr(b, lambda H: H.rat_b) for b in history], c='slategray', label='Robot')
    plt.plot([Pr(b, lambda H: H.rat_a) for b in history], c='goldenrod', ls='--', label='Cheese')
    plt.title('Rational?')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.yticks([0, 1])
    plt.legend()

    plt.subplot(1, 5, 5)
    plt.plot([EE(b, lambda H: vqs[H][2][s], lambda H: H.rat_a and H.rat_b) for b, s in zip(history, states)], c='slategray', label='Robot')
#   plt.legend()
#   plt.twinx()
    plt.plot([EE(b, lambda H: vqs[H][0][s], lambda H: H.rat_a and H.rat_b) for b, s in zip(history, states)], c='goldenrod', ls='--', label='Cheese*')
    plt.title('Value function')
    plt.xlabel('Time')
    plt.ylabel('Long-term reward')

    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname)
        plt.close()
        with open('out/modelout.txt', 'a') as f:
            f.write(f'''{{\
  "name": "{fname}",\
  "help": {Pr(particles, lambda H: H.rho_o > 0)},\
  "hinder": {Pr(particles, lambda H: H.rho_o < 0)},\
  "indifferent": {Pr(particles, lambda H: H.rho_o == 0)}\
  }}
''')

turn_t = namedtuple('turn_t', 'a_action a_outcome b_action b_outcome deus')
all_turns = [turn_t(*x) for x in itertools.product(Actions, [0, 1], Actions, [0], [None])] + [
    turn_t(None, None, None, None, destination) for destination in [(1, 3), (1, 1), (1, 5), (3, 1), (3, 5)]
]
flash_turn = turn_t(Actions.STAY, 0, Actions.SOUTH, 0, None)
flash_count = 1

def apply_turn(s, turn):
    transitions = []
    
    if turn.deus is not None:
        if find(s, *turn.deus) == '.' and maze[s.ry][s.rx] == '#':
            s0 = s
            s = s0._replace(rx=turn.deus[0], ry=turn.deus[1])
            transitions.append( (s0, None, 'r', s) )
            transitions.append( (s, None, None, s) )
        else:
            s0 = s
            transitions.append( (s0, None, None, s0) )
            transitions.append( (s0, None, None, s0) )
        return s, transitions

    L = latent_t(1, 1, 1, 1, 0, True, True)
    s0 = s
    r, trs = step_a(s, turn.a_action, L)
    s = trs[min(turn.a_outcome, len(trs) - 1)][0]
    transitions.append( (s0, turn.a_action, 'a', s) )

    s0 = s
    r, trs = step_b(s, turn.b_action, L)
    s = trs[min(turn.b_outcome, len(trs) - 1)][0]
    transitions.append( (s0, turn.b_action, 'b', s) )
    
    return s, transitions

def unpack_turns(s, turns):
    transitions = []
    for turn in turns:
        s, ts = apply_turn(s, turn)
        transitions += ts

    return transitions


pair_t = namedtuple('pair_t', 'head tail')
be_t = namedtuple('be_t', 's0 turns s history score')

def pair_to_list(pair):
    if pair is None:
        return []
    z = pair_to_list(pair.tail)
    z.append(pair.head)
    return z

def score_history(history, which, runtime, transitions):
    mid = runtime
    both_rat_p = lambda H: H.rat_a and H.rat_b
    rat = sum([Pr(b, both_rat_p) for i, b in enumerate(history)])
    if which == 'help-then-hinder':
        return rat + 0.1 * sum([
            Pr(b, lambda H: (H.rho_o > 0 if i < mid else H.rho_o < 0), both_rat_p) for i, b in enumerate(history)
        ])
    elif which == 'hinder-then-help':
        return rat + 0.1 * sum([
            Pr(b, lambda H: (H.rho_o > 0 if i > mid else H.rho_o < 0), both_rat_p) for i, b in enumerate(history)
        ])
    elif which == 'help':
        return rat + sum([
            Pr(b, lambda H: H.rho_o > 0, both_rat_p) for i, b in enumerate(history)
        ])
    elif which == 'hinder':
        return rat + sum([
            Pr(b, lambda H: H.rho_o < 0, both_rat_p) for i, b in enumerate(history)
        ])
    elif which == 'indifferent':
        return rat + sum([
            Pr(b, lambda H: H.rho_o == 0, both_rat_p) for i, b in enumerate(history)
        ])
    elif which == 'hinder-then-goal':
        return rat + sum([
            Pr(b, lambda H: H.agy == 1 and (H.rho_o < 0 if i < mid else H.rho_o == 0), both_rat_p) for i, b in enumerate(history)
        ])
    elif which == 'help-then-goal':
        return rat + sum([
            Pr(b, lambda H: H.agy == 1 and (H.rho_o > 0 if i < mid else H.rho_o == 0), both_rat_p) for i, b in enumerate(history)
        ])
    elif which == 'irony':
        return (
            rat +
            + sum([Pr(b, lambda H: H.agy == 5, both_rat_p) for i, b in enumerate(history)])
            + sum([Pr(b, lambda H: H.rho_o > 0, lambda H: both_rat_p(H) and H.agy == 1) for i, b in enumerate(history)])
            + sum([Pr(b, lambda H: H.rho_o < 0, lambda H: both_rat_p(H) and H.agy == 5) for i, b in enumerate(history)])
        )
    elif which == 'cinderella':
        scene_p = lambda H: True # and H.rho_o > 2 and H.agy == 1 and H.bgy == 1
        return (
            rat
            + 1e-2 * sum([np.sin(1.5 * i / mid * np.pi) * EE(b, lambda H: vqs[H][2][s], lambda H: scene_p(H) and both_rat_p(H)) for i, (b, s) in enumerate(zip(history, [t[-1] for t in transitions]))])
            - 0.1 * sum(-sum(p[L] * math.log(q[L]) for L in hypothesis_space) for p, q in zip(history, history[1:]))
        )
    elif which == 'flashback-help':
        return (
            rat
            + 0.1 * sum([
                Pr(b, lambda H: H.rho_o > 0, both_rat_p) for i, b in enumerate(history)
            ])
            + 1 * sum([int(tr[0].ay == tr[0].by) for tr in transitions[-flash_count:]])
            + 1 * sum([int(tr[0].ax == tr[0].bx + 1) for tr in transitions[-flash_count:]])
        )
    elif which == 'flashback-hinder':
        return (
            rat
            + 0.1 * sum([
                Pr(b, lambda H: H.rho_o < 0, both_rat_p) for i, b in enumerate(history)
            ])
            + 1 * sum([int(tr[0].ay == tr[0].by) for tr in transitions[-flash_count:]])
            + 1 * sum([int(tr[0].ax == tr[0].bx + 1) for tr in transitions[-flash_count:]])
        )
    else:
        raise ValueError(which)
    return rat

def score_turns(turns, verbose=False):
    out = 0
    den = sum([1 for turn in turns if turn.a_action != Actions.STAY][-5:])
    num = sum([1 * (turn.a_outcome == 0) for turn in turns if turn.a_action != Actions.STAY][-5:])
    if den >= 5:
        so = (num + 1) / (den + 2)
        out += -1 * np.abs(so - 0.6)
    # out += sum((1 if turn.deus is not None else 0) for turn in turns)
    return out

def advance_be(be, which, runtime):
    out = []
    s, turns, s_current, history, score = be
    for turn in all_turns:
        sp, ts = apply_turn(s_current, turn)
        history_new = history
        for tr in ts:
            history_new = pair_t(advance_transition(history_new.head, tr), history_new)
        turns_new = pair_t(turn, turns)

        if which.startswith('flashback'):
            turns_virtual = turns_new
            history_virtual = history_new
            s_virtual = sp
            for _ in range(flash_count):
                s_virtual, ts = apply_turn(s_virtual, flash_turn)
                for tr in ts:
                    history_virtual = pair_t(
                        advance_transition(history_virtual.head, tr),
                        history_virtual
                    )
                    turns_virtual = pair_t(flash_turn, turns_virtual)
            turns_virtual_list = pair_to_list(turns_virtual)
            score_h = score_history(pair_to_list(history_virtual), which, runtime, unpack_turns(s, turns_virtual_list))
            score_t = score_turns(turns_virtual_list)
        else:
            turns_new_list = pair_to_list(turns_new)
            score_h = score_history(pair_to_list(history_new), which, runtime, unpack_turns(s, turns_new_list))
            score_t = score_turns(turns_new_list)

#       turns_new_list = pair_to_list(turns_new)
#       score_h = score_history(pair_to_list(history_new), which, runtime, unpack_turns(s, turns_new_list))
#       score_t = score_turns(turns_new_list)

        score_new = score_h + score_t
        
        out.append( be_t(s, turns_new, sp, history_new, score_new) )
    return out


def lean_search(s0, which, runtime, beam_cuts):
    beam = []
    beam.append( be_t(s0, None, s0, pair_t(init_particles, None), 0.) )
    for t in range(runtime):
        beam = sum(map(functools.partial(advance_be, which=which, runtime=runtime), beam), start=[])
        beam.sort(key=lambda st: st[4])
        beam = beam[-beam_cuts:]
    return beam[-1]

def lean_search_many(seed, which, runtime, deus, beam_size, beam_cuts, video_mode, fname):
    global all_turns
    import random
    random.seed(seed)

    if not deus:
      all_turns = [turn for turn in all_turns if turn.deus is None]

    s0s = []
    while len(s0s) < beam_size:
        s = state_t(*random.choice(S))
        if not state_valid(s):
            continue
        if deus and maze[s.ry][s.rx] != '#':
            continue
        s0s.append(s)

    with Pool(len(os.sched_getaffinity(0)) - 1) as pool:
        bests = list(tqdm(pool.imap_unordered(
            functools.partial(lean_search, which=which, runtime=runtime, beam_cuts=beam_cuts),
            # lambda s0: lean_search(s0, which, int(random.gauss(40, 10)), beam_cuts),
        s0s), total=len(s0s)))
    
    bests.sort(key=lambda st: st.score / len(pair_to_list(st.turns)))
    s, best_turns, sp, history, score = bests[-1]
    print(score)
    if which.startswith('flashback'):
        for _ in range(flash_count):
            best_turns = pair_t(flash_turn, best_turns)
    transitions = unpack_turns(s, pair_to_list(best_turns))

    os.makedirs(f'out/{fname}/', exist_ok=True)
    graph_history(transitions, f'out/{fname}/graph.pdf')
    show_video(transitions, fname, mode=video_mode)


def beam_search(seed=0, which=None, beam_size=500, beam_cuts=len(all_turns), video_mode='cairo', fname='out'):
    import random
    random.seed(seed)

    beam = []
    while len(beam) < beam_size:
        s = state_t(*random.choice(S))
        if not state_valid(s):
            continue
        beam.append( be_t(s, None, s, pair_t(init_particles, None), 0.) )

    for t in tqdm(range(40)):
        with Pool(len(os.sched_getaffinity(0)) - 1) as pool:
            beam = sum(
                pool.imap_unordered(functools.partial(advance_be, which=which), beam),
                start=[]
            )

        beam.sort(key=lambda st: st[4])
        new_beam = []
        beam_counts = {}
        while len(new_beam) < beam_size and beam != []:
            be = beam.pop()
            s0 = be[0]
            bc = beam_counts.get(s0, 0)
            if bc < beam_cuts:
                beam_counts[s0] = bc + 1
                new_beam.insert(0, be)
        # print(len(new_beam), len(set(  be.s0 for be in new_beam  )))
        beam = new_beam
        # beam = beam[-beam_size:]
    
    s, best_turns, sp, history, score = beam[-1]
    print(score)
    transitions = unpack_turns(s, pair_to_list(best_turns))
    graph_history(transitions, f'{fname}.pdf')
    show_video(transitions, fname, mode=video_mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the inverse inverse planner')
    parser.add_argument('--seed', metavar='seed', type=int,
                        help='Which random seed to try')
    parser.add_argument('--which', metavar='which', type=str,
                        help='Which story to make')
    parser.add_argument('--runtime', metavar='runtime', type=int, default=40,
                        help='Length of video in turns')
    parser.add_argument('--beam_size', metavar='beam_size', type=int, default=500,
                        help='Beam size')
    parser.add_argument('--beam_cuts', metavar='beam_cuts', type=int, default=len(all_turns),
                        help='Beam cuts')
    parser.add_argument('--video_mode', metavar='video_mode', type=str, default='cairo',
                        help='Video rendering mode (cairo or pov)')
    parser.add_argument('--deus', action=argparse.BooleanOptionalAction)
    parser.add_argument('--prefix', metavar='prefix', type=str, default='',
                        help='Prefix in output files')
    args = parser.parse_args()
    print(args)

    # beam_search(args.seed, args.which, args.beam_size, args.beam_cuts, args.video_mode, 'out')
    lean_search_many(args.seed, args.which, args.runtime, args.deus, args.beam_size, args.beam_cuts, args.video_mode, f'out-{args.prefix}{args.which}-{args.seed:04}')
