from i2p import *
import random

def make_plan(seed, fname, alignment):
    random.seed(seed)
    while True:
        ag = random.choice([(1, 1), (1, 5)])
        bg = random.choice([(1, 1), (1, 5), (3, 3)])
        L = latent_t(* ag + bg, random.choice([+3, +1]) * alignment, True, True)
        if L in vqs:
            break
    V_a, Q_a, V_b, Q_b = vqs[L]

    while True:
        s = state_t(*random.choice(S))
        if state_valid(s):
            break

    transitions = []
    for _ in range(7):
        s0 = s
        a = max(Actions, key=lambda a: Q_a[s][a])
        r, trs = step_a(s, a, L)
        s = random.choices(
            [s_ for s_, p in trs],
            weights=[p for s_, p in trs]
        )[0]
        transitions.append( (s0, a, 'a', s) )
        s0 = s
        a = max(Actions, key=lambda a: Q_b[s][a])
        r, trs = step_b(s, a, L)
        s = random.choices(
            [s_ for s_, p in trs],
            weights=[p for s_, p in trs]
        )[0]
        transitions.append( (s0, a, 'b', s) )

    os.makedirs(f'out/{fname}/', exist_ok=True)
    graph_history(transitions, f'out/{fname}/graph.pdf')
    show_video(transitions, fname, mode='pov')

if __name__ == '__main__':
#   os.system('rm out/modelout.txt')
    for seed in range(20):
        make_plan(seed=seed, fname=f'out-plan-help-{seed:04}', alignment=+1)
        make_plan(seed=seed, fname=f'out-plan-hinder-{seed:04}', alignment=-1)
        make_plan(seed=seed, fname=f'out-plan-indifferent-{seed:04}', alignment=0)
