from collections import namedtuple
state_t = namedtuple('state', 'xs vs ms ss ls ks theta omega mus')
mdp_t = namedtuple('mdp', 'init observe step terminate')

def make_init(lamp_block_m_mul):
    @jax.jit
    def init():
        xs = np.array([
            [-0.8, -0.8],
            [+0.8, -0.8],
            [0., 0.0],

            [-0.4, 0.5],
            [+0.4, 0.5],
            [0., 1.0],

            [-0.4, 1.5],  #
            [+0.4, 1.5],
            [0., 2.0],

            [-0.4, 2.5],
            [+0.4, 2.5],

            [-3., -1.0],
            [-3., 0.],  #
            [-4., -1.0],
            [-4., 0.],
            [-5., -1.0],
            [-5., 0.],
            [-6., -1.0],
            [-6., 0.]
        ])

        ss = np.array([
            [0, 1], [1, 2], [0, 2],
            [2, 3], [2, 4], [3, 4],
            [3, 5], [4, 5],
            [5, 6], [5, 7], [6, 7],
            [6, 8], [7, 8],
            [8, 9], [8, 10], [9, 10],

            [0, 3], [3, 6], [6, 9],
            [1, 4], [4, 7], [7, 10],

            [11, 12], [11, 13], [11, 14],
            [12, 13], [12, 14],

            [13, 14], [13, 15], [13, 16],
            [14, 15], [14, 16],

            [15, 16], [15, 17], [15, 18],
            [16, 17], [16, 18], [17, 18],

            [6, 12]
        ])

        xs = (xs + np.array([[-4., 0.4]])) * np.array([[5.0, 4.0]])
        vs = np.zeros_like(xs)
        ms = np.ones(len(xs))[..., None] * config.lamp_m
        ms = ms.at[0:2].mul(5.)
        mus = np.ones_like(ms) * config.lamp_mu_seesaw

        ms = ms.at[11:].mul(lamp_block_m_mul)
        mus = mus.at[11:].mul(config.lamp_block_mu_mul)

        ls = np.sqrt(np.sum((xs[ss[:, 0]] - xs[ss[:, 1]]) ** 2, axis=-1, keepdims=True))
        ks = np.ones_like(ls) * config.lamp_k
        ks = ks.at[-1].mul(0.2)
        ks = ks.at[config.lamp_actuators].mul(config.lamp_k_actuator_mul)
        theta = 0.1
        omega = 0.
        return state_t(xs, vs, ms, ss, ls, ks, theta, omega, mus)

    return init


@jax.jit
def step(state, action):
    for _ in range(config.lamp_substep):
        state = step_(state, action)
    return state

@jax.jit
def step_(state, action):
    xs, vs, ms, ss, ls, ks, theta, omega, mus = state
    ls_0 = ls
    ls = ls.at[config.lamp_actuators, :].mul(
        1. + config.lamp_max_actuation * np.tanh(action[..., None]),
        unique_indices=True,
        indices_are_sorted=True
    )

    # Force from gravity
    F_g = config.lamp_g * np.array([0., -1.]) * ms

    # Force from springs
    a_xs = xs[ss[:, 0]]
    b_xs = xs[ss[:, 1]]
    a_vs = vs[ss[:, 0]]
    b_vs = vs[ss[:, 1]]
    dxs = a_xs - b_xs
    dvs = a_vs - b_vs
    ls_ = np.sqrt(np.sum(dxs ** 2, axis=-1, keepdims=True))
    dv_ = np.sum(dxs * dvs, axis=-1, keepdims=True) / ls_
    magnitudes = ks * np.clip(ls_ - ls, -1., 1.) + config.lamp_beta * (dv_)

    fs = magnitudes * (dxs / ls_)

    F_s = -(np.eye(len(ms))[ss[:, 0]][..., None] * fs[:, None, :]).sum(axis=0) +\
          +(np.eye(len(ms))[ss[:, 1]][..., None] * fs[:, None, :]).sum(axis=0)


    # Detect collisions
    normal = np.array([-np.sin(theta), np.cos(theta)])[None, ...]
    parallel = np.array([np.cos(theta), np.sin(theta)])[None, ...]
    ys = np.tan(theta) * xs[:, 0]
    rs = xs[:, 0] / np.cos(theta)
    collisions = (xs[:, 1] < ys)  # & (np.abs(rs) < config.lamp_r_seesaw)

    # Apply collision dynamics TODO: damping
    dys = ys[:] - xs[:, 1]  ##
    F_c = dys[..., None] * np.cos(theta) * config.lamp_k_seesaw
    ## F_s[collisions] += F_c * normal
    ## F_s = F_s.at[collisions].add(F_c * normal)
    F_s = F_s + np.where(collisions[..., None], F_c * normal, 0.)

    # Apply friction
    vps = np.sum(vs[:] * parallel, axis=-1, keepdims=True)  ##
    F_f = -np.sign(vps) * F_c[:] * mus  #config.lamp_mu_seesaw
    ## F_s[collisions] += F_f * parallel
    ## F_s = F_s.at[collisions].add(F_f * parallel)
    F_s = F_s + np.where(collisions[..., None], F_f * parallel, 0.)

    # tau = np.sum(np.where(collisions, -F_c[:, 0] * rs, 0.))  ##
    # omega = omega + tau / config.lamp_I_seesaw
    # omega = (omega + 1. / 500 / config.lamp_substep) % 1.0

    # Step Euler integrator
    Fs = F_g + F_s
    as_ = Fs / ms
    vs = vs + as_ * config.lamp_dt
    xs = xs + vs * config.lamp_dt
    # theta = theta + omega * config.lamp_dt
    # theta = np.clip(theta, -np.pi / 4, +np.pi / 4)

    return state_t(xs, vs, ms, ss, ls_0, ks, theta, omega, mus)

@jax.jit
def observe(state):
    return np.concatenate([
        # state.xs.reshape(-1),
        np.array([state.omega]),
        (state.xs[:, 0] - state.xs[(0,), 0]).reshape(-1),
        (state.xs[:, 1] - np.tan(state.theta) * state.xs[0, 0]).reshape(-1),
        state.vs.reshape(-1)
    ])

@jax.jit
def reward(state, action, old_state):
    return (
        1.0 * np.mean(state.xs[9, 1] - np.tan(state.theta) * state.xs[0, 0]) +
        0.5 * np.mean(state.vs[:11, 0] * state.ms[:11, 0]) +
        -0.1 * np.mean(state.vs[:11] ** 2) +
        -0.1 * np.mean(action ** 2)
    )

@jax.jit
def terminate(state, action, old_state, fresh_state):
    return jax.lax.cond(
        state.xs[9, 1] < 1.5,
        lambda _: (fresh_state, 0., 0.),
        lambda _: (state, reward(state, action, old_state), 1.),
        ()
    )


init = make_init(0.1)
state = init()



def draw(state, t, fname='out', msg='', boxcolor=(0.4, 0.3, 0.1, 1)):
    with cairo.ImageSurface(cairo.FORMAT_ARGB32, 600, 300) as surface:
        ctx = cairo.Context(surface)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        ctx.set_source_rgba(0.8, 0.8, 1 if state.omega < 0.5 else 0, 1)
        if msg.startswith('boom'):
            ctx.set_source_rgba(0.9, 0.14, 0, 1)
            dt = t - int(msg.split('-')[1])
            if dt < 30:
                ctx.translate(10 * np.sin(dt * np.pi / 4), 10 * np.sin(dt * np.pi / 5))
        ctx.rectangle(0, 0, 600, 300)
        ctx.fill()

        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.save()
        ctx.translate(12, 12)
        # ctx.show_text(msg)
        ctx.restore()

        ctx.translate(300, 150)
        ctx.scale(4, -4)
        
        ctx.set_source_rgba(0, 0.3, 0, 1)
        ctx.move_to(-80, -80 * np.tan(state.theta))
        ctx.line_to(+80, +80 * np.tan(state.theta))
        ctx.line_to(+80, -80)
        ctx.line_to(-80, -80)
        ctx.fill()
        
        box_path = [11, 12, 18, 17]
        ctx.move_to(state.xs[box_path[0], 0], state.xs[box_path[0], 1])
        for n in box_path:
            ctx.line_to(state.xs[n, 0], state.xs[n, 1])
        ctx.line_to(state.xs[box_path[0], 0], state.xs[box_path[0], 1])
        ctx.set_source_rgba(*boxcolor)
        ctx.fill_preserve()
        ctx.set_line_width(0.5)
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.stroke()
        
        ctx.move_to(state.xs[6, 0], state.xs[6, 1])
        ctx.line_to(state.xs[12, 0], state.xs[12, 1])
        ctx.set_source_rgba(0.4, 0.4, 0.4, 1)
        ctx.stroke()
        
        guy_path = [0, 1, 2, 4, 5, 7, 8, 10, 9, 8, 6, 5, 3, 2, 0]
        ctx.move_to(state.xs[guy_path[0], 0], state.xs[guy_path[0], 1])
        for n in guy_path:
            ctx.line_to(state.xs[n, 0], state.xs[n, 1])
        ctx.set_source_rgba(0.8, 0.2, 0.4, 1)
        ctx.fill_preserve()
        ctx.set_line_width(0.5)
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.stroke()

        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.set_line_width(0.1)

        for i, (s0, s1) in enumerate(state.ss.tolist()):
            if i not in config.lamp_actuators.tolist():
                ctx.move_to(state.xs[s0][0], state.xs[s0][1])
                ctx.line_to(state.xs[s1][0], state.xs[s1][1])
        # ctx.stroke()
        surface.write_to_png(f'{fname}.png')


def make_video(mdp, theta, phi, seed=0, name='', frames=600, skip=1, boxcolor=None):
    os.system('rm out/*.png')
    
    if boxcolor is None:
        import random
        boxcolor = [random.random() for _ in range(3)] + [1]

    @jax.jit
    def loop(s, key):
        key, subkey = jax.random.split(key)
        a = policy_net.apply(theta, mdp.observe(s)[None, ...])[0]
        a = jax.random.normal(subkey, a[0].shape) * 0.1 * a[1] + a[0]
        s = mdp.step(s, a)
        s, r, _ = mdp.terminate(s, a, s, mdp.init())
        return s, key

    key = jax.random.PRNGKey(seed)
    s = mdp.init()
    for t in tqdm(range(frames + 1)):
        if t % skip == 0:
            draw(jax.device_get(s), t, f'out/{t:09}', boxcolor=boxcolor)
            # plt.figure()
            # draw(jax.device_get(s))
            # plt.title(f't = {t}')
            # plt.savefig(f'out/{t:09}.jpg')
            # plt.clf()
        
        s, key = loop(s, key)

    os.system(f'''../../ffmpeg-git-20220722-i686-static/ffmpeg -hide_banner -loglevel error -framerate 60 -y -pattern_type glob -i 'out/*.png' -c:v libx264 -pix_fmt yuv420p out-{name}.mp4''')
    return Video(f'out-{name}.mp4', html_attributes='controls loop autoplay')