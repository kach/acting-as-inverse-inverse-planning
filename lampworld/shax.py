def shax(mdp, init_theta=None, init_phi=None):
    
    init_batch = [mdp.init() for _ in range(config.shac_N)]
    init_batch = state_t(
        *(np.stack(p) for p in zip(*init_batch))
    )

    def policy_loss(key, s, theta, phi):
        total_reward = np.zeros(s.xs.shape[0]) * 0.
        tm = np.ones_like(total_reward)

        x = (key, s, total_reward, tm)
        def loop(t, x):
            key, s, total_reward, tm = x
            key, subkey = jax.random.split(key)
            a = policy_net.apply(theta, jax.vmap(mdp.observe)(s))
            a = jax.random.normal(subkey, a[:, 0, :].shape) * a[:, 1, :] + a[:, 0, :]
            s0 = s
            s = jax.vmap(mdp.step, in_axes=(0, 0))(s, a)
            s, r, tm_ = jax.vmap(mdp.terminate)(s, a, s0, init_batch)
            tm = tm * tm_
            total_reward = total_reward + tm * r * config.shac_gamma ** t
            return key, s, total_reward, tm
        key, s, total_reward, tm = jax.lax.fori_loop(0, config.shac_h, loop, x)
        total_reward = total_reward + tm * (
            value_net.apply(phi, jax.vmap(mdp.observe)(s)) *
            config.shac_gamma ** config.shac_h
        )
        return -np.mean(total_reward / config.shac_h), (s, total_reward)

    def value_loss(s0, phi, total_reward):
        return np.mean(((value_net.apply(phi, jax.vmap(mdp.observe)(s0)) - total_reward) / config.shac_h) ** 2)

    value_optimizer = optax.adam(
        learning_rate=config.shac_value_lr,
        b1=config.shac_adam_beta1, b2=config.shac_adam_beta2
    )
    policy_optimizer = optax.adam(
        learning_rate=config.shac_policy_lr,
        b1=config.shac_adam_beta1, b2=config.shac_adam_beta2
    )

    @jax.jit
    def shac_value_step(s0, phi, total_reward, value_opt_state):
        vloss, d_vloss = jax.value_and_grad(value_loss, argnums=1)(
            s0, phi, total_reward
        )
        value_updates, value_opt_state = value_optimizer.update(d_vloss, value_opt_state)
        phi = optax.apply_updates(phi, value_updates)

        return vloss, value_opt_state, phi

    @jax.jit
    def shac_policy_step(key, s, theta, phi, policy_opt_state):
        (ploss, (s, total_reward)), d_ploss = jax.value_and_grad(policy_loss, argnums=2, has_aux=True)(
            key, s, theta, phi
        )
        policy_updates, policy_opt_state = policy_optimizer.update(d_ploss, policy_opt_state)
        theta = optax.apply_updates(theta, policy_updates)

        return s, theta, policy_opt_state, ploss, total_reward
    
    
    
    
    
    
    
    
    

    key = jax.random.PRNGKey(0)
    s = init_batch

    key, subkey = jax.random.split(key)
    theta = init_theta or policy_net.init(subkey, jax.vmap(mdp.observe)(init_batch))
    policy_opt_state = policy_optimizer.init(theta)

    key, subkey = jax.random.split(key)
    phi = init_phi or value_net.init(subkey, jax.vmap(mdp.observe)(init_batch))

    plosses = []
    vlosses = []

    for m in tqdm(range(config.shac_M + 1)):
        key, subkey = jax.random.split(key)

        s0 = s
        s, theta, policy_opt_state, ploss, total_reward = shac_policy_step(
            subkey, s, theta, phi, policy_opt_state
        )


        value_opt_state = value_optimizer.init(phi)
        phi0 = phi
        for i in range(config.shac_value_iters):
            for b in range(config.shac_value_batches):
                vloss, value_opt_state, phi = shac_value_step(
                    s0, phi, total_reward, value_opt_state
                )

        phi = jax.tree_util.tree_map(
            lambda x, y: x * config.shac_alpha + y * (1 - config.shac_alpha),
            phi0, phi
        )

        plosses.append(ploss)
        vlosses.append(vloss)
        # if m > 0 and m % 100 == 0:
        #     print(np.mean(np.array(plosses[-100:])), np.mean(np.array(vlosses[-100:])))

        if m % config.shac_reset_freq == 0:
            s = init_batch
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.xlabel('Time')
    plt.ylabel('Actor loss (-1 * avg reward)')
    plt.plot(plosses)
    plt.subplot(1, 2, 2)
    plt.plot(vlosses)
    plt.xlabel('Time')
    plt.ylabel('Critic loss')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    return theta, phi