class config:
    lamp_substep = 8
    lamp_dt = 0.1 / lamp_substep
    lamp_g = 1.2
    lamp_m = 10.
    lamp_k = 500.
    lamp_beta = 20.0
    lamp_k_seesaw = 2_000.
    lamp_mu_seesaw = 0.3
    lamp_r_seesaw = 50.
    lamp_I_seesaw = 1_000.
    lamp_actuators = np.array([16, 17, 18, 19, 20, 21])
    lamp_max_actuation = 0.25
    lamp_k_actuator_mul = 1.0
    
    lamp_block_m_mul = 0.1
    lamp_block_mu_mul = 5.0

    shac_policy_layers = [128, 64]
    shac_value_layers = [128, 64]
    shac_h = 32
    shac_N = 64
    shac_policy_lr = 0.0002 #0.002
    shac_value_lr = 0.0005
    shac_adam_beta1 = 0.7
    shac_adam_beta2 = 0.95
    shac_gamma = 0.99
    shac_lambda = 0.95
    shac_alpha = 0.2  # 0.995
    shac_value_iters = 16
    shac_value_batches = 4
    shac_M = 2_000 * 3
    shac_reset_freq = 10


class Policy(nn.Module):
    @nn.compact
    def __call__(self, x):
        for size in config.shac_policy_layers:
            x = nn.elu(nn.Dense(size)(x))
        x = nn.Dense(len(config.lamp_actuators) * 2)(x)  # 2 for mean/stdev
        return x.reshape(x.shape[0], 2, -1)

class Value(nn.Module):
    @nn.compact
    def __call__(self, x):
        for size in config.shac_value_layers:
            x = nn.elu(nn.Dense(size)(x))
        x = nn.Dense(1)(x)
        return x.reshape(x.shape[0])

policy_net = Policy()
value_net = Value()