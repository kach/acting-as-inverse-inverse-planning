import jax
import jax.numpy as np

import flax
import flax.linen as nn

import optax

from IPython.display import display, Image, Video, Audio

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

import os
import cairo

import notificationstation as ns
ns.login(
    'https://hooks.slack.com/services/REDACTED',
    'REDACTED'
)