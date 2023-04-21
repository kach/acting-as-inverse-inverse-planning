# Acting as Inverse Inverse Planning

This repository contains source code to accompany the SIGGRAPH paper [Acting as
Inverse Inverse Planning](https://doi.org/10.1145/3588432.3591510) (Chandra,
Li, Tenenbaum, and Ragan-Kelley 2023).

> Great storytellers know how to take us on a journey. They direct characters
> to act---not necessarily in the most rational way---but rather in a way that
> leads to interesting situations, and ultimately creates an impactful
> experience for the audience members looking on.
> 
> If audience experience is what matters, can we help artists and animators
> _directly_ craft such experiences, independent of the concrete character
> actions needed to evoke those experiences? In this paper, we offer a novel
> computational framework for such tools. Our key idea is to optimize
> animations with respect to _simulated_ audience members' experiences.
> 
> To simulate the audience, we borrow an established principle from cognitive
> science: that human social intuition can be modeled as "inverse planning,"
> the task of inferring an agent's (hidden) goals from its (observed) actions.
> Building on this model, we treat storytelling as "_inverse_ inverse
> planning," the task of choosing actions to manipulate an inverse planner's
> inferences.
> 
> Our framework is grounded in literary theory, naturally capturing many
> storytelling elements from first principles. We give a series of examples to
> demonstrate this, with supporting evidence from human subject studies.

```bibtex
@InProceedings{chandra2023acting,
  title = {Acting as Inverse Inverse Planning},
  author = {Kartik Chandra and Tzu-Mao Li and Joshua Tenenbaum and Jonathan Ragan-Kelley},
  booktitle = {Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings (SIGGRAPH '23 Conference Proceedings)},
  month = {aug},
  year = {2023},
  doi = {10.1145/3588432.3591510}
}
```

---

**Contents**

Kitchen domain
- `gridworld/hoh.py` - definition of the grid-world and planner
- `gridworld/train_all.sh` - trains the planner on all possible hypotheses (takes a few hours)
- `gridworld/regp.py` - runs the planner to create animations with naïve planner
- `gridworld/i2p.py` - inverse inverse planner
- `gridworld/i2p.sbatch` - sample slurm script to run inverse inverse planner
- `gridworld/draw_*.py` - utilities to render animations

Hill domain
- `lampworld/model.py` - defines the mass-spring system and implements a differentiable simulator
- `lampworld/shax.py` - our implementation of Short-Horizon Actor-Critic
- `lampworld/SHAX.ipynb` - train controllers for lamp
- `lampworld/Mime.ipynb` - optimize trajectories to mime heavy box
- `lampworld/Story.ipynb` - optimize trajectory to "stumble"
- `lampworld/config.py` - config
- `lampworld/imports.py` - common imports
- `lampworld/notificationstation.py` - utility to notify when jobs complete
- `lampworld/weights/*` - pre-trained weights

Dependencies
- `python3`, `numpy`, `matplotlib`, `jax`, `flax`, `optax`, `cairo`, `tqdm`
- `ffmpeg`, `pov-ray`
