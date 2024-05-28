import pickle
from math import ceil
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from einops import rearrange, repeat
from jax.tree_util import tree_leaves
from oetils import LivePlot, init_plotting
from orbax.checkpoint import PyTreeCheckpointer
from pink import PinkNoiseProcess
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm, trange

from pontryagin.sm import Ant, CartPole, CartPoleModel, Cheetah, GymEnv, GymSystem, Integrator, MLPModel


def learn_model(
        system, model, params, u0, rng, epochs=100, n_traj=100, batch_size=100, lr=2e-3, wd=1e-3, proj_po=False,
        save=None, load=None, monitor=10, plot=True):
    """Learn a dynamics model from random rollouts"""
    rng, key = jax.random.split(rng)
    np_rng = np.random.default_rng(key[1].item())

    if load:
        data = jnp.load(load)
    else:
        # Collect data
        pn = PinkNoiseProcess((n_traj, *u0.T.shape), scale=1, rng=np_rng)
        t = repeat(jnp.arange(len(u0)), 't -> n t d', n=n_traj, d=1)
        u = rearrange(pn.sample(len(u0)), 'n d t -> n t d')
        y, _ = system.rollout(u, rng, tqdm)
        x = system.compress(y)
        data = rearrange(jnp.concatenate([t, x[:, :-1], u, x[:, 1:]], axis=-1), 'n t d -> (n t) d')
        if save:
            jnp.save(save, data)

    val = data[-(len(data) // 10):]
    data = data[:-(len(data) // 10)]
    N = len(data)

    # Initialize optimizer
    optimizer = optax.adamw(lr, weight_decay=wd)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss(params, D):
        return optax.huber_loss(jax.vmap(lambda t, x, u: model.apply(params, t, x, u, method=model.f))(
            D[:, :1].ravel().astype(int),  # t
            D[:, 1 : 1 + system.dx],  # x
            D[:, 1 + system.dx : 1 + system.dx + system.du]),  # u
            D[:, -system.dx:]  # x'
        ).mean()

    @jax.jit
    def step(params, opt_state, data):
        g = jax.grad(loss)(params, data)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        if proj_po:
            # Project onto positive orthant
            params = jax.tree_map(nn.relu, params)
        return params, opt_state

    if plot:
        # Set up live plot
        losses = np.full((2, epochs * N // batch_size // monitor), np.nan)
        fig, ax = plt.subplots()
        ax.set_title('Dynamics model training')
        ax.set_ylim(1e-5, 1e1)
        ax.set_xlim(0, epochs)
        ax.set_yscale('log')
        xs = np.linspace(0, epochs, losses.shape[1])
        num = ax.text(0.02, 0.97, '', transform=ax.transAxes, ha='left', va='top', animated=True)
        (ln1,) = ax.plot(xs, losses[0], animated=True, label='Training loss')
        (ln2,) = ax.plot(xs, losses[1], animated=True, label='Validation loss')
        ax.legend()
        lp = LivePlot(fig.canvas, [ln1, ln2, num], .01)
        plt.show(block=False)
        plt.pause(.1)

    # Training
    for k in trange(epochs):
        data = np_rng.permutation(data)
        for i in (pb := trange(N // batch_size, leave=False)):
            batch = data[i*batch_size : (i+1)*batch_size]
            params, opt_state = step(params, opt_state, batch)

            if plot and (k*(N // batch_size) + i + 1) % monitor == 0:
                # Monitoring
                losses[0, (k*(N // batch_size) + i) // monitor] = loss(params, data)
                losses[1, (k*(N // batch_size) + i) // monitor] = loss(params, val)
                ln1.set_ydata(losses[0])
                ln2.set_ydata(losses[1])
                if isinstance(model, CartPoleModel):
                    pb.write(f"Training loss: {losses[0, (k*(N // batch_size) + i) // monitor]:.3e}, "
                             f"params: {[f'{x:.2f}' for x in tree_leaves(params)]}")
                num.set_text(f'Epoch {k + (i + 1)*batch_size / N:.1f}')
                lp.update()

    return params


def optimize_trajectory(
        u0, system, rng, method, steps, model=None, monitor=None, checkpoint=None, path=None, lr=1e-3, samples=10,
        std=1e-3, stop_G=None, eta=0.1, s0=1e-3, alpha=0.95, elites=.1, plot=True):
    """Learn an action sequence"""
    if model is None and hasattr(system, 'model'):
        model, params = (system.model, system.params) if model is None else model
    samples = samples if method != 'rls' else 1  # RLS uses only one sample
    perturb = jax.jit(lambda z, std, rng: z + std * jax.random.normal(rng, (samples, *z.shape)))
    sim = jax.jit(lambda u: model.rollout(params, u))
    gym_env = isinstance(system, GymSystem)

    def forward(u, rng):
        # Forward pass
        y, r = system.rollout(u, rng)
        x = system.compress(y)
        return x, r

    @jax.jit
    def backward(x, dfdx, dfdu, drdx, drdu, drT):
        # Backward pass
        p = jnp.zeros_like(x).at[-1].set(drT)
        p = jax.lax.fori_loop(1, len(x) + 1, lambda i, l: l.at[-i - 1].set(drdx[-i] + dfdx[-i].T @ l[-i]), p)

        # Compute action gradients
        g = jax.vmap(lambda dldu, dfdu, p: dldu + dfdu.T @ p)(drdu, dfdu, p[1:])
        return g

    @jax.jit
    def reward_grads(x, u):
        # Compute reward gradients
        t = jnp.arange(len(u))
        drdx = jax.vmap(jax.grad(system.r, 1))(t, x[:-1], u)
        drdu = jax.vmap(jax.grad(system.r, 2))(t, x[:-1], u)
        drT = jax.grad(system.rT)(x[-1])
        return drdx, drdu, drT

    @jax.jit
    def model_jacs(x, u):
        # Compute model Jacobians
        f = lambda t, x, u: model.apply(params, t, x, u, method=model.f)
        t = jnp.arange(len(u))
        dfdx = jax.vmap(jax.jacobian(f, 1))(t, x[:-1], u)
        dfdu = jax.vmap(jax.jacobian(f, 2))(t, x[:-1], u)
        return dfdx, dfdu

    @jax.jit
    def mad(u):
        return jax.grad(lambda u: model.rollout(params, u)[1].sum())(u)

    @jax.jit
    def ls_jacs(x, u, xs, us):
        # Estimate Jacobians using finite differences
        dx = rearrange(xs - x, 'n t d -> t n d')[:-1]
        du = rearrange(us - u, 'n t d -> t n d')
        dxu = jnp.concatenate([dx, du], -1)
        dx_ = rearrange(xs - x, 'n t d -> t n d')[1:]
        AB = jax.vmap(jnp.linalg.lstsq)(dxu, dx_)[0]
        A = jax.vmap(jnp.transpose)(AB[:, :system.dx])
        B = jax.vmap(jnp.transpose)(AB[:, system.dx:])
        return A, B

    @jax.jit
    def lslin_jacs(xs, us):
        # Estimate Jacobians by fitting a linear dynamics model to perturbed data
        x_ = rearrange(xs, 'n t d -> t n d')[1:]
        xs = rearrange(xs, 'n t d -> t n d')[:-1]
        us = rearrange(us, 'n t d -> t n d')
        xu = jnp.concatenate([xs, us, jnp.ones((*xs.shape[:-1], 1))], -1)
        F = jax.vmap(jnp.linalg.lstsq)(xu, x_)[0]
        A = jax.vmap(jnp.transpose)(F[:, :system.dx])
        B = jax.vmap(jnp.transpose)(F[:, system.dx:-1])
        return A, B

    @jax.jit
    def lms_jacs(x, u, F):
        # Least mean squares update of Jacobians
        x_ = x[1:]
        xu = jnp.concatenate([x[:-1], u, jnp.ones((len(u), 1))], -1)
        g = jax.vmap(lambda xu, x_, F: (F @ xu - x_)[:, None] @ xu[None, :])(xu, x_, F)
        g = jnp.clip(g, -1, 1)
        F = F - eta * g
        A = F[..., :system.dx]
        B = F[..., system.dx:-1]
        return A, B, F

    @jax.jit
    def rls_jacs(x, u, F, S):
        # Recursive least squares update of Jacobians
        x_ = x[1:]
        xu = jnp.concatenate([x[:-1], u, jnp.ones((len(u), 1))], -1)
        S0 = jnp.eye(F.shape[-1]) * s0
        S = jax.vmap(lambda S, xu: alpha * S + (1 - alpha) * S0 + xu[:, None] @ xu[None, :])(S, xu)
        F = jax.vmap(lambda F, S, xu, x_: F + (x_ - F @ xu)[:, None] @ jnp.linalg.solve(S, xu)[None, :])(F, S, xu, x_)
        A = F[..., :system.dx]
        B = F[..., system.dx:-1]
        return A, B, F, S

    @jax.jit
    def fd(u, G, us, Gs):
        # Naive finite difference method that ignores the dynamic structure
        du = rearrange(us - u, 'n t d -> n (t d)')
        dG = Gs - G
        g = jnp.linalg.lstsq(du, dG)[0]
        g = rearrange(g, '(t d) -> t d', t=us.shape[1])
        return g

    @jax.jit
    def update(g, opt_state, u):
        updates, opt_state = optimizer.update(-g, opt_state, u)
        u = optax.apply_updates(u, updates)
        return u, opt_state

    def grad(u, aux, rng_env, rng_opt):
        # Compute return and gradient
        jacs = None
        std = aux[0]

        if method == 'mad':
            # Use approximate forward pass (model autodiff)
            g = mad(u)

        elif method == 'fd':
            # Get the gradient estimate directly from the finite differences approximation of the objective
            rng_env, key_env = jax.random.split(rng_env)
            _, r = forward(u, key_env)  # Reference trajectory
            us = perturb(u, std, rng_opt)[:-1]  # Sample noisy actions (remove one to account for reference trajectory)
            _, rs = forward(us, rng_env)
            G = r.sum()
            g = fd(u, G, us, rs.sum(-1))

        elif method == 'pbt':
            # Use exact forward pass (planning by trying)
            x, r = forward(u, rng_env)
            jacs = model_jacs(x, u)
            grads = reward_grads(x, u)
            g = backward(x, *jacs, *grads)

        elif method == 'ls':
            # Estimate Jacobians from a least squares fit of perturbed trajectories
            rng_env, key_env = jax.random.split(rng_env)
            x, r = forward(u, key_env)  # Reference trajectory
            us = perturb(u, std, rng_opt)[:-1]  # Sample noisy actions (remove one to account for reference trajectory)
            xs, _ = forward(us, rng_env)
            jacs = ls_jacs(x, u, xs, us)
            grads = reward_grads(x, u)
            g = backward(x, *jacs, *grads)

        elif method == 'lslin':
            # Estimate Jacobians from a least squares linear model of perturbed trajectories
            rng_env, key_env = jax.random.split(rng_env)
            x, r = forward(u, key_env)  # Reference trajectory
            us = perturb(u, std, rng_opt)[:-1]  # Sample noisy actions (remove one to account for reference trajectory)
            xs, rs = forward(us, rng_env)
            jacs = lslin_jacs(xs, us)
            grads = reward_grads(x, u)
            g = backward(x, *jacs, *grads)

        elif method == 'lms':
            # Least mean squares.
            u = perturb(u, std, rng_opt)[0]
            x, r = forward(u, rng_env)
            F = aux[1]
            *jacs, F = lms_jacs(x, u, F)
            aux[1] = F
            grads = reward_grads(x, u)
            g = backward(x, *jacs, *grads)

        elif method == 'rls':
            # Recursive least squares.
            u = perturb(u, std, rng_opt)[0]
            x, r = forward(u, rng_env)
            F, S = aux[1:]
            *jacs, F, S = rls_jacs(x, u, F, S)
            aux[1:] = F, S
            grads = reward_grads(x, u)
            g = backward(x, *jacs, *grads)

        return g, aux, jacs

    def cem(u, std, rng_env, rng_opt):
        us = perturb(u, std, rng_opt)
        _, rs = forward(us, rng_env)
        Gs = rs.sum(-1)
        n_elites = ceil(elites * samples)
        best = jnp.argpartition(Gs, -n_elites)[-n_elites:]
        u = us[best].mean(0)
        std = us[best].std(0)
        return u, std

    def step(u, opt_state, aux, rng_env, rng_opt):
        # Optimization step
        if method == 'cem':
            std = aux[0]
            u, std = cem(u, std, rng_env, rng_opt)
            # aux[0] = std  # makes it worse. also, not compatible with std schedule.
        else:
            g, aux, _ = grad(u, aux, rng_env, rng_opt)
            u, opt_state = update(g, opt_state, u)
        return u, opt_state, aux

    def init_monitor(rng):
        rng_env1, rng_env2, rng_opt = jax.random.split(rng, 3)
        if gym_env:
            rng_env1 = rng_env2 = rng_monitor

        g_approx = jacs = None
        if method != 'cem':
            # Get approximate gradient
            g_approx, *_, jacs = grad(u, aux, rng_env1, rng_opt)

        std = None
        if method == 'cem':
            std = aux[0]

        x_mod = None
        if method == 'pbt' or method == 'mad':
            # Get model forward pass
            x_mod, _ = sim(u)

        A = B = None
        if jacs is not None:
            A, B = jacs

        # Query system for data to plot
        r, plotting, _ = system.monitor(u, rng_env2, g_approx=g_approx, x_mod=x_mod, std=std, A=A, B=B, plot=plot)

        # Plot trajectory optimization status
        if plot:
            lpls, sys_lines, ylim = plotting
            fig, ax = plt.subplots(num=2)
            ax.set_title("Trajectory optimization")
            ax.set_xlabel("$t$")
            ax.set_ylim(-ylim, ylim)
            lines = []
            for y, label, fmt in sys_lines:
                lines.append(ax.plot(y, fmt, animated=True, label=label)[0])
            num = ax.text(0.02, 0.97, '', transform=ax.transAxes, ha='left', va='top', animated=True)
            lines.append(num)
            ax.legend(loc='lower right', ncol=2)
            lp = LivePlot(fig.canvas, lines, .01)
            plt.show(block=False)
            plt.pause(.1)
            return r.sum(), ((lp, lines),) + lpls

        return r.sum(), None

    def monitoring(rng):
        rng_env1, rng_env2, rng_opt = jax.random.split(rng, 3)
        if gym_env:
            rng_env1 = rng_env2 = rng_monitor

        g_approx = jacs = None
        if method != 'cem':
            # Get approximate gradient
            g_approx, *_, jacs = grad(u, aux, rng_env1, rng_opt)

        std = None
        if method == 'cem':
            std = aux[0]

        x_mod = None
        if method == 'pbt' or method == 'mad':
            # Get model forward pass
            x_mod, r_mod = sim(u)

        A = B = None
        if jacs is not None:
            A, B = jacs

        # Query system for data to plot
        r_sys, text, _ = system.monitor(u, rng_env2, lpls, g_approx=g_approx, x_mod=x_mod, std=std, A=A, B=B, plot=plot)
        if plot:
            lp, lines = lpls[0]
            lines[-1].set_text(f'Iteration {i:3d}\nReturn: {r_sys.sum():.2f} (best: {ret_max:.2f})')
            lp.update()

        # Update live plot and status text
        if method == 'pbt':
            text.update({'model return': f"{r_mod.sum():.2f}"})
        text.update({
            'return': f"{r_sys.sum():.2f}",
            'best' + (' (model return)' if method == 'mad' else ''): f"{ret_max:.2f}"
        })
        if not plot:
            print(
                f"Step {i:{int(np.log10(steps) + 1)}d}. " + ", ".join(f"{k}: {v}" for k, v in text.items()), flush=True)

        return r_sys.sum()

    # Initialize optimizer (not used by CEM)
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1),
        optax.adam(lr),
        optax.zero_nans(),
    )
    opt_state = optimizer.init(u0)

    # Initialize optimization method-specific variables
    std_rate = (std[1] / std[0])**(1 / steps) if not isinstance(std, float) else 1
    std = std[0] if not isinstance(std, float) else std
    aux = [std]
    if method in ['lms', 'rls']:
        rng, *keys = jax.random.split(rng, 3)
        A = (repeat(np.eye(system.dx), 'x y -> t x y', t=len(u0))
             + 0.01 * jax.random.normal(keys[0], (len(u0), system.dx, system.dx)))
        B = 0.01 * jax.random.normal(keys[1], (len(u0), system.dx, system.du))
        c = jnp.zeros((len(u0), system.dx))
        F = jnp.concatenate([A, B, c[..., None]], -1)
        if method == 'rls':
            S = repeat(jnp.eye(F.shape[-1]), 'x y -> n x y', n=len(F)) * s0
            aux = [std, F, S]
        else:
            aux = [std, F]
    elif method == 'cem':
        aux = [std * jnp.ones_like(u0)]

    # Initialize other variables
    i = 0
    u, ret_max = u0, -np.inf
    best = returns = None
    rng, rng_env, rng_opt = jax.random.split(rng, 3)
    if gym_env:
        rngs = np.random.default_rng(rng[1].item()).spawn(2)
        rng_env = key_env = rngs[0]
        rng_monitor = rngs[1]
    if monitor:
        print("Compiling...")
        rng, key = jax.random.split(rng)
        ret, lpls = init_monitor(key)
        returns = np.zeros(steps // monitor + 1)
        returns[0] = ret

    # Restore from checkpoint
    if checkpoint and path:
        try:
            opt_state_struct = jax.tree_util.tree_structure(opt_state)
            cp = PyTreeCheckpointer().restore(path / 'checkpoint')
            i, u, ret_max, best, returns, opt_state, aux, rng, rng_env_state, rng_opt = cp
            if gym_env:
                rng_env.bit_generator.state = jax.tree_map(
                    lambda x, y: type(x)(y), rng_env.bit_generator.state, rng_env_state)
            else:
                rng_env = rng_env_state
            opt_state = jax.tree_util.tree_unflatten(opt_state_struct, jax.tree_util.tree_leaves(opt_state))
            print(f"Restored from checkpoint at step {i}.")
        except FileNotFoundError:
            pass

    # Trajectory optimization loop
    for i in (pb := trange(i + 1, steps + 1)):
        aux[0] *= std_rate
        if not gym_env:
            rng_env, key_env = jax.random.split(rng_env)
        rng_opt, key_opt = jax.random.split(rng_opt)
        u, opt_state, aux = step(u, opt_state, aux, key_env, key_opt)

        if monitor and i % monitor == 0:
            rng, key = jax.random.split(rng)
            ret = monitoring(key)
            if ret > ret_max:
                ret_max, best =  ret, u
            returns[i // monitor] = ret

        if checkpoint and path and i % checkpoint == 0:
            rng_env_state = jax.tree_map(str, rng_env.bit_generator.state) if gym_env else rng_env
            PyTreeCheckpointer().save(path / 'checkpoint',
                (i, u, ret_max, best, returns, opt_state, aux, rng, rng_env_state, rng_opt), force=True)

    if best is not None:
        u = best
    if path:
        jnp.save(path / 'actions.npy', u)
        if monitor:
            np.save(path / 'returns.npy', returns)
    if monitor:
        monitoring(rng)
    return u, returns, i


class MonitorCallback(BaseCallback):
    def __init__(self, monitor, env):
        super().__init__()
        self.monitor = monitor
        self.env = env
        self.i = 0
        self.u = None
        self.ret_max = -np.inf
        self.returns_cl = []
        self.returns_ol = []

    def monitoring(self):
        # Closed-loop rollout
        obs, _ = self.env.reset()
        done = False
        actions = []
        rewards = []
        ep_len = 0
        terminated = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(action)
            obs, r, term, done, _ = self.env.step(action, val=True)
            if not terminated:
                ep_len += 1
            terminated |= term
            rewards.append(r)
        u = jnp.array(actions)
        ret_cl = sum(rewards)
        self.returns_cl.append(ret_cl)

        # Open-loop rollout
        obs, _ = self.env.reset()
        done = False
        rewards = []
        for action in u:
            obs, r, _, done, _ = self.env.step(action, val=True)
            rewards.append(r)
        ret_ol = sum(rewards)
        self.returns_ol.append(ret_ol)

        if ret_ol > self.ret_max:
            self.ret_max, self.u = ret_ol, u
        print(f"Step {self.i:5d}. episode length: {ep_len}, closed-loop return: {ret_cl:.2f}, "
              f"open-loop return: {ret_ol:.2f}, best: {self.ret_max:.2f}", flush=True)

    def dump(self):
        return self.i, self.u, self.ret_max, self.returns_cl, self.returns_ol

    def load(self, data):
        self.i, self.u, self.ret_max, self.returns_cl, self.returns_ol = data
        self.i -= 1

    def _on_step(self):
        if self.i % self.monitor == 0:
            self.monitoring()
        self.i += 1
        return True

    def _on_training_end(self):
        self.monitoring()


class CheckpointCallback(BaseCallback):
    def __init__(self, checkpoint, path, monitor, rngs):
        super().__init__()
        self.checkpoint = checkpoint
        self.path = path
        self.monitor = monitor
        self.rngs = rngs
        self.i = 0

    def checkpointing(self):
        self.model.save(self.path / 'model.zip')
        self.model.save_replay_buffer(self.path / 'replay_buffer.pkl')
        with open(self.path / 'monitor.pkl', 'wb') as f:
            pickle.dump(self.monitor.dump(), f)
        with open(self.path / 'rngs.pkl', 'wb') as f:
            pickle.dump(self.rngs, f)

    @staticmethod
    def restore(path):
        model = SAC.load(path / 'model.zip')
        model.load_replay_buffer(path / 'replay_buffer.pkl')
        with open(path / 'monitor.pkl', 'rb') as f:
            monitor_data = pickle.load(f)
        with open(path / 'rngs.pkl', 'rb') as f:
            rngs = pickle.load(f)
        return model, monitor_data, rngs

    def _on_step(self):
        if self.i % self.checkpoint == 0:
            self.checkpointing()
        self.i += 1
        return True

    def _on_training_end(self):
        self.checkpointing()


def train_policy(system, rng, horizon, steps, monitor, checkpoint, path, ent_coef='auto'):
    """Train a policy using SAC"""
    restored = False
    if checkpoint and path:
        try:
            policy, monitor_data, rngs = CheckpointCallback.restore(path)
            training_env = GymEnv(system, rngs[0], horizon)
            eval_env = GymEnv(system, rngs[1], horizon)
            policy.set_env(training_env)
            monitor = MonitorCallback(monitor, eval_env)
            monitor.load(monitor_data)
            print(f"Restored from checkpoint at step {monitor.i}.")
            restored = True
        except FileNotFoundError:
            pass

    if not restored:
        keys = jax.random.split(rng, 2)
        rngs = np.random.default_rng(keys[0][1].item()).spawn(2)
        training_env = GymEnv(system, rngs[0], horizon)
        eval_env = GymEnv(system, rngs[1], horizon)
        policy = SAC("MlpPolicy", training_env, ent_coef=ent_coef, verbose=2, seed=keys[1][1].item())
        monitor = MonitorCallback(monitor, eval_env)

    callbacks = [monitor]
    if checkpoint and path:
        callbacks.append(CheckpointCallback(checkpoint, path, monitor, rngs))
    policy.learn(total_timesteps=steps - monitor.i, log_interval=None, progress_bar=True, callback=callbacks)

    if path:
        jnp.save(path / 'actions.npy', monitor.u)
        np.save(path / 'returns_cl.npy', np.array(monitor.returns_cl))
        np.save(path / 'returns_ol.npy', np.array(monitor.returns_ol))

    return monitor.u, np.array(monitor.returns_ol), monitor.i


def run(seed, env, method, u0, steps, monitor, checkpoint, model=None, plot=False, env_kwargs=None, method_kwargs=None,
        model_kwargs=None, sac_kwargs=None, train_kwargs=None, train_model=False, load_model=None, path=None):
    """Run an experiment"""
    if plot:
        init_plotting(show=True)
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)

    env_kwargs = env_kwargs or {}
    method_kwargs = method_kwargs or {}


    # Initialize system
    if env == 'integrator':
        env = Integrator(**env_kwargs)
    elif env == 'cartpole':
        env = CartPole(**env_kwargs)
    elif env == 'ant':
        env = Ant(**env_kwargs)
    elif env == 'cheetah':
        env = Cheetah(**env_kwargs)

    # Initial action sequence
    u0 = u0[1] * jax.random.normal(keys[0], (u0[0], env.du))

    # Initialize model
    if model == 'mlp':
        model = MLPModel(env, **model_kwargs)
    elif model == 'cartpole':
        model = CartPoleModel(env, **model_kwargs)
    if model is not None:
        params = model.init(keys[1], 0, env.x0, u0[0])

        # Train dynamics model
        if load_model:
            params = PyTreeCheckpointer().restore((Path(path) if path else Path.cwd()) / '..' / load_model / 'model')
        elif train_model:
            params = learn_model(env, model, params, u0, keys[2], plot=plot, **train_kwargs)
            if path is not None:
                PyTreeCheckpointer().save(path / 'model', params)

        model = (model, params)

    # Learn action sequence
    if method != 'sac':
        u, rets, i = optimize_trajectory(
            u0, env, keys[3], method, steps, model, monitor, checkpoint, path, plot=plot, **method_kwargs)
    else:
        u, rets, i = train_policy(env, keys[3], len(u0), steps, monitor, checkpoint, path, **sac_kwargs)

    if plot:
        rng = np.random.default_rng(keys[4][1].item()) if isinstance(env, GymSystem) else keys[4]
        env.render(u, rng)

    return {
        "avg_return": rets.mean(),
        "max_return": rets.max(),
        "iterations": i
    }
