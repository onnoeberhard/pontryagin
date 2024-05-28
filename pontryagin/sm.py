"""Systems and models."""
from collections import deque
from dataclasses import field
from functools import partial

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from oetils.plotting import LivePlot
from jax.scipy.spatial.transform import Rotation as R


class Model(nn.Module):
    system: object
    convert: bool = True    # Whether this model uses compressed states

    def __call__(self, t, x, u):
        return self.f(t, x, u)

    def reset(self):
        x0 = self.system._reset()
        if self.convert:
            x0 = self.system.compress(x0)
        return x0

    def f(self, t, x, u):
        if self.convert:
            x = self.system.expand(x)
        x = self.system._f(t, x, u)
        if self.convert:
            x = self.system.compress(x)
        return x

    def r(self, t, x, u):
        if self.convert:
            x = self.system.expand(x)
        return self.system.r(t, x, u)

    def rT(self, x):
        if self.convert:
            x = self.system.expand(x)
        return self.system.rT(x)

    def rollout(self, params, u):
        def step(t, xr):
            x, r = xr
            r = r.at[t].set(self.apply(params, t, x[t], u[t], method=self.r))
            x = x.at[t+1].set(self.apply(params, t, x[t], u[t], method=self.f))
            return x, r

        x0 = self.reset()
        x = jnp.zeros((len(u) + 1, *x0.shape))
        x = x.at[0].set(x0)
        r = jnp.zeros(len(u) + 1)
        x, r = jax.lax.fori_loop(0, len(u), step, (x, r))
        r = r.at[-1].set(self.apply(params, x[-1], method=self.rT))
        return x, r


class System:
    def _reset(self):
        return self.reset(jax.random.PRNGKey(0))

    def _f(self, t, x, u):
        return self.f(t, x, u, jax.random.PRNGKey(0))

    def expand(self, x):
        return x

    def compress(self, x):
        return x

    def rollout(self, u, rng, pb=None):
        if u.ndim == 2:
            return self._rollout(u, rng)
        x = np.zeros((u.shape[0], u.shape[1] + 1, self.dx))
        r = np.zeros((u.shape[0], u.shape[1] + 1))
        pb = pb or (lambda x: x)
        keys = jax.random.split(rng, len(u))
        for i in pb(range(len(u))):
            x[i], r[i] = self.rollout(u[i], keys[i])
        return x, r

    def _rollout(self, u, rng):
        x = np.zeros((len(u)+1, self.dx))
        r = np.zeros(len(u)+1)
        x[0] = self.reset(rng)
        for t, ut in enumerate(u):
            x[t + 1] = self.f(t, x[t], ut, rng)
            r[t] = self.r(t, x[t], ut)
        r[-1] = self.rT(x[-1])
        return x, r

    def _stats(self, u, rng, A=None, B=None):
        x_sys, r = self.rollout(u, rng)
        return x_sys, r, None, None, None

    def stats(self, u, rng, g_approx=None, x_mod=None, A=None, B=None):
        x_sys, r, g_true, A_error, B_error = self._stats(u, rng, A, B)
        data = {
            'x_sys': x_sys,
            'x_mod': x_mod,
            'u': u,
            'r': r,
            'g_true': g_true,
            'g_approx': g_approx,
            'A_error': A_error,
            'B_error': B_error,
        }
        labels = {}

        labels['x_sys'] = '$x$' if x_sys.shape[1] == 1 else r'$\|x\|$'
        x_sys = x_sys[:, 0] if x_sys.shape[1] == 1 else np.linalg.norm(x_sys, axis=-1)

        if x_mod is not None:
            labels['x_mod'] = r'$\tilde x$' if x_mod.shape[1] == 1 else r'$\|\tilde x\|$'
            x_mod = x_mod[:, 0] if x_mod.shape[1] == 1 else np.linalg.norm(x_mod, axis=-1)

        labels['u'] = r'$u$' if u.shape[1] == 1 else r'$\|u\|$'
        u = u[:, 0] if u.shape[1] == 1 else np.linalg.norm(u, axis=-1)

        if g_approx is not None:
            labels['g_approx'] = r'$\tilde\nabla J$' if g_approx.shape[1] == 1 else r'$\|\tilde\nabla J\|$'
            g_approx = g_approx[:, 0] if g_approx.shape[1] == 1 else np.linalg.norm(g_approx, axis=-1)

        if g_true is not None:
            labels['g_true'] = r'$\nabla J$' if g_true.shape[1] == 1 else r'$\|\nabla J\|$'
            g_true = g_true[:, 0] if g_true.shape[1] == 1 else np.linalg.norm(g_true, axis=-1)

        return x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data

    def monitor(self, u, rng, lpls=None, g_approx=None, x_mod=None, std=None, A=None, B=None, plot=True):
        x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data = self.stats(u, rng, g_approx, x_mod, A, B)
        if lpls is not None:
            i = 0
            lines = lpls[0][1]
            lines[0 + i].set_ydata(x_sys)
            if x_mod is not None:
                lines[0 + i + 1].set_ydata(x_mod)
                i += 1
            lines[1 + i].set_ydata(u)
            if std is not None:
                lines[1 + i + 1].set_ydata(u + std)
                lines[1 + i + 2].set_ydata(u - std)
                i += 2
            lines[2 + i].set_ydata(r)
            if g_true is not None:
                lines[2 + i + 1].set_ydata(g_true)
                i += 1
            if g_approx is not None:
                lines[2 + i + 1].set_ydata(g_approx)
                i += 1
            if A_error is not None and B_error is not None:
                lines[2 + i + 1].set_ydata(A_error)
                lines[2 + i + 2].set_ydata(B_error)
                i += 2
            return r, {}, data
        elif plot:
            lines = [
                (x_sys, labels['x_sys'], 'C0'),
                (u, labels['u'], 'C1'),
                (r, r'$r$', 'C3'),
            ]
            if std is not None:
                lines.insert(2, (u + std, None, 'C1--'))
                lines.insert(3, (u - std, None, 'C1--'))
            if x_mod is not None:
                lines.insert(1, (x_mod, labels['x_mod'], 'C0--'))
            if g_true is not None:
                lines.append((g_true, labels['g_true'], 'C2'))
            if g_approx is not None:
                lines.append((g_approx, labels['g_approx'], 'C2--'))
            if A_error is not None and B_error is not None:
                lines += [
                    (A_error, r'$\|\partial_x\tilde f - \partial_x f\|$', 'C4'),
                    (B_error, r'$\|\partial_u\tilde f - \partial_u f\|$', 'C5'),
                ]

            lpls = ()
            return r, (lpls, lines, 5), data
        else:
            return r, {}, data


class JaxSystem(System):
    @partial(jax.jit, static_argnums=(0,3))
    def rollout(self, u, rng, pb=None):
        if u.ndim > 2:
            rng = jax.random.split(rng, len(u))
            return jax.vmap(self.rollout)(u, rng)
        return self._rollout(u, rng)

    @partial(jax.jit, static_argnums=(0,))
    def _rollout(self, u, rng):
        def step(t, xrr):
            x, r, rng = xrr
            rng, key = jax.random.split(rng)
            r = r.at[t].set(self.r(t, x[t], u[t]))
            x = x.at[t+1].set(self.f(t, x[t], u[t], key))
            return x, r, rng

        rng, key = jax.random.split(rng)
        x0 = self.reset(key)
        x = jnp.zeros((len(u) + 1, *x0.shape))
        x = x.at[0].set(x0)
        r = jnp.zeros(len(u) + 1)
        x, r, _ = jax.lax.fori_loop(0, len(u), step, (x, r, rng))
        r = r.at[-1].set(self.rT(x[-1]))
        return x, r

    @partial(jax.jit, static_argnums=(0,))
    def _stats_init(self, u, rng):
        x_sys, r = self.rollout(u, rng)
        cost = lambda u: self.rollout(u, rng)[1].sum()
        g_true = jax.grad(cost)(u)
        return x_sys, r, g_true

    @partial(jax.jit, static_argnums=(0,))
    def jac_error(self, u, x, A, B):
        t = jnp.arange(len(u))
        A_error = jnp.linalg.norm(A - jax.vmap(jax.jacobian(self._f, 1))(t, x[:-1], u), axis=(1, 2))
        B_error = jnp.linalg.norm(B - jax.vmap(jax.jacobian(self._f, 2))(t, x[:-1], u), axis=(1, 2))
        return A_error, B_error

    def _stats(self, u, rng, A=None, B=None):
        x_sys, r, g_true = self._stats_init(u, rng)
        A_error, B_error = None, None
        if A is not None and B is not None:
            A_error, B_error = self.jac_error(u, x_sys, A, B)
        return x_sys, r, g_true, A_error, B_error


class Integrator(JaxSystem):
    def __init__(self, goal, double=False, reg=1e-1, noise=0, bound=None):
        n = len(goal)
        self.dx = n
        self.du = n if not double else n // 2
        self.dt = 1
        self.goal = jnp.asarray(goal)
        self.x0 = jnp.zeros(n)
        self.double = double
        self.reg = reg
        self.noise = noise
        self.bound = bound if bound is not None else jnp.inf
        self.model = Model(self)
        self.params = self.model.init(jax.random.PRNGKey(0), 0, self.x0, jnp.zeros(self.du))

    def _reset(self):
        return self.x0

    def reset(self, rng):
        x0 = self._reset()
        if self.noise:
            x0 += self.noise * jax.random.normal(rng, x0.shape)
        return x0

    def _f(self, t, x, u):
        if self.double:
            v = x[self.dx // 2:] + u*self.dt
            z = x[:self.dx // 2] + v*self.dt
            x = jnp.r_[z, v]
        else:
            x += u*self.dt
        return jnp.clip(x, -self.bound, self.bound)

    def f(self, t, x, u, rng):
        x = self._f(t, x, u)
        if self.noise:
            x += self.noise * jax.random.normal(rng, x.shape)
        return jnp.clip(x, -self.bound, self.bound)

    def r(self, t, x, u):
        return -self.reg * jnp.linalg.norm(u)**2

    def rT(self, x):
        return -jnp.linalg.norm(x - self.goal)**2

    def monitor(self, u, lpls=None, g_approx=None, x_mod=None, std=None, A=None, B=None, plot=True):
        r, tp, data = super().monitor(u, lpls, g_approx, x_mod, std, A, B, plot)
        x_sys = data['x_sys']
        x_mod = data['x_mod']

        if lpls is not None and len(lpls) > 1:
            lp, lines = lpls[1]
            lines[0].set_xdata(x_sys[:, 0])
            lines[0].set_ydata(x_sys[:, 1])
            if x_mod is not None:
                lines[1].set_xdata(x_mod[:, 0])
                lines[1].set_ydata(x_mod[:, 1])
            lp.update()
            return r, {}, data
        elif plot:
            _, lines, _ = tp
            lpls = []
            if self.dx == 2 or self.double and self.dx == 4:
                # Plot 2d map
                fig, ax = plt.subplots(figsize=(4, 4))
                b = abs(self.goal).max()
                ax.set_xlim(-1.05*b, 1.05*b)
                ax.set_ylim(-1.05*b, 1.05*b)
                ax.set_aspect('equal')
                if self.double and self.dx == 2:
                    ax.set_xlabel('$x$')
                    ax.set_ylabel(r'$\dot x$')
                else:
                    ax.set_xlabel('$x_1$')
                    ax.set_ylabel('$x_2$')
                map_lines = []
                map_lines.append(ax.plot(x_sys[:, 0], x_sys[:, 1], 'C0', label=r'$x$', animated=True)[0])
                if x_mod is not None:
                    map_lines.append(ax.plot(x_mod[:, 0], x_mod[:, 1], 'C0--', label=r'$\hat x$', animated=True)[0])
                ax.plot([self.goal[0]], [self.goal[1]], '.', c='red', label=r'Goal')
                ax.legend()
                lp = LivePlot(fig.canvas, map_lines, .01)
                plt.show(block=False)
                plt.pause(.1)
                lpls = ((lp, map_lines),)
            return r, (lpls, lines, 1.5*jnp.linalg.norm(self.goal)), data
        else:
            return r, {}, data

    def render(self, u, rng):
        plt.show()


class CartPole(JaxSystem):
    def __init__(self, *args, noise=0, **kwargs):
        self.model = CartPoleModel(self, *args, **kwargs)
        self.noise = noise
        self.x0 = self.model.x0
        self.dx = len(self.x0)
        self.du = 1
        self.params = self.model.init(jax.random.PRNGKey(0), 0, self.x0, jnp.zeros(self.du))

    def _reset(self):
        return self.x0

    def reset(self, rng):
        x0 = self._reset()
        if self.noise:
            x0 += self.noise * jax.random.normal(rng, x0.shape)
        return x0

    def _f(self, t, x, u):
        return self.model.apply(self.params, t, x, u)

    def f(self, t, x, u, rng):
        x = self._f(t, x, u)
        if self.noise:
            x += self.noise * jax.random.normal(rng, x.shape)
        return x

    def r(self, t, x, u):
        return -1e-3 * u[0]**2

    def rT(self, x):
        return -(abs(x[0]) + abs(x[2]) + abs(x[3]) + abs((x[1] - jnp.pi)%(2 * jnp.pi) - jnp.pi))

    def stats(self, u, rng, g_approx=None, x_mod=None, A=None, B=None):
        x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data = super().stats(
            u, rng, g_approx, x_mod, A, B)
        labels['x_sys'] = r'$\theta$'
        x_sys = data['x_sys'][:, 1] % (2 * jnp.pi) - jnp.pi
        if x_mod is not None:
            labels['x_mod'] = r'$\tilde\theta$'
            x_mod = data['x_mod'][:, 1] % (2 * jnp.pi) - jnp.pi
        return x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data

    def monitor(self, u, rng, lpls=None, g_approx=None, x_mod=None, std=None, A=None, B=None, plot=True):
        r, tp, data = super().monitor(u, rng, lpls, g_approx, x_mod, std, A, B, plot)
        x_sys = data['x_sys']

        if lpls is not None:
            lp, lines = lpls[1]
            R = self.params['params']['R']
            x_up = x_sys[:, 0] - R * jnp.sin(x_sys[:, 1])
            y_up = R * jnp.cos(x_sys[:, 1])
            x_car = x_sys[:, 0]
            subidx = np.r_[0:len(u):(len(u)//10), len(u)-1]
            for i, j in enumerate(subidx):
                lines[i].set_xdata([x_car[j], x_up[j]])
                lines[i].set_ydata([0, y_up[j]])
            lines[-1].set_xdata(x_up)
            lines[-1].set_ydata(y_up)
            lp.update()
            return r, {}, data
        elif plot:
            _, lines, _ = tp
            R = self.params['params']['R']
            L = self.params['params']['L']
            H = 3
            fig, ax = plt.subplots(figsize=(1.5*H, H))
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(-1.05 * 1.5 * R, 1.05 * 1.5 * R)
            ax.set_ylim(-1.05 * R, 1.05 * R)
            ax.plot([-L, L], [0, 0], lw=2, c='k')    # Bar
            x_up = x_sys[:, 0] - R * jnp.sin(x_sys[:, 1])
            y_up = R * jnp.cos(x_sys[:, 1])
            x_car = x_sys[:, 0]
            subidx = np.r_[0:len(u):(len(u)//10), len(u)-1]
            pendulum_lines = ()
            for i in subidx:
                pendulum_lines += ax.plot(
                    [x_car[i], x_up[i]], [0, y_up[i]], 'o-', lw=2, c='blue',
                    alpha=0.1 + 0.5*(i / (len(u) - 1)) + 0.4*(i == len(u)-1))[0],
            pendulum_lines += ax.plot(x_up, y_up, '.-', lw=1, ms=2, c='red')[0],
            lp = LivePlot(fig.canvas, pendulum_lines, .01)
            plt.show(block=False)
            plt.pause(.1)
            lpls = ((lp, pendulum_lines),)
            return r, (lpls, lines, 5), data
        else:
            return r, {}, data

    def render(self, u, rng, T=3):
        u_ = jnp.zeros((T * len(u), 1)).at[:len(u)].set(u)
        x, _ = self.model.rollout(self.params, u_)
        R = self.params['params']['R']
        L = self.params['params']['L']

        xn = min(x[:, 0].min(), (x[:, 0] - R * jnp.sin(x[:, 1])).min())
        xp = max(x[:, 0].max(), (x[:, 0] - R * jnp.sin(x[:, 1])).max())
        xn, xp = jnp.where(jnp.isnan(jnp.r_[xn, xp]), jnp.r_[-R, R], jnp.r_[xn, xp])
        H = 3
        if 1/4 < (aspect := (xp - xn) / (2 * R)) < 4:
            fig, ax = plt.subplots(figsize=(H * aspect, H))
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        elif aspect <= 1/4:
            fig, ax = plt.subplots(figsize=(H, H))
        else:
            fig, ax = plt.subplots(figsize=(H * 4, H))
        ax.set_xlim(xn - 0.05*(xp - xn), xp + 0.05*(xp - xn))
        ax.set_ylim(-1.05 * R, 1.05 * R)

        line, = ax.plot([], [], 'o-', lw=2, c='blue')
        trace, = ax.plot([], [], '.-', lw=1, ms=2, c='red')
        ax.plot([-L, L], [0, 0], lw=2, c='k')    # Bar
        time_text = ax.text(0.03, 0.97, '', transform=ax.transAxes, ha='left', va='top')
        forced_text = ax.text(0.03, 0.03, '', transform=ax.transAxes, ha='left', va='bottom')
        history_x, history_y = deque(maxlen=200), deque(maxlen=200)

        def animate(i):
            line.set_linestyle('-')
            forced_text.set_text("")
            if i >= len(u):
                line.set_linestyle(':')
                forced_text.set_text("unforced")
            if i >= len(u_):
                return line, trace, time_text, forced_text
            thisx = [x[i, 0], x[i, 0] - R * jnp.sin(x[i, 1])]
            thisy = [0, R * jnp.cos(x[i, 1])]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[1])
            history_y.appendleft(thisy[1])

            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            time_text.set_text(f'$t = {i*self.model.dt:.1f}$\n$u = {u_[i, 0]:.1f}$')
            return line, trace, time_text, forced_text

        _ = animation.FuncAnimation(
            fig, animate, (T + 1) * len(u), interval=self.model.dt * 1000, blit=True)
        plt.show()


class CartPoleModel(Model):
    g: float = 9.81
    x0: jnp.ndarray = field(default_factory=lambda: jnp.r_[0, jnp.pi, 0, 0])
    dt: float = 0.01
    res: int = 10
    real: bool = True
    sigma: float = 0
    params: dict = field(default_factory=lambda:
        {'m': 0.1, 'M': 1, 'R': 0.5, 'L': 1, 'kappa': 0.01, 'gamma': 0.01, 'force': 50})
    bound: float = 1e6
    contact: float = 0.  # 1e4

    def setup(self):
        def init_param(rng, key):
            rand = jnp.exp(jax.random.normal(rng) * self.sigma)
            rand = jax.lax.while_loop(lambda x: key == 'L' and x > 1, lambda x: x / 2, rand)     # must have L < real L
            param = rand * (self.params[key] if self.real else 1)
            return param

        self.m = self.param('m', partial(init_param, key='m'))
        self.M = self.param('M', partial(init_param, key='M'))
        self.R = self.param('R', partial(init_param, key='R'))
        self.L = self.param('L', partial(init_param, key='L'))
        self.kappa = self.param('kappa', partial(init_param, key='kappa'))
        self.gamma = self.param('gamma', partial(init_param, key='gamma'))
        self.force = self.param('force', partial(init_param, key='force'))

    def dz(self, x, u):
        dt = self.dt / self.res
        u = jnp.clip(u, -1, 1) * self.force
        df = x[2:]
        df = jnp.where(x[:2] + df*dt > self.bound, (self.bound - x[:2])/dt, df)
        df = jnp.where(x[:2] + df*dt < -self.bound, (-self.bound - x[:2])/dt, df)
        return df

    def dv(self, x, u):
        dt = self.dt / self.res
        u = jnp.clip(u, -1, 1) * self.force
        r, th, dr, dth = x
        m, M, R, L, g, kappa, gamma = self.m, self.M, self.R, self.L, self.g, self.kappa, self.gamma
        u -= (abs(r) > L) * (dr * r > 0) * (r - jnp.sign(r)*L) * self.res * self.contact    # Edge contact force
        df = jnp.r_[
            (-(m * R * dth**2 * jnp.sin(th)) + (m * g * jnp.sin(th) * jnp.cos(th)) - kappa*dr
                - (gamma * dth * jnp.cos(th) / R) + u) / (M + m - (m * jnp.cos(th)**2)),
            (-(m * R * dth**2 * jnp.sin(th) * jnp.cos(th)) + (M + m)*(g * jnp.sin(th)) - (kappa * dr * jnp.cos(th))
                - gamma*(M + m)*(dth / m / R) + (u * jnp.cos(th))) / (R * (M + m - (m * jnp.cos(th)**2)))
        ]
        df = jnp.where(x[2:] + df*dt > self.bound, (self.bound - x[2:])/dt, df)
        df = jnp.where(x[2:] + df*dt < -self.bound, (-self.bound - x[2:])/dt, df)
        return df

    def f(self, t, x, u):
        def step(_, x):
            x = x + jnp.r_[0, 0, self.dv(x, u)]*self.dt/self.res
            x = x + jnp.r_[self.dz(x, u), 0, 0]*self.dt/self.res
            return x
        return jax.lax.fori_loop(0, self.res, step, x)


class MLPModel(Model):
    layers: list = field(default_factory=lambda: [16, 16])

    def setup(self):
        self.net = nn.Sequential(sum(([nn.Dense(k), nn.gelu] for k in self.layers), []) + [nn.Dense(self.system.dx)])

    def f(self, t, x, u):
        return x + self.net(jnp.r_[x, u])


class GymSystem(System):
    def __init__(self, env, copy_x=False, **kwargs):
        self.gen_env = lambda render_mode=None: gym.make(env, render_mode=render_mode, **kwargs)
        self.env = self.gen_env()
        self.copy_x = copy_x
        self.x0 = self._reset()
        self.dx = len(self.x0)
        self.du = self.env.action_space.shape[0]

    @partial(jax.jit, static_argnums=(0,))
    def r_health(self, z, lo, hi):
        s = 10  # slope
        return jax.lax.cond(
            z < lo, lambda: s * (z - lo), lambda: jax.lax.cond(z > hi, lambda: -s * (z - hi), lambda: 0.))

    def r(self, t, x, u):
        return 0.

    def rT(self, x):
        return 0.

    def _reset(self):
        return self.reset(np.random.default_rng(0))

    def reset(self, rng):
        self.env.np_random = rng
        x0 = self.env.reset()[0]
        return np.r_[x0[0], x0] if self.copy_x else x0

    def f(self, t, x, u, rng):
        self.env.np_random = rng
        x_ = self.env.step(u)[0]
        return np.r_[x[1], x_] if self.copy_x else x_

    def render(self, u, rng):
        env = self.gen_env(render_mode='human')
        env.np_random = rng
        while True:
            env.reset()
            for ut in u:
                env.step(ut)


class Ant(GymSystem):
    def __init__(self, noise=0, healthy=True, discrete_health=False, euthanasia=False, orig_r=False, **kwargs):
        super().__init__('Ant-v4', copy_x=True, reset_noise_scale=noise, terminate_when_unhealthy=False,
            exclude_current_positions_from_observation=False, **kwargs)
        self.healthy = healthy
        self.discrete_health = discrete_health
        self.euthanasia = euthanasia
        self.orig_r = orig_r
        self.rem_dx = 3  # First 3 dimensions should be removed from observation

    def health(self, t, x, u, val=False):
        cos = R.from_quat(jnp.concatenate([x[5:8], x[4:5]], -1)).as_matrix() @ jnp.r_[0, 0, 1] @ jnp.r_[0, 0, 1]
        health_z = self.r_health(x[3], 0.2, 1)
        health_cos = self.r_health(cos, np.cos(np.pi / 4), 1)
        health = health_z + health_cos
        if self.orig_r:
            return (health_z == 0).astype(float)
        return (health == 0).astype(float) if self.discrete_health and not val else health

    def r(self, t, x, u, val=False):
        dt = self.env.unwrapped.dt
        return (
            (x[1] - x[0]) / dt             # velocity
            - 0.5 * jnp.linalg.norm(u)**2  # control cost
            + (self.health(t, x, u, val) if self.healthy else 0.)    # health
        )

    def stats(self, u, rng, g_approx=None, x_mod=None, A=None, B=None):
        x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data = super().stats(
            u, rng, g_approx, x_mod, A, B)
        labels['x_sys'] = r'$x$-position'
        x_sys = data['x_sys'][:, 1]
        return x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data


class Cheetah(GymSystem):
    def __init__(self, noise=0, healthy=True, discrete_health=False, euthanasia=False, orig_r=False, **kwargs):
        super().__init__('HalfCheetah-v4', copy_x=True, reset_noise_scale=noise,
            exclude_current_positions_from_observation=False, **kwargs)
        self.healthy = healthy
        self.discrete_health = discrete_health
        self.euthanasia = euthanasia
        self.orig_r = orig_r
        self.rem_dx = 2

    def health(self, t, x, u, val=False):
        health = self.r_health(x[3], -np.pi / 4, np.pi / 4)
        if self.orig_r:
            return 0.
        return (health == 0).astype(float) if self.discrete_health and not val else health

    def r(self, t, x, u, val=False):
        dt = self.env.unwrapped.dt
        return (
            (x[1] - x[0]) / dt             # velocity
            - 0.1 * jnp.linalg.norm(u)**2  # control cost
            + (self.health(t, x, u, val) if self.healthy else 0.)    # health
        )

    def stats(self, u, rng, g_approx=None, x_mod=None, A=None, B=None):
        x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data = super().stats(
            u, rng, g_approx, x_mod, A, B)
        labels['x_sys'] = r'$x$-position'
        x_sys = data['x_sys'][:, 1]
        return x_sys, x_mod, u, r, g_true, g_approx, A_error, B_error, labels, data


class GymEnv(gym.Env):
    def __init__(self, system, rng, horizon):
        self.system = system
        self.rng = rng
        self.np_random = rng
        self.horizon = horizon
        bound = 1  # Correct for both Ant and Cheetah
        self.action_space = gym.spaces.Box(-bound, bound, (self.system.du,))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.system.dx - self.system.rem_dx,))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.x = self.system.reset(self.rng)
        return self.x[self.system.rem_dx:], {}

    def step(self, action, val=False):
        reward = self.system.r(self.t, self.x, action, val=val)
        self.x = self.system.f(self.t, self.x, action, self.rng)
        self.t += 1
        truncated = self.t >= self.horizon
        terminated = self.system.euthanasia and self.system.health(self.t, self.x, action) < 0
        return self.x[self.system.rem_dx:], reward, terminated, truncated, {}

    def render(self):
        raise NotImplementedError

    def rollout_policy(self, policy, val=True):
        observations = []
        rewards = []
        actions = []
        done = False
        obs, _ = self.reset()
        observations.append(obs)
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            actions.append(action)
            obs, r, _, done, _ = self.step(action, val=val)
            observations.append(obs)
            rewards.append(r)
        return np.stack(observations), np.stack(rewards), np.stack(actions)

    def render_policy(self, policy):
        self.system.env = self.system.gen_env(render_mode='human')
        try:
            while True:
                obs, _ = self.reset()
                done = False
                ret = 0
                while not done:
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, r, _, done, _ = self.step(action)
                    ret += r
                print(f'Return: {ret:.2f}')
        except:
            self.system.env = self.system.gen_env()
            raise
