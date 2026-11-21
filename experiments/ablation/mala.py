"""
Metropolis-adjusted Langevin algorithm (MALA) with adaptive step size.

targets p(x|y) = p(y|x) p(x). The gradient of the posterior is supplied by
the caller (likelihood gradient + prior gradient), so the same sampler runs
for both the reference forward model and the frequency-attenuated surrogate.
"""

import numpy as np


def make_posterior(likelihood_grad, prior):
    """Return a callable log_prob_grad(x) -> (log_p, grad) for the posterior."""
    def f(x):
        g_ll, ll = likelihood_grad(x)
        lp = prior.log_p(x)
        gp = prior.grad(x)
        return ll + lp, g_ll + gp
    return f


class MALA:
    def __init__(self, log_prob_grad, x0, step_size=0.05,
                 target_accept=(0.5, 0.8), adapt_every=50, seed=0):
        self.f = log_prob_grad
        self.x = np.asarray(x0, dtype=float).ravel().copy()
        self.eps = step_size
        self.target = target_accept
        self.adapt_every = adapt_every
        self.rng = np.random.default_rng(seed)
        self.lp, self.g = self.f(self.x)

    def step(self):
        """One MALA step. Returns (accepted: bool, x: current state)."""
        n = len(self.x)
        eps = self.eps
        # proposal
        eta = self.rng.standard_normal(n)
        x_star = self.x + 0.5 * eps ** 2 * self.g + eps * eta

        lp_star, g_star = self.f(x_star)

        # log acceptance (Metropolis with asymmetric Langevin proposal)
        # log q(x*|x) = -||x* - x - (eps^2/2) g||^2 / (2 eps^2)
        # log q(x|x*) = -||x - x* - (eps^2/2) g*||^2 / (2 eps^2)
        mu_fwd = self.x + 0.5 * eps ** 2 * self.g
        mu_bwd = x_star + 0.5 * eps ** 2 * g_star
        logq_fwd = -0.5 * np.sum((x_star - mu_fwd) ** 2) / eps ** 2
        logq_bwd = -0.5 * np.sum((self.x - mu_bwd) ** 2) / eps ** 2
        log_alpha = (lp_star - self.lp) + (logq_bwd - logq_fwd)

        accepted = False
        if np.log(self.rng.random()) < log_alpha:
            self.x = x_star
            self.lp = lp_star
            self.g = g_star
            accepted = True
        return accepted, self.x.copy()

    def run(self, n_burnin, n_iter, thin=1, progress=False):
        """Run MALA. Returns (samples, accept_rate, accept_history)."""
        accepts = []
        for it in range(n_burnin):
            a, _ = self.step()
            accepts.append(a)
            # adaptive step size during burn-in
            if (it + 1) % self.adapt_every == 0:
                acc = np.mean(accepts[-self.adapt_every:])
                if acc < self.target[0]:
                    self.eps *= 0.9
                elif acc > self.target[1]:
                    self.eps *= 1.1

        samples = []
        for it in range(n_iter):
            a, x = self.step()
            accepts.append(a)
            if (it + 1) % thin == 0:
                samples.append(x)
        samples = np.array(samples)
        return samples, float(np.mean(accepts)), np.array(accepts)
