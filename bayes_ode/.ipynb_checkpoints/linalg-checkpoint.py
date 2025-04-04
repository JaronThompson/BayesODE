from functools import partial

import numpy.typing as npt
import jax.numpy as jnp
from jax import random, jit, jacfwd, jacrev, grad, vmap
from jax.numpy.linalg import inv
from jax.nn import tanh, sigmoid, relu
from jax.experimental.ode import odeint


# define ode_model that takes as input the initial condition the latent variables
@partial(jit, static_argnums=(0,))
def ode_model(system: callable, 
              tf: float, 
              x: jnp.ndarray, 
              u: jnp.ndarray, 
              z: jnp.ndarray) -> jnp.ndarray:
    
    # unpack data and integration time
    t_span = jnp.linspace(0., tf, 10)

    # integrate ODE
    y_hat = odeint(system, jnp.array(x[0]), t_span, u, z,
                   rtol=1.4e-8, atol=1.4e-8, mxstep=10000, hmax=jnp.inf)

    # y_hat is the ode_model estimate of observed variable y
    return y_hat[-1]

# invertible, differentiable function to map noise to ode_model parameters
@jit
def T(y: jnp.ndarray, lmbda: jnp.ndarray) -> jnp.ndarray:
    # mean and standard deviation
    mu, log_s = lmbda.at[:len(lmbda) // 2].get(), lmbda.at[len(lmbda) // 2:].get()

    # convert to z
    z = mu + jnp.exp(log_s) * y

    return z

# batch sampling of ode model parameters
@jit
def batch_T(y_batch: jnp.ndarray, lmbda: jnp.ndarray) -> jnp.ndarray:
    return vmap(T, (0, None))(y_batch, lmbda)

# log of absolute value of determinant of Jacobian of T w.r.t. y
@jit
def log_abs_det(lmbda: jnp.ndarray) -> jnp.ndarray:
    log_s = lmbda.at[(len(lmbda) // 2):].get()
    return jnp.sum(log_s)

# evaluate negative log prior
@jit
def neg_log_prior(z_prior: jnp.ndarray, z: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp

# gradient of negative log prior
@jit
def grad_neg_log_prior(z_prior: jnp.ndarray, z: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    return jacrev(neg_log_prior, 1)(z_prior, z, alpha)


# evaluate negative log likelihood
@partial(jit, static_argnums=(0,))
def neg_log_likelihood(system: callable, 
                       z: jnp.ndarray, 
                       tf: float, 
                       x: jnp.ndarray, 
                       u: jnp.ndarray, 
                       nu2: jnp.ndarray, 
                       sigma2: jnp.ndarray) -> jnp.ndarray:

    # ode_model prediction
    y = ode_model(system, tf, x, u, z)

    # residuals
    res = jnp.nan_to_num(x[-1] - y)

    # predicted variance
    var = nu2 + sigma2 * jnp.nan_to_num(y) ** 2

    # likelihood
    lp = jnp.sum(res ** 2 / var / 2. + jnp.log(var) / 2.)

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_neg_log_likelihood(system: callable, 
                            z: jnp.ndarray, 
                            tf: float, 
                            x: jnp.ndarray, 
                            u: jnp.ndarray, 
                            nu2: jnp.ndarray, 
                            sigma2: jnp.ndarray) -> jnp.ndarray:
    return jacrev(neg_log_likelihood, 1)(system, z, tf, x, u, nu2, sigma2)


# evaluate negative log prior
@jit
def neg_log_prior_lmbda(z_prior: jnp.ndarray, 
                        y: jnp.ndarray, 
                        alpha: jnp.ndarray, 
                        lmbda: jnp.ndarray) -> jnp.ndarray:

    # sample z
    z = T(y, lmbda)

    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of negative log prior
@jit
def grad_neg_log_prior_lmbda(z_prior: jnp.ndarray, 
                             y: jnp.ndarray, 
                             alpha: jnp.ndarray, 
                             lmbda: jnp.ndarray) -> jnp.ndarray:
    return jacrev(neg_log_prior_lmbda, -1)(z_prior, y, alpha, lmbda)


# evaluate negative log likelihood
@partial(jit, static_argnums=(0,))
def neg_log_likelihood_lmbda(system: callable, 
                             y: jnp.ndarray, 
                             tf: float, 
                             x: jnp.ndarray, 
                             u: jnp.ndarray, 
                             nu2: jnp.ndarray, 
                             sigma2: jnp.ndarray, 
                             lmbda: jnp.ndarray) -> jnp.ndarray:

    # sample z
    z = T(y, lmbda)

    # likelihood
    lp = neg_log_likelihood(system, z, tf, x, u, nu2, sigma2)

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_neg_log_likelihood_lmbda(system: callable, 
                                  y: jnp.ndarray, 
                                  tf: float, 
                                  x: jnp.ndarray, 
                                  u: jnp.ndarray, 
                                  nu2: jnp.ndarray, 
                                  sigma2: jnp.ndarray, 
                                  lmbda: jnp.ndarray) -> jnp.ndarray:
    return jacrev(neg_log_likelihood_lmbda, -1)(system, y, tf, x, u, nu2, sigma2, lmbda)

# negative log posterior
def nlp(self, z: jnp.ndarray) -> jnp.ndarray:

    # prior
    self.NLP = neg_log_prior(self.prior_mean, z, self.alpha)

    # likelihood
    for tf, x, u in zip(self.T, self.X, self.U):
        self.NLP += neg_log_likelihood(self.system, z, tf, x, u, self.nu2, self.sigma2)

    # return NLP
    return self.NLP

# approximation to evidence (log probability of the data) 
def approx_evidence(self) -> jnp.ndarray:

    # posterior entropy
    self.log_evidence = log_abs_det(self.lmbda) - nlp(self, self.z)

    return self.log_evidence