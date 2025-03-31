from .data_processing import *
from .optimizers import *

import numpy.typing as npt
import jax.numpy as jnp
from .linalg import *

class Model:
    """
    Model class that takes an ODE model function and training data.

    system is the ode where system(x: jnp.ndarray, t: float, inputs: jnp.ndarray, params: jnp.ndarray)
    dataframe contains data with columns ['Treatments', 'Time', 'x_1', ..., 'u_1', ...]
    sys_vars is a list of system variable names 'x_1', ...
    inputs is a list of system input names 'u_1', ...
    params is an array of initial parameter guess 
    prior is an array of same length as params that defines prior mean for each parameter (typically zero)
    alpha is the precision of the prior (larger = more regularization)
    nu2 is the variance of constant Gaussian noise (larger = more regularization)
    sigma2 is the percent variance of Gaussian noise (larger = more regularization)
    """

    def __init__(self,
                 system: callable,
                 dataframe: pd.DataFrame,
                 sys_vars: list[str],
                 inputs: list[str], 
                 params: npt.NDArray | jnp.ndarray,
                 prior: npt.NDArray | jnp.ndarray,
                 alpha: npt.NDArray | jnp.ndarray | float = 1.0,
                 nu2: npt.NDArray | jnp.ndarray | float = 0.01,
                 sigma2: npt.NDArray | jnp.ndarray | float = 0.01):

        # System of differential equations
        self.system: callable = system

        # Processed data
        self.sys_vars: list[str] = sys_vars
        self.T, self.X, self.U = process_df(dataframe, sys_vars, inputs)

        # Scale data based on max measured value
        self.X_scale = 1.  #  float = np.max(self.X, axis=0)[-1] if self.X.size else 1.0
        self.X /= self.X_scale

        # Parameter prior
        self.prior_mean: npt.NDArray = prior

        # Problem dimension
        self.d: int = len(params)

        # Prior and measurement precision
        self.alpha: npt.NDArray = (np.full(self.d, alpha) if isinstance(alpha, (float, int)) 
                                   else np.asarray(alpha))
        
        self.nu2: npt.NDArray = (np.full(len(sys_vars), nu2) if isinstance(nu2, (float, int)) 
                                 else np.asarray(nu2))

        self.sigma2: npt.NDArray = (np.full(len(sys_vars), sigma2) if isinstance(sigma2, (float, int)) 
                                    else np.asarray(sigma2))

        # Initial parameter guess
        self.z: jnp.ndarray = jnp.array(params)
        self.lmbda: jnp.ndarray = jnp.append(self.z, jnp.log(jnp.ones(self.d) / 10.))

    def fit_posterior_EM(self, 
                         n_sample_sgd: int=1, 
                         n_sample_hypers: int=100, 
                         patience: int=3, 
                         lr: float=1e-3, 
                         max_iterations: int=100):

        # optimize parameter posterior
        print("Updating posterior...")
        f = fit_ADAM(self, n_sample_sgd, lr=lr)

        # init evidence, fail count, iteration count
        previdence = np.copy(f[-1])
        fails = 0
        t = 0
        while fails < patience and t < max_iterations:

            # update iteration count
            t += 1

            # update prior and measurement precision estimate
            print("Updating hyperparameters...")
            update_hypers(self, n_sample = n_sample_hypers)

            # optimize parameter posterior
            print("Updating posterior...")
            f = fit_ADAM(self, n_sample_sgd, lr=lr)

            # check convergence
            if self.log_evidence <= previdence:
                fails += 1
            previdence = np.copy(self.log_evidence)

    def predict_point(self, x0, u, t_eval):

        # mean of posterior
        z = T(np.zeros(self.d), self.lmbda)

        return self.X_scale * odeint(self.system, x0 / self.X_scale, t_eval, u, z)
        
    def predict_sample(self, x0, u, t_eval, n_sample=21):

        # sample noise
        y = np.random.randn(n_sample, self.d)

        # posterior predictive
        predictions = []
        for yi in y:

            # sample from posterior 
            zi = T(yi, self.lmbda)
            
            # sample prediction
            prediction = odeint(self.system, x0 / self.X_scale, t_eval, u, zi)
            predictions.append(self.X_scale * prediction)

        return np.stack(predictions)

    def predict_prior(self, x0, u, t_eval, n_sample=21):

        # sample from prior
        y = np.random.randn(n_sample, self.d)
        z = self.prior_mean + np.sqrt(1. / self.alpha) * y

        # sample from prior predictive
        predictions = []
        for zi in z:
            predictions.append(self.X_scale * odeint(self.system, x0 / self.X_scale, t_eval, u, zi))

        return np.stack(predictions)

    def param_df(self,):

        # mean and standard deviation of posterior
        mean = np.array(self.lmbda[:self.d])
        stdv = np.array(np.exp(self.lmbda[self.d:]))

        # save parameter dataframe
        df_param = pd.DataFrame()
        df_param['mean'] = mean
        df_param['stdv'] = stdv

        return df_param