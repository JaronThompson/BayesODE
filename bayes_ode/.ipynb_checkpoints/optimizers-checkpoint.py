import numpy as np
from .convergence import check_convergence
from .linalg import *
from tqdm import tqdm 

# ADAM optimizer 
def fit_ADAM(self, 
             n_sample: int = 3, 
             lr: float = 1e-3, 
             beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, 
             max_epochs: int = 100000, tol: float = 1e-3, patience: int = 3) -> list[float]:
    
    """
    ADAM optimizer for minimizing a function.

    Parameters:
    - n_sample: Number of samples to draw from approximate posterior
    - lr (learning_rate): Step size for the optimization.
    - beta1: Exponential decay rate for the first moment estimate.
    - beta2: Exponential decay rate for the second moment estimate.
    - epsilon: Small constant to prevent division by zero.
    - max_epochs: Maximum number of iterations over full data set.
    - tol: Tolerance to stop optimization when the change in parameters is below this value.
    - patience: Maximum number of times to allow mis-step in objective 

    """
    m = np.zeros_like(self.lmbda)
    v = np.zeros_like(self.lmbda)
    t = 0
    epoch = 0
    passes = 0
    fails = 0

    # order of samples
    N = len(self.X)
    order = np.arange(N)

    # save best parameters
    best_params = np.copy(self.lmbda)

    # initialize function evaluations
    f = []

    while epoch <= max_epochs and passes < patience:

        if epoch % 5 == 0:

            # check convergence
            f.append(approx_evidence(self))
            convergence = (f[-1] - np.mean(f[-10:])) / np.abs(np.mean(f[-10:]))

            # determine slope of elbo over time
            if len(f) > 2:
                slope = check_convergence(f[-10:])
            else:
                slope = 1.

            # check tolerance
            if abs(slope) < tol:
                passes += 1
                print(f"pass {passes}")
            else:
                passes = 0

            # save parameters if improved
            if f[-1] >= np.max(f):
                best_params = np.copy(self.lmbda)

            # if slope is negative, add to fail count
            if slope < 0:
                fails += 1
                print(f"fail {fails}")
            else:
                fails = 0

            # if fails exceeds patience, return best parameters
            if fails == patience:
                self.lmbda = jnp.array(best_params)
                self.z = self.lmbda.at[:self.d].get()
                return f

            print("Epoch {:.0f}, ELBO: {:.3f}, Slope: {:.3f}, Convergence: {:.5f}".format(epoch, f[-1], slope, convergence))
        epoch += 1

        # update at each sample
        np.random.shuffle(order)
        for sample_index in order:

            # gradient of entropy of approximate distribution w.r.t log_s
            gradient = np.append(np.zeros(self.d), -np.ones(self.d)) / N

            # sample parameters
            y = np.random.randn(n_sample, self.d)

            # gradient of negative log posterior
            for yi in y:

                # prior
                gradient += grad_neg_log_prior_lmbda(self.prior_mean, yi, self.alpha, self.lmbda) / N / n_sample

                # gradient of negative log likelihood
                grad_nll = np.nan_to_num(grad_neg_log_likelihood_lmbda(self.system,
                                                                        yi,
                                                                        self.T[sample_index],
                                                                        self.X[sample_index],
                                                                        self.U[sample_index],
                                                                        self.nu2, self.sigma2,
                                                                        self.lmbda)) / n_sample

                # ignore exploding gradients
                gradient += np.where(np.abs(grad_nll) < 1000, grad_nll, 0.)

            # moment estimation
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # adjust moments based on number of iterations
            t += 1
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # take step
            self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)  # / np.sqrt(t)
            self.z = self.lmbda.at[:self.d].get()

    return f

# EM algorithm to update hyperparameters
def update_hypers(self, n_sample: int = 100):

    # create dictionaries of estimated/empirical moments for each output
    Z = {}
    Y2 = {}
    for j, var in enumerate(self.sys_vars):
        Z[j] = []
        Y2[j] = []

    # approximate expected error
    y = np.random.randn(n_sample, self.d)
    z = batch_T(y, self.lmbda)
    for zi in tqdm(z):

        # loop over each sample in dataset
        for tf, x, u in zip(self.T, self.X, self.U):

            # integrate ODE
            t_hat = ode_model(self.system, tf, x, u, zi)

            # make sure predictions aren't NaN or inf
            if not (np.any(np.isnan(t_hat)) or np.any(np.isinf(t_hat))):

                # clip negatives to zero
                t_hat = jnp.clip(t_hat, 0., np.inf)

                # Determine error: set to zero if either the data is NaN or the model prediction is NaN
                y_error = np.nan_to_num(x[-1] - t_hat)

                # estimate noise
                for j, (y_j, f_j, e_j) in enumerate(zip(x[-1], t_hat, y_error)):
                    if y_j > 0:
                        Y2[j].append(y_j ** 2)
                        Z[j].append(e_j ** 2)

    # solve for noise parameters
    for j, var in enumerate(self.sys_vars):
        y2 = np.array(Y2[j])
        z = np.array(Z[j])
        B = np.vstack((np.ones_like(y2), y2)).T
        a = (np.linalg.inv(B.T @ B) @ B.T) @ z
        self.nu2[j] = np.max([a[0], 1e-4])
        self.sigma2[j] = np.max([a[1], 1e-4])

    # update alpha
    var = jnp.exp(self.lmbda.at[self.d:].get()) ** 2
    self.alpha = 1. / ((self.z - self.prior_mean) ** 2 + var + 1e-4)