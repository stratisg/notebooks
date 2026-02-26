import matplotlib.pyplot as plt
import pyro
import torch
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import MCMC, NUTS
from pyro.optim import Adam


def generate_data(mean, std, n_measurements):
    """
    True generating process.
    """

    return Normal(mean, std).sample((n_measurements,))


def model_normal(mu_mean, std_mean, std_likelihood, measurements):
    """
    Prior distribution of the mean.
    """
    mean = pyro.sample("mean", Normal(mu_mean, std_mean))
    n_samples = len(measurements)
    with pyro.plate("measurements", n_samples):
        pyro.sample("y", Normal(mean, std_likelihood),
                    obs=measurements)
    
    return mean


def sampling(model, sampler_args, model_args):
    """
    MCMC sampling.
    """
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, **sampler_args)
    mcmc.run(**model_args)
    mcmc.summary()

    return mcmc.get_samples()
    


def variational_inference(model, guide, model_args, optim_args, n_steps=1000,
                          print_epoch=100):
    """
    Variational inference loop.
    """
    pyro.clear_param_store()
    optimizer = Adam(optim_args)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    l_loss = torch.zeros(n_steps)
    for i_step in range(n_steps):
        l_loss[i_step] = svi.step(**model_args)
        if i_step % print_epoch == 0:
            print(79 * "=")
            print(f"Epoch: {i_step}\tLoss: {l_loss[i_step]:.3E}")
    # Print last loss.
    print(79 * "=")
    print(f"Epoch: {i_step}\tLoss: {l_loss[i_step]:.3E}")

    # Visualize the loss.
    l_epochs = torch.arange(n_steps)
    plt.figure("Loss")
    if l_loss.any() <= 0:
        plt.plot(l_epochs, l_loss)
    else:
        plt.semilogy(l_epochs, l_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_markov.png", dpi=600)