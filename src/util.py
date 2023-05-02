import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

import gpytorch


def exact_mll(predicted_distribution, target, gp):
    """Calculate negative marginal log-likelihood of exact model."""
    data_size = target.shape[-1]
    loss = -predicted_distribution.log_prob(target).sum()

    if isinstance(gp, nn.ModuleList):
        for gp_ in gp:
            # Add additional terms (SGPR / learned inducing points,
            # heteroskedastic likelihood models)
            for added_loss_term in gp_.added_loss_terms():
                loss += added_loss_term.loss()

            # Add log probs of priors on the (functions of) parameters
            for _, prior, closure, _ in gp_.named_priors():
                loss += prior.log_prob(closure()).sum()
    else:
        for added_loss_term in gp.added_loss_terms():
            loss += added_loss_term.loss()

        # Add log probs of priors on the (functions of) parameters
        for _, prior, closure, _ in gp.named_priors():
            loss += prior.log_prob(closure()).sum()

    return loss / data_size


class Delta(gpytorch.distributions.Delta):
    """Delta Distribution."""

    def __init__(self, validate_args=False, *args, **kwargs):
        super().__init__(validate_args=validate_args, *args, **kwargs)

    def __str__(self):
        """Get string of Delta distribution."""
        return f"Delta loc: {self.v}"

    def entropy(self):
        """Return entropy of distribution."""
        return torch.zeros(self.batch_shape)
    

def tensor_to_distribution(args, **kwargs):

    if not isinstance(args, tuple):
        return Categorical(logits=args)
    elif torch.all(args[1] == 0):
        if kwargs.get("add_noise", False):
            noise_clip = kwargs.get("noise_clip", np.inf)
            policy_noise = kwargs.get("policy_noise", 1)
            try:
                policy_noise = policy_noise()
            except TypeError:
                pass
            mean = args[0] + (torch.randn_like(args[0]) * policy_noise).clamp(
                -noise_clip, noise_clip
            )
        else:
            mean = args[0]
        return Delta(v=mean, event_dim=min(1, mean.dim()))
    else:
        if kwargs.get("tanh", False):
            d = TransformedDistribution(
                MultivariateNormal(args[0], scale_tril=args[1]), [TanhTransform()]
            )
        else:
            d = MultivariateNormal(args[0], scale_tril=args[1])
        return d


def add_data_to_gp(gp_model, new_inputs, new_targets):
    """Add new data points to an existing GP model.
    Once available, gp_model.get_fantasy_model should be preferred over this.
    """
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    targets = torch.cat((gp_model.train_targets, new_targets), dim=-1)
    gp_model.set_train_data(inputs, targets, strict=False)


def add_input_to_gp(gp_model, new_inputs):
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    gp_model.set_train_data(inputs, strict=False)