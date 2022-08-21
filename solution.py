import sys
from dac4automlcomp.policy import DACPolicy, DeterministicPolicy
from examples.ac_for_dac.schedulers import Configurable, Serializable

LR = .005
RF = .05


class CustomLRPolicy(Configurable, Serializable, DeterministicPolicy, DACPolicy):

    def __init__(self):
        pass

    def act(self, state):
        # The base learning rate is given by applying a linear decay
        base_lr = LR * (1 - state['step'] / self.cutoff)

        # If the validation loss increases, further reduce the learning rate
        if state['validation_loss'] is not None:
            val_loss = state['validation_loss'].detach().mean().cpu().item()
            if self.last_val_loss < val_loss:
                self.reduction_factor *= RF
            self.last_val_loss = val_loss

        return base_lr * self.reduction_factor

    def reset(self, instance):
        self.reduction_factor = 1.
        self.last_val_loss = sys.float_info.max
        self.cutoff = instance.cutoff


def load_solution(path):
    return CustomLRPolicy()
