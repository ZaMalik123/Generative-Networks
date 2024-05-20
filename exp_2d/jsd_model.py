import torch
import utils
import losses
import networks
from collections import OrderedDict
from torch.autograd import Variable
from base_model import Base

"20230526: ZMALIK, JSD model"

class JSD(Base):
    "JSD based model"
    def get_data(self, config):
        """override z with gz in the case gen=T"""
        z = utils.to_var(next(self.z_generator))
        gz = self.g(z) if config.gen else z
        r = utils.to_var(next(self.r_generator))
        return r, gz

    def define_d(self, config):
        """Discriminator for JSD model"""
        self.phi = networks.get_d(config)
        self.d_optimizer = networks.get_optim(self.phi.parameters(), config.d_lr, config)

    def psi(self, y):
        return self.phi(y)

    def get_dux(self, config, reverse=False):
        """Compute derivative of the discriminator"""
        x = Variable(x.data, requires_grad=True)
        if reverse:
            ux = self.psi(x)
        else:
            ux = self.phi(x)
        dux = torch.autograd.grad(outputs=ux, inputs=x,
                                  grad_outputs=utils.get_ones(ux.size()),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        return dux

    def follow_ode(self, x, y, ux, vy, config):
        """Generate a sample by following the governing ODE"""
        blank = []
        return blank

    
    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        """Compute the discriminator loss"""
        # ux = D(x), the discriminator evaluated on samples from the target
        # vy = 1 - D(y), used for training
        dloss = torch.nn.BCELoss()
        return dloss(ux, torch.ones_like(ux)) + dloss(vy, torch.zeros_like(vy))

    def calc_gloss(self, x, y, ux, vy, config):
        """Computes the generator loss"""
        if config.follow_ode: # Explicitly follow ODE and do MSE fitting
          yn1 = self.follow_ode(x, y, ux, vy)
          if config.shuffle:
            yn1 = utils.shuffle(yn1, y)
          yn1 = yn1.detach()
          gloss = torch.nn.MSELoss()
          return gloss(y,yn1); 
        else:
          gloss = torch.nn.BCELoss()
          return gloss(vy, torch.ones_like(vy)) #Original update rule from Goodfellow et al

    ## Model statistics
    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        stats['loss/disc'] = self.d_loss
        if config.gen:
            stats['loss/gen'] = self.g_loss
        return stats

    def get_networks(self, config):
        nets = OrderedDict([('phi', self.phi)])
        if config.gen:
            nets['gen'] = self.g
        return nets