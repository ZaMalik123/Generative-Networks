import torch
import utils
import losses
import networks
import itertools
from collections import OrderedDict
from base_model import Base

class W1(Base):
    """Wasserstein-1 based models including WGAN-GP/WGAN-LP"""

    def get_data(self, config):
        """override z with gz"""
        z = utils.to_var(next(self.z_generator)[0])
        gz = self.g(z)
        r = utils.to_var(next(self.r_generator)[0])
        if gz.size() != r.size():
            z = utils.to_var(next(self.z_generator)[0])
            gz = self.g(z)
            r = utils.to_var(next(self.r_generator)[0])
        # ZMALIK 20240404 Include previous sample, if needed/available
        if hasattr(self, 'g_min1'):
          g_min1_z = self.g_min1(z)
        else:
          g_min1_z = []
        return r, gz, g_min1_z, z # 20240404 ZMALIK: Unpack 4 values, not 2.

    def define_d(self, config):
        self.phi = networks.get_d(config)
        self.d_optimizer = networks.get_optim(self.phi.parameters(),config.d_lr, config)

    def psi(self, y):
        return -self.phi(y)

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        d_loss = -torch.mean(ux + vy)
        d_loss += losses.gp_loss(x, y, self.phi, config.lambda_gp, clamp=config.clamp)
        return d_loss

    # 20240426 ZMALIK: New option for generator update
    def calc_gloss(self, x, y, y1, ux, vy, config):
       if config.follow_ode: # Explicitly follow ODE and do MSE fitting
          gloss = torch.nn.MSELoss()
          return gloss(y,y1); 
       else:
          return torch.mean(vy) #Original update rule found in Leygonie et al

    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        stats['loss/disc'] = self.d_loss
        stats['loss/gen'] = self.g_loss
        stats['w1_loss'] = self.wp_loss
        return stats

    def get_networks(self):
        nets = OrderedDict([('phi', self.phi)])
        nets['gen'] = self.g
        return nets
