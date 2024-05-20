import torch
import utils
import losses
import networks
import itertools
from collections import OrderedDict
from base_model import Base

class W2(Base):
    """Wasserstein-2 based model W2GAN"""

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
        self.phi, self.eps = networks.get_d(config), networks.get_d(config)
        self.d_optimizer = networks.get_optim(itertools.chain(list(self.phi.parameters()),
                                                              list(self.eps.parameters())),
                                                              config.d_lr, config)

    def psi(self, y):
        return -self.phi(y) + self.eps(y)

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        d_loss = -torch.mean(ux + vy)
        if config.ineq:
            d_loss += losses.ineq_loss(x, y, ux, vy, self.cost, config.lambda_ineq)
        if config.ineq_interp:
            d_loss += losses.calc_interp_ineq(x, y, self.phi, self.psi, self.cost, config.lambda_ineq, losses.ineq_loss)
        if config.eq_phi:
            d_loss += losses.calc_eq(x, tx, self.phi, self.psi, self.cost, config.lambda_eq)
        if config.eq_psi:
            d_loss += losses.calc_eq(ty, y, self.phi, self.psi, self.cost, config.lambda_eq)
        if config.lambda_eps > 0.0:
            d_loss += config.lambda_eps * torch.mean((torch.clamp(self.psi(y), min=0))**2)
        return d_loss

    # 20240404 ZMALIK: Commented out old calc_gloss code
    #def calc_gloss(self, x, y, ux, vy, config):
    #    return torch.mean(vy)

    # 20240404 ZMALIK: Explicitly compute MSE loss
    def calc_gloss(self, x, y, y1, ux, vy, config):
        """Computes generator loss by either original update rule or MSE fitting.
        No shuffling implemented for high dimensional experiments."""
        if config.follow_ode: # Explicitly follow ODE and do MSE fitting
          gloss = torch.nn.MSELoss()
          return gloss(y,y1); 
        else:
          return torch.mean(vy) #Original update rule found in Leygonie et al

    def calc_w2loss(self, ux, vy, config):
        """
        Compute W2 distance between generated sample and target sample, using 
        the KP computed from the discriminator.
        """
        return torch.mean(ux+vy)
    
    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        stats['loss/disc'] = self.d_loss
        stats['loss/gen'] = self.g_loss
        stats['w2_loss'] = self.wp_loss
        return stats

    def get_networks(self):
        nets = OrderedDict([('phi', self.phi),
                            ('eps', self.eps)])
        nets['gen'] = self.g
        return nets
