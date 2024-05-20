import torch
import utils
import functools
import losses
import networks
from torch.autograd import Variable
from collections import OrderedDict


class Base(object):
    def __init__(self, config, r_loader, z_loader):
        self.config = config
        self.cost = functools.partial(losses.cost, l=config.l, p=config.p) # Obtain function that behaves like cost with given parameters
        self.z_generator = z_loader
        self.r_generator = r_loader
        self.fixed_z = utils.to_var(next(self.z_generator))
        self.fixed_r = utils.to_var(next(self.r_generator))
        self.delta_t = config.delta_t
        self.define_model(config)

    def get_fixed_data(self):
        return self.fixed_r, self.fixed_z

    def get_data(self, config):
        z = utils.to_var(next(self.z_generator))
        r = utils.to_var(next(self.r_generator))
        return r, z

    def get_cost(self):
        return self.cost

    def define_model(self, config):
        self.define_d(config)
        if config.gen:
            self.define_g(config)

    def define_d(self, config):
        raise NotImplementedError("Please Implement this method")

    def define_g(self, config):
        self.g = networks.get_g(config)
        self.g_optimizer = networks.get_optim(self.g.parameters(), config.g_lr, config)
 
    def get_tx(self, x, reverse=False):
        x = Variable(x.data, requires_grad=True)
        if reverse:
            ux = self.psi(x)
        else:
            ux = self.phi(x)
        dux = torch.autograd.grad(outputs=ux, inputs=x,
                                  grad_outputs=utils.get_ones(ux.size()),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        Tx = x - dux
        return Tx

    # 20230519 Make a function that computes the gradient of the discriminator
    # 20230526 ZMALIK -> Commented out code now sent to individual models
    # 20230621 ZMALIK -> Uncommented code for use in base model
    def get_dux(self, x, reverse=False, prev=False):
        x = Variable(x.data, requires_grad=True)
        if prev: # Obtain gradient of previous discriminator
            if reverse:
                ux = self.psi_min1(x)
            else:
                ux = self.phi_min1(x)
        else: # Obtain gradient of current discriminator
            if reverse:
                ux = self.psi(x)
            else:
                ux = self.phi(x)
        dux = torch.autograd.grad(outputs=ux, inputs=x,
                                  grad_outputs=utils.get_ones(ux.size()),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        return dux

    def follow_ode(self, x, y, y_min1, ux, vy, config):
        dvy = self.get_dux(y) # Obtain gradient of current discriminator
        #dvy_hist_list = dvy_hist.get_dvy_hist_list() # Obtain list of previous dvy
        #size = len(dvy_hist_list) # Obtain step number (in training iteration)
        #dvy = self.get_dux(y) # Obtain most recent discriminator gradient
        #dvy_hist_list.append(dvy) # Update current list
        #if size == 0:
        #  yn1 = self.forward_euler(y, dvy_hist_list, config)
        if hasattr(self, 'g_min1') == 0:
          # Perform forward Euler if we do not yet have access to the previous generator
          yn1 = self.forward_euler(y, dvy, config)
        elif config.ode_step == 'ab1':
          # Perform Adams-Bashford-1 time step
          dvy_min1 = self.get_dux(y_min1, prev=True) # Obtain gradient of previous discriminator
          yn1 = self.adams_bashford_1(y, dvy, dvy_min1, config)
        #elif config.ode_step == 'ab2':
          # Perform Adams-Bashford-2 time step
        #  if size == 1:
            # Perform 1 step of AB1 before doing AB2
        #    yn1 = self.adams_bashford_1(y, dvy_hist_list, config)
        #  else:
        #    yn1 = self.adams_bashford_2(y, dvy_hist_list, config)
        else:
          # Default to forward Euler
          yn1 = self.forward_euler(y, dvy, config)
        #if size == 5:
        #  dvy_hist_list.pop(0)
        #dvy_hist.set_dvy_hist_list(dvy_hist_list) # Update the shared data
        #import pdb
        #pdb.set_trace()
        return yn1

    def forward_euler(self, y, dvy, config):
      # Forward Euler rule to obtain next sample of points
        yn1 = y + self.delta_t * dvy
        return yn1

    def adams_bashford_1(self, y, dvy, dvy_min1, config):
      # AB1 rule to obtain next sample of points
      yn1 = y + self.delta_t * (1.5 * dvy - 0.5 * dvy_min1)
      return yn1

    #def adams_bashford_2(self, y, dvy_hist_list, config):
    #  yn1 = y + self.delta_t * ((23/12) * dvy_hist_list[-1] - (4/3) * dvy_hist_list[-2] + (5/12) * dvy_hist_list[-3])
    #  return yn1

    def train_iter(self, config):
        self.set_phi_min1(config) # 20230718ZMALIK: Save previous discriminator before updating
        for it in range(config.d_iters):
            self.train_diter(config)
        if config.gen: 
            self.train_giter(config)
        
    def train_diter(self, config):
        self.d_optimizer.zero_grad()
        x, y, y_min1, z = self.get_data(config)
        # this is good for computational reasons:
        x, y = x.detach(), y.detach()
        tx, ty = self.get_tx(x), self.get_tx(y, reverse=True)
        ux, vy = self.phi(x), self.psi(y)
        d_loss = self.calc_dloss(x, y, tx, ty, ux, vy, config)
        d_loss.backward()
        self.d_optimizer.step()
        self.d_loss = d_loss.data.item()

    def train_giter(self, config):
        # 20230828 Return z values as well
        x, y, y_min1, z = self.get_data(config) # Default option 20230718 ZMALIK: Produce sample from previous generator as well
        tx, ty = self.get_tx(x), self.get_tx(y, reverse=True)
        ux, vy = self.phi(x), self.psi(y)
        if config.follow_ode:
          y1 = self.follow_ode(x, y, y_min1, ux, vy, config) # Sample produced by following ODE dynamics
          y1 = y1.detach()
          self.set_g_min1(config) #20230718 ZMALIK: Save previous generator AFTER following the ODE
        else:
          y1 = []
        #with torch.no_grad():
        for it in range(config.g_iters):
          self.g_optimizer.zero_grad()
          g_loss = self.calc_gloss(x, y, y1, ux, vy, config) # Now with option to shuffle!
          g_loss.backward(retain_graph=True)
          self.g_optimizer.step()
          a= list(self.g.parameters())[0].grad
          self.g_loss = g_loss.data.item()
          # Produce new y values using the SAME z values.
          y = self.g(z)
            #import pdb
            #pdb.set_trace()

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        raise NotImplementedError("Please Implement this method")

    def calc_gloss(self, x, y, ux, vy, config):
        raise NotImplementedError("Please Implement this method")

    def set_phi_min1(self, config):
        '''
        20230718ZMALIK: Save the current discriminator and set it to be the previous one
        '''
        self.phi_min1 = self.phi

    def set_psi_min1(self, config):
        '''
        20230718ZMALIK: Save the current discriminator and set it to be the previous one
        '''
        self.psi_min1 = self.psi

    def set_g_min1(self, config):
        '''
        20230718 ZMALIK: Save the current generator and set it to be the previous one
        '''
        self.g_min1 = self.g

    ## model statistics
    def get_stats(self,  config):
        raise NotImplementedError("Please Implement this method")

    def get_visuals(self, config):
        gz = self.g(self.fixed_z) if config.gen else self.fixed_z
        x, y = self.fixed_r, gz
        tx, ty = self.get_tx(x), self.get_tx(y, reverse=True)
        images = OrderedDict([('X', x),
                              ('TX', tx),
                              ('Y', y),
                              ('TY', ty)])
        if config.gen:
            images['ZY'] = self.fixed_z
        return images
