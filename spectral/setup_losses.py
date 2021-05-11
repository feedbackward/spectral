'''Setup: loss functions used for training and evaluation.'''

## External modules.
from copy import deepcopy
import numpy as np
from scipy.special import erf

## Internal modules.
from mml.losses import Loss
from mml.losses.absolute import Absolute
from mml.losses.classification import Zero_One
from mml.losses.logistic import Logistic
from mml.losses.quadratic import Quadratic


###############################################################################


## All the helper functions for our specialized losses.

def sigma_default(u, c=1.0):
    '''
    '''
    return c * np.exp(-c*(1.0-u)) / (1.0-np.exp(-c))


def cdf_helper(u, model, loss_base, X=None, y=None):
    '''
    A helper to be used in the driver script.
    Estimates the loss CDF (loss evaluated at "model"),
    and evalutes that CDF estimator at "u".
    '''

    ref_losses = loss_base(model=model, X=X, y=y)
    
    ## Shape checks.
    if u.ndim == 2 and ref_losses.ndim == 2:
        if u.shape[1] == 1 and ref_losses.shape[1] == 1:
            out = np.mean(u >= ref_losses.T, axis=1, keepdims=True)
            return out
        else:
            raise ValueError("u and ref_losses have wrong 2nd axis.")
    else:
        raise ValueError("u and ref_losses have mismatched ndim.")


def cdf_approx_helper(u, model, loss_base, X=None, y=None):
    '''
    '''
    ref_losses = loss_base(model=model, X=X, y=y)
    return cdf_approx_static(u=u,
                             shift=np.mean(ref_losses),
                             scale=np.std(ref_losses))


def pdf_approx_helper(u, model, loss_base, X=None, y=None):
    '''
    '''
    ref_losses = loss_base(model=model, X=X, y=y)
    return pdf_approx_static(u=u,
                             shift=np.mean(ref_losses),
                             scale=np.std(ref_losses))


def cdf_approx_static(u, shift=0.0, scale=1.0):
    '''
    '''
    plus = (u+shift)/(scale*np.sqrt(2))
    minus = (u-shift)/(scale*np.sqrt(2))
    return (erf(plus)+erf(minus))/2
    

def pdf_approx_static(u, shift=0.0, scale=1.0):
    '''
    '''
    plus = (u+shift)/scale
    minus = (u-shift)/scale
    coef = 1.0 / (scale*np.sqrt(2*np.pi))
    return coef * (np.exp(-minus**2/2) + np.exp(-plus**2/2))


def get_noise(shape, rg):
    '''
    '''
    norm = 0
    while norm < 0.0001:
        vec = rg.normal(size=shape)
        norm = np.sqrt(np.sum(vec**2))

    return vec / norm
    #return rg.uniform(size=shape)


## Our loss classes for learning under spectral risks.

class S_Risk(Loss):
    '''
    Spectral risk based losses.
    Essentially, a special loss class that takes
    a base loss, spectral density, and empirical CDF
    as arguments for construction.

    The result is a loss object which amounts to an
    unbiased estimate of the spectral risk, though
    conditioned on the data used to compute the
    CDF estimate.

    - loss_base: the base loss object.
    - sigma: spectral density function.
    - cdf: empirical cdf function.
    - gamma: the perturbation parameter.
    - rg: random generator object.
    '''
    
    def __init__(self, loss_base, sigma, cdf, gamma, rg, name=None):
        '''
        '''
        loss_name = "S_Risk x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.sigma = sigma
        self.cdf = cdf
        self.gamma = gamma
        self.rg = rg
        return None
    
    
    def func(self, model, X, y):
        '''
        '''
        losses = self.loss(model=model, X=X, y=y)
        return losses * self.sigma(self.cdf(u=losses, model=model))
    
    
    def grad(self, model, X, y):
        '''
        '''
        ## Prepare shifted model, init the grad dict.
        loss_grads = {}
        model_shifted = deepcopy(model)
        for pn, p in model_shifted.paras.items():
            U = get_noise(shape=p.shape, rg=self.rg)
            p += self.gamma * U
            loss_grads[pn] = (p.size/self.gamma) * np.expand_dims(a=U, axis=0)
        
        ## Compute the modified loss at shifted model.
        losses = self.func(model=model_shifted, X=X, y=y)
        ldim = losses.ndim
        
        ## Compute the (smoothed fn) gradient estimate.
        for pn in loss_grads:
            gdim = loss_grads[pn].ndim
            if ldim > gdim:
                raise ValueError("Axis dimensions are wrong; ldim > gdim.")
            elif ldim < gdim:
                losses_exp = np.expand_dims(a=losses,
                                            axis=tuple(range(ldim,gdim)))
                loss_grads[pn] = loss_grads[pn] * losses_exp
            else:
                loss_grads[pn] = loss_grads[pn] * losses
        
        ## Return the gradients for all parameters being optimized.
        return loss_grads


class S_Risk_Fast(Loss):
    '''
    Fast approximation to S_Risk.
    Assumes that sigma is the exponential aversion function.
    '''
    
    def __init__(self, loss_base, sigma, cdf, pdf, name=None):
        '''
        '''
        loss_name = "S_Risk_Fast x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.sigma = sigma
        self.cdf = cdf
        self.pdf = pdf
        return None
    
    
    def func(self, model, X, y):
        '''
        '''
        losses = self.loss(model=model, X=X, y=y)
        return losses * self.sigma(cdf_approx_static(u=losses))
    
    
    def grad(self, model, X, y):
        '''
        '''
        
        ## Initial computations.
        loss_grads = self.loss.grad(model=model, X=X, y=y)
        losses = self.loss(model=model, X=X, y=y)
        TMP_CDF_VALS = self.cdf(u=losses, model=model)
        #print("DBDB TMP_CDF_VALS", TMP_CDF_VALS)
        coef = self.sigma(TMP_CDF_VALS)
        #print("DBDB losses", losses)
        coef *= 1.0 + losses*self.pdf(u=losses, model=model)
        #print("DBDB coef", coef)
        ldim = coef.ndim

        ## Compute the gradient estimate.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ldim > gdim:
                raise ValueError("Axis dimensions are wrong; ldim > gdim.")
            elif ldim < gdim:
                coef_exp = np.expand_dims(a=coef,
                                          axis=tuple(range(ldim,gdim)))
                g *= coef_exp
            else:
                g *= coef
            
        ## Return the gradients for all parameters being optimized.
        return loss_grads


## A dictionary of instantiated losses.

dict_losses = {
    "absolute": Absolute(),
    "logistic": Logistic(),
    "quadratic": Quadratic(),
    "zero_one": Zero_One()
}


def get_base(name):
    '''
    '''
    return dict_losses[name]


def get_loss(name, cdf=None, pdf=None, **kwargs):
    '''
    '''
    if kwargs["fast"]:
        return S_Risk_Fast(loss_base=dict_losses[name],
                           sigma=kwargs["sigma"],
                           cdf=cdf,
                           pdf=pdf)
    else:
        return S_Risk(loss_base=dict_losses[name],
                  sigma=kwargs["sigma"],
                  cdf=cdf,
                  gamma=kwargs["gamma"],
                  rg=kwargs["rg"])


###############################################################################
