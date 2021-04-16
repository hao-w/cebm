def nats_to_params(nat1, nat2):
    """
    convert a Gaussian natural parameters its distritbuion parameters,
    mu = - 0.5 *  (nat1 / nat2), sigma = (- 0.5 / nat2).sqrt()
    input:
        nat1 : natural parameter which correspond to x,
        nat2 : natural parameter which correspond to x^2.      
    return values:
        mu : mean of a Gaussian
        sigma : standard deviation of a Gaussian
    """
    mu = - 0.5 * nat1 / nat2
    sigma = (- 0.5 / nat2).sqrt()
    return mu, sigma

def params_to_nats(mu, sigma):
    """
        convert a Gaussian distribution parameters to the natrual parameters
        nat1 = mean / sigma**2, nat2 = - 1 / (2 * sigma**2)
        input:
            mu : mean of a Gaussian
            sigma : standard deviation of a Gaussian
        return values:
            nat1 : natural parameter which correspond to x,
            nat2 : natural parameter which correspond to x^2.
    """
    nat1 = mu / (sigma**2)
    nat2 = - 0.5 / (sigma**2)
    return nat1, nat2