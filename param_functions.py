def sparse_crf_anneal_param():

    # Storage convention: 'weight','sigma','subtract_eps'

    params = []
    steps = 15
    weight_start = 1.0
    weight_end = 1.0

    weight_delta = (weight_end-weight_start)/(steps-1)

    eps_start = 0.0   # set to positive value if repulsion (negative weights) is desired
    eps_end  = 0.0    # this parameter is not described in the paper because we did not use it

    eps_delta = (eps_start - eps_end) / (steps - 1)

    sigma_start = 0.05
    sigma_end = 0.15
    sigma_delta = (sigma_end - sigma_start) / (steps - 1)

    for i in range(steps):
        params.append([weight_start+weight_delta*i, sigma_start+sigma_delta*i, eps_start - eps_delta*i ])

    return params

def sparse_crf_to_list_dict(fn):

    param_list = []
    params = fn()

    for p in params:
        param_list.append({'weight': p[0], 'sigma': p[1], 'subtract_eps': p[2]})

    return param_list

