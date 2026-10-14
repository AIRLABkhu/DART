from algorithms.sac import SAC
from algorithms.drq import DrQ
from algorithms.svea_c import SVEA_C


algorithm = {
	'sac': SAC,
	'drq': DrQ,
	'svea_c':SVEA_C,
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
