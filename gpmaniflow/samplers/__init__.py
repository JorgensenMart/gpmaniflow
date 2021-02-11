from .matheron import sample_matheron

def initialize_sampler(from_df = False, num_samples = 1, num_basis = 512):
    return from_df, num_samples, num_basis