import numpy as np

params = [{4179340454199820289, 3},
          {1945555039024054273, 5},
          {180143985094819841, 6},
          {31525197391593473, 3},
          {7881299347898369, 6},
          {4222124650659841, 19},
          {3799912185593857, 5}]

def get_params(seed = 123):
    rng = np.random.default_rng(seed)
    return rng.choice(params)
