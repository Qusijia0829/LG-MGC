
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig
import torch
def build_dalle(path = None):

    prior_config = TrainDiffusionPriorConfig.from_json_path("  ").prior
    prior = prior_config.create()

    prior_model_state = torch.load("   ",map_location=torch.device('cpu'))
    prior.load_state_dict(prior_model_state, strict=True)
    prior.to('cuda')

    return  prior
