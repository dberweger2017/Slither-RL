"""
Export a trained PPO MultiInputPolicy to ONNX format for in-browser inference.
"""
import os
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from train import SlitherFeatureExtractor

CHECKPOINT_DIR = "checkpoints"


class OnnxablePolicy(torch.nn.Module):
    """
    SB3's PPO policy outputs distributions for training. We just want the deterministic 
    action prediction for inference. This wrapper extracts the features and passes 
    them through the policy net to yield the exact float actions.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs_map, obs_state):
        # 1. Feature extraction (CNN + MLP) bypassing dicts which break ONNX tracing
        cnn_features = self.features_extractor.cnn_fc(self.features_extractor.cnn(obs_map))
        state_features = self.features_extractor.state_mlp(obs_state)
        features = torch.cat([cnn_features, state_features], dim=1)
        
        # 2. Pass through the policy architecture
        latent_pi = self.mlp_extractor.policy_net(features)
        
        # 3. Get action means (deterministic actions)
        mean_actions = self.action_net(latent_pi)

        # Squash to [-1, 1] if the policy action distribution uses Tanh
        if hasattr(self.policy.action_dist, 'squash_output') and self.policy.action_dist.squash_output:
            mean_actions = torch.tanh(mean_actions)
            
        return mean_actions


def export_model(model_path, output_path="slither_policy.onnx"):
    print(f"Loading model from {model_path} bypassing SB3's PPO.load()...")
    
    import zipfile
    import tempfile
    
    from slither_gym import SlitherEnv
    from train import SlitherFeatureExtractor

    env = SlitherEnv(num_scripted=0, num_selfplay=0)
    policy_kwargs = {
        'features_extractor_class': SlitherFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': 576},
        'net_arch': dict(pi=[256, 128], vf=[256, 128]),
    }
    model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, device='cpu')
    
    # Extract just the policy.pth from the SB3 zip
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extract('policy.pth', tmpdir)
        
        state_dict_path = os.path.join(tmpdir, 'policy.pth')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.policy.load_state_dict(state_dict)
    
    print("Model loaded successfully.")
    
    # Wrap it
    onnxable_model = OnnxablePolicy(model.policy)
    onnxable_model.eval()
    
    # Create dummy inputs that match the observation space exactly
    # map: [1, 5, 84, 84] float32
    # state: [1, 8] float32
    dummy_map = torch.zeros(1, 5, 84, 84, dtype=torch.float32)
    dummy_state = torch.zeros(1, 8, dtype=torch.float32)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        onnxable_model,
        (dummy_map, dummy_state),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['map', 'state'],
        output_names=['action']
    )
    
    print(f"✅ Export complete! Saved to {output_path}")

    # Verify ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model checked and validated successfully.")
    except ImportError:
        print("Install 'onnx' library to validate the model (pip install onnx).")
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/policy_final.zip",
                        help="Path to the PPO model zip file")
    parser.add_argument("--out", type=str, default="slither_policy.onnx",
                        help="Output ONNX file path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Could not find model at {args.model}")
        
        # Try finding the latest numbered checkpoint if policy_final doesn't exist
        if not os.path.exists(CHECKPOINT_DIR):
            return
            
        files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("policy_")])
        if files:
            args.model = os.path.join(CHECKPOINT_DIR, files[-1])
            print(f"Using latest checkpoint instead: {args.model}")
        else:
            return

    export_model(args.model, args.out)

if __name__ == "__main__":
    main()
