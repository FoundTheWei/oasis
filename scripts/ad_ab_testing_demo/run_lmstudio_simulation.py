#!/usr/bin/env python3
# Run ad testing simulation with LMStudio

import asyncio
import os
import sys
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scripts.ad_ab_testing_demo.ad_testing_simulation import running

# Create log directory if it doesn't exist
os.makedirs("./log", exist_ok=True)

# Create ad results directory if it doesn't exist
os.makedirs("./ad_results", exist_ok=True)

# Load the LMStudio configuration
config_path = os.path.join(os.path.dirname(__file__), "ad_testing_lmstudio.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Extract configuration sections
data_params = cfg.get("data", {})
simulation_params = cfg.get("simulation", {})
model_configs = cfg.get("model", {})
inference_params = cfg.get("inference", {})

# Run the simulation
if __name__ == "__main__":
    print("Starting ad testing simulation with LMStudio...")
    print(f"Configuration: {config_path}")
    print(f"Using model: {inference_params.get('model_type', 'unknown')}")
    print(f"Server: {inference_params.get('server_url', [{'host': 'unknown'}])[0]['host']}:{inference_params.get('server_url', [{'ports': ['unknown']}])[0]['ports'][0]}")
    
    # Run the simulation
    asyncio.run(
        running(
            **data_params,
            **simulation_params,
            model_configs=model_configs,
            inference_configs=inference_params,
        ),
        debug=True,
    ) 