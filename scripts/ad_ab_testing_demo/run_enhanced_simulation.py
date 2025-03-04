#!/usr/bin/env python3
# Enhanced Ad Testing Simulation Runner

import asyncio
import os
import sys
import yaml
import json
import argparse
from datetime import datetime
from pathlib import Path
from colorama import Fore, Back, Style, init
import requests
import time

# Initialize colorama
init(autoreset=True)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scripts.ad_ab_testing_demo.ad_testing_simulation import running
from scripts.ad_ab_testing_demo.ad_metrics_reporter import AdMetricsReporter

def parse_arguments():
    """Parse command line arguments for the simulation."""
    parser = argparse.ArgumentParser(description='Run enhanced ad testing simulation')
    
    parser.add_argument('--config', type=str, default='ad_testing_lmstudio.yaml',
                      help='Configuration file name (default: ad_testing_lmstudio.yaml)')
    
    parser.add_argument('--timesteps', type=int, default=None,
                      help='Override number of timesteps in simulation')
    
    parser.add_argument('--output-dir', type=str, default='./ad_results',
                      help='Directory to save simulation results (default: ./ad_results)')
    
    parser.add_argument('--campaign-id', type=str, default=None,
                      help='Override campaign ID from ad config')
    
    parser.add_argument('--demographics', action='store_true',
                      help='Enable demographic-based simulation')
    
    parser.add_argument('--sentiment', action='store_true',
                      help='Enable sentiment analysis in simulation')
    
    parser.add_argument('--conversion-tracking', action='store_true',
                      help='Enable detailed conversion funnel tracking')
    
    parser.add_argument('--competitor-analysis', action='store_true',
                      help='Enable competitor ad analysis')
    
    parser.add_argument('--generate-report', action='store_true',
                      help='Generate comprehensive report after simulation')
    
    parser.add_argument('--interactive', action='store_true',
                      help='Enable interactive mode to adjust simulation parameters during runtime')
    
    return parser.parse_args()

def load_configuration(config_name):
    """Load configuration from the specified YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), config_name)
    
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg, config_path
    except FileNotFoundError:
        print(f"{Fore.RED}Configuration file not found: {config_path}{Fore.RESET}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"{Fore.RED}Error parsing YAML configuration: {e}{Fore.RESET}")
        sys.exit(1)

def override_config_with_args(cfg, args):
    """Override configuration with command-line arguments."""
    # Apply command-line overrides to configuration
    if args.timesteps:
        cfg['simulation']['num_timesteps'] = args.timesteps
        
    if args.output_dir:
        cfg['simulation']['ad_results_dir'] = args.output_dir
        
    if args.campaign_id:
        # We'll need to read and modify the ad config file
        ad_config_path = cfg['data']['ad_config_path']
        try:
            with open(ad_config_path, 'r') as f:
                ad_config = json.load(f)
            
            # Update campaign ID
            ad_config['campaign_id'] = args.campaign_id
            
            # Write back the modified config
            with open(ad_config_path, 'w') as f:
                json.dump(ad_config, f, indent=2)
                
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Failed to update campaign ID in ad config: {e}{Fore.RESET}")
    
    # Enable additional features based on arguments
    if args.demographics and 'analytics' in cfg:
        cfg['analytics']['segment_analysis'] = True
        
    if args.sentiment and 'analytics' in cfg:
        cfg['analytics']['sentiment_analysis'] = True
        
    if args.conversion_tracking and 'analytics' in cfg:
        cfg['analytics']['track_conversion_funnel'] = True
        
    if args.competitor_analysis and 'analytics' in cfg:
        cfg['analytics']['competitor_analysis'] = True
        
    return cfg

def setup_directories(cfg):
    """Set up necessary directories for the simulation."""
    # Create log directory
    os.makedirs("./log", exist_ok=True)
    
    # Create ad results directory
    results_dir = cfg['simulation']['ad_results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Update config to use the timestamped directory
    cfg['simulation']['ad_results_dir'] = run_dir
    
    return run_dir, timestamp

def print_simulation_info(cfg, args, run_dir, timestamp):
    """Print information about the simulation that will be run."""
    print(f"\n{Fore.CYAN}{'='*80}{Fore.RESET}")
    print(f"{Fore.CYAN}Enhanced Ad Testing Simulation{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*80}{Fore.RESET}")
    
    # Basic configuration info
    print(f"\n{Fore.WHITE}Configuration:{Fore.RESET} {args.config}")
    
    # Get ad configuration
    ad_config_path = cfg['data']['ad_config_path']
    try:
        with open(ad_config_path, 'r') as f:
            ad_config = json.load(f)
        campaign_id = ad_config.get('campaign_id', 'unknown')
        campaign_name = ad_config.get('campaign_name', 'unknown')
        variants = ad_config.get('variants', [])
    except Exception:
        campaign_id = 'unknown'
        campaign_name = 'unknown'
        variants = []
    
    print(f"{Fore.WHITE}Campaign:{Fore.RESET} {campaign_name} (ID: {campaign_id})")
    print(f"{Fore.WHITE}Model:{Fore.RESET} {cfg['inference'].get('model_type', 'unknown')}")
    print(f"{Fore.WHITE}API Server:{Fore.RESET} {cfg['inference'].get('server_url', [{'host': 'unknown'}])[0]['host']}:{cfg['inference'].get('server_url', [{'ports': ['unknown']}])[0]['ports'][0]}")
    print(f"{Fore.WHITE}Timesteps:{Fore.RESET} {cfg['simulation'].get('num_timesteps', 0)}")
    print(f"{Fore.WHITE}Variants:{Fore.RESET} {len(variants)}")
    print(f"{Fore.WHITE}Output Directory:{Fore.RESET} {run_dir}")
    print(f"{Fore.WHITE}Run ID:{Fore.RESET} {timestamp}")
    
    # Enhanced features
    print(f"\n{Fore.YELLOW}Enhanced Features:{Fore.RESET}")
    
    if 'analytics' in cfg:
        analytics = cfg['analytics']
        print(f"  - Demographic Analysis: {'✅' if analytics.get('segment_analysis', False) else '❌'}")
        print(f"  - Sentiment Analysis: {'✅' if analytics.get('sentiment_analysis', False) else '❌'}")
        print(f"  - Conversion Tracking: {'✅' if analytics.get('track_conversion_funnel', False) else '❌'}")
        print(f"  - Competitor Analysis: {'✅' if analytics.get('competitor_analysis', False) else '❌'}")
    else:
        print(f"  {Fore.RED}No analytics features configured{Fore.RESET}")
    
    print(f"\n{Fore.CYAN}{'='*80}{Fore.RESET}\n")

def check_lmstudio_connection(cfg):
    """Check if LMStudio is running and accessible"""
    if 'inference' not in cfg:
        return False
    
    inference = cfg['inference']
    if inference.get('is_openai_model', False):
        # If using OpenAI, skip this check
        return True
    
    if 'server_url' not in inference:
        return False
    
    server_urls = inference['server_url']
    if not server_urls or not isinstance(server_urls, list):
        return False
    
    # Try to connect to the first server
    host = server_urls[0].get('host', 'localhost')
    ports = server_urls[0].get('ports', [1234])
    
    if not ports:
        return False
    
    url = f"http://{host}:{ports[0]}/v1/models"
    
    try:
        print(f"{Fore.YELLOW}Checking connection to LMStudio at {url}...{Fore.RESET}")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"{Fore.GREEN}Successfully connected to LMStudio!{Fore.RESET}")
            return True
        else:
            print(f"{Fore.RED}LMStudio returned status code {response.status_code}{Fore.RESET}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}Failed to connect to LMStudio: {str(e)}{Fore.RESET}")
        return False

async def run_simulation(cfg):
    """Run the ad testing simulation with the provided configuration."""
    # Extract configuration sections
    data_params = cfg.get("data", {}).copy()  # Create a copy to avoid modifying the original
    simulation_params = cfg.get("simulation", {}).copy()  # Create a copy to avoid modifying the original
    model_configs = cfg.get("model", {}) or {}  # Ensure we have a dict even if None
    inference_params = cfg.get("inference", {})
    analytics_params = cfg.get("analytics", {})
    
    # Move demographic_segments from data_params to model_configs
    if 'demographic_segments' in data_params:
        model_configs['demographic_segments'] = data_params.pop('demographic_segments')
    
    # Move custom simulation parameters to model_configs
    custom_sim_params = [
        'enable_retargeting', 
        'ad_fatigue_factor', 
        'seasonal_effects', 
        'time_of_day_effects'
    ]
    
    for param in custom_sim_params:
        if param in simulation_params:
            if 'simulation_features' not in model_configs:
                model_configs['simulation_features'] = {}
            model_configs['simulation_features'][param] = simulation_params.pop(param)
    
    # Add analytics params to model_configs if they exist
    if analytics_params:
        model_configs["analytics"] = analytics_params
    
    # Keep only the parameters that the running function accepts
    allowed_params = [
        'db_path', 'user_path', 'pair_path', 'ad_config_path', 'num_timesteps',
        'clock_factor', 'recsys_type', 'controllable_user', 'allow_self_rating',
        'show_score', 'max_rec_post_len', 'activate_prob', 'ad_exposure_prob',
        'follow_post_agent', 'mute_post_agent', 'refresh_rec_post_count',
        'round_post_num', 'action_space_file_path', 'ad_results_dir'
    ]
    
    # Filter simulation params to only include those the function accepts
    filtered_simulation_params = {k: v for k, v in simulation_params.items() if k in allowed_params}
    
    # Run the simulation
    await running(
        **data_params,
        **filtered_simulation_params,
        model_configs=model_configs,
        inference_configs=inference_params,
    )

def generate_reports(run_dir, campaign_id):
    """Generate comprehensive reports using the AdMetricsReporter."""
    print(f"\n{Fore.CYAN}Generating comprehensive reports...{Fore.RESET}")
    
    try:
        reporter = AdMetricsReporter(run_dir)
        reporter.generate_comprehensive_report(campaign_id)
        print(f"{Fore.GREEN}Reports generated successfully in {run_dir}/reports{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}Error generating reports: {e}{Fore.RESET}")

async def main():
    """Main entry point for the enhanced simulation runner."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    cfg, config_path = load_configuration(args.config)
    
    # Override configuration with command-line arguments
    cfg = override_config_with_args(cfg, args)
    
    # Set up directories
    run_dir, timestamp = setup_directories(cfg)
    
    # Get campaign ID from ad config
    try:
        with open(cfg['data']['ad_config_path'], 'r') as f:
            ad_config = json.load(f)
        campaign_id = ad_config.get('campaign_id', 'unknown')
    except Exception:
        campaign_id = 'unknown'
    
    # Print simulation information
    print_simulation_info(cfg, args, run_dir, timestamp)
    
    # Save the effective configuration
    effective_config_path = os.path.join(run_dir, f"effective_config_{timestamp}.yaml")
    with open(effective_config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    # Check LMStudio connection if using local model
    if not cfg['inference'].get('is_openai_model', False):
        if not check_lmstudio_connection(cfg):
            print(f"\n{Fore.RED}ERROR: Cannot connect to LMStudio.{Fore.RESET}")
            print(f"{Fore.YELLOW}Please make sure LMStudio is running with API server enabled on the configured port.{Fore.RESET}")
            print(f"{Fore.YELLOW}You can start LMStudio, load a model, and enable API server from the 'Local Server' tab.{Fore.RESET}")
            print(f"{Fore.YELLOW}Alternatively, you can modify the configuration to use OpenAI models instead.{Fore.RESET}")
            return
    
    # Run the simulation
    try:
        await run_simulation(cfg)
        
        # Generate reports if requested
        if args.generate_report:
            generate_reports(run_dir, campaign_id)
        
        print(f"\n{Fore.GREEN}Simulation completed successfully!{Fore.RESET}")
        print(f"{Fore.GREEN}Results saved to: {run_dir}{Fore.RESET}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during simulation: {str(e)}{Fore.RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main(), debug=True) 