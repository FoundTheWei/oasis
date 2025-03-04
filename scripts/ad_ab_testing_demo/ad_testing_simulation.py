# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from colorama import Back, Fore, Style, init
from tqdm import tqdm
from yaml import safe_load

# Initialize colorama
init(autoreset=True)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import (gen_control_agents_with_data,
                                                 generate_reddit_agents)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType
from oasis.ad_testing import AdMetricsTracker
from oasis.ad_testing.ad_campaign import AdCampaign
from oasis.ad_testing.ad_variant import AdVariant
from scripts.ad_ab_testing_demo.ad_metrics_reporter import AdMetricsReporter

# Setup logging
social_log = logging.getLogger(name="social")
social_log.setLevel("DEBUG")
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
file_handler = logging.FileHandler(f"./log/ad_testing-{str(now)}.log",
                                   encoding="utf-8")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(stream_handler)

# Parse arguments
parser = argparse.ArgumentParser(description="Arguments for ad A/B testing simulation.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="./scripts/ad_ab_testing_demo/ad_testing.yaml",
)

# Set default paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "ad_testing.db")
DEFAULT_USER_PATH = os.path.join(DATA_DIR, "reddit", "user_data_36.json")
DEFAULT_PAIR_PATH = os.path.join(DATA_DIR, "emall", "product.json")
DEFAULT_AD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ad_variants.json")


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    user_path: str | None = DEFAULT_USER_PATH,
    pair_path: str | None = DEFAULT_PAIR_PATH,
    ad_config_path: str | None = DEFAULT_AD_CONFIG_PATH,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "reddit",
    controllable_user: bool = True,
    allow_self_rating: bool = False,
    show_score: bool = True,
    max_rec_post_len: int = 20,
    activate_prob: float = 0.5,
    ad_exposure_prob: float = 0.8,
    follow_post_agent: bool = False,
    mute_post_agent: bool = False,
    model_configs: dict[str, Any] | None = None,
    inference_configs: dict[str, Any] | None = None,
    refresh_rec_post_count: int = 5,
    round_post_num: int = 5,
    action_space_file_path: str = None,
    ad_results_dir: str = "./ad_results",
) -> None:
    """Run the ad A/B testing simulation
    
    Args:
        db_path: Path to the database
        user_path: Path to user profiles
        pair_path: Path to initial posts
        ad_config_path: Path to ad configuration
        num_timesteps: Number of simulation timesteps
        clock_factor: Time scaling factor
        recsys_type: Recommendation system type
        controllable_user: Whether to use a controllable user
        allow_self_rating: Allow users to rate their own posts
        show_score: Show post scores
        max_rec_post_len: Max number of posts to recommend
        activate_prob: Probability of agent activation
        ad_exposure_prob: Probability of showing an ad to an active agent
        follow_post_agent: Whether agents follow the post agent
        mute_post_agent: Whether agents mute the post agent
        model_configs: Model configurations
        inference_configs: Inference configurations
        refresh_rec_post_count: Number of posts to refresh
        round_post_num: Number of posts per round
        action_space_file_path: Path to action space prompt
        ad_results_dir: Directory to save ad results
    """
    # Set defaults for paths if not provided
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    user_path = DEFAULT_USER_PATH if user_path is None else user_path
    pair_path = DEFAULT_PAIR_PATH if pair_path is None else pair_path
    ad_config_path = DEFAULT_AD_CONFIG_PATH if ad_config_path is None else ad_config_path
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize clock and simulation time
    start_time = datetime(2024, 8, 6, 8, 0)
    clock = Clock(k=clock_factor)
    
    # Initialize channels
    twitter_channel = Channel()
    inference_channel = Channel()
    
    # Load action space prompt
    print(f"Loading action space prompt from: {action_space_file_path}")
    with open(action_space_file_path, "r", encoding="utf-8") as file:
        action_space_prompt = file.read()

    # Initialize platform
    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        allow_self_rating=allow_self_rating,
        show_score=show_score,
        recsys_type=recsys_type,
        max_rec_post_len=max_rec_post_len,
        refresh_rec_post_count=refresh_rec_post_count,
    )
    
    # Add some sample products
    await infra.sign_up_product(1, "GlowPod")
    await infra.sign_up_product(2, "Mistify")
    await infra.sign_up_product(3, "ZenCloud")

    # Start platform
    twitter_task = asyncio.create_task(infra.running())

    # Check if using OpenAI model
    if inference_configs.get("model_type", "")[:3] == "gpt":
        is_openai_model = True
    else:
        is_openai_model = inference_configs.get("is_openai_model", False)
    
    # Generate agents
    if not controllable_user:
        raise ValueError("Uncontrollable user is not supported")
    else:
        agent_graph, id_mapping = await gen_control_agents_with_data(
            twitter_channel,
            1,  # One controllable agent
        )
        agent_graph = await generate_reddit_agents(
            user_path,
            twitter_channel,
            inference_channel,
            agent_graph,
            id_mapping,
            follow_post_agent,
            mute_post_agent,
            action_space_prompt,
            inference_configs["model_type"],
            is_openai_model,
        )
    
    # Load initial posts
    with open(pair_path, "r") as f:
        pairs = json.load(f)
    
    # Load ad configurations
    with open(ad_config_path, "r") as f:
        ad_config = json.load(f)
    
    # Create ad variants
    ad_variants = []
    for variant_config in ad_config["variants"]:
        variant = AdVariant(
            variant_id=variant_config["id"],
            headline=variant_config["headline"],
            body_text=variant_config["body_text"],
            image_url=variant_config.get("image_url"),
            cta_text=variant_config.get("cta_text", "Learn More"),
            target_demographics=variant_config.get("targeting")
        )
        ad_variants.append(variant)
    
    # Create campaign
    campaign = AdCampaign(
        campaign_id=ad_config["campaign_id"],
        name=ad_config["campaign_name"],
        variants=ad_variants,
        daily_budget=ad_config.get("daily_budget", 100.0)
    )
    
    # Create metrics tracker
    metrics_tracker = AdMetricsTracker(ad_results_dir)
    metrics_tracker.add_campaign(campaign)
    
    print(f"{Fore.GREEN}Starting ad A/B testing simulation with {len(agent_graph.get_agents())} agents{Fore.RESET}")
    print(f"{Fore.BLUE}Testing {len(ad_variants)} ad variants:{Fore.RESET}")
    for variant in ad_variants:
        print(f"  - {variant.variant_id}: {variant.headline}")
    
    # Run simulation for specified number of timesteps
    for timestep in range(num_timesteps):
        os.environ["TIME_STAMP"] = str(timestep + 1)
        if timestep == 0:
            start_time_0 = datetime.now()
        
        print(Back.GREEN + f"Timestep: {timestep + 1}/{num_timesteps}" + Back.RESET)
        social_log.info(f"Timestep: {timestep + 1}.")

        # Create initial posts in first timestep
        post_agent = agent_graph.get_agent(0)
        if timestep == 0:
            post_count = min(3, len(pairs))
            for i in range(post_count):
                await post_agent.perform_action_by_data(
                    "create_post", content=pairs[i]["content"])
                social_log.info(f"Created post with content: {pairs[i]['content'][:50]}...")

        # Update recommendation table
        await infra.update_rec_table()
        social_log.info("Updated recommendation table")
        
        # Process non-controllable agents
        social_tasks = []
        ad_tasks = []
        
        non_controllable_agents = [(node_id, agent) for node_id, agent 
                                   in agent_graph.get_agents() 
                                   if agent.user_info.is_controllable is False]
        
        # Process each agent
        for node_id, agent in tqdm(non_controllable_agents, desc="Processing agents"):
            # Determine if agent is active this timestep
            if random.random() < activate_prob:
                # Determine if agent sees an ad
                if random.random() < ad_exposure_prob:
                    # Select appropriate variant for agent
                    profile = agent.user_info.profile
                    ad_variant = campaign.select_variant_for_agent(profile)
                    
                    if ad_variant:
                        # Process ad for this agent using direct API approach
                        # Record impression
                        campaign.record_impression(ad_variant.variant_id, datetime.now())
                        
                        # Create prompt for LLM to determine if agent would click the ad
                        ad_prompt = (
                            f"You are shown the following advertisement:\n"
                            f"Headline: {ad_variant.headline}\n"
                            f"Content: {ad_variant.body_text}\n"
                            f"Call to Action: {ad_variant.cta_text}\n\n"
                            f"Based on your profile and interests, would you click on this ad? "
                            f"Please respond with either 'Yes, I would click this ad because...' or "
                            f"'No, I would not click this ad because...' and explain your reasoning."
                        )
                        
                        # Get system message from agent
                        system_message = agent.system_message.content
                        
                        # Prepare messages in OpenAI format
                        messages = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": ad_prompt}
                        ]
                        
                        # Get server URL from inference configs
                        server_url = inference_configs.get("server_url", [{"host": "127.0.0.1", "ports": [1234]}])[0]
                        host = server_url.get("host", "127.0.0.1")
                        port = server_url.get("ports", [1234])[0]
                        
                        url = f"http://{host}:{port}/v1/chat/completions"
                        
                        # Get model name from inference configs or use a default
                        model_name = None
                        if isinstance(inference_configs.get("model_type"), str):
                            model_name = inference_configs.get("model_type")
                        # If no model specified, use what we see in LMStudio UI
                        if not model_name:
                            model_name = "unsloth/deepseek-r1-distill-llama-8b"
                        
                        payload = {
                            "model": model_name,
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 500
                        }
                        
                        headers = {
                            "Content-Type": "application/json"
                        }
                        
                        # Create a task for processing this ad
                        ad_tasks.append({
                            "agent_id": agent.agent_id,
                            "variant_id": ad_variant.variant_id,
                            "url": url,
                            "payload": payload,
                            "headers": headers
                        })
                else:
                    # Regular social media actions
                    social_tasks.append(agent.perform_action_by_llm())
        
        # Execute all ad processing tasks
        if ad_tasks:
            print(f"Processing {len(ad_tasks)} ad exposures...")
            ad_clicks = 0
            
            # Process each ad task directly
            for task in ad_tasks:
                agent_id = task["agent_id"]
                variant_id = task["variant_id"]
                url = task["url"]
                payload = task["payload"]
                headers = task["headers"]
                
                try:
                    social_log.info(f"Agent {agent_id} processing ad: {variant_id}")
                    print(f"Agent {agent_id} processing ad variant: {variant_id}")
                    
                    # Add better timeout and error handling
                    response = requests.post(
                        url, 
                        headers=headers, 
                        data=json.dumps(payload), 
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                        social_log.info(f"Got LMStudio response for agent {agent_id}: {content[:100]}...")
                        print(f"Received response for agent {agent_id}")
                        
                        # Record click if agent decides to click
                        if "yes" in content.lower() and "click" in content.lower():
                            campaign.record_click(variant_id, datetime.now())
                            social_log.info(f"Agent {agent_id} clicked ad: {variant_id}")
                            metrics_tracker.record_ad_click(agent_id, variant_id, content)
                            ad_clicks += 1
                            print(f"👆 Agent {agent_id} clicked ad: {variant_id}")
                        else:
                            social_log.info(f"Agent {agent_id} did not click ad: {variant_id}")
                            metrics_tracker.record_ad_impression(agent_id, variant_id, content)
                            print(f"👁️ Agent {agent_id} viewed but did not click ad: {variant_id}")
                    else:
                        social_log.error(f"LMStudio API error: {response.status_code}, {response.text}")
                        print(f"❌ API error for agent {agent_id}: {response.status_code}")
                except requests.exceptions.Timeout:
                    social_log.error(f"Timeout processing ad for agent {agent_id}")
                    print(f"⏱️ Timeout processing ad for agent {agent_id}")
                except requests.exceptions.ConnectionError:
                    social_log.error(f"Connection error for agent {agent_id}. Check if LMStudio is running.")
                    print(f"❌ Connection error for agent {agent_id}. Check if LMStudio is running at {url}")
                except Exception as e:
                    social_log.error(f"Error getting agent {agent_id} response to ad: {e}")
                    print(f"❌ Error processing ad for agent {agent_id}: {str(e)}")
                    import traceback
                    social_log.error(traceback.format_exc())
            
            print(f"Ad clicks: {ad_clicks} ({(ad_clicks/len(ad_tasks))*100:.1f}% CTR)")
        
        # Execute social tasks
        if social_tasks:
            random.shuffle(social_tasks)
            print(f"Processing {len(social_tasks)} social actions...")
            await asyncio.gather(*social_tasks)

        # Calculate clock factor after first timestep
        if timestep == 0:
            time_difference = datetime.now() - start_time_0
            two_hours_in_seconds = timedelta(hours=2).total_seconds()
            clock_factor = two_hours_in_seconds / time_difference.total_seconds()
            clock.k = clock_factor
            social_log.info(f"Set clock factor to: {clock_factor}")
        
        # Print interim results
        if (timestep + 1) % 1 == 0 or timestep == num_timesteps - 1:
            print("\nInterim Campaign Results:")
            metrics_tracker.print_campaign_results(campaign.campaign_id)

    # Save final results
    metrics_tracker.save_all_results(campaign.campaign_id)
    print(f"{Fore.GREEN}Final results saved to: {ad_results_dir}{Fore.RESET}")
    
    # Shutdown
    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await twitter_task
    social_log.info("Ad A/B testing simulation completed!")


if __name__ == "__main__":
    args = parser.parse_args()

    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        model_configs = cfg.get("model")
        inference_params = cfg.get("inference")
        
        # Create log directory if it doesn't exist
        os.makedirs("./log", exist_ok=True)
        
        asyncio.run(
            running(
                **data_params,
                **simulation_params,
                model_configs=model_configs,
                inference_configs=inference_params,
            ),
            debug=True,
        )
    else:
        print(f"Config file not found: {args.config_path}")
        print("Using default configuration")
        asyncio.run(running()) 