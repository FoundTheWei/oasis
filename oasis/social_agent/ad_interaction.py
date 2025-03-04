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
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

from camel.messages import BaseMessage
from camel.memories import MemoryRecord
from camel.types import OpenAIBackendRole

from oasis.ad_testing.ad_variant import AdVariant
from oasis.ad_testing.ad_campaign import AdCampaign

# Setup logger
if "sphinx" not in __import__("sys").modules:
    ad_log = logging.getLogger(name="social.ad")
    ad_log.setLevel("DEBUG")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"./log/social.ad-{str(now)}.log")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
    ad_log.addHandler(file_handler)


async def process_ad(agent, ad_variant: AdVariant, campaign: AdCampaign, current_time: datetime):
    """Process an ad shown to the agent and determine response
    
    Args:
        agent: The SocialAgent instance
        ad_variant: The ad variant being shown
        campaign: The campaign the ad belongs to
        current_time: The current simulation time
        
    Returns:
        True if the agent clicked the ad, False otherwise
    """
    # Record impression
    campaign.record_impression(ad_variant.variant_id, current_time)
    
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
    
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=ad_prompt
    )
    
    # Add ad interaction to agent memory
    agent.memory.write_record(
        MemoryRecord(
            message=user_msg,
            role_at_backend=OpenAIBackendRole.USER,
        ))
    
    # Get LLM response
    openai_messages, _ = agent.memory.get_context()
    if not openai_messages or openai_messages[0]["role"] != agent.system_message.role_name:
        openai_messages = [{
            "role": agent.system_message.role_name,
            "content": agent.system_message.content,
        }] + [user_msg.to_openai_user_message()]
    
    ad_log.info(f"Agent {agent.agent_id} processing ad: {ad_variant.variant_id}")
    
    response = None
    content = None
    try:
        if agent.is_openai_model:
            response = agent.model_backend.run(openai_messages)
            content = response.choices[0].message.content
        else:
            # Use direct API call to LMStudio for more reliable interaction
            import requests
            import json
            import sys
            
            # Get server URL from agent configuration
            server_url = agent.inference_configs.get("server_url", [{"host": "127.0.0.1", "ports": [1234]}])[0]
            host = server_url.get("host", "127.0.0.1")
            port = server_url.get("ports", [1234])[0]
            
            url = f"http://{host}:{port}/v1/chat/completions"
            
            payload = {
                "model": "unsloth/deepseek-r1-distill-llama-8b",  # This should be configurable
                "messages": openai_messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            ad_log.info(f"Sending direct API request to LMStudio at {url}")
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                ad_log.info(f"Got LMStudio response: {content[:100]}...")
            else:
                ad_log.error(f"LMStudio API error: {response.status_code}, {response.text}")
                return False
    except Exception as e:
        ad_log.error(f"Error getting agent response to ad: {e}")
        import traceback
        ad_log.error(traceback.format_exc())
        return False
    
    if content is None:
        ad_log.error(f"No content received from LLM for agent {agent.agent_id}")
        return False
    
    ad_log.info(f"Agent {agent.agent_id} response to ad: {content}")
    
    # Record click if agent decides to click
    if "yes" in content.lower() and "click" in content.lower():
        campaign.record_click(ad_variant.variant_id, current_time)
        ad_log.info(f"Agent {agent.agent_id} clicked ad: {ad_variant.variant_id}")
        
        # Add agent's reply to memory
        agent_msg = BaseMessage.make_assistant_message(
            role_name="Assistant",
            content=content
        )
        agent.memory.write_record(
            MemoryRecord(
                message=agent_msg, 
                role_at_backend=OpenAIBackendRole.ASSISTANT
            )
        )
        
        return True
    return False 