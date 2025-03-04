#!/usr/bin/env python3
# Test script to directly test the ad interaction with LMStudio API

import sys
import os
import json
import requests
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from oasis.ad_testing.ad_variant import AdVariant
from oasis.ad_testing.ad_campaign import AdCampaign

class MockAgent:
    """Mock agent for testing purposes"""
    def __init__(self, agent_id=1):
        self.agent_id = agent_id
        self.inference_configs = {
            "server_url": [{"host": "127.0.0.1", "ports": [1234]}]
        }
        self.is_openai_model = False
        
        # Mock profile
        self.user_profile = {
            "age": 30,
            "gender": "female",
            "interests": ["shopping", "technology", "fashion"]
        }

def create_test_ad():
    """Create a test ad variant"""
    return AdVariant(
        variant_id="variant_a",
        headline="Summer Sale: 30% Off Everything!",
        body_text="Beat the heat with our biggest summer sale ever. Limited time only!",
        image_url="images/summer_sale_1.jpg",
        cta_text="Shop Now",
        target_demographics={"age_groups": ["millennials", "gen_z"]}
    )

def create_test_campaign():
    """Create a test campaign"""
    ad = create_test_ad()
    return AdCampaign(
        campaign_id="test_campaign", 
        name="Test Campaign",
        variants=[ad],
        daily_budget=100.0,
        cost_per_impression=0.01,
        cost_per_click=0.5
    )

def process_ad_direct(agent, ad_variant, campaign):
    """Directly process an ad with LMStudio API"""
    # Record impression
    campaign.record_impression(ad_variant.variant_id, datetime.now())
    
    # Create prompt for LLM
    ad_prompt = (
        f"You are shown the following advertisement:\n"
        f"Headline: {ad_variant.headline}\n"
        f"Content: {ad_variant.body_text}\n"
        f"Call to Action: {ad_variant.cta_text}\n\n"
        f"Based on your profile and interests, would you click on this ad? "
        f"Please respond with either 'Yes, I would click this ad because...' or "
        f"'No, I would not click this ad because...' and explain your reasoning."
    )
    
    # System message to provide context about the user
    system_message = f"You are a 30-year-old female who is interested in shopping, technology, and fashion."
    
    # Prepare messages in OpenAI format
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": ad_prompt}
    ]
    
    # Get server URL
    server_url = agent.inference_configs.get("server_url", [{"host": "127.0.0.1", "ports": [1234]}])[0]
    host = server_url.get("host", "127.0.0.1")
    port = server_url.get("ports", [1234])[0]
    
    url = f"http://{host}:{port}/v1/chat/completions"
    
    payload = {
        "model": "unsloth/deepseek-r1-distill-llama-8b",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending direct API request to LMStudio at {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"Got LMStudio response: {content}")
            
            # Record click if agent decides to click
            if "yes" in content.lower() and "click" in content.lower():
                campaign.record_click(ad_variant.variant_id, datetime.now())
                print(f"Agent clicked ad: {ad_variant.variant_id}")
                return True
            else:
                print(f"Agent did not click ad: {ad_variant.variant_id}")
                return False
        else:
            print(f"LMStudio API error: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"Error getting agent response to ad: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Testing direct API interaction with LMStudio...")
    agent = MockAgent()
    ad = create_test_ad()
    campaign = create_test_campaign()
    
    result = process_ad_direct(agent, ad, campaign)
    
    if result:
        print("\nTest completed successfully - Agent clicked the ad!")
    else:
        print("\nTest completed - Agent did not click the ad.")
        
    # Print campaign metrics
    print("\nCampaign Metrics:")
    today = datetime.now().strftime("%Y-%m-%d")
    daily_metrics = campaign.get_daily_metrics(today)
    
    if daily_metrics and ad.variant_id in daily_metrics:
        variant_metrics = daily_metrics[ad.variant_id]
        print(f"Impressions: {variant_metrics.get('impressions', 0)}")
        print(f"Clicks: {variant_metrics.get('clicks', 0)}")
        if variant_metrics.get('impressions', 0) > 0:
            ctr = variant_metrics.get('clicks', 0) / variant_metrics.get('impressions', 1) * 100
            print(f"CTR: {ctr:.2f}%")
    else:
        print("No metrics available for today") 