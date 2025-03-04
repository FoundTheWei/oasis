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
from typing import Dict, Any, Optional


class AdVariant:
    """Represents a single ad variant for A/B testing"""

    def __init__(
        self,
        variant_id: str,
        headline: str,
        body_text: str,
        image_url: Optional[str] = None,
        cta_text: str = "Learn More",
        target_demographics: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an ad variant
        
        Args:
            variant_id: Unique identifier for this variant
            headline: The headline/title of the ad
            body_text: Main content of the ad
            image_url: Optional URL to an image for the ad
            cta_text: Call to action text
            target_demographics: Optional targeting criteria
        """
        self.variant_id = variant_id
        self.headline = headline
        self.body_text = body_text
        self.image_url = image_url
        self.cta_text = cta_text
        self.target_demographics = target_demographics or {}
        
        # Metrics
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.costs = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ad variant to dictionary for display to agents"""
        return {
            "type": "advertisement",
            "variant_id": self.variant_id,
            "headline": self.headline,
            "body_text": self.body_text,
            "image_url": self.image_url,
            "cta": self.cta_text
        }
    
    def matches_targeting(self, agent_profile: Dict[str, Any]) -> bool:
        """Check if agent matches targeting criteria
        
        Args:
            agent_profile: Agent's profile data
            
        Returns:
            True if the agent matches targeting criteria or if no criteria set
        """
        if not self.target_demographics:
            return True
            
        for key, values in self.target_demographics.items():
            if key in agent_profile:
                if isinstance(values, list):
                    if agent_profile[key] not in values:
                        return False
                else:
                    if agent_profile[key] != values:
                        return False
        return True 