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
from typing import List, Dict, Any
from datetime import datetime
import random

from oasis.ad_testing.ad_variant import AdVariant


class AdCampaign:
    """Manages ad variants and tracks campaign metrics"""
    
    def __init__(
        self,
        campaign_id: str,
        name: str,
        variants: List[AdVariant],
        daily_budget: float = 100.0,
        cost_per_impression: float = 0.01,
        cost_per_click: float = 0.5,
    ):
        """Initialize an ad campaign
        
        Args:
            campaign_id: Unique identifier for the campaign
            name: Display name of the campaign
            variants: List of ad variants to test
            daily_budget: Budget limit per day
            cost_per_impression: Cost each time ad is shown
            cost_per_click: Cost each time ad is clicked
        """
        self.campaign_id = campaign_id
        self.name = name
        self.variants = variants
        self.daily_budget = daily_budget
        self.cost_per_impression = cost_per_impression
        self.cost_per_click = cost_per_click
        self.total_spent = 0.0
        self.metrics_by_day = {}
    
    def select_variant_for_agent(self, agent_profile: Dict[str, Any]) -> AdVariant:
        """Select appropriate ad variant for an agent based on targeting and A/B split
        
        Args:
            agent_profile: Agent's profile information for targeting
            
        Returns:
            Selected ad variant or None if no matching variants
        """
        # Filter variants that match targeting
        matching_variants = [v for v in self.variants if v.matches_targeting(agent_profile)]
        if not matching_variants:
            return None
        
        # Randomly select one variant (for A/B testing)
        return random.choice(matching_variants)
    
    def record_impression(self, variant_id: str, timestamp: datetime):
        """Record an ad impression
        
        Args:
            variant_id: ID of the variant that was shown
            timestamp: When the impression occurred
        """
        day_key = timestamp.strftime("%Y-%m-%d")
        if day_key not in self.metrics_by_day:
            self.metrics_by_day[day_key] = {
                v.variant_id: {"impressions": 0, "clicks": 0, "conversions": 0, "costs": 0.0}
                for v in self.variants
            }
        
        for variant in self.variants:
            if variant.variant_id == variant_id:
                variant.impressions += 1
                variant.costs += self.cost_per_impression
                self.total_spent += self.cost_per_impression
                
                self.metrics_by_day[day_key][variant_id]["impressions"] += 1
                self.metrics_by_day[day_key][variant_id]["costs"] += self.cost_per_impression
                break
    
    def record_click(self, variant_id: str, timestamp: datetime):
        """Record an ad click
        
        Args:
            variant_id: ID of the variant that was clicked
            timestamp: When the click occurred
        """
        day_key = timestamp.strftime("%Y-%m-%d")
        if day_key not in self.metrics_by_day:
            self.metrics_by_day[day_key] = {
                v.variant_id: {"impressions": 0, "clicks": 0, "conversions": 0, "costs": 0.0}
                for v in self.variants
            }
        
        for variant in self.variants:
            if variant.variant_id == variant_id:
                variant.clicks += 1
                variant.costs += self.cost_per_click
                self.total_spent += self.cost_per_click
                
                self.metrics_by_day[day_key][variant_id]["clicks"] += 1
                self.metrics_by_day[day_key][variant_id]["costs"] += self.cost_per_click
                break
    
    def get_daily_metrics(self, day: str = None) -> Dict[str, Dict[str, Any]]:
        """Get metrics for a specific day
        
        Args:
            day: Day in format "YYYY-MM-DD" or None for all days
            
        Returns:
            Dictionary of metrics by variant
        """
        if day is None:
            # Return latest day
            if not self.metrics_by_day:
                return {}
            day = max(self.metrics_by_day.keys())
        
        return self.metrics_by_day.get(day, {}) 