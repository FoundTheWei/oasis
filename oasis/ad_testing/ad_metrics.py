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
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

from oasis.ad_testing.ad_campaign import AdCampaign


class AdMetricsTracker:
    """Tracks and analyzes ad campaign performance"""
    
    def __init__(self, output_dir: str = "./ad_results"):
        """Initialize the metrics tracker
        
        Args:
            output_dir: Directory to save results and visualizations
        """
        self.campaigns = {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def add_campaign(self, campaign: AdCampaign):
        """Add a campaign to track
        
        Args:
            campaign: The campaign to track
        """
        self.campaigns[campaign.campaign_id] = campaign
    
    def record_ad_impression(self, agent_id: str, variant_id: str, content: str):
        """Record an ad impression
        
        Args:
            agent_id: ID of the agent who saw the impression
            variant_id: ID of the ad variant shown
            content: Content of the ad
        """
        # Find the campaign that contains this variant
        for campaign_id, campaign in self.campaigns.items():
            for variant in campaign.variants:
                if variant.variant_id == variant_id:
                    campaign.record_impression(variant_id, datetime.now())
                    return
    
    def record_ad_click(self, agent_id: str, variant_id: str, content: str):
        """Record an ad click
        
        Args:
            agent_id: ID of the agent who clicked
            variant_id: ID of the ad variant clicked
            content: Content of the ad
        """
        # Find the campaign that contains this variant
        for campaign_id, campaign in self.campaigns.items():
            for variant in campaign.variants:
                if variant.variant_id == variant_id:
                    campaign.record_click(variant_id, datetime.now())
                    return
    
    def get_campaign_summary(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get summary metrics for a campaign
        
        Args:
            campaign_id: ID of the campaign to summarize
            
        Returns:
            Dictionary of summary metrics or None if campaign not found
        """
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            return None
        
        summary = {
            "campaign_name": campaign.name,
            "total_spent": campaign.total_spent,
            "variants": []
        }
        
        for variant in campaign.variants:
            ctr = (variant.clicks / variant.impressions * 100) if variant.impressions > 0 else 0
            summary["variants"].append({
                "variant_id": variant.variant_id,
                "headline": variant.headline,
                "impressions": variant.impressions,
                "clicks": variant.clicks,
                "ctr": round(ctr, 2),
                "cost": round(variant.costs, 2),
                "cpc": round((variant.costs / variant.clicks) if variant.clicks > 0 else 0, 2)
            })
        
        return summary
    
    def generate_comparison_report(self, campaign_id: str) -> Optional[pd.DataFrame]:
        """Generate a DataFrame comparing variant performance
        
        Args:
            campaign_id: ID of the campaign to report on
            
        Returns:
            DataFrame with comparison data or None if campaign not found
        """
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            return None
        
        data = []
        for variant in campaign.variants:
            ctr = (variant.clicks / variant.impressions * 100) if variant.impressions > 0 else 0
            cpc = (variant.costs / variant.clicks) if variant.clicks > 0 else 0
            
            data.append({
                "Variant": variant.variant_id,
                "Headline": variant.headline,
                "Impressions": variant.impressions,
                "Clicks": variant.clicks,
                "CTR (%)": round(ctr, 2),
                "Cost ($)": round(variant.costs, 2),
                "CPC ($)": round(cpc, 2)
            })
        
        return pd.DataFrame(data)
    
    def plot_variant_performance(self, campaign_id: str, metric: str = "CTR (%)"):
        """Generate a bar chart of variant performance
        
        Args:
            campaign_id: ID of the campaign to visualize
            metric: Which metric to plot
        """
        df = self.generate_comparison_report(campaign_id)
        if df is None:
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(df["Headline"], df[metric])
        plt.title(f"Ad Variant Performance - {metric}")
        plt.ylabel(metric)
        plt.xlabel("Ad Variants")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Ensure file path doesn't have spaces or special chars
        safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        save_path = os.path.join(self.output_dir, f"campaign_{campaign_id}_{safe_metric}.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved performance chart to {save_path}")
    
    def save_all_results(self, campaign_id: str):
        """Save all results to files
        
        Args:
            campaign_id: ID of the campaign to save results for
        """
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            return
        
        # Save metrics summary
        summary = self.get_campaign_summary(campaign_id)
        summary_path = os.path.join(self.output_dir, f"campaign_{campaign_id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save comparison report
        df = self.generate_comparison_report(campaign_id)
        if df is not None:
            csv_path = os.path.join(self.output_dir, f"campaign_{campaign_id}_results.csv")
            df.to_csv(csv_path, index=False)
        
        # Generate all plots
        self.plot_variant_performance(campaign_id, "CTR (%)")
        self.plot_variant_performance(campaign_id, "Impressions")
        self.plot_variant_performance(campaign_id, "Clicks")
        self.plot_variant_performance(campaign_id, "Cost ($)")
        
        print(f"All results saved to {self.output_dir}")
        
    def print_campaign_results(self, campaign_id: str):
        """Print campaign results to console"""
        df = self.generate_comparison_report(campaign_id)
        if df is None:
            print(f"Campaign {campaign_id} not found")
            return
        
        print("\n=== CAMPAIGN RESULTS ===")
        print(df.to_string(index=False))
        print("\n") 