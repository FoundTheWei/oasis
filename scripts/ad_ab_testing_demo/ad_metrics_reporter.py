#!/usr/bin/env python3
# Ad Metrics Reporter - Comprehensive analytics for ad testing

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AdMetricsReporter:
    """
    Advanced reporting system for ad campaign testing results.
    Generates comprehensive analytics and visualizations for ad designers and campaign managers.
    """
    
    def __init__(self, results_dir: str = "./ad_results"):
        """
        Initialize the metrics reporter with the directory containing test results.
        
        Args:
            results_dir: Directory where ad test results are stored
        """
        self.results_dir = results_dir
        self.campaign_data = None
        self.variant_data = None
        self.user_data = None
        self.interaction_data = None
        self.sentiment_data = None
        self.conversion_data = None
        self.demographic_data = None
        
        # Create output directories
        self.reports_dir = os.path.join(results_dir, "reports")
        self.charts_dir = os.path.join(results_dir, "charts")
        self.dashboards_dir = os.path.join(results_dir, "dashboards")
        
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.dashboards_dir, exist_ok=True)

    def load_data(self, campaign_id: str) -> None:
        """
        Load all data for a specific campaign.
        
        Args:
            campaign_id: ID of the campaign to analyze
        """
        # Load campaign level data
        campaign_file = os.path.join(self.results_dir, f"{campaign_id}_summary.json")
        if os.path.exists(campaign_file):
            with open(campaign_file, 'r') as f:
                self.campaign_data = json.load(f)
        
        # Load variant performance data
        variants_file = os.path.join(self.results_dir, f"{campaign_id}_variants.json")
        if os.path.exists(variants_file):
            with open(variants_file, 'r') as f:
                self.variant_data = json.load(f)
        
        # Load user interaction data
        interactions_file = os.path.join(self.results_dir, f"{campaign_id}_interactions.json")
        if os.path.exists(interactions_file):
            with open(interactions_file, 'r') as f:
                self.interaction_data = json.load(f)
                
        # Load user demographic data
        demographics_file = os.path.join(self.results_dir, f"{campaign_id}_demographics.json")
        if os.path.exists(demographics_file):
            with open(demographics_file, 'r') as f:
                self.demographic_data = json.load(f)
                
        # Load sentiment analysis data
        sentiment_file = os.path.join(self.results_dir, f"{campaign_id}_sentiment.json")
        if os.path.exists(sentiment_file):
            with open(sentiment_file, 'r') as f:
                self.sentiment_data = json.load(f)
                
        # Load conversion funnel data
        conversion_file = os.path.join(self.results_dir, f"{campaign_id}_conversion.json")
        if os.path.exists(conversion_file):
            with open(conversion_file, 'r') as f:
                self.conversion_data = json.load(f)
                
    def generate_variant_comparison(self) -> pd.DataFrame:
        """
        Generate a comparison of all ad variants.
        
        Returns:
            DataFrame with comparative metrics for all variants
        """
        if not self.variant_data:
            return pd.DataFrame()
            
        # Convert to pandas DataFrame for easier analysis
        variants_df = pd.DataFrame(self.variant_data)
        
        # Calculate key metrics
        if 'impressions' in variants_df.columns and 'clicks' in variants_df.columns:
            variants_df['ctr'] = variants_df['clicks'] / variants_df['impressions']
            
        if 'clicks' in variants_df.columns and 'conversions' in variants_df.columns:
            variants_df['conversion_rate'] = variants_df['conversions'] / variants_df['clicks']
            
        if 'spend' in variants_df.columns and 'conversions' in variants_df.columns:
            variants_df['cost_per_acquisition'] = variants_df['spend'] / variants_df['conversions']
            
        if 'revenue' in variants_df.columns and 'spend' in variants_df.columns:
            variants_df['roi'] = (variants_df['revenue'] - variants_df['spend']) / variants_df['spend']
            
        return variants_df
        
    def generate_demographic_insights(self) -> Dict[str, pd.DataFrame]:
        """
        Generate insights based on demographic segments.
        
        Returns:
            Dictionary of DataFrames with demographic analysis
        """
        if not self.demographic_data or not self.variant_data:
            return {}
            
        insights = {}
        
        # Convert to DataFrames
        demo_df = pd.DataFrame(self.demographic_data)
        variant_df = pd.DataFrame(self.variant_data)
        
        # Group by demographic segments
        for segment in demo_df['segment'].unique():
            segment_data = demo_df[demo_df['segment'] == segment]
            
            # Get performance by demographic segment
            segment_performance = segment_data.groupby('variant_id').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'engagement_time': 'mean'
            }).reset_index()
            
            # Calculate rates
            segment_performance['ctr'] = segment_performance['clicks'] / segment_performance['impressions']
            segment_performance['cvr'] = segment_performance['conversions'] / segment_performance['clicks']
            
            insights[segment] = segment_performance
            
        return insights
        
    def generate_sentiment_analysis(self) -> pd.DataFrame:
        """
        Analyze sentiment data for comments and reactions.
        
        Returns:
            DataFrame with sentiment analysis by variant
        """
        if not self.sentiment_data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(self.sentiment_data)
        
        # Group by variant and sentiment
        sentiment_analysis = sentiment_df.groupby(['variant_id', 'sentiment']).size().unstack().reset_index()
        
        # Calculate sentiment scores
        if all(col in sentiment_analysis.columns for col in ['positive', 'neutral', 'negative']):
            total = sentiment_analysis['positive'] + sentiment_analysis['neutral'] + sentiment_analysis['negative']
            sentiment_analysis['sentiment_score'] = (
                sentiment_analysis['positive'] - sentiment_analysis['negative']
            ) / total
            
        return sentiment_analysis
        
    def generate_conversion_funnel(self) -> pd.DataFrame:
        """
        Analyze the conversion funnel by variant.
        
        Returns:
            DataFrame with conversion funnel metrics
        """
        if not self.conversion_data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        funnel_df = pd.DataFrame(self.conversion_data)
        
        # Group by variant
        funnel_analysis = funnel_df.groupby('variant_id').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'add_to_cart': 'sum',
            'checkout_initiated': 'sum',
            'purchase_completed': 'sum'
        }).reset_index()
        
        # Calculate drop-off rates
        funnel_analysis['impression_to_click'] = funnel_analysis['clicks'] / funnel_analysis['impressions']
        funnel_analysis['click_to_cart'] = funnel_analysis['add_to_cart'] / funnel_analysis['clicks']
        funnel_analysis['cart_to_checkout'] = funnel_analysis['checkout_initiated'] / funnel_analysis['add_to_cart']
        funnel_analysis['checkout_to_purchase'] = funnel_analysis['purchase_completed'] / funnel_analysis['checkout_initiated']
        funnel_analysis['overall_conversion'] = funnel_analysis['purchase_completed'] / funnel_analysis['impressions']
        
        return funnel_analysis
    
    def create_variant_comparison_chart(self, save_path: Optional[str] = None) -> None:
        """
        Create a visual comparison of ad variants performance.
        
        Args:
            save_path: Path to save the chart
        """
        if not self.variant_data:
            return
            
        variants_df = self.generate_variant_comparison()
        
        # Create a subplot with 2 rows and 2 columns
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=("Click-Through Rate by Variant", 
                           "Conversion Rate by Variant",
                           "Cost per Acquisition by Variant",
                           "Return on Ad Spend by Variant")
        )
        
        # Add bar charts for each metric
        if 'ctr' in variants_df.columns:
            fig.add_trace(
                go.Bar(x=variants_df['variant_id'], y=variants_df['ctr'], name="CTR"),
                row=1, col=1
            )
            
        if 'conversion_rate' in variants_df.columns:
            fig.add_trace(
                go.Bar(x=variants_df['variant_id'], y=variants_df['conversion_rate'], name="CVR"),
                row=1, col=2
            )
            
        if 'cost_per_acquisition' in variants_df.columns:
            fig.add_trace(
                go.Bar(x=variants_df['variant_id'], y=variants_df['cost_per_acquisition'], name="CPA"),
                row=2, col=1
            )
            
        if 'roi' in variants_df.columns:
            fig.add_trace(
                go.Bar(x=variants_df['variant_id'], y=variants_df['roi'], name="ROAS"),
                row=2, col=2
            )
            
        # Update layout
        fig.update_layout(
            title_text="Ad Variant Performance Comparison",
            height=800,
            width=1000
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
            fig.write_image(save_path.replace('.html', '.png'))
        
    def create_demographic_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Create a heatmap showing variant performance across demographic segments.
        
        Args:
            save_path: Path to save the chart
        """
        if not self.demographic_data or not self.variant_data:
            return
            
        # Create a pivot table for the heatmap
        demo_df = pd.DataFrame(self.demographic_data)
        
        # Calculate CTR by segment and variant
        segment_variant_ctr = demo_df.groupby(['segment', 'variant_id']).apply(
            lambda x: x['clicks'].sum() / x['impressions'].sum() if x['impressions'].sum() > 0 else 0
        ).reset_index(name='ctr')
        
        # Create pivot table
        pivot_df = segment_variant_ctr.pivot(index='segment', columns='variant_id', values='ctr')
        
        # Create heatmap
        fig = px.imshow(
            pivot_df, 
            labels=dict(x="Ad Variant", y="Demographic Segment", color="CTR"),
            title="Click-Through Rate by Demographic Segment and Ad Variant",
            color_continuous_scale="Viridis"
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            width=800
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
            fig.write_image(save_path.replace('.html', '.png'))
            
    def create_sentiment_analysis_chart(self, save_path: Optional[str] = None) -> None:
        """
        Create a visualization of sentiment analysis.
        
        Args:
            save_path: Path to save the chart
        """
        if not self.sentiment_data:
            return
            
        sentiment_df = pd.DataFrame(self.sentiment_data)
        
        # Create stacked bar chart of sentiment counts
        sentiment_counts = sentiment_df.groupby(['variant_id', 'sentiment']).size().unstack()
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Add traces for each sentiment
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_counts.columns:
                fig.add_trace(
                    go.Bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts[sentiment],
                        name=sentiment.capitalize()
                    )
                )
                
        # Update layout
        fig.update_layout(
            title="Sentiment Analysis by Ad Variant",
            xaxis_title="Ad Variant",
            yaxis_title="Count",
            barmode='stack',
            height=500,
            width=800
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
            fig.write_image(save_path.replace('.html', '.png'))
            
    def create_conversion_funnel_chart(self, save_path: Optional[str] = None) -> None:
        """
        Create a visualization of the conversion funnel.
        
        Args:
            save_path: Path to save the chart
        """
        if not self.conversion_data:
            return
            
        funnel_df = self.generate_conversion_funnel()
        
        # Create a subplot for each variant
        fig = make_subplots(
            rows=len(funnel_df),
            cols=1,
            subplot_titles=[f"Conversion Funnel: {variant}" for variant in funnel_df['variant_id']],
            vertical_spacing=0.1
        )
        
        # Add funnel chart for each variant
        for i, (_, row) in enumerate(funnel_df.iterrows(), 1):
            values = [
                row['impressions'],
                row['clicks'],
                row['add_to_cart'],
                row['checkout_initiated'],
                row['purchase_completed']
            ]
            
            fig.add_trace(
                go.Funnel(
                    name=row['variant_id'],
                    y=['Impressions', 'Clicks', 'Add to Cart', 'Checkout', 'Purchase'],
                    x=values,
                    textinfo="value+percent initial"
                ),
                row=i,
                col=1
            )
            
        # Update layout
        fig.update_layout(
            title_text="Conversion Funnel by Ad Variant",
            height=300 * len(funnel_df),
            width=800
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
            fig.write_image(save_path.replace('.html', '.png'))
            
    def generate_comprehensive_report(self, campaign_id: str) -> None:
        """
        Generate a comprehensive report with all analyses and visualizations.
        
        Args:
            campaign_id: ID of the campaign to analyze
        """
        # Load all data
        self.load_data(campaign_id)
        
        # Generate CSV reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Variant comparison
        variant_df = self.generate_variant_comparison()
        variant_csv = os.path.join(self.reports_dir, f"{campaign_id}_variant_comparison_{timestamp}.csv")
        if not variant_df.empty:
            variant_df.to_csv(variant_csv, index=False)
            
        # Demographic insights
        demo_insights = self.generate_demographic_insights()
        for segment, data in demo_insights.items():
            demo_csv = os.path.join(self.reports_dir, f"{campaign_id}_{segment}_analysis_{timestamp}.csv")
            data.to_csv(demo_csv, index=False)
            
        # Sentiment analysis
        sentiment_df = self.generate_sentiment_analysis()
        sentiment_csv = os.path.join(self.reports_dir, f"{campaign_id}_sentiment_analysis_{timestamp}.csv")
        if not sentiment_df.empty:
            sentiment_df.to_csv(sentiment_csv, index=False)
            
        # Conversion funnel
        funnel_df = self.generate_conversion_funnel()
        funnel_csv = os.path.join(self.reports_dir, f"{campaign_id}_conversion_funnel_{timestamp}.csv")
        if not funnel_df.empty:
            funnel_df.to_csv(funnel_csv, index=False)
            
        # Generate charts
        variant_chart = os.path.join(self.charts_dir, f"{campaign_id}_variant_comparison_{timestamp}.html")
        self.create_variant_comparison_chart(variant_chart)
        
        demo_chart = os.path.join(self.charts_dir, f"{campaign_id}_demographic_heatmap_{timestamp}.html")
        self.create_demographic_heatmap(demo_chart)
        
        sentiment_chart = os.path.join(self.charts_dir, f"{campaign_id}_sentiment_analysis_{timestamp}.html")
        self.create_sentiment_analysis_chart(sentiment_chart)
        
        funnel_chart = os.path.join(self.charts_dir, f"{campaign_id}_conversion_funnel_{timestamp}.html")
        self.create_conversion_funnel_chart(funnel_chart)
        
        # Generate executive summary
        summary = {
            "campaign_id": campaign_id,
            "report_generated": timestamp,
            "total_variants": len(variant_df) if not variant_df.empty else 0,
            "best_performing_variant": self._get_best_variant(variant_df) if not variant_df.empty else None,
            "demographic_insights": self._summarize_demographics(demo_insights),
            "sentiment_overview": self._summarize_sentiment(sentiment_df) if not sentiment_df.empty else None,
            "conversion_overview": self._summarize_conversion(funnel_df) if not funnel_df.empty else None,
            "reports_generated": {
                "csv_reports": [os.path.basename(f) for f in [variant_csv, sentiment_csv, funnel_csv] if f],
                "charts": [os.path.basename(f) for f in [variant_chart, demo_chart, sentiment_chart, funnel_chart] if f]
            }
        }
        
        # Save summary
        summary_path = os.path.join(self.reports_dir, f"{campaign_id}_executive_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Comprehensive report generated for campaign {campaign_id}")
        print(f"Executive summary saved to: {summary_path}")
            
    def _get_best_variant(self, variant_df: pd.DataFrame) -> Dict[str, Any]:
        """Determine the best performing variant based on multiple metrics."""
        if variant_df.empty:
            return {}
            
        metrics = {}
        
        # Best CTR
        if 'ctr' in variant_df.columns:
            best_ctr = variant_df.loc[variant_df['ctr'].idxmax()]
            metrics['best_ctr'] = {
                'variant_id': best_ctr['variant_id'],
                'ctr': best_ctr['ctr']
            }
            
        # Best conversion rate
        if 'conversion_rate' in variant_df.columns:
            best_cvr = variant_df.loc[variant_df['conversion_rate'].idxmax()]
            metrics['best_conversion_rate'] = {
                'variant_id': best_cvr['variant_id'],
                'conversion_rate': best_cvr['conversion_rate']
            }
            
        # Best ROI
        if 'roi' in variant_df.columns:
            best_roi = variant_df.loc[variant_df['roi'].idxmax()]
            metrics['best_roi'] = {
                'variant_id': best_roi['variant_id'],
                'roi': best_roi['roi']
            }
            
        return metrics
        
    def _summarize_demographics(self, demo_insights: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Summarize demographic insights."""
        if not demo_insights:
            return {}
            
        summary = {}
        
        for segment, df in demo_insights.items():
            if df.empty:
                continue
                
            if 'ctr' in df.columns:
                best_variant = df.loc[df['ctr'].idxmax()]
                summary[segment] = {
                    'best_variant': best_variant['variant_id'],
                    'ctr': best_variant['ctr']
                }
                
        return summary
        
    def _summarize_sentiment(self, sentiment_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize sentiment analysis."""
        if sentiment_df.empty:
            return {}
            
        summary = {}
        
        if 'sentiment_score' in sentiment_df.columns:
            best_sentiment = sentiment_df.loc[sentiment_df['sentiment_score'].idxmax()]
            summary['best_sentiment'] = {
                'variant_id': best_sentiment['variant_id'],
                'sentiment_score': best_sentiment['sentiment_score']
            }
            
        return summary
        
    def _summarize_conversion(self, funnel_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize conversion funnel analysis."""
        if funnel_df.empty:
            return {}
            
        summary = {}
        
        if 'overall_conversion' in funnel_df.columns:
            best_conversion = funnel_df.loc[funnel_df['overall_conversion'].idxmax()]
            summary['best_overall_conversion'] = {
                'variant_id': best_conversion['variant_id'],
                'overall_conversion': best_conversion['overall_conversion']
            }
            
        return summary


if __name__ == "__main__":
    # Example usage
    reporter = AdMetricsReporter("./ad_results")
    reporter.generate_comprehensive_report("summer_sale_2023") 