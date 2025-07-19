#!/usr/bin/env python3
"""
Biological Data Volatility Analysis Tool
Measures and analyzes volatility patterns in biological/environmental data
to create "BioVIX" style indicators for market prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BioVolatilityAnalyzer:
    """
    Analyzes volatility patterns in biological data streams
    Creates volatility indices similar to VIX but for ecosystem data
    """
    
    def __init__(self):
        self.volatility_metrics = {}
        self.bio_vix_components = {}
        self.volatility_regimes = {}
        
    def calculate_rolling_volatility(self, data: pd.Series, window: int = 20, 
                                   method: str = 'std') -> pd.Series:
        """
        Calculate rolling volatility using various methods
        """
        if method == 'std':
            # Standard rolling standard deviation
            return data.rolling(window=window).std()
        
        elif method == 'parkinson':
            # Parkinson volatility estimator (requires high/low data)
            # For single series, use rolling max/min as proxy
            rolling_max = data.rolling(window=window).max()
            rolling_min = data.rolling(window=window).min()
            return np.sqrt(np.log(rolling_max / rolling_min) ** 2 / (4 * np.log(2)))
        
        elif method == 'garman_klass':
            # Garman-Klass estimator adaptation for biological data
            rolling_max = data.rolling(window=window).max()
            rolling_min = data.rolling(window=window).min()
            rolling_mean = data.rolling(window=window).mean()
            
            return np.sqrt(
                0.5 * (np.log(rolling_max / rolling_min) ** 2) - 
                (2 * np.log(2) - 1) * (np.log(rolling_mean / rolling_min) ** 2)
            )
        
        elif method == 'returns_vol':
            # Volatility of returns (for time series with trend)
            returns = data.pct_change()
            return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        elif method == 'mad':
            # Mean Absolute Deviation
            return data.rolling(window=window).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
        
        else:
            raise ValueError(f"Unknown volatility method: {method}")
    
    def calculate_regime_volatility(self, data: pd.Series, regime_threshold: float = 0.5) -> pd.DataFrame:
        """
        Calculate volatility in different regimes (high/low activity periods)
        """
        # Identify regimes based on data levels
        data_normalized = (data - data.mean()) / data.std()
        high_regime = data_normalized > regime_threshold
        low_regime = data_normalized < -regime_threshold
        normal_regime = (~high_regime) & (~low_regime)
        
        regime_data = pd.DataFrame({
            'value': data,
            'regime': 'normal',
            'normalized': data_normalized
        })
        
        regime_data.loc[high_regime, 'regime'] = 'high'
        regime_data.loc[low_regime, 'regime'] = 'low'
        
        # Calculate volatility within each regime
        regime_volatilities = {}
        for regime in ['high', 'normal', 'low']:
            regime_subset = regime_data[regime_data['regime'] == regime]['value']
            if len(regime_subset) > 5:
                regime_volatilities[f'{regime}_vol'] = regime_subset.std()
                regime_volatilities[f'{regime}_count'] = len(regime_subset)
            else:
                regime_volatilities[f'{regime}_vol'] = np.nan
                regime_volatilities[f'{regime}_count'] = 0
        
        return regime_data, regime_volatilities
    
    def calculate_behavioral_volatility(self, animal_data: pd.DataFrame) -> Dict:
        """
        Calculate volatility specific to animal behavior patterns
        """
        behavioral_vol = {}
        
        if 'animal_count' in animal_data.columns:
            # Population volatility - sudden changes in animal numbers
            count_changes = animal_data['animal_count'].pct_change().abs()
            behavioral_vol['population_instability'] = count_changes.rolling(7).mean()
            
            # Clustering coefficient - how animals group vs spread out
            if len(animal_data) > 20:
                # Simulate clustering based on count variance
                clustering = animal_data['animal_count'].rolling(7).std() / animal_data['animal_count'].rolling(7).mean()
                behavioral_vol['clustering_volatility'] = clustering
        
        if 'activity_level' in animal_data.columns:
            # Activity volatility - erratic vs consistent behavior
            activity_vol = self.calculate_rolling_volatility(animal_data['activity_level'], window=7)
            behavioral_vol['activity_volatility'] = activity_vol
            
            # Circadian disruption - deviation from expected daily patterns
            if len(animal_data) > 24:
                hourly_pattern = animal_data.groupby(animal_data.index.hour)['activity_level'].mean()
                expected_pattern = np.tile(hourly_pattern.values, len(animal_data) // 24 + 1)[:len(animal_data)]
                circadian_disruption = np.abs(animal_data['activity_level'].values - expected_pattern)
                behavioral_vol['circadian_disruption'] = pd.Series(circadian_disruption, index=animal_data.index)
        
        if 'species_diversity' in animal_data.columns:
            # Biodiversity instability
            diversity_vol = self.calculate_rolling_volatility(animal_data['species_diversity'], window=7)
            behavioral_vol['biodiversity_volatility'] = diversity_vol
        
        return behavioral_vol
    
    def calculate_vegetation_volatility(self, vegetation_data: pd.DataFrame) -> Dict:
        """
        Calculate volatility specific to vegetation/phenology patterns
        """
        vegetation_vol = {}
        
        if 'greenness_index' in vegetation_data.columns:
            # Greenness volatility - sudden changes in vegetation health
            greenness_vol = self.calculate_rolling_volatility(vegetation_data['greenness_index'], window=14)
            vegetation_vol['greenness_volatility'] = greenness_vol
            
            # Phenological timing volatility - deviation from expected seasonal patterns
            doy = vegetation_data.index.dayofyear  # Day of year
            
            # Expected seasonal pattern (simplified sine wave)
            expected_seasonal = 0.35 + 0.15 * np.sin(2 * np.pi * (doy - 80) / 365)
            phenology_deviation = np.abs(vegetation_data['greenness_index'].values - expected_seasonal)
            vegetation_vol['phenological_stress'] = pd.Series(phenology_deviation, index=vegetation_data.index)
        
        if 'seasonal_change_rate' in vegetation_data.columns:
            # Growth rate volatility
            growth_vol = self.calculate_rolling_volatility(vegetation_data['seasonal_change_rate'], window=7)
            vegetation_vol['growth_rate_volatility'] = growth_vol
        
        return vegetation_vol
    
    def calculate_environmental_volatility(self, environmental_data: pd.DataFrame) -> Dict:
        """
        Calculate volatility in environmental conditions
        """
        env_vol = {}
        
        # Temperature volatility
        if 'temperature' in environmental_data.columns:
            temp_vol = self.calculate_rolling_volatility(environmental_data['temperature'], window=7)
            env_vol['temperature_volatility'] = temp_vol
            
            # Temperature shock indicator (rapid changes)
            temp_shocks = environmental_data['temperature'].diff().abs()
            env_vol['temperature_shocks'] = temp_shocks.rolling(3).max()
        
        # Pressure volatility
        if 'pressure' in environmental_data.columns:
            pressure_vol = self.calculate_rolling_volatility(environmental_data['pressure'], window=7)
            env_vol['pressure_volatility'] = pressure_vol
        
        # Humidity volatility
        if 'humidity' in environmental_data.columns:
            humidity_vol = self.calculate_rolling_volatility(environmental_data['humidity'], window=7)
            env_vol['humidity_volatility'] = humidity_vol
        
        # Weather uncertainty index (combination of multiple variables)
        weather_vars = [col for col in environmental_data.columns 
                       if col in ['temperature', 'pressure', 'humidity', 'wind_speed']]
        
        if len(weather_vars) >= 2:
            # Calculate correlation matrix stability
            rolling_corr_std = []
            for i in range(30, len(environmental_data)):
                window_data = environmental_data[weather_vars].iloc[i-30:i]
                corr_matrix = window_data.corr().values
                # Use standard deviation of correlation matrix as uncertainty measure
                corr_std = np.std(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                rolling_corr_std.append(corr_std)
            
            env_vol['weather_uncertainty'] = pd.Series(
                [np.nan] * 30 + rolling_corr_std, 
                index=environmental_data.index
            )
        
        return env_vol
    
    def create_bio_vix_index(self, biological_data: Dict[str, pd.DataFrame], 
                           weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Create a VIX-style volatility index for biological data
        """
        
        if weights is None:
            weights = {
                'behavioral': 0.3,
                'vegetation': 0.3,
                'environmental': 0.25,
                'regime': 0.15
            }
        
        all_volatilities = {}
        
        # Calculate all volatility components
        for data_type, data in biological_data.items():
            if 'animal' in data_type.lower() or 'wildlife' in data_type.lower():
                behavioral_vol = self.calculate_behavioral_volatility(data)
                for key, series in behavioral_vol.items():
                    all_volatilities[f'behavioral_{key}'] = series
                    
            elif 'vegetation' in data_type.lower() or 'plant' in data_type.lower():
                vegetation_vol = self.calculate_vegetation_volatility(data)
                for key, series in vegetation_vol.items():
                    all_volatilities[f'vegetation_{key}'] = series
                    
            elif 'environmental' in data_type.lower() or 'weather' in data_type.lower():
                env_vol = self.calculate_environmental_volatility(data)
                for key, series in env_vol.items():
                    all_volatilities[f'environmental_{key}'] = series
        
        # Combine volatilities into DataFrame
        volatility_df = pd.DataFrame(all_volatilities)
        
        if volatility_df.empty:
            print("No volatility data calculated")
            return pd.DataFrame()
        
        # Normalize all volatilities to 0-1 scale
        scaler = StandardScaler()
        normalized_vol = pd.DataFrame(
            scaler.fit_transform(volatility_df.fillna(0)),
            index=volatility_df.index,
            columns=volatility_df.columns
        )
        
        # Calculate component indices
        behavioral_cols = [col for col in normalized_vol.columns if col.startswith('behavioral_')]
        vegetation_cols = [col for col in normalized_vol.columns if col.startswith('vegetation_')]
        environmental_cols = [col for col in normalized_vol.columns if col.startswith('environmental_')]
        
        bio_vix_components = pd.DataFrame(index=normalized_vol.index)
        
        if behavioral_cols:
            bio_vix_components['behavioral_vix'] = normalized_vol[behavioral_cols].mean(axis=1)
        else:
            bio_vix_components['behavioral_vix'] = 0
            
        if vegetation_cols:
            bio_vix_components['vegetation_vix'] = normalized_vol[vegetation_cols].mean(axis=1)
        else:
            bio_vix_components['vegetation_vix'] = 0
            
        if environmental_cols:
            bio_vix_components['environmental_vix'] = normalized_vol[environmental_cols].mean(axis=1)
        else:
            bio_vix_components['environmental_vix'] = 0
        
        # Calculate regime volatility (variance of the combined index)
        if len(bio_vix_components.columns) > 0:
            temp_index = bio_vix_components.mean(axis=1)
            regime_vol = self.calculate_rolling_volatility(temp_index, window=20)
            bio_vix_components['regime_vix'] = regime_vol / regime_vol.std()  # Normalize
        else:
            bio_vix_components['regime_vix'] = 0
        
        # Calculate final BioVIX index
        bio_vix_components['bio_vix'] = (
            weights['behavioral'] * bio_vix_components['behavioral_vix'] +
            weights['vegetation'] * bio_vix_components['vegetation_vix'] +
            weights['environmental'] * bio_vix_components['environmental_vix'] +
            weights['regime'] * bio_vix_components['regime_vix']
        )
        
        # Scale to VIX-like range (0-100)
        bio_vix_components['bio_vix_scaled'] = (
            (bio_vix_components['bio_vix'] - bio_vix_components['bio_vix'].min()) /
            (bio_vix_components['bio_vix'].max() - bio_vix_components['bio_vix'].min()) * 100
        )
        
        self.bio_vix_components = bio_vix_components
        
        return bio_vix_components
    
    def calculate_volatility_clustering(self, volatility_series: pd.Series, 
                                      threshold_percentile: float = 0.8) -> Dict:
        """
        Detect volatility clustering (periods of high/low volatility persistence)
        """
        threshold = volatility_series.quantile(threshold_percentile)
        high_vol_periods = volatility_series > threshold
        
        # Find clusters of high volatility
        clusters = []
        in_cluster = False
        cluster_start = None
        
        for i, is_high_vol in enumerate(high_vol_periods):
            if is_high_vol and not in_cluster:
                # Start of new cluster
                cluster_start = i
                in_cluster = True
            elif not is_high_vol and in_cluster:
                # End of cluster
                clusters.append((cluster_start, i-1))
                in_cluster = False
        
        # Handle case where series ends in high volatility
        if in_cluster:
            clusters.append((cluster_start, len(high_vol_periods)-1))
        
        # Calculate cluster statistics
        if clusters:
            cluster_lengths = [end - start + 1 for start, end in clusters]
            avg_cluster_length = np.mean(cluster_lengths)
            max_cluster_length = max(cluster_lengths)
            cluster_frequency = len(clusters) / len(volatility_series) * 100  # clusters per 100 observations
        else:
            avg_cluster_length = 0
            max_cluster_length = 0
            cluster_frequency = 0
        
        return {
            'clusters': clusters,
            'avg_cluster_length': avg_cluster_length,
            'max_cluster_length': max_cluster_length,
            'cluster_frequency': cluster_frequency,
            'current_in_cluster': high_vol_periods.iloc[-1] if len(high_vol_periods) > 0 else False
        }
    
    def calculate_volatility_asymmetry(self, data: pd.Series, volatility: pd.Series) -> Dict:
        """
        Calculate asymmetry in volatility (leverage effect analog)
        """
        # Calculate returns/changes
        returns = data.pct_change().dropna()
        vol_aligned = volatility.loc[returns.index]
        
        # Split into positive and negative changes
        positive_changes = returns[returns > 0]
        negative_changes = returns[returns < 0]
        
        pos_vol = vol_aligned.loc[positive_changes.index]
        neg_vol = vol_aligned.loc[negative_changes.index]
        
        asymmetry_stats = {
            'positive_change_vol_mean': pos_vol.mean() if len(pos_vol) > 0 else np.nan,
            'negative_change_vol_mean': neg_vol.mean() if len(neg_vol) > 0 else np.nan,
            'volatility_asymmetry_ratio': (neg_vol.mean() / pos_vol.mean()) if len(pos_vol) > 0 and len(neg_vol) > 0 else np.nan
        }
        
        # Asymmetry ratio > 1 means negative changes have higher volatility (stress response)
        return asymmetry_stats
    
    def detect_volatility_regime_changes(self, volatility_series: pd.Series, 
                                       window: int = 30) -> pd.DataFrame:
        """
        Detect structural breaks in volatility patterns
        """
        regime_data = pd.DataFrame(index=volatility_series.index)
        regime_data['volatility'] = volatility_series
        
        # Rolling mean and std
        regime_data['vol_mean'] = volatility_series.rolling(window=window).mean()
        regime_data['vol_std'] = volatility_series.rolling(window=window).std()
        
        # Z-score for current volatility
        regime_data['vol_zscore'] = (
            (volatility_series - regime_data['vol_mean']) / regime_data['vol_std']
        )
        
        # Identify regime changes (significant deviations)
        regime_data['regime_change'] = np.abs(regime_data['vol_zscore']) > 2
        
        # Classify current regime
        recent_mean = regime_data['vol_mean'].iloc[-window:].mean()
        current_vol = volatility_series.iloc[-5:].mean()  # Last 5 observations
        
        if current_vol > recent_mean * 1.5:
            current_regime = 'high_volatility'
        elif current_vol < recent_mean * 0.5:
            current_regime = 'low_volatility'
        else:
            current_regime = 'normal_volatility'
        
        regime_data['current_regime'] = current_regime
        
        return regime_data
    
    def plot_bio_volatility_analysis(self, bio_vix_data: pd.DataFrame, 
                                   market_data: Optional[pd.DataFrame] = None):
        """
        Create comprehensive volatility visualization
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. BioVIX Components Over Time
        if not bio_vix_data.empty:
            components = ['behavioral_vix', 'vegetation_vix', 'environmental_vix', 'regime_vix']
            available_components = [col for col in components if col in bio_vix_data.columns]
            
            for component in available_components:
                axes[0, 0].plot(bio_vix_data.index, bio_vix_data[component], 
                              label=component.replace('_', ' ').title(), alpha=0.7)
            
            axes[0, 0].set_title('BioVIX Components Over Time')
            axes[0, 0].legend()
            axes[0, 0].set_ylabel('Volatility Index')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Overall BioVIX vs Market VIX (if available)
        if 'bio_vix_scaled' in bio_vix_data.columns:
            axes[0, 1].plot(bio_vix_data.index, bio_vix_data['bio_vix_scaled'], 
                          'g-', label='BioVIX', linewidth=2)
            
            if market_data is not None and 'VIX' in market_data.columns:
                # Align market data
                aligned_market = market_data.reindex(bio_vix_data.index, method='ffill')
                axes[0, 1].plot(aligned_market.index, aligned_market['VIX'], 
                              'r-', label='Market VIX', alpha=0.7)
            
            axes[0, 1].set_title('BioVIX vs Market Volatility')
            axes[0, 1].legend()
            axes[0, 1].set_ylabel('Volatility Index')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Volatility Distribution
        if 'bio_vix_scaled' in bio_vix_data.columns:
            axes[1, 0].hist(bio_vix_data['bio_vix_scaled'].dropna(), 
                          bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_title('BioVIX Distribution')
            axes[1, 0].set_xlabel('BioVIX Level')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add percentile lines
            percentiles = [10, 25, 50, 75, 90]
            for p in percentiles:
                value = bio_vix_data['bio_vix_scaled'].quantile(p/100)
                axes[1, 0].axvline(value, color='red', linestyle='--', alpha=0.5)
                axes[1, 0].text(value, axes[1, 0].get_ylim()[1]*0.8, f'{p}%', 
                              rotation=90, verticalalignment='bottom')
        
        # 4. Volatility Clustering Analysis
        if 'bio_vix_scaled' in bio_vix_data.columns:
            clustering_info = self.calculate_volatility_clustering(bio_vix_data['bio_vix_scaled'])
            
            # Plot volatility with cluster periods highlighted
            axes[1, 1].plot(bio_vix_data.index, bio_vix_data['bio_vix_scaled'], 
                          'b-', alpha=0.7, label='BioVIX')
            
            # Highlight high volatility clusters
            threshold = bio_vix_data['bio_vix_scaled'].quantile(0.8)
            axes[1, 1].axhline(threshold, color='red', linestyle='--', alpha=0.5, label='High Vol Threshold')
            
            for start_idx, end_idx in clustering_info['clusters']:
                start_date = bio_vix_data.index[start_idx]
                end_date = bio_vix_data.index[end_idx]
                axes[1, 1].axvspan(start_date, end_date, alpha=0.2, color='red')
            
            axes[1, 1].set_title('Volatility Clustering Analysis')
            axes[1, 1].legend()
            axes[1, 1].set_ylabel('BioVIX')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Regime Analysis
        if 'bio_vix_scaled' in bio_vix_data.columns:
            regime_data = self.detect_volatility_regime_changes(bio_vix_data['bio_vix_scaled'])
            
            axes[2, 0].plot(regime_data.index, regime_data['volatility'], 'b-', alpha=0.7, label='BioVIX')
            axes[2, 0].plot(regime_data.index, regime_data['vol_mean'], 'r-', alpha=0.7, label='Rolling Mean')
            
            # Highlight regime changes
            regime_changes = regime_data[regime_data['regime_change']]
            if not regime_changes.empty:
                axes[2, 0].scatter(regime_changes.index, regime_changes['volatility'], 
                                 color='red', s=50, label='Regime Changes', zorder=5)
            
            axes[2, 0].set_title('Volatility Regime Detection')
            axes[2, 0].legend()
            axes[2, 0].set_ylabel('BioVIX')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Current Volatility Dashboard
        if 'bio_vix_scaled' in bio_vix_data.columns:
            current_vix = bio_vix_data['bio_vix_scaled'].iloc[-1]
            mean_vix = bio_vix_data['bio_vix_scaled'].mean()
            std_vix = bio_vix_data['bio_vix_scaled'].std()
            
            # Create dashboard-style display
            axes[2, 1].bar(['Current', 'Mean', 'Mean+1σ', 'Mean+2σ'], 
                          [current_vix, mean_vix, mean_vix + std_vix, mean_vix + 2*std_vix],
                          color=['blue', 'green', 'orange', 'red'], alpha=0.7)
            
            axes[2, 1].set_title('Current BioVIX Level')
            axes[2, 1].set_ylabel('BioVIX Value')
            axes[2, 1].grid(True, alpha=0.3)
            
            # Add interpretation text
            if current_vix > mean_vix + 2*std_vix:
                interpretation = "EXTREME STRESS"
                color = 'red'
            elif current_vix > mean_vix + std_vix:
                interpretation = "HIGH STRESS"
                color = 'orange'
            elif current_vix < mean_vix - std_vix:
                interpretation = "LOW STRESS"
                color = 'green'
            else:
                interpretation = "NORMAL"
                color = 'blue'
            
            axes[2, 1].text(0.5, 0.9, interpretation, transform=axes[2, 1].transAxes,
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        plt.show()
    
    def generate_volatility_signals(self, bio_vix_data: pd.DataFrame) -> Dict:
        """
        Generate trading signals based on volatility patterns
        """
        if bio_vix_data.empty or 'bio_vix_scaled' not in bio_vix_data.columns:
            return {'error': 'No BioVIX data available'}
        
        current_vix = bio_vix_data['bio_vix_scaled'].iloc[-1]
        recent_vix = bio_vix_data['bio_vix_scaled'].tail(7).mean()
        historical_mean = bio_vix_data['bio_vix_scaled'].mean()
        historical_std = bio_vix_data['bio_vix_scaled'].std()
        
        # Calculate volatility trend
        vix_trend = bio_vix_data['bio_vix_scaled'].tail(5).diff().mean()
        
        # Volatility clustering analysis
        clustering_info = self.calculate_volatility_clustering(bio_vix_data['bio_vix_scaled'])
        
        # Generate signals
        signals = {}
        
        # Mean reversion signal
        if current_vix > historical_mean + 2 * historical_std:
            signals['mean_reversion'] = 'BUY'  # Extreme fear, expect reversion
        elif current_vix < historical_mean - historical_std:
            signals['mean_reversion'] = 'SELL'  # Complacency, expect volatility increase
        else:
            signals['mean_reversion'] = 'HOLD'
        
        # Trend following signal
        if vix_trend > 0.5:
            signals['trend_following'] = 'SELL'  # Rising volatility, risk-off
        elif vix_trend < -0.5:
            signals['trend_following'] = 'BUY'  # Falling volatility, risk-on
        else:
            signals['trend_following'] = 'HOLD'
        
        # Clustering signal
        if clustering_info['current_in_cluster']:
            signals['clustering'] = 'SELL'  # In high volatility cluster
        else:
            signals['clustering'] = 'BUY'   # Not in cluster
        
        # Regime signal
        regime_data = self.detect_volatility_regime_changes(bio_vix_data['bio_vix_scaled'])
        current_regime = regime_data['current_regime'].iloc[-1]
        
        if current_regime == 'high_volatility':
            signals['regime'] = 'SELL'
        elif current_regime == 'low_volatility':
            signals['regime'] = 'BUY'
        else:
            signals['regime'] = 'HOLD'
        
        # Consensus signal
        buy_signals = sum(1 for s in signals.values() if s == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s == 'SELL')
        
        if buy_signals > sell_signals:
            consensus = 'BUY'
        elif sell_signals > buy_signals:
            consensus = 'SELL'
        else:
            consensus = 'HOLD'
        
        return {
            'signals': signals,
            'consensus': consensus,
            'confidence': abs(buy_signals - sell_signals) / len(signals),
            'current_vix': current_vix,
            'historical_percentile': stats.percentileofscore(bio_vix_data['bio_vix_scaled'], current_vix),
            'regime': current_regime,
            'clustering_active': clustering_info['current_in_cluster'],
            'interpretation': self._interpret_vix_level(current_vix, historical_mean, historical_std)
        }
    
    def _interpret_vix_level(self, current_vix: float, mean_vix: float, std_vix: float) -> str:
        """Interpret current BioVIX level"""
        if current_vix > mean_vix + 2 * std_vix:
            return "EXTREME ECOSYSTEM STRESS - Major disruption likely"
        elif current_vix > mean_vix + std_vix:
            return "HIGH ECOSYSTEM STRESS - Significant uncertainty"
        elif current_vix < mean_vix - std_vix:
            return "LOW ECOSYSTEM STRESS - Stable conditions"
        else:
            return "NORMAL ECOSYSTEM STRESS - Baseline conditions"

# Demo function to showcase bio-volatility analysis
def demo_bio_volatility_analysis():
    """
    Demonstrate biological volatility analysis with simulated data
    """
    print("🧬 BIOLOGICAL VOLATILITY ANALYSIS DEMO 🧬\n")
    
    # Create simulated biological data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Simulate wildlife data with volatility clustering
    np.random.seed(42)
    n_days = len(dates)
    
    # Base patterns with regime changes
    regime_changes = [100, 200, 300]  # Days where volatility regime changes
    volatility_regimes = [0.1, 0.3, 0.15, 0.4]  # Different volatility levels
    
    wildlife_data = pd.DataFrame(index=dates)
    
    # Generate animal count with volatility clustering
    animal_counts = []
    current_vol = volatility_regimes[0]
    regime_idx = 0
    
    for i in range(n_days):
        # Check for regime change
        if regime_idx < len(regime_changes) and i >= regime_changes[regime_idx]:
            regime_idx += 1
            current_vol = volatility_regimes[regime_idx]
        
        # Generate count with current volatility
        if i == 0:
            base_count = 50
        else:
            # Add trend + seasonal + volatility
            seasonal = 10 * np.sin(2 * np.pi * i / 365)
            trend = 0.01 * i
            noise = np.random.normal(0, current_vol * 20)
            base_count = max(10, animal_counts[-1] + trend + seasonal + noise)
        
        animal_counts.append(base_count)
    
    wildlife_data['animal_count'] = animal_counts
    
    # Add other wildlife metrics
    wildlife_data['species_diversity'] = (
        8 + 3 * np.sin(2 * np.pi * np.arange(n_days) / 365) + 
        np.random.normal(0, 1, n_days)
    )
    
    wildlife_data['activity_level'] = (
        0.6 + 0.2 * np.sin(2 * np.pi * np.arange(n_days) / 365) +
        0.1 * np.random.normal(0, 1, n_days)
    )
    
    # Simulate vegetation data
    vegetation_data = pd.DataFrame(index=dates)
    
    # Greenness index with phenological patterns
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    vegetation_data['greenness_index'] = (
        0.35 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) +
        0.05 * np.random.normal(0, 1, n_days)
    )
    
    vegetation_data['seasonal_change_rate'] = vegetation_data['greenness_index'].diff()
    
    # Simulate environmental data
    environmental_data = pd.DataFrame(index=dates)
    environmental_data['temperature'] = (
        15 + 10 * np.sin(2 * np.pi * day_of_year / 365) +
        5 * np.random.normal(0, 1, n_days)
    )
    
    environmental_data['pressure'] = (
        1013 + 20 * np.random.normal(0, 1, n_days)
    )
    
    environmental_data['humidity'] = (
        60 + 20 * np.sin(2 * np.pi * day_of_year / 365) +
        10 * np.random.normal(0, 1, n_days)
    )
    
    # Initialize analyzer
    analyzer = BioVolatilityAnalyzer()
    
    # Create BioVIX index
    biological_data = {
        'wildlife': wildlife_data,
        'vegetation': vegetation_data,
        'environmental': environmental_data
    }
    
    print("1. Calculating BioVIX components...")
    bio_vix_data = analyzer.create_bio_vix_index(biological_data)
    
    if not bio_vix_data.empty:
        print(f"✓ Created BioVIX with {len(bio_vix_data)} observations")
        
        # Show current BioVIX statistics
        current_vix = bio_vix_data['bio_vix_scaled'].iloc[-1]
        mean_vix = bio_vix_data['bio_vix_scaled'].mean()
        std_vix = bio_vix_data['bio_vix_scaled'].std()
        
        print(f"\n=== BIOVIX STATISTICS ===")
        print(f"Current BioVIX: {current_vix:.2f}")
        print(f"Historical Mean: {mean_vix:.2f}")
        print(f"Historical Std: {std_vix:.2f}")
        print(f"Current Percentile: {stats.percentileofscore(bio_vix_data['bio_vix_scaled'], current_vix):.1f}%")
        
        # Analyze volatility clustering
        print("\n2. Analyzing volatility clustering...")
        clustering_info = analyzer.calculate_volatility_clustering(bio_vix_data['bio_vix_scaled'])
        
        print(f"✓ Found {len(clustering_info['clusters'])} high volatility clusters")
        print(f"Average cluster length: {clustering_info['avg_cluster_length']:.1f} days")
        print(f"Currently in cluster: {clustering_info['current_in_cluster']}")
        
        # Detect regime changes
        print("\n3. Detecting volatility regimes...")
        regime_data = analyzer.detect_volatility_regime_changes(bio_vix_data['bio_vix_scaled'])
        current_regime = regime_data['current_regime'].iloc[-1]
        
        print(f"✓ Current volatility regime: {current_regime}")
        
        # Generate trading signals
        print("\n4. Generating volatility-based trading signals...")
        signals = analyzer.generate_volatility_signals(bio_vix_data)
        
        print(f"✓ Generated signals with {signals['confidence']:.1%} confidence")
        print(f"Consensus: {signals['consensus']}")
        print(f"Interpretation: {signals['interpretation']}")
        
        # Show detailed signals
        print(f"\n=== DETAILED SIGNALS ===")
        for signal_type, signal in signals['signals'].items():
            print(f"{signal_type.replace('_', ' ').title()}: {signal}")
        
        # Create visualizations
        print("\n5. Creating volatility visualizations...")
        
        # Simulate some market data for comparison
        market_data = pd.DataFrame(index=dates)
        market_data['VIX'] = (
            20 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 100) +
            10 * np.random.normal(0, 1, n_days)
        )
        
        analyzer.plot_bio_volatility_analysis(bio_vix_data, market_data)
        
        print("\n✅ Bio-volatility analysis complete!")
        
        return analyzer, bio_vix_data, signals
    
    else:
        print("❌ Failed to create BioVIX index")
        return None, None, None

# Advanced volatility analysis functions
def calculate_volatility_term_structure(bio_vix_data: pd.DataFrame, 
                                      terms: List[int] = [7, 14, 30, 60]) -> pd.DataFrame:
    """
    Calculate volatility term structure (different time horizons)
    Similar to VIX term structure in options markets
    """
    if 'bio_vix_scaled' not in bio_vix_data.columns:
        return pd.DataFrame()
    
    term_structure = pd.DataFrame(index=bio_vix_data.index)
    
    for term in terms:
        if len(bio_vix_data) >= term:
            # Calculate forward-looking volatility for each term
            rolling_vol = bio_vix_data['bio_vix_scaled'].rolling(window=term).std()
            term_structure[f'bio_vix_{term}d'] = rolling_vol
    
    # Calculate term structure slope (contango vs backwardation)
    if len(terms) >= 2:
        short_term = f'bio_vix_{terms[0]}d'
        long_term = f'bio_vix_{terms[-1]}d'
        
        if short_term in term_structure.columns and long_term in term_structure.columns:
            term_structure['term_structure_slope'] = (
                term_structure[long_term] - term_structure[short_term]
            )
            
            # Positive slope = contango (normal), negative = backwardation (stress)
            term_structure['market_structure'] = term_structure['term_structure_slope'].apply(
                lambda x: 'contango' if x > 0 else 'backwardation' if x < 0 else 'flat'
            )
    
    return term_structure

def calculate_volatility_risk_premium(bio_vix_data: pd.DataFrame, 
                                    realized_window: int = 30) -> pd.DataFrame:
    """
    Calculate the difference between implied (BioVIX) and realized volatility
    """
    if 'bio_vix_scaled' not in bio_vix_data.columns:
        return pd.DataFrame()
    
    risk_premium_data = pd.DataFrame(index=bio_vix_data.index)
    
    # Use bio_vix as "implied" volatility
    risk_premium_data['implied_vol'] = bio_vix_data['bio_vix_scaled']
    
    # Calculate realized volatility from bio_vix changes
    bio_vix_returns = bio_vix_data['bio_vix_scaled'].pct_change()
    risk_premium_data['realized_vol'] = (
        bio_vix_returns.rolling(window=realized_window).std() * np.sqrt(252) * 100
    )
    
    # Risk premium = implied - realized
    risk_premium_data['volatility_risk_premium'] = (
        risk_premium_data['implied_vol'] - risk_premium_data['realized_vol']
    )
    
    # Positive risk premium = fear premium, negative = complacency
    risk_premium_data['market_sentiment'] = risk_premium_data['volatility_risk_premium'].apply(
        lambda x: 'fearful' if x > 5 else 'complacent' if x < -5 else 'neutral'
    )
    
    return risk_premium_data

if __name__ == "__main__":
    analyzer, bio_vix_data, signals = demo_bio_volatility_analysis()
    
    if bio_vix_data is not None:
        print("\n" + "="*60)
        print("ADVANCED VOLATILITY ANALYSIS")
        print("="*60)
        
        # Calculate term structure
        print("\n6. Calculating volatility term structure...")
        term_structure = calculate_volatility_term_structure(bio_vix_data)
        
        if not term_structure.empty and 'market_structure' in term_structure.columns:
            current_structure = term_structure['market_structure'].iloc[-1]
            print(f"✓ Current term structure: {current_structure}")
            
            if current_structure == 'backwardation':
                print("   → Indicates near-term stress/uncertainty")
            elif current_structure == 'contango':
                print("   → Indicates normal market conditions")
        
        # Calculate risk premium
        print("\n7. Calculating volatility risk premium...")
        risk_premium = calculate_volatility_risk_premium(bio_vix_data)
        
        if not risk_premium.empty and 'market_sentiment' in risk_premium.columns:
            current_sentiment = risk_premium['market_sentiment'].iloc[-1]
            current_premium = risk_premium['volatility_risk_premium'].iloc[-1]
            print(f"✓ Current sentiment: {current_sentiment}")
            print(f"   Risk premium: {current_premium:.2f}")
            
            if current_sentiment == 'fearful':
                print("   → Ecosystem stress creating fear premium")
            elif current_sentiment == 'complacent':
                print("   → Low perceived risk, potential for surprise")
        
        print(f"\n=== FINAL RECOMMENDATIONS ===")
        print(f"BioVIX Level: {bio_vix_data['bio_vix_scaled'].iloc[-1]:.1f}")
        print(f"Trading Signal: {signals['consensus']}")
        print(f"Confidence: {signals['confidence']:.1%}")
        print(f"Key Insight: {signals['interpretation']}")
        
        print(f"\n💡 Next Steps:")
        print(f"1. Monitor BioVIX for regime changes")
        print(f"2. Use volatility clustering for timing")
        print(f"3. Watch term structure for stress signals")
        print(f"4. Apply mean reversion strategies during extreme levels")
