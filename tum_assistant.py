#!/usr/bin/env python3
"""
TUM Production Forecasting Assistant
Analyzes historical TUM data and forecasts future production.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats

# Default data path
DEFAULT_DATA_PATH = r"\\p1fs002\CSI - Technical Services\TUM Data for Accountants FY27"

# Site configurations with their CSV filenames
SITES = {
    "mt_whaleback": "Mt Whaleback.csv",
    "area_c": "Area C.csv",
    "roy_hill": "Roy Hill Bravo.csv",
    "iron_valley": "Iron Valley.csv",
    "wodgina": "Wodgina.csv",
    "sanjiv_ridge": "Sanjiv Ridge.csv",
    "rod_ore": "Rod Ore.csv",
    "granites": "Granites.csv",
    "hope_downs_4": "Hope Downs 4.csv",
    "kcgm_main": "KCGM-Main Plant.csv",
    "kcgm_charlotte": "KCGM - Mt Charlotte.csv",
    "west_angelas_1": "West Angelas Plant 1.csv",
    "west_angelas_2": "West Angelas Plant 2.csv",
    "key_metrics": "Key Production Metrics.csv",
}


def parse_percentage(val):
    """Convert percentage string to float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val) if val <= 1 else float(val) / 100
    val = str(val).strip().replace('%', '')
    try:
        num = float(val)
        return num / 100 if num > 1 else num
    except ValueError:
        return np.nan


def parse_month(val):
    """Parse month string to datetime."""
    if pd.isna(val):
        return None
    val = str(val).strip()

    # Handle "Jul-24" format
    for fmt in ['%b-%y', '%B-%y', '%b-%Y', '%B-%Y']:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def load_site_data(site_key: str, data_path: str = DEFAULT_DATA_PATH) -> Optional[pd.DataFrame]:
    """Load and normalize data for a specific site."""
    if site_key not in SITES:
        print(f"Unknown site: {site_key}")
        return None

    filepath = Path(data_path) / SITES[site_key]
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Remove empty rows
    df = df.dropna(how='all')

    # Standardize column names
    col_mapping = {
        'Crushed Tonnes': 'tonnes',
        'Tonnes': 'tonnes',
        'Run Hours': 'run_hours',
        'TPH': 'tph',
        'Availability': 'availability',
        'Utilisation': 'utilisation',
        'Effective Utilisation': 'effective_utilisation',
        'Planned Maint': 'planned_maint',
        'Unplanned': 'unplanned',
        'Internal Delays': 'internal_delays',
        'External Delays': 'external_delays',
        'Lump Tonnes': 'lump_tonnes',
        'Fines Tonnes': 'fines_tonnes',
        'Sum of LUMP': 'lump_tonnes',
        'Sum of FINES': 'fines_tonnes',
    }

    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # Parse month column
    if 'Month' in df.columns:
        df['date'] = df['Month'].apply(parse_month)
        if df['date'].isna().all() and 'Year' in df.columns:
            # Handle Year, Month as separate columns
            df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B', errors='coerce')

    # Parse percentage columns
    pct_cols = ['availability', 'utilisation', 'effective_utilisation']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_percentage)

    # Ensure numeric columns
    numeric_cols = ['tonnes', 'run_hours', 'tph', 'planned_maint', 'unplanned',
                    'internal_delays', 'external_delays', 'lump_tonnes', 'fines_tonnes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by date
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    df['site'] = site_key
    return df


def load_all_sites(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load data from all sites into a single DataFrame."""
    all_data = []
    for site_key in SITES:
        if site_key == 'key_metrics':
            continue
        df = load_site_data(site_key, data_path)
        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate summary statistics for a site."""
    stats_dict = {}

    numeric_cols = ['tonnes', 'run_hours', 'tph', 'availability', 'utilisation',
                    'effective_utilisation', 'lump_tonnes', 'fines_tonnes']

    for col in numeric_cols:
        if col in df.columns and df[col].notna().any():
            stats_dict[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'last_3m_avg': df[col].tail(3).mean(),
                'last_6m_avg': df[col].tail(6).mean(),
            }

    return stats_dict


def forecast_linear(series: pd.Series, periods: int = 6) -> tuple:
    """Simple linear regression forecast."""
    valid_data = series.dropna()
    if len(valid_data) < 3:
        return None, None, None

    x = np.arange(len(valid_data))
    y = valid_data.values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Forecast future periods
    future_x = np.arange(len(valid_data), len(valid_data) + periods)
    forecast = slope * future_x + intercept

    # Confidence interval (95%)
    confidence = 1.96 * std_err * np.sqrt(1 + 1/len(valid_data) +
                                           (future_x - x.mean())**2 / ((x - x.mean())**2).sum())

    return forecast, confidence, r_value**2


def forecast_moving_average(series: pd.Series, periods: int = 6, window: int = 3) -> np.ndarray:
    """Moving average forecast with trend adjustment."""
    valid_data = series.dropna()
    if len(valid_data) < window:
        return None

    # Calculate moving average
    ma = valid_data.rolling(window=window).mean()

    # Calculate trend from last few MAs
    recent_ma = ma.tail(window).values
    if len(recent_ma) >= 2:
        trend = (recent_ma[-1] - recent_ma[0]) / (len(recent_ma) - 1)
    else:
        trend = 0

    # Forecast with trend
    last_ma = ma.iloc[-1]
    forecast = [last_ma + trend * (i + 1) for i in range(periods)]

    return np.array(forecast)


def forecast_seasonal(series: pd.Series, periods: int = 6) -> np.ndarray:
    """Seasonal forecast using year-over-year patterns."""
    valid_data = series.dropna()
    if len(valid_data) < 12:
        # Not enough data for seasonal analysis
        return forecast_moving_average(series, periods)

    # Calculate seasonal indices (monthly patterns)
    n_complete_years = len(valid_data) // 12
    if n_complete_years < 1:
        return forecast_moving_average(series, periods)

    # Use last complete year's seasonal pattern
    seasonal_pattern = valid_data.tail(12).values
    overall_avg = valid_data.mean()

    # Calculate seasonal indices
    monthly_avg = []
    for m in range(12):
        month_values = valid_data.iloc[m::12]
        monthly_avg.append(month_values.mean() if len(month_values) > 0 else overall_avg)

    seasonal_indices = np.array(monthly_avg) / overall_avg

    # Apply recent trend
    recent_avg = valid_data.tail(6).mean()
    trend_factor = recent_avg / overall_avg

    # Generate forecast
    start_month = len(valid_data) % 12
    forecast = []
    for i in range(periods):
        month_idx = (start_month + i) % 12
        forecast.append(recent_avg * seasonal_indices[month_idx])

    return np.array(forecast)


def generate_forecast(df: pd.DataFrame, periods: int = 6, method: str = 'ensemble') -> pd.DataFrame:
    """Generate production forecast for a site."""
    if df.empty or 'tonnes' not in df.columns:
        return pd.DataFrame()

    results = []

    # Get last date
    if 'date' in df.columns and df['date'].notna().any():
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                      periods=periods, freq='MS')
    else:
        future_dates = [f"Period +{i+1}" for i in range(periods)]

    # Forecast key metrics
    metrics_to_forecast = ['tonnes', 'tph', 'availability', 'utilisation',
                           'effective_utilisation', 'lump_tonnes', 'fines_tonnes']

    for metric in metrics_to_forecast:
        if metric not in df.columns or df[metric].isna().all():
            continue

        series = df[metric]

        if method == 'linear':
            forecast, confidence, r2 = forecast_linear(series, periods)
        elif method == 'moving_average':
            forecast = forecast_moving_average(series, periods)
            confidence, r2 = None, None
        elif method == 'seasonal':
            forecast = forecast_seasonal(series, periods)
            confidence, r2 = None, None
        else:  # ensemble
            # Combine methods
            linear_fc, _, r2 = forecast_linear(series, periods)
            ma_fc = forecast_moving_average(series, periods)
            seasonal_fc = forecast_seasonal(series, periods)

            forecasts = [f for f in [linear_fc, ma_fc, seasonal_fc] if f is not None]
            if forecasts:
                forecast = np.mean(forecasts, axis=0)
                confidence = np.std(forecasts, axis=0) * 1.96
            else:
                forecast, confidence = None, None

        if forecast is not None:
            for i, date in enumerate(future_dates):
                result = {
                    'date': date,
                    'metric': metric,
                    'forecast': forecast[i],
                    'method': method,
                }
                if confidence is not None:
                    result['lower_bound'] = forecast[i] - confidence[i] if isinstance(confidence, np.ndarray) else forecast[i] - confidence
                    result['upper_bound'] = forecast[i] + confidence[i] if isinstance(confidence, np.ndarray) else forecast[i] + confidence
                results.append(result)

    return pd.DataFrame(results)


def export_to_excel(df: pd.DataFrame, forecast_df: pd.DataFrame, stats: dict,
                    output_path: str, site_name: str):
    """Export analysis results to Excel."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Historical data
        df.to_excel(writer, sheet_name='Historical Data', index=False)

        # Forecast
        if not forecast_df.empty:
            # Pivot forecast for easier reading
            forecast_pivot = forecast_df.pivot(index='date', columns='metric', values='forecast')
            forecast_pivot.to_excel(writer, sheet_name='Forecast')

            # Full forecast details
            forecast_df.to_excel(writer, sheet_name='Forecast Details', index=False)

        # Statistics
        stats_rows = []
        for metric, values in stats.items():
            for stat_name, stat_value in values.items():
                stats_rows.append({
                    'Metric': metric,
                    'Statistic': stat_name,
                    'Value': stat_value
                })
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        # Summary sheet
        summary_data = {
            'Item': ['Site', 'Data Points', 'Date Range', 'Export Date'],
            'Value': [
                site_name,
                len(df),
                f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

    print(f"Exported to: {output_path}")


def print_summary(df: pd.DataFrame, site_name: str):
    """Print summary statistics to console."""
    print(f"\n{'='*60}")
    print(f"TUM Summary: {site_name}")
    print(f"{'='*60}")

    if df.empty:
        print("No data available.")
        return

    print(f"\nData Period: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "")
    print(f"Total Records: {len(df)}")

    if 'tonnes' in df.columns:
        print(f"\n--- Production (Tonnes) ---")
        print(f"  Total:       {df['tonnes'].sum():,.0f}")
        print(f"  Monthly Avg: {df['tonnes'].mean():,.0f}")
        print(f"  Last 3M Avg: {df['tonnes'].tail(3).mean():,.0f}")
        print(f"  Min:         {df['tonnes'].min():,.0f}")
        print(f"  Max:         {df['tonnes'].max():,.0f}")

    if 'tph' in df.columns:
        print(f"\n--- Throughput (TPH) ---")
        print(f"  Average:     {df['tph'].mean():,.1f}")
        print(f"  Last 3M Avg: {df['tph'].tail(3).mean():,.1f}")

    if 'availability' in df.columns:
        print(f"\n--- Availability ---")
        print(f"  Average:     {df['availability'].mean()*100:.1f}%")
        print(f"  Last 3M Avg: {df['availability'].tail(3).mean()*100:.1f}%")

    if 'effective_utilisation' in df.columns:
        print(f"\n--- Effective Utilisation ---")
        print(f"  Average:     {df['effective_utilisation'].mean()*100:.1f}%")
        print(f"  Last 3M Avg: {df['effective_utilisation'].tail(3).mean()*100:.1f}%")


def print_forecast(forecast_df: pd.DataFrame):
    """Print forecast results to console."""
    if forecast_df.empty:
        print("\nNo forecast available.")
        return

    print(f"\n{'='*60}")
    print("Production Forecast")
    print(f"{'='*60}")

    # Pivot for display
    metrics = ['tonnes', 'tph', 'availability', 'utilisation']

    for metric in metrics:
        metric_data = forecast_df[forecast_df['metric'] == metric]
        if metric_data.empty:
            continue

        print(f"\n--- {metric.replace('_', ' ').title()} ---")
        for _, row in metric_data.iterrows():
            date_str = row['date'].strftime('%b-%Y') if hasattr(row['date'], 'strftime') else str(row['date'])
            value = row['forecast']

            if metric in ['availability', 'utilisation', 'effective_utilisation']:
                print(f"  {date_str}: {value*100:.1f}%")
            elif metric == 'tonnes':
                print(f"  {date_str}: {value:,.0f}")
            else:
                print(f"  {date_str}: {value:,.1f}")


def list_sites():
    """List available sites."""
    print("\nAvailable Sites:")
    print("-" * 40)
    for key, filename in SITES.items():
        print(f"  {key:20s} -> {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="TUM Production Forecasting Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tum_assistant.py --list                     # List available sites
  python tum_assistant.py --site mt_whaleback        # Analyze Mt Whaleback
  python tum_assistant.py --site area_c --forecast 6 # 6-month forecast for Area C
  python tum_assistant.py --site roy_hill --export   # Export Roy Hill to Excel
  python tum_assistant.py --all --export             # Export all sites
        """
    )

    parser.add_argument('--site', '-s', type=str, help='Site to analyze (use --list to see options)')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all sites')
    parser.add_argument('--list', '-l', action='store_true', help='List available sites')
    parser.add_argument('--forecast', '-f', type=int, default=6, help='Forecast periods (months, default: 6)')
    parser.add_argument('--method', '-m', type=str, default='ensemble',
                        choices=['linear', 'moving_average', 'seasonal', 'ensemble'],
                        help='Forecasting method (default: ensemble)')
    parser.add_argument('--export', '-e', action='store_true', help='Export to Excel')
    parser.add_argument('--output', '-o', type=str, help='Output directory for exports')
    parser.add_argument('--data-path', '-d', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to TUM data files')

    args = parser.parse_args()

    if args.list:
        list_sites()
        return

    if not args.site and not args.all:
        parser.print_help()
        print("\nError: Please specify --site or --all")
        return

    # Set output directory
    output_dir = args.output or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    sites_to_process = list(SITES.keys()) if args.all else [args.site]

    for site_key in sites_to_process:
        if site_key == 'key_metrics' and args.all:
            continue

        print(f"\nProcessing: {site_key}")

        # Load data
        df = load_site_data(site_key, args.data_path)
        if df is None or df.empty:
            print(f"  No data found for {site_key}")
            continue

        # Calculate statistics
        stats = calculate_statistics(df)

        # Generate forecast
        forecast_df = generate_forecast(df, periods=args.forecast, method=args.method)

        # Print summary
        print_summary(df, site_key)

        if not forecast_df.empty:
            print_forecast(forecast_df)

        # Export if requested
        if args.export:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f"TUM_Forecast_{site_key}_{timestamp}.xlsx")
            export_to_excel(df, forecast_df, stats, output_file, site_key)

    print(f"\n{'='*60}")
    print("Analysis complete.")


if __name__ == '__main__':
    main()
