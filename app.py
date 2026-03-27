"""
TUM Production Forecasting Assistant - Web UI
Local web-based interface for TUM analysis and forecasting.
With interactive scenario modeling for contract changes.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="TUM Forecasting Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default data path
DEFAULT_DATA_PATH = r"\\p1fs002\CSI - Technical Services\TUM Data for Accountants FY27"

# Site configurations
SITES = {
    "west_angelas_1": ("West Angelas Plant 1", "West Angelas Plant 1.csv"),
    "west_angelas_2": ("West Angelas Plant 2", "West Angelas Plant 2.csv"),
    "mt_whaleback": ("Mt Whaleback", "Mt Whaleback.csv"),
    "area_c": ("Area C", "Area C.csv"),
    "roy_hill": ("Roy Hill Bravo", "Roy Hill Bravo.csv"),
    "iron_valley": ("Iron Valley", "Iron Valley.csv"),
    "wodgina": ("Wodgina", "Wodgina.csv"),
    "sanjiv_ridge": ("Sanjiv Ridge", "Sanjiv Ridge.csv"),
    "rod_ore": ("Rod Ore", "Rod Ore.csv"),
    "granites": ("Granites", "Granites.csv"),
    "hope_downs_4": ("Hope Downs 4", "Hope Downs 4.csv"),
    "kcgm_main": ("KCGM Main Plant", "KCGM-Main Plant.csv"),
    "kcgm_charlotte": ("KCGM Mt Charlotte", "KCGM - Mt Charlotte.csv"),
}

HISTORICAL_FILES = {
    "west_angelas_1": "West Angelas Plant 1 - Historical.csv",
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


@st.cache_data
def load_site_data(site_key: str, data_path: str = DEFAULT_DATA_PATH):
    """Load data for a specific site."""
    if site_key not in SITES:
        return None, None

    site_name, filename = SITES[site_key]
    filepath = Path(data_path) / filename

    if not filepath.exists():
        return None, site_name

    df = pd.read_csv(filepath, encoding='utf-8-sig')  # Handle BOM
    df = df.dropna(how='all')

    # Drop any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Normalize column names - handle 'Crushed Tonnes' variant
    if 'Crushed Tonnes' in df.columns and 'Tonnes' not in df.columns:
        df = df.rename(columns={'Crushed Tonnes': 'Tonnes'})

    # Parse percentage columns
    pct_cols = ['Availability', 'Utilisation', 'Effective Utilisation']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_percentage)

    return df, site_name


@st.cache_data
def load_historical_data(site_key: str, data_path: str = DEFAULT_DATA_PATH):
    """Load historical data if available."""
    if site_key not in HISTORICAL_FILES:
        return None

    filepath = Path(data_path) / HISTORICAL_FILES[site_key]
    if not filepath.exists():
        return None

    df = pd.read_csv(filepath, encoding='utf-8-sig')  # Handle BOM
    df = df.dropna(how='all')

    # Drop any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Normalize column names - handle 'Crushed Tonnes' variant
    if 'Crushed Tonnes' in df.columns and 'Tonnes' not in df.columns:
        df = df.rename(columns={'Crushed Tonnes': 'Tonnes'})

    pct_cols = ['Availability', 'Utilisation', 'Effective Utilisation']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_percentage)

    return df


def apply_scenario_adjustments(df, tph_adjustment, availability_adjustment, utilisation_adjustment):
    """Apply scenario adjustments to create modified forecast."""
    df_adjusted = df.copy()

    # Apply TPH adjustment (percentage change)
    if 'TPH' in df_adjusted.columns:
        df_adjusted['TPH'] = df_adjusted['TPH'] * (1 + tph_adjustment / 100)

    # Apply availability adjustment (absolute change in percentage points)
    if 'Availability' in df_adjusted.columns:
        df_adjusted['Availability'] = np.clip(
            df_adjusted['Availability'] + (availability_adjustment / 100),
            0, 1
        )

    # Apply utilisation adjustment (absolute change in percentage points)
    if 'Utilisation' in df_adjusted.columns:
        df_adjusted['Utilisation'] = np.clip(
            df_adjusted['Utilisation'] + (utilisation_adjustment / 100),
            0, 1
        )

    # Recalculate Effective Utilisation
    if 'Availability' in df_adjusted.columns and 'Utilisation' in df_adjusted.columns:
        df_adjusted['Effective Utilisation'] = df_adjusted['Availability'] * df_adjusted['Utilisation']

    # Recalculate Tonnes based on TPH and Run Hours
    # Tonnes = TPH * Run Hours
    if 'TPH' in df_adjusted.columns and 'Run Hours' in df_adjusted.columns:
        df_adjusted['Tonnes'] = df_adjusted['TPH'] * df_adjusted['Run Hours']

    return df_adjusted


def calculate_required_changes(current_tonnes, target_tonnes, df):
    """Calculate what changes are needed to meet target."""
    if current_tonnes == 0:
        return {}

    ratio = target_tonnes / current_tonnes

    avg_tph = df['TPH'].mean()
    avg_availability = df['Availability'].mean()
    avg_utilisation = df['Utilisation'].mean()
    total_run_hours = df['Run Hours'].sum()

    # Option A: Increase TPH only
    required_tph = avg_tph * ratio
    tph_change_pct = (ratio - 1) * 100

    # Option B: Increase Run Hours only
    required_run_hours = total_run_hours * ratio
    hours_change_pct = (ratio - 1) * 100

    # Option C: Balanced approach (split between TPH and hours)
    balanced_factor = np.sqrt(ratio)
    balanced_tph = avg_tph * balanced_factor
    balanced_hours_factor = balanced_factor

    return {
        'ratio': ratio,
        'tph_only': {
            'required_tph': required_tph,
            'change_pct': tph_change_pct,
        },
        'hours_only': {
            'required_hours': required_run_hours,
            'change_pct': hours_change_pct,
        },
        'balanced': {
            'tph_change_pct': (balanced_factor - 1) * 100,
            'hours_change_pct': (balanced_factor - 1) * 100,
        }
    }


def forecast_ensemble(series: pd.Series, periods: int = 6):
    """Generate ensemble forecast."""
    valid_data = series.dropna()
    if len(valid_data) < 3:
        return None

    x = np.arange(len(valid_data))
    y = valid_data.values
    slope, intercept = np.polyfit(x, y, 1)
    linear_fc = slope * np.arange(len(valid_data), len(valid_data) + periods) + intercept

    window = min(3, len(valid_data))
    ma = valid_data.rolling(window=window).mean()
    last_ma = ma.iloc[-1]
    trend = (ma.iloc[-1] - ma.iloc[-window]) / window if len(ma) >= window else 0
    ma_fc = np.array([last_ma + trend * (i + 1) for i in range(periods)])

    forecast = (linear_fc + ma_fc) / 2
    return forecast


def create_html_report(df_forecast, df_historical, site_name, contract_tonnes, scenario_params, lump_split=0.4):
    """Create HTML report mimicking CSI Mining Services TUM Report format."""
    from datetime import datetime
    import base64

    total_tonnes = df_forecast['Tonnes'].sum()
    avg_availability = df_forecast['Availability'].mean() * 100
    avg_utilisation = df_forecast['Utilisation'].mean() * 100
    total_run_hours = df_forecast['Run Hours'].sum()
    avg_tph = df_forecast['TPH'].mean()

    gap = contract_tonnes - total_tonnes
    gap_status = "ON TRACK" if abs(gap) < contract_tonnes * 0.03 else ("SHORTFALL" if gap > 0 else "SURPLUS")
    gap_message = "Forecast meets contract requirements." if gap_status == "ON TRACK" else (
        f"Forecast is {abs(gap):,.0f} tonnes below target. Review operational parameters." if gap > 0 else
        f"Forecast exceeds contract requirements. Maintain current operational parameters."
    )

    # Calculate unavailable hours
    planned_maint = df_forecast['Planned Maint'].sum() if 'Planned Maint' in df_forecast.columns else 0
    unplanned = df_forecast['Unplanned'].sum() if 'Unplanned' in df_forecast.columns else 0
    internal_delays = df_forecast['Internal Delays'].sum() if 'Internal Delays' in df_forecast.columns else 0
    external_delays = df_forecast['External Delays'].sum() if 'External Delays' in df_forecast.columns else 0

    # Build forecast table rows
    forecast_rows = ""
    for _, row in df_forecast.iterrows():
        month = row.get('Month', '')
        tonnes = row.get('Tonnes', 0)
        tph = row.get('TPH', 0)
        lump = tonnes * lump_split
        fines = tonnes * (1 - lump_split)
        run_hrs = row.get('Run Hours', 0)
        avail = row.get('Availability', 0) * 100
        util = row.get('Utilisation', 0) * 100
        pm = row.get('Planned Maint', 0)
        upm = row.get('Unplanned', 0)
        intd = row.get('Internal Delays', 0)
        extd = row.get('External Delays', 0)

        forecast_rows += f"""
        <tr>
            <td>{month}</td>
            <td>{tonnes:,.0f}</td>
            <td>{tph:,.0f}</td>
            <td>{lump:,.0f}</td>
            <td>{fines:,.0f}</td>
            <td>{run_hrs:,.0f}</td>
            <td>{avail:.1f}%</td>
            <td>{util:.1f}%</td>
            <td>{pm:.0f}</td>
            <td>{upm:.0f}</td>
            <td>{intd:.0f}</td>
            <td>{extd:.0f}</td>
        </tr>"""

    # Build historical table rows if available
    historical_rows = ""
    if df_historical is not None and len(df_historical) > 0:
        for _, row in df_historical.iterrows():
            month = row.get('Month', '')
            tonnes = row.get('Tonnes', 0)
            tph = row.get('TPH', 0)
            run_hrs = row.get('Run Hours', 0)
            avail = row.get('Availability', 0) * 100
            util = row.get('Utilisation', 0) * 100
            pm = row.get('Planned Maint', 0)
            upm = row.get('Unplanned', 0)
            intd = row.get('Internal Delays', 0)
            extd = row.get('External Delays', 0)

            historical_rows += f"""
            <tr>
                <td>{month}</td>
                <td>{tonnes:,.0f}</td>
                <td>{tph:,.0f}</td>
                <td>{run_hrs:,.0f}</td>
                <td>{avail:.1f}%</td>
                <td>{util:.1f}%</td>
                <td>{pm:.0f}</td>
                <td>{upm:.0f}</td>
                <td>{intd:.0f}</td>
                <td>{extd:.0f}</td>
            </tr>"""

    historical_section = ""
    if historical_rows:
        hist_total = df_historical['Tonnes'].sum()
        hist_avg_avail = df_historical['Availability'].mean() * 100
        hist_avg_util = df_historical['Utilisation'].mean() * 100
        historical_section = f"""
        <div class="page-break"></div>
        <div class="section-title">Detailed Monthly Historical Breakdown</div>
        <table>
            <thead>
                <tr>
                    <th>Month</th>
                    <th>Tonnes</th>
                    <th>TPH</th>
                    <th>Run Hours</th>
                    <th>Availability</th>
                    <th>Utilisation</th>
                    <th>Planned Maint</th>
                    <th>Unplanned</th>
                    <th>Internal</th>
                    <th>External</th>
                </tr>
            </thead>
            <tbody>
                <tr class="total-row">
                    <td><strong>Total/Avg</strong></td>
                    <td><strong>{hist_total:,.0f}</strong></td>
                    <td><strong>{df_historical['TPH'].mean():,.0f}</strong></td>
                    <td><strong>{df_historical['Run Hours'].sum():,.0f}</strong></td>
                    <td><strong>{hist_avg_avail:.1f}%</strong></td>
                    <td><strong>{hist_avg_util:.1f}%</strong></td>
                    <td><strong>{df_historical['Planned Maint'].sum():,.0f}</strong></td>
                    <td><strong>{df_historical['Unplanned'].sum():,.0f}</strong></td>
                    <td><strong>{df_historical['Internal Delays'].sum():,.0f}</strong></td>
                    <td><strong>{df_historical['External Delays'].sum():,.0f}</strong></td>
                </tr>
                {historical_rows}
            </tbody>
        </table>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>CSI TUM Report - {site_name}</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a1a; color: #fff; }}
            .header {{ background: #1a1a1a; padding: 30px 40px; border-bottom: 3px solid #333; }}
            .header-title {{ font-size: 28px; font-weight: bold; color: #fff; }}
            .header-subtitle {{ font-size: 14px; color: #888; margin-top: 5px; }}
            .header-right {{ float: right; text-align: right; }}
            .project-name {{ font-size: 18px; color: #fff; }}
            .generated-date {{ font-size: 12px; color: #888; }}
            .content {{ padding: 30px 40px; background: #1a1a1a; }}
            .kpi-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .kpi-card {{ flex: 1; background: #2a2a2a; border-radius: 8px; padding: 20px; text-align: center; }}
            .kpi-label {{ font-size: 12px; color: #888; text-transform: uppercase; margin-bottom: 10px; }}
            .kpi-value {{ font-size: 32px; font-weight: bold; color: #4CAF50; }}
            .kpi-value.blue {{ color: #2196F3; }}
            .kpi-value.orange {{ color: #FF9800; }}
            .kpi-value.purple {{ color: #9C27B0; }}
            .section-title {{ font-size: 20px; font-weight: bold; margin: 30px 0 15px 0; color: #fff; }}
            .assumptions-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #2a2a2a; padding: 20px; border-radius: 8px; }}
            .assumption-item {{ font-size: 13px; }}
            .assumption-label {{ color: #888; }}
            .assumption-value {{ color: #fff; font-weight: bold; }}
            .gap-box {{ background: #2a2a2a; border-radius: 8px; padding: 20px; margin: 20px 0; }}
            .gap-status {{ display: inline-block; padding: 8px 16px; border-radius: 4px; font-weight: bold; }}
            .gap-status.on-track {{ background: #1b5e20; color: #4CAF50; }}
            .gap-status.shortfall {{ background: #b71c1c; color: #f44336; }}
            .gap-status.surplus {{ background: #1565c0; color: #2196F3; }}
            .gap-detail {{ margin-left: 20px; display: inline-block; color: #888; }}
            .gap-message {{ margin-top: 10px; color: #aaa; font-size: 14px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 12px; }}
            th {{ background: #333; color: #fff; padding: 10px 8px; text-align: left; font-weight: bold; }}
            td {{ padding: 8px; border-bottom: 1px solid #333; color: #ddd; }}
            tr:hover {{ background: #2a2a2a; }}
            .total-row {{ background: #333; font-weight: bold; }}
            .total-row td {{ color: #fff; }}
            .page-break {{ page-break-before: always; margin-top: 40px; }}
            .methodology {{ background: #2a2a2a; padding: 25px; border-radius: 8px; }}
            .methodology h3 {{ color: #4CAF50; margin: 15px 0 8px 0; font-size: 14px; }}
            .methodology p {{ color: #aaa; font-size: 13px; margin-left: 15px; line-height: 1.6; }}
            .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; border-top: 1px solid #333; margin-top: 40px; }}
            @media print {{
                body {{ background: #fff; color: #000; }}
                .header {{ background: #333; }}
                .kpi-card, .assumptions-grid, .gap-box, .methodology {{ background: #f5f5f5; }}
                th {{ background: #333; }}
                td {{ border-bottom: 1px solid #ddd; color: #333; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-right">
                <div class="project-name">Project: {site_name}</div>
                <div class="generated-date">Generated: {datetime.now().strftime('%d/%m/%Y')}</div>
            </div>
            <div class="header-title">CSI MINING SERVICES</div>
            <div class="header-subtitle">TIME USAGE MODEL V1.0</div>
        </div>

        <div class="content">
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">Total Production</div>
                    <div class="kpi-value">{total_tonnes:,.0f} t</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg Availability</div>
                    <div class="kpi-value blue">{avg_availability:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg Utilisation</div>
                    <div class="kpi-value orange">{avg_utilisation:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Run Hours</div>
                    <div class="kpi-value purple">{total_run_hours:,.0f} hrs</div>
                </div>
            </div>

            <div class="section-title">Model Assumptions</div>
            <div class="assumptions-grid">
                <div class="assumption-item">
                    <span class="assumption-label">Site:</span><br>
                    <span class="assumption-value">{site_name}</span>
                </div>
                <div class="assumption-item">
                    <span class="assumption-label">Target TPH:</span><br>
                    <span class="assumption-value">{avg_tph:,.0f}</span>
                </div>
                <div class="assumption-item">
                    <span class="assumption-label">Contract Tonnes:</span><br>
                    <span class="assumption-value">{contract_tonnes:,.0f}</span>
                </div>
                <div class="assumption-item">
                    <span class="assumption-label">Forecast Year:</span><br>
                    <span class="assumption-value">FY27</span>
                </div>
                <div class="assumption-item">
                    <span class="assumption-label">Operation:</span><br>
                    <span class="assumption-value">24h</span>
                </div>
                <div class="assumption-item">
                    <span class="assumption-label">Lump Split:</span><br>
                    <span class="assumption-value">{lump_split*100:.0f}%</span>
                </div>
                <div class="assumption-item">
                    <span class="assumption-label">Fines Split:</span><br>
                    <span class="assumption-value">{(1-lump_split)*100:.0f}%</span>
                </div>
                <div class="assumption-item">
                    <span class="assumption-label">Commodity:</span><br>
                    <span class="assumption-value">Iron Ore</span>
                </div>
            </div>

            <div class="section-title">Scenario & Gap Analysis</div>
            <div class="gap-box">
                <span class="gap-status {'on-track' if gap_status == 'ON TRACK' else 'shortfall' if gap_status == 'SHORTFALL' else 'surplus'}">
                    Status: {gap_status}
                </span>
                <span class="gap-detail">Gap: {gap:+,.0f} tonnes {'surplus' if gap < 0 else 'shortfall' if gap > 0 else ''}</span>
                <div class="gap-message">{gap_message}</div>
            </div>

            <div class="section-title">Detailed Monthly Forecast</div>
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Tonnes</th>
                        <th>TPH</th>
                        <th>Lump ({lump_split*100:.0f}%)</th>
                        <th>Fines ({(1-lump_split)*100:.0f}%)</th>
                        <th>Run Hours</th>
                        <th>Availability</th>
                        <th>Utilisation</th>
                        <th>Planned Maint</th>
                        <th>Unplanned</th>
                        <th>Internal</th>
                        <th>External</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="total-row">
                        <td><strong>Total/Avg</strong></td>
                        <td><strong>{total_tonnes:,.0f}</strong></td>
                        <td><strong>{avg_tph:,.0f}</strong></td>
                        <td><strong>{total_tonnes * lump_split:,.0f}</strong></td>
                        <td><strong>{total_tonnes * (1-lump_split):,.0f}</strong></td>
                        <td><strong>{total_run_hours:,.0f}</strong></td>
                        <td><strong>{avg_availability:.1f}%</strong></td>
                        <td><strong>{avg_utilisation:.1f}%</strong></td>
                        <td><strong>{planned_maint:,.0f}</strong></td>
                        <td><strong>{unplanned:,.0f}</strong></td>
                        <td><strong>{internal_delays:,.0f}</strong></td>
                        <td><strong>{external_delays:,.0f}</strong></td>
                    </tr>
                    {forecast_rows}
                </tbody>
            </table>

            {historical_section}

            <div class="page-break"></div>
            <div class="section-title">Calculation Methodology</div>
            <div class="methodology">
                <h3>1. Calendar Time (Base)</h3>
                <p>Total theoretical opportunity (Days in Month x 24h). The maximum 'inventory of time'.</p>

                <h3>2. Unavailable Time (Asset Cannot Run)</h3>
                <p>Planned Maintenance: Scheduled shuts (Fixed logic).<br>
                Unplanned Downtime: Breakdowns (Stochastic logic, affected by seasonality).<br>
                Metric: Availability % = (Total - Unavailable) / Total.</p>

                <h3>3. Unproductive Time (Asset Can Run, But Isn't)</h3>
                <p>Internal Delays: Operational inefficiencies (e.g., blockages).<br>
                External Delays: Supply chain constraints (e.g., no feed).<br>
                Adhoc Delays: Shift changes, crib breaks, minor stops.<br>
                Metric: Utilisation % = Run Hours / Available Hours.</p>

                <h3>4. Productive Time (Run Hours)</h3>
                <p>Total Hours - Unavailable - Unproductive. The only revenue-generating time window.</p>

                <h3>5. Production Rate (Throughput)</h3>
                <p>Based on Target TPH, adjusted for:<br>
                - Seasonal Impact: TPH reduction during wet months.<br>
                - Ramp-Up: Reduced efficiency after major shuts (>24h).<br>
                Product Split: Lump (Premium) vs Fines (Standard) calculated from total.</p>
            </div>

            <div class="footer">
                CSI Mining Services - Time Usage Model Report | Generated by TUM Forecasting Assistant
            </div>
        </div>
    </body>
    </html>
    """

    return html


def create_excel_export(df_forecast, df_adjusted, df_historical, site_name, contract_tonnes, scenario_params, lump_split=0.5):
    """Create Excel file with calculations and scenario comparison."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Original forecast data
        df_export = df_forecast.copy()
        df_export['Lump Tonnes'] = df_export['Tonnes'] * lump_split
        df_export['Fines Tonnes'] = df_export['Tonnes'] * (1 - lump_split)
        df_export.to_excel(writer, sheet_name='Original Forecast', index=False)

        # Adjusted forecast data
        df_adj_export = df_adjusted.copy()
        df_adj_export['Lump Tonnes'] = df_adj_export['Tonnes'] * lump_split
        df_adj_export['Fines Tonnes'] = df_adj_export['Tonnes'] * (1 - lump_split)
        df_adj_export.to_excel(writer, sheet_name='Adjusted Forecast', index=False)

        # Historical data
        if df_historical is not None:
            df_historical.to_excel(writer, sheet_name='Historical Data', index=False)

        # Scenario comparison
        comparison_data = {
            'Metric': ['Total Tonnes', 'Avg TPH', 'Avg Availability', 'Avg Utilisation',
                       'Avg Eff Utilisation', 'Contract Target', 'Gap vs Target'],
            'Original': [
                df_forecast['Tonnes'].sum(),
                df_forecast['TPH'].mean(),
                df_forecast['Availability'].mean(),
                df_forecast['Utilisation'].mean(),
                df_forecast['Effective Utilisation'].mean(),
                contract_tonnes,
                contract_tonnes - df_forecast['Tonnes'].sum()
            ],
            'Adjusted': [
                df_adjusted['Tonnes'].sum(),
                df_adjusted['TPH'].mean(),
                df_adjusted['Availability'].mean(),
                df_adjusted['Utilisation'].mean(),
                df_adjusted['Effective Utilisation'].mean(),
                contract_tonnes,
                contract_tonnes - df_adjusted['Tonnes'].sum()
            ],
            'Change': [
                df_adjusted['Tonnes'].sum() - df_forecast['Tonnes'].sum(),
                df_adjusted['TPH'].mean() - df_forecast['TPH'].mean(),
                df_adjusted['Availability'].mean() - df_forecast['Availability'].mean(),
                df_adjusted['Utilisation'].mean() - df_forecast['Utilisation'].mean(),
                df_adjusted['Effective Utilisation'].mean() - df_forecast['Effective Utilisation'].mean(),
                0,
                (contract_tonnes - df_adjusted['Tonnes'].sum()) - (contract_tonnes - df_forecast['Tonnes'].sum())
            ]
        }
        pd.DataFrame(comparison_data).to_excel(writer, sheet_name='Scenario Comparison', index=False)

        # Scenario parameters
        params_data = {
            'Parameter': ['TPH Adjustment (%)', 'Availability Adjustment (pp)', 'Utilisation Adjustment (pp)',
                          'Lump Split (%)', 'Contract Tonnes'],
            'Value': [
                scenario_params.get('tph_adj', 0),
                scenario_params.get('avail_adj', 0),
                scenario_params.get('util_adj', 0),
                lump_split * 100,
                contract_tonnes
            ]
        }
        pd.DataFrame(params_data).to_excel(writer, sheet_name='Scenario Parameters', index=False)

    output.seek(0)
    return output


# =============================================================================
# MAIN APP
# =============================================================================

st.title("⚙️ TUM Production Forecasting Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Site Selection")
    site_key = st.selectbox(
        "Select Site",
        options=list(SITES.keys()),
        format_func=lambda x: SITES[x][0]
    )

    st.markdown("---")

    # Contract parameters
    st.header("Contract Parameters")
    contract_tonnes = st.number_input(
        "Contract Tonnes Target",
        min_value=0,
        value=5000000 if "west_angelas_1" in site_key else 1750000 if "west_angelas_2" in site_key else 1000000,
        step=100000,
        format="%d",
        help="Target production for the contract period"
    )

    lump_split = st.slider("Lump Split %", 0, 100, 50, help="Percentage of production as lump") / 100

    st.markdown("---")

    # Scenario Adjustments
    st.header("Scenario Adjustments")
    st.caption("Adjust parameters to model different scenarios")

    # Check if TUM adjustment was applied
    default_tph_adj = st.session_state.get('tph_adj_value', 0)

    tph_adjustment = st.slider(
        "TPH Adjustment (%)",
        min_value=-50,
        max_value=50,
        value=int(default_tph_adj),
        step=1,
        help="Percentage change to throughput rate"
    )

    availability_adjustment = st.slider(
        "Availability Adjustment (pp)",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        help="Percentage point change to availability"
    )

    utilisation_adjustment = st.slider(
        "Utilisation Adjustment (pp)",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        help="Percentage point change to utilisation"
    )

    # Reset button
    if st.button("Reset Adjustments"):
        tph_adjustment = 0
        availability_adjustment = 0
        utilisation_adjustment = 0
        st.rerun()

    st.markdown("---")
    st.caption(f"Data: {DEFAULT_DATA_PATH}")

# Load data
df_original, site_name = load_site_data(site_key)
df_hist = load_historical_data(site_key)

if df_original is None:
    st.error(f"Could not load data for {site_name}. Check the data path.")
    st.stop()

# Apply scenario adjustments
df = apply_scenario_adjustments(df_original, tph_adjustment, availability_adjustment, utilisation_adjustment)

# Main content
st.header(f"📊 {site_name}")

# Check if adjustments are active
adjustments_active = tph_adjustment != 0 or availability_adjustment != 0 or utilisation_adjustment != 0

if adjustments_active:
    st.info(f"📊 **Scenario Active:** TPH {tph_adjustment:+.0f}% | Availability {availability_adjustment:+.0f}pp | Utilisation {utilisation_adjustment:+.0f}pp")

# KPI Cards - Show both original and adjusted if adjustments active
col1, col2, col3, col4 = st.columns(4)

total_tonnes = df['Tonnes'].sum()
original_tonnes = df_original['Tonnes'].sum()
avg_availability = df['Availability'].mean()
avg_utilisation = df['Utilisation'].mean()
total_run_hours = df['Run Hours'].sum()

with col1:
    delta_tonnes = total_tonnes - original_tonnes if adjustments_active else total_tonnes - contract_tonnes
    delta_label = "vs Original" if adjustments_active else "vs Target"
    st.metric(
        "Total Production",
        f"{total_tonnes:,.0f} t",
        delta=f"{delta_tonnes:+,.0f} ({delta_label})"
    )

with col2:
    orig_avail = df_original['Availability'].mean()
    delta_avail = (avg_availability - orig_avail) * 100 if adjustments_active else None
    st.metric(
        "Avg Availability",
        f"{avg_availability * 100:.1f}%",
        delta=f"{delta_avail:+.1f}pp" if delta_avail else None
    )

with col3:
    orig_util = df_original['Utilisation'].mean()
    delta_util = (avg_utilisation - orig_util) * 100 if adjustments_active else None
    st.metric(
        "Avg Utilisation",
        f"{avg_utilisation * 100:.1f}%",
        delta=f"{delta_util:+.1f}pp" if delta_util else None
    )

with col4:
    gap = contract_tonnes - total_tonnes
    gap_status = "ON TARGET" if abs(gap) < 10000 else ("SHORT" if gap > 0 else "OVER")
    st.metric(
        "Gap vs Contract",
        f"{gap:+,.0f} t",
        delta=gap_status,
        delta_color="normal" if gap_status == "ON TARGET" else ("inverse" if gap > 0 else "off")
    )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast Data", "🎯 Scenario Planner", "📜 Historical", "🔮 Forecast", "📥 Export"])

with tab1:
    st.subheader("FY27 Forecast Data" + (" (Adjusted)" if adjustments_active else ""))

    # Show calculations
    with st.expander("📐 View Calculations", expanded=False):
        calc_df = pd.DataFrame({
            'Metric': ['Total Tonnes', 'Total Run Hours', 'Avg TPH', 'Avg Availability',
                       'Avg Utilisation', 'Avg Eff Utilisation', 'Total Lump', 'Total Fines'],
            'Formula': ['SUM(Tonnes)', 'SUM(Run Hours)', 'AVERAGE(TPH)', 'AVERAGE(Availability)',
                        'AVERAGE(Utilisation)', 'Availability × Utilisation',
                        f'Total × {lump_split:.0%}', f'Total × {1-lump_split:.0%}'],
            'Result': [
                f"{df['Tonnes'].sum():,.0f}",
                f"{df['Run Hours'].sum():,.0f}",
                f"{df['TPH'].mean():,.1f}",
                f"{df['Availability'].mean()*100:.1f}%",
                f"{df['Utilisation'].mean()*100:.1f}%",
                f"{df['Effective Utilisation'].mean()*100:.1f}%",
                f"{df['Tonnes'].sum() * lump_split:,.0f}",
                f"{df['Tonnes'].sum() * (1-lump_split):,.0f}",
            ]
        })
        st.dataframe(calc_df, use_container_width=True, hide_index=True)

    # Monthly data table
    st.dataframe(
        df.style.format({
            'Tonnes': '{:,.0f}',
            'Run Hours': '{:,.1f}',
            'TPH': '{:,.1f}',
            'Availability': '{:.1%}',
            'Utilisation': '{:.1%}',
            'Effective Utilisation': '{:.1%}',
            'Planned Maint': '{:.0f}',
            'Unplanned': '{:.0f}',
            'Internal Delays': '{:.0f}',
            'External Delays': '{:.0f}',
        }),
        use_container_width=True,
        hide_index=True
    )

    # Production chart - compare original vs adjusted
    st.subheader("Production Trend")

    if adjustments_active:
        fig_prod = go.Figure()
        fig_prod.add_trace(go.Bar(name='Original', x=df_original['Month'], y=df_original['Tonnes'], marker_color='#cccccc'))
        fig_prod.add_trace(go.Bar(name='Adjusted', x=df['Month'], y=df['Tonnes'], marker_color='#1f77b4'))
        fig_prod.update_layout(barmode='group', xaxis_title='Month', yaxis_title='Tonnes', legend_title='Scenario')
    else:
        fig_prod = px.bar(df, x='Month', y='Tonnes', color_discrete_sequence=['#1f77b4'])
        fig_prod.update_layout(xaxis_title='Month', yaxis_title='Tonnes', showlegend=False)

    fig_prod.update_yaxes(tickformat=',')
    fig_prod.add_hline(y=contract_tonnes/len(df), line_dash="dash", line_color="red",
                       annotation_text=f"Target Avg: {contract_tonnes/len(df):,.0f}")
    st.plotly_chart(fig_prod, use_container_width=True)

    # Availability & Utilisation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Availability")
        fig_avail = px.line(df, x='Month', y='Availability', markers=True, color_discrete_sequence=['#2ca02c'])
        fig_avail.update_layout(xaxis_title='Month', yaxis_title='Availability %', showlegend=False)
        fig_avail.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_avail, use_container_width=True)

    with col2:
        st.subheader("Utilisation")
        fig_util = px.line(df, x='Month', y='Utilisation', markers=True, color_discrete_sequence=['#ff7f0e'])
        fig_util.update_layout(xaxis_title='Month', yaxis_title='Utilisation %', showlegend=False)
        fig_util.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_util, use_container_width=True)

with tab2:
    st.subheader("🎯 Scenario Planner")
    st.markdown("Use this tool to adjust TUM forecasts to meet specific targets.")

    # TUM Adjustment Section
    st.markdown("### Direct TUM Adjustment")
    st.markdown("Scale the entire forecast proportionally to hit a specific tonnage target.")

    col_tum1, col_tum2, col_tum3 = st.columns(3)
    with col_tum1:
        current_tum = original_tonnes
        st.metric("Current TUM Forecast", f"{current_tum:,.0f} t")

    with col_tum2:
        target_tum = st.number_input(
            "Target TUM (tonnes)",
            min_value=0,
            value=int(contract_tonnes),
            step=100000,
            format="%d",
            help="Enter your desired total production target"
        )

    with col_tum3:
        if current_tum > 0:
            scale_factor = target_tum / current_tum
            adjustment_pct = (scale_factor - 1) * 100
            st.metric("Required Adjustment", f"{adjustment_pct:+.1f}%",
                      delta="Decrease" if adjustment_pct < 0 else "Increase")

    # Show what the adjustment means
    if current_tum > 0 and target_tum != current_tum:
        st.markdown("---")
        col_adj1, col_adj2, col_adj3 = st.columns(3)

        current_avg_tph = df_original['TPH'].mean()
        new_tph = current_avg_tph * scale_factor

        with col_adj1:
            st.markdown("**TPH Adjustment**")
            st.markdown(f"- Current: **{current_avg_tph:,.0f}** TPH")
            st.markdown(f"- New: **{new_tph:,.0f}** TPH")

        with col_adj2:
            st.markdown("**Production Impact**")
            st.markdown(f"- From: **{current_tum:,.0f}** t")
            st.markdown(f"- To: **{target_tum:,.0f}** t")
            st.markdown(f"- Change: **{target_tum - current_tum:+,.0f}** t")

        with col_adj3:
            st.markdown("**Apply Adjustment**")
            if st.button("📊 Apply TUM Adjustment", key="apply_tum_adj"):
                st.session_state['tph_adj_value'] = adjustment_pct
                st.rerun()

        # Preview adjusted monthly breakdown
        with st.expander("📋 Preview Adjusted Monthly Forecast", expanded=False):
            df_preview = df_original.copy()
            df_preview['Adjusted Tonnes'] = df_preview['Tonnes'] * scale_factor
            df_preview['Adjusted TPH'] = df_preview['TPH'] * scale_factor
            preview_cols = ['Month', 'Tonnes', 'Adjusted Tonnes', 'TPH', 'Adjusted TPH']
            available_cols = [c for c in preview_cols if c in df_preview.columns]
            st.dataframe(
                df_preview[available_cols].style.format({
                    'Tonnes': '{:,.0f}',
                    'Adjusted Tonnes': '{:,.0f}',
                    'TPH': '{:,.0f}',
                    'Adjusted TPH': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            st.markdown(f"**Adjusted Total: {df_preview['Adjusted Tonnes'].sum():,.0f} tonnes**")

    st.markdown("---")
    st.markdown("### Gap Analysis")

    # Current status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Forecast", f"{original_tonnes:,.0f} t")
    with col2:
        st.metric("Contract Target", f"{contract_tonnes:,.0f} t")
    with col3:
        gap = contract_tonnes - original_tonnes
        st.metric("Gap", f"{gap:+,.0f} t", delta="Shortfall" if gap > 0 else "Surplus")

    st.markdown("---")

    # Calculate required changes
    if gap != 0:
        changes = calculate_required_changes(original_tonnes, contract_tonnes, df_original)

        st.subheader("Options to Meet Target")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Option A: Increase TPH Only**")
            st.markdown(f"""
            - Current Avg TPH: **{df_original['TPH'].mean():,.0f}**
            - Required TPH: **{changes['tph_only']['required_tph']:,.0f}**
            - Change needed: **{changes['tph_only']['change_pct']:+.1f}%**
            """)
            if st.button("Apply TPH Change", key="apply_tph"):
                st.session_state['tph_adj'] = changes['tph_only']['change_pct']
                st.rerun()

        with col2:
            st.markdown("**Option B: Increase Run Hours Only**")
            st.markdown(f"""
            - Current Total Hours: **{df_original['Run Hours'].sum():,.0f}**
            - Required Hours: **{changes['hours_only']['required_hours']:,.0f}**
            - Change needed: **{changes['hours_only']['change_pct']:+.1f}%**
            """)
            st.caption("(Adjust via Availability/Utilisation)")

        with col3:
            st.markdown("**Option C: Balanced Approach**")
            st.markdown(f"""
            - TPH change: **{changes['balanced']['tph_change_pct']:+.1f}%**
            - Hours change: **{changes['balanced']['hours_change_pct']:+.1f}%**
            - Split evenly between both
            """)
            if st.button("Apply Balanced Change", key="apply_balanced"):
                st.session_state['tph_adj'] = changes['balanced']['tph_change_pct']
                st.rerun()
    else:
        st.success("Current forecast meets the contract target!")

    st.markdown("---")

    # Sensitivity Analysis
    st.subheader("Sensitivity Analysis")
    st.markdown("See how changes in key parameters affect total production:")

    # Create sensitivity table
    sensitivity_data = []
    for tph_chg in [-10, -5, 0, 5, 10]:
        for avail_chg in [-5, 0, 5]:
            df_temp = apply_scenario_adjustments(df_original, tph_chg, avail_chg, 0)
            tonnes = df_temp['Tonnes'].sum()
            sensitivity_data.append({
                'TPH Change (%)': tph_chg,
                'Availability Change (pp)': avail_chg,
                'Total Tonnes': tonnes,
                'vs Target': tonnes - contract_tonnes,
                'Meets Target': '✓' if tonnes >= contract_tonnes else '✗'
            })

    sens_df = pd.DataFrame(sensitivity_data)
    st.dataframe(
        sens_df.style.format({
            'Total Tonnes': '{:,.0f}',
            'vs Target': '{:+,.0f}'
        }).apply(lambda x: ['background-color: #d4edda' if v == '✓' else '' for v in x], subset=['Meets Target']),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("Historical Performance")

    if df_hist is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Historical Total", f"{df_hist['Tonnes'].sum():,.0f} t")
        with col2:
            st.metric("Avg Availability", f"{df_hist['Availability'].mean()*100:.1f}%")
        with col3:
            st.metric("Avg Utilisation", f"{df_hist['Utilisation'].mean()*100:.1f}%")
        with col4:
            st.metric("Months of Data", f"{len(df_hist)}")

        with st.expander("📐 View Calculations", expanded=False):
            hist_calc_df = pd.DataFrame({
                'Metric': ['Total Tonnes', 'Total Run Hours', 'Avg TPH', 'Avg Availability', 'Avg Utilisation'],
                'Formula': ['SUM(Tonnes)', 'SUM(Run Hours)', 'AVERAGE(TPH)', 'AVERAGE(Availability)', 'AVERAGE(Utilisation)'],
                'Result': [
                    f"{df_hist['Tonnes'].sum():,.0f}",
                    f"{df_hist['Run Hours'].sum():,.0f}",
                    f"{df_hist['TPH'].mean():,.1f}",
                    f"{df_hist['Availability'].mean()*100:.1f}%",
                    f"{df_hist['Utilisation'].mean()*100:.1f}%",
                ]
            })
            st.dataframe(hist_calc_df, use_container_width=True, hide_index=True)

        st.dataframe(
            df_hist.style.format({
                'Tonnes': '{:,.0f}',
                'Run Hours': '{:,.1f}',
                'TPH': '{:,.1f}',
                'Availability': '{:.1%}',
                'Utilisation': '{:.1%}',
                'Effective Utilisation': '{:.1%}',
                'Planned Maint': '{:.0f}',
                'Unplanned': '{:.0f}',
                'Internal Delays': '{:.0f}',
                'External Delays': '{:.0f}',
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Historical Production Trend")
        fig_hist = px.bar(df_hist, x='Month', y='Tonnes', color_discrete_sequence=['#ff7f0e'])
        fig_hist.update_layout(xaxis_title='Month', yaxis_title='Tonnes', showlegend=False)
        fig_hist.update_yaxes(tickformat=',')
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No historical data available for this site.")

with tab4:
    st.subheader("Production Forecast")

    forecast_periods = st.slider("Forecast Periods (months)", 1, 12, 6, key="fc_periods")

    if 'Tonnes' in df.columns:
        forecast = forecast_ensemble(df['Tonnes'], forecast_periods)

        if forecast is not None:
            forecast_months = [f"Month +{i+1}" for i in range(forecast_periods)]
            forecast_df = pd.DataFrame({'Period': forecast_months, 'Forecast Tonnes': forecast})

            with st.expander("📐 Forecast Methodology", expanded=False):
                st.markdown("""
                **Ensemble Forecast Method:**
                1. **Linear Regression:** `y = slope × x + intercept`
                2. **Moving Average:** `MA = AVERAGE(last 3 months) + trend`
                3. **Ensemble:** `Final = (Linear + MA) / 2`
                """)

            st.dataframe(
                forecast_df.style.format({'Forecast Tonnes': '{:,.0f}'}),
                use_container_width=True,
                hide_index=True
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecast Total", f"{forecast.sum():,.0f} t")
            with col2:
                st.metric("Forecast Avg/Month", f"{forecast.mean():,.0f} t")
            with col3:
                current_avg = df['Tonnes'].mean()
                change = ((forecast.mean() - current_avg) / current_avg) * 100
                st.metric("vs Current Avg", f"{change:+.1f}%")

            st.subheader("Actual vs Forecast")
            all_months = list(df['Month']) + forecast_months

            fig_combined = go.Figure()
            fig_combined.add_trace(go.Bar(name='Actual', x=df['Month'].tolist(), y=df['Tonnes'].tolist(), marker_color='#1f77b4'))
            fig_combined.add_trace(go.Bar(name='Forecast', x=forecast_months, y=forecast.tolist(), marker_color='#ff7f0e'))
            fig_combined.update_layout(xaxis_title='Period', yaxis_title='Tonnes', legend_title='Type')
            fig_combined.update_yaxes(tickformat=',')
            st.plotly_chart(fig_combined, use_container_width=True)
        else:
            st.warning("Not enough data to generate forecast.")

with tab5:
    st.subheader("Export Report")

    # Gap Analysis
    st.markdown("**Gap Analysis:**")
    gap = contract_tonnes - total_tonnes
    gap_pct = (gap / contract_tonnes) * 100 if contract_tonnes > 0 else 0

    gap_df = pd.DataFrame({
        'Metric': ['Contract Target', 'Forecast Total', 'Gap (Tonnes)', 'Gap (%)'],
        'Value': [f"{contract_tonnes:,.0f}", f"{total_tonnes:,.0f}", f"{gap:,.0f}", f"{gap_pct:.1f}%"],
        'Calculation': ['User Input', 'SUM(Monthly Tonnes)', 'Contract - Forecast', '(Contract - Forecast) / Contract × 100']
    })
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

    status = "ON TARGET" if abs(gap) < 10000 else ("UNDER TARGET" if gap > 0 else "OVER TARGET")
    if status == "ON TARGET":
        st.success(f"Status: {status}")
    elif status == "UNDER TARGET":
        st.warning(f"Status: {status} - {abs(gap):,.0f} tonnes shortfall")
    else:
        st.info(f"Status: {status} - {abs(gap):,.0f} tonnes surplus")

    st.markdown("---")

    # Scenario summary
    if adjustments_active:
        st.markdown("**Scenario Applied:**")
        st.markdown(f"- TPH Adjustment: {tph_adjustment:+.0f}%")
        st.markdown(f"- Availability Adjustment: {availability_adjustment:+.0f}pp")
        st.markdown(f"- Utilisation Adjustment: {utilisation_adjustment:+.0f}pp")
        st.markdown(f"- Production Change: {total_tonnes - original_tonnes:+,.0f} tonnes")
        st.markdown("---")

    scenario_params = {
        'tph_adj': tph_adjustment,
        'avail_adj': availability_adjustment,
        'util_adj': utilisation_adjustment
    }

    st.markdown("### Export Options")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("**CSI TUM Report (HTML)**")
        st.caption("Professional report matching CSI Mining Services format")

        # Generate HTML report
        html_report = create_html_report(df, df_hist, site_name, contract_tonnes, scenario_params, lump_split)

        st.download_button(
            label="📄 Download HTML Report",
            data=html_report,
            file_name=f"CSI_TUM_Report_{site_key.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html"
        )
        st.caption("Open in browser, then Print > Save as PDF")

    with col_exp2:
        st.markdown("**Excel Data Export**")
        st.caption("Raw data with calculations for further analysis")

        excel_file = create_excel_export(df_original, df, df_hist, site_name, contract_tonnes, scenario_params, lump_split)

        st.download_button(
            label="📥 Download Excel Report",
            data=excel_file,
            file_name=f"TUM_Report_{site_key}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.caption("Includes: Forecast, Historical, Scenario Comparison")

# Footer
st.markdown("---")
st.caption(f"TUM Forecasting Assistant | Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
