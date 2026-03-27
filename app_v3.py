"""
TUM Forecasting Assistant v3.0
Time Usage Model for Crushing Plant Performance Forecasting

Built-in database with all CSI Mining sites.
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

# Data path
DATA_PATH = Path(r"\\p1fs002\CSI - Technical Services\TUM Data for Accountants FY27")

# Site configurations
SITES = {
    "mt_whaleback": {
        "name": "Mt Whaleback - Crushing Circuit",
        "file": "Mt Whaleback - Historical.csv",
        "commodity": "Iron Ore",
        "default_contract": 12000000,
        "default_lump_split": 0.40
    },
    "area_c": {
        "name": "Area C - Crushing Circuit",
        "file": "Area C.csv",
        "commodity": "Iron Ore",
        "default_contract": 8000000,
        "default_lump_split": 0.35
    },
    "iron_valley": {
        "name": "Iron Valley - Crushing Circuit",
        "file": "Iron Valley.csv",
        "commodity": "Iron Ore",
        "default_contract": 5000000,
        "default_lump_split": 0.30
    },
    "roy_hill": {
        "name": "Roy Hill - Crushing Circuit",
        "file": "Roy Hill Bravo.csv",
        "commodity": "Iron Ore",
        "default_contract": 4000000,
        "default_lump_split": 0.40
    },
    "wodgina": {
        "name": "Wodgina - Crushing Circuit",
        "file": "Wodgina.csv",
        "commodity": "Lithium",
        "default_contract": 5000000,
        "default_lump_split": 0.0
    },
    "sanjiv_ridge": {
        "name": "Sanjiv Ridge - Crushing Circuit",
        "file": "Sanjiv Ridge.csv",
        "commodity": "Iron Ore",
        "default_contract": 5500000,
        "default_lump_split": 0.50
    },
    "granites": {
        "name": "Granites - Crushing Circuit",
        "file": "Granites.csv",
        "commodity": "Gold",
        "default_contract": 2500000,
        "default_lump_split": 0.0
    },
    "rod_ore": {
        "name": "Rod Ore - Crushing Circuit",
        "file": "Rod Ore.csv",
        "commodity": "Iron Ore",
        "default_contract": 6500000,
        "default_lump_split": 0.40
    },
    "hope_downs_4": {
        "name": "Hope Downs 4 - Crushing Circuit",
        "file": "Hope Downs 4.csv",
        "commodity": "Iron Ore",
        "default_contract": 4000000,
        "default_lump_split": 0.45
    },
    "kcgm_main": {
        "name": "KCGM - Main Plant",
        "file": "KCGM-Main Plant.csv",
        "commodity": "Gold",
        "default_contract": 3500000,
        "default_lump_split": 0.0
    },
    "west_angelas_1": {
        "name": "West Angelas - Plant 1",
        "file": "West Angelas Plant 1.csv",
        "commodity": "Iron Ore",
        "default_contract": 5000000,
        "default_lump_split": 0.40
    },
    "west_angelas_2": {
        "name": "West Angelas - Plant 2",
        "file": "West Angelas Plant 2.csv",
        "commodity": "Iron Ore",
        "default_contract": 2000000,
        "default_lump_split": 0.40
    },
}

# Constants
CALENDAR_HOURS = {
    'Jul': 744, 'Aug': 744, 'Sep': 720, 'Oct': 744,
    'Nov': 720, 'Dec': 744, 'Jan': 744, 'Feb': 672,
    'Mar': 744, 'Apr': 720, 'May': 744, 'Jun': 720
}
FY_MONTHS = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# Commodity options
COMMODITIES = ['Iron Ore', 'Lithium', 'Gold', 'Garnet']


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
def load_site_data(site_key):
    """Load historical data for a site."""
    if site_key not in SITES:
        return None, "Site not found"

    site_config = SITES[site_key]
    filepath = DATA_PATH / site_config["file"]

    if not filepath.exists():
        return None, f"Data file not found: {filepath}"

    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Normalize column names
        col_map = {
            'Crushed Tonnes': 'Tonnes',
            'Planned Maint.': 'Planned Maint',
            'Unplanned Maint.': 'Unplanned'
        }
        df = df.rename(columns=col_map)

        # Parse percentages
        for col in ['Availability', 'Utilisation', 'Effective Utilisation']:
            if col in df.columns:
                df[col] = df[col].apply(parse_percentage)

        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


def calculate_historical_stats(df):
    """Calculate statistics from historical data."""
    stats = {}

    # Column name variations
    planned_col = 'Planned Maint' if 'Planned Maint' in df.columns else 'Planned Maint.'
    unplanned_col = 'Unplanned' if 'Unplanned' in df.columns else 'Unplanned Maint.'

    # Maintenance averages (per month)
    stats['avg_planned_maint'] = df[planned_col].mean() if planned_col in df.columns else 0
    stats['avg_unplanned'] = df[unplanned_col].mean() if unplanned_col in df.columns else 0
    stats['avg_internal_delays'] = df['Internal Delays'].mean() if 'Internal Delays' in df.columns else 0
    stats['avg_external_delays'] = df['External Delays'].mean() if 'External Delays' in df.columns else 0

    # Performance averages
    stats['avg_tph'] = df['TPH'].mean() if 'TPH' in df.columns else 0
    stats['avg_availability'] = df['Availability'].mean() if 'Availability' in df.columns else 0
    stats['avg_utilisation'] = df['Utilisation'].mean() if 'Utilisation' in df.columns else 0
    stats['avg_run_hours'] = df['Run Hours'].mean() if 'Run Hours' in df.columns else 0
    stats['avg_tonnes_per_month'] = df['Tonnes'].mean() if 'Tonnes' in df.columns else 0

    # Totals
    stats['total_tonnes'] = df['Tonnes'].sum() if 'Tonnes' in df.columns else 0
    stats['total_run_hours'] = df['Run Hours'].sum() if 'Run Hours' in df.columns else 0
    stats['months_of_data'] = len(df)

    # Calculate total unavailable/unproductive time per month
    stats['total_unavailable_per_month'] = stats['avg_planned_maint'] + stats['avg_unplanned']
    stats['total_unproductive_per_month'] = stats['avg_internal_delays'] + stats['avg_external_delays']

    return stats


def generate_fy_forecast(contract_tonnes, lump_split, stats, forecast_year="27"):
    """Generate FY forecast based on historical stats and contract target."""

    forecast = []
    total_run_hours = 0

    # Calculate run hours for each month based on historical averages
    for month in FY_MONTHS:
        cal_hours = CALENDAR_HOURS[month]

        # Available hours = Calendar - Planned - Unplanned
        available_hours = cal_hours - stats['avg_planned_maint'] - stats['avg_unplanned']
        available_hours = max(0, available_hours)

        # Run hours = Available - Internal - External
        run_hours = available_hours - stats['avg_internal_delays'] - stats['avg_external_delays']
        run_hours = max(0, run_hours)

        # KPIs
        availability = available_hours / cal_hours if cal_hours > 0 else 0
        utilisation = run_hours / available_hours if available_hours > 0 else 0

        forecast.append({
            'month': month,
            'calendar_hours': cal_hours,
            'available_hours': available_hours,
            'run_hours': run_hours,
            'availability': availability,
            'utilisation': utilisation,
            'planned_maint': stats['avg_planned_maint'],
            'unplanned': stats['avg_unplanned'],
            'internal_delays': stats['avg_internal_delays'],
            'external_delays': stats['avg_external_delays']
        })

        total_run_hours += run_hours

    # Calculate required TPH
    required_tph = contract_tonnes / total_run_hours if total_run_hours > 0 else 0

    # Build final forecast DataFrame
    forecast_data = []
    for f in forecast:
        year_suffix = "-26" if f['month'] in ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] else f"-{forecast_year}"
        tonnes = required_tph * f['run_hours']

        forecast_data.append({
            'Month': f"{f['month']}{year_suffix}",
            'Tonnes': round(tonnes),
            'TPH': round(required_tph),
            'Lump Tonnes': round(tonnes * lump_split),
            'Fines Tonnes': round(tonnes * (1 - lump_split)),
            'Run Hours': round(f['run_hours']),
            'Run Hours / Day': round(f['run_hours'] / (f['calendar_hours'] / 24), 2),
            'Availability': f['availability'],
            'Utilisation': f['utilisation'],
            'Effective Utilisation': f['availability'] * f['utilisation'],
            'Planned Maint': round(f['planned_maint']),
            'Unplanned': round(f['unplanned']),
            'Internal Delays': round(f['internal_delays']),
            'External Delays': round(f['external_delays'])
        })

    df_forecast = pd.DataFrame(forecast_data)

    return df_forecast, required_tph, total_run_hours


def create_html_report(df_forecast, df_historical, site_name, contract_tonnes, lump_split, stats, required_tph, commodity):
    """Generate HTML report matching CSI Mining Services format."""

    total_tonnes = df_forecast['Tonnes'].sum()
    avg_availability = df_forecast['Availability'].mean() * 100
    avg_utilisation = df_forecast['Utilisation'].mean() * 100
    total_run_hours = df_forecast['Run Hours'].sum()

    gap = total_tonnes - contract_tonnes
    gap_status = "ON TRACK" if abs(gap) < contract_tonnes * 0.01 else ("SURPLUS" if gap > 0 else "SHORTFALL")

    tph_diff_pct = ((required_tph - stats['avg_tph']) / stats['avg_tph'] * 100) if stats['avg_tph'] > 0 else 0

    # Check if Iron Ore (show lump/fines)
    is_iron_ore = commodity == "Iron Ore"

    # Build forecast rows
    forecast_rows = ""
    for _, row in df_forecast.iterrows():
        if is_iron_ore:
            forecast_rows += f"""<tr>
            <td>{row['Month']}</td><td>{row['Tonnes']:,.0f}</td><td>{row['TPH']:,.0f}</td>
            <td>{row['Lump Tonnes']:,.0f}</td><td>{row['Fines Tonnes']:,.0f}</td>
            <td>{row['Run Hours']:,.0f}</td><td>{row['Run Hours / Day']:.2f}</td>
            <td>{row['Availability']*100:.1f}%</td><td>{row['Utilisation']*100:.1f}%</td>
            <td>{row['Planned Maint']:.0f}</td><td>{row['Unplanned']:.0f}</td>
            <td>{row['Internal Delays']:.0f}</td><td>{row['External Delays']:.0f}</td>
        </tr>"""
        else:
            forecast_rows += f"""<tr>
            <td>{row['Month']}</td><td>{row['Tonnes']:,.0f}</td><td>{row['TPH']:,.0f}</td>
            <td>{row['Run Hours']:,.0f}</td><td>{row['Run Hours / Day']:.2f}</td>
            <td>{row['Availability']*100:.1f}%</td><td>{row['Utilisation']*100:.1f}%</td>
            <td>{row['Planned Maint']:.0f}</td><td>{row['Unplanned']:.0f}</td>
            <td>{row['Internal Delays']:.0f}</td><td>{row['External Delays']:.0f}</td>
        </tr>"""

    # Build historical rows
    historical_rows = ""
    if df_historical is not None and len(df_historical) > 0:
        for _, row in df_historical.iterrows():
            planned = row.get('Planned Maint', row.get('Planned Maint.', 0))
            unplanned = row.get('Unplanned', row.get('Unplanned Maint.', 0))
            historical_rows += f"""<tr>
                <td>{row.get('Month', '')}</td><td>{row.get('Tonnes', 0):,.0f}</td><td>{row.get('TPH', 0):,.0f}</td>
                <td>{row.get('Run Hours', 0):,.0f}</td>
                <td>{row.get('Availability', 0)*100:.1f}%</td><td>{row.get('Utilisation', 0)*100:.1f}%</td>
                <td>{planned:.0f}</td><td>{unplanned:.0f}</td>
                <td>{row.get('Internal Delays', 0):.0f}</td><td>{row.get('External Delays', 0):.0f}</td>
            </tr>"""

    historical_section = f"""
    <div class="section-title">Detailed Monthly Historical Breakdown</div>
    <table>
        <thead><tr>
            <th>Month</th><th>Tonnes</th><th>TPH</th><th>Run Hrs</th>
            <th>Avail</th><th>Util</th><th>Planned</th><th>Unplanned</th><th>Internal</th><th>External</th>
        </tr></thead>
        <tbody>
            <tr class="total-row">
                <td><strong>Total/Avg</strong></td>
                <td><strong>{stats['total_tonnes']:,.0f}</strong></td>
                <td><strong>{stats['avg_tph']:,.0f}</strong></td>
                <td><strong>{stats['total_run_hours']:,.0f}</strong></td>
                <td><strong>{stats['avg_availability']*100:.1f}%</strong></td>
                <td><strong>{stats['avg_utilisation']*100:.1f}%</strong></td>
                <td><strong>{stats['avg_planned_maint']*stats['months_of_data']:,.0f}</strong></td>
                <td><strong>{stats['avg_unplanned']*stats['months_of_data']:,.0f}</strong></td>
                <td><strong>{stats['avg_internal_delays']*stats['months_of_data']:,.0f}</strong></td>
                <td><strong>{stats['avg_external_delays']*stats['months_of_data']:,.0f}</strong></td>
            </tr>
            {historical_rows}
        </tbody>
    </table>
    """ if df_historical is not None else ""

    warning_html = f"""
    <div class="warning-box">
        <strong>⚠️ TPH Warning:</strong> Required TPH ({required_tph:,.0f}) is {abs(tph_diff_pct):.1f}%
        {'higher' if tph_diff_pct > 0 else 'lower'} than historical average ({stats['avg_tph']:,.0f}).
        Please verify this target is achievable.
    </div>
    """ if abs(tph_diff_pct) > 15 else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CSI TUM Report - {site_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a1a; color: #fff; font-size: 11px; }}
        .header {{ background: #1a1a1a; padding: 20px 30px; border-bottom: 2px solid #333; overflow: hidden; }}
        .header-title {{ font-size: 24px; font-weight: bold; }}
        .header-subtitle {{ font-size: 12px; color: #666; }}
        .header-right {{ float: right; text-align: right; }}
        .project-name {{ font-size: 16px; font-weight: bold; }}
        .generated-date {{ font-size: 11px; color: #666; }}
        .content {{ padding: 20px 30px; }}
        .kpi-container {{ display: flex; gap: 15px; margin-bottom: 25px; }}
        .kpi-card {{ flex: 1; background: #2a2a2a; border-radius: 6px; padding: 15px; text-align: center; }}
        .kpi-label {{ font-size: 10px; color: #666; text-transform: uppercase; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 26px; font-weight: bold; }}
        .kpi-value.green {{ color: #4CAF50; }}
        .kpi-value.blue {{ color: #2196F3; }}
        .kpi-value.orange {{ color: #FF9800; }}
        .kpi-value.purple {{ color: #9C27B0; }}
        .section-title {{ font-size: 16px; font-weight: bold; margin: 25px 0 12px 0; }}
        .assumptions-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px 20px; background: #2a2a2a; padding: 15px 20px; border-radius: 6px; }}
        .assumption-item {{ display: flex; justify-content: space-between; font-size: 11px; }}
        .assumption-label {{ color: #888; }}
        .assumption-value {{ font-weight: 600; }}
        .gap-box {{ background: #2a2a2a; border-radius: 6px; padding: 15px 20px; margin: 15px 0; display: flex; align-items: center; gap: 15px; }}
        .gap-status {{ padding: 6px 14px; border-radius: 4px; font-weight: bold; }}
        .gap-status.on-track {{ background: #1b5e20; color: #4CAF50; }}
        .gap-status.shortfall {{ background: #b71c1c; color: #f44336; }}
        .gap-status.surplus {{ background: #0d47a1; color: #2196F3; }}
        .gap-detail {{ color: #888; }}
        .warning-box {{ background: #4a3000; border: 1px solid #ff9800; border-radius: 6px; padding: 15px; margin: 15px 0; }}
        .warning-box strong {{ color: #ff9800; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 9px; margin: 10px 0; }}
        th {{ background: #333; padding: 6px 3px; text-align: right; font-weight: 600; }}
        th:first-child {{ text-align: left; }}
        td {{ padding: 5px 3px; border-bottom: 1px solid #333; text-align: right; }}
        td:first-child {{ text-align: left; }}
        .total-row {{ background: #2d4a2d; }}
        .total-row td {{ color: #4CAF50; font-weight: bold; }}
        .methodology {{ background: #2a2a2a; padding: 20px; border-radius: 6px; margin-top: 25px; }}
        .methodology h3 {{ color: #4CAF50; font-size: 13px; margin: 12px 0 6px 0; }}
        .methodology h3:first-child {{ margin-top: 0; }}
        .methodology p {{ color: #aaa; font-size: 11px; margin-left: 12px; line-height: 1.5; }}
        .footer {{ text-align: center; padding: 15px; color: #555; font-size: 10px; border-top: 1px solid #333; margin-top: 30px; }}
        .page-break {{ page-break-before: always; }}
        @media print {{ body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
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
            <div class="kpi-card"><div class="kpi-label">Total Production</div><div class="kpi-value green">{total_tonnes:,.0f} t</div></div>
            <div class="kpi-card"><div class="kpi-label">Avg Availability</div><div class="kpi-value blue">{avg_availability:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Avg Utilisation</div><div class="kpi-value orange">{avg_utilisation:.1f}%</div></div>
            <div class="kpi-card"><div class="kpi-label">Total Run Hours</div><div class="kpi-value purple">{total_run_hours:,.0f} hrs</div></div>
        </div>

        <div class="section-title">Model Assumptions</div>
        <div class="assumptions-grid">
            <div class="assumption-item"><span class="assumption-label">Site:</span><span class="assumption-value">{site_name}</span></div>
            <div class="assumption-item"><span class="assumption-label">Required TPH:</span><span class="assumption-value">{required_tph:,.0f}</span></div>
            <div class="assumption-item"><span class="assumption-label">Historical Avg TPH:</span><span class="assumption-value">{stats['avg_tph']:,.0f}</span></div>
            <div class="assumption-item"><span class="assumption-label">Avg Planned Maint:</span><span class="assumption-value">{stats['avg_planned_maint']:.1f} h/mo</span></div>
            <div class="assumption-item"><span class="assumption-label">Forecast Year:</span><span class="assumption-value">FY27</span></div>
            <div class="assumption-item"><span class="assumption-label">Lump Split:</span><span class="assumption-value">{f'{lump_split*100:.0f}%' if is_iron_ore else 'N/A'}</span></div>
            <div class="assumption-item"><span class="assumption-label">Avg Unplanned:</span><span class="assumption-value">{stats['avg_unplanned']:.1f} h/mo</span></div>
            <div class="assumption-item"><span class="assumption-label">Avg Internal Delays:</span><span class="assumption-value">{stats['avg_internal_delays']:.1f} h/mo</span></div>
            <div class="assumption-item"><span class="assumption-label">Operation:</span><span class="assumption-value">24h</span></div>
            <div class="assumption-item"><span class="assumption-label">Fines Split:</span><span class="assumption-value">{f'{(1-lump_split)*100:.0f}%' if is_iron_ore else 'N/A'}</span></div>
            <div class="assumption-item"><span class="assumption-label">Avg External Delays:</span><span class="assumption-value">{stats['avg_external_delays']:.1f} h/mo</span></div>
            <div class="assumption-item"><span class="assumption-label">Historical Data:</span><span class="assumption-value">{stats['months_of_data']} months</span></div>
            <div class="assumption-item"><span class="assumption-label">Commodity:</span><span class="assumption-value">{commodity}</span></div>
            <div class="assumption-item"><span class="assumption-label">Contract Tonnes:</span><span class="assumption-value">{contract_tonnes:,.0f}</span></div>
            <div class="assumption-item"></div>
            <div class="assumption-item"></div>
        </div>

        {warning_html}

        <div class="section-title">Scenario & Gap Analysis</div>
        <div class="gap-box">
            <span class="gap-status {gap_status.lower().replace(' ', '-')}">Status: {gap_status}</span>
            <span class="gap-detail">Gap: {gap:+,.0f} tonnes {'surplus' if gap > 0 else 'shortfall' if gap < 0 else ''}</span>
        </div>
        <p style="color:#888; font-size:11px;">{'Forecast meets contract requirements.' if gap_status == 'ON TRACK' else 'Review operational parameters to meet target.' if gap_status == 'SHORTFALL' else 'Forecast exceeds contract requirements.'}</p>

        <div class="section-title">Detailed Monthly Forecast - FY27</div>
        <table>
            <thead><tr>
                <th>Month</th><th>Tonnes</th><th>TPH</th>{'<th>Lump</th><th>Fines</th>' if is_iron_ore else ''}
                <th>Run Hrs</th><th>Hrs/Day</th><th>Avail</th><th>Util</th>
                <th>Planned</th><th>Unplanned</th><th>Internal</th><th>External</th>
            </tr></thead>
            <tbody>
                <tr class="total-row">
                    <td><strong>Total</strong></td>
                    <td><strong>{total_tonnes:,.0f}</strong></td>
                    <td><strong>{required_tph:,.0f}</strong></td>
                    {f"<td><strong>{df_forecast['Lump Tonnes'].sum():,.0f}</strong></td><td><strong>{df_forecast['Fines Tonnes'].sum():,.0f}</strong></td>" if is_iron_ore else ''}
                    <td><strong>{total_run_hours:,.0f}</strong></td>
                    <td><strong>{total_run_hours/365:.2f}</strong></td>
                    <td><strong>{avg_availability:.1f}%</strong></td>
                    <td><strong>{avg_utilisation:.1f}%</strong></td>
                    <td><strong>{df_forecast['Planned Maint'].sum():.0f}</strong></td>
                    <td><strong>{df_forecast['Unplanned'].sum():.0f}</strong></td>
                    <td><strong>{df_forecast['Internal Delays'].sum():.0f}</strong></td>
                    <td><strong>{df_forecast['External Delays'].sum():.0f}</strong></td>
                </tr>
                {forecast_rows}
            </tbody>
        </table>

        {historical_section}

        <div class="methodology">
            <h3>Calculation Methodology</h3>
            <p><strong>1. Available Hours</strong> = Calendar Hours - Planned Maint - Unplanned Maint</p>
            <p><strong>2. Availability %</strong> = Available Hours / Calendar Hours</p>
            <p><strong>3. Run Hours</strong> = Available Hours - Internal Delays - External Delays</p>
            <p><strong>4. Utilisation %</strong> = Run Hours / Available Hours</p>
            <p><strong>5. Required TPH</strong> = Contract Tonnes / Total Run Hours = {contract_tonnes:,.0f} / {total_run_hours:,.0f} = <strong>{required_tph:,.0f}</strong></p>
            <p><strong>6. Monthly Tonnes</strong> = TPH × Run Hours</p>
        </div>

        <div class="footer">CSI Mining Services - Time Usage Model | Generated by TUM Forecasting Assistant v3.0</div>
    </div>
</body>
</html>"""

    return html


# =============================================================================
# MAIN APP
# =============================================================================

st.title("⚙️ TUM Forecasting Assistant")
st.markdown("**Time Usage Model for Crushing Plant Performance Forecasting**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Site Selection")

    site_key = st.selectbox(
        "Select Site",
        options=list(SITES.keys()),
        format_func=lambda x: SITES[x]["name"]
    )

    site_config = SITES[site_key]

    st.markdown("---")
    st.header("🎯 Forecast Targets")

    contract_tonnes = st.number_input(
        "Contract Tonnes Target",
        min_value=0,
        value=site_config["default_contract"],
        step=100000,
        format="%d"
    )

    st.markdown("---")
    st.header("🏭 Commodity")

    # Get default commodity index
    default_commodity = site_config.get("commodity", "Iron Ore")
    default_idx = COMMODITIES.index(default_commodity) if default_commodity in COMMODITIES else 0

    selected_commodity = st.selectbox(
        "Select Commodity",
        options=COMMODITIES,
        index=default_idx
    )

    # Only show lump/fines split for Iron Ore
    if selected_commodity == "Iron Ore":
        lump_split = st.slider(
            "Lump Split %",
            min_value=0,
            max_value=100,
            value=int(site_config["default_lump_split"] * 100)
        ) / 100
        st.caption(f"Fines Split: {(1-lump_split)*100:.0f}%")
    else:
        lump_split = 0.0
        st.info(f"No lump/fines split for {selected_commodity}")

    st.markdown("---")
    st.caption(f"Data: {DATA_PATH}")

# Load data
df_historical, error = load_site_data(site_key)

if error:
    st.error(f"Error: {error}")
    st.stop()

# Calculate stats
stats = calculate_historical_stats(df_historical)

# Display
commodity_icons = {'Iron Ore': '🔶', 'Lithium': '🔋', 'Gold': '🥇', 'Garnet': '💎'}
st.header(f"📍 {site_config['name']}")
st.caption(f"{commodity_icons.get(selected_commodity, '⚙️')} **Commodity:** {selected_commodity}")

# Historical Summary
st.subheader("📜 Historical Performance")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Historical Avg TPH", f"{stats['avg_tph']:,.0f}")
with col2:
    st.metric("Avg Availability", f"{stats['avg_availability']*100:.1f}%")
with col3:
    st.metric("Avg Utilisation", f"{stats['avg_utilisation']*100:.1f}%")
with col4:
    st.metric("Months of Data", f"{stats['months_of_data']}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Planned Maint", f"{stats['avg_planned_maint']:.0f} hrs/mo")
with col2:
    st.metric("Avg Unplanned", f"{stats['avg_unplanned']:.0f} hrs/mo")
with col3:
    st.metric("Avg Internal Delays", f"{stats['avg_internal_delays']:.0f} hrs/mo")
with col4:
    st.metric("Avg External Delays", f"{stats['avg_external_delays']:.0f} hrs/mo")

with st.expander("📋 View Historical Data"):
    st.dataframe(df_historical, use_container_width=True, hide_index=True)

st.markdown("---")

# Generate Forecast
st.subheader("🔮 FY27 Forecast")

df_forecast, required_tph, total_run_hours = generate_fy_forecast(contract_tonnes, lump_split, stats)

# TPH comparison
tph_diff = ((required_tph - stats['avg_tph']) / stats['avg_tph'] * 100) if stats['avg_tph'] > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Required TPH",
        f"{required_tph:,.0f}",
        delta=f"{tph_diff:+.1f}% vs historical",
        delta_color="normal" if abs(tph_diff) < 10 else "inverse"
    )
with col2:
    st.metric("Total Run Hours", f"{total_run_hours:,.0f}")
with col3:
    st.metric("Forecast Total", f"{df_forecast['Tonnes'].sum():,.0f} t")
with col4:
    gap = df_forecast['Tonnes'].sum() - contract_tonnes
    st.metric("Gap vs Target", f"{gap:+,.0f} t")

if abs(tph_diff) > 15:
    st.warning(f"⚠️ Required TPH ({required_tph:,.0f}) is {abs(tph_diff):.1f}% {'higher' if tph_diff > 0 else 'lower'} than historical average ({stats['avg_tph']:,.0f}). Verify target is achievable.")

# Forecast table - hide Lump/Fines for non-Iron Ore
if selected_commodity == "Iron Ore":
    display_cols = df_forecast.columns.tolist()
    format_dict = {
        'Tonnes': '{:,.0f}', 'TPH': '{:,.0f}',
        'Lump Tonnes': '{:,.0f}', 'Fines Tonnes': '{:,.0f}',
        'Run Hours': '{:,.0f}', 'Run Hours / Day': '{:.2f}',
        'Availability': '{:.1%}', 'Utilisation': '{:.1%}', 'Effective Utilisation': '{:.1%}',
        'Planned Maint': '{:.0f}', 'Unplanned': '{:.0f}',
        'Internal Delays': '{:.0f}', 'External Delays': '{:.0f}'
    }
else:
    # Hide Lump/Fines columns for non-Iron Ore commodities
    display_cols = [c for c in df_forecast.columns if c not in ['Lump Tonnes', 'Fines Tonnes']]
    format_dict = {
        'Tonnes': '{:,.0f}', 'TPH': '{:,.0f}',
        'Run Hours': '{:,.0f}', 'Run Hours / Day': '{:.2f}',
        'Availability': '{:.1%}', 'Utilisation': '{:.1%}', 'Effective Utilisation': '{:.1%}',
        'Planned Maint': '{:.0f}', 'Unplanned': '{:.0f}',
        'Internal Delays': '{:.0f}', 'External Delays': '{:.0f}'
    }

st.dataframe(
    df_forecast[display_cols].style.format(format_dict),
    use_container_width=True, hide_index=True
)

# Charts
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(df_forecast, x='Month', y='Tonnes', color_discrete_sequence=['#4CAF50'])
    fig.update_layout(title='Monthly Production Forecast', xaxis_title='', yaxis_title='Tonnes')
    fig.update_yaxes(tickformat=',')
    fig.add_hline(y=contract_tonnes/12, line_dash="dash", line_color="red", annotation_text="Target Avg")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_forecast['Month'], y=df_forecast['Availability']*100, name='Availability', line=dict(color='#2196F3')))
    fig.add_trace(go.Scatter(x=df_forecast['Month'], y=df_forecast['Utilisation']*100, name='Utilisation', line=dict(color='#FF9800')))
    fig.update_layout(title='Availability & Utilisation', xaxis_title='', yaxis_title='%')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Export
st.subheader("📥 Export Report")

col1, col2 = st.columns(2)

with col1:
    html_report = create_html_report(
        df_forecast, df_historical, site_config['name'], contract_tonnes,
        lump_split, stats, required_tph, selected_commodity
    )
    st.download_button(
        "📄 Download HTML Report",
        data=html_report,
        file_name=f"CSI_TUM_Report_{site_config['name'].replace(' ', '_').replace('-', '')}_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html"
    )
    st.caption("Open in browser → Print → Save as PDF")

with col2:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_forecast.to_excel(writer, sheet_name='FY27 Forecast', index=False)
        df_historical.to_excel(writer, sheet_name='Historical Data', index=False)
    output.seek(0)
    st.download_button(
        "📥 Download Excel Data",
        data=output,
        file_name=f"TUM_Data_{site_config['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Footer
st.markdown("---")
st.caption(f"TUM Forecasting Assistant v3.0 | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
