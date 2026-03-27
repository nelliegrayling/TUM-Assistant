"""
TUM Forecasting Assistant v2.0
Time Usage Model for Crushing Plant Performance Forecasting

Purpose: Use historical maintenance data to forecast crushing plant performance
and calculate the TPH required to meet contract tonnage targets.
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

# Constants
CALENDAR_HOURS_PER_MONTH = {
    'Jul': 31 * 24, 'Aug': 31 * 24, 'Sep': 30 * 24, 'Oct': 31 * 24,
    'Nov': 30 * 24, 'Dec': 31 * 24, 'Jan': 31 * 24, 'Feb': 28 * 24,
    'Mar': 31 * 24, 'Apr': 30 * 24, 'May': 31 * 24, 'Jun': 30 * 24
}

FY_MONTHS = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']


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


def load_uploaded_data(uploaded_file):
    """Load data from uploaded file (CSV or Excel)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel."

        # Clean up
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Normalize column names
        if 'Crushed Tonnes' in df.columns and 'Tonnes' not in df.columns:
            df = df.rename(columns={'Crushed Tonnes': 'Tonnes'})

        # Parse percentage columns
        pct_cols = ['Availability', 'Utilisation', 'Effective Utilisation']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_percentage)

        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def calculate_historical_averages(df):
    """Calculate average maintenance hours and performance metrics from historical data."""
    averages = {}

    # Maintenance averages
    if 'Planned Maint' in df.columns:
        averages['planned_maint'] = df['Planned Maint'].mean()
    elif 'Planned Maint.' in df.columns:
        averages['planned_maint'] = df['Planned Maint.'].mean()
    else:
        averages['planned_maint'] = 0

    if 'Unplanned' in df.columns:
        averages['unplanned'] = df['Unplanned'].mean()
    elif 'Unplanned Maint.' in df.columns:
        averages['unplanned'] = df['Unplanned Maint.'].mean()
    else:
        averages['unplanned'] = 0

    averages['internal_delays'] = df['Internal Delays'].mean() if 'Internal Delays' in df.columns else 0
    averages['external_delays'] = df['External Delays'].mean() if 'External Delays' in df.columns else 0

    # Performance averages
    averages['avg_tph'] = df['TPH'].mean() if 'TPH' in df.columns else 0
    averages['avg_availability'] = df['Availability'].mean() if 'Availability' in df.columns else 0
    averages['avg_utilisation'] = df['Utilisation'].mean() if 'Utilisation' in df.columns else 0
    averages['avg_run_hours'] = df['Run Hours'].mean() if 'Run Hours' in df.columns else 0
    averages['avg_tonnes'] = df['Tonnes'].mean() if 'Tonnes' in df.columns else 0

    # Total historical
    averages['total_tonnes'] = df['Tonnes'].sum() if 'Tonnes' in df.columns else 0
    averages['total_run_hours'] = df['Run Hours'].sum() if 'Run Hours' in df.columns else 0
    averages['months_of_data'] = len(df)

    return averages


def generate_forecast(contract_tonnes, lump_split, historical_averages, forecast_year="27"):
    """Generate FY forecast based on historical averages and contract target."""

    forecast_data = []

    # Use historical averages for maintenance
    avg_planned = historical_averages['planned_maint']
    avg_unplanned = historical_averages['unplanned']
    avg_internal = historical_averages['internal_delays']
    avg_external = historical_averages['external_delays']

    total_run_hours = 0
    monthly_run_hours = []

    # Calculate run hours for each month
    for month in FY_MONTHS:
        calendar_hours = CALENDAR_HOURS_PER_MONTH[month]

        # Available hours (after maintenance)
        unavailable = avg_planned + avg_unplanned
        available_hours = calendar_hours - unavailable

        # Run hours (after delays)
        unproductive = avg_internal + avg_external
        run_hours = available_hours - unproductive
        run_hours = max(0, run_hours)  # Can't be negative

        # Calculate KPIs
        availability = available_hours / calendar_hours if calendar_hours > 0 else 0
        utilisation = run_hours / available_hours if available_hours > 0 else 0

        monthly_run_hours.append({
            'month': month,
            'calendar_hours': calendar_hours,
            'available_hours': available_hours,
            'run_hours': run_hours,
            'availability': availability,
            'utilisation': utilisation,
            'planned_maint': avg_planned,
            'unplanned': avg_unplanned,
            'internal_delays': avg_internal,
            'external_delays': avg_external
        })

        total_run_hours += run_hours

    # Calculate required TPH to meet contract target
    required_tph = contract_tonnes / total_run_hours if total_run_hours > 0 else 0

    # Build forecast with tonnes
    for m in monthly_run_hours:
        month_suffix = "-26" if m['month'] in ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] else f"-{forecast_year}"
        tonnes = required_tph * m['run_hours']
        lump_tonnes = tonnes * lump_split
        fines_tonnes = tonnes * (1 - lump_split)

        forecast_data.append({
            'Month': f"{m['month']}{month_suffix}",
            'Tonnes': round(tonnes),
            'TPH': round(required_tph),
            'Lump Tonnes': round(lump_tonnes),
            'Fines Tonnes': round(fines_tonnes),
            'Run Hours': round(m['run_hours']),
            'Run Hours / Day': round(m['run_hours'] / (CALENDAR_HOURS_PER_MONTH[m['month']] / 24), 2),
            'Availability': m['availability'],
            'Utilisation': m['utilisation'],
            'Effective Utilisation': m['availability'] * m['utilisation'],
            'Planned Maint': round(m['planned_maint']),
            'Unplanned': round(m['unplanned']),
            'Internal Delays': round(m['internal_delays']),
            'External Delays': round(m['external_delays'])
        })

    df_forecast = pd.DataFrame(forecast_data)

    return df_forecast, required_tph, total_run_hours


def create_html_report(df_forecast, df_historical, site_name, contract_tonnes, lump_split,
                       historical_averages, required_tph):
    """Create HTML report matching CSI Mining Services format."""

    total_tonnes = df_forecast['Tonnes'].sum()
    avg_availability = df_forecast['Availability'].mean() * 100
    avg_utilisation = df_forecast['Utilisation'].mean() * 100
    total_run_hours = df_forecast['Run Hours'].sum()

    gap = contract_tonnes - total_tonnes
    gap_status = "ON TRACK" if abs(gap) < contract_tonnes * 0.01 else ("SHORTFALL" if gap > 0 else "SURPLUS")

    # Planned maintenance totals
    total_planned = df_forecast['Planned Maint'].sum()
    total_unplanned = df_forecast['Unplanned'].sum()
    total_internal = df_forecast['Internal Delays'].sum()
    total_external = df_forecast['External Delays'].sum()

    # Build forecast rows
    forecast_rows = ""
    for _, row in df_forecast.iterrows():
        forecast_rows += f"""
        <tr>
            <td>{row['Month']}</td>
            <td>{row['Tonnes']:,.0f}</td>
            <td>{row['TPH']:,.0f}</td>
            <td>{row['Lump Tonnes']:,.0f}</td>
            <td>{row['Fines Tonnes']:,.0f}</td>
            <td>{row['Run Hours']:,.0f}</td>
            <td>{row['Run Hours / Day']:.2f}</td>
            <td>{row['Availability']*100:.1f}%</td>
            <td>{row['Utilisation']*100:.1f}%</td>
            <td>{row['Planned Maint']:.0f}</td>
            <td>{row['Unplanned']:.0f}</td>
            <td>{row['Internal Delays']:.0f}</td>
            <td>{row['External Delays']:.0f}</td>
        </tr>"""

    # Build historical rows if available
    historical_section = ""
    if df_historical is not None and len(df_historical) > 0:
        hist_rows = ""
        for _, row in df_historical.iterrows():
            planned = row.get('Planned Maint', row.get('Planned Maint.', 0))
            unplanned = row.get('Unplanned', row.get('Unplanned Maint.', 0))
            hist_rows += f"""
            <tr>
                <td>{row.get('Month', '')}</td>
                <td>{row.get('Tonnes', 0):,.0f}</td>
                <td>{row.get('TPH', 0):,.0f}</td>
                <td>{row.get('Run Hours', 0):,.0f}</td>
                <td>{row.get('Availability', 0)*100:.1f}%</td>
                <td>{row.get('Utilisation', 0)*100:.1f}%</td>
                <td>{planned:.0f}</td>
                <td>{unplanned:.0f}</td>
                <td>{row.get('Internal Delays', 0):.0f}</td>
                <td>{row.get('External Delays', 0):.0f}</td>
            </tr>"""

        historical_section = f"""
        <div class="section-title">Detailed Monthly Historical Breakdown</div>
        <table>
            <thead>
                <tr>
                    <th>Month</th><th>Tonnes</th><th>TPH</th><th>Run Hours</th>
                    <th>Availability</th><th>Utilisation</th>
                    <th>Planned Maint</th><th>Unplanned</th><th>Internal</th><th>External</th>
                </tr>
            </thead>
            <tbody>
                <tr class="total-row">
                    <td><strong>Total/Avg</strong></td>
                    <td><strong>{historical_averages['total_tonnes']:,.0f}</strong></td>
                    <td><strong>{historical_averages['avg_tph']:,.0f}</strong></td>
                    <td><strong>{historical_averages['total_run_hours']:,.0f}</strong></td>
                    <td><strong>{historical_averages['avg_availability']*100:.1f}%</strong></td>
                    <td><strong>{historical_averages['avg_utilisation']*100:.1f}%</strong></td>
                    <td><strong>{historical_averages['planned_maint']*historical_averages['months_of_data']:,.0f}</strong></td>
                    <td><strong>{historical_averages['unplanned']*historical_averages['months_of_data']:,.0f}</strong></td>
                    <td><strong>{historical_averages['internal_delays']*historical_averages['months_of_data']:,.0f}</strong></td>
                    <td><strong>{historical_averages['external_delays']*historical_averages['months_of_data']:,.0f}</strong></td>
                </tr>
                {hist_rows}
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
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a1a; color: #fff; font-size: 11px; }}
            .page {{ page-break-after: always; }}
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
            .section-title {{ font-size: 16px; font-weight: bold; margin: 20px 0 12px 0; }}
            .assumptions-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px 20px; background: #2a2a2a; padding: 15px 20px; border-radius: 6px; }}
            .assumption-item {{ display: flex; justify-content: space-between; font-size: 11px; }}
            .assumption-label {{ color: #888; }}
            .assumption-value {{ font-weight: 600; }}
            .gap-box {{ background: #2a2a2a; border-radius: 6px; padding: 15px 20px; margin: 15px 0; }}
            .gap-status {{ display: inline-block; padding: 6px 14px; border-radius: 4px; font-weight: bold; }}
            .gap-status.on-track {{ background: #1b5e20; color: #4CAF50; }}
            .gap-status.shortfall {{ background: #b71c1c; color: #f44336; }}
            .gap-status.surplus {{ background: #0d47a1; color: #2196F3; }}
            .gap-detail {{ margin-left: 20px; color: #888; }}
            .warning-box {{ background: #4a3000; border: 1px solid #ff9800; border-radius: 6px; padding: 15px; margin: 15px 0; }}
            .warning-box strong {{ color: #ff9800; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 9px; margin: 10px 0; }}
            th {{ background: #333; padding: 6px 3px; text-align: right; font-weight: 600; }}
            th:first-child {{ text-align: left; }}
            td {{ padding: 5px 3px; border-bottom: 1px solid #333; text-align: right; }}
            td:first-child {{ text-align: left; }}
            .total-row {{ background: #2d4a2d; }}
            .total-row td {{ color: #4CAF50; font-weight: bold; }}
            .section-row {{ background: #252525; }}
            .section-row td {{ color: #888; font-size: 8px; text-transform: uppercase; }}
            .methodology {{ background: #2a2a2a; padding: 20px; border-radius: 6px; margin-top: 20px; }}
            .methodology h3 {{ color: #4CAF50; font-size: 12px; margin: 12px 0 6px 0; }}
            .methodology p {{ color: #aaa; font-size: 11px; margin-left: 12px; line-height: 1.5; }}
            .footer {{ text-align: center; padding: 15px; color: #555; font-size: 10px; border-top: 1px solid #333; margin-top: 20px; }}
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
                <div class="kpi-card">
                    <div class="kpi-label">Total Production</div>
                    <div class="kpi-value green">{total_tonnes:,.0f} t</div>
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
                <div class="assumption-item"><span class="assumption-label">Site:</span><span class="assumption-value">{site_name}</span></div>
                <div class="assumption-item"><span class="assumption-label">Calculated TPH:</span><span class="assumption-value">{required_tph:,.0f}</span></div>
                <div class="assumption-item"><span class="assumption-label">Historical Avg TPH:</span><span class="assumption-value">{historical_averages['avg_tph']:,.0f}</span></div>
                <div class="assumption-item"><span class="assumption-label">Avg Planned Maint:</span><span class="assumption-value">{historical_averages['planned_maint']:.1f} h/mo</span></div>
                <div class="assumption-item"><span class="assumption-label">Forecast Year:</span><span class="assumption-value">FY27</span></div>
                <div class="assumption-item"><span class="assumption-label">Lump Split:</span><span class="assumption-value">{lump_split*100:.0f}%</span></div>
                <div class="assumption-item"><span class="assumption-label">Avg Unplanned:</span><span class="assumption-value">{historical_averages['unplanned']:.1f} h/mo</span></div>
                <div class="assumption-item"><span class="assumption-label">Avg Internal Delays:</span><span class="assumption-value">{historical_averages['internal_delays']:.1f} h/mo</span></div>
                <div class="assumption-item"><span class="assumption-label">Operation:</span><span class="assumption-value">24h</span></div>
                <div class="assumption-item"><span class="assumption-label">Fines Split:</span><span class="assumption-value">{(1-lump_split)*100:.0f}%</span></div>
                <div class="assumption-item"><span class="assumption-label">Avg External Delays:</span><span class="assumption-value">{historical_averages['external_delays']:.1f} h/mo</span></div>
                <div class="assumption-item"><span class="assumption-label">Historical Data:</span><span class="assumption-value">{historical_averages['months_of_data']} months</span></div>
                <div class="assumption-item"><span class="assumption-label">Commodity:</span><span class="assumption-value">Iron Ore</span></div>
                <div class="assumption-item"><span class="assumption-label">Contract Tonnes:</span><span class="assumption-value">{contract_tonnes:,.0f}</span></div>
                <div class="assumption-item"></div>
                <div class="assumption-item"></div>
            </div>

            {"<div class='warning-box'><strong>⚠️ TPH Warning:</strong> Calculated TPH (" + f"{required_tph:,.0f}" + ") differs from historical average (" + f"{historical_averages['avg_tph']:,.0f}" + ") by " + f"{abs(required_tph - historical_averages['avg_tph']) / historical_averages['avg_tph'] * 100:.1f}%" + ". Please verify this target is achievable.</div>" if historical_averages['avg_tph'] > 0 and abs(required_tph - historical_averages['avg_tph']) / historical_averages['avg_tph'] > 0.15 else ""}

            <div class="section-title">Scenario & Gap Analysis</div>
            <div class="gap-box">
                <span class="gap-status {'on-track' if gap_status == 'ON TRACK' else 'shortfall' if gap_status == 'SHORTFALL' else 'surplus'}">{gap_status}</span>
                <span class="gap-detail">Gap: {gap:+,.0f} tonnes</span>
            </div>

            <div class="section-title">Detailed Monthly Forecast - FY27</div>
            <table>
                <thead>
                    <tr>
                        <th>Month</th><th>Tonnes</th><th>TPH</th><th>Lump</th><th>Fines</th>
                        <th>Run Hrs</th><th>Hrs/Day</th><th>Avail</th><th>Util</th>
                        <th>Planned</th><th>Unplanned</th><th>Internal</th><th>External</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="total-row">
                        <td><strong>Total</strong></td>
                        <td><strong>{total_tonnes:,.0f}</strong></td>
                        <td><strong>{required_tph:,.0f}</strong></td>
                        <td><strong>{total_tonnes * lump_split:,.0f}</strong></td>
                        <td><strong>{total_tonnes * (1-lump_split):,.0f}</strong></td>
                        <td><strong>{total_run_hours:,.0f}</strong></td>
                        <td><strong>{total_run_hours / 365:.2f}</strong></td>
                        <td><strong>{avg_availability:.1f}%</strong></td>
                        <td><strong>{avg_utilisation:.1f}%</strong></td>
                        <td><strong>{total_planned:.0f}</strong></td>
                        <td><strong>{total_unplanned:.0f}</strong></td>
                        <td><strong>{total_internal:.0f}</strong></td>
                        <td><strong>{total_external:.0f}</strong></td>
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

            <div class="footer">
                CSI Mining Services - Time Usage Model | Generated by TUM Forecasting Assistant v2.0
            </div>
        </div>
    </body>
    </html>
    """

    return html


# =============================================================================
# MAIN APP
# =============================================================================

st.title("⚙️ TUM Forecasting Assistant v2.0")
st.markdown("**Time Usage Model for Crushing Plant Performance Forecasting**")
st.markdown("---")

# Sidebar - Inputs
with st.sidebar:
    st.header("📊 Configuration")

    site_name = st.text_input("Site Name", value="Mt Whaleback - Crushing Circuit")

    st.markdown("---")

    st.subheader("📁 Historical Data")
    uploaded_file = st.file_uploader(
        "Upload Historical Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with historical maintenance and performance data"
    )

    st.markdown("---")

    st.subheader("🎯 Forecast Targets")

    contract_tonnes = st.number_input(
        "Contract Tonnes Target",
        min_value=0,
        value=12000000,
        step=100000,
        format="%d",
        help="Target total production for the forecast year"
    )

    lump_split = st.slider(
        "Lump Split %",
        min_value=0,
        max_value=100,
        value=40,
        help="Percentage of production as lump ore"
    ) / 100

    st.caption(f"Fines Split: {(1-lump_split)*100:.0f}%")

# Main content
if uploaded_file is not None:
    # Load data
    df_historical, error = load_uploaded_data(uploaded_file)

    if error:
        st.error(error)
    else:
        st.success(f"✅ Loaded {len(df_historical)} months of historical data")

        # Calculate historical averages
        hist_avg = calculate_historical_averages(df_historical)

        # Display historical summary
        st.header("📜 Historical Data Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Historical Avg TPH", f"{hist_avg['avg_tph']:,.0f}")
        with col2:
            st.metric("Avg Availability", f"{hist_avg['avg_availability']*100:.1f}%")
        with col3:
            st.metric("Avg Utilisation", f"{hist_avg['avg_utilisation']*100:.1f}%")
        with col4:
            st.metric("Months of Data", f"{hist_avg['months_of_data']}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Planned Maint", f"{hist_avg['planned_maint']:.1f} hrs/mo")
        with col2:
            st.metric("Avg Unplanned", f"{hist_avg['unplanned']:.1f} hrs/mo")
        with col3:
            st.metric("Avg Internal Delays", f"{hist_avg['internal_delays']:.1f} hrs/mo")
        with col4:
            st.metric("Avg External Delays", f"{hist_avg['external_delays']:.1f} hrs/mo")

        with st.expander("📋 View Historical Data", expanded=False):
            st.dataframe(df_historical, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Generate Forecast
        st.header("🔮 FY27 Forecast")

        df_forecast, required_tph, total_run_hours = generate_forecast(
            contract_tonnes, lump_split, hist_avg
        )

        # TPH Comparison
        tph_diff_pct = (required_tph - hist_avg['avg_tph']) / hist_avg['avg_tph'] * 100 if hist_avg['avg_tph'] > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Required TPH",
                f"{required_tph:,.0f}",
                delta=f"{tph_diff_pct:+.1f}% vs historical",
                delta_color="normal" if abs(tph_diff_pct) < 10 else "inverse"
            )
        with col2:
            st.metric("Total Run Hours", f"{total_run_hours:,.0f}")
        with col3:
            st.metric("Forecast Total", f"{df_forecast['Tonnes'].sum():,.0f} t")

        # Warning if TPH is significantly different
        if abs(tph_diff_pct) > 15:
            st.warning(f"⚠️ **TPH Warning:** Required TPH ({required_tph:,.0f}) is {abs(tph_diff_pct):.1f}% {'higher' if tph_diff_pct > 0 else 'lower'} than historical average ({hist_avg['avg_tph']:,.0f}). Please verify this target is achievable.")

        # Forecast Table
        st.subheader("Monthly Breakdown")

        # Format for display
        display_df = df_forecast.copy()
        st.dataframe(
            display_df.style.format({
                'Tonnes': '{:,.0f}',
                'TPH': '{:,.0f}',
                'Lump Tonnes': '{:,.0f}',
                'Fines Tonnes': '{:,.0f}',
                'Run Hours': '{:,.0f}',
                'Run Hours / Day': '{:.2f}',
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

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Monthly Production")
            fig_tonnes = px.bar(
                df_forecast, x='Month', y='Tonnes',
                color_discrete_sequence=['#4CAF50']
            )
            fig_tonnes.update_layout(xaxis_title='Month', yaxis_title='Tonnes')
            fig_tonnes.update_yaxes(tickformat=',')
            fig_tonnes.add_hline(
                y=contract_tonnes/12, line_dash="dash", line_color="red",
                annotation_text=f"Target Avg: {contract_tonnes/12:,.0f}"
            )
            st.plotly_chart(fig_tonnes, use_container_width=True)

        with col2:
            st.subheader("Availability & Utilisation")
            fig_kpi = go.Figure()
            fig_kpi.add_trace(go.Scatter(
                x=df_forecast['Month'], y=df_forecast['Availability']*100,
                name='Availability', mode='lines+markers', line=dict(color='#2196F3')
            ))
            fig_kpi.add_trace(go.Scatter(
                x=df_forecast['Month'], y=df_forecast['Utilisation']*100,
                name='Utilisation', mode='lines+markers', line=dict(color='#FF9800')
            ))
            fig_kpi.update_layout(xaxis_title='Month', yaxis_title='%', legend_title='Metric')
            st.plotly_chart(fig_kpi, use_container_width=True)

        st.markdown("---")

        # Export
        st.header("📥 Export Report")

        html_report = create_html_report(
            df_forecast, df_historical, site_name, contract_tonnes,
            lump_split, hist_avg, required_tph
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report,
                file_name=f"CSI_TUM_Report_{site_name.replace(' ', '_').replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )
            st.caption("Open in browser → Print → Save as PDF")

        with col2:
            # Excel export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_forecast.to_excel(writer, sheet_name='FY27 Forecast', index=False)
                df_historical.to_excel(writer, sheet_name='Historical Data', index=False)
            output.seek(0)

            st.download_button(
                label="📥 Download Excel Data",
                data=output,
                file_name=f"TUM_Data_{site_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("👈 Upload historical data in the sidebar to generate a forecast.")

    st.markdown("""
    ### How to use this app:

    1. **Upload Historical Data** - CSV or Excel file with columns:
       - Month, Tonnes, TPH, Run Hours
       - Availability, Utilisation
       - Planned Maint, Unplanned, Internal Delays, External Delays

    2. **Set Contract Target** - Enter the target tonnes for the forecast year

    3. **Set Lump/Fines Split** - Percentage of production as lump ore

    4. **Generate Forecast** - The app will:
       - Calculate average maintenance hours from historical data
       - Determine run hours based on maintenance pattern
       - Back-calculate TPH required to meet your target
       - Generate monthly breakdown with all KPIs

    5. **Export Report** - Download HTML report (matches CSI format) or Excel data
    """)

# Footer
st.markdown("---")
st.caption(f"TUM Forecasting Assistant v2.0 | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
