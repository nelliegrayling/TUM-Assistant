"""
TUM Forecasting Assistant v4.0
Time Usage Model for Crushing Plant Performance Forecasting

Reliability improvements:
- Data validation on load
- Offline mode with local caching
- Audit trail logging
- Division by zero protection
- Data freshness indicators
- Input bounds validation
- Forecast comparison view
- Export versioning
- Enhanced sanity checks
- Automatic backups
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import os
import json
import pickle
import logging
from logging.handlers import RotatingFileHandler
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONFIGURATION
# =============================================================================

# Page config
st.set_page_config(
    page_title="TUM Forecasting Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
APP_DIR = Path(__file__).parent
DATA_PATH = Path(r"\\p1fs002\CSI - Technical Services\TUM Data for Accountants FY27")
CACHE_DIR = APP_DIR / ".cache"
LOG_DIR = APP_DIR / "logs"
BACKUP_DIR = APP_DIR / "backups"
EXPORT_DIR = APP_DIR / "exports"

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
COMMODITIES = ['Iron Ore', 'Lithium', 'Gold', 'Garnet']

# Validation thresholds
MAX_TPH = 5000
MAX_MONTHLY_HOURS = 744
TPH_WARNING_THRESHOLD = 15  # percent
TPH_ERROR_THRESHOLD = 30  # percent

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure audit logging with rotation."""
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger('TUM_Audit')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            LOG_DIR / "tum_audit.log",
            maxBytes=5*1024*1024,
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

audit_logger = setup_logging()

def log_action(action: str, details: dict = None):
    """Log a user action with optional details."""
    user = os.environ.get('USERNAME', 'unknown')
    msg = f"USER={user} | ACTION={action}"
    if details:
        detail_str = ' | '.join(f'{k}={v}' for k, v in details.items())
        msg += f" | {detail_str}"
    audit_logger.info(msg)

# =============================================================================
# CACHING FUNCTIONS (Offline Mode)
# =============================================================================

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)

def get_cache_metadata_path():
    return CACHE_DIR / "cache_metadata.json"

def load_cache_metadata() -> dict:
    """Load cache metadata."""
    meta_path = get_cache_metadata_path()
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache_metadata(metadata: dict):
    """Save cache metadata."""
    ensure_cache_dir()
    with open(get_cache_metadata_path(), 'w') as f:
        json.dump(metadata, f, indent=2)

def cache_site_data(site_key: str, df: pd.DataFrame):
    """Cache site data locally for offline use."""
    ensure_cache_dir()
    cache_path = CACHE_DIR / f"{site_key}_data.pkl"
    df.to_pickle(cache_path)

    metadata = load_cache_metadata()
    metadata[site_key] = {
        "cached_at": datetime.now().isoformat(),
        "rows": len(df),
        "file": str(cache_path)
    }
    save_cache_metadata(metadata)
    log_action("DATA_CACHED", {"site": site_key, "rows": len(df)})

def load_cached_data(site_key: str):
    """Load cached data if available. Returns (df, cached_at) or (None, None)."""
    cache_path = CACHE_DIR / f"{site_key}_data.pkl"
    if cache_path.exists():
        try:
            df = pd.read_pickle(cache_path)
            metadata = load_cache_metadata()
            cached_at_str = metadata.get(site_key, {}).get("cached_at")
            cached_at = datetime.fromisoformat(cached_at_str) if cached_at_str else None
            return df, cached_at
        except Exception as e:
            log_action("CACHE_LOAD_ERROR", {"site": site_key, "error": str(e)})
    return None, None

# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================

def backup_forecast(site_key: str, df_forecast: pd.DataFrame, contract: float):
    """Create backup of forecast before generating new one."""
    BACKUP_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now()
    backup_file = BACKUP_DIR / f"{site_key}_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

    backup_data = {
        'site': site_key,
        'timestamp': timestamp.isoformat(),
        'contract_tonnes': contract,
        'total_tonnes': float(df_forecast['Tonnes'].sum()),
        'required_tph': float(df_forecast['TPH'].iloc[0]),
        'forecast': df_forecast.to_dict(orient='records')
    }

    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2, default=str)

    # Cleanup old backups (keep last 10 per site)
    site_backups = sorted(BACKUP_DIR.glob(f"{site_key}_backup_*.json"))
    for old_backup in site_backups[:-10]:
        old_backup.unlink()

    log_action("FORECAST_BACKED_UP", {"site": site_key, "file": backup_file.name})
    return backup_file

def get_previous_forecast(site_key: str):
    """Get the most recent backup for comparison."""
    site_backups = sorted(BACKUP_DIR.glob(f"{site_key}_backup_*.json"))
    if site_backups:
        try:
            with open(site_backups[-1], 'r') as f:
                return json.load(f)
        except:
            pass
    return None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero/NaN denominator."""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator

def parse_percentage(val):
    """Convert percentage string to float (0-1 range)."""
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

def generate_versioned_filename(base_name: str, extension: str) -> str:
    """Generate unique filename with version number to prevent overwrites."""
    EXPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d')

    version = 1
    while True:
        if version == 1:
            filename = f"{base_name}_{timestamp}.{extension}"
        else:
            filename = f"{base_name}_{timestamp}_v{version}.{extension}"

        if not (EXPORT_DIR / filename).exists():
            return filename
        version += 1

        if version > 99:
            filename = f"{base_name}_{timestamp}_{datetime.now().strftime('%H%M%S')}.{extension}"
            return filename

# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df: pd.DataFrame, site_key: str) -> list:
    """
    Validate loaded data for anomalies.
    Returns list of warning/error messages.
    """
    warnings = []

    # TPH validation
    if 'TPH' in df.columns:
        high_tph = df[df['TPH'] > MAX_TPH]
        if len(high_tph) > 0:
            warnings.append(f"DATA: {len(high_tph)} rows with TPH > {MAX_TPH}")

        negative_tph = df[df['TPH'] < 0]
        if len(negative_tph) > 0:
            warnings.append(f"ERROR: {len(negative_tph)} rows with negative TPH")

    # Availability validation (should be 0-1 after parsing)
    if 'Availability' in df.columns:
        invalid_avail = df[(df['Availability'] > 1.0) | (df['Availability'] < 0)]
        if len(invalid_avail) > 0:
            warnings.append(f"DATA: {len(invalid_avail)} rows with invalid Availability (not 0-100%)")

    # Utilisation validation
    if 'Utilisation' in df.columns:
        invalid_util = df[(df['Utilisation'] > 1.0) | (df['Utilisation'] < 0)]
        if len(invalid_util) > 0:
            warnings.append(f"DATA: {len(invalid_util)} rows with invalid Utilisation (not 0-100%)")

    # Negative values check
    numeric_cols = ['Tonnes', 'Run Hours', 'Planned Maint', 'Unplanned',
                    'Internal Delays', 'External Delays']
    for col in numeric_cols:
        if col in df.columns:
            negative = df[df[col] < 0]
            if len(negative) > 0:
                warnings.append(f"DATA: {len(negative)} rows with negative {col}")

    # Hours consistency check
    if 'Run Hours' in df.columns:
        excessive_hours = df[df['Run Hours'] > MAX_MONTHLY_HOURS]
        if len(excessive_hours) > 0:
            warnings.append(f"DATA: {len(excessive_hours)} rows with Run Hours > {MAX_MONTHLY_HOURS}")

    if warnings:
        log_action("DATA_VALIDATION_WARNINGS", {"site": site_key, "count": len(warnings)})

    return warnings

def validate_inputs(contract_tonnes: float, lump_split: float,
                   commodity: str, stats: dict) -> list:
    """Validate user inputs and return warnings."""
    warnings = []

    if contract_tonnes <= 0:
        warnings.append("Contract tonnes must be positive")

    if lump_split < 0 or lump_split > 1:
        warnings.append("Lump split must be between 0% and 100%")

    if commodity == "Iron Ore" and lump_split == 0:
        warnings.append("Iron Ore typically has a lump split > 0%")

    if commodity != "Iron Ore" and lump_split > 0:
        warnings.append(f"{commodity} typically has no lump/fines split")

    # Check if contract is reasonable compared to historical
    if stats.get('total_tonnes', 0) > 0:
        annual_historical = stats['total_tonnes'] / stats.get('months_of_data', 12) * 12
        if contract_tonnes < annual_historical * 0.1:
            warnings.append("Contract target seems unusually low compared to historical data")
        elif contract_tonnes > annual_historical * 5:
            warnings.append("Contract target seems unusually high compared to historical data")

    return warnings

def generate_sanity_warnings(required_tph: float, stats: dict, contract_tonnes: float) -> list:
    """Generate sanity check warnings for forecast."""
    warnings = []

    if stats.get('avg_tph', 0) > 0:
        tph_diff_pct = safe_divide(required_tph - stats['avg_tph'], stats['avg_tph'], 0) * 100

        if tph_diff_pct > TPH_ERROR_THRESHOLD:
            warnings.append({
                'level': 'error',
                'message': f"Required TPH ({required_tph:,.0f}) is {tph_diff_pct:.0f}% HIGHER "
                          f"than historical ({stats['avg_tph']:,.0f}). This may be unachievable."
            })
        elif tph_diff_pct > TPH_WARNING_THRESHOLD:
            warnings.append({
                'level': 'warning',
                'message': f"Required TPH is {tph_diff_pct:.0f}% higher than historical average. Verify achievability."
            })
        elif tph_diff_pct < -TPH_ERROR_THRESHOLD:
            warnings.append({
                'level': 'info',
                'message': f"Required TPH is {abs(tph_diff_pct):.0f}% lower than historical. "
                          f"Consider if this leaves capacity for additional contracts."
            })

    # Check if contract is achievable
    if stats.get('avg_tph', 0) > 0 and stats.get('months_of_data', 0) > 0:
        avg_monthly_run_hours = safe_divide(stats.get('total_run_hours', 0), stats['months_of_data'], 0)
        max_achievable = stats['avg_tph'] * avg_monthly_run_hours * 12
        if contract_tonnes > max_achievable * 1.2:
            warnings.append({
                'level': 'warning',
                'message': f"Contract ({contract_tonnes:,.0f}t) exceeds 120% of estimated max capacity ({max_achievable:,.0f}t/year)"
            })

    return warnings

# =============================================================================
# DATA FRESHNESS
# =============================================================================

def get_data_freshness(site_key: str):
    """Get last modification time and freshness status of source data file."""
    if site_key not in SITES:
        return None, "unknown"

    filepath = DATA_PATH / SITES[site_key]["file"]

    try:
        if filepath.exists():
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age = datetime.now() - mtime

            if age.days > 30:
                status = "stale"
            elif age.days > 7:
                status = "warning"
            else:
                status = "fresh"

            return mtime, status
    except:
        pass
    return None, "unavailable"

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_site_data(site_key: str):
    """
    Load historical data for a site with offline fallback.
    Returns: (df, error_message, using_cache, validation_warnings)
    """
    if site_key not in SITES:
        return None, "Site not found", False, []

    site_config = SITES[site_key]
    filepath = DATA_PATH / site_config["file"]
    using_cache = False
    validation_warnings = []

    try:
        if filepath.exists():
            # Network available - load fresh data
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

            # Validate data
            validation_warnings = validate_data(df, site_key)

            # Cache for offline use
            cache_site_data(site_key, df)

            log_action("DATA_LOADED", {"site": site_key, "source": "network", "rows": len(df)})
            return df, None, False, validation_warnings
        else:
            raise FileNotFoundError("Network path not available")

    except Exception as e:
        # Try loading from cache
        cached_df, cached_at = load_cached_data(site_key)
        if cached_df is not None:
            log_action("DATA_LOADED", {"site": site_key, "source": "cache", "cached_at": str(cached_at)})
            cache_msg = f"Using cached data from {cached_at.strftime('%d/%m/%Y %H:%M')}" if cached_at else "Using cached data"
            validation_warnings = validate_data(cached_df, site_key)
            return cached_df, cache_msg, True, validation_warnings

        log_action("DATA_LOAD_FAILED", {"site": site_key, "error": str(e)})
        return None, f"Error: {str(e)} (no cache available)", False, []

# =============================================================================
# STATISTICS CALCULATION
# =============================================================================

def calculate_historical_stats(df: pd.DataFrame) -> dict:
    """Calculate statistics from historical data with safe operations."""
    stats = {}

    # Column name variations
    planned_col = 'Planned Maint' if 'Planned Maint' in df.columns else 'Planned Maint.'
    unplanned_col = 'Unplanned' if 'Unplanned' in df.columns else 'Unplanned Maint.'

    # Maintenance averages (per month) with NaN handling
    stats['avg_planned_maint'] = df[planned_col].mean() if planned_col in df.columns and len(df[planned_col].dropna()) > 0 else 0
    stats['avg_unplanned'] = df[unplanned_col].mean() if unplanned_col in df.columns and len(df[unplanned_col].dropna()) > 0 else 0
    stats['avg_internal_delays'] = df['Internal Delays'].mean() if 'Internal Delays' in df.columns and len(df['Internal Delays'].dropna()) > 0 else 0
    stats['avg_external_delays'] = df['External Delays'].mean() if 'External Delays' in df.columns and len(df['External Delays'].dropna()) > 0 else 0

    # Performance averages
    stats['avg_tph'] = df['TPH'].mean() if 'TPH' in df.columns and len(df['TPH'].dropna()) > 0 else 0
    stats['avg_availability'] = df['Availability'].mean() if 'Availability' in df.columns and len(df['Availability'].dropna()) > 0 else 0
    stats['avg_utilisation'] = df['Utilisation'].mean() if 'Utilisation' in df.columns and len(df['Utilisation'].dropna()) > 0 else 0
    stats['avg_run_hours'] = df['Run Hours'].mean() if 'Run Hours' in df.columns and len(df['Run Hours'].dropna()) > 0 else 0
    stats['avg_tonnes_per_month'] = df['Tonnes'].mean() if 'Tonnes' in df.columns and len(df['Tonnes'].dropna()) > 0 else 0

    # Totals
    stats['total_tonnes'] = df['Tonnes'].sum() if 'Tonnes' in df.columns else 0
    stats['total_run_hours'] = df['Run Hours'].sum() if 'Run Hours' in df.columns else 0
    stats['months_of_data'] = len(df)

    # Derived stats
    stats['total_unavailable_per_month'] = stats['avg_planned_maint'] + stats['avg_unplanned']
    stats['total_unproductive_per_month'] = stats['avg_internal_delays'] + stats['avg_external_delays']

    return stats

# =============================================================================
# FORECAST GENERATION
# =============================================================================

def generate_fy_forecast(contract_tonnes: float, lump_split: float, stats: dict, forecast_year: str = "27"):
    """Generate FY forecast based on historical stats and contract target."""

    forecast = []
    total_run_hours = 0

    for month in FY_MONTHS:
        cal_hours = CALENDAR_HOURS[month]

        # Available hours = Calendar - Planned - Unplanned
        available_hours = cal_hours - stats['avg_planned_maint'] - stats['avg_unplanned']
        available_hours = max(0, available_hours)

        # Run hours = Available - Internal - External
        run_hours = available_hours - stats['avg_internal_delays'] - stats['avg_external_delays']
        run_hours = max(0, run_hours)

        # KPIs with safe division
        availability = safe_divide(available_hours, cal_hours, 0)
        utilisation = safe_divide(run_hours, available_hours, 0)

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

    # Calculate required TPH with safe division
    required_tph = safe_divide(contract_tonnes, total_run_hours, 0)

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
            'Run Hours / Day': round(safe_divide(f['run_hours'], f['calendar_hours'] / 24, 0), 2),
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

# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def create_html_report(df_forecast, df_historical, site_name, contract_tonnes,
                       lump_split, stats, required_tph, commodity):
    """Generate HTML report matching CSI Mining Services format."""

    total_tonnes = df_forecast['Tonnes'].sum()
    avg_availability = df_forecast['Availability'].mean() * 100
    avg_utilisation = df_forecast['Utilisation'].mean() * 100
    total_run_hours = df_forecast['Run Hours'].sum()

    gap = total_tonnes - contract_tonnes
    gap_status = "ON TRACK" if abs(gap) < contract_tonnes * 0.01 else ("SURPLUS" if gap > 0 else "SHORTFALL")

    tph_diff_pct = safe_divide(required_tph - stats['avg_tph'], stats['avg_tph'], 0) * 100

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
            avail = row.get('Availability', 0)
            util = row.get('Utilisation', 0)
            avail_pct = avail * 100 if avail <= 1 else avail
            util_pct = util * 100 if util <= 1 else util
            historical_rows += f"""<tr>
                <td>{row.get('Month', '')}</td><td>{row.get('Tonnes', 0):,.0f}</td><td>{row.get('TPH', 0):,.0f}</td>
                <td>{row.get('Run Hours', 0):,.0f}</td>
                <td>{avail_pct:.1f}%</td><td>{util_pct:.1f}%</td>
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
        <strong>TPH Warning:</strong> Required TPH ({required_tph:,.0f}) is {abs(tph_diff_pct):.1f}%
        {'higher' if tph_diff_pct > 0 else 'lower'} than historical average ({stats['avg_tph']:,.0f}).
        Please verify this target is achievable.
    </div>
    """ if abs(tph_diff_pct) > TPH_WARNING_THRESHOLD else ""

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
        @media print {{ body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-right">
            <div class="project-name">Project: {site_name}</div>
            <div class="generated-date">Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
        </div>
        <div class="header-title">CSI MINING SERVICES</div>
        <div class="header-subtitle">TIME USAGE MODEL V4.0</div>
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
        </div>

        {warning_html}

        <div class="section-title">Scenario & Gap Analysis</div>
        <div class="gap-box">
            <span class="gap-status {gap_status.lower().replace(' ', '-')}">Status: {gap_status}</span>
            <span class="gap-detail">Gap: {gap:+,.0f} tonnes {'surplus' if gap > 0 else 'shortfall' if gap < 0 else ''}</span>
        </div>

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
                    <td><strong>{safe_divide(total_run_hours, 365, 0):.2f}</strong></td>
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
            <p><strong>6. Monthly Tonnes</strong> = TPH x Run Hours</p>
        </div>

        <div class="footer">CSI Mining Services - Time Usage Model | Generated by TUM Forecasting Assistant v4.0</div>
    </div>
</body>
</html>"""

    return html

# =============================================================================
# MAIN APP
# =============================================================================

st.title("TUM Forecasting Assistant v4.0")
st.markdown("**Time Usage Model for Crushing Plant Performance Forecasting**")

# Initialize session state for forecast history
if 'forecast_history' not in st.session_state:
    st.session_state.forecast_history = {}

# Sidebar
with st.sidebar:
    st.header("Site Selection")

    site_key = st.selectbox(
        "Select Site",
        options=list(SITES.keys()),
        format_func=lambda x: SITES[x]["name"]
    )

    site_config = SITES[site_key]

    # Data freshness indicator
    data_mtime, freshness_status = get_data_freshness(site_key)
    if data_mtime:
        age_days = (datetime.now() - data_mtime).days
        freshness_colors = {"fresh": "green", "warning": "orange", "stale": "red"}
        st.caption(f"Data updated: {data_mtime.strftime('%d/%m/%Y')} ({age_days}d ago)")
        if freshness_status == "stale":
            st.warning("Data >30 days old")

    st.markdown("---")
    st.header("Forecast Targets")

    contract_tonnes = st.number_input(
        "Contract Tonnes Target",
        min_value=100000,
        max_value=50000000,
        value=site_config["default_contract"],
        step=100000,
        format="%d"
    )

    st.markdown("---")
    st.header("Commodity")

    default_commodity = site_config.get("commodity", "Iron Ore")
    default_idx = COMMODITIES.index(default_commodity) if default_commodity in COMMODITIES else 0

    selected_commodity = st.selectbox(
        "Select Commodity",
        options=COMMODITIES,
        index=default_idx
    )

    # Lump/fines split for Iron Ore only
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
df_historical, error, using_cache, validation_warnings = load_site_data(site_key)

# Show cache/offline warning
if using_cache:
    st.warning(f"OFFLINE MODE: {error}")

if error and not using_cache:
    st.error(f"Error: {error}")
    st.stop()

# Show validation warnings
if validation_warnings:
    with st.expander(f"Data Validation Warnings ({len(validation_warnings)})", expanded=False):
        for warn in validation_warnings:
            st.warning(warn)

# Calculate stats
stats = calculate_historical_stats(df_historical)

# Input validation warnings
input_warnings = validate_inputs(contract_tonnes, lump_split, selected_commodity, stats)
for warn in input_warnings:
    st.warning(warn)

# Display header
commodity_icons = {'Iron Ore': '🔶', 'Lithium': '🔋', 'Gold': '🥇', 'Garnet': '💎'}
st.header(f"{site_config['name']}")
st.caption(f"{commodity_icons.get(selected_commodity, '⚙️')} **Commodity:** {selected_commodity}")

# Historical Summary
st.subheader("Historical Performance")

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

with st.expander("View Historical Data"):
    st.dataframe(df_historical, use_container_width=True, hide_index=True)

st.markdown("---")

# Generate Forecast
st.subheader("FY27 Forecast")

df_forecast, required_tph, total_run_hours = generate_fy_forecast(contract_tonnes, lump_split, stats)

# Backup previous forecast
backup_forecast(site_key, df_forecast, contract_tonnes)

# Log forecast generation
log_action("FORECAST_GENERATED", {
    "site": site_key,
    "contract": contract_tonnes,
    "required_tph": round(required_tph),
    "total_tonnes": df_forecast['Tonnes'].sum()
})

# Save to forecast history for comparison
if site_key not in st.session_state.forecast_history:
    st.session_state.forecast_history[site_key] = []
st.session_state.forecast_history[site_key].append({
    'timestamp': datetime.now(),
    'contract': contract_tonnes,
    'total_tonnes': df_forecast['Tonnes'].sum(),
    'required_tph': required_tph
})
st.session_state.forecast_history[site_key] = st.session_state.forecast_history[site_key][-5:]

# TPH comparison
tph_diff = safe_divide(required_tph - stats['avg_tph'], stats['avg_tph'], 0) * 100

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

# Sanity check warnings
sanity_warnings = generate_sanity_warnings(required_tph, stats, contract_tonnes)
for warn in sanity_warnings:
    if warn['level'] == 'error':
        st.error(warn['message'])
    elif warn['level'] == 'warning':
        st.warning(warn['message'])
    else:
        st.info(warn['message'])

# Forecast table
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

# Comparison with previous forecast
st.markdown("---")
st.subheader("Forecast Comparison")

prev_forecast = get_previous_forecast(site_key)
if prev_forecast and prev_forecast.get('total_tonnes'):
    prev_total = prev_forecast['total_tonnes']
    curr_total = df_forecast['Tonnes'].sum()
    diff = curr_total - prev_total
    diff_pct = safe_divide(diff, prev_total, 0) * 100
    prev_time = datetime.fromisoformat(prev_forecast['timestamp']) if prev_forecast.get('timestamp') else None

    col1, col2, col3 = st.columns(3)
    with col1:
        time_str = prev_time.strftime('%d/%m/%Y %H:%M') if prev_time else "Unknown"
        st.metric("Previous Forecast", f"{prev_total:,.0f} t", help=f"Generated {time_str}")
    with col2:
        st.metric("Current Forecast", f"{curr_total:,.0f} t")
    with col3:
        st.metric("Change", f"{diff:+,.0f} t", delta=f"{diff_pct:+.1f}%")
else:
    st.info("No previous forecast to compare. Generate another forecast to see comparison.")

st.markdown("---")

# Export
st.subheader("Export Report")

col1, col2 = st.columns(2)

with col1:
    html_report = create_html_report(
        df_forecast, df_historical, site_config['name'], contract_tonnes,
        lump_split, stats, required_tph, selected_commodity
    )

    base_name = f"CSI_TUM_Report_{site_config['name'].replace(' ', '_').replace('-', '')}"
    html_filename = generate_versioned_filename(base_name, "html")

    if st.download_button(
        "Download HTML Report",
        data=html_report,
        file_name=html_filename,
        mime="text/html"
    ):
        log_action("EXPORT_HTML", {"site": site_config['name'], "file": html_filename})

    st.caption("Open in browser -> Print -> Save as PDF")

with col2:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_forecast.to_excel(writer, sheet_name='FY27 Forecast', index=False)
        df_historical.to_excel(writer, sheet_name='Historical Data', index=False)
    output.seek(0)

    excel_base = f"TUM_Data_{site_config['name'].replace(' ', '_')}"
    excel_filename = generate_versioned_filename(excel_base, "xlsx")

    if st.download_button(
        "Download Excel Data",
        data=output,
        file_name=excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        log_action("EXPORT_EXCEL", {"site": site_config['name'], "file": excel_filename})

# Footer
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"TUM Forecasting Assistant v4.0 | {datetime.now().strftime('%d/%m/%Y %H:%M')}")
with col2:
    if st.button("View Audit Log"):
        log_file = LOG_DIR / "tum_audit.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()[-50:]  # Last 50 entries
            st.text_area("Recent Audit Log", "".join(lines), height=300)
        else:
            st.info("No audit log yet")
