"""
Unit tests for TUM Forecasting Assistant calculations.
Run with: python -m pytest test_calculations.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app_v4 import (
    parse_percentage,
    calculate_historical_stats,
    generate_fy_forecast,
    safe_divide,
    validate_data,
    validate_inputs,
    generate_sanity_warnings,
    CALENDAR_HOURS,
    FY_MONTHS
)


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_divide_by_zero_returns_default(self):
        assert safe_divide(10, 0) == 0.0

    def test_divide_by_zero_custom_default(self):
        assert safe_divide(10, 0, -1) == -1

    def test_nan_numerator_returns_default(self):
        assert safe_divide(np.nan, 2) == 0.0

    def test_nan_denominator_returns_default(self):
        assert safe_divide(10, np.nan) == 0.0

    def test_both_nan_returns_default(self):
        assert safe_divide(np.nan, np.nan) == 0.0

    def test_negative_numbers(self):
        assert safe_divide(-10, 2) == -5.0

    def test_float_precision(self):
        result = safe_divide(1, 3)
        assert abs(result - 0.333333) < 0.001


class TestParsePercentage:
    """Tests for parse_percentage function."""

    def test_string_with_percent_sign(self):
        assert parse_percentage("95.5%") == 0.955

    def test_string_without_percent_sign(self):
        assert parse_percentage("95.5") == 0.955

    def test_float_as_decimal(self):
        assert parse_percentage(0.955) == 0.955

    def test_float_as_percentage(self):
        assert parse_percentage(95.5) == 0.955

    def test_integer_percentage(self):
        assert parse_percentage(95) == 0.95

    def test_zero(self):
        assert parse_percentage(0) == 0.0

    def test_hundred_percent(self):
        assert parse_percentage("100%") == 1.0

    def test_nan_returns_nan(self):
        assert pd.isna(parse_percentage(np.nan))

    def test_invalid_string_returns_nan(self):
        assert pd.isna(parse_percentage("invalid"))

    def test_empty_string_returns_nan(self):
        assert pd.isna(parse_percentage(""))

    def test_whitespace_handling(self):
        assert parse_percentage("  95.5%  ") == 0.955


class TestValidateData:
    """Tests for validate_data function."""

    def test_detects_high_tph(self):
        df = pd.DataFrame({'TPH': [2000, 6000, 2100]})
        warnings = validate_data(df, 'test_site')
        assert any('TPH > 5000' in w for w in warnings)

    def test_detects_negative_tph(self):
        df = pd.DataFrame({'TPH': [2000, -500, 2100]})
        warnings = validate_data(df, 'test_site')
        assert any('negative TPH' in w for w in warnings)

    def test_detects_negative_tonnes(self):
        df = pd.DataFrame({'Tonnes': [1000000, -500000, 1200000]})
        warnings = validate_data(df, 'test_site')
        assert any('negative Tonnes' in w for w in warnings)

    def test_detects_invalid_availability_over_100(self):
        df = pd.DataFrame({'Availability': [0.95, 1.5, 0.92]})
        warnings = validate_data(df, 'test_site')
        assert any('Availability' in w for w in warnings)

    def test_detects_negative_availability(self):
        df = pd.DataFrame({'Availability': [0.95, -0.1, 0.92]})
        warnings = validate_data(df, 'test_site')
        assert any('Availability' in w for w in warnings)

    def test_detects_excessive_run_hours(self):
        df = pd.DataFrame({'Run Hours': [500, 800, 520]})
        warnings = validate_data(df, 'test_site')
        assert any('Run Hours > 744' in w for w in warnings)

    def test_valid_data_no_warnings(self):
        df = pd.DataFrame({
            'TPH': [2000, 2100, 2050],
            'Availability': [0.95, 0.92, 0.94],
            'Utilisation': [0.80, 0.78, 0.79],
            'Tonnes': [1000000, 1100000, 1050000],
            'Run Hours': [500, 520, 510]
        })
        warnings = validate_data(df, 'test_site')
        assert len(warnings) == 0


class TestValidateInputs:
    """Tests for validate_inputs function."""

    @pytest.fixture
    def sample_stats(self):
        return {
            'total_tonnes': 12000000,
            'months_of_data': 12,
            'avg_tph': 2000
        }

    def test_warns_on_zero_contract(self, sample_stats):
        warnings = validate_inputs(0, 0.4, 'Iron Ore', sample_stats)
        assert any('positive' in w for w in warnings)

    def test_warns_on_negative_contract(self, sample_stats):
        warnings = validate_inputs(-1000000, 0.4, 'Iron Ore', sample_stats)
        assert any('positive' in w for w in warnings)

    def test_warns_on_iron_ore_no_lump_split(self, sample_stats):
        warnings = validate_inputs(12000000, 0, 'Iron Ore', sample_stats)
        assert any('lump split > 0' in w for w in warnings)

    def test_warns_on_non_iron_ore_with_lump_split(self, sample_stats):
        warnings = validate_inputs(5000000, 0.4, 'Lithium', sample_stats)
        assert any('no lump/fines split' in w for w in warnings)

    def test_warns_on_very_low_contract(self, sample_stats):
        warnings = validate_inputs(100000, 0.4, 'Iron Ore', sample_stats)
        assert any('unusually low' in w for w in warnings)

    def test_warns_on_very_high_contract(self, sample_stats):
        warnings = validate_inputs(100000000, 0.4, 'Iron Ore', sample_stats)
        assert any('unusually high' in w for w in warnings)

    def test_valid_inputs_no_warnings(self, sample_stats):
        warnings = validate_inputs(12000000, 0.4, 'Iron Ore', sample_stats)
        # May have some warnings but shouldn't have critical ones
        assert not any('positive' in w for w in warnings)


class TestGenerateSanityWarnings:
    """Tests for generate_sanity_warnings function."""

    @pytest.fixture
    def sample_stats(self):
        return {
            'avg_tph': 2000,
            'total_run_hours': 6000,
            'months_of_data': 12
        }

    def test_warns_on_tph_30_percent_higher(self, sample_stats):
        required_tph = 2800  # 40% higher
        warnings = generate_sanity_warnings(required_tph, sample_stats, 12000000)
        assert any(w['level'] == 'error' for w in warnings)

    def test_warns_on_tph_20_percent_higher(self, sample_stats):
        required_tph = 2400  # 20% higher
        warnings = generate_sanity_warnings(required_tph, sample_stats, 12000000)
        assert any(w['level'] == 'warning' for w in warnings)

    def test_info_on_tph_much_lower(self, sample_stats):
        required_tph = 1200  # 40% lower
        warnings = generate_sanity_warnings(required_tph, sample_stats, 6000000)
        assert any(w['level'] == 'info' for w in warnings)

    def test_no_warnings_on_similar_tph(self, sample_stats):
        required_tph = 2100  # 5% higher
        warnings = generate_sanity_warnings(required_tph, sample_stats, 10500000)
        high_priority_warnings = [w for w in warnings if w['level'] in ['error', 'warning']]
        assert len(high_priority_warnings) == 0


class TestCalculateHistoricalStats:
    """Tests for calculate_historical_stats function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'Month': ['Jul-24', 'Aug-24', 'Sep-24'],
            'Tonnes': [1000000, 1100000, 1050000],
            'TPH': [2000, 2100, 2050],
            'Run Hours': [500, 520, 510],
            'Availability': [0.95, 0.92, 0.94],
            'Utilisation': [0.80, 0.78, 0.79],
            'Planned Maint': [20, 30, 25],
            'Unplanned': [10, 15, 12],
            'Internal Delays': [5, 8, 6],
            'External Delays': [150, 160, 155]
        })

    def test_avg_tph_calculation(self, sample_df):
        stats = calculate_historical_stats(sample_df)
        expected_avg_tph = (2000 + 2100 + 2050) / 3
        assert abs(stats['avg_tph'] - expected_avg_tph) < 1

    def test_months_of_data(self, sample_df):
        stats = calculate_historical_stats(sample_df)
        assert stats['months_of_data'] == 3

    def test_total_tonnes(self, sample_df):
        stats = calculate_historical_stats(sample_df)
        assert stats['total_tonnes'] == 3150000

    def test_total_run_hours(self, sample_df):
        stats = calculate_historical_stats(sample_df)
        assert stats['total_run_hours'] == 1530

    def test_avg_availability(self, sample_df):
        stats = calculate_historical_stats(sample_df)
        expected = (0.95 + 0.92 + 0.94) / 3
        assert abs(stats['avg_availability'] - expected) < 0.01

    def test_avg_utilisation(self, sample_df):
        stats = calculate_historical_stats(sample_df)
        expected = (0.80 + 0.78 + 0.79) / 3
        assert abs(stats['avg_utilisation'] - expected) < 0.01

    def test_avg_planned_maint(self, sample_df):
        stats = calculate_historical_stats(sample_df)
        expected = (20 + 30 + 25) / 3
        assert abs(stats['avg_planned_maint'] - expected) < 1

    def test_handles_empty_dataframe(self):
        empty_df = pd.DataFrame()
        stats = calculate_historical_stats(empty_df)
        assert stats['months_of_data'] == 0
        assert stats['avg_tph'] == 0


class TestGenerateFyForecast:
    """Tests for generate_fy_forecast function."""

    @pytest.fixture
    def sample_stats(self):
        return {
            'avg_planned_maint': 25,
            'avg_unplanned': 12,
            'avg_internal_delays': 6,
            'avg_external_delays': 155,
            'avg_tph': 2050,
            'avg_availability': 0.94,
            'avg_utilisation': 0.79,
            'avg_run_hours': 510,
            'avg_tonnes_per_month': 1050000,
            'total_tonnes': 3150000,
            'total_run_hours': 1530,
            'months_of_data': 3,
            'total_unavailable_per_month': 37,
            'total_unproductive_per_month': 161
        }

    def test_generates_12_months(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.4, sample_stats)
        assert len(df) == 12

    def test_all_months_present(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.4, sample_stats)
        months = [m.split('-')[0] for m in df['Month']]
        for expected_month in FY_MONTHS:
            assert expected_month in months

    def test_total_matches_contract_within_tolerance(self, sample_stats):
        contract = 12000000
        df, tph, hours = generate_fy_forecast(contract, 0.4, sample_stats)
        # Should be within 0.1% of contract (rounding differences)
        assert abs(df['Tonnes'].sum() - contract) < contract * 0.001

    def test_lump_fines_split_40_60(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.4, sample_stats)
        total_tonnes = df['Tonnes'].sum()
        total_lump = df['Lump Tonnes'].sum()
        total_fines = df['Fines Tonnes'].sum()
        # Lump should be ~40% of total
        assert abs(total_lump / total_tonnes - 0.4) < 0.01
        # Fines should be ~60% of total
        assert abs(total_fines / total_tonnes - 0.6) < 0.01

    def test_no_lump_fines_when_split_is_zero(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.0, sample_stats)
        assert df['Lump Tonnes'].sum() == 0
        assert df['Fines Tonnes'].sum() == df['Tonnes'].sum()

    def test_tph_is_consistent_across_months(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.4, sample_stats)
        unique_tph = df['TPH'].unique()
        assert len(unique_tph) == 1  # All months should have same TPH

    def test_availability_in_valid_range(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.4, sample_stats)
        assert all(df['Availability'] >= 0)
        assert all(df['Availability'] <= 1)

    def test_utilisation_in_valid_range(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.4, sample_stats)
        assert all(df['Utilisation'] >= 0)
        assert all(df['Utilisation'] <= 1)

    def test_run_hours_positive(self, sample_stats):
        df, tph, hours = generate_fy_forecast(12000000, 0.4, sample_stats)
        assert all(df['Run Hours'] >= 0)

    def test_handles_zero_contract(self, sample_stats):
        df, tph, hours = generate_fy_forecast(0, 0.4, sample_stats)
        assert tph == 0
        assert df['Tonnes'].sum() == 0


class TestCalendarHours:
    """Tests for CALENDAR_HOURS constant."""

    def test_all_months_defined(self):
        for month in FY_MONTHS:
            assert month in CALENDAR_HOURS

    def test_february_has_672_hours(self):
        # 28 days * 24 hours = 672 (non-leap year)
        assert CALENDAR_HOURS['Feb'] == 672

    def test_31_day_months_have_744_hours(self):
        months_31_days = ['Jul', 'Aug', 'Oct', 'Dec', 'Jan', 'Mar', 'May']
        for month in months_31_days:
            assert CALENDAR_HOURS[month] == 744

    def test_30_day_months_have_720_hours(self):
        months_30_days = ['Sep', 'Nov', 'Apr', 'Jun']
        for month in months_30_days:
            assert CALENDAR_HOURS[month] == 720

    def test_total_calendar_hours_in_year(self):
        total = sum(CALENDAR_HOURS.values())
        # 365 days * 24 hours = 8760
        assert total == 8760


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
