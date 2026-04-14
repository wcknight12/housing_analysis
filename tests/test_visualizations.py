"""Tests for the visualizations module."""

import os
import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # non-interactive backend for tests

from housing_analysis.data_loader import load_data
from housing_analysis.analysis import run_full_analysis
from housing_analysis.visualizations import (
    plot_price_distribution,
    plot_price_vs_sqft,
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_affordability,
    plot_price_per_sqft_by_city,
    plot_school_vs_price,
    plot_age_vs_price,
    generate_all_charts,
)


@pytest.fixture(scope="module")
def df():
    return load_data(n_samples=200, random_seed=2)


@pytest.fixture(scope="module")
def results(df):
    return run_full_analysis(df)


def _is_figure(obj) -> bool:
    return isinstance(obj, plt.Figure)


class TestIndividualCharts:
    def test_price_distribution_returns_figure(self, df):
        fig = plot_price_distribution(df)
        assert _is_figure(fig)
        plt.close(fig)

    def test_price_vs_sqft_returns_figure(self, df):
        fig = plot_price_vs_sqft(df)
        assert _is_figure(fig)
        plt.close(fig)

    def test_feature_importance_returns_figure(self, results):
        fig = plot_feature_importance(results["combined_importance"])
        assert _is_figure(fig)
        plt.close(fig)

    def test_correlation_heatmap_returns_figure(self, df):
        fig = plot_correlation_heatmap(df)
        assert _is_figure(fig)
        plt.close(fig)

    def test_affordability_returns_figure(self, results):
        fig = plot_affordability(results["affordability"])
        assert _is_figure(fig)
        plt.close(fig)

    def test_price_per_sqft_returns_figure(self, df):
        fig = plot_price_per_sqft_by_city(df)
        assert _is_figure(fig)
        plt.close(fig)

    def test_school_vs_price_returns_figure(self, df):
        fig = plot_school_vs_price(df)
        assert _is_figure(fig)
        plt.close(fig)

    def test_age_vs_price_returns_figure(self, df):
        fig = plot_age_vs_price(df)
        assert _is_figure(fig)
        plt.close(fig)


class TestSaveFunctionality:
    def test_saves_file_when_path_provided(self, df, tmp_path):
        save_path = str(tmp_path / "test_chart.png")
        fig = plot_price_distribution(df, save_path=save_path)
        plt.close(fig)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 0


class TestGenerateAllCharts:
    def test_returns_dict(self, df, results, tmp_path):
        paths = generate_all_charts(df, results, output_dir=str(tmp_path))
        assert isinstance(paths, dict)

    def test_all_charts_created(self, df, results, tmp_path):
        paths = generate_all_charts(df, results, output_dir=str(tmp_path))
        expected_charts = {
            "price_distribution", "price_vs_sqft", "feature_importance",
            "correlation_heatmap", "affordability", "price_per_sqft",
            "school_vs_price", "age_vs_price",
        }
        assert expected_charts.issubset(set(paths.keys()))

    def test_chart_files_exist(self, df, results, tmp_path):
        paths = generate_all_charts(df, results, output_dir=str(tmp_path))
        for name, path in paths.items():
            assert os.path.isfile(path), f"Chart file missing: {name} -> {path}"

    def test_chart_files_non_empty(self, df, results, tmp_path):
        paths = generate_all_charts(df, results, output_dir=str(tmp_path))
        for name, path in paths.items():
            assert os.path.getsize(path) > 0, f"Chart file is empty: {name}"

    def test_creates_output_directory(self, df, results, tmp_path):
        out_dir = str(tmp_path / "new_subdir")
        generate_all_charts(df, results, output_dir=out_dir)
        assert os.path.isdir(out_dir)
