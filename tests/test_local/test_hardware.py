"""Tests for hardware detection and model recommendations."""

from __future__ import annotations

from stateloom.local.hardware import HardwareInfo, detect_hardware, recommend_models


class TestHardwareInfo:
    def test_defaults(self):
        info = HardwareInfo()
        assert info.ram_gb == 0.0
        assert info.gpu_name == ""
        assert info.os_name == ""

    def test_with_values(self):
        info = HardwareInfo(ram_gb=16.0, gpu_name="RTX 4090", gpu_vram_gb=24.0)
        assert info.ram_gb == 16.0
        assert info.gpu_vram_gb == 24.0


class TestDetectHardware:
    def test_returns_hardware_info(self):
        info = detect_hardware()
        assert isinstance(info, HardwareInfo)
        assert info.os_name != ""
        assert info.arch != ""
        assert info.cpu_count > 0

    def test_disk_free_positive(self):
        info = detect_hardware()
        assert info.disk_free_gb > 0


class TestRecommendModels:
    def test_no_memory_returns_empty(self):
        hw = HardwareInfo(ram_gb=1.0, gpu_vram_gb=0.0)
        # With 1GB RAM and 2GB headroom, nothing fits
        result = recommend_models(hw)
        assert result == []

    def test_small_memory_returns_ultra_light(self):
        hw = HardwareInfo(ram_gb=4.0, gpu_vram_gb=0.0)
        result = recommend_models(hw)
        assert len(result) > 0
        # All returned models should fit
        for m in result:
            assert m["size_gb"] <= 2.0  # 4GB - 2GB headroom

    def test_medium_memory_includes_more(self):
        hw = HardwareInfo(ram_gb=10.0, gpu_vram_gb=0.0)
        result = recommend_models(hw)
        tiers = {m["tier"] for m in result}
        assert "ultra-light" in tiers
        assert "light" in tiers

    def test_gpu_vram_used_when_available(self):
        hw = HardwareInfo(ram_gb=8.0, gpu_vram_gb=24.0)
        result = recommend_models(hw)
        # With 24GB VRAM, should include medium models
        sizes = [m["size_gb"] for m in result]
        assert max(sizes) > 4.0

    def test_sorted_by_tier_and_size(self):
        hw = HardwareInfo(ram_gb=32.0, gpu_vram_gb=0.0)
        result = recommend_models(hw)
        tier_order = {"ultra-light": 0, "light": 1, "medium": 2, "heavy": 3}
        for i in range(len(result) - 1):
            t1 = tier_order[result[i]["tier"]]
            t2 = tier_order[result[i + 1]["tier"]]
            if t1 == t2:
                assert result[i]["size_gb"] <= result[i + 1]["size_gb"]
            else:
                assert t1 <= t2

    def test_auto_detect_hardware(self):
        """recommend_models with no argument auto-detects hardware."""
        result = recommend_models()
        assert isinstance(result, list)
