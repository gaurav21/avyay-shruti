"""Tests for shruti.config module."""

import os
import pytest
from unittest.mock import patch


class TestConfig:
    """Test Config class behavior."""

    def test_default_values(self):
        """Config has sensible defaults."""
        from shruti.config import Config

        assert Config.CHUNK_SIZE == int(os.getenv("CHUNK_SIZE", "500"))
        assert Config.CHUNK_OVERLAP == int(os.getenv("CHUNK_OVERLAP", "75"))
        assert Config.WHISPER_MODEL == os.getenv("WHISPER_MODEL", "whisper-large-v3")
        assert "MiniLM" in Config.EMBEDDING_MODEL or Config.EMBEDDING_MODEL == os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    def test_validate_raises_without_api_key(self, monkeypatch):
        """validate() raises when GROQ_API_KEY is empty."""
        monkeypatch.setenv("GROQ_API_KEY", "")
        # Re-import to pick up new env
        from shruti.config import Config
        Config.GROQ_API_KEY = ""
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            Config.validate()

    def test_validate_passes_with_api_key(self, monkeypatch):
        """validate() succeeds when GROQ_API_KEY is set."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        from shruti.config import Config
        Config.GROQ_API_KEY = "test-key"
        Config.validate()  # Should not raise

    def test_get_chroma_persist_path_returns_path(self):
        """get_chroma_persist_path returns a resolved Path."""
        from shruti.config import Config
        path = Config.get_chroma_persist_path()
        assert path.is_absolute()

    def test_ensure_directories(self, tmp_path, monkeypatch):
        """ensure_directories creates required dirs."""
        from shruti.config import Config
        Config.CHROMA_PERSIST_DIR = str(tmp_path / "chroma_test")
        Config.AUDIO_OUTPUT_DIR = str(tmp_path / "audio_test")
        Config.ensure_directories()
        assert (tmp_path / "chroma_test").exists()
        assert (tmp_path / "audio_test").exists()

    def test_get_config_returns_config(self):
        """get_config() returns a Config instance."""
        from shruti.config import get_config
        c = get_config()
        assert hasattr(c, "GROQ_API_KEY")

    def test_setup_directories_callable(self, tmp_path):
        """setup_directories() is callable without error."""
        from shruti.config import Config, setup_directories
        Config.CHROMA_PERSIST_DIR = str(tmp_path / "sd_test")
        Config.AUDIO_OUTPUT_DIR = str(tmp_path / "sd_audio")
        setup_directories()
