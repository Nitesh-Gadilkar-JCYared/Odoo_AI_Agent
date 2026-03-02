"""
config.py
─────────
Loads credentials from .env file automatically.
Values from .env are used as defaults in the UI —
so you never have to type them again after restart.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this file
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


class Config:
    # ── Odoo ──
    ODOO_URL      = os.getenv("ODOO_URL", "")
    ODOO_DB       = os.getenv("ODOO_DB", "")
    ODOO_USER     = os.getenv("ODOO_USER", "")
    ODOO_PASSWORD = os.getenv("ODOO_PASSWORD", "")

    # ── LLM (free) ──
    GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    @classmethod
    def is_fully_configured(cls) -> bool:
        """Returns True if all required fields are present in .env"""
        return all([
            cls.ODOO_URL,
            cls.ODOO_DB,
            cls.ODOO_USER,
            cls.ODOO_PASSWORD,
            cls.GROQ_API_KEY,
        ])