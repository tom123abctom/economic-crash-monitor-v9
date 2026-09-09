import os

source_dir = r"C:\Users\tomba\.gemini\antigravity\scratch\economic-crash-monitor"

def read_clean(rel_path):
    p = os.path.join(source_dir, rel_path)
    with open(p, "r", encoding="utf-8") as f:
        lines = []
        for l in f.readlines():
            stripped = l.strip()
            if stripped.startswith("sys.path.append") or stripped.startswith("sys.path.insert"):
                continue
            if stripped.startswith("from app.") or stripped.startswith("from indicators.") or stripped.startswith("from data_sources.") or stripped.startswith("from models.") or stripped.startswith("from alerts."):
                continue
            if stripped in ["import sys", "import os", "import yaml", "import pandas as pd", "import numpy as np", "import requests"]:
                continue
            lines.append(l)
        return "".join(lines)

header = """import os
import sys
import sqlite3
import yaml
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import yfinance as yf

"""

full_code = header
full_code += "# ==================== DB MODULE ====================\n" + read_clean("app/indicators/db.py") + "\n\n"
full_code += "# ==================== SCORING MODULE ====================\n" + read_clean("app/models/scoring.py") + "\n\n"
full_code += "# ==================== FETCH MODULE ====================\n" + read_clean("app/data_sources/fetch_data.py") + "\n\n"
full_code += "# ==================== BACKTEST MODULE ====================\n" + read_clean("app/models/backtest.py") + "\n\n"
full_code += "# ==================== ALERTS MODULE ====================\n" + read_clean("app/alerts/email_alert.py") + "\n\n"
full_code += "# ==================== DASHBOARD MODULE ====================\n" + read_clean("app/dashboard/main.py") + "\n\n"

# Write standalone streamlit_app.py
with open(os.path.join(source_dir, "streamlit_app.py"), "w", encoding="utf-8") as f:
    f.write(full_code)

# Write main.py wrapper
with open(os.path.join(source_dir, "main.py"), "w", encoding="utf-8") as f:
    f.write('from streamlit_app import main\n\nif __name__ == "__main__":\n    main()\n')

print("Successfully generated self-contained standalone app!")
