import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Fabric Dashboard", layout="wide")

# Load data
with open("dashboard_input.json", "r") as f:
    data = json.load(f)

image_path = data["image_path"]
fabric = data["fabric"]
score = data["score"]

# Title
st.markdown("## 🌿 Fabric Sustainability Overview")

# Layout
col1, col2 = st.columns([1, 1])

# Left: Image
with col1:
    st.image(image_path, use_container_width=True)

# Right: Fancy HTML using components.html()
html_content = f"""
<div style="
    background-color: #f0f4f8;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-family: 'Segoe UI', sans-serif;
">
    <h2 style="margin-top: 0; font-size: 32px; color: #1f4e79;">🧵 Fabric Type</h2>
    <p style="font-size: 28px; margin-bottom: 2rem; color: #1f4e79; text-transform: uppercase;">
    {fabric}</p>

    <h2 style="font-size: 32px; color: #2e7d32;">🌱 Sustainability Score</h2>
    <p style="font-size: 36px; font-weight: bold; color: #2e7d32;">{score} / 10</p>
</div>
"""

with col2:
    components.html(html_content, height=300)