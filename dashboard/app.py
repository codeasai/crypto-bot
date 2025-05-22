import streamlit as st
from pathlib import Path
import sys
import os

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ตั้งค่า environment variable
os.environ["PYTHONPATH"] = str(project_root)

# ตั้งค่า page config
st.set_page_config(
    page_title="Crypto Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css_file = project_root / "dashboard" / "assets" / "css" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    load_css()
    
    # Sidebar
    st.sidebar.title("Crypto Trading Bot")
    logo_path = project_root / "dashboard" / "assets" / "img" / "logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=100)
    
    # Main content
    st.title("Welcome to Crypto Trading Bot Dashboard")
    st.markdown("""
    ### Overview
    This dashboard provides tools for:
    - Data analysis and visualization
    - Model training and evaluation
    - Backtesting strategies
    - Live trading monitoring
    - System configuration
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Models", "3")
    with col2:
        st.metric("Total Trades", "1,234")
    with col3:
        st.metric("Success Rate", "65%")

if __name__ == "__main__":
    main() 