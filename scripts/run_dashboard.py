import os
import subprocess
from pathlib import Path

def run_dashboard():
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "dashboard" / "app.py"
    
    # ตรวจสอบว่าไฟล์ app.py มีอยู่จริง
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return
    
    # ตั้งค่า environment variables
    os.environ["PYTHONPATH"] = str(project_root)
    
    # รัน Streamlit app
    os.chdir(str(project_root))
    try:
        subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
    except FileNotFoundError:
        print("Error: streamlit command not found. Please install streamlit using: pip install streamlit")

if __name__ == "__main__":
    run_dashboard() 