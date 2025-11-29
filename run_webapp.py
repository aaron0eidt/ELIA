# A simple launcher script for the web app.

import subprocess
import sys
import os

def main():
    print("LLM Attribution Analysis Web App")
    print("=" * 50)
    
    # Check if the script is being run from the correct directory.
    if not os.path.exists("web_app.py"):
        print("Error: web_app.py not found!")
        print("Please run this script from the Bachelor Arbeit directory.")
        return
    
    # Check if streamlit is installed.
    try:
        import streamlit
        print("Streamlit found")
    except ImportError:
        print("Streamlit not found. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("Starting the web application...")
    print("The app will open in your browser at http://localhost:8501")
    print("To stop the app, press Ctrl+C isn this terminal")
    print("=" * 50)
    
    # Run the streamlit app.
    try:
        subprocess.run(["streamlit", "run", "web_app.py"])
    except KeyboardInterrupt:
        print("\nWeb app stopped. Goodbye!")
    except FileNotFoundError:
        print("Error: streamlit command not found.")
        print("Please install streamlit: pip install streamlit")

if __name__ == "__main__":
    main() 