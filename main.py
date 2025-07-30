#!/usr/bin/env python3
"""
NewsBot Intelligence System - One-Click Launcher
Automatically sets up and launches the complete NewsBot system.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        if e.output:
            print(f"Output: {e.output}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"Python version: {sys.version.split()[0]} - Compatible")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\nInstalling Python dependencies...")
    
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies from requirements.txt")
    else:
        print("requirements.txt not found. Installing core packages...")
        packages = [
            "pandas>=2.0.3",
            "numpy>=1.26.4", 
            "matplotlib>=3.7.2",
            "seaborn>=0.13.1",
            "scikit-learn>=1.7.1",
            "nltk>=3.9.1",
            "spacy>=3.8.4",
            "textblob>=0.19.0",
            "vaderSentiment>=3.3.2",
            "plotly>=6.2.0",
            "streamlit>=1.42.2",
            "requests>=2.32.3"
        ]
        
        for package in packages:
            if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package.split('>=')[0]}"):
                print(f"Warning: Failed to install {package}")
        return True

def setup_nltk_data():
    """Download required NLTK data."""
    print("\nSetting up NLTK data...")
    
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    try:
        import nltk
        nltk_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'vader_lexicon', 'omw-1.4']
        for package in nltk_packages:
            try:
                # Check if package is already downloaded
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' 
                             else f'corpora/{package}' if package in ['stopwords', 'wordnet', 'omw-1.4'] 
                             else f'taggers/{package}' if package == 'averaged_perceptron_tagger'
                             else f'vader_lexicon' if package == 'vader_lexicon'
                             else package)
                print(f"{package} already available")
            except LookupError:
                # Package not found, download it
                try:
                    nltk.download(package, quiet=True)
                    print(f"Downloaded {package}")
                except Exception as e:
                    print(f"Warning: Could not download {package}: {e}")
        return True
    except ImportError:
        print("NLTK not available - will be installed with dependencies")
        return True

def setup_spacy_models():
    """Download spaCy models."""
    print("\nSetting up spaCy models...")
    
    try:
        import spacy
    except ImportError:
        print("spaCy not available - will be installed with dependencies")
        return True
    
    models = ["en_core_web_sm", "en_core_web_md"]
    for model in models:
        try:
            # Check if model is already installed
            spacy.load(model)
            print(f"{model} already available")
        except OSError:
            # Model not found, install it
            if not run_command(f"{sys.executable} -m spacy download {model}", f"Installing spaCy {model}"):
                print(f"Warning: Could not install {model}")
                # Try alternative for en_core_web_md if it fails
                if model == "en_core_web_md":
                    try:
                        spacy.load("en_core_web_sm")
                        print("Using en_core_web_sm as fallback")
                    except OSError:
                        run_command(f"{sys.executable} -m spacy download en_core_web_sm", "Installing fallback spaCy model")

def create_directories():
    """Create necessary project directories."""
    print("\nCreating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "outputs/visualizations",
        "outputs/reports",
        "outputs/analysis_results",
        "dashboard/assets"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def download_and_prepare_data():
    """Download and prepare the BBC News dataset."""
    print("Acquiring BBC News Dataset...")
    
    if Path("data_acquisition.py").exists():
        return run_command(f"{sys.executable} data_acquisition.py", "Downloading and preparing dataset")
    else:
        print("Error: data_acquisition.py not found!")
        print("The BBC News dataset is required for this analysis.")
        print("Please ensure data_acquisition.py is in the project directory.")
        return False

def check_analysis_results():
    """Check if analysis results exist for the dashboard."""
    required_files = [
        "outputs/analysis_results/sentiment_results.json",
        "outputs/analysis_results/classification_results.json",
        "data/processed/newsbot_dataset.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def run_notebook_analysis():
    """Run the Jupyter notebook to generate analysis results."""
    print("\nRunning analysis notebook to generate dashboard data...")
    print("This may take a few minutes...")
    
    notebook_path = Path("NewsBot_Intelligence_System.ipynb")
    if not notebook_path.exists():
        print("Error: Notebook file not found!")
        return False
    
    try:
        # Try using nbconvert to execute the notebook
        cmd = f"{sys.executable} -m jupyter nbconvert --to notebook --execute --inplace NewsBot_Intelligence_System.ipynb"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        print("Notebook analysis completed successfully!")
        return True
    except subprocess.TimeoutExpired:
        print("Warning: Notebook execution timed out. Try running it manually.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error running notebook: {e}")
        print("You can run the notebook manually using: jupyter notebook NewsBot_Intelligence_System.ipynb")
        return False
    except FileNotFoundError:
        print("Jupyter not found. Installing jupyter...")
        if run_command(f"{sys.executable} -m pip install jupyter nbconvert", "Installing Jupyter"):
            return run_notebook_analysis()  # Retry after installation
        return False

def verify_installation():
    """Verify that all components work."""
    print("\nVerifying installation...")
    
    try:
        # Test core libraries
        import pandas
        import numpy
        import matplotlib
        import streamlit
        print("Core libraries working")
        
        # Test NLP libraries
        try:
            import nltk
            import spacy
            print("NLP libraries working")
        except ImportError as e:
            print(f"Warning: Some NLP libraries not available: {e}")
        
        # Check if data exists
        if Path("data/processed/newsbot_dataset.csv").exists():
            print("Dataset ready")
        else:
            print("Warning: Dataset not found")
        
        return True
        
    except ImportError as e:
        print(f"Installation verification failed: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\nLaunching NewsBot Dashboard...")
    print("The dashboard will open in your default web browser")
    print("Use Ctrl+C to stop the dashboard")
    
    dashboard_path = Path("dashboard/newsbot_dashboard.py")
    if dashboard_path.exists():
        try:
            # Launch Streamlit
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)
        except KeyboardInterrupt:
            print("\nDashboard stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"Error launching dashboard: {e}")
            print("Try running manually: streamlit run dashboard/newsbot_dashboard.py")
    else:
        print("Dashboard file not found. Please check the installation.")

def main():
    """Main launcher function."""
    print("=" * 60)
    print("NewsBot Intelligence System - One-Click Launcher")
    print("Course: ITAI2373 - Natural Language Processing")
    print("Team: Martin Demel & Jiri Musil")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    if script_dir != Path("."):
        os.chdir(script_dir)
        print(f"Changed to directory: {script_dir.absolute()}")
    
    # Step 1: Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Step 2: Install dependencies
    print("\nStep 1/7: Installing dependencies...")
    if not install_dependencies():
        print("Warning: Some dependencies failed to install")
    
    # Step 3: Setup NLTK
    print("\nStep 2/7: Setting up NLTK...")
    setup_nltk_data()
    
    # Step 4: Setup spaCy
    print("\nStep 3/7: Setting up spaCy...")
    setup_spacy_models()
    
    # Step 5: Create directories
    print("\nStep 4/7: Creating directories...")
    create_directories()
    
    # Step 6: Download data
    print("\nStep 5/7: Preparing dataset...")
    download_and_prepare_data()
    
    # Step 7: Check if analysis results exist
    print("\nStep 6/7: Checking analysis results...")
    results_exist, missing_files = check_analysis_results()
    
    if not results_exist:
        print("Warning: Dashboard data not found. Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        
        print("\nAuto-generating analysis results by running notebook...")
        if run_notebook_analysis():
            print("Analysis complete! Dashboard data generated.")
        else:
            print("Could not generate analysis data automatically.")
            print("Please run the notebook manually and then restart this launcher.")
    else:
        print("Analysis results found - Dashboard ready!")
    
    # Step 8: Verify installation
    print("\nStep 7/7: Verifying installation...")
    if verify_installation():
        print("\nSetup completed successfully!")
        
        # Check one more time if we have the required data
        results_exist, _ = check_analysis_results()
        
        if results_exist:
            print("\nLaunching Interactive Dashboard...")
            launch_dashboard()
        else:
            print("\nDashboard data still missing. Choose an option:")
            print("1. Launch Dashboard anyway (limited functionality)")
            print("2. Open Jupyter Notebook to run analysis manually")
            print("3. Exit")
            
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    launch_dashboard()
                elif choice == "2":
                    if Path("NewsBot_Intelligence_System.ipynb").exists():
                        print("Opening Jupyter notebook...")
                        subprocess.run([sys.executable, "-m", "jupyter", "notebook", "NewsBot_Intelligence_System.ipynb"])
                    else:
                        print("Notebook file not found")
                elif choice == "3":
                    print("Exiting...")
                else:
                    print("Invalid choice. Launching dashboard...")
                    launch_dashboard()
                    
            except KeyboardInterrupt:
                print("\nExiting...")
        
    else:
        print("\nSetup completed with warnings. Please check the output above.")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 