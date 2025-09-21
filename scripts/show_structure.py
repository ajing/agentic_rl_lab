#!/usr/bin/env python3
"""
Simple script to show the organized scripts directory structure.
"""

import os
from pathlib import Path

def show_directory_structure():
    """Display the organized scripts directory structure."""
    base_dir = Path(__file__).parent
    
    print("📁 Scripts Directory Organization")
    print("=" * 50)
    
    directories = {
        "data_preparation": "📁 Data Preparation & Indexing",
        "week1": "📁 Week 1: Basic RAG Pipeline", 
        "week2": "📁 Week 2: RL Environment",
        "week3_4": "📁 Week 3-4: Reward Modeling",
        "week5_6": "📁 Week 5-6: Real Data Training",
        "training": "📁 General Training Scripts",
        "evaluation": "📁 Evaluation & Benchmarking",
        "testing": "📁 Testing & Validation"
    }
    
    for dir_name, description in directories.items():
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"\n{description}")
            print("-" * len(description))
            
            # List Python files in the directory
            py_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.py')])
            for py_file in py_files:
                print(f"  📄 {py_file}")
    
    print(f"\n📄 README.md - Documentation for all scripts")
    print(f"📄 show_structure.py - This structure display script")

if __name__ == "__main__":
    show_directory_structure()
