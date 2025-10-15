#!/usr/bin/env python3
"""
Debug script để kiểm tra wizard data storage
"""

import streamlit as st
from src.data_manager import DataManager
from wizard_ui.session_manager import SessionManager

def debug_wizard_data():
    """Debug wizard data storage"""
    print("=== Debugging Wizard Data Storage ===")
    
    # Test 1: Create SessionManager
    try:
        print("1. Creating SessionManager...")
        session_manager = SessionManager()
        print("   SUCCESS: SessionManager created")
        
        # Test 2: Set wizard data
        try:
            print("\n2. Setting wizard data...")
            import pandas as pd
            test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
            
            session_manager.set_wizard_data('dataset', test_df)
            session_manager.set_wizard_data('dataset_name', 'test_dataset')
            print("   SUCCESS: Wizard data set")
            
            # Test 3: Get wizard data
            try:
                print("\n3. Getting wizard data...")
                dataset = session_manager.get_wizard_data('dataset')
                dataset_name = session_manager.get_wizard_data('dataset_name')
                
                if dataset is not None:
                    print(f"   SUCCESS: Dataset retrieved - Shape: {dataset.shape}")
                    print(f"   SUCCESS: Dataset name: {dataset_name}")
                else:
                    print("   ERROR: Dataset is None")
                    
            except Exception as e:
                print(f"   ERROR: Error getting wizard data: {e}")
                
        except Exception as e:
            print(f"   ERROR: Error setting wizard data: {e}")
            
    except Exception as e:
        print(f"   ERROR: Error creating SessionManager: {e}")
    
    # Test 4: Check session state directly
    try:
        print("\n4. Checking session state directly...")
        if hasattr(st, 'session_state'):
            if 'wizard_data' in st.session_state:
                wizard_data = st.session_state.wizard_data
                print(f"   SUCCESS: wizard_data exists with keys: {list(wizard_data.keys())}")
                
                if 'dataset' in wizard_data:
                    dataset = wizard_data['dataset']
                    if dataset is not None:
                        print(f"   SUCCESS: Dataset in session state - Shape: {dataset.shape}")
                    else:
                        print("   ERROR: Dataset in session state is None")
                else:
                    print("   ERROR: 'dataset' key not found in wizard_data")
            else:
                print("   ERROR: 'wizard_data' not found in session state")
        else:
            print("   ERROR: st.session_state does not exist")
            
    except Exception as e:
        print(f"   ERROR: Error checking session state: {e}")

if __name__ == "__main__":
    debug_wizard_data()
