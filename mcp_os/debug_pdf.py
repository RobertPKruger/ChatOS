# debug_pdf.py - Test the PDF functionality directly
"""
Debug script to test PDF functionality and see what's happening
"""

import os
import sys
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_tools import AppToolsManager

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_pdf_functionality():
    """Test PDF opening functionality step by step"""
    print("=== TESTING PDF FUNCTIONALITY ===\n")
    
    # Initialize app manager
    app_manager = AppToolsManager("apps_config.json")
    
    print("1. Testing PDF file detection:")
    desktop_path = os.path.expanduser("~/OneDrive/Desktop")
    print(f"   Looking in: {desktop_path}")
    print(f"   Directory exists: {os.path.exists(desktop_path)}")
    
    if os.path.exists(desktop_path):
        all_files = os.listdir(desktop_path)
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        print(f"   All files: {len(all_files)}")
        print(f"   PDF files found: {len(pdf_files)}")
        for pdf in pdf_files:
            print(f"     - {pdf}")
    
    print("\n2. Testing app configuration:")
    acrobat_config = app_manager.get_app_config("acrobat")
    if acrobat_config:
        print(f"   Acrobat configured: ✅")
        print(f"   Paths: {acrobat_config.get('paths', [])}")
        
        # Test if any path exists
        for path in acrobat_config.get('paths', []):
            expanded_path = path.replace('{username}', os.getenv('USERNAME', ''))
            if path.startswith('C:\\') and os.path.exists(expanded_path):
                print(f"   Found executable: {expanded_path} ✅")
                break
            elif not path.startswith('C:\\'):
                print(f"   Non-path entry: {path}")
        else:
            print("   No executable paths found ❌")
    else:
        print(f"   Acrobat configured: ❌")
    
    print("\n3. Testing find_pdf_files method:")
    pdf_files = app_manager.find_pdf_files("~/Desktop")
    print(f"   PDFs found by method: {len(pdf_files)}")
    for pdf in pdf_files:
        print(f"     - {pdf}")
        print(f"       Exists: {os.path.exists(pdf)}")
    
    print("\n4. Testing app launching (dry run):")
    if pdf_files:
        first_pdf = pdf_files[0]
        print(f"   Would open: {first_pdf}")
        print(f"   With app: acrobat")
        
        # Test the actual method
        result = app_manager.open_file_with_app(first_pdf, "acrobat")
        print(f"   Result: {result}")
    else:
        print("   No PDFs to test with")
    
    print("\n5. Testing direct launch_app method:")
    result = app_manager.launch_app("acrobat")
    print(f"   Launch acrobat result: {result}")
    
    print("\n6. Manual test - checking if acrobat is in PATH:")
    import shutil
    acrobat_in_path = shutil.which("AcroRd32.exe") or shutil.which("Acrobat.exe")
    print(f"   Acrobat in PATH: {acrobat_in_path}")

if __name__ == "__main__":
    test_pdf_functionality()