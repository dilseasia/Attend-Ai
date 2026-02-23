#!/usr/bin/env python3
"""
Script to rename Abhishek_Vasudev_1940 to Abhishek_Vasudev_1941 
across all date folders in recognized_photos

INSTRUCTIONS:
1. Save this script in your project-backend folder (or anywhere)
2. Run it with: python rename_folders_v2.py
"""

import os
import sys

# Old and new folder names
OLD_NAME = "Harojdh Sandhu_2561"
NEW_NAME = "Harjodh Sandhu_2561"

def find_recognized_photos_folder():
    """Find the recognized_photos folder"""
    # Try current directory first
    if os.path.exists("recognized_photos"):
        print("✓ Found 'recognized_photos' in current directory")
        return "recognized_photos"
    
    # Try one level up
    if os.path.exists("../recognized_photos"):
        print("✓ Found 'recognized_photos' one level up")
        return "../recognized_photos"
    
    # Try common locations
    common_paths = [
        "./project-backend/recognized_photos",
        "../project-backend/recognized_photos",
        "../../project-backend/recognized_photos",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✓ Found 'recognized_photos' at: {path}")
            return path
    
    # Ask user to provide path
    print("\n⚠ Could not automatically find 'recognized_photos' folder.")
    print("\nPlease provide the path in one of these formats:")
    print("  - Relative: recognized_photos")
    print("  - Relative: ../project-backend/recognized_photos")
    print("  - Absolute: C:\\Users\\YourName\\project-backend\\recognized_photos")
    print("  - Absolute: /home/user/project-backend/recognized_photos")
    
    path = input("\nEnter path to 'recognized_photos' folder: ").strip()
    
    # Remove quotes if user copied path with quotes
    path = path.strip('"').strip("'")
    
    if os.path.exists(path):
        print(f"✓ Path verified: {path}")
        return path
    else:
        print(f"\n✗ Error: Path '{path}' does not exist!")
        print("\nTroubleshooting:")
        print("  1. Check if the path is correct")
        print("  2. Make sure you're using the right slashes (/ or \\)")
        print("  3. Verify the folder actually exists")
        return None

def show_folder_structure(base_path):
    """Show what's inside the recognized_photos folder"""
    print(f"\n📁 Contents of '{base_path}':")
    print("-" * 70)
    
    try:
        items = os.listdir(base_path)
        if not items:
            print("  (empty folder)")
            return
        
        for item in sorted(items)[:10]:  # Show first 10 items
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Check if OLD_NAME exists inside this date folder
                old_path = os.path.join(item_path, OLD_NAME)
                if os.path.exists(old_path):
                    print(f"  📂 {item}/ → Contains '{OLD_NAME}' ✓")
                else:
                    print(f"  📂 {item}/ → Does NOT contain '{OLD_NAME}'")
            else:
                print(f"  📄 {item}")
        
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more items")
    except Exception as e:
        print(f"  ✗ Error reading folder: {e}")
    
    print("-" * 70)

def rename_folders(base_path):
    """Rename all instances of the old folder name to the new name"""
    
    renamed_count = 0
    errors = []
    skipped = []
    not_found = []
    
    print(f"\n🔍 Scanning '{base_path}' for date folders...")
    
    # Iterate through all date folders
    try:
        all_items = os.listdir(base_path)
        date_folders = sorted([f for f in all_items 
                             if os.path.isdir(os.path.join(base_path, f))])
    except Exception as e:
        print(f"✗ Error reading base directory: {e}")
        return
    
    if not date_folders:
        print("✗ No date folders found!")
        return
    
    print(f"✓ Found {len(date_folders)} folders\n")
    print("Processing:")
    print("-" * 70)
    
    for date_folder in date_folders:
        date_folder_path = os.path.join(base_path, date_folder)
        old_folder_path = os.path.join(date_folder_path, OLD_NAME)
        new_folder_path = os.path.join(date_folder_path, NEW_NAME)
        
        # Check if the old folder exists
        if os.path.exists(old_folder_path):
            # Check if new name already exists
            if os.path.exists(new_folder_path):
                skip_msg = f"{date_folder}: Target '{NEW_NAME}' already exists"
                print(f"⚠ {skip_msg}")
                skipped.append(skip_msg)
                continue
            
            try:
                # Rename the folder
                os.rename(old_folder_path, new_folder_path)
                print(f"✓ {date_folder}: {OLD_NAME} → {NEW_NAME}")
                renamed_count += 1
            except Exception as e:
                error_msg = f"{date_folder}: {str(e)}"
                print(f"✗ {error_msg}")
                errors.append(error_msg)
        else:
            not_found.append(date_folder)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  ✓ Folders successfully renamed: {renamed_count}")
    
    if not_found:
        print(f"  ℹ Folders where '{OLD_NAME}' was not found: {len(not_found)}")
    
    if skipped:
        print(f"  ⚠ Folders skipped (target exists): {len(skipped)}")
    
    if errors:
        print(f"  ✗ Errors encountered: {len(errors)}")
        print("\nERROR DETAILS:")
        for error in errors:
            print(f"  • {error}")
    elif renamed_count > 0:
        print("\n  ✓✓✓ All operations completed successfully! ✓✓✓")
    elif not_found and renamed_count == 0:
        print(f"\n  ℹ No folders were renamed because '{OLD_NAME}' was not found")
        print(f"  ℹ Please check if the folder name is correct")
    
    print("="*70)

def main():
    print("="*70)
    print("FOLDER RENAME UTILITY - v2 (Diagnostic Version)")
    print("="*70)
    print(f"This script will rename:")
    print(f"  FROM: {OLD_NAME}")
    print(f"  TO:   {NEW_NAME}")
    print("="*70)
    
    # Find the recognized_photos folder
    base_path = find_recognized_photos_folder()
    
    if not base_path:
        print("\n✗ Cannot proceed without a valid path.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Show folder structure for verification
    show_folder_structure(base_path)
    
    # Confirm with user
    print(f"\n📍 Base path: {os.path.abspath(base_path)}")
    print("\n⚠ IMPORTANT: This will rename folders permanently!")
    confirm = input("\nProceed with renaming? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n✗ Operation cancelled by user.")
        input("\nPress Enter to exit...")
        sys.exit(0)
    
    # Perform the rename
    rename_folders(base_path)
    
    input("\n\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Operation cancelled by user (Ctrl+C)")
        input("\nPress Enter to exit...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

