#!/usr/bin/env python3

import subprocess
import time
import re
import os
import sys
import json
from pathlib import Path

# Configuration
CLOUDFLARE_DIR = "./nohupAttendanceSystem"  # Directory where cloudflare runs
BASE_URL_FILE = "./baseurl.json"  # Your baseurl.json file
BACKEND_PORT = 8000
NOHUP_FILE = "nohup.out"
MAX_WAIT_TIME = 30  # seconds

# Remote Server Configuration
REMOTE_HOST = "10.8.14.74"
REMOTE_USER = "attendai"
REMOTE_PASSWORD = "Tg5^vP1!zR8cHn3Xy0&"
REMOTE_PROJECT_DIR = "/var/www/html/attendai"
REMOTE_BASE_URL_FILE = f"{REMOTE_PROJECT_DIR}/baseurl.json"

def start_cloudflare_tunnel():
    """Start cloudflared tunnel in background from nohupAttendanceSystem directory"""
    print("🚀 Starting Cloudflare tunnel...")
    
    # Create directory if it doesn't exist
    os.makedirs(CLOUDFLARE_DIR, exist_ok=True)
    
    # Change to cloudflare directory
    nohup_path = os.path.join(CLOUDFLARE_DIR, NOHUP_FILE)
    
    # Remove old nohup.out if exists
    if os.path.exists(nohup_path):
        os.remove(nohup_path)
        print(f"🗑️  Removed old {nohup_path}")
    
    # Start tunnel from the nohupAttendanceSystem directory
    cmd = f"cd {CLOUDFLARE_DIR} && nohup cloudflared tunnel --url http://localhost:{BACKEND_PORT} > {NOHUP_FILE} 2>&1 &"
    subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    
    print(f"⏳ Waiting for tunnel to establish...")

def extract_tunnel_url():
    """Extract tunnel URL from nohup.out in nohupAttendanceSystem directory"""
    nohup_path = os.path.join(CLOUDFLARE_DIR, NOHUP_FILE)
    start_time = time.time()
    
    while time.time() - start_time < MAX_WAIT_TIME:
        if not os.path.exists(nohup_path):
            time.sleep(1)
            continue
            
        try:
            with open(nohup_path, 'r') as f:
                content = f.read()
                # Look for cloudflare URL
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', content)
                if match:
                    return match.group(0)
        except Exception as e:
            print(f"⚠️  Error reading file: {e}")
        
        print(".", end="", flush=True)
        time.sleep(2)
    
    print()  # New line after dots
    return None

def update_baseurl_json(tunnel_url):
    """Update the baseurl.json file with new tunnel URL"""
    
    if not os.path.exists(BASE_URL_FILE):
        print(f"❌ Error: {BASE_URL_FILE} not found!")
        return False
    
    try:
        # Read the JSON file
        with open(BASE_URL_FILE, 'r') as f:
            data = json.load(f)
        
        # Backup original file
        backup_file = f"{BASE_URL_FILE}.backup"
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update the base_url field
        old_url = data.get('base_url', 'N/A')
        data['base_url'] = tunnel_url
        
        # Write updated content with proper formatting
        with open(BASE_URL_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Updated LOCAL {BASE_URL_FILE}")
        print(f"   Old URL: {old_url}")
        print(f"   New URL: {tunnel_url}")
        print(f"📝 Backup saved as {backup_file}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {BASE_URL_FILE}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        return False

def check_sshpass_installed():
    """Check if sshpass is installed"""
    try:
        subprocess.run(['which', 'sshpass'], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_sshpass():
    """Attempt to install sshpass"""
    print("\n⚠️  sshpass is not installed. Attempting to install...")
    try:
        # Try apt-get (Debian/Ubuntu)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'sshpass'], check=True)
        print("✅ sshpass installed successfully")
        return True
    except:
        try:
            # Try yum (CentOS/RHEL)
            subprocess.run(['sudo', 'yum', 'install', '-y', 'sshpass'], check=True)
            print("✅ sshpass installed successfully")
            return True
        except:
            print("❌ Could not install sshpass automatically")
            print("   Please install manually: sudo apt-get install sshpass")
            return False

def update_remote_baseurl(tunnel_url):
    """Update baseurl.json on remote server via SSH"""
    print("\n" + "=" * 60)
    print("🌐 Updating REMOTE server...")
    print("=" * 60)
    
    # Check if sshpass is installed
    if not check_sshpass_installed():
        if not install_sshpass():
            print("❌ Cannot update remote server without sshpass")
            return False
    
    try:
        # Create temporary file with updated JSON
        temp_file = "/tmp/baseurl_temp.json"
        
        # First, fetch the current remote file
        print(f"📥 Fetching current remote file...")
        fetch_cmd = [
            'sshpass', '-p', REMOTE_PASSWORD,
            'scp',
            '-o', 'StrictHostKeyChecking=no',
            f'{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE_URL_FILE}',
            temp_file
        ]
        
        result = subprocess.run(fetch_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to fetch remote file: {result.stderr}")
            return False
        
        # Read and update the JSON
        with open(temp_file, 'r') as f:
            data = json.load(f)
        
        # Store old URLs for logging
        old_baseurl = data.get('baseurl', 'N/A')
        old_base_url = data.get('base_url', 'N/A')
        
        # Remove old keys if they exist
        if 'baseurl' in data:
            del data['baseurl']
        if 'base_url' in data:
            del data['base_url']
        
        # Set only the correct field
        data['baseurl'] = tunnel_url
        
        # Write updated JSON (only baseurl field, properly formatted)
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Create backup on remote server
        print(f"💾 Creating backup on remote server...")
        backup_cmd = [
            'sshpass', '-p', REMOTE_PASSWORD,
            'ssh',
            '-o', 'StrictHostKeyChecking=no',
            f'{REMOTE_USER}@{REMOTE_HOST}',
            f'cp {REMOTE_BASE_URL_FILE} {REMOTE_BASE_URL_FILE}.backup'
        ]
        subprocess.run(backup_cmd, capture_output=True)
        
        # Upload the updated file
        print(f"📤 Uploading updated file to remote server...")
        upload_cmd = [
            'sshpass', '-p', REMOTE_PASSWORD,
            'scp',
            '-o', 'StrictHostKeyChecking=no',
            temp_file,
            f'{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE_URL_FILE}'
        ]
        
        result = subprocess.run(upload_cmd, capture_output=True, text=True)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if result.returncode == 0:
            print(f"✅ Updated REMOTE {REMOTE_BASE_URL_FILE}")
            if old_baseurl != 'N/A':
                print(f"   Old baseurl: {old_baseurl}")
            if old_base_url != 'N/A':
                print(f"   Old base_url: {old_base_url}")
            print(f"   Removed old keys and set new baseurl: {tunnel_url}")
            print(f"   Host: {REMOTE_HOST}")
            print(f"📝 Remote backup saved as {REMOTE_BASE_URL_FILE}.backup")
            return True
        else:
            print(f"❌ Failed to upload file: {result.stderr}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in remote file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error updating remote server: {e}")
        return False

def show_current_url():
    """Show current URL from nohup.out if it exists"""
    nohup_path = os.path.join(CLOUDFLARE_DIR, NOHUP_FILE)
    if os.path.exists(nohup_path):
        try:
            with open(nohup_path, 'r') as f:
                content = f.read()
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', content)
                if match:
                    print(f"📡 Existing tunnel URL: {match.group(0)}")
        except:
            pass

def show_current_baseurl():
    """Show current base_url from baseurl.json"""
    if os.path.exists(BASE_URL_FILE):
        try:
            with open(BASE_URL_FILE, 'r') as f:
                data = json.load(f)
                current_url = data.get('base_url', 'N/A')
                print(f"📄 Current LOCAL base_url: {current_url}")
        except:
            pass

def show_remote_baseurl():
    """Show current base_url from remote server"""
    print(f"📡 Fetching REMOTE base_url...")
    
    if not check_sshpass_installed():
        print("⚠️  sshpass not installed, skipping remote check")
        return
    
    try:
        cmd = [
            'sshpass', '-p', REMOTE_PASSWORD,
            'ssh',
            '-o', 'StrictHostKeyChecking=no',
            f'{REMOTE_USER}@{REMOTE_HOST}',
            f'cat {REMOTE_BASE_URL_FILE}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Check both possible field names
            baseurl = data.get('baseurl', None)
            base_url = data.get('base_url', None)
            
            if baseurl:
                print(f"📄 Current REMOTE baseurl: {baseurl}")
            if base_url:
                print(f"📄 Current REMOTE base_url: {base_url}")
            if not baseurl and not base_url:
                print(f"⚠️  No baseurl or base_url found in remote file")
        else:
            print(f"⚠️  Could not fetch remote base_url")
    except Exception as e:
        print(f"⚠️  Error fetching remote base_url: {e}")

def main():
    print("=" * 60)
    print("Cloudflare Tunnel Auto-Updater (Local + Remote)")
    print("=" * 60)
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"📂 Cloudflare directory: {CLOUDFLARE_DIR}")
    print(f"📄 Local Base URL file: {BASE_URL_FILE}")
    print(f"🌐 Remote Host: {REMOTE_HOST}")
    print(f"📄 Remote Base URL file: {REMOTE_BASE_URL_FILE}")
    print("=" * 60)
    
    # Show current baseurl.json content (local and remote)
    show_current_baseurl()
    show_remote_baseurl()
    
    # Check if cloudflared is already running
    try:
        result = subprocess.run(['pgrep', '-f', 'cloudflared'], capture_output=True)
        if result.returncode == 0:
            print("\n⚠️  Cloudflared is already running!")
            print("🔍 Checking for existing URL...")
            show_current_url()
            response = input("\nDo you want to restart it? (y/n): ").lower()
            if response == 'y':
                print("🛑 Stopping existing cloudflared process...")
                subprocess.run(['pkill', 'cloudflared'])
                time.sleep(2)
            else:
                print("✋ Keeping existing tunnel")
                sys.exit(0)
    except:
        pass
    
    # Start tunnel
    start_cloudflare_tunnel()
    
    # Extract URL
    tunnel_url = extract_tunnel_url()
    
    if not tunnel_url:
        nohup_path = os.path.join(CLOUDFLARE_DIR, NOHUP_FILE)
        print(f"\n❌ Could not extract tunnel URL.")
        print(f"📋 Please check: cat {nohup_path}")
        sys.exit(1)
    
    print(f"\n📡 Tunnel URL found: {tunnel_url}")
    
    # Update local baseurl.json file
    if not update_baseurl_json(tunnel_url):
        sys.exit(1)
    
    # Ask if user wants to update remote server
    print("\n" + "=" * 60)
    response = input("🌐 Do you want to update the REMOTE server as well? (y/n): ").lower()
    
    remote_success = False
    if response == 'y':
        remote_success = update_remote_baseurl(tunnel_url)
    else:
        print("⏭️  Skipping remote server update")
    
    print("\n" + "=" * 60)
    print("✨ Setup Complete!")
    print("=" * 60)
    print(f"🌐 Tunnel URL: {tunnel_url}")
    print(f"📂 Tunnel running in: {CLOUDFLARE_DIR}")
    print(f"📄 Logs: {os.path.join(CLOUDFLARE_DIR, NOHUP_FILE)}")
    print(f"✅ LOCAL updated: {BASE_URL_FILE}")
    if remote_success:
        print(f"✅ REMOTE updated: {REMOTE_HOST}:{REMOTE_BASE_URL_FILE}")
    print("\n📝 Useful commands:")
    print(f"   View local JSON: cat {BASE_URL_FILE}")
    print(f"   View logs: cat {os.path.join(CLOUDFLARE_DIR, NOHUP_FILE)}")
    print(f"   Stop tunnel: pkill cloudflared")
    print(f"   Check status: pgrep -a cloudflared")
    if check_sshpass_installed():
        print(f"   View remote JSON: sshpass -p '{REMOTE_PASSWORD}' ssh {REMOTE_USER}@{REMOTE_HOST} 'cat {REMOTE_BASE_URL_FILE}'")
    print("=" * 60)

if __name__ == "__main__":
    main()