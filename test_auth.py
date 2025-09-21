#!/usr/bin/env python3
"""
Test PhysioNet Authentication
Tests if the provided credentials can authenticate with PhysioNet.
"""

import sys
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_physionet_auth():
    """Test PhysioNet authentication."""
    username = "uragul500@gmail.com"
    password = "Ragul@4321"
    
    logger.info("🔐 Testing PhysioNet authentication...")
    
    try:
        session = requests.Session()
        
        # Get login page
        login_url = "https://physionet.org/login/"
        logger.info("Fetching login page...")
        
        login_page = session.get(login_url, timeout=30)
        login_page.raise_for_status()
        
        # Extract CSRF token
        csrf_token = None
        for line in login_page.text.split('\n'):
            if 'csrfmiddlewaretoken' in line and 'value=' in line:
                csrf_token = line.split('value="')[1].split('"')[0]
                break
        
        if not csrf_token:
            logger.error("❌ Could not find CSRF token")
            return False
            
        logger.info(f"✓ Found CSRF token: {csrf_token[:10]}...")
        
        # Attempt login
        login_data = {
            'username': username,
            'password': password,
            'csrfmiddlewaretoken': csrf_token
        }
        
        logger.info("Attempting login...")
        login_response = session.post(login_url, data=login_data, allow_redirects=True, timeout=30)
        
        # Check login success
        if 'login' in login_response.url.lower() and login_response.status_code == 200:
            logger.error("❌ Login failed - still on login page")
            logger.error(f"Response URL: {login_response.url}")
            return False
        elif login_response.status_code != 200:
            logger.error(f"❌ Login failed with status code: {login_response.status_code}")
            return False
        else:
            logger.info("✅ Login successful!")
            logger.info(f"Redirected to: {login_response.url}")
            
            # Test access to MIMIC-IV page
            mimic_url = "https://physionet.org/content/mimiciv/2.2/"
            logger.info("Testing access to MIMIC-IV page...")
            
            mimic_response = session.get(mimic_url, timeout=30)
            if mimic_response.status_code == 200:
                logger.info("✅ Successfully accessed MIMIC-IV page!")
                return True
            else:
                logger.error(f"❌ Could not access MIMIC-IV page: {mimic_response.status_code}")
                return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Network error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_physionet_auth()
    if success:
        print("\n🎉 Authentication test passed! Ready to download MIMIC-IV data.")
    else:
        print("\n💥 Authentication test failed. Please check credentials and try again.")
        sys.exit(1)