"""
Job Scraper for Discord Notifications
Scrapes RemoteOK for entry/associate automation & operations roles
Sends individual notifications to Discord via webhook
Runs automatically via GitHub Actions every hour
"""

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION - Job Search Criteria
# ============================================================================

# Discord webhook URL (loaded from environment variable)
WEBHOOK_URL = os.getenv('JOB_WEBHOOK_URL')

# Keywords to INCLUDE (job must have at least one of these)
INCLUDE_KEYWORDS = [
    'automation', 'n8n', 'python', 'operations', 'administration',
    'coordinator', 'associate', 'entry level'
]

# Keywords to EXCLUDE (job must NOT have any of these)
EXCLUDE_KEYWORDS = [
    'customer service', 'sales',
    'senior', 'lead', 'director', 'manager', 'principal'
]

# Requirements (RemoteOK is remote-only by default)
MUST_BE_REMOTE = False

# File to track jobs already sent (prevents duplicates)
SENT_JOBS_FILE = 'sent_jobs.json'


# ============================================================================
# DUPLICATE TRACKING
# ============================================================================

def load_sent_jobs():
    """Load the list of job IDs we've already sent to Discord"""
    if os.path.exists(SENT_JOBS_FILE):
        with open(SENT_JOBS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_sent_job(job_id):
    """Add a job ID to the sent list"""
    sent_jobs = load_sent_jobs()
    if job_id not in sent_jobs:
        sent_jobs.append(job_id)
        with open(SENT_JOBS_FILE, 'w') as f:
            json.dump(sent_jobs, f, indent=2)


def is_already_sent(job_id):
    """Check if we've already sent this job"""
    return job_id in load_sent_jobs()


# ============================================================================
# JOB SCRAPING
# ============================================================================

def fetch_remoteok_jobs():
    """Fetch recent jobs from RemoteOK API"""
    url = "https://remoteok.com/api"
    
    try:
        # RemoteOK requires a user agent
        headers = {'User-Agent': 'Job Scraper Bot'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # First item is metadata, skip it
        jobs = response.json()[1:]
        
        print(f"✅ Fetched {len(jobs)} jobs from RemoteOK")
        return jobs
        
    except Exception as e:
        print(f"❌ Error fetching jobs: {e}")
        return []


# ============================================================================
# FILTERING LOGIC
# ============================================================================

def matches_criteria(job):
    """Check if a job matches your search criteria"""
    
    # Get job details (case-insensitive search)
    title = job.get('position', '').lower()
    description = job.get('description', '').lower()
    tags = [tag.lower() for tag in job.get('tags', [])]
    
    # Combine all text for keyword searching
    all_text = f"{title} {description} {' '.join(tags)}"
    
    # Check: Must be remote (disabled since RemoteOK is remote-only)
    if MUST_BE_REMOTE and not job.get('remote', False):
        return False
    
    # Check: Must have at least one INCLUDE keyword
    has_include = any(keyword.lower() in all_text for keyword in INCLUDE_KEYWORDS)
    if not has_include:
        return False
    
    # Check: Must NOT have any EXCLUDE keywords
    has_exclude = any(keyword.lower() in all_text for keyword in EXCLUDE_KEYWORDS)
    if has_exclude:
        return False
    
    return True


# ============================================================================
# DISCORD NOTIFICATION
# ============================================================================

def send_to_discord(job):
    """Send a job posting to Discord as a rich embed"""
    
    # Extract job details
    title = job.get('position', 'Unknown Position')
    company = job.get('company', 'Unknown Company')
    location = job.get('location', 'Remote')
    url = f"https://remoteok.com/remote-jobs/{job.get('id', '')}"
    tags = job.get('tags', [])[:5]  # First 5 tags only
    salary = job.get('salary_range', 'Not specified')
    
    # Create Discord embed
    embed = {
        "title": f"💼 {title}",
        "url": url,
        "color": 0x00FF9F,  # Cyan/green - matches your bot theme
        "fields": [
            {
                "name": "🏢 Company",
                "value": company,
                "inline": True
            },
            {
                "name": "📍 Location",
                "value": location,
                "inline": True
            },
            {
                "name": "💰 Salary",
                "value": salary,
                "inline": True
            },
            {
                "name": "🏷️ Tags",
                "value": ", ".join(tags) if tags else "None",
                "inline": False
            }
        ],
        "footer": {
            "text": f"RemoteOK | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
    }
    
    payload = {
        "embeds": [embed]
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ Sent to Discord: {title} at {company}")
        return True
        
    except Exception as e:
        print(f"❌ Error sending to Discord: {e}")
        return False


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main function - scrape, filter, and send jobs"""
    
    print("\n" + "="*60)
    print("🔍 Job Scraper Starting...")
    print("="*60 + "\n")
    
    # Fetch jobs from RemoteOK
    jobs = fetch_remoteok_jobs()
    
    if not jobs:
        print("❌ No jobs fetched. Exiting.")
        return
    
    # Filter and send matching jobs
    new_jobs_sent = 0
    
    for job in jobs:
        job_id = job.get('id')
        
        # Skip if we've already sent this job
        if is_already_sent(job_id):
            continue
        
        # Check if job matches criteria
        if matches_criteria(job):
            # Send to Discord
            if send_to_discord(job):
                # Mark as sent
                save_sent_job(job_id)
                new_jobs_sent += 1
                
                # Stop after sending 10 jobs per run (rate limiting)
                if new_jobs_sent >= 10:
                    print("\n⚠️ Reached 10 jobs limit for this run")
                    break
    
    print(f"\n✅ Complete! Sent {new_jobs_sent} new job(s) to Discord")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
