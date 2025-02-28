#!/usr/bin/env python3
"""
Script to set up the PostgreSQL database for the chatbot.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the database setup function
from api.database import setup_db

def main():
    # Load environment variables
    load_dotenv()
    
    print("Setting up database...")
    
    # Set up the database
    setup_db()
    
    print("Database setup completed successfully!")

if __name__ == "__main__":
    main()