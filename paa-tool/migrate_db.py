#!/usr/bin/env python3
"""
Database migration script to add missing columns to serp_scores and serp_features tables
"""

import sqlite3
import os
from datetime import datetime

def migrate_database():
    """Add missing columns to the database"""
    
    # Get the database path
    db_path = "data/serp.db"
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found!")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if serp_scores table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='serp_scores'")
        if not cursor.fetchone():
            print("serp_scores table does not exist!")
            return False
        
        # Get current columns for serp_scores
        cursor.execute("PRAGMA table_info(serp_scores)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        # Define new columns to add to serp_scores
        new_serp_scores_columns = [
            ("share_of_voice", "REAL DEFAULT 0.0"),
            ("percentile_30d", "REAL DEFAULT 0.0"),
            ("percentile_90d", "REAL DEFAULT 0.0"),
            ("competitor_benchmark", "REAL DEFAULT 0.0"),
            ("potential_ceiling", "REAL DEFAULT 0.0")
        ]
        
        # Add missing columns to serp_scores
        for column_name, column_type in new_serp_scores_columns:
            if column_name not in existing_columns:
                print(f"Adding column {column_name} to serp_scores table...")
                cursor.execute(f"ALTER TABLE serp_scores ADD COLUMN {column_name} {column_type}")
                print(f"✓ Added {column_name}")
            else:
                print(f"Column {column_name} already exists in serp_scores")
        
        # Check if serp_features table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='serp_features'")
        if not cursor.fetchone():
            print("serp_features table does not exist!")
            return False
        
        # Get current columns for serp_features
        cursor.execute("PRAGMA table_info(serp_features)")
        existing_features_columns = [row[1] for row in cursor.fetchall()]
        
        # Define new columns to add to serp_features
        new_serp_features_columns = [
            ("position", "INTEGER"),
            ("domain", "VARCHAR(255)"),
            ("ownership_gained", "BOOLEAN DEFAULT 0"),
            ("ownership_lost", "BOOLEAN DEFAULT 0")
        ]
        
        # Add missing columns to serp_features
        for column_name, column_type in new_serp_features_columns:
            if column_name not in existing_features_columns:
                print(f"Adding column {column_name} to serp_features table...")
                cursor.execute(f"ALTER TABLE serp_features ADD COLUMN {column_name} {column_type}")
                print(f"✓ Added {column_name}")
            else:
                print(f"Column {column_name} already exists in serp_features")
        
        # Commit changes
        conn.commit()
        print("\n✓ Database migration completed successfully!")
        
        # Verify the changes for serp_scores
        cursor.execute("PRAGMA table_info(serp_scores)")
        columns = cursor.fetchall()
        print("\nCurrent serp_scores table structure:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Verify the changes for serp_features
        cursor.execute("PRAGMA table_info(serp_features)")
        columns = cursor.fetchall()
        print("\nCurrent serp_features table structure:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    print("Starting database migration...")
    success = migrate_database()
    if success:
        print("\nMigration completed successfully!")
    else:
        print("\nMigration failed!") 