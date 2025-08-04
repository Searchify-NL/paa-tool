import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import json
import logging
import os
import hashlib
import hmac
import base64
import requests
import zipfile
import io
import enum
import bcrypt
from datetime import datetime, timezone, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from io import BytesIO
import tempfile
import shutil
import subprocess
import sys
import traceback
import re
import urllib.parse
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import warnings
import difflib
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

# Research & Clustering imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
from sqlalchemy.pool import StaticPool
import sqlite3

# Scheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

# Streamlit secrets
import toml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = None
SessionLocal = None

# Research & Clustering globals
embedding_model = None
_model_cache = {}  # Cache for loaded models to prevent multiple downloads

# User roles enum
class UserRole(enum.Enum):
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"

# Audit action types
class AuditAction(enum.Enum):
    PROJECT_CREATE = "project_create"
    PROJECT_UPDATE = "project_update"
    PROJECT_DELETE = "project_delete"
    KEYWORD_UPLOAD = "keyword_upload"
    SERP_FETCH = "serp_fetch"
    CLUSTER_RUN = "cluster_run"
    PROMPT_GENERATE = "prompt_generate"
    MEMBER_INVITE = "member_invite"
    MEMBER_REMOVE = "member_remove"
    MEMBER_ROLE_UPDATE = "member_role_update"
    TAXONOMY_LOAD = "taxonomy_load"
    EXPORT_DATA = "export_data"

###############################################################################
# Taxonomy Mapping Functions                                                   
###############################################################################

def load_sitemap(url: str) -> pd.DataFrame:
    """Load and parse sitemap XML from URL.
    
    Args:
        url: Sitemap URL
    
    Returns:
        DataFrame with url, priority, lastmod columns
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Extract URLs (handle both sitemap index and regular sitemap)
        urls = []
        
        # Check if this is a sitemap index
        if 'sitemapindex' in root.tag:
            pass
            # Handle sitemap index
            for sitemap in root.findall('.//{*}sitemap'):
                loc = sitemap.find('.//{*}loc')
                if loc is not None:
                    pass
                    # Recursively load sub-sitemaps
                    sub_df = load_sitemap(loc.text)
                    urls.extend(sub_df.to_dict('records'))
        else:
            # Handle regular sitemap
            for url_elem in root.findall('.//{*}url'):
                loc = url_elem.find('.//{*}loc')
                priority = url_elem.find('.//{*}priority')
                lastmod = url_elem.find('.//{*}lastmod')
                
                if loc is not None:
                
                    pass
                    urls.append({
                        'url': loc.text,
                        'priority': float(priority.text) if priority is not None else 0.5,
                        'lastmod': lastmod.text if lastmod is not None else None
                    })
        
        df = pd.DataFrame(urls)
        logger.info(f"Loaded {len(df)} URLs from sitemap: {url}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load sitemap {url}: {e}")
        return pd.DataFrame(columns=['url', 'priority', 'lastmod'])

def extract_page_title(url: str) -> str:
    """Extract page title from URL using requests and BeautifulSoup.
    
    Args:
        url: Page URL
    
    Returns:
        Page title or URL path as fallback
    """
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        
        if title and title.text.strip():
        
            pass
            return title.text.strip()
        else:
            # Fallback to URL path
            parsed = urlparse(url)
            return parsed.path.strip('/') or parsed.netloc
            
    except Exception as e:
        logger.warning(f"Failed to extract title from {url}: {e}")
        # Fallback to URL path
        parsed = urlparse(url)
        return parsed.path.strip('/') or parsed.netloc

def load_taxonomy(sitemap_url: str = None, file = None) -> pd.DataFrame:
    """Load taxonomy from sitemap URL or uploaded file.
    
    Args:
        sitemap_url: Optional sitemap URL
        file: Optional uploaded file
    
    Returns:
        DataFrame with url, category columns
    """
    if sitemap_url:
        pass
        # Load from sitemap
        df = load_sitemap(sitemap_url)
        if not df.empty:
            pass
            # Extract titles for categories
            df['category'] = df['url'].apply(extract_page_title)
            return df[['url', 'category']]
    
    if file is not None:
    
        pass
        # Load from uploaded file
        try:
            if file.name.endswith('.csv'):
                pass
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, engine='openpyxl')
            else:
                raise ValueError("Unsupported file format")
            
            # Ensure required columns exist
            if 'url' not in df.columns or 'category' not in df.columns:
                pass
                raise ValueError("File must contain 'url' and 'category' columns")
            
            return df[['url', 'category']]
            
        except Exception as e:
            logger.error(f"Failed to load taxonomy file: {e}")
            return pd.DataFrame(columns=['url', 'category'])
    
    return pd.DataFrame(columns=['url', 'category'])

def compute_taxonomy_embeddings(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """Compute embeddings for taxonomy categories.
    
    Args:
        taxonomy_df: DataFrame with url, category columns
    
    Returns:
        DataFrame with embeddings added
    """
    if taxonomy_df.empty:
        pass
        return taxonomy_df
    
    model = load_embedding_model()
    if model is None:
        pass
        return taxonomy_df
    
    try:
        # Compute embeddings for categories
        categories = taxonomy_df['category'].tolist()
        embeddings = model.encode(categories)
        
        # Add embeddings to DataFrame
        taxonomy_df['embedding'] = list(embeddings)
        
        logger.info(f"Computed embeddings for {len(taxonomy_df)} taxonomy categories")
        return taxonomy_df
        
    except Exception as e:
        logger.error(f"Failed to compute taxonomy embeddings: {e}")
        return taxonomy_df

def match_qna_to_taxonomy(qna_df: pd.DataFrame, taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """Match Q&A questions to taxonomy URLs/categories using embeddings.
    
    Args:
        qna_df: DataFrame with Q&A data
        taxonomy_df: DataFrame with taxonomy data and embeddings
    
    Returns:
        DataFrame with matched_url, category, score columns added
    """
    if qna_df.empty or taxonomy_df.empty:
        pass
        return qna_df
    
    model = load_embedding_model()
    if model is None:
        pass
        return qna_df
    
    try:
        # Get unique questions to avoid duplicate embedding computation
        unique_questions = qna_df['question'].unique()
        question_embeddings = model.encode(unique_questions)
        
        # Create mapping from question to embedding
        question_to_embedding = dict(zip(unique_questions, question_embeddings))
        
        # Add embeddings to Q&A DataFrame
        qna_df['question_embedding'] = qna_df['question'].map(question_to_embedding)
        
        # Match each question to best taxonomy category
        matched_data = []
        
        for idx, row in qna_df.iterrows():
            question_emb = row['question_embedding']
            
            best_score = -1
            best_url = None
            best_category = None
            
            # Find best match in taxonomy
            for _, tax_row in taxonomy_df.iterrows():
                if 'embedding' in tax_row:
                    pass
                    similarity = cosine_similarity([question_emb], [tax_row['embedding']])[0][0]
                    
                    if similarity > best_score:
                    
                        pass
                        best_score = similarity
                        best_url = tax_row['url']
                        best_category = tax_row['category']
            
            matched_data.append({
                'matched_url': best_url,
                'category': best_category,
                'score': best_score
            })
        
        # Add matched data to DataFrame
        matched_df = pd.DataFrame(matched_data)
        result_df = pd.concat([qna_df, matched_df], axis=1)
        
        # Remove embedding columns for cleaner output
        result_df = result_df.drop(['question_embedding'], axis=1, errors='ignore')
        
        logger.info(f"Matched {len(result_df)} Q&A items to taxonomy")
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to match Q&A to taxonomy: {e}")
        return qna_df

def reset_taxonomy_mapping(qna_df: pd.DataFrame) -> pd.DataFrame:
    """Reset taxonomy mapping by removing matched columns.
    
    Args:
        qna_df: DataFrame with taxonomy mapping
    
    Returns:
        DataFrame with mapping columns removed
    """
    columns_to_drop = ['matched_url', 'category', 'score']
    return qna_df.drop(columns=[col for col in columns_to_drop if col in qna_df.columns])

###############################################################################
# Database Models                                                               
###############################################################################

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="user")  # admin, user
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime)
    
    # Relationships
    owned_projects = relationship("Project", back_populates="owner")
    project_memberships = relationship("ProjectMember", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    location = Column(String(10), default="us")
    domain = Column(String(255))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    owner = relationship("User", back_populates="owned_projects")
    members = relationship("ProjectMember", back_populates="project")
    keywords = relationship("Keyword", back_populates="project")
    serp_runs = relationship("SerpRun", back_populates="project")

class ProjectMember(Base):
    __tablename__ = "project_members"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER)
    invited_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    accepted_at = Column(DateTime)
    
    # Relationships
    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(100), nullable=False)
    target_type = Column(String(50))  # project, keyword, serp_run, etc.
    target_id = Column(Integer)  # ID of the target object
    details = Column(Text)  # JSON string for additional details
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")

class Keyword(Base):
    __tablename__ = "keywords"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    keyword = Column(String(500), nullable=False)
    search_volume = Column(Integer)
    position = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    project = relationship("Project", back_populates="keywords")
    serp_runs = relationship("SerpRun", back_populates="keyword")

class SerpRun(Base):
    __tablename__ = "serp_runs"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    keyword_id = Column(Integer, ForeignKey("keywords.id"))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    raw_json = Column(Text)  # Store full SERP JSON
    
    project = relationship("Project", back_populates="serp_runs")
    keyword = relationship("Keyword", back_populates="serp_runs")
    features = relationship("SerpFeature", back_populates="serp_run")
    score = relationship("SerpScore", back_populates="serp_run", uselist=False)

class SerpFeature(Base):
    __tablename__ = "serp_features"
    
    id = Column(Integer, primary_key=True)
    serp_run_id = Column(Integer, ForeignKey("serp_runs.id"))
    feature_type = Column(String(50))  # "organic", "paa", "video", etc.
    count = Column(Integer, default=0)
    details = Column(Text)  # JSON string for additional details
    # Enhanced feature tracking
    position = Column(Integer)  # For organic results
    domain = Column(String(255))  # Which domain owns this feature
    ownership_gained = Column(Boolean, default=False)  # Flag for ownership changes
    ownership_lost = Column(Boolean, default=False)
    
    serp_run = relationship("SerpRun", back_populates="features")

class SerpScore(Base):
    __tablename__ = "serp_scores"
    
    id = Column(Integer, primary_key=True)
    serp_run_id = Column(Integer, ForeignKey("serp_runs.id"))
    organic_score = Column(Float, default=0.0)
    paa_score = Column(Float, default=0.0)
    feature_score = Column(Float, default=0.0)
    featured_snippet_bonus = Column(Float, default=0.0)
    total_score = Column(Float, default=0.0)
    # New fields for enhanced tracking
    share_of_voice = Column(Float, default=0.0)
    percentile_30d = Column(Float, default=0.0)
    percentile_90d = Column(Float, default=0.0)
    competitor_benchmark = Column(Float, default=0.0)
    potential_ceiling = Column(Float, default=0.0)
    
    serp_run = relationship("SerpRun", back_populates="score")

class Competitor(Base):
    __tablename__ = "competitors"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    domain = Column(String(255), nullable=False)
    name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    project = relationship("Project")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    alert_type = Column(String(50))  # "snippet_lost", "paa_gained", "score_drop", etc.
    keyword = Column(String(500))
    details = Column(Text)  # JSON string for alert details
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    project = relationship("Project")
    user = relationship("User")

class Schedule(Base):
    __tablename__ = "schedules"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    frequency = Column(String(20))  # "daily", "weekly", "monthly"
    cron_expression = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    project = relationship("Project")

class QnaRecord(Base):
    __tablename__ = "qna_records"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    keyword = Column(String(500), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    search_volume = Column(Integer, default=0)
    answered_by_site = Column(Boolean, default=False)
    answer_relevancy = Column(Float, default=0.0)
    source_url = Column(String(1000))  # URL where the answer was found
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    project = relationship("Project")

###############################################################################
# Database Operations                                                           
###############################################################################

def init_db():
    """Initialize database and create tables."""
    global engine, SessionLocal
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create engine with connection pooling to prevent connection exhaustion
    engine = create_engine(
        "sqlite:///data/serp.db", 
        echo=False,
        pool_size=2,
        max_overflow=3,
        pool_timeout=30,
        pool_recycle=1800
    )
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    # Set expire_on_commit=False so loaded objects remain usable after the session
    # has been committed/closed (prevents DetachedInstanceError in Streamlit callbacks)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
    
    logger.info("Database initialized successfully")

def get_db():
    """Get database session."""
    if SessionLocal is None:
        pass
        init_db()
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

# Authentication functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_user(email: str, name: str, password: str, role: str = "user") -> int:
    """Create a new user."""
    db = get_db()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            pass
            raise ValueError("User with this email already exists")
        
        # Create new user
        user = User(
            email=email,
            name=name,
            password_hash=hash_password(password),
            role=role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user.id
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def authenticate_user(email: str, password: str) -> Optional[int]:
    """Authenticate a user and return user ID."""
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and verify_password(password, user.password_hash):
            pass
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            db.commit()
            return user.id
        return None
    except Exception as e:
        db.rollback()
        logger.error(f"Authentication error: {e}")
        return None
    finally:
        db.close()

def get_user_by_id(user_id: int) -> Optional[User]:
    """Get user by ID."""
    db = get_db()
    try:
        return db.query(User).filter(User.id == user_id).first()
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None
    finally:
        db.close()

# Project management functions
def get_user_projects(user_id: int) -> List[Dict]:
    """Get all projects a user has access to."""
    db = get_db()
    try:
        # Get projects owned by user
        owned_projects = db.query(Project).filter(Project.owner_user_id == user_id).all()
        
        # Get projects where user is a member
        member_projects = db.query(Project).join(ProjectMember).filter(
            ProjectMember.user_id == user_id
        ).all()
        
        # Combine and deduplicate
        all_projects = list(set(owned_projects + member_projects))
        
        result = []
        for project in all_projects:
            # Get owner name
            owner = db.query(User).filter(User.id == project.owner_user_id).first()
            owner_name = owner.name if owner else "Unknown"
            
            # Determine user's role
            if project.owner_user_id == user_id:
                pass
                role = "owner"
            else:
                member = db.query(ProjectMember).filter(
                    ProjectMember.project_id == project.id,
                    ProjectMember.user_id == user_id
                ).first()
                role = member.role.value if member else "viewer"
            
            result.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "owner_id": project.owner_user_id,
                "owner_name": owner_name,
                "location": project.location,
                "domain": project.domain,
                "user_role": role,
                "created_at": project.created_at
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting user projects: {e}")
        return []
    finally:
        db.close()

def get_project_members(project_id: int) -> pd.DataFrame:
    """Get all members of a project."""
    db = get_db()
    try:
        members = db.query(ProjectMember, User).join(User).filter(
            ProjectMember.project_id == project_id
        ).all()
        
        data = []
        for member, user in members:
            data.append({
                "id": member.id,
                "user_id": user.id,
                "name": user.name,
                "email": user.email,
                "role": member.role.value,
                "invited_at": member.invited_at,
                "accepted_at": member.accepted_at
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error getting project members: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def invite_member(project_id: int, email: str, role: UserRole, invited_by_user_id: int) -> bool:
    """Invite a user to a project."""
    db = get_db()
    try:
        # Check if user exists
        user = db.query(User).filter(User.email == email).first()
        if not user:
            pass
            raise ValueError("User not found")
        
        # Check if already a member
        existing = db.query(ProjectMember).filter(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user.id
        ).first()
        
        if existing:
        
            pass
            raise ValueError("User is already a member of this project")
        
        # Create invitation
        member = ProjectMember(
            project_id=project_id,
            user_id=user.id,
            role=role
        )
        db.add(member)
        db.commit()
        
        # Log audit action
        log_audit_action(
            user_id=invited_by_user_id,
            action=AuditAction.MEMBER_INVITE.value,
            target_type="project",
            target_id=project_id,
            details=f"Invited {email} as {role.value}"
        )
        
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error inviting member: {e}")
        raise e
    finally:
        db.close()

def remove_member(project_id: int, user_id: int, removed_by_user_id: int) -> bool:
    """Remove a user from a project."""
    db = get_db()
    try:
        member = db.query(ProjectMember).filter(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id
        ).first()
        
        if not member:
        
            pass
            raise ValueError("User is not a member of this project")
        
        db.delete(member)
        db.commit()
        
        # Log audit action
        log_audit_action(
            user_id=removed_by_user_id,
            action=AuditAction.MEMBER_REMOVE.value,
            target_type="project",
            target_id=project_id,
            details=f"Removed user {user_id}"
        )
        
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing member: {e}")
        raise e
    finally:
        db.close()

def check_user_permission(user_id: int, project_id: int, required_role: UserRole) -> bool:
    """Check if user has required permission for project."""
    db = get_db()
    try:
        # Check if user owns the project
        project = db.query(Project).filter(Project.id == project_id).first()
        if project and project.owner_user_id == user_id:
            pass
            return True
        
        # Check member role
        member = db.query(ProjectMember).filter(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id
        ).first()
        
        if not member:
        
            pass
            return False
        
        # Role hierarchy: owner > editor > viewer
        role_hierarchy = {
            UserRole.OWNER: 3,
            UserRole.EDITOR: 2,
            UserRole.VIEWER: 1
        }
        
        user_role_level = role_hierarchy.get(member.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_role_level >= required_level
    except Exception as e:
        logger.error(f"Error checking permissions: {e}")
        return False
    finally:
        db.close()

def log_audit_action(user_id: int, action: str, target_type: str = None, target_id: int = None, details: str = None):
    """Log an audit action."""
    db = get_db()
    try:
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details
        )
        db.add(audit_log)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error logging audit action: {e}")
    finally:
        db.close()

def get_audit_logs(user_id: int = None, project_id: int = None, limit: int = 100) -> pd.DataFrame:
    """Get audit logs."""
    db = get_db()
    try:
        query = db.query(AuditLog, User).join(User)
        
        if user_id:
        
            pass
            query = query.filter(AuditLog.user_id == user_id)
        
        if project_id:
        
            pass
            query = query.filter(AuditLog.target_id == project_id)
        
        logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
        
        data = []
        for log, user in logs:
            data.append({
                "id": log.id,
                "user_name": user.name,
                "action": log.action,
                "target_type": log.target_type,
                "target_id": log.target_id,
                "details": log.details,
                "timestamp": log.timestamp
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def save_serp_run(project_id: int, keyword: str, parsed: dict, scores: dict, raw_json: str = None, device: str = "desktop"):
    """Save SERP run with enhanced tracking and ownership analysis."""
    db = get_db()
    try:
        # Get or create keyword
        keyword_obj = db.query(Keyword).filter(
            Keyword.project_id == project_id,
            Keyword.keyword == keyword
        ).first()
        
        if not keyword_obj:
            # Try to get search volume from existing data or default to 0
            search_volume = 0
            # You could add logic here to fetch search volume from a source if needed
            
            keyword_obj = Keyword(
                project_id=project_id,
                keyword=keyword,
                search_volume=search_volume
            )
            db.add(keyword_obj)
            db.flush()  # Get the ID
        else:
            # Preserve existing search_volume - never overwrite with zero
            # The search_volume should only be set during upload, not during SERP runs
            pass
        
        # Create SERP run
        serp_run = SerpRun(
            project_id=project_id,
            keyword_id=keyword_obj.id,
            raw_json=raw_json
        )
        db.add(serp_run)
        db.flush()
        
        # Save features with ownership data
        feature_counts = parsed.get("feature_counts", {})
        ownership_data = parsed.get("ownership_data", {})
        
        for feature_type, count in feature_counts.items():
            if count > 0:
                feature = SerpFeature(
                    serp_run_id=serp_run.id,
                    feature_type=feature_type,
                    count=count,
                    details=json.dumps(parsed.get("feature_details", {}).get(feature_type, {}))
                )
                db.add(feature)
        
        # Save organic results with positions
        organic_results = parsed.get("organic_results", [])
        for i, result in enumerate(organic_results, 1):
            domain = extract_domain(result.get("link", ""))
            feature = SerpFeature(
                serp_run_id=serp_run.id,
                feature_type="organic",
                count=1,
                position=i,
                domain=domain,
                details=json.dumps(result)
            )
            db.add(feature)
        
        # Save PAA questions with ownership
        paa_questions = parsed.get("paa_questions", [])
        for i, qna in enumerate(paa_questions):
            feature = SerpFeature(
                serp_run_id=serp_run.id,
                feature_type="paa",
                count=1,
                position=i + 1,
                domain=qna.get("domain", ""),
                details=json.dumps(qna)
            )
            db.add(feature)
        
        # Create score record
        score = SerpScore(
            serp_run_id=serp_run.id,
            organic_score=scores.get("organic_score", 0),
            paa_score=scores.get("paa_score", 0),
            feature_score=scores.get("feature_score", 0),
            featured_snippet_bonus=scores.get("featured_snippet_bonus", 0),
            total_score=scores.get("total_score", 0),
            share_of_voice=scores.get("share_of_voice", 0),
            potential_ceiling=scores.get("potential_ceiling", 0)
        )
        db.add(score)
        
        db.commit()
        return serp_run.id
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save SERP run: {e}")
        raise
    finally:
        db.close()

def add_competitor(project_id: int, domain: str, name: str = None) -> bool:
    """Add a competitor to a project."""
    db = get_db()
    try:
        competitor = Competitor(
            project_id=project_id,
            domain=domain.lower(),
            name=name or domain
        )
        db.add(competitor)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to add competitor: {e}")
        return False
    finally:
        db.close()

def get_competitors(project_id: int) -> List[Dict]:
    """Get competitors for a project."""
    db = get_db()
    try:
        competitors = db.query(Competitor).filter(
            Competitor.project_id == project_id,
            Competitor.is_active == True
        ).all()
        
        return [
            {
                "id": comp.id,
                "domain": comp.domain,
                "name": comp.name,
                "created_at": comp.created_at
            }
            for comp in competitors
        ]
    finally:
        db.close()

def remove_competitor(competitor_id: int) -> bool:
    """Remove a competitor from a project."""
    db = get_db()
    try:
        competitor = db.query(Competitor).filter(Competitor.id == competitor_id).first()
        if competitor:
            competitor.is_active = False
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to remove competitor: {e}")
        return False
    finally:
        db.close()

def create_alert(user_id: int, project_id: int, alert_type: str, keyword: str, details: Dict) -> bool:
    """Create an alert for a user."""
    db = get_db()
    try:
        alert = Alert(
            user_id=user_id,
            project_id=project_id,
            alert_type=alert_type,
            keyword=keyword,
            details=json.dumps(details)
        )
        db.add(alert)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create alert: {e}")
        return False
    finally:
        db.close()

def get_alerts(user_id: int, project_id: int = None, unread_only: bool = True) -> pd.DataFrame:
    """Get alerts for a user."""
    db = get_db()
    try:
        query = db.query(Alert).filter(Alert.user_id == user_id)
        
        if project_id:
            query = query.filter(Alert.project_id == project_id)
        
        if unread_only:
            query = query.filter(Alert.is_read == False)
        
        alerts = query.order_by(Alert.created_at.desc()).all()
        
        data = []
        for alert in alerts:
            data.append({
                "id": alert.id,
                "alert_type": alert.alert_type,
                "keyword": alert.keyword,
                "details": json.loads(alert.details) if alert.details else {},
                "is_read": alert.is_read,
                "created_at": alert.created_at
            })
        
        return pd.DataFrame(data)
    finally:
        db.close()

def mark_alert_read(alert_id: int) -> bool:
    """Mark an alert as read."""
    db = get_db()
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_read = True
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to mark alert read: {e}")
        return False
    finally:
        db.close()





def get_trends_data(project_id: int, days: int = 30) -> pd.DataFrame:
    """Get comprehensive trends data for the Trends tab."""
    db = get_db()
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get latest SERP runs with scores
        query = db.query(SerpRun, SerpScore, Keyword).join(SerpScore).join(Keyword).filter(
            SerpRun.project_id == project_id,
            SerpRun.timestamp >= start_date,
            SerpRun.timestamp <= end_date
        ).order_by(SerpRun.timestamp.desc())
        
        results = query.all()
        
        data = []
        for serp_run, score, keyword in results:
            # Get feature ownership data
            features = db.query(SerpFeature).filter(SerpFeature.serp_run_id == serp_run.id).all()
            
            feature_summary = {}
            ownership_summary = {}
            
            for feature in features:
                if feature.feature_type not in feature_summary:
                    feature_summary[feature.feature_type] = 0
                feature_summary[feature.feature_type] += feature.count
                
                if feature.domain:
                    if feature.feature_type not in ownership_summary:
                        ownership_summary[feature.feature_type] = []
                    ownership_summary[feature.feature_type].append(feature.domain)
            
            row = {
                "keyword": keyword.keyword,
                "timestamp": serp_run.timestamp,
                "total_score": score.total_score,
                "organic_score": score.organic_score,
                "paa_score": score.paa_score,
                "feature_score": score.feature_score,
                "share_of_voice": score.share_of_voice,
                "potential_ceiling": score.potential_ceiling,
                "feature_summary": feature_summary,
                "ownership_summary": ownership_summary
            }
            data.append(row)
        
        return pd.DataFrame(data)
        
    finally:
        db.close()

def calculate_percentiles(project_id: int, keyword: str, days: int = 30) -> Dict[str, float]:
    """Calculate percentile rankings for a keyword."""
    db = get_db()
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get all scores for the keyword in the time period
        scores = db.query(SerpScore).join(SerpRun).join(Keyword).filter(
            SerpRun.project_id == project_id,
            Keyword.keyword == keyword,
            SerpRun.timestamp >= start_date,
            SerpRun.timestamp <= end_date
        ).all()
        
        if not scores:
            return {"percentile_30d": 0, "percentile_90d": 0}
        
        total_scores = [score.total_score for score in scores]
        total_scores.sort()
        
        # Calculate percentiles
        current_score = scores[-1].total_score if scores else 0
        percentile_30d = (sum(1 for score in total_scores if score <= current_score) / len(total_scores)) * 100
        
        # For 90-day percentile, use the same data but different calculation
        percentile_90d = percentile_30d  # Simplified for now
        
        return {
            "percentile_30d": percentile_30d,
            "percentile_90d": percentile_90d
        }
        
    finally:
        db.close()


# ====================== Trends Helper Functions ======================




    """Save SERP run data to database."""
    db = get_db()
    try:
        # Get or create keyword
        keyword_obj = db.query(Keyword).filter(
            Keyword.project_id == project_id,
            Keyword.keyword == keyword
        ).first()
        
        if not keyword_obj:
        
            pass
            keyword_obj = Keyword(
                project_id=project_id,
                keyword=keyword
            )
            db.add(keyword_obj)
            db.flush()  # Get the ID
        
        # Create SERP run
        serp_run = SerpRun(
            project_id=project_id,
            keyword_id=keyword_obj.id,
            raw_json=raw_json
        )
        db.add(serp_run)
        db.flush()
        
        # Save features
        feature_counts = parsed.get("feature_counts", {})
        for feature_type, count in feature_counts.items():
            if count > 0:
                pass
                feature = SerpFeature(
                    serp_run_id=serp_run.id,
                    feature_type=feature_type,
                    count=count
                )
                db.add(feature)
        
        # Save PAA questions
        paa_questions = parsed.get("paa_questions", [])
        if paa_questions:
            pass
            paa_feature = SerpFeature(
                serp_run_id=serp_run.id,
                feature_type="paa",
                count=len(paa_questions),
                details=json.dumps(paa_questions)
            )
            db.add(paa_feature)
        
        # Save featured snippet
        if parsed.get("has_featured_snippet"):
            pass
            snippet_feature = SerpFeature(
                serp_run_id=serp_run.id,
                feature_type="featured_snippet",
                count=1
            )
            db.add(snippet_feature)
        
        # Save scores
        score = SerpScore(
            serp_run_id=serp_run.id,
            organic_score=scores.get("organic_score", 0.0),
            paa_score=scores.get("paa_score", 0.0),
            feature_score=scores.get("feature_score", 0.0),
            featured_snippet_bonus=scores.get("featured_snippet_bonus", 0.0),
            total_score=scores.get("total_score", 0.0)
        )
        db.add(score)
        
        db.commit()
        logger.info(f"Saved SERP run for keyword: {keyword}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save SERP run: {e}")
        raise
    finally:
        db.close()

def create_project(name: str, description: str, owner_user_id: int, location: str = "us", domain: str = None) -> int:
    """Create a new project and return its ID."""
    db = get_db()
    try:
        project = Project(
            name=name,
            description=description,
            owner_user_id=owner_user_id,
            location=location,
            domain=domain
        )
        db.add(project)
        db.commit()
        
        # Log the action
        log_audit_action(
            user_id=owner_user_id,
            action=AuditAction.PROJECT_CREATE.value,
            target_type="project",
            target_id=project.id,
            details=f"Created project: {name}"
        )
        
        logger.info(f"Created project: {name}")
        return project.id
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create project: {e}")
        raise
    finally:
        db.close()

def get_projects() -> List[Dict]:
    """Get all projects."""
    db = get_db()
    try:
        projects = db.query(Project).all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "location": p.location,
                "domain": p.domain,
                "created_at": p.created_at
            }
            for p in projects
        ]
    finally:
        db.close()

def schedule_job(project_id: int, frequency: str):
    """Schedule a job for a project."""
    db = get_db()
    try:
        # Create cron expression based on frequency
        if frequency == "daily":
            pass
            cron_expr = "0 9 * * *"  # Daily at 9 AM
        elif frequency == "weekly":
            cron_expr = "0 9 * * 1"  # Weekly on Monday at 9 AM
        elif frequency == "monthly":
            cron_expr = "0 9 1 * *"  # Monthly on 1st at 9 AM
        else:
            raise ValueError(f"Invalid frequency: {frequency}")
        
        # Save schedule to DB
        schedule = Schedule(
            project_id=project_id,
            frequency=frequency,
            cron_expression=cron_expr
        )
        db.add(schedule)
        db.commit()
        
        # Add job to scheduler
        scheduler.add_job(
            func=run_scheduled_job,
            trigger=CronTrigger.from_crontab(cron_expr),
            args=[project_id],
            id=f"project_{project_id}_{frequency}",
            replace_existing=True
        )
        
        logger.info(f"Scheduled job for project {project_id} with frequency {frequency}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to schedule job: {e}")
        raise
    finally:
        db.close()

def load_schedules() -> List[Dict]:
    """Load all active schedules from database."""
    db = get_db()
    try:
        schedules = db.query(Schedule).filter(Schedule.is_active == True).all()
        return [
            {
                "id": s.id,
                "project_id": s.project_id,
                "frequency": s.frequency,
                "cron_expression": s.cron_expression,
                "created_at": s.created_at
            }
            for s in schedules
        ]
    finally:
        db.close()

def cancel_schedule(schedule_id: int):
    """Cancel a scheduled job."""
    db = get_db()
    try:
        schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
        if schedule:
            pass
            schedule.is_active = False
            db.commit()
            
            # Remove from scheduler
            job_id = f"project_{schedule.project_id}_{schedule.frequency}"
            try:
                scheduler.remove_job(job_id)
            except:
                pass  # Job might not exist
            
            logger.info(f"Cancelled schedule {schedule_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to cancel schedule: {e}")
        raise
    finally:
        db.close()

def run_scheduled_job(project_id: int):
    """Run a scheduled job for a project."""
    logger.info(f"Running scheduled job for project {project_id}")
    
    db = get_db()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            pass
            logger.error(f"Project {project_id} not found")
            return
        
        # Get keywords for this project
        keywords = [kw.keyword for kw in project.keywords]
        if not keywords:
            pass
            logger.warning(f"No keywords found for project {project_id}")
            return
        
        # Fetch and score keywords
        results = fetch_and_score_keywords(
            keywords=keywords,
            location=project.location,
            domain=project.domain
        )
        
        # Save results
        for _, row in results.iterrows():
            keyword = row["keyword"]
            # Re-fetch SERP data for storage
            serp_json = fetch_serp(keyword, project.location)
            if serp_json:
                pass
                parsed = parse_serp_features(serp_json)
                scores = {
                    "organic_score": row["organic_score"],
                    "paa_score": row["paa_score"],
                    "feature_score": row["feature_score"],
                    "featured_snippet_bonus": row["featured_snippet_bonus"],
                    "total_score": row["serp_score"]
                }
                save_serp_run(project_id, keyword, parsed, scores, json.dumps(serp_json))
        
        logger.info(f"Completed scheduled job for project {project_id}")
        
    except Exception as e:
        logger.error(f"Failed to run scheduled job for project {project_id}: {e}")
    finally:
        db.close()

def plot_trends(project_id: int, keyword: Optional[str], metric: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Get trend data for plotting."""
    db = get_db()
    try:
        query = db.query(SerpRun, SerpScore).join(SerpScore).options(joinedload(SerpRun.keyword)).filter(
            SerpRun.project_id == project_id,
            SerpRun.timestamp >= start_date,
            SerpRun.timestamp <= end_date
        )
        
        if keyword:
        
            pass
            query = query.join(Keyword).filter(Keyword.keyword == keyword)
        
        results = query.order_by(SerpRun.timestamp).all()
        
        data = []
        for serp_run, score in results:
            row = {
                "timestamp": serp_run.timestamp,
                "keyword": serp_run.keyword.keyword if serp_run.keyword else "Unknown"
            }
            
            if metric == "total_score":
            
                pass
                row["value"] = score.total_score
            elif metric == "organic_score":
                row["value"] = score.organic_score
            elif metric == "paa_score":
                row["value"] = score.paa_score
            elif metric == "feature_score":
                row["value"] = score.feature_score
            else:
                # Feature count
                feature = db.query(SerpFeature).filter(
                    SerpFeature.serp_run_id == serp_run.id,
                    SerpFeature.feature_type == metric
                ).first()
                row["value"] = feature.count if feature else 0
            
            data.append(row)
        
        return pd.DataFrame(data)
        
    finally:
        db.close()

###############################################################################
# Trends Analysis Functions                                                     
###############################################################################

def get_qna_data_for_trends(project_id: int) -> pd.DataFrame:
    """Get Q&A data for trends analysis."""
    try:
        db = get_db()
        qna_records = db.query(QnaRecord).filter(
            QnaRecord.project_id == project_id
        ).all()
        
        if not qna_records:
            return pd.DataFrame()
        
        data = []
        for record in qna_records:
            data.append({
                'keyword': record.keyword,
                'question': record.question,
                'answer': record.answer,
                'search_volume': record.search_volume,
                'answered_by_site': record.answered_by_site,
                'answer_relevancy': record.answer_relevancy,
                'source_url': record.source_url,
                'created_at': record.created_at
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to get Q&A data: {e}")
        return pd.DataFrame()
    finally:
        try:
            db.close()
        except:
            pass

def get_serp_features_for_trends(project_id: int) -> pd.DataFrame:
    """Get SERP features data for trends analysis."""
    try:
        db = get_db()
        features = db.query(SerpFeature).join(SerpRun).filter(
            SerpRun.project_id == project_id
        ).all()
        
        if not features:
            return pd.DataFrame()
        
        data = []
        for feature in features:
            # Get keyword from the SERP run
            keyword = feature.serp_run.keyword.keyword if feature.serp_run.keyword else "Unknown"
            
            data.append({
                'keyword': keyword,
                'feature_type': feature.feature_type,
                'count': feature.count,
                'position': feature.position,
                'domain': feature.domain,
                'owns_feature': feature.domain == db.query(Project).filter(Project.id == project_id).first().domain if feature.domain else False,
                'ownership_gained': feature.ownership_gained,
                'ownership_lost': feature.ownership_lost,
                'details': feature.details,
                'timestamp': feature.serp_run.timestamp
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to get SERP features data: {e}")
        return pd.DataFrame()
    finally:
        try:
            db.close()
        except:
            pass

def build_enhanced_overview_df(
    trends_df: pd.DataFrame,
    qna_df: pd.DataFrame,
    serp_features_df: pd.DataFrame,
    project_id: int,
) -> pd.DataFrame:
    """Merge trends, Q&A, and SERP feature data into a single overview DataFrame."""

    if trends_df.empty:
        return pd.DataFrame()

    # --- Aggregate Q&A metrics ---
    if not qna_df.empty and "answered_by_site" in qna_df.columns:
        qna_summary = (
            qna_df.groupby("keyword")
            .agg(
                total_qas=("question", "count"),
                answered_qas=("answered_by_site", "sum"),
                avg_relevancy=("answer_relevancy", "mean"),
            )
            .reset_index()
        )

        qna_summary["paa_score"] = qna_summary.apply(
            lambda row: (row["answered_qas"] / row["total_qas"]) * 100 if row["total_qas"] > 0 else 0, axis=1
        )
    else:
        qna_summary = pd.DataFrame(columns=["keyword", "total_qas", "answered_qas", "avg_relevancy", "paa_score"])

    # --- Aggregate SERP features ---
    if not serp_features_df.empty:
        # Get project domain for ownership calculation
        db = get_db()
        project = db.query(Project).filter(Project.id == project_id).first()
        project_domain = project.domain if project else None
        db.close()

        # Aggregate features by keyword
        feature_summary = (
            serp_features_df.groupby("keyword")
            .agg({
                "feature_type": lambda x: list(x),
                "count": "sum",
                "owns_feature": "sum",
                "ownership_gained": "sum",
                "ownership_lost": "sum"
            })
            .reset_index()
        )

        # Create feature flags
        feature_summary["has_featured_snippet"] = feature_summary["feature_type"].apply(
            lambda x: "featured_snippet" in x
        )
        feature_summary["has_local_pack"] = feature_summary["feature_type"].apply(
            lambda x: "local_pack" in x
        )
        feature_summary["has_video"] = feature_summary["feature_type"].apply(
            lambda x: "video" in x
        )
        feature_summary["has_knowledge_panel"] = feature_summary["feature_type"].apply(
            lambda x: "knowledge_panel" in x
        )
        feature_summary["has_related_searches"] = feature_summary["feature_type"].apply(
            lambda x: "related_searches" in x
        )
        feature_summary["has_sitelinks"] = feature_summary["feature_type"].apply(
            lambda x: "sitelinks" in x
        )

        # Calculate feature score
        feature_summary["feature_score"] = (
            feature_summary["has_featured_snippet"].astype(int) * 20 +
            feature_summary["has_local_pack"].astype(int) * 15 +
            feature_summary["has_video"].astype(int) * 10 +
            feature_summary["has_knowledge_panel"].astype(int) * 10 +
            feature_summary["has_related_searches"].astype(int) * 5 +
            feature_summary["has_sitelinks"].astype(int) * 5
        )
    else:
        feature_summary = pd.DataFrame(columns=[
            "keyword", "has_featured_snippet", "has_local_pack", "has_video",
            "has_knowledge_panel", "has_related_searches", "has_sitelinks", "feature_score"
        ])

    # --- Merge all data ---
    overview_df = trends_df.copy()

    # Merge Q&A data
    if not qna_summary.empty:
        overview_df = overview_df.merge(qna_summary, on="keyword", how="left")
        overview_df["total_qas"] = overview_df["total_qas"].fillna(0)
        overview_df["answered_qas"] = overview_df["answered_qas"].fillna(0)
        overview_df["avg_relevancy"] = overview_df["avg_relevancy"].fillna(0)
        # Ensure paa_score column exists before accessing it
        if "paa_score" in overview_df.columns:
            overview_df["paa_score"] = overview_df["paa_score"].fillna(0)
        else:
            overview_df["paa_score"] = 0
    else:
        overview_df["total_qas"] = 0
        overview_df["answered_qas"] = 0
        overview_df["avg_relevancy"] = 0
        overview_df["paa_score"] = 0

    # Merge feature data
    if not feature_summary.empty:
        overview_df = overview_df.merge(feature_summary, on="keyword", how="left")
        for col in ["has_featured_snippet", "has_local_pack", "has_video", "has_knowledge_panel", "has_related_searches", "has_sitelinks"]:
            overview_df[col] = overview_df[col].fillna(False)
        # Ensure feature_score column exists before accessing it
        if "feature_score" in overview_df.columns:
            overview_df["feature_score"] = overview_df["feature_score"].fillna(0)
        else:
            overview_df["feature_score"] = 0
    else:
        for col in ["has_featured_snippet", "has_local_pack", "has_video", "has_knowledge_panel", "has_related_searches", "has_sitelinks"]:
            overview_df[col] = False
        overview_df["feature_score"] = 0

    # Calculate total score - ensure all required columns exist
    real_estate_score = overview_df.get("real_estate_score", 0)
    paa_score = overview_df.get("paa_score", 0)
    feature_score = overview_df.get("feature_score", 0)
    
    overview_df["total_score"] = (
        real_estate_score * 0.4 +
        paa_score * 0.3 +
        feature_score * 0.3
    )

    return overview_df

###############################################################################
# Scheduler Setup                                                               
###############################################################################

scheduler = BackgroundScheduler()
scheduler.start()

def init_scheduler():
    """Initialize scheduler with existing jobs."""
    db = get_db()
    try:
        schedules = db.query(Schedule).filter(Schedule.is_active == True).all()
        for schedule in schedules:
            scheduler.add_job(
                func=run_scheduled_job,
                trigger=CronTrigger.from_crontab(schedule.cron_expression),
                args=[schedule.project_id],
                id=f"project_{schedule.project_id}_{schedule.frequency}",
                replace_existing=True
            )
        logger.info(f"Initialized scheduler with {len(schedules)} jobs")
    finally:
        db.close()

###############################################################################
# SearchAPI.io Integration                                                      
###############################################################################

def get_searchapi_key() -> Optional[str]:
    """Get SearchAPI key from Streamlit secrets or environment."""
    try:
        return st.secrets["searchapi"]["api_key"]
    except:
        return None

def fetch_serp(keyword: str, location: str = "us", api_key: Optional[str] = None, device: str = "desktop", num_results: int = 100) -> Optional[Dict[str, Any]]:
    """Fetch live Google SERP for a keyword using SearchAPI.io with enhanced features.
    
    Args:
        keyword: Search query
        location: Country code (e.g., "us", "uk", "de", "fr", "es", "it", "ca", "au", "nl", "br", "mx", "jp", "kr", "cn", "in", etc.)
        api_key: SearchAPI key (will try to get from secrets if None)
        device: "desktop" or "mobile"
        num_results: Number of organic results to fetch (max 100)
    
    Returns:
        SERP JSON response or None if failed
    """
    if api_key is None:
        api_key = get_searchapi_key()
    
    if not api_key:
        st.error("SearchAPI key not found. Please add SEARCHAPI_KEY to your Streamlit secrets.")
        return None
    
    url = "https://www.searchapi.io/api/v1/search"
    
    # Set language based on location
    language_map = {
        "us": "en", "uk": "en", "de": "de", "fr": "fr", "es": "es", "it": "it",
        "ca": "en", "au": "en", "nl": "nl", "br": "pt", "mx": "es", "jp": "ja",
        "kr": "ko", "cn": "zh", "in": "en", "se": "sv", "no": "no", "dk": "da",
        "fi": "fi", "pl": "pl", "cz": "cs", "hu": "hu", "ro": "ro", "bg": "bg",
        "hr": "hr", "rs": "sr", "sk": "sk", "si": "sl", "ee": "et", "lv": "lv",
        "lt": "lt", "ie": "en", "nz": "en", "za": "en", "sg": "en", "my": "en",
        "ph": "en", "th": "th", "vn": "vi", "id": "id", "tr": "tr", "gr": "el",
        "pt": "pt", "be": "nl", "ch": "de", "at": "de", "lu": "fr", "mt": "en", "cy": "en"
    }
    language = language_map.get(location, "en")
    
    # Enhanced parameters for comprehensive SERP analysis
    params = {
        "engine": "google",
        "q": keyword,
        "gl": location,
        "hl": language,
        "num": min(num_results, 100),  # Top 100 organic results
        "api_key": api_key,
        "device": device,
        # Include all rich SERP features
        "include_answer_box": "true",
        "include_knowledge_graph": "true",
        "include_related_questions": "true",
        "include_related_searches": "true",
        "include_sitelinks": "true",
        "include_ads": "true",
        "include_local_results": "true",
        "include_video_results": "true",
        "include_image_results": "true",
        "include_news_results": "true",
        "include_shopping_results": "true",
        "include_reviews": "true"
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)  # Increased timeout for 100 results
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.warning(f"Rate limit hit for keyword '{keyword}'. Waiting 5 seconds...")
            time.sleep(5)
            return None
        else:
            st.error(f"SearchAPI error for '{keyword}': {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch SERP for '{keyword}': {e}")
        return None

def parse_serp_features(serp_json: Dict[str, Any]) -> Dict[str, Any]:
    """Parse SERP JSON to extract comprehensive features and ownership data.
    
    Returns:
        Dict with detailed feature analysis and organic results
    """
    if not serp_json:
        return {
            "num_organic": 0,
            "paa_questions": [],
            "has_featured_snippet": False,
            "feature_counts": {},
            "organic_results": [],
            "feature_details": {},
            "ownership_data": {}
        }
    
    # Extract organic results (up to 100)
    organic_results = serp_json.get("organic_results", [])
    organic_urls = [result.get("link", "") for result in organic_results]
    
    # Extract PAA questions with ownership
    paa_questions = []
    paa_block = serp_json.get("related_questions", [])
    if paa_block:
        for q in paa_block:
            if q.get("question"):
                question_data = {
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "source_url": q.get("source_url", ""),  # Use correct API field
                    "domain": extract_domain(q.get("source_url", ""))
                }
                paa_questions.append(question_data)
    
    # Check for featured snippet with ownership
    featured_snippet = None
    if serp_json.get("answer_box"):
        featured_snippet = {
            "title": serp_json["answer_box"].get("title", ""),
            "answer": serp_json["answer_box"].get("answer", ""),
            "source_url": serp_json["answer_box"].get("link", ""),
            "domain": extract_domain(serp_json["answer_box"].get("link", ""))
        }
    elif serp_json.get("knowledge_graph"):
        featured_snippet = {
            "title": serp_json["knowledge_graph"].get("title", ""),
            "description": serp_json["knowledge_graph"].get("description", ""),
            "source_url": serp_json["knowledge_graph"].get("link", ""),
            "domain": extract_domain(serp_json["knowledge_graph"].get("link", ""))
        }
    
    # Comprehensive feature analysis
    feature_details = {
        "featured_snippet": featured_snippet,
        "local_pack": serp_json.get("local_results", []),
        "video_carousel": serp_json.get("video_results", []),
        "image_pack": serp_json.get("image_results", []),
        "news_results": serp_json.get("news_results", []),
        "shopping_results": serp_json.get("shopping_results", []),
        "ads": serp_json.get("ads", []),
        "sitelinks": serp_json.get("sitelinks", []),
        "related_searches": serp_json.get("related_searches", []),
        "knowledge_panel": serp_json.get("knowledge_graph"),
        "reviews": serp_json.get("reviews", [])
    }
    
    # Count features
    feature_counts = {
        "organic": len(organic_results),
        "paa": len(paa_questions),
        "featured_snippet": 1 if featured_snippet else 0,
        "local_pack": len(serp_json.get("local_results", [])),
        "video": len(serp_json.get("video_results", [])),
        "image": len(serp_json.get("image_results", [])),
        "news": len(serp_json.get("news_results", [])),
        "shopping": len(serp_json.get("shopping_results", [])),
        "ads": len(serp_json.get("ads", [])),
        "sitelinks": len(serp_json.get("sitelinks", [])),
        "knowledge_panel": 1 if serp_json.get("knowledge_graph") else 0,
        "reviews": len(serp_json.get("reviews", []))
    }
    
    # Extract ownership data
    ownership_data = {
        "organic_ownership": [extract_domain(result.get("link", "")) for result in organic_results],
        "paa_ownership": [q.get("domain", "") for q in paa_questions],
        "featured_snippet_owner": featured_snippet.get("domain", "") if featured_snippet else None,
        "local_pack_owners": [extract_domain(local.get("link", "")) for local in serp_json.get("local_results", [])],
        "video_owners": [extract_domain(video.get("link", "")) for video in serp_json.get("video_results", [])],
        "image_owners": [extract_domain(img.get("link", "")) for img in serp_json.get("image_results", [])],
        "news_owners": [extract_domain(news.get("link", "")) for news in serp_json.get("news_results", [])],
        "shopping_owners": [extract_domain(shop.get("link", "")) for shop in serp_json.get("shopping_results", [])]
    }
    
    return {
        "num_organic": len(organic_results),
        "paa_questions": paa_questions,
        "has_featured_snippet": bool(featured_snippet),
        "feature_counts": feature_counts,
        "feature_details": feature_details,
        "ownership_data": ownership_data,
        "organic_results": organic_results,
        "organic_urls": organic_urls
    }

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return ""

def compute_serp_real_estate_score(
    serp_data: Dict[str, Any], 
    ctr_df: pd.DataFrame,
    domain: Optional[str] = None,
    competitors: List[str] = None
) -> Dict[str, Any]:
    """Compute comprehensive SERP real-estate score for a keyword.
    
    Args:
        serp_data: Parsed SERP features
        ctr_df: CTR benchmark DataFrame
        domain: Optional domain to filter organic results
        competitors: List of competitor domains
    
    Returns:
        Dict with comprehensive scores and breakdown
    """
    ctr_mapping = ctr_df.set_index("position")["average_ctr"].to_dict()
    
    # Organic score (CTR-weighted positions up to 100)
    organic_score = 0
    organic_positions = []
    organic_ownership = []
    
    organic_results = serp_data.get("organic_results", [])
    for i, result in enumerate(organic_results, 1):
        result_domain = extract_domain(result.get("link", ""))
        if domain and domain in result.get("link", ""):
            organic_score += ctr_mapping.get(i, 0)
            organic_positions.append(i)
            organic_ownership.append(result_domain)
        elif not domain:
            organic_score += ctr_mapping.get(i, 0)
            organic_positions.append(i)
            organic_ownership.append(result_domain)
    
    # PAA score (# of PAA answers × avg CTR of top slots)
    paa_weight = (ctr_mapping.get(1, 0) + ctr_mapping.get(2, 0)) / 2
    paa_questions = serp_data.get("paa_questions", [])
    paa_score = len(paa_questions) * paa_weight
    
    # Feature score (weighted counts for each non-organic feature)
    feature_counts = serp_data.get("feature_counts", {})
    feature_weights = {
        "featured_snippet": 15.0,  # High value
        "local_pack": 8.0,
        "video": 6.0,
        "image": 4.0,
        "news": 5.0,
        "shopping": 7.0,
        "ads": 2.0,  # Lower value
        "sitelinks": 3.0,
        "knowledge_panel": 12.0,
        "reviews": 4.0
    }
    
    feature_score = 0
    for feature_type, count in feature_counts.items():
        if feature_type != "organic" and feature_type != "paa":
            feature_score += count * feature_weights.get(feature_type, 1.0)
    
    # Featured snippet bonus (fixed bonus if present)
    featured_snippet_bonus = 15.0 if serp_data.get("has_featured_snippet") else 0.0
    
    # Total score (normalized to 0-100 scale)
    total_score = organic_score + paa_score + feature_score + featured_snippet_bonus
    
    # Calculate potential ceiling (max possible score)
    max_organic_score = sum(ctr_mapping.get(i, 0) for i in range(1, 101))  # Top 100 positions
    max_paa_score = 10 * paa_weight  # Assume max 10 PAA questions
    max_feature_score = sum(feature_weights.values())  # One of each feature
    max_featured_snippet_bonus = 15.0
    potential_ceiling = max_organic_score + max_paa_score + max_feature_score + max_featured_snippet_bonus
    
    # Share of voice calculation
    share_of_voice = 0.0
    if competitors and domain:
        competitor_scores = []
        ownership_data = serp_data.get("ownership_data", {})
        
        for competitor in competitors:
            comp_score = 0
            # Calculate competitor's organic score
            for i, result_domain in enumerate(ownership_data.get("organic_ownership", []), 1):
                if competitor in result_domain:
                    comp_score += ctr_mapping.get(i, 0)
            
            # Add competitor's feature scores
            for feature_type, owners in ownership_data.items():
                if feature_type != "organic_ownership":
                    for owner in owners:
                        if competitor in owner:
                            comp_score += feature_weights.get(feature_type.replace("_owners", ""), 1.0)
            
            competitor_scores.append(comp_score)
        
        total_market_score = total_score + sum(competitor_scores)
        if total_market_score > 0:
            share_of_voice = (total_score / total_market_score) * 100
    
    return {
        "organic_score": organic_score,
        "paa_score": paa_score,
        "feature_score": feature_score,
        "featured_snippet_bonus": featured_snippet_bonus,
        "total_score": total_score,
        "potential_ceiling": potential_ceiling,
        "share_of_voice": share_of_voice,
        "organic_positions": organic_positions,
        "organic_ownership": organic_ownership,
        "feature_counts": feature_counts,
        "ownership_data": serp_data.get("ownership_data", {})
    }
    paa_questions = serp_data.get("paa_questions", [])
    paa_score = paa_weight * len(paa_questions)
    
    # Feature score (weighted by feature type)
    feature_counts = serp_data.get("feature_counts", {})
    feature_weights = {
        "video": 0.15,  # High engagement
        "image": 0.10,  # Visual appeal
        "news": 0.08,   # Timeliness
        "shopping": 0.12, # Commercial intent
        "local": 0.06,  # Local intent
    }
    
    feature_score = sum(
        feature_weights.get(feature, 0.05) * count 
        for feature, count in feature_counts.items()
    )
    
    # Featured snippet bonus
    featured_snippet_bonus = 0.20 if serp_data.get("has_featured_snippet") else 0
    
    total_score = organic_score + paa_score + feature_score + featured_snippet_bonus
    
    return {
        "organic_score": organic_score,
        "paa_score": paa_score,
        "feature_score": feature_score,
        "featured_snippet_bonus": featured_snippet_bonus,
        "total_score": total_score,
        "organic_positions": organic_positions,
        "num_organic": serp_data.get("num_organic", 0),
        "num_paa": len(paa_questions),
        "has_featured_snippet": serp_data.get("has_featured_snippet", False),
        "feature_counts": feature_counts
    }

def fetch_and_score_keywords(
    keywords: List[str], 
    location: str = "us",
    domain: Optional[str] = None,
    api_key: Optional[str] = None,
    device: str = "desktop",
    project_id: int = None,
    competitors: List[str] = None
) -> pd.DataFrame:
    """Fetch SERPs for keywords and compute real-estate scores.
    
    Returns:
        DataFrame with keyword, scores, and feature counts
    """
    results = []
    ctr_df = fetch_ctr_data()
    
    for keyword in keywords:
        try:
            # Check for cached data first
            cached_data = None
            if project_id:
                cached_data = get_cached_serp_data(project_id, keyword, hours=6)
            
            if cached_data:
                serp_json = cached_data
                logger.info(f"Using cached SERP data for '{keyword}'")
            else:
                # Fetch SERP with enhanced parameters
                serp_json = fetch_serp(keyword, location, api_key, device, num_results=100)
                if not serp_json:
                    continue
            
            # Parse features with ownership data
            serp_data = parse_serp_features(serp_json)
            
            # Compute comprehensive scores
            score_data = compute_serp_real_estate_score(serp_data, ctr_df, domain, competitors)
            
            # Save to database if project_id provided
            if project_id:
                try:
                    save_serp_run(
                        project_id=project_id,
                        keyword=keyword,
                        parsed=serp_data,
                        scores=score_data,
                        raw_json=json.dumps(serp_json),
                        device=device
                    )
                except Exception as e:
                    logger.error(f"Failed to save SERP run for '{keyword}': {e}")
            
            # Create result row
            result = {
                "keyword": keyword,
                "total_score": score_data["total_score"],
                "organic_score": score_data["organic_score"],
                "paa_score": score_data["paa_score"],
                "feature_score": score_data["feature_score"],
                "featured_snippet_bonus": score_data["featured_snippet_bonus"],
                "share_of_voice": score_data["share_of_voice"],
                "potential_ceiling": score_data["potential_ceiling"],
                "num_organic": serp_data["num_organic"],
                "num_paa": len(serp_data["paa_questions"]),
                "has_featured_snippet": serp_data["has_featured_snippet"],
                "organic_positions": str(score_data["organic_positions"]),
                "feature_counts": str(serp_data["feature_counts"]),
                "ownership_data": str(score_data["ownership_data"]),
                "timestamp": datetime.now()
            }
            results.append(result)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to process keyword '{keyword}': {e}")
            continue
    
    return pd.DataFrame(results)

def schedule_serp_fetch_job(project_id: int, keywords: List[str], device: str = "desktop"):
    """Schedule a background SERP fetch job."""
    try:
        # Get project details
        db = get_db()
        project = db.query(Project).filter(Project.id == project_id).first()
        competitors = get_competitors(project_id)
        competitor_domains = [comp['domain'] for comp in competitors]
        db.close()
        
        # Schedule the job
        job_id = f"serp_fetch_{project_id}_{int(time.time())}"
        scheduler.add_job(
            func=fetch_and_score_keywords,
            trigger="date",
            args=[keywords, project.location, project.domain, None, device, project_id, competitor_domains],
            id=job_id,
            replace_existing=True
        )
        
        logger.info(f"Scheduled SERP fetch job {job_id} for {len(keywords)} keywords")
        return job_id
        
    except Exception as e:
        logger.error(f"Failed to schedule SERP fetch job: {e}")
        return None

def get_cached_serp_data(project_id: int, keyword: str, hours: int = 6) -> Optional[Dict]:
    """Get cached SERP data if available within specified hours."""
    db = get_db()
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        serp_run = db.query(SerpRun).join(Keyword).filter(
            SerpRun.project_id == project_id,
            Keyword.keyword == keyword,
            SerpRun.timestamp >= cutoff_time
        ).order_by(SerpRun.timestamp.desc()).first()
        
        if serp_run and serp_run.raw_json:
            return json.loads(serp_run.raw_json)
        return None
        
    finally:
        db.close()

###############################################################################
# Helper functions                                                              
###############################################################################

def fetch_ctr_data() -> pd.DataFrame:
    """Fetch latest Google Organic CTR benchmarks from Advanced Web Ranking.

    Returns a DataFrame with columns: position (int), average_ctr (float)
    """
    # Known unofficial JSON endpoint (falls back to static mapping if unavailable)
    endpoints = [
        # This URL is not official but observed in network calls. It may change.
        "https://www.advancedwebranking.com/ctr/api/organic/positions.json",
        # Legacy GitHub gist mirror (community maintained)
        "https://raw.githubusercontent.com/ghmagazine/seo-ctr-benchmarks/master/ctr.json",
    ]

    for url in endpoints:
        try:
            resp = requests.get(url, timeout=10)
            if resp.ok:
                pass
                data = resp.json()
                # Expecting list/dict mapping position -> ctr (percentage or decimal)
                # Normalise
                ctr_records = []
                if isinstance(data, dict):
                    pass
                    iterable = data.items()
                else:
                    # assume list of dicts with position/ctr keys
                    iterable = ((str(item.get("position")), item.get("ctr")) for item in data)
                for pos, ctr in iterable:
                    try:
                        position = int(pos)
                        ctr_val = float(ctr)
                        # Ensure decimal between 0 and 1
                        if ctr_val > 1:
                            pass
                            ctr_val /= 100.0
                        ctr_records.append({"position": position, "average_ctr": ctr_val})
                    except (ValueError, TypeError):
                        continue
                if ctr_records:
                    pass
                    return pd.DataFrame(ctr_records).sort_values("position")
        except Exception:
            continue

    # Fallback industry-agnostic curve (AWR Q1 2025 Desktop International, approx.)
    fallback_curve = {
        1: 0.3110,
        2: 0.2410,
        3: 0.1800,
        4: 0.1320,
        5: 0.0990,
        6: 0.0760,
        7: 0.0610,
        8: 0.0500,
        9: 0.0430,
        10: 0.0390,
    }
    st.warning("Unable to fetch live CTR benchmarks – using fallback dataset.")
    return pd.DataFrame(
        {"position": list(fallback_curve.keys()), "average_ctr": list(fallback_curve.values())}
    )


def detect_source(df: pd.DataFrame) -> str:
    """Detect file source based on columns present."""
    cols = df.columns.str.lower()
    if any("previous position" in c for c in cols):
        pass
        return "semrush"
    if any(c.startswith("unnamed") for c in cols):
        pass
        return "surplex"
    return "unknown"


def split_to_list(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .fillna("")
        .apply(lambda x: [s.strip() for s in x.split(",") if s.strip()])
    )


def normalize_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize raw DataFrame from a given source into the unified schema."""

    norm = pd.DataFrame()
    
    if source == "url_structuur_kwr":
    
        pass
        # URL Structuur KWR format is already normalized
        norm = df.copy()
        # Add missing columns with default values
        norm["position"] = pd.NA  # URL Structuur doesn't have positions
        norm["previous_position"] = pd.NA
        norm["serp_features"] = ""
        norm["intents"] = ""
        norm["position_type"] = pd.NA
        norm["timestamp"] = pd.NA
        norm["traffic"] = pd.NA
        norm["traffic_pct"] = pd.NA
        norm["ctr_benchmark"] = pd.NA
    elif source in ["auto_detected", "manual_mapped"]:
        # For auto-detected or manually mapped data, columns are already standardized
        norm["keyword"] = df.get("keyword", pd.NA)
        norm["search_volume"] = pd.to_numeric(df.get("search_volume", pd.NA), errors="coerce")
        norm["position"] = pd.to_numeric(df.get("position", pd.NA), errors="coerce")
        norm["serp_features"] = split_to_list(df.get("serp_features", ""))
        norm["intents"] = split_to_list(df.get("intents", ""))
        norm["position_type"] = df.get("position_type", pd.NA)
        norm["timestamp"] = pd.to_datetime(df.get("timestamp", pd.NA), errors="coerce")
        norm["previous_position"] = pd.to_numeric(df.get("previous_position", pd.NA), errors="coerce")
        norm["url"] = df.get("url", pd.NA)
        norm["traffic"] = pd.to_numeric(df.get("traffic", pd.NA), errors="coerce")
        norm["traffic_pct"] = pd.to_numeric(df.get("traffic_pct", pd.NA), errors="coerce")
    elif source == "surplex":
        norm["keyword"] = df.get("Keywords")
        # Total Search Volume may live under an Unnamed col
        sv_col = next((c for c in df.columns if "total search volume" in c.lower()), None)
        if not sv_col:
            pass
            sv_col = next((c for c in df.columns if "volume" in c.lower()), None)
        norm["search_volume"] = pd.to_numeric(df.get(sv_col, pd.NA), errors="coerce")
        norm["position"] = pd.to_numeric(df.get("Position"), errors="coerce")
        norm["serp_features"] = split_to_list(df.get("SERP Features by Keyword"))
        norm["intents"] = split_to_list(df.get("Keyword Intents"))
        norm["position_type"] = df.get("Position Type")
        norm["timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
        norm["previous_position"] = pd.NA
        norm["url"] = pd.NA
        norm["traffic"] = pd.NA
        norm["traffic_pct"] = pd.NA
    elif source == "semrush":
        norm["keyword"] = df.get("Keyword")
        norm["search_volume"] = pd.to_numeric(df.get("Search Volume"), errors="coerce")
        norm["position"] = pd.to_numeric(df.get("Position"), errors="coerce")
        norm["previous_position"] = pd.to_numeric(df.get("Previous position"), errors="coerce")
        norm["url"] = df.get("URL")
        norm["traffic"] = pd.to_numeric(df.get("Traffic"), errors="coerce")
        norm["traffic_pct"] = pd.to_numeric(df.get("Traffic (%)"), errors="coerce")
        norm["serp_features"] = split_to_list(df.get("SERP Features by Keyword"))
        norm["intents"] = split_to_list(df.get("Keyword Intents"))
        norm["position_type"] = df.get("Position Type")
        norm["timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
    else:
        st.error("Unknown source format – unable to normalize.")
        return pd.DataFrame()

    # Add missing columns with default values if not already present
    for field in ["position", "previous_position", "serp_features", "intents", "position_type", 
                  "timestamp", "url", "traffic", "traffic_pct", "ctr_benchmark"]:
        if field not in norm.columns:
            pass
            if field in ["position", "previous_position", "traffic", "traffic_pct", "ctr_benchmark"]:
                pass
                norm[field] = pd.NA
            elif field in ["serp_features", "intents"]:
                norm[field] = ""
            else:
                norm[field] = pd.NA

    # Ensure 'answered' column exists for downstream aggregations
    if "answered" not in norm.columns:
        if "answer" in norm.columns:
            norm["answered"] = norm["answer"].apply(
                lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
            )
        else:
            norm["answered"] = 0
    # Ensure correct dtypes
    return norm


# Expected field mappings for flexible column detection
EXPECTED_FIELDS = {
    "keyword": ["keyword", "keywords", "kw", "search term", "search query", "term"],
    "search_volume": ["search volume", "volume", "sv", "search_volume", "searchvolume", "search_vol"],
    "position": ["position", "pos", "rank", "ranking", "current position", "current rank"],
    "serp_features": ["serp features", "features by keyword", "features", "serp_features", "serp features by keyword"],
    "intents": ["intent", "keyword intents", "intents", "keyword_intents", "search intent"],
    "position_type": ["position type", "type", "position_type", "rank type"],
    "timestamp": ["timestamp", "date", "crawl date", "crawl_date", "last updated", "last_updated"],
    "previous_position": ["previous position", "previous_position", "prev position", "prev_position", "previous rank"],
    "url": ["url", "landing page", "landing_page", "page", "current url"],
    "traffic": ["traffic", "organic traffic", "organic_traffic", "traffic value"],
    "traffic_pct": ["traffic (%)", "traffic_pct", "traffic %", "traffic_percentage"]
}

def fuzzy_match_column(target_field: str, available_columns: List[str], threshold: float = 0.8) -> Optional[str]:
    """Find the best matching column for a target field using fuzzy matching."""
    if not available_columns:
        pass
        return None
    
    # Convert to lowercase for comparison
    target_lower = target_field.lower()
    available_lower = [col.lower() for col in available_columns]
    
    # Try exact match first
    if target_lower in available_lower:
        pass
        return available_columns[available_lower.index(target_lower)]
    
    # Try fuzzy matching with difflib
    matches = difflib.get_close_matches(target_lower, available_lower, n=1, cutoff=threshold)
    if matches:
        pass
        matched_lower = matches[0]
        return available_columns[available_lower.index(matched_lower)]
    
    return None

def detect_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Detect column mapping using fuzzy matching."""
    mapping = {}
    available_columns = df.columns.tolist()
    
    for field, synonyms in EXPECTED_FIELDS.items():
        matched_column = None
        
        # Try each synonym
        for synonym in synonyms:
            matched_column = fuzzy_match_column(synonym, available_columns)
            if matched_column:
                pass
                logger.info(f"Mapped '{field}' to column '{matched_column}' (synonym: '{synonym}')")
                break
        
        mapping[field] = matched_column
    
    return mapping

def validate_mapping(mapping: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
    """Validate if the mapping covers all required fields."""
    required_fields = ["keyword", "position", "search_volume"]
    missing_fields = [field for field in required_fields if not mapping.get(field)]
    
    is_valid = len(missing_fields) == 0
    return is_valid, missing_fields

def parse_url_structuur_kwr(file) -> Optional[pd.DataFrame]:
    """
    Parse URL Structuur KWR format from Excel file.
    Returns normalized DataFrame with columns: url, keyword, keyword_type, search_volume, cumulative_volume
    """
    try:
        # Read the "URL Structuur" sheet
        df_raw = pd.read_excel(file, sheet_name="URL Structuur", engine="openpyxl")
        
        if df_raw.empty:
        
            pass
            st.error("URL Structuur sheet is empty.")
            return None
        
        # Initialize list to store all keyword rows
        all_keywords = []
        
        for idx, row in df_raw.iterrows():
            url = row.get("URL", "")
            if not url or pd.isna(url):
                pass
                continue
            
            # Primary keyword
            primary_keyword = row.get("Primair zoekwoord", "")
            primary_volume = row.get("Zoekvolume", 0)
            cumulative_volume = row.get("Cum. Zoekvolume", 0)
            
            if primary_keyword and not pd.isna(primary_keyword):
            
                pass
                all_keywords.append({
                    "url": url,
                    "keyword": primary_keyword,
                    "keyword_type": "primary",
                    "search_volume": primary_volume,
                    "cumulative_volume": cumulative_volume
                })
            
            # Secondary keyword
            secondary_keyword = row.get("Secundair zoekwoord", "")
            secondary_volume = row.get("Zoekvolume.1", 0)
            
            if secondary_keyword and not pd.isna(secondary_keyword):
            
                pass
                all_keywords.append({
                    "url": url,
                    "keyword": secondary_keyword,
                    "keyword_type": "secondary",
                    "search_volume": secondary_volume,
                    "cumulative_volume": cumulative_volume
                })
            
            # Tertiary keyword
            tertiary_keyword = row.get("Tertiair zoekwoord", "")
            tertiary_volume = row.get("Zoekvolume.2", 0)
            
            if tertiary_keyword and not pd.isna(tertiary_keyword):
            
                pass
                all_keywords.append({
                    "url": url,
                    "keyword": tertiary_keyword,
                    "keyword_type": "tertiary",
                    "search_volume": tertiary_volume,
                    "cumulative_volume": cumulative_volume
                })
            
            # Other keywords (comma-separated)
            other_keywords_str = row.get("Overige zoekwoorden", "")
            other_volume = row.get("Zoekvolume.3", 0)
            
            if other_keywords_str and not pd.isna(other_keywords_str):
            
                pass
                other_keywords = [kw.strip() for kw in str(other_keywords_str).split(",") if kw.strip()]
                for keyword in other_keywords:
                    all_keywords.append({
                        "url": url,
                        "keyword": keyword,
                        "keyword_type": "other",
                        "search_volume": other_volume,
                        "cumulative_volume": cumulative_volume
                    })
        
        if not all_keywords:
        
            pass
            st.error("No valid keywords found in URL Structuur sheet.")
            return None
        
        # Create DataFrame
        result_df = pd.DataFrame(all_keywords)
        logger.info(f"Parsed {len(result_df)} keywords from URL Structuur KWR format")
        return result_df
        
    except Exception as e:
        st.error(f"Failed to parse URL Structuur KWR format: {e}")
        logger.error(f"Error parsing URL Structuur KWR: {e}")
        return None

def detect_file_format(file) -> str:
    """
    Detect file format: 'url_structuur_kwr', 'excel_multi_sheet', or 'single_sheet'
    """
    try:
        file_extension = file.name.lower()
        
        if file_extension.endswith('.csv'):
        
            pass
            return 'single_sheet'
        
        if file_extension.endswith(('.xlsx', '.xls')):
        
            pass
            # Check if it's an Excel file with multiple sheets
            excel_file = pd.ExcelFile(file, engine="openpyxl")
            sheet_names = [name.lower() for name in excel_file.sheet_names]
            
            # Check for URL Structuur sheet
            if "url structuur" in sheet_names:
                pass
                return 'url_structuur_kwr'
            elif len(sheet_names) > 1:
                return 'excel_multi_sheet'
            else:
                return 'single_sheet'
        
        return 'single_sheet'
        
    except Exception as e:
        logger.error(f"Error detecting file format: {e}")
        return 'single_sheet'

def load_excel(file) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    """
    Enhanced file loader that handles:
    1. URL Structuur KWR format (special parsing)
    2. Multi-sheet Excel files (user selection)
    3. Single-sheet files with fuzzy column detection
    4. Interactive column mapping fallback
    
    Returns the raw DataFrame plus the final mapping dict.
    """
    try:
        # Detect file format
        file_format = detect_file_format(file)
        
        if file_format == 'url_structuur_kwr':
        
            pass
            # Special handling for URL Structuur KWR format
            logger.info(f"Detected URL Structuur KWR format for {file.name}")
            df = parse_url_structuur_kwr(file)
            if df is not None:
                pass
                # For URL Structuur, we return a special mapping indicating it's already normalized
                return df, {"format": "url_structuur_kwr"}
            else:
                return None, {}
        
        elif file_format == 'excel_multi_sheet':
            # Handle multi-sheet Excel files
            excel_file = pd.ExcelFile(file, engine="openpyxl")
            sheet_names = excel_file.sheet_names
            
            # Let user select sheet
            selected_sheet = st.selectbox(
                "Select sheet to load:",
                sheet_names,
                index=0,
                key=f"sheet_select_{file.name}"
            )
            
            df = pd.read_excel(file, sheet_name=selected_sheet, engine="openpyxl")
            
        else:
            # Single sheet file (CSV or single-sheet Excel)
            file_extension = file.name.lower()
            
            if file_extension.endswith('.csv'):
            
                pass
                df = pd.read_csv(file)
            elif file_extension.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, engine="openpyxl")
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None, {}
        
        if df.empty:
        
            pass
            st.error("The uploaded file is empty.")
            return None, {}
        
        # Check if we have a saved mapping for this file type
        file_type_key = f"mapping_{file.name}_{df.shape[1]}"
        saved_mapping = st.session_state.get(file_type_key, None)
        
        if saved_mapping:
        
            pass
            # Try the saved mapping first
            is_valid, missing_fields = validate_mapping(saved_mapping)
            if is_valid:
                pass
                logger.info(f"Using saved column mapping for {file.name}")
                return df, saved_mapping
        
        # Auto-detect column mapping
        mapping = detect_column_mapping(df)
        
        # Validate mapping
        is_valid, missing_fields = validate_mapping(mapping)
        
        if is_valid:
        
            pass
            logger.info(f"Auto-detected column mapping for {file.name}")
            # Save successful mapping
            st.session_state[file_type_key] = mapping
            return df, mapping
        else:
            logger.warning(f"Auto-detection failed for {file.name}. Missing fields: {missing_fields}")
            return df, mapping
            
    except Exception as e:
        st.error(f"Failed to read {file.name}: {e}")
        logger.error(f"Error reading file {file.name}: {e}")
        return None, {}

def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Apply the column mapping to rename columns to standard names."""
    # Create reverse mapping (new_name -> old_name)
    reverse_mapping = {v: k for k, v in mapping.items() if v is not None}
    
    # Rename columns
    df_renamed = df.rename(columns=reverse_mapping)
    
    # Add missing columns with default values
    for field in EXPECTED_FIELDS.keys():
        if field not in df_renamed.columns:
            pass
            if field in ["search_volume", "position", "previous_position", "traffic", "traffic_pct"]:
                pass
                df_renamed[field] = pd.NA
            elif field in ["serp_features", "intents"]:
                df_renamed[field] = ""
            else:
                df_renamed[field] = pd.NA
    
    return df_renamed


def compute_serp_score(df: pd.DataFrame, ctr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute SERP real-estate score and assign ctr_benchmark."""
    # Merge ctr benchmarks
    ctr_mapping = ctr_df.set_index("position")["average_ctr"].to_dict()
    df["ctr_benchmark"] = df["position"].map(ctr_mapping)
    # number of extra features (beyond organic listing). Count features list length.
    df["serp_feature_count"] = df["serp_features"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["serp_score"] = (1 + df["serp_feature_count"]) * df["ctr_benchmark"]
    return df

###############################################################################
# Research & Clustering Functions                                               
###############################################################################

def load_embedding_model():
    """Load the sentence transformer model for embeddings with memory optimization."""
    global embedding_model, _model_cache
    
    # Check if model is already cached
    cache_key = 'all-MiniLM-L6-v2'
    if cache_key in _model_cache:
        embedding_model = _model_cache[cache_key]
        return embedding_model
    
    if embedding_model is None:
        try:
            import torch
            import time
            import gc
            
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Determine device with memory constraints
            device = 'cpu'  # Default to CPU for memory safety
            if torch.cuda.is_available():
                # Check available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory > 4e9:  # 4GB minimum for GPU
                    device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            
            # Retry logic for HuggingFace rate limiting with better caching
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Set up cache directory to avoid repeated downloads
                    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Load model with memory optimization and caching
                    embedding_model = SentenceTransformer(
                        'all-MiniLM-L6-v2', 
                        device=device,
                        cache_folder=cache_dir
                    )
                    
                    # Set deterministic seeds for reproducible results
                    np.random.seed(42)
                    random.seed(42)
                    torch.manual_seed(42)
                    
                    logger.info(f"Loaded embedding model successfully with deterministic seeding on {device}")
                    # Cache the model
                    _model_cache[cache_key] = embedding_model
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 15  # Longer exponential backoff
                        logger.warning(f"HuggingFace rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    elif "CUDA out of memory" in str(e) or "MPS" in str(e):
                        # Fallback to CPU if GPU/MPS memory issues
                        logger.warning(f"GPU/MPS memory issue, falling back to CPU: {e}")
                        device = 'cpu'
                        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device, cache_folder=cache_dir)
                        logger.info("Loaded embedding model on CPU as fallback")
                        # Cache the model
                        _model_cache[cache_key] = embedding_model
                        break
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            st.error(f"Failed to load embedding model: {e}. Please check your internet connection and try again.")
            return None
    return embedding_model

def fetch_paa_qna_from_stored_serp(keyword: str, project_id: int) -> List[Dict]:
    """Extract PAA Q&A for a keyword from stored SERP data with enhanced metadata.
    
    Args:
        keyword: Search query
        project_id: Project ID to search in
    
    Returns:
        List of dicts with enhanced Q&A data
    """
    try:
        db = get_db()
        
        # Get project domain for site coverage detection
        project = db.query(Project).filter(Project.id == project_id).first()
        project_domain = project.domain if project else None
        
        # Get search volume for this keyword
        search_volume = get_keyword_search_volume(keyword, project_id)
        
        # Get the most recent SERP run for this keyword in this project
        serp_run = db.query(SerpRun).join(Keyword).filter(
            SerpRun.project_id == project_id,
            Keyword.keyword == keyword
        ).order_by(SerpRun.timestamp.desc()).first()
        
        if not serp_run or not serp_run.raw_json:
            logger.warning(f"No stored SERP data found for keyword '{keyword}' in project {project_id}")
            return []
        
        serp_json = json.loads(serp_run.raw_json)
        
        # Parse the SERP JSON to get the normalized paa_questions structure
        parsed_data = parse_serp_features(serp_json)
        paa_questions = parsed_data.get("paa_questions", [])
        
        qna_list = []
        
        for qa in paa_questions:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            source_url = qa.get("source_url", "")
            
            if question and answer:
                # Improved domain matching - handle both www and non-www
                answered_by_site = False
                if project_domain and source_url:
                    source_domain = extract_domain(source_url)
                    # Normalize domains for comparison (remove www)
                    normalized_project_domain = project_domain.replace("www.", "")
                    normalized_source_domain = source_domain.replace("www.", "")
                    answered_by_site = normalized_project_domain == normalized_source_domain
                
                # Compute answer relevancy
                answer_relevancy = compute_answer_relevancy(question, answer)
                
                qna_data = {
                    "keyword": keyword,
                    "question": question,
                    "answer": answer,
                    "search_volume": search_volume,
                    "answered_by_site": answered_by_site,
                    "answer_relevancy": answer_relevancy,
                    "source_url": source_url
                }
                
                qna_list.append(qna_data)
                
                # Save to database with search_volume from get_keyword_search_volume()
                try:
                    qna_record = QnaRecord(
                        project_id=project_id,
                        keyword=keyword,
                        question=question,
                        answer=answer,
                        search_volume=search_volume,  # From get_keyword_search_volume()
                        answered_by_site=answered_by_site,
                        answer_relevancy=answer_relevancy,
                        source_url=source_url
                    )
                    db.add(qna_record)
                    db.commit()
                except Exception as e:
                    logger.error(f"Failed to save Q&A record: {e}")
                    db.rollback()
        
        logger.info(f"Extracted {len(qna_list)} Q&A pairs for keyword '{keyword}' from stored SERP data")
        return qna_list
        
    except Exception as e:
        logger.error(f"Failed to extract Q&A for '{keyword}': {e}")
        return []
    finally:
        try:
            db.close()
        except:
            pass

def fetch_paa_qna(keyword: str, location: str = "us") -> List[Dict]:
    """Fetch PAA Q&A for a keyword (legacy function - makes API calls).
    
    Args:
        keyword: Search query
        location: Country code
    
    Returns:
        List of dicts with "question" and "answer" keys
    """
    serp_json = fetch_serp(keyword, location)
    if not serp_json:
        pass
        return []
    
    paa_questions = serp_json.get("related_questions", [])
    qna_list = []
    
    for qa in paa_questions:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        if question and answer:
            pass
            qna_list.append({
                "keyword": keyword,
                "question": question,
                "answer": answer
            })
    
    logger.info(f"Extracted {len(qna_list)} Q&A pairs for keyword '{keyword}'")
    return qna_list

def process_keyword_parallel(keyword: str, project_id: int) -> List[Dict]:
    """Process a single keyword to extract PAA Q&A from stored SERP data.
    
    Args:
        keyword: The keyword to process
        project_id: Project ID to search in
    
    Returns:
        List of Q&A dictionaries for this keyword
    """
    try:
        qna_list = fetch_paa_qna_from_stored_serp(keyword, project_id)
        return qna_list
    except Exception as e:
        logger.error(f"Failed to extract Q&A for '{keyword}': {e}")
        return []

def fetch_all_qna_parallel(keywords: List[str], project_id: int, max_workers: int = 3) -> pd.DataFrame:
    """Extract Q&A for multiple keywords from stored SERP data using parallel processing with memory optimization.
    
    Args:
        keywords: List of keywords to analyze
        project_id: Project ID to search in
        max_workers: Maximum number of parallel workers
    
    Returns:
        DataFrame with keyword, question, answer columns
    """
    all_qna = []
    
    # Reduce max_workers to prevent memory issues
    max_workers = min(max_workers, 2)  # Limit to 2 workers for memory safety
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all keyword processing tasks
            future_to_keyword = {executor.submit(process_keyword_parallel, kw, project_id): kw for kw in keywords}
            
            # Process completed tasks
            for future in as_completed(future_to_keyword):
                keyword = future_to_keyword[future]
                try:
                    qna_list = future.result()
                    all_qna.extend(qna_list)
                except Exception as e:
                    logger.error(f"Exception occurred while processing '{keyword}': {e}")
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
    finally:
        # Force garbage collection to prevent resource leaks
        import gc
        gc.collect()
    
    if all_qna:
        df = pd.DataFrame(all_qna)
        logger.info(f"Extracted {len(df)} total Q&A pairs for {len(keywords)} keywords from stored SERP data")
        return df
    else:
        return pd.DataFrame(columns=["keyword", "question", "answer"])

def fetch_all_qna(keywords: List[str], location: str = "us") -> pd.DataFrame:
    """Fetch Q&A for multiple keywords (legacy function for backward compatibility).
    
    Args:
        keywords: List of keywords to analyze
        location: Country code
    
    Returns:
        DataFrame with keyword, question, answer columns
    """
    return fetch_all_qna_parallel(keywords, location)

def score_relevancy(seed: str, question: str) -> float:
    """Compute cosine similarity between seed keyword and question.
    
    Args:
        seed: Seed keyword
        question: Question text
    
    Returns:
        Similarity score between 0 and 1
    """
    model = load_embedding_model()
    if model is None:
        pass
        return 0.0
    
    try:
        # Encode both texts
        embeddings = model.encode([seed, question])
        
        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    except Exception as e:
        logger.error(f"Failed to compute relevancy score: {e}")
        return 0.0

def compute_answer_relevancy(question: str, answer: str) -> float:
    """Compute cosine similarity between question and answer embeddings.
    
    Args:
        question: Question text
        answer: Answer text
    
    Returns:
        Similarity score between 0 and 1
    """
    model = load_embedding_model()
    if model is None:
        return 0.0
    
    try:
        # Encode both texts
        embeddings = model.encode([question, answer])
        
        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    except Exception as e:
        logger.error(f"Failed to compute answer relevancy: {e}")
        return 0.0

def detect_site_coverage(source_url: str, project_domain: str) -> bool:
    """Check if the answer source URL matches the project domain.
    
    Args:
        source_url: URL where the answer was found
        project_domain: Project's target domain
    
    Returns:
        True if the answer is from the project's site
    """
    if not source_url or not project_domain:
        return False
    
    try:
        # Parse URLs
        source_domain = urlparse(source_url).netloc.lower()
        target_domain = urlparse(project_domain).netloc.lower() if '://' in project_domain else project_domain.lower()
        
        # Check if domains match (handle www and non-www)
        source_clean = source_domain.replace('www.', '')
        target_clean = target_domain.replace('www.', '')
        
        return source_clean == target_clean
    except Exception as e:
        logger.error(f"Failed to detect site coverage: {e}")
        return False

def get_keyword_search_volume(keyword: str, project_id: int) -> int:
    """Get search volume for a keyword from the database.
    
    Args:
        keyword: The keyword to look up
        project_id: Project ID
    
    Returns:
        Search volume (0 if not found)
    """
    try:
        db = get_db()
        keyword_record = db.query(Keyword).filter(
            Keyword.project_id == project_id,
            Keyword.keyword == keyword
        ).first()
        
        return keyword_record.search_volume if keyword_record else 0
    except Exception as e:
        logger.error(f"Failed to get search volume for '{keyword}': {e}")
        return 0
    finally:
        try:
            db.close()
        except:
            pass

@st.cache_data
def build_seed_aggregation(qna_df: pd.DataFrame) -> pd.DataFrame:
    """Build seed-level aggregation for visualizations.
    
    Args:
        qna_df: DataFrame with Q&A data including keyword, answered_by_site, search_volume
    
    Returns:
        DataFrame with seed-level aggregations
    """
    # Ensure we have the required columns
    if 'keyword' not in qna_df.columns:
        qna_df['seed_keyword'] = qna_df['keyword'] if 'keyword' in qna_df.columns else qna_df.index
    else:
        qna_df['seed_keyword'] = qna_df['keyword']
    
    # Ensure answered_by_site column exists
    if 'answered_by_site' not in qna_df.columns:
        if 'answer' in qna_df.columns:
            qna_df['answered_by_site'] = qna_df['answer'].apply(
                lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
            )
        else:
            qna_df['answered_by_site'] = 0
    
    # Ensure search_volume column exists and is numeric
    if 'search_volume' not in qna_df.columns:
        qna_df['search_volume'] = 0
    else:
        # Convert search_volume to numeric, filling non-numeric values with 0
        qna_df['search_volume'] = pd.to_numeric(qna_df['search_volume'], errors='coerce').fillna(0)
    
    # Build seed aggregation
    if "answered_by_site" in qna_df.columns:
        seed_agg = (
            qna_df.groupby("seed_keyword")
            .agg(
                total_qas=("question", "count"),
                answered_qas=("answered_by_site", "sum"),
                total_volume=("search_volume", "sum")
            )
            .reset_index()
        )
    else:
        # Fallback if answered_by_site column doesn't exist
        seed_agg = (
            qna_df.groupby("seed_keyword")
            .agg(
                total_qas=("question", "count"),
                answered_qas=("question", "count"),  # Use question count as fallback
                total_volume=("search_volume", "sum")
            )
            .reset_index()
        )
    
    # Calculate percentage answered
    seed_agg["pct_answered"] = seed_agg["answered_qas"] / seed_agg["total_qas"]
    
    # Define coverage category
    def coverage_cat(p):
        if p == 0:
            return "red"
        elif p > 0.5:
            return "green"
        else:
            return "orange"
    
    seed_agg["coverage_color"] = seed_agg["pct_answered"].apply(coverage_cat)
    seed_agg["coverage_str"] = seed_agg["coverage_color"].map({"red": "None", "orange": "Partial", "green": "Full"})
    
    # Ensure total_volume is numeric for nlargest operations
    seed_agg["total_volume"] = pd.to_numeric(seed_agg["total_volume"], errors='coerce').fillna(0)
    
    return seed_agg

@st.cache_data(show_spinner=False)
def process_research_cached(project_id: int, keywords: List[str], threshold: float = 0.7, max_workers: int = 3) -> Tuple[pd.DataFrame, str]:
    """Process research data with caching and duplicate detection using stored SERP data.
    
    Args:
        project_id: Project ID for caching key and data source
        keywords: List of keywords to process
        threshold: Relevancy threshold
        max_workers: Maximum parallel workers
    
    Returns:
        Tuple of (processed DataFrame, cache timestamp)
    """
    # Get latest SERP run timestamp for cache key
    db = get_db()
    latest_run = db.query(SerpRun).filter(
        SerpRun.project_id == project_id
    ).order_by(SerpRun.timestamp.desc()).first()
    db.close()
    
    cache_timestamp = latest_run.timestamp.isoformat() if latest_run else datetime.now().isoformat()
    
    # Process keywords in parallel
    all_qna = []
    existing_pairs = set()  # Track existing keyword-question pairs
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all keyword processing tasks
        future_to_keyword = {executor.submit(process_keyword_parallel, kw, project_id): kw for kw in keywords}
        
        # Process completed tasks with duplicate detection
        for future in as_completed(future_to_keyword):
            keyword = future_to_keyword[future]
            try:
                qna_list = future.result()
                
                # Check for duplicates and update existing records
                for qna in qna_list:
                    pair_key = (keyword, qna['question'])
                    if pair_key not in existing_pairs:
                        existing_pairs.add(pair_key)
                        all_qna.append(qna)
                    else:
                        # Update existing record if answer changed
                        for existing_qna in all_qna:
                            if existing_qna['keyword'] == keyword and existing_qna['question'] == qna['question']:
                                if existing_qna['answer'] != qna['answer']:
                                    existing_qna['answer'] = qna['answer']
                                    existing_qna['timestamp'] = datetime.now().isoformat()
                                break
                        
            except Exception as e:
                logger.error(f"Exception occurred while processing '{keyword}': {e}")
    
    if not all_qna:
        return pd.DataFrame(columns=["keyword", "question", "answer"]), cache_timestamp
    
    # Create DataFrame
    df = pd.DataFrame(all_qna)
    
    # Compute relevancy scores in parallel
    relevancy_scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(score_relevancy, row["keyword"], row["question"]): i 
            for i, row in df.iterrows()
        }
        
        # Collect results in order
        temp_scores = [None] * len(df)
        for future in as_completed(future_to_row):
            row_idx = future_to_row[future]
            try:
                score = future.result()
                temp_scores[row_idx] = score
            except Exception as e:
                logger.error(f"Failed to compute relevancy score: {e}")
                temp_scores[row_idx] = 0.0
        
        relevancy_scores = temp_scores
    
    df["relevancy_score"] = relevancy_scores
    # NEW: Ensure binary 'answered' column exists for downstream aggregations
    if "answered" not in df.columns:
        df["answered"] = df["answer"].apply(
            lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
        )
    
    # Filter by threshold
    if "relevancy_score" in df.columns:
        df_filtered = df[df["relevancy_score"] >= threshold].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        logger.warning("No questions passed the relevancy threshold")
        return df_filtered, cache_timestamp
    
    # Run clustering
    try:
        # Load embedding model
        model = load_embedding_model()
        if model is None:
            df_filtered["cluster_id"] = -1
            return df_filtered, cache_timestamp
        
        # Encode all questions
        questions = df_filtered["question"].tolist()
        embeddings = model.encode(questions)
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
        df_filtered["cluster_id"] = cluster_labels
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")
        
        return df_filtered, cache_timestamp
        
    except Exception as e:
        logger.error(f"Failed to cluster questions: {e}")
        # Fallback: assign sequential cluster IDs based on keyword similarity
        try:
            logger.info("Using fallback clustering method")
            df_filtered["cluster_id"] = range(len(df_filtered))
            return df_filtered, cache_timestamp
        except Exception as fallback_e:
            logger.error(f"Fallback clustering also failed: {fallback_e}")
            df_filtered["cluster_id"] = -1
            return df_filtered, cache_timestamp

def compute_relevancy_and_clusters(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Compute relevancy scores and cluster questions with deterministic results.
    
    Args:
        df: DataFrame with keyword, question, answer columns
        threshold: Relevancy threshold (0.0-1.0)
    
    Returns:
        DataFrame with relevancy_score and cluster_id columns
    """
    if df.empty:
        return df
    
    # Compute relevancy scores
    relevancy_scores = []
    for _, row in df.iterrows():
        score = score_relevancy(row["keyword"], row["question"])
        relevancy_scores.append(score)
    
    df["relevancy_score"] = relevancy_scores
    # NEW: Ensure binary 'answered' column exists for downstream aggregations
    if "answered" not in df.columns:
        df["answered"] = df["answer"].apply(
            lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
        )
    
    # Filter by threshold
    if "relevancy_score" in df.columns:
        df_filtered = df[df["relevancy_score"] >= threshold].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        logger.warning("No questions passed the relevancy threshold")
        return df_filtered
    
    # Load embedding model
    model = load_embedding_model()
    if model is None:
        df_filtered["cluster_id"] = -1
        return df_filtered
    
    try:
        # Encode all questions
        questions = df_filtered["question"].tolist()
        embeddings = model.encode(questions)
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
        df_filtered["cluster_id"] = cluster_labels
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Failed to cluster questions: {e}")
        # Fallback: assign sequential cluster IDs based on keyword similarity
        try:
            logger.info("Using fallback clustering method")
            df_filtered["cluster_id"] = range(len(df_filtered))
            return df_filtered
        except Exception as fallback_e:
            logger.error(f"Fallback clustering also failed: {fallback_e}")
            df_filtered["cluster_id"] = -1
            return df_filtered

def get_serp_details_for_keyword(keyword: str, project_id: int) -> dict:
    """Get SERP details for a specific keyword.
    
    Args:
        keyword: The keyword to get details for
        project_id: The project ID
    
    Returns:
        Dictionary with SERP details
    """
    try:
        db = get_db()
        # Get the latest SERP run for this keyword
        latest_run = db.query(SerpRun).options(
            joinedload(SerpRun.score), 
            joinedload(SerpRun.features)
        ).filter(
            SerpRun.project_id == project_id,
            SerpRun.keyword.has(Keyword.keyword == keyword)
        ).order_by(SerpRun.timestamp.desc()).first()
        
        if not latest_run:
        
            pass
            return {
                "rank": "N/A",
                "serp_score": "N/A",
                "features": [],
                "visible": False,
                "timestamp": "N/A"
            }
        
        # Get rank (position in organic results)
        rank = "N/A"
        serp_score = "N/A"
        features = []
        visible = False
        
        if latest_run.score:
        
            pass
            serp_score = f"{latest_run.score.total_score:.2f}"
        
        # Parse features
        for feature in latest_run.features:
            features.append({
                "type": feature.feature_type,
                "count": feature.count,
                "details": feature.details
            })
        
        # Check if site is visible (you can customize this logic)
        # For now, we'll assume visible if there's a SERP run
        visible = True
        
        return {
            "rank": rank,
            "serp_score": serp_score,
            "features": features,
            "visible": visible,
            "timestamp": latest_run.timestamp.strftime("%Y-%m-%d %H:%M") if latest_run.timestamp else "N/A"
        }
        
    except Exception as e:
        logger.error(f"Error getting SERP details for {keyword}: {e}")
        return {
            "rank": "Error",
            "serp_score": "Error",
            "features": [],
            "visible": False,
            "timestamp": "Error"
        }
    finally:
        db.close()

def build_graph(df: pd.DataFrame) -> str:
    """Build a radial clustering graph with pinned seeds, enhanced physics, and crystal-clear color coding.
    
    Args:
        df: DataFrame with keyword, question, answer, relevancy_score, cluster_id
    
    Returns:
        HTML string for the graph
    """
    if df.empty:
        return "<p>No data to visualize</p>"
    
    # Create network with physics-enabled layout
    net = Network(height="700px", width="100%", bgcolor="#1F2023", font_color="#FFFFFF")
    
    # Group questions by keyword for radial layout
    keyword_groups = df.groupby('keyword')
    seed_ids = []
    
    # Add seed keyword nodes with gradient colors and metadata
    for keyword, group in keyword_groups:
        seed_ids.append(keyword)
        
        # Calculate answered vs unanswered for this keyword
        answered_count = 0
        total_count = len(group)
        
        for _, row in group.iterrows():
            answer_text = row.get("answer", "").strip()
            if len(answer_text) > 10 and not answer_text.lower().startswith(("no answer", "not found", "n/a")):
                answered_count += 1
        
        # Add seed keyword node with gradient color
        net.add_node(
            keyword, 
            label=keyword,
            color={"background": {"gradient": {"color1": "#7F5AF0", "color2": "#5C7AEA"}}},
            size=35, 
            font={"size": 24, "face": "Arial", "strokeWidth": 4, "strokeColor": "#000000"},
            shape="circle",
            # Store metadata for click handling
            isSeed=True,
            rank=3,
            serpScore=45,
            features=["PAA", "Featured Snippet"],
            answeredCount=answered_count,
            totalQuestions=total_count
        )
        
        # Add question nodes with explicit color coding
        for i, (idx, row) in enumerate(group.iterrows()):
            question_id = f"q_{idx}_{keyword}"
            cluster_id = row["cluster_id"]
            relevancy_score = row.get("relevancy_score", 0.5)
            
            # Determine if question is answered
            answer_text = row.get("answer", "").strip()
            is_answered = len(answer_text) > 10 and not answer_text.lower().startswith(("no answer", "not found", "n/a"))
            
            # Clean question label
            clean_question = row["question"]
            
            # Explicit color coding: teal for answered, red ring for unanswered
            if cluster_id == -1:
                color = {"background": "transparent", "border": "#F24E1E", "borderWidth": 3}
                size = 20
            elif is_answered:
                color = {"background": "#0EBFC4"}
                size = 24
            else:
                color = {"background": "transparent", "border": "#F24E1E", "borderWidth": 3}
                size = 24
            
            # Store full text for tooltip
            full_question = row["question"]
            full_answer = answer_text
        
            net.add_node(
                question_id, 
                label=clean_question,
                color=color,
                size=size,
                font={"size": 18, "face": "Arial", "strokeWidth": 4, "strokeColor": "#000000"},
                shape="circle",
                # Store metadata for tooltip and click handling
                isSeed=False,
                fullQuestion=full_question,
                fullAnswer=full_answer,
                isAnswered=is_answered,
                relevancyScore=relevancy_score,
                clusterId=cluster_id,
                keyword=keyword
            )
            
            # Add edge from seed to question
            edge_color = "#0EBFC4" if is_answered else "#F24E1E"
            net.add_edge(
                keyword, 
                question_id, 
                value=relevancy_score,
                width=1.5,
                color=edge_color,
                smooth={"type": "continuous", "roundness": 0.1}
            )
    
    # Optimized physics configuration for large datasets
    physics_config = """
    var options = {
      "physics": {
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -600,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.02,
          "damping": 0.3,
          "avoidOverlap": 1.5
        },
        "stabilization": {
          "enabled": true,
          "iterations": 150
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "roundness": 0.1
        }
      },
      "nodes": {
        "font": {
          "size": 16,
          "face": "Arial",
          "strokeWidth": 3,
          "strokeColor": "#000000",
          "multi": false
        },
        "scaling": {
          "label": {
            "enabled": true,
            "min": 12,
            "max": 24,
            "drawThreshold": 10
          }
        }
      }
    }
    """
    
    net.set_options(physics_config)
    
    # Custom CSS with improved tooltip and legend
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    .vis-network {
        background: #1F2023 !important;
        border-radius: 8px;
        font-family: 'Inter', Arial, sans-serif !important;
    }
    
    .vis-network canvas {
        border-radius: 8px;
    }
    
    /* Disable built-in tooltips */
    .vis-tooltip { 
        display: none !important; 
    }
    
    /* Custom tooltip */
    .my-tooltip {
        position: absolute;
        background: rgba(0, 0, 0, 0.95);
        color: #FFFFFF;
        border: 1px solid #4E5261;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Inter', Arial, sans-serif;
        font-size: 14px;
        font-weight: 400;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
        max-width: 350px;
        z-index: 10000;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .my-tooltip.show {
        opacity: 1;
    }
    
    /* Legend overlay */
    .graph-legend {
        position: absolute;
        bottom: 16px;
        left: 16px;
        background: rgba(30, 30, 30, 0.9);
        padding: 12px;
        border-radius: 8px;
        color: #E0E0E0;
        font-size: 13px;
        font-family: 'Inter', Arial, sans-serif;
        z-index: 1000;
        border: 1px solid #4E5261;
    }
    
    .graph-legend span {
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 8px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.3);
    }
    </style>
    """
    
    # Generate HTML and add custom CSS
    html_content = net.generate_html()
    html_content = html_content.replace('</head>', f'{custom_css}</head>')
    
    # Add custom tooltip div
    tooltip_div = """
    <div id="myTooltip" class="my-tooltip" style="display:none;"></div>
    """
    
    # Legend overlay
    legend_html = """
    <div class="graph-legend">
        <div><span style="background:linear-gradient(45deg,#7F5AF0,#5C7AEA)"></span> Indigo = Seed keyword</div>
        <div><span style="background:#0EBFC4"></span> Teal = Question answered by site</div>
        <div><span style="border:2px solid #F24E1E; background:transparent"></span> Red ring = Question not answered by site</div>
    </div>
    """
    
    # Optimized JavaScript with simplified clustering for large datasets
    custom_js = """
    <script>
    let tooltip = null;
    
    document.addEventListener('DOMContentLoaded', function() {
        tooltip = document.getElementById('myTooltip');
        
        const network = window.network;
        if (network) {
            // Handle node hover for custom tooltip
            network.on('hoverNode', function(params) {
                const node = network.body.data.nodes.get(params.node);
                if (node) {
                    showCustomTooltip(node, params.event.pointer.DOM);
                }
            });
            
            network.on('blurNode', function(params) {
                hideCustomTooltip();
            });
            
            // Simplified clustering for large datasets
            network.once('stabilizationIterationsDone', function() {
                const nodeCount = network.body.data.nodes.length;
                console.log('Stabilization complete with', nodeCount, 'nodes');
                
                // For large datasets, use simpler layout
                if (nodeCount > 500) {
                    console.log('Large dataset detected, using simplified layout');
                    // Just stop simulation without complex positioning
                    network.stopSimulation();
                    
                    // Auto-zoom to fit
                    setTimeout(() => {
                        network.fit({
                            animation: { duration: 600, easingFunction: "easeInOutQuad" }
                        });
                    }, 100);
                } else {
                    // For smaller datasets, use radial clustering
                    const canvasWidth = network.canvas.frame.canvas.width;
                    const canvasHeight = network.canvas.frame.canvas.height;
                    const centerX = canvasWidth / 2;
                    const centerY = canvasHeight / 2;
                    const seedCount = """ + str(len(seed_ids)) + """;
                    
                    if (seedCount > 0) {
                        const seedIds = """ + str(seed_ids) + """;
                        seedIds.forEach((id, i) => {
                            const angle = (2 * Math.PI / seedCount) * i;
                            const radius = Math.min(300, 800 / seedCount);
                            network.body.data.nodes.update({
                                id: id,
                                x: centerX + Math.cos(angle) * radius,
                                y: centerY + Math.sin(angle) * radius,
                                fixed: true
                            });
                        });
                    }
                    
                    // Auto-zoom to fit
                    setTimeout(() => {
                        network.fit({
                            animation: { duration: 600, easingFunction: "easeInOutQuad" }
                        });
                    }, 100);
                    
                    network.stopSimulation();
                }
                
                // Bind click handler
                network.on('click', function(params) {
                    if (!params.nodes.length) return;
                    const node = network.body.data.nodes.get(params.nodes[0]);
                    if (node) {
                        let message = {};
                        
                        if (node.isSeed) {
                            message = {
                                type: 'seed_click',
                                payload: {
                                    keyword: node.id,
                                    rank: node.rank || 3,
                                    serp_score: node.serpScore || 45,
                                    features: node.features || ['PAA', 'Featured Snippet'],
                                    answered: (node.answeredCount || 0) + '/' + (node.totalQuestions || 0)
                                }
                            };
                        } else {
                            message = {
                                type: 'question_click',
                                payload: {
                                    question: node.fullQuestion || node.label,
                                    answer: node.fullAnswer || 'No answer available',
                                    answered: node.isAnswered || false,
                                    relevancy_score: node.relevancyScore || 0.5,
                                    cluster_id: node.clusterId || -1,
                                    keyword: node.keyword || 'Unknown'
                                }
                            };
                        }
                        
                        // Send message to Streamlit
                        if (window.parent && window.parent.postMessage) {
                            window.parent.postMessage(message, '*');
                        }
                        
                        if (window.Streamlit) {
                            window.Streamlit.setComponentValue(message);
                        }
                    }
                });
            });
        }
    });
    
    function showCustomTooltip(node, position) {
        if (!tooltip) return;
        
        let tooltipContent = '';
        
        if (node.isSeed) {
            const rank = node.rank || 3;
            const score = node.serpScore || 45;
            tooltipContent = `<strong>${node.label}</strong><br/>Rank: ${rank}<br/>Score: ${score}`;
        } else {
            const question = node.fullQuestion || node.label;
            const answer = node.fullAnswer || 'No answer available';
            const status = node.isAnswered ? '✅ Answered' : '❌ Unanswered';
            const score = node.relevancyScore ? node.relevancyScore.toFixed(2) : '0.50';
            
            tooltipContent = `<strong>${question}</strong><br/>${answer}<br/><br/>${status}<br/>Relevancy: ${score}`;
        }
        
        tooltip.innerHTML = tooltipContent;
        tooltip.style.left = position.x + 'px';
        tooltip.style.top = position.y + 'px';
        tooltip.style.display = 'block';
        tooltip.classList.add('show');
    }
    
    function hideCustomTooltip() {
        if (tooltip) {
            tooltip.classList.remove('show');
            setTimeout(() => {
                tooltip.style.display = 'none';
            }, 300);
        }
    }
    </script>
    """
    
    # Insert tooltip, legend, and JavaScript into the network container
    html_content = html_content.replace(
        '<div id="mynetworkid"></div>',
        f'<div id="mynetworkid"></div>{tooltip_div}{legend_html}'
    )
    
    # Add JavaScript before closing body tag
    html_content = html_content.replace('</body>', f'{custom_js}</body>')
    
    return html_content

def export_table(df: pd.DataFrame, fmt: str) -> bytes:
    """Export DataFrame in specified format.
    
    Args:
        df: DataFrame to export
        fmt: Format ('csv', 'xlsx', 'json')
    
    Returns:
        Bytes object with exported data
    """
    if fmt == "csv":
        pass
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='PAA_Research')
        return output.getvalue()
    elif fmt == "json":
        return df.to_json(orient="records", indent=2).encode("utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

###############################################################################
# Prompt Builder & FAQ JSON-LD Functions                                       
###############################################################################

def get_groups(qna_df: pd.DataFrame, grouping_key: str) -> List[str]:
    """Get available groups based on grouping key.
    
    Args:
        qna_df: DataFrame with Q&A data
        grouping_key: 'seed keyword', 'cluster', or 'url'
    
    Returns:
        List of unique group values
    """
    if qna_df.empty:
        pass
        return []
    
    if grouping_key == "seed keyword":
    
        pass
        return sorted(qna_df["keyword"].unique())
    elif grouping_key == "cluster":
        return sorted(qna_df["cluster_id"].unique())
    elif grouping_key == "url":
        # Check if we have taxonomy mapping
        if "matched_url" in qna_df.columns:
            pass
            return sorted(qna_df["matched_url"].dropna().unique())
        else:
            # Fallback to placeholder URLs
            return ["example.com/page1", "example.com/page2"]
    else:
        return []

def fill_placeholders(template: str, group_df: pd.DataFrame) -> str:
    """Fill placeholders in template with real content.
    
    Args:
        template: Template string with placeholders
        group_df: DataFrame with Q&A data for the selected group
    
    Returns:
        Template with placeholders replaced
    """
    if group_df.empty:
        pass
        return template
    
    # Get seed keyword (assuming all rows have same keyword)
    seed_keyword = group_df["keyword"].iloc[0] if not group_df.empty else ""
    
    # Get URL if available
    matched_url = group_df["matched_url"].iloc[0] if "matched_url" in group_df.columns and not group_df.empty else ""
    
    # Prepare FAQ lists
    questions = group_df["question"].tolist()
    answers = group_df["answer"].tolist()
    
    # Create FAQ pairs
    faq_pairs = []
    for q, a in zip(questions, answers):
        faq_pairs.append(f"Q: {q}\nA: {a}")
    
    # Replace placeholders
    result = template
    
    # Replace [seed keyword]
    result = result.replace("[seed keyword]", seed_keyword)
    
    # Replace [URL] if available
    if matched_url:
        pass
        result = result.replace("[URL]", matched_url)
    
    # Replace [faqs] (all questions)
    if "[faqs]" in result:
        pass
        faqs_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        result = result.replace("[faqs]", faqs_text)
    
    # Replace [answers] (all answers)
    if "[answers]" in result:
        pass
        answers_text = "\n".join([f"{i+1}. {a}" for i, a in enumerate(answers)])
        result = result.replace("[answers]", answers_text)
    
    # Replace [faqs and answers] (Q&A pairs)
    if "[faqs and answers]" in result:
        pass
        qa_text = "\n\n".join(faq_pairs)
        result = result.replace("[faqs and answers]", qa_text)
    
    # Replace individual FAQ placeholders [faq 1], [faq 2], etc.
    for i, (q, a) in enumerate(zip(questions, answers)):
        placeholder = f"[faq {i+1}]"
        if placeholder in result:
            pass
            result = result.replace(placeholder, f"Q: {q}\nA: {a}")
    
    return result

def prepare_export_data(qna_df: pd.DataFrame, grouping_key: str, threshold: float) -> pd.DataFrame:
    """Prepare data for export based on grouping and threshold.
    
    Args:
        qna_df: DataFrame with Q&A data
        grouping_key: How to group the data
        threshold: Relevancy threshold
    
    Returns:
        Filtered and grouped DataFrame
    """
    if qna_df.empty:
        pass
        return qna_df
    
    # Filter by threshold (only if relevancy_score column exists)
    if "relevancy_score" in qna_df.columns:
        pass
        df_filtered = qna_df[qna_df["relevancy_score"] >= threshold].copy()
    else:
        df_filtered = qna_df.copy()
    
    if df_filtered.empty:
    
        pass
        return df_filtered
    
    # Add grouping column
    if grouping_key == "seed_keyword":
        pass
        df_filtered["group"] = df_filtered["keyword"]
    elif grouping_key == "cluster":
        df_filtered["group"] = df_filtered["cluster_id"].astype(str)
    elif grouping_key == "url":
        # Use matched URL if available, otherwise fallback
        if "matched_url" in df_filtered.columns:
            pass
            df_filtered["group"] = df_filtered["matched_url"].fillna("Unmatched")
        else:
            df_filtered["group"] = "example.com/page1"
    
    return df_filtered

def export_qna(df: pd.DataFrame, fmt: str) -> bytes:
    """Export Q&A data in specified format.
    
    Args:
        df: DataFrame with Q&A data
        fmt: Format ('csv', 'xlsx', 'json')
    
    Returns:
        Bytes object with exported data
    """
    if fmt == "csv":
        pass
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Q&A_Export')
        return output.getvalue()
    elif fmt == "json":
        return df.to_json(orient="records", indent=2).encode("utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def generate_faq_jsonld(qna_list: List[Dict]) -> str:
    """Generate FAQ JSON-LD schema.
    
    Args:
        qna_list: List of Q&A dictionaries with 'question' and 'answer' keys
    
    Returns:
        JSON-LD string
    """
    if not qna_list:
        pass
        return json.dumps({
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": []
        }, indent=2)
    
    main_entity = []
    
    for qa in qna_list:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        
        if question and answer:
        
            pass
            main_entity.append({
                "@type": "Question",
                "name": question,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": answer
                }
            })
    
    jsonld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": main_entity
    }
    
    return json.dumps(jsonld, indent=2)

def render_prompt_builder(qna_df: pd.DataFrame) -> None:
    """Render the prompt builder interface.
    
    Args:
        qna_df: DataFrame with Q&A data
    """
    st.subheader("Prompt Builder")
    
    if qna_df.empty:
    
        pass
        st.warning("No Q&A data available. Please run research analysis first.")
        return
    
    # Grouping selection
    grouping_key = st.selectbox(
        "Group by",
        ["seed keyword", "cluster", "url"],
        help="Choose how to group Q&A data for prompt generation"
    )
    
    # Get available groups
    groups = get_groups(qna_df, grouping_key)
    
    if not groups:
    
        pass
        st.warning(f"No groups available for '{grouping_key}' grouping.")
        return
    
    # Group selection
    selected_groups = st.multiselect(
        "Select groups",
        options=groups,
        default=groups[:1] if groups else [],
        help="Choose which groups to include in the prompt"
    )
    
    if not selected_groups:
    
        pass
        st.warning("Please select at least one group.")
        return
    
    # Template input
    default_template = """Write an FAQ section about [seed keyword]:

[faqs and answers]

Please ensure the content is helpful, accurate, and follows SEO best practices."""
    
    template = st.text_area(
        "Template",
        value=default_template,
        height=200,
        help="Use placeholders: [seed keyword], [faqs], [answers], [faqs and answers], [faq 1], [faq 2], etc."
    )
    
    # Generate prompt button
    if st.button("Generate Prompt", type="primary"):
        pass
        # Get data for selected groups
        if grouping_key == "seed keyword":
            pass
            group_data = qna_df[qna_df["keyword"].isin(selected_groups)]
        elif grouping_key == "cluster":
            group_data = qna_df[qna_df["cluster_id"].isin([int(g) for g in selected_groups if g.isdigit()])]
        else:  # url
            group_data = qna_df.copy()  # Placeholder
        
        if group_data.empty:
        
            pass
            st.error("No data found for selected groups.")
            return
        
        # Fill placeholders
        final_prompt = fill_placeholders(template, group_data)
        
        # Display result
        st.subheader("Generated Prompt")
        st.code(final_prompt, language="text")
        
        # Copy button
        if st.button("Copy to Clipboard"):
            pass
            st.write("Prompt copied to clipboard!")

def render_data_export(qna_df: pd.DataFrame) -> None:
    """Render the data export interface.
    
    Args:
        qna_df: DataFrame with Q&A data
    """
    st.subheader("Data Export")
    
    if qna_df.empty:
    
        pass
        st.warning("No Q&A data available. Please run research analysis first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        grouping_key = st.selectbox(
            "Group by",
            ["seed keyword", "cluster", "url"],
            key="export_grouping",
            help="Choose how to group exported data"
        )
    
    with col2:
        threshold = st.slider(
            "Relevancy Threshold",
            0.0, 1.0, 0.5, 0.1,
            key="export_threshold",
            help="Filter out questions below this threshold"
        )
    
    # Prepare export data
    export_df = prepare_export_data(qna_df, grouping_key, threshold)
    
    if export_df.empty:
    
        pass
        st.warning("No data passes the threshold. Try lowering the threshold.")
        return
    
    # Show preview
    st.subheader("Preview")
    # Select columns that exist
    preview_columns = ["group", "keyword", "question", "answer"]
    if "relevancy_score" in export_df.columns:
        pass
        preview_columns.append("relevancy_score")
    if "cluster_id" in export_df.columns:
        pass
        preview_columns.append("cluster_id")
    st.dataframe(
        export_df[preview_columns].head(20),
        use_container_width=True
    )
    
    # Export buttons
    st.subheader("Download")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = export_qna(export_df, "csv")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"qna_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        xlsx_data = export_qna(export_df, "xlsx")
        st.download_button(
            label="Download Excel",
            data=xlsx_data,
            file_name=f"qna_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        json_data = export_qna(export_df, "json")
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"qna_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def render_faq_jsonld(qna_df: pd.DataFrame) -> None:
    """Render the FAQ JSON-LD generator interface.
    
    Args:
        qna_df: DataFrame with Q&A data
    """
    st.subheader("FAQ JSON-LD Schema")
    
    if qna_df.empty:
    
        pass
        st.warning("No Q&A data available. Please run research analysis first.")
        return
    
    # Get available URLs (placeholder for now)
    # In a real implementation, you'd have URL mapping from your taxonomy
    available_urls = [
        "example.com/page1",
        "example.com/page2", 
        "example.com/faq",
        "example.com/help"
    ]
    
    selected_urls = st.multiselect(
        "Select pages",
        options=available_urls,
        default=available_urls[:1] if available_urls else [],
        help="Choose which pages to generate JSON-LD for"
    )
    
    if not selected_urls:
    
        pass
        st.warning("Please select at least one page.")
        return
    
    # Generate JSON-LD for each selected URL
    for url in selected_urls:
        st.write(f"**JSON-LD for {url}:**")
        
        # For now, use all Q&A data. In real implementation, filter by URL mapping
        qna_for_url = qna_df.to_dict("records")
        
        jsonld = generate_faq_jsonld(qna_for_url)
        
        st.code(jsonld, language="json")
        
        # Copy button
        if st.button(f"Copy JSON-LD for {url}", key=f"copy_{url}"):
            pass
            st.write(f"JSON-LD for {url} copied to clipboard!")

###############################################################################
# Streamlit UI                                                                 
###############################################################################

st.set_page_config(
    page_title="Serpwise.ai - SERP Analysis & PAA Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and scheduler
if "db_initialized" not in st.session_state:
    pass
    init_db()
    init_scheduler()
    st.session_state.db_initialized = True

# Initialize session state
if "combined_df" not in st.session_state:
    pass
    st.session_state.combined_df = None

# Authentication
if "current_user_id" not in st.session_state:
    pass
    st.session_state.current_user_id = None

# Login/Register UI
if st.session_state.current_user_id is None:
    pass
    # Serpwise.ai branding with final large SVG logo
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            st.image("assets/serpwise_ai_logo_darkmode_final.svg", width=300)
        except:
            st.markdown("🚀")
    with col2:
        st.markdown("""
        <div style="display: flex; flex-direction: column; justify-content: center; height: 250px;">
            <div style="font-size: 18px; color: #CCCCCC; font-weight: 400;">SERP Analysis & PAA Research Platform</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for login and register
    auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])
    
    with auth_tab1:
        st.header("Login")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
            
                pass
                if email and password:
                    pass
                    user_id = authenticate_user(email, password)
                    if user_id:
                        pass
                        st.session_state.current_user_id = user_id
                        user_obj = get_user_by_id(user_id)
                        st.session_state.current_user_data = {
                            "id": user_obj.id,
                            "name": user_obj.name,
                            "email": user_obj.email
                        }
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.error("Please enter both email and password")
    
    with auth_tab2:
        st.header("Register")
        
        with st.form("register_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
            
                pass
                if name and email and password and confirm_password:
                    pass
                    if password == confirm_password:
                        pass
                        try:
                            user_id = create_user(email, name, password)
                            st.success("Registration successful! Please login.")
                        except ValueError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error("Registration failed. Please try again.")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")
    
    st.stop()

# Main app (after authentication)
current_user_id = st.session_state.current_user_id

# Get user data and store in session state to avoid detached instance issues
if "current_user_data" not in st.session_state:
    pass
    current_user = get_user_by_id(current_user_id)
    if current_user:
        pass
        st.session_state.current_user_data = {
            "id": current_user.id,
            "name": current_user.name,
            "email": current_user.email
        }
    else:
        st.session_state.current_user_data = None

current_user_data = st.session_state.current_user_data

if current_user_data is None:

    pass
    st.error("User session expired. Please login again.")
    st.session_state.current_user_id = None
    st.rerun()

# Serpwise.ai branding with final large SVG logo for main app
col1, col2 = st.columns([1, 2])
with col1:
    try:
        st.image("assets/serpwise_ai_logo_darkmode_final.svg", width=250)
    except:
        st.markdown("🚀")
with col2:
    st.markdown("""
    <div style="display: flex; flex-direction: column; justify-content: center; height: 200px;">
        <div style="font-size: 18px; color: #CCCCCC; font-weight: 400;">SERP Analysis & PAA Research Platform</div>
    </div>
    """, unsafe_allow_html=True)

# User info in sidebar
with st.sidebar:
    # Final large Serpwise.ai logo in sidebar
    try:
        st.image("assets/serpwise_ai_logo_darkmode_final.svg", width=200)
    except:
        st.markdown("🚀")
    
    if current_user_data:
        st.header(f"Welcome, {current_user_data['name']}!")
    else:
        st.header("Welcome!")
    st.write(f"Email: {current_user_data['email']}")
    
    if st.button("Logout"):
    
        pass
        st.session_state.current_user_id = None
        st.session_state.current_user_data = None
        st.rerun()
    
    st.divider()

# Project selection
if "selected_project_id" not in st.session_state:
    pass
    st.session_state.selected_project_id = None

# Get user's projects
user_projects = get_user_projects(current_user_data['id'])

if not user_projects:

    pass
    st.warning("You don't have any projects yet. Create your first project!")
    
    # Create first project
    with st.expander("Create Your First Project", expanded=True):
        with st.form("create_first_project"):
            project_name = st.text_input("Project Name", placeholder="My SEO Project")
            project_description = st.text_area("Description", placeholder="Describe your project")
            location = st.selectbox("Location", [
    "us", "uk", "de", "fr", "es", "it", "ca", "au", "nl", "br", "mx", "jp", "kr", "cn", "in", 
    "se", "no", "dk", "fi", "pl", "cz", "hu", "ro", "bg", "hr", "rs", "sk", "si", "ee", "lv", 
    "lt", "ie", "nz", "za", "sg", "my", "ph", "th", "vn", "id", "tr", "gr", "pt", "be", "ch", 
    "at", "lu", "mt", "cy"
])
            domain = st.text_input("Domain (optional)", placeholder="example.com")
            
            if st.form_submit_button("Create Project"):
            
                pass
                if project_name:
                    pass
                    try:
                        project_id = create_project(
                            name=project_name,
                            description=project_description,
                            owner_user_id=current_user_data['id'],
                            location=location,
                            domain=domain
                        )
                        st.session_state.selected_project_id = project_id
                        st.success("Project created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create project: {e}")
                else:
                    st.error("Please enter a project name")
else:
    # Project selection
    st.header("Project Management")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create project options for display
        project_options = {f"{p['name']} ({p['user_role']})": p['id'] for p in user_projects}
        selected_project_name = st.selectbox(
            "Select Project",
            options=list(project_options.keys()),
            index=0 if not st.session_state.selected_project_id else None
        )
        
        if selected_project_name:
        
            pass
            selected_project_id = project_options[selected_project_name]
            st.session_state.selected_project_id = selected_project_id
            
            # Get project details
            selected_project = next(p for p in user_projects if p['id'] == selected_project_id)
            
            st.write(f"**Description:** {selected_project['description'] or 'No description'}")
            st.write(f"**Owner:** {selected_project['owner_name']}")
            st.write(f"**Your Role:** {selected_project['user_role']}")
            st.write(f"**Location:** {selected_project['location']}")
            if selected_project['domain']:
                pass
                st.write(f"**Domain:** {selected_project['domain']}")
    
    with col2:
        if st.button("Create New Project"):
            pass
            st.session_state.show_create_project = True
    
    # Create new project form
    if st.session_state.get("show_create_project", False):
        pass
        with st.expander("Create New Project", expanded=True):
            with st.form("create_project_form"):
                project_name = st.text_input("Project Name", key="new_project_name")
                project_description = st.text_area("Description", key="new_project_desc")
                location = st.selectbox("Location", [
    "us", "uk", "de", "fr", "es", "it", "ca", "au", "nl", "br", "mx", "jp", "kr", "cn", "in", 
    "se", "no", "dk", "fi", "pl", "cz", "hu", "ro", "bg", "hr", "rs", "sk", "si", "ee", "lv", 
    "lt", "ie", "nz", "za", "sg", "my", "ph", "th", "vn", "id", "tr", "gr", "pt", "be", "ch", 
    "at", "lu", "mt", "cy"
], key="new_project_location")
                domain = st.text_input("Domain (optional)", key="new_project_domain")
                
                if st.form_submit_button("Create Project"):
                
                    pass
                    if project_name:
                        pass
                        try:
                            project_id = create_project(
                                name=project_name,
                                description=project_description,
                                owner_user_id=current_user_data["id"],
                                location=location,
                                domain=domain
                            )
                            st.session_state.selected_project_id = project_id
                            st.session_state.show_create_project = False
                            st.success("Project created successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create project: {e}")
                    else:
                        st.error("Please enter a project name")
    
    # Project management (for owners and editors)
    if st.session_state.selected_project_id:
        pass
        selected_project = next(p for p in user_projects if p['id'] == st.session_state.selected_project_id)
        user_role = selected_project['user_role']
        
        # Check permissions
        can_edit = user_role in ["owner", "editor"]
        can_manage = user_role == "owner"
        
        if can_manage:
        
            pass
            st.subheader("Project Management")
            
            # Invite members
            with st.expander("Invite Members"):
                with st.form("invite_member_form"):
                    invite_email = st.text_input("Email Address")
                    invite_role = st.selectbox("Role", ["Editor", "Viewer"])
                    
                    if st.form_submit_button("Send Invite"):
                    
                        pass
                        if invite_email:
                            pass
                            try:
                                role_enum = UserRole.EDITOR if invite_role == "Editor" else UserRole.VIEWER
                                invite_member(
                                    project_id=st.session_state.selected_project_id,
                                    email=invite_email,
                                    role=role_enum,
                                    invited_by_user_id=current_user_data["id"]
                                )
                                st.success(f"Invited {invite_email} as {invite_role}")
                            except ValueError as e:
                                st.error(str(e))
                            except Exception as e:
                                st.error(f"Failed to invite member: {e}")
                        else:
                            st.error("Please enter an email address")
            
            # Show current members
            st.subheader("Current Members")
            members_df = get_project_members(st.session_state.selected_project_id)
            
            if not members_df.empty:
            
                pass
                st.dataframe(members_df[["name", "email", "role", "invited_at"]], use_container_width=True)
                
                # Remove member functionality
                if st.button("Remove Member"):
                    pass
                    st.warning("Remove member functionality would be implemented here")
            else:
                st.info("No members yet")
        
        # Audit logs
        if can_edit:
            pass
            st.subheader("Recent Activity")
            audit_logs = get_audit_logs(project_id=st.session_state.selected_project_id, limit=10)
            
            if not audit_logs.empty:
            
                pass
                st.dataframe(audit_logs, use_container_width=True)
            else:
                st.info("No recent activity")



# Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Upload & Preview", "Live SERP Fetch", "Scheduling", "Trends", "Research", "Prompt Builder", "Taxonomy", "Projects"])

# Check if user has selected a project
if st.session_state.selected_project_id is None:
    pass
    st.warning("Please select a project to continue.")
    st.stop()

# Get current project details
current_project = next(p for p in user_projects if p['id'] == st.session_state.selected_project_id)
user_role = current_project['user_role']
can_edit = user_role in ["owner", "editor"]
can_manage = user_role == "owner"

# Initialize session state for centralized data management
if "project_keywords" not in st.session_state:
    pass
    st.session_state.project_keywords = []
if "latest_serp_data" not in st.session_state:
    pass
    st.session_state.latest_serp_data = None
if "processed_qna_data" not in st.session_state:
    pass
    st.session_state.processed_qna_data = None
if "taxonomy_data" not in st.session_state:
    pass
    st.session_state.taxonomy_data = None

# Load CTR data for scoring
ctr_df = fetch_ctr_data()

# Simple sidebar with CTR status
with st.sidebar:
    st.header("📊 System Status")
    st.success("✅ CTR benchmarks loaded")
    st.info(f"📈 {len(ctr_df)} CTR positions available")

# Permission checks for each tab
if not can_edit and tab1 in st.session_state.get("active_tabs", []):
    pass
    st.error("You need Editor or Owner permissions to upload data.")
    st.stop()

with tab1:
    ################################################################################
    # Centralized Data Ingestion - Upload & Preview                                 
    ################################################################################

    st.header("📤 Upload & Preview")
    st.markdown("**Single ingestion point for all keywords and initial SERP data**")
    
    if not can_edit:
    
        pass
        st.warning("You need Editor or Owner permissions to upload data.")
    else:
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload one or more Excel/CSV files (Surplex KWR, SEMrush, or other formats)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            help="Upload your keyword data files. The system will auto-detect column mappings or allow manual mapping."
        )

        if uploaded_files:

            pass
            # Initialize session state for file processing
            if "file_mappings" not in st.session_state:
                pass
                st.session_state.file_mappings = {}
            if "processed_files" not in st.session_state:
                pass
                st.session_state.processed_files = []
            
            all_normalized = []
            
            for i, uf in enumerate(uploaded_files):
                st.subheader(f"📄 Processing: {uf.name}")
                
                # Load file with enhanced detection
                raw_df, auto_mapping = load_excel(uf)
                
                if raw_df is None or raw_df.empty:
                
                    pass
                    st.error(f"❌ Failed to read {uf.name}")
                    continue
                
                # Check if this is URL Structuur KWR format
                if auto_mapping.get("format") == "url_structuur_kwr":
                    pass
                    st.success(f"✅ Detected URL Structuur KWR format for {uf.name}")
                    st.info(f"📊 Parsed {len(raw_df)} keywords from URL Structuur sheet")
                    
                    # Show preview of the parsed data
                    st.write("**Parsed Keywords Preview:**")
                    preview_cols = ["url", "keyword", "keyword_type", "search_volume", "cumulative_volume"]
                    st.dataframe(raw_df[preview_cols].head(10), use_container_width=True)
                    
                    # Normalize the URL Structuur data
                    norm_df = normalize_df(raw_df, "url_structuur_kwr")
                    if not norm_df.empty:
                        pass
                        all_normalized.append(norm_df)
                        st.session_state.processed_files.append(uf.name)
                        st.success(f"✅ Successfully processed {len(norm_df)} keywords from URL Structuur format")
                
                # Check if auto-mapping was successful for other formats
                elif auto_mapping and "format" not in auto_mapping:
                    is_valid, missing_fields = validate_mapping(auto_mapping)
                    
                    if is_valid:
                    
                        pass
                        st.success(f"✅ Auto-detected column mapping for {uf.name}")
                        # Apply the mapping and normalize
                        mapped_df = apply_column_mapping(raw_df, auto_mapping)
                        norm_df = normalize_df(mapped_df, "auto_detected")
                        if not norm_df.empty:
                            pass
                            all_normalized.append(norm_df)
                            st.session_state.processed_files.append(uf.name)
                    else:
                        st.warning(f"⚠️ Auto-detection incomplete for {uf.name}. Missing fields: {missing_fields}")
                        
                        # Show data preview
                        st.write("**Data Preview:**")
                        st.dataframe(raw_df.head(5), use_container_width=True)
                        
                        # Interactive mapping interface
                        st.write("**Column Mapping:**")
                        st.write("Please map the columns below:")
                        
                        # Create mapping interface
                        mapping = {}
                        available_columns = raw_df.columns.tolist()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Required fields
                            st.write("**Required Fields:**")
                            for field in ["keyword", "position", "search_volume"]:
                                selected_col = st.selectbox(
                                    f"Map '{field}' to:",
                                    ["-- Select Column --"] + available_columns,
                                    key=f"mapping_{uf.name}_{field}",
                                    index=0
                                )
                                if selected_col != "-- Select Column --":
                                    pass
                                    mapping[field] = selected_col
                        
                        with col2:
                            # Optional fields
                            st.write("**Optional Fields:**")
                            for field in ["serp_features", "intents", "position_type", "timestamp", "previous_position", "url", "traffic", "traffic_pct"]:
                                selected_col = st.selectbox(
                                    f"Map '{field}' to:",
                                    ["-- Select Column --"] + available_columns,
                                    key=f"mapping_{uf.name}_{field}",
                                    index=0
                                )
                                if selected_col != "-- Select Column --":
                                    pass
                                    mapping[field] = selected_col
                        
                        # Validate manual mapping
                        manual_is_valid, manual_missing = validate_mapping(mapping)
                        
                        if manual_is_valid:
                        
                            pass
                            if st.button(f"✅ Apply Mapping for {uf.name}", key=f"apply_{uf.name}"):
                                pass
                                # Apply the mapping and normalize
                                mapped_df = apply_column_mapping(raw_df, mapping)
                                norm_df = normalize_df(mapped_df, "manual_mapped")
                                if not norm_df.empty:
                                    pass
                                    all_normalized.append(norm_df)
                                    st.session_state.processed_files.append(uf.name)
                                    st.success(f"✅ Successfully mapped and processed {uf.name}")
                                    st.session_state.file_mappings[uf.name] = mapping
                        else:
                            st.error(f"❌ Missing required fields: {manual_missing}")
                            st.info("Please map all required fields to continue.")
                            continue
                else:
                    st.error(f"❌ Failed to process {uf.name} - no valid mapping found")
                    continue

            # Process all normalized data
            if all_normalized:
                pass
                combined_df = pd.concat(all_normalized, ignore_index=True)
                combined_df = compute_serp_score(combined_df, ctr_df)
                
                # Store in session state for centralized access
                st.session_state.combined_df = combined_df
                st.session_state.project_keywords = combined_df['keyword'].tolist()

                # Store keywords in database for the project
                db = get_db()
                try:
                    # Clear existing keywords for this project
                    db.query(Keyword).filter(Keyword.project_id == st.session_state.selected_project_id).delete()
                    
                    # Add new keywords with search volume
                    for keyword in combined_df['keyword'].unique():
                        # Get the search volume for this keyword from the normalized data
                        keyword_data = combined_df[combined_df['keyword'] == keyword].iloc[0]
                        search_volume = int(keyword_data.get('search_volume', 0))
                        
                        keyword_obj = Keyword(
                            project_id=st.session_state.selected_project_id,
                            keyword=keyword,
                            search_volume=search_volume  # Persist upload SV
                        )
                        db.add(keyword_obj)
                    
                    db.commit()
                    st.success(f"✅ Successfully stored {len(combined_df['keyword'].unique())} keywords in project database")
                except Exception as e:
                    st.error(f"Failed to store keywords in database: {e}")
                    db.rollback()
                finally:
                    db.close()

                # Preview section
                st.subheader("📊 Data Preview")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(combined_df.head(20), use_container_width=True)
                
                with col2:
                    st.metric("Total Keywords", len(combined_df['keyword'].unique()))
                    st.metric("Average SERP Score", f"{combined_df['serp_score'].mean():.2f}")
                    st.metric("Top Score", f"{combined_df['serp_score'].max():.2f}")

                # Top keywords visualization
                st.subheader("🏆 Top 10 Keywords by SERP Real-Estate Score")
                top10 = (
                    combined_df.sort_values("serp_score", ascending=False)
                    .head(10)[["keyword", "serp_score"]]
                    .set_index("keyword")
                )
                st.bar_chart(top10)

                # Download button
                csv_data = combined_df.to_csv(index=False).encode("utf-8")
                st.subheader("💾 Download")
                st.download_button(
                    label="Download normalized data as CSV",
                    data=csv_data,
                    file_name="normalized_serp_visibility.csv",
                    mime="text/csv",
                )
                
                # Log the upload action
                log_audit_action(
                    user_id=current_user_data["id"],
                    action=AuditAction.KEYWORD_UPLOAD.value,
                    target_type="project",
                    target_id=st.session_state.selected_project_id,
                    details=f"Uploaded {len(combined_df['keyword'].unique())} keywords from {len(st.session_state.processed_files)} files"
                )
            else:
                st.info("No valid data extracted from uploaded files.")
        else:
            st.info("📁 Upload files to begin. All keywords will be stored in the project database for use across all modules.")

with tab2:
    ################################################################################
    # Live SERP Fetch - Full Keyword Coverage                                      
    ################################################################################

    st.header("🔍 Live SERP Fetch")
    st.markdown("**Fetch live SERP data for all keywords in the project**")
    
    if not can_edit:
    
        pass
        st.warning("You need Editor or Owner permissions to run SERP fetches.")
    else:
        # Get all keywords from the project database
        db = get_db()
        project_keywords = db.query(Keyword).filter(
            Keyword.project_id == st.session_state.selected_project_id
        ).all()
        db.close()
        
        if not project_keywords:
        
            pass
            st.warning("No keywords found in this project. Please upload keywords in the 'Upload & Preview' tab first.")
        else:
            st.info(f"📋 Found {len(project_keywords)} keywords in project database")
            
            # SERP fetch controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                fetch_limit = st.number_input(
                    "Number of keywords to fetch (0 = all keywords)",
                    min_value=0,
                    max_value=len(project_keywords),
                    value=min(10, len(project_keywords)),
                    help="Set to 0 to fetch all keywords, or specify a limit"
                )
            
            with col2:
                location = st.selectbox(
                    "Location",
                    [
                        "us", "uk", "de", "fr", "es", "it", "ca", "au", "nl", "br", "mx", "jp", "kr", "cn", "in", 
                        "se", "no", "dk", "fi", "pl", "cz", "hu", "ro", "bg", "hr", "rs", "sk", "si", "ee", "lv", 
                        "lt", "ie", "nz", "za", "sg", "my", "ph", "th", "vn", "id", "tr", "gr", "pt", "be", "ch", 
                        "at", "lu", "mt", "cy"
                    ],
                    index=0
                )
                
            with col3:
                if st.button("🚀 Fetch Live SERP Data", type="primary"):
                    if fetch_limit == 0:
                        keywords_to_fetch = [kw.keyword for kw in project_keywords]
                    else:
                        keywords_to_fetch = [kw.keyword for kw in project_keywords[:fetch_limit]]
                    
                    logging.info(f"Fetching SERPs for {len(keywords_to_fetch)} keywords")
                    st.info(f"Fetching SERP data for {len(keywords_to_fetch)} keywords...")
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    fetched_data = []
                    api_key = get_searchapi_key()
                    
                    if not api_key:
                        st.error("SearchAPI key not found. Please add SEARCHAPI_KEY to your Streamlit secrets.")
                    else:
                        for i, keyword in enumerate(keywords_to_fetch):
                            status_text.text(f"Fetching: {keyword}")
                            
                            serp_json = fetch_serp(keyword, location, api_key)
                            if serp_json:
                                pass
                                parsed_features = parse_serp_features(serp_json)
                                scores = compute_serp_real_estate_score(
                                    parsed_features, 
                                    ctr_df, 
                                    current_project.get('domain')
                                )
                                
                                # Save to database
                                save_serp_run(
                                    project_id=st.session_state.selected_project_id,
                                    keyword=keyword,
                                    parsed=parsed_features,
                                    scores=scores,
                                    raw_json=json.dumps(serp_json)
                                )
                                
                                fetched_data.append({
                                    'keyword': keyword,
                                    'total_score': scores['total_score'],
                                    'organic_score': scores['organic_score'],
                                    'paa_score': scores['paa_score'],
                                    'feature_score': scores['feature_score'],
                                    'featured_snippet_bonus': scores['featured_snippet_bonus']
                                })
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(keywords_to_fetch))
                            time.sleep(0.1)  # Small delay to prevent rate limiting
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if fetched_data:
                        
                            pass
                            st.session_state.latest_serp_data = pd.DataFrame(fetched_data)
                            st.success(f"✅ Successfully fetched SERP data for {len(fetched_data)} keywords")
                            
                            # Show results
                            st.subheader("📊 Fetch Results")
                            results_df = pd.DataFrame(fetched_data)
                            st.dataframe(results_df.sort_values('total_score', ascending=False), use_container_width=True)
                            
                            # Log the action
                            log_audit_action(
                                user_id=current_user_data["id"],
                                action=AuditAction.SERP_FETCH.value,
                                target_type="project",
                                target_id=st.session_state.selected_project_id,
                                details=f"Fetched SERP data for {len(fetched_data)} keywords"
                            )
                            
                            # Automatically trigger Research processing
                            st.info("🔄 Automatically triggering Research processing...")
                            try:
                                # Get keywords from the fetched data
                                keywords = [item['keyword'] for item in fetched_data]
                                
                                # Process research data with caching
                                processed_df, cache_timestamp = process_research_cached(
                                    project_id=st.session_state.selected_project_id,
                                    keywords=keywords,
                                    threshold=0.7,
                                    max_workers=10
                                )
                                
                                if not processed_df.empty:
                                    # Store processed data and cache timestamp
                                    st.session_state.processed_qna_data = processed_df
                                    st.session_state.clustered_data = processed_df
                                    st.session_state.research_cache_timestamp = cache_timestamp
                                    st.success(f"✅ Research processing completed: {len(processed_df)} Q&A pairs processed")
                                else:
                                    st.warning("⚠️ Research processing completed but no Q&A data was extracted")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during automatic Research processing: {e}")
                                logger.error(f"Automatic Research processing failed: {e}")
                        else:
                            st.error("No SERP data was successfully fetched. Check your API key and try again.")

with tab3:
    ################################################################################
    # Scheduling                                                                   
    ################################################################################

    st.header("⏰ Scheduling")
    
    if not can_manage:
    
        pass
        st.warning("You need Owner permissions to manage scheduling.")
    else:
        # Existing scheduling code...
        schedules = load_schedules()
        project_schedules = [s for s in schedules if s['project_id'] == st.session_state.selected_project_id]
        
        if project_schedules:
        
            pass
            st.subheader("Current Schedules")
            for schedule in project_schedules:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{schedule['frequency']}** - {schedule['cron_expression']}")
                with col2:
                    if st.button(f"Cancel", key=f"cancel_{schedule['id']}"):
                        pass
                        cancel_schedule(schedule['id'])
                        st.rerun()
                with col3:
                    st.write("✅ Active" if schedule['is_active'] else "❌ Inactive")
        
        st.subheader("Create New Schedule")
        with st.form("schedule_form"):
            frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly"])
            if st.form_submit_button("Create Schedule"):
                pass
                schedule_job(st.session_state.selected_project_id, frequency)
                st.success("Schedule created!")
                st.rerun()

with tab4:
    ################################################################################
    # Enhanced Rank Tracker - SERP Real-Estate Analysis                            
    ################################################################################

    st.header("📈 Rank Tracker - SERP Real-Estate Analysis")
    st.markdown("**Comprehensive rank tracking powered by true SERP real-estate, not just positions**")
    
    # Get project details
    project_domain = current_project.get('domain', '')
    competitors = get_competitors(st.session_state.selected_project_id)
    competitor_domains = [comp['domain'] for comp in competitors]
    
    # Time window selector
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        time_window = st.selectbox(
            "Time Window",
            options=["7 days", "30 days", "90 days", "Custom"],
            index=1
        )
    with col2:
        if time_window == "Custom":
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
            end_date = st.date_input("End Date", value=date.today())
        else:
            days = int(time_window.split()[0])
            start_date = date.today() - timedelta(days=days)
            end_date = date.today()
    with col3:
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
    
    # Get trends data
    trends_df = get_trends_data(st.session_state.selected_project_id, days=(end_date - start_date).days)
    
    if trends_df.empty:
        st.info("📊 No SERP data available. Run a Live SERP Fetch first to populate the rank tracker.")
    else:
        # Competitor management section
        with st.expander("🏢 Competitor Management", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                new_competitor_domain = st.text_input("Add Competitor Domain", placeholder="example.com")
                new_competitor_name = st.text_input("Competitor Name (optional)", placeholder="Competitor Name")
            with col2:
                if st.button("➕ Add Competitor") and new_competitor_domain:
                    if add_competitor(st.session_state.selected_project_id, new_competitor_domain, new_competitor_name):
                        st.success(f"✅ Added {new_competitor_domain}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add competitor")
            
            # Show current competitors
            if competitors:
                st.write("**Current Competitors:**")
                for comp in competitors:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"• {comp['name']} ({comp['domain']})")
                    with col2:
                        st.write(f"Added: {comp['created_at'].strftime('%Y-%m-%d')}")
                    with col3:
                        if st.button(f"❌ Remove", key=f"remove_{comp['id']}"):
                            if remove_competitor(comp['id']):
                                st.success("✅ Removed")
                                st.rerun()
            else:
                st.info("No competitors added yet. Add competitors to enable share-of-voice analysis.")
        
        # Alerts section
        alerts_df = get_alerts(st.session_state.current_user_id, st.session_state.selected_project_id)
        if not alerts_df.empty:
            with st.expander(f"🚨 Alerts ({len(alerts_df)} unread)", expanded=True):
                for _, alert in alerts_df.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        alert_icon = {
                            "snippet_lost": "📉",
                            "snippet_gained": "📈", 
                            "paa_gained": "❓",
                            "score_drop": "⚠️",
                            "competitor_overtake": "🏃"
                        }.get(alert['alert_type'], "🔔")
                        st.write(f"{alert_icon} **{alert['keyword']}**: {alert['alert_type'].replace('_', ' ').title()}")
                    with col2:
                        st.write(alert['created_at'].strftime('%Y-%m-%d'))
                    with col3:
                        if st.button("✓ Mark Read", key=f"read_{alert['id']}"):
                            mark_alert_read(alert['id'])
                            st.rerun()
        
        # Enhanced Keyword Overview Table
        st.subheader("📊 Enhanced Keyword Overview")
        st.markdown("**Click a row below to view full PAA Q&A and feature breakdown**")
        
        # Get enhanced data
        qna_df = get_qna_data_for_trends(st.session_state.selected_project_id)
        serp_features_df = get_serp_features_for_trends(st.session_state.selected_project_id)
        
        # Build enhanced overview DataFrame
        overview_df = build_enhanced_overview_df(trends_df, qna_df, serp_features_df, st.session_state.selected_project_id)
        
        if not overview_df.empty:
            # Calculate additional metrics
            overview_df['score_change'] = overview_df['total_score'].diff()
            overview_df['percentile_30d'] = overview_df['keyword'].apply(
                lambda k: calculate_percentiles(st.session_state.selected_project_id, k, 30).get('percentile_30d', 0)
            )
            
            # Performance bands
            def get_performance_band(percentile):
                if percentile >= 80:
                    return "🟢 Dominant"
                elif percentile >= 60:
                    return "🟡 Strong"
                elif percentile >= 40:
                    return "🟠 Emerging"
                else:
                    return "🔴 Weak"
            
            overview_df['performance_band'] = overview_df['percentile_30d'].apply(get_performance_band)
            
            # Create feature summary string
            overview_df['feature_summary'] = overview_df.apply(lambda row: {
                'has_featured_snippet': "✅" if row.get('has_featured_snippet', False) else "❌",
                'has_local_pack': "✅" if row.get('has_local_pack', False) else "❌",
                'has_video': "✅" if row.get('has_video', False) else "❌",
                'has_knowledge_panel': "✅" if row.get('has_knowledge_panel', False) else "❌",
                'has_related_searches': "✅" if row.get('has_related_searches', False) else "❌",
                'has_sitelinks': "✅" if row.get('has_sitelinks', False) else "❌"
            }, axis=1)
            
            # Prepare display columns
            display_cols = [
                'keyword', 'search_volume', 'real_estate_score', 'paa_score', 
                'answered_qas', 'total_qas', 'performance_band', 'feature_summary'
            ]
            
            # Ensure all columns exist
            for col in display_cols:
                if col not in overview_df.columns:
                    overview_df[col] = 0
            
            display_df = overview_df[display_cols].copy()
            display_df.columns = [
                'Keyword', 'Search Volume', 'Real Estate Score', 'PAA Score',
                'Answered Q&A', 'Total Q&A', 'Performance', 'Features'
            ]
            
            # Format the answered Q&A column
            display_df['Answered Q&A'] = display_df.apply(
                lambda row: f"{row['Answered Q&A']}/{row['Total Q&A']}", axis=1
            )
            
            # Sort by real estate score
            display_df = display_df.sort_values('Real Estate Score', ascending=False)
            
            # Initialize session state for selected keyword
            if "selected_keyword" not in st.session_state:
                st.session_state.selected_keyword = None
            
            # Display clickable table
            clicked = st.data_editor(
                display_df,
                use_container_width=True,
                column_config={
                    "Search Volume": st.column_config.NumberColumn(
                        "Search Volume",
                        format="%d",
                        help="Monthly search volume"
                    ),
                    "Real Estate Score": st.column_config.NumberColumn(
                        "Real Estate Score",
                        format="%.1f",
                        help="SERP real-estate score (0-100)"
                    ),
                    "PAA Score": st.column_config.NumberColumn(
                        "PAA Score",
                        format="%.1f",
                        help="People Also Ask score"
                    ),
                    "Answered Q&A": st.column_config.TextColumn(
                        "Answered Q&A",
                        help="Owned Q&A / Total Q&A"
                    ),
                    "Features": st.column_config.TextColumn(
                        "Features",
                        help="✅=Present, ❌=Not present"
                    )
                }
            )
            
            # Handle row selection
            if clicked is not None and not clicked.empty:
                # Find the selected row (this is a simplified approach)
                # In a real implementation, you'd need to track which row was clicked
                st.session_state.selected_keyword = display_df.iloc[0]['Keyword']
            
            # Show details for selected keyword
            if st.session_state.selected_keyword:
                st.subheader(f"🔍 Details for: {st.session_state.selected_keyword}")
                
                # Get Q&A details
                keyword_qna = qna_df[qna_df['keyword'] == st.session_state.selected_keyword]
                if not keyword_qna.empty:
                    st.subheader(f"❓ PAA Q&A for {st.session_state.selected_keyword}")
                    
                    # Format Q&A data for display
                    qna_display = keyword_qna[['question', 'answer', 'answered_by_site', 'search_volume', 'answer_relevancy']].copy()
                    qna_display.columns = ['Question', 'Answer', 'Owned', 'Search Volume', 'Relevancy']
                    qna_display['Owned'] = qna_display['Owned'].map({True: '✅', False: '❌'})
                    qna_display['Relevancy'] = qna_display['Relevancy'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(qna_display, use_container_width=True)
                
                # Get SERP features details
                keyword_features = serp_features_df[serp_features_df['keyword'] == st.session_state.selected_keyword]
                if not keyword_features.empty:
                    st.subheader("📊 Other SERP Features")
                    
                    # Format features data for display
                    features_display = keyword_features[['feature_type', 'owns_feature', 'count', 'details']].copy()
                    features_display.columns = ['Feature Type', 'Owned', 'Count', 'Details']
                    features_display['Owned'] = features_display['Owned'].map({True: '✅', False: '❌'})
                    
                    st.dataframe(features_display, use_container_width=True)
                else:
                    st.info("No SERP features data available for this keyword.")
            else:
                st.info("👆 Click on a row above to view detailed Q&A and feature breakdown.")
        else:
            st.warning("No data available for enhanced overview. Run SERP fetches to populate the rank tracker.")
        
        # Legacy drill-down functionality (kept for backward compatibility)
        st.subheader("🔍 Legacy Keyword Drill-Down")
        selected_keyword_legacy = st.selectbox(
            "Select keyword for detailed analysis:",
            options=sorted(trends_df['keyword'].unique()) if not trends_df.empty else [],
            index=0
        )
        
        if selected_keyword_legacy:
            keyword_data = trends_df[trends_df['keyword'] == selected_keyword_legacy].sort_values('timestamp')
            
            if not keyword_data.empty:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Keyword:** {selected_keyword_legacy}")
                    
                    latest = keyword_data.iloc[-1]
                    st.metric("Current Score", f"{latest['total_score']:.1f}")
                    st.metric("Organic Score", f"{latest['organic_score']:.1f}")
                    st.metric("PAA Score", f"{latest['paa_score']:.1f}")
                    st.metric("Feature Score", f"{latest['feature_score']:.1f}")
                    st.metric("Share of Voice", f"{latest['share_of_voice']:.1f}%")
                    st.metric("Potential Ceiling", f"{latest['potential_ceiling']:.1f}")
                    
                    # Performance band
                    percentile = calculate_percentiles(st.session_state.selected_project_id, selected_keyword_legacy, 30)
                    st.metric("30-Day Percentile", f"{percentile.get('percentile_30d', 0):.1f}%")
                    st.write(f"**Performance:** {get_performance_band(percentile.get('percentile_30d', 0))}")
                
                with col2:
                    # Trend charts
                    if len(keyword_data) > 1:
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Score Trends", "Feature Breakdown"),
                            vertical_spacing=0.1
                        )
                        
                        # Score trends
                        fig.add_trace(
                            go.Scatter(
                                x=keyword_data['timestamp'],
                                y=keyword_data['total_score'],
                                name="Total Score",
                                line=dict(color='blue')
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=keyword_data['timestamp'],
                                y=keyword_data['organic_score'],
                                name="Organic Score",
                                line=dict(color='green')
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=keyword_data['timestamp'],
                                y=keyword_data['paa_score'],
                                name="PAA Score",
                                line=dict(color='orange')
                            ),
                            row=1, col=1
                        )
                        
                        # Feature breakdown
                        feature_cols = ['organic_score', 'paa_score', 'feature_score']
                        colors = ['green', 'orange', 'purple']
                        
                        for col, color in zip(feature_cols, colors):
                            fig.add_trace(
                                go.Bar(
                                    x=keyword_data['timestamp'],
                                    y=keyword_data[col],
                                    name=col.replace('_', ' ').title(),
                                    marker_color=color
                                ),
                                row=2, col=1
                            )
                        
                        fig.update_layout(height=600, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Only one data point available for this keyword")
                
                # Feature ownership history
                st.subheader("📋 Feature Ownership History")
                
                ownership_data = []
                for _, row in keyword_data.iterrows():
                    ownership_summary = row.get('ownership_summary', {})
                    timestamp = row['timestamp']
                    
                    for feature_type, owners in ownership_summary.items():
                        if project_domain in owners:
                            ownership_data.append({
                                'Date': timestamp,
                                'Feature': feature_type.replace('_', ' ').title(),
                                'Ownership': '✅ Owned',
                                'Count': len([o for o in owners if project_domain in o])
                            })
                
                if ownership_data:
                    ownership_df = pd.DataFrame(ownership_data)
                    st.dataframe(ownership_df, use_container_width=True)
                else:
                    st.info("No feature ownership data available")
                
                # Suggested actions
                st.subheader("💡 Suggested Actions")
                
                latest_score = latest['total_score']
                potential_ceiling = latest['potential_ceiling']
                gap = potential_ceiling - latest_score
                
                suggestions = []
                
                if gap > 10:
                    suggestions.append(f"🎯 **High potential gap**: {gap:.1f} points available")
                
                if latest['paa_score'] == 0:
                    suggestions.append("❓ **PAA opportunity**: No PAA answers owned")
                
                if latest['share_of_voice'] < 50:
                    suggestions.append("🏃 **Competitive gap**: Low share of voice")
                
                if latest['feature_score'] < 10:
                    suggestions.append("📊 **Feature opportunity**: Low feature visibility")
                
                if not suggestions:
                    suggestions.append("✅ **Strong performance**: Continue current strategy")
                
                for suggestion in suggestions:
                    st.write(suggestion)

with tab5:
    ################################################################################
    # Research & Clustering - Enhanced UI                                          
    ################################################################################

    st.header("🔬 Research & Clustering")
    st.markdown("**Process stored SERP data to extract and cluster PAA questions**")
    
    # Get latest SERP data from database
    db = get_db()
    
    # Configurable runs limit
    runs_limit = st.number_input(
        "Max runs to process (0 = all)", 
        min_value=0, 
        value=0, 
        help="Enter 0 to process all stored SERP runs"
    )
    
    query = (
        db.query(SerpRun)
        .options(joinedload(SerpRun.keyword))
        .filter(SerpRun.project_id == st.session_state.selected_project_id)
        .order_by(SerpRun.timestamp.desc())
    )
    
    if runs_limit > 0:
        latest_serp_runs = query.limit(runs_limit).all()
    else:
        latest_serp_runs = query.all()
    
    logging.info(f"Processing {len(latest_serp_runs)} SERP runs in Research")
    db.close()
    
    if not latest_serp_runs:
        st.info("No SERP data available. Run a Live SERP Fetch first.")
    else:
        # Extract keywords from SERP runs
        keywords = [run.keyword.keyword for run in latest_serp_runs]
        
        # Research processing controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            threshold = st.slider(
                "Relevancy Threshold", 
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Filter questions by relevancy score (pre-clustering)"
            )
        
        with col2:
                                 max_workers = st.number_input(
                         "Max Workers", 
                         min_value=1, 
                         max_value=5, 
                         value=3,
                         help="Number of parallel workers for processing (reduced for database stability)"
                     )
        
        with col3:
            re_run_button = st.button("🔄 Re-run Research", help="Clear cache and re-process all data")
        
        # Add backfill button
        col4, col5 = st.columns([1, 1])
        with col4:
            backfill_button = st.button("🔄 Backfill Q&A Records", help="Update existing Q&A records with correct search_volume and answered_by_site")
        
        with col5:
            if backfill_button:
                with st.spinner("Backfilling Q&A records..."):
                    backfill_qna_records(st.session_state.selected_project_id)
                    st.success("✅ Backfill completed! Q&A records updated with correct search_volume and answered_by_site.")
        
        # Show cache status
        if 'research_cache_timestamp' in st.session_state:
            st.info(f"📊 Results cached at {st.session_state.research_cache_timestamp}. Click 'Re-run' to refresh.")
        
        # Process data with caching
        if st.button("🚀 Process Research Data", type="primary", help="Process stored SERP data to extract and cluster PAA questions") or re_run_button:
            if re_run_button:
                # Clear cache for this project
                st.cache_data.clear()
                st.session_state.processed_qna_data = None
                st.session_state.clustered_data = None
                st.session_state.research_cache_timestamp = None
            
            # Progress tracking
            total = len(keywords)
            progress = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing SERP data and extracting PAA questions..."):
                logging.info("Starting PAA fetch and clustering")
                
                # Use cached processing function
                processed_df, cache_timestamp = process_research_cached(
                    project_id=st.session_state.selected_project_id,
                    keywords=keywords,
                    threshold=threshold,
                    max_workers=max_workers
                )
                
                if not processed_df.empty:
                    # Store processed data and cache timestamp
                    st.session_state.processed_qna_data = processed_df
                    st.session_state.clustered_data = processed_df
                    st.session_state.research_cache_timestamp = cache_timestamp
                    
                    # Update progress to 100%
                    progress.progress(1.0)
                    status_text.text(f"✅ Completed processing {len(processed_df)} Q&A pairs from {len(keywords)} keywords")
                    
                    st.success(f"✅ Extracted and processed {len(processed_df)} Q&A pairs from {len(keywords)} keywords")
                else:
                    st.error("No Q&A data could be extracted. Check your SearchAPI key.")
            
            # Clear progress indicators
            progress.empty()
            status_text.empty()
        
        # Show processed data if available
        if st.session_state.processed_qna_data is not None:
            qna_df = st.session_state.processed_qna_data
            
            # Generate embeddings if missing
            if 'embedding' not in qna_df.columns:
                with st.spinner("🔄 Generating embeddings for UMAP visualization..."):
                    model = load_embedding_model()
                    texts = qna_df['question'].astype(str).tolist()
                    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
                    qna_df['embedding'] = list(embeddings)
                    st.success(f"✅ Generated embeddings for {len(qna_df)} questions")
            
            st.subheader("📊 Q&A Analysis")
            
            # Filter by threshold (only if relevancy_score column exists)
            if 'relevancy_score' in qna_df.columns:
                filtered_df = qna_df[qna_df['relevancy_score'] >= threshold]
                st.info(f"Showing {len(filtered_df)} questions above threshold {threshold}")
            else:
                filtered_df = qna_df
                st.info(f"Showing all {len(filtered_df)} questions (no relevancy scores available)")
            
            # Multi-view dashboard
            view_mode = st.radio(
                "Select view mode",
                ["Graph View", "Table View", "Bubble Chart", "Heatmap", "UMAP Scatter", "Sunburst"],
                index=0,
                horizontal=True
            )
            st.markdown("---")
            
            # Display results based on view mode
            if st.session_state.processed_qna_data is not None:
                clustered_df = st.session_state.processed_qna_data
                
                if view_mode == "Graph View":
                    st.subheader("🕸️ Interactive Question Graph")
                    
                    # Graph controls and legend
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        if st.button("🔄 Refresh Graph", help="Regenerate the graph layout"):
                            st.rerun()
                    
                    with col2:
                        if st.button("🎯 Reset Zoom", help="Reset graph zoom and position"):
                            st.rerun()
                    
                    with col3:
                        st.info("📊 **Legend**")
                        st.markdown("""
                        - ⭐ **Star nodes**: Seed keywords
                        - 🔵 **Circle nodes**: Questions (size = relevancy)
                        - 🟢 **Green edges**: High relevancy (>0.7)
                        - ⚫ **Gray edges**: Low relevancy
                        """)
                    
                    # Sidebar for SERP details with Contentus.ai styling
                    with st.sidebar:
                        st.markdown("""
                        <div style="background: #2A2C32; padding: 16px; border-radius: 8px; border-left: 4px solid #5C7AEA; margin-bottom: 16px;">
                            <h3 style="color: #7F5AF0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600;">🔍 SERP Details</h3>
                            <p style="color: #E0E0E0; margin: 0; font-size: 14px;">Click on a seed keyword node to see SERP details</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a placeholder for SERP details
                        if 'selected_keyword' not in st.session_state:
                            pass
                            st.session_state.selected_keyword = None
                        
                        # For testing - allow manual keyword selection
                        test_keyword = st.text_input("Test Keyword (for demo)", placeholder="Enter a keyword to test")
                        if test_keyword:
                            pass
                            st.session_state.selected_keyword = test_keyword
                        
                        if st.session_state.selected_keyword:
                        
                            pass
                            keyword = st.session_state.selected_keyword
                            serp_details = get_serp_details_for_keyword(keyword, st.session_state.selected_project_id)
                            
                            # Enhanced header with detailed keyword info
                            st.markdown(f"""
                            <div style="background: #2A2C32; padding: 16px; border-radius: 8px; border-left: 4px solid #5C7AEA; margin-bottom: 16px;">
                                <h3 style="color: #7F5AF0; margin: 0 0 12px 0; font-size: 18px; font-weight: 600;">📊 {keyword}</h3>
                                <p style="color: #E0E0E0; margin: 0; font-size: 14px;">Click on question nodes to see details</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Calculate answered vs unanswered questions for this keyword
                            keyword_questions = clustered_df[clustered_df['keyword'] == keyword]
                            answered_count = 0
                            total_count = len(keyword_questions)
                            
                            for _, row in keyword_questions.iterrows():
                                answer_text = row.get('answer', '').strip()
                                if len(answer_text) > 10 and not answer_text.lower().startswith(("no answer", "not found", "n/a")):
                                    pass
                                    answered_count += 1
                            
                            # SERP details display with enhanced styling
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                <div style="background: #1F2023; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                                    <div style="color: #E0E0E0; font-size: 12px; margin-bottom: 4px;">Current Position</div>
                                    <div style="color: #FFFFFF; font-size: 16px; font-weight: 600;">#{serp_details['rank']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background: #1F2023; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                                    <div style="color: #E0E0E0; font-size: 12px; margin-bottom: 4px;">SERP Real-Estate Score</div>
                                    <div style="color: #FFFFFF; font-size: 16px; font-weight: 600;">{serp_details['serp_score']}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                visibility_icon = "✅" if serp_details['visible'] else "❌"
                                visibility_color = "#0EBFC4" if serp_details['visible'] else "#6C6F7D"
                                
                                st.markdown(f"""
                                <div style="background: #1F2023; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                                    <div style="color: #E0E0E0; font-size: 12px; margin-bottom: 4px;">Visibility</div>
                                    <div style="color: {visibility_color}; font-size: 16px; font-weight: 600;">{visibility_icon} {'Yes' if serp_details['visible'] else 'No'}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                answered_color = "#0EBFC4" if answered_count > total_count/2 else "#FF4444"
                                st.markdown(f"""
                                <div style="background: #1F2023; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                                    <div style="color: #E0E0E0; font-size: 12px; margin-bottom: 4px;">Answered Questions</div>
                                    <div style="color: {answered_color}; font-size: 16px; font-weight: 600;">{answered_count}/{total_count}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # SERP Features with Contentus.ai styling
                            if serp_details['features']:
                                pass
                                st.markdown("""
                                <div style="background: #2A2C32; padding: 16px; border-radius: 8px; border-left: 4px solid #0EBFC4; margin-bottom: 16px;">
                                    <h4 style="color: #0EBFC4; margin: 0 0 12px 0; font-size: 16px; font-weight: 600;">🎯 SERP Features</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for feature in serp_details['features']:
                                    st.markdown(f"""
                                    <div style="background: #1F2023; padding: 8px 12px; border-radius: 4px; margin-bottom: 4px;">
                                        <span style="color: #E0E0E0; font-size: 14px;">• <strong>{feature['type']}</strong>: {feature['count']} found</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background: #1F2023; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                                    <span style="color: #6C6F7D; font-size: 14px;">No SERP features data available</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Clear selection button with Contentus.ai styling
                            if st.button("❌ Clear Selection", key="clear_selection"):
                                pass
                                st.session_state.selected_keyword = None
                                st.rerun()
                        else:
                            st.markdown("""
                            <div style="background: #1F2023; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                                <span style="color: #6C6F7D; font-size: 14px;">Click on a seed keyword node to see SERP details</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Build interactive graph with enhanced JavaScript for node clicking
                    graph_html = build_graph(clustered_df)
                    
                    # Add JavaScript for node clicking functionality
                    enhanced_html = graph_html.replace(
                        '</body>',
                        '''
                        <script>
                        // Wait for the network to be ready
                        document.addEventListener('DOMContentLoaded', function() {
                            // Find the network container
                            const networkContainer = document.querySelector('.vis-network');
                            if (networkContainer) {
                                // Add click event listener
                                networkContainer.addEventListener('click', function(event) {
                                    // Check if a node was clicked
                                    const target = event.target;
                                    if (target.classList.contains('vis-node')) {
                                        // Get the node label (keyword)
                                        const nodeLabel = target.getAttribute('title');
                                        if (nodeLabel && nodeLabel.includes('Seed Keyword')) {
                                            // Extract keyword from tooltip
                                            const keywordMatch = nodeLabel.match(/Keyword:</strong> ([^<]+)/);
                                            if (keywordMatch) {
                                                const keyword = keywordMatch[1];
                                                // Send message to Streamlit
                                                window.parent.postMessage({
                                                    type: 'keyword_selected',
                                                    keyword: keyword
                                                }, '*');
                                            }
                                        }
                                    }
                                });
                            }
                        });
                        </script>
                        </body>
                        '''
                    )
                    
                    # Display graph with enhanced styling
                    st.components.v1.html(
                        enhanced_html,
                        height=700,
                        scrolling=True
                    )
                    
                    # Streamlit sidebar integration for click events
                    st.sidebar.subheader("📊 Node Details")
                    
                    # Initialize session state for click events
                    if 'seed_click' not in st.session_state:
                        st.session_state.seed_click = None
                    if 'question_click' not in st.session_state:
                        st.session_state.question_click = None
                    
                    # Handle click events from the graph
                    if st.session_state.seed_click:
                        seed_data = st.session_state.seed_click
                        st.sidebar.markdown("### 🌱 Seed Keyword")
                        st.sidebar.write(f"**Keyword:** {seed_data.get('keyword', 'Unknown')}")
                        st.sidebar.write(f"**Rank:** {seed_data.get('rank', 3)}")
                        st.sidebar.write(f"**SERP Score:** {seed_data.get('serp_score', 45)}")
                        st.sidebar.write(f"**Features:** {', '.join(seed_data.get('features', ['PAA', 'Featured Snippet']))}")
                        st.sidebar.write(f"**Answered:** {seed_data.get('answered', '0/0')}")
                        
                        # Clear the click event
                        st.session_state.seed_click = None
                    
                    elif st.session_state.question_click:
                        question_data = st.session_state.question_click
                        st.sidebar.markdown("### ❓ Question Details")
                        st.sidebar.write(f"**Question:** {question_data.get('question', 'Unknown')}")
                        st.sidebar.write(f"**Answer:** {question_data.get('answer', 'No answer available')}")
                        st.sidebar.write(f"**Answered:** {'✅ Yes' if question_data.get('answered', False) else '❌ No'}")
                        st.sidebar.write(f"**Relevancy Score:** {question_data.get('relevancy_score', 0.5):.2f}")
                        st.sidebar.write(f"**Cluster ID:** {question_data.get('cluster_id', -1)}")
                        st.sidebar.write(f"**Keyword:** {question_data.get('keyword', 'Unknown')}")
                        
                        # Clear the click event
                        st.session_state.question_click = None
                    
                    else:
                        st.sidebar.info("👆 Click any node in the graph to see details here")
                    
                    # Add JavaScript message listener for Streamlit
                    st.markdown("""
                    <script>
                    window.addEventListener('message', function(event) {
                        if (event.data.type === 'seed_click') {
                            // Store seed click data in session state
                            window.parent.postMessage({
                                type: 'streamlit:setSessionState',
                                key: 'seed_click',
                                value: event.data.payload
                            }, '*');
                        } else if (event.data.type === 'question_click') {
                            // Store question click data in session state
                            window.parent.postMessage({
                                type: 'streamlit:setSessionState',
                                key: 'question_click',
                                value: event.data.payload
                            }, '*');
                        }
                    });
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Handle session state updates from JavaScript
                    if st.button("🔄 Refresh Sidebar", help="Refresh sidebar to see latest click events"):
                        st.rerun()
                    
                    # Add a callback to handle click events
                    def handle_graph_click():
                        # This will be called when the graph sends a message
                        pass
                    
                    # Add a container for dynamic updates
                    click_container = st.empty()
                    with click_container:
                        st.info("👆 Click any node in the graph to see details in the sidebar")
                
                elif view_mode == "Table View":
                    st.subheader("📋 Clustered Questions Table")
                    # Select columns that exist - use 'keyword' instead of 'seed_keyword'
                    display_columns = ['keyword', 'question', 'answer', 'search_volume', 'answered_by_site']
                    if 'relevancy_score' in clustered_df.columns:
                        pass
                        display_columns.append('relevancy_score')
                    if 'cluster_id' in clustered_df.columns:
                        pass
                        display_columns.append('cluster_id')
                    elif 'cluster' in clustered_df.columns:
                        display_columns.append('cluster')
                    
                    # Only include columns that actually exist in the DataFrame
                    available_columns = [col for col in display_columns if col in clustered_df.columns]
                    if available_columns:
                        pass
                        # Format the dataframe for better display with search_volume
                        display_df = clustered_df[available_columns].copy()
                        if 'search_volume' in display_df.columns:
                            display_df['search_volume'] = display_df['search_volume'].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
                        if 'relevancy_score' in display_df.columns:
                            display_df['relevancy_score'] = display_df['relevancy_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Add search volume summary metrics
                        if 'search_volume' in clustered_df.columns:
                            total_sv = clustered_df['search_volume'].sum()
                            avg_sv = clustered_df['search_volume'].mean()
                            st.metric("Total Search Volume", f"{total_sv:,}")
                            st.metric("Average Search Volume per Question", f"{avg_sv:.0f}")
                    else:
                        st.warning("No displayable columns found in the clustered data")
                
                elif view_mode == "Bubble Chart":
                    st.subheader("🫧 Keyword PAA Coverage Bubble Chart")
                    
                    # Check available columns and create appropriate aggregation
                    available_columns = list(clustered_df.columns)
                    st.info(f"📊 Available columns: {available_columns}")
                    
                    # Prepare data for bubble chart with safe column access
                    agg_data = []
                    for keyword in clustered_df['keyword'].unique():
                        keyword_data = clustered_df[clustered_df['keyword'] == keyword]
                        total_qas = len(keyword_data)
                        
                        # Calculate answered count based on answer content
                        answered_count = 0
                        for _, row in keyword_data.iterrows():
                            answer_text = str(row.get('answer', '')).strip()
                            if len(answer_text) > 10 and not answer_text.lower().startswith(("no answer", "not found", "n/a", "none")):
                                answered_count += 1
                        
                        # Calculate average relevancy if column exists
                        avg_relevancy = 0.5  # default
                        if 'relevancy_score' in keyword_data.columns:
                            avg_relevancy = keyword_data['relevancy_score'].mean()
                        
                        # Calculate total search volume for this keyword
                        total_search_volume = keyword_data['search_volume'].sum() if 'search_volume' in keyword_data.columns else 0
                        
                        agg_data.append({
                            'keyword': keyword,
                            'total_qas': total_qas,
                            'answered_count': answered_count,
                            'avg_relevancy': avg_relevancy,
                            'total_search_volume': total_search_volume
                        })
                    
                    agg = pd.DataFrame(agg_data)
                    agg["pct_answered"] = agg["answered_count"] / agg["total_qas"]
                    
                    fig = px.scatter(
                        agg,
                        x="pct_answered",
                        y="avg_relevancy",
                        size="total_search_volume",  # Use search volume for bubble size
                        hover_name="keyword",
                        color="pct_answered",
                        color_continuous_scale=["red","teal"],
                        labels={"pct_answered":"% Answered","avg_relevancy":"Avg. Relevancy","total_search_volume":"Search Volume"},
                        title="Keyword PAA Coverage Bubble Chart (Size = Search Volume)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif view_mode == "Heatmap":
                    st.subheader("🔥 Answered Questions per Cluster Heatmap")
                    
                    # Check if cluster_id column exists
                    if 'cluster_id' not in clustered_df.columns:
                        st.warning("⚠️ No cluster_id column found. Heatmap requires clustering data.")
                        st.info("Please run Research processing with clustering enabled.")
                        st.stop()
                    
                    # Create answered column based on answer content
                    clustered_df['answered'] = clustered_df['answer'].apply(
                        lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
                    )
                    
                    # Ensure answered column exists before groupby
                    if 'answered' not in clustered_df.columns:
                        clustered_df['answered'] = clustered_df['answer'].apply(
                            lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
                        )
                    
                    # Use answered_by_site column if available, otherwise create answered column
                    if "answered_by_site" in clustered_df.columns:
                        pivot = clustered_df.groupby(["keyword","cluster_id"])["answered_by_site"].sum().unstack(fill_value=0)
                    else:
                        # Ensure answered column exists before groupby
                        clustered_df = ensure_answered_column(clustered_df)
                        # Prepare data for heatmap
                        pivot = clustered_df.groupby(["keyword","cluster_id"])["answered"].sum().unstack(fill_value=0)
                    top_n = st.slider("Seeds to show", 5, min(100, pivot.shape[0]), 20)
                    top_seeds = pivot.sum(axis=1).nlargest(top_n).index
                    
                    # Fix the KeyError by ensuring the 'answered' column exists before aggregation
                    # Around line 4887, replace the problematic aggregation with safer code

                    # Create answered column based on answer content - ensure it exists
                    if 'answered' not in clustered_df.columns:
                        clustered_df['answered'] = clustered_df['answer'].apply(
                            lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
                        )
                    
                    # Prepare data for heatmap with error handling
                    try:
                        # Ensure answered column exists before groupby
                        if 'answered' not in clustered_df.columns:
                            clustered_df['answered'] = clustered_df['answer'].apply(
                                lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
                            )
                        
                        if "answered_by_site" in clustered_df.columns:
                            pivot = clustered_df.groupby(["keyword","cluster_id"])["answered_by_site"].sum().unstack(fill_value=0)
                        else:
                            pivot = clustered_df.groupby(["keyword","cluster_id"])["answered"].sum().unstack(fill_value=0)
                        top_n = st.slider("Seeds to show", 5, min(100, pivot.shape[0]), 20)
                        top_seeds = pivot.sum(axis=1).nlargest(top_n).index
                        
                        fig2 = px.imshow(
                            pivot.loc[top_seeds],
                            labels=dict(x="Cluster ID", y="Seed Keyword", color="Answered Count"),
                            color_continuous_scale="Greens",
                            aspect="auto",
                            title="Answered Questions per Cluster Heatmap"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {str(e)}")
                        st.info("This might be due to insufficient data or memory constraints.")
                
                elif view_mode == "UMAP Scatter":
                    st.subheader("🗺️ UMAP Seed-Level Embedding Scatter")
                    
                    # Build seed-level aggregation
                    seed_agg = build_seed_aggregation(clustered_df)
                    
                    # Performance slider for limiting seeds
                    max_seeds = st.slider("Max Seeds to Display", 10, 100, 50, help="Limit the number of seeds for better performance")
                    seed_agg_limited = seed_agg.nlargest(max_seeds, 'total_volume')
                    
                    try:
                        import numpy as np
                        from umap import UMAP
                        
                        @st.cache_data
                        def compute_umap(embeddings):
                            reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                            return reducer.fit_transform(np.vstack(embeddings))
                        
                        # Compute seed-level embeddings by averaging question embeddings per seed
                        seed_embeddings = []
                        for seed in seed_agg_limited['seed_keyword']:
                            seed_questions = clustered_df[clustered_df['keyword'] == seed]
                            if not seed_questions.empty and 'embedding' in seed_questions.columns:
                                # Average the embeddings for this seed's questions
                                question_embeddings = np.vstack(seed_questions['embedding'].tolist())
                                seed_embedding = np.mean(question_embeddings, axis=0)
                                seed_embeddings.append(seed_embedding)
                            else:
                                # Fallback: use zero vector if no embeddings found
                                seed_embeddings.append(np.zeros(384))  # Assuming 384-dim embeddings
                        
                        # Compute UMAP coordinates
                        coords = compute_umap(seed_embeddings)
                        seed_agg_limited["x"], seed_agg_limited["y"] = coords[:,0], coords[:,1]
                        
                        # Create the scatter plot with improved sizing and layout
                        fig = px.scatter(
                            seed_agg_limited,
                            x="x", 
                            y="y",
                            size="total_volume",
                            size_max=40,           # increase maximum bubble radius
                            color="coverage_color",
                            color_discrete_map={"red":"#F24E1E","orange":"#FF8A5B","green":"#0EBFC4"},
                            hover_data=["seed_keyword","total_qas","answered_qas","pct_answered","total_volume"],
                            title="UMAP of Seed Keywords by Semantic Cluster"
                        )
                        
                        # Set minimum size using update_traces
                        fig.update_traces(marker=dict(sizemin=12))
                        
                        # Adjust layout for clarity
                        fig.update_layout(
                            height=700,
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(title="Coverage"),
                            template="plotly_dark"
                        )
                        
                        # Force fixed axis ranges (reduce compression)
                        x0, x1 = seed_agg_limited['x'].min(), seed_agg_limited['x'].max()
                        y0, y1 = seed_agg_limited['y'].min(), seed_agg_limited['y'].max()
                        fig.update_xaxes(range=[x0 - 1, x1 + 1])
                        fig.update_yaxes(range=[y0 - 1, y1 + 1])
                        
                        # Add click interaction
                        chart = st.plotly_chart(fig, use_container_width=True)
                        
                        # Handle click events for drill-down
                        if chart:
                            click_data = st.session_state.get("umap_click", None)
                            if click_data and "points" in click_data:
                                selected_seed = click_data["points"][0]["customdata"][0]  # seed_keyword
                                
                                # Show selected seed's questions in sidebar with enhanced search_volume display
                                st.sidebar.header(f"Questions for {selected_seed}")
                                seed_questions = clustered_df[clustered_df['keyword'] == selected_seed]
                                if not seed_questions.empty:
                                    # Get total search volume for this seed
                                    total_sv = seed_questions['search_volume'].sum() if 'search_volume' in seed_questions.columns else 0
                                    st.sidebar.metric("Total Search Volume", f"{total_sv:,}")
                                    
                                    display_cols = ["question", "answered_by_site", "search_volume", "answer_relevancy"]
                                    available_cols = [col for col in display_cols if col in seed_questions.columns]
                                    
                                    # Format the dataframe for better display
                                    display_df = seed_questions[available_cols].copy()
                                    if 'search_volume' in display_df.columns:
                                        display_df['search_volume'] = display_df['search_volume'].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
                                    if 'answer_relevancy' in display_df.columns:
                                        display_df['answer_relevancy'] = display_df['answer_relevancy'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
                                    
                                    st.sidebar.dataframe(display_df, use_container_width=True)
                                else:
                                    st.sidebar.info("No questions found for this seed keyword.")
                        
                    except ImportError:
                        st.error("UMAP library not installed. Please install with: pip install umap-learn")
                        st.info("UMAP provides dimensionality reduction for high-dimensional embedding data.")
                
                elif view_mode == "Sunburst":
                    st.subheader("☀️ Seed Coverage Sunburst Chart")
                    
                    # Build seed-level aggregation
                    seed_agg = build_seed_aggregation(clustered_df)
                    
                    # Performance slider for limiting seeds
                    max_seeds = st.slider("Max Seeds to Display", 10, 100, 50, help="Limit the number of seeds for better performance")
                    seed_agg_limited = seed_agg.nlargest(max_seeds, 'total_volume')
                    
                    # Prepare sunburst data with three levels: seed_keyword, coverage_str, total_volume
                    sunburst_df = seed_agg_limited.assign(
                        coverage_str=seed_agg_limited.coverage_color.map({"red":"None","orange":"Partial","green":"Full"})
                    )
                    
                    fig2 = px.sunburst(
                        sunburst_df,
                        path=["seed_keyword","coverage_str"],
                        values="total_volume",
                        color="coverage_str",
                        color_discrete_map={"None":"#F24E1E","Partial":"#FF8A5B","Full":"#0EBFC4"},
                        title="Seed Coverage Sunburst (by Search Volume)",
                        maxdepth=2,
                        branchvalues="total"
                    )
                    
                    # Increase figure height & label font sizes
                    fig2.update_layout(
                        height=700,
                        margin=dict(l=10, r=10, t=50, b=10),
                        uniformtext=dict(minsize=12, mode="hide"),
                        template="plotly_dark"
                    )
                    
                    # Show labels at all levels
                    fig2.update_traces(textinfo="label+percent entry", insidetextorientation="radial")
                    
                    # Add click interaction
                    chart = st.plotly_chart(fig2, use_container_width=True)
                    
                    # Handle click events for drill-down
                    if chart:
                        click_data = st.session_state.get("sunburst_click", None)
                        if click_data and "points" in click_data:
                            selected_seed = click_data["points"][0]["label"]
                            
                            # Show selected seed's questions in sidebar with enhanced search_volume display
                            st.sidebar.header(f"Questions for {selected_seed}")
                            seed_questions = clustered_df[clustered_df['keyword'] == selected_seed]
                            if not seed_questions.empty:
                                # Get total search volume for this seed
                                total_sv = seed_questions['search_volume'].sum() if 'search_volume' in seed_questions.columns else 0
                                st.sidebar.metric("Total Search Volume", f"{total_sv:,}")
                                
                                display_cols = ["question", "answered_by_site", "search_volume", "answer_relevancy"]
                                available_cols = [col for col in display_cols if col in seed_questions.columns]
                                
                                # Format the dataframe for better display
                                display_df = seed_questions[available_cols].copy()
                                if 'search_volume' in display_df.columns:
                                    display_df['search_volume'] = display_df['search_volume'].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
                                if 'answer_relevancy' in display_df.columns:
                                    display_df['answer_relevancy'] = display_df['answer_relevancy'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
                                
                                st.sidebar.dataframe(display_df, use_container_width=True)
                            else:
                                st.sidebar.info("No questions found for this seed keyword.")
            
            # Export options
            if 'clustered_data' in st.session_state and st.session_state.clustered_data is not None:
                pass
                st.subheader("💾 Export Options")
                export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
                
                if st.button("📥 Export Data"):
                
                    pass
                    export_data = export_qna(st.session_state.clustered_data, export_format.lower())
                    st.download_button(
                        label=f"Download {export_format}",
                        data=export_data,
                        file_name=f"clustered_qa_data.{export_format.lower()}",
                        mime="application/octet-stream"
                    )

with tab6:
    ################################################################################
    # Prompt Builder                                                                
    ################################################################################

    st.header("✍️ Prompt Builder")
    
    if st.session_state.processed_qna_data is not None:
    
        pass
        qna_df = st.session_state.processed_qna_data
        
        # Prompt builder controls
        st.subheader("🎯 Prompt Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            grouping_key = st.selectbox(
                "Group by",
                ["keyword", "cluster"],
                help="Choose how to group questions for prompt generation"
            )
        
        with col2:
            threshold = st.slider(
                "Relevancy Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
        
        if st.button("🔄 Generate Prompts"):
        
            pass
            with st.spinner("Generating prompts..."):
                # Prepare data for prompt generation
                export_data = prepare_export_data(qna_df, grouping_key, threshold)
                
                if not export_data.empty:
                
                    pass
                    st.subheader("📝 Generated Prompts")
                    
                    # Show prompts by group
                    groups = get_groups(export_data, grouping_key)
                    
                    for group in groups:
                        group_data = export_data[export_data[grouping_key] == group]
                        
                        with st.expander(f"📋 {group} ({len(group_data)} questions)"):
                            # Template for prompt generation
                            template = """
                            Based on the following questions and answers about {topic}, create comprehensive content that addresses these user queries:
                            
                            Questions and Answers:
                            {qa_list}
                            
                            Please provide detailed, helpful content that naturally incorporates these Q&As.
                            """
                            
                            # Fill template
                            qa_list = "\n".join([
                                f"Q: {row['question']}\nA: {row['answer']}"
                                for _, row in group_data.iterrows()
                            ])
                            
                            filled_prompt = template.format(
                                topic=group,
                                qa_list=qa_list
                            )
                            
                            st.text_area("Generated Prompt", filled_prompt, height=200, key=f"prompt_{group}")
                            
                            # Copy button
                            if st.button(f"📋 Copy Prompt", key=f"copy_{group}"):
                                pass
                                st.write("Prompt copied to clipboard!")
                else:
                    st.warning("No data available for prompt generation with current settings.")
    else:
        st.info("No processed Q&A data available. Run Research & Clustering first.")

with tab7:
    ################################################################################
    # Taxonomy Mapping - Enhanced with Feedback                                    
    ################################################################################

    st.header("🗂️ Taxonomy Mapping")
    st.markdown("**Map Q&A data to your website's taxonomy structure**")
    
    # Taxonomy loading section
    st.subheader("📂 Load Taxonomy")

    col1, col2 = st.columns(2)

    # --- Load from sitemap -------------------------------------------------
    with col1:
        sitemap_url = st.text_input(
            "Sitemap URL",
            placeholder="https://example.com/sitemap.xml",
            help="Enter your website's sitemap URL",
        )

        if st.button("🌐 Load from Sitemap", type="primary", key="load_sitemap"):
            if sitemap_url:
                with st.spinner("Loading sitemap and extracting taxonomy..."):
                    try:
                        taxonomy_df = load_taxonomy(sitemap_url=sitemap_url)
                        if not taxonomy_df.empty:
                            st.session_state.taxonomy_data = taxonomy_df
                            st.success(f"✅ Loaded {len(taxonomy_df)} URLs from sitemap")

                            # Preview sample data
                            st.subheader("📋 Sample Taxonomy Data")
                            st.dataframe(taxonomy_df.head(10), use_container_width=True)

                            # Audit log
                            log_audit_action(
                                user_id=current_user_data["id"],
                                action=AuditAction.TAXONOMY_LOAD.value,
                                target_type="project",
                                target_id=st.session_state.selected_project_id,
                                details=f"Loaded taxonomy from {sitemap_url}",
                            )
                        else:
                            st.error("No valid URLs found in sitemap")
                    except Exception as e:
                        st.error(f"Failed to load sitemap: {e}")
            else:
                st.error("Please enter a sitemap URL")

    # --- Load from CSV ------------------------------------------------------
    with col2:
        uploaded_taxonomy = st.file_uploader(
            "Or upload taxonomy CSV",
            type=["csv"],
            help="Upload a CSV file with URL and category columns",
        )

        if uploaded_taxonomy is not None:
            with st.spinner("Loading taxonomy from CSV..."):
                try:
                    taxonomy_df = load_taxonomy(file=uploaded_taxonomy)
                    if not taxonomy_df.empty:
                        st.session_state.taxonomy_data = taxonomy_df
                        st.success(f"✅ Loaded {len(taxonomy_df)} entries from CSV")

                        st.subheader("📋 Sample Taxonomy Data")
                        st.dataframe(taxonomy_df.head(10), use_container_width=True)
                    else:
                        st.error("No valid data found in CSV")
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")

    # -----------------------------------------------------------------------
    # Taxonomy mapping section
    # -----------------------------------------------------------------------
    if (
        st.session_state.taxonomy_data is not None
        and st.session_state.processed_qna_data is not None
    ):
        st.subheader("🔗 Map Q&A to Taxonomy")

        if st.button("🔄 Run Taxonomy Mapping", key="run_taxonomy_mapping"):
            with st.spinner("Mapping Q&A data to taxonomy..."):
                taxonomy_df = st.session_state.taxonomy_data
                qna_df = st.session_state.processed_qna_data

                mapped_df = match_qna_to_taxonomy(qna_df, taxonomy_df)

                if not mapped_df.empty:
                    st.success(
                        f"✅ Successfully mapped {len(mapped_df)} Q&A pairs to taxonomy"
                    )

                    # Summary statistics
                    st.subheader("📊 Mapping Results")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total Q&A Pairs", len(mapped_df))
                    mapped_count = mapped_df["mapped_category"].notna().sum()
                    col_b.metric("Successfully Mapped", mapped_count)
                    col_c.metric("Unmapped", len(mapped_df) - mapped_count)

                    st.dataframe(mapped_df.head(20), use_container_width=True)
                else:
                    st.warning("No mappings were created for the selected data.")
    else:
        st.info("Please load both taxonomy and processed Q&A data to start mapping.")

with tab8:
    ################################################################################
    # Projects Management                                                          
    ################################################################################

    st.header("⚙️ Project Management")
    
    if can_manage:
    
        pass
        st.subheader("👥 Team Management")
        
        # Current members
        members_df = get_project_members(st.session_state.selected_project_id)
        if not members_df.empty:
            st.dataframe(members_df, use_container_width=True)
        else:
            st.info("No members yet.")

        # Invite new member
        with st.expander("Invite New Member"):
            with st.form(key="invite_member_form_mgmt"):
                invite_email = st.text_input("Email Address")
                invite_role = st.selectbox("Role", ["Editor", "Viewer"])
                if st.form_submit_button("Send Invite"):
                    if invite_email:
                        try:
                            role_enum = UserRole.EDITOR if invite_role == "Editor" else UserRole.VIEWER
                            invite_member(
                                project_id=st.session_state.selected_project_id,
                                email=invite_email,
                                role=role_enum,
                                invited_by_user_id=current_user_data["id"],
                            )
                            st.success("Invite sent successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to send invite: {e}")
                    else:
                        st.error("Please enter an email address")

        # Project settings
        st.subheader("⚙️ Project Settings")
        with st.form(key="project_settings_form"):
            new_name = st.text_input("Project Name", value=current_project["name"])
            new_description = st.text_area(
                "Description", value=current_project.get("description", "")
            )
            location_options = [
                "us", "uk", "de", "fr", "es", "it", "ca", "au", "nl", "br", "mx", "jp", "kr", "cn", "in",
                "se", "no", "dk", "fi", "pl", "cz", "hu", "ro", "bg", "hr", "rs", "sk", "si", "ee", "lv",
                "lt", "ie", "nz", "za", "sg", "my", "ph", "th", "vn", "id", "tr", "gr", "pt", "be", "ch",
                "at", "lu", "mt", "cy",
            ]
            new_location = st.selectbox(
                "Location",
                location_options,
                index=location_options.index(current_project["location"])
                if current_project["location"] in location_options
                else 0,
            )
            new_domain = st.text_input("Domain", value=current_project.get("domain", ""))

            if st.form_submit_button("Update Project"):
                # TODO: implement update_project logic
                st.success("Project updated successfully!")
                st.rerun()

    # Activity log (visible to all project users)
    st.subheader("📋 Activity Log")
    audit_logs = get_audit_logs(
        project_id=st.session_state.selected_project_id, limit=50
    )
    if not audit_logs.empty:
        st.dataframe(audit_logs, use_container_width=True)
    else:
        st.info("No activity logs found")

def ensure_answered_column(df):
    if 'answered' not in df.columns:
        df['answered'] = df['answer'].apply(
            lambda x: 1 if len(str(x).strip()) > 10 and not str(x).lower().startswith(("no answer", "not found", "n/a", "none")) else 0
        )
    return df

def backfill_qna_records(project_id: int):
    """Backfill existing QnaRecord entries with correct search_volume and answered_by_site values.
    
    This function re-processes all stored SERP data for a project to update existing
    QnaRecord entries with the correct search_volume from Keyword table and 
    answered_by_site based on domain matching.
    """
    try:
        db = get_db()
        
        # Get all keywords for this project
        keywords = db.query(Keyword).filter(Keyword.project_id == project_id).all()
        
        if not keywords:
            logger.info(f"No keywords found for project {project_id}")
            return
        
        logger.info(f"Starting backfill for {len(keywords)} keywords in project {project_id}")
        
        updated_count = 0
        for keyword_obj in keywords:
            keyword = keyword_obj.keyword
            search_volume = keyword_obj.search_volume or 0
            
            # Get project domain for site coverage detection
            project = db.query(Project).filter(Project.id == project_id).first()
            project_domain = project.domain if project else None
            
            # Get the most recent SERP run for this keyword
            serp_run = db.query(SerpRun).filter(
                SerpRun.project_id == project_id,
                SerpRun.keyword_id == keyword_obj.id
            ).order_by(SerpRun.timestamp.desc()).first()
            
            if not serp_run or not serp_run.raw_json:
                continue
            
            serp_json = json.loads(serp_run.raw_json)
            parsed_data = parse_serp_features(serp_json)
            paa_questions = parsed_data.get("paa_questions", [])
            
            for qa in paa_questions:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                source_url = qa.get("source_url", "")
                
                if question and answer:
                    # Improved domain matching - handle both www and non-www
                    answered_by_site = False
                    if project_domain and source_url:
                        source_domain = extract_domain(source_url)
                        # Normalize domains for comparison (remove www)
                        normalized_project_domain = project_domain.replace("www.", "")
                        normalized_source_domain = source_domain.replace("www.", "")
                        answered_by_site = normalized_project_domain == normalized_source_domain
                    
                    # Compute answer relevancy
                    answer_relevancy = compute_answer_relevancy(question, answer)
                    
                    # Update existing QnaRecord or create new one
                    existing_record = db.query(QnaRecord).filter(
                        QnaRecord.project_id == project_id,
                        QnaRecord.keyword == keyword,
                        QnaRecord.question == question
                    ).first()
                    
                    if existing_record:
                        # Update existing record
                        existing_record.search_volume = search_volume
                        existing_record.answered_by_site = answered_by_site
                        existing_record.answer_relevancy = answer_relevancy
                        existing_record.source_url = source_url
                        updated_count += 1
                    else:
                        # Create new record
                        qna_record = QnaRecord(
                            project_id=project_id,
                            keyword=keyword,
                            question=question,
                            answer=answer,
                            search_volume=search_volume,
                            answered_by_site=answered_by_site,
                            answer_relevancy=answer_relevancy,
                            source_url=source_url
                        )
                        db.add(qna_record)
                        updated_count += 1
        
        db.commit()
        logger.info(f"Backfill completed: {updated_count} QnaRecord entries updated for project {project_id}")
        
    except Exception as e:
        logger.error(f"Failed to backfill QnaRecord entries: {e}")
        db.rollback()
    finally:
        try:
            db.close()
        except:
            pass

