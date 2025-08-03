"""
Feedback Management System for Betty AI Assistant

This module handles user feedback collection, storage, and analytics
for continuous improvement of Betty's responses and capabilities.
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path


class FeedbackManager:
    """Manages user feedback storage and retrieval for Betty AI Assistant."""
    
    def __init__(self, db_path: str = "data/betty_feedback.db"):
        """Initialize the feedback manager with database connection."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the feedback database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    betty_response TEXT NOT NULL,
                    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('thumbs_up', 'thumbs_down')),
                    feedback_details TEXT,
                    response_quality_score REAL,
                    obt_compliance_score REAL,
                    response_length INTEGER,
                    contains_outcome BOOLEAN,
                    contains_kpi BOOLEAN,
                    contains_gps_tier BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_agent TEXT,
                    ip_hash TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_id ON feedback(conversation_id)")
    
    def generate_conversation_id(self, user_message: str, betty_response: str) -> str:
        """Generate a unique conversation ID based on message content."""
        content = f"{user_message[:100]}{betty_response[:100]}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def analyze_response_quality(self, betty_response: str) -> Dict[str, Any]:
        """Analyze Betty's response for quality metrics."""
        metrics = {
            'response_length': len(betty_response),
            'contains_outcome': False,
            'contains_kpi': False,
            'contains_gps_tier': False,
            'obt_compliance_score': 0.0,
            'response_quality_score': 0.0
        }
        
        response_lower = betty_response.lower()
        
        # Check for OBT elements
        outcome_indicators = ['outcome:', 'what:', 'result:', 'achieve', 'measured']
        kpi_indicators = ['kpi:', 'goal:', 'measurement:', 'metric', 'target']
        gps_indicators = ['gps tier:', 'destination', 'highway', 'main street', 'county road']
        
        metrics['contains_outcome'] = any(indicator in response_lower for indicator in outcome_indicators)
        metrics['contains_kpi'] = any(indicator in response_lower for indicator in kpi_indicators)
        metrics['contains_gps_tier'] = any(indicator in response_lower for indicator in gps_indicators)
        
        # Calculate OBT compliance score (0-1)
        obt_score = 0.0
        if metrics['contains_outcome']:
            obt_score += 0.4
        if metrics['contains_kpi']:
            obt_score += 0.3
        if metrics['contains_gps_tier']:
            obt_score += 0.3
        
        metrics['obt_compliance_score'] = obt_score
        
        # Calculate overall response quality (simplified scoring)
        quality_score = 0.5  # Base score
        if metrics['response_length'] > 50:  # Substantial response
            quality_score += 0.2
        if metrics['contains_outcome']:
            quality_score += 0.1
        if metrics['contains_kpi']:
            quality_score += 0.1
        if metrics['contains_gps_tier']:
            quality_score += 0.1
        
        metrics['response_quality_score'] = min(quality_score, 1.0)
        
        return metrics
    
    def record_feedback(
        self,
        session_id: str,
        user_message: str,
        betty_response: str,
        feedback_type: str,
        feedback_details: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Record user feedback for a conversation."""
        
        conversation_id = self.generate_conversation_id(user_message, betty_response)
        metrics = self.analyze_response_quality(betty_response)
        
        # Hash IP address for privacy
        ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16] if ip_address else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback (
                    session_id, conversation_id, user_message, betty_response,
                    feedback_type, feedback_details, response_quality_score,
                    obt_compliance_score, response_length, contains_outcome,
                    contains_kpi, contains_gps_tier, user_agent, ip_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, conversation_id, user_message, betty_response,
                feedback_type, feedback_details, metrics['response_quality_score'],
                metrics['obt_compliance_score'], metrics['response_length'],
                metrics['contains_outcome'], metrics['contains_kpi'],
                metrics['contains_gps_tier'], user_agent, ip_hash
            ))
        
        return conversation_id
    
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary statistics for feedback over the specified period."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get feedback counts
            cursor.execute("""
                SELECT 
                    feedback_type,
                    COUNT(*) as count,
                    AVG(response_quality_score) as avg_quality,
                    AVG(obt_compliance_score) as avg_obt_compliance
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY feedback_type
            """.format(days))
            
            feedback_counts = {row['feedback_type']: dict(row) for row in cursor.fetchall()}
            
            # Get overall metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(response_quality_score) as avg_quality,
                    AVG(obt_compliance_score) as avg_obt_compliance,
                    AVG(response_length) as avg_response_length,
                    SUM(CASE WHEN contains_outcome THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as outcome_percentage,
                    SUM(CASE WHEN contains_kpi THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as kpi_percentage,
                    SUM(CASE WHEN contains_gps_tier THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as gps_tier_percentage
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days))
            
            overall_metrics = dict(cursor.fetchone())
            
            return {
                'feedback_counts': feedback_counts,
                'overall_metrics': overall_metrics,
                'period_days': days
            }
    
    def get_recent_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent feedback entries with details."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id, session_id, conversation_id, user_message, betty_response,
                    feedback_type, feedback_details, response_quality_score,
                    obt_compliance_score, timestamp, contains_outcome,
                    contains_kpi, contains_gps_tier
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify areas for improvement based on negative feedback."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Find patterns in negative feedback
            cursor.execute("""
                SELECT 
                    user_message,
                    betty_response,
                    feedback_details,
                    response_quality_score,
                    obt_compliance_score,
                    timestamp
                FROM feedback 
                WHERE feedback_type = 'thumbs_down'
                ORDER BY timestamp DESC 
                LIMIT 20
            """)
            
            negative_feedback = [dict(row) for row in cursor.fetchall()]
            
            # Find low-scoring responses
            cursor.execute("""
                SELECT 
                    user_message,
                    betty_response,
                    response_quality_score,
                    obt_compliance_score,
                    timestamp
                FROM feedback 
                WHERE response_quality_score < 0.5 OR obt_compliance_score < 0.3
                ORDER BY response_quality_score ASC 
                LIMIT 10
            """)
            
            low_scoring = [dict(row) for row in cursor.fetchall()]
            
            return {
                'negative_feedback': negative_feedback,
                'low_scoring_responses': low_scoring
            }


# Global instance for easy import
feedback_manager = FeedbackManager()