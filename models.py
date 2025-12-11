"""
Database models for storing audio analysis history.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalysisHistory(Base):
    """Model for storing song analysis history."""
    
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    duration = Column(Float)
    tempo = Column(Float)
    sample_rate = Column(Integer)
    
    vocals_percent = Column(Float)
    drums_percent = Column(Float)
    bass_percent = Column(Float)
    guitar_percent = Column(Float)
    piano_percent = Column(Float)
    other_percent = Column(Float)
    
    band_energies = Column(JSON)
    raw_scores = Column(JSON)
    
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'filename': self.filename,
            'duration': self.duration,
            'tempo': self.tempo,
            'sample_rate': self.sample_rate,
            'percentages': {
                'vocals': self.vocals_percent,
                'drums': self.drums_percent,
                'bass': self.bass_percent,
                'guitar': self.guitar_percent,
                'piano': self.piano_percent,
                'other': self.other_percent
            },
            'band_energies': self.band_energies,
            'raw_scores': self.raw_scores,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None
        }


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def save_analysis(filename, results):
    """Save analysis results to database."""
    db = SessionLocal()
    try:
        percentages = results['percentages']
        
        band_energies_clean = convert_to_python_types(results.get('band_energies', {}))
        raw_scores_clean = convert_to_python_types(results.get('raw_scores', {}))
        
        analysis = AnalysisHistory(
            filename=filename,
            duration=float(results['duration']),
            tempo=float(results['tempo']),
            sample_rate=int(results['sample_rate']),
            vocals_percent=float(percentages['vocals']),
            drums_percent=float(percentages['drums']),
            bass_percent=float(percentages['bass']),
            guitar_percent=float(percentages['guitar']),
            piano_percent=float(percentages['piano']),
            other_percent=float(percentages['other']),
            band_energies=band_energies_clean,
            raw_scores=raw_scores_clean
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis.id
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_all_analyses():
    """Get all analysis history."""
    db = SessionLocal()
    try:
        analyses = db.query(AnalysisHistory).order_by(AnalysisHistory.analyzed_at.desc()).all()
        return [a.to_dict() for a in analyses]
    finally:
        db.close()


def get_analysis_by_id(analysis_id):
    """Get specific analysis by ID."""
    db = SessionLocal()
    try:
        analysis = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
        return analysis.to_dict() if analysis else None
    finally:
        db.close()


def delete_analysis(analysis_id):
    """Delete an analysis from history."""
    db = SessionLocal()
    try:
        analysis = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
        if analysis:
            db.delete(analysis)
            db.commit()
            return True
        return False
    finally:
        db.close()


def get_analyses_for_comparison(analysis_ids):
    """Get multiple analyses for comparison."""
    db = SessionLocal()
    try:
        analyses = db.query(AnalysisHistory).filter(AnalysisHistory.id.in_(analysis_ids)).all()
        return [a.to_dict() for a in analyses]
    finally:
        db.close()
