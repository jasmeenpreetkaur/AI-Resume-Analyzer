"""
Data Loading Utilities for AI Resume Analyzer

This module provides convenient functions to load datasets used throughout
the project. Use this to maintain consistent data access patterns.

Usage:
    from src.data_loader import DataLoader
    
    loader = DataLoader()
    resumes_df = loader.load_resume_csv()
    jobs_df = loader.load_jobs_csv()
    esco_skills = loader.load_esco_skills()
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Centralized data loading for the AI Resume Analyzer project.
    
    Handles all dataset loading with consistent error handling and
    data validation.
    """
    
    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize DataLoader and locate data directory.
        
        Args:
            data_root: Path to data directory. If None, auto-detects.
        """
        self.data_root = data_root or self._find_data_root()
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        
        logger.info(f"DataLoader initialized with root: {self.data_root}")
        
        # Define all dataset paths
        self.paths = {
            'resume_csv': self.data_root / 'resume_dataset' / 'Resume' / 'Resume.csv',
            'jobs_csv': self.data_root / 'jobs' / 'all_job_post.csv',
            'resume_pdfs_dir': self.data_root / 'resume_dataset' / 'data' / 'data',
            'esco_skills': self.data_root / 'external' / 'skills_en.csv',
            'esco_occupations': self.data_root / 'external' / 'occupations_en.csv',
            'esco_occ_skill_relations': self.data_root / 'external' / 'occupationSkillRelations_en.csv',
            'esco_skill_hierarchy': self.data_root / 'external' / 'skillsHierarchy_en.csv',
            'esco_skill_relations': self.data_root / 'external' / 'skillSkillRelations_en.csv',
        }
    
    @staticmethod
    def _find_data_root() -> Path:
        """
        Auto-detect data root directory using multiple strategies.
        
        Returns:
            Path to data directory
            
        Raises:
            FileNotFoundError: If data directory cannot be found
        """
        # Strategy 1: Current working directory
        cwd_data = Path.cwd() / 'data'
        if cwd_data.exists():
            return cwd_data
        
        # Strategy 2: Relative to this script
        script_dir = Path(__file__).resolve().parent
        script_data = script_dir / 'data'
        if script_data.exists():
            return script_data
        
        # Strategy 3: If in src/, check parent
        if script_dir.name == 'src':
            parent_data = script_dir.parent / 'data'
            if parent_data.exists():
                return parent_data
        
        # Strategy 4: Check parent of parent
        grandparent_data = script_dir.parent.parent / 'data'
        if grandparent_data.exists():
            return grandparent_data
        
        raise FileNotFoundError(
            f"Could not find 'data' directory. Searched: {Path.cwd()}, {script_dir}"
        )
    
    def load_resume_csv(self) -> pd.DataFrame:
        """
        Load resume dataset CSV.
        
        Returns:
            DataFrame with 2,484 resumes and 20 columns
            
        Columns: ID, Category, Name, Email, Phone_Number, Websites, Resume,
                Resume_str, Experience, Skills, Education, Certifications,
                Projects, Achievements, Links, and others
        """
        return self._load_csv(self.paths['resume_csv'], 'Resume')
    
    def load_jobs_csv(self) -> pd.DataFrame:
        """
        Load job descriptions dataset CSV.
        
        Returns:
            DataFrame with 1,167 job postings
        """
        return self._load_csv(self.paths['jobs_csv'], 'Jobs')
    
    def load_esco_skills(self) -> pd.DataFrame:
        """
        Load ESCO skills master list.
        
        Returns:
            DataFrame with ~13,485 skill definitions
            
        Key columns: skill_id, preferred_label, description, skill_type
        """
        return self._load_csv(self.paths['esco_skills'], 'ESCO Skills')
    
    def load_esco_occupations(self) -> pd.DataFrame:
        """
        Load ESCO occupations master list.
        
        Returns:
            DataFrame with ~2,942 occupation definitions
            
        Key columns: occupation_id, preferred_label, description
        """
        return self._load_csv(self.paths['esco_occupations'], 'ESCO Occupations')
    
    def load_esco_occupation_skill_relations(self) -> pd.DataFrame:
        """
        Load ESCO occupation-to-skill mappings.
        
        Returns:
            DataFrame with ~200k+ occupation-skill relationships
            
        Key columns: occupation_id, skill_id, relation_type
        """
        return self._load_csv(
            self.paths['esco_occ_skill_relations'],
            'ESCO Occupation-Skill Relations'
        )
    
    def load_esco_skill_hierarchy(self) -> pd.DataFrame:
        """
        Load ESCO skill hierarchy (taxonomy).
        
        Returns:
            DataFrame with ~13,485 rows representing skill hierarchy
            
        Key columns: skill_id, parent_skill_id (for building skill tree)
        """
        return self._load_csv(self.paths['esco_skill_hierarchy'], 'ESCO Skill Hierarchy')
    
    def load_esco_skill_relations(self) -> pd.DataFrame:
        """
        Load ESCO skill-to-skill relationships.
        
        Returns:
            DataFrame with ~42k+ skill relationships
            
        Key columns: skill_id_1, skill_id_2, relation_type
        """
        return self._load_csv(self.paths['esco_skill_relations'], 'ESCO Skill Relations')
    
    def load_esco_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all ESCO datasets at once.
        
        Returns:
            Dictionary with keys: 'skills', 'occupations', 'occ_skill_relations',
                                   'skill_hierarchy', 'skill_relations'
        """
        logger.info("Loading all ESCO datasets...")
        return {
            'skills': self.load_esco_skills(),
            'occupations': self.load_esco_occupations(),
            'occ_skill_relations': self.load_esco_occupation_skill_relations(),
            'skill_hierarchy': self.load_esco_skill_hierarchy(),
            'skill_relations': self.load_esco_skill_relations(),
        }
    
    def get_resume_pdfs(self, category: Optional[str] = None) -> List[Path]:
        """
        Get paths to resume PDF files.
        
        Args:
            category: Filter by career category (ENGINEERING, FINANCE, etc.)
                     If None, returns all PDFs.
        
        Returns:
            List of Path objects to PDF files
        """
        pdfs_dir = self.paths['resume_pdfs_dir']
        
        if not pdfs_dir.exists():
            logger.warning(f"PDF directory not found: {pdfs_dir}")
            return []
        
        if category:
            category_dir = pdfs_dir / category
            if not category_dir.exists():
                logger.warning(f"Category {category} not found")
                return []
            pdfs = sorted(category_dir.glob('*.pdf'))
        else:
            pdfs = sorted(pdfs_dir.rglob('*.pdf'))
        
        logger.info(f"Found {len(pdfs)} PDFs" + (f" in {category}" if category else ""))
        return pdfs
    
    def get_pdf_categories(self) -> List[str]:
        """
        Get list of available resume PDF categories.
        
        Returns:
            List of category names (ENGINEERING, FINANCE, etc.)
        """
        pdfs_dir = self.paths['resume_pdfs_dir']
        if not pdfs_dir.exists():
            return []
        
        categories = [d.name for d in pdfs_dir.iterdir() if d.is_dir()]
        return sorted(categories)
    
    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that all required dataset files exist.
        
        Returns:
            Dictionary with file paths as keys and existence status as values
        """
        status = {}
        for name, path in self.paths.items():
            if 'dir' in name:  # It's a directory
                status[name] = path.exists()
            else:  # It's a file
                status[name] = path.exists()
        
        return status
    
    def print_validation_report(self) -> None:
        """Print a validation report of all dataset paths."""
        status = self.validate_paths()
        
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT")
        print("="*80)
        print(f"Data Root: {self.data_root}\n")
        
        all_valid = True
        for name, exists in status.items():
            symbol = "✅" if exists else "❌"
            path = self.paths[name]
            print(f"{symbol} {name:40s} {str(path)}")
            if not exists:
                all_valid = False
        
        print("\n" + "="*80)
        if all_valid:
            print("✅ All datasets found and ready to use!")
        else:
            print("⚠️  Some datasets are missing. Check paths above.")
        print("="*80 + "\n")
    
    @staticmethod
    def _load_csv(filepath: Path, dataset_name: str) -> pd.DataFrame:
        """
        Load CSV with error handling.
        
        Args:
            filepath: Path to CSV file
            dataset_name: Name for logging purposes
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If CSV parsing fails
        """
        if not filepath.exists():
            raise FileNotFoundError(f"{dataset_name} file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            logger.info(f"✅ Loaded {dataset_name}: {len(df):,} rows × {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load {dataset_name}: {str(e)}")
            raise


# Convenience functions for quick access
def load_resumes() -> pd.DataFrame:
    """Quick load of resume CSV."""
    return DataLoader().load_resume_csv()


def load_jobs() -> pd.DataFrame:
    """Quick load of jobs CSV."""
    return DataLoader().load_jobs_csv()


def load_esco() -> Dict[str, pd.DataFrame]:
    """Quick load of all ESCO datasets."""
    return DataLoader().load_esco_all()


if __name__ == "__main__":
    """
    Example usage and validation when run directly.
    """
    print("\n" + "="*80)
    print("DATA LOADER - QUICK START EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize loader
    loader = DataLoader()
    
    # Validate all paths
    loader.print_validation_report()
    
    # Load datasets
    print("Loading datasets...\n")
    
    try:
        resumes = loader.load_resume_csv()
        print(f"Resumes DataFrame shape: {resumes.shape}")
        print(f"Columns: {list(resumes.columns)[:5]}... (showing first 5)\n")
        
        jobs = loader.load_jobs_csv()
        print(f"Jobs DataFrame shape: {jobs.shape}")
        print(f"Columns: {list(jobs.columns)[:5]}... (showing first 5)\n")
        
        skills = loader.load_esco_skills()
        print(f"ESCO Skills shape: {skills.shape}\n")
        
        occupations = loader.load_esco_occupations()
        print(f"ESCO Occupations shape: {occupations.shape}\n")
        
        # Show available categories
        categories = loader.get_pdf_categories()
        print(f"Available PDF Categories: {', '.join(categories)}\n")
        
        # Show sample PDFs
        for category in categories[:2]:  # First 2 categories
            pdfs = loader.get_resume_pdfs(category)
            print(f"{category}: {len(pdfs)} resumes")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you're in the project root directory or have set PYTHONPATH correctly.")
    
    print("\n" + "="*80)
    print("✅ Data loader is working correctly!")
    print("="*80 + "\n")