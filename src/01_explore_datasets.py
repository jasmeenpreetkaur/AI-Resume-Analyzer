"""
Dataset Exploration Script for AI Resume Analyzer & Career Intelligence System

Purpose:
    Automatically discover and analyze all CSV files in the data/ directory.
    Generate comprehensive statistics without modifying any datasets.

Author: AI Resume Analyzer Team
Phase: 2 - Dataset Exploration
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime


class DatasetExplorer:
    """
    Discovers, analyzes, and reports on all datasets in the project.
    """
    
    def __init__(self, data_root: Path = None):
        """
        Initialize the explorer with a data root directory.
        
        Args:
            data_root: Path to the data directory. Defaults to ./data/
        """
        self.data_root = data_root or Path("./data")
        self.csv_files: List[Path] = []
        self.pdf_files: List[Path] = []
        self.analysis_results: Dict = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def discover_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Recursively discover all CSV and PDF files in data_root.
        
        Returns:
            Tuple of (csv_files, pdf_files)
        """
        if not self.data_root.exists():
            print(f"⚠️  Data root directory not found: {self.data_root}")
            return [], []
        
        # Discover CSV files
        self.csv_files = sorted(self.data_root.rglob("*.csv"))
        
        # Discover PDF files
        self.pdf_files = sorted(self.data_root.rglob("*.pdf"))
        
        return self.csv_files, self.pdf_files
    
    def analyze_csv(self, filepath: Path) -> Dict:
        """
        Analyze a single CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            df = pd.read_csv(filepath, low_memory=False)
            
            analysis = {
                "file_path": str(filepath),
                "relative_path": str(filepath.relative_to(self.data_root)),
                "file_size_mb": filepath.stat().st_size / (1024 * 1024),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_values_percent": (
                    (df.isnull().sum() / len(df) * 100).round(2).to_dict()
                ),
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_rows_percent": round(
                    (df.duplicated().sum() / len(df) * 100), 2
                ),
                "first_3_rows": df.head(3).to_dict(orient="records"),
                "status": "✓ Successfully analyzed"
            }
            
            return analysis
            
        except Exception as e:
            return {
                "file_path": str(filepath),
                "relative_path": str(filepath.relative_to(self.data_root)),
                "status": f"✗ Error: {str(e)}"
            }
    
    def explore_all(self) -> Dict:
        """
        Execute full exploration of all datasets.
        
        Returns:
            Dictionary with complete analysis results
        """
        print("\n" + "="*80)
        print("AI RESUME ANALYZER - DATASET EXPLORATION")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Data Root: {self.data_root.resolve()}")
        print("="*80 + "\n")
        
        # Discover files
        csv_files, pdf_files = self.discover_files()
        
        print(f"📊 DISCOVERY RESULTS")
        print(f"   CSV files found: {len(csv_files)}")
        print(f"   PDF files found: {len(pdf_files)}")
        print(f"   Total files: {len(csv_files) + len(pdf_files)}\n")
        
        # Analyze CSVs
        print(f"🔍 ANALYZING CSV FILES ({len(csv_files)} files)\n")
        print("-" * 80)
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}] {csv_file.relative_to(self.data_root)}")
            print("-" * 80)
            
            analysis = self.analyze_csv(csv_file)
            self.analysis_results[str(csv_file.relative_to(self.data_root))] = analysis
            
            if analysis.get("status") == "✓ Successfully analyzed":
                self._print_csv_analysis(analysis)
            else:
                print(f"   {analysis.get('status')}")
        
        # PDF Summary with categorization
        print(f"\n\n{'='*80}")
        print(f"📄 PDF FILES SUMMARY")
        print(f"{'='*80}")
        print(f"   Total PDF files: {len(pdf_files)}")
        
        if pdf_files:
            # Categorize PDFs
            pdf_by_category = {}
            for pdf in pdf_files:
                # Extract category from path
                parts = pdf.parts
                category = "OTHER"
                for part in parts:
                    if part in ["ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", 
                               "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"]:
                        category = part
                        break
                
                if category not in pdf_by_category:
                    pdf_by_category[category] = []
                pdf_by_category[category].append(pdf)
            
            # Print by category
            print(f"\n   Distribution by Career Category:")
            sorted_cats = sorted(pdf_by_category.items(), key=lambda x: len(x[1]), reverse=True)
            for category, pdfs in sorted_cats:
                pct = round((len(pdfs) / len(pdf_files) * 100), 1)
                print(f"      • {category:25s}: {len(pdfs):4d} PDFs ({pct:5.1f}%)")
        
        # Dataset Summary
        print(f"\n\n{'='*80}")
        print(f"📋 DATASET SUMMARY")
        print(f"{'='*80}")
        self._print_summary()
        
        return self.analysis_results
    
    def _print_csv_analysis(self, analysis: Dict) -> None:
        """
        Pretty print analysis for a single CSV file.
        
        Args:
            analysis: Analysis dictionary
        """
        print(f"   Status: {analysis.get('status')}")
        print(f"   Size: {analysis.get('file_size_mb', 0):.2f} MB")
        print(f"   Rows: {analysis.get('num_rows', 0):,}")
        print(f"   Columns: {analysis.get('num_columns', 0)}")
        
        # Column names
        cols = analysis.get('column_names', [])
        if cols:
            print(f"\n   Column Names ({len(cols)}):")
            for col in cols:
                print(f"      • {col}")
        
        # Data types
        dtypes = analysis.get('data_types', {})
        if dtypes:
            print(f"\n   Data Types:")
            for col, dtype in dtypes.items():
                print(f"      • {col}: {dtype}")
        
        # Missing values
        missing = analysis.get('missing_values', {})
        missing_pct = analysis.get('missing_values_percent', {})
        has_missing = any(v > 0 for v in missing.values())
        if has_missing:
            print(f"\n   ⚠️  Missing Values:")
            for col, count in missing.items():
                if count > 0:
                    pct = missing_pct.get(col, 0)
                    print(f"      • {col}: {count} ({pct}%)")
        else:
            print(f"\n   ✓ No missing values")
        
        # Duplicates
        dup_count = analysis.get('duplicate_rows', 0)
        dup_pct = analysis.get('duplicate_rows_percent', 0)
        if dup_count > 0:
            print(f"\n   ⚠️  Duplicate Rows: {dup_count} ({dup_pct}%)")
        else:
            print(f"\n   ✓ No duplicate rows")
        
        # First 3 rows
        first_rows = analysis.get('first_3_rows', [])
        if first_rows:
            print(f"\n   First 3 Rows:")
            for idx, row in enumerate(first_rows, 1):
                print(f"\n      Row {idx}:")
                for key, value in row.items():
                    # Truncate long values
                    val_str = str(value)
                    if len(val_str) > 60:
                        val_str = val_str[:57] + "..."
                    print(f"         {key}: {val_str}")
    
    def _print_summary(self) -> None:
        """Print summary statistics of all datasets."""
        if not self.analysis_results:
            print("   No CSV files analyzed.")
            return
        
        print(f"\n   Successful Analyses: ", end="")
        successful = sum(
            1 for r in self.analysis_results.values()
            if r.get("status") == "✓ Successfully analyzed"
        )
        print(f"{successful}/{len(self.analysis_results)}")
        
        print(f"\n   Dataset Categories:")
        
        # Resume dataset
        resume_files = [
            k for k in self.analysis_results.keys()
            if "resume" in k.lower()
        ]
        if resume_files:
            print(f"\n      📝 Resume Dataset: {len(resume_files)} file(s)")
            for f in resume_files:
                rows = self.analysis_results[f].get('num_rows', 'N/A')
                print(f"         • {f} ({rows} rows)")
        
        # Job dataset
        job_files = [
            k for k in self.analysis_results.keys()
            if "job" in k.lower()
        ]
        if job_files:
            print(f"\n      💼 Job Dataset: {len(job_files)} file(s)")
            for f in job_files:
                rows = self.analysis_results[f].get('num_rows', 'N/A')
                print(f"         • {f} ({rows} rows)")
        
        # ESCO dataset
        esco_files = [
            k for k in self.analysis_results.keys()
            if "esco" in k.lower() or k.startswith("external/")
        ]
        if esco_files:
            print(f"\n      🏢 ESCO Dataset: {len(esco_files)} file(s)")
            for f in esco_files:
                rows = self.analysis_results[f].get('num_rows', 'N/A')
                print(f"         • {f} ({rows} rows)")
        
        # PDF Summary by category
        if self.pdf_files:
            print(f"\n   📄 PDF FILES BY CATEGORY:")
            pdf_by_category = {}
            for pdf in self.pdf_files:
                # Extract category from path (e.g., ENGINEERING, FINANCE, etc.)
                parts = pdf.parts
                category = "OTHER"
                for part in parts:
                    if part in ["ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", 
                               "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"]:
                        category = part
                        break
                
                if category not in pdf_by_category:
                    pdf_by_category[category] = 0
                pdf_by_category[category] += 1
            
            # Sort by count descending
            sorted_cats = sorted(pdf_by_category.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_cats:
                print(f"      • {category}: {count} PDFs")
    
    def save_report(self, output_path: Path = None) -> Path:
        """
        Save analysis results to a JSON report.
        
        Args:
            output_path: Path to save the report. Defaults to reports/exploration.json
            
        Returns:
            Path to the saved report
        """
        output_path = output_path or Path("./reports/exploration_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": self.timestamp,
            "data_root": str(self.data_root.resolve()),
            "csv_files_count": len(self.csv_files),
            "pdf_files_count": len(self.pdf_files),
            "analyses": self.analysis_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n\n✅ Report saved to: {output_path}\n")
        return output_path


def find_data_root() -> Path:
    """
    Find the data root directory by searching up from the current location.
    
    Tries multiple strategies:
    1. Check if data/ exists in current working directory
    2. Check if data/ exists relative to script location
    3. Check if we're in a src/ subdirectory and data is in parent
    
    Returns:
        Path to data directory
        
    Raises:
        FileNotFoundError: If data directory cannot be found
    """
    # Strategy 1: Current working directory
    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data
    
    # Strategy 2: Relative to script location
    script_dir = Path(__file__).resolve().parent
    script_data = script_dir / "data"
    if script_data.exists():
        return script_data
    
    # Strategy 3: If script is in src/, check parent directory
    if script_dir.name == "src":
        parent_data = script_dir.parent / "data"
        if parent_data.exists():
            return parent_data
    
    # Strategy 4: Check parent of parent
    grandparent_data = script_dir.parent.parent / "data"
    if grandparent_data.exists():
        return grandparent_data
    
    # Not found
    raise FileNotFoundError(
        f"Could not find 'data' directory. "
        f"Searched from: {Path.cwd().resolve()}, {script_dir.resolve()}"
    )


def main():
    """Main execution function."""
    try:
        # Find data root automatically
        data_root = find_data_root()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nMake sure the project structure is:")
        print("   ai_resume_analyzer/")
        print("   ├── data/")
        print("   ├── src/")
        print("   └── 01_explore_datasets.py (or in src/)")
        return
    
    # Initialize explorer
    explorer = DatasetExplorer(data_root=data_root)
    
    # Run exploration
    explorer.explore_all()
    
    # Save report
    explorer.save_report()
    
    print("="*80)
    print("✅ DATASET EXPLORATION COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("   1. Review the output above and the JSON report")
    print("   2. Identify relevant files for Phase 3 (Data Cleaning)")
    print("   3. Document which ESCO files to use")
    print("   4. Note columns to preserve and handle missing values")
    print("\n")


if __name__ == "__main__":
    main()