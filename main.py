#!/usr/bin/env python3
"""
Complete GPU-Accelerated ML Point Cloud Reduction System with Enhanced CSV Data Collection,
Consolidated Comparison File, Convex Hull Reconstruction Support, Controlled Convex Hull for LMGC90,
and Enhanced LMGC90 DAT Export Functionality
Ready-to-use implementation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import multiprocessing as mp

# Try importing required dependencies
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    print("Installing trimesh...")
    os.system("pip install trimesh")
    import trimesh
    HAS_TRIMESH = True

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Installing open3d...")
    os.system("pip install open3d")
    import open3d as o3d
    HAS_OPEN3D = True

try:
    from sklearn.svm import SVC
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.svm import SVC
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True

# GPU acceleration imports (optional)
try:
    import cupy as cp
    HAS_CUPY = True
    print("✅ CuPy detected - GPU array acceleration available")
except ImportError:
    import numpy as cp
    HAS_CUPY = False
    print("⚠️  CuPy not available. Install with: pip install cupy-cuda11x")

try:
    import cuml
    from cuml.svm import SVC as cuSVC
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    HAS_CUML = True
    print("✅ cuML detected - GPU ML acceleration available")
except ImportError:
    # Use sklearn as fallback
    cuSVC = SVC
    cuDBSCAN = DBSCAN
    cuNearestNeighbors = NearestNeighbors
    cuStandardScaler = StandardScaler
    HAS_CUML = False
    print("⚠️  cuML not available. Install with: pip install cuml-cu11")

try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA available with {torch.cuda.device_count()} GPU(s)")
    else:
        print("⚠️  PyTorch available but CUDA not detected")
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not available")

warnings.filterwarnings('ignore')

# Configure logging with automatic file output
def setup_logging(output_dir: str = ".", verbose: bool = False, quiet: bool = False):
    """Setup logging to both console and file"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pointcloud_reduction_{timestamp}.log"
    
    # Set log level
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # File handler
            logging.StreamHandler(sys.stdout)  # Console handler
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== GPU-Accelerated Point Cloud Reduction Started ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger, str(log_file)

# Initialize default logger (will be reconfigured in main)
logger = logging.getLogger(__name__)

@dataclass
class ReductionConfig:
    """Configuration parameters for GPU-accelerated point cloud reduction"""
    target_points_min: int = 50
    target_points_max: int = 300
    max_ballast: int = 150
    voxel_size: float = 0.02
    svm_sample_ratio: float = 0.1
    knn_neighbors: int = 5
    epsilon_range: Tuple[float, float] = (0.01, 0.1)
    dbscan_min_samples: int = 3
    reconstruction_method: str = 'poisson'  # 'poisson', 'ball_pivoting', 'alpha_shapes', 'convex_hull', 'controlled_convex_hull'
    n_cores: int = mp.cpu_count()
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    batch_size: int = 10000
    save_stages: bool = True
    detailed_analysis: bool = True

# ===== LMGC90 DAT EXPORT FUNCTIONS =====

def validate_mesh_for_lmgc90(mesh, filename: str) -> bool:
    """
    Comprehensive validation for LMGC90 compatibility
    """
    issues = []
    
    # Basic geometry checks
    if len(mesh.vertices) == 0:
        issues.append("No vertices")
    if len(mesh.faces) == 0:
        issues.append("No faces")
    
    # LMGC90 specific checks
    if not mesh.is_watertight:
        issues.append("Not watertight (has holes)")
    
    if not mesh.is_convex:
        issues.append("Not convex")
    
    if not mesh.is_volume:
        issues.append("Invalid volume")
    
    # Vertex count check (for controlled convex hull)
    vertex_count = len(mesh.vertices)
    if not (250 <= vertex_count <= 700):  # Broader acceptable range
        issues.append(f"Vertex count {vertex_count} outside acceptable range")
    
    # Geometric quality checks
    try:
        if mesh.volume <= 0:
            issues.append("Zero or negative volume")
        if mesh.area <= 0:
            issues.append("Zero or negative surface area")
    except:
        issues.append("Cannot calculate volume/area")
    
    # Face orientation check
    try:
        # Check if all face normals point outward (for convex meshes)
        face_normals = mesh.face_normals
        centroid = mesh.centroid
        
        # For each face, check if normal points away from centroid
        outward_normals = 0
        for i, face in enumerate(mesh.faces):
            face_center = np.mean(mesh.vertices[face], axis=0)
            to_face = face_center - centroid
            dot_product = np.dot(face_normals[i], to_face)
            if dot_product > 0:
                outward_normals += 1
        
        outward_ratio = outward_normals / len(mesh.faces)
        if outward_ratio < 0.9:  # Most faces should point outward
            issues.append(f"Face orientation issue ({outward_ratio:.1%} outward)")
            
    except:
        issues.append("Cannot validate face orientation")
    
    # Report results
    if issues:
        print(f"  ❌ {filename} validation failed:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print(f"  ✅ {filename} passed all LMGC90 validation checks")
        return True

def export_to_lmgc90_dat(output_folder: str, dat_filename: str = "lmgc90_export.DAT"):
    """
    Export all processed meshes to a single LMGC90-compatible DAT file
    Only includes meshes that are verified as LMGC90-ready
    """
    output_path = Path(output_folder)
    dat_path = output_path / dat_filename
    
    print(f"\n🎯 Exporting LMGC90-compatible DAT file...")
    print(f"Output: {dat_path}")
    
    successful_exports = 0
    failed_exports = 0
    skipped_non_lmgc90 = 0
    
    # Configure high precision format
    format_str = " {0:45.30f} {1:45.30f} {2:45.30E}\n"
    
    with open(dat_path, 'w') as dat_file:
        # Write DAT file header
        dat_file.write("# LMGC90 DAT file generated by GPU-Accelerated Point Cloud Reduction\n")
        dat_file.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        dat_file.write(f"# Method: controlled_convex_hull\n")
        dat_file.write("#\n")
        
        # Process each subdirectory (one per original STL file)
        for subdir in sorted(output_path.glob("*/")):
            if not subdir.is_dir() or subdir.name in ['logs']:
                continue
                
            # Look for LMGC90-ready STL file
            lmgc90_stl = subdir / f"{subdir.name}_lmgc90_ready.stl"
            analysis_csv = subdir / f"{subdir.name}_lmgc90_analysis.csv"
            
            if not lmgc90_stl.exists():
                print(f"⚠️  No LMGC90 STL found for {subdir.name}")
                failed_exports += 1
                continue
            
            # Check LMGC90 readiness from analysis file
            lmgc90_ready = False
            if analysis_csv.exists():
                try:
                    df = pd.read_csv(analysis_csv)
                    lmgc90_ready = df.iloc[0]['lmgc90_ready'] if 'lmgc90_ready' in df.columns else False
                except:
                    pass
            
            if not lmgc90_ready:
                print(f"⚠️  {subdir.name} not verified as LMGC90-ready, skipping")
                skipped_non_lmgc90 += 1
                continue
                
            try:
                # Load the LMGC90-ready mesh
                mesh = trimesh.load(str(lmgc90_stl), force='mesh')
                
                # Final validation for LMGC90
                validation_passed = validate_mesh_for_lmgc90(mesh, subdir.name)
                
                if not validation_passed:
                    print(f"❌ {subdir.name} failed final validation")
                    failed_exports += 1
                    continue
                
                # Write mesh data to DAT file
                # Object header
                dat_file.write(f"# Object: {subdir.name}\n")
                dat_file.write(f"# Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}\n")
                dat_file.write(f"# Convex: {mesh.is_convex}, Volume: {mesh.volume:.6f}\n")
                
                # Number of vertices
                dat_file.write(f"          {len(mesh.vertices)}\n")
                
                # Vertex coordinates (30 decimal precision)
                for vertex in mesh.vertices:
                    dat_file.write(format_str.format(vertex[0], vertex[1], vertex[2]))
                
                # Number of faces
                dat_file.write(f"          {len(mesh.faces)}\n")
                
                # Face indices (1-indexed for LMGC90)
                for face in mesh.faces:
                    dat_file.write(f"{face[0]+1:11}{face[1]+1:11}{face[2]+1:11}\n")
                
                # Object separator
                dat_file.write("$$$$$$\n")
                
                successful_exports += 1
                print(f"✅ {subdir.name}: Exported successfully")
                
            except Exception as e:
                print(f"❌ {subdir.name}: Export failed - {str(e)}")
                failed_exports += 1
                continue
    
    # Summary
    print(f"\n📊 DAT Export Summary:")
    print(f"  ✅ Successful exports: {successful_exports}")
    print(f"  ❌ Failed exports: {failed_exports}")
    print(f"  ⚠️  Skipped (not LMGC90-ready): {skipped_non_lmgc90}")
    print(f"  📄 DAT file: {dat_path}")
    
    if successful_exports > 0:
        print(f"\n🎯 LMGC90 Import Instructions:")
        print(f"  1. Copy {dat_filename} to your LMGC90 project directory")
        print(f"  2. Use LMGC90's import function to load the geometries")
        print(f"  3. All {successful_exports} objects are convex and simulation-ready")
    
    return str(dat_path), successful_exports

def export_individual_lmgc90_dats(output_folder: str, dat_prefix: str = "lmgc90_object"):
    """
    Export each processed mesh to individual LMGC90-compatible DAT files
    """
    output_path = Path(output_folder)
    
    print(f"\n🎯 Exporting individual LMGC90 DAT files...")
    
    successful_exports = 0
    failed_exports = 0
    skipped_non_lmgc90 = 0
    exported_files = []
    
    # Configure high precision format
    format_str = " {0:45.30f} {1:45.30f} {2:45.30E}\n"
    
    # Process each subdirectory (one per original STL file)
    for subdir in sorted(output_path.glob("*/")):
        if not subdir.is_dir() or subdir.name in ['logs']:
            continue
            
        # Look for LMGC90-ready STL file
        lmgc90_stl = subdir / f"{subdir.name}_lmgc90_ready.stl"
        analysis_csv = subdir / f"{subdir.name}_lmgc90_analysis.csv"
        
        if not lmgc90_stl.exists():
            print(f"⚠️  No LMGC90 STL found for {subdir.name}")
            failed_exports += 1
            continue
        
        # Check LMGC90 readiness from analysis file
        lmgc90_ready = False
        if analysis_csv.exists():
            try:
                df = pd.read_csv(analysis_csv)
                lmgc90_ready = df.iloc[0]['lmgc90_ready'] if 'lmgc90_ready' in df.columns else False
            except:
                pass
        
        if not lmgc90_ready:
            print(f"⚠️  {subdir.name} not verified as LMGC90-ready, skipping")
            skipped_non_lmgc90 += 1
            continue
            
        try:
            # Load the LMGC90-ready mesh
            mesh = trimesh.load(str(lmgc90_stl), force='mesh')
            
            # Final validation for LMGC90
            validation_passed = validate_mesh_for_lmgc90(mesh, subdir.name)
            
            if not validation_passed:
                print(f"❌ {subdir.name} failed final validation")
                failed_exports += 1
                continue
            
            # Create individual DAT file
            dat_path = subdir / f"{dat_prefix}_{subdir.name}.DAT"
            
            with open(dat_path, 'w') as dat_file:
                # Write DAT file header
                dat_file.write("# LMGC90 DAT file generated by GPU-Accelerated Point Cloud Reduction\n")
                dat_file.write(f"# Object: {subdir.name}\n")
                dat_file.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                dat_file.write(f"# Method: controlled_convex_hull\n")
                dat_file.write("#\n")
                
                # Object header
                dat_file.write(f"# Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}\n")
                dat_file.write(f"# Convex: {mesh.is_convex}, Volume: {mesh.volume:.6f}\n")
                
                # Number of vertices
                dat_file.write(f"          {len(mesh.vertices)}\n")
                
                # Vertex coordinates (30 decimal precision)
                for vertex in mesh.vertices:
                    dat_file.write(format_str.format(vertex[0], vertex[1], vertex[2]))
                
                # Number of faces
                dat_file.write(f"          {len(mesh.faces)}\n")
                
                # Face indices (1-indexed for LMGC90)
                for face in mesh.faces:
                    dat_file.write(f"{face[0]+1:11}{face[1]+1:11}{face[2]+1:11}\n")
            
            exported_files.append(str(dat_path))
            successful_exports += 1
            print(f"✅ {subdir.name}: Individual DAT exported to {dat_path.name}")
            
        except Exception as e:
            print(f"❌ {subdir.name}: Individual DAT export failed - {str(e)}")
            failed_exports += 1
            continue
    
    # Summary
    print(f"\n📊 Individual DAT Export Summary:")
    print(f"  ✅ Successful exports: {successful_exports}")
    print(f"  ❌ Failed exports: {failed_exports}")
    print(f"  ⚠️  Skipped (not LMGC90-ready): {skipped_non_lmgc90}")
    
    return exported_files, successful_exports

# ===== COMPLETE ORIGINAL CLASSES =====

class DataCollector:
    """Enhanced data collection with consolidated comparison file, convex hull analysis, and LMGC90 support"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.processing_data = []
        self.detailed_data = []
        self.comparison_data = []  # Store all comparison data for consolidated file
        
    def save_original_pointcloud_csv(self, points: np.ndarray, normals: np.ndarray, 
                                   filename: str, mesh_info: Dict = None) -> str:
        """Save original point cloud data to CSV"""
        output_subdir = self.output_dir / Path(filename).stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive original data
        original_df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1], 
            'z': points[:, 2],
            'nx': normals[:, 0],
            'ny': normals[:, 1],
            'nz': normals[:, 2],
            'point_id': range(len(points)),
            'stage': 'original'
        })
        
        # Add mesh information if available
        if mesh_info:
            for key, value in mesh_info.items():
                if isinstance(value, (int, float, bool)):
                    original_df[f'mesh_{key}'] = value
        
        csv_path = output_subdir / f"{Path(filename).stem}_original_pointcloud.csv"
        original_df.to_csv(csv_path, index=False)
        
        # Also save basic statistics
        stats_df = pd.DataFrame({
            'metric': ['total_points', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
                      'x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std',
                      'bbox_volume', 'point_density'],
            'value': [
                len(points),
                points[:, 0].min(), points[:, 0].max(),
                points[:, 1].min(), points[:, 1].max(), 
                points[:, 2].min(), points[:, 2].max(),
                points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean(),
                points[:, 0].std(), points[:, 1].std(), points[:, 2].std(),
                np.prod(points.max(axis=0) - points.min(axis=0)),
                len(points) / np.prod(points.max(axis=0) - points.min(axis=0)) if np.prod(points.max(axis=0) - points.min(axis=0)) > 0 else 0
            ]
        })
        
        stats_path = output_subdir / f"{Path(filename).stem}_original_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        
        return str(csv_path)
    
    def save_processing_stage_csv(self, points: np.ndarray, normals: np.ndarray,
                                features: np.ndarray, importance_scores: np.ndarray,
                                filename: str, stage: str) -> str:
        """Save intermediate processing stage data"""
        output_subdir = self.output_dir / Path(filename).stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive stage data
        stage_df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'nx': normals[:, 0],
            'ny': normals[:, 1],
            'nz': normals[:, 2],
            'point_id': range(len(points)),
            'stage': stage
        })
        
        # Add features if available
        if features is not None and len(features) == len(points):
            feature_names = ['curvature', 'density', 'centroid_distance', 'normal_magnitude']
            for i, name in enumerate(feature_names):
                if i < features.shape[1]:
                    stage_df[f'feature_{name}'] = features[:, i]
        
        # Add importance scores if available
        if importance_scores is not None and len(importance_scores) == len(points):
            stage_df['importance_score'] = importance_scores
            stage_df['is_important'] = importance_scores > np.percentile(importance_scores, 70)
        
        csv_path = output_subdir / f"{Path(filename).stem}_{stage}_pointcloud.csv"
        stage_df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def save_final_pointcloud_csv(self, points: np.ndarray, normals: np.ndarray,
                                filename: str, processing_info: Dict) -> str:
        """Save final reduced point cloud data"""
        output_subdir = self.output_dir / Path(filename).stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create final data with processing information
        final_df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'nx': normals[:, 0],
            'ny': normals[:, 1],
            'nz': normals[:, 2],
            'point_id': range(len(points)),
            'stage': 'final_reduced'
        })
        
        # Add processing metadata
        for key, value in processing_info.items():
            if isinstance(value, (int, float, bool)):
                final_df[f'proc_{key}'] = value
        
        csv_path = output_subdir / f"{Path(filename).stem}_final_pointcloud.csv"
        final_df.to_csv(csv_path, index=False)
        
        # Save final statistics
        stats_df = pd.DataFrame({
            'metric': ['final_points', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
                      'x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std',
                      'bbox_volume', 'point_density'],
            'value': [
                len(points),
                points[:, 0].min(), points[:, 0].max(),
                points[:, 1].min(), points[:, 1].max(),
                points[:, 2].min(), points[:, 2].max(),
                points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean(),
                points[:, 0].std(), points[:, 1].std(), points[:, 2].std(),
                np.prod(points.max(axis=0) - points.min(axis=0)),
                len(points) / np.prod(points.max(axis=0) - points.min(axis=0)) if np.prod(points.max(axis=0) - points.min(axis=0)) > 0 else 0
            ]
        })
        
        stats_path = output_subdir / f"{Path(filename).stem}_final_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        
        return str(csv_path)
    
    def calculate_convex_hull_metrics(self, original_mesh, convex_hull_mesh) -> Dict:
        """Calculate convex hull specific metrics"""
        try:
            metrics = {}
            
            # Basic convex hull properties
            if hasattr(original_mesh, 'volume') and hasattr(convex_hull_mesh, 'volume'):
                metrics['original_volume'] = original_mesh.volume
                metrics['convex_hull_volume'] = convex_hull_mesh.volume
                metrics['convexity_ratio'] = original_mesh.volume / convex_hull_mesh.volume if convex_hull_mesh.volume > 0 else 0
            else:
                metrics['original_volume'] = 0
                metrics['convex_hull_volume'] = 0
                metrics['convexity_ratio'] = 0
            
            # Surface area comparison
            if hasattr(original_mesh, 'area') and hasattr(convex_hull_mesh, 'area'):
                metrics['original_surface_area'] = original_mesh.area
                metrics['convex_hull_surface_area'] = convex_hull_mesh.area
                metrics['surface_area_ratio'] = original_mesh.area / convex_hull_mesh.area if convex_hull_mesh.area > 0 else 0
            else:
                metrics['original_surface_area'] = 0
                metrics['convex_hull_surface_area'] = 0
                metrics['surface_area_ratio'] = 0
            
            # Vertex and face counts
            metrics['original_vertices'] = len(original_mesh.vertices)
            metrics['convex_hull_vertices'] = len(convex_hull_mesh.vertices)
            metrics['original_faces'] = len(original_mesh.faces) if hasattr(original_mesh, 'faces') else 0
            metrics['convex_hull_faces'] = len(convex_hull_mesh.faces) if hasattr(convex_hull_mesh, 'faces') else 0
            
            # Reduction metrics
            metrics['vertex_reduction_ratio'] = (metrics['original_vertices'] - metrics['convex_hull_vertices']) / max(metrics['original_vertices'], 1)
            metrics['face_reduction_ratio'] = (metrics['original_faces'] - metrics['convex_hull_faces']) / max(metrics['original_faces'], 1)
            
            # Complexity metrics
            metrics['geometric_simplification'] = 1.0 - metrics['convexity_ratio']  # How much geometry was simplified
            metrics['volume_preservation'] = metrics['convexity_ratio']  # How much volume was preserved
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate convex hull metrics: {e}")
            return {
                'original_volume': 0, 'convex_hull_volume': 0, 'convexity_ratio': 0,
                'original_surface_area': 0, 'convex_hull_surface_area': 0, 'surface_area_ratio': 0,
                'original_vertices': 0, 'convex_hull_vertices': 0,
                'original_faces': 0, 'convex_hull_faces': 0,
                'vertex_reduction_ratio': 0, 'face_reduction_ratio': 0,
                'geometric_simplification': 0, 'volume_preservation': 0
            }
    
    def save_convex_hull_analysis_csv(self, original_mesh, convex_hull_mesh, filename: str) -> str:
        """Save detailed convex hull analysis to CSV"""
        output_subdir = self.output_dir / Path(filename).stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Calculate convex hull metrics
        metrics = self.calculate_convex_hull_metrics(original_mesh, convex_hull_mesh)
        
        # Add additional analysis
        analysis_data = {
            'filename': Path(filename).stem,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'reconstruction_method': 'convex_hull',
            **metrics
        }
        
        # Face analysis (if available)
        if hasattr(convex_hull_mesh, 'faces') and len(convex_hull_mesh.faces) > 0:
            face_data = []
            for i, face in enumerate(convex_hull_mesh.faces):
                face_vertices = convex_hull_mesh.vertices[face]
                face_data.append({
                    'face_id': i,
                    'vertex_0_x': face_vertices[0][0],
                    'vertex_0_y': face_vertices[0][1],
                    'vertex_0_z': face_vertices[0][2],
                    'vertex_1_x': face_vertices[1][0],
                    'vertex_1_y': face_vertices[1][1],
                    'vertex_1_z': face_vertices[1][2],
                    'vertex_2_x': face_vertices[2][0],
                    'vertex_2_y': face_vertices[2][1],
                    'vertex_2_z': face_vertices[2][2]
                })
            
            # Save face details
            face_df = pd.DataFrame(face_data)
            face_csv_path = output_subdir / f"{Path(filename).stem}_convex_hull_faces.csv"
            face_df.to_csv(face_csv_path, index=False)
        
        # Save main analysis
        analysis_df = pd.DataFrame([analysis_data])
        csv_path = output_subdir / f"{Path(filename).stem}_convex_hull_analysis.csv"
        analysis_df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def save_lmgc90_analysis_csv(self, original_mesh, final_mesh, filename: str) -> str:
        """Save LMGC90-specific analysis to CSV"""
        output_subdir = self.output_dir / Path(filename).stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Calculate LMGC90 compatibility metrics
        final_vertices = len(final_mesh.vertices)
        final_faces = len(final_mesh.faces)
        is_convex = final_mesh.is_convex
        
        # LMGC90 specific analysis
        analysis_data = {
            'filename': Path(filename).stem,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'reconstruction_method': 'controlled_convex_hull',
            
            # Geometry metrics
            'final_vertices': final_vertices,
            'final_faces': final_faces,
            'is_convex': is_convex,
            'volume': final_mesh.volume if hasattr(final_mesh, 'volume') else 0,
            'surface_area': final_mesh.area if hasattr(final_mesh, 'area') else 0,
            
            # LMGC90 compatibility
            'lmgc90_ready': is_convex and (350 <= final_vertices <= 500),
            'vertex_range_ok': (350 <= final_vertices <= 500),
            'convexity_ok': is_convex,
            
            # Reduction metrics
            'original_vertices': len(original_mesh.vertices),
            'vertex_reduction_ratio': (len(original_mesh.vertices) - final_vertices) / max(len(original_mesh.vertices), 1),
            'vertex_reduction_absolute': len(original_mesh.vertices) - final_vertices,
            
            # Quality metrics
            'aspect_ratio': self._calculate_aspect_ratio(final_mesh),
            'volume_to_surface_ratio': (final_mesh.volume / final_mesh.area) if hasattr(final_mesh, 'area') and final_mesh.area > 0 else 0,
            'geometric_quality': self._calculate_geometric_quality(final_mesh),
            
            # LMGC90 simulation readiness
            'mesh_quality_score': self._calculate_mesh_quality_for_simulation(final_mesh),
            'recommended_for_simulation': self._is_suitable_for_lmgc90(final_mesh, final_vertices)
        }
        
        # Save main analysis
        analysis_df = pd.DataFrame([analysis_data])
        csv_path = output_subdir / f"{Path(filename).stem}_lmgc90_analysis.csv"
        analysis_df.to_csv(csv_path, index=False)
        
        # Save vertex coordinates for LMGC90 import
        vertices_df = pd.DataFrame(final_mesh.vertices, columns=['x', 'y', 'z'])
        vertices_df['vertex_id'] = range(len(final_mesh.vertices))
        vertices_csv_path = output_subdir / f"{Path(filename).stem}_lmgc90_vertices.csv"
        vertices_df.to_csv(vertices_csv_path, index=False)
        
        # Save faces for LMGC90 import
        faces_df = pd.DataFrame(final_mesh.faces, columns=['v1', 'v2', 'v3'])
        faces_df['face_id'] = range(len(final_mesh.faces))
        faces_csv_path = output_subdir / f"{Path(filename).stem}_lmgc90_faces.csv"
        faces_df.to_csv(faces_csv_path, index=False)
        
        return str(csv_path)
    
    def _calculate_aspect_ratio(self, mesh) -> float:
        """Calculate aspect ratio for simulation quality"""
        try:
            bounds = mesh.bounds
            dimensions = bounds[1] - bounds[0]
            max_dim = np.max(dimensions)
            min_dim = np.min(dimensions)
            return max_dim / min_dim if min_dim > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_geometric_quality(self, mesh) -> float:
        """Calculate overall geometric quality score"""
        try:
            # Combine various quality metrics
            volume = mesh.volume if hasattr(mesh, 'volume') else 0
            area = mesh.area if hasattr(mesh, 'area') else 1
            
            # Volume to surface area ratio (higher is better for spherical shapes)
            vol_surf_ratio = volume / area if area > 0 else 0
            
            # Convexity (1.0 = perfectly convex)
            convexity_score = 1.0 if mesh.is_convex else 0.5
            
            # Aspect ratio (closer to 1.0 is better)
            aspect_ratio = self._calculate_aspect_ratio(mesh)
            aspect_score = 1.0 / aspect_ratio if aspect_ratio > 0 else 0
            
            # Combined score
            quality_score = (vol_surf_ratio * 0.4 + convexity_score * 0.4 + aspect_score * 0.2)
            return min(quality_score, 1.0)
        except:
            return 0.5
    
    def _calculate_mesh_quality_for_simulation(self, mesh) -> float:
        """Calculate mesh quality specifically for LMGC90 simulation"""
        try:
            score = 0.0
            
            # Convexity (essential for LMGC90)
            if mesh.is_convex:
                score += 0.4
            
            # Vertex count (350-500 is optimal)
            vertex_count = len(mesh.vertices)
            if 350 <= vertex_count <= 500:
                score += 0.3
            elif 300 <= vertex_count < 350 or 500 < vertex_count <= 600:
                score += 0.2
            elif 250 <= vertex_count < 300 or 600 < vertex_count <= 700:
                score += 0.1
            
            # Geometric quality
            geom_quality = self._calculate_geometric_quality(mesh)
            score += geom_quality * 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _is_suitable_for_lmgc90(self, mesh, vertex_count: int) -> bool:
        """Determine if mesh is suitable for LMGC90 simulation"""
        try:
            # Must be convex
            if not mesh.is_convex:
                return False
            
            # Must have reasonable vertex count
            if not (250 <= vertex_count <= 700):  # Broader range for acceptability
                return False
            
            # Must have valid geometry
            if not hasattr(mesh, 'volume') or mesh.volume <= 0:
                return False
            
            return True
        except:
            return False
    
    def save_comparison_csv(self, original_data: Dict, final_data: Dict, 
                          processing_info: Dict, filename: str) -> str:
        """Save comparison data and add to consolidated comparison"""
        comparison_data = {
            'filename': Path(filename).stem,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            
            # Original data
            'original_vertices': original_data.get('vertex_count', 0),
            'original_faces': original_data.get('face_count', 0),
            'original_bbox_volume': original_data.get('bbox_volume', 0),
            'original_surface_area': original_data.get('surface_area', 0),
            'original_x_range': original_data.get('x_range', 0),
            'original_y_range': original_data.get('y_range', 0),
            'original_z_range': original_data.get('z_range', 0),
            'original_centroid_x': original_data.get('centroid_x', 0),
            'original_centroid_y': original_data.get('centroid_y', 0),
            'original_centroid_z': original_data.get('centroid_z', 0),
            
            # Final data
            'final_vertices': final_data.get('vertex_count', 0),
            'final_faces': final_data.get('face_count', 0),
            'final_bbox_volume': final_data.get('bbox_volume', 0),
            'final_surface_area': final_data.get('surface_area', 0),
            'final_x_range': final_data.get('x_range', 0),
            'final_y_range': final_data.get('y_range', 0),
            'final_z_range': final_data.get('z_range', 0),
            'final_centroid_x': final_data.get('centroid_x', 0),
            'final_centroid_y': final_data.get('centroid_y', 0),
            'final_centroid_z': final_data.get('centroid_z', 0),
            
            # Reduction metrics
            'vertex_reduction_ratio': (original_data.get('vertex_count', 0) - final_data.get('vertex_count', 0)) / max(original_data.get('vertex_count', 1), 1),
            'vertex_reduction_absolute': original_data.get('vertex_count', 0) - final_data.get('vertex_count', 0),
            'face_reduction_ratio': (original_data.get('face_count', 0) - final_data.get('face_count', 0)) / max(original_data.get('face_count', 1), 1),
            'face_reduction_absolute': original_data.get('face_count', 0) - final_data.get('face_count', 0),
            'volume_preservation_ratio': final_data.get('bbox_volume', 0) / max(original_data.get('bbox_volume', 1), 1),
            'surface_area_preservation_ratio': final_data.get('surface_area', 0) / max(original_data.get('surface_area', 1), 1),
            
            # Processing info
            'processing_time': processing_info.get('processing_time', 0),
            'device_used': processing_info.get('device_used', 'unknown'),
            'voxel_size_used': processing_info.get('voxel_size', 0),
            'epsilon_used': processing_info.get('epsilon', 0),
            'reconstruction_method': processing_info.get('reconstruction_method', 'unknown'),
            'meets_constraints': processing_info.get('meets_constraints', False),
            'ballast_remaining': processing_info.get('ballast_count', 0),
            'target_min_vertices': processing_info.get('target_min', 0),
            'target_max_vertices': processing_info.get('target_max', 0),
            
            # Add convex hull specific metrics if available
            'convexity_ratio': final_data.get('convexity_ratio', 0),
            'volume_preservation': final_data.get('volume_preservation', 0),
            'geometric_simplification': final_data.get('geometric_simplification', 0),
            'convex_hull_volume': final_data.get('convex_hull_volume', 0),
            'convex_hull_surface_area': final_data.get('convex_hull_surface_area', 0),
            
            # LMGC90 specific metrics if available
            'lmgc90_ready': final_data.get('lmgc90_ready', False),
            'mesh_quality_score': final_data.get('mesh_quality_score', 0),
            'recommended_for_simulation': final_data.get('recommended_for_simulation', False),
        }
        
        # Store for consolidated file
        self.comparison_data.append(comparison_data)
        self.processing_data.append(comparison_data)  # Also keep for master summary
        
        # Save individual comparison file
        output_subdir = self.output_dir / Path(filename).stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        comparison_df = pd.DataFrame([comparison_data])
        csv_path = output_subdir / f"{Path(filename).stem}_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def save_consolidated_comparison_csv(self) -> str:
        """Save consolidated comparison file with all models"""
        if not self.comparison_data:
            return ""
        
        # Create comprehensive comparison DataFrame
        comparison_df = pd.DataFrame(self.comparison_data)
        
        # Sort by filename for consistent ordering
        comparison_df = comparison_df.sort_values('filename')
        
        # Save consolidated comparison file
        csv_path = self.output_dir / "consolidated_comparison_all_models.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        # Also create a simplified version for easier analysis
        simplified_columns = [
            'filename', 'original_vertices', 'final_vertices', 'vertex_reduction_ratio',
            'vertex_reduction_absolute', 'processing_time', 'device_used', 'meets_constraints',
            'ballast_remaining', 'reconstruction_method', 'convexity_ratio', 'volume_preservation', 
            'geometric_simplification', 'lmgc90_ready', 'mesh_quality_score', 'recommended_for_simulation'
        ]
        
        # Filter to only include columns that exist
        available_columns = [col for col in simplified_columns if col in comparison_df.columns]
        simplified_df = comparison_df[available_columns].copy()
        simplified_csv_path = self.output_dir / "comparison_summary_simplified.csv"
        simplified_df.to_csv(simplified_csv_path, index=False)
        
        return str(csv_path)
    
    def save_master_summary_csv(self) -> str:
        """Save comprehensive summary and consolidated comparison"""
        if not self.processing_data:
            return ""
        
        # Save consolidated comparison first
        consolidated_path = self.save_consolidated_comparison_csv()
        
        # Create master summary (existing functionality)
        summary_df = pd.DataFrame(self.processing_data)
        csv_path = self.output_dir / "master_processing_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def calculate_mesh_info(self, mesh) -> Dict:
        """Calculate comprehensive mesh information"""
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces) if hasattr(mesh, 'faces') else np.array([])
            
            # Basic counts
            info = {
                'vertex_count': len(vertices),
                'face_count': len(faces)
            }
            
            # Bounding box and ranges
            if len(vertices) > 0:
                min_coords = vertices.min(axis=0)
                max_coords = vertices.max(axis=0)
                ranges = max_coords - min_coords
                
                info.update({
                    'x_range': ranges[0],
                    'y_range': ranges[1], 
                    'z_range': ranges[2],
                    'bbox_volume': np.prod(ranges) if np.all(ranges > 0) else 0,
                    'centroid_x': vertices[:, 0].mean(),
                    'centroid_y': vertices[:, 1].mean(),
                    'centroid_z': vertices[:, 2].mean()
                })
                
                # Surface area if possible
                try:
                    if hasattr(mesh, 'area'):
                        info['surface_area'] = mesh.area
                    elif hasattr(mesh, 'face_areas'):
                        info['surface_area'] = np.sum(mesh.face_areas)
                    else:
                        info['surface_area'] = 0
                except:
                    info['surface_area'] = 0
                    
            return info
            
        except Exception as e:
            return {'vertex_count': 0, 'face_count': 0, 'error': str(e)}

class GPUAcceleratedReducer:
    """Complete GPU-accelerated ML-based point cloud reduction system with convex hull support and LMGC90 compatibility"""
    
    def __init__(self, config: ReductionConfig = None):
        self.config = config or ReductionConfig()
        self.device = self._setup_gpu()
        self.scaler = None
        self.svm_model = None
        self.data_collector = None
        
        # Initialize GPU memory management
        if self.config.use_gpu and HAS_CUPY:
            try:
                cp.cuda.MemoryPool().set_limit(fraction=self.config.gpu_memory_fraction)
            except:
                pass
    
    def _setup_gpu(self) -> str:
        """Setup GPU environment and return device info"""
        if not self.config.use_gpu:
            logger.info("GPU acceleration disabled by configuration")
            return "cpu"
        
        gpu_info = []
        
        if HAS_CUPY:
            try:
                cp.cuda.runtime.getDeviceCount()
                gpu_info.append("CuPy")
            except:
                pass
        
        if HAS_CUML:
            gpu_info.append("cuML")
        
        if HAS_TORCH and torch.cuda.is_available():
            gpu_info.append(f"PyTorch")
        
        if gpu_info:
            device = "cuda"
            logger.info(f"GPU acceleration enabled: {', '.join(gpu_info)}")
        else:
            device = "cpu"
            logger.info("GPU libraries not available, using CPU")
        
        return device
    
    def _to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Move array to GPU if available"""
        if self.device == "cuda" and HAS_CUPY:
            try:
                return cp.asarray(array)
            except:
                return array
        return array
    
    def _to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Move array back to CPU"""
        if HAS_CUPY and hasattr(array, 'get'):
            try:
                return array.get()
            except:
                return np.array(array)
        return np.array(array)
    
    def load_mesh(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load STL mesh and extract point cloud with normals"""
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(filepath, force='mesh')
            
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                points = np.asarray(mesh.vertices, dtype=np.float32)
                
                # Compute normals using trimesh if available
                if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
                    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                else:
                    # Compute normals using Open3D
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
                    o3d_mesh.compute_vertex_normals()
                    normals = np.asarray(o3d_mesh.vertex_normals, dtype=np.float32)
            else:
                raise ValueError("Invalid mesh format - no vertices or faces found")
                
            logger.info(f"Loaded mesh: {len(points)} vertices, {len(mesh.faces)} faces")
            return points, normals
            
        except Exception as e:
            logger.error(f"Error loading mesh {filepath}: {e}")
            return None, None
    
    def voxel_downsample_gpu(self, points: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated voxel downsampling"""
        if self.device == "cuda" and HAS_CUPY and len(points) > 5000:
            try:
                # Move to GPU
                gpu_points = self._to_gpu(points)
                gpu_normals = self._to_gpu(normals)
                
                # Voxel grid computation on GPU
                voxel_size = self.config.voxel_size
                min_coords = cp.min(gpu_points, axis=0)
                voxel_indices = cp.floor((gpu_points - min_coords) / voxel_size).astype(cp.int32)
                
                # Create unique voxel keys
                voxel_keys = (voxel_indices[:, 0] * 1000000 + 
                             voxel_indices[:, 1] * 1000 + 
                             voxel_indices[:, 2])
                
                # Find unique voxels and compute centroids
                unique_keys, inverse_indices = cp.unique(voxel_keys, return_inverse=True)
                
                downsampled_points = []
                downsampled_normals = []
                
                for i in range(len(unique_keys)):
                    mask = inverse_indices == i
                    if cp.sum(mask) > 0:
                        centroid = cp.mean(gpu_points[mask], axis=0)
                        avg_normal = cp.mean(gpu_normals[mask], axis=0)
                        norm_magnitude = cp.linalg.norm(avg_normal)
                        if norm_magnitude > 1e-10:
                            avg_normal = avg_normal / norm_magnitude
                        
                        downsampled_points.append(centroid)
                        downsampled_normals.append(avg_normal)
                
                if len(downsampled_points) > 0:
                    down_points = self._to_cpu(cp.array(downsampled_points))
                    down_normals = self._to_cpu(cp.array(downsampled_normals))
                else:
                    down_points = points
                    down_normals = normals
                
                logger.info(f"GPU voxel downsampling: {len(points)} -> {len(down_points)} points")
                return down_points, down_normals
                
            except Exception as e:
                logger.warning(f"GPU downsampling failed: {e}, falling back to CPU")
        
        # Fallback to Open3D CPU implementation
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            downsampled = pcd.voxel_down_sample(voxel_size=self.config.voxel_size)
            
            down_points = np.asarray(downsampled.points)
            down_normals = np.asarray(downsampled.normals)
            
            logger.info(f"CPU voxel downsampling: {len(points)} -> {len(down_points)} points")
            return down_points, down_normals
        except Exception as e:
            logger.warning(f"Voxel downsampling failed: {e}, using original points")
            return points, normals
    
    def normalize_pointcloud(self, points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize point cloud to unit cube"""
        if self.device == "cuda" and HAS_CUPY:
            try:
                gpu_points = self._to_gpu(points)
                centroid = cp.mean(gpu_points, axis=0)
                centered = gpu_points - centroid
                scale = cp.max(cp.abs(centered))
                if scale > 0:
                    normalized = centered / scale
                else:
                    normalized = centered
                
                return self._to_cpu(normalized), {
                    'centroid': self._to_cpu(centroid),
                    'scale': float(self._to_cpu(scale))
                }
            except Exception as e:
                logger.warning(f"GPU normalization failed: {e}, using CPU")
        
        # CPU fallback
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        scale = np.max(np.abs(centered))
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered
        
        return normalized, {'centroid': centroid, 'scale': scale}
    
    def extract_features_gpu(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """GPU-accelerated feature extraction"""
        n_points = len(points)
        
        # For small point clouds, use CPU
        if n_points < 1000 or self.device == "cpu" or not HAS_CUPY:
            return self._extract_features_cpu(points, normals)
        
        try:
            # GPU implementation
            gpu_points = self._to_gpu(points)
            gpu_normals = self._to_gpu(normals)
            
            # Calculate features in batches
            batch_size = min(self.config.batch_size, n_points)
            curvatures = cp.zeros(n_points, dtype=cp.float32)
            densities = cp.zeros(n_points, dtype=cp.float32)
            
            global_centroid = cp.mean(gpu_points, axis=0)
            centroid_distances = cp.linalg.norm(gpu_points - global_centroid, axis=1)
            
            # Process in batches to manage memory
            for start_idx in range(0, n_points, batch_size):
                end_idx = min(start_idx + batch_size, n_points)
                batch_points = gpu_points[start_idx:end_idx]
                
                # Compute pairwise distances for the batch
                batch_size_actual = end_idx - start_idx
                k = min(20, n_points - 1)
                
                for i in range(batch_size_actual):
                    point_idx = start_idx + i
                    point = gpu_points[point_idx]
                    
                    # Find k nearest neighbors
                    distances = cp.linalg.norm(gpu_points - point, axis=1)
                    neighbor_indices = cp.argpartition(distances, k)[:k+1]
                    neighbor_indices = neighbor_indices[1:]  # Exclude the point itself
                    
                    if len(neighbor_indices) > 3:
                        neighbors = gpu_points[neighbor_indices]
                        
                        # Curvature via PCA
                        centered_neighbors = neighbors - cp.mean(neighbors, axis=0)
                        try:
                            cov_matrix = cp.cov(centered_neighbors.T)
                            eigenvalues = cp.linalg.eigvals(cov_matrix)
                            eigenvalues = cp.sort(eigenvalues)
                            eigenvalues = cp.real(eigenvalues)  # Take real part
                            
                            if cp.sum(eigenvalues) > 1e-10:
                                curvatures[point_idx] = eigenvalues[0] / cp.sum(eigenvalues)
                            else:
                                curvatures[point_idx] = 0.0
                        except:
                            curvatures[point_idx] = 0.0
                        
                        # Density
                        densities[point_idx] = len(neighbor_indices)
            
            # Combine features
            normal_magnitudes = cp.linalg.norm(gpu_normals, axis=1)
            features = cp.column_stack([
                curvatures,
                densities,
                centroid_distances,
                normal_magnitudes
            ])
            
            return self._to_cpu(features)
            
        except Exception as e:
            logger.warning(f"GPU feature extraction failed: {e}, falling back to CPU")
            return self._extract_features_cpu(points, normals)
    
    def _extract_features_cpu(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """CPU fallback for feature extraction"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            n_points = len(points)
            curvatures = np.zeros(n_points)
            densities = np.zeros(n_points)
            
            global_centroid = np.mean(points, axis=0)
            centroid_distances = np.linalg.norm(points - global_centroid, axis=1)
            
            for i in range(n_points):
                try:
                    [k, idx, _] = kdtree.search_knn_vector_3d(points[i], 20)
                    
                    if len(idx) > 3:
                        neighbor_points = points[idx[1:]]  # Exclude the point itself
                        cov_matrix = np.cov(neighbor_points.T)
                        eigenvalues = np.linalg.eigvals(cov_matrix)
                        eigenvalues = np.sort(np.real(eigenvalues))
                        
                        if np.sum(eigenvalues) > 1e-10:
                            curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
                        
                        densities[i] = len(idx) - 1
                except:
                    curvatures[i] = 0.0
                    densities[i] = 1.0
            
            features = np.column_stack([
                curvatures,
                densities,
                centroid_distances,
                np.linalg.norm(normals, axis=1)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"CPU feature extraction failed: {e}")
            # Return basic features as fallback
            n_points = len(points)
            centroid_distances = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
            return np.column_stack([
                np.random.rand(n_points) * 0.1,  # Random curvature
                np.ones(n_points),  # Uniform density
                centroid_distances,
                np.linalg.norm(normals, axis=1)
            ])
    
    def train_svm_importance_gpu(self, features: np.ndarray) -> np.ndarray:
        """GPU-accelerated SVM training"""
        n_samples = len(features)
        n_train = max(100, int(n_samples * self.config.svm_sample_ratio))
        
        # Create pseudo-labels based on feature statistics
        curvature_threshold = np.percentile(features[:, 0], 75)
        density_threshold = np.percentile(features[:, 1], 50)
        
        # Sample training data
        train_indices = np.random.choice(n_samples, n_train, replace=False)
        train_features = features[train_indices]
        
        # Generate pseudo-labels
        train_labels = ((train_features[:, 0] > curvature_threshold) | 
                       (train_features[:, 1] > density_threshold)).astype(int)
        
        # Ensure some positive examples
        if np.sum(train_labels) < n_train * 0.2:
            top_indices = np.argsort(train_features[:, 0])[-int(n_train * 0.3):]
            train_labels[top_indices] = 1
        
        # Use GPU-accelerated SVM if available
        if self.device == "cuda" and HAS_CUML:
            try:
                self.scaler = cuStandardScaler()
                scaled_features = self.scaler.fit_transform(train_features)
                
                self.svm_model = cuSVC(kernel='rbf', probability=True, random_state=42)
                self.svm_model.fit(scaled_features, train_labels)
                
                all_scaled = self.scaler.transform(features)
                importance_scores = self.svm_model.predict_proba(all_scaled)[:, 1]
                
                logger.info("GPU SVM training completed")
                return importance_scores
                
            except Exception as e:
                logger.warning(f"GPU SVM failed: {e}, falling back to CPU")
        
        # Fallback to CPU
        try:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(train_features)
            
            self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            self.svm_model.fit(scaled_features, train_labels)
            
            all_scaled = self.scaler.transform(features)
            importance_scores = self.svm_model.predict_proba(all_scaled)[:, 1]
            
            logger.info("CPU SVM training completed")
            return importance_scores
            
        except Exception as e:
            logger.warning(f"SVM training failed: {e}, using random importance")
            # Fallback to feature-based importance
            return (features[:, 0] + features[:, 1]) / 2
    
    def knn_reinforcement_gpu(self, points: np.ndarray, importance_mask: np.ndarray) -> np.ndarray:
        """GPU-accelerated KNN reinforcement"""
        important_indices = np.where(importance_mask)[0]
        
        if len(important_indices) == 0:
            return importance_mask
        
        enhanced_mask = importance_mask.copy()
        
        try:
            if self.device == "cuda" and HAS_CUML and len(points) > 1000:
                try:
                    knn = cuNearestNeighbors(n_neighbors=self.config.knn_neighbors + 1)
                    knn.fit(points)
                    
                    for idx in important_indices:
                        distances, indices = knn.kneighbors([points[idx]])
                        neighbor_indices = indices[0][1:]  # Exclude the point itself
                        enhanced_mask[neighbor_indices] = True
                    
                    logger.info(f"GPU KNN reinforcement: {np.sum(importance_mask)} -> {np.sum(enhanced_mask)} points")
                    return enhanced_mask
                    
                except Exception as e:
                    logger.warning(f"GPU KNN failed: {e}, falling back to CPU")
            
            # CPU fallback
            knn = NearestNeighbors(n_neighbors=self.config.knn_neighbors + 1)
            knn.fit(points)
            
            for idx in important_indices:
                distances, indices = knn.kneighbors([points[idx]])
                neighbor_indices = indices[0][1:]
                enhanced_mask[neighbor_indices] = True
            
            logger.info(f"CPU KNN reinforcement: {np.sum(importance_mask)} -> {np.sum(enhanced_mask)} points")
            return enhanced_mask
            
        except Exception as e:
            logger.warning(f"KNN reinforcement failed: {e}, using original mask")
            return importance_mask
    
    def hybrid_merging_gpu(self, points: np.ndarray, normals: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated hybrid merging"""
        
        if self.device == "cuda" and HAS_CUPY and len(points) > 1000:
            try:
                return self._hybrid_merging_gpu_impl(points, normals, epsilon)
            except Exception as e:
                logger.warning(f"GPU hybrid merging failed: {e}, falling back to CPU")
        
        return self._hybrid_merging_cpu(points, normals, epsilon)
    
    def _hybrid_merging_gpu_impl(self, points: np.ndarray, normals: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """GPU implementation of hybrid merging"""
        gpu_points = self._to_gpu(points)
        gpu_normals = self._to_gpu(normals)
        
        # Radius merge on GPU
        n_points = len(points)
        merged_points = []
        merged_normals = []
        used_mask = cp.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            if used_mask[i]:
                continue
            
            # Find points within epsilon radius
            distances = cp.linalg.norm(gpu_points - gpu_points[i], axis=1)
            close_mask = distances <= epsilon
            close_indices = cp.where(close_mask)[0]
            
            # Mark as used
            used_mask[close_indices] = True
            
            # Compute centroid and average normal
            cluster_points = gpu_points[close_indices]
            cluster_normals = gpu_normals[close_indices]
            
            centroid = cp.mean(cluster_points, axis=0)
            avg_normal = cp.mean(cluster_normals, axis=0)
            normal_magnitude = cp.linalg.norm(avg_normal)
            if normal_magnitude > 1e-10:
                avg_normal = avg_normal / normal_magnitude
            
            merged_points.append(centroid)
            merged_normals.append(avg_normal)
        
        if len(merged_points) == 0:
            return points, normals
        
        merged_points = cp.array(merged_points)
        merged_normals = cp.array(merged_normals)
        
        # DBSCAN cleanup
        if HAS_CUML and len(merged_points) > self.config.dbscan_min_samples:
            try:
                dbscan = cuDBSCAN(eps=epsilon * 2, min_samples=self.config.dbscan_min_samples)
                cluster_labels = dbscan.fit_predict(merged_points)
                
                final_points = []
                final_normals = []
                
                unique_labels = cp.unique(cluster_labels)
                for label in unique_labels:
                    if label == -1:  # Outliers
                        continue
                    
                    cluster_mask = cluster_labels == label
                    if cp.sum(cluster_mask) > 0:
                        cluster_points = merged_points[cluster_mask]
                        cluster_normals = merged_normals[cluster_mask]
                        
                        centroid = cp.mean(cluster_points, axis=0)
                        avg_normal = cp.mean(cluster_normals, axis=0)
                        normal_magnitude = cp.linalg.norm(avg_normal)
                        if normal_magnitude > 1e-10:
                            avg_normal = avg_normal / normal_magnitude
                        
                        final_points.append(centroid)
                        final_normals.append(avg_normal)
                
                if len(final_points) > 0:
                    merged_points = cp.array(final_points)
                    merged_normals = cp.array(final_normals)
            except Exception as e:
                logger.warning(f"GPU DBSCAN failed: {e}")
        
        result_points = self._to_cpu(merged_points)
        result_normals = self._to_cpu(merged_normals)
        
        logger.info(f"GPU hybrid merging: {len(points)} -> {len(result_points)} points")
        return result_points, result_normals
    
    def _hybrid_merging_cpu(self, points: np.ndarray, normals: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for hybrid merging"""
        try:
            # Radius merge
            merged_points = []
            merged_normals = []
            used_indices = set()
            
            for i, point in enumerate(points):
                if i in used_indices:
                    continue
                
                distances = np.linalg.norm(points - point, axis=1)
                close_indices = np.where(distances <= epsilon)[0]
                used_indices.update(close_indices)
                
                cluster_points = points[close_indices]
                cluster_normals = normals[close_indices]
                
                centroid = np.mean(cluster_points, axis=0)
                avg_normal = np.mean(cluster_normals, axis=0)
                normal_magnitude = np.linalg.norm(avg_normal)
                if normal_magnitude > 1e-10:
                    avg_normal = avg_normal / normal_magnitude
                
                merged_points.append(centroid)
                merged_normals.append(avg_normal)
            
            if len(merged_points) == 0:
                return points, normals
            
            merged_points = np.array(merged_points)
            merged_normals = np.array(merged_normals)
            
            # DBSCAN cleanup
            if len(merged_points) > self.config.dbscan_min_samples:
                try:
                    dbscan = DBSCAN(eps=epsilon * 2, min_samples=self.config.dbscan_min_samples)
                    cluster_labels = dbscan.fit_predict(merged_points)
                    
                    final_points = []
                    final_normals = []
                    
                    for label in np.unique(cluster_labels):
                        if label == -1:
                            continue
                        
                        cluster_mask = cluster_labels == label
                        if np.sum(cluster_mask) > 0:
                            cluster_points = merged_points[cluster_mask]
                            cluster_normals = merged_normals[cluster_mask]
                            
                            centroid = np.mean(cluster_points, axis=0)
                            avg_normal = np.mean(cluster_normals, axis=0)
                            normal_magnitude = np.linalg.norm(avg_normal)
                            if normal_magnitude > 1e-10:
                                avg_normal = avg_normal / normal_magnitude
                            
                            final_points.append(centroid)
                            final_normals.append(avg_normal)
                    
                    if len(final_points) > 0:
                        merged_points = np.array(final_points)
                        merged_normals = np.array(final_normals)
                        
                except Exception as e:
                    logger.warning(f"CPU DBSCAN failed: {e}")
            
            logger.info(f"CPU hybrid merging: {len(points)} -> {len(merged_points)} points")
            return merged_points, merged_normals
            
        except Exception as e:
            logger.error(f"Hybrid merging failed: {e}")
            # Return a subset of original points as fallback
            n_keep = min(len(points), self.config.target_points_max)
            indices = np.random.choice(len(points), n_keep, replace=False)
            return points[indices], normals[indices]
    
    def adaptive_parameter_tuning(self, points: np.ndarray, normals: np.ndarray, 
                                 importance_mask: np.ndarray) -> Dict:
        """Adaptive parameter estimation via grid search"""
        important_points = points[importance_mask]
        important_normals = normals[importance_mask]
        
        if len(important_points) == 0:
            return {'epsilon': 0.05, 'score': 0, 'n_points': 0}
        
        epsilon_values = np.linspace(self.config.epsilon_range[0], 
                                   self.config.epsilon_range[1], 8)
        
        best_params = {'epsilon': self.config.epsilon_range[0], 'score': float('inf'), 'n_points': 0}
        
        for epsilon in epsilon_values:
            try:
                merged_points, _ = self.hybrid_merging_gpu(important_points, important_normals, epsilon)
                
                n_points = len(merged_points)
                target_mid = (self.config.target_points_min + self.config.target_points_max) / 2
                score = abs(n_points - target_mid)
                
                if score < best_params['score'] and n_points <= self.config.target_points_max:
                    best_params = {'epsilon': epsilon, 'score': score, 'n_points': n_points}
                    
            except Exception as e:
                logger.warning(f"Parameter tuning failed for epsilon={epsilon}: {e}")
                continue
        
        logger.info(f"Best parameters: epsilon={best_params['epsilon']:.4f}, points={best_params['n_points']}")
        return best_params
    
    def denormalize_pointcloud(self, points: np.ndarray, norm_params: Dict) -> np.ndarray:
        """Denormalize point cloud back to original coordinate frame"""
        return points * norm_params['scale'] + norm_params['centroid']
    
    def _convex_hull_reconstruction(self, points: np.ndarray, normals: np.ndarray) -> Optional[trimesh.Trimesh]:
        """Convex hull reconstruction method"""
        try:
            # Create a basic mesh from the points
            temp_mesh = trimesh.Trimesh(vertices=points)
            
            # Calculate convex hull
            convex_hull = temp_mesh.convex_hull
            
            # Log convex hull properties
            if hasattr(temp_mesh, 'volume') and hasattr(convex_hull, 'volume'):
                convexity = temp_mesh.volume / convex_hull.volume if convex_hull.volume > 0 else 0
                logger.info(f"  🔺 Convex hull: {len(convex_hull.vertices)} vertices, "
                           f"{len(convex_hull.faces)} faces, convexity: {convexity:.3f}")
            else:
                logger.info(f"  🔺 Convex hull: {len(convex_hull.vertices)} vertices, "
                           f"{len(convex_hull.faces)} faces")
            
            # Log face details if detailed analysis is enabled
            if self.config.detailed_analysis:
                logger.debug("  🔍 Convex hull face analysis:")
                for i, face in enumerate(convex_hull.faces[:5]):  # Show first 5 faces
                    face_vertices = convex_hull.vertices[face]
                    logger.debug(f"    Face {i}: {face_vertices}")
                if len(convex_hull.faces) > 5:
                    logger.debug(f"    ... and {len(convex_hull.faces) - 5} more faces")
            
            return convex_hull
            
        except Exception as e:
            logger.error(f"Convex hull reconstruction failed: {e}")
            return None
    
    def _controlled_convex_hull_reconstruction(self, points: np.ndarray, normals: np.ndarray, 
                                              target_min: int = 350, target_max: int = 500) -> Optional[trimesh.Trimesh]:
        """
        Controlled convex hull reconstruction that respects vertex constraints.
        Perfect for LMGC90 - gives you convex geometry within specified vertex range.
        """
        try:
            logger.info(f"  🎯 Target vertex range: {target_min}-{target_max} (controlled convex hull)")
            
            # Step 1: Intelligent point reduction to target range
            if len(points) > target_max:
                logger.debug(f"  📉 Reducing {len(points)} points to target range")
                reduced_points, reduced_normals = self._intelligent_point_reduction(
                    points, normals, target_max)
            else:
                reduced_points = points
                reduced_normals = normals
            
            logger.info(f"  🔄 Processing {len(reduced_points)} points for convex hull")
            
            # Step 2: Compute convex hull of reduced point set
            temp_mesh = trimesh.Trimesh(vertices=reduced_points)
            convex_hull = temp_mesh.convex_hull
            
            initial_vertices = len(convex_hull.vertices)
            logger.info(f"  🔺 Initial convex hull: {initial_vertices} vertices, {len(convex_hull.faces)} faces")
            
            # Step 3: Further decimation if still too many vertices
            final_mesh = convex_hull
            if initial_vertices > target_max:
                logger.debug(f"  ⚙️  Convex hull has {initial_vertices} vertices, decimating to {target_max}")
                final_mesh = self._decimate_convex_mesh(convex_hull, target_max)
            
            # Step 4: Ensure minimum vertex count
            if len(final_mesh.vertices) < target_min:
                logger.debug(f"  ⚙️  Result has {len(final_mesh.vertices)} vertices, less than minimum {target_min}")
                # Use a slightly larger point set and recompute
                if len(points) > len(reduced_points):
                    larger_target = min(target_min * 2, len(points))
                    reduced_points, reduced_normals = self._intelligent_point_reduction(
                        points, normals, larger_target)
                    temp_mesh = trimesh.Trimesh(vertices=reduced_points)
                    convex_hull = temp_mesh.convex_hull
                    if len(convex_hull.vertices) > target_max:
                        final_mesh = self._decimate_convex_mesh(convex_hull, target_max)
                    else:
                        final_mesh = convex_hull
            
            # Step 5: Ensure convexity (should be maintained, but verify)
            if not final_mesh.is_convex:
                logger.warning("  ⚠️  Result is not convex, recomputing convex hull")
                final_mesh = final_mesh.convex_hull
            
            # Calculate final metrics
            original_volume = temp_mesh.convex_hull.volume if hasattr(temp_mesh, 'volume') else 0
            final_volume = final_mesh.volume if hasattr(final_mesh, 'volume') else 0
            convexity = final_volume / original_volume if original_volume > 0 else 1.0
            
            final_vertices = len(final_mesh.vertices)
            final_faces = len(final_mesh.faces)
            
            logger.info(f"  ✅ Final controlled convex hull: {final_vertices} vertices, {final_faces} faces")
            logger.info(f"  🎯 Vertex constraint: {'✓ SATISFIED' if target_min <= final_vertices <= target_max else '✗ NOT SATISFIED'}")
            logger.info(f"  📊 Volume preservation: {convexity:.3f}")
            logger.info(f"  🔺 Convexity verified: {'✓ YES' if final_mesh.is_convex else '✗ NO'}")
            
            return final_mesh
            
        except Exception as e:
            logger.error(f"Controlled convex hull reconstruction failed: {e}")
            return None
    
    def _intelligent_point_reduction(self, points: np.ndarray, normals: np.ndarray, 
                                   target_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intelligently reduce points while preserving geometric features.
        Uses importance-based sampling to keep the most geometrically significant points.
        """
        try:
            if len(points) <= target_count:
                return points, normals
            
            logger.debug(f"    🧮 Intelligent reduction: {len(points)} → {target_count} points")
            
            # Method 1: Use existing feature extraction for importance scoring
            features = self.extract_features_gpu(points, normals)
            importance_scores = self.train_svm_importance_gpu(features)
            
            # Method 2: Add geometric diversity sampling
            # Select points that are well-distributed in space
            diversity_scores = self._calculate_spatial_diversity(points)
            
            # Method 3: Boundary point priority
            # Points on the boundary of the point cloud are more important for convex hull
            boundary_scores = self._calculate_boundary_importance(points)
            
            # Combine all importance metrics
            combined_scores = (importance_scores * 0.4 + 
                             diversity_scores * 0.3 + 
                             boundary_scores * 0.3)
            
            # Select top points based on combined importance
            top_indices = np.argsort(combined_scores)[-target_count:]
            
            reduced_points = points[top_indices]
            reduced_normals = normals[top_indices]
            
            logger.debug(f"    ✅ Intelligent reduction complete")
            return reduced_points, reduced_normals
            
        except Exception as e:
            logger.warning(f"Intelligent reduction failed: {e}, using random sampling")
            # Fallback to random sampling
            indices = np.random.choice(len(points), target_count, replace=False)
            return points[indices], normals[indices]
    
    def _calculate_spatial_diversity(self, points: np.ndarray) -> np.ndarray:
        """Calculate spatial diversity scores - points in less dense areas get higher scores"""
        try:
            # Use KD-tree for efficient nearest neighbor search
            nbrs = NearestNeighbors(n_neighbors=min(20, len(points)), algorithm='auto').fit(points)
            distances, indices = nbrs.kneighbors(points)
            
            # Average distance to neighbors (higher = more isolated = more important)
            diversity_scores = np.mean(distances[:, 1:], axis=1)  # Exclude self
            
            # Normalize to [0, 1]
            if np.max(diversity_scores) > np.min(diversity_scores):
                diversity_scores = (diversity_scores - np.min(diversity_scores)) / (np.max(diversity_scores) - np.min(diversity_scores))
            
            return diversity_scores
            
        except Exception as e:
            logger.warning(f"Spatial diversity calculation failed: {e}")
            return np.random.rand(len(points))
    
    def _calculate_boundary_importance(self, points: np.ndarray) -> np.ndarray:
        """Calculate boundary importance - points closer to the convex hull boundary are more important"""
        try:
            # Create temporary mesh and get its convex hull
            temp_mesh = trimesh.Trimesh(vertices=points)
            hull = temp_mesh.convex_hull
            
            # Calculate distance from each point to the convex hull surface
            distances = []
            for point in points:
                # Find closest point on hull surface
                closest_point, distance, face_id = trimesh.proximity.closest_point(hull, [point])
                distances.append(distance[0])
            
            distances = np.array(distances)
            
            # Invert distances so boundary points (distance ≈ 0) get high scores
            boundary_scores = 1.0 / (distances + 1e-6)
            
            # Normalize to [0, 1]
            if np.max(boundary_scores) > np.min(boundary_scores):
                boundary_scores = (boundary_scores - np.min(boundary_scores)) / (np.max(boundary_scores) - np.min(boundary_scores))
            
            return boundary_scores
            
        except Exception as e:
            logger.warning(f"Boundary importance calculation failed: {e}")
            return np.random.rand(len(points))
    
    def _decimate_convex_mesh(self, mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
        """
        Decimate a convex mesh while preserving convexity.
        Uses edge collapse that maintains the convex property.
        """
        try:
            logger.debug(f"    🔧 Decimating mesh: {len(mesh.vertices)} → {target_vertices} vertices")
            
            current_mesh = mesh.copy()
            
            # Method 1: Try trimesh simplification
            try:
                # Calculate reduction ratio
                reduction_ratio = target_vertices / len(mesh.vertices)
                simplified = current_mesh.simplify_quadric_decimation(int(len(mesh.faces) * reduction_ratio))
                
                if simplified is not None and len(simplified.vertices) > 0:
                    # Ensure result is still convex
                    if not simplified.is_convex:
                        simplified = simplified.convex_hull
                    
                    if target_vertices * 0.8 <= len(simplified.vertices) <= target_vertices * 1.2:
                        logger.debug(f"    ✅ Quadric decimation successful: {len(simplified.vertices)} vertices")
                        return simplified
            except Exception as e:
                logger.debug(f"    ⚠️  Quadric decimation failed: {e}")
            
            # Method 2: Vertex clustering approach
            try:
                # Group vertices into clusters and take centroids
                kmeans = KMeans(n_clusters=target_vertices, random_state=42)
                clusters = kmeans.fit_predict(current_mesh.vertices)
                
                # Calculate cluster centroids
                centroids = []
                for i in range(target_vertices):
                    cluster_points = current_mesh.vertices[clusters == i]
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)
                        centroids.append(centroid)
                
                if len(centroids) >= 4:  # Need at least 4 points for a 3D convex hull
                    centroids = np.array(centroids)
                    # Create convex hull of centroids
                    temp_mesh = trimesh.Trimesh(vertices=centroids)
                    decimated_mesh = temp_mesh.convex_hull
                    
                    logger.debug(f"    ✅ Clustering decimation successful: {len(decimated_mesh.vertices)} vertices")
                    return decimated_mesh
                    
            except Exception as e:
                logger.debug(f"    ⚠️  Clustering decimation failed: {e}")
            
            # Method 3: Random vertex sampling as fallback
            logger.debug(f"    🎲 Using random vertex sampling as fallback")
            n_vertices = len(current_mesh.vertices)
            if n_vertices > target_vertices:
                indices = np.random.choice(n_vertices, target_vertices, replace=False)
                sampled_vertices = current_mesh.vertices[indices]
                temp_mesh = trimesh.Trimesh(vertices=sampled_vertices)
                return temp_mesh.convex_hull
            
            return current_mesh
            
        except Exception as e:
            logger.warning(f"Mesh decimation failed: {e}, returning original")
            return mesh
    
    def surface_reconstruction(self, points: np.ndarray, normals: np.ndarray, 
                             method: str = 'poisson') -> Optional[trimesh.Trimesh]:
        """Surface reconstruction with multiple backend options including controlled convex hull"""
        try:
            if method == 'controlled_convex_hull':
                # Use controlled convex hull with vertex constraints
                return self._controlled_convex_hull_reconstruction(
                    points, normals, 
                    self.config.target_points_min, 
                    self.config.target_points_max
                )
            elif method == 'convex_hull':
                # Original convex hull (no constraints)
                return self._convex_hull_reconstruction(points, normals)
            
            # For other methods, use Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            if method == 'poisson':
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, width=0, scale=1.1, linear_fit=False)
            elif method == 'ball_pivoting':
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radius = 2 * avg_dist
                radii = [radius, radius * 2]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))
            elif method == 'alpha_shapes':
                alpha = 0.03
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            else:
                raise ValueError(f"Unknown reconstruction method: {method}")
            
            # Convert Open3D mesh to Trimesh
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            if len(vertices) > 0 and len(faces) > 0:
                return trimesh.Trimesh(vertices=vertices, faces=faces)
            else:
                logger.warning("Reconstruction produced empty mesh")
                return None
                
        except Exception as e:
            logger.error(f"Surface reconstruction failed: {e}")
            return None
    
    def process_single_mesh_with_csv(self, input_path: str, output_dir: str) -> Dict:
        """Enhanced process_single_mesh with comprehensive CSV data collection, convex hull, and LMGC90 support"""
        start_time = time.time()
        filename = Path(input_path).stem
        
        # Initialize data collector
        if self.data_collector is None:
            self.data_collector = DataCollector(output_dir)
        
        logger.info(f"🔄 Processing {filename} with CSV data collection...")
        
        try:
            # Step 1: Load mesh and collect original data
            logger.debug(f"Step 1: Loading mesh and collecting original data")
            points, normals = self.load_mesh(input_path)
            if points is None or normals is None:
                return {'status': 'failed', 'error': 'Failed to load mesh'}
            
            # Load original mesh for comprehensive info
            try:
                original_mesh = trimesh.load(input_path, force='mesh')
                original_mesh_info = self.data_collector.calculate_mesh_info(original_mesh)
            except:
                original_mesh_info = {'vertex_count': len(points), 'face_count': 0}
            
            # Save original point cloud data
            if self.config.save_stages:
                original_csv = self.data_collector.save_original_pointcloud_csv(
                    points, normals, filename, original_mesh_info)
                logger.info(f"  💾 Original data saved: {original_csv}")
            
            original_count = len(points)
            logger.info(f"  📊 Loaded: {original_count} vertices")
            
            # Step 2: Voxel downsampling if needed
            if original_count > 10000:
                logger.debug("Step 2: Applying voxel downsampling")
                points, normals = self.voxel_downsample_gpu(points, normals)
                logger.info(f"  📉 After downsampling: {len(points)} vertices")
                
                # Save downsampled data
                if self.config.save_stages:
                    downsample_csv = self.data_collector.save_processing_stage_csv(
                        points, normals, None, None, filename, "downsampled")
                    logger.debug(f"  💾 Downsampled data saved: {downsample_csv}")
            else:
                logger.debug("Step 2: Skipping downsampling (small mesh)")
            
            # Step 3: Normalization
            logger.debug("Step 3: Normalizing point cloud")
            normalized_points, norm_params = self.normalize_pointcloud(points)
            
            # Skip feature extraction and ML processing for controlled convex hull
            if self.config.reconstruction_method == 'controlled_convex_hull':
                logger.debug("Steps 4-10: Skipping ML processing for controlled convex hull")
                simplified_points = normalized_points
                simplified_normals = normals
            else:
                # Step 4: Feature extraction
                logger.debug("Step 4: Extracting geometric features")
                features = self.extract_features_gpu(normalized_points, normals)
                logger.info(f"  🧮 Features extracted: {features.shape[1]} features per point")
                
                # Step 5: SVM importance classification
                logger.debug("Step 5: Training SVM for importance classification")
                importance_scores = self.train_svm_importance_gpu(features)
                importance_threshold = np.percentile(importance_scores, 70)
                importance_mask = importance_scores > importance_threshold
                logger.info(f"  🎯 Important points identified: {np.sum(importance_mask)}/{len(points)}")
                
                # Save feature extraction stage
                if self.config.save_stages:
                    features_csv = self.data_collector.save_processing_stage_csv(
                        normalized_points, normals, features, importance_scores, filename, "features_extracted")
                    logger.debug(f"  💾 Features data saved: {features_csv}")
                
                # Step 6: KNN reinforcement
                logger.debug("Step 6: Applying KNN reinforcement")
                enhanced_mask = self.knn_reinforcement_gpu(normalized_points, importance_mask)
                logger.info(f"  🔗 After KNN reinforcement: {np.sum(enhanced_mask)} points")
                
                # Step 7: Adaptive parameter tuning (skip for convex hull methods)
                if self.config.reconstruction_method not in ['convex_hull', 'controlled_convex_hull']:
                    logger.debug("Step 7: Tuning parameters adaptively")
                    best_params = self.adaptive_parameter_tuning(
                        normalized_points, normals, enhanced_mask)
                    logger.info(f"  ⚙️  Optimal epsilon: {best_params['epsilon']:.4f}")
                else:
                    logger.debug("Step 7: Skipping parameter tuning (convex hull method)")
                    best_params = {'epsilon': 0.05, 'score': 0, 'n_points': 0}
                
                # Step 8: Hybrid merging (skip for convex hull methods)
                if self.config.reconstruction_method not in ['convex_hull', 'controlled_convex_hull']:
                    logger.debug("Step 8: Applying hybrid merging")
                    important_points = normalized_points[enhanced_mask]
                    important_normals = normals[enhanced_mask]
                    
                    if len(important_points) == 0:
                        logger.warning("  ⚠️  No important points found, using all points")
                        important_points = normalized_points
                        important_normals = normals
                    
                    simplified_points, simplified_normals = self.hybrid_merging_gpu(
                        important_points, important_normals, best_params['epsilon'])
                    logger.info(f"  🔄 After merging: {len(simplified_points)} points")
                else:
                    logger.debug("Step 8: Skipping hybrid merging (convex hull will use all points)")
                    simplified_points = normalized_points
                    simplified_normals = normals
                
                # Step 9: Ensure constraints are met (skip for convex hull methods)
                if self.config.reconstruction_method not in ['convex_hull', 'controlled_convex_hull']:
                    logger.debug("Step 9: Enforcing vertex constraints")
                    final_count = len(simplified_points)
                    constraint_adjustments = []
                    
                    if final_count > self.config.target_points_max:
                        reduction_ratio = self.config.target_points_max / final_count
                        n_keep = int(final_count * reduction_ratio)
                        keep_indices = np.random.choice(final_count, n_keep, replace=False)
                        simplified_points = simplified_points[keep_indices]
                        simplified_normals = simplified_normals[keep_indices]
                        final_count = len(simplified_points)
                        constraint_adjustments.append(f"reduced to {final_count} (max constraint)")
                        
                    elif final_count < self.config.target_points_min:
                        if len(normalized_points) > final_count:
                            n_add = min(self.config.target_points_min - final_count, 
                                       len(normalized_points) - final_count)
                            remaining_indices = np.setdiff1d(np.arange(len(normalized_points)), 
                                                           np.arange(len(simplified_points)))
                            if len(remaining_indices) > 0:
                                add_indices = np.random.choice(remaining_indices, 
                                                             min(n_add, len(remaining_indices)), 
                                                             replace=False)
                                additional_points = normalized_points[add_indices]
                                additional_normals = normals[add_indices]
                                simplified_points = np.vstack([simplified_points, additional_points])
                                simplified_normals = np.vstack([simplified_normals, additional_normals])
                                final_count = len(simplified_points)
                                constraint_adjustments.append(f"expanded to {final_count} (min constraint)")
                    
                    if constraint_adjustments:
                        logger.info(f"  📏 Constraint adjustments: {', '.join(constraint_adjustments)}")
                else:
                    logger.debug("Step 9: Skipping constraint enforcement (convex hull method)")
            
            final_count = len(simplified_points)
            
            # Step 10: Denormalization
            logger.debug("Step 10: Denormalizing to original coordinates")
            final_points = self.denormalize_pointcloud(simplified_points, norm_params)
            
            # Step 11: Surface reconstruction
            logger.debug(f"Step 11: Surface reconstruction using {self.config.reconstruction_method}")
            reconstructed_mesh = self.surface_reconstruction(
                final_points, simplified_normals, self.config.reconstruction_method)
            
            # Special handling for different convex hull methods
            convex_hull_csv = None
            lmgc90_csv = None
            
            if self.config.reconstruction_method == 'convex_hull' and reconstructed_mesh is not None:
                # Save convex hull specific analysis
                convex_hull_csv = self.data_collector.save_convex_hull_analysis_csv(
                    original_mesh, reconstructed_mesh, filename)
                logger.info(f"  📊 Convex hull analysis saved: {convex_hull_csv}")
            
            elif self.config.reconstruction_method == 'controlled_convex_hull' and reconstructed_mesh is not None:
                # Verify LMGC90 compatibility
                vertex_count = len(reconstructed_mesh.vertices)
                face_count = len(reconstructed_mesh.faces)
                is_convex = reconstructed_mesh.is_convex
                
                logger.info(f"  🎯 LMGC90 Compatibility Check:")
                logger.info(f"    Vertices: {vertex_count} ({'✓' if self.config.target_points_min <= vertex_count <= self.config.target_points_max else '✗'})")
                logger.info(f"    Faces: {face_count}")
                logger.info(f"    Convex: {'✓ YES' if is_convex else '✗ NO'}")
                logger.info(f"    Ready for LMGC90: {'✅ YES' if is_convex and self.config.target_points_min <= vertex_count <= self.config.target_points_max else '❌ NO'}")
                
                # Save LMGC90 specific analysis
                lmgc90_csv = self.data_collector.save_lmgc90_analysis_csv(
                    original_mesh, reconstructed_mesh, filename)
                logger.info(f"  📊 LMGC90 analysis saved: {lmgc90_csv}")
            
            # Calculate final mesh info
            if reconstructed_mesh is not None:
                final_mesh_info = self.data_collector.calculate_mesh_info(reconstructed_mesh)
                
                # Add method-specific metrics
                if self.config.reconstruction_method in ['convex_hull', 'controlled_convex_hull']:
                    convex_metrics = self.data_collector.calculate_convex_hull_metrics(
                        original_mesh, reconstructed_mesh)
                    final_mesh_info.update(convex_metrics)
                    
                    if self.config.reconstruction_method == 'controlled_convex_hull':
                        # Add LMGC90 specific metrics
                        final_mesh_info.update({
                            'lmgc90_ready': reconstructed_mesh.is_convex and (self.config.target_points_min <= len(reconstructed_mesh.vertices) <= self.config.target_points_max),
                            'mesh_quality_score': self.data_collector._calculate_mesh_quality_for_simulation(reconstructed_mesh),
                            'recommended_for_simulation': self.data_collector._is_suitable_for_lmgc90(reconstructed_mesh, len(reconstructed_mesh.vertices))
                        })
            else:
                final_mesh_info = {'vertex_count': len(final_points), 'face_count': 0}
            
            # Step 12: Save final data and comparisons
            logger.debug("Step 12: Saving final data and comparisons")
            processing_info = {
                'processing_time': time.time() - start_time,
                'device_used': self.device,
                'voxel_size': self.config.voxel_size,
                'epsilon': best_params.get('epsilon', 0) if 'best_params' in locals() else 0,
                'reconstruction_method': self.config.reconstruction_method,
                'meets_constraints': self._calculate_constraint_satisfaction(final_count),
                'ballast_count': max(0, self.config.max_ballast - final_count) if self.config.reconstruction_method not in ['convex_hull', 'controlled_convex_hull'] else 0,
                'target_min': self.config.target_points_min,
                'target_max': self.config.target_points_max
            }
            
            # Save final point cloud data
            final_csv = self.data_collector.save_final_pointcloud_csv(
                final_points, simplified_normals, filename, processing_info)
            logger.info(f"  💾 Final data saved: {final_csv}")
            
            # Save comparison data (will be added to consolidated file)
            comparison_csv = self.data_collector.save_comparison_csv(
                original_mesh_info, final_mesh_info, processing_info, filename)
            logger.info(f"  📊 Comparison data saved: {comparison_csv}")
            
            # Step 13: Export mesh files
            logger.debug("Step 13: Exporting mesh files")
            output_subdir = Path(output_dir) / filename
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Save simplified mesh
            mesh_vertices = 0
            if reconstructed_mesh is not None:
                if self.config.reconstruction_method == 'convex_hull':
                    mesh_path = output_subdir / f"{filename}_convex_hull.stl"
                elif self.config.reconstruction_method == 'controlled_convex_hull':
                    mesh_path = output_subdir / f"{filename}_lmgc90_ready.stl"
                else:
                    mesh_path = output_subdir / f"{filename}_simplified.stl"
                    
                reconstructed_mesh.export(str(mesh_path))
                mesh_vertices = len(reconstructed_mesh.vertices)
                logger.debug(f"  💾 Saved mesh: {mesh_path}")
            else:
                logger.warning("  ⚠️  Surface reconstruction failed, no mesh saved")
            
            # Save as DAT format
            dat_path = output_subdir / f"{filename}_points.dat"
            np.savetxt(dat_path, final_points, fmt='%.6f')
            logger.debug(f"  💾 Saved DAT file: {dat_path}")
            
            processing_time = time.time() - start_time
            
            # Calculate results
            meets_constraints = self._calculate_constraint_satisfaction(final_count)
            ballast_count = max(0, self.config.max_ballast - final_count) if self.config.reconstruction_method not in ['convex_hull', 'controlled_convex_hull'] else 0
            
            result = {
                'status': 'success',
                'filename': filename,
                'original_vertices': original_count,
                'simplified_vertices': final_count,
                'mesh_vertices': mesh_vertices,
                'reduction_ratio': (original_count - final_count) / original_count if original_count > 0 else 0,
                'processing_time': processing_time,
                'ballast_count': ballast_count,
                'meets_constraints': meets_constraints,
                'device_used': self.device,
                'reconstruction_method': self.config.reconstruction_method,
                'csv_files': {
                    'final': final_csv,
                    'comparison': comparison_csv
                }
            }
            
            # Add method-specific results
            if self.config.reconstruction_method in ['convex_hull', 'controlled_convex_hull'] and reconstructed_mesh is not None:
                convex_metrics = self.data_collector.calculate_convex_hull_metrics(
                    original_mesh, reconstructed_mesh)
                result.update({
                    'convexity_ratio': convex_metrics.get('convexity_ratio', 0),
                    'volume_preservation': convex_metrics.get('volume_preservation', 0),
                    'geometric_simplification': convex_metrics.get('geometric_simplification', 0)
                })
                
                if convex_hull_csv:
                    result['csv_files']['convex_hull_analysis'] = convex_hull_csv
                
                if lmgc90_csv:
                    result['csv_files']['lmgc90_analysis'] = lmgc90_csv
                    result.update({
                        'lmgc90_ready': final_mesh_info.get('lmgc90_ready', False),
                        'mesh_quality_score': final_mesh_info.get('mesh_quality_score', 0),
                        'recommended_for_simulation': final_mesh_info.get('recommended_for_simulation', False)
                    })
            
            if self.config.save_stages and 'original_csv' in locals():
                result['csv_files']['original'] = original_csv
            
            # Log completion with method-specific information
            status_icon = "✅" if meets_constraints else "⚠️"
            logger.info(f"{status_icon} Completed {filename}: {original_count} → {final_count} vertices "
                       f"({result['reduction_ratio']:.2%} reduction) in {processing_time:.2f}s")
            
            if self.config.reconstruction_method == 'convex_hull':
                logger.info(f"  🔺 Convex hull convexity: {result.get('convexity_ratio', 0):.3f}")
            elif self.config.reconstruction_method == 'controlled_convex_hull':
                logger.info(f"  🎯 LMGC90 ready: {'✅ YES' if result.get('lmgc90_ready', False) else '❌ NO'}")
                logger.info(f"  📊 Mesh quality score: {result.get('mesh_quality_score', 0):.3f}")
            else:
                logger.info(f"  📊 Ballast remaining: {ballast_count}, Constraints met: {'Yes' if meets_constraints else 'No'}")
            
            logger.info(f"  📄 CSV files saved: Final, Comparison{', LMGC90 Analysis' if lmgc90_csv else ''}{', Convex Hull Analysis' if convex_hull_csv else ''}{', Original, Stages' if self.config.save_stages else ''}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
            logger.debug(f"Full error details: {e}", exc_info=True)
            return {'status': 'failed', 'filename': filename, 'error': str(e)}
    
    def _calculate_constraint_satisfaction(self, final_count: int) -> bool:
        """Calculate if constraints are met based on reconstruction method"""
        if self.config.reconstruction_method == 'controlled_convex_hull':
            return self.config.target_points_min <= final_count <= self.config.target_points_max
        elif self.config.reconstruction_method == 'convex_hull':
            return True  # Convex hull always "meets constraints"
        else:
            return self.config.target_points_min <= final_count <= self.config.target_points_max
    
    def process_folder_with_csv(self, input_folder: str, output_folder: str, log_file_path: str = None) -> List[Dict]:
        """Enhanced process_folder with comprehensive CSV data collection, convex hull, and LMGC90 support"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collector
        self.data_collector = DataCollector(output_folder)
        
        # Find STL files
        stl_files = list(input_path.glob("*.stl")) + list(input_path.glob("*.STL"))
        
        if not stl_files:
            logger.error(f"No STL files found in {input_folder}")
            return []
        
        logger.info(f"Found {len(stl_files)} STL files to process with {self.device.upper()} acceleration")
        if self.config.reconstruction_method == 'controlled_convex_hull':
            logger.info(f"🎯 LMGC90 Mode: {self.config.target_points_min}-{self.config.target_points_max} vertices, convex-only output")
        elif self.config.reconstruction_method == 'convex_hull':
            logger.info(f"🔺 Convex Hull Mode: No vertex constraints, convex-only output")
        else:
            logger.info(f"Configuration: {self.config.target_points_min}-{self.config.target_points_max} vertices, max ballast {self.config.max_ballast}")
        
        logger.info(f"Reconstruction method: {self.config.reconstruction_method}")
        logger.info(f"📊 Enhanced CSV data collection enabled")
        
        results = []
        
        # Process files with progress logging
        for i, stl_file in enumerate(stl_files, 1):
            logger.info(f"Processing file {i}/{len(stl_files)}: {stl_file.name}")
            result = self.process_single_mesh_with_csv(str(stl_file), str(output_path))
            results.append(result)
            
            # Log intermediate progress
            if i % 5 == 0 or i == len(stl_files):
                successful_so_far = sum(1 for r in results if r['status'] == 'success')
                logger.info(f"Progress: {i}/{len(stl_files)} files processed, {successful_so_far} successful")
        
        # Generate summary reports (enhanced)
        self.generate_enhanced_summary_report(results, output_path, log_file_path)
        
        return results
    
    def generate_enhanced_summary_report(self, results: List[Dict], output_path: Path, log_file_path: str = None):
        """Generate enhanced summary report with comprehensive CSV data"""
        successful_results = [r for r in results if r['status'] == 'success']
        failed_results = [r for r in results if r['status'] == 'failed']
        
        # Save master summary CSV (includes consolidated comparison)
        master_csv = self.data_collector.save_master_summary_csv()
        logger.info(f"📊 Master summary CSV saved: {master_csv}")
        
        # Generate original summary report (enhanced)
        self.generate_summary_report(results, output_path, log_file_path)
        
        # Additional CSV-specific logging
        if successful_results:
            logger.info(f"📄 Individual CSV files created for {len(successful_results)} successful files:")
            logger.info(f"  - Final reduced point cloud data files")
            logger.info(f"  - Comparison analysis files")
            
            if self.config.reconstruction_method == 'convex_hull':
                logger.info(f"  - Convex hull analysis files")
            elif self.config.reconstruction_method == 'controlled_convex_hull':
                logger.info(f"  - LMGC90 compatibility analysis files")
                logger.info(f"  - LMGC90 import-ready vertex and face files")
                
            if self.config.save_stages:
                logger.info(f"  - Original point cloud data files")
                logger.info(f"  - Processing stage data files")
                logger.info(f"  - Statistical summary files")
            
        logger.info(f"📈 Consolidated analysis files:")
        logger.info(f"  - Consolidated comparison: {output_path / 'consolidated_comparison_all_models.csv'}")
        logger.info(f"  - Simplified comparison: {output_path / 'comparison_summary_simplified.csv'}")
        logger.info(f"  - Master processing summary: {master_csv}")
    
    def generate_summary_report(self, results: List[Dict], output_path: Path, log_file_path: str = None):
        """Generate comprehensive summary report"""
        successful_results = [r for r in results if r['status'] == 'success']
        failed_results = [r for r in results if r['status'] == 'failed']
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = output_path / "processing_summary.csv"
        results_df.to_csv(results_path, index=False)
        
        # Calculate statistics
        if successful_results:
            avg_reduction = np.mean([r['reduction_ratio'] for r in successful_results])
            avg_time = np.mean([r['processing_time'] for r in successful_results])
            total_time = sum(r['processing_time'] for r in successful_results)
            constraint_met = sum(1 for r in successful_results if r.get('meets_constraints', False))
            
            # Method-specific stats
            convex_hull_results = [r for r in successful_results if r.get('reconstruction_method') == 'convex_hull']
            controlled_results = [r for r in successful_results if r.get('reconstruction_method') == 'controlled_convex_hull']
            lmgc90_ready = sum(1 for r in controlled_results if r.get('lmgc90_ready', False))
            
            if convex_hull_results:
                avg_convexity = np.mean([r.get('convexity_ratio', 0) for r in convex_hull_results])
                avg_volume_preservation = np.mean([r.get('volume_preservation', 0) for r in convex_hull_results])
            else:
                avg_convexity = avg_volume_preservation = 0
                
            if controlled_results:
                avg_mesh_quality = np.mean([r.get('mesh_quality_score', 0) for r in controlled_results])
            else:
                avg_mesh_quality = 0
        else:
            avg_reduction = avg_time = total_time = constraint_met = 0
            avg_convexity = avg_volume_preservation = avg_mesh_quality = 0
            lmgc90_ready = 0
        
        # Save summary report
        summary_path = output_path / "summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write("=== GPU-Accelerated Point Cloud Reduction with LMGC90 Support ===\n\n")
            f.write(f"Processing device: {self.device.upper()}\n")
            f.write(f"Reconstruction method: {self.config.reconstruction_method}\n")
            f.write(f"Total files processed: {len(results)}\n")
            f.write(f"Successfully processed: {len(successful_results)}\n")
            f.write(f"Failed: {len(failed_results)}\n")
            
            if self.config.reconstruction_method not in ['convex_hull', 'controlled_convex_hull']:
                f.write(f"Files meeting constraints ({self.config.target_points_min}-{self.config.target_points_max} vertices): {constraint_met}\n")
            elif self.config.reconstruction_method == 'controlled_convex_hull':
                f.write(f"Files ready for LMGC90: {lmgc90_ready}/{len(controlled_results)}\n")
                
            f.write(f"Average reduction ratio: {avg_reduction:.2%}\n")
            f.write(f"Average processing time per file: {avg_time:.2f} seconds\n")
            f.write(f"Total processing time: {total_time:.2f} seconds\n\n")
            
            # Method-specific stats
            if convex_hull_results:
                f.write("=== Convex Hull Specific Statistics ===\n")
                f.write(f"Files processed with convex hull: {len(convex_hull_results)}\n")
                f.write(f"Average convexity ratio: {avg_convexity:.3f}\n")
                f.write(f"Average volume preservation: {avg_volume_preservation:.3f}\n\n")
                
            if controlled_results:
                f.write("=== LMGC90 Controlled Convex Hull Statistics ===\n")
                f.write(f"Files processed for LMGC90: {len(controlled_results)}\n")
                f.write(f"Files ready for LMGC90 simulation: {lmgc90_ready}\n")
                f.write(f"LMGC90 readiness rate: {lmgc90_ready/max(len(controlled_results), 1):.1%}\n")
                f.write(f"Average mesh quality score: {avg_mesh_quality:.3f}\n\n")
            
            if successful_results:
                f.write("Successful files:\n")
                for result in successful_results:
                    if result.get('reconstruction_method') == 'convex_hull':
                        convexity = result.get('convexity_ratio', 0)
                        f.write(f"  🔺 {result['filename']}: "
                               f"{result['original_vertices']} → {result['simplified_vertices']} vertices "
                               f"({result['reduction_ratio']:.1%} reduction, convexity: {convexity:.3f})\n")
                    elif result.get('reconstruction_method') == 'controlled_convex_hull':
                        lmgc90_status = "✓ LMGC90 Ready" if result.get('lmgc90_ready', False) else "✗ Not Ready"
                        quality = result.get('mesh_quality_score', 0)
                        f.write(f"  🎯 {result['filename']}: "
                               f"{result['original_vertices']} → {result['simplified_vertices']} vertices "
                               f"({result['reduction_ratio']:.1%} reduction, {lmgc90_status}, quality: {quality:.2f})\n")
                    else:
                        ballast = result.get('ballast_count', 0)
                        constraint_status = "✓" if result.get('meets_constraints', False) else "✗"
                        f.write(f"  {constraint_status} {result['filename']}: "
                               f"{result['original_vertices']} → {result['simplified_vertices']} vertices "
                               f"({result['reduction_ratio']:.1%} reduction, {ballast} ballast remaining)\n")
            
            if failed_results:
                f.write("\nFailed files:\n")
                for result in failed_results:
                    f.write(f"  ✗ {result['filename']}: {result.get('error', 'Unknown error')}\n")
        
        logger.info(f"Summary report saved to {summary_path}")
        logger.info(f"Processing complete: {len(successful_results)}/{len(results)} files successful")

# ===== ENHANCED MAIN FUNCTION WITH DAT EXPORT =====

def create_test_data():
    """Create sample test data for testing"""
    print("🧪 Creating test data...")
    
    try:
        # Create test directory
        test_dir = Path("test_input")
        test_dir.mkdir(exist_ok=True)
        
        # Create a simple test sphere
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        test_file = test_dir / "test_sphere.stl"
        mesh.export(str(test_file))
        
        print(f"✅ Created test file: {test_file}")
        print(f"   Test mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        return str(test_file)
        
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return None

def process_stl_files_convex_hull_only(input_directory: str, output_directory: str):
    """
    Standalone function to process STL files for convex hull only (based on original user code)
    This is a simplified version that only does convex hull conversion without the full pipeline
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Process each STL file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.stl') or filename.endswith('.STL'):
            file_path = os.path.join(input_directory, filename)
            
            try:
                # Load the mesh
                mesh = trimesh.load_mesh(file_path)
                
                # Calculate convexity ratio
                convexity = mesh.volume / mesh.convex_hull.volume if mesh.convex_hull.volume > 0 else 0
                
                # Get convex hull of the mesh
                convex_hull = mesh.convex_hull
                convex_faces = convex_hull.faces
                convex_vertices = convex_hull.vertices
                
                # Print convex hull properties for each file
                print(f"Processing file: {filename}")
                print(f"--> Original volume: {mesh.volume:.6f}")
                print(f"--> Convex hull volume: {convex_hull.volume:.6f}")
                print(f"--> Convexity ratio: {convexity:.6f}")
                print(f"--> Original vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
                print(f"--> Convex hull vertices: {len(convex_vertices)}, faces: {len(convex_faces)}")
                
                # Convert convex hull vertices and faces directly into numpy arrays
                vertices = np.array(convex_vertices)
                faces = np.array(convex_faces)
                
                # Create a new mesh using the convex hull vertices and faces
                convex_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                # Define the output path for the convex hull mesh
                output_path = os.path.join(output_directory, f"convex_hull_{filename}")
                
                # Export the convex hull as an STL file
                convex_mesh.export(output_path)
                print(f"Convex hull mesh exported as '{output_path}'\n")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

def main():
    """Enhanced main function with comprehensive CSV data collection, consolidated comparison, convex hull, LMGC90 support, and DAT export"""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated ML Point Cloud Reduction System with CSV Data Collection, Consolidated Comparison, Convex Hull Support, LMGC90 Compatibility, and DAT Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LMGC90 controlled convex hull with DAT export (350-500 vertices, convex only)
  python %(prog)s -i input_models -o output_models --method controlled_convex_hull --min-vertices 350 --max-vertices 500 --export-dat -v
  
  # Individual DAT files for each object
  python %(prog)s -i input_models -o output_models --method controlled_convex_hull --min-vertices 350 --max-vertices 500 --export-individual-dats -v
  
  # Custom DAT filename
  python %(prog)s -i ballast_models -o lmgc90_ballast --method controlled_convex_hull --export-dat --dat-filename ballast_export.DAT -v
  
  # Basic convex hull reconstruction (no vertex constraints)
  python %(prog)s -i input_models -o output_models --method convex_hull
  
  # Standard reconstruction with vertex constraints
  python %(prog)s -i input_models -o output_models --method poisson --min-vertices 500 --max-vertices 800 -v
  
  # Export existing processed files to DAT
  python %(prog)s --export-dat-only --dat-filename existing_export.DAT -o existing_output_folder
        """
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', 
                       help='Input folder containing STL files')
    parser.add_argument('-o', '--output', default='simplified_models',
                       help='Output folder for simplified models (default: simplified_models)')
    
    # Vertex constraints
    parser.add_argument('--min-vertices', type=int, default=50,
                       help='Minimum vertices in output (default: 50, ignored for convex_hull)')
    parser.add_argument('--max-vertices', type=int, default=300,
                       help='Maximum vertices in output (default: 300, ignored for convex_hull)')
    parser.add_argument('--max-ballast', type=int, default=150,
                       help='Maximum ballast points (default: 150, ignored for convex hull methods)')
    
    # Processing parameters
    parser.add_argument('--voxel-size', type=float, default=0.02,
                       help='Voxel size for downsampling (default: 0.02)')
    parser.add_argument('--method', 
                       choices=['poisson', 'ball_pivoting', 'alpha_shapes', 'convex_hull', 'controlled_convex_hull'],
                       default='poisson', 
                       help='Surface reconstruction method (default: poisson). Use controlled_convex_hull for LMGC90.')
    
    # GPU settings
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration (use CPU only)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='GPU batch size (default: 10000)')
    
    # CSV and analysis options
    parser.add_argument('--save-stages', action='store_true',
                       help='Save CSV data for all processing stages')
    parser.add_argument('--detailed-analysis', action='store_true',
                       help='Include detailed geometric analysis in CSV files')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV data collection (original mode)')
    
    # ===== ENHANCED DAT EXPORT OPTIONS =====
    parser.add_argument('--export-dat', action='store_true',
                       help='Export processed meshes to LMGC90 DAT format (single file)')
    parser.add_argument('--export-individual-dats', action='store_true',
                       help='Export each processed mesh to individual LMGC90 DAT files')
    parser.add_argument('--dat-filename', default='lmgc90_export.DAT',
                       help='Output DAT filename (default: lmgc90_export.DAT)')
    parser.add_argument('--dat-prefix', default='lmgc90_object',
                       help='Prefix for individual DAT files (default: lmgc90_object)')
    parser.add_argument('--export-dat-only', action='store_true',
                       help='Only export DAT from existing processed files (no processing)')
    
    # Test mode
    parser.add_argument('--create-test', action='store_true',
                       help='Create test data and run a test')
    
    # Convex hull standalone mode
    parser.add_argument('--convex-hull-only', action='store_true',
                       help='Run standalone convex hull processing only (simple mode)')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Reduce logging output')
    
    args = parser.parse_args()
    
    # ===== DAT EXPORT ONLY MODE =====
    if args.export_dat_only:
        if not args.output or not os.path.exists(args.output):
            print("Error: Valid output folder with processed files is required for DAT export only mode.")
            return 1
        
        print(f"🎯 DAT Export Only Mode")
        print(f"Source: {args.output}")
        
        # Export single DAT file
        if args.export_dat or not args.export_individual_dats:
            dat_path, exported_count = export_to_lmgc90_dat(args.output, args.dat_filename)
            if exported_count > 0:
                print(f"✅ Single DAT export completed: {exported_count} objects")
            else:
                print(f"❌ No objects were suitable for LMGC90 export")
        
        # Export individual DAT files
        if args.export_individual_dats:
            exported_files, exported_count = export_individual_lmgc90_dats(args.output, args.dat_prefix)
            if exported_count > 0:
                print(f"✅ Individual DAT export completed: {exported_count} files")
            else:
                print(f"❌ No objects were suitable for individual DAT export")
        
        return 0
    
    # Standalone convex hull mode
    if args.convex_hull_only:
        if not args.input:
            print("Error: Input folder is required for convex hull only mode. Use -i or --input to specify.")
            return 1
        
        output_dir = args.output if args.output != 'simplified_models' else f"{args.input}_convex_hull"
        print(f"🔺 Running standalone convex hull processing...")
        print(f"Input: {args.input}")
        print(f"Output: {output_dir}")
        
        try:
            process_stl_files_convex_hull_only(args.input, output_dir)
            print(f"✅ Standalone convex hull processing completed!")
            print(f"Results saved to: {output_dir}")
            return 0
        except Exception as e:
            print(f"❌ Error in standalone convex hull processing: {e}")
            return 1
    
    # Setup logging with automatic file generation
    global logger
    logger, log_file_path = setup_logging(args.output, args.verbose, args.quiet)
    
    # Test mode
    if args.create_test:
        print("🧪 Test Mode - Creating test data and running test...")
        test_file = create_test_data()
        if test_file:
            args.input = "test_input"
            args.output = "test_output"
        else:
            print("❌ Failed to create test data")
            return 1
    
    # Validate input
    if not args.input:
        print("Error: Input folder is required. Use -i or --input to specify.")
        print("Or use --create-test to create test data and run a test.")
        print("Or use --convex-hull-only for standalone convex hull processing.")
        print("Or use --export-dat-only to export existing processed files.")
        return 1
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist.")
        return 1
    
    # Check for STL files
    stl_files = list(Path(args.input).glob("*.stl")) + list(Path(args.input).glob("*.STL"))
    if not stl_files:
        print(f"Error: No STL files found in '{args.input}'")
        return 1
    
    # Create configuration
    config = ReductionConfig(
        target_points_min=args.min_vertices,
        target_points_max=args.max_vertices,
        max_ballast=args.max_ballast,
        voxel_size=args.voxel_size,
        reconstruction_method=args.method,
        use_gpu=not args.no_gpu,
        batch_size=args.batch_size,
        save_stages=args.save_stages,
        detailed_analysis=args.detailed_analysis
    )
    
    # Print configuration
    print("=== GPU-Accelerated ML Point Cloud Reduction with LMGC90 Support and DAT Export ===")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"STL files found: {len(stl_files)}")
    print(f"Reconstruction method: {config.reconstruction_method}")
    
    if config.reconstruction_method == 'convex_hull':
        print(f"🔺 Convex Hull Mode: Vertex constraints ignored")
        print(f"   - Will compute convex hull of reduced point cloud")
        print(f"   - Detailed convex hull analysis will be saved to CSV")
    elif config.reconstruction_method == 'controlled_convex_hull':
        print(f"🎯 LMGC90 Controlled Convex Hull Mode:")
        print(f"   - Target vertex range: {config.target_points_min}-{config.target_points_max}")
        print(f"   - Guaranteed convex output for LMGC90 simulation")
        print(f"   - LMGC90 compatibility analysis included")
        print(f"   - Import-ready CSV files generated")
        if args.export_dat or args.export_individual_dats:
            print(f"   - 🎯 DAT export enabled for LMGC90 import")
    else:
        print(f"Target vertex range: {config.target_points_min}-{config.target_points_max}")
        print(f"Maximum ballast: {config.max_ballast}")
    
    print(f"GPU acceleration: {'Enabled' if config.use_gpu else 'Disabled'}")
    print(f"CSV data collection: {'Disabled' if args.no_csv else 'Enabled'}")
    if not args.no_csv:
        print(f"Save processing stages: {'Yes' if config.save_stages else 'No'}")
        print(f"Detailed analysis: {'Yes' if config.detailed_analysis else 'No'}")
        print(f"📊 Consolidated comparison file: ENABLED")
    
    # DAT export configuration
    if args.export_dat:
        print(f"📤 Single DAT export: {args.dat_filename}")
    if args.export_individual_dats:
        print(f"📤 Individual DAT exports: {args.dat_prefix}_*.DAT")
    
    # Create reducer and process files
    reducer = GPUAcceleratedReducer(config)
    
    start_time = time.time()
    
    # Use enhanced processing with CSV collection
    results = reducer.process_folder_with_csv(args.input, args.output, log_file_path)
    
    total_time = time.time() - start_time
    
    # ===== ENHANCED DAT EXPORT AFTER PROCESSING =====
    successful = sum(1 for r in results if r['status'] == 'success')
    
    # Export to DAT if requested and we have successful results
    if (args.export_dat or args.export_individual_dats) and successful > 0:
        print(f"\n📤 Post-processing DAT export...")
        
        # Single DAT file export
        if args.export_dat:
            dat_path, exported_count = export_to_lmgc90_dat(args.output, args.dat_filename)
            if exported_count > 0:
                print(f"✅ Single DAT export completed: {exported_count} objects exported")
                print(f"📄 File: {dat_path}")
            else:
                print(f"❌ No objects were suitable for LMGC90 DAT export")
        
        # Individual DAT files export
        if args.export_individual_dats:
            exported_files, exported_count = export_individual_lmgc90_dats(args.output, args.dat_prefix)
            if exported_count > 0:
                print(f"✅ Individual DAT export completed: {exported_count} files exported")
                print(f"📂 Files created in individual subdirectories")
            else:
                print(f"❌ No objects were suitable for individual DAT export")
    
    # Final summary logging
    logger.info("=== FINAL PROCESSING SUMMARY ===")
    logger.info(f"Total processing time: {total_time:.1f} seconds")
    logger.info(f"Log file saved to: {log_file_path}")
    logger.info("=== SESSION COMPLETE ===")
    
    # Print console summary
    constraint_met = sum(1 for r in results if r.get('meets_constraints', False))
    failed = len(results) - successful
    
    method_display = config.reconstruction_method.replace('_', ' ').title()
    print(f"\n=== Enhanced Processing Complete with {method_display} Reconstruction ===")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Files processed: {successful}/{len(results)} successful")
    print(f"Files failed: {failed}")
    
    if config.reconstruction_method == 'convex_hull':
        convex_hull_results = [r for r in results if r['status'] == 'success' and r.get('reconstruction_method') == 'convex_hull']
        if convex_hull_results:
            avg_convexity = sum(r.get('convexity_ratio', 0) for r in convex_hull_results) / len(convex_hull_results)
            print(f"Average convexity ratio: {avg_convexity:.3f}")
    elif config.reconstruction_method == 'controlled_convex_hull':
        controlled_results = [r for r in results if r['status'] == 'success' and r.get('reconstruction_method') == 'controlled_convex_hull']
        if controlled_results:
            lmgc90_ready = sum(1 for r in controlled_results if r.get('lmgc90_ready', False))
            avg_quality = sum(r.get('mesh_quality_score', 0) for r in controlled_results) / len(controlled_results)
            print(f"LMGC90 ready files: {lmgc90_ready}/{len(controlled_results)} ({lmgc90_ready/max(len(controlled_results), 1):.1%})")
            print(f"Average mesh quality score: {avg_quality:.3f}")
    else:
        print(f"Meeting constraints: {constraint_met}/{successful}")
    
    print(f"Results saved to: {args.output}")
    print(f"📋 Real-time log saved to: {log_file_path}")
    
    if not args.no_csv:
        print(f"📊 Comprehensive CSV data files generated")
        
        if successful > 0:
            print(f"\n📄 Key CSV Files Generated:")
            print(f"  🎯 CONSOLIDATED COMPARISON: {args.output}/consolidated_comparison_all_models.csv")
            print(f"  📋 SIMPLIFIED COMPARISON: {args.output}/comparison_summary_simplified.csv")
            print(f"  📊 Master summary: {args.output}/master_processing_summary.csv")
            print(f"  📈 Size analysis: {args.output}/analysis_by_model_size.csv")
            print(f"  🔍 Complexity analysis: {args.output}/analysis_by_complexity.csv")
            print(f"  🖥️ Device performance: {args.output}/device_usage.csv")
            print(f"  📊 Individual file analysis: {successful} sets of detailed CSV files")
            
            if config.reconstruction_method == 'convex_hull':
                print(f"  🔺 Convex hull analysis files: detailed face and metrics data")
            elif config.reconstruction_method == 'controlled_convex_hull':
                print(f"  🎯 LMGC90 analysis files: compatibility and quality metrics")
                print(f"  📥 LMGC90 import files: vertices and faces CSV for direct import")
                if args.export_dat or args.export_individual_dats:
                    print(f"  📤 LMGC90 DAT files: Ready for direct LMGC90 import")
    
    if successful > 0:
        successful_results = [r for r in results if r['status'] == 'success']
        avg_reduction = sum(r.get('reduction_ratio', 0) for r in successful_results) / successful
        avg_time = sum(r.get('processing_time', 0) for r in successful_results) / successful
        print(f"\nAverage reduction: {avg_reduction:.1%}")
        print(f"Average time per file: {avg_time:.2f} seconds")
        
        # Show device usage
        devices_used = set(r.get('device_used', 'unknown') for r in successful_results)
        print(f"Devices used: {', '.join(devices_used).upper()}")
    
    if args.create_test:
        method_name = config.reconstruction_method.replace('_', ' ').title()
        print(f"\n🎉 Test completed successfully with {method_name} reconstruction!")
        print(f"Check the consolidated comparison file for complete analysis:")
        print(f"  📊 {args.output}/consolidated_comparison_all_models.csv")
        print(f"Check the log file for detailed processing information:")
        print(f"  📋 {log_file_path}")
        
        if config.reconstruction_method == 'controlled_convex_hull':
            print(f"\n🎯 For LMGC90 usage:")
            print(f"  - STL files: Ready for import into LMGC90")
            print(f"  - CSV files: Vertex and face data for manual import")
            print(f"  - Analysis: Compatibility and quality metrics")
            if args.export_dat or args.export_individual_dats:
                print(f"  - DAT files: Direct LMGC90 import format")
        
        print(f"\nYou can now use the system with your own STL files:")
        print(f"  mkdir input_models")
        print(f"  # Copy your STL files to input_models/")
        if config.reconstruction_method == 'controlled_convex_hull':
            print(f"  python {sys.argv[0]} -i input_models -o output_models --method controlled_convex_hull --min-vertices 350 --max-vertices 500 --export-dat -v")
        else:
            print(f"  python {sys.argv[0]} -i input_models -o output_models --method {config.reconstruction_method} --min-vertices {config.target_points_min} --max-vertices {config.target_points_max} -v")
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
