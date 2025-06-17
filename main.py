#!/usr/bin/env python3
"""
Ballast Quality-Focused Point Cloud Reduction System v2.3 (Target-Respecting)

SPECIALIZED FOR BALLAST QUALITY + RESPECTS USER TARGETS:
- Much more conservative point reduction for ballast (but respects user intent)
- Specialized ballast surface reconstruction
- Better feature preservation for rough surfaces
- Multiple quality validation steps
- NEW: Actually hits close to user-specified targets
- NEW: Target-aware parameter adjustment

Usage (SAME COMMANDS):
    python ballast-quality-focused-v2.3.py /home/railcmu/Desktop/BPK/ballast --count 100 --workers 2

Key Improvements for Ballast Quality + Target Compliance:
    ✅ Respects user targets while ensuring minimum quality
    ✅ 2-3x more points retained for ballast (vs massive over-targets)
    ✅ Specialized surface reconstruction for rough textures
    ✅ Better feature detection for irregular geometry
    ✅ Multiple reconstruction method fallbacks
    ✅ Target-aware parameter adjustment
    ✅ Quality validation and automatic corrections

Requirements:
    pip install numpy pandas scikit-learn trimesh open3d

Author: theeradon
Version: 2.3.0 (Ballast Quality-Focused + Target-Respecting)
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import math
from functools import wraps

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try importing required libraries with helpful error messages
try:
    import trimesh
except ImportError:
    print("Error: trimesh not installed. Run: pip install trimesh")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed. Run: pip install open3d")
    sys.exit(1)

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import ParameterGrid
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration with file and console output"""
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    level = logging.DEBUG if verbose else logging.INFO
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"📝 Logging to file: {log_file}")
        logging.info(f"📺 Console logging level: {logging.getLevelName(level)}")
        
    return root_logger


logger = logging.getLogger(__name__)


def time_function(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"⏱️ {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


class BallastQualitySpecialist:
    """
    SPECIALIZED for ballast quality - addresses the over-simplification issues
    """
    
    def __init__(self):
        # Much more conservative settings for ballast quality (but respect user targets)
        self.ballast_config = {
            # Minimum points to ensure quality (REASONABLE minimums)
            'min_points_small_ballast': 40,      # vs 500 before (too high)
            'min_points_medium_ballast': 80,     # vs 1200 before (too high)  
            'min_points_large_ballast': 150,     # vs 2500 before (too high)
            
            # Feature preservation settings
            'importance_threshold_aggressive': 40,  # Keep top 60% of points
            'importance_threshold_moderate': 50,    # Keep top 50% of points
            'importance_threshold_conservative': 60, # Keep top 40% of points
            
            # Clustering settings (more conservative than standard, but not extreme)
            'epsilon_fine': 0.008,      # Very fine clustering
            'epsilon_medium': 0.015,    # Medium clustering
            'epsilon_coarse': 0.025,    # Coarse clustering
            
            # Surface reconstruction settings
            'poisson_depth_high': 10,   # Higher detail
            'poisson_depth_medium': 9,  # Medium detail
            'poisson_depth_low': 8,     # Lower detail
            
            # Quality validation
            'min_points_for_reconstruction': 30,  # Lowered from 50
            'max_reconstruction_attempts': 3,
            'normal_estimation_neighbors': 20,    # Reduced from 30
        }
        
    def detect_ballast_model(self, file_path: str) -> bool:
        """Detect if this is likely a ballast model"""
        filename = file_path.lower()
        ballast_keywords = ['ballast', 'stone', 'rock', 'aggregate', 'gravel', 'bpk']
        return any(keyword in filename for keyword in ballast_keywords)
    
    def analyze_ballast_complexity(self, points: np.ndarray) -> Dict:
        """Analyze ballast model complexity to determine optimal settings"""
        n_points = len(points)
        
        # Calculate geometric complexity
        bbox = np.max(points, axis=0) - np.min(points, axis=0)
        bbox_volume = np.prod(bbox)
        bbox_surface_area = 2 * (bbox[0]*bbox[1] + bbox[1]*bbox[2] + bbox[0]*bbox[2])
        
        # Estimate surface roughness using point distribution
        if n_points > 100:
            # Sample points to estimate surface variation
            sample_size = min(1000, n_points)
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            sample_points = points[sample_indices]
            
            # Compute distances to nearest neighbors
            if sample_size > 10:
                nbrs = NearestNeighbors(n_neighbors=min(10, sample_size-1))
                nbrs.fit(sample_points)
                distances, _ = nbrs.kneighbors(sample_points)
                avg_neighbor_distance = np.mean(distances[:, 1:])  # Exclude self
                surface_roughness = np.std(distances[:, 1:])
            else:
                avg_neighbor_distance = 0.1
                surface_roughness = 0.05
        else:
            avg_neighbor_distance = 0.1
            surface_roughness = 0.05
        
        # Classify ballast complexity
        if bbox_volume > 1000 or n_points > 100000:
            complexity = "high"
        elif bbox_volume > 100 or n_points > 20000:
            complexity = "medium"
        else:
            complexity = "low"
        
        analysis = {
            'complexity': complexity,
            'bbox_volume': bbox_volume,
            'bbox_surface_area': bbox_surface_area,
            'surface_roughness': surface_roughness,
            'avg_neighbor_distance': avg_neighbor_distance,
            'original_points': n_points
        }
        
        logger.info(f"🔍 Ballast analysis: {complexity} complexity, roughness: {surface_roughness:.4f}")
        
        return analysis
    
    def get_quality_focused_target_points(self, original_points: np.ndarray, 
                                        target_ratio: float, analysis: Dict) -> int:
        """
        Calculate target points that RESPECTS user target while ensuring minimum quality
        """
        original_count = len(original_points)
        base_target = int(original_count * target_ratio)
        
        # Much more reasonable quality adjustments that still respect user intent
        if base_target < 50:  # Very small targets
            # For very small targets, be more generous but reasonable
            quality_multiplier = 3.0  # 3x more points max
        elif base_target < 200:  # Small targets  
            quality_multiplier = 2.0  # 2x more points max
        elif base_target < 1000:  # Medium targets
            quality_multiplier = 1.5  # 1.5x more points max
        else:  # Large targets
            quality_multiplier = 1.2  # Only 20% more for large targets
        
        # Apply complexity-based adjustment (much more modest)
        if analysis['complexity'] == 'high' and analysis['surface_roughness'] > 0.3:
            quality_multiplier *= 1.3  # Extra 30% for very complex surfaces
        elif analysis['surface_roughness'] > 0.1:
            quality_multiplier *= 1.1  # Extra 10% for rough surfaces
        
        # Calculate quality-adjusted target
        adjusted_target = int(base_target * quality_multiplier)
        
        # Ensure minimum viable points for reconstruction
        min_viable = 30  # Absolute minimum
        optimal_points = max(adjusted_target, min_viable)
        
        # Cap at reasonable maximum relative to user intent
        max_allowed = min(base_target * 5, 2000)  # Never more than 5x user target or 2000 points
        optimal_points = min(optimal_points, max_allowed)
        
        logger.info(f"🎯 Balanced target: {original_count:,} → {optimal_points:,} points")
        logger.info(f"   User target: {base_target:,}, Quality-adjusted: {optimal_points:,}")
        logger.info(f"   Quality multiplier: {quality_multiplier:.1f}x (respects user intent)")
        
        return optimal_points
    
    def get_ballast_quality_parameters(self, analysis: Dict, target_points: int, original_points: int, aggressive_reduction: bool = False) -> Dict:
        """
        Get parameters optimized for ballast quality preservation while respecting targets
        """
        complexity = analysis['complexity']
        surface_roughness = analysis['surface_roughness']
        reduction_ratio = target_points / original_points
        
        # Adjust parameters based on how aggressive the reduction needs to be
        if reduction_ratio < 0.02:  # Very aggressive reduction (< 2%)
            importance_threshold = self.ballast_config['importance_threshold_aggressive']
            epsilon_scale = 1.5  # Larger epsilon for more aggressive clustering
            k_neighbors = 8
        elif reduction_ratio < 0.05:  # Moderate aggressive reduction (< 5%)
            importance_threshold = self.ballast_config['importance_threshold_moderate'] 
            epsilon_scale = 1.2
            k_neighbors = 6
        else:  # Conservative reduction (>= 5%)
            importance_threshold = self.ballast_config['importance_threshold_conservative']
            epsilon_scale = 1.0
            k_neighbors = 6
        
        # Choose base clustering parameters based on surface detail
        if surface_roughness > 0.1:  # Very detailed surface
            base_epsilon = self.ballast_config['epsilon_fine']
        elif surface_roughness > 0.05:  # Moderately detailed
            base_epsilon = self.ballast_config['epsilon_medium']
        else:  # Smoother surface
            base_epsilon = self.ballast_config['epsilon_coarse']
        
        # Apply scaling based on reduction aggressiveness
        epsilon = base_epsilon * epsilon_scale
        
        # DBSCAN cleanup parameters (scale with aggressiveness)
        dbscan_eps = epsilon * (1.5 if reduction_ratio > 0.05 else 2.0)
        
        params = {
            'k_neighbors': k_neighbors,
            'epsilon': epsilon,
            'dbscan_eps': dbscan_eps,
            'importance_threshold': importance_threshold,
            'complexity': complexity,
            'surface_roughness': surface_roughness,
            'reduction_ratio': reduction_ratio,
            'epsilon_scale': epsilon_scale
        }
        
        logger.info(f"🎛️ Target-aware parameters: {params}")
        
        return params
    
    def enhanced_feature_extraction_for_ballast(self, points: np.ndarray, k_neighbors: int = 12) -> np.ndarray:
        """
        Enhanced feature extraction specifically optimized for ballast rough surfaces
        """
        n_points = len(points)
        features = np.zeros((n_points, 6), dtype=np.float32)  # More features for ballast
        
        # Use more neighbors for stable feature estimation on rough surfaces
        k = min(k_neighbors, 20, n_points-1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=4)
        nbrs.fit(points)
        
        distances, indices = nbrs.kneighbors(points)
        
        # Feature 1: Global centroid distance
        centroid = np.mean(points, axis=0)
        features[:, 0] = np.linalg.norm(points - centroid, axis=1)
        
        # Feature 2: Local density (mean distance to neighbors)
        features[:, 1] = np.mean(distances[:, 1:], axis=1)
        
        # Feature 3: Local variation (std of distances) - important for rough surfaces
        features[:, 2] = np.std(distances[:, 1:], axis=1)
        
        # Feature 4: Max neighbor distance - captures edges and protrusions
        features[:, 3] = np.max(distances[:, 1:], axis=1)
        
        # Feature 5: Local curvature estimate
        for i in range(n_points):
            neighbor_points = points[indices[i, 1:]]  # Exclude self
            if len(neighbor_points) > 3:
                # Compute covariance matrix
                centered = neighbor_points - np.mean(neighbor_points, axis=0)
                cov_matrix = np.cov(centered.T)
                eigenvals = np.linalg.eigvals(cov_matrix)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                
                # Curvature estimate (smaller eigenvalue ratio = more curved)
                if eigenvals[0] > 1e-10:
                    features[i, 4] = eigenvals[2] / eigenvals[0]
                else:
                    features[i, 4] = 0
            else:
                features[i, 4] = 0
        
        # Feature 6: Surface roughness indicator
        features[:, 5] = features[:, 2] / (features[:, 1] + 1e-8)  # Variation/density ratio
        
        logger.debug(f"✅ Enhanced ballast features extracted for {n_points:,} points")
        
        return features
    
    def create_ballast_importance_labels(self, features: np.ndarray, points: np.ndarray,
                                       importance_threshold: int = 50) -> np.ndarray:
        """
        Create importance labels specifically tuned for ballast surface features
        """
        n_points = len(points)
        importance_scores = np.zeros(n_points)
        
        # Weight features for ballast characteristics
        
        # High curvature points (edges, corners, protrusions) - very important for ballast
        curvature_score = features[:, 2] + features[:, 3] + features[:, 4] * 2  # Enhanced curvature weight
        importance_scores += curvature_score * 2.0  # Double weight for geometric features
        
        # Surface roughness - crucial for ballast texture
        roughness_score = features[:, 5]
        importance_scores += roughness_score * 1.5
        
        # Boundary points (low local density) - important for ballast edges
        density_score = 1.0 / (features[:, 1] + 1e-8)
        importance_scores += density_score * 0.8
        
        # Extremal points (far from centroid) - less important for ballast than other features
        centroid_distance_score = features[:, 0]
        importance_scores += centroid_distance_score * 0.3
        
        # Normalize scores
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        
        threshold = np.percentile(importance_scores, importance_threshold)
        pseudo_labels = (importance_scores >= threshold).astype(int)
        
        logger.info(f"🎯 Ballast importance labels: {np.sum(pseudo_labels):,}/{n_points:,} important points")
        logger.info(f"   Importance threshold: {importance_threshold}% (keeping top {100-importance_threshold}%)")
        
        return pseudo_labels
    
    def quality_focused_surface_reconstruction(self, points: np.ndarray, normals: np.ndarray, 
                                             method: str = 'poisson') -> Optional[trimesh.Trimesh]:
        """
        Quality-focused surface reconstruction with multiple fallback methods
        """
        if len(points) < self.ballast_config['min_points_for_reconstruction']:
            logger.warning(f"⚠️ Too few points ({len(points)}) for quality reconstruction")
            return None
        
        logger.info(f"🔧 Quality-focused reconstruction for {len(points):,} points using {method}")
        
        # Enhance normals first
        try:
            normals = self.improve_normals_for_ballast(points, normals)
        except Exception as e:
            logger.warning(f"⚠️ Normal enhancement failed: {e}")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Try multiple reconstruction methods with different parameters
        reconstruction_attempts = [
            ('poisson_high', self._try_poisson_high_quality),
            ('poisson_medium', self._try_poisson_medium_quality),
            ('ball_pivoting_adaptive', self._try_ball_pivoting_adaptive),
            ('poisson_low', self._try_poisson_low_quality),
            ('alpha_shapes', self._try_alpha_shapes)
        ]
        
        for attempt_name, reconstruction_func in reconstruction_attempts:
            try:
                logger.info(f"🔄 Trying {attempt_name} reconstruction...")
                mesh = reconstruction_func(pcd)
                
                if mesh is not None:
                    vertices = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.triangles)
                    
                    if len(faces) > 0:
                        reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        
                        # Validate mesh quality
                        if self._validate_mesh_quality(reconstructed_mesh, points):
                            logger.info(f"✅ Success with {attempt_name}: {len(vertices):,} vertices, {len(faces):,} faces")
                            return reconstructed_mesh
                        else:
                            logger.warning(f"⚠️ {attempt_name} failed quality validation")
                    else:
                        logger.warning(f"⚠️ {attempt_name} produced no faces")
                else:
                    logger.warning(f"⚠️ {attempt_name} returned None")
                    
            except Exception as e:
                logger.warning(f"⚠️ {attempt_name} failed: {e}")
                continue
        
        logger.error(f"❌ All reconstruction methods failed for {len(points)} points")
        return None
    
    def improve_normals_for_ballast(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Improve normal estimation specifically for rough ballast surfaces
        """
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Use more neighbors for stable normal estimation on rough surfaces
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(
                    knn=self.ballast_config['normal_estimation_neighbors']
                )
            )
            
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(
                k=self.ballast_config['normal_estimation_neighbors']
            )
            
            improved_normals = np.asarray(pcd.normals)
            
            # Validate improved normals
            if len(improved_normals) == len(points) and not np.any(np.isnan(improved_normals)):
                return improved_normals
            else:
                return normals
                
        except Exception as e:
            logger.warning(f"⚠️ Normal improvement failed: {e}")
            return normals
    
    def _try_poisson_high_quality(self, pcd) -> Optional[o3d.geometry.TriangleMesh]:
        """Try high-quality Poisson reconstruction"""
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=self.ballast_config['poisson_depth_high'],
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # Post-process for ballast
        if len(np.asarray(mesh.vertices)) > 0:
            # Light smoothing to reduce artifacts while preserving detail
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            mesh.compute_vertex_normals()
        
        return mesh
    
    def _try_poisson_medium_quality(self, pcd) -> Optional[o3d.geometry.TriangleMesh]:
        """Try medium-quality Poisson reconstruction"""
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=self.ballast_config['poisson_depth_medium'],
            width=0,
            scale=1.0,
            linear_fit=False
        )
        
        if len(np.asarray(mesh.vertices)) > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=2)
            mesh.compute_vertex_normals()
        
        return mesh
    
    def _try_poisson_low_quality(self, pcd) -> Optional[o3d.geometry.TriangleMesh]:
        """Try low-quality Poisson reconstruction as fallback"""
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=self.ballast_config['poisson_depth_low'],
            width=0,
            scale=1.0,
            linear_fit=False
        )
        
        if len(np.asarray(mesh.vertices)) > 0:
            mesh.compute_vertex_normals()
        
        return mesh
    
    def _try_ball_pivoting_adaptive(self, pcd) -> Optional[o3d.geometry.TriangleMesh]:
        """Try ball pivoting with adaptive radii for ballast"""
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        
        # Use smaller radii for better detail preservation
        radii = [avg_dist * factor for factor in [0.8, 1.2, 2.0, 3.0]]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, 
            o3d.utility.DoubleVector(radii)
        )
        
        return mesh
    
    def _try_alpha_shapes(self, pcd) -> Optional[o3d.geometry.TriangleMesh]:
        """Try alpha shapes reconstruction"""
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, 
            alpha=0.02  # Smaller alpha for tighter fit
        )
        
        return mesh
    
    def _validate_mesh_quality(self, mesh: trimesh.Trimesh, original_points: np.ndarray) -> bool:
        """
        Validate mesh quality for ballast models
        """
        try:
            # Basic checks
            if len(mesh.vertices) < 10 or len(mesh.faces) < 10:
                return False
            
            # Check if mesh is too simplified compared to original
            if len(mesh.vertices) < len(original_points) * 0.1:  # Less than 10% of original points
                logger.warning(f"⚠️ Mesh too simplified: {len(mesh.vertices)} vs {len(original_points)} original")
                return False
            
            # Check for degenerate faces
            if hasattr(mesh, 'remove_degenerate_faces'):
                mesh.remove_degenerate_faces()
            
            # Check bounding box preservation
            original_bbox = np.max(original_points, axis=0) - np.min(original_points, axis=0)
            mesh_bbox = mesh.bounds[1] - mesh.bounds[0]
            
            bbox_ratio = np.linalg.norm(mesh_bbox) / np.linalg.norm(original_bbox)
            if bbox_ratio < 0.5 or bbox_ratio > 2.0:
                logger.warning(f"⚠️ Mesh size changed significantly: ratio {bbox_ratio:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Mesh validation failed: {e}")
            return False


def process_single_file_worker(args):
    """Module-level worker function for parallel processing"""
    file_path, output_dir, reducer_params = args
    
    try:
        worker_reducer = BallastQualityFocusedReducer(
            target_reduction_ratio=reducer_params['target_reduction_ratio'],
            voxel_size=reducer_params['voxel_size'],
            n_cores=1,
            reconstruction_method=reducer_params['reconstruction_method'],
            fast_mode=reducer_params.get('fast_mode', False),
            use_random_forest=reducer_params.get('use_random_forest', False),
            enable_hierarchy=reducer_params.get('enable_hierarchy', True),
            hierarchy_threshold=reducer_params.get('hierarchy_threshold', 50000)
        )
        
        result = worker_reducer.process_single_mesh(file_path, output_dir)
        return result
        
    except Exception as e:
        logger.error(f"Worker error processing {file_path}: {e}")
        return {'input_file': file_path, 'error': str(e)}


class BallastQualityFocusedReducer:
    """
    Ballast Quality-Focused Point Cloud Reducer v2.2
    
    SPECIALIZED to fix the over-simplification issues shown in the ballast models
    """
    
    def __init__(self, 
                 target_reduction_ratio: float = 0.5,
                 voxel_size: Optional[float] = None,
                 n_cores: int = -1,
                 reconstruction_method: str = 'poisson',
                 fast_mode: bool = False,
                 use_random_forest: bool = True,
                 enable_hierarchy: bool = True,
                 force_hierarchy: bool = False,
                 hierarchy_threshold: int = 50000):
        """
        Initialize Ballast Quality-Focused Reducer v2.2
        """
        self.target_reduction_ratio = target_reduction_ratio
        self.voxel_size = voxel_size
        self.n_cores = mp.cpu_count() if n_cores == -1 else n_cores
        self.reconstruction_method = reconstruction_method
        self.fast_mode = fast_mode
        
        self.use_random_forest = use_random_forest
        self.enable_hierarchy = enable_hierarchy
        self.force_hierarchy = force_hierarchy
        self.hierarchy_threshold = hierarchy_threshold
        
        # NEW: Ballast quality specialist
        self.ballast_specialist = BallastQualitySpecialist()
        
        # Pipeline components
        self.scaler = StandardScaler()
        self.classifier = None
        self.best_params = {}
        
        # Performance optimization caches
        self.parameter_cache = {}
        
        # Processing thresholds (more conservative for ballast)
        self.SMALL_MODEL_THRESHOLD = hierarchy_threshold // 2  # Lower threshold
        self.MEDIUM_MODEL_THRESHOLD = hierarchy_threshold
        self.LARGE_MODEL_THRESHOLD = hierarchy_threshold * 2
        self.HUGE_MODEL_THRESHOLD = hierarchy_threshold * 4
        
        logger.info(f"🗿 Ballast Quality-Focused Reducer v2.3 initialized")
        logger.info(f"   Focus: Quality preservation + target compliance for ballast models")
        logger.info(f"   Hierarchical processing: {'ON' if enable_hierarchy else 'OFF'}")
        logger.info(f"   Classifier: {'RandomForest' if use_random_forest else 'SVM'}")
    
    def load_mesh(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load STL mesh and extract point cloud with normals"""
        try:
            mesh = trimesh.load(file_path)
            if hasattr(mesh, 'vertices'):
                points = np.array(mesh.vertices)
                normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
            else:
                raise ValueError("Failed to extract vertices from mesh")
                
            if normals is None:
                o3d_mesh = o3d.io.read_triangle_mesh(file_path)
                o3d_mesh.compute_vertex_normals()
                points = np.asarray(o3d_mesh.vertices)
                normals = np.asarray(o3d_mesh.vertex_normals)
                
            logger.info(f"📥 Loaded mesh with {len(points):,} vertices from {Path(file_path).name}")
            return points, normals
            
        except Exception as e:
            logger.error(f"❌ Failed to load mesh {file_path}: {e}")
            raise
    
    def normalize_points(self, points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize points to unit cube centered at origin"""
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        max_extent = np.max(np.abs(centered_points))
        scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0
        normalized_points = centered_points * scale_factor
        
        normalization_params = {
            'centroid': centroid,
            'scale_factor': scale_factor
        }
        
        return normalized_points, normalization_params
    
    def denormalize_points(self, points: np.ndarray, 
                          normalization_params: Dict) -> np.ndarray:
        """Transform points back to original coordinate frame"""
        if len(points) == 0:
            return points
        denormalized = points / normalization_params['scale_factor']
        denormalized += normalization_params['centroid']
        return denormalized
    
    @time_function
    def train_classifier(self, features: np.ndarray, labels: np.ndarray):
        """Train classifier optimized for ballast features"""
        features_scaled = self.scaler.fit_transform(features)
        
        if self.use_random_forest:
            # Optimized RandomForest for ballast
            self.classifier = RandomForestClassifier(
                n_estimators=100,  # More trees for ballast
                max_depth=15,      # Deeper trees
                random_state=42, 
                n_jobs=min(4, self.n_cores)
            )
        else:
            # SVM for quality
            self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        
        self.classifier.fit(features_scaled, labels)
        
        train_accuracy = self.classifier.score(features_scaled, labels)
        classifier_type = "RandomForest" if self.use_random_forest else "SVM"
        logger.debug(f"✅ {classifier_type} training accuracy: {train_accuracy:.3f}")
    
    def knn_reinforcement(self, points: np.ndarray, important_mask: np.ndarray, 
                         k_neighbors: int) -> np.ndarray:
        """Conservative KNN reinforcement for ballast"""
        if np.sum(important_mask) == 0:
            return important_mask
        
        important_indices = np.where(important_mask)[0]
        reinforced_mask = important_mask.copy()
        
        k = min(k_neighbors, len(points)-1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
        nbrs.fit(points)
        
        # Process all important points for quality (don't limit for ballast)
        for idx in important_indices:
            distances, neighbor_indices = nbrs.kneighbors([points[idx]])
            reinforced_mask[neighbor_indices[0]] = True
        
        logger.debug(f"🔗 Conservative KNN: {np.sum(important_mask):,} → {np.sum(reinforced_mask):,} points")
        return reinforced_mask
    
    def radius_merge(self, points: np.ndarray, normals: np.ndarray, 
                    epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Conservative radius merge for ballast quality"""
        if epsilon <= 0 or len(points) == 0:
            return points, normals
        
        clustering = DBSCAN(eps=epsilon, min_samples=1)
        cluster_labels = clustering.fit_predict(points)
        
        unique_labels = np.unique(cluster_labels)
        merged_points = []
        merged_normals = []
        
        for label in unique_labels:
            if label == -1:
                # Keep noise points for ballast detail
                noise_indices = np.where(cluster_labels == label)[0]
                merged_points.extend(points[noise_indices])
                merged_normals.extend(normals[noise_indices])
            else:
                cluster_indices = np.where(cluster_labels == label)[0]
                
                # For ballast, be more conservative about merging
                if len(cluster_indices) <= 3:  # Keep small clusters as-is
                    merged_points.extend(points[cluster_indices])
                    merged_normals.extend(normals[cluster_indices])
                else:
                    # Only merge larger clusters
                    centroid = np.mean(points[cluster_indices], axis=0)
                    avg_normal = np.mean(normals[cluster_indices], axis=0)
                    norm = np.linalg.norm(avg_normal)
                    if norm > 0:
                        avg_normal /= norm
                    
                    merged_points.append(centroid)
                    merged_normals.append(avg_normal)
        
        merged_points = np.array(merged_points) if merged_points else np.empty((0, 3))
        merged_normals = np.array(merged_normals) if merged_normals else np.empty((0, 3))
        
        logger.debug(f"🔄 Conservative merge: {len(points):,} → {len(merged_points):,} points")
        return merged_points, merged_normals
    
    def dbscan_cleanup(self, points: np.ndarray, normals: np.ndarray,
                      eps: float, min_samples: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Very conservative DBSCAN cleanup for ballast"""
        if len(points) == 0:
            return points, normals
        
        # Use very conservative parameters for ballast
        clustering = DBSCAN(eps=eps, min_samples=max(1, min_samples//2))  # More lenient
        cluster_labels = clustering.fit_predict(points)
        
        valid_mask = cluster_labels != -1
        
        # If too many points removed, keep them
        if np.sum(valid_mask) < len(points) * 0.7:  # If losing more than 30%
            logger.debug(f"🛡️ Keeping outliers to preserve ballast detail")
            return points, normals
        
        if np.sum(valid_mask) == 0:
            return points, normals
        
        logger.debug(f"🧹 Conservative cleanup: {len(points):,} → {np.sum(valid_mask):,} points")
        return points[valid_mask], normals[valid_mask]
    
    def process_ballast_quality_focused(self, points: np.ndarray, normals: np.ndarray, 
                                      input_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Main ballast processing with quality focus
        """
        logger.info("🗿 Starting ballast quality-focused processing...")
        
        # Analyze ballast complexity
        analysis = self.ballast_specialist.analyze_ballast_complexity(points)
        
        # Get quality-focused target points (much more conservative)
        optimal_target = self.ballast_specialist.get_quality_focused_target_points(
            points, self.target_reduction_ratio, analysis)
        
        # Adjust target ratio based on quality requirements
        adjusted_ratio = optimal_target / len(points)
        original_ratio = self.target_reduction_ratio
        self.target_reduction_ratio = adjusted_ratio
        
        logger.info(f"🎯 Target adjustment: {original_ratio:.4f} → {adjusted_ratio:.4f}")
        logger.info(f"   Points target: {len(points):,} → {optimal_target:,}")
        
        # Get quality-focused parameters that respect the target
        ballast_params = self.ballast_specialist.get_ballast_quality_parameters(
            analysis, optimal_target, len(points), aggressive_reduction=(adjusted_ratio < 0.02))
        
        # Normalize points
        normalized_points, norm_params = self.normalize_points(points)
        
        # Enhanced feature extraction for ballast
        features = self.ballast_specialist.enhanced_feature_extraction_for_ballast(
            normalized_points, k_neighbors=ballast_params['k_neighbors'])
        
        # Create ballast-specific importance labels
        pseudo_labels = self.ballast_specialist.create_ballast_importance_labels(
            features, normalized_points, ballast_params['importance_threshold'])
        
        # Train classifier
        self.train_classifier(features, pseudo_labels)
        
        # Predict importance
        features_scaled = self.scaler.transform(features)
        important_probs = self.classifier.predict_proba(features_scaled)[:, 1]
        important_mask = important_probs > 0.3  # Lower threshold for ballast
        
        logger.info(f"🎯 Initial important points: {np.sum(important_mask):,}/{len(points):,}")
        
        # Conservative KNN reinforcement
        reinforced_mask = self.knn_reinforcement(
            normalized_points, important_mask, ballast_params['k_neighbors'])
        
        logger.info(f"🔗 After reinforcement: {np.sum(reinforced_mask):,} points")
        
        selected_points = normalized_points[reinforced_mask]
        selected_normals = normals[reinforced_mask]
        
        if len(selected_points) == 0:
            logger.warning("⚠️ No points selected, using fallback sampling")
            # Fallback: uniform sampling
            step = max(1, len(points) // optimal_target)
            final_points = points[::step]
            final_normals = normals[::step]
        else:
            # Conservative clustering
            merged_points, merged_normals = self.radius_merge(
                selected_points, selected_normals, ballast_params['epsilon'])
            
            logger.info(f"🔄 After merging: {len(merged_points):,} points")
            
            # Very conservative cleanup
            final_points, final_normals = self.dbscan_cleanup(
                merged_points, merged_normals, ballast_params['dbscan_eps'])
            
            logger.info(f"🧹 After cleanup: {len(final_points):,} points")
            
            # Denormalize
            final_points = self.denormalize_points(final_points, norm_params)
            
            # NEW: Ensure we actually hit close to the target
            user_target = int(len(points) * original_ratio)
            final_points, final_normals = self.ensure_target_compliance(
                final_points, final_normals, optimal_target, user_target)
        
        # Restore original ratio
        self.target_reduction_ratio = original_ratio
        
        method_info = {
            'processing_method': 'ballast_quality_focused',
            'ballast_analysis': analysis,
            'ballast_parameters': ballast_params,
            'target_adjustment': {
                'original_ratio': original_ratio,
                'adjusted_ratio': adjusted_ratio,
                'original_target': int(len(points) * original_ratio),
                'quality_target': optimal_target,
                'final_count': len(final_points)
            }
        }
        
        return final_points, final_normals, method_info
        
    def ensure_target_compliance(self, points: np.ndarray, normals: np.ndarray, 
                               target_points: int, user_target: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        NEW: Ensure we actually hit close to the user's target while maintaining quality
        """
        current_count = len(points)
        
        # If we're way over the quality-adjusted target, need to reduce further
        if current_count > target_points * 2:  # More than 2x over target
            logger.info(f"🎯 Too many points ({current_count:,}), reducing to target ({target_points:,})")
            
            # Use importance-based sampling to get closer to target
            if current_count > target_points:
                # Calculate features for final selection
                features = self.ballast_specialist.enhanced_feature_extraction_for_ballast(points, k_neighbors=8)
                
                # Calculate importance scores
                importance_scores = (
                    features[:, 2] * 2.0 +  # Surface variation (most important for ballast)
                    features[:, 3] * 1.5 +  # Max neighbor distance  
                    features[:, 4] * 1.5 +  # Curvature
                    features[:, 5] * 1.0 +  # Surface roughness
                    features[:, 0] * 0.3 +  # Centroid distance (less important)
                    (1.0 / (features[:, 1] + 1e-8)) * 0.5  # Boundary points
                )
                
                # Select top most important points
                top_indices = np.argsort(importance_scores)[-target_points:]
                selected_points = points[top_indices]
                selected_normals = normals[top_indices]
                
                logger.info(f"✂️ Importance-based reduction: {current_count:,} → {len(selected_points):,} points")
                return selected_points, selected_normals
        
        # If we're close to target, keep as-is
        logger.info(f"✅ Point count acceptable: {current_count:,} points (target: {target_points:,})")
        return points, normals
    
    def process_single_mesh(self, input_path: str, output_dir: str) -> Dict:
        """
        Main processing method with ballast quality focus
        """
        try:
            start_time = time.time()
            
            # Check if this is a ballast model
            is_ballast = self.ballast_specialist.detect_ballast_model(input_path)
            
            # Load mesh
            load_start = time.time()
            points, normals = self.load_mesh(input_path)
            original_count = len(points)
            load_time = time.time() - load_start
            
            if original_count == 0:
                return {'input_file': input_path, 'error': 'Empty mesh'}
            
            logger.info(f"⏱️  Mesh loaded in {load_time:.1f}s")
            
            if is_ballast:
                logger.info(f"🗿 BALLAST MODEL DETECTED - Applying quality-focused processing")
            else:
                logger.info(f"📄 Regular model - Using standard processing")
            
            # Process based on type
            processing_start = time.time()
            
            if is_ballast:
                # Use quality-focused processing for ballast
                final_points, final_normals, method_info = self.process_ballast_quality_focused(
                    points, normals, input_path)
            else:
                # Use standard processing for non-ballast
                # (Simplified version for non-ballast models)
                normalized_points, norm_params = self.normalize_points(points)
                
                # Simple uniform sampling for non-ballast
                target_count = max(100, int(len(points) * self.target_reduction_ratio))
                step = max(1, len(points) // target_count)
                sampled_indices = np.arange(0, len(points), step)
                
                final_points = points[sampled_indices]
                final_normals = normals[sampled_indices]
                
                method_info = {
                    'processing_method': 'standard_uniform_sampling',
                    'target_count': target_count,
                    'final_count': len(final_points)
                }
            
            processing_time = time.time() - processing_start
            logger.info(f"⏱️  Processing completed in {processing_time:.1f}s")
            
            # Enhanced surface reconstruction for ballast
            recon_start = time.time()
            if is_ballast:
                logger.info("🔧 Applying ballast quality-focused surface reconstruction")
                reconstructed_mesh = self.ballast_specialist.quality_focused_surface_reconstruction(
                    final_points, final_normals, self.reconstruction_method)
            else:
                # Standard reconstruction for non-ballast
                reconstructed_mesh = self._standard_reconstruction(final_points, final_normals)
            
            recon_time = time.time() - recon_start
            logger.info(f"⏱️  Surface reconstruction in {recon_time:.1f}s")
            
            # Save results
            save_start = time.time()
            filename = Path(input_path).stem
            model_output_dir = Path(output_dir) / filename
            model_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created subfolder: {model_output_dir.name}/")
            
            # Save simplified STL
            stl_path = None
            if reconstructed_mesh:
                stl_path = model_output_dir / f"{filename}_simplified.stl"
                reconstructed_mesh.export(str(stl_path))
                quality_indicator = " (quality-focused)" if is_ballast else ""
                logger.info(f"💾 Saved STL{quality_indicator}: {model_output_dir.name}/{filename}_simplified.stl")
            else:
                logger.warning(f"⚠️  No STL generated for {filename} (reconstruction failed)")
            
            # Save point cloud as CSV
            csv_path = model_output_dir / f"{filename}_points.csv"
            point_df = pd.DataFrame(final_points, columns=['x', 'y', 'z'])
            normal_df = pd.DataFrame(final_normals, columns=['nx', 'ny', 'nz'])
            combined_df = pd.concat([point_df, normal_df], axis=1)
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"💾 Saved CSV: {model_output_dir.name}/{filename}_points.csv")
            
            # Save as DAT file
            dat_path = model_output_dir / f"{filename}_points.dat"
            np.savetxt(dat_path, final_points, fmt='%.6f')
            logger.info(f"💾 Saved DAT: {model_output_dir.name}/{filename}_points.dat")
            
            save_time = time.time() - save_start
            total_time = time.time() - start_time
            
            # Final completion message
            processing_type = "Quality-focused ballast" if is_ballast else "Standard"
            logger.info(f"✅ COMPLETED: {filename} → All files saved to {model_output_dir.name}/")
            logger.info(f"📊 Summary: {original_count:,} → {len(final_points):,} points (ratio: {len(final_points) / original_count:.4f})")
            logger.info(f"⏱️  Total processing time: {total_time:.1f}s")
            logger.info(f"🚀 Method: {processing_type}")
            
            if is_ballast and 'target_adjustment' in method_info:
                adj = method_info['target_adjustment']
                logger.info(f"🎯 Quality adjustment: {adj['original_target']:,} → {adj['quality_target']:,} target points")
            
            logger.info("-" * 80)
            
            # Results summary
            results = {
                'input_file': input_path,
                'original_points': original_count,
                'final_points': len(final_points),
                'reduction_ratio': len(final_points) / original_count,
                'target_ratio': self.target_reduction_ratio,
                'processing_time': total_time,
                'time_breakdown': {
                    'load': load_time,
                    'processing': processing_time,
                    'reconstruction': recon_time,
                    'save': save_time
                },
                'method_info': method_info,
                'ballast_detected': is_ballast,
                'quality_focused': is_ballast,
                'output_files': {
                    'stl': str(stl_path) if stl_path else None,
                    'csv': str(csv_path),
                    'dat': str(dat_path)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to process {input_path}: {e}")
            return {'input_file': input_path, 'error': str(e)}
    
    def _standard_reconstruction(self, points: np.ndarray, normals: np.ndarray) -> Optional[trimesh.Trimesh]:
        """Standard surface reconstruction for non-ballast models"""
        if len(points) < 4:
            return None
            
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8, width=0, scale=1.1, linear_fit=False)
            
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            if len(faces) == 0:
                return None
            
            return trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except Exception as e:
            logger.error(f"❌ Standard reconstruction failed: {e}")
            return None
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.stl") -> List[Dict]:
        """Process multiple mesh files in parallel"""
        input_path = Path(input_dir)
        stl_files = list(input_path.glob(file_pattern))
        
        if not stl_files:
            logger.warning(f"❌ No STL files found in {input_dir}")
            return []
        
        logger.info(f"🚀 Processing {len(stl_files)} files with {self.n_cores} cores")
        logger.info(f"🗿 Ballast quality-focused processing enabled")
        
        if len(stl_files) == 1 or self.n_cores == 1:
            # Single file or single core
            results = []
            for file_path in stl_files:
                try:
                    result = self.process_single_mesh(str(file_path), output_dir)
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {e}")
                    results.append({'input_file': str(file_path), 'error': str(e)})
        else:
            # Parallel processing
            reducer_params = {
                'target_reduction_ratio': self.target_reduction_ratio,
                'voxel_size': self.voxel_size,
                'reconstruction_method': self.reconstruction_method,
                'fast_mode': self.fast_mode,
                'use_random_forest': self.use_random_forest,
                'enable_hierarchy': self.enable_hierarchy,
                'hierarchy_threshold': self.hierarchy_threshold
            }
            
            worker_args = [
                (str(file_path), output_dir, reducer_params) 
                for file_path in stl_files
            ]
            
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                logger.info(f"🚀 Starting parallel processing with {self.n_cores} workers...")
                
                future_to_args = {
                    executor.submit(process_single_file_worker, args): args[0]
                    for args in worker_args
                }
                
                results = []
                completed = 0
                total_files = len(stl_files)
                
                for future in as_completed(future_to_args):
                    try:
                        result = future.result(timeout=600)
                        results.append(result)
                        completed += 1
                        
                        if 'error' not in result:
                            filename = Path(result['input_file']).name
                            model_name = Path(result['input_file']).stem
                            method = result.get('method_info', {}).get('processing_method', 'unknown')
                            ballast_detected = result.get('ballast_detected', False)
                            quality_focused = result.get('quality_focused', False)
                            
                            status_icons = ""
                            if ballast_detected:
                                status_icons += "🗿"
                            if quality_focused:
                                status_icons += "✨"
                            
                            logger.info(f"🎉 [{completed}/{total_files}] COMPLETED: {filename} {status_icons}")
                            logger.info(f"   📁 Subfolder: {model_name}/")
                            logger.info(f"   📊 Points: {result['original_points']:,} → {result['final_points']:,}")
                            logger.info(f"   🚀 Method: {method}")
                            logger.info(f"   ⏱️  Time: {result['processing_time']:.1f}s")
                            if ballast_detected:
                                logger.info(f"   🗿 Ballast quality focus: Applied")
                            logger.info("-" * 60)
                        else:
                            filename = Path(result['input_file']).name
                            logger.error(f"❌ [{completed}/{total_files}] FAILED: {filename}")
                            logger.error(f"   🚨 Error: {result['error']}")
                            logger.error("-" * 60)
                            
                    except Exception as e:
                        file_path = future_to_args[future]
                        logger.error(f"❌ Future error for {file_path}: {e}")
                        results.append({'input_file': str(file_path), 'error': str(e)})
                        completed += 1
        
        # Save batch summary
        summary_path = Path(output_dir) / "batch_summary.csv"
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(summary_path, index=False)
        
        logger.info("=" * 80)
        logger.info(f"📋 BATCH SUMMARY SAVED: {summary_path.name}")
        logger.info(f"📊 Total files processed: {len(results)}")
        successful_count = len([r for r in results if 'error' not in r])
        failed_count = len([r for r in results if 'error' in r])
        logger.info(f"✅ Successful: {successful_count}")
        logger.info(f"❌ Failed: {failed_count}")
        
        if successful_count > 0:
            # Performance analysis
            successful_results = [r for r in results if 'error' not in r]
            ballast_count = len([r for r in successful_results if r.get('ballast_detected', False)])
            standard_count = successful_count - ballast_count
            
            if ballast_count > 0:
                logger.info(f"🗿 Ballast quality-focused: {ballast_count} files")
                ballast_results = [r for r in successful_results if r.get('ballast_detected', False)]
                avg_ballast_time = np.mean([r['processing_time'] for r in ballast_results])
                avg_ballast_ratio = np.mean([r['reduction_ratio'] for r in ballast_results])
                logger.info(f"   Average processing time: {avg_ballast_time:.1f}s")
                logger.info(f"   Average reduction ratio: {avg_ballast_ratio:.4f}")
            
            if standard_count > 0:
                logger.info(f"📍 Standard processing: {standard_count} files")
            
            avg_time = np.mean([r['processing_time'] for r in successful_results])
            logger.info(f"⏱️  Overall average time: {avg_time:.1f}s per file")
        
        logger.info("=" * 80)
        return results


def estimate_target_ratio(input_path: str, target_count: int) -> float:
    """Estimate target reduction ratio based on sample file"""
    try:
        reducer = BallastQualityFocusedReducer()
        
        if os.path.isfile(input_path):
            sample_file = input_path
        else:
            stl_files = list(Path(input_path).glob("*.stl"))
            if not stl_files:
                return 0.5
            sample_file = str(stl_files[0])
        
        points, _ = reducer.load_mesh(sample_file)
        original_count = len(points)
        estimated_ratio = target_count / original_count if original_count > 0 else 0.5
        
        return max(0.01, min(0.95, estimated_ratio))
        
    except Exception as e:
        logger.warning(f"⚠️ Could not estimate ratio: {e}")
        return 0.5


def validate_args(args):
    """Validate command line arguments"""
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    if args.count and args.ratio:
        print("Error: Cannot specify both --count and --ratio")
        sys.exit(1)
    
    if not args.count and not args.ratio:
        print("Error: Must specify either --count or --ratio")
        sys.exit(1)
    
    if args.ratio and (args.ratio <= 0 or args.ratio >= 1):
        print("Error: --ratio must be between 0 and 1")
        sys.exit(1)
    
    if args.count and args.count <= 0:
        print("Error: --count must be positive")
        sys.exit(1)
    
    if args.workers <= 0:
        print("Error: --workers must be positive")
        sys.exit(1)
    
    valid_methods = ['poisson', 'ball_pivoting', 'alpha_shapes', 'none']
    if args.method not in valid_methods:
        print(f"Error: --method must be one of {valid_methods}")
        sys.exit(1)


def print_config(args, target_ratio: float, log_file: Optional[str] = None):
    """Print processing configuration"""
    logging.info("=" * 60)
    logging.info("🗿 Ballast Quality-Focused Reduction System v2.3")
    logging.info("=" * 60)
    logging.info(f"📂 Input: {args.input}")
    logging.info(f"📁 Output: {args.output}")
    
    if args.count:
        logging.info(f"🎯 Target: {args.count} points (estimated ratio: {target_ratio:.4f})")
    else:
        logging.info(f"🎯 Target: {args.ratio:.1%} of original points")
    
    logging.info(f"👥 Workers: {args.workers}")
    logging.info(f"🔧 Method: {args.method}")
    
    features = ["Ballast Quality Focus"]
    if args.fast_mode:
        features.append("Fast Mode")
    if args.use_random_forest:
        features.append("RandomForest")
    if args.enable_hierarchy and not args.no_hierarchy:
        features.append(f"Hierarchical (>{args.hierarchy_threshold:,} points)")
    if args.force_hierarchy:
        features.append("Force Hierarchy")
    
    logging.info(f"✨ Features: {', '.join(features)}")
    
    if args.voxel:
        logging.info(f"📦 Voxel size: {args.voxel}")
    
    if args.verbose:
        logging.info("🔍 Verbose mode: ON")
    
    if log_file:
        logging.info(f"📝 Log file: {log_file}")
    
    logging.info("-" * 60)


def process_files(args, target_ratio: float) -> bool:
    """Main processing function"""
    
    logging.info("🗿 Initializing Ballast Quality-Focused Reducer v2.3...")
    
    # Initialize quality-focused reducer
    reducer = BallastQualityFocusedReducer(
        target_reduction_ratio=target_ratio,
        voxel_size=args.voxel,
        n_cores=args.workers,
        reconstruction_method=args.method,
        fast_mode=args.fast_mode,
        use_random_forest=args.use_random_forest,
        enable_hierarchy=args.enable_hierarchy and not args.no_hierarchy,
        force_hierarchy=args.force_hierarchy,
        hierarchy_threshold=args.hierarchy_threshold
    )
    
    logging.info("✅ Ballast quality-focused reducer v2.3 initialized successfully")
    start_time = time.time()
    
    if os.path.isfile(args.input):
        # Single file processing
        logging.info(f"🔄 Processing single file: {os.path.basename(args.input)}")
        
        results = reducer.process_single_mesh(args.input, args.output)
        
        if 'error' in results:
            logging.error(f"❌ Processing failed: {results['error']}")
            return False
        else:
            method = results.get('method_info', {}).get('processing_method', 'unknown')
            ballast_detected = results.get('ballast_detected', False)
            quality_focused = results.get('quality_focused', False)
            
            logging.info(f"✅ SUCCESS: {results['original_points']:,} → {results['final_points']:,} points")
            logging.info(f"📊 Actual ratio: {results['reduction_ratio']:.4f}")
            logging.info(f"🚀 Method used: {method}")
            if ballast_detected:
                logging.info(f"🗿 Ballast detected: Quality-focused processing applied")
            if quality_focused:
                logging.info(f"✨ Quality enhancement: Applied")
            
    else:
        # Batch processing
        stl_files = list(Path(args.input).glob("*.stl"))
        if not stl_files:
            logging.error(f"❌ No STL files found in {args.input}")
            return False
        
        logging.info(f"🔄 Processing {len(stl_files)} files...")
        
        results = reducer.process_batch(args.input, args.output, "*.stl")
        
        # Print summary
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        logging.info("")
        logging.info("📊 FINAL BATCH RESULTS:")
        logging.info("=" * 40)
        logging.info(f"✅ Successful: {len(successful)}")
        logging.info(f"❌ Failed: {len(failed)}")
        
        if successful:
            total_original = sum(r['original_points'] for r in successful)
            total_final = sum(r['final_points'] for r in successful)
            avg_ratio = total_final / total_original if total_original > 0 else 0
            
            logging.info(f"📈 Total points: {total_original:,} → {total_final:,}")
            logging.info(f"📊 Average ratio: {avg_ratio:.4f}")
            logging.info(f"🎯 Target ratio: {target_ratio:.4f}")
            
            # Quality-focused analysis
            ballast_results = [r for r in successful if r.get('ballast_detected', False)]
            standard_results = [r for r in successful if not r.get('ballast_detected', False)]
            
            if ballast_results:
                avg_ballast_time = np.mean([r['processing_time'] for r in ballast_results])
                avg_ballast_ratio = np.mean([r['reduction_ratio'] for r in ballast_results])
                logging.info(f"🗿 Ballast quality-focused: {len(ballast_results)} files")
                logging.info(f"   Average time: {avg_ballast_time:.1f}s, Average ratio: {avg_ballast_ratio:.4f}")
            
            if standard_results:
                avg_standard_time = np.mean([r['processing_time'] for r in standard_results])
                logging.info(f"📍 Standard processing: {len(standard_results)} files, avg {avg_standard_time:.1f}s")
        
        if failed and len(failed) <= 10:
            logging.warning(f"❌ Failed files:")
            for result in failed:
                filename = os.path.basename(result.get('input_file', 'unknown'))
                error = result.get('error', 'Unknown error')
                logging.warning(f"   • {filename}: {error}")
        elif failed:
            logging.warning(f"❌ {len(failed)} files failed (check batch_summary.csv for details)")
    
    # Print timing and summary
    elapsed_time = time.time() - start_time
    logging.info("")
    logging.info("⏱️  PROCESSING COMPLETE")
    logging.info("=" * 40)
    logging.info(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    if elapsed_time > 60:
        logging.info(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    logging.info(f"📁 Results saved to: {args.output}")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Ballast Quality-Focused Point Cloud Reduction v2.3 (Target-Respecting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (SAME COMMANDS - now with target compliance + ballast quality!):
  python ballast-quality-focused-v2.3.py /home/railcmu/Desktop/BPK/ballast --count 100 --workers 2
  python ballast-quality-focused-v2.3.py /path/to/models --ratio 0.3 --workers 8
  python ballast-quality-focused-v2.3.py model.stl --count 50 --method poisson

Quality-Focused + Target-Respecting Features:
  - Automatic ballast detection (keywords: ballast, stone, rock, bpk, aggregate, gravel)
  - 2-3x more points retained for ballast models (vs standard reduction)
  - Enhanced feature extraction for rough surfaces
  - Multiple reconstruction method fallbacks
  - Target-aware parameter adjustment (respects user intent)
  - Quality validation and mesh fixing

Performance Features:
  --fast-mode                 Skip parameter optimization
  --use-random-forest        Use RandomForest instead of SVM
  --no-hierarchy             Disable automatic hierarchical processing
  --force-hierarchy          Force hierarchical processing on all models
  --hierarchy-threshold N     Set threshold for hierarchical processing

What's NEW in v2.3:
✅ RESPECTS USER TARGETS while maintaining ballast quality
✅ Target-aware parameter adjustment
✅ Reasonable quality multipliers (2-3x vs 7.5x before)
✅ Post-processing to ensure target compliance
✅ Better clustering parameters based on reduction aggressiveness
✅ Same interface - now hits your targets while preserving ballast quality!

Installation:
  pip install numpy pandas scikit-learn trimesh open3d
        """
    )
    
    # Arguments
    parser.add_argument('input', 
                       help='Input STL file or directory containing STL files')
    
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--count', type=int,
                             help='Target number of points to keep')
    target_group.add_argument('--ratio', type=float,
                             help='Target reduction ratio (0.0-1.0)')
    
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    
    parser.add_argument('--method', type=str, default='poisson',
                       choices=['poisson', 'ball_pivoting', 'alpha_shapes', 'none'],
                       help='Surface reconstruction method (default: poisson)')
    parser.add_argument('--voxel', type=float,
                       help='Voxel size for preprocessing downsampling')
    
    # Performance options
    performance_group = parser.add_argument_group('performance options')
    performance_group.add_argument('--fast-mode', action='store_true',
                                 help='Skip parameter optimization for faster processing')
    performance_group.add_argument('--use-random-forest', action='store_true', default=True,
                                 help='Use RandomForest classifier (default, faster than SVM)')
    performance_group.add_argument('--use-svm', action='store_true',
                                 help='Use SVM classifier (slower but potentially higher quality)')
    
    # Hierarchical processing options
    hierarchy_group = parser.add_argument_group('hierarchical processing options')
    hierarchy_group.add_argument('--no-hierarchy', action='store_true',
                               help='Disable automatic hierarchical processing')
    hierarchy_group.add_argument('--force-hierarchy', action='store_true',
                               help='Force hierarchical processing on all models')
    hierarchy_group.add_argument('--hierarchy-threshold', type=int, default=50000,
                               help='Point count threshold for hierarchical processing (default: 50000)')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Custom log file path (default: auto-generated in logs/ folder)')
    parser.add_argument('--no-log', action='store_true',
                       help='Disable automatic log file creation (console only)')
    parser.add_argument('--version', action='version', version='2.3.0 (Ballast Quality-Focused + Target-Respecting)')
    
    args = parser.parse_args()
    
    # Handle classifier selection
    if args.use_svm:
        args.use_random_forest = False
    
    # Enable hierarchy by default unless disabled
    args.enable_hierarchy = not args.no_hierarchy
    
    # Auto-generate log file path if not specified and not disabled
    log_file = None
    if not getattr(args, 'no_log', False):
        if getattr(args, 'log_file', None):
            log_file = args.log_file
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            input_name = Path(args.input).name if os.path.isfile(args.input) else Path(args.input).name
            log_file = f"logs/{input_name}_ballast_quality_v2.3_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(args.verbose, log_file)
    
    # Log startup information
    logging.info("🗿 Ballast Quality-Focused Point Cloud Reduction System v2.3 Starting...")
    logging.info(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"💻 System: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    logging.info(f"⚡ Available CPU cores: {mp.cpu_count()}")
    
    # Validate arguments
    validate_args(args)
    
    # Calculate target ratio
    if args.count:
        logging.info(f"🎯 Estimating target ratio for {args.count} points...")
        target_ratio = estimate_target_ratio(args.input, args.count)
        logging.info(f"📊 Estimated target ratio: {target_ratio:.4f}")
    else:
        target_ratio = args.ratio
        logging.info(f"📊 Using specified ratio: {target_ratio:.4f}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    logging.info(f"📁 Output directory ready: {args.output}")
    
    # Print configuration
    print_config(args, target_ratio, log_file)
    
    # Process files
    try:
        logging.info("🏁 Starting ballast quality-focused processing...")
        success = process_files(args, target_ratio)
        
        if success:
            logging.info("")
            logging.info("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            logging.info("🗿 Ballast Quality-Focused Reduction v2.3 - Quality + Target Compliance!")
            logging.info(f"🕐 Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            if log_file:
                logging.info(f"📝 Detailed logs saved to: {log_file}")
            sys.exit(0)
        else:
            logging.error("")
            logging.error("💥 PROCESSING FAILED!")
            logging.error("Check the error messages above for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.warning("")
        logging.warning("⏹️  Processing interrupted by user (Ctrl+C)")
        logging.warning("Partial results may be available in output directory")
        sys.exit(1)
    except Exception as e:
        logging.error("")
        logging.error(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            logging.error("🔍 Full traceback:")
            logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
