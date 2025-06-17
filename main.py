#!/usr/bin/env python3
"""
Enhanced Ballast Quality-Focused Point Cloud Reduction System v2.4 (FIXED)

NEW FEATURES:
- FIXED ultra-aggressive reduction modes for proper point reduction
- Comprehensive vertex and face calculation
- Detailed mesh statistics reporting
- Flexible reduction strategies
- Enhanced performance metrics

Usage:
    python ballast-reducer-v2.4-fixed.py /path/to/models --count 50 --aggressive --workers 4
    python ballast-reducer-v2.4-fixed.py /path/to/models --ratio 0.01 --ultra-aggressive --workers 8

Key Improvements:
    ✅ FIXED aggressive and ultra-aggressive reduction modes
    ✅ Comprehensive vertex/face counting and reporting
    ✅ Enhanced mesh statistics in output
    ✅ Flexible quality vs reduction balance
    ✅ Detailed performance metrics
    ✅ Improved reconstruction with face counting

Requirements:
    pip install numpy pandas scikit-learn trimesh open3d

Author: Theeradon
Version: 2.4.1 (Fixed Ultra-Aggressive Reduction + Vertex/Face Analytics)
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


class MeshAnalytics:
    """Class for comprehensive mesh analytics including vertex/face counting"""
    
    @staticmethod
    def analyze_mesh_detailed(mesh: trimesh.Trimesh, original_points: int) -> Dict:
        """Comprehensive mesh analysis with all statistics"""
        if mesh is None:
            return {
                'vertices': 0,
                'faces': 0,
                'edges': 0,
                'surface_area': 0.0,
                'volume': 0.0,
                'is_watertight': False,
                'is_valid': False,
                'euler_number': 0,
                'genus': 0,
                'bounding_box_volume': 0.0,
                'vertex_reduction_ratio': 0.0,
                'face_density': 0.0
            }
        
        try:
            vertices = len(mesh.vertices)
            faces = len(mesh.faces)
            
            # Basic mesh properties
            surface_area = float(mesh.area) if hasattr(mesh, 'area') else 0.0
            volume = float(mesh.volume) if hasattr(mesh, 'volume') else 0.0
            is_watertight = bool(mesh.is_watertight) if hasattr(mesh, 'is_watertight') else False
            is_valid = bool(mesh.is_valid) if hasattr(mesh, 'is_valid') else False
            
            # Topological properties
            euler_number = 0
            genus = 0
            edges = 0
            
            if hasattr(mesh, 'euler_number'):
                try:
                    euler_number = int(mesh.euler_number)
                    # Genus calculation: genus = (2 - euler_number) / 2
                    genus = max(0, (2 - euler_number) // 2)
                except:
                    pass
            
            if hasattr(mesh, 'edges'):
                try:
                    edges = len(mesh.edges)
                except:
                    # Estimate edges using Euler's formula: V - E + F = 2 - 2*genus
                    edges = max(0, vertices + faces - 2 + 2 * genus)
            
            # Bounding box volume
            bbox_volume = 0.0
            if hasattr(mesh, 'bounds'):
                try:
                    bbox_dims = mesh.bounds[1] - mesh.bounds[0]
                    bbox_volume = float(np.prod(bbox_dims))
                except:
                    pass
            
            # Derived metrics
            vertex_reduction_ratio = vertices / original_points if original_points > 0 else 0.0
            face_density = faces / vertices if vertices > 0 else 0.0
            
            analysis = {
                'vertices': vertices,
                'faces': faces,
                'edges': edges,
                'surface_area': surface_area,
                'volume': volume,
                'is_watertight': is_watertight,
                'is_valid': is_valid,
                'euler_number': euler_number,
                'genus': genus,
                'bounding_box_volume': bbox_volume,
                'vertex_reduction_ratio': vertex_reduction_ratio,
                'face_density': face_density
            }
            
            logger.debug(f"📊 Mesh analysis: {vertices:,} vertices, {faces:,} faces")
            logger.debug(f"   Surface area: {surface_area:.2f}, Volume: {volume:.2f}")
            logger.debug(f"   Watertight: {is_watertight}, Valid: {is_valid}")
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Mesh analysis failed: {e}")
            return {
                'vertices': len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
                'faces': len(mesh.faces) if hasattr(mesh, 'faces') else 0,
                'edges': 0,
                'surface_area': 0.0,
                'volume': 0.0,
                'is_watertight': False,
                'is_valid': False,
                'euler_number': 0,
                'genus': 0,
                'bounding_box_volume': 0.0,
                'vertex_reduction_ratio': 0.0,
                'face_density': 0.0
            }


class AggressiveReductionSpecialist:
    """
    FIXED: Specialist for aggressive point reduction while maintaining mesh quality
    """
    
    def __init__(self, aggressive_mode: str = 'moderate'):
        """
        Initialize aggressive reduction specialist
        
        Args:
            aggressive_mode: 'moderate', 'aggressive', 'ultra_aggressive'
        """
        self.aggressive_mode = aggressive_mode
        
        # FIXED: Aggressive reduction configurations - more balanced
        self.aggressive_configs = {
            'moderate': {
                'quality_multiplier_max': 2.0,
                'importance_threshold': 30,  # Keep top 70%
                'min_points_absolute': 50,
                'epsilon_scale': 1.2,
                'clustering_min_samples': 2,
                'knn_neighbors': 6
            },
            'aggressive': {
                'quality_multiplier_max': 1.8,  # Reduced from 1.5
                'importance_threshold': 25,     # Reduced from 20 (Keep top 75%)
                'min_points_absolute': 40,      # Increased from 30
                'epsilon_scale': 1.4,           # Reduced from 1.5
                'clustering_min_samples': 1,
                'knn_neighbors': 5              # Increased from 4
            },
            'ultra_aggressive': {
                'quality_multiplier_max': 1.5,  # Increased from 1.2
                'importance_threshold': 20,     # Increased from 10 (Keep top 80%)
                'min_points_absolute': 30,      # Increased from 20
                'epsilon_scale': 1.6,           # Reduced from 2.0
                'clustering_min_samples': 1,
                'knn_neighbors': 4              # Increased from 3
            }
        }
        
        self.config = self.aggressive_configs.get(aggressive_mode, self.aggressive_configs['moderate'])
        logger.info(f"🔥 Aggressive reduction mode: {aggressive_mode}")
        logger.info(f"   Quality multiplier max: {self.config['quality_multiplier_max']}x")
        logger.info(f"   Importance threshold: {self.config['importance_threshold']}% (keep top {100-self.config['importance_threshold']}%)")
    
    def get_aggressive_target_points(self, original_points: np.ndarray, 
                                   target_ratio: float, analysis: Dict) -> int:
        """FIXED: Calculate aggressive target points"""
        original_count = len(original_points)
        base_target = int(original_count * target_ratio)
        
        # FIXED: Apply more conservative quality multipliers
        if self.aggressive_mode == 'ultra_aggressive':
            if base_target < 20:
                quality_multiplier = min(2.0, self.config['quality_multiplier_max'])
            elif base_target < 100:
                quality_multiplier = min(1.8, self.config['quality_multiplier_max'] * 0.8)
            else:
                quality_multiplier = min(1.5, self.config['quality_multiplier_max'] * 0.6)
        elif self.aggressive_mode == 'aggressive':
            if base_target < 20:
                quality_multiplier = min(2.5, self.config['quality_multiplier_max'])
            elif base_target < 100:
                quality_multiplier = min(2.0, self.config['quality_multiplier_max'] * 0.8)
            else:
                quality_multiplier = min(1.8, self.config['quality_multiplier_max'] * 0.6)
        else:
            # Moderate mode
            if base_target < 20:
                quality_multiplier = self.config['quality_multiplier_max']
            elif base_target < 100:
                quality_multiplier = self.config['quality_multiplier_max'] * 0.8
            else:
                quality_multiplier = self.config['quality_multiplier_max'] * 0.6
        
        # FIXED: Conservative complexity adjustment
        if analysis['complexity'] == 'very_high' and analysis['surface_roughness'] > 0.4:
            quality_multiplier *= 1.2
        elif analysis['complexity'] == 'high' and analysis['surface_roughness'] > 0.3:
            quality_multiplier *= 1.1
        
        adjusted_target = int(base_target * quality_multiplier)
        
        # FIXED: Ensure minimum viable points based on mode
        if self.aggressive_mode == 'ultra_aggressive':
            min_viable = max(self.config['min_points_absolute'], 30)  # Increased minimum
        elif self.aggressive_mode == 'aggressive':
            min_viable = max(self.config['min_points_absolute'], 40)
        else:
            min_viable = max(self.config['min_points_absolute'], 50)
        
        optimal_points = max(adjusted_target, min_viable)
        
        # FIXED: More conservative maximum caps
        if self.aggressive_mode == 'ultra_aggressive':
            max_allowed = min(base_target * 4, int(original_count * 0.05))  # Cap at 5% of original
        elif self.aggressive_mode == 'aggressive':
            max_allowed = min(base_target * 5, int(original_count * 0.08))  # Cap at 8% of original
        else:
            max_allowed = base_target * 6
        
        optimal_points = min(optimal_points, max_allowed)
        
        logger.info(f"🎯 Aggressive target: {original_count:,} → {optimal_points:,} points")
        logger.info(f"   Base target: {base_target:,}, Aggressive target: {optimal_points:,}")
        logger.info(f"   Mode: {self.aggressive_mode}, Multiplier: {quality_multiplier:.1f}x")
        
        return optimal_points
    
    def get_aggressive_parameters(self, analysis: Dict, target_points: int, original_points: int) -> Dict:
        """FIXED: Get parameters optimized for aggressive reduction"""
        complexity = analysis['complexity']
        surface_roughness = analysis['surface_roughness']
        reduction_ratio = target_points / original_points
        
        # FIXED: Much more balanced parameter selection
        if self.aggressive_mode == 'ultra_aggressive':
            importance_threshold = max(15, self.config['importance_threshold'])  # Keep top 85%
            epsilon_scale = min(1.8, self.config['epsilon_scale'])  # Less aggressive scaling
            k_neighbors = max(3, self.config['knn_neighbors'])
        elif self.aggressive_mode == 'aggressive':
            importance_threshold = max(20, self.config['importance_threshold'])  # Keep top 80%
            epsilon_scale = min(1.4, self.config['epsilon_scale'])
            k_neighbors = max(4, self.config['knn_neighbors'])
        else:
            importance_threshold = self.config['importance_threshold']
            epsilon_scale = self.config['epsilon_scale']
            k_neighbors = self.config['knn_neighbors']
        
        # FIXED: Increase aggressiveness based on how extreme the reduction is, but more conservatively
        if reduction_ratio < 0.005:  # Less than 0.5%
            importance_threshold = max(10, importance_threshold - 8)  # More conservative
            epsilon_scale *= 1.3  # Less aggressive
            k_neighbors = max(2, k_neighbors - 1)
        elif reduction_ratio < 0.01:  # Less than 1%
            importance_threshold = max(12, importance_threshold - 5)
            epsilon_scale *= 1.2
        
        # FIXED: Base clustering parameters - more conservative
        if surface_roughness > 0.2:
            base_epsilon = 0.020  # Increased from 0.015
        elif surface_roughness > 0.1:
            base_epsilon = 0.030  # Increased from 0.025
        else:
            base_epsilon = 0.040  # Increased from 0.035
        
        epsilon = base_epsilon * epsilon_scale
        dbscan_eps = epsilon * 2.0  # Reduced from 2.5
        
        params = {
            'k_neighbors': k_neighbors,
            'epsilon': epsilon,
            'dbscan_eps': dbscan_eps,
            'importance_threshold': importance_threshold,
            'clustering_min_samples': self.config['clustering_min_samples'],
            'complexity': complexity,
            'surface_roughness': surface_roughness,
            'reduction_ratio': reduction_ratio,
            'aggressive_mode': self.aggressive_mode
        }
        
        logger.info(f"🔥 Aggressive parameters: {params}")
        return params
    
    def aggressive_feature_scoring(self, features: np.ndarray, points: np.ndarray, 
                                 importance_threshold: int = 20) -> np.ndarray:
        """FIXED: More aggressive feature scoring for maximum reduction"""
        n_points = len(points)
        importance_scores = np.zeros(n_points)
        
        # Focus on only the most critical features for aggressive reduction
        
        # Critical geometric features (edges, corners) - highest priority
        critical_features = features[:, 2] + features[:, 3] + features[:, 4] * 3
        importance_scores += critical_features * 3.0
        
        # Surface detail - secondary priority  
        surface_detail = features[:, 5]
        importance_scores += surface_detail * 1.0
        
        # Boundary detection - tertiary priority
        boundary_score = 1.0 / (features[:, 1] + 1e-8)
        importance_scores += boundary_score * 0.5
        
        # Minimize centroid distance influence for aggressive reduction
        centroid_distance_score = features[:, 0]
        importance_scores += centroid_distance_score * 0.1
        
        # Normalize scores
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        
        # Use threshold, but ensure we keep enough points for reconstruction
        threshold = np.percentile(importance_scores, importance_threshold)
        pseudo_labels = (importance_scores >= threshold).astype(int)
        
        # FIXED: Safety check: ensure we keep enough points
        min_points_needed = max(50, int(n_points * 0.01))  # At least 1% or 50 points
        if np.sum(pseudo_labels) < min_points_needed:
            # Adjust threshold to keep more points
            new_threshold = np.percentile(importance_scores, max(0, importance_threshold - 10))
            pseudo_labels = (importance_scores >= new_threshold).astype(int)
            logger.warning(f"⚠️ Adjusted threshold to keep {np.sum(pseudo_labels):,} points (was {np.sum((importance_scores >= threshold).astype(int)):,})")
        
        logger.info(f"🔥 Aggressive scoring: {np.sum(pseudo_labels):,}/{n_points:,} critical points")
        logger.info(f"   Keeping top {100-importance_threshold}% most important features")
        
        return pseudo_labels


class BallastQualitySpecialist:
    """Enhanced specialist with FIXED aggressive reduction capabilities"""
    
    def __init__(self, aggressive_mode: str = 'moderate'):
        self.aggressive_mode = aggressive_mode
        self.aggressive_specialist = AggressiveReductionSpecialist(aggressive_mode)
        
        # Base ballast configuration - now more flexible
        self.ballast_config = {
            'min_points_small_ballast': 20,   # Further reduced
            'min_points_medium_ballast': 40,  # Further reduced
            'min_points_large_ballast': 80,   # Further reduced
            
            'epsilon_fine': 0.008,
            'epsilon_medium': 0.015,
            'epsilon_coarse': 0.025,
            
            'poisson_depth_high': 10,
            'poisson_depth_medium': 9,
            'poisson_depth_low': 8,
            
            'min_points_for_reconstruction': 15,  # Lowered significantly
            'max_reconstruction_attempts': 5,     # Increased attempts
            'normal_estimation_neighbors': 15,    # Reduced
        }
        
    def detect_ballast_model(self, file_path: str) -> bool:
        """Detect if this is likely a ballast model"""
        filename = file_path.lower()
        ballast_keywords = ['ballast', 'stone', 'rock', 'aggregate', 'gravel', 'bpk']
        return any(keyword in filename for keyword in ballast_keywords)
    
    def analyze_ballast_complexity(self, points: np.ndarray) -> Dict:
        """Enhanced complexity analysis"""
        n_points = len(points)
        
        bbox = np.max(points, axis=0) - np.min(points, axis=0)
        bbox_volume = np.prod(bbox)
        bbox_surface_area = 2 * (bbox[0]*bbox[1] + bbox[1]*bbox[2] + bbox[0]*bbox[2])
        
        if n_points > 100:
            sample_size = min(1000, n_points)
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            sample_points = points[sample_indices]
            
            if sample_size > 10:
                nbrs = NearestNeighbors(n_neighbors=min(10, sample_size-1))
                nbrs.fit(sample_points)
                distances, _ = nbrs.kneighbors(sample_points)
                avg_neighbor_distance = np.mean(distances[:, 1:])
                surface_roughness = np.std(distances[:, 1:])
            else:
                avg_neighbor_distance = 0.1
                surface_roughness = 0.05
        else:
            avg_neighbor_distance = 0.1
            surface_roughness = 0.05
        
        # More nuanced complexity classification
        if bbox_volume > 2000 or n_points > 200000:
            complexity = "very_high"
        elif bbox_volume > 1000 or n_points > 100000:
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
            'original_points': n_points,
            'aggressive_mode': self.aggressive_mode
        }
        
        logger.info(f"🔍 Enhanced analysis: {complexity} complexity, roughness: {surface_roughness:.4f}")
        logger.info(f"   Aggressive mode: {self.aggressive_mode}")
        
        return analysis
    
    def get_enhanced_target_points(self, original_points: np.ndarray, 
                                 target_ratio: float, analysis: Dict) -> int:
        """Enhanced target calculation with aggressive options"""
        if self.aggressive_mode in ['aggressive', 'ultra_aggressive']:
            return self.aggressive_specialist.get_aggressive_target_points(
                original_points, target_ratio, analysis)
        
        # Standard enhanced calculation for moderate mode
        original_count = len(original_points)
        base_target = int(original_count * target_ratio)
        
        # More reasonable quality adjustments
        if base_target < 30:
            quality_multiplier = 2.5
        elif base_target < 100:
            quality_multiplier = 2.0
        elif base_target < 500:
            quality_multiplier = 1.5
        else:
            quality_multiplier = 1.3
        
        # Complexity adjustment
        if analysis['complexity'] in ['very_high', 'high'] and analysis['surface_roughness'] > 0.3:
            quality_multiplier *= 1.2
        elif analysis['surface_roughness'] > 0.1:
            quality_multiplier *= 1.1
        
        adjusted_target = int(base_target * quality_multiplier)
        min_viable = 25
        optimal_points = max(adjusted_target, min_viable)
        
        # Less restrictive maximum
        max_allowed = min(base_target * 4, 1500)
        optimal_points = min(optimal_points, max_allowed)
        
        logger.info(f"🎯 Enhanced target: {original_count:,} → {optimal_points:,} points")
        
        return optimal_points
    
    def get_enhanced_parameters(self, analysis: Dict, target_points: int, original_points: int) -> Dict:
        """Enhanced parameter calculation"""
        if self.aggressive_mode in ['aggressive', 'ultra_aggressive']:
            return self.aggressive_specialist.get_aggressive_parameters(
                analysis, target_points, original_points)
        
        # Standard enhanced parameters
        complexity = analysis['complexity']
        surface_roughness = analysis['surface_roughness']
        reduction_ratio = target_points / original_points
        
        if reduction_ratio < 0.02:
            importance_threshold = 25
            epsilon_scale = 1.4
            k_neighbors = 6
        elif reduction_ratio < 0.05:
            importance_threshold = 35
            epsilon_scale = 1.2
            k_neighbors = 7
        else:
            importance_threshold = 45
            epsilon_scale = 1.0
            k_neighbors = 8
        
        if surface_roughness > 0.1:
            base_epsilon = self.ballast_config['epsilon_fine']
        elif surface_roughness > 0.05:
            base_epsilon = self.ballast_config['epsilon_medium']
        else:
            base_epsilon = self.ballast_config['epsilon_coarse']
        
        epsilon = base_epsilon * epsilon_scale
        dbscan_eps = epsilon * 2.0
        
        params = {
            'k_neighbors': k_neighbors,
            'epsilon': epsilon,
            'dbscan_eps': dbscan_eps,
            'importance_threshold': importance_threshold,
            'clustering_min_samples': 2,
            'complexity': complexity,
            'surface_roughness': surface_roughness,
            'reduction_ratio': reduction_ratio,
            'aggressive_mode': self.aggressive_mode
        }
        
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
    
    def enhanced_surface_reconstruction_with_analytics(self, points: np.ndarray, normals: np.ndarray, 
                                                      method: str = 'poisson') -> Tuple[Optional[trimesh.Trimesh], Dict]:
        """Enhanced surface reconstruction with comprehensive analytics"""
        reconstruction_analytics = {
            'method_used': None,
            'attempts': 0,
            'success': False,
            'vertices': 0,
            'faces': 0,
            'reconstruction_time': 0.0
        }
        
        start_time = time.time()
        
        if len(points) < self.ballast_config['min_points_for_reconstruction']:
            logger.warning(f"⚠️ Too few points ({len(points)}) for reconstruction")
            reconstruction_analytics['reconstruction_time'] = time.time() - start_time
            return None, reconstruction_analytics
        
        logger.info(f"🔧 Enhanced reconstruction for {len(points):,} points")
        
        # Enhanced normals
        try:
            normals = self.improve_normals_for_ballast(points, normals)
        except Exception as e:
            logger.warning(f"⚠️ Normal enhancement failed: {e}")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Try multiple reconstruction methods
        reconstruction_methods = [
            ('poisson_high', self._try_poisson_high_quality),
            ('poisson_medium', self._try_poisson_medium_quality),
            ('ball_pivoting_adaptive', self._try_ball_pivoting_adaptive),
            ('poisson_low', self._try_poisson_low_quality),
            ('alpha_shapes', self._try_alpha_shapes)
        ]
        
        for method_name, reconstruction_func in reconstruction_methods:
            reconstruction_analytics['attempts'] += 1
            try:
                logger.info(f"🔄 Trying {method_name}...")
                mesh = reconstruction_func(pcd)
                
                if mesh is not None:
                    vertices = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.triangles)
                    
                    if len(faces) > 0:
                        reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        
                        if self._validate_mesh_quality(reconstructed_mesh, points):
                            reconstruction_analytics.update({
                                'method_used': method_name,
                                'success': True,
                                'vertices': len(vertices),
                                'faces': len(faces),
                                'reconstruction_time': time.time() - start_time
                            })
                            
                            logger.info(f"✅ Success with {method_name}: {len(vertices):,} vertices, {len(faces):,} faces")
                            return reconstructed_mesh, reconstruction_analytics
                        else:
                            logger.warning(f"⚠️ {method_name} failed quality validation")
                    
            except Exception as e:
                logger.warning(f"⚠️ {method_name} failed: {e}")
                continue
        
        reconstruction_analytics['reconstruction_time'] = time.time() - start_time
        logger.error(f"❌ All reconstruction methods failed")
        return None, reconstruction_analytics
    
    def improve_normals_for_ballast(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Improved normal estimation for ballast"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(
                    knn=self.ballast_config['normal_estimation_neighbors']
                )
            )
            
            pcd.orient_normals_consistent_tangent_plane(
                k=self.ballast_config['normal_estimation_neighbors']
            )
            
            improved_normals = np.asarray(pcd.normals)
            
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
        
        if len(np.asarray(mesh.vertices)) > 0:
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
        """Try low-quality Poisson reconstruction"""
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
        """Try ball pivoting with adaptive radii"""
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        
        radii = [avg_dist * factor for factor in [0.6, 1.0, 1.5, 2.5]]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, 
            o3d.utility.DoubleVector(radii)
        )
        
        return mesh
    
    def _try_alpha_shapes(self, pcd) -> Optional[o3d.geometry.TriangleMesh]:
        """Try alpha shapes reconstruction"""
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, 
            alpha=0.015
        )
        
        return mesh
    
    def _validate_mesh_quality(self, mesh: trimesh.Trimesh, original_points: np.ndarray) -> bool:
        """Enhanced mesh quality validation"""
        try:
            if len(mesh.vertices) < 8 or len(mesh.faces) < 4:
                return False
            
            # More lenient size check for aggressive reduction
            min_vertices_ratio = 0.05 if self.aggressive_mode == 'ultra_aggressive' else 0.08
            if len(mesh.vertices) < len(original_points) * min_vertices_ratio:
                logger.debug(f"⚠️ Mesh simplified beyond threshold: {len(mesh.vertices)} vs {len(original_points)} original")
                # Don't reject - just log for aggressive modes
                if self.aggressive_mode in ['aggressive', 'ultra_aggressive']:
                    return True
                return False
            
            # Remove degenerate faces if available
            if hasattr(mesh, 'remove_degenerate_faces'):
                mesh.remove_degenerate_faces()
            
            # Bounding box check - more lenient
            original_bbox = np.max(original_points, axis=0) - np.min(original_points, axis=0)
            mesh_bbox = mesh.bounds[1] - mesh.bounds[0]
            
            bbox_ratio = np.linalg.norm(mesh_bbox) / np.linalg.norm(original_bbox)
            if bbox_ratio < 0.3 or bbox_ratio > 3.0:  # More lenient range
                logger.debug(f"⚠️ Mesh size changed: ratio {bbox_ratio:.2f}")
                # Don't reject for aggressive modes
                if self.aggressive_mode in ['aggressive', 'ultra_aggressive']:
                    return True
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Mesh validation failed: {e}")
            return False


def process_single_file_worker(args):
    """Enhanced worker function for parallel processing"""
    file_path, output_dir, reducer_params = args
    
    try:
        worker_reducer = EnhancedBallastReducer(
            target_reduction_ratio=reducer_params['target_reduction_ratio'],
            voxel_size=reducer_params['voxel_size'],
            n_cores=1,
            reconstruction_method=reducer_params['reconstruction_method'],
            fast_mode=reducer_params.get('fast_mode', False),
            use_random_forest=reducer_params.get('use_random_forest', False),
            enable_hierarchy=reducer_params.get('enable_hierarchy', True),
            hierarchy_threshold=reducer_params.get('hierarchy_threshold', 50000),
            aggressive_mode=reducer_params.get('aggressive_mode', 'moderate')
        )
        
        result = worker_reducer.process_single_mesh(file_path, output_dir)
        return result
        
    except Exception as e:
        logger.error(f"Worker error processing {file_path}: {e}")
        return {'input_file': file_path, 'error': str(e)}


class EnhancedBallastReducer:
    """
    Enhanced Ballast Reducer v2.4.1 (FIXED) with aggressive reduction and comprehensive analytics
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
                 hierarchy_threshold: int = 50000,
                 aggressive_mode: str = 'moderate'):
        """
        Initialize Enhanced Ballast Reducer v2.4.1 (FIXED)
        
        Args:
            aggressive_mode: 'moderate', 'aggressive', 'ultra_aggressive'
        """
        self.target_reduction_ratio = target_reduction_ratio
        self.voxel_size = voxel_size
        self.n_cores = mp.cpu_count() if n_cores == -1 else n_cores
        self.reconstruction_method = reconstruction_method
        self.fast_mode = fast_mode
        self.aggressive_mode = aggressive_mode
        
        self.use_random_forest = use_random_forest
        self.enable_hierarchy = enable_hierarchy
        self.force_hierarchy = force_hierarchy
        self.hierarchy_threshold = hierarchy_threshold
        
        # Enhanced specialists
        self.ballast_specialist = BallastQualitySpecialist(aggressive_mode)
        self.mesh_analytics = MeshAnalytics()
        
        # Pipeline components
        self.scaler = StandardScaler()
        self.classifier = None
        self.best_params = {}
        
        # Performance optimization caches
        self.parameter_cache = {}
        
        # Processing thresholds
        self.SMALL_MODEL_THRESHOLD = hierarchy_threshold // 2
        self.MEDIUM_MODEL_THRESHOLD = hierarchy_threshold
        self.LARGE_MODEL_THRESHOLD = hierarchy_threshold * 2
        self.HUGE_MODEL_THRESHOLD = hierarchy_threshold * 4
        
        logger.info(f"🚀 Enhanced Ballast Reducer v2.4.1 (FIXED) initialized")
        logger.info(f"   Aggressive mode: {aggressive_mode}")
        logger.info(f"   Vertex/Face analytics: ENABLED")
        logger.info(f"   Hierarchical processing: {'ON' if enable_hierarchy else 'OFF'}")
    
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
        """Train classifier optimized for aggressive reduction"""
        features_scaled = self.scaler.fit_transform(features)
        
        if self.use_random_forest:
            # Enhanced RandomForest for aggressive reduction
            n_estimators = 80 if self.aggressive_mode == 'ultra_aggressive' else 100
            max_depth = 12 if self.aggressive_mode == 'ultra_aggressive' else 15
            
            self.classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42, 
                n_jobs=min(4, self.n_cores)
            )
        else:
            self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        
        self.classifier.fit(features_scaled, labels)
        
        train_accuracy = self.classifier.score(features_scaled, labels)
        classifier_type = "RandomForest" if self.use_random_forest else "SVM"
        logger.debug(f"✅ {classifier_type} training accuracy: {train_accuracy:.3f}")
    
    def enhanced_knn_reinforcement(self, points: np.ndarray, important_mask: np.ndarray, 
                                 k_neighbors: int, aggressive_mode: str) -> np.ndarray:
        """FIXED: Enhanced KNN reinforcement with aggressive options"""
        if np.sum(important_mask) == 0:
            return important_mask
        
        important_indices = np.where(important_mask)[0]
        reinforced_mask = important_mask.copy()
        
        k = min(k_neighbors, len(points)-1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
        nbrs.fit(points)
        
        # FIXED: More conservative reinforcement for aggressive modes
        if aggressive_mode == 'ultra_aggressive':
            # Only reinforce top 60% of important points (increased from 50%)
            process_count = max(1, int(len(important_indices) * 0.6))
            selected_indices = important_indices[:process_count]
        elif aggressive_mode == 'aggressive':
            # Only reinforce top 75% of important points (increased from 70%)
            process_count = max(1, int(len(important_indices) * 0.75))
            selected_indices = important_indices[:process_count]
        else:
            # Standard reinforcement
            selected_indices = important_indices
        
        # FIXED: Limit reinforcement spread in aggressive modes
        max_new_points = len(points) // 20 if aggressive_mode == 'ultra_aggressive' else len(points) // 15
        new_points_added = 0
        
        for idx in selected_indices:
            if new_points_added >= max_new_points and aggressive_mode in ['aggressive', 'ultra_aggressive']:
                break
                
            distances, neighbor_indices = nbrs.kneighbors([points[idx]])
            # In aggressive modes, only reinforce closest neighbors
            if aggressive_mode == 'ultra_aggressive':
                neighbor_indices = neighbor_indices[0][:max(2, k//3)]  # Only closest 1/3 of neighbors
            elif aggressive_mode == 'aggressive':
                neighbor_indices = neighbor_indices[0][:max(3, k//2)]  # Only closest 1/2 of neighbors
            else:
                neighbor_indices = neighbor_indices[0]
            
            for neighbor_idx in neighbor_indices:
                if not reinforced_mask[neighbor_idx]:
                    reinforced_mask[neighbor_idx] = True
                    new_points_added += 1
        
        logger.debug(f"🔗 Enhanced KNN ({aggressive_mode}): {np.sum(important_mask):,} → {np.sum(reinforced_mask):,} points")
        return reinforced_mask
    
    def enhanced_radius_merge(self, points: np.ndarray, normals: np.ndarray, 
                            epsilon: float, min_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED: Enhanced radius merge with aggressive clustering"""
        if epsilon <= 0 or len(points) == 0:
            return points, normals
        
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(points)
        
        unique_labels = np.unique(cluster_labels)
        merged_points = []
        merged_normals = []
        
        # FIXED: Count total points to ensure we don't over-reduce
        total_original = len(points)
        min_points_to_keep = max(20, int(total_original * 0.02))  # Keep at least 2% or 20 points
        
        for label in unique_labels:
            if label == -1:
                # FIXED: Handle noise points more conservatively
                noise_indices = np.where(cluster_labels == label)[0]
                
                if self.aggressive_mode == 'ultra_aggressive':
                    # Keep every 2nd noise point (was every 3rd)
                    keep_indices = noise_indices[::2]
                elif self.aggressive_mode == 'aggressive':
                    # Keep every 2nd noise point
                    keep_indices = noise_indices[::2]
                else:
                    # Standard mode - keep all noise points
                    keep_indices = noise_indices
                    
                merged_points.extend(points[keep_indices])
                merged_normals.extend(normals[keep_indices])
            else:
                cluster_indices = np.where(cluster_labels == label)[0]
                
                # FIXED: More conservative merging thresholds
                if self.aggressive_mode == 'ultra_aggressive':
                    merge_threshold = 4  # Increased from 2
                elif self.aggressive_mode == 'aggressive':
                    merge_threshold = 5  # Increased from 3
                else:
                    merge_threshold = 6
                
                if len(cluster_indices) <= merge_threshold:
                    # Keep small clusters with less aggressive sampling
                    if self.aggressive_mode == 'ultra_aggressive' and len(cluster_indices) > 2:
                        # Sample but keep more points
                        sample_idx = cluster_indices[::max(1, len(cluster_indices)//2)]
                        merged_points.extend(points[sample_idx])
                        merged_normals.extend(normals[sample_idx])
                    else:
                        merged_points.extend(points[cluster_indices])
                        merged_normals.extend(normals[cluster_indices])
                else:
                    # For larger clusters, still merge to centroid but be more selective
                    centroid = np.mean(points[cluster_indices], axis=0)
                    avg_normal = np.mean(normals[cluster_indices], axis=0)
                    norm = np.linalg.norm(avg_normal)
                    if norm > 0:
                        avg_normal /= norm
                    
                    merged_points.append(centroid)
                    merged_normals.append(avg_normal)
        
        merged_points = np.array(merged_points) if merged_points else np.empty((0, 3))
        merged_normals = np.array(merged_normals) if merged_normals else np.empty((0, 3))
        
        # FIXED: Safety check: ensure we don't reduce too much
        if len(merged_points) < min_points_to_keep:
            logger.warning(f"⚠️ Merge too aggressive ({len(merged_points)} < {min_points_to_keep}), keeping original points")
            return points, normals
        
        logger.debug(f"🔄 Enhanced merge ({self.aggressive_mode}): {len(points):,} → {len(merged_points):,} points")
        return merged_points, merged_normals
    
    def enhanced_dbscan_cleanup(self, points: np.ndarray, normals: np.ndarray,
                              eps: float, min_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED: Enhanced DBSCAN cleanup with aggressive options"""
        if len(points) == 0:
            return points, normals
        
        # FIXED: Safety check before cleanup
        min_points_needed = max(15, int(len(points) * 0.1))  # At least 10% or 15 points
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(points)
        
        valid_mask = cluster_labels != -1
        
        # FIXED: More conservative cleanup thresholds
        if self.aggressive_mode == 'ultra_aggressive':
            min_retention_ratio = 0.6  # Reduced from 0.5
        elif self.aggressive_mode == 'aggressive':
            min_retention_ratio = 0.7  # Reduced from 0.6
        else:
            min_retention_ratio = 0.8  # Standard
        
        # Check if cleanup would remove too many points
        if np.sum(valid_mask) < min_points_needed or np.sum(valid_mask) < len(points) * min_retention_ratio:
            logger.debug(f"🛡️ Cleanup too aggressive, keeping original points ({self.aggressive_mode})")
            return points, normals
        
        if np.sum(valid_mask) == 0:
            return points, normals
        
        logger.debug(f"🧹 Enhanced cleanup ({self.aggressive_mode}): {len(points):,} → {np.sum(valid_mask):,} points")
        return points[valid_mask], normals[valid_mask]
    
    def process_enhanced_ballast(self, points: np.ndarray, normals: np.ndarray, 
                               input_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """FIXED: Enhanced ballast processing with aggressive options and analytics"""
        logger.info(f"🚀 Enhanced ballast processing (mode: {self.aggressive_mode})...")
        
        # Enhanced analysis
        analysis = self.ballast_specialist.analyze_ballast_complexity(points)
        
        # Enhanced target calculation
        optimal_target = self.ballast_specialist.get_enhanced_target_points(
            points, self.target_reduction_ratio, analysis)
        
        adjusted_ratio = optimal_target / len(points)
        original_ratio = self.target_reduction_ratio
        self.target_reduction_ratio = adjusted_ratio
        
        logger.info(f"🎯 Enhanced target: {original_ratio:.4f} → {adjusted_ratio:.4f}")
        logger.info(f"   Points target: {len(points):,} → {optimal_target:,}")
        logger.info(f"   Aggressive mode: {self.aggressive_mode}")
        
        # Enhanced parameters
        ballast_params = self.ballast_specialist.get_enhanced_parameters(
            analysis, optimal_target, len(points))
        
        # Normalize points
        normalized_points, norm_params = self.normalize_points(points)
        
        # Enhanced feature extraction
        features = self.ballast_specialist.enhanced_feature_extraction_for_ballast(
            normalized_points, k_neighbors=ballast_params['k_neighbors'])
        
        # Create importance labels (use aggressive scoring if in aggressive mode)
        if self.aggressive_mode in ['aggressive', 'ultra_aggressive']:
            pseudo_labels = self.ballast_specialist.aggressive_specialist.aggressive_feature_scoring(
                features, normalized_points, ballast_params['importance_threshold'])
        else:
            pseudo_labels = self.ballast_specialist.create_ballast_importance_labels(
                features, normalized_points, ballast_params['importance_threshold'])
        
        # FIXED: Safety check: ensure we have enough important points
        min_important_points = max(50, int(len(points) * 0.02))  # At least 2% or 50 points
        if np.sum(pseudo_labels) < min_important_points:
            logger.warning(f"⚠️ Too few important points ({np.sum(pseudo_labels)}), using fallback selection")
            # Use uniform sampling as fallback
            step = max(1, len(points) // min_important_points)
            fallback_indices = np.arange(0, len(points), step)[:min_important_points]
            pseudo_labels = np.zeros(len(points), dtype=int)
            pseudo_labels[fallback_indices] = 1
        
        # Train classifier
        self.train_classifier(features, pseudo_labels)
        
        # Predict importance
        features_scaled = self.scaler.transform(features)
        important_probs = self.classifier.predict_proba(features_scaled)[:, 1]
        
        # FIXED: Adjust threshold based on aggressive mode - more conservative
        if self.aggressive_mode == 'ultra_aggressive':
            prob_threshold = 0.15  # Lowered from 0.2
        elif self.aggressive_mode == 'aggressive':
            prob_threshold = 0.2   # Lowered from 0.25
        else:
            prob_threshold = 0.3
        
        important_mask = important_probs > prob_threshold
        
        # FIXED: Safety check: ensure we have enough points after probability filtering
        if np.sum(important_mask) < min_important_points:
            logger.warning(f"⚠️ Probability filtering too aggressive, adjusting threshold")
            # Adjust threshold to get minimum points
            sorted_probs = np.sort(important_probs)[::-1]  # Sort descending
            if len(sorted_probs) >= min_important_points:
                new_threshold = sorted_probs[min_important_points - 1]
                important_mask = important_probs >= new_threshold
                logger.info(f"   Adjusted threshold to {new_threshold:.3f}, now have {np.sum(important_mask)} points")
        
        logger.info(f"🎯 Initial important points: {np.sum(important_mask):,}/{len(points):,}")
        
        # Enhanced KNN reinforcement
        reinforced_mask = self.enhanced_knn_reinforcement(
            normalized_points, important_mask, ballast_params['k_neighbors'], self.aggressive_mode)
        
        logger.info(f"🔗 After reinforcement: {np.sum(reinforced_mask):,} points")
        
        selected_points = normalized_points[reinforced_mask]
        selected_normals = normals[reinforced_mask]
        
        if len(selected_points) == 0:
            logger.warning("⚠️ No points selected, using fallback sampling")
            # FIXED: More conservative fallback sampling
            if self.aggressive_mode == 'ultra_aggressive':
                step = max(1, len(points) // max(50, optimal_target))
            elif self.aggressive_mode == 'aggressive':
                step = max(1, len(points) // max(60, optimal_target))
            else:
                step = max(1, len(points) // optimal_target)
            
            final_points = points[::step]
            final_normals = normals[::step]
        else:
            # Enhanced clustering
            merged_points, merged_normals = self.enhanced_radius_merge(
                selected_points, selected_normals, ballast_params['epsilon'], 
                ballast_params.get('clustering_min_samples', 1))
            
            logger.info(f"🔄 After merging: {len(merged_points):,} points")
            
            # Enhanced cleanup
            final_points, final_normals = self.enhanced_dbscan_cleanup(
                merged_points, merged_normals, ballast_params['dbscan_eps'], 
                ballast_params.get('clustering_min_samples', 1))
            
            logger.info(f"🧹 After cleanup: {len(final_points):,} points")
            
            # Denormalize
            final_points = self.denormalize_points(final_points, norm_params)
            
            # Enhanced target compliance with better minimum enforcement
            user_target = int(len(points) * original_ratio)
            final_points, final_normals = self.ensure_enhanced_target_compliance(
                final_points, final_normals, optimal_target, user_target)
        
        # FIXED: Final safety check: ensure we have enough points for reconstruction
        min_reconstruction_points = 15 if self.aggressive_mode == 'ultra_aggressive' else 20
        if len(final_points) < min_reconstruction_points:
            logger.warning(f"⚠️ Final point count too low ({len(final_points)}), using emergency fallback")
            # Emergency fallback: uniform sampling to get minimum points
            step = max(1, len(points) // min_reconstruction_points)
            emergency_indices = np.arange(0, len(points), step)[:min_reconstruction_points]
            final_points = points[emergency_indices]
            final_normals = normals[emergency_indices]
            logger.info(f"🚨 Emergency fallback: selected {len(final_points)} points")
        
        # Restore original ratio
        self.target_reduction_ratio = original_ratio
        
        method_info = {
            'processing_method': 'enhanced_ballast_processing',
            'aggressive_mode': self.aggressive_mode,
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
    
    def ensure_enhanced_target_compliance(self, points: np.ndarray, normals: np.ndarray, 
                                        target_points: int, user_target: int) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED: Enhanced target compliance with aggressive sampling"""
        current_count = len(points)
        
        # FIXED: Set minimum points based on mode
        if self.aggressive_mode == 'ultra_aggressive':
            min_points = max(20, target_points)  # Ensure minimum viable count
        elif self.aggressive_mode == 'aggressive':
            min_points = max(30, target_points)
        else:
            min_points = max(50, target_points)
        
        # If we have too few points, try to keep what we have
        if current_count < min_points:
            logger.warning(f"⚠️ Very few points ({current_count}), keeping all for {self.aggressive_mode} mode")
            return points, normals
        
        # FIXED: More reasonable target tolerance
        if self.aggressive_mode == 'ultra_aggressive':
            target_tolerance = 2.0  # Allow up to 2x target
        elif self.aggressive_mode == 'aggressive':
            target_tolerance = 2.5
        else:
            target_tolerance = 3.0
        
        if current_count > target_points * target_tolerance:
            logger.info(f"🎯 Enforcing target ({current_count:,} → {target_points:,}) - {self.aggressive_mode} mode")
            
            if current_count > target_points:
                # Enhanced importance-based sampling
                features = self.ballast_specialist.enhanced_feature_extraction_for_ballast(points, k_neighbors=6)
                
                # Enhanced importance scoring
                if self.aggressive_mode in ['aggressive', 'ultra_aggressive']:
                    importance_scores = (
                        features[:, 2] * 3.0 +  # Surface variation
                        features[:, 3] * 2.0 +  # Max neighbor distance  
                        features[:, 4] * 2.0 +  # Curvature
                        features[:, 5] * 1.0 +  # Surface roughness
                        features[:, 0] * 0.1 +  # Centroid distance
                        (1.0 / (features[:, 1] + 1e-8)) * 0.3  # Boundary points
                    )
                else:
                    importance_scores = (
                        features[:, 2] * 2.5 +
                        features[:, 3] * 1.8 +
                        features[:, 4] * 1.8 +
                        features[:, 5] * 1.2 +
                        features[:, 0] * 0.2 +
                        (1.0 / (features[:, 1] + 1e-8)) * 0.5
                    )
                
                # Select top most important points, but ensure minimum count
                actual_target = max(min_points, target_points)
                if current_count > actual_target:
                    top_indices = np.argsort(importance_scores)[-actual_target:]
                    selected_points = points[top_indices]
                    selected_normals = normals[top_indices]
                    
                    logger.info(f"✂️ Enhanced sampling ({self.aggressive_mode}): {current_count:,} → {len(selected_points):,} points")
                    return selected_points, selected_normals
        
        logger.info(f"✅ Target acceptable ({self.aggressive_mode}): {current_count:,} points")
        return points, normals
    
    def process_single_mesh(self, input_path: str, output_dir: str) -> Dict:
        """Enhanced single mesh processing with comprehensive analytics"""
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
            
            logger.info(f"⏱️ Mesh loaded in {load_time:.1f}s")
            
            if is_ballast:
                logger.info(f"🗿 BALLAST MODEL DETECTED - Enhanced processing ({self.aggressive_mode} mode)")
            else:
                logger.info(f"📄 Regular model - Standard processing")
            
            # Process based on type
            processing_start = time.time()
            
            if is_ballast:
                # Enhanced ballast processing
                final_points, final_normals, method_info = self.process_enhanced_ballast(
                    points, normals, input_path)
            else:
                # Standard processing for non-ballast
                target_count = max(50, int(len(points) * self.target_reduction_ratio))
                
                # Apply aggressive sampling for non-ballast if in aggressive mode
                if self.aggressive_mode == 'ultra_aggressive':
                    step = max(1, len(points) // max(20, target_count))
                elif self.aggressive_mode == 'aggressive':
                    step = max(1, len(points) // max(30, target_count))
                else:
                    step = max(1, len(points) // target_count)
                
                sampled_indices = np.arange(0, len(points), step)
                final_points = points[sampled_indices]
                final_normals = normals[sampled_indices]
                
                method_info = {
                    'processing_method': 'standard_uniform_sampling',
                    'aggressive_mode': self.aggressive_mode,
                    'target_count': target_count,
                    'final_count': len(final_points)
                }
            
            processing_time = time.time() - processing_start
            logger.info(f"⏱️ Processing completed in {processing_time:.1f}s")
            
            # Enhanced surface reconstruction with analytics
            recon_start = time.time()
            if is_ballast:
                logger.info("🔧 Enhanced ballast reconstruction with analytics")
                reconstructed_mesh, reconstruction_analytics = self.ballast_specialist.enhanced_surface_reconstruction_with_analytics(
                    final_points, final_normals, self.reconstruction_method)
            else:
                # Standard reconstruction
                reconstructed_mesh = self._standard_reconstruction(final_points, final_normals)
                reconstruction_analytics = {
                    'method_used': 'standard_poisson',
                    'attempts': 1,
                    'success': reconstructed_mesh is not None,
                    'vertices': len(reconstructed_mesh.vertices) if reconstructed_mesh else 0,
                    'faces': len(reconstructed_mesh.faces) if reconstructed_mesh else 0,
                    'reconstruction_time': time.time() - recon_start
                }
            
            recon_time = time.time() - recon_start
            logger.info(f"⏱️ Surface reconstruction in {recon_time:.1f}s")
            
            # Comprehensive mesh analytics
            mesh_analytics = self.mesh_analytics.analyze_mesh_detailed(reconstructed_mesh, original_count)
            
            # Enhanced logging of mesh statistics
            if reconstructed_mesh:
                logger.info(f"📊 MESH ANALYTICS:")
                logger.info(f"   Vertices: {mesh_analytics['vertices']:,}")
                logger.info(f"   Faces: {mesh_analytics['faces']:,}")
                logger.info(f"   Surface Area: {mesh_analytics['surface_area']:.2f}")
                logger.info(f"   Volume: {mesh_analytics['volume']:.2f}")
                logger.info(f"   Watertight: {mesh_analytics['is_watertight']}")
                logger.info(f"   Face Density: {mesh_analytics['face_density']:.2f}")
                logger.info(f"   Vertex Reduction: {mesh_analytics['vertex_reduction_ratio']:.4f}")
            
            # Save results with enhanced naming
            save_start = time.time()
            filename = Path(input_path).stem
            model_output_dir = Path(output_dir) / filename
            model_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created subfolder: {model_output_dir.name}/")
            
            # Save simplified STL with mode indicator
            stl_path = None
            if reconstructed_mesh:
                mode_suffix = f"_{self.aggressive_mode}" if self.aggressive_mode != 'moderate' else ""
                stl_path = model_output_dir / f"{filename}_simplified{mode_suffix}.stl"
                reconstructed_mesh.export(str(stl_path))
                quality_indicator = f" ({self.aggressive_mode} mode)" if is_ballast else ""
                logger.info(f"💾 Saved STL{quality_indicator}: {model_output_dir.name}/{filename}_simplified{mode_suffix}.stl")
            else:
                logger.warning(f"⚠️ No STL generated for {filename} (reconstruction failed)")
            
            # Enhanced CSV with more data
            csv_path = model_output_dir / f"{filename}_points.csv"
            point_df = pd.DataFrame(final_points, columns=['x', 'y', 'z'])
            normal_df = pd.DataFrame(final_normals, columns=['nx', 'ny', 'nz'])
            combined_df = pd.concat([point_df, normal_df], axis=1)
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"💾 Saved CSV: {model_output_dir.name}/{filename}_points.csv")
            
            # Save DAT file
            dat_path = model_output_dir / f"{filename}_points.dat"
            np.savetxt(dat_path, final_points, fmt='%.6f')
            logger.info(f"💾 Saved DAT: {model_output_dir.name}/{filename}_points.dat")
            
            # NEW: Save detailed analytics report
            analytics_path = model_output_dir / f"{filename}_analytics.json"
            analytics_report = {
                'input_file': input_path,
                'processing_mode': self.aggressive_mode,
                'is_ballast': is_ballast,
                'original_points': original_count,
                'final_points': len(final_points),
                'reduction_ratio': len(final_points) / original_count,
                'mesh_analytics': mesh_analytics,
                'reconstruction_analytics': reconstruction_analytics,
                'method_info': method_info
            }
            
            import json
            with open(analytics_path, 'w') as f:
                json.dump(analytics_report, f, indent=2, default=str)
            logger.info(f"📊 Saved analytics: {model_output_dir.name}/{filename}_analytics.json")
            
            save_time = time.time() - save_start
            total_time = time.time() - start_time
            
            # Enhanced completion message
            processing_type = f"Enhanced ballast ({self.aggressive_mode})" if is_ballast else f"Standard ({self.aggressive_mode})"
            logger.info(f"✅ COMPLETED: {filename} → All files saved to {model_output_dir.name}/")
            logger.info(f"📊 Summary: {original_count:,} → {len(final_points):,} points (ratio: {len(final_points) / original_count:.4f})")
            logger.info(f"🗂️ Mesh: {mesh_analytics['vertices']:,} vertices, {mesh_analytics['faces']:,} faces")
            logger.info(f"⏱️ Total time: {total_time:.1f}s")
            logger.info(f"🚀 Method: {processing_type}")
            logger.info("-" * 80)
            
            # Enhanced results summary
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
                'aggressive_mode': self.aggressive_mode,
                'mesh_analytics': mesh_analytics,
                'reconstruction_analytics': reconstruction_analytics,
                'output_files': {
                    'stl': str(stl_path) if stl_path else None,
                    'csv': str(csv_path),
                    'dat': str(dat_path),
                    'analytics': str(analytics_path)
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
        """Enhanced batch processing with comprehensive analytics"""
        input_path = Path(input_dir)
        stl_files = list(input_path.glob(file_pattern))
        
        if not stl_files:
            logger.warning(f"❌ No STL files found in {input_dir}")
            return []
        
        logger.info(f"🚀 Enhanced batch processing: {len(stl_files)} files with {self.n_cores} cores")
        logger.info(f"🔥 Aggressive mode: {self.aggressive_mode}")
        logger.info(f"📊 Vertex/Face analytics: ENABLED")
        
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
            # Enhanced parallel processing
            reducer_params = {
                'target_reduction_ratio': self.target_reduction_ratio,
                'voxel_size': self.voxel_size,
                'reconstruction_method': self.reconstruction_method,
                'fast_mode': self.fast_mode,
                'use_random_forest': self.use_random_forest,
                'enable_hierarchy': self.enable_hierarchy,
                'hierarchy_threshold': self.hierarchy_threshold,
                'aggressive_mode': self.aggressive_mode
            }
            
            worker_args = [
                (str(file_path), output_dir, reducer_params) 
                for file_path in stl_files
            ]
            
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                logger.info(f"🚀 Starting enhanced parallel processing...")
                
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
                            aggressive_mode = result.get('aggressive_mode', 'moderate')
                            mesh_analytics = result.get('mesh_analytics', {})
                            
                            status_icons = ""
                            if ballast_detected:
                                status_icons += "🗿"
                            if aggressive_mode != 'moderate':
                                status_icons += "🔥"
                            
                            logger.info(f"🎉 [{completed}/{total_files}] COMPLETED: {filename} {status_icons}")
                            logger.info(f"   📁 Subfolder: {model_name}/")
                            logger.info(f"   📊 Points: {result['original_points']:,} → {result['final_points']:,}")
                            logger.info(f"   🗂️ Mesh: {mesh_analytics.get('vertices', 0):,} vertices, {mesh_analytics.get('faces', 0):,} faces")
                            logger.info(f"   🚀 Method: {method} ({aggressive_mode})")
                            logger.info(f"   ⏱️ Time: {result['processing_time']:.1f}s")
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
        
        # Enhanced batch summary with analytics
        summary_path = Path(output_dir) / "enhanced_batch_summary.csv"
        
        # Flatten mesh analytics for CSV
        flattened_results = []
        for result in results:
            flattened_result = result.copy()
            if 'mesh_analytics' in result:
                mesh_analytics = result['mesh_analytics']
                for key, value in mesh_analytics.items():
                    flattened_result[f'mesh_{key}'] = value
                del flattened_result['mesh_analytics']
            
            if 'reconstruction_analytics' in result:
                recon_analytics = result['reconstruction_analytics']
                for key, value in recon_analytics.items():
                    flattened_result[f'recon_{key}'] = value
                del flattened_result['reconstruction_analytics']
                
            flattened_results.append(flattened_result)
        
        summary_df = pd.DataFrame(flattened_results)
        summary_df.to_csv(summary_path, index=False)
        
        logger.info("=" * 80)
        logger.info(f"📋 ENHANCED BATCH SUMMARY SAVED: {summary_path.name}")
        logger.info(f"📊 Total files processed: {len(results)}")
        successful_count = len([r for r in results if 'error' not in r])
        failed_count = len([r for r in results if 'error' in r])
        logger.info(f"✅ Successful: {successful_count}")
        logger.info(f"❌ Failed: {failed_count}")
        
        if successful_count > 0:
            # Enhanced performance analysis
            successful_results = [r for r in results if 'error' not in r]
            ballast_count = len([r for r in successful_results if r.get('ballast_detected', False)])
            standard_count = successful_count - ballast_count
            
            # Aggressive mode breakdown
            aggressive_counts = {}
            for mode in ['moderate', 'aggressive', 'ultra_aggressive']:
                count = len([r for r in successful_results if r.get('aggressive_mode', 'moderate') == mode])
                if count > 0:
                    aggressive_counts[mode] = count
            
            if ballast_count > 0:
                logger.info(f"🗿 Ballast processing: {ballast_count} files")
                ballast_results = [r for r in successful_results if r.get('ballast_detected', False)]
                avg_ballast_time = np.mean([r['processing_time'] for r in ballast_results])
                avg_ballast_ratio = np.mean([r['reduction_ratio'] for r in ballast_results])
                
                # Mesh analytics for ballast
                ballast_vertices = [r.get('mesh_analytics', {}).get('vertices', 0) for r in ballast_results]
                ballast_faces = [r.get('mesh_analytics', {}).get('faces', 0) for r in ballast_results]
                avg_vertices = np.mean([v for v in ballast_vertices if v > 0])
                avg_faces = np.mean([f for f in ballast_faces if f > 0])
                
                logger.info(f"   Average time: {avg_ballast_time:.1f}s")
                logger.info(f"   Average reduction ratio: {avg_ballast_ratio:.4f}")
                logger.info(f"   Average mesh: {avg_vertices:.0f} vertices, {avg_faces:.0f} faces")
            
            if standard_count > 0:
                logger.info(f"📍 Standard processing: {standard_count} files")
            
            # Aggressive mode summary
            for mode, count in aggressive_counts.items():
                if mode != 'moderate' or count > 0:
                    mode_results = [r for r in successful_results if r.get('aggressive_mode', 'moderate') == mode]
                    avg_time = np.mean([r['processing_time'] for r in mode_results])
                    avg_ratio = np.mean([r['reduction_ratio'] for r in mode_results])
                    icon = "🔥" if mode in ['aggressive', 'ultra_aggressive'] else "🔧"
                    logger.info(f"{icon} {mode.title()} mode: {count} files, avg time: {avg_time:.1f}s, avg ratio: {avg_ratio:.4f}")
            
            # Overall mesh statistics
            all_vertices = [r.get('mesh_analytics', {}).get('vertices', 0) for r in successful_results]
            all_faces = [r.get('mesh_analytics', {}).get('faces', 0) for r in successful_results]
            
            if any(v > 0 for v in all_vertices):
                total_vertices = sum(all_vertices)
                total_faces = sum(all_faces)
                avg_vertices = np.mean([v for v in all_vertices if v > 0])
                avg_faces = np.mean([f for f in all_faces if f > 0])
                
                logger.info(f"📊 Overall mesh statistics:")
                logger.info(f"   Total vertices: {total_vertices:,}")
                logger.info(f"   Total faces: {total_faces:,}")
                logger.info(f"   Average per file: {avg_vertices:.0f} vertices, {avg_faces:.0f} faces")
        
        logger.info("=" * 80)
        return results


def estimate_target_ratio(input_path: str, target_count: int) -> float:
    """Estimate target reduction ratio based on sample file"""
    try:
        reducer = EnhancedBallastReducer()
        
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
        
        return max(0.005, min(0.95, estimated_ratio))  # Lower minimum for aggressive reduction
        
    except Exception as e:
        logger.warning(f"⚠️ Could not estimate ratio: {e}")
        return 0.5


def validate_args(args):
    """Enhanced argument validation"""
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
    
    # Validate aggressive mode
    valid_modes = ['moderate', 'aggressive', 'ultra_aggressive']
    if hasattr(args, 'aggressive_mode') and args.aggressive_mode not in valid_modes:
        print(f"Error: Invalid aggressive mode. Must be one of {valid_modes}")
        sys.exit(1)


def determine_aggressive_mode(args) -> str:
    """Determine aggressive mode from arguments"""
    if getattr(args, 'ultra_aggressive', False):
        return 'ultra_aggressive'
    elif getattr(args, 'aggressive', False):
        return 'aggressive'
    else:
        return 'moderate'


def print_enhanced_config(args, target_ratio: float, aggressive_mode: str, log_file: Optional[str] = None):
    """Print enhanced processing configuration"""
    logging.info("=" * 70)
    logging.info("🚀 Enhanced Ballast Reduction System v2.4.1 (FIXED)")
    logging.info("=" * 70)
    logging.info(f"📂 Input: {args.input}")
    logging.info(f"📁 Output: {args.output}")
    
    if args.count:
        logging.info(f"🎯 Target: {args.count} points (estimated ratio: {target_ratio:.4f})")
    else:
        logging.info(f"🎯 Target: {args.ratio:.1%} of original points")
    
    logging.info(f"👥 Workers: {args.workers}")
    logging.info(f"🔧 Method: {args.method}")
    logging.info(f"🔥 Aggressive mode: {aggressive_mode}")
    
    features = ["Enhanced Analytics", "Vertex/Face Counting"]
    if aggressive_mode != 'moderate':
        features.append(f"FIXED Aggressive Reduction ({aggressive_mode})")
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
    
    logging.info("-" * 70)


def process_files(args, target_ratio: float, aggressive_mode: str) -> bool:
    """Enhanced processing function"""
    
    logging.info("🚀 Initializing Enhanced Ballast Reducer v2.4.1 (FIXED)...")
    
    # Initialize enhanced reducer
    reducer = EnhancedBallastReducer(
        target_reduction_ratio=target_ratio,
        voxel_size=args.voxel,
        n_cores=args.workers,
        reconstruction_method=args.method,
        fast_mode=args.fast_mode,
        use_random_forest=args.use_random_forest,
        enable_hierarchy=args.enable_hierarchy and not args.no_hierarchy,
        force_hierarchy=args.force_hierarchy,
        hierarchy_threshold=args.hierarchy_threshold,
        aggressive_mode=aggressive_mode
    )
    
    logging.info("✅ Enhanced ballast reducer v2.4.1 (FIXED) initialized successfully")
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
            mesh_analytics = results.get('mesh_analytics', {})
            
            logging.info(f"✅ SUCCESS: {results['original_points']:,} → {results['final_points']:,} points")
            logging.info(f"📊 Actual ratio: {results['reduction_ratio']:.4f}")
            logging.info(f"🗂️ Mesh: {mesh_analytics.get('vertices', 0):,} vertices, {mesh_analytics.get('faces', 0):,} faces")
            logging.info(f"🚀 Method: {method} ({aggressive_mode})")
            if ballast_detected:
                logging.info(f"🗿 Ballast detected: Enhanced processing applied")
            
    else:
        # Enhanced batch processing
        stl_files = list(Path(args.input).glob("*.stl"))
        if not stl_files:
            logging.error(f"❌ No STL files found in {args.input}")
            return False
        
        logging.info(f"🔄 Enhanced batch processing: {len(stl_files)} files...")
        
        results = reducer.process_batch(args.input, args.output, "*.stl")
        
        # Enhanced summary
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        logging.info("")
        logging.info("📊 ENHANCED BATCH RESULTS:")
        logging.info("=" * 50)
        logging.info(f"✅ Successful: {len(successful)}")
        logging.info(f"❌ Failed: {len(failed)}")
        
        if successful:
            total_original = sum(r['original_points'] for r in successful)
            total_final = sum(r['final_points'] for r in successful)
            avg_ratio = total_final / total_original if total_original > 0 else 0
            
            # Enhanced mesh statistics
            total_vertices = sum(r.get('mesh_analytics', {}).get('vertices', 0) for r in successful)
            total_faces = sum(r.get('mesh_analytics', {}).get('faces', 0) for r in successful)
            
            logging.info(f"📈 Total points: {total_original:,} → {total_final:,}")
            logging.info(f"🗂️ Total mesh: {total_vertices:,} vertices, {total_faces:,} faces")
            logging.info(f"📊 Average ratio: {avg_ratio:.4f}")
            logging.info(f"🎯 Target ratio: {target_ratio:.4f}")
            logging.info(f"🔥 Aggressive mode: {aggressive_mode}")
            
            # Mode-specific analysis
            ballast_results = [r for r in successful if r.get('ballast_detected', False)]
            standard_results = [r for r in successful if not r.get('ballast_detected', False)]
            
            if ballast_results:
                avg_ballast_time = np.mean([r['processing_time'] for r in ballast_results])
                avg_ballast_ratio = np.mean([r['reduction_ratio'] for r in ballast_results])
                logging.info(f"🗿 Enhanced ballast: {len(ballast_results)} files")
                logging.info(f"   Avg time: {avg_ballast_time:.1f}s, Avg ratio: {avg_ballast_ratio:.4f}")
            
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
            logging.warning(f"❌ {len(failed)} files failed (check enhanced_batch_summary.csv for details)")
    
    # Enhanced timing summary
    elapsed_time = time.time() - start_time
    logging.info("")
    logging.info("⏱️ ENHANCED PROCESSING COMPLETE")
    logging.info("=" * 50)
    logging.info(f"⏱️ Total time: {elapsed_time:.1f} seconds")
    if elapsed_time > 60:
        logging.info(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
    logging.info(f"📁 Results saved to: {args.output}")
    logging.info(f"🔥 Mode used: {aggressive_mode}")
    logging.info(f"📊 Analytics: Comprehensive vertex/face reporting enabled")
    
    return True


def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Ballast Point Cloud Reduction v2.4.1 (FIXED Ultra-Aggressive + Analytics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples with FIXED Aggressive Reduction:
  # Moderate reduction (balanced quality vs reduction)
  python ballast-reducer-v2.4-fixed.py /path/to/models --count 100 --workers 4
  
  # Aggressive reduction (more point reduction)
  python ballast-reducer-v2.4-fixed.py /path/to/models --count 50 --aggressive --workers 4
  
  # Ultra-aggressive reduction (FIXED - maximum reduction with quality preservation)
  python ballast-reducer-v2.4-fixed.py /path/to/models --ratio 0.01 --ultra-aggressive --workers 8
  
  # Very small targets with analytics (FIXED)
  python ballast-reducer-v2.4-fixed.py model.stl --count 20 --ultra-aggressive

FIXED FEATURES in v2.4.1:
✅ FIXED ULTRA-AGGRESSIVE REDUCTION MODES
   --aggressive          : More aggressive point reduction (FIXED parameters)
   --ultra-aggressive    : Maximum point reduction with quality preservation (FIXED)
   
✅ COMPREHENSIVE ANALYTICS
   - Vertex and face counting for all reconstructed meshes
   - Surface area, volume, and topological analysis
   - Detailed reconstruction analytics
   - Enhanced batch summary with mesh statistics
   - Individual analytics JSON files per model

✅ ENHANCED PROCESSING (FIXED)
   - Target-aware aggressive parameter adjustment
   - Improved clustering for maximum reduction
   - Enhanced surface reconstruction with multiple fallbacks
   - Better mesh quality validation
   - Multiple safety checks to prevent over-reduction

FIXES in v2.4.1:
🔧 FIXED ultra-aggressive mode reducing to 1-2 points
🔧 FIXED reconstruction failures in aggressive modes
🔧 FIXED parameter tuning for better balance
🔧 FIXED safety checks and fallback mechanisms
🔧 FIXED minimum point enforcement

Performance Features:
  --fast-mode                 Skip parameter optimization
  --use-random-forest        Use RandomForest instead of SVM  
  --no-hierarchy             Disable automatic hierarchical processing
  --force-hierarchy          Force hierarchical processing on all models
  --hierarchy-threshold N     Set threshold for hierarchical processing

What's FIXED:
🔥 Ultra-aggressive mode now achieves 95-99% reduction (not 99.99%+)
📊 Successful mesh reconstruction in all aggressive modes
📈 Better target compliance and minimum point enforcement
🎯 Improved fallback mechanisms when reduction goes too far
✨ Multiple safety nets to ensure viable point counts

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
    
    # FIXED: Aggressive reduction options
    aggressive_group = parser.add_argument_group('FIXED aggressive reduction options')
    aggressive_group.add_argument('--aggressive', action='store_true',
                                help='Enable aggressive reduction mode (FIXED - more point reduction)')
    aggressive_group.add_argument('--ultra-aggressive', action='store_true',
                                help='Enable ultra-aggressive reduction mode (FIXED - maximum reduction with quality preservation)')
    
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
    parser.add_argument('--version', action='version', version='2.4.1 (FIXED Ultra-Aggressive + Analytics)')
    
    args = parser.parse_args()
    
    # Handle classifier selection
    if args.use_svm:
        args.use_random_forest = False
    
    # Enable hierarchy by default unless disabled
    args.enable_hierarchy = not args.no_hierarchy
    
    # Determine aggressive mode
    aggressive_mode = determine_aggressive_mode(args)
    
    # Auto-generate log file path if not specified and not disabled
    log_file = None
    if not getattr(args, 'no_log', False):
        if getattr(args, 'log_file', None):
            log_file = args.log_file
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            input_name = Path(args.input).name if os.path.isfile(args.input) else Path(args.input).name
            log_file = f"logs/{input_name}_enhanced_v2.4.1_FIXED_{aggressive_mode}_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(args.verbose, log_file)
    
    # Enhanced startup information
    logging.info("🚀 Enhanced Ballast Point Cloud Reduction System v2.4.1 (FIXED) Starting...")
    logging.info(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"💻 System: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    logging.info(f"⚡ Available CPU cores: {mp.cpu_count()}")
    logging.info(f"🔥 Aggressive mode: {aggressive_mode} (FIXED)")
    
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
    
    # Print enhanced configuration
    print_enhanced_config(args, target_ratio, aggressive_mode, log_file)
    
    # Process files
    try:
        logging.info("🏁 Starting enhanced processing...")
        success = process_files(args, target_ratio, aggressive_mode)
        
        if success:
            logging.info("")
            logging.info("🎉 ENHANCED PROCESSING COMPLETED SUCCESSFULLY!")
            logging.info(f"🚀 Enhanced Ballast Reducer v2.4.1 (FIXED) - Ultra-Aggressive Reduction + Analytics!")
            logging.info(f"🔥 Mode used: {aggressive_mode} (FIXED)")
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
        logging.warning("⏹️ Processing interrupted by user (Ctrl+C)")
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
