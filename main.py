#!/usr/bin/env python3
"""
Complete Machine Learning-based Point Cloud Reduction System

Usage:
    python point_cloud_reduction_system.py /path/to/models --count 50 --workers 12
    python point_cloud_reduction_system.py /path/to/models --ratio 0.3 --workers 8
    python point_cloud_reduction_system.py model.stl --count 100 --method poisson

Requirements:
    pip install numpy pandas scikit-learn trimesh open3d

Author: Theeradon
Version: 1.0.0
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
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import ParameterGrid
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration with file and console output"""
    
    # Create logs directory if saving to file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always detailed in file
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Log the logging setup
        logging.info(f"📝 Logging to file: {log_file}")
        logging.info(f"📺 Console logging level: {logging.getLevelName(level)}")
        
    return root_logger


# Configure initial logging (will be reconfigured in main())
logger = logging.getLogger(__name__)


def process_single_file_worker(args):
    """
    Module-level worker function for parallel processing.
    This function can be pickled and sent to worker processes.
    
    Args:
        args: tuple of (file_path, output_dir, reducer_params)
    """
    file_path, output_dir, reducer_params = args
    
    try:
        # Create a new reducer instance for this worker
        worker_reducer = PointCloudReducer(
            target_reduction_ratio=reducer_params['target_reduction_ratio'],
            voxel_size=reducer_params['voxel_size'],
            n_cores=1,  # Each worker uses 1 core to avoid nested parallelism
            reconstruction_method=reducer_params['reconstruction_method'],
            fast_mode=reducer_params.get('fast_mode', False)
        )
        
        # Process the file
        result = worker_reducer.process_single_mesh(file_path, output_dir)
        return result
        
    except Exception as e:
        logger.error(f"Worker error processing {file_path}: {e}")
        return {'input_file': file_path, 'error': str(e)}


class PointCloudReducer:
    """
    Advanced Machine Learning-based Point Cloud Reduction System
    
    Implements a multi-stage pipeline for intelligent 3D model simplification:
    1. Importance Classification (SVM)
    2. Local Continuity (KNN Reinforcement) 
    3. Hybrid Merging (Radius + DBSCAN)
    4. Adaptive Parameter Estimation
    5. Multi-Resolution Preprocessing
    6. Flexible Surface Reconstruction
    """
    
    def __init__(self, 
                 target_reduction_ratio: float = 0.5,
                 voxel_size: Optional[float] = None,
                 n_cores: int = -1,
                 reconstruction_method: str = 'poisson',
                 fast_mode: bool = False):
        """
        Initialize the Point Cloud Reducer
        
        Args:
            target_reduction_ratio: Target ratio of points to keep (0.0-1.0)
            voxel_size: Voxel size for preprocessing downsampling 
            n_cores: Number of CPU cores to use (-1 for all)
            reconstruction_method: 'poisson', 'ball_pivoting', or 'alpha_shapes'
            fast_mode: Skip parameter optimization for faster processing
        """
        self.target_reduction_ratio = target_reduction_ratio
        self.voxel_size = voxel_size
        self.n_cores = mp.cpu_count() if n_cores == -1 else n_cores
        self.reconstruction_method = reconstruction_method
        self.fast_mode = fast_mode
        
        # Pipeline components
        self.scaler = StandardScaler()
        self.svm_classifier = None
        self.best_params = {}
        
        # Cache for optimization
        self._feature_cache = {}
        
    def load_mesh(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load STL mesh and extract point cloud with normals"""
        try:
            # Try trimesh first (better STL support)
            mesh = trimesh.load(file_path)
            if hasattr(mesh, 'vertices'):
                points = np.array(mesh.vertices)
                normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
            else:
                raise ValueError("Failed to extract vertices from mesh")
                
            # Fallback to Open3D if trimesh fails
            if normals is None:
                o3d_mesh = o3d.io.read_triangle_mesh(file_path)
                o3d_mesh.compute_vertex_normals()
                points = np.asarray(o3d_mesh.vertices)
                normals = np.asarray(o3d_mesh.vertex_normals)
                
            logger.info(f"Loaded mesh with {len(points)} vertices from {file_path}")
            return points, normals
            
        except Exception as e:
            logger.error(f"Failed to load mesh {file_path}: {e}")
            raise
    
    def voxel_downsample(self, points: np.ndarray, normals: np.ndarray, 
                        voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-resolution preprocessing via voxel downsampling"""
        if voxel_size is None:
            return points, normals
            
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Voxel downsample
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        
        down_points = np.asarray(downsampled_pcd.points)
        down_normals = np.asarray(downsampled_pcd.normals)
        
        logger.info(f"Voxel downsampled from {len(points)} to {len(down_points)} points")
        return down_points, down_normals
    
    def normalize_points(self, points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize points to unit cube centered at origin"""
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Scale to fit in unit cube
        max_extent = np.max(np.abs(centered_points))
        scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0
        normalized_points = centered_points * scale_factor
        
        normalization_params = {
            'centroid': centroid,
            'scale_factor': scale_factor
        }
        
        return normalized_points, normalization_params
    
    def extract_features(self, points: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
        """Extract geometric features for each point"""
        n_points = len(points)
        features = np.zeros((n_points, 5))  # 5 features per point
        
        # Build KD-tree for efficient neighbor search
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, n_points-1))
        nbrs.fit(points)
        
        # Global centroid distance
        centroid = np.mean(points, axis=0)
        features[:, 0] = np.linalg.norm(points - centroid, axis=1)
        
        # Parallel feature extraction for efficiency
        def compute_local_features(point_idx):
            distances, indices = nbrs.kneighbors([points[point_idx]])
            neighbors = points[indices[0]]
            
            # Local density (average distance to k nearest neighbors)
            local_density = np.mean(distances[0][1:])  # Exclude self
            
            # Curvature estimation via PCA
            if len(neighbors) > 3:
                centered_neighbors = neighbors - np.mean(neighbors, axis=0)
                cov_matrix = np.cov(centered_neighbors.T)
                eigenvalues = np.linalg.eigvals(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
                
                # Curvature measures
                linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
                planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
                sphericity = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
            else:
                linearity = planarity = sphericity = 0
                
            return point_idx, local_density, linearity, planarity, sphericity
        
        # Process in batches for memory efficiency
        batch_size = min(1000, n_points)
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_indices = range(i, end_idx)
            
            with ThreadPoolExecutor(max_workers=min(4, self.n_cores)) as executor:
                results = list(executor.map(compute_local_features, batch_indices))
            
            for point_idx, local_density, linearity, planarity, sphericity in results:
                features[point_idx, 1] = local_density
                features[point_idx, 2] = linearity
                features[point_idx, 3] = planarity
                features[point_idx, 4] = sphericity
        
        logger.info(f"Extracted features for {n_points} points")
        return features
    
    def create_pseudo_labels(self, features: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Create pseudo-labels for SVM training based on geometric importance"""
        n_points = len(points)
        
        # Combine multiple importance criteria
        importance_scores = np.zeros(n_points)
        
        # High curvature points (edges, corners)
        curvature_score = features[:, 2] + features[:, 3]  # linearity + planarity
        importance_scores += curvature_score
        
        # Boundary points (low local density)
        density_score = 1.0 / (features[:, 1] + 1e-8)  # Inverse density
        importance_scores += 0.5 * density_score
        
        # Extremal points (far from centroid)
        centroid_distance_score = features[:, 0]
        importance_scores += 0.3 * centroid_distance_score
        
        # Convert to binary labels (top percentile as important)
        threshold = np.percentile(importance_scores, 70)  # Top 30% as important
        pseudo_labels = (importance_scores >= threshold).astype(int)
        
        logger.info(f"Created pseudo-labels: {np.sum(pseudo_labels)} important points out of {n_points}")
        return pseudo_labels
    
    def train_svm_classifier(self, features: np.ndarray, labels: np.ndarray):
        """Train SVM classifier for importance prediction"""
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train SVM with RBF kernel
        self.svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_classifier.fit(features_scaled, labels)
        
        # Log training accuracy
        train_accuracy = self.svm_classifier.score(features_scaled, labels)
        logger.info(f"SVM training accuracy: {train_accuracy:.3f}")
    
    def knn_reinforcement(self, points: np.ndarray, important_mask: np.ndarray, 
                         k_neighbors: int) -> np.ndarray:
        """Grow important regions using KNN to ensure local continuity"""
        if np.sum(important_mask) == 0:
            return important_mask
        
        important_indices = np.where(important_mask)[0]
        reinforced_mask = important_mask.copy()
        
        # Build KD-tree
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(points)-1))
        nbrs.fit(points)
        
        # For each important point, include its k nearest neighbors
        for idx in important_indices:
            distances, neighbor_indices = nbrs.kneighbors([points[idx]])
            reinforced_mask[neighbor_indices[0]] = True
        
        logger.info(f"KNN reinforcement expanded from {np.sum(important_mask)} to {np.sum(reinforced_mask)} points")
        return reinforced_mask
    
    def radius_merge(self, points: np.ndarray, normals: np.ndarray, 
                    epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Merge points within epsilon radius using spatial clustering"""
        if epsilon <= 0 or len(points) == 0:
            return points, normals
        
        # Use DBSCAN for spatial clustering
        clustering = DBSCAN(eps=epsilon, min_samples=1)
        cluster_labels = clustering.fit_predict(points)
        
        # Compute cluster centroids
        unique_labels = np.unique(cluster_labels)
        merged_points = []
        merged_normals = []
        
        for label in unique_labels:
            if label == -1:  # Noise points
                noise_indices = np.where(cluster_labels == label)[0]
                merged_points.extend(points[noise_indices])
                merged_normals.extend(normals[noise_indices])
            else:
                cluster_indices = np.where(cluster_labels == label)[0]
                centroid = np.mean(points[cluster_indices], axis=0)
                avg_normal = np.mean(normals[cluster_indices], axis=0)
                # Normalize the averaged normal
                norm = np.linalg.norm(avg_normal)
                if norm > 0:
                    avg_normal /= norm
                
                merged_points.append(centroid)
                merged_normals.append(avg_normal)
        
        merged_points = np.array(merged_points) if merged_points else np.empty((0, 3))
        merged_normals = np.array(merged_normals) if merged_normals else np.empty((0, 3))
        
        logger.info(f"Radius merge reduced from {len(points)} to {len(merged_points)} points")
        return merged_points, merged_normals
    
    def dbscan_cleanup(self, points: np.ndarray, normals: np.ndarray,
                      eps: float, min_samples: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Apply DBSCAN to remove outliers and recombine clusters"""
        if len(points) == 0:
            return points, normals
            
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(points)
        
        # Keep only non-noise points and compute cluster centroids
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) == 0:
            return points, normals
        
        valid_points = points[valid_mask]
        valid_normals = normals[valid_mask]
        valid_labels = cluster_labels[valid_mask]
        
        # Compute cluster centroids
        unique_labels = np.unique(valid_labels)
        cleaned_points = []
        cleaned_normals = []
        
        for label in unique_labels:
            cluster_indices = np.where(valid_labels == label)[0]
            centroid = np.mean(valid_points[cluster_indices], axis=0)
            avg_normal = np.mean(valid_normals[cluster_indices], axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                avg_normal /= norm
            
            cleaned_points.append(centroid)
            cleaned_normals.append(avg_normal)
        
        cleaned_points = np.array(cleaned_points) if cleaned_points else np.empty((0, 3))
        cleaned_normals = np.array(cleaned_normals) if cleaned_normals else np.empty((0, 3))
        
        logger.info(f"DBSCAN cleanup: {len(points)} -> {len(cleaned_points)} points")
        return cleaned_points, cleaned_normals
    
    def adaptive_parameter_search(self, points: np.ndarray, normals: np.ndarray,
                                features: np.ndarray, target_count: int, 
                                fast_mode: bool = False) -> Dict:
        """Grid search for optimal parameters to achieve target point count"""
        
        if fast_mode:
            # Fast mode: use reasonable defaults without grid search
            logger.info(f"Fast mode: Using default parameters for target count: {target_count}")
            return {
                'k_neighbors': 10,
                'epsilon': 0.05,
                'dbscan_eps': 0.08,
                'final_count': target_count
            }
        
        # Simplified parameter grid for speed
        param_grid = {
            'k_neighbors': [5, 10, 15],  # Reduced from more options
            'epsilon': [0.02, 0.05, 0.08],  # Reduced from 5 to 3 options
            'dbscan_eps': [0.03, 0.08]  # Reduced from 4 to 2 options
        }
        
        best_params = None
        best_score = float('inf')
        
        logger.info(f"Starting parameter search for target count: {target_count}")
        total_combinations = len(list(ParameterGrid(param_grid)))
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        start_time = time.time()
        tested = 0
        
        for params in ParameterGrid(param_grid):
            try:
                tested += 1
                elapsed = time.time() - start_time
                if tested > 1:
                    avg_time_per_combo = elapsed / tested
                    remaining_combos = total_combinations - tested
                    eta = remaining_combos * avg_time_per_combo
                    logger.info(f"Parameter search progress: {tested}/{total_combinations} "
                              f"(ETA: {eta:.1f}s)")
                
                # Create pseudo-labels and train SVM
                pseudo_labels = self.create_pseudo_labels(features, points)
                features_scaled = self.scaler.fit_transform(features)
                
                temp_svm = SVC(kernel='rbf', probability=True, random_state=42)
                temp_svm.fit(features_scaled, pseudo_labels)
                
                # Predict importance
                important_probs = temp_svm.predict_proba(features_scaled)[:, 1]
                important_mask = important_probs > 0.5
                
                # Apply KNN reinforcement
                reinforced_mask = self.knn_reinforcement(points, important_mask, 
                                                       params['k_neighbors'])
                
                # Select points and apply merging
                selected_points = points[reinforced_mask]
                selected_normals = normals[reinforced_mask]
                
                if len(selected_points) == 0:
                    continue
                
                # Radius merge
                merged_points, merged_normals = self.radius_merge(
                    selected_points, selected_normals, params['epsilon'])
                
                # DBSCAN cleanup
                final_points, final_normals = self.dbscan_cleanup(
                    merged_points, merged_normals, params['dbscan_eps'])
                
                # Score based on distance from target
                current_count = len(final_points)
                score = abs(current_count - target_count)
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    best_params['final_count'] = current_count
                    logger.info(f"New best: {best_params} (score: {best_score})")
                
            except Exception as e:
                logger.warning(f"Parameter combination failed: {params}, Error: {e}")
                continue
        
        if best_params is None:
            # Fallback parameters
            best_params = {
                'k_neighbors': 10,
                'epsilon': 0.05,
                'dbscan_eps': 0.08,
                'final_count': len(points)
            }
        
        search_time = time.time() - start_time
        logger.info(f"Parameter search completed in {search_time:.1f}s")
        logger.info(f"Best parameters found: {best_params}")
        return best_params
    
    def denormalize_points(self, points: np.ndarray, 
                          normalization_params: Dict) -> np.ndarray:
        """Transform points back to original coordinate frame"""
        if len(points) == 0:
            return points
        denormalized = points / normalization_params['scale_factor']
        denormalized += normalization_params['centroid']
        return denormalized
    
    def surface_reconstruction(self, points: np.ndarray, normals: np.ndarray) -> Optional[trimesh.Trimesh]:
        """Reconstruct surface mesh from simplified point cloud"""
        if len(points) < 4:  # Need at least 4 points for reconstruction
            logger.warning("Too few points for surface reconstruction")
            return None
            
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        try:
            if self.reconstruction_method == 'poisson':
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, width=0, scale=1.1, linear_fit=False)
                
            elif self.reconstruction_method == 'ball_pivoting':
                # Estimate radius for ball pivoting
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radius = 2 * avg_dist
                
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector([radius, radius * 2]))
                
            elif self.reconstruction_method == 'alpha_shapes':
                # Alpha shapes approximation using Delaunay
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.1)
            
            else:
                raise ValueError(f"Unknown reconstruction method: {self.reconstruction_method}")
            
            # Convert to trimesh
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            if len(faces) == 0:
                logger.warning("Surface reconstruction produced no faces")
                return None
            
            reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            logger.info(f"Reconstructed mesh with {len(vertices)} vertices and {len(faces)} faces")
            return reconstructed_mesh
            
        except Exception as e:
            logger.error(f"Surface reconstruction failed: {e}")
            return None
    
    def process_single_mesh(self, input_path: str, output_dir: str) -> Dict:
        """Process a single mesh file through the complete pipeline"""
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Track processing time
            start_time = time.time()
            
            # Load mesh
            load_start = time.time()
            points, normals = self.load_mesh(input_path)
            original_count = len(points)
            load_time = time.time() - load_start
            
            if original_count == 0:
                return {'input_file': input_path, 'error': 'Empty mesh'}
            
            logger.info(f"⏱️  Mesh loaded in {load_time:.1f}s")
            
            # Voxel downsampling (optional)
            if self.voxel_size:
                downsample_start = time.time()
                points, normals = self.voxel_downsample(points, normals, self.voxel_size)
                downsample_time = time.time() - downsample_start
                logger.info(f"⏱️  Voxel downsampling in {downsample_time:.1f}s")
            
            # Normalize points
            norm_start = time.time()
            normalized_points, norm_params = self.normalize_points(points)
            norm_time = time.time() - norm_start
            logger.info(f"⏱️  Normalization in {norm_time:.1f}s")
            
            # Extract features
            feature_start = time.time()
            features = self.extract_features(normalized_points)
            feature_time = time.time() - feature_start
            logger.info(f"⏱️  Feature extraction in {feature_time:.1f}s")
            
            # Calculate target count
            target_count = max(4, int(len(points) * self.target_reduction_ratio))
            
            # Adaptive parameter search
            param_start = time.time()
            best_params = self.adaptive_parameter_search(
                normalized_points, normals, features, target_count, self.fast_mode)
            param_time = time.time() - param_start
            
            if self.fast_mode:
                logger.info(f"⏱️  Fast mode (no parameter search): {param_time:.1f}s")
            else:
                logger.info(f"⏱️  Parameter optimization in {param_time:.1f}s")
            
            # Apply best parameters
            pipeline_start = time.time()
            pseudo_labels = self.create_pseudo_labels(features, normalized_points)
            self.train_svm_classifier(features, pseudo_labels)
            
            # Predict importance
            features_scaled = self.scaler.transform(features)
            important_probs = self.svm_classifier.predict_proba(features_scaled)[:, 1]
            important_mask = important_probs > 0.5
            
            # KNN reinforcement
            reinforced_mask = self.knn_reinforcement(
                normalized_points, important_mask, best_params['k_neighbors'])
            
            # Select points
            selected_points = normalized_points[reinforced_mask]
            selected_normals = normals[reinforced_mask]
            
            if len(selected_points) == 0:
                return {'input_file': input_path, 'error': 'No points selected'}
            
            # Hybrid merging
            merged_points, merged_normals = self.radius_merge(
                selected_points, selected_normals, best_params['epsilon'])
            
            final_points, final_normals = self.dbscan_cleanup(
                merged_points, merged_normals, best_params['dbscan_eps'])
            
            if len(final_points) == 0:
                return {'input_file': input_path, 'error': 'All points eliminated during processing'}
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"⏱️  ML pipeline in {pipeline_time:.1f}s")
            
            # Denormalize
            final_points = self.denormalize_points(final_points, norm_params)
            
            # Surface reconstruction
            recon_start = time.time()
            reconstructed_mesh = self.surface_reconstruction(final_points, final_normals)
            recon_time = time.time() - recon_start
            logger.info(f"⏱️  Surface reconstruction in {recon_time:.1f}s")
            
            # Save results - Create subdirectory for each model
            save_start = time.time()
            filename = Path(input_path).stem
            model_output_dir = output_path / filename
            
            # Create subdirectory with progress message
            model_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created subfolder: {model_output_dir.name}/")
            
            # Save simplified STL
            stl_path = None
            if reconstructed_mesh:
                stl_path = model_output_dir / f"{filename}_simplified.stl"
                reconstructed_mesh.export(str(stl_path))
                logger.info(f"💾 Saved STL: {model_output_dir.name}/{filename}_simplified.stl")
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
            
            # Final completion message for this file
            logger.info(f"✅ COMPLETED: {filename} → All files saved to {model_output_dir.name}/")
            logger.info(f"📊 Summary: {original_count:,} → {len(final_points):,} points (ratio: {len(final_points) / original_count:.3f})")
            logger.info(f"⏱️  Total processing time: {total_time:.1f}s")
            logger.info(f"⏱️  Breakdown: Load({load_time:.1f}s) + Features({feature_time:.1f}s) + "
                       f"Params({param_time:.1f}s) + Pipeline({pipeline_time:.1f}s) + "
                       f"Recon({recon_time:.1f}s) + Save({save_time:.1f}s)")
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
                    'features': feature_time,
                    'parameters': param_time,
                    'pipeline': pipeline_time,
                    'reconstruction': recon_time,
                    'save': save_time
                },
                'parameters': best_params,
                'output_files': {
                    'stl': str(stl_path) if stl_path else None,
                    'csv': str(csv_path),
                    'dat': str(dat_path)
                }
            }
            
            logger.info(f"Successfully processed {filename}: {original_count} -> {len(final_points)} points")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {e}")
            return {'input_file': input_path, 'error': str(e)}
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.stl") -> List[Dict]:
        """Process multiple mesh files in parallel"""
        input_path = Path(input_dir)
        stl_files = list(input_path.glob(file_pattern))
        
        if not stl_files:
            logger.warning(f"No STL files found in {input_dir}")
            return []
        
        logger.info(f"Processing {len(stl_files)} files with {self.n_cores} cores")
        
        # TRUE PARALLEL PROCESSING - Multiple files simultaneously
        if len(stl_files) == 1 or self.n_cores == 1:
            # Single file or single core - no need for parallel processing
            results = []
            for file_path in stl_files:
                try:
                    result = self.process_single_mesh(str(file_path), output_dir)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append({'input_file': str(file_path), 'error': str(e)})
        else:
            # Multiple files and multiple cores - use parallel processing
            
            # Prepare reducer parameters for workers
            reducer_params = {
                'target_reduction_ratio': self.target_reduction_ratio,
                'voxel_size': self.voxel_size,
                'reconstruction_method': self.reconstruction_method
            }
            
            # Prepare arguments for worker processes
            worker_args = [
                (str(file_path), output_dir, reducer_params) 
                for file_path in stl_files
            ]
            
            # Use ProcessPoolExecutor for true parallel file processing
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                logger.info(f"🚀 Starting parallel processing with {self.n_cores} workers...")
                
                # Submit all files to the worker pool using the module-level function
                future_to_args = {
                    executor.submit(process_single_file_worker, args): args[0]  # args[0] is file_path
                    for args in worker_args
                }
                
                results = []
                completed = 0
                total_files = len(stl_files)
                
                # Collect results as they complete
                from concurrent.futures import as_completed
                for future in as_completed(future_to_args):
                    try:
                        result = future.result(timeout=600)  # 10 minute timeout per file
                        results.append(result)
                        completed += 1
                        
                        # Progress tracking with detailed save information
                        if 'error' not in result:
                            filename = Path(result['input_file']).name
                            model_name = Path(result['input_file']).stem
                            logger.info(f"🎉 [{completed}/{total_files}] BATCH COMPLETED: {filename}")
                            logger.info(f"   📁 Subfolder: {model_name}/")
                            logger.info(f"   📊 Points: {result['original_points']:,} → {result['final_points']:,}")
                            logger.info(f"   📈 Ratio: {result['reduction_ratio']:.3f}")
                            
                            # Show saved files
                            if result['output_files']['stl']:
                                logger.info(f"   💾 Files saved: STL ✅ CSV ✅ DAT ✅")
                            else:
                                logger.info(f"   💾 Files saved: STL ❌ CSV ✅ DAT ✅")
                            logger.info("-" * 60)
                        else:
                            filename = Path(result['input_file']).name
                            logger.error(f"❌ [{completed}/{total_files}] BATCH FAILED: {filename}")
                            logger.error(f"   🚨 Error: {result['error']}")
                            logger.error("-" * 60)
                            
                    except Exception as e:
                        file_path = future_to_args[future]
                        logger.error(f"Future error for {file_path}: {e}")
                        results.append({'input_file': str(file_path), 'error': str(e)})
                        completed += 1
        
        # Save batch summary with progress message
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
        logger.info("=" * 80)
        return results


def estimate_target_ratio(input_path: str, target_count: int) -> float:
    """Estimate target reduction ratio based on sample file"""
    try:
        reducer = PointCloudReducer()
        
        # Find a sample file to estimate point count
        if os.path.isfile(input_path):
            sample_file = input_path
        else:
            # Find first STL file in directory
            stl_files = list(Path(input_path).glob("*.stl"))
            if not stl_files:
                return 0.5  # Default ratio
            sample_file = str(stl_files[0])
        
        # Load sample and estimate
        points, _ = reducer.load_mesh(sample_file)
        original_count = len(points)
        estimated_ratio = target_count / original_count if original_count > 0 else 0.5
        
        # Clamp to reasonable bounds
        return max(0.01, min(0.95, estimated_ratio))
        
    except Exception as e:
        logger.warning(f"Could not estimate ratio: {e}")
        return 0.5  # Fallback


def validate_args(args):
    """Validate command line arguments"""
    # Check input path exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Check target specification
    if args.count and args.ratio:
        print("Error: Cannot specify both --count and --ratio")
        sys.exit(1)
    
    if not args.count and not args.ratio:
        print("Error: Must specify either --count or --ratio")
        sys.exit(1)
    
    # Validate ratio range
    if args.ratio and (args.ratio <= 0 or args.ratio >= 1):
        print("Error: --ratio must be between 0 and 1")
        sys.exit(1)
    
    # Validate count
    if args.count and args.count <= 0:
        print("Error: --count must be positive")
        sys.exit(1)
    
    # Validate workers
    if args.workers <= 0:
        print("Error: --workers must be positive")
        sys.exit(1)
    
    # Validate reconstruction method
    valid_methods = ['poisson', 'ball_pivoting', 'alpha_shapes']
    if args.method not in valid_methods:
        print(f"Error: --method must be one of {valid_methods}")
        sys.exit(1)


def print_header():
    """Print application header"""
    # Header is now handled in print_config function
    pass


def print_config(args, target_ratio: float, log_file: Optional[str] = None):
    """Print processing configuration"""
    logging.info("=" * 60)
    logging.info("🔧 Point Cloud Reduction System - ML Pipeline")
    logging.info("=" * 60)
    logging.info(f"📂 Input: {args.input}")
    logging.info(f"📁 Output: {args.output}")
    
    if args.count:
        logging.info(f"🎯 Target: {args.count} points (estimated ratio: {target_ratio:.3f})")
    else:
        logging.info(f"🎯 Target: {args.ratio:.1%} of original points")
    
    logging.info(f"👥 Workers: {args.workers}")
    logging.info(f"🔧 Method: {args.method}")
    
    if args.voxel:
        logging.info(f"📦 Voxel size: {args.voxel}")
    
    if args.verbose:
        logging.info("🔍 Verbose mode: ON")
    
    if log_file:
        logging.info(f"📝 Log file: {log_file}")
    
    logging.info("-" * 60)


def process_files(args, target_ratio: float) -> bool:
    """Main processing function"""
    
    # Log processing start
    logging.info("🚀 Initializing Point Cloud Reducer...")
    
    # Initialize reducer
    reducer = PointCloudReducer(
        target_reduction_ratio=target_ratio,
        voxel_size=args.voxel,
        n_cores=args.workers,
        reconstruction_method=args.method
    )
    
    logging.info("✅ Reducer initialized successfully")
    start_time = time.time()
    
    # Check if input is single file or directory
    if os.path.isfile(args.input):
        # Single file processing
        logging.info(f"🔄 Processing single file: {os.path.basename(args.input)}")
        
        results = reducer.process_single_mesh(args.input, args.output)
        
        if 'error' in results:
            logging.error(f"❌ Processing failed: {results['error']}")
            return False
        else:
            logging.info(f"✅ SUCCESS: {results['original_points']:,} → {results['final_points']:,} points")
            logging.info(f"📊 Actual ratio: {results['reduction_ratio']:.3f}")
            
    else:
        # Batch processing
        stl_files = list(Path(args.input).glob("*.stl"))
        if not stl_files:
            logging.error(f"❌ No STL files found in {args.input}")
            return False
        
        logging.info(f"🔄 Processing {len(stl_files)} files...")
        logging.info(f"⚡ Target: ~{int(len(stl_files) * target_ratio * 50000)} total points from {len(stl_files)} files")
        
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
            logging.info(f"📊 Average ratio: {avg_ratio:.3f}")
            logging.info(f"🎯 Target ratio: {target_ratio:.3f}")
            
            # Performance metrics
            if len(successful) > 1:
                avg_original = total_original / len(successful)
                avg_final = total_final / len(successful)
                logging.info(f"📊 Average per file: {avg_original:,.0f} → {avg_final:.0f} points")
        
        if failed and len(failed) <= 10:  # Show details for small number of failures
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
    
    # Calculate performance stats
    if os.path.isdir(args.input):
        stl_count = len(list(Path(args.input).glob("*.stl")))
        if stl_count > 0 and elapsed_time > 0:
            files_per_second = stl_count / elapsed_time
            logging.info(f"⚡ Performance: {files_per_second:.2f} files/second")
            if args.workers > 1:
                speedup = min(args.workers, stl_count)
                logging.info(f"🚀 Estimated speedup: ~{speedup:.1f}x with {args.workers} workers")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Reduce 3D point clouds using machine learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory with automatic logging (recommended)
  python point_cloud_reduction_system.py /home/user/models --count 50 --workers 12
  
  # Process with custom log file
  python point_cloud_reduction_system.py /home/user/models --ratio 0.3 --workers 8 --log-file my_custom.log
  
  # Process without any log file (console only)
  python point_cloud_reduction_system.py /home/user/models --count 100 --no-log
  
  # Verbose mode with automatic logging
  python point_cloud_reduction_system.py /data/models --ratio 0.2 --workers 16 --verbose

Auto-logging:
  By default, detailed logs are automatically saved to:
  logs/[input_name]_processing_[timestamp].log
  
  Use --no-log to disable automatic logging
  Use --log-file to specify custom log location

Installation:
  pip install numpy pandas scikit-learn trimesh open3d
        """
    )
    
    # Required arguments
    parser.add_argument('input', 
                       help='Input STL file or directory containing STL files')
    
    # Target specification (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--count', type=int,
                             help='Target number of points to keep')
    target_group.add_argument('--ratio', type=float,
                             help='Target reduction ratio (0.0-1.0)')
    
    # Processing options
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    
    # Advanced options
    parser.add_argument('--method', type=str, default='poisson',
                       choices=['poisson', 'ball_pivoting', 'alpha_shapes'],
                       help='Surface reconstruction method (default: poisson)')
    parser.add_argument('--voxel', type=float,
                       help='Voxel size for preprocessing downsampling')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Custom log file path (default: auto-generated in logs/ folder)')
    parser.add_argument('--no-log', action='store_true',
                       help='Disable automatic log file creation (console only)')
    parser.add_argument('--version', action='version', version='1.0.0')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Auto-generate log file path if not specified and not disabled
    log_file = None
    if not getattr(args, 'no_log', False):
        if getattr(args, 'log_file', None):
            # User specified custom log file
            log_file = args.log_file
        else:
            # Auto-generate log file with timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            input_name = Path(args.input).name if os.path.isfile(args.input) else Path(args.input).name
            log_file = f"logs/{input_name}_processing_{timestamp}.log"
    
    # Setup logging with optional file output
    logger = setup_logging(args.verbose, log_file)
    
    # Log startup information
    logging.info("🚀 Point Cloud Reduction System Starting...")
    logging.info(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"💻 System: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    logging.info(f"⚡ Available CPU cores: {mp.cpu_count()}")
    
    # Validate arguments
    validate_args(args)
    
    # Calculate target ratio
    if args.count:
        logging.info(f"🎯 Estimating target ratio for {args.count} points...")
        target_ratio = estimate_target_ratio(args.input, args.count)
        logging.info(f"📊 Estimated target ratio: {target_ratio:.3f}")
    else:
        target_ratio = args.ratio
        logging.info(f"📊 Using specified ratio: {target_ratio:.3f}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    logging.info(f"📁 Output directory ready: {args.output}")
    
    # Print header and config
    print_config(args, target_ratio, log_file)
    
    # Process files
    try:
        logging.info("🏁 Starting file processing...")
        success = process_files(args, target_ratio)
        
        if success:
            logging.info("")
            logging.info("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
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
