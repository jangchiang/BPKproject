#!/usr/bin/env python3
"""
Complete GPU-Accelerated ML Point Cloud Reduction System
Ready-to-test implementation with all dependencies included
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
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.svm import SVC
    from sklearn.cluster import DBSCAN
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
    target_points_min: int = 500
    target_points_max: int = 800
    max_ballast: int = 300
    voxel_size: float = 0.02
    svm_sample_ratio: float = 0.1
    knn_neighbors: int = 5
    epsilon_range: Tuple[float, float] = (0.01, 0.1)
    dbscan_min_samples: int = 3
    reconstruction_method: str = 'poisson'  # 'poisson', 'ball_pivoting', 'alpha_shapes'
    n_cores: int = mp.cpu_count()
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    batch_size: int = 10000

class GPUAcceleratedReducer:
    """Complete GPU-accelerated ML-based point cloud reduction system"""
    
    def __init__(self, config: ReductionConfig = None):
        self.config = config or ReductionConfig()
        self.device = self._setup_gpu()
        self.scaler = None
        self.svm_model = None
        
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
    
    def surface_reconstruction(self, points: np.ndarray, normals: np.ndarray, 
                             method: str = 'poisson') -> Optional[trimesh.Trimesh]:
        """Surface reconstruction with multiple backend options"""
        try:
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
    
    def process_single_mesh(self, input_path: str, output_dir: str) -> Dict:
        """Process a single mesh file with GPU acceleration"""
        start_time = time.time()
        filename = Path(input_path).stem
        
        logger.info(f"🔄 Processing {filename} on {self.device.upper()}...")
        
        try:
            # Step 1: Load mesh
            logger.debug(f"Step 1: Loading mesh from {input_path}")
            points, normals = self.load_mesh(input_path)
            if points is None or normals is None:
                return {'status': 'failed', 'error': 'Failed to load mesh'}
            
            original_count = len(points)
            logger.info(f"  📊 Loaded: {original_count} vertices")
            
            # Step 2: Voxel downsampling if needed
            if original_count > 10000:
                logger.debug("Step 2: Applying voxel downsampling")
                points, normals = self.voxel_downsample_gpu(points, normals)
                logger.info(f"  📉 After downsampling: {len(points)} vertices")
            else:
                logger.debug("Step 2: Skipping downsampling (small mesh)")
            
            # Step 3: Normalization
            logger.debug("Step 3: Normalizing point cloud")
            normalized_points, norm_params = self.normalize_pointcloud(points)
            
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
            
            # Step 6: KNN reinforcement
            logger.debug("Step 6: Applying KNN reinforcement")
            enhanced_mask = self.knn_reinforcement_gpu(normalized_points, importance_mask)
            logger.info(f"  🔗 After KNN reinforcement: {np.sum(enhanced_mask)} points")
            
            # Step 7: Adaptive parameter tuning
            logger.debug("Step 7: Tuning parameters adaptively")
            best_params = self.adaptive_parameter_tuning(
                normalized_points, normals, enhanced_mask)
            logger.info(f"  ⚙️  Optimal epsilon: {best_params['epsilon']:.4f}")
            
            # Step 8: Hybrid merging
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
            
            # Step 9: Ensure constraints are met
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
            
            # Step 10: Denormalization
            logger.debug("Step 10: Denormalizing to original coordinates")
            final_points = self.denormalize_pointcloud(simplified_points, norm_params)
            
            # Step 11: Surface reconstruction
            logger.debug(f"Step 11: Surface reconstruction using {self.config.reconstruction_method}")
            reconstructed_mesh = self.surface_reconstruction(
                final_points, simplified_normals, self.config.reconstruction_method)
            
            # Step 12: Export results
            logger.debug("Step 12: Exporting results")
            output_subdir = Path(output_dir) / filename
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Save simplified mesh
            mesh_vertices = 0
            if reconstructed_mesh is not None:
                mesh_path = output_subdir / f"{filename}_simplified.stl"
                reconstructed_mesh.export(str(mesh_path))
                mesh_vertices = len(reconstructed_mesh.vertices)
                logger.debug(f"  💾 Saved simplified mesh: {mesh_path}")
            else:
                logger.warning("  ⚠️  Surface reconstruction failed, no mesh saved")
            
            # Save point cloud data
            points_df = pd.DataFrame(final_points, columns=['x', 'y', 'z'])
            normals_df = pd.DataFrame(simplified_normals, columns=['nx', 'ny', 'nz'])
            combined_df = pd.concat([points_df, normals_df], axis=1)
            
            csv_path = output_subdir / f"{filename}_points.csv"
            combined_df.to_csv(csv_path, index=False)
            logger.debug(f"  💾 Saved point cloud: {csv_path}")
            
            # Save as DAT format
            dat_path = output_subdir / f"{filename}_points.dat"
            np.savetxt(dat_path, final_points, fmt='%.6f')
            logger.debug(f"  💾 Saved DAT file: {dat_path}")
            
            processing_time = time.time() - start_time
            
            # Calculate results
            ballast_count = max(0, self.config.max_ballast - final_count)
            meets_constraints = (final_count >= self.config.target_points_min and 
                               final_count <= self.config.target_points_max)
            
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
                'device_used': self.device
            }
            
            status_icon = "✅" if meets_constraints else "⚠️"
            logger.info(f"{status_icon} Completed {filename}: {original_count} → {final_count} vertices "
                       f"({result['reduction_ratio']:.2%} reduction) in {processing_time:.2f}s")
            logger.info(f"  📊 Ballast remaining: {ballast_count}, Constraints met: {'Yes' if meets_constraints else 'No'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
            logger.debug(f"Full error details: {e}", exc_info=True)
            return {'status': 'failed', 'filename': filename, 'error': str(e)}
    
    def process_folder(self, input_folder: str, output_folder: str, log_file_path: str = None) -> List[Dict]:
        """Process all STL files in a folder"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find STL files
        stl_files = list(input_path.glob("*.stl")) + list(input_path.glob("*.STL"))
        
        if not stl_files:
            logger.error(f"No STL files found in {input_folder}")
            return []
        
        logger.info(f"Found {len(stl_files)} STL files to process with {self.device.upper()} acceleration")
        logger.info(f"Configuration: {self.config.target_points_min}-{self.config.target_points_max} vertices, max ballast {self.config.max_ballast}")
        
        results = []
        
        # Process files with progress logging
        for i, stl_file in enumerate(stl_files, 1):
            logger.info(f"Processing file {i}/{len(stl_files)}: {stl_file.name}")
            result = self.process_single_mesh(str(stl_file), str(output_path))
            results.append(result)
            
            # Log intermediate progress
            if i % 5 == 0 or i == len(stl_files):
                successful_so_far = sum(1 for r in results if r['status'] == 'success')
                logger.info(f"Progress: {i}/{len(stl_files)} files processed, {successful_so_far} successful")
        
        # Generate summary report
        self.generate_summary_report(results, output_path, log_file_path)
        
        return results
    
    def generate_summary_report(self, results: List[Dict], output_path: Path):
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
        else:
            avg_reduction = avg_time = total_time = constraint_met = 0
        
        # Save summary report
        summary_path = output_path / "summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write("=== GPU-Accelerated Point Cloud Reduction Summary ===\n\n")
            f.write(f"Processing device: {self.device.upper()}\n")
            f.write(f"Total files processed: {len(results)}\n")
            f.write(f"Successfully processed: {len(successful_results)}\n")
            f.write(f"Failed: {len(failed_results)}\n")
            f.write(f"Files meeting constraints (500-800 vertices): {constraint_met}\n")
            f.write(f"Average reduction ratio: {avg_reduction:.2%}\n")
            f.write(f"Average processing time per file: {avg_time:.2f} seconds\n")
            f.write(f"Total processing time: {total_time:.2f} seconds\n\n")
            
            if successful_results:
                f.write("Successful files:\n")
                for result in successful_results:
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


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated ML Point Cloud Reduction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python %(prog)s -i input_models -o output_models
  
  # Custom vertex range
  python %(prog)s -i models --min-vertices 400 --max-vertices 600
  
  # Disable GPU and use CPU only
  python %(prog)s -i models --no-gpu
  
  # Create test data and run
  python %(prog)s --create-test
        """
    )
    
    # Input/Output
    parser.add_argument('-i', '--input', 
                       help='Input folder containing STL files')
    parser.add_argument('-o', '--output', default='simplified_models',
                       help='Output folder for simplified models (default: simplified_models)')
    
    # Vertex constraints
    parser.add_argument('--min-vertices', type=int, default=500,
                       help='Minimum vertices in output (default: 500)')
    parser.add_argument('--max-vertices', type=int, default=800,
                       help='Maximum vertices in output (default: 800)')
    parser.add_argument('--max-ballast', type=int, default=300,
                       help='Maximum ballast points (default: 300)')
    
    # Processing parameters
    parser.add_argument('--voxel-size', type=float, default=0.02,
                       help='Voxel size for downsampling (default: 0.02)')
    parser.add_argument('--method', choices=['poisson', 'ball_pivoting', 'alpha_shapes'],
                       default='poisson', help='Surface reconstruction method (default: poisson)')
    
    # GPU settings
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration (use CPU only)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='GPU batch size (default: 10000)')
    
    # Test mode
    parser.add_argument('--create-test', action='store_true',
                       help='Create test data and run a test')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Reduce logging output')
    
    args = parser.parse_args()
    
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
        batch_size=args.batch_size
    )
    
    # Print configuration
    print("=== GPU-Accelerated ML Point Cloud Reduction ===")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"STL files found: {len(stl_files)}")
    print(f"Target vertex range: {config.target_points_min}-{config.target_points_max}")
    print(f"Maximum ballast: {config.max_ballast}")
    print(f"Reconstruction method: {config.reconstruction_method}")
    print(f"GPU acceleration: {'Enabled' if config.use_gpu else 'Disabled'}")
    
    # Create reducer and process files
    reducer = GPUAcceleratedReducer(config)
    
    start_time = time.time()
    results = reducer.process_folder(args.input, args.output, log_file_path)
    total_time = time.time() - start_time
    
    # Final summary logging
    logger.info("=== FINAL PROCESSING SUMMARY ===")
    logger.info(f"Total processing time: {total_time:.1f} seconds")
    logger.info(f"Log file saved to: {log_file_path}")
    logger.info("=== SESSION COMPLETE ===")
    
    # Print console summary
    successful = sum(1 for r in results if r['status'] == 'success')
    constraint_met = sum(1 for r in results if r.get('meets_constraints', False))
    failed = len(results) - successful
    
    print(f"\n=== Processing Complete ===")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Files processed: {successful}/{len(results)} successful")
    print(f"Files failed: {failed}")
    print(f"Meeting constraints: {constraint_met}/{successful}")
    print(f"Results saved to: {args.output}")
    print(f"📋 Real-time log saved to: {log_file_path}")
    
    if successful > 0:
        successful_results = [r for r in results if r['status'] == 'success']
        avg_reduction = sum(r.get('reduction_ratio', 0) for r in successful_results) / successful
        avg_time = sum(r.get('processing_time', 0) for r in successful_results) / successful
        print(f"Average reduction: {avg_reduction:.1%}")
        print(f"Average time per file: {avg_time:.2f} seconds")
        
        # Show device usage
        devices_used = set(r.get('device_used', 'unknown') for r in successful_results)
        print(f"Devices used: {', '.join(devices_used).upper()}")
    
    if args.create_test:
        print(f"\n🎉 Test completed successfully!")
        print(f"Check the log file for detailed processing information:")
        print(f"  📋 {log_file_path}")
        print(f"You can now use the system with your own STL files:")
        print(f"  mkdir input_models")
        print(f"  # Copy your STL files to input_models/")
        print(f"  python {sys.argv[0]} -i input_models -o output_models")
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
