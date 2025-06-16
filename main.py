import os  # for file/directory operations
import glob  # to find files by pattern
import argparse  # for command-line argument parsing
import numpy as np  # numerical operations
import trimesh  # to load STL meshes
from sklearn.decomposition import PCA  # for curvature estimation
from sklearn.svm import SVC  # for classification
from sklearn.neighbors import NearestNeighbors  # for KNN operations
from sklearn.cluster import DBSCAN  # for density-based clustering
import open3d as o3d  # point cloud and mesh processing

# Utility: visualize point cloud with optional color

def visualize(points: np.ndarray, mask: np.ndarray = None, title: str = "PointCloud"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if mask is not None:
        colors = np.zeros((len(points), 3))
        colors[mask] = [1, 0, 0]
        colors[~mask] = [0.7, 0.7, 0.7]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([pcd], window_name=title)

class PointCloudLoader:
    """
    Loads a mesh file and extracts its vertex point cloud.
    """
    def __init__(self, filename: str):
        self.filename = filename

    def load(self) -> np.ndarray:
        mesh = trimesh.load(self.filename)
        return np.array(mesh.vertices)

class Preprocessor:
    """
    Normalizes point clouds: p' = (p - μ) / R_max
    """
    def normalize(self, points: np.ndarray) -> np.ndarray:
        centroid = points.mean(axis=0)
        pts_centered = points - centroid
        max_range = np.max(np.ptp(pts_centered, axis=0))
        return pts_centered / max_range

class FeatureExtractor:
    """
    Extracts geometric features per point:
     - curvature = λ_min / Σλ_i
     - density = |N_r(p)|
     - dist_centroid = ||p - μ||₂
    """
    def __init__(self, radius: float = 0.05):
        self.radius = radius

    def compute(self, points: np.ndarray) -> np.ndarray:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(self.radius))
        nbrs = NearestNeighbors(radius=self.radius).fit(points)
        features = []
        centroid = points.mean(axis=0)
        for pt in points:
            idx = nbrs.radius_neighbors([pt], return_distance=False)[0]
            local_pts = points[idx]
            pca = PCA(n_components=3).fit(local_pts)
            eigen = np.sort(pca.explained_variance_)
            curvature = eigen[0] / eigen.sum() if eigen.sum() > 0 else 0
            density = len(idx)
            dist_centroid = np.linalg.norm(pt - centroid)
            features.append([curvature, density, dist_centroid])
        return np.array(features)

class SVMClassifier:
    """
    Trains an SVM: f(x) = sign(∑_i α_i y_i K(x_i, x) + b).
    """
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True)

    def train(self, features: np.ndarray, labels: np.ndarray):
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

class KNNReinforcement:
    """
    Expands important set via K-nearest neighbors: N_k(p).
    """
    def __init__(self, k: int = 5):
        self.k = k

    def expand(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(points)
        idx_sel = np.where(mask)[0]
        neighbors = nbrs.kneighbors(points[idx_sel], return_distance=False)
        expanded = set(idx_sel.tolist())
        for nbr in neighbors:
            expanded.update(nbr.tolist())
        mask_expanded = np.zeros_like(mask, dtype=bool)
        mask_expanded[list(expanded)] = True
        return mask_expanded

class HybridMerger:
    """
    Merges points by DBSCAN clustering: c_j = (1/|C_j|) ∑_{p∈C_j} p
    """
    def __init__(self, eps: float = 0.02):
        self.eps = eps

    def merge(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        selected = points[mask]
        db = DBSCAN(eps=self.eps, min_samples=1).fit(selected)
        centroids = [selected[db.labels_ == lbl].mean(axis=0)
                     for lbl in np.unique(db.labels_)]
        return np.array(centroids)

class ParameterEstimator:
    """
    Auto-tunes parameters to meet either:
      - target_ratio: |S| ≈ α * |I|  or
      - explicit target_count: |S| ≈ N_target
    """
    def __init__(self, eps_values=None, k_values=None, target_ratio: float = 0.2, target_count: int = None):
        self.eps_values = eps_values or np.linspace(0.005, 0.05, 10)
        self.k_values = k_values or [3, 5, 7, 9]
        self.target_ratio = target_ratio
        self.target_count = target_count

    def tune(self, points: np.ndarray, mask: np.ndarray) -> dict:
        best_score = -np.inf
        best_params = {'eps': self.eps_values[0], 'k': self.k_values[0]}
        original_count = mask.sum()
        if self.target_count is not None:
            target = self.target_count
        else:
            target = int(original_count * self.target_ratio)
        for eps in self.eps_values:
            for k in self.k_values:
                mask_exp = KNNReinforcement(k=k).expand(points, mask)
                merged = HybridMerger(eps=eps).merge(points, mask_exp)
                score = -abs(len(merged) - target)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'k': k}
        return best_params

class SurfaceReconstructor:
    """
    Mesh reconstruction methods: Poisson, BPA, Alpha shape.
    """
    def __init__(self, method: str = 'poisson'):
        self.method = method

    def reconstruct(self, points: np.ndarray) -> o3d.geometry.TriangleMesh:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if self.method == 'poisson':
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        elif self.method == 'bpa':
            radii = o3d.utility.DoubleVector([0.005, 0.01, 0.02])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        else:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.03)
        return mesh

class Exporter:
    """
    Outputs simplified mesh (*.stl) and vertices (*.csv).
    """
    def export(self, mesh: o3d.geometry.TriangleMesh, out_dir: str):
        mesh.compute_vertex_normals()
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(out_dir)
        o3d.io.write_triangle_mesh(os.path.join(out_dir, f"{base}.stl"), mesh)
        np.savetxt(os.path.join(out_dir, f"{base}.csv"), np.asarray(mesh.vertices), delimiter=',')

class SimplificationPipeline:
    """
    Full pipeline with optional visualization and user-defined reduction.
    """
    def __init__(self, filepath: str, labels: np.ndarray, output_base: str,
                 visualize_steps: bool = False, target_count: int = None, target_ratio: float = 0.2):
        self.filepath = filepath
        self.labels = labels
        self.output_base = output_base
        self.visualize = visualize_steps
        self.target_count = target_count
        self.target_ratio = target_ratio

    def run(self):
        points = PointCloudLoader(self.filepath).load()
        if self.visualize:
            visualize(points, title='Raw Point Cloud')
        pts_norm = Preprocessor().normalize(points)
        if self.visualize:
            visualize(pts_norm, title='Normalized Cloud')
        features = FeatureExtractor().compute(pts_norm)
        clf = SVMClassifier()
        clf.train(features, self.labels)
        mask = clf.predict(features).astype(bool)
        if self.visualize:
            visualize(pts_norm, mask, title='SVM Important Points')
        estimator = ParameterEstimator(target_ratio=self.target_ratio, target_count=self.target_count)
        params = estimator.tune(pts_norm, mask)
        mask_exp = KNNReinforcement(k=params['k']).expand(pts_norm, mask)
        if self.visualize:
            visualize(pts_norm, mask_exp, title='KNN Expanded Mask')
        simplified = HybridMerger(eps=params['eps']).merge(pts_norm, mask_exp)
        if self.visualize:
            visualize(simplified, title='Simplified Points')
        mesh = SurfaceReconstructor(method='poisson').reconstruct(simplified)
        if self.visualize:
            o3d.visualization.draw_geometries([mesh], window_name='Reconstructed Mesh')
        base = os.path.splitext(os.path.basename(self.filepath))[0]
        out_dir = os.path.join(self.output_base, base)
        Exporter().export(mesh, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch simplify STL files with customizable reduction and display each step.")
    parser.add_argument('input_folder', help='Folder with .stl files')
    parser.add_argument('--visualize', action='store_true', help='Show GUI at each stage')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Target reduction ratio α (default: 0.2)')
    parser.add_argument('--count', type=int,
                        help='Explicit target number of points after simplification')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = os.path.join(input_folder, 'simplified')
    stl_files = glob.glob(os.path.join(input_folder, '*.stl'))

    for stl in stl_files:
        print(f"Processing {stl}...")
        points = PointCloudLoader(stl).load()
        labels = np.ones(len(points))
        pipeline = SimplificationPipeline(
            stl, labels, output_folder,
            visualize_steps=args.visualize,
            target_count=args.count,
            target_ratio=args.ratio)
        pipeline.run()
        print(f"Saved output to {os.path.join(output_folder, os.path.splitext(os.path.basename(stl))[0])}")
