Controlled Convex Hull Reconstruction for LMGC90
Purpose
The controlled convex hull reconstruction method is specifically designed for LMGC90 discrete element simulation software, which requires:
Convex geometry only (no concave shapes)
Specific vertex count range (typically 350-500 vertices for optimal performance)
High-quality mesh suitable for physics simulation
Step-by-Step Process
Step 1: Intelligent Point Reduction
python
if len(points) > target_max:
    reduced_points, reduced_normals = self._intelligent_point_reduction(
        points, normals, target_max)
What it does:
If the input has too many points (>500), intelligently reduces them
Uses three scoring methods combined:
Importance scores: SVM-based geometric feature importance
Diversity scores: Spatial distribution (keeps well-spaced points)
Boundary scores: Points closer to convex hull boundary get priority
Why it's important:
Simple random sampling would lose important geometric features
This preserves shape characteristics while reducing complexity
Step 2: Initial Convex Hull Computation
python
temp_mesh = trimesh.Trimesh(vertices=reduced_points)
convex_hull = temp_mesh.convex_hull
What it does:
Creates a convex hull from the reduced point set
Uses trimesh's robust convex hull algorithm
Key insight:
Convex hull of reduced points ≠ reduced convex hull of original
This approach maintains better geometric fidelity
Step 3: Mesh Decimation (if needed)
python
if initial_vertices > target_max:
    final_mesh = self._decimate_convex_mesh(convex_hull, target_max)
What it does:
If the convex hull still has too many vertices, applies decimation
Uses multiple decimation strategies:
Quadric decimation: Preserves geometry while reducing vertices
Vertex clustering: Groups nearby vertices and uses centroids
Random sampling: Fallback method
Convexity preservation:
After each decimation, recomputes convex hull to ensure convexity
Critical for LMGC90 compatibility
Step 4: Minimum Vertex Enforcement
python
if len(final_mesh.vertices) < target_min:
    # Use larger point set and recompute
What it does:
If result has too few vertices (<350), starts over with more points
Ensures minimum complexity for stable simulation
Step 5: Final Validation
python
if not final_mesh.is_convex:
    final_mesh = final_mesh.convex_hull
What it does:
Double-checks convexity (should always be true, but safety check)
Recomputes convex hull if somehow non-convex
Key Differences from Standard Convex Hull
Aspect	Standard Convex Hull	Controlled Convex Hull
Vertex Count	Whatever results naturally	Strictly controlled (350-500)
Point Selection	Uses all points	Intelligent reduction first
Quality Focus	Basic convexity	Simulation-ready quality
Validation	Minimal	Comprehensive LMGC90 checks
The Three Scoring Methods in Detail
1. Importance Scores (40% weight)
Uses SVM to classify geometrically important points
Based on curvature, density, centroid distance, normal magnitude
Keeps points that define the shape's character
2. Diversity Scores (30% weight)
python
def _calculate_spatial_diversity(self, points: np.ndarray) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=min(20, len(points)))
    distances, indices = nbrs.kneighbors(points)
    diversity_scores = np.mean(distances[:, 1:], axis=1)  # Higher = more isolated
Points in less dense areas get higher scores
Prevents clustering all selected points in one region
Maintains overall shape proportions
3. Boundary Scores (30% weight)
python
def _calculate_boundary_importance(self, points: np.ndarray) -> np.ndarray:
    temp_mesh = trimesh.Trimesh(vertices=points)
    hull = temp_mesh.convex_hull
    # Calculate distance from each point to hull surface
    # Closer to boundary = higher score
Points near the convex hull boundary are prioritized
These points are most critical for defining the final convex shape
Interior points are less important for convex hull geometry
Decimation Strategies
Method 1: Quadric Decimation
Best quality: Minimizes geometric error during reduction
How it works: Calculates error quadrics for each vertex, removes vertices that cause least distortion
When it fails: Sometimes doesn't preserve convexity
Method 2: Vertex Clustering
Most reliable: Always produces valid results
How it works: K-means clustering of vertices, use cluster centroids
Trade-off: Less precise geometry but guaranteed to work
Method 3: Random Sampling
Fallback only: When other methods fail
Simple but crude: Randomly select target number of vertices
