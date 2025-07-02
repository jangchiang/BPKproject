# 3. Methodology

## 3.1 Algorithm Overview

The controlled convex hull reconstruction algorithm addresses the fundamental challenge of generating simulation-ready convex hulls with predictable complexity while preserving geometric fidelity. Our approach transforms complex 3D meshes into DEM-compatible convex geometries through a five-stage sequential pipeline:

1. **Intelligent Point Reduction**: Multi-criteria selection of geometrically significant points
2. **Initial Convex Hull Computation**: Standard convex hull generation from reduced point set
3. **Adaptive Mesh Decimation**: Constraint-aware vertex reduction using hierarchical strategies
4. **Minimum Vertex Enforcement**: Ensuring adequate geometric complexity for stable simulation
5. **Final Validation**: Comprehensive convexity and quality verification

The core innovation lies in performing intelligent point reduction *before* convex hull computation, rather than attempting to decimate an already-computed hull. This pre-processing strategy preserves geometric fidelity more effectively because original shape characteristics guide the selection process.

## 3.2 Intelligent Point Reduction

### 3.2.1 Multi-Criteria Scoring Framework

We employ a weighted combination of three complementary scoring criteria to identify geometrically significant points. Each criterion addresses different aspects of shape preservation and spatial distribution.

**Geometric Importance Score (40% weight):**
$$I_g(p_i) = SVM_{importance}(\mathbf{f}(p_i))$$

The feature vector $\mathbf{f}(p_i)$ encapsulates local geometric properties:
- **Local curvature**: Computed via PCA of k-nearest neighbors to identify shape-defining regions
- **Point density**: Neighborhood population within radius r, indicating geometric complexity
- **Centroid distance**: Global position relative to mesh centroid, preserving overall proportions
- **Normal magnitude**: Surface orientation significance for boundary definition

A Support Vector Machine classifier distinguishes geometrically important points using pseudo-labels derived from feature statistics, with importance thresholds set at the 70th percentile of curvature and 50th percentile of density values.

**Spatial Diversity Score (30% weight):**
$$I_s(p_i) = \frac{1}{k}\sum_{j=1}^{k} d(p_i, p_{j})$$

This criterion promotes selection of spatially distributed points by computing the average distance to k-nearest neighbors. Points in less dense regions receive higher scores, preventing clustering and maintaining global shape proportions.

**Boundary Proximity Score (30% weight):**
$$I_b(p_i) = \frac{1}{d_{hull}(p_i) + \epsilon}$$

Where $d_{hull}(p_i)$ represents the distance from point $p_i$ to the convex hull surface of the original point cloud. Points closer to the boundary receive priority as they are most critical for defining the final convex shape.

### 3.2.2 Combined Selection Strategy

The final importance score integrates all three criteria:
$$I_{total}(p_i) = 0.4 \cdot I_g(p_i) + 0.3 \cdot I_s(p_i) + 0.3 \cdot I_b(p_i)$$

Points are ranked by their total importance score, and the top-scoring points are selected until the target vertex count is reached. This ensures that the reduced point set maintains essential geometric characteristics while respecting spatial distribution constraints.

## 3.3 Convex Hull Generation and Constraint Enforcement

### 3.3.1 Initial Convex Hull Computation

The reduced point set undergoes standard convex hull computation using robust algorithms (QuickHull implementation via Trimesh library). This initial hull typically satisfies vertex count constraints due to the pre-reduction phase, but may require further refinement.

### 3.3.2 Adaptive Mesh Decimation

When the initial convex hull exceeds maximum vertex constraints, we apply a three-tier hierarchical decimation strategy:

**Primary Strategy - Quadric Error Decimation:**
We adapt Garland & Heckbert's quadric error metric approach, computing error quadrics for each vertex and iteratively removing vertices that minimize geometric distortion. The algorithm is modified to maintain convexity by recomputing the convex hull after significant decimation steps.

**Secondary Strategy - Clustering-Based Decimation:**
When quadric decimation fails to preserve convexity or achieve target vertex counts:
1. Apply k-means clustering to existing vertices (k = target vertex count)
2. Replace each cluster with its geometric centroid
3. Compute convex hull of centroids

This approach guarantees both convexity and precise vertex count control.

**Fallback Strategy - Constrained Random Sampling:**
Random vertex sampling with immediate convex hull recomputation ensures algorithm robustness when other methods fail, though with reduced geometric fidelity.

### 3.3.3 Minimum Vertex Enforcement

If the resulting mesh has fewer vertices than the minimum constraint (typically 350 for LMGC90), we iteratively:
1. Increase the input point set size for re-reduction
2. Recompute the convex hull
3. Verify vertex count compliance
4. Repeat until adequate complexity is achieved

This ensures sufficient geometric detail for stable DEM simulation while maintaining convexity.

## 3.4 Quality Assessment and Validation

### 3.4.1 Geometric Quality Metrics

**Convexity Preservation Ratio:**
$$C_{ratio} = \frac{V_{original}}{V_{convex\_hull}}$$
Measures how much of the original volume is preserved in the convex approximation.

**Geometric Fidelity:**
$$F_{metric} = 1 - \frac{|A_{original} - A_{final}|}{A_{original}}$$
Quantifies surface area preservation as an indicator of shape similarity.

**Volume-to-Surface Ratio:**
$$VSR = \frac{V_{final}}{A_{final}}$$
Indicates geometric efficiency and mesh quality for collision detection algorithms.

### 3.4.2 Constraint Compliance Verification

**Vertex Constraint Satisfaction:**
$$V_{constraint} = \begin{cases} 
1 & \text{if } V_{min} \leq V_{final} \leq V_{max} \\
0 & \text{otherwise}
\end{cases}$$

**Convexity Verification:**
Rigorous testing using computational geometry algorithms to ensure perfect convexity of the final mesh.

### 3.4.3 LMGC90 Simulation Readiness

**Comprehensive Readiness Score:**
$$S_{LMGC90} = 0.4 \cdot C_{convex} + 0.3 \cdot V_{constraint} + 0.3 \cdot F_{metric}$$

Where $C_{convex}$ is a binary convexity indicator. This score provides a single metric for DEM simulation suitability, combining geometric quality with technical constraints.

**Mesh Quality Assessment:**
Additional quality metrics include aspect ratio analysis, face normal consistency, and geometric regularity measures specifically relevant to collision detection efficiency in discrete element simulations.

## 3.5 Implementation Considerations

### 3.5.1 Computational Complexity

The algorithm exhibits O(n log n) complexity for the convex hull computation and O(n²) for the feature extraction phase, where n is the number of input vertices. The multi-criteria scoring adds minimal overhead compared to the geometric computations.

### 3.5.2 Parameter Sensitivity

Key parameters include:
- k-nearest neighbors count for feature extraction (typically 20)
- SVM training sample ratio (10% of total points)
- Scoring criterion weights (40%/30%/30% distribution)
- Vertex count constraints (350-500 for LMGC90)

These parameters can be adapted for different DEM software requirements or specific application domains.

### 3.5.3 Robustness and Error Handling

The hierarchical decimation strategy ensures algorithm robustness through multiple fallback mechanisms. Quality validation at each stage prevents propagation of geometric errors, and comprehensive logging facilitates debugging and parameter optimization.
