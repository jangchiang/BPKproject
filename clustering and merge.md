# 3.6 Hybrid Merging and Clustering Methodology

## 3.6.1 Theoretical Foundation

The hybrid merging methodology addresses the challenge of intelligent point cloud simplification while preserving geometric features through a dual-stage clustering approach. This method combines radius-based point merging with density-based spatial clustering to achieve optimal point reduction that maintains both local geometric relationships and global spatial coherence.

**Problem Formulation:**
Given a point set $P = \{p_1, p_2, ..., p_n\}$ with corresponding normals $N = \{n_1, n_2, ..., n_n\}$, the objective is to find a reduced set $P' \subset P$ such that:

$$\min |P'| \text{ subject to } \mathcal{G}(P, P') \geq \tau$$

where $\mathcal{G}(P, P')$ represents geometric fidelity preservation and $\tau$ is the quality threshold.

## 3.6.2 Dual-Stage Clustering Strategy

### Stage 1: Radius-Based Merging

The first stage performs local geometric clustering using spatial proximity criteria. Points within radius $\epsilon$ are grouped into clusters, with each cluster represented by its geometric centroid:

$$C_r = \{C_i : C_i = \{p_j \in P : \|p_j - c_i\| \leq \epsilon\}\}$$

where $c_i$ are cluster centers determined iteratively. The optimal radius $\epsilon$ is estimated using nearest neighbor distance analysis:

$$\epsilon_{opt} = \text{percentile}(\{d_{nn}(p_i)\}_{i=1}^n, 50)$$

where $d_{nn}(p_i)$ represents the distance to the nearest neighbor of point $p_i$.

**Centroid Computation:**
For each cluster $C_i$, the representative point and normal are computed as:

$$\mathbf{c_i} = \frac{1}{|C_i|}\sum_{p_j \in C_i} p_j$$

$$\mathbf{n_i} = \frac{\sum_{p_j \in C_i} n_j}{\|\sum_{p_j \in C_i} n_j\|}$$

### Stage 2: DBSCAN Refinement

The second stage applies Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to the merged points from Stage 1, providing global spatial coherence and noise removal:

$$C_d = \{C_j : |C_j| \geq \text{min\_samples} \text{ and } \rho(C_j) \geq \rho_{min}\}$$

where $\rho(C_j)$ represents the density of cluster $C_j$ and $\rho_{min}$ is the minimum density threshold.

**Parameter Selection:**
DBSCAN parameters are adaptively determined:
- $\epsilon_{DBSCAN} = 2.0 \times \epsilon_{radius}$
- $\text{min\_samples} = \max(3, \lceil 0.01 \times |P_1| \rceil)$

where $P_1$ is the point set after radius merging.

## 3.6.3 GPU-Accelerated Implementation

For computational efficiency with large point clouds ($n > 1000$), GPU acceleration is employed using CUDA libraries:

**Distance Computation:**
```
For each point p_i:
    distances = ||P - p_i||_2  // Vectorized on GPU
    neighbors = {p_j : distances[j] ≤ ε}
    cluster_centroid = mean(neighbors)
```

**Performance Optimization:**
- Memory coalescing for optimal GPU utilization
- Batch processing to manage GPU memory constraints
- Automatic CPU fallback for resource-limited scenarios

## 3.6.4 Quality Assessment Metrics

### Geometric Fidelity Preservation

**Reduction Efficiency:**
$$R_{eff} = \frac{|P| - |P'|}{|P|}$$

**Coverage Quality:**
$$Q_{cov} = \frac{1}{|P|}\sum_{p_i \in P} \exp\left(-\frac{\min_{p_j \in P'} \|p_i - p_j\|}{\sigma}\right)$$

where $\sigma$ is the characteristic length scale of the point cloud.

**Spatial Distribution Preservation:**
$$S_{dist} = 1 - \frac{\text{std}(d_{nn}(P'))}{\text{mean}(d_{nn}(P'))}$$

where $d_{nn}(P')$ represents nearest neighbor distances in the reduced set.

### Boundary Completeness

**Boundary Representation Ratio:**
$$B_{ratio} = \frac{|P' \cap B_{hull}|}{|B_{hull}|}$$

where $B_{hull}$ denotes points on or near the convex hull boundary.

## 3.6.5 Adaptive Parameter Selection

The algorithm adapts parameters based on point cloud characteristics:

**Density-Based Adaptation:**
$$\epsilon_{adaptive} = \epsilon_{base} \times \left(1 + \alpha \cdot \log\left(\frac{\rho_{local}}{\rho_{global}}\right)\right)$$

where $\rho_{local}$ and $\rho_{global}$ are local and global point densities, and $\alpha = 0.2$ is the adaptation coefficient.

**Complexity-Aware Scaling:**
For high-curvature regions, the radius is reduced to preserve geometric detail:
$$\epsilon_{local} = \epsilon_{adaptive} \times (1 - \beta \cdot \mathcal{K}(p_i))$$

where $\mathcal{K}(p_i)$ is the local curvature measure and $\beta = 0.3$ is the curvature sensitivity parameter.

## 3.6.6 Integration with Convex Hull Pipeline

The hybrid merging serves as a preprocessing step before convex hull generation, ensuring that the input point set is optimally distributed for high-quality hull computation while maintaining geometric significance.

**Pipeline Integration:**
1. **Input**: Important points from multi-criteria selection
2. **Radius merging**: Local geometric clustering
3. **DBSCAN refinement**: Global spatial optimization
4. **Output**: Reduced point set for convex hull computation

**Quality Validation:**
The merged point set is validated against geometric preservation criteria before proceeding to convex hull generation. If quality metrics fall below threshold values, parameters are automatically adjusted and the process is repeated.

## 3.6.7 Computational Complexity

The hybrid merging algorithm exhibits computational complexity of:
- **Radius merging**: $O(n^2)$ for naive implementation, $O(n \log n)$ with spatial indexing
- **DBSCAN refinement**: $O(n \log n)$ for well-distributed data
- **Overall complexity**: $O(n \log n)$ with optimized implementation

**Memory Requirements:**
Space complexity is $O(n)$ for point storage and $O(k)$ for cluster management, where $k \ll n$ is the number of clusters.

## 3.6.8 Robustness and Error Handling

The algorithm incorporates multiple fallback strategies to ensure robust operation:

1. **Parameter failure**: Automatic parameter re-estimation using alternative percentiles
2. **GPU memory exhaustion**: Graceful degradation to CPU processing with batch management
3. **Clustering failure**: Fallback to uniform subsampling with importance weighting

**Convergence Guarantee:**
The iterative parameter adaptation process is guaranteed to converge due to the bounded nature of the geometric domain and the monotonic improvement in quality metrics.

This hybrid merging methodology provides an essential preprocessing step that bridges the gap between raw point cloud data and convex hull generation requirements, ensuring optimal geometric fidelity while maintaining computational efficiency suitable for large-scale DEM simulation applications.
