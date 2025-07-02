# 3. Methodology

## 3.1 Algorithm Overview

The controlled convex hull reconstruction methodology addresses the fundamental challenge of generating DEM-compatible convex geometries with predictable vertex complexity while preserving geometric fidelity. Our approach employs a novel two-stage sequential framework that transforms complex 3D meshes into simulation-ready convex hulls through intelligent preprocessing followed by constraint-aware hull generation.

**Core Innovation:**
Unlike traditional approaches that compute full convex hulls followed by post-processing decimation, our method performs intelligent point reduction *prior* to convex hull computation. This pre-processing strategy preserves geometric significance more effectively by incorporating shape characteristics into the selection process before hull generation.

**Two-Stage Framework:**
1. **Stage 1 - Intelligent Point Reduction**: Multi-criteria selection of geometrically significant points using machine learning techniques
2. **Stage 2 - Convex Hull with Constraints**: Constraint-aware convex hull generation with vertex count enforcement

## 3.2 Stage 1: Intelligent Point Reduction

### 3.2.1 Problem Formulation

Given an input point cloud $P = \{p_1, p_2, ..., p_n\}$ with corresponding normals $N = \{n_1, n_2, ..., n_n\}$, Stage 1 seeks to identify an optimal subset $P' \subset P$ that maximizes geometric significance while preparing for efficient convex hull generation:

$$\max_{P' \subset P} \mathcal{I}(P') \text{ subject to } |P'| \leq \tau_{max}$$

where $\mathcal{I}(P')$ represents the aggregate importance score and $\tau_{max}$ is the maximum point threshold.

### 3.2.2 Multi-Criteria Importance Assessment

**Feature Vector Construction:**
Each point $p_i$ is characterized by a four-dimensional feature vector:
$$\mathbf{f}(p_i) = [C(p_i), D(p_i), G(p_i), N(p_i)]^T$$

where:
- $C(p_i)$: Local curvature via PCA eigenvalue analysis
- $D(p_i)$: Point density within neighborhood radius
- $G(p_i)$: Normalized distance to global centroid
- $N(p_i)$: Surface normal magnitude

**Support Vector Machine Classification:**
Geometric importance is assessed using an unsupervised SVM with pseudo-labeling strategy:
$$I_g(p_i) = \text{SVM}_{rbf}(\mathbf{f}(p_i))$$

Pseudo-labels are generated using statistical thresholds:
$$\text{label}(p_i) = \begin{cases} 
1 & \text{if } C(p_i) > \text{percentile}_{75}(C) \text{ or } D(p_i) > \text{percentile}_{50}(D) \\
0 & \text{otherwise}
\end{cases}$$

**Spatial Diversity Scoring:**
To ensure well-distributed point selection:
$$I_s(p_i) = \frac{1}{k}\sum_{j=1}^{k} d(p_i, \text{kNN}_j(p_i))$$

where $\text{kNN}_j(p_i)$ represents the $j$-th nearest neighbor of point $p_i$.

**Boundary Proximity Assessment:**
Points near the convex hull boundary receive priority:
$$I_b(p_i) = \frac{1}{d_{\text{hull}}(p_i) + \epsilon}$$

where $d_{\text{hull}}(p_i)$ is the distance from $p_i$ to the convex hull surface.

### 3.2.3 Combined Importance Scoring

The final importance score integrates all three criteria through weighted combination:
$$I_{\text{total}}(p_i) = 0.4 \cdot I_g(p_i) + 0.3 \cdot I_s(p_i) + 0.3 \cdot I_b(p_i)$$

Points are ranked by $I_{\text{total}}$ and the top-scoring subset forms the seed selection $S_{\text{seed}}$.

### 3.2.4 K-Nearest Neighbors Reinforcement

To enhance spatial coherence and prevent isolated point selection:
$$S_{\text{enhanced}} = S_{\text{seed}} \cup \bigcup_{p \in S_{\text{seed}}} \text{kNN}_5(p)$$

This reinforcement strategy ensures adequate point density along geometric features while maintaining spatial connectivity.

### 3.2.5 Hybrid Merging and Clustering

**Radius-Based Merging:**
Points within radius $\epsilon$ are grouped into local clusters:
$$C_r = \{C_i : C_i = \{p_j \in S_{\text{enhanced}} : \|p_j - c_i\| \leq \epsilon\}\}$$

Each cluster is represented by its geometric centroid:
$$\mathbf{c_i} = \frac{1}{|C_i|}\sum_{p_j \in C_i} p_j$$

**DBSCAN Refinement:**
Density-Based Spatial Clustering removes noise and ensures global coherence:
$$C_{\text{final}} = \{C_j : |C_j| \geq \text{min\_samples} \text{ and } \rho(C_j) \geq \rho_{\min}\}$$

where $\rho(C_j)$ represents cluster density. Parameters are adaptively selected as:
- $\epsilon_{\text{DBSCAN}} = 2.0 \times \epsilon_{\text{radius}}$
- $\text{min\_samples} = \max(3, \lceil 0.01 \times |S_{\text{enhanced}}| \rceil)$

### 3.2.6 Stage 1 Output

Stage 1 produces a geometrically significant, spatially coherent point subset $P_{\text{reduced}}$ with typical reduction ratios of 70-90% while preserving essential shape characteristics.

## 3.3 Stage 2: Convex Hull with Constraints

### 3.3.1 Constraint-Aware Hull Generation

Stage 2 transforms the reduced point set into a convex hull satisfying strict vertex count constraints:
$$V_{\min} \leq |\text{vertices}(\text{ConvexHull}(P_{\text{reduced}}))| \leq V_{\max}$$

For LMGC90 compatibility: $V_{\min} = 350$, $V_{\max} = 500$.

### 3.3.2 Initial Convex Hull Computation

The reduced point set undergoes standard convex hull generation:
$$H_{\text{initial}} = \text{ConvexHull}(P_{\text{reduced}})$$

Due to the intelligent preprocessing in Stage 1, $H_{\text{initial}}$ typically satisfies vertex constraints, though refinement may be required.

### 3.3.3 Adaptive Constraint Enforcement

**Excess Vertex Decimation ($|V| > V_{\max}$):**
When the initial hull exceeds maximum constraints, hierarchical decimation is applied:

*Primary Strategy - Quadric Error Decimation:*
Vertices are iteratively removed based on geometric error minimization while maintaining convexity through periodic hull recomputation.

*Secondary Strategy - K-Means Clustering:*
$$\text{vertices}_{\text{new}} = \text{KMeans}(\text{vertices}_{\text{current}}, k=V_{\max}).\text{cluster\_centers\_}$$

This approach guarantees exact vertex count while preserving spatial distribution.

*Fallback Strategy - Constrained Random Sampling:*
Random vertex selection with immediate convex hull validation ensures algorithm robustness.

**Minimum Vertex Enforcement ($|V| < V_{\min}$):**
When vertex count is insufficient, the input point set is iteratively expanded:
1. Increase $\tau_{\max}$ in Stage 1
2. Recompute $P_{\text{reduced}}$ with larger subset
3. Generate new convex hull
4. Validate vertex count compliance

### 3.3.4 Convexity Validation and Quality Assessment

**Convexity Verification:**
$$\text{is\_convex}(H) = \begin{cases} 
\text{True} & \text{if } H \text{ satisfies convexity constraints} \\
\text{False} & \text{otherwise}
\end{cases}$$

Non-convex results trigger automatic hull recomputation.

**Quality Metrics:**
Geometric preservation is quantified through multiple criteria:

*Convexity Preservation Ratio:*
$$C_{\text{ratio}} = \frac{V_{\text{original}}}{V_{\text{convex\_hull}}}$$

*Geometric Fidelity:*
$$F_{\text{metric}} = 1 - \frac{|A_{\text{original}} - A_{\text{final}}|}{A_{\text{original}}}$$

*Constraint Satisfaction:*
$$V_{\text{constraint}} = \begin{cases} 
1 & \text{if } V_{\min} \leq |V_{\text{final}}| \leq V_{\max} \\
0 & \text{otherwise}
\end{cases}$$

*LMGC90 Simulation Readiness:*
$$S_{\text{LMGC90}} = 0.4 \cdot C_{\text{convex}} + 0.3 \cdot V_{\text{constraint}} + 0.3 \cdot F_{\text{metric}}$$

where $C_{\text{convex}}$ is a binary convexity indicator.

## 3.4 Computational Complexity and Implementation

### 3.4.1 Algorithmic Complexity

**Stage 1 Complexity:**
- Feature extraction: $O(n^2)$ for neighborhood computations
- SVM training: $O(m^3)$ where $m$ is training sample size
- KNN reinforcement: $O(n \log n)$ with spatial indexing
- Hybrid merging: $O(n \log n)$ with efficient clustering

**Stage 2 Complexity:**
- Convex hull computation: $O(k \log k)$ where $k = |P_{\text{reduced}}|$
- Constraint enforcement: $O(v^2)$ where $v$ is vertex count
- Overall complexity: $O(n^2)$ dominated by Stage 1 feature extraction

### 3.4.2 GPU Acceleration

For computational efficiency with large point clouds ($n > 5000$), GPU acceleration is employed using CUDA libraries for:
- Parallel distance computations in feature extraction
- Accelerated KNN search operations  
- Vectorized clustering algorithms
- Automatic CPU fallback for resource-constrained scenarios

### 3.4.3 Parameter Configuration

Key algorithmic parameters include:
- Feature extraction neighborhood size: $k = 20$
- SVM training sample ratio: 10% of total points
- Importance score weights: 40%/30%/30% distribution
- KNN reinforcement neighbors: $k = 5$
- DBSCAN minimum samples: $\max(3, \lceil 0.01n \rceil)$

## 3.5 Integration and Validation

### 3.5.1 Pipeline Integration

The two-stage methodology integrates seamlessly with preprocessing (voxel downsampling) and post-processing (quality validation) components, forming a complete DEM-compatible mesh generation pipeline.

### 3.5.2 Robustness and Error Handling

Multiple fallback mechanisms ensure algorithmic robustness:
- Hierarchical decimation strategies in Stage 2
- Adaptive parameter adjustment based on intermediate results  
- Quality-driven iterative refinement
- Comprehensive validation at each processing stage

This two-stage controlled convex hull methodology provides a principled approach to generating DEM-compatible convex geometries that balances geometric fidelity preservation with computational efficiency while guaranteeing constraint satisfaction for simulation applications.
