import numpy as np
import scipy as sp
import seaborn as sb
import copy


def cut_to_size(points,size):
    points=points[points[::,0]<=size[0]]
    points=points[points[::,0]>=-size[0]]
    points=points[points[::,1]<=size[1]]
    points=points[points[::,1]>=-size[1]]
    return points


def generate_cryratl_points(size,shape,orientation):
    v1=np.array([1,0])
    v2=np.array([shape[0]/2, np.sqrt(3)/2*shape[1]])
    MaxPos=int(round(2*(max(size)/min([np.sqrt(3)/2*shape[1],1]))))
    RotationMat=np.array([[np.cos(orientation), np.sin(orientation)],[-np.sin(orientation),np.cos(orientation)]])
    point_pos=np.zeros((((2*MaxPos)**2),2))
    index=0;
    for n in range(-MaxPos,MaxPos):
        for m in range(-MaxPos,MaxPos):
            tempv= np.dot(RotationMat, n*v1+m*v2)
            point_pos[index]=tempv
            index+=1
    DM=sp.spatial.Delaunay( cut_to_size(point_pos,(size[0]+2,size[1]+2)))
    DM.centroids=np.array([np.mean(DM.points[tri],0) for tri in DM.simplices])
    DM.goods_bool=np.array([((abs(cent[0])<=size[0]) and (abs(cent[1])<=size[1])) for cent in DM.centroids])
    DM.good_idxs=np.where(DM.goods_bool==True)
    DM.all_simplices= DM.simplices
    DM.simplices=DM.simplices[DM.good_idxs]
    return DM

def generate_foam_points2(size,eta):
    v1=np.array([1,0])
    v2=np.array([1/2, np.sqrt(3)/2])
    MaxPos=int(round(2*(max(size)/min([np.sqrt(3)/2,1]))))
    point_pos=np.zeros((((2*MaxPos)**2),2))
    index=0;
    for n in range(-MaxPos,MaxPos):
        for m in range(-MaxPos,MaxPos):
            theta = 2*np.pi + np.random.rand()
            tempv=n*v1+m*v2 +np.array([np.cos(theta),np.sin(theta)])
            point_pos[index]=tempv
            index+=1
    DM=sp.spatial.Delaunay( cut_to_size(point_pos,(size[0]+2,size[1]+2)))
    DM.centroids=np.array([np.mean(DM.points[tri],0) for tri in DM.simplices])
    DM.goods_bool=np.array([((abs(cent[0])<=size[0]) and (abs(cent[1])<=size[1])) for cent in DM.centroids])
    DM.good_idxs=np.where(DM.goods_bool==True)
    DM.all_simplices= DM.simplices
    DM.simplices=DM.simplices[DM.good_idxs]
    return DM

def generate_foam_points(size,eta):
    Max=max(size)
    DM = generate_cryratl_points(size,(1,1),0);
    counter=0;
    for point in DM.points:
        theta= 2 * np.pi *  np.random.rand()
        DM.points[counter]=point + eta * np.array([np.cos(theta), np.sin(theta)])
        counter+=1
    return DM

def find_triangle_edges(triangle):
    return np.array([(a, b) for idx, a in enumerate(triangle) for b in triangle[idx + 1:]])

def add_edges_to_triangulation(triangulation):
    triangulation.edges= np.array([find_triangle_edges(tri) for tri in triangulation.simplices])
    triangulation.rigidities = [[] for tri in triangulation.simplices]
    triangulation.rest_lenghts = [[] for tri in triangulation.simplices]
    return 0


# ---------------------------------------------------------------------------
# Vectorized helpers for the Woodbury-optimized solver
# ---------------------------------------------------------------------------

def _compute_local_tensors_vectorized(triangulation):
    """Compute bare elastic tensors for all triangles at once (vectorized).

    Returns (N, 5) array of local tensor components [a0, a1, a2, a3, a4].

    Each triangle's elastic tensor is normalized by its area (1/Omega_s),
    so that it represents the elastic energy DENSITY (per unit area).
    This ensures that small triangles (denser spring network per unit area)
    correctly get a larger elastic modulus than large triangles.
    """
    N = len(triangulation.simplices)
    edges = triangulation.edges  # (N, 3, 2) node-index pairs
    positions = triangulation.points

    # Edge vectors: (N, 3, 2)
    node_a = edges[:, :, 0]
    node_b = edges[:, :, 1]
    vecs = positions[node_a] - positions[node_b]  # (N, 3, 2)

    vx = vecs[:, :, 0]  # (N, 3)
    vy = vecs[:, :, 1]  # (N, 3)

    # Rigidities: default [1,1,1] per triangle
    rigs = np.ones((N, 3))
    for i, r in enumerate(triangulation.rigidities):
        if len(r) > 0:
            rigs[i] = r

    # Length squared: default = actual edge length squared
    length2 = np.sum(vecs**2, axis=2)  # (N, 3)
    for i, rl in enumerate(triangulation.rest_lenghts):
        if len(rl) > 0:
            length2[i] = np.array(rl)**2

    # Compute triangle areas for proper normalization
    tri_pts = positions[triangulation.simplices]  # (N, 3, 2)
    v1 = tri_pts[:, 1] - tri_pts[:, 0]  # (N, 2)
    v2 = tri_pts[:, 2] - tri_pts[:, 0]  # (N, 2)
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])  # (N,)
    triangulation.triangle_areas = areas

    # Use 1/area instead of 1/16 for correct energy density normalization
    factor = rigs / length2 / areas[:, None]  # (N, 3)

    local_tensors = np.column_stack([
        np.sum(factor * vx**4,             axis=1),  # a0: (1,1,1,1)
        np.sum(factor * vx**3 * vy,        axis=1),  # a1: (1,1,1,2)
        np.sum(factor * vx**2 * vy**2,     axis=1),  # a2: (1,1,2,2)
        np.sum(factor * vx * vy**3,        axis=1),  # a3: (1,2,2,2)
        np.sum(factor * vy**4,             axis=1),  # a4: (2,2,2,2)
    ])  # (N, 5)

    return local_tensors


def _batch_to_9x9(vecs5):
    """Convert (N, 5) tensor components to (N, 9, 9) dense matrices.

    The 9x9 matrix has block structure (3x3 blocks, each scalar * I_3):
        [[a0*I    2*a1*I   a2*I ],
         [a1*I    2*a2*I   a3*I ],
         [a2*I    2*a3*I   a4*I ]]
    i.e. kron(M, I_3) where M = [[a0, 2a1, a2], [a1, 2a2, a3], [a2, 2a3, a4]]
    """
    N = vecs5.shape[0]
    mat = np.zeros((N, 9, 9))

    a0 = vecs5[:, 0]
    a1 = vecs5[:, 1]
    a2 = vecs5[:, 2]
    a3 = vecs5[:, 3]
    a4 = vecs5[:, 4]

    for i in range(3):
        mat[:, i,     i]     = a0       # block (0,0): a0 * I_3
        mat[:, i,     3+i]   = 2*a1     # block (0,1): 2a1 * I_3
        mat[:, i,     6+i]   = a2       # block (0,2): a2 * I_3
        mat[:, 3+i,   i]     = a1       # block (1,0): a1 * I_3
        mat[:, 3+i,   3+i]   = 2*a2     # block (1,1): 2a2 * I_3
        mat[:, 3+i,   6+i]   = a3       # block (1,2): a3 * I_3
        mat[:, 6+i,   i]     = a2       # block (2,0): a2 * I_3
        mat[:, 6+i,   3+i]   = 2*a3     # block (2,1): 2a3 * I_3
        mat[:, 6+i,   6+i]   = a4       # block (2,2): a4 * I_3

    return mat


def _batch_to_9vec(vecs5):
    """Convert (N, 5) tensor components to (N, 9) vectors.

    Pattern: [a0, a1, a2, a1, a2, a3, a2, a3, a4]
    """
    a0 = vecs5[:, 0]
    a1 = vecs5[:, 1]
    a2 = vecs5[:, 2]
    a3 = vecs5[:, 3]
    a4 = vecs5[:, 4]
    return np.column_stack([a0, a1, a2, a1, a2, a3, a2, a3, a4])


def _woodbury_solve(A_blocks, B_blocks, dA_vecs, area_weights=None):
    """Solve (A - B_full) W = -dA using Woodbury decomposition.

    A_full = block_diag(A_1, ..., A_N)  [9N x 9N, block diagonal]
    B_full = U @ V  where:
        U = [I_9; I_9; ...; I_9]                          [9N x 9]
        V = [w_1*B_1, w_2*B_2, ..., w_N*B_N]              [9 x 9N]
    with w_t = Omega_t / V_total (area weights).

    By Woodbury: (A - UV)^{-1} = A^{-1} + A^{-1} U (I - V A^{-1} U)^{-1} V A^{-1}

    Parameters:
        A_blocks: (N, 9, 9) per-triangle A matrices
        B_blocks: (N, 9, 9) per-triangle delta-A matrices (B = dA matrices)
        dA_vecs:  (N, 9) per-triangle dA vectors
        area_weights: (N,) area weights Omega_s / V_total. If None, uses 1/N.

    Returns:
        W: (N, 9) solution vectors per triangle
    """
    N = A_blocks.shape[0]

    if area_weights is None:
        area_weights = np.full(N, 1.0 / N)

    # Step 1: Invert each 9x9 A block
    # Regularize to handle degenerate triangles (nearly collinear edges)
    eps = 1e-14 * np.max(np.abs(A_blocks))
    A_blocks_reg = A_blocks + eps * np.eye(9)[None, :, :]
    A_inv = np.linalg.inv(A_blocks_reg)  # (N, 9, 9)

    # Step 2: y_i = A_i^{-1} @ dA_i for each triangle
    y = np.einsum('nij,nj->ni', A_inv, dA_vecs)  # (N, 9)

    # Step 3: V @ y = sum_j w_j * B_j @ y_j  (a 9-vector)
    Vy = np.einsum('n,nij,nj->i', area_weights, B_blocks, y)  # (9,)

    # Step 4: S = V @ A^{-1} @ U = sum_j w_j * B_j @ A_j^{-1}  (9x9)
    S = np.einsum('n,nij,njk->ik', area_weights, B_blocks, A_inv)  # (9, 9)

    # Step 5: Solve (I_9 - S) z = Vy
    z = np.linalg.solve(np.eye(9) - S, Vy)  # (9,)

    # Step 6: W_i = -(y_i + A_i^{-1} @ z)
    # Note: no 1/N factor because U_s = I_9 (area weights absorbed into V)
    correction = np.einsum('nij,j->ni', A_inv, z)  # (N, 9)
    W = -(y + correction)  # (N, 9)

    return W


def _compute_actual_elastic_tensor_vectorized(bare_tensors, Ws):
    """Compute actual elastic tensor for all triangles (vectorized).

    Uses the 4-index tensor contraction:
        C_{mnab} = A_{mnab} + A_{mnij}W_{ijab} + A_{abij}W_{ijmn} + A_{klij}W_{ijab}W_{klmn}

    The 4x4 matrix uses index mapping: M[(m-1)*2+(a-1), (n-1)*2+(b-1)] = A_{m,n,a,b}
    So M.reshape(2,2,2,2) gives T[m,a,n,b] = A_{m+1, n+1, a+1, b+1}.

    Parameters:
        bare_tensors: (N, 5) bare elastic tensor components
        Ws: (N, 9) W solution vectors

    Returns:
        (N, 6) actual elastic tensor components per triangle
    """
    N = bare_tensors.shape[0]
    a = bare_tensors
    w = Ws

    # Build A_mat (N, 4, 4) from 5-component bare tensor
    A_mat = np.zeros((N, 4, 4))
    A_mat[:, 0, 0] = a[:, 0]
    A_mat[:, 0, 1] = a[:, 1]
    A_mat[:, 0, 2] = a[:, 1]
    A_mat[:, 0, 3] = a[:, 2]
    A_mat[:, 1, 0] = a[:, 1]
    A_mat[:, 1, 1] = a[:, 2]
    A_mat[:, 1, 2] = a[:, 2]
    A_mat[:, 1, 3] = a[:, 3]
    A_mat[:, 2, 0] = a[:, 1]
    A_mat[:, 2, 1] = a[:, 2]
    A_mat[:, 2, 2] = a[:, 2]
    A_mat[:, 2, 3] = a[:, 3]
    A_mat[:, 3, 0] = a[:, 2]
    A_mat[:, 3, 1] = a[:, 3]
    A_mat[:, 3, 2] = a[:, 3]
    A_mat[:, 3, 3] = a[:, 4]

    # Build W_mat (N, 4, 4) from 9-component W vector
    W_mat = np.zeros((N, 4, 4))
    W_mat[:, 0, 0] = w[:, 0]
    W_mat[:, 0, 1] = w[:, 1]
    W_mat[:, 0, 2] = w[:, 3]
    W_mat[:, 0, 3] = w[:, 4]
    W_mat[:, 1, 0] = w[:, 1]
    W_mat[:, 1, 1] = w[:, 2]
    W_mat[:, 1, 2] = w[:, 4]
    W_mat[:, 1, 3] = w[:, 5]
    W_mat[:, 2, 0] = w[:, 3]
    W_mat[:, 2, 1] = w[:, 4]
    W_mat[:, 2, 2] = w[:, 6]
    W_mat[:, 2, 3] = w[:, 7]
    W_mat[:, 3, 0] = w[:, 4]
    W_mat[:, 3, 1] = w[:, 5]
    W_mat[:, 3, 2] = w[:, 7]
    W_mat[:, 3, 3] = w[:, 8]

    # Reshape to 4-index tensors: (N, 2, 2, 2, 2)
    # T[m, a, n, b] = A_{m+1, n+1, a+1, b+1} (the physical elastic tensor)
    A4 = A_mat.reshape(N, 2, 2, 2, 2)
    W4 = W_mat.reshape(N, 2, 2, 2, 2)

    # C_{mnab} = A_{mnab} + sum_{ij} A_{mnij}*W_{ijab} + sum_{ij} A_{abij}*W_{ijmn}
    #          + sum_{ijkl} A_{klij}*W_{ijab}*W_{klmn}
    # In T notation: T[m,a,n,b] corresponds to physical index (m+1, n+1, a+1, b+1)
    # S2: sum over (i,j) of T_A[m,i,n,j] * T_W[i,a,j,b]
    S2 = np.einsum('tminj,tiajb->tmanb', A4, W4)
    # S3: sum over (i,j) of T_A[a,i,b,j] * T_W[i,m,j,n]
    S3 = np.einsum('taibj,timjn->tmanb', A4, W4)
    # S4: sum over (i,j,k,l) of T_A[k,i,l,j] * T_W[i,a,j,b] * T_W[k,m,l,n]
    S4 = np.einsum('tkilj,tiajb,tkmln->tmanb', A4, W4, W4)

    C4 = A4 + S2 + S3 + S4  # (N, 2, 2, 2, 2)
    C_mat = C4.reshape(N, 4, 4)

    # Extract 6 independent components:
    # C[1111]=M[0,0], C[1112]=M[0,1], C[1122]=M[1,1], C[2112]=M[1,2], C[2122]=M[2,3], C[2222]=M[3,3]
    actual = np.column_stack([
        C_mat[:, 0, 0],
        C_mat[:, 0, 1],
        C_mat[:, 1, 1],
        C_mat[:, 1, 2],
        C_mat[:, 2, 3],
        C_mat[:, 3, 3],
    ])  # (N, 6)

    return actual


def analyze_elastic_struct(triangulation):
    """Main analysis function — Woodbury-optimized version.

    Replaces the O(N^3) sparse inverse/solve with an O(N) Woodbury decomposition
    exploiting the fact that A is block-diagonal and B is rank-9.
    """
    # Step 1: Build edge lists
    add_edges_to_triangulation(triangulation)

    # Step 2: Compute local bare elastic tensors (vectorized)
    # (also computes and stores triangulation.triangle_areas)
    triangulation.BareElasticTensor = _compute_local_tensors_vectorized(triangulation)
    N = len(triangulation.simplices)
    areas = triangulation.triangle_areas  # (N,)
    area_weights = areas / areas.sum()    # (N,) normalized to sum to 1

    # Step 3: Compute delta tensors (area-weighted mean)
    mean_tensor = np.average(triangulation.BareElasticTensor, weights=areas, axis=0)
    triangulation.delta_tensor = triangulation.BareElasticTensor - mean_tensor

    # Step 4: Build (N, 9, 9) A and B block matrices and (N, 9) dA vectors
    A_blocks = _batch_to_9x9(triangulation.BareElasticTensor)  # (N, 9, 9)
    B_blocks = _batch_to_9x9(triangulation.delta_tensor)       # (N, 9, 9)
    dA_vecs  = _batch_to_9vec(triangulation.delta_tensor)      # (N, 9)

    # Step 5: Woodbury solve (area-weighted coupling)
    triangulation.Ws = _woodbury_solve(A_blocks, B_blocks, dA_vecs,
                                       area_weights=area_weights)  # (N, 9)

    # Step 6: Compute actual elastic tensor (vectorized)
    triangulation.ActualElasticTensor = _compute_actual_elastic_tensor_vectorized(
        triangulation.BareElasticTensor, triangulation.Ws
    )  # (N, 6)

    # Step 7: Area-weighted mean over all triangles
    triangulation.totalElasticTensor = np.average(
        triangulation.ActualElasticTensor, weights=areas, axis=0
    )

    # Step 8: Poisson's ratio and Young's modulus
    C = triangulation.totalElasticTensor
    triangulation.PoissonsRatio = (
        (C[2] * C[3] - C[1] * C[4]) /
        (C[0] * C[3] - C[1]**2)
    )
    triangulation.YoungsModulus = (
        (C[2]**2 * C[3] - 2*C[1]*C[2]*C[4] + C[1]**2 * C[5] +
         C[0] * (C[4]**2 - C[3]*C[5])) /
        (C[1]**2 - C[0] * C[3])
    )

    return 0


# ---------------------------------------------------------------------------
# Original (non-vectorized) functions preserved for reference/compatibility
# ---------------------------------------------------------------------------

def bare_elastic(multiindex,edge_list,poslist):
    mu=multiindex[0];
    nu=multiindex[1];
    alpha=multiindex[2];
    beta=multiindex[3];
    norm= 3/ len(edge_list)
    com=0;
    for idx,edge in enumerate(edge_list):
        vec=np.array(poslist[edge[0]]-poslist[edge[1]])
        com+=norm * (vec[mu]*vec[nu]*vec[alpha]*vec[beta])/np.dot(vec,vec)
    return com

def bare_elastic_local_tensor_componenets(multiindex,triangle_edges,poslist,rigidities=[],rest_lenghts=[]):
    mu=multiindex[0]-1;
    nu=multiindex[1]-1;
    alpha=multiindex[2]-1;
    beta=multiindex[3]-1;
    com=0;
    restlength_flag=1
    if rigidities ==[]:
        rigidities=[1,1,1]
    if rest_lenghts ==[]:
        restlength_flag=0
    for idx,edge in enumerate(triangle_edges):
        vec=np.array(poslist[edge[0]]-poslist[edge[1]])
        if restlength_flag:
            length2= rest_lenghts[idx]**2
        else:
            length2=(1-restlength_flag)*(np.dot(vec,vec))
        com+= rigidities[idx]*(vec[mu]*vec[nu]*vec[alpha]*vec[beta])/length2 /16
    return com

def flatten_edges_remove_duplicates(edgelist):
    all_edges=copy.deepcopy(edgelist)
    if edgelist.shape[1]==3:
        all_edges=np.reshape(all_edges,(len(all_edges)*3,2))
    for idx,edge in enumerate(all_edges):
        if edge[0]>edge[1]:
            t=edge[0]
            all_edges[idx,0]=edge[1]
            all_edges[idx,1]=t
    index_list=[0]
    edge_list_final=[all_edges[0]]
    for idx,edge in enumerate(all_edges[1::]):
        tests=edge==edge_list_final
        test_final= np.array([test.all() for  test in tests])
        if ~(test_final).any():
            index_list.append(idx+1)
            edge_list_final.append(edge)
    return np.array(edge_list_final)


def Amat(Avec):
    return np.array([[ Avec[0], Avec[1], Avec[1], Avec[2]],
                      [Avec[1], Avec[2], Avec[2], Avec[3]],
                      [Avec[1], Avec[2], Avec[2], Avec[3]],
                      [Avec[2], Avec[3], Avec[3], Avec[4]]])
def Wmat(Avec):
    return np.array([[ Avec[0], Avec[1], Avec[3], Avec[4]],
                      [Avec[1], Avec[2], Avec[4], Avec[5]],
                      [Avec[3], Avec[4], Avec[6], Avec[7]],
                      [Avec[4], Avec[5], Avec[7], Avec[8]]])

def silly_fun():
    return "this is silly indeed"
