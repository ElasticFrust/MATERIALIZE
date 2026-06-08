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
            # point_pos[[index,1]]=tempv[[1]]
            index+=1
            # print(tempv[[0]])
            # print(point_pos)
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
            # point_pos[[index,1]]=tempv[[1]]
            index+=1
            # print(tempv[[0]])
            # print(point_pos)
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

def add_edges_to_triangulation(triangulation): ## void, adds "edges" list
    triangulation.edges= np.array([find_triangle_edges(tri) for tri in triangulation.simplices])
    triangulation.rigidities = [[] for tri in triangulation.simplices]
    triangulation.rest_lenghts = [[] for tri in triangulation.simplices]
    return 0
    

def flatten_edges_remove_duplicates(edgelist): # As name implies, but actually we dont really need it. 
    all_edges=copy.deepcopy(edgelist)
    if edgelist.shape[1]==3:
        all_edges=np.reshape(all_edges,(len(all_edges)*3,2))
    #firs reorder so that lower index first
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
    
    
def bare_elastic(multiindex,edge_list,poslist): # This one is not needed either. We need to calculate the bare value for each triangel. Then we should summ them up however we deem fit.
    mu=multiindex[0];
    nu=multiindex[1];
    alpha=multiindex[2];
    beta=multiindex[3];    
    norm= 3/ len(edge_list) # factor of 3 because we want 1/number of triangles, but we work on edges we didnt even have to remove dups
    com=0;
    for idx,edge in enumerate(edge_list):
        vec=np.array(poslist[edge[0]]-poslist[edge[1]])
        com+=norm * (vec[mu]*vec[nu]*vec[alpha]*vec[beta])/np.dot(vec,vec) 
    return com
    
def bare_elastic_local_tensor_componenets(multiindex,triangle_edges,poslist,rigidities=[],rest_lenghts=[]): #Calculating A(s) for specific triangle given 4-index (mu,nu,alpha,beta)
    # REMEBER that greek indices are in base 1 while code in base 0
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

def add_local_tensors_to_triangulation(triangulation_with_edges_list): # Add local elastic A(s) tensor to triangulation structures. Tensor is given in vectorized  componenets: A1= A4x4(1,1)=A_rank4(1,1,1,1), A2=A4x4(1,2)=A4x4(1,3)=A_rank4(1,1,1,2), A3=A4x4(1,4)=A_rank4(1,2,1,2), A4=A4x4(2,4)=A_rank4(1,2,2,2), A5=A4x4(4,4)=A_rank4(2,2,2,2),
    positions=triangulation_with_edges_list.points
    triangulation_with_edges_list.BareElasticTensor=np.array([np.array([bare_elastic_local_tensor_componenets([1,1,1,1],tri_Edges,positions,
                                                                                                triangulation_with_edges_list.rigidities[idx],
                                                                                                triangulation_with_edges_list.rest_lenghts[idx]),
                                                                        bare_elastic_local_tensor_componenets([1,1,1,2],tri_Edges,positions,
                                                                                                triangulation_with_edges_list.rigidities[idx],
                                                                                                triangulation_with_edges_list.rest_lenghts[idx]),
                                                                        bare_elastic_local_tensor_componenets([1,1,2,2],tri_Edges,positions,
                                                                                                triangulation_with_edges_list.rigidities[idx],
                                                                                                triangulation_with_edges_list.rest_lenghts[idx]),
                                                                        bare_elastic_local_tensor_componenets([1,2,2,2],tri_Edges,positions,
                                                                                                triangulation_with_edges_list.rigidities[idx],
                                                                                                triangulation_with_edges_list.rest_lenghts[idx]),
                                                                        bare_elastic_local_tensor_componenets([2,2,2,2],tri_Edges,positions,
                                                                                                triangulation_with_edges_list.rigidities[idx],
                                                                                                triangulation_with_edges_list.rest_lenghts[idx])]) 
                                                            for idx,tri_Edges in enumerate(triangulation_with_edges_list.edges)])
    return 0

def add_delta_As(triangulation):
    mean_tansor=np.mean(triangulation.BareElasticTensor,0)
#print(mean_tansor)
    triangulation.delta_tensor = np.array([local_tensor-mean_tansor for local_tensor in triangulation.BareElasticTensor])
    return 0

def to_vectorized_matrix(mat):
    data = np.array([[mat[2],mat[2],mat[2],0,0,0,0,0,0],
                     [mat[1],mat[1],mat[1],2*mat[3],2*mat[3],2*mat[3],0,0,0],
                     [mat[0],mat[0],mat[0],2*mat[2],2*mat[2],2*mat[2],mat[4],mat[4],mat[4]],
                     [0,0,0, 2*mat[1],2*mat[1],2*mat[1],mat[3],mat[3],mat[3]],
                     [0,0,0,0,0,0,mat[2],mat[2],mat[2]]
                     ])
    offsets = np.array([-6,-3,0,3,6])
    return sp.sparse.dia_matrix((data,offsets),shape=(9,9))

def to_9vector(mat):
    return np.array([mat[0],mat[1],mat[2],mat[1],mat[2],mat[3],mat[2],mat[3],mat[4]])
    

def create_A_and_B_matrices_local(triangulation):
    triangulation.A_matrix = [to_vectorized_matrix(A) for A in triangulation.BareElasticTensor]
    triangulation.B_matrix = [to_vectorized_matrix(dA) for dA in triangulation.delta_tensor]
    return 0

def creat_dA_vec(triangulation):
    triangulation.dA_9vector = [to_9vector(dA) for dA in triangulation.delta_tensor]
    return 0

def calc_actual_elastic_tensor(A, W): #calculate the actual elastic tensor given A (as a 5 component) ans W(9 comps)
        # The result has 6 components so that C[1111]=C1, C[1112]=C2, C[1122]=C3, C[2112]=C4, C[2122]= C5, C[2222]=C6 all others a related by the symmetries: C[abcd]=C[bacd]=C[abdc]=C[cdab]
    return np.array([A[0]*(1 + W[0])**2 + 4*W[3]*(A[1]*(1 + W[0]) + A[2]*W[3]) + 2*(A[2]*(1 + W[0]) + 2*A[3]*W[3])*W[5] + A[4]*W[5]**2 ,

                     A[0]*(1 + W[0])*W[1] + 2*A[2]*W[3]*(1 + 2*W[4]) + A[1]*(1 + W[0] + 2*W[1]*W[3] + 2*(1 + W[0])*W[4]) + 
                                                             A[3]*W[5] + (A[2]*W[1] + 2*A[3]*W[4])*W[5] + (A[2]*(1 + W[0]) + 2*A[3]*W[3] + A[4]*W[5])*W[7],

                     -(A[0]*(1 + W[0])*W[2] + 2*(A[3] + A[1]*W[2])*W[3] + A[4]*W[5] + 2*(A[1] + A[1]*W[0] + A[3]*W[5])*W[6] + 
                                                             (2*A[3]*W[3] + A[4]*W[5])*W[8] + A[2]*(1 + W[2]*W[5] + 4*W[3]*W[6] + W[8] + W[0]*(1 + W[8]))),

                     W[1]*(A[0]*W[1] + A[1]*(2 + 4*W[4])) + 2*A[3]*(1 + 2*W[4])*W[7] + A[4]*W[7]**2 + A[2]*((1 + 2*W[4])**2 + 2*W[1]*W[7]),

                     W[2]*(A[1] + A[0]*W[1] + 2*A[1]*W[4]) + 2*A[1]*W[1]*W[6] + A[4]*W[7]*(1 + W[8]) + A[2]*((2 + 4*W[4])*W[6] + W[2]*W[7] + W[1]*(1 + W[8])) +
                                                              A[3]*(1 + 2*W[6]*W[7] + W[8] + 2*W[4]*(1 + W[8])),

                     A[0]*W[2]**2 + 4*A[3]*W[6] + 4*A[1]*W[2]*W[6] + 4*A[3]*W[6]*W[8] + A[4]*(1 + W[8])**2 + 2*A[2]*(W[2] + 2*W[6]**2 + W[2]*W[8])
                    ])
    

def add_actual_elastic_tensor(triangulation):
    triangulation.ActualElasticTensor = np.array([calc_actual_elastic_tensor(triangulation.BareElasticTensor[idx], triangulation.Ws[idx]) for idx,simp in enumerate(triangulation.simplices)])
    return 0



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
def mat4ind(A,m,n,a,b):
    return A[(m-1)*2+(a-1),(n-1)*2+(b-1)]

def actualelastic4ind(A,W,m,n,a,b):
    S1=mat4ind(A,m,n,a,b)
    S2=0
    S3=0;
    S4=0;
    for i in range(1,3):
        for j in range(1,3):
            S2+= mat4ind(A,m,n,i,j)*mat4ind(W,i,j,a,b)
            S3+= mat4ind(A,a,b,i,j)*mat4ind(W,i,j,m,n)
    for i in range(1,3):
        for j in range(1,3):
            for k in range(1,3):
                for l in range(1,3):
                    S4+=mat4ind(A,k,l,i,j)*mat4ind(W,i,j,a,b)*mat4ind(W,k,l,m,n)
    return S1+S2+S3+S4

def actualmat(A,W):
    return np.array(([[actualelastic4ind(A,W,1,1,1,1), actualelastic4ind(A,W,1,1,1,2), actualelastic4ind(A,W,1,2,1,1), actualelastic4ind(A,W,1,2,1,2)],
                      [actualelastic4ind(A,W,1,1,2,1), actualelastic4ind(A,W,1,1,2,2), actualelastic4ind(A,W,1,2,2,1), actualelastic4ind(A,W,1,2,2,2)],
                      [actualelastic4ind(A,W,2,1,1,1), actualelastic4ind(A,W,2,1,1,2), actualelastic4ind(A,W,2,2,1,1), actualelastic4ind(A,W,2,2,1,2)],
                      [actualelastic4ind(A,W,2,1,2,1), actualelastic4ind(A,W,2,1,2,2), actualelastic4ind(A,W,2,2,2,1), actualelastic4ind(A,W,2,2,2,2)]]))  
def actualmat_to_6vec(mat):
    return np.array([mat[0,0],mat[0,1],mat[1,1],mat[1,2],mat[2,3],mat[3,3]])

def add_actual_matrix(triangulation):
    triangulation.Amatrix=np.array([Amat(avec) for avec in triangulation.BareElasticTensor])
    triangulation.Wmatrix=np.array([Wmat(wvec) for wvec in triangulation.Ws])
    triangulation.actual_elastic_tensor_matrix=np.array([actualmat(triangulation.Amatrix[idx],triangulation.Wmatrix[idx]) for idx,tri in enumerate(triangulation.simplices)])
    triangulation.ActualElasticTensor=np.array([actualmat_to_6vec(mat) for mat in triangulation.actual_elastic_tensor_matrix])
    return 0
        


def analyze_elastic_struct(triangulation): # This is the main function
    add_edges_to_triangulation(triangulation) # make sure triangulation has edge lists
    add_local_tensors_to_triangulation(triangulation) # make sure triangulatio has local elastic tensor
    add_delta_As(triangulation) # calculate \delta A(s)
    create_A_and_B_matrices_local(triangulation) # create sparse 9x9 matrices of A and dA (B)
    creat_dA_vec(triangulation) # create the 9vector dA
    triangulation.dA_vec_all = np.block(triangulation.dA_9vector) #the whole dA vector (1,9xN)
    triangulation.A_matrix_all = sp.sparse.block_diag(triangulation.A_matrix) # the  9N x 9N deal
    one_vec=1/len(triangulation.simplices) * np.ones(len(triangulation.simplices))
    one_vec=one_vec[...,None]
    triangulation.B_matrix_all = sp.sparse.kron(one_vec,sp.sparse.block_array([triangulation.B_matrix]))
    triangulation.C_matrix_all = sp.sparse.linalg.inv(triangulation.A_matrix_all-triangulation.B_matrix_all)
    triangulation.Ws_with_inv = - np.reshape(triangulation.C_matrix_all @ triangulation.dA_vec_all,(-1,9))
    triangulation.Ws_with_spsolve=-np.reshape(sp.sparse.linalg.spsolve(triangulation.A_matrix_all-triangulation.B_matrix_all,triangulation.dA_vec_all,'Natural'),(-1,9))
    triangulation.Ws=triangulation.Ws_with_spsolve
    #add_actual_elastic_tensor(triangulation)  ## FUNCTION HAS AN ERROR
    add_actual_matrix(triangulation)
    triangulation.totalElasticTensor = np.mean(triangulation.ActualElasticTensor,0)
    #directional poisson and youngs, assuming stretching along the 'y' (or "2") direction
    triangulation.PoissonsRatio = ((triangulation.totalElasticTensor[2] * triangulation.totalElasticTensor[3] - 
                                    triangulation.totalElasticTensor[1] * triangulation.totalElasticTensor[4]) / 
                                    (triangulation.totalElasticTensor[0] * triangulation.totalElasticTensor[3] - triangulation.totalElasticTensor[1]**2))
    triangulation.YoungsModulus =  (((triangulation.totalElasticTensor[2]**2) * triangulation.totalElasticTensor[3] - 
                                     2 * triangulation.totalElasticTensor[1] * triangulation.totalElasticTensor[2] * triangulation.totalElasticTensor[4] 
                                      + (triangulation.totalElasticTensor[1]**2) * triangulation.totalElasticTensor[5] + 
                                     triangulation.totalElasticTensor[0] * (triangulation.totalElasticTensor[4]**2 - triangulation.totalElasticTensor[3]
                                                                             * triangulation.totalElasticTensor[5]))/ 
                                     (triangulation.totalElasticTensor[1]**2 - triangulation.totalElasticTensor[0] *
                                      triangulation.totalElasticTensor[3]))
    return 0


                       
        
    


def silly_fun():
    return "this is silly indeed"