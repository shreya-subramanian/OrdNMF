import os
import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
sys.path.append("C:/Users/DELL/Desktop/ordnmf_project/OrdNMF/model/OrdNMF")
from OrdNMF import OrdNMF  # Ensure this is correctly set up

# Function to prepare matrix
def prepare_matrix(df, label_col='watch_ratio', user_col='user_id', item_col='video_id'):
    scaler = MinMaxScaler()
    df[label_col] = scaler.fit_transform(df[[label_col]])
    df_aggregated = df.groupby([user_col, item_col], as_index=False)[label_col].mean()
    user_item_matrix = df_aggregated.pivot(index=user_col, columns=item_col, values=label_col).fillna(0)
    user_mapping = {idx: user for idx, user in enumerate(user_item_matrix.index)}
    item_mapping = {idx: item for idx, item in enumerate(user_item_matrix.columns)}
    return user_item_matrix.values, user_mapping, item_mapping

# Load data
base_folder = r"C:\Users\DELL\Desktop\ordnmf_project\KuaiRec 2.0\data"
big_matrix_path = os.path.join(base_folder, "big_matrix.csv")
small_matrix_path = os.path.join(base_folder, "small_matrix.csv")

big_matrix = pd.read_csv(big_matrix_path)
small_matrix = pd.read_csv(small_matrix_path)

big_matrix_prepared, big_user_map, big_item_map = prepare_matrix(big_matrix)
small_matrix_prepared, small_user_map, small_item_map = prepare_matrix(small_matrix)

# RMSE Function
def RMSE(true_matrix, pred_matrix, mask):
    diff = true_matrix[mask] - pred_matrix[mask]
    return np.sqrt(np.mean(diff**2))

# Sparsity Testing
test_sparsity_values = [1e-2, 1e-1, 2e-1, 3e-1]

for sv in test_sparsity_values:
    
    # Create a sparse version of the matrix
    sparser_array = small_matrix_prepared.copy() #copy of small matrix
    total_elements = sparser_array.size #calculates the number of elements in the matrix 
    num_to_mask = int(sv * total_elements) #calculates the number of elemnts to mask with NA values based on sparisity level
    indices = np.array(random.sample(range(total_elements), num_to_mask)) # chooses indices randomly from flattened 1D array to mask 
    row_indices, col_indices = np.unravel_index(indices, sparser_array.shape) #unravles back to 2D 
    sparser_array[row_indices, col_indices] = np.nan # replaces with NaN 
    
    # Convert to sparse CSR matrix for OrdNMF
    dense_matrix_sparse = csr_matrix(np.nan_to_num(sparser_array)) #creates csr matrix, thats what ordNMF implementation takes in 
    
    # Initialize and fit OrdNMF
    K = 5   #latency features 
    model = OrdNMF(K=K) 
    delta = np.linspace(0, 1, 6)  # Example delta, delta evenly spaces out ratings so if we have (1-5) it does (0,0.2,0.4,0.6,0.8,1) matches to category based on value 
    model.delta = delta
    model.fit(dense_matrix_sparse, T=6, precision=10**-5, seed=0, verbose=True, save=False)
    
    # Compute RMSE
    Ew, Eh = model.Ew, model.Eh  #Ew, rows(so in our case users) to each category, Eh does columns(in our case movies) to each genre score 
    pred_matrix = Ew.dot(Eh.T)  #final_matrix is their dot_product
    known_mask = ~np.isnan(sparser_array) # boolean array true if non empty in orginal array, false otheriwse 
    rmse = RMSE(small_matrix_prepared, pred_matrix, known_mask) #how close original to predicted 
    
    # Effective sparsity
    effective_sparsity = np.mean(np.isnan(sparser_array)) 
    print(effective_sparsity,rmse)