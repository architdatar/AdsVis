#%%
#%load_ext autoreload
#%autoreload 2

import os 
import sys
import numpy as np
import scipy 
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import dash
import dash_core_components as dcc
import dash_html_components as html

pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 800)
pd.set_option('display.expand_frame_repr', False)

if sys.platform == 'win32':
    home = 'D:\\'
else:
    home=os.path.expanduser('~')

from functools import reduce

import seaborn as sns
from copy import deepcopy

from scipy.interpolate import griddata as gd

sys.path.append(os.path.join(home, 'repo', 'research_current', 'VisSoft'))
plt.style.use(os.path.join(home, 'repo', 'mplstyles', 'mypaper.mplstyle'))
#plt.style.use(os.path.join(home, 'repo', 'mplstyles', 'mypresentation.mplstyle'))

#sys.path.append(os.path.join(home, 'repo', 'WaterProject', 'NVT_W_method'))
#sys.path.insert(0, os.path.join(home, "Software", "flat_histogram_analysis"))

#from utils_movie import read_framework_file, read_movie_file, write_to_point3D #, plot_molecule
from ads_vis.plotly_utils_test import plot_molecule, compute_RDF, make_cell_parameter_matrix, make_3d_histograms
from ads_vis.network_utils import make_NNN, plot_network_traces

import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import re
import multiprocessing
import copy

import plotly as px
np.random.seed(1)

df_data = pd.read_excel(os.path.join(home, "repo", "WaterProject", "NVT_W_method", "waterData_combined.xlsx"), \
    skiprows=2, header=0, usecols=[0, 1, 3])

for name_root in df_data["Name"].values:

    name = f"{name_root}_fullCoRE"

    print(name)

    try:
        t1 = time.time()
        #frame_file_name = f"{home}/repo/WaterProject/NVT_W_method/snapshots/{name}_frame.pdb"
        frame_file_name = f"{home}/Downloads/movie_files_processed/{name_root}_frame.pdb"

        with open(frame_file_name, "r") as pdb_frame:
            properties_array = pdb_frame.readlines()[1].split()[1:]
        print(properties_array)
        [A, B, C, alpha_deg, beta_deg, gamma_deg] = [float(parameter) for parameter in properties_array]
        print([A, B, C, alpha_deg, beta_deg, gamma_deg])
        frame_properties_dict = {"A": A, "B": B, "C": C, "alpha_deg": alpha_deg, \
            "beta_deg": beta_deg, "gamma_deg": gamma_deg}
        #Properties of the frame. they need to be defined somewhere in the code.
        frame_cell_parameter_matrix = make_cell_parameter_matrix(frame_properties_dict)

        df_frame = pd.read_table(frame_file_name, delim_whitespace=True, skiprows=2, 
            usecols=[1, 2, 4, 5, 6], names=["AtomNum", "AtomType", "x", "y", "z"])

        df_frame = df_frame.query("x>3 & x<18 & y>0 & y<13")

        t2 = time.time()
        #plotting the frame molecule. 
        fig = plot_molecule(name, df_frame)
        print(f"T2: {t2 - t1}")

        movie_file_name = f"{home}/Downloads/movie_files_processed/{name_root}_T298_1_918_movie_processed.txt"

        O_pos_df = pd.read_table(movie_file_name, \
                    delim_whitespace=True, skiprows=1, usecols=[4,5,6], names=["Ow_x", "Ow_y", "Ow_z"]) #Read directly from a bash output. 

        O_pos_array = O_pos_df[["Ow_x", "Ow_y", "Ow_z"]].values

        #3D histogram
        t3d_a = time.time()

        hist , mid_point_mesh = make_3d_histograms(O_pos_array, A, B, C)

        fig.update_layout(width=14 * 96 * 0.5, height=12 * 96 * 0.5, autosize=False)
        t3d_b = time.time()
        print(f"3D histogram time: {t3d_b-t3d_a:.3f}")

        t_net_a = time.time()

        #Create an array of low free energies and compute the graph. 
        mesh_array = np.column_stack(( mid_point_mesh[0].flatten(), mid_point_mesh[1].flatten(),\
            mid_point_mesh[2].flatten(), -np.log(hist.flatten()) ))
        mesh_array_df = pd.DataFrame(mesh_array, columns=["x", "y", "z", "FreeEnergy"])
        energy_cutoff=10
        low_energy_mask = mesh_array_df["FreeEnergy"] < energy_cutoff
        mesh_array_df = mesh_array_df[low_energy_mask]

        #Here, we will implement the nearest neighbor network. Later, this function 
        #can be transferred to a different file for network utils. 
        sample_size=min(mesh_array_df.shape[0], 1000)
        NNN = make_NNN(mesh_array_df, frame_cell_parameter_matrix, size=sample_size)
        nodes , edges, node_info = NNN

        print(f"Edges: {len(edges)}, Vertices: {node_info.shape[0]}, Beta index: {len(edges) /node_info.shape[0] : .2f}")
        with open("beta_index_vals_fullCoRE.txt", "a+") as outfile:
            outfile.write(f"{name} Edges: {len(edges)}, Vertices: {node_info.shape[0]},\
            Beta index: {len(edges) /node_info.shape[0] : .2f}\n")

        t_net_b = time.time()
        print(f"Network computation time: {t_net_b - t_net_a}")
    except Exception as e:
        print(f"{name} {e}")
        continue


    """
    trdf_a = time.time()
    #RDFs
    #gr, hist_bins = compute_RDF(copy.deepcopy(O_pos_array[1::50, :]), frame_properties_dict)
    gr, hist_bins = compute_RDF(mesh_array_df[["x", "y", "z"]].values, frame_properties_dict, \
        dr=1, sample_size=min(mesh_array_df.shape[0], 1000))

    trdf_b = time.time()
    print(f"RDF time: {trdf_b-trdf_a:.3f}")

    ##########
    #Here, we also add the ability to visualize the energy map in the structure. For instance, we 
    #have externally computed the interaction energy of the water molecule with the framework which we will now plot 
    #visualize. 

    energy_file_name = f"{home}/repo/WaterProject/NVT_W_method/free_energy_data/{name}_DDEC-o_298_energy-comb.txt"
    ener_df = pd.read_csv(energy_file_name, header=0, usecols=[1,2,3,4,6,7,9,10]) #Read directly from a bash output. 

    #remove duplicates and only select those without energy drifts.
    ener_df["LastWarning"] = ener_df["LastWarning"].replace(np.nan, "Normal", regex=True)
    ener_df["LastWarning"] = ener_df["LastWarning"].astype("str").astype("category")
    ener_df = ener_df[ener_df["LastWarning"]=="Normal"]
    ener_df.drop_duplicates("Position_index", keep="last", inplace=True)

    #filter for low energy points. 
    #ener_df = ener_df.query("Energy<=0")

    #visualize the energy map. 
    x = ener_df["x"].values
    y = ener_df["y"].values
    z = ener_df["z"].values
    v = ener_df["Energy"].values
    _ , energy_mid_point_mesh = make_3d_histograms(ener_df[["x", "y", "z"]].values, A, B, C)
    X, Y, Z = energy_mid_point_mesh
    V = gd((x,y,z), v, (X.flatten(),Y.flatten(),Z.flatten()), method='nearest')

    #Now, we will also add a network and create the points appropriately. 
    ener_mesh_array = np.column_stack((X.flatten(), Y.flatten(), Z.flatten(), V ))
    ener_mesh_array_df = pd.DataFrame(ener_mesh_array, columns=["x", "y", "z", "Energy"])
    #energy_cutoff=-5000
    low_energy_mask = (ener_mesh_array_df["Energy"] - ener_mesh_array_df["Energy"].min()) < 2500
    ener_mesh_array_df = ener_mesh_array_df[low_energy_mask]

    sample_size=min(ener_mesh_array_df.shape[0], 1000)
    NNN = make_NNN(ener_mesh_array_df, frame_cell_parameter_matrix,
            size=sample_size)

    ener_nodes , ener_edges, ener_node_info = NNN

    #We will now plot these on the figure. This can also be made into a function and written as a utility. 
    print(f"Edges: {len(ener_edges)}, Vertices: {ener_node_info.shape[0]}, Beta index: {len(ener_edges) /ener_node_info.shape[0] : .2f}")
    with open("ener_beta_index_vals_fullCoRE.txt", "a+") as outfile:
        outfile.write(f"{name} Edges: {len(ener_edges)}, Vertices: {ener_node_info.shape[0]},\
        Beta index: {len(ener_edges) / ener_node_info.shape[0] : .2f}\n")

    """
# %%
