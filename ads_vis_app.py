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

#name_root = "NIKJOH"
#name_root = "YOMBAE"
#name_root = "INOVEN"
#name_root = "QIGBIR"

#name_root = "VAHSON" 
#name_root = "XUPSAE" 

#name_root = "AFOVIB"
#name_root = "GIYTIS"
#name_root = "MIHBAG"
#name_root = "REFTUT"
#name_root = "NIKDAM"

#name_root = "KULRIT"
#name_root = "PEJFOA"
#name_root = "MEXJAC"
#name_root = "XIBZUF" # small sample size
#name_root = "NUTQAV"

"""
# MOFs
name = f"{name_root}_fullCoRE"
frame_file_name = f"{home}/repo/WaterProject/NVT_W_method/snapshots/{name}_frame.pdb"
movie_file_name = f"{home}/repo/WaterProject/NVT_W_method/snapshots/{name}_T298_1_NVT_jobs_movie_processed.txt"
"""

#CAU jobs.
#name = "CAU-10-42x42x30-0P"
#name = "CAU-10-42x42x30-10P-1"
#name = "CAU-10-42x42x30-30P-1"
#name = "CAU-10-42x42x30-50P-1"
#name = "CAU-10-42x42x30-70P-1"
name = "CAU-10-42x42x30-100P"

frame_file_name = f"{home}/repo/WaterProject/CAU_calcualtions/Snapshots/{name}_frame.pdb"
movie_file_name = f"{home}/repo/WaterProject/CAU_calcualtions/Snapshots/{name}_T298_1_750_movie-comb_processed.txt"


print(name)
t1 = time.time()

with open(frame_file_name, "r") as pdb_frame:
    properties_array = pdb_frame.readlines()[1].split()[1:]
print(properties_array)
[A, B, C, alpha_deg, beta_deg, gamma_deg] = [float(parameter) for parameter in properties_array]
print([A, B, C, alpha_deg, beta_deg, gamma_deg])
frame_properties_dict = {"A": A, "B": B, "C": C, "alpha_deg": alpha_deg, \
    "beta_deg": beta_deg, "gamma_deg": gamma_deg}
#Properties of the frame. they need to be defined somewhere in the code.
frame_cell_parameter_matrix = make_cell_parameter_matrix(frame_properties_dict)
frame_dims_perpendicular = frame_cell_parameter_matrix.diagonal()

df_frame = pd.read_table(frame_file_name, delim_whitespace=True, skiprows=2, 
    usecols=[1, 2, 4, 5, 6], names=["AtomNum", "AtomType", "x", "y", "z"])

df_frame = df_frame.query("x>3 & x<3 & y>0 & y<0")
#df_frame = df_frame.query("x>3 & x<18 & y>0 & y<13")

t2 = time.time()
#plotting the frame molecule. 
fig = plot_molecule(name, df_frame)
print(f"T2: {t2 - t1}")

#movie_file_name = f"snapshots/{name}_T298_1_918_movie_processed.txt"
#movie_file_name = f"{home}/repo/WaterProject/NVT_W_method/snapshots/{name}_T298_1_918_movie-comb_processed.txt"

O_pos_df = pd.read_table(movie_file_name, \
            delim_whitespace=True, skiprows=1, usecols=[4,5,6], names=["Ow_x", "Ow_y", "Ow_z"]) #Read directly from a bash output. 

#O_pos_df = O_pos_df.query("Ow_x>5 & Ow_x<16 & Ow_y>0.01 & Ow_y<11.8")
O_pos_df = O_pos_df.query("Ow_x>15 & Ow_x<27.5 & Ow_y>35 & Ow_y<42")
O_pos_array = O_pos_df[["Ow_x", "Ow_y", "Ow_z"]].values

#3D histogram
t3d_a = time.time()

def make_3d_histograms(O_pos_array, A, B, C):
    bins = [tuple(np.arange(0, A, 1)), 
    tuple(np.arange(0, B, 1)), tuple(np.arange(0, C, 1))]
    hist, edges = np.histogramdd(O_pos_array, bins=bins, density=True)
    hist[hist==0] = np.exp(-100)
    #compute the mid-points. 
    mid_points = [(edge_val[:-1] + edge_val[1:]) /2 for edge_val in edges]

    #create 3D mesh to match with the 3D histograms (hopefully it matches)
    #mid_point_mesh = np.meshgrid(mid_points[0], mid_points[1], mid_points[2])
    mid_point_mesh = np.meshgrid(mid_points[0], mid_points[1], mid_points[2], indexing="ij")
    return hist, mid_point_mesh

hist , mid_point_mesh = make_3d_histograms(O_pos_array, A, B, C)

#to test if things are reasonable. 
fig.add_trace(go.Volume(
    x=mid_point_mesh[0].flatten(),
    y=mid_point_mesh[1].flatten(),
    z=mid_point_mesh[2].flatten(),
    value=-np.log(hist.flatten()),
    isomin=0,
    isomax=50,
    cmin=0, cmax=50,
    opacity=1.0, #needs to be small to see through all surfaces
    surface_count=17,
    # needs to be a large number for good volume rendering
    colorscale=px.colors.diverging.BrBG,
    name="volume_ads"
    ))

fig.update_layout(width=14 * 96 * 0.5, height=12 * 96 * 0.5, autosize=False, 
paper_bgcolor="white", plot_bgcolor="white"
    )
fig.update_scenes(xaxis_visible=False, yaxis_visible=False, \
    zaxis_visible=False, camera = dict(center=dict(x=0, y=0, z=0),
                            eye=dict(x=0, y=1.25, z=0.0),) #projection=dict(type="orthographic"))
    )

t3d_b = time.time()
print(f"3D histogram time: {t3d_b-t3d_a:.3f}")


t_net_a = time.time()

#Create an array of low free energies and compute the graph. 
mesh_array = np.column_stack((mid_point_mesh[0].flatten(), mid_point_mesh[1].flatten(),\
     mid_point_mesh[2].flatten(), -np.log(hist.flatten()) ))
mesh_array_df = pd.DataFrame(mesh_array, columns=["x", "y", "z", "FreeEnergy"])
energy_cutoff=10
low_energy_mask = mesh_array_df["FreeEnergy"] < energy_cutoff
mesh_array_df = mesh_array_df[low_energy_mask]

if False:
    #Here, we will implement the nearest neighbor network. Later, this function 
    #can be transferred to a different file for network utils. 
    sample_size=min(mesh_array_df.shape[0], 1000)
    NNN = make_NNN(mesh_array_df, frame_cell_parameter_matrix, size=sample_size)
    nodes , edges, node_info = NNN

    #Add the network traces
    plot_network_traces(NNN, fig)

    print(f"Edges: {len(edges)}, Vertices: {node_info.shape[0]}, Beta index: {len(edges) /node_info.shape[0] : .2f}")
    with open("beta_index_vals.txt", "a+") as outfile:
        outfile.write(f"{name} Edges: {len(edges)}, Vertices: {node_info.shape[0]},\
        Beta index: {len(edges) /node_info.shape[0] : .2f}\n")

    t_net_b = time.time()
    print(f"Network computation time: {t_net_b - t_net_a}")
    
#save the main figure. 
#fig.write_image(f"{home}/repo/WaterProject/NVT_W_method/snapshots/vis_images/{name}_vis.jpg",\ 
#    width=900, height=900)
fig.write_image(f"{home}/repo/WaterProject/CAU_calcualtions/images/{name}_vis.jpg", 
    width=1800, height=1800)

#Histograms
#fig_hist = make_subplots(rows=1, cols=3)
fig_hist = make_subplots(rows=2, cols=2)
fig_hist.update_layout(width=14 * 96 * 0.8, height=12 * 96 * 0.8, autosize=False)

#We will now also make 1D histograms
for dim_ind, dim in enumerate(["x", "y", "z"]):
    hist_1D, edges_1D = np.histogram(O_pos_df[f"Ow_{dim}"], bins=np.arange(0, A, 1))
    fig_hist.add_trace(go.Scatter(x=(edges_1D[:-1] + edges_1D[1:])/2, y=hist_1D), #mode="markers+lines",
        row=dim_ind//2 + 1, col=dim_ind % 2 + 1 )
    fig_hist.update_xaxes(title_text=dim, row=dim_ind//2 + 1, col=dim_ind % 2 + 1)
    fig_hist.update_yaxes(title_text="Hist", row=dim_ind//2 + 1, col=dim_ind % 2 + 1 )

free_energies = -np.log(hist.flatten())
hist_en_1D, edges_en_1D = np.histogram(free_energies, bins =  np.linspace(min(free_energies)*1.2, 10) )
fig_hist.add_trace(go.Scatter(x=(edges_en_1D[:-1]+edges_en_1D[1:])/2, y=hist_en_1D), 
    row=2, col=2)
fig_hist.update_xaxes(title_text="Free energy", row=2, col=2)
fig_hist.update_yaxes(title_text="Hist", row=2, col=2)


trdf_a = time.time()
#RDFs
fig_rdf = go.Figure()
#gr, hist_bins = compute_RDF(copy.deepcopy(O_pos_array[1::50, :]), frame_properties_dict)
gr, hist_bins = compute_RDF(mesh_array_df[["x", "y", "z"]].values, frame_properties_dict, \
    dr=1, sample_size=min(mesh_array_df.shape[0], 1000))

#fig_rdf.add_trace(go.Scatter(x=hist_bins[1:], y=gr, mode="markers+lines"))
#fig_rdf.update_xaxes(title_text="r", range=(2.5, 12))
#fig_rdf.update_yaxes(title_text="g(r)", range=(0, 5))

fig_rdf.add_trace(go.Scatter(x=hist_bins[1:], y=gr, mode="markers+lines"))
fig_rdf.update_xaxes(title_text="r", range=(0, 12))
fig_rdf.update_yaxes(title_text="g(r)", range=(-0.5, 5))
fig_rdf.update_layout(autosize=False) #width=7 * 96 * 0.5, height=6 * 96 * 0.5

trdf_b = time.time()
print(f"RDF time: {trdf_b-trdf_a:.3f}")

"""
#VISIT visualization. 
#write everything to dataframe and then into a point 3D file for visualization in Visit. 
mesh_df = pd.DataFrame.from_dict({"x": mid_point_mesh[0].flatten(), 
                                 "y": mid_point_mesh[1].flatten(), 
                                 "z": mid_point_mesh[2].flatten(), 
                                 "value": hist.flatten()})
write_to_point3D(mesh_df, f"Snapshots/{name}_density.3D")
"""

##########
if False:

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

    ener_vol_trace = go.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=V,      isomin=-5000,
        isomax=-3000, cmin=-5000, cmax=0, opacity=0.5, surface_count=17, colorscale=px.colors.diverging.BrBG, 
        visible=True, name="energy_map")

    fig.add_trace(ener_vol_trace)

#turn off the volume map. 
if False:
    next(fig.select_traces({"name": "volume_ads"})).update(visible=False)
    next(fig.select_traces({"name": "network_trace"})).update(visible=False)
    next(fig.select_traces({"name": "network_points_trace"})).update(visible=False)

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
    ener_network_trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines', \
                            line=dict(width = 10, color="black"), visible=True,
                            name="ener_network_trace")
    for (i, j) in ener_edges:
        ener_cart_distance_array  = ener_node_info.loc[i, ["x", "y", "z"]].values - ener_node_info.loc[j, ["x", "y", "z"]].values
        ener_cart_distance = np.sqrt(np.sum(ener_cart_distance_array**2))
        if ener_cart_distance < 4 :
            ener_network_trace['x'] += (ener_node_info.loc[i, "x"], ener_node_info.loc[j, "x"], None)
            ener_network_trace['y'] += (ener_node_info.loc[i, "y"], ener_node_info.loc[j, "y"], None)
            ener_network_trace['z'] += (ener_node_info.loc[i, "z"], ener_node_info.loc[j, "z"], None)

    #We add another trace to show the points. 
    ener_network_points_trace = go.Scatter3d(x=ener_node_info["x"].values, y=ener_node_info["y"].values, 
        z=ener_node_info["z"].values, \
            mode='markers', marker=dict(color='black', size=7, opacity=1),
            name='ener_network_points_trace', visible=True)

    fig.add_traces([ener_network_trace, ener_network_points_trace])
    print(f"Edges: {len(ener_edges)}, Vertices: {ener_node_info.shape[0]}, Beta index: {len(ener_edges) /ener_node_info.shape[0] : .2f}")
    with open("ener_beta_index_vals.txt", "a+") as outfile:
        outfile.write(f"{name} Edges: {len(ener_edges)}, Vertices: {ener_node_info.shape[0]},\
        Beta index: {len(ener_edges) / ener_node_info.shape[0] : .2f}\n")


#%%
#from dash_bootstrap_components import dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    # All elements from the top of the page
    html.H1(children='AdsVis'), 
    html.Div(children=name),
    html.Div([
        html.Div([
            dcc.Graph(
                id='graph1',
                figure=fig
            ),  
            dcc.RangeSlider(
            id='my-range-slider',
            min=0,
            max=50,
            #step=5,
            step=None,
            #value=[0, 10], #50
            value=[0, 50], 
            #marks={0:"0", 25: "25", 50:"50"}
            marks={float(val) : {"label": f"{val}", 'style': {'color': '#77b0b1'}} for val in np.arange(0, 51, 5)}
        ),
        html.Div(id='output-container-range-slider')
        ], className='six columns'), 
        html.Div([
            html.Div(children='''
                Free energy distribution. 
            '''),
            dcc.Graph(
                id='graph2',
                figure=fig_hist
            ),
        ], className='six columns'),
    ], className='row'),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.Div(children='''
            RDFs
        '''),
        dcc.Graph(
            id='graph3',
            figure=fig_rdf
        ),  
    ], className='row'),
])

#Defining actions for slider
@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value')])
def update_output(value):
    next(fig.select_traces({"name": "volume_ads"})).update(isomin=value[0], isomax=value[1])
    #select_data_free_en(isomin=value[0], isomax=value[1])
    #print('You have selected "{}"'.format(value))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False)
    #pass


"""
Issues to be addressed with the app.
1. histogram isn't being binned in 3D.The math for that is rather hard. So, currently, we don't 
see the full histogram. The easy way to solve this is to convert everything to fractional, make 
histogram, and then convert everything back and then plot. 
In the most general sense, it can also be seen as convolution. 
7. (done): will need to confirm now. Take a high loading case: compare w/ RASPA for triclinic. 
Our code will currently have to do this framewise, which we haven't yet added. But, we can write a function
for that easily. Just compute RDF for each frame and then average it. Ideally, it should work. 
Make RDF: see if there are any effects for the previous study as well.
2. Appearance: set correct colors for the atoms and the correct sizes. 

bonds: strategy: make cylidners: page on phone. We can rotate cylinders as shown. 
https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

3. Make sure that the box and/or unit cell are shown. 
5. Set figure sizes properly: adjust layout to bottom-left corner of the screen.
6. Make 1D (done) and 2D histograms to show the molecular density (even x, y, z) is fine.  

"""
# %%
