#%%

import numpy as np
import pandas as pd
import re
import plotly
import plotly.graph_objects as go
import plotly.express as px
import os 
import copy

_dirname = os.path.dirname(__file__)

"""
We develop the backend for the app. 
"""

def plot_molecule(molecule_name, structures_df, fig=None):
    """Creates a 3D plot of the molecule
    Inputs:
    molecule_name: name of the molecule
    structures_df : dataframe with atomic coordinates. "Atom", "x", "y", "z"
    Basic structure of code taken from: 
    https://www.kaggle.com/mykolazotko/3d-visualization-of-molecules-with-plotly/notebook

    """
    
    atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)  

    #cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red')
    color_scheme = "Hex_Jmol"
    atom_color_df = pd.read_table(f"{_dirname}/color_schemes.dat", skiprows=1, header=0, 
        delim_whitespace=True, usecols=[1, 3, 4])
    atom_color_df = atom_color_df[["ElementSymbol", color_scheme]]
    atom_color_df[color_scheme] = atom_color_df[color_scheme].apply(lambda x: f"#{x}")
    atom_color_dict = atom_color_df.set_index("ElementSymbol").to_dict()[color_scheme]

    #molecule = structures_df[structures_df.molecule_name == molecule_name]
    molecule = structures_df
    coordinates = molecule[['x', 'y', 'z']].values
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    z_coordinates = coordinates[:, 2]
    elements = molecule.AtomType.tolist()
    #radii = [atomic_radii[element] for element in elements]
    radii = [atomic_radii[element] if element in atomic_radii.keys() else 1.2 for element in elements]

    #colors = [cpk_colors[element] if element in cpk_colors.keys() else "black" for element in elements ]
    atom_colors = [atom_color_dict[element] if element in atom_color_dict.keys() else atom_color_dict["other"] for element in elements ]
    
    def get_bonds():
        """Generates a set of bonds from atomic cartesian coordinates"""
        ids = np.arange(coordinates.shape[0])
        bonds = dict()
        coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids
        
        for _ in range(len(ids)):
            coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
            radii_compare = np.roll(radii_compare, -1, axis=0)
            ids_compare = np.roll(ids_compare, -1, axis=0)
            distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)
            bond_distances = (radii + radii_compare) * 1.3
            mask = np.logical_and(distances > 0.1, distances <  bond_distances)
            distances = distances.round(2)
            new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}
            bonds.update(new_bonds)
        return bonds            


    def get_atom_trace():
        """Creates an atom trace for the plot"""
        markers = dict(color=atom_colors, size=[20*radius for radius in radii], symbol='circle', opacity=1.0, \
            line=dict(color='white', width=0))  
        trace = go.Scatter3d(x=x_coordinates, y=y_coordinates, z=z_coordinates, mode='markers', marker=markers,
                             text=elements, name='atoms_trace', visible=True)
        return trace

    def get_bond_trace():
        """"Creates a bond trace for the plot"""
        trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines', #marker=dict(color='grey', size=7, opacity=1),
                              line=dict(width = 10, color="gray"), visible=True,
                             name="bonds_trace")
        for i, j in bonds.keys():
            trace['x'] += (x_coordinates[i], x_coordinates[j], None)
            trace['y'] += (y_coordinates[i], y_coordinates[j], None)
            trace['z'] += (z_coordinates[i], z_coordinates[j], None)
        return trace
    
    bonds = get_bonds()

    atoms_trace = get_atom_trace()
    bonds_trace = get_bond_trace()

    """
    zipped = zip(range(len(elements)), x_coordinates, y_coordinates, z_coordinates)
    annotations_id = [dict(text=num, x=x, y=y, z=z, showarrow=False, yshift=15, font = dict(color = "blue"))
                   for num, x, y, z in zipped]
    
    annotations_length = []
    for (i, j), dist in bonds.items():
        x_middle, y_middle, z_middle = (coordinates[i] + coordinates[j])/2
        annotation = dict(text=dist, x=x_middle, y=y_middle, z=z_middle, showarrow=False, yshift=15)
        annotations_length.append(annotation)   
    """

    updatemenus = list([
        dict(buttons=list([
                #  dict(label = 'Atom indices',
                #       method = 'relayout',
                #       args = [{'scene.annotations': annotations_id}]),
                #  dict(label = 'Bond lengths',
                #       method = 'relayout',
                #       args = [{'scene.annotations': annotations_length}]),
                #  dict(label = 'Atom indices & Bond lengths',
                #       method = 'relayout',
                #       args = [{'scene.annotations': annotations_id + annotations_length}]),
                # dict(label = 'Hide all',
                #      method = 'relayout',
                #      args = [{'scene.annotations': []}])
                dict(method='restyle',
                    label="Frameworks",
                    #visible=True,
                    args=[{'visible':True},[i for i,x in enumerate([atoms_trace, bonds_trace]) if (x.name == "atoms_trace" or x.name == "bonds_trace")]],
                    args2=[{'visible':'legendonly'},[i for i,x in enumerate([atoms_trace, bonds_trace]) if (x.name == "atoms_trace" or x.name == "bonds_trace")]]
                        )

                 ]),
                 direction = 'down',
                 xanchor = 'left',
                 yanchor = 'top'
            ),        
    ])

    data = [atoms_trace, bonds_trace]
    #axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, titlefont=dict(color='white'))
    axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=True, titlefont=dict(color='white'),
        )
    layout = dict(scene=dict(xaxis=axis_params.update({"title_text": "x"}), 
        yaxis=axis_params.update({"title_text": "x"}), zaxis=axis_params.update({"title_text": "x"})), # annotations=annotations_id 
                 showlegend=False, updatemenus=updatemenus, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)') #margin=dict(r=0, l=0, b=0, t=0)

    #This will be modified later to externally supply a figure instance. 
    #fig = go.Figure(data=data, layout=layout)
    if fig is None:
        fig = go.Figure()
     
    fig.add_traces(data)
    fig.update_layout(layout)
    #fig.show()
    ##iplot(fig) 
    return fig   


def make_cell_parameter_matrix(properties_dict):
    """
    Makes a cell paramter matrix from the property dict.
    """
    [A, B, C] = [properties_dict["A"], properties_dict["B"], properties_dict["C"] ]
    [alpha, beta, gamma] = np.array([np.deg2rad(properties_dict['alpha_deg']), 
        np.deg2rad(properties_dict['beta_deg']), np.deg2rad(properties_dict['gamma_deg'])])
    #Cell matrix parameters. 
    tempD=(np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma)

    cell_parameter_matrix = np.array([[A, B*np.cos(gamma), C*np.cos(beta)], 
        [0, B*np.sin(gamma), C*tempD], [0, 0, C*np.sqrt(1-(np.cos(beta))**2-tempD**2)]])
    return cell_parameter_matrix


def compute_periodic_distance(atom1_coords, atom2_coords, 
    frame_dims_perpendicular):
    """
    Computes periodic distances between the two atoms in a given framework.
    """

    absolute_distances = np.abs(atom1_coords - atom2_coords)
    #print(f"A: {absolute_distances.shape}") 

    periodic_absolute_distance_x = np.min(np.column_stack((absolute_distances[:, 0], \
        frame_dims_perpendicular[0]-absolute_distances[:, 0])), axis=1)
    periodic_absolute_distance_y = np.min(np.column_stack((absolute_distances[:, 1], \
        frame_dims_perpendicular[1]-absolute_distances[:, 1])), axis=1)
    periodic_absolute_distance_z = np.min(np.column_stack((absolute_distances[:, 2], \
        frame_dims_perpendicular[2]-absolute_distances[:, 2])), axis=1)

    periodic_absolute_distances = np.column_stack((periodic_absolute_distance_x,\
        periodic_absolute_distance_y, periodic_absolute_distance_z))

    distances = np.linalg.norm(periodic_absolute_distances, axis=1)
    return distances

def compute_RDF(coordinates, frame_properties_dict, dr=1, radius_for_rdf=12, sample_size=500):
    #Next, we will write a for loop to compute the RDF. Ultimately, we will also
    #make this a function. 

    coordinates = copy.deepcopy(coordinates)

    #randomly select sample_size number of points. 
    selection_indices = np.random.choice(coordinates.shape[0], size=sample_size, replace=False)
    coordinates = coordinates[selection_indices, :]

    #compute certain properties for the framework. 
    frame_cell_parameter_matrix = make_cell_parameter_matrix(frame_properties_dict)
    frame_dims_perpendicular = frame_cell_parameter_matrix.diagonal()
    a = frame_cell_parameter_matrix[:, 0]
    b = frame_cell_parameter_matrix[:, 1]
    c = frame_cell_parameter_matrix[:, 2]
    V = np.dot(a, np.cross(b, c)) #Volume of the parallelopeipid

    #initialize distances and histogram:
    hist_bins = np.arange(0, radius_for_rdf, dr)
    nr = np.zeros(hist_bins.shape[0]-1)
    
    for roll_id in np.arange(1, coordinates.shape[0], 1):
    #for roll_id in np.arange(1, 2, 1):
        distances = compute_periodic_distance(coordinates, \
            np.roll(coordinates, roll_id, axis=0), frame_dims_perpendicular)
        nr_new, _ = np.histogram(distances, bins = hist_bins)
        nr += nr_new

    nr /= coordinates.shape[0]  
    #Finally, we will compute the RDF. 
    gr = nr / hist_bins[1:]**2 / (4 * np.pi * dr * coordinates.shape[0] / V)
    return gr, hist_bins

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



if __name__=="__main__":
    import sys
    if sys.platform == 'win32':
        home = 'D:\\'
    else:
        home=os.path.expanduser('~')
    import time
    import matplotlib.pyplot as plt

    atom_color_df = pd.read_table("color_schemes.dat", skiprows=1, header=0, 
    delim_whitespace=True, usecols=[0, 1, 3, 4])

    #name_root = "NIKJOH"
    name_root = "VAHSON" 

    name = f"{name_root}_fullCoRE"

    print(name)

    frame_file_name = f"{home}/repo/WaterProject/NVT_W_method/snapshots/{name}_frame.pdb"

    with open(frame_file_name, "r") as pdb_frame:
        properties_array = pdb_frame.readlines()[1].split()[1:]
    print(properties_array)
    [A, B, C, alpha_deg, beta_deg, gamma_deg] = [float(parameter) for parameter in properties_array]
    print([A, B, C, alpha_deg, beta_deg, gamma_deg])
    frame_properties_dict = {"A": A, "B": B, "C": C, "alpha_deg": alpha_deg, \
        "beta_deg": beta_deg, "gamma_deg": gamma_deg}
    df_frame = pd.read_table(frame_file_name, delim_whitespace=True, skiprows=2, 
        usecols=[1, 2, 4, 5, 6], names=["AtomNum", "AtomType", "x", "y", "z"])
    

    test_array = np.random.random((7,3))
    #d = np.min(np.column_stack((test_array[0], test_array[1])), axis=1)

    movie_file_name = f"{home}/repo/WaterProject/NVT_W_method/snapshots/{name}_T298_1_918_movie-comb_processed.txt"
    O_pos_df = pd.read_table(movie_file_name, \
            delim_whitespace=True, skiprows=1, usecols=[4,5,6], names=["Ow_x", "Ow_y", "Ow_z"]) #Read directly from a bash output. 
    #O_pos_df = O_pos_df.query("Ow_x>5 & Ow_x<16 & Ow_y>0.01 & Ow_y<11.8")
    O_pos_array = O_pos_df[["Ow_x", "Ow_y", "Ow_z"]].values

    gr, hist_bins = compute_RDF(coordinates, frame_properties_dict)

    plt.plot(hist_bins[1:], gr)
    plt.ylim((0, 5))
    plt.xlim((3, 12))

    #to parallelize: distribute id ranges and roll accordingly: and roll to 
# %%
