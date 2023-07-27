import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Union, List, Dict
import shapely.ops
from tiresias.data import LABO_GALERIE_SHAPEFILE


FILEPATH = LABO_GALERIE_SHAPEFILE['filepath']
CRS_INFO = LABO_GALERIE_SHAPEFILE['crs_info']
COLUMN_ID = LABO_GALERIE_SHAPEFILE['column_id']

def load_shapefile(filepath: str = FILEPATH, column_id: str = COLUMN_ID) -> List[gpd.GeoDataFrame]:
    """
    Load the file from the specified `filepath` and check for uniqueness of the specified identifier.
    If the identifier is not unique, the corresponding geometries will be geometrically merged.
    """
    raw_gdf = gpd.read_file(filepath, crs_wkt=CRS_INFO)
    return raw_gdf, check_id_unique(gdf=raw_gdf, column_id=column_id)


def check_id_unique(gdf: gpd.GeoDataFrame, column_id: str) -> gpd.GeoDataFrame:
    """
    Check if the specified identifier in the GeoDataFrame is unique.
    If it is not unique, merge the geometries of non-unique identifiers.
    """
    if gdf[column_id].is_unique:
        return gdf
    else:
        id_nonunique = { k:v for (k,v) in gdf[column_id].value_counts().to_dict().items() if v >1}
        return merge_geodataframe(gdf=gdf, id_nonunique=id_nonunique, column_id=column_id)
        

def merge_geodataframe(gdf: gpd.GeoDataFrame, id_nonunique: Dict[str, List[str]], column_id: str) -> gpd.GeoDataFrame:
    """ 
    Return a new GeoDataFrame with merged geometries and only id and geometry columns.
    """
    grouped = gdf.groupby(column_id)  
    new_geometries = []
    new_identifiants = []
    for identifier, group in grouped:
        # TODO check that identifier should be in 3 digits and avoid GAN1 GAN2 etc
        if len(identifier)>3:
            continue
        if identifier in id_nonunique.keys():
            merged_geometry = group['geometry'].unary_union
            new_geometries.append(merged_geometry)
            new_identifiants.append(identifier)
        else:
            new_geometries.append(group['geometry'].iloc[0])
            new_identifiants.append(identifier) 
    new_gdf = gpd.GeoDataFrame({column_id: new_identifiants, 'geometry': new_geometries}, crs=gdf.crs)
    return new_gdf


def get_neighbors(gdf, distance_max=0.1, column_id=COLUMN_ID ) -> Dict[str, List[str]]:
    """
    Return neighbors for each item in the GeoDataFrame within a maximum distance.

    Parameters:
        gdf: The GeoDataFrame containing the geometries to find neighbors for.
        distance_max: The maximum distance for considering items as neighbors. Default is 0.1.
        column_id: The name of the column containing the unique identifiers. Default is COLUMN_ID.

    """
    relations_neigbhors = {}
    geometry_by_id = gdf.set_index(column_id)['geometry'].apply(lambda x: x).to_dict()   
    for reference_id, reference_geometry in geometry_by_id.items():
        neighbors = []
        for others_id, others_geometry in geometry_by_id.items():
            if reference_id == others_id:
                continue  # Ne pas comparer avec lui-même
            distance = reference_geometry.distance(others_geometry)
            if distance < distance_max:
                neighbors.append(others_id)

        relations_neigbhors[reference_id] = neighbors
    
    return relations_neigbhors


def display_id_simple(gdf: gpd.GeoDataFrame, id_value: Union[str, int],  neighbors_display: bool = True, column_id: str = COLUMN_ID):
    """
    Display the selected GeoDataFrame element with the given `id_value`.
    
    Parameters:
        gdf: The GeoDataFrame containing the elements to display.
        id_value: The identifier (unique value) of the element to display. It can be either a string or an integer.
        neighbors_display: Whether to display neighboring elements. Default is True.
        column_id: The name of the column containing the unique identifiers. Default is COLUMN_ID.

    """
    selected_elements = gdf[gdf[column_id] == id_value]
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, facecolor='blue', edgecolor='black')
    if neighbors_display:
        neighbors = get_neighbors(gdf)[id_value]
        for neighbor in neighbors:
            gdf[gdf[column_id] == neighbor].plot(color='red', ax=ax)
    selected_elements.plot(color='yellow', ax=ax)
    ax.annotate("Nord", xy=(0.1, 0.95), xycoords='axes fraction', xytext=(0.1, 0.85),
                arrowprops=dict(arrowstyle="fancy", color="black"), fontsize=15, fontweight='bold')
    x, y = selected_elements.geometry.iloc[0].centroid.xy
    plt.annotate(f'{id_value}', xy=(x[0], y[0]), xytext=(x[0]+2.0, y[0]+2.0), fontsize=15, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.title("Localisation OCR")
    plt.show()

def display_id(gdf: gpd.GeoDataFrame, id_value: Union[str, int], neighbors_display: bool = True, buffer_size: float = 10, column_id: str = COLUMN_ID):
    """
    Display the selected GeoDataFrame element with the given `id_value`, and optionally its closest neighbors within a range of `buffer_size`.

    Parameters:
        gdf: The GeoDataFrame containing the elements to display.
        id_value: The identifier (unique value) of the element to display.
                                    It can be either a string or an integer.
        neighbors_display: Whether to display neighboring elements in red color.
        buffer_size: The buffer size in meters to add around the reference point when cropping neighbor geometries.
                                       Default is 10 meters.
        column_id: The name of the column containing the unique identifiers.

    """
    LABO_COLOR = 'blue'
    OCR_COLOR = 'yellow'
    NEIGHBOR_COLOR = 'red'

    selected_elements = gdf[gdf[column_id] == id_value]
    selected_geometry = selected_elements.iloc[0]['geometry']
    fig, ax = plt.subplots(figsize=(10, 10))
    # display labo
    gdf.plot(ax=ax, facecolor='blue', edgecolor='black')
    # display neighbors
    if neighbors_display:
        neighbors = get_neighbors(gdf)[id_value]
        for neighbor in neighbors:
            neighbor_element = gdf[gdf[column_id] == neighbor]
            neighbor_geometry = neighbor_element.iloc[0]['geometry']
            reference_point = shapely.ops.nearest_points(selected_geometry, neighbor_geometry)[0]
            cropped_geometry = neighbor_geometry.intersection(reference_point.buffer(buffer_size))
            gpd.GeoDataFrame({column_id: [neighbor], 'geometry': [cropped_geometry]}).plot(color=NEIGHBOR_COLOR, ax=ax)
    # display selected galerie
    selected_elements.plot(color='yellow', ax=ax)            
    # add North arrow
    ax.annotate("Nord", xy=(0.1, 0.95), xycoords='axes fraction', xytext=(0.1, 0.85),
                arrowprops=dict(arrowstyle="fancy", color="black"), fontsize=15, fontweight='bold')
    x, y = selected_elements.geometry.iloc[0].centroid.xy
    
    plt.annotate(f'{id_value}', xy=(x[0], y[0]), xytext=(x[0]+2.0, y[0]+2.0), fontsize=15, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.title("Localisation OCR")

    # Add the legend with red and yellow rectangles
    target_legend = mpatches.Patch(color=OCR_COLOR, label=f'OCR info -> {id_value}')
    neighbor_legend = mpatches.Patch(color=NEIGHBOR_COLOR, label='Extension possible')
    labo_legend = mpatches.Patch(color=LABO_COLOR, label='Laboratoire')
    plt.legend(handles=[target_legend, neighbor_legend, labo_legend], loc='lower left')

    plt.show()


def main():
    return

if __name__ =="main":
    main()