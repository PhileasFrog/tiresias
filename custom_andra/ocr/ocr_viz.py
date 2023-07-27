import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import pandas as pd
from PIL import Image
from config_andra import OCR_DISPLAY_VAR
import custom_andra.utils.geo_info
import shapely.ops


NEIGHBOR_COLOR = OCR_DISPLAY_VAR['neighbor_color']
OCR_COLOR = OCR_DISPLAY_VAR['ocr_color']
LABO_COLOR = OCR_DISPLAY_VAR['labo_color']
COLUMN_ID = OCR_DISPLAY_VAR['column_id']


def ocr_figure_raw(ax, image_path: str):
    ax.imshow(Image.open(image_path))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{image_path}')

def ocr_figure_detection(ax, image_path: str, ocr_pred_galerie_df: pd.DataFrame):
    ax.imshow(Image.open(image_path))
    for _, row in ocr_pred_galerie_df.iterrows():
        polygon_detected = row.det_polygons
        galerie_name = row.rec_texts
        x_coords = polygon_detected[::2]
        y_coords = polygon_detected[1::2]
        ax.plot(x_coords + [x_coords[0]], y_coords+ [y_coords[0]], 'k-')
        ax.fill(x_coords + [x_coords[0]], y_coords+ [y_coords[0]], alpha=0.4, color='green')
        xy = min(x_coords) + (max(x_coords) - min(x_coords))/2 , min(y_coords) - 10
        ax.annotate(f'{galerie_name}', xy=xy, fontsize=10, fontweight='bold', color='red')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Galerie detection')

def ocr_figure_map_unique(ax, galerie_gdf: gpd.GeoDataFrame, detected_galerie_name: str, neighbors_display: bool =True, buffer_size: int = 10):

    galerie_gdf.plot(ax=ax, facecolor=LABO_COLOR, edgecolor='black')
    detected_galerie_gdf = galerie_gdf.loc[galerie_gdf[COLUMN_ID] == detected_galerie_name]
    centroid_point = detected_galerie_gdf.centroid.iloc[0]
    x, y = centroid_point.x, centroid_point.y
    detected_galerie_gdf = galerie_gdf.loc[galerie_gdf[COLUMN_ID] == detected_galerie_name]['geometry']
    detected_galerie_gdf.plot(color=OCR_COLOR, ax=ax)  
    ax.set_title("Localisation OCR")
    ax.annotate("Nord", xy=(0.1, 0.95), xycoords='axes fraction', xytext=(0.1, 0.85),
                arrowprops=dict(arrowstyle="fancy", color="black"), fontsize=15, fontweight='bold')


    if neighbors_display:
        neighbors = custom_andra.utils.geo_info.get_neighbors(galerie_gdf)
        neighbors = neighbors[detected_galerie_name]
        for neighbor in neighbors:
            neighbor_gdf = galerie_gdf.loc[galerie_gdf[COLUMN_ID] == neighbor].geometry
            reference_point = shapely.ops.nearest_points(neighbor_gdf.iloc[0], detected_galerie_gdf.iloc[0])[0]
            cropped_geometry = neighbor_gdf.intersection(reference_point.buffer(buffer_size))
            gpd.GeoDataFrame({COLUMN_ID: [neighbor], 'geometry': [cropped_geometry.geometry.iloc[0]]}).plot(color=NEIGHBOR_COLOR, ax=ax)

    ax.annotate(f'{detected_galerie_name}', xy=(x, y), xytext=(x, y+20), fontsize=15, fontweight='bold')


    # # Add the legend with red and yellow rectangles
    target_legend = mpatches.Patch(color=OCR_COLOR, label=f'OCR info -> {detected_galerie_name}')
    neighbor_legend = mpatches.Patch(color=NEIGHBOR_COLOR, label='Extension possible')
    labo_legend = mpatches.Patch(color=LABO_COLOR, label='Laboratoire')
    plt.legend(handles=[target_legend, neighbor_legend, labo_legend], loc='lower left')


def plot_ocr_results(image_path: str, ocr_pred_galerie_df: pd.DataFrame, galerie_gdf: gpd.GeoDataFrame, detected_galerie_name: str):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 20))
    ocr_figure_raw(ax1, image_path=image_path)
    ocr_figure_detection(ax2, image_path=image_path, ocr_pred_galerie_df=ocr_pred_galerie_df)
    ocr_figure_map_unique(ax3, galerie_gdf=galerie_gdf, detected_galerie_name=detected_galerie_name)
    plt.show()