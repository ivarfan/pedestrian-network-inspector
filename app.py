"""
Pedestrian Network Quality Inspector
A Streamlit application for analyzing and visualizing pedestrian network quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import json
import zipfile
import tempfile
import os
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from network_analyzer import NetworkAnalyzer, load_network_from_shapefile
from osm_comparison import (OSMComparator, bounds_from_gdf, OSMNX_AVAILABLE, 
                            compute_offset_arrows, compute_tiled_error_density, get_worst_tiles,
                            compute_network_quality_score)

# Page configuration
st.set_page_config(
    page_title="Pedestrian Network Quality Inspector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode
st.markdown("""
<style>
    /* Force dark mode */
    :root {
        color-scheme: dark;
    }
    
    /* Main app background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
        border-right: 1px solid #2d3139;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #fafafa;
    }
    
    /* Sidebar title */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #60a5fa;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d3139;
    }
    
    /* Sidebar region info */
    .sidebar-region {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #262b36;
        border-radius: 0.5rem;
        word-break: break-all;
    }
    
    /* Navigation styling */
    .nav-section {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #2d3139;
    }
    
    .nav-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #6b7280;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    /* Custom navigation item styling */
    .nav-item {
        padding: 0.5rem 0.75rem;
        margin: 0.125rem 0;
        border-radius: 0.375rem;
        cursor: pointer;
        color: #9ca3af;
        font-size: 0.95rem;
        transition: all 0.15s ease;
    }
    
    .nav-item:hover {
        background: #262b36;
        color: #fafafa;
    }
    
    .nav-item-active {
        background: #1e3a5f !important;
        color: #60a5fa !important;
        font-weight: 500;
    }
    
    /* Sidebar radio navigation styling */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.125rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        padding: 0.5rem 0.75rem !important;
        margin: 0.125rem 0;
        border-radius: 0.5rem;
        background: transparent;
        cursor: pointer;
        border: 1px solid transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: #262b36;
    }
    
    /* Selected state - matching the Edge Length style */
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: #1e3a5f !important;
        border: 1px solid #2d4a6f !important;
        border-radius: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] p,
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] span {
        color: #60a5fa !important;
        font-weight: 500;
    }
    
    /* Main content text color */
    .main .block-container {
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #60a5fa;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    
    .quality-good { color: #34d399; }
    .quality-medium { color: #fbbf24; }
    .quality-poor { color: #f87171; }
    
    .score-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Upload page title */
    .upload-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #60a5fa;
        text-align: center;
        margin-top: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* File uploader styling - dark mode */
    .upload-area [data-testid="stFileUploader"] > section {
        border: 2px dashed #60a5fa;
        border-radius: 0.75rem;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1d24 0%, #262b36 100%);
        transition: all 0.3s ease;
    }
    
    .upload-area [data-testid="stFileUploader"] > section:hover {
        border-color: #93c5fd;
        background: linear-gradient(135deg, #1e3a5f 0%, #1e40af20 100%);
    }
    
    /* Feature grid - dark mode */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .feature-item {
        background: #1a1d24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #60a5fa;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    .feature-item h4 {
        margin: 0 0 0.5rem 0;
        color: #60a5fa;
        font-size: 0.9rem;
    }
    
    .feature-item p {
        margin: 0;
        color: #9ca3af;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    /* Streamlit elements dark mode overrides */
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label, .stCheckbox label {
        color: #fafafa !important;
    }
    
    .stDataFrame {
        background: #1a1d24;
    }
    
    /* Info/Warning/Error/Success boxes */
    .stAlert {
        background-color: transparent;
        border: none;
        box-shadow: none;
    }
    
    /* Keep the inner alert box styled */
    .stAlert [data-baseweb="notification"] {
        background-color: #1a1d24;
        border: 1px solid #2d3139;
    }
    
    /* Remove outer container styling */
    [data-testid="stNotification"] {
        background-color: transparent !important;
    }
    
    .element-container:has(.stAlert) {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a1d24;
        color: #fafafa;
    }
    
    /* Divider */
    hr {
        border-color: #2d3139;
    }
    
    /* Tab styling in main content */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #1a1d24;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-bottom: none !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9ca3af;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: 1px solid #2d3139;
        border-bottom: 1px solid #2d3139 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #262b36;
        color: #fafafa;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important;
        color: #60a5fa !important;
        border: 1px solid #1e3a5f !important;
        border-bottom: 1px solid #1e3a5f !important;
    }
    
    /* Remove tab highlight/underline */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
        background: transparent !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #fafafa;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }
    
    /* Caption text */
    .stCaption {
        color: #6b7280;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1e3a5f;
        color: #fafafa;
        border: 1px solid #2d3139;
    }
    
    .stButton > button:hover {
        background-color: #1e40af;
        border-color: #60a5fa;
    }
    
    /* Hide sidebar on upload page */
    .hide-sidebar [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# Dark mode plotly template
PLOTLY_DARK_TEMPLATE = {
    'layout': {
        'paper_bgcolor': '#0e1117',
        'plot_bgcolor': '#1a1d24',
        'font': {'color': '#fafafa'},
        'title': {'font': {'color': '#fafafa'}},
        'xaxis': {
            'gridcolor': '#2d3139',
            'linecolor': '#2d3139',
            'tickfont': {'color': '#9ca3af'},
            'title': {'font': {'color': '#9ca3af'}}
        },
        'yaxis': {
            'gridcolor': '#2d3139',
            'linecolor': '#2d3139',
            'tickfont': {'color': '#9ca3af'},
            'title': {'font': {'color': '#9ca3af'}}
        },
        'legend': {'font': {'color': '#fafafa'}},
        'colorway': ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#fb7185']
    }
}


def apply_dark_theme(fig):
    """Apply dark theme to a plotly figure."""
    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1a1d24',
        font=dict(color='#fafafa'),
        title=dict(font=dict(color='#fafafa')),
        xaxis=dict(
            gridcolor='#2d3139',
            linecolor='#2d3139',
            tickfont=dict(color='#9ca3af'),
            title=dict(font=dict(color='#9ca3af'))
        ),
        yaxis=dict(
            gridcolor='#2d3139',
            linecolor='#2d3139',
            tickfont=dict(color='#9ca3af'),
            title=dict(font=dict(color='#9ca3af'))
        ),
        legend=dict(font=dict(color='#fafafa'))
    )
    return fig


def get_quality_color(score: float) -> str:
    """Return color based on quality score."""
    if score >= 70:
        return '#34d399'  # Green (updated for dark mode)
    elif score >= 40:
        return '#fbbf24'  # Yellow (updated for dark mode)
    else:
        return '#f87171'  # Red (updated for dark mode)


def extract_network_from_zip(uploaded_file) -> tuple:
    """Extract network files from uploaded zip."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Find shapefile
            shp_file = None
            json_file = None
            
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.shp'):
                        shp_file = os.path.join(root, file)
                    elif file.endswith('.json') and 'info' in file.lower():
                        json_file = os.path.join(root, file)
            
            if shp_file:
                network_gdf = gpd.read_file(shp_file)
                
                # Load info.json if available
                bbox = None
                if json_file:
                    with open(json_file, 'r') as f:
                        info = json.load(f)
                        if 'bbox' in info:
                            bbox = tuple(info['bbox'])
                
                return network_gdf, bbox, None
            else:
                return None, None, "No shapefile found in zip"
                
    except Exception as e:
        return None, None, str(e)


def add_bbox_to_map(m: folium.Map, bbox: tuple, show: bool = True) -> None:
    """Add bounding box rectangle to map. bbox is (south, north, west, east)."""
    if bbox is None or not show:
        return
    south, north, west, east = bbox
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color='#fbbf24',
        fill=False,
        weight=2,
        dash_array='10, 5',
        popup='Detection Region'
    ).add_to(m)


def get_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Get UTM CRS for accurate distance calculations."""
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf_wgs = gdf.to_crs(epsg=4326)
    centroid = gdf_wgs.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    return f"EPSG:{32600 + utm_zone}" if centroid.y >= 0 else f"EPSG:{32700 + utm_zone}"


def get_edge_lengths_in_meters(network_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Calculate edge lengths in meters by projecting to UTM."""
    if network_gdf.crs is None:
        network_gdf = network_gdf.set_crs(epsg=4326)
    
    centroid = network_gdf.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}" if centroid.y >= 0 else f"EPSG:{32700 + utm_zone}"
    
    try:
        network_proj = network_gdf.to_crs(utm_crs)
        return network_proj.geometry.length
    except:
        return network_gdf.geometry.length * 111000


def create_network_map_with_layers(network_gdf: gpd.GeoDataFrame,
                                    dead_ends_gdf: gpd.GeoDataFrame = None,
                                    intersections_gdf: gpd.GeoDataFrame = None,
                                    component_gdf: gpd.GeoDataFrame = None,
                                    all_nodes_gdf: gpd.GeoDataFrame = None,
                                    node_mode: str = 'plain',
                                    bbox: tuple = None,
                                    center: tuple = None,
                                    zoom: int = 15,
                                    show_bbox: bool = True) -> folium.Map:
    """Create interactive map with multiple layers."""
    
    network_wgs = network_gdf.to_crs(epsg=4326)
    
    if center is None:
        bounds = network_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # Network edges
    edges_layer = folium.FeatureGroup(name='Network Edges', show=True)
    for idx, row in network_wgs.iterrows():
        if row.geometry is not None:
            coords = []
            if row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords.extend([(c[1], c[0]) for c in line.coords])
            if coords:
                folium.PolyLine(coords, weight=3, color='#60a5fa', opacity=0.8).add_to(edges_layer)
    edges_layer.add_to(m)
    
    # Helper function for proportional radius
    def get_node_radius(degree, is_proportional):
        if is_proportional:
            # degree 1 -> 3, degree 2 -> 5, degree 3 -> 7, degree 4 -> 9, etc. (capped at 15)
            return min(3 + (degree - 1) * 2, 15)
        else:
            # Fixed sizes: dead-ends 5, normal 3, intersections 5
            if degree == 1:
                return 5
            elif degree == 2:
                return 3
            else:
                return 5
    
    is_proportional = node_mode == 'proportional'
    
    # Dead-end nodes (red)
    if dead_ends_gdf is not None and len(dead_ends_gdf) > 0:
        dead_ends_layer = folium.FeatureGroup(name='Dead-end Nodes (red)', show=True)
        dead_ends_wgs = dead_ends_gdf.to_crs(epsg=4326)
        for idx, row in dead_ends_wgs.iterrows():
            if row.geometry is not None:
                radius = get_node_radius(1, is_proportional)
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=radius, color='#f87171', fill=True,
                    fillColor='#f87171', fillOpacity=0.8,
                    popup=f"Dead-end Node {row.get('node_id', idx)}, Degree: 1"
                ).add_to(dead_ends_layer)
        dead_ends_layer.add_to(m)
    
    # Normal nodes (degree 2) - gray
    if all_nodes_gdf is not None and len(all_nodes_gdf) > 0:
        normal_nodes = all_nodes_gdf[all_nodes_gdf['degree'] == 2]
        if len(normal_nodes) > 0:
            normal_layer = folium.FeatureGroup(name='Normal Nodes (gray)', show=False)
            normal_wgs = normal_nodes.to_crs(epsg=4326)
            for idx, row in normal_wgs.iterrows():
                if row.geometry is not None:
                    radius = get_node_radius(2, is_proportional)
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=radius, color='#888888', fill=True,
                        fillColor='#888888', fillOpacity=0.6,
                        popup=f"Degree 2"
                    ).add_to(normal_layer)
            normal_layer.add_to(m)
    
    # Intersection nodes (green)
    if intersections_gdf is not None and len(intersections_gdf) > 0:
        intersections_layer = folium.FeatureGroup(name='Intersection Nodes (green)', show=True)
        intersections_wgs = intersections_gdf.to_crs(epsg=4326)
        for idx, row in intersections_wgs.iterrows():
            if row.geometry is not None:
                degree = row.get('degree', 3)
                radius = get_node_radius(degree, is_proportional)
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=radius, color='#34d399', fill=True,
                    fillColor='#34d399', fillOpacity=0.8,
                    popup=f"Intersection Node {row.get('node_id', idx)}, Degree: {degree}"
                ).add_to(intersections_layer)
        intersections_layer.add_to(m)
    
    # Connected components
    if component_gdf is not None and len(component_gdf) > 0:
        components_layer = folium.FeatureGroup(name='Connected Components', show=False)
        component_wgs = component_gdf.to_crs(epsg=4326)
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
                  '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
        
        unique_comps = component_wgs['component_id'].unique()
        for comp_id in unique_comps:
            comp_edges = component_wgs[component_wgs['component_id'] == comp_id]
            is_oversized = comp_edges['is_oversized'].iloc[0] if 'is_oversized' in comp_edges.columns else False
            edge_count = comp_edges['edge_count'].iloc[0] if 'edge_count' in comp_edges.columns else len(comp_edges)
            
            color = '#cccccc' if is_oversized else colors[int(comp_id) % len(colors)]
            opacity = 0.4 if is_oversized else 0.9
            
            for idx, row in comp_edges.iterrows():
                if row.geometry is not None:
                    coords = [(c[1], c[0]) for c in row.geometry.coords]
                    if coords:
                        folium.PolyLine(
                            coords, weight=4, color=color, opacity=opacity,
                            popup=f"Component {comp_id} ({edge_count} edges)"
                        ).add_to(components_layer)
        components_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def create_comparison_map(network_gdf: gpd.GeoDataFrame,
                          comparison_results: dict,
                          bbox: tuple = None,
                          center: tuple = None,
                          show_bbox: bool = True) -> folium.Map:
    """Create map comparing Tile2Net and OSM networks."""
    
    network_wgs = network_gdf.to_crs(epsg=4326)
    
    if center is None:
        bounds = network_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # AI matched (True Positives) - Blue
    ai_matched_layer = folium.FeatureGroup(name='Tile2Net Matched (Blue)', show=True)
    ai_matched = comparison_results.get('ai_matched', gpd.GeoDataFrame())
    if len(ai_matched) > 0:
        ai_matched_wgs = ai_matched.to_crs(epsg=4326)
        for idx, row in ai_matched_wgs.iterrows():
            if row.geometry is not None and row.geometry.geom_type in ['LineString', 'MultiLineString']:
                coords = []
                if row.geometry.geom_type == 'LineString':
                    coords = [(c[1], c[0]) for c in row.geometry.coords]
                else:
                    for line in row.geometry.geoms:
                        coords.extend([(c[1], c[0]) for c in line.coords])
                if coords:
                    folium.PolyLine(coords, weight=4, color='#60a5fa', opacity=0.8).add_to(ai_matched_layer)
    ai_matched_layer.add_to(m)
    
    # AI unmatched (False Positives) - Red
    ai_unmatched_layer = folium.FeatureGroup(name='Tile2Net Only (Red)', show=True)
    ai_unmatched = comparison_results.get('ai_unmatched', gpd.GeoDataFrame())
    if len(ai_unmatched) > 0:
        ai_unmatched_wgs = ai_unmatched.to_crs(epsg=4326)
        for idx, row in ai_unmatched_wgs.iterrows():
            if row.geometry is not None and row.geometry.geom_type in ['LineString', 'MultiLineString']:
                coords = []
                if row.geometry.geom_type == 'LineString':
                    coords = [(c[1], c[0]) for c in row.geometry.coords]
                else:
                    for line in row.geometry.geoms:
                        coords.extend([(c[1], c[0]) for c in line.coords])
                if coords:
                    folium.PolyLine(coords, weight=4, color='#f87171', opacity=0.8).add_to(ai_unmatched_layer)
    ai_unmatched_layer.add_to(m)
    
    # OSM matched - Green
    osm_layer = folium.FeatureGroup(name='OSM Matched (Green)', show=True)
    osm_matched = comparison_results.get('osm_matched', gpd.GeoDataFrame())
    if len(osm_matched) > 0:
        osm_matched_wgs = osm_matched.to_crs(epsg=4326)
        for idx, row in osm_matched_wgs.iterrows():
            if row.geometry is not None and row.geometry.geom_type in ['LineString', 'MultiLineString']:
                coords = []
                if row.geometry.geom_type == 'LineString':
                    coords = [(c[1], c[0]) for c in row.geometry.coords]
                else:
                    for line in row.geometry.geoms:
                        coords.extend([(c[1], c[0]) for c in line.coords])
                if coords:
                    folium.PolyLine(coords, weight=3, color='#34d399', opacity=0.7).add_to(osm_layer)
    osm_layer.add_to(m)
    
    # OSM unmatched (False Negatives) - Orange
    osm_unmatched_layer = folium.FeatureGroup(name='OSM Only (Orange)', show=True)
    osm_unmatched = comparison_results.get('osm_unmatched', gpd.GeoDataFrame())
    if len(osm_unmatched) > 0:
        osm_unmatched_wgs = osm_unmatched.to_crs(epsg=4326)
        for idx, row in osm_unmatched_wgs.iterrows():
            if row.geometry is not None and row.geometry.geom_type in ['LineString', 'MultiLineString']:
                coords = []
                if row.geometry.geom_type == 'LineString':
                    coords = [(c[1], c[0]) for c in row.geometry.coords]
                else:
                    for line in row.geometry.geoms:
                        coords.extend([(c[1], c[0]) for c in line.coords])
                if coords:
                    folium.PolyLine(coords, weight=3, color='#fd7e14', opacity=0.7).add_to(osm_unmatched_layer)
    osm_unmatched_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def create_gap_diagnostics_map(network_gdf: gpd.GeoDataFrame,
                               near_miss_data: dict,
                               bbox: tuple = None,
                               center: tuple = None,
                               show_bbox: bool = True) -> folium.Map:
    """Create map showing near-miss endpoints and missing links."""
    
    network_wgs = network_gdf.to_crs(epsg=4326)
    
    if center is None:
        bounds = network_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # Network in light gray
    edges_layer = folium.FeatureGroup(name='Network Edges', show=True)
    for idx, row in network_wgs.iterrows():
        if row.geometry is not None:
            coords = []
            if row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords.extend([(c[1], c[0]) for c in line.coords])
            if coords:
                folium.PolyLine(coords, weight=2, color='#aaaaaa', opacity=0.5).add_to(edges_layer)
    edges_layer.add_to(m)
    
    # Missing links
    missing_links_layer = folium.FeatureGroup(name='Missing Links (gaps)', show=True)
    missing_links_gdf = near_miss_data.get('missing_links', gpd.GeoDataFrame())
    if len(missing_links_gdf) > 0:
        missing_wgs = missing_links_gdf.to_crs(epsg=4326)
        for idx, row in missing_wgs.iterrows():
            if row.geometry is not None:
                coords = [(c[1], c[0]) for c in row.geometry.coords]
                if coords:
                    folium.PolyLine(
                        coords, weight=3, color='#fb7185', opacity=0.9,
                        dash_array='5, 5',
                        popup=f"Gap: {row.get('distance', 0):.2f} m"
                    ).add_to(missing_links_layer)
                    for coord in coords:
                        folium.CircleMarker(
                            location=coord, radius=4, color='#fb7185',
                            fill=True, fillColor='#fb7185', fillOpacity=0.8
                        ).add_to(missing_links_layer)
    missing_links_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def create_intersection_angles_map(network_gdf: gpd.GeoDataFrame,
                                    angles_gdf: gpd.GeoDataFrame,
                                    bbox: tuple = None,
                                    center: tuple = None,
                                    show_bbox: bool = True) -> folium.Map:
    """Create map showing intersection angles."""
    
    network_wgs = network_gdf.to_crs(epsg=4326)
    angles_wgs = angles_gdf.to_crs(epsg=4326)
    
    if center is None:
        bounds = network_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # Network in light gray
    edges_layer = folium.FeatureGroup(name='Network Edges', show=True)
    for idx, row in network_wgs.iterrows():
        if row.geometry is not None:
            coords = []
            if row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords.extend([(c[1], c[0]) for c in line.coords])
            if coords:
                folium.PolyLine(coords, weight=2, color='#cccccc', opacity=0.6).add_to(edges_layer)
    edges_layer.add_to(m)
    
    # Intersections colored by angle quality
    intersections_layer = folium.FeatureGroup(name='Intersections', show=True)
    for idx, row in angles_wgs.iterrows():
        if row.geometry is not None:
            score = row.get('right_angle_score', 0)
            min_angle = row.get('min_angle', 90)
            
            if score > 0.5 or (min_angle > 60 and min_angle < 120):
                color = '#34d399'
            elif min_angle < 30 or min_angle > 150:
                color = '#f87171'
            else:
                color = '#fbbf24'
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8, color=color, fill=True,
                fillColor=color, fillOpacity=0.8,
                popup=f"Degree: {row['degree']}<br>Min angle: {min_angle:.1f}°<br>Max angle: {row['max_angle']:.1f}°"
            ).add_to(intersections_layer)
    intersections_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def create_edge_length_map(network_gdf: gpd.GeoDataFrame,
                           edges_by_length: dict,
                           bbox: tuple = None,
                           center: tuple = None,
                           show_bbox: bool = True) -> folium.Map:
    """Create map showing edges colored by length."""
    
    all_edges = edges_by_length['all_edges'].to_crs(epsg=4326)
    
    if center is None:
        bounds = all_edges.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # Normal edges
    normal_layer = folium.FeatureGroup(name='Normal Edges', show=True)
    normal_edges = all_edges[all_edges['category'] == 'normal']
    for idx, row in normal_edges.iterrows():
        if row.geometry is not None:
            coords = [(c[1], c[0]) for c in row.geometry.coords]
            if coords:
                folium.PolyLine(coords, weight=2, color='#888888', opacity=0.5,
                               popup=f"Length: {row['length']:.2f}").add_to(normal_layer)
    normal_layer.add_to(m)
    
    # Short edges
    short_layer = folium.FeatureGroup(name='Very Short Edges', show=True)
    short_edges = edges_by_length['short_edges'].to_crs(epsg=4326) if len(edges_by_length['short_edges']) > 0 else gpd.GeoDataFrame()
    for idx, row in short_edges.iterrows():
        if row.geometry is not None:
            coords = [(c[1], c[0]) for c in row.geometry.coords]
            if coords:
                folium.PolyLine(coords, weight=5, color='#f87171', opacity=0.9,
                               popup=f"Length: {row['length']:.2f}").add_to(short_layer)
    short_layer.add_to(m)
    
    # Long edges
    long_layer = folium.FeatureGroup(name='Very Long Edges', show=True)
    long_edges = edges_by_length['long_edges'].to_crs(epsg=4326) if len(edges_by_length['long_edges']) > 0 else gpd.GeoDataFrame()
    for idx, row in long_edges.iterrows():
        if row.geometry is not None:
            coords = [(c[1], c[0]) for c in row.geometry.coords]
            if coords:
                folium.PolyLine(coords, weight=5, color='#00bcd4', opacity=0.9,
                               popup=f"Length: {row['length']:.2f}").add_to(long_layer)
    long_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def create_loops_spurs_map(network_gdf: gpd.GeoDataFrame,
                           small_loops: gpd.GeoDataFrame,
                           short_spurs: gpd.GeoDataFrame,
                           bbox: tuple = None,
                           center: tuple = None,
                           show_bbox: bool = True) -> folium.Map:
    """Create map highlighting small loops and short spurs."""
    
    network_wgs = network_gdf.to_crs(epsg=4326)
    
    if center is None:
        bounds = network_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # Network edges
    edges_layer = folium.FeatureGroup(name='Network Edges', show=True)
    for idx, row in network_wgs.iterrows():
        if row.geometry is not None:
            coords = []
            if row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords.extend([(c[1], c[0]) for c in line.coords])
            if coords:
                folium.PolyLine(coords, weight=2, color='#aaaaaa', opacity=0.5).add_to(edges_layer)
    edges_layer.add_to(m)
    
    # Small loops
    loops_layer = folium.FeatureGroup(name='Small Loops', show=True)
    if len(small_loops) > 0:
        loops_wgs = small_loops.to_crs(epsg=4326)
        for idx, row in loops_wgs.iterrows():
            if row.geometry is not None and row.geometry.geom_type == 'Polygon':
                coords = [(c[1], c[0]) for c in row.geometry.exterior.coords]
                if coords:
                    folium.Polygon(
                        locations=coords, color='#00bcd4', fill=True,
                        fillColor='#00bcd4', fillOpacity=0.3, weight=3,
                        popup=f"Loop perimeter: {row['perimeter']:.2f}"
                    ).add_to(loops_layer)
    loops_layer.add_to(m)
    
    # Short spurs
    spurs_layer = folium.FeatureGroup(name='Short Spurs', show=True)
    if len(short_spurs) > 0:
        spurs_wgs = short_spurs.to_crs(epsg=4326)
        for idx, row in spurs_wgs.iterrows():
            if row.geometry is not None:
                coords = [(c[1], c[0]) for c in row.geometry.coords]
                if coords:
                    folium.PolyLine(
                        coords, weight=5, color='#f87171', opacity=0.9,
                        popup=f"Spur length: {row['length']:.2f}"
                    ).add_to(spurs_layer)
    spurs_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def create_error_density_map(network_gdf: gpd.GeoDataFrame,
                             tiles_gdf: gpd.GeoDataFrame,
                             bbox: tuple = None,
                             center: tuple = None,
                             show_bbox: bool = True) -> folium.Map:
    """Create choropleth map showing error density by tile."""
    
    network_wgs = network_gdf.to_crs(epsg=4326)
    tiles_wgs = tiles_gdf.to_crs(epsg=4326)
    
    if center is None:
        bounds = network_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # Error density tiles
    tiles_layer = folium.FeatureGroup(name='Error Density Tiles', show=True)
    max_errors = tiles_wgs['total_errors'].max() if len(tiles_wgs) > 0 else 1
    
    for idx, row in tiles_wgs.iterrows():
        if row.geometry is not None and row.geometry.geom_type == 'Polygon':
            coords = [(c[1], c[0]) for c in row.geometry.exterior.coords]
            total_errors = row['total_errors']
            
            if total_errors == 0:
                folium.Polygon(
                    locations=coords, color='#4b5563', fill=False,
                    weight=1, opacity=0.5
                ).add_to(tiles_layer)
            else:
                ratio = total_errors / max_errors if max_errors > 0 else 0
                opacity = 0.3 + ratio * 0.5
                folium.Polygon(
                    locations=coords, color='#f87171', fill=True,
                    fillColor='#f87171', fillOpacity=opacity, weight=1,
                    popup=f"FP: {row['false_positives']}, FN: {row['false_negatives']}<br>"
                           f"Dead-ends: {row['dead_ends']}, Near-misses: {row['near_misses']}<br>"
                           f"Total: {total_errors}"
                ).add_to(tiles_layer)
    tiles_layer.add_to(m)
    
    # Network edges
    edges_layer = folium.FeatureGroup(name='Network Edges', show=True)
    for idx, row in network_wgs.iterrows():
        if row.geometry is not None:
            coords = []
            if row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords.extend([(c[1], c[0]) for c in line.coords])
            if coords:
                folium.PolyLine(coords, weight=2, color='#9ca3af', opacity=0.6).add_to(edges_layer)
    edges_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def create_offset_arrows_map(osm_gdf: gpd.GeoDataFrame,
                              predicted_gdf: gpd.GeoDataFrame,
                              arrows_gdf: gpd.GeoDataFrame,
                              bbox: tuple = None,
                              center: tuple = None,
                              show_bbox: bool = True) -> folium.Map:
    """Create map showing offset arrows from OSM centroids to nearest predicted points."""
    from shapely.ops import nearest_points
    
    osm_wgs = osm_gdf.to_crs(epsg=4326)
    predicted_wgs = predicted_gdf.to_crs(epsg=4326)
    arrows_wgs = arrows_gdf.to_crs(epsg=4326) if len(arrows_gdf) > 0 else gpd.GeoDataFrame()
    
    if center is None:
        bounds = osm_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    
    # Set map background to dark gray for Plain Background option
    m.get_root().html.add_child(folium.Element('<style>.leaflet-container { background-color: #1a1d24 !important; }</style>'))
    
    # Base layers (mutually exclusive - radio buttons)
    folium.TileLayer(tiles='', name='Plain Background', attr=' ', overlay=False).add_to(m)
    folium.TileLayer(tiles='CartoDB dark_matter', name='Street Map', overlay=False).add_to(m)
    folium.TileLayer(
        tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
        attr='NYC DoITT', name='Satellite (NYC 2024)', overlay=False,
        maxNativeZoom=20, maxZoom=22
    ).add_to(m)
    
    # OSM reference network in thin gray
    osm_layer = folium.FeatureGroup(name='OSM Reference (gray)', show=True)
    for idx, row in osm_wgs.iterrows():
        if row.geometry is not None:
            coords = []
            if row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords.extend([(c[1], c[0]) for c in line.coords])
            if coords:
                folium.PolyLine(coords, weight=2, color='#999999', opacity=0.6).add_to(osm_layer)
    osm_layer.add_to(m)
    
    # Predicted network in blue
    predicted_layer = folium.FeatureGroup(name='Tile2Net (blue)', show=True)
    for idx, row in predicted_wgs.iterrows():
        if row.geometry is not None:
            coords = []
            if row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords.extend([(c[1], c[0]) for c in line.coords])
            if coords:
                folium.PolyLine(coords, weight=3, color='#60a5fa', opacity=0.7).add_to(predicted_layer)
    predicted_layer.add_to(m)
    
    # Offset arrows colored by distance
    if len(arrows_wgs) > 0:
        arrows_layer = folium.FeatureGroup(name='Offset Arrows', show=True)
        max_dist = arrows_wgs['distance'].max() if 'distance' in arrows_wgs.columns else 1
        
        for idx, row in arrows_wgs.iterrows():
            if row.geometry is not None and row.geometry.geom_type == 'LineString':
                coords = [(c[1], c[0]) for c in row.geometry.coords]
                if len(coords) >= 2:
                    dist = row.get('distance', 0)
                    # Color: green (short) to red (long)
                    ratio = min(dist / max_dist, 1.0) if max_dist > 0 else 0
                    # Interpolate from green to red
                    r = int(255 * ratio)
                    g = int(255 * (1 - ratio))
                    color = f'#{r:02x}{g:02x}00'
                    
                    # Draw arrow line
                    folium.PolyLine(
                        coords, weight=2, color=color, opacity=0.8,
                        popup=f"Offset: {dist:.2f} m"
                    ).add_to(arrows_layer)
                    
                    # Draw arrow head (small circle at end)
                    folium.CircleMarker(
                        location=coords[-1], radius=3,
                        color=color, fill=True, fillColor=color, fillOpacity=0.8
                    ).add_to(arrows_layer)
        arrows_layer.add_to(m)
    
    add_bbox_to_map(m, bbox, show_bbox)
    folium.LayerControl().add_to(m)
    return m


def main():
    # Initialize session state for file management
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
        st.session_state.network_gdf = None
        st.session_state.bbox = None
        st.session_state.region_str = None
    
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Overview"
    
    if 'show_bbox' not in st.session_state:
        st.session_state.show_bbox = True
    
    # FULL SCREEN UPLOAD VIEW (when no data loaded)
    if st.session_state.uploaded_data is None:
        # Hide sidebar on upload page
        st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none; }
            [data-testid="stSidebarCollapsedControl"] { display: none; }
        </style>
        """, unsafe_allow_html=True)
        
        # Title and subtitle
        st.markdown("""
        <div class="upload-title">Pedestrian Network Quality Inspector</div>
        <div class="upload-subtitle">
            Analyze and visualize the quality of AI-generated pedestrian networks from Tile2Net. 
            Compare with OpenStreetMap, identify gaps, and assess network topology.
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader - centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drag and drop your Tile2Net ZIP file here, or click to browse",
                type=['zip'],
                key="file_uploader_main",
                help="Upload a ZIP file containing network shapefile (.shp, .shx, .dbf, .prj) and optionally info.json"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle file upload
        if uploaded_file is not None:
            with st.spinner("Loading network..."):
                network_gdf, bbox, error = extract_network_from_zip(uploaded_file)
            
            if error:
                st.error(f"Error loading file: {error}")
                return
            
            if network_gdf is None or len(network_gdf) == 0:
                st.error("No valid network data found.")
                return
            
            # Store in session state
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            st.session_state.uploaded_data = file_id
            st.session_state.network_gdf = network_gdf
            st.session_state.bbox = bbox
            
            # Create region string
            if bbox is not None:
                st.session_state.region_str = f"[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]"
            else:
                network_bounds = network_gdf.to_crs(epsg=4326).total_bounds
                st.session_state.region_str = f"[{network_bounds[1]:.4f}, {network_bounds[3]:.4f}, {network_bounds[0]:.4f}, {network_bounds[2]:.4f}]"
            st.rerun()
        
        # Feature descriptions
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.markdown("""
            <div class="feature-grid">
                <div class="feature-item">
                    <h4>Quality Scoring</h4>
                    <p>Automated scoring for connectivity, dead-ends, and intersections</p>
                </div>
                <div class="feature-item">
                    <h4>Network Topology</h4>
                    <p>Analyze degree distribution, centrality, and connected components</p>
                </div>
                <div class="feature-item">
                    <h4>Gap Detection</h4>
                    <p>Find near-miss endpoints and potential connection issues</p>
                </div>
                <div class="feature-item">
                    <h4>OSM Comparison</h4>
                    <p>Compare with OpenStreetMap to measure precision and recall</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # ANALYSIS VIEW (when data is loaded)
    network_gdf = st.session_state.network_gdf
    bbox = st.session_state.bbox
    
    # SIDEBAR NAVIGATION
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Pedestrian Network<br>Quality Inspector</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="sidebar-region"><strong>Region:</strong><br>{st.session_state.region_str}</div>', unsafe_allow_html=True)
        
        if st.button("Change Location", key="change_region_btn", use_container_width=True):
            st.session_state.uploaded_data = None
            st.session_state.network_gdf = None
            st.session_state.bbox = None
            st.session_state.region_str = None
            st.session_state.selected_tab = "Overview"
            # Clear OSM comparison cache too
            if 'osm_comparison_result' in st.session_state:
                del st.session_state.osm_comparison_result
            if 'osm_bounds' in st.session_state:
                del st.session_state.osm_bounds
            if 'osm_comparator' in st.session_state:
                del st.session_state.osm_comparator
            # Clear NQS cache
            if 'nqs_result' in st.session_state:
                del st.session_state.nqs_result
            if 'nqs_bounds' in st.session_state:
                del st.session_state.nqs_bounds
            if 'nqs_osm_network' in st.session_state:
                del st.session_state.nqs_osm_network
            st.rerun()
        
        st.markdown('<div class="nav-section"><div class="nav-label">Navigation</div></div>', unsafe_allow_html=True)
        
        # Tab navigation using radio buttons
        tab_options = [
            "Overview",
            "Graph Metrics", 
            "Network Map",
            "Gap Diagnostics",
            "Geometry Checks",
            "Edge Analysis",
            "OSM Comparison"
        ]
        
        selected = st.radio(
            "Select View",
            tab_options,
            index=tab_options.index(st.session_state.selected_tab) if st.session_state.selected_tab in tab_options else 0,
            label_visibility="collapsed",
            key="nav_radio"
        )
        
        current_tab = selected
        st.session_state.selected_tab = current_tab
        
        # Spacer to push settings to bottom
        st.markdown("<div style='flex-grow: 1; min-height: 2rem;'></div>", unsafe_allow_html=True)
        
        # Map settings at bottom
        st.markdown('<div class="nav-section"><div class="nav-label">Map Settings</div></div>', unsafe_allow_html=True)
        st.session_state.show_bbox = st.checkbox("Show Bounding Box", value=st.session_state.show_bbox, key="bbox_toggle")
    
    # Analyze
    with st.spinner("Analyzing network..."):
        analyzer = NetworkAnalyzer(network_gdf)
        basic_stats = analyzer.get_basic_stats()
        components = analyzer.get_connected_components()
        dead_ends = analyzer.get_dead_ends()
        intersections = analyzer.get_intersection_nodes()
        degree_dist = analyzer.get_node_degree_distribution()
        betweenness = analyzer.compute_betweenness_centrality()
        connectivity = analyzer.compute_connectivity_metrics()
        quality_scores = analyzer.get_quality_score()
        component_gdf = analyzer.get_component_gdf()
        anomalies = analyzer.detect_geometric_anomalies()
        edge_analysis = analyzer.analyze_edge_geometry()
        all_nodes_degree = analyzer.get_all_nodes_with_degree()
        edge_lengths_meters = get_edge_lengths_in_meters(network_gdf)
    
    # Render content based on selected tab
    if current_tab == "Overview":
        st.header("Network Quality Overview")
        
        # Compute NQS if OSM is available
        nqs_result = None
        if OSMNX_AVAILABLE:
            # Check if we already have the NQS result cached
            if 'nqs_result' not in st.session_state or st.session_state.get('nqs_bounds') != bbox:
                with st.spinner("Computing Network Quality Score (comparing with OSM ground truth)..."):
                    try:
                        bounds = bbox if bbox is not None else bounds_from_gdf(network_gdf.to_crs(epsg=4326))
                        
                        # Fetch OSM data
                        comparator = OSMComparator(bounds)
                        osm_network = comparator.fetch_osm_network()
                        
                        if osm_network is not None and len(osm_network) > 0:
                            # Project to UTM for accurate distance calculations
                            utm_crs = get_utm_crs(network_gdf)
                            network_proj = network_gdf.to_crs(utm_crs)
                            osm_proj = osm_network.to_crs(utm_crs)
                            
                            # Compute NQS with default parameters
                            nqs_result = compute_network_quality_score(
                                gt_network=osm_proj,
                                pred_network=network_proj,
                                distance_tolerance=4.0,  # 4 meters
                                overlap_threshold=0.5,   # 50% overlap required
                                sample_interval=1.5,     # Sample every 1.5 meters
                                d0=1.0,                  # 1 meter scale for geometry score
                                weights={'f1': 0.5, 'topo': 0.3, 'geom': 0.2}
                            )
                            
                            # Store for later use
                            st.session_state.nqs_result = nqs_result
                            st.session_state.nqs_bounds = bbox
                            st.session_state.nqs_osm_network = osm_network
                    except Exception as e:
                        st.warning(f"Could not compute NQS: {str(e)}. Showing basic network statistics.")
                        nqs_result = None
            else:
                nqs_result = st.session_state.nqs_result
        
        # Display NQS scores if available
        if nqs_result is not None:
            st.subheader("Network Quality Score (NQS)")
            
            # Main NQS score
            nqs_score = nqs_result['NQS']
            nqs_color = get_quality_color(nqs_score)
            
            # Get other scores
            f1_score = nqs_result['F1'] * 100
            f1_color = get_quality_color(f1_score)
            topo_score = nqs_result['S_topo'] * 100
            topo_color = get_quality_color(topo_score)
            geom_score = nqs_result['S_geom'] * 100
            geom_color = get_quality_color(geom_score)
            
            # 2-column layout: NQS on left (large), other scores on right
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown(f"""
                <div style="text-align:center; padding:2.5rem 2rem; border-radius:0.5rem; background-color:{nqs_color}20; border: 3px solid {nqs_color};">
                    <h1 style="color:{nqs_color}; margin:0; font-size:4.5rem; font-weight:bold;">{nqs_score:.1f}</h1>
                    <p style="margin:0.5rem 0 0 0; font-weight:bold; font-size:1.2rem; color:#fafafa;">NQS (0-100)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_right:
                # F1 Score
                st.markdown(f"""
                <div style="padding:0.75rem; border-radius:0.5rem; background-color:{f1_color}20; border: 2px solid {f1_color}; margin-bottom: 8px;">
                    <div style="display:flex; align-items:center; justify-content:center; gap:1rem;">
                        <span style="color:{f1_color}; font-size:2rem; font-weight:bold;">{f1_score:.1f}</span>
                        <div style="text-align:left;">
                            <p style="margin:0; font-weight:bold; color:#fafafa;">F1 Score</p>
                            <p style="margin:0; font-size:0.8rem; color:#9ca3af;">Coverage & Purity</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Topology Score
                st.markdown(f"""
                <div style="padding:0.75rem; border-radius:0.5rem; background-color:{topo_color}20; border: 2px solid {topo_color}; margin-bottom: 8px;">
                    <div style="display:flex; align-items:center; justify-content:center; gap:1rem;">
                        <span style="color:{topo_color}; font-size:2rem; font-weight:bold;">{topo_score:.1f}</span>
                        <div style="text-align:left;">
                            <p style="margin:0; font-weight:bold; color:#fafafa;">Topology Score</p>
                            <p style="margin:0; font-size:0.8rem; color:#9ca3af;">Structure Match</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Geometry Score
                st.markdown(f"""
                <div style="padding:0.75rem; border-radius:0.5rem; background-color:{geom_color}20; border: 2px solid {geom_color};">
                    <div style="display:flex; align-items:center; justify-content:center; gap:1rem;">
                        <span style="color:{geom_color}; font-size:2rem; font-weight:bold;">{geom_score:.1f}</span>
                        <div style="text-align:left;">
                            <p style="margin:0; font-weight:bold; color:#fafafa;">Geometry Score</p>
                            <p style="margin:0; font-size:0.8rem; color:#9ca3af;">Alignment Quality</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Spacer
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            
            # Interpretation
            st.info(f"**Interpretation:** {nqs_result['interpretation']}")
            
            st.divider()
            
            # Detailed score breakdown
            with st.expander("Score Calculation Details", expanded=True):
                st.markdown(f"""
                ### Network Quality Score Formula
                
                **NQS = 100 × (0.5 × F1 + 0.3 × S_topo + 0.2 × S_geom)**
                
                The ground truth is OpenStreetMap (OSM) pedestrian network.
                
                ---
                
                ### 1. Coverage & Purity (F1 Score): {nqs_result['F1']*100:.1f}%
                
                **Edge Matching Process:**
                - Distance tolerance: {nqs_result['params']['distance_tolerance']} meters
                - Overlap threshold: {nqs_result['params']['overlap_threshold']*100:.0f}% of sampled points must match
                - Sample interval: {nqs_result['params']['sample_interval']} meters
                
                **Length Statistics:**
                - Ground Truth (OSM) total length: {nqs_result['L_GT']:.1f} m
                - GT matched length: {nqs_result['L_match_GT']:.1f} m ({(nqs_result['L_match_GT']/nqs_result['L_GT']*100) if nqs_result['L_GT'] > 0 else 0:.1f}%)
                - Predicted (Tile2Net) total length: {nqs_result['L_pred']:.1f} m
                - Predicted matched length: {nqs_result['L_match_pred']:.1f} m ({(nqs_result['L_match_pred']/nqs_result['L_pred']*100) if nqs_result['L_pred'] > 0 else 0:.1f}%)
                
                **Metrics:**
                - **Recall (Coverage)** = L_match_GT / L_GT = **{nqs_result['recall']*100:.1f}%**
                  - *"How much of the OSM network did we capture?"*
                - **Precision (Purity)** = L_match_pred / L_pred = **{nqs_result['precision']*100:.1f}%**
                  - *"How much of our predictions are actually real?"*
                - **F1** = 2 × P × R / (P + R) = **{nqs_result['F1']*100:.1f}%**
                
                ---
                
                ### 2. Topology Score: {nqs_result['S_topo']*100:.1f}%
                
                **Connected Components (β₀):**
                - Ground Truth: {nqs_result['beta0_GT']} components
                - Predicted: {nqs_result['beta0_pred']} components
                - S_comp = 1 - |diff| / max(GT, 1) = **{nqs_result['S_comp']*100:.1f}%**
                
                **Independent Cycles (β₁):**
                - Ground Truth: {nqs_result['beta1_GT']} cycles
                - Predicted: {nqs_result['beta1_pred']} cycles
                - S_cycle = 1 - |diff| / max(GT, 1) = **{nqs_result['S_cycle']*100:.1f}%**
                
                **S_topo = (S_comp + S_cycle) / 2 = {nqs_result['S_topo']*100:.1f}%**
                
                ---
                
                ### 3. Geometric Alignment Score: {nqs_result['S_geom']*100:.1f}%
                
                **Offset Distance:**
                - Median distance from matched GT edges to predicted: **{nqs_result['d_med']:.2f} m**
                - Mean distance: {nqs_result['d_mean']:.2f} m
                
                **S_geom = exp(-d_med / d₀) where d₀ = {nqs_result['params']['d0']} m**
                
                *Lower median distance = higher geometric alignment score*
                """)
            
            # Score interpretation guide
            with st.expander("Score Interpretation Guide"):
                st.markdown("""
                | NQS Range | Quality | Description |
                |-----------|---------|-------------|
                | **90-100** | 🟢 Excellent | Very close to ground truth in edges, structure, and geometry |
                | **80-90** | 🟢 Very Good | Likely usable; small local issues |
                | **60-80** | 🟡 Usable | Noticeable gaps, false positives, or structural differences |
                | **Below 60** | 🔴 Needs Work | Significant missing or spurious structure |
                
                **Component Scores:**
                - **F1 (50% weight)**: Measures overall coverage and purity. High F1 means you captured most of the real network without hallucinating extra edges.
                - **Topology (30% weight)**: Measures structural similarity. Similar component count and cycle count means the network "feels" the same.
                - **Geometry (20% weight)**: Measures positional accuracy. Lower median offset means better alignment with the true path locations.
                """)
        
        else:
            # Fallback to basic scores when OSM not available
            st.subheader("Basic Quality Scores")
            st.info("OSM comparison not available. Showing basic network metrics.")
            
            col1, col2, col3, col4 = st.columns(4)
            
            for i, (col, (name, key)) in enumerate(zip(
                [col1, col2, col3, col4],
                [("Overall", "overall"), ("Connectivity", "connectivity"), 
                 ("Dead-End", "dead_end"), ("Intersection", "intersection")]
            )):
                with col:
                    score = quality_scores[key]
                    color = get_quality_color(score)
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem; border-radius:0.5rem; background-color:{color}20; border: 2px solid {color}">
                        <h2 style="color:{color}; margin:0">{score:.1f}</h2>
                        <p style="margin:0; font-weight:bold">{name}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        st.subheader("Key Statistics")
        
        # Calculate total and average length in meters
        total_length_m = edge_lengths_meters.sum()
        avg_length_m = edge_lengths_meters.mean() if len(edge_lengths_meters) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Edges", basic_stats['num_edges'])
            st.metric("Total Nodes", basic_stats['num_nodes'])
        with col2:
            st.metric("Total Length", f"{total_length_m:.0f} m")
            st.metric("Avg Edge Length", f"{avg_length_m:.2f} m")
        with col3:
            st.metric("Connected Components", len(components))
            st.metric("Largest Component", f"{connectivity['largest_component_ratio']*100:.1f}%")
        with col4:
            st.metric("Dead Ends", len(dead_ends))
            st.metric("Intersections", len(intersections))
        
        # Issues summary based on NQS if available
        st.divider()
        st.subheader("Issues Summary")
        issues = []
        
        if nqs_result is not None:
            if nqs_result['recall'] < 0.7:
                issues.append(f"Low coverage: Only {nqs_result['recall']*100:.1f}% of OSM network captured")
            if nqs_result['precision'] < 0.7:
                issues.append(f"Low purity: {(1-nqs_result['precision'])*100:.1f}% of predicted edges are false positives")
            if nqs_result['S_comp'] < 0.7:
                issues.append(f"Component mismatch: GT has {nqs_result['beta0_GT']} components, predicted has {nqs_result['beta0_pred']}")
            if nqs_result['S_cycle'] < 0.7:
                issues.append(f"Cycle mismatch: GT has {nqs_result['beta1_GT']} cycles, predicted has {nqs_result['beta1_pred']}")
            if nqs_result['d_med'] > 2.0:
                issues.append(f"Poor geometric alignment: Median offset is {nqs_result['d_med']:.2f}m")
        else:
            if len(components) > 1:
                issues.append(f"Network has {len(components)} disconnected components")
            if len(dead_ends) > basic_stats['num_nodes'] * 0.3:
                issues.append(f"High proportion of dead-ends ({len(dead_ends)/basic_stats['num_nodes']*100:.1f}%)")
            if quality_scores['connectivity'] < 50:
                issues.append("Low connectivity score - network is fragmented")
        
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("Network quality looks good! No major issues detected.")
    
    elif current_tab == "Graph Metrics":
        st.header("Graph Metrics Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Degree Distribution")
            if degree_dist['degrees']:
                degree_colors = []
                for d in degree_dist['degrees']:
                    if d == 1:
                        degree_colors.append('#f87171')
                    elif d == 2:
                        degree_colors.append('#9ca3af')
                    else:
                        degree_colors.append('#34d399')
                
                fig = go.Figure(data=[go.Bar(x=degree_dist['degrees'], y=degree_dist['counts'], marker_color=degree_colors)])
                fig.update_layout(xaxis_title='Node Degree', yaxis_title='Count', title='Distribution of Node Degrees', showlegend=False)
                apply_dark_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                **Legend:** <span style="color:#f87171">●</span> Dead-ends (degree 1) | 
                <span style="color:#9ca3af">●</span> Normal (degree 2) | 
                <span style="color:#34d399">●</span> Intersections (degree ≥3)
                
                **Stats:** Mean: {degree_dist['stats']['mean']:.2f}, Max: {degree_dist['stats']['max']}
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Network Indices")
            st.metric("Alpha Index (Circuitry)", f"{connectivity['alpha_index']:.3f}",
                     help="0 = tree structure, 1 = maximum circuits")
            st.metric("Beta Index", f"{connectivity['beta_index']:.3f}",
                     help="Ratio of edges to nodes")
            st.metric("Gamma Index", f"{connectivity['gamma_index']:.3f}",
                     help="Ratio of actual to max possible edges")
            st.metric("Average Clustering", f"{connectivity['average_clustering']:.3f}",
                     help="Triangle formation measure")
        
        if connectivity['alpha_index'] == 0 or connectivity['average_clustering'] == 0:
            st.info("**Note:** Alpha Index = 0 means no cycles. Clustering = 0 means no triangles. Both common in pedestrian networks.")
        
        st.divider()
        st.subheader("Betweenness Centrality")
        if betweenness:
            fig = px.histogram(x=list(betweenness.values()), nbins=50, 
                              title='Distribution of Node Betweenness Centrality',
                              labels={'x': 'Betweenness Centrality', 'y': 'Count'})
            apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            st.markdown("**Most Central Nodes:**")
            st.dataframe(pd.DataFrame(top_nodes, columns=['Node ID', 'Betweenness Centrality']), use_container_width=True)
    
    elif current_tab == "Network Map":
        st.header("Interactive Network Map")
        st.markdown("**Layer Controls:** Use the layer control (top right) to toggle visibility.")
        st.caption("Note: Map interactions may cause brief reloads due to Streamlit's rendering model.")
        
        node_mode = st.radio("Node Display Mode", ['Plain', 'Size proportional to degree'], horizontal=True)
        enable_max_edges = st.checkbox("Filter components by max edges", value=False)
        max_edges_threshold = st.slider("Max edges", 1, 500, 50, disabled=not enable_max_edges)
        
        if enable_max_edges:
            filtered_component_gdf = analyzer.get_component_gdf_with_edge_filter(max_edges=max_edges_threshold)
        else:
            filtered_component_gdf = component_gdf.copy()
            filtered_component_gdf['is_oversized'] = False
            filtered_component_gdf['edge_count'] = 1
        
        m = create_network_map_with_layers(
            network_gdf, dead_ends_gdf=dead_ends, intersections_gdf=intersections,
            component_gdf=filtered_component_gdf, all_nodes_gdf=all_nodes_degree,
            node_mode='proportional' if 'proportional' in node_mode.lower() else 'plain',
            bbox=bbox, show_bbox=st.session_state.show_bbox
        )
        st_folium(m, width=1200, height=500, key="network_map")
        
        st.subheader("Node Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dead-ends (degree 1)", len(all_nodes_degree[all_nodes_degree['degree'] == 1]), help="Red nodes")
        with col2:
            st.metric("Normal nodes (degree 2)", len(all_nodes_degree[all_nodes_degree['degree'] == 2]), help="Gray nodes (hidden)")
        with col3:
            st.metric("Intersections (degree ≥3)", len(all_nodes_degree[all_nodes_degree['degree'] >= 3]), help="Green nodes")
        
        st.divider()
        st.subheader("Connected Components")
        if len(components) > 1:
            st.warning(f"Network has {len(components)} disconnected components!")
            comp_df = pd.DataFrame(components).drop(columns=['node_ids'])
            st.dataframe(comp_df, use_container_width=True)
            
            isolated = analyzer.get_isolated_subgraphs(min_size_threshold=5)
            if isolated:
                st.subheader("Isolated Subgraphs (< 5 nodes)")
                st.warning(f"Found {len(isolated)} small isolated subgraphs")
                st.dataframe(pd.DataFrame(isolated).drop(columns=['node_ids']), use_container_width=True)
        else:
            st.success("Network is fully connected (single component)")
            st.metric("Total Nodes", components[0]['num_nodes'])
            st.metric("Total Edges", components[0]['num_edges'])
    
    elif current_tab == "Gap Diagnostics":
        st.header("Gap & Snapping Diagnostics")
        st.markdown("Identifies potential gaps where endpoints are close but not connected.")
        
        try:
            utm_crs = get_utm_crs(network_gdf)
            network_proj = network_gdf.to_crs(utm_crs)
            is_projected = True
        except:
            network_proj = network_gdf
            is_projected = False
        
        gap_threshold = st.slider("Near-miss threshold (meters)", 0.5, 20.0, 2.0, help="Max distance for near-miss")
        
        if is_projected:
            analyzer_proj = NetworkAnalyzer(network_proj)
            near_miss_data = analyzer_proj.find_near_miss_endpoints(max_distance=gap_threshold)
            if len(near_miss_data['endpoints']) > 0:
                near_miss_data['endpoints'] = near_miss_data['endpoints'].to_crs(network_gdf.crs)
            if len(near_miss_data['missing_links']) > 0:
                near_miss_data['missing_links'] = near_miss_data['missing_links'].to_crs(network_gdf.crs)
        else:
            near_miss_data = analyzer.find_near_miss_endpoints(max_distance=gap_threshold)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Endpoints (dead-ends)", len(near_miss_data['endpoints']))
        with col2:
            st.metric("Near-miss Pairs Found", len(near_miss_data['near_miss_pairs']))
        
        if len(near_miss_data['near_miss_pairs']) > 0:
            st.warning(f"Found **{len(near_miss_data['near_miss_pairs'])}** near-miss endpoint pairs!")
            st.markdown("**Interpretation:** Clusters at intersections → snapping threshold too tight")
        else:
            st.success("No near-miss endpoints found at this threshold.")
        
        m_gaps = create_gap_diagnostics_map(network_gdf, near_miss_data, bbox=bbox, show_bbox=st.session_state.show_bbox)
        st_folium(m_gaps, width=1200, height=500, key="gap_map")
        
        if len(near_miss_data['missing_links']) > 0:
            st.subheader("Near-miss Endpoint Pairs")
            display_df = near_miss_data['missing_links'][['distance', 'node1', 'node2']].copy()
            display_df['distance'] = display_df['distance'].round(2)
            display_df.columns = ['Distance (m)', 'Node 1', 'Node 2']
            st.dataframe(display_df.head(20), use_container_width=True)
    
    elif current_tab == "Geometry Checks":
        st.header("Intersection Angle Analysis")
        st.markdown("""
        **Color coding:** Green = ~90° angles, Yellow = moderate, Red = extreme (<30° or >150°)
        """)
        
        angles_gdf = analyzer.compute_intersection_angles()
        
        if len(angles_gdf) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Intersections", len(angles_gdf))
            with col2:
                st.metric("Good Angle Intersections", len(angles_gdf[angles_gdf['right_angle_score'] > 0.5]))
            with col3:
                st.metric("Extreme Angle Intersections", len(angles_gdf[(angles_gdf['min_angle'] < 30) | (angles_gdf['min_angle'] > 150)]))
            
            m_angles = create_intersection_angles_map(network_gdf, angles_gdf, bbox=bbox, show_bbox=st.session_state.show_bbox)
            st_folium(m_angles, width=1200, height=450, key="angles_map")
            
            fig = px.histogram(angles_gdf, x='min_angle', nbins=36, title='Distribution of Minimum Angles at Intersections',
                              labels={'min_angle': 'Minimum Angle (degrees)', 'count': 'Count'})
            fig.add_vline(x=30, line_dash="dash", line_color="#f87171", annotation_text="Acute threshold")
            fig.add_vline(x=90, line_dash="dash", line_color="#34d399", annotation_text="Right angle")
            apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No intersections (degree ≥ 3) found in the network.")
    
    elif current_tab == "Edge Analysis":
        st.header("Edge Length & Topology Analysis")
        tab_length, tab_loops = st.tabs(["Edge Length", "Loops & Spurs"])
        
        with tab_length:
            st.subheader("Edge Length Visualization")
            st.markdown("**Colors:** Red = very short, Cyan = very long, Gray = normal")
            
            try:
                utm_crs = get_utm_crs(network_gdf)
                network_proj = network_gdf.to_crs(utm_crs)
                analyzer_proj = NetworkAnalyzer(network_proj)
                is_projected = True
            except:
                analyzer_proj = analyzer
                is_projected = False
            
            col1, col2 = st.columns(2)
            with col1:
                short_thresh = st.slider("Short edge threshold (m)", 0.1, 20.0, 5.0)
            with col2:
                long_thresh = st.slider("Long edge threshold (m)", 50.0, 500.0, 100.0)
            
            edges_by_length = analyzer_proj.get_edges_by_length(short_threshold=short_thresh, long_threshold=long_thresh)
            
            if is_projected:
                for key in ['all_edges', 'short_edges', 'long_edges']:
                    if len(edges_by_length[key]) > 0:
                        edges_by_length[key] = edges_by_length[key].to_crs(network_gdf.crs)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Very Short Edges", len(edges_by_length['short_edges']))
            with col2:
                st.metric("Normal Edges", len(edges_by_length['all_edges'][edges_by_length['all_edges']['category'] == 'normal']))
            with col3:
                st.metric("Very Long Edges", len(edges_by_length['long_edges']))
            
            m_length = create_edge_length_map(network_gdf, edges_by_length, bbox=bbox, show_bbox=st.session_state.show_bbox)
            st_folium(m_length, width=1200, height=450, key="edge_length_map")
        
        with tab_loops:
            st.subheader("Small Loops & Short Spurs Detection")
            st.markdown("**Detection:** Cyan polygons = small loops, Red lines = short spurs")
            
            col1, col2 = st.columns(2)
            with col1:
                loop_thresh = st.slider("Max loop perimeter (m)", 10.0, 200.0, 50.0)
            with col2:
                spur_thresh = st.slider("Max spur length (m)", 1.0, 50.0, 10.0)
            
            if is_projected:
                small_loops = analyzer_proj.find_small_loops(max_perimeter=loop_thresh)
                short_spurs = analyzer_proj.find_short_spurs(max_length=spur_thresh)
                if len(small_loops) > 0:
                    small_loops = small_loops.to_crs(network_gdf.crs)
                if len(short_spurs) > 0:
                    short_spurs = short_spurs.to_crs(network_gdf.crs)
            else:
                small_loops = analyzer.find_small_loops(max_perimeter=loop_thresh)
                short_spurs = analyzer.find_short_spurs(max_length=spur_thresh)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Small Loops Found", len(small_loops))
            with col2:
                st.metric("Short Spurs Found", len(short_spurs))
            
            m_loops = create_loops_spurs_map(network_gdf, small_loops, short_spurs, bbox=bbox, show_bbox=st.session_state.show_bbox)
            st_folium(m_loops, width=1200, height=450, key="loops_spurs_map")
    
    elif current_tab == "OSM Comparison":
        st.header("Network Comparison with OpenStreetMap")
        
        if not OSMNX_AVAILABLE:
            st.warning("OSM comparison requires `osmnx`. Install with: `pip install osmnx`")
            return
        
        bounds = bbox if bbox is not None else bounds_from_gdf(network_gdf.to_crs(epsg=4326))
        
        if 'osm_comparison_result' not in st.session_state:
            st.session_state.osm_comparison_result = None
            st.session_state.osm_bounds = None
            st.session_state.osm_comparator = None
        
        need_rerun = (st.session_state.osm_comparison_result is None or st.session_state.osm_bounds != bounds)
        
        if need_rerun:
            with st.spinner("Fetching OSM data and comparing networks..."):
                try:
                    comparator = OSMComparator(bounds)
                    comparator.fetch_osm_network()
                    comparison = comparator.calculate_coverage_metrics(network_gdf)
                    st.session_state.osm_comparison_result = comparison
                    st.session_state.osm_bounds = bounds
                    st.session_state.osm_comparator = comparator
                except Exception as e:
                    st.error(f"Error during OSM comparison: {str(e)}")
                    return
        
        comparison = st.session_state.osm_comparison_result
        
        if comparison and comparison.get('osm_available', False):
            st.subheader("Coverage Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", f"{comparison['precision']*100:.1f}%", help="Correctness")
            with col2:
                st.metric("Recall", f"{comparison['recall']*100:.1f}%", help="Completeness")
            with col3:
                st.metric("F1 Score", f"{comparison['f1_score']*100:.1f}%", help="Harmonic mean")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tile2Net Edges", comparison['ai_total_edges'])
            with col2:
                st.metric("OSM Edges", comparison['osm_total_edges'])
            with col3:
                st.metric("Tile2Net Only (FP)", len(comparison.get('ai_unmatched', [])))
            with col4:
                st.metric("OSM Only (FN)", len(comparison.get('osm_unmatched', [])))
            
            st.divider()
            osm_tabs = st.tabs(["Comparison Map", "Offset Arrows", "Error Density"])
            
            with osm_tabs[0]:
                st.subheader("Comparison Map")
                st.markdown("""
                **Legend:**
                - **Blue** = Tile2Net edges that spatially overlap with OSM edges (True Positives from Tile2Net perspective)
                - **Green** = OSM edges that spatially overlap with Tile2Net edges (True Positives from OSM perspective)
                - **Red** = Tile2Net edges with NO nearby OSM edge (False Positives - Tile2Net detected paths that don't exist in OSM)
                - **Orange** = OSM edges with NO nearby Tile2Net edge (False Negatives - real paths that Tile2Net missed)
                
                *Blue and Green often overlap but may differ slightly because matching is done by buffering each network 
                and checking intersection. An edge might match in one direction but not the other due to geometry differences.*
                """)
                m = create_comparison_map(network_gdf, comparison, bbox=bbox, show_bbox=st.session_state.show_bbox)
                st_folium(m, width=1200, height=500, key="osm_comparison_map")
            
            with osm_tabs[1]:
                st.subheader("Offset Arrows")
                st.markdown("""
                **What this shows:** Arrows from each OSM segment centroid to the closest point on the Tile2Net network.
                
                - **Arrow color:** Green = small offset (good alignment), Red = large offset (poor alignment)
                - **Gray lines:** OSM reference network
                - **Blue lines:** Tile2Net predicted network
                
                **What it tells you:**
                - Systematic spatial bias (e.g., all predicted lines shifted toward building side)
                - Poor geometric alignment at curvy paths or roundabouts
                - Areas where Tile2Net consistently over/under-shoots the true path location
                """)
                
                osm_network = st.session_state.osm_comparator.osm_network if st.session_state.osm_comparator else None
                
                if osm_network is not None and len(osm_network) > 0:
                    max_arrow_dist = st.slider("Max arrow distance (m)", 5, 50, 20, 
                                               help="Only show arrows shorter than this distance")
                    
                    with st.spinner("Computing offset arrows..."):
                        try:
                            utm_crs = get_utm_crs(network_gdf)
                            osm_proj = osm_network.to_crs(utm_crs)
                            network_proj = network_gdf.to_crs(utm_crs)
                            
                            # Compute arrows using the function from osm_comparison
                            arrows_gdf = compute_offset_arrows(osm_proj, network_proj, max_distance=max_arrow_dist)
                            
                            if len(arrows_gdf) > 0:
                                # Convert back to original CRS
                                arrows_gdf = arrows_gdf.to_crs(network_gdf.crs)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Arrows", len(arrows_gdf))
                                with col2:
                                    st.metric("Mean Offset", f"{arrows_gdf['distance'].mean():.2f} m")
                                with col3:
                                    st.metric("Max Offset", f"{arrows_gdf['distance'].max():.2f} m")
                                
                                m_arrows = create_offset_arrows_map(
                                    osm_network, network_gdf, arrows_gdf, bbox=bbox, show_bbox=st.session_state.show_bbox
                                )
                                st_folium(m_arrows, width=1200, height=500, key="offset_arrows_map")
                                
                                # Histogram of offsets
                                fig = px.histogram(
                                    arrows_gdf, x='distance', nbins=30,
                                    title='Distribution of Offset Distances',
                                    labels={'distance': 'Offset Distance (m)', 'count': 'Count'}
                                )
                                fig.add_vline(x=arrows_gdf['distance'].mean(), line_dash="dash", 
                                             line_color="#60a5fa", annotation_text="Mean")
                                apply_dark_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No offset arrows computed. OSM segments may be too far from Tile2Net network.")
                        except Exception as e:
                            st.error(f"Error computing offset arrows: {str(e)}")
            
            with osm_tabs[2]:
                st.subheader("Tiled Error Density Map")
                st.markdown("Grid tiles colored by error count. Empty tiles show outline only.")
                
                osm_network = st.session_state.osm_comparator.osm_network if st.session_state.osm_comparator else None
                
                if osm_network is not None and len(osm_network) > 0:
                    tile_size = st.slider("Tile size (m)", 25, 200, 50)
                    
                    # Cache key based on bounds and tile size - only recompute when these change
                    density_cache_key = f"density_{hash(str(bounds))}_{tile_size}"
                    
                    # Check if we need to recompute
                    if 'density_cache_key' not in st.session_state or st.session_state.density_cache_key != density_cache_key:
                        with st.spinner("Computing error density..."):
                            try:
                                utm_crs = get_utm_crs(network_gdf)
                                osm_proj = osm_network.to_crs(utm_crs)
                                network_proj = network_gdf.to_crs(utm_crs)
                                
                                analyzer_proj = NetworkAnalyzer(network_proj)
                                dead_ends_proj = analyzer_proj.get_dead_ends()
                                near_miss_data = analyzer_proj.find_near_miss_endpoints(max_distance=2.0)
                                
                                comparator_proj = OSMComparator(bounds)
                                comparator_proj.osm_network = osm_proj
                                comparison_proj = comparator_proj.compare_with_network(network_proj, buffer_distance=5.0, use_projected=False)
                                
                                tiles_gdf = compute_tiled_error_density(
                                    network_proj, osm_proj, comparison_proj, tile_size=tile_size,
                                    dead_ends_gdf=dead_ends_proj,
                                    near_misses_gdf=near_miss_data.get('missing_links', gpd.GeoDataFrame())
                                )
                                
                                if len(tiles_gdf) > 0:
                                    tiles_gdf = tiles_gdf.to_crs(network_gdf.crs)
                                
                                # Get worst tiles for gallery
                                worst_tiles = get_worst_tiles(tiles_gdf, n_tiles=10, min_errors=0)
                                if len(worst_tiles) > 0:
                                    worst_tiles = worst_tiles.to_crs(epsg=4326)
                                
                                # Convert networks to WGS84 for gallery display
                                network_wgs = network_gdf.to_crs(epsg=4326)
                                osm_wgs = osm_network.to_crs(epsg=4326)
                                
                                # Store in session state
                                st.session_state.density_cache_key = density_cache_key
                                st.session_state.density_tiles_gdf = tiles_gdf
                                st.session_state.density_worst_tiles = worst_tiles
                                st.session_state.density_network_wgs = network_wgs
                                st.session_state.density_osm_wgs = osm_wgs
                                st.session_state.gallery_selected_tile = 0
                            except Exception as e:
                                st.error(f"Error computing density: {str(e)}")
                                st.session_state.density_tiles_gdf = None
                    
                    # Use cached data
                    if 'density_tiles_gdf' in st.session_state and st.session_state.density_tiles_gdf is not None:
                        tiles_gdf = st.session_state.density_tiles_gdf
                        
                        if len(tiles_gdf) > 0:
                            tiles_with_errors = tiles_gdf[tiles_gdf['total_errors'] > 0]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Tiles", len(tiles_gdf))
                            with col2:
                                st.metric("Tiles with Errors", len(tiles_with_errors))
                            with col3:
                                if len(tiles_with_errors) > 0:
                                    st.metric("Max Errors/Tile", tiles_with_errors['total_errors'].max())
                            
                            m_density = create_error_density_map(network_gdf, tiles_gdf, bbox=bbox, show_bbox=st.session_state.show_bbox)
                            st_folium(m_density, width=1200, height=500, key="error_density_map")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total FP", int(tiles_gdf['false_positives'].sum()))
                            with col2:
                                st.metric("Total FN", int(tiles_gdf['false_negatives'].sum()))
                            with col3:
                                st.metric("Total Dead-ends", int(tiles_gdf['dead_ends'].sum()))
                            with col4:
                                st.metric("Total Near-misses", int(tiles_gdf['near_misses'].sum()))
                            
                            # --- Error Gallery Section ---
                            st.markdown("---")
                            st.subheader("Error Gallery")
                            st.markdown("**Worst tiles by error count** - Blue = Tile2Net, Green = OSM")
                            
                            worst_tiles = st.session_state.density_worst_tiles
                            network_wgs = st.session_state.density_network_wgs
                            osm_wgs = st.session_state.density_osm_wgs
                            
                            if len(worst_tiles) > 0:
                                # Initialize selected tile if not set
                                if 'gallery_selected_tile' not in st.session_state:
                                    st.session_state.gallery_selected_tile = 0
                                
                                # Create tile selector buttons in a grid
                                st.markdown("**Select a tile to view:**")
                                for row_start in range(0, min(10, len(worst_tiles)), 5):
                                    cols = st.columns(5)
                                    for i, col in enumerate(cols):
                                        idx = row_start + i
                                        if idx < len(worst_tiles):
                                            tile = worst_tiles.iloc[idx]
                                            with col:
                                                # Highlight selected tile
                                                is_selected = (idx == st.session_state.gallery_selected_tile)
                                                btn_type = "primary" if is_selected else "secondary"
                                                if st.button(
                                                    f"#{idx+1}: {tile['total_errors']} err",
                                                    key=f"tile_btn_{idx}",
                                                    type=btn_type,
                                                    use_container_width=True
                                                ):
                                                    st.session_state.gallery_selected_tile = idx
                                                    st.rerun()
                                                st.caption(f"FP:{tile['false_positives']} FN:{tile['false_negatives']}")
                                
                                st.markdown("---")
                                
                                # Display only the selected tile's map
                                selected_idx = st.session_state.gallery_selected_tile
                                if selected_idx < len(worst_tiles):
                                    tile = worst_tiles.iloc[selected_idx]
                                    
                                    col_info, col_map = st.columns([1, 3])
                                    
                                    with col_info:
                                        st.markdown(f"### Tile #{selected_idx + 1}")
                                        st.metric("Total Errors", tile['total_errors'])
                                        st.metric("False Positives", tile['false_positives'])
                                        st.metric("False Negatives", tile['false_negatives'])
                                        if 'dead_ends' in tile:
                                            st.metric("Dead Ends", int(tile['dead_ends']))
                                        if 'near_misses' in tile:
                                            st.metric("Near Misses", int(tile['near_misses']))
                                    
                                    with col_map:
                                        bounds_tile = tile.geometry.bounds
                                        center = [(bounds_tile[1] + bounds_tile[3]) / 2, (bounds_tile[0] + bounds_tile[2]) / 2]
                                        
                                        mini_m = folium.Map(
                                            location=center, zoom_start=18,
                                            tiles=None
                                        )
                                        folium.TileLayer(
                                            tiles='https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_2024/MapServer/tile/{z}/{y}/{x}',
                                            attr='NYC DoITT', maxNativeZoom=20, maxZoom=22
                                        ).add_to(mini_m)
                                        
                                        # Draw OSM edges (green)
                                        from shapely.geometry import box
                                        tile_box = box(bounds_tile[0], bounds_tile[1], bounds_tile[2], bounds_tile[3])
                                        for _, osm_row in osm_wgs.iterrows():
                                            if osm_row.geometry is not None and tile_box.intersects(osm_row.geometry):
                                                coords = []
                                                if osm_row.geometry.geom_type == 'LineString':
                                                    coords = [(c[1], c[0]) for c in osm_row.geometry.coords]
                                                elif osm_row.geometry.geom_type == 'MultiLineString':
                                                    for line in osm_row.geometry.geoms:
                                                        coords.extend([(c[1], c[0]) for c in line.coords])
                                                if coords:
                                                    folium.PolyLine(coords, weight=3, color='#34d399', opacity=0.8).add_to(mini_m)
                                        
                                        # Draw Tile2Net edges (blue)
                                        for _, net_row in network_wgs.iterrows():
                                            if net_row.geometry is not None and tile_box.intersects(net_row.geometry):
                                                coords = []
                                                if net_row.geometry.geom_type == 'LineString':
                                                    coords = [(c[1], c[0]) for c in net_row.geometry.coords]
                                                elif net_row.geometry.geom_type == 'MultiLineString':
                                                    for line in net_row.geometry.geoms:
                                                        coords.extend([(c[1], c[0]) for c in line.coords])
                                                if coords:
                                                    folium.PolyLine(coords, weight=3, color='#60a5fa', opacity=0.8).add_to(mini_m)
                                        
                                        # Draw tile boundary
                                        folium.Rectangle(
                                            bounds=[[bounds_tile[1], bounds_tile[0]], [bounds_tile[3], bounds_tile[2]]],
                                            color='#fbbf24', fill=False, weight=2
                                        ).add_to(mini_m)
                                        st_folium(mini_m, width=600, height=450, key="gallery_single_map")
                            else:
                                st.success("No tiles with errors found!")
        else:
            st.error("Could not fetch OSM data for comparison.")


if __name__ == "__main__":
    main()
