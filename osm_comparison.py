"""
OpenStreetMap Comparison Module
Fetches OSM pedestrian network data and compares with AI-generated networks.
Includes Network Quality Score (NQS) computation for network evaluation.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union, nearest_points
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    ox.settings.timeout = 30  # 30 second timeout
    ox.settings.log_console = False
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


class OSMComparator:
    """Compare AI-generated network with OpenStreetMap data."""
    
    def __init__(self, bounds: tuple, crs: str = "EPSG:4326"):
        """
        Initialize with geographic bounds.
        
        Args:
            bounds: (min_lat, max_lat, min_lon, max_lon) or (south, north, west, east)
            crs: Coordinate reference system
        """
        self.bounds = bounds  # (south, north, west, east)
        self.crs = crs
        self.osm_network = None
        self.osm_nodes = None
        
    def fetch_osm_network(self, network_type: str = 'walk') -> gpd.GeoDataFrame:
        """
        Fetch pedestrian network from OpenStreetMap.
        
        Args:
            network_type: Type of network ('walk', 'bike', 'all')
        """
        if not OSMNX_AVAILABLE:
            raise ImportError("osmnx is required for OSM comparison. Install with: pip install osmnx")
        
        south, north, west, east = self.bounds
        
        # Create bounding box - osmnx uses (left, bottom, right, top) = (west, south, east, north)
        bbox = (west, south, east, north)
        
        try:
            # Fetch network graph
            G = ox.graph_from_bbox(
                bbox=bbox, 
                network_type=network_type, 
                simplify=True,
                truncate_by_edge=True
            )
            
            # Convert to GeoDataFrames
            nodes, edges = ox.graph_to_gdfs(G)
            
            # Reset index for edges
            edges = edges.reset_index()
            nodes = nodes.reset_index()
            
            self.osm_network = edges
            self.osm_nodes = nodes
            
            return edges
            
        except Exception as e:
            print(f"Error fetching OSM data: {e}")
            # Return empty GeoDataFrame
            return gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)
    
    def get_osm_sidewalks(self) -> gpd.GeoDataFrame:
        """Fetch specifically sidewalk data from OSM."""
        if not OSMNX_AVAILABLE:
            raise ImportError("osmnx is required for OSM comparison.")
        
        south, north, west, east = self.bounds
        
        try:
            # Custom filter for sidewalks
            custom_filter = '["highway"~"footway|pedestrian|path|sidewalk|crossing"]'
            
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north),  # (left, bottom, right, top)
                custom_filter=custom_filter,
                simplify=True
            )
            
            _, edges = ox.graph_to_gdfs(G)
            return edges.reset_index()
            
        except Exception as e:
            print(f"Error fetching sidewalks: {e}")
            return gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)
    
    def compare_with_network(self, 
                             ai_network: gpd.GeoDataFrame, 
                             buffer_distance: float = 0.00005,  # ~5 meters in degrees
                             use_projected: bool = True) -> dict:
        """
        Compare AI-generated network with OSM network.
        
        Args:
            ai_network: AI-generated network GeoDataFrame
            buffer_distance: Distance for spatial matching
            use_projected: Whether to project to a local CRS for accurate distance calculation
        """
        if self.osm_network is None:
            self.fetch_osm_network()
        
        if self.osm_network is None or len(self.osm_network) == 0:
            return {
                'osm_available': False,
                'message': 'No OSM data available for comparison'
            }
        
        # Ensure same CRS
        ai_net = ai_network.copy()
        osm_net = self.osm_network.copy()
        
        if ai_net.crs != osm_net.crs:
            osm_net = osm_net.to_crs(ai_net.crs)
        
        # Project to local CRS for accurate distance calculations if needed
        if use_projected:
            # Use UTM zone based on centroid
            centroid = ai_net.unary_union.centroid
            utm_crs = f"EPSG:{32600 + int((centroid.x + 180) / 6) + 1}"
            try:
                ai_net_proj = ai_net.to_crs(utm_crs)
                osm_net_proj = osm_net.to_crs(utm_crs)
                buffer_dist = buffer_distance * 111000  # Convert degrees to meters roughly
            except:
                ai_net_proj = ai_net
                osm_net_proj = osm_net
                buffer_dist = buffer_distance
        else:
            ai_net_proj = ai_net
            osm_net_proj = osm_net
            buffer_dist = buffer_distance
        
        # Calculate lengths
        ai_total_length = ai_net_proj.geometry.length.sum()
        osm_total_length = osm_net_proj.geometry.length.sum()
        
        # Spatial matching: AI edges that intersect OSM edges
        osm_buffered = osm_net_proj.copy()
        osm_buffered['geometry'] = osm_buffered.geometry.buffer(buffer_dist)
        osm_union = unary_union(osm_buffered.geometry)
        
        ai_matched = ai_net_proj[ai_net_proj.geometry.intersects(osm_union)].copy()
        ai_unmatched = ai_net_proj[~ai_net_proj.geometry.intersects(osm_union)].copy()
        
        # AI edges buffered for reverse matching
        ai_buffered = ai_net_proj.copy()
        ai_buffered['geometry'] = ai_buffered.geometry.buffer(buffer_dist)
        ai_union = unary_union(ai_buffered.geometry)
        
        osm_matched = osm_net_proj[osm_net_proj.geometry.intersects(ai_union)].copy()
        osm_unmatched = osm_net_proj[~osm_net_proj.geometry.intersects(ai_union)].copy()
        
        # Convert back to original CRS for output
        results = {
            'osm_available': True,
            'ai_matched': ai_matched.to_crs(ai_network.crs) if len(ai_matched) > 0 else gpd.GeoDataFrame(),
            'ai_unmatched': ai_unmatched.to_crs(ai_network.crs) if len(ai_unmatched) > 0 else gpd.GeoDataFrame(),
            'osm_matched': osm_matched.to_crs(ai_network.crs) if len(osm_matched) > 0 else gpd.GeoDataFrame(),
            'osm_unmatched': osm_unmatched.to_crs(ai_network.crs) if len(osm_unmatched) > 0 else gpd.GeoDataFrame(),
            'ai_match_ratio': len(ai_matched) / len(ai_net_proj) if len(ai_net_proj) > 0 else 0,
            'osm_match_ratio': len(osm_matched) / len(osm_net_proj) if len(osm_net_proj) > 0 else 0,
            'ai_length_matched_ratio': ai_matched.geometry.length.sum() / ai_total_length if ai_total_length > 0 else 0,
            'osm_length_matched_ratio': osm_matched.geometry.length.sum() / osm_total_length if osm_total_length > 0 else 0,
            'ai_total_edges': len(ai_net_proj),
            'osm_total_edges': len(osm_net_proj),
            'ai_total_length': ai_total_length,
            'osm_total_length': osm_total_length
        }
        
        return results
    
    def calculate_coverage_metrics(self, ai_network: gpd.GeoDataFrame, timeout: int = 30) -> dict:
        """
        Calculate coverage metrics comparing AI and OSM networks.
        
        Args:
            ai_network: AI-generated network GeoDataFrame
            timeout: Timeout in seconds for OSM fetch
        """
        import signal
        
        # Simple timeout handling for the fetch
        try:
            comparison = self.compare_with_network(ai_network)
        except Exception as e:
            return {
                'osm_available': False,
                'message': f'Comparison failed: {str(e)}'
            }
        
        if not comparison.get('osm_available', False):
            return comparison
        
        # Precision: How much of AI network matches OSM (correctness)
        precision = comparison['ai_length_matched_ratio']
        
        # Recall: How much of OSM network is captured by AI (completeness)
        recall = comparison['osm_length_matched_ratio']
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            **comparison,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'interpretation': {
                'precision': f"AI network correctness: {precision*100:.1f}% matches OSM",
                'recall': f"OSM coverage by AI: {recall*100:.1f}%",
                'f1': f"Overall agreement score: {f1*100:.1f}%"
            }
        }


def get_osm_network_for_bounds(bounds: tuple, network_type: str = 'walk') -> gpd.GeoDataFrame:
    """
    Convenience function to get OSM network for given bounds.
    
    Args:
        bounds: (south, north, west, east) or (min_lat, max_lat, min_lon, max_lon)
        network_type: Type of network to fetch
    """
    comparator = OSMComparator(bounds)
    return comparator.fetch_osm_network(network_type)


def bounds_from_gdf(gdf: gpd.GeoDataFrame, buffer: float = 0.001) -> tuple:
    """Extract bounds from GeoDataFrame with optional buffer."""
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    # Return as (south, north, west, east) = (miny, maxy, minx, maxx)
    return (
        bounds[1] - buffer,  # south
        bounds[3] + buffer,  # north
        bounds[0] - buffer,  # west
        bounds[2] + buffer   # east
    )


def compute_offset_arrows(reference_gdf: gpd.GeoDataFrame, 
                          predicted_gdf: gpd.GeoDataFrame,
                          max_distance: float = 50.0) -> gpd.GeoDataFrame:
    """
    Compute offset arrows from reference segment centroids to closest points on predicted network.
    
    Args:
        reference_gdf: Reference network (e.g., OSM)
        predicted_gdf: Predicted network (e.g., Tile2Net)
        max_distance: Maximum distance to consider for arrows (in CRS units)
        
    Returns:
        GeoDataFrame with arrows (LineStrings), distances, and colors
    """
    from shapely.ops import nearest_points
    
    if len(reference_gdf) == 0 or len(predicted_gdf) == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'distance', 'ref_centroid'])
    
    # Ensure same CRS
    if reference_gdf.crs != predicted_gdf.crs:
        predicted_gdf = predicted_gdf.to_crs(reference_gdf.crs)
    
    # Create union of predicted network for nearest point calculation
    predicted_union = unary_union(predicted_gdf.geometry)
    
    arrows_data = []
    
    for idx, row in reference_gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        
        # Get centroid of reference segment
        centroid = row.geometry.centroid
        
        # Find nearest point on predicted network
        try:
            nearest_pt = nearest_points(centroid, predicted_union)[1]
            distance = centroid.distance(nearest_pt)
            
            if distance <= max_distance and distance > 0:
                # Create arrow from centroid to nearest point
                arrow = LineString([centroid, nearest_pt])
                arrows_data.append({
                    'geometry': arrow,
                    'distance': distance,
                    'ref_centroid': centroid,
                    'nearest_point': nearest_pt
                })
        except Exception:
            continue
    
    if arrows_data:
        return gpd.GeoDataFrame(arrows_data, crs=reference_gdf.crs)
    return gpd.GeoDataFrame(columns=['geometry', 'distance', 'ref_centroid', 'nearest_point'])


def compute_tiled_error_density(ai_network: gpd.GeoDataFrame,
                                osm_network: gpd.GeoDataFrame,
                                comparison_results: dict,
                                tile_size: float = 100.0,
                                dead_ends_gdf: gpd.GeoDataFrame = None,
                                near_misses_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    Divide the region into tiles and compute error density for each.
    
    Args:
        ai_network: AI-generated network
        osm_network: OSM reference network
        comparison_results: Results from compare_with_network
        tile_size: Size of each tile in CRS units (meters if projected)
        dead_ends_gdf: GeoDataFrame of dead-end nodes
        near_misses_gdf: GeoDataFrame of near-miss endpoint pairs
        
    Returns:
        GeoDataFrame with tile polygons and error counts
    """
    from shapely.geometry import box
    
    # Get bounds from both networks
    all_geoms = []
    if len(ai_network) > 0:
        all_geoms.append(ai_network.unary_union)
    if len(osm_network) > 0:
        all_geoms.append(osm_network.unary_union)
    
    if not all_geoms:
        return gpd.GeoDataFrame(columns=['geometry', 'false_positives', 'false_negatives', 
                                          'dead_ends', 'near_misses', 'error_density'])
    
    combined_bounds = unary_union(all_geoms).bounds
    minx, miny, maxx, maxy = combined_bounds
    
    # Create grid of tiles
    tiles_data = []
    
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile_box = box(x, y, x + tile_size, y + tile_size)
            
            # Count false positives (AI edges not in OSM)
            false_positives = 0
            if 'ai_unmatched' in comparison_results and len(comparison_results['ai_unmatched']) > 0:
                ai_unmatched = comparison_results['ai_unmatched']
                false_positives = sum(1 for _, row in ai_unmatched.iterrows() 
                                     if row.geometry is not None and tile_box.intersects(row.geometry))
            
            # Count false negatives (OSM edges not in AI)
            false_negatives = 0
            if 'osm_unmatched' in comparison_results and len(comparison_results['osm_unmatched']) > 0:
                osm_unmatched = comparison_results['osm_unmatched']
                false_negatives = sum(1 for _, row in osm_unmatched.iterrows() 
                                      if row.geometry is not None and tile_box.intersects(row.geometry))
            
            # Count dead ends in tile
            dead_end_count = 0
            if dead_ends_gdf is not None and len(dead_ends_gdf) > 0:
                dead_end_count = sum(1 for _, row in dead_ends_gdf.iterrows() 
                                     if row.geometry is not None and tile_box.contains(row.geometry))
            
            # Count near-miss pairs in tile
            near_miss_count = 0
            if near_misses_gdf is not None and len(near_misses_gdf) > 0:
                near_miss_count = sum(1 for _, row in near_misses_gdf.iterrows() 
                                      if row.geometry is not None and tile_box.intersects(row.geometry))
            
            total_errors = false_positives + false_negatives + dead_end_count + near_miss_count
            
            tiles_data.append({
                'geometry': tile_box,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'dead_ends': dead_end_count,
                'near_misses': near_miss_count,
                'total_errors': total_errors,
                'tile_x': x,
                'tile_y': y
            })
            
            y += tile_size
        x += tile_size
    
    if tiles_data:
        tiles_gdf = gpd.GeoDataFrame(tiles_data, crs=ai_network.crs if ai_network.crs else osm_network.crs)
        
        # Calculate error density as percentage of max errors
        max_errors = tiles_gdf['total_errors'].max()
        tiles_gdf['error_density'] = tiles_gdf['total_errors'] / max_errors if max_errors > 0 else 0
        
        return tiles_gdf
    
    return gpd.GeoDataFrame(columns=['geometry', 'false_positives', 'false_negatives', 
                                      'dead_ends', 'near_misses', 'total_errors', 'error_density'])


def get_worst_tiles(tiles_gdf: gpd.GeoDataFrame, 
                    n_tiles: int = 9,
                    min_errors: int = 1) -> gpd.GeoDataFrame:
    """
    Get the N worst tiles by error count for the error gallery.
    
    Args:
        tiles_gdf: GeoDataFrame from compute_tiled_error_density
        n_tiles: Number of worst tiles to return
        min_errors: Minimum error count to consider
        
    Returns:
        GeoDataFrame with the worst tiles sorted by error count
    """
    if len(tiles_gdf) == 0:
        return tiles_gdf
    
    # Filter tiles with at least min_errors
    filtered = tiles_gdf[tiles_gdf['total_errors'] >= min_errors].copy()
    
    # Sort by total errors descending
    sorted_tiles = filtered.sort_values('total_errors', ascending=False)
    
    return sorted_tiles.head(n_tiles)


# =============================================================================
# Network Quality Score (NQS) Functions
# =============================================================================

def sample_points_along_edge(geometry, sample_interval: float = 1.5) -> list:
    """
    Sample points along a LineString or MultiLineString at regular intervals.
    
    Args:
        geometry: Shapely LineString or MultiLineString
        sample_interval: Distance between sample points (in CRS units, typically meters)
        
    Returns:
        List of Point objects sampled along the edge
    """
    if geometry is None or geometry.is_empty:
        return []
    
    points = []
    
    if geometry.geom_type == 'MultiLineString':
        lines = list(geometry.geoms)
    else:
        lines = [geometry]
    
    for line in lines:
        if line is None or line.is_empty:
            continue
        
        length = line.length
        if length == 0:
            continue
        
        # Sample points along the line
        num_samples = max(2, int(length / sample_interval) + 1)
        for i in range(num_samples):
            fraction = i / (num_samples - 1) if num_samples > 1 else 0
            point = line.interpolate(fraction, normalized=True)
            points.append(point)
    
    return points


def compute_edge_matching(gt_network: gpd.GeoDataFrame,
                          pred_network: gpd.GeoDataFrame,
                          distance_tolerance: float = 4.0,
                          overlap_threshold: float = 0.5,
                          sample_interval: float = 1.5) -> dict:
    """
    Match edges between ground truth and predicted networks using point sampling.
    
    For each GT edge:
    - Sample points along the edge
    - For each sample point, find the nearest predicted edge
    - If at least overlap_threshold fraction of points are within distance_tolerance,
      mark the GT edge as "matched"
    
    Do the same symmetrically for predicted edges.
    
    Args:
        gt_network: Ground truth network (OSM) GeoDataFrame
        pred_network: Predicted network (Tile2Net) GeoDataFrame
        distance_tolerance: Maximum distance to consider a point as matched (meters)
        overlap_threshold: Minimum fraction of points that must match (0-1)
        sample_interval: Distance between sample points (meters)
        
    Returns:
        Dictionary with matching results including matched/unmatched edges and lengths
    """
    if len(gt_network) == 0 or len(pred_network) == 0:
        return {
            'gt_matched': gpd.GeoDataFrame(),
            'gt_unmatched': gt_network.copy() if len(gt_network) > 0 else gpd.GeoDataFrame(),
            'pred_matched': gpd.GeoDataFrame(),
            'pred_unmatched': pred_network.copy() if len(pred_network) > 0 else gpd.GeoDataFrame(),
            'L_GT': 0,
            'L_match_GT': 0,
            'L_pred': 0,
            'L_match_pred': 0,
            'gt_edge_distances': {},  # Median distances for each matched GT edge
            'pred_edge_distances': {}  # Median distances for each matched pred edge
        }
    
    # Ensure same CRS
    if gt_network.crs != pred_network.crs:
        pred_network = pred_network.to_crs(gt_network.crs)
    
    # Create union of predicted network for nearest point calculations
    pred_union = unary_union(pred_network.geometry)
    gt_union = unary_union(gt_network.geometry)
    
    # Match GT edges to predicted network
    gt_matched_indices = []
    gt_unmatched_indices = []
    gt_edge_distances = {}  # Store median distances for matched GT edges
    
    for idx, row in gt_network.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            gt_unmatched_indices.append(idx)
            continue
        
        # Sample points along GT edge
        sample_points = sample_points_along_edge(row.geometry, sample_interval)
        
        if len(sample_points) == 0:
            gt_unmatched_indices.append(idx)
            continue
        
        # Calculate distances to nearest predicted edge for each sample point
        distances = []
        for pt in sample_points:
            try:
                nearest_pt = nearest_points(pt, pred_union)[1]
                dist = pt.distance(nearest_pt)
                distances.append(dist)
            except:
                distances.append(float('inf'))
        
        # Count how many points are within tolerance
        within_tolerance = sum(1 for d in distances if d <= distance_tolerance)
        match_ratio = within_tolerance / len(sample_points)
        
        if match_ratio >= overlap_threshold:
            gt_matched_indices.append(idx)
            # Store median distance for geometry score
            valid_distances = [d for d in distances if d != float('inf')]
            if valid_distances:
                gt_edge_distances[idx] = np.median(valid_distances)
        else:
            gt_unmatched_indices.append(idx)
    
    # Match predicted edges to GT network (symmetric)
    pred_matched_indices = []
    pred_unmatched_indices = []
    pred_edge_distances = {}
    
    for idx, row in pred_network.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            pred_unmatched_indices.append(idx)
            continue
        
        # Sample points along predicted edge
        sample_points = sample_points_along_edge(row.geometry, sample_interval)
        
        if len(sample_points) == 0:
            pred_unmatched_indices.append(idx)
            continue
        
        # Calculate distances to nearest GT edge for each sample point
        distances = []
        for pt in sample_points:
            try:
                nearest_pt = nearest_points(pt, gt_union)[1]
                dist = pt.distance(nearest_pt)
                distances.append(dist)
            except:
                distances.append(float('inf'))
        
        # Count how many points are within tolerance
        within_tolerance = sum(1 for d in distances if d <= distance_tolerance)
        match_ratio = within_tolerance / len(sample_points)
        
        if match_ratio >= overlap_threshold:
            pred_matched_indices.append(idx)
            valid_distances = [d for d in distances if d != float('inf')]
            if valid_distances:
                pred_edge_distances[idx] = np.median(valid_distances)
        else:
            pred_unmatched_indices.append(idx)
    
    # Create result GeoDataFrames
    gt_matched = gt_network.loc[gt_matched_indices].copy() if gt_matched_indices else gpd.GeoDataFrame()
    gt_unmatched = gt_network.loc[gt_unmatched_indices].copy() if gt_unmatched_indices else gpd.GeoDataFrame()
    pred_matched = pred_network.loc[pred_matched_indices].copy() if pred_matched_indices else gpd.GeoDataFrame()
    pred_unmatched = pred_network.loc[pred_unmatched_indices].copy() if pred_unmatched_indices else gpd.GeoDataFrame()
    
    # Calculate lengths
    L_GT = gt_network.geometry.length.sum()
    L_match_GT = gt_matched.geometry.length.sum() if len(gt_matched) > 0 else 0
    L_pred = pred_network.geometry.length.sum()
    L_match_pred = pred_matched.geometry.length.sum() if len(pred_matched) > 0 else 0
    
    return {
        'gt_matched': gt_matched,
        'gt_unmatched': gt_unmatched,
        'pred_matched': pred_matched,
        'pred_unmatched': pred_unmatched,
        'L_GT': L_GT,
        'L_match_GT': L_match_GT,
        'L_pred': L_pred,
        'L_match_pred': L_match_pred,
        'gt_edge_distances': gt_edge_distances,
        'pred_edge_distances': pred_edge_distances
    }


def compute_f1_score(matching_results: dict) -> dict:
    """
    Compute length-weighted precision, recall, and F1 score.
    
    Recall (coverage) = L_match_GT / L_GT
    Precision (purity) = L_match_pred / L_pred
    F1 = 2 * P * R / (P + R)
    
    Args:
        matching_results: Results from compute_edge_matching
        
    Returns:
        Dictionary with recall, precision, and f1 scores (all 0-1)
    """
    L_GT = matching_results['L_GT']
    L_match_GT = matching_results['L_match_GT']
    L_pred = matching_results['L_pred']
    L_match_pred = matching_results['L_match_pred']
    
    # Recall: How much of GT did we capture?
    recall = L_match_GT / L_GT if L_GT > 0 else 0
    
    # Precision: How much of predicted is actually real?
    precision = L_match_pred / L_pred if L_pred > 0 else 0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1
    }


def build_network_graph(network_gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build a NetworkX graph from a network GeoDataFrame.
    
    Args:
        network_gdf: GeoDataFrame with LineString geometries
        
    Returns:
        NetworkX Graph
    """
    G = nx.Graph()
    nodes = {}
    node_id = 0
    
    for idx, row in network_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        if geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            lines = [geom]
        
        for line in lines:
            if line is None or line.is_empty:
                continue
            
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            
            start_pt = coords[0]
            end_pt = coords[-1]
            
            # Round coordinates for node matching
            start_key = (round(start_pt[0], 6), round(start_pt[1], 6))
            end_key = (round(end_pt[0], 6), round(end_pt[1], 6))
            
            if start_key not in nodes:
                nodes[start_key] = node_id
                G.add_node(node_id, pos=start_key)
                node_id += 1
            
            if end_key not in nodes:
                nodes[end_key] = node_id
                G.add_node(node_id, pos=end_key)
                node_id += 1
            
            G.add_edge(nodes[start_key], nodes[end_key], length=line.length)
    
    return G


def compute_topology_score(gt_network: gpd.GeoDataFrame,
                           pred_network: gpd.GeoDataFrame) -> dict:
    """
    Compute topology subscore based on connected components (beta0) and cycles (beta1).
    
    S_comp = 1 - min(1, |beta0_pred - beta0_GT| / max(beta0_GT, 1))
    S_cycle = 1 - min(1, |beta1_pred - beta1_GT| / max(beta1_GT, 1))
    S_topo = (S_comp + S_cycle) / 2
    
    beta0 = number of connected components
    beta1 = number of independent cycles = edges - nodes + components (Euler formula for graphs)
    
    Args:
        gt_network: Ground truth network GeoDataFrame
        pred_network: Predicted network GeoDataFrame
        
    Returns:
        Dictionary with component and cycle scores
    """
    # Build graphs
    gt_graph = build_network_graph(gt_network)
    pred_graph = build_network_graph(pred_network)
    
    # Compute beta0 (connected components)
    beta0_GT = nx.number_connected_components(gt_graph) if gt_graph.number_of_nodes() > 0 else 0
    beta0_pred = nx.number_connected_components(pred_graph) if pred_graph.number_of_nodes() > 0 else 0
    
    # Compute beta1 (independent cycles/cyclomatic complexity)
    # beta1 = edges - nodes + components (for undirected graphs)
    n_GT = gt_graph.number_of_nodes()
    m_GT = gt_graph.number_of_edges()
    beta1_GT = m_GT - n_GT + beta0_GT if n_GT > 0 else 0
    
    n_pred = pred_graph.number_of_nodes()
    m_pred = pred_graph.number_of_edges()
    beta1_pred = m_pred - n_pred + beta0_pred if n_pred > 0 else 0
    
    # Compute component score
    diff_comp = abs(beta0_pred - beta0_GT) / max(beta0_GT, 1)
    diff_comp = min(1, diff_comp)
    S_comp = 1 - diff_comp
    
    # Compute cycle score
    diff_cycle = abs(beta1_pred - beta1_GT) / max(beta1_GT, 1)
    diff_cycle = min(1, diff_cycle)
    S_cycle = 1 - diff_cycle
    
    # Combined topology score
    S_topo = (S_comp + S_cycle) / 2
    
    return {
        'beta0_GT': beta0_GT,
        'beta0_pred': beta0_pred,
        'beta1_GT': beta1_GT,
        'beta1_pred': beta1_pred,
        'S_comp': S_comp,
        'S_cycle': S_cycle,
        'S_topo': S_topo
    }


def compute_geometric_score(matching_results: dict, d0: float = 1.0) -> dict:
    """
    Compute geometric alignment subscore.
    
    For matched GT edges, compute the median distance to their matched predicted edges.
    S_geom = exp(-d_med / d0)
    
    Args:
        matching_results: Results from compute_edge_matching (includes gt_edge_distances)
        d0: Scale parameter for exponential decay (default 1 meter)
        
    Returns:
        Dictionary with geometric score and median distance
    """
    gt_edge_distances = matching_results.get('gt_edge_distances', {})
    
    if len(gt_edge_distances) == 0:
        return {
            'd_med': float('inf'),
            'S_geom': 0
        }
    
    # Compute median of all edge median distances
    all_distances = list(gt_edge_distances.values())
    d_med = np.median(all_distances)
    
    # Compute geometric score with exponential decay
    S_geom = np.exp(-d_med / d0)
    
    return {
        'd_med': d_med,
        'd_mean': np.mean(all_distances),
        'd_max': np.max(all_distances),
        'S_geom': S_geom
    }


def compute_network_quality_score(gt_network: gpd.GeoDataFrame,
                                   pred_network: gpd.GeoDataFrame,
                                   distance_tolerance: float = 4.0,
                                   overlap_threshold: float = 0.5,
                                   sample_interval: float = 1.5,
                                   d0: float = 1.0,
                                   weights: dict = None) -> dict:
    """
    Compute the complete Network Quality Score (NQS).
    
    NQS = 100 * (w_f1 * F1 + w_topo * S_topo + w_geom * S_geom)
    
    Default weights: 50% F1, 30% topology, 20% geometry
    
    Args:
        gt_network: Ground truth network (OSM) GeoDataFrame - should be in projected CRS (meters)
        pred_network: Predicted network (Tile2Net) GeoDataFrame - should be in projected CRS (meters)
        distance_tolerance: Distance tolerance for edge matching (meters)
        overlap_threshold: Minimum fraction of points that must match (0-1)
        sample_interval: Distance between sample points (meters)
        d0: Scale parameter for geometric score exponential decay
        weights: Dictionary with 'f1', 'topo', 'geom' weights (must sum to 1)
        
    Returns:
        Dictionary with all scores and component metrics
    """
    if weights is None:
        weights = {'f1': 0.5, 'topo': 0.3, 'geom': 0.2}
    
    # Ensure same CRS
    if len(pred_network) > 0 and len(gt_network) > 0:
        if pred_network.crs != gt_network.crs:
            pred_network = pred_network.to_crs(gt_network.crs)
    
    # Step 1: Edge matching
    matching_results = compute_edge_matching(
        gt_network, pred_network,
        distance_tolerance=distance_tolerance,
        overlap_threshold=overlap_threshold,
        sample_interval=sample_interval
    )
    
    # Step 2: F1 score
    f1_results = compute_f1_score(matching_results)
    
    # Step 3: Topology score
    topo_results = compute_topology_score(gt_network, pred_network)
    
    # Step 4: Geometric score
    geom_results = compute_geometric_score(matching_results, d0=d0)
    
    # Step 5: Combined NQS
    F1 = f1_results['f1']
    S_topo = topo_results['S_topo']
    S_geom = geom_results['S_geom']
    
    NQS = 100 * (weights['f1'] * F1 + weights['topo'] * S_topo + weights['geom'] * S_geom)
    
    # Interpretation
    if NQS >= 90:
        interpretation = "Excellent - Very close to ground truth in edges, structure, and geometry"
    elif NQS >= 80:
        interpretation = "Very Good - Likely usable with small local issues"
    elif NQS >= 60:
        interpretation = "Usable - Noticeable gaps, false positives, or structural differences"
    else:
        interpretation = "Needs Work - Significant missing or spurious structure"
    
    return {
        # Main scores
        'NQS': NQS,
        'F1': F1,
        'S_topo': S_topo,
        'S_geom': S_geom,
        'interpretation': interpretation,
        
        # F1 components
        'recall': f1_results['recall'],
        'precision': f1_results['precision'],
        
        # Topology components
        'beta0_GT': topo_results['beta0_GT'],
        'beta0_pred': topo_results['beta0_pred'],
        'beta1_GT': topo_results['beta1_GT'],
        'beta1_pred': topo_results['beta1_pred'],
        'S_comp': topo_results['S_comp'],
        'S_cycle': topo_results['S_cycle'],
        
        # Geometry components
        'd_med': geom_results['d_med'],
        'd_mean': geom_results.get('d_mean', 0),
        
        # Length statistics
        'L_GT': matching_results['L_GT'],
        'L_match_GT': matching_results['L_match_GT'],
        'L_pred': matching_results['L_pred'],
        'L_match_pred': matching_results['L_match_pred'],
        
        # Matched/unmatched edges for visualization
        'gt_matched': matching_results['gt_matched'],
        'gt_unmatched': matching_results['gt_unmatched'],
        'pred_matched': matching_results['pred_matched'],
        'pred_unmatched': matching_results['pred_unmatched'],
        
        # Parameters used
        'params': {
            'distance_tolerance': distance_tolerance,
            'overlap_threshold': overlap_threshold,
            'sample_interval': sample_interval,
            'd0': d0,
            'weights': weights
        }
    }
