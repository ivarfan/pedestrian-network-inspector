"""
Network Analysis Module for Pedestrian Network Quality Assessment
Provides topological and geometric analysis of pedestrian networks.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import linemerge, unary_union, nearest_points
from scipy import stats
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')


class NetworkAnalyzer:
    """Analyzes pedestrian network topology and geometry quality."""
    
    def __init__(self, network_gdf: gpd.GeoDataFrame):
        """
        Initialize analyzer with network GeoDataFrame.
        
        Args:
            network_gdf: GeoDataFrame with LineString geometries representing network edges
        """
        self.network_gdf = network_gdf.copy()
        self.graph = None
        self.nodes_gdf = None
        self._build_graph()
        
    def _build_graph(self):
        """Build NetworkX graph from network GeoDataFrame."""
        self.graph = nx.Graph()
        
        # Extract unique nodes from line endpoints
        nodes = {}
        node_id = 0
        
        for idx, row in self.network_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
                
            if isinstance(geom, MultiLineString):
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
                start_key = (round(start_pt[0], 7), round(start_pt[1], 7))
                end_key = (round(end_pt[0], 7), round(end_pt[1], 7))
                
                if start_key not in nodes:
                    nodes[start_key] = node_id
                    self.graph.add_node(node_id, pos=start_key, x=start_key[0], y=start_key[1])
                    node_id += 1
                    
                if end_key not in nodes:
                    nodes[end_key] = node_id
                    self.graph.add_node(node_id, pos=end_key, x=end_key[0], y=end_key[1])
                    node_id += 1
                    
                # Add edge with length attribute
                length = line.length
                self.graph.add_edge(
                    nodes[start_key], 
                    nodes[end_key], 
                    length=length,
                    geometry=line,
                    edge_idx=idx
                )
        
        # Create nodes GeoDataFrame
        node_data = []
        for node_id, data in self.graph.nodes(data=True):
            node_data.append({
                'node_id': node_id,
                'geometry': Point(data['x'], data['y']),
                'x': data['x'],
                'y': data['y']
            })
        
        if node_data:
            self.nodes_gdf = gpd.GeoDataFrame(node_data, crs=self.network_gdf.crs)
        else:
            self.nodes_gdf = gpd.GeoDataFrame(columns=['node_id', 'geometry', 'x', 'y'])
            
    def get_basic_stats(self) -> dict:
        """Get basic network statistics."""
        return {
            'num_edges': self.graph.number_of_edges(),
            'num_nodes': self.graph.number_of_nodes(),
            'total_length': sum(d.get('length', 0) for _, _, d in self.graph.edges(data=True)),
            'avg_edge_length': np.mean([d.get('length', 0) for _, _, d in self.graph.edges(data=True)]) if self.graph.number_of_edges() > 0 else 0,
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0
        }
    
    def get_connected_components(self) -> list:
        """Get connected components information."""
        components = list(nx.connected_components(self.graph))
        component_info = []
        
        for i, comp in enumerate(components):
            subgraph = self.graph.subgraph(comp)
            total_length = sum(d.get('length', 0) for _, _, d in subgraph.edges(data=True))
            component_info.append({
                'component_id': i,
                'num_nodes': len(comp),
                'num_edges': subgraph.number_of_edges(),
                'total_length': total_length,
                'node_ids': list(comp)
            })
            
        return sorted(component_info, key=lambda x: x['num_nodes'], reverse=True)
    
    def get_isolated_subgraphs(self, min_size_threshold: int = 3) -> list:
        """
        Identify isolated (small) subgraphs that may indicate issues.
        
        Args:
            min_size_threshold: Components with fewer nodes than this are flagged
        """
        components = self.get_connected_components()
        return [c for c in components if c['num_nodes'] < min_size_threshold]
    
    def get_dead_ends(self) -> gpd.GeoDataFrame:
        """Find dead-end nodes (degree 1 nodes)."""
        dead_ends = []
        
        for node_id in self.graph.nodes():
            degree = self.graph.degree(node_id)
            if degree == 1:
                data = self.graph.nodes[node_id]
                dead_ends.append({
                    'node_id': node_id,
                    'geometry': Point(data['x'], data['y']),
                    'x': data['x'],
                    'y': data['y'],
                    'degree': degree
                })
                
        if dead_ends:
            return gpd.GeoDataFrame(dead_ends, crs=self.network_gdf.crs)
        return gpd.GeoDataFrame(columns=['node_id', 'geometry', 'x', 'y', 'degree'])
    
    def get_node_degree_distribution(self) -> dict:
        """Get node degree distribution."""
        degrees = [d for _, d in self.graph.degree()]
        
        if not degrees:
            return {'degrees': [], 'counts': [], 'stats': {}}
            
        unique, counts = np.unique(degrees, return_counts=True)
        
        return {
            'degrees': unique.tolist(),
            'counts': counts.tolist(),
            'stats': {
                'mean': np.mean(degrees),
                'median': np.median(degrees),
                'max': max(degrees),
                'min': min(degrees),
                'std': np.std(degrees)
            }
        }
    
    def compute_betweenness_centrality(self) -> dict:
        """Compute betweenness centrality for nodes."""
        if self.graph.number_of_nodes() < 2:
            return {}
        return nx.betweenness_centrality(self.graph)
    
    def compute_edge_betweenness(self) -> dict:
        """Compute edge betweenness centrality."""
        if self.graph.number_of_edges() < 1:
            return {}
        return nx.edge_betweenness_centrality(self.graph)
    
    def get_intersection_nodes(self, min_degree: int = 3) -> gpd.GeoDataFrame:
        """Find intersection nodes (nodes with degree >= min_degree)."""
        intersections = []
        
        for node_id in self.graph.nodes():
            degree = self.graph.degree(node_id)
            if degree >= min_degree:
                data = self.graph.nodes[node_id]
                intersections.append({
                    'node_id': node_id,
                    'geometry': Point(data['x'], data['y']),
                    'x': data['x'],
                    'y': data['y'],
                    'degree': degree
                })
                
        if intersections:
            return gpd.GeoDataFrame(intersections, crs=self.network_gdf.crs)
        return gpd.GeoDataFrame(columns=['node_id', 'geometry', 'x', 'y', 'degree'])
    
    def analyze_edge_geometry(self) -> gpd.GeoDataFrame:
        """Analyze geometric properties of edges."""
        edge_data = []
        
        for u, v, data in self.graph.edges(data=True):
            geom = data.get('geometry')
            if geom is None:
                continue
                
            length = data.get('length', geom.length)
            
            # Calculate sinuosity (actual length / straight line distance)
            coords = list(geom.coords)
            straight_dist = Point(coords[0]).distance(Point(coords[-1]))
            sinuosity = length / straight_dist if straight_dist > 0 else 1.0
            
            # Count vertices (complexity)
            num_vertices = len(coords)
            
            edge_data.append({
                'start_node': u,
                'end_node': v,
                'length': length,
                'sinuosity': sinuosity,
                'num_vertices': num_vertices,
                'geometry': geom
            })
            
        if edge_data:
            return gpd.GeoDataFrame(edge_data, crs=self.network_gdf.crs)
        return gpd.GeoDataFrame(columns=['start_node', 'end_node', 'length', 'sinuosity', 'num_vertices', 'geometry'])
    
    def detect_geometric_anomalies(self, 
                                    min_length: float = 1.0,
                                    max_sinuosity: float = 1.5,
                                    length_percentile_threshold: float = 5) -> dict:
        """
        Detect geometric anomalies in the network.
        
        Args:
            min_length: Minimum expected edge length (in CRS units)
            max_sinuosity: Maximum expected sinuosity
            length_percentile_threshold: Flag edges below this percentile as too short
        """
        edge_analysis = self.analyze_edge_geometry()
        
        if len(edge_analysis) == 0:
            return {
                'too_short': gpd.GeoDataFrame(),
                'too_sinuous': gpd.GeoDataFrame(),
                'very_long': gpd.GeoDataFrame()
            }
        
        # Calculate thresholds
        length_lower = np.percentile(edge_analysis['length'], length_percentile_threshold)
        length_upper = np.percentile(edge_analysis['length'], 100 - length_percentile_threshold)
        
        anomalies = {
            'too_short': edge_analysis[edge_analysis['length'] < max(min_length, length_lower)].copy(),
            'too_sinuous': edge_analysis[edge_analysis['sinuosity'] > max_sinuosity].copy(),
            'very_long': edge_analysis[edge_analysis['length'] > length_upper].copy()
        }
        
        return anomalies
    
    def compute_connectivity_metrics(self) -> dict:
        """Compute network connectivity metrics."""
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()
        
        if n < 2:
            return {
                'alpha_index': 0,
                'beta_index': 0,
                'gamma_index': 0,
                'average_clustering': 0,
                'num_connected_components': 0,
                'largest_component_ratio': 0
            }
        
        # Alpha index: (m - n + 1) / (2n - 5) for planar graphs
        alpha = (m - n + 1) / (2 * n - 5) if n > 2 else 0
        
        # Beta index: m / n
        beta = m / n if n > 0 else 0
        
        # Gamma index: m / (3 * (n - 2)) for planar graphs
        gamma = m / (3 * (n - 2)) if n > 2 else 0
        
        # Connected components
        components = list(nx.connected_components(self.graph))
        largest_comp = max(components, key=len) if components else set()
        
        return {
            'alpha_index': max(0, min(1, alpha)),  # Clamp to [0, 1]
            'beta_index': beta,
            'gamma_index': max(0, min(1, gamma)),  # Clamp to [0, 1]
            'average_clustering': nx.average_clustering(self.graph) if n > 2 else 0,
            'num_connected_components': len(components),
            'largest_component_ratio': len(largest_comp) / n if n > 0 else 0
        }
    
    def get_component_gdf(self) -> gpd.GeoDataFrame:
        """Create GeoDataFrame with component labels for visualization."""
        components = list(nx.connected_components(self.graph))
        
        # Create node to component mapping
        node_to_comp = {}
        for comp_id, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = comp_id
        
        # Add component info to edges
        edge_data = []
        for u, v, data in self.graph.edges(data=True):
            geom = data.get('geometry')
            if geom is not None:
                edge_data.append({
                    'component_id': node_to_comp.get(u, -1),
                    'geometry': geom,
                    'length': data.get('length', 0)
                })
        
        if edge_data:
            gdf = gpd.GeoDataFrame(edge_data, crs=self.network_gdf.crs)
            return gdf
        return gpd.GeoDataFrame(columns=['component_id', 'geometry', 'length'])
    
    def get_component_gdf_with_edge_filter(self, max_edges: int = None) -> gpd.GeoDataFrame:
        """
        Create GeoDataFrame with component labels, graying out components above max_edges.
        
        Args:
            max_edges: Maximum edges threshold. Components with more edges are marked as 'oversized'.
        """
        components = list(nx.connected_components(self.graph))
        
        # Create node to component mapping and calculate component sizes
        node_to_comp = {}
        comp_edge_counts = {}
        
        for comp_id, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = comp_id
            subgraph = self.graph.subgraph(comp)
            comp_edge_counts[comp_id] = subgraph.number_of_edges()
        
        # Add component info to edges
        edge_data = []
        for u, v, data in self.graph.edges(data=True):
            geom = data.get('geometry')
            if geom is not None:
                comp_id = node_to_comp.get(u, -1)
                edge_count = comp_edge_counts.get(comp_id, 0)
                is_oversized = max_edges is not None and edge_count > max_edges
                
                edge_data.append({
                    'component_id': comp_id,
                    'geometry': geom,
                    'length': data.get('length', 0),
                    'edge_count': edge_count,
                    'is_oversized': is_oversized
                })
        
        if edge_data:
            gdf = gpd.GeoDataFrame(edge_data, crs=self.network_gdf.crs)
            return gdf
        return gpd.GeoDataFrame(columns=['component_id', 'geometry', 'length', 'edge_count', 'is_oversized'])
    
    def get_all_nodes_with_degree(self) -> gpd.GeoDataFrame:
        """Get all nodes with their degree for visualization."""
        node_data = []
        
        for node_id in self.graph.nodes():
            degree = self.graph.degree(node_id)
            data = self.graph.nodes[node_id]
            node_data.append({
                'node_id': node_id,
                'geometry': Point(data['x'], data['y']),
                'x': data['x'],
                'y': data['y'],
                'degree': degree
            })
        
        if node_data:
            return gpd.GeoDataFrame(node_data, crs=self.network_gdf.crs)
        return gpd.GeoDataFrame(columns=['node_id', 'geometry', 'x', 'y', 'degree'])
    
    def find_near_miss_endpoints(self, max_distance: float = 2.0) -> dict:
        """
        Find pairs of endpoints that are close but not connected (potential gap issues).
        
        Args:
            max_distance: Maximum distance (in CRS units) to consider as near-miss.
            
        Returns:
            Dictionary with endpoints, near-miss pairs, and their connecting lines.
        """
        # Get all endpoints (degree 1 nodes)
        endpoints = []
        endpoint_node_ids = []
        
        for node_id in self.graph.nodes():
            if self.graph.degree(node_id) == 1:
                data = self.graph.nodes[node_id]
                endpoints.append((data['x'], data['y']))
                endpoint_node_ids.append(node_id)
        
        if len(endpoints) < 2:
            return {
                'endpoints': gpd.GeoDataFrame(columns=['node_id', 'geometry', 'x', 'y']),
                'near_miss_pairs': [],
                'missing_links': gpd.GeoDataFrame(columns=['geometry', 'distance', 'node1', 'node2'])
            }
        
        # Build KD-tree for efficient nearest neighbor search
        coords = np.array(endpoints)
        tree = cKDTree(coords)
        
        # Find pairs within max_distance
        pairs = tree.query_pairs(max_distance)
        
        near_miss_pairs = []
        missing_links = []
        
        for i, j in pairs:
            pt1 = endpoints[i]
            pt2 = endpoints[j]
            node1 = endpoint_node_ids[i]
            node2 = endpoint_node_ids[j]
            
            # Check if they're not already connected
            if not self.graph.has_edge(node1, node2):
                dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                near_miss_pairs.append({
                    'node1': node1,
                    'node2': node2,
                    'distance': dist
                })
                missing_links.append({
                    'geometry': LineString([pt1, pt2]),
                    'distance': dist,
                    'node1': node1,
                    'node2': node2
                })
        
        # Create endpoint GeoDataFrame
        endpoint_gdf = gpd.GeoDataFrame([
            {
                'node_id': endpoint_node_ids[i],
                'geometry': Point(endpoints[i]),
                'x': endpoints[i][0],
                'y': endpoints[i][1]
            }
            for i in range(len(endpoints))
        ], crs=self.network_gdf.crs)
        
        missing_links_gdf = gpd.GeoDataFrame(
            missing_links, crs=self.network_gdf.crs
        ) if missing_links else gpd.GeoDataFrame(columns=['geometry', 'distance', 'node1', 'node2'])
        
        return {
            'endpoints': endpoint_gdf,
            'near_miss_pairs': near_miss_pairs,
            'missing_links': missing_links_gdf
        }
    
    def compute_intersection_angles(self) -> gpd.GeoDataFrame:
        """
        Compute angles between incident edges at each intersection (degree >= 3).
        
        Returns:
            GeoDataFrame with intersection nodes and angle statistics.
        """
        intersection_data = []
        
        for node_id in self.graph.nodes():
            degree = self.graph.degree(node_id)
            if degree < 3:
                continue
            
            node_data = self.graph.nodes[node_id]
            node_pt = np.array([node_data['x'], node_data['y']])
            
            # Get directions of all incident edges
            directions = []
            edge_endpoints = []
            
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.edges[node_id, neighbor]
                geom = edge_data.get('geometry')
                
                if geom is None:
                    neighbor_data = self.graph.nodes[neighbor]
                    direction = np.array([neighbor_data['x'] - node_pt[0],
                                         neighbor_data['y'] - node_pt[1]])
                else:
                    # Get the direction from the actual edge geometry
                    coords = list(geom.coords)
                    # Find which end is the current node
                    start_pt = np.array(coords[0][:2])
                    end_pt = np.array(coords[-1][:2])
                    
                    if np.linalg.norm(start_pt - node_pt) < np.linalg.norm(end_pt - node_pt):
                        # Node is at start, direction is towards second point
                        direction = np.array(coords[1][:2]) - start_pt
                    else:
                        # Node is at end, direction is towards second-to-last point
                        direction = np.array(coords[-2][:2]) - end_pt
                
                norm = np.linalg.norm(direction)
                if norm > 0:
                    directions.append(direction / norm)
                    edge_endpoints.append(direction)
            
            # Calculate angles between all pairs of directions
            angles = []
            for i in range(len(directions)):
                for j in range(i + 1, len(directions)):
                    cos_angle = np.clip(np.dot(directions[i], directions[j]), -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    angles.append(angle_deg)
            
            if angles:
                # Check for near-right angles
                near_right_angles = sum(1 for a in angles if 80 <= a <= 100)
                acute_angles = sum(1 for a in angles if a < 30)
                obtuse_angles = sum(1 for a in angles if a > 150)
                
                # Score: higher is better (more right angles, fewer extreme angles)
                right_angle_score = near_right_angles / len(angles) if angles else 0
                
                intersection_data.append({
                    'node_id': node_id,
                    'geometry': Point(node_pt),
                    'x': node_pt[0],
                    'y': node_pt[1],
                    'degree': degree,
                    'min_angle': min(angles),
                    'max_angle': max(angles),
                    'mean_angle': np.mean(angles),
                    'near_right_angles': near_right_angles,
                    'acute_angles': acute_angles,
                    'obtuse_angles': obtuse_angles,
                    'right_angle_score': right_angle_score,
                    'edge_directions': edge_endpoints
                })
        
        if intersection_data:
            return gpd.GeoDataFrame(intersection_data, crs=self.network_gdf.crs)
        return gpd.GeoDataFrame(columns=['node_id', 'geometry', 'x', 'y', 'degree',
                                          'min_angle', 'max_angle', 'mean_angle',
                                          'near_right_angles', 'acute_angles', 'obtuse_angles',
                                          'right_angle_score', 'edge_directions'])
    
    def get_edges_by_length(self, short_threshold: float = 5.0, long_threshold: float = None) -> dict:
        """
        Categorize edges by length for visualization.
        
        Args:
            short_threshold: Edges shorter than this are considered "very short"
            long_threshold: Edges longer than this are considered "very long". 
                           If None, uses 95th percentile.
        """
        edge_data = []
        lengths = []
        
        for u, v, data in self.graph.edges(data=True):
            geom = data.get('geometry')
            if geom is None:
                continue
            length = data.get('length', geom.length)
            lengths.append(length)
            edge_data.append({
                'start_node': u,
                'end_node': v,
                'geometry': geom,
                'length': length
            })
        
        if not edge_data:
            return {
                'all_edges': gpd.GeoDataFrame(columns=['geometry', 'length', 'category']),
                'short_edges': gpd.GeoDataFrame(columns=['geometry', 'length']),
                'long_edges': gpd.GeoDataFrame(columns=['geometry', 'length'])
            }
        
        if long_threshold is None:
            long_threshold = np.percentile(lengths, 95)
        
        for ed in edge_data:
            if ed['length'] < short_threshold:
                ed['category'] = 'very_short'
            elif ed['length'] > long_threshold:
                ed['category'] = 'very_long'
            else:
                ed['category'] = 'normal'
        
        all_edges = gpd.GeoDataFrame(edge_data, crs=self.network_gdf.crs)
        short_edges = all_edges[all_edges['category'] == 'very_short'].copy()
        long_edges = all_edges[all_edges['category'] == 'very_long'].copy()
        
        return {
            'all_edges': all_edges,
            'short_edges': short_edges,
            'long_edges': long_edges,
            'short_threshold': short_threshold,
            'long_threshold': long_threshold
        }
    
    def find_small_loops(self, max_perimeter: float = 50.0) -> gpd.GeoDataFrame:
        """
        Find small loops (cycles) below a perimeter threshold.
        
        Args:
            max_perimeter: Maximum perimeter to consider as "small loop"
        """
        loops_data = []
        
        try:
            # Find all simple cycles
            cycles = list(nx.simple_cycles(self.graph.to_directed()))
            
            # Filter to unique undirected cycles (each cycle appears twice in directed version)
            seen_cycles = set()
            unique_cycles = []
            
            for cycle in cycles:
                if len(cycle) < 3:  # Need at least 3 nodes for a loop
                    continue
                # Normalize cycle representation
                cycle_key = tuple(sorted(cycle))
                if cycle_key not in seen_cycles:
                    seen_cycles.add(cycle_key)
                    unique_cycles.append(cycle)
            
            for cycle in unique_cycles[:1000]:  # Limit to prevent very long computation
                # Calculate perimeter
                perimeter = 0
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if self.graph.has_edge(u, v):
                        edge_data = self.graph.edges[u, v]
                        perimeter += edge_data.get('length', 0)
                
                if perimeter > 0 and perimeter < max_perimeter:
                    # Create polygon from cycle nodes
                    cycle_coords = []
                    for node_id in cycle:
                        node_data = self.graph.nodes[node_id]
                        cycle_coords.append((node_data['x'], node_data['y']))
                    cycle_coords.append(cycle_coords[0])  # Close the loop
                    
                    if len(cycle_coords) >= 4:
                        loops_data.append({
                            'geometry': Polygon(cycle_coords),
                            'perimeter': perimeter,
                            'num_nodes': len(cycle),
                            'node_ids': cycle
                        })
        except Exception as e:
            # Cycle finding can fail on certain graph structures
            pass
        
        if loops_data:
            return gpd.GeoDataFrame(loops_data, crs=self.network_gdf.crs)
        return gpd.GeoDataFrame(columns=['geometry', 'perimeter', 'num_nodes', 'node_ids'])
    
    def find_short_spurs(self, max_length: float = 10.0) -> gpd.GeoDataFrame:
        """
        Find short dead-end spurs (edges connected to degree-1 nodes).
        
        Args:
            max_length: Maximum length to consider as "short spur"
        """
        spurs_data = []
        
        for node_id in self.graph.nodes():
            if self.graph.degree(node_id) != 1:
                continue
            
            # Get the single connected edge
            neighbors = list(self.graph.neighbors(node_id))
            if not neighbors:
                continue
            
            neighbor = neighbors[0]
            edge_data = self.graph.edges[node_id, neighbor]
            length = edge_data.get('length', 0)
            
            if length < max_length:
                geom = edge_data.get('geometry')
                if geom is not None:
                    node_data = self.graph.nodes[node_id]
                    spurs_data.append({
                        'geometry': geom,
                        'length': length,
                        'dead_end_node': node_id,
                        'dead_end_x': node_data['x'],
                        'dead_end_y': node_data['y']
                    })
        
        if spurs_data:
            return gpd.GeoDataFrame(spurs_data, crs=self.network_gdf.crs)
        return gpd.GeoDataFrame(columns=['geometry', 'length', 'dead_end_node', 'dead_end_x', 'dead_end_y'])
    
    def get_quality_score(self) -> dict:
        """
        Calculate overall network quality score (0-100).
        Higher is better.
        
        Score formulas:
        - Connectivity: (nodes_in_largest_component / total_nodes) * 100
        - Dead-End: max(0, (1 - dead_end_ratio * 2)) * 100
        - Intersection: min(100, intersection_ratio * 200)
        - Overall: Connectivity * 0.40 + Dead-End * 0.30 + Intersection * 0.30
        """
        scores = {}
        
        # Connectivity score (based on connected components ratio)
        connectivity = self.compute_connectivity_metrics()
        scores['connectivity'] = connectivity['largest_component_ratio'] * 100
        
        # Dead-end score (fewer dead ends = better)
        dead_ends = self.get_dead_ends()
        n_nodes = self.graph.number_of_nodes()
        dead_end_ratio = len(dead_ends) / n_nodes if n_nodes > 0 else 1
        scores['dead_end'] = max(0, (1 - dead_end_ratio * 2)) * 100  # Penalize dead ends
        
        # Intersection quality (good networks have proper intersections)
        intersections = self.get_intersection_nodes(min_degree=3)
        intersection_ratio = len(intersections) / n_nodes if n_nodes > 0 else 0
        scores['intersection'] = min(100, intersection_ratio * 200)  # Target ~50% intersection ratio
        
        # Overall score (without geometry)
        weights = {'connectivity': 0.40, 'dead_end': 0.30, 'intersection': 0.30}
        scores['overall'] = sum(scores[k] * weights[k] for k in weights)
        
        return scores


def load_network_from_shapefile(shapefile_path: str) -> gpd.GeoDataFrame:
    """Load network from shapefile."""
    return gpd.read_file(shapefile_path)


def compare_networks(network1: gpd.GeoDataFrame, 
                     network2: gpd.GeoDataFrame, 
                     buffer_distance: float = 10) -> dict:
    """
    Compare two networks and find differences.
    
    Args:
        network1: First network (e.g., AI-generated)
        network2: Second network (e.g., OSM reference)
        buffer_distance: Buffer distance for spatial matching (in CRS units)
    
    Returns:
        Dictionary with comparison results
    """
    # Ensure same CRS
    if network1.crs != network2.crs:
        network2 = network2.to_crs(network1.crs)
    
    # Buffer network2 for spatial join
    network2_buffered = network2.copy()
    network2_buffered['geometry'] = network2_buffered.geometry.buffer(buffer_distance)
    
    # Find matching and non-matching edges
    joined = gpd.sjoin(network1, network2_buffered, how='left', predicate='intersects')
    
    matched_idx = joined[~joined['index_right'].isna()].index.unique()
    unmatched_idx = joined[joined['index_right'].isna()].index.unique()
    
    # Edges in network1 not in network2
    only_in_network1 = network1.loc[unmatched_idx].copy()
    
    # Edges matching
    matched = network1.loc[matched_idx].copy()
    
    # Similarly for network2
    network1_buffered = network1.copy()
    network1_buffered['geometry'] = network1_buffered.geometry.buffer(buffer_distance)
    
    joined2 = gpd.sjoin(network2, network1_buffered, how='left', predicate='intersects')
    unmatched_idx2 = joined2[joined2['index_right'].isna()].index.unique()
    only_in_network2 = network2.loc[unmatched_idx2].copy()
    
    return {
        'matched': matched,
        'only_in_network1': only_in_network1,
        'only_in_network2': only_in_network2,
        'match_ratio_network1': len(matched) / len(network1) if len(network1) > 0 else 0,
        'match_ratio_network2': (len(network2) - len(only_in_network2)) / len(network2) if len(network2) > 0 else 0
    }
