#!/usr/bin/env python3
"""
Compute cooperation factors for all maps and save to lookup file.
Contains the expensive max-flow calculation logic.

Usage:
nohup python compute_cooperation_factor.py > compute_cooperation_factor.log 2>&1 &
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from collections import deque


def calculate_cooperation_factor_expensive(map_nr, maps_directory):
    """
    Calculate cooperation factor using expensive max-flow analysis.
    
    Args:
        map_nr: Map identifier
        maps_directory: Path to maps directory
        
    Returns:
        float: Cooperation factor (0.2-2.0)
    """
    # Load map structure
    map_txt_path = os.path.join(maps_directory, f'{map_nr}.txt')
    if not os.path.exists(map_txt_path):
        print(f"Warning: Map file {map_txt_path} not found. Using default cooperation factor 1.0")
        return 1.0
    
    with open(map_txt_path, 'r') as f:
        map_lines = [line.rstrip('\n') for line in f.readlines()]
    
    rows, cols = len(map_lines), len(map_lines[0]) if map_lines else 0
    
    # Create binary accessibility matrix (1=walkable, 0=blocked)
    accessibility = np.zeros((rows, cols), dtype=int)
    
    # Identify key locations
    agents = []
    resources = []  # Dispensers (T, P, C, X)
    work_stations = []  # Cutting boards (B), Counters (M)
    delivery_points = []  # Delivery (D)
    
    for i, line in enumerate(map_lines):
        for j, char in enumerate(line):
            if char in [' ', '1', '2', 'T', 'P', 'C', 'X', 'B', 'M', 'D']:
                accessibility[i, j] = 1
                
                if char in ['1', '2']:
                    agents.append((i, j))
                elif char in ['T', 'P', 'C', 'X']:
                    resources.append((i, j))
                elif char in ['B', 'M']:
                    work_stations.append((i, j))
                elif char == 'D':
                    delivery_points.append((i, j))
    
    # BFS helper function for pathfinding
    def bfs_shortest_path(start, end):
        queue = deque([(start[0], start[1], 0)])
        visited = set()
        visited.add(start)
        
        while queue:
            x, y, dist = queue.popleft()
            if (x, y) == end:
                return dist
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < rows and 0 <= ny < cols and 
                    accessibility[nx, ny] == 1 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
        
        return float('inf')  # No path found
    
    # Calculate cooperation metrics
    
    # 1. SPATIAL CONSTRAINT METRIC (0.0-1.0)
    total_cells = rows * cols
    walkable_cells = np.sum(accessibility)
    spatial_constraint = 1.0 - (walkable_cells / total_cells) if total_cells > 0 else 0.0
    
    # 2. BOTTLENECK METRIC (0.0-1.0)
    def find_bottlenecks():
        bottlenecks = 0
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if accessibility[i, j] == 1:  # Walkable cell
                    # Count walkable neighbors
                    neighbors = [
                        accessibility[i-1, j], accessibility[i+1, j],
                        accessibility[i, j-1], accessibility[i, j+1]
                    ]
                    walkable_neighbors = sum(neighbors)
                    
                    # Bottleneck: walkable cell with ≤2 walkable orthogonal neighbors
                    if walkable_neighbors <= 2:
                        bottlenecks += 1
        
        return bottlenecks / walkable_cells if walkable_cells > 0 else 0.0
    
    bottleneck_density = find_bottlenecks()
    
    # 3. PATH DIVERSITY METRIC (0.0-1.0) - Max-flow version
    def calculate_path_diversity():
        """Calculate path diversity using max-flow to find independent routes."""
        if len(agents) < 2:
            return 0.5  # Default for incomplete maps
        
        def build_graph_with_node_splitting():
            """Build graph with node splitting for node-disjoint paths."""
            # Create mapping from (row, col) to node indices
            node_map = {}
            node_count = 0
            
            # Each walkable cell gets two nodes: in-node and out-node
            for i in range(rows):
                for j in range(cols):
                    if accessibility[i, j] == 1:
                        node_map[(i, j)] = (node_count, node_count + 1)  # (in_node, out_node)
                        node_count += 2
            
            # Build adjacency list with capacities
            graph = {}
            for node in range(node_count):
                graph[node] = {}
            
            # Add edges
            for i in range(rows):
                for j in range(cols):
                    if accessibility[i, j] == 1:
                        in_node, out_node = node_map[(i, j)]
                        
                        # Internal edge from in_node to out_node (capacity 1 for node splitting)
                        graph[in_node][out_node] = 1
                        
                        # Edges to neighbors
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < rows and 0 <= nj < cols and 
                                accessibility[ni, nj] == 1):
                                neighbor_in, neighbor_out = node_map[(ni, nj)]
                                # Connect this cell's out_node to neighbor's in_node
                                graph[out_node][neighbor_in] = 1
            
            return graph, node_map
        
        def edmonds_karp_max_flow(graph, source, sink):
            """Edmonds-Karp algorithm for maximum flow."""
            def bfs_find_path():
                visited = set([source])
                queue = deque([(source, [source])])
                
                while queue:
                    node, path = queue.popleft()
                    
                    for neighbor in graph[node]:
                        if neighbor not in visited and graph[node][neighbor] > 0:
                            new_path = path + [neighbor]
                            
                            if neighbor == sink:
                                return new_path
                            
                            visited.add(neighbor)
                            queue.append((neighbor, new_path))
                
                return None
            
            max_flow_value = 0
            
            # Keep finding augmenting paths until none exist
            while True:
                path = bfs_find_path()
                if not path:
                    break
                
                # Find minimum capacity along the path
                flow = float('inf')
                for i in range(len(path) - 1):
                    flow = min(flow, graph[path[i]][path[i + 1]])
                
                # Update residual capacities
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    graph[u][v] -= flow
                    
                    # Add reverse edge
                    if v not in graph:
                        graph[v] = {}
                    if u not in graph[v]:
                        graph[v][u] = 0
                    graph[v][u] += flow
                
                max_flow_value += flow
            
            return max_flow_value
        
        def calculate_route_independence():
            """Calculate average number of independent routes between key locations."""
            if len(agents) < 2:
                return 1.0  # Single agent can't have route conflicts
            
            graph, node_map = build_graph_with_node_splitting()
            
            route_diversities = []
            
            # Test independence between pairs of agents
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    agent1_pos = agents[i]
                    agent2_pos = agents[j]
                    
                    if agent1_pos in node_map and agent2_pos in node_map:
                        # Source is agent1's out_node, sink is agent2's in_node
                        source = node_map[agent1_pos][1]  # out_node
                        sink = node_map[agent2_pos][0]    # in_node
                        
                        # Create a fresh copy of the graph for this flow calculation
                        graph_copy = {}
                        for node in graph:
                            graph_copy[node] = graph[node].copy()
                        
                        max_flow = edmonds_karp_max_flow(graph_copy, source, sink)
                        route_diversities.append(max_flow)
            
            # Also test agent to key resource independence
            key_locations = resources + work_stations + delivery_points
            for agent in agents:
                for location in key_locations[:3]:  # Limit to avoid too many calculations
                    if agent in node_map and location in node_map:
                        source = node_map[agent][1]      # agent out_node
                        sink = node_map[location][0]     # location in_node
                        
                        graph_copy = {}
                        for node in graph:
                            graph_copy[node] = graph[node].copy()
                        
                        max_flow = edmonds_karp_max_flow(graph_copy, source, sink)
                        route_diversities.append(max_flow)
            
            if not route_diversities:
                return 1.0
            
            # Average number of independent paths
            avg_independence = np.mean(route_diversities)
            
            # Normalize: 1 path = high constraint (1.0), 3+ paths = low constraint (0.0)
            normalized_constraint = max(0.0, min(1.0, (3.0 - avg_independence) / 2.0))
            
            return normalized_constraint
        
        return calculate_route_independence()
    
    path_constraint = calculate_path_diversity()
    
    # 4. WORKSPACE OVERLAP METRIC (0.0-1.0)
    def calculate_workspace_overlap():
        if len(agents) < 2:
            return 0.0
        
        max_distance = max(rows, cols) // 2
        
        agent_workspaces = []
        for agent in agents:
            workspace = set()
            queue = deque([(agent[0], agent[1], 0)])
            visited = set([agent])
            
            while queue:
                x, y, dist = queue.popleft()
                if dist <= max_distance:
                    workspace.add((x, y))
                    
                    if dist < max_distance:
                        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < rows and 0 <= ny < cols and 
                                accessibility[nx, ny] == 1 and (nx, ny) not in visited):
                                visited.add((nx, ny))
                                queue.append((nx, ny, dist + 1))
            
            agent_workspaces.append(workspace)
        
        # Calculate overlap between agent workspaces
        if len(agent_workspaces) >= 2:
            overlap = len(agent_workspaces[0].intersection(agent_workspaces[1]))
            total_workspace = len(agent_workspaces[0].union(agent_workspaces[1]))
            return overlap / total_workspace if total_workspace > 0 else 0.0
        
        return 0.0
    
    workspace_overlap = calculate_workspace_overlap()
    
    # 5. CRITICAL PATH DEPENDENCY (0.0-1.0)
    def calculate_critical_dependency():
        if not delivery_points or len(agents) < 2:
            return 0.0
        
        # Check shared resource dependencies
        shared_resources = 0
        total_resources = len(resources) + len(work_stations)
        
        if total_resources > 0:
            for resource in resources + work_stations:
                accessible_agents = 0
                for agent in agents:
                    if bfs_shortest_path(agent, resource) != float('inf'):
                        accessible_agents += 1
                
                if accessible_agents > 1:
                    shared_resources += 1
            
            return shared_resources / total_resources
        
        return 0.0
    
    critical_dependency = calculate_critical_dependency()
    
    # WEIGHTED COMBINATION OF METRICS
    weights = {
        'spatial_constraint': 0.02,      # Basic layout constraint
        'bottleneck_density': 2.0,      # Physical bottlenecks
        'path_constraint': 0.8,         # Path diversity/flexibility
        'workspace_overlap': 0.07,       # Operational area conflicts  
        'critical_dependency': 0.03      # Infrastructure sharing
    }
    
    cooperation_score = (
        weights['spatial_constraint'] * spatial_constraint +
        weights['bottleneck_density'] * bottleneck_density +
        weights['path_constraint'] * path_constraint +
        weights['workspace_overlap'] * workspace_overlap +
        weights['critical_dependency'] * critical_dependency
    )
    
    # Transform to cooperation factor range [0.2, 2.0]
    baseline_factor = 0.1  # Minimum cooperation (very open maps)
    max_factor = 2.0      # Maximum cooperation (very constrained maps)
    
    cooperation_factor = baseline_factor + (max_factor - baseline_factor) * cooperation_score

    # Add map-specific adjustments based on naming patterns
    if "baseline" in map_nr:
        cooperation_factor *= 0.2  # Baseline maps are designed to be more open
    elif "semiencouraged" in map_nr:
        cooperation_factor *= 0.7  # Semi-encouraged maps have some cooperation elements
    elif "encouraged" in map_nr:
        cooperation_factor *= 1.0  # Forced cooperation maps
    elif "corridor" in map_nr:
        cooperation_factor *= 1.3  # Corridor maps have bottlenecks
    elif "extreme" in map_nr:
        cooperation_factor *= 1.8  # Extreme maps

    # Clamp to valid range
    cooperation_factor = max(0.2, min(2.0, cooperation_factor))
    
    # Debug information
    print(f"Cooperation factor calculation for {map_nr}:")
    print(f"  Spatial constraint: {spatial_constraint:.3f}")
    print(f"  Bottleneck density: {bottleneck_density:.3f}")
    print(f"  Path constraint: {path_constraint:.3f}")
    print(f"  Workspace overlap: {workspace_overlap:.3f}")
    print(f"  Critical dependency: {critical_dependency:.3f}")
    print(f"  Final cooperation factor: {cooperation_factor:.3f}")
    
    return cooperation_factor


def compute_all_cooperation_factors():
    """Compute cooperation factors for all map files in the maps directory."""
    
    maps_dir = Path(__file__).parent
    output_file = maps_dir / 'cooperation_factors.json'
    
    print("Scanning for map files...")
    map_files = glob.glob(str(maps_dir / "*.txt"))
    map_files = [f for f in map_files if not f.endswith('_info.txt')]  # Exclude info files
    
    cooperation_factors = {}
    
    print(f"\nFound {len(map_files)} map files. Computing cooperation factors...")
    print("=" * 60)
    
    for map_file in sorted(map_files):
        map_name = Path(map_file).stem  # Get filename without .txt extension
        
        try:
            print(f"\nProcessing: {map_name}")
            
            # Calculate cooperation factor using the expensive method
            cooperation_factor = calculate_cooperation_factor_expensive(map_name, str(maps_dir))
            
            cooperation_factors[map_name] = cooperation_factor
            print(f"✓ {map_name}: {cooperation_factor:.3f}")
            
        except Exception as e:
            print(f"✗ Error processing {map_name}: {e}")
            cooperation_factors[map_name] = 1.0  # Default fallback
    
    # Save to JSON file
    print(f"\nSaving cooperation factors to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cooperation_factors, f, indent=2, sort_keys=True)
    
    print(f"✓ Successfully saved {len(cooperation_factors)} cooperation factors")
    print(f"✓ File saved to: {output_file}")
    
    # Print summary statistics
    values = list(cooperation_factors.values())
    print(f"\nSummary Statistics:")
    print(f"  Min cooperation factor: {min(values):.3f}")
    print(f"  Max cooperation factor: {max(values):.3f}")
    print(f"  Mean cooperation factor: {sum(values)/len(values):.3f}")
    
    return cooperation_factors


if __name__ == "__main__":
    compute_all_cooperation_factors()