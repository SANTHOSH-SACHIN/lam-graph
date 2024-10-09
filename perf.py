import time
import psutil
import os
import pandas as pd
import numpy as np
import json
import networkx as nx

node_dtype = np.dtype([
    ('id', np.int32),
    ('type', 'S20'),
    ('name', 'S50'),
    ('revenue', np.float32),
    ('market_share', np.float32),
    ('price', np.float32),
    ('lead_time', np.int16),
    ('production_cost', np.float32),
    ('importance', np.int8),
    ('level', np.int8),
    ('quantity', np.int32),
    ('supplier', 'S20')
])

edge_dtype = np.dtype([
    ('source_id', np.int32),
    ('target_id', np.int32),
    ('quantity', np.int32)
])

def generate_nodes(node_types, total_nodes, attr_ranges, suppliers):
    node_counts = {}
    remaining_nodes = total_nodes

    for node_type in node_types:
        if 'count' in node_type:
            node_counts[node_type['name']] = node_type['count']
            remaining_nodes -= node_type['count']
        elif 'count_percentage' in node_type:
            count = int(total_nodes * node_type['count_percentage'])
            node_counts[node_type['name']] = count
            remaining_nodes -= count

    # Ensure remaining nodes are assigned to "Make/Purchase Part"
    if 'Make/Purchase Part' not in node_counts:
        node_counts['Make/Purchase Part'] = remaining_nodes

    nodes = np.zeros(total_nodes, dtype=node_dtype)
    nodes['id'] = np.arange(total_nodes)
    nodes['name'] = np.array([f'Node_{i}' for i in range(total_nodes)]).astype('S50')
    nodes['supplier'] = np.random.choice(suppliers, total_nodes).astype('S20')

    current_index = 0
    for node_type in node_types:
        count = node_counts[node_type['name']]
        end_index = current_index + count
        nodes['type'][current_index:end_index] = node_type['name']

        # Handle level assignment specifically for Modules
        if node_type['name'] == "Module":
            nodes['level'][current_index:end_index] = np.random.randint(3, 4, count)  # Example level assignment
        elif 'level' in node_type:
            nodes['level'][current_index:end_index] = node_type['level']
        elif 'level_range' in node_type:
            nodes['level'][current_index:end_index] = np.random.randint(
                node_type['level_range'][0], node_type['level_range'][1] + 1, count
            )

        current_index = end_index

    # Vectorized assignment for attributes
    for attr, (min_val, max_val) in attr_ranges.items():
        if attr in ['importance', 'lead_time', 'quantity']:
            nodes[attr] = np.random.randint(min_val, max_val + 1, total_nodes)
        else:
            nodes[attr] = np.random.uniform(min_val, max_val, total_nodes)

    return nodes

def generate_edges(node_types, nodes, total_edges, max_edge_weight=100):
    edges = np.zeros(total_edges, dtype=edge_dtype)

    # Create a mapping of node types to their IDs for easy access
    type_to_ids = {node_type['name']: [] for node_type in node_types}

    for node in nodes:
        type_to_ids[node['type'].decode('utf-8')].append(node['id'])

    edge_count = 0
    for node_type in node_types:
        current_ids = type_to_ids[node_type['name']]
        for child_name in node_type.get('children', []):
            child_ids = type_to_ids.get(child_name, [])
            if not child_ids:
                continue
            for parent_id in current_ids:
                target_id = np.random.choice(child_ids)
                edges[edge_count] = (parent_id, target_id, np.random.randint(1, max_edge_weight))
                edge_count += 1
                if edge_count >= total_edges:
                    return edges[:edge_count]  # Return early if we've generated enough edges

    # If we need more edges, create random edges between existing nodes
    while edge_count < total_edges:
        source_id = np.random.choice(nodes['id'])
        target_id = np.random.choice(nodes['id'])
        if source_id != target_id:  # Avoid self-loops
            edges[edge_count] = (source_id, target_id, np.random.randint(1, max_edge_weight))
            edge_count += 1

    return edges[:edge_count]
def generate_edges(node_types, nodes, total_edges):
    edges = np.zeros(total_edges, dtype=edge_dtype)

    # Create a mapping of node types to their IDs for easy access
    type_to_ids = {node_type['name']: [] for node_type in node_types}
    
    for node in nodes:
        type_to_ids[node['type'].decode('utf-8')].append(node['id'])
    
    edge_count = 0
    
    # Generate structured edges based on parent-child relationships
    for node_type in node_types:
        current_ids = type_to_ids[node_type['name']]
        
        for child_name in node_type.get('children', []):
            child_ids = type_to_ids.get(child_name, [])
            if not child_ids:
                continue
            
            num_edges_per_parent = min(len(current_ids), len(child_ids))
            parent_indices = np.random.choice(current_ids, size=num_edges_per_parent, replace=True)
            child_indices = np.random.choice(child_ids, size=num_edges_per_parent, replace=True)

            edges[edge_count:edge_count + num_edges_per_parent]['source_id'] = parent_indices
            edges[edge_count:edge_count + num_edges_per_parent]['target_id'] = child_indices
            edges[edge_count:edge_count + num_edges_per_parent]['quantity'] = np.random.randint(1, 100, num_edges_per_parent)

            edge_count += num_edges_per_parent
            
            if edge_count >= total_edges:
                return edges[:edge_count]

    # If we need more edges, create random edges between existing nodes
    while edge_count < total_edges:
        source_id = np.random.choice(nodes['id'])
        target_id = np.random.choice(nodes['id'])
        if source_id != target_id:  # Avoid self-loops
            edges[edge_count] = (source_id, target_id, np.random.randint(1, 100))
            edge_count += 1

    return edges[:edge_count]

def generate_graph(config):
    total_nodes = config['total_nodes']
    total_edges = config['total_edges']
    
    # Generate nodes and edges without threading for simplicity; you can add it back later if needed.
    nodes = generate_nodes(config['node_types'], total_nodes, config['attribute_ranges'], config['suppliers'])
    
    edges = generate_edges(config['node_types'], nodes, total_edges)

    print("Number of nodes: ", len(nodes))
    print("Number of edges: ", len(edges))
    
    return nodes, edges


def save_to_mmap(data, filename):
    mmap_file = np.memmap(filename, dtype=data.dtype, mode='w+', shape=data.shape)
    mmap_file[:] = data[:]
    mmap_file.flush()

def load_mmap_data(filename, dtype):
    return np.memmap(filename, dtype=dtype, mode='r')

# Function to create the graph
G = nx.DiGraph()
def create_graph(nodes, edges, max_nodes=50000000):
    node_sample = np.random.choice(range(len(nodes)), min(max_nodes, len(nodes)), replace=False)
    for i in node_sample:
        node = nodes[i]
        G.add_node(node['id'], **{k: node[k].decode('utf-8') if isinstance(node[k], bytes) else node[k] for k in node.dtype.names})

    for edge in edges:
        if edge['source_id'] in G.nodes and edge['target_id'] in G.nodes:
            G.add_edge(edge['source_id'], edge['target_id'], quantity=edge['quantity'])
    print("number of nodes: ", len(G.nodes))
    print("number of edges: ", len(G.edges))
    return G

import matplotlib.pyplot as plt

# Define the total_nodes and total_edges combinations
total_nodes = [1000, 10000, 100000, 1000000]
total_edges = [2500, 25000, 250000, 2500000]

# Initialize lists to store the time and memory readings
time_readings = []
memory_readings = []

# Loop through each combination of total_nodes and total_edges
for nodes, edges in zip(total_nodes, total_edges):
    config = {
  "total_nodes": nodes,
  "total_edges": edges,
  "node_types": [
    {
      "name": "Business Group",
      "level": 0,
      "count": 1,
      "children": ["Product Family"]
    },
    {
      "name": "Product Family",
      "level": 1,
      "count": 4,
      "children": ["Product Offering"]
    },
    {
      "name": "Product Offering",
      "level": 2,
      "count": 10,
      "children": ["Module", "Make/Purchase Part"]
    },
    {
      "name": "Module",
      "level": 3,
      "count_percentage": 0.4,
      "children": ["Make/Purchase Part"]
    },
    {
      "name": "Make/Purchase Part",
      "level_range": [4, 10],
      "children": []
    }
  ],
  "suppliers": ["Supplier A", "Supplier B", "Supplier C", "Supplier D", "Supplier E"],
  "attribute_ranges": {
    "revenue": [100000, 1000000],
    "market_share": [0, 0.5],
    "price": [10, 1000],
    "lead_time": [1, 30],
    "production_cost": [1, 500],
    "importance": [1, 10],
    "quantity": [0, 1000]
  }
}


    # Generate nodes and edges
    
for i,j in zip(total_nodes, total_edges):
    # Measure time and memory before graph construction
        config = {
  "total_nodes": i,
  "total_edges": j,
  "node_types": [
    {
      "name": "Business Group",
      "level": 0,
      "count": 1,
      "children": ["Product Family"]
    },
    {
      "name": "Product Family",
      "level": 1,
      "count": 4,
      "children": ["Product Offering"]
    },
    {
      "name": "Product Offering",
      "level": 2,
      "count": 10,
      "children": ["Module", "Make/Purchase Part"]
    },
    {
      "name": "Module",
      "level": 3,
      "count_percentage": 0.4,
      "children": ["Make/Purchase Part"]
    },
    {
      "name": "Make/Purchase Part",
      "level_range": [4, 10],
      "children": []
    }
  ],
  "suppliers": ["Supplier A", "Supplier B", "Supplier C", "Supplier D", "Supplier E"],
  "attribute_ranges": {
    "revenue": [100000, 1000000],
    "market_share": [0, 0.5],
    "price": [10, 1000],
    "lead_time": [1, 30],
    "production_cost": [1, 500],
    "importance": [1, 10],
    "quantity": [0, 1000]
  }
        }       
        nodes, edges = generate_graph(config)

        G = create_graph(nodes, edges)
        # nodes, edges = generate_graph(config)
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss 
        
        degree_centrality = nx.degree_centrality(G)
        
        # Betweenness Centrality
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Closeness Centrality
        closeness_centrality = nx.closeness_centrality(G)

        # Measure time and memory after graph construction
        end_time = time.time()
        mem_after = process.memory_info().rss

        # Calculate time and memory readings
        time_reading = end_time - start_time
        memory_reading = (mem_after - mem_before) / (1024 ** 2)

        # Append the time and memory readings to the lists
        time_readings.append(time_reading)
        memory_readings.append(memory_reading)
import matplotlib.pyplot as plt

# Create a separate plot for Node Size vs. Execution Time
plt.figure(figsize=(8, 6))
plt.plot(total_nodes, time_readings, marker='o', label='Execution Time')
plt.xlabel('Total Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('Node Size vs. Execution Time')
plt.grid(True)
plt.legend()
plt.show()

# Create a separate plot for Node Size vs. Memory Usage
plt.figure(figsize=(8, 6))
plt.plot(total_nodes, memory_readings, marker='o', label='Memory Usage', color='orange')
plt.xlabel('Total Nodes')
plt.ylabel('Memory Usage (MB)')
plt.title('Node Size vs. Memory Usage')
plt.grid(True)
plt.legend()
plt.show()
