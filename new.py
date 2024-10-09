import streamlit as st
import numpy as np
import json
from datetime import datetime
import time
import psutil
import os
import networkx as nx
import plotly.graph_objects as go
import concurrent.futures
import multiprocessing


st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                width: 70%;
                margin-left: 15%;
            }
            </style>
            """, unsafe_allow_html=True)

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

# Function to get node colors based on node type
def get_node_colors(G):
    color_map = {
        'Business Group': '#FF4136',
        'Product Family': '#FF851B',
        'Product Offering': '#FFDC00',
        'Module': '#2ECC40',
        'Make/Purchase Part': '#0074D9'
    }
    return [color_map[G.nodes[node]['type']] for node in G.nodes()]

# Function to visualize the graph
def visualize_graph(G):
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=get_node_colors(G),
            size=10,
            line_width=2))

    node_text = []
    for node in G.nodes():
        node_info = G.nodes[node]
        text = f"ID: {node}<br>"
        text += "<br>".join([f"{k}: {v}" for k, v in node_info.items()])
        node_text.append(text)

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Knowledge Graph Visualization',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig

st.title("Hierarchical Graph Generator")

with st.expander("Graph Files Generation", expanded=False):
    uploaded_file = st.file_uploader("### Upload Configuration JSON", type="json")

    if uploaded_file is not None:
        config_data = uploaded_file.read()

    try:
        config = json.loads(config_data)

        if st.button("Generate Graph"):
            start_time = datetime.now()
            
            nodes, edges = generate_graph(config)
            end_time = datetime.now()

            save_to_mmap(nodes, 'nodes.mmap')
            save_to_mmap(edges, 'edges.mmap')

            memory_used_nodes = nodes.nbytes 
            memory_used_edges = edges.nbytes 

            unique_types, counts = np.unique(nodes['type'], return_counts=True)
            type_count_dict = dict(zip(unique_types.astype(str), counts))

            st.success("Graph generated and saved successfully!")
            st.write(f"Total Nodes: {len(nodes)}")
            st.write(f"Total Edges: {len(edges)}")
            st.write(f"Graph generated in {(end_time - start_time).total_seconds()} seconds")
            st.write(f"Memory Used (Nodes): {memory_used_nodes/(1024**2):.2f} MB")
            st.write(f"Memory Used (Edges): {memory_used_edges/(1024**2):.2f} MB")
            
            st.subheader("Node Count by Type")
            for node_type, count in type_count_dict.items():
                st.write(f"{node_type}: {count}")

    except Exception as e:
        st.error(f"Error loading configuration: {e}")

with st.expander("Graph Visualization", expanded=False):
    st.write("### Visualize the generated graph with interactive Plotly display.")

    if st.button("Load and Visualize Graph"):
        # Load nodes and edges from memory-mapped files
        st.write("Loading data from memory-mapped files...")
        nodes = load_mmap_data('nodes.mmap', node_dtype)
        edges = load_mmap_data('edges.mmap', edge_dtype)

        # Measure memory before graph construction
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss 

        st.write("Creating graph...")
        G = create_graph(nodes, edges)

        # Measure time and memory after graph construction
        end_time = time.time()
        mem_after = process.memory_info().rss

        st.write(f"Graph construction completed in {end_time - start_time:.2f} seconds")
        st.write(f"Memory used for graph construction: {(mem_after - mem_before)/(1024**2) :.2f} MB")

        # Measure time and memory for visualization
        st.write("Visualizing graph...")
        start_viz_time = time.time()
        mem_before_viz = process.memory_info().rss

        fig = visualize_graph(G)

        end_viz_time = time.time()
        mem_after_viz = process.memory_info().rss
        st.write(f"Graph visualization completed in {end_viz_time - start_viz_time:.2f} seconds")
        st.write(f"Memory used for graph visualization: {(mem_after_viz - mem_before_viz) :.2f} MB")

        # Display the Plotly graph in the Streamlit app
        st.plotly_chart(fig)
        
with st.expander("Graph Queries", expanded=False):
    nodes = load_mmap_data('nodes.mmap', node_dtype)
    edges = load_mmap_data('edges.mmap', edge_dtype)

    
    tabs = st.tabs([
        "Find Shortest Path", 
        "Check Path Existence", 
        "Node Distribution at Level", 
        "Degree Distribution by Level", 
        "Supplier Distribution", 
        "Subgraph Extraction and Visualization"
    ])

# Tab for Finding Shortest Path
    with tabs[0]:
        source_id = st.number_input("Enter source node ID", min_value=0)
        target_id = st.number_input("Enter target node ID", min_value=0)

        if st.button("Find Shortest Path"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 
                G = create_graph(nodes, edges)
                if nx.has_path(G, source_id, target_id):
                    shortest_path = nx.shortest_path(G, source=source_id, target=target_id)
                    result_text = f"Shortest path between {source_id} and {target_id}: {shortest_path}"

                    # Visualize the shortest path
                    path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
                    
                    # Create a subgraph for the path
                    path_subgraph = G.edge_subgraph(path_edges).copy()

                    end_time = time.time()
                    mem_after = process.memory_info().rss 

                    st.write(result_text)
                    st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                    st.write(f"Memory taken: {(mem_after - mem_before)/(1024**2) :.4f} MB")

                    # Visualize the shortest path subgraph
                    fig_path_subgraph = visualize_graph(path_subgraph)
                    st.plotly_chart(fig_path_subgraph)

                else:
                    result_text = f"No path exists between {source_id} and {target_id}"
                    end_time = time.time()
                    mem_after = process.memory_info().rss 

                    st.write(result_text)
                    st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                    st.write(f"Memory taken: {(mem_after - mem_before)/(1024**2) :.4f} MB")

            except Exception as e:
                st.error(f"Error during shortest path calculation: {e}")

    # Tab for Checking Path Existence
    with tabs[1]:
        source_id = st.number_input("Enter source node ID for path check", min_value=0, key="source_check")
        target_id = st.number_input("Enter target node ID for path check", min_value=0, key="target_check")

        if st.button("Check Path Existence"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 
                G = create_graph(nodes, edges)
                path_exists = nx.has_path(G, source=source_id, target=target_id)
                result_text = f"Path exists between {source_id} and {target_id}: {path_exists}"

                end_time = time.time()
                mem_after = process.memory_info().rss 

                st.write(result_text)
                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {(mem_after - mem_before)/(1024**2) :.4f} MB")

            except Exception as e:
                st.error(f"Error during path existence check: {e}")

    # Tab for Node Distribution at Level
    with tabs[2]:
        level = st.number_input("Enter level to check node distribution", min_value=0, max_value=11)

        if st.button("Node Distribution at Level"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 
                G = create_graph(nodes, edges)
                level_nodes = [node for node in G.nodes if G.nodes[node]['level'] == level]
                end_time = time.time()
                mem_after = process.memory_info().rss 

                st.write(f"Total nodes at level {level}: {len(level_nodes)}")
                st.write(f"Nodes: {level_nodes}")
                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {(mem_after - mem_before)/(1024**2) :.4f} MB")

            except Exception as e:
                st.error(f"Error during node distribution query: {e}")

    # Tab for Degree Distribution by Level
    with tabs[3]:
        if st.button("Degree Distribution by Level"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 
                G = create_graph(nodes, edges)
                degree_distribution = {}
                for node in G.nodes:
                    node_level = G.nodes[node]['level']
                    degree = G.degree(node)
                    if node_level not in degree_distribution:
                        degree_distribution[node_level] = []
                    degree_distribution[node_level].append(degree)

                end_time = time.time()
                mem_after = process.memory_info().rss 

                for level, degrees in degree_distribution.items():
                    avg_degree = np.mean(degrees)
                    st.write(f"Level {level}: Average degree: {avg_degree:.2f}")

                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {abs(mem_after - mem_before)/(1024**2) :.4f} MB")

            except Exception as e:
                st.error(f"Error during degree distribution query: {e}")

    # Tab for Supplier Distribution
    with tabs[4]:
        if st.button("Supplier Distribution"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 
                G = create_graph(nodes, edges)
                supplier_counts = {}
                for node in G.nodes:
                    supplier = G.nodes[node]['supplier']
                    if supplier not in supplier_counts:
                        supplier_counts[supplier] = 0
                    supplier_counts[supplier] += 1

                end_time = time.time()
                mem_after = process.memory_info().rss

                st.write("Supplier Distribution:")
                for supplier, count in supplier_counts.items():
                    st.write(f"Supplier {supplier}: {count} nodes")

                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {(mem_after - mem_before)/(1024**2) :.4f} MB")

            except Exception as e:
                st.error(f"Error during supplier distribution query: {e}")

    # Subgraph Extraction and Visualization
    with tabs[5]:
        node_id = st.number_input("Enter node ID for subgraph extraction", min_value=0)
        radius = st.number_input("Enter radius for subgraph extraction", min_value=1)

        if st.button("Subgraph Extraction and Visualization"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 
                G = create_graph(nodes, edges)
                # Use nx.bfs_tree for more efficient subgraph extraction
                subgraph_nodes = nx.bfs_tree(G, source=node_id, depth_limit=radius)
                subgraph = G.subgraph(subgraph_nodes)

                end_time = time.time()
                mem_after = process.memory_info().rss 

                st.write(f"Subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")

                # Visualize the subgraph
                fig_subgraph = visualize_graph(subgraph)

                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {(mem_after - mem_before)/(1024**2) :.4f} MB")

                # Display the plotly chart of the subgraph
                st.plotly_chart(fig_subgraph)

            except Exception as e:
                st.error(f"Error during subgraph extraction: {e}")


with st.expander("Graph Centralities", expanded=False):
    if st.button("Calculate Centralities"):
        # Load nodes and edges from memory-mapped files
        st.write("Loading data from memory-mapped files...")
        nodes = load_mmap_data('nodes.mmap', node_dtype)
        edges = load_mmap_data('edges.mmap', edge_dtype)

        # Measure memory before graph construction
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss 

        st.write("Creating graph...")
        G = create_graph(nodes, edges)

        # Measure time and memory after graph construction
        end_time = time.time()
        mem_after = process.memory_info().rss

        st.write(f"Graph construction completed in {end_time - start_time:.2f} seconds")
        st.write(f"Memory used for graph construction: {(mem_after - mem_before) / (1024 ** 2):.2f} MB")

        # Calculate centralities
        st.write("Calculating centralities...")
        
        start_centrality_time = time.time()
        
        # Degree Centrality
        degree_centrality = nx.degree_centrality(G)
        
        # Betweenness Centrality
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Closeness Centrality
        closeness_centrality = nx.closeness_centrality(G)
        
        end_centrality_time = time.time()

        mem_after_centrality = process.memory_info().rss

        st.write(f"Centralities calculated in {end_centrality_time - start_centrality_time:.2f} seconds")
        st.write(f"Memory used for centralities calculation: {(mem_after_centrality - mem_before) / (1024 ** 2):.2f} MB")

        # Displaying results
        st.subheader("Degree Centrality")
        for node, centrality in degree_centrality.items():
            st.write(f"Node {node}: Centrality {centrality:.4f}")

        st.subheader("Betweenness Centrality")
        for node, centrality in betweenness_centrality.items():
            st.write(f"Node {node}: Centrality {centrality:.4f}")

        st.subheader("Closeness Centrality")
        for node, centrality in closeness_centrality.items():
            st.write(f"Node {node}: Centrality {centrality:.4f}")


import pandas as pd
with st.expander("Manage Nodes", expanded=False):
    option = st.radio("Choose an action:", ("Add Nodes", "Delete Nodes"))
if option == "Add Nodes":
    st.write("Upload a CSV file containing node features (excluding ID) and parent-child relationships.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame for user confirmation
        st.write("Data Preview:")
        st.dataframe(df)

        if st.button("Add Nodes"):
            try:
                # Measure time and memory before adding nodes
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 

                # Load existing nodes and edges
                nodes = load_mmap_data('nodes.mmap', node_dtype)
                edges = load_mmap_data('edges.mmap', edge_dtype)

                # Determine the current maximum ID for new nodes
                current_max_id = nodes['id'].max() + 1

                # Prepare new nodes
                new_nodes = np.zeros(len(df), dtype=node_dtype)
                new_nodes['id'] = np.arange(current_max_id, current_max_id + len(df))
                
                # Fill in other features from the DataFrame
                for column in df.columns:
                    if column in node_dtype.names[1:]:  # Exclude 'id'
                        new_nodes[column] = df[column].values

                # Append new nodes to existing nodes
                combined_nodes = np.concatenate((nodes, new_nodes))
                
                # Save updated nodes back to memory-mapped file
                save_to_mmap(combined_nodes, 'nodes.mmap')

                # Process parent-child relationships to create edges
                for index, row in df.iterrows():
                    parent_id = row.get('parent_id')
                    child_id = row.get('child_id')

                    if pd.notna(parent_id):  # Ensure parent ID is present
                        # Create an edge from parent to the newly added node
                        edges = np.append(edges, np.array([(parent_id, current_max_id + index, 1)], dtype=edge_dtype))

                    if pd.notna(child_id):  # Check if child ID is present before creating an outgoing edge
                        edges = np.append(edges, np.array([(current_max_id + index, child_id, 1)], dtype=edge_dtype))

                # Save updated edges back to memory-mapped file
                save_to_mmap(edges, 'edges.mmap')

                end_time = time.time()
                mem_after = process.memory_info().rss

                st.success("Nodes added successfully!")
                st.write(f"Time taken to add nodes: {end_time - start_time:.4f} seconds")
                st.write(f"Memory used during addition: {(mem_after - mem_before) / (1024 ** 2):.4f} MB")

            except Exception as e:
                st.error(f"Error adding nodes: {e}")
                
    elif option == "Delete Nodes":
        st.write("Upload a CSV file containing Node IDs to delete (separated by commas).")
        delete_file = st.file_uploader("Choose a CSV file for deletion", type="csv")

        if delete_file is not None:
            # Read the CSV file into a DataFrame
            delete_df = pd.read_csv(delete_file, header=None)
            delete_ids = delete_df[0].astype(int).tolist()  # Assuming IDs are in the first column

            if st.button("Delete Nodes"):
                try:
                    # Measure time and memory before deletion
                    start_time = time.time()
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss 

                    # Load existing nodes and edges
                    nodes = load_mmap_data('nodes.mmap', node_dtype)
                    edges = load_mmap_data('edges.mmap', edge_dtype)

                    # Create a set of IDs to delete for quick lookup
                    ids_to_delete = set(delete_ids)

                    # Filter out the nodes that are not in the deletion list
                    filtered_nodes = nodes[~np.isin(nodes['id'], ids_to_delete)]

                    # Create a mask for edges to keep only those that do not connect to deleted nodes
                    filtered_edges = edges[~np.isin(edges['source_id'], ids_to_delete) & ~np.isin(edges['target_id'], ids_to_delete)]

                    # Save updated nodes and edges back to memory-mapped files
                    save_to_mmap(filtered_nodes, 'nodes.mmap')
                    save_to_mmap(filtered_edges, 'edges.mmap')

                    end_time = time.time()
                    mem_after = process.memory_info().rss

                    st.success(f"{ids_to_delete} Nodes and their corresponding edges deleted successfully!")
                    st.write(f"Time taken to delete nodes: {end_time - start_time:.4f} seconds")
                    st.write(f"Memory used during deletion: {(mem_after - mem_before) / (1024 ** 2):.4f} MB")

                except Exception as e:
                    st.error(f"Error deleting nodes: {e}")