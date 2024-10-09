import streamlit as st
import numpy as np
import json
from datetime import datetime
import time
import psutil
import time
import os
import networkx as nx
import plotly.graph_objects as go
import concurrent.futures
from numba import jit

st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                width: 70%;
                margin-left: 15%;
            }
            </style>
            """, unsafe_allow_html=True)

# Define data types for nodes and edges
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

import concurrent.futures

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
        
        if 'level' in node_type:
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

def generate_graph(config):
    total_nodes = config['total_nodes']
    total_edges = config['total_edges']
    node_types = config['node_types']
    suppliers = config['suppliers']
    attr_ranges = config['attribute_ranges']

    # Use multithreading to parallelize node and edge generation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_nodes = executor.submit(generate_nodes, node_types, total_nodes, attr_ranges, suppliers)
        future_edges = executor.submit(generate_edges, node_types, future_nodes.result(), total_edges)

    nodes = future_nodes.result()
    edges = future_edges.result()
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

                memory_used_nodes = nodes.nbytes / (1024 ** 2)
                memory_used_edges = edges.nbytes / (1024 ** 2)

                unique_types, counts = np.unique(nodes['type'], return_counts=True)
                type_count_dict = dict(zip(unique_types.astype(str), counts))

                st.success("Graph generated and saved successfully!")
                st.write(f"Total Nodes: {len(nodes)}")
                st.write(f"Total Edges: {len(edges)}")
                st.write(f"Graph generated in {(end_time - start_time).total_seconds()} seconds")
                st.write(f"Memory Used (Nodes): {memory_used_nodes:.2f} MB")
                st.write(f"Memory Used (Edges): {memory_used_edges:.2f} MB")
                
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
        st.write(f"Memory used for graph construction: {mem_after - mem_before:.2f} MB")

        # Measure time and memory for visualization
        st.write("Visualizing graph...")
        start_viz_time = time.time()

        fig = visualize_graph(G)

        end_viz_time = time.time()
        st.write(f"Graph visualization completed in {end_viz_time - start_viz_time:.2f} seconds")

        # Display the Plotly graph in the Streamlit app
        st.plotly_chart(fig)

with st.expander("Graph Queries", expanded=False):

    nodes = load_mmap_data('nodes.mmap', node_dtype)
    edges = load_mmap_data('edges.mmap', edge_dtype)

    G = create_graph(nodes, edges)
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

                if nx.has_path(G, source_id, target_id):
                    shortest_path = nx.shortest_path(G, source=source_id, target=target_id)
                    result_text = f"Shortest path between {source_id} and {target_id}: {shortest_path}"
                else:
                    result_text = f"No path exists between {source_id} and {target_id}"

                end_time = time.time()
                mem_after = process.memory_info().rss 

                st.write(result_text)
                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {mem_after - mem_before:.4f} MB")

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

                path_exists = nx.has_path(G, source=source_id, target=target_id)
                result_text = f"Path exists between {source_id} and {target_id}: {path_exists}"

                end_time = time.time()
                mem_after = process.memory_info().rss 

                st.write(result_text)
                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {mem_after - mem_before:.4f} MB")

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

                level_nodes = [node for node in G.nodes if G.nodes[node]['level'] == level]
                end_time = time.time()
                mem_after = process.memory_info().rss 

                st.write(f"Total nodes at level {level}: {len(level_nodes)}")
                st.write(f"Nodes: {level_nodes}")
                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {mem_after - mem_before:.4f} MB")

            except Exception as e:
                st.error(f"Error during node distribution query: {e}")

    # Tab for Degree Distribution by Level
    with tabs[3]:
        if st.button("Degree Distribution by Level"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 

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
                st.write(f"Memory taken: {mem_after - mem_before:.4f} MB")

            except Exception as e:
                st.error(f"Error during degree distribution query: {e}")

    # Tab for Supplier Distribution
    with tabs[4]:
        if st.button("Supplier Distribution"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 

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
                st.write(f"Memory taken: {mem_after - mem_before:.4f} MB")

            except Exception as e:
                st.error(f"Error during supplier distribution query: {e}")

        #Subgraph Extraction and Visualization
    with tabs[5]:
        node_id = st.number_input("Enter node ID for subgraph extraction", min_value=0)
        radius = st.number_input("Enter radius for subgraph extraction", min_value=1)

        if st.button("Subgraph Extraction and Visualization"):
            try:
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss 

                # Use nx.bfs_tree for more efficient subgraph extraction
                subgraph_nodes = nx.bfs_tree(G, source=node_id, depth_limit=radius)
                subgraph = G.subgraph(subgraph_nodes)

                end_time = time.time()
                mem_after = process.memory_info().rss 

                st.write(f"Subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")

                # Visualize the subgraph
                fig_subgraph = visualize_graph(subgraph)

                st.write(f"Time taken: {end_time - start_time:.4f} seconds")
                st.write(f"Memory taken: {mem_after - mem_before:.4f} MB")

                # Display the plotly chart of the subgraph
                st.plotly_chart(fig_subgraph)

            except Exception as e:
                st.error(f"Error during subgraph extraction: {e}")
