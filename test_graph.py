from app import all_passages
from src.research_analyzer import analyze_connections
from src.connection_graph import build_connection_graph

connections = analyze_connections(all_passages, top_n=15)

figure = build_connection_graph(connections)

figure.write_html("outputs/research_connection_map.html")

print("Graph generated successfully!")
print("Open: outputs/research_connection_map.html")
