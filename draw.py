from langchain_core.runnables.graph import MermaidDrawMethod
from PIL import Image

def show_graph(app):
    # Generate the PNG data from the graph
    png_data = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )

    # Save the PNG data to a local file
    file_path = "langgraph/graph/graph.png"
    with open(file_path, "wb") as f:
        f.write(png_data)

    # Open and display the image using Pillow
    image = Image.open(file_path)
    image.show()