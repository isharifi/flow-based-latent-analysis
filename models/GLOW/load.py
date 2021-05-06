import tensorflow as tf
from models.GLOW.model import Glow_Model

def load_from_graph(graph_dir):
    generator = Glow_Model(graph_dir)
    return generator