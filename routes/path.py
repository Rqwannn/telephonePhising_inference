from app import api
from app.model import *

def AI_API_PATH():
    api.add_resource(Inference, "/stream", endpoint="inference.post", methods=["POST"])
    api.add_resource(Inference, "/save_model", endpoint="inference.get", methods=["GET"])
