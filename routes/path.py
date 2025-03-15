from app import api
from app.model import *
from app.InsightAI import *
from app.InsightOpenAI import *
from app.controller.auth import *

def AI_API_PATH():
    api.add_resource(InsightAI, "/stream_insight", endpoint="insightai.post", methods=["POST"])
    api.add_resource(InsightOpenAI, "/stream_insight_openai", endpoint="insightopenai.post", methods=["POST"])
    api.add_resource(Inference, "/stream", endpoint="inference.post", methods=["POST"])
    api.add_resource(Inference, "/save_model", endpoint="inference.get", methods=["GET"])

    api.add_resource(Login, "/api/v1/engkol/resource/login")
    api.add_resource(Register, "/api/v1/engkol/resource/register")
    api.add_resource(Register, "/api/v1/engkol/resource/register/<string:id>", endpoint='Register.delete', methods=["DELETE"])