from flask import request, abort, jsonify
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage

import torch
from huggingface_hub import HfApi, hf_hub_download

from PIL import Image
import numpy as np

from dotenv import load_dotenv
import joblib
import os

class Inference(Resource):
    def __init__(self):
        pass

    def get(self):
        pass
    
    def post(self):
        pass