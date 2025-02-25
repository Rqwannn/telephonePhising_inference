from app.websocket.audio_realtime import *

class SocketManager:
    def __init__(self, socketio_server):
        self.register_handlers = audio_socket_handlers(socketio_server)