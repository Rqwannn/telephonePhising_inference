from app import create_app
from flask_socketio import SocketIO
import os

app = create_app()
socketio_server = SocketIO(app, cors_allowed_origins="*")

if __name__ == "__main__":
    # app.run(debug=True) # False If Production

    from app.websocket.handler import SocketManager

    SocketManager(socketio_server)
    socketio_server.run(app, debug=True, host='0.0.0.0', port=5000)