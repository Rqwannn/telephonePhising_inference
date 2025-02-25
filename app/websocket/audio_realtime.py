from flask_socketio import send, emit

values = {
    'slider1': 25,
    'slider2': 0,
}

def audio_socket_handlers(socketio_server):

    @socketio_server.on('connect')
    def test_connect():
        emit('after connect',  {'data':'Lets dance'})

    @socketio_server.on('Slider value changed')
    def value_changed(message):
        values[message['who']] = message['data']
        emit('update value', message, broadcast=True)
