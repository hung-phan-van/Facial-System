
from flask_socketio import SocketIO
import time
from flask import Flask, request, jsonify, redirect, url_for, render_template, Response
import threading
import os

g_background_thread = None
app = Flask(__name__)
# NOTE: must set async_handlers = False => handle event by event in sequence
my_flask_socketio = SocketIO(app, async_handlers = False)

g_recent_streaming_time = time.time()

@my_flask_socketio.on('streaming_client_send_time_to_monitoring_process', namespace='')
def record_latest_time(data):
    global g_recent_streaming_time
  
    print('Receiving time from client: ',g_recent_streaming_time )
    g_recent_streaming_time = data['time']

def background_thread():
    global g_recent_streaming_time
    while True:
        if (time.time() - g_recent_streaming_time > 8):
            print('*********** We need to restart process here')
            # os.system('pm2 restart 0')
            os.system('pm2 restart client_sent_image')
            time.sleep(10)
        else:
            print('ok - no need to restart server')
        time.sleep(3)


# @my_flask_socketio.on('connect', namespace='/test')  # /test podría ser /trucks o /requests tambien.
@my_flask_socketio.on('connect')  # /test podría ser /trucks o /requests tambien.
def test_connect():
    print("Testing connection")
    return

if __name__ == '__main__':
    print("server running")
    thread1 = threading.Thread(target=background_thread)
    thread1.start()
    my_flask_socketio.run(app, host='0.0.0.0', port=8000)


