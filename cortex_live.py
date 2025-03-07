import asyncio
import websockets
import json
import time
import multiprocessing
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from multiprocessing import Queue

from RNN_GRU import classify_realtime_data_rnn_gru
from robot_control import RobotControl

client_id = "Npi8UYcSbt5S9UYOgcTPXMk617VlHLwaISMrADu0"
client_secret = "q4f73wdxWThS9vc1GE8OBTjx35Di3yxEv2sAfGQfOfeRXkcdLIgBd9LgtuPsFObhX4h6HkppTu6zuJJ5xkP9jOiiJYvLgJDa91NMbcOsznELuTgTUnYdLxVnpBtGWJpY"
should_stop = False
url = "wss://localhost:6868"


# Creating tkinter window for live prediction
class RealtimeClassificationWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-time  Classification Result")
        self.label = tk.Label(self.root, text="Waiting for classification...", font=("Arial", 20))
        self.label.pack(padx=20, pady=20)

    def update_classification(self, result):
        self.label.config(text=f"Classification Result: {result}")
        self.root.update()

    def run(self):
        self.root.mainloop()


# Create and run the tkinter window
window = RealtimeClassificationWindow()


async def send(ws, request):
    message = json.dumps(request)
    print("Sending:", message)
    await ws.send(message)


async def receive(ws):
    message = await ws.recv()
    print("Received:", message)
    return json.loads(message)


def create_session_request(cortex_token, headset_id):
    request = {
        "jsonrpc": "2.0",
        "method": "createSession",
        "params": {
            "cortexToken": cortex_token,
            "status": "active",
            "headset": headset_id
        },
        "id": 3
    }
    return request


def create_error_message():
    root = tk.Tk()
    root.withdraw()  # hides the root window
    messagebox.showerror("Error", "Headset is disconnected. Please check.")
    root.destroy()


async def receive_and_display_data(ws, cortex, queue, start_time, channel_list):
    global should_stop
    rob = RobotControl()
    command_queue = Queue(maxsize=100)
    chunks_queue = Queue(maxsize=100)
    buffer = []
    BUFFER_SIZE = 128
    classification_map = {
        0: "Left",
        1: "Right",
        2: "Stop"
    }

    while not should_stop:
        data = json.loads(await ws.recv())
        if 'eeg' in data:
            print(f"EEG Data: {data}")
            sample = data['eeg']
            sample_list = []
            for item in sample:
                if isinstance(item, list):
                    sample_list.extend(item)
                else:
                    sample_list.append(item)

            # for live mode
            eeg_df = pd.DataFrame([sample_list], columns=channel_list[:-1])
            buffer.append(eeg_df)
            if len(buffer) == BUFFER_SIZE:
                chunk_df = pd.concat(buffer, axis=0)
                # Adding chunks of data needed for classification
                chunks_queue.put(chunk_df)
                buffer = []
                print(chunk_df.shape)

                while not chunks_queue.empty():
                    # preprocessing steps here
                    eeg_data = chunks_queue.get()
                    preprocessed_eeg_data = []  # call function here or do preprocessing here.
                    async with command_queue.put(classify_realtime_data_rnn_gru(preprocessed_eeg_data)):
                        while not command_queue.empty():
                            command = await command_queue.get()
                            rob.command_robot(classification_map[command])
                            window.update_classification(classification_map[command])
                        rob.command_sm(classification_map[2])


class Cortex:
    def __init__(self, queue, mode="SSVEP_final"):  # mode = SSVEP or SSVEP_final or MI
        self.process = None
        self.queue = queue
        self.mode = mode

    def start(self):
        self.process = multiprocessing.Process(target=self.run, args=(self.queue,))
        self.process.start()

    def stop(self):
        global should_stop
        should_stop = True
        if self.process is not None:
            self.process.join()

    def run(self, queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main(queue))
        loop.close()

    async def main(self, queue):
        global should_stop
        should_stop = False

        async with websockets.connect(url) as ws:
            try:
                auth_request = {
                    "jsonrpc": "2.0",
                    "method": "authorize",
                    "params": {
                        "clientId": client_id,
                        "clientSecret": client_secret
                    },
                    "id": 1
                }
                await send(ws, auth_request)
                response = await receive(ws)
                cortex_token = response["result"]["cortexToken"]

                query_headsets_request = {
                    "jsonrpc": "2.0",
                    "method": "queryHeadsets",
                    "params": {},
                    "id": 2
                }
                await send(ws, query_headsets_request)
                response = await receive(ws)
                if response["result"]:
                    headset_id = response["result"][0]["id"]
                else:
                    create_error_message()
                    return
                headset_id = response["result"][0]["id"]

                session_request = create_session_request(cortex_token, headset_id)
                await send(ws, session_request)
                response = await receive(ws)
                session_id = response["result"]["id"]

                sub_request = {
                    "jsonrpc": "2.0",
                    "method": "subscribe",
                    "params": {
                        "cortexToken": cortex_token,
                        "session": session_id,
                        "streams": ["eeg"]
                    },
                    "id": 4
                }
                await send(ws, sub_request)
                extended_result = await receive(ws)
                await receive(ws)
                channel_list = extended_result["result"]["success"][0]["cols"]

                start_time = time.time()  # Record the start time
                await receive_and_display_data(ws, self, queue, start_time, channel_list)

            except KeyError:
                create_error_message()


if __name__ == "__main__":
    queue = Queue()
    cortex = Cortex(queue, "SSVEP_final")  # or "SSVEP" or "MI"
    cortex.start()
    time.sleep(2)
