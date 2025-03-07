import asyncio
import websockets
import json
import csv
import os, time
import multiprocessing
import datetime
import tkinter as tk
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from multiprocessing import Queue
##from FFT_SVM import classify_realtime_data
##from RNN_GRU import classify_realtime_data_rnn_gru


client_id = "Npi8UYcSbt5S9UYOgcTPXMk617VlHLwaISMrADu0"
client_secret = "q4f73wdxWThS9vc1GE8OBTjx35Di3yxEv2sAfGQfOfeRXkcdLIgBd9LgtuPsFObhX4h6HkppTu6zuJJ5xkP9jOiiJYvLgJDa91NMbcOsznELuTgTUnYdLxVnpBtGWJpY"
should_stop = False
url = "wss://localhost:6868"

#creating tkinter window for live prediction
'''
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
'''
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

#async def receive_and_write_data(ws, writer, cortex, queue, start_time, channel_list):
async def receive_and_write_data(ws, writer, cortex, queue, start_time):

    global should_stop
    '''
    buffer = []
    BUFFER_SIZE = 128
    classification_map = {
    0: "Left",
    1: "Right",
    2: "Stop"
    }
    '''
    state = "Rest"  # Initialize state
    cycle_count = 0  # Initialize cycle_count
    state_count = 0  # Initialize state_count
    elapsed_time = 0
    start_time = time.time()  # Initialize start_time
    while not should_stop:
        data = json.loads(await ws.recv())
        if 'eeg' in data:
            print(f"EEG Data: {data}")
            sample = data['eeg']
            sample_time = data['time']
            sample_list = []
            for item in sample:
                if isinstance(item, list):
                    sample_list.extend(item)
                else:
                    sample_list.append(item)
            
            #for live mode
            '''
            print(sample_list)
            print(channel_list)
            eeg_df = pd.DataFrame([sample_list], columns=channel_list[:-1])
            buffer.append(eeg_df)
            if len(buffer) == BUFFER_SIZE:
                
                chunk_df = pd.concat(buffer, axis=0)                
                print(chunk_df.shape)
                classification_result = classify_realtime_data_rnn_gru(chunk_df)
                window.update_classification(classification_map[classification_result[0]])
                buffer = []
            '''  
            sample_list.append(sample_time)
            sample_list.append(datetime.datetime.fromtimestamp(sample_time).strftime('%Y-%m-%d %H:%M:%S'))

            if cortex.mode == "SSVEP":

                elapsed_time = time.time() - start_time

                if elapsed_time < 10:
                    state = "Rest"
                elif elapsed_time < 24:
                    state = "Left"
                elif elapsed_time < 30:
                    state = "Rest"
                elif elapsed_time < 44:
                    state = "Right"
                elif elapsed_time < 50:
                    state = "Rest"
                elif elapsed_time < 64:
                    state = "Left"
                elif elapsed_time < 70:
                    state = "Rest"
                elif elapsed_time < 84:
                    state = "Right"
                elif elapsed_time < 90:
                    state = "Rest"
                elif elapsed_time < 104:
                    state = "Left"
                elif elapsed_time < 110:
                    state = "Rest"
                elif elapsed_time < 124:
                    state = "Right"
                elif elapsed_time < 130:
                    state = "Rest"
            

                
                # Check if 130 seconds have passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= 130:
                    should_stop = True

                
                if not queue.empty():
                    cortex.current_label = queue.get() 
            
            elif cortex.mode == "SSVEP_final":
                elapsed_time = time.time() - start_time

                if elapsed_time < 10:
                    state = "Rest"
                elif elapsed_time < 24:
                    state = "Left"
                elif elapsed_time < 30:
                    state = "Rest"
                elif elapsed_time < 44:
                    state = "Right"
                elif elapsed_time < 50:
                    state = "Rest"
                elif elapsed_time < 64:
                    state = "Stop"
                elif elapsed_time < 70:
                    state = "Rest"
                elif elapsed_time < 84:
                    state = "Left"
                elif elapsed_time < 90:
                    state = "Rest"
                elif elapsed_time < 104:
                    state = "Right"
                elif elapsed_time < 110:
                    state = "Rest"
                elif elapsed_time < 124:
                    state = "Stop"
                elif elapsed_time < 130:
                    state = "Rest"
                elif elapsed_time < 144:
                    state = "Left"
                elif elapsed_time < 150:
                    state = "Rest"
                elif elapsed_time < 164:
                    state = "Right"
                elif elapsed_time < 170:
                    state = "Rest"
                elif elapsed_time < 184:
                    state = "Stop"
                elif elapsed_time < 190:
                    state = "Rest"

                
                # Check if 190 seconds have passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= 190:
                    should_stop = True

                
                if not queue.empty():
                    cortex.current_label = queue.get() 
                    
            elif cortex.mode == "MI":
                elapsed_time = time.time() - start_time
                if elapsed_time < 10:
                    state = "Rest"
                elif elapsed_time < 52:
                    state = "Forward"
                
                
                # Check if 190 seconds have passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= 52:
                    should_stop = True

                
                if not queue.empty():
                    cortex.current_label = queue.get() 
                    
            elif cortex.mode == "Four_class":
                elapsed_time = time.time() - start_time

                if elapsed_time < 10:
                    state = "Rest"
                elif elapsed_time < 24:
                    state = "Left"
                elif elapsed_time < 30:
                    state = "Rest"
                elif elapsed_time < 44:
                    state = "Right"
                elif elapsed_time < 50:
                    state = "Rest"
                elif elapsed_time < 64:
                    state = "Stop"
                elif elapsed_time < 70:    
                    state = "Rest"
                elif elapsed_time < 84:
                    state = "Forward"
                elif elapsed_time < 90:
                    state = "Rest"
                elif elapsed_time < 104:
                    state ="Left"
                elif elapsed_time < 110:
                    state = "Rest"
                elif elapsed_time < 124:
                    state = "Right"
                elif elapsed_time < 130:
                    state = "Rest"
                elif elapsed_time < 144:
                    state = "Stop"
                elif elapsed_time < 150:
                    state = "Rest"
                elif elapsed_time < 164:
                    state = "Forward"
                elif elapsed_time < 170:
                    state = "Rest"
                elif elapsed_time < 184:
                    state = "Left"
                elif elapsed_time < 190:
                    state = "Rest"
                elif elapsed_time < 204:
                    state = "Right"
                elif elapsed_time < 210:
                    state = "Rest"
                elif elapsed_time < 224:
                    state = "Stop"
                elif elapsed_time < 230:
                    state = "Rest"
                elif elapsed_time < 244:
                    state = "Forward"
                elif elapsed_time < 250:
                    state = "Rest"
                
                # Check if 250 seconds have passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= 250:
                    should_stop = True

                
                if not queue.empty():
                    cortex.current_label = queue.get() 
            

            sample_list.append(state)  # Append the state as "New Label"
            writer.writerow(sample_list)


class Cortex:
    def __init__(self, queue, data_dir, mode = "SSVEP_final"):   # mode = SSVEP or SSVEP_final or MI
        self.process = None
        self.current_label = None
        self.queue = queue
        self.mode = mode
        self.data_dir = data_dir 
        
    def start(self, current_label):
        self.current_label = current_label
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

        if not os.path.exists(self.data_dir):  
            os.makedirs(self.data_dir)

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
                print('__________________________________________________')
                print(extended_result["result"]["success"][0]["cols"])
                print('__________________________________________________')
                channel_list = extended_result["result"]["success"][0]["cols"]
                
                # Creating timestamp string
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                csv_filename = os.path.join(self.data_dir, f'eeg_data_{timestamp}.csv')

                with open(csv_filename, 'w', newline='') as csvfile:
                    #fieldnames = channel_list +['Human Readable Time']+['label']+['New Label']
                    fieldnames = channel_list +['Human Readable Time']+['New Label']                    
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(fieldnames)
                    start_time = time.time()  # Record the start time
                    await receive_and_write_data(ws, writer, self, queue, start_time)
                    #await receive_and_write_data(ws, writer, self, queue, start_time, channel_list)

            
            except KeyError:
                create_error_message()


if __name__ == "__main__":
    queue = Queue()
    cortex = Cortex(queue, "SSVEP_final")  # or "SSVEP" or "MI"
    cortex.start("Label")
    time.sleep(2)
