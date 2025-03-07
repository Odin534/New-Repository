import tkinter as tk
import time
import datetime
import multiprocessing
import threading
import csv
import os
import ctypes
import json
from cortex import Cortex
from queue import Queue
from decimal import Decimal, ROUND_DOWN

current_label = None
stimuli_complete = False  # Flag to track completion of flickering
current_dir = os.getcwd()

with open('config.json', 'r') as file:
    config = json.load(file)
    #profiles = json.load(file)
current_profile = config['current_profile']
data_dir = config['data_dir']
print(f"Current data_dir: {current_profile}")
print(f"Current data_dir: {data_dir}")
#current_profile = profiles[-1] # Get the current profile (assumes the current profile is the last one in the list)
#data_dir = os.path.join(current_dir, 'Profile', current_profile)

# Screen dimensions and colors
#width, height = 1000, 800  # Fixed screen size
white = "#ffffff"
black = "#000000"
red = "#FF0000"

# Arrow and button dimensions
arrow_size = 100
button_size = 100

def draw_arrow(canvas, arrow, pos):
    if arrow == "left":
        canvas.create_polygon(
            pos[0], pos[1] + arrow_size // 2,
            pos[0] + arrow_size, pos[1],
            pos[0] + arrow_size, pos[1] + arrow_size,
            fill=white
        )
    elif arrow == "right":
        canvas.create_polygon(
            pos[0] + arrow_size, pos[1] + arrow_size // 2,
            pos[0], pos[1],
            pos[0], pos[1] + arrow_size,
            fill=white
        )

def draw_button(canvas, pos, color):
    canvas.create_rectangle(
        pos[0], pos[1],
        pos[0] + button_size, pos[1] + button_size,
        fill=color
    )

def flicker_object(canvas, object, frequency, duration, object_positions, color, log_file, queue, cortex, ipc_queue):
    global current_label
    current_label = object  # Set the global variable to the current arrow direction
    cortex.current_label = current_label  # Set the cortex object's current_label attribute
    period = 1 / frequency
    end_time = time.time() + duration

    label_mapping = {"left": 0, "right": 1, "stop": 2, "rest": "Rest"}

    while time.time() < end_time:
        
        if object in ["left", "right"]:
            draw_arrow(canvas, object, object_positions[object])
        else:
            draw_button(canvas, object_positions[object], color)

        log_file.write(f"{time.time():.2f}: {object}\n")  # Write timestamp and object to the log file

        # Write to CSV file
        timestamp = Decimal(time.time()).quantize(Decimal("0.00"), rounding=ROUND_DOWN)
        current_time = datetime.datetime.fromtimestamp(float(timestamp))
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Send the current_label to the cortex.py script through ipc_queue
        ipc_queue.put(current_label)
        queue.put({"Unix Timestamp": str(timestamp), "Human Readable Timestamp": current_time_str, "Label": label_mapping[current_label]})
        canvas.update()
        time.sleep(period / 2 - 0.001)

        canvas.delete("all")
        current_label = "rest"  # Set the global variable and cortex object's current_label attribute to "rest"
        cortex.current_label = current_label
        ipc_queue.put(current_label)  # Send the "rest" label to the cortex.py script through ipc_queue
        queue.put({"Unix Timestamp": str(timestamp), "Human Readable Timestamp": current_time_str, "Label": "Rest"})
        canvas.update()
        time.sleep(period / 2 - 0.001)

    if object == "right":  # Check if the last object in the sequence is shown
        global stimuli_complete
        stimuli_complete = True
        ipc_queue.put("stimuli_complete")

def write_to_csv(queue):
    with open('data.csv', 'w', newline='') as csvfile:
        fieldnames = ['Unix Timestamp', 'Human Readable Timestamp', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        last_written_time = None
        while True:
            row = queue.get()
            if row is None:
                break
            unix_timestamp = Decimal(row['Unix Timestamp'])
            label = row['Label']
            current_time_str = row['Human Readable Timestamp']
            if last_written_time is None or unix_timestamp > Decimal(last_written_time):
                writer.writerow({'Unix Timestamp': str(unix_timestamp), 'Human Readable Timestamp': current_time_str, 'Label': label})
                last_written_time = str(unix_timestamp)

def get_screen_dimensions():
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height
    #return 1000, 800

def close_window(root):
    global stimuli_complete
    stimuli_complete = True
    root.destroy()

def run_visual_stimuli(start_event, cortex, ipc_queue):
    try:
        root = tk.Tk()  # Create the root window
        root.title("Visual Stimuli Display Screen")
        width, height = get_screen_dimensions()
        root.geometry(f"{width}x{height}")
        
        control_frame = tk.Frame(root, bg=black)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        #close_button = tk.Button(control_frame, text="Close", command=lambda: close_window(root), bg=black, fg=white)
        #close_button.pack(side=tk.RIGHT, padx=10, pady=10)

        canvas = tk.Canvas(root, width=width, height=height, bg=black)
        canvas.pack()

        object_positions = {
            "left": ((width // 2) - arrow_size * 2, (height // 2) - arrow_size // 2),
            "right": ((width // 2) + arrow_size, (height // 2) - arrow_size // 2),
            "stop": ((width // 2) - button_size // 2, (height // 2) - button_size // 2),
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(current_dir, "log.txt")
        csv_file_path = os.path.join(current_dir, "data.csv")
        
        cortex.start(current_label)  # Start the Cortex data recording


        with open(log_file_path, "w") as log_file, open(csv_file_path, "w", newline='') as csv_file:
            object_order = ["left", "right", "stop"]
            frequencies = [11, 13, 9]
            duration = 14
            gap = 6

            queue = Queue()
            writer_thread = threading.Thread(target=write_to_csv, args=(queue,))

            writer_thread.start()

            queue.put({'Unix Timestamp': Decimal(time.time()).quantize(Decimal("0.00"), rounding=ROUND_DOWN),
                       'Human Readable Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       'Label': "Rest"})
            #time.sleep(10)  # Rest for 10 seconds

            countdown_time = 10
            countdown_label = tk.Label(root, text="Please relax yourself. The flickering starts in :", font=("Arial", 20), fg=white, bg=black)
            countdown_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
            while countdown_time > 0:
                canvas.delete("all")
                canvas.create_text(width // 2, height // 2, text=f"Timer: {countdown_time}", fill=white, font=("Arial", 24))
                root.update()
                time.sleep(1)
                countdown_time -= 1

            canvas.delete("all")
            countdown_label.destroy()


            for _ in range(3):
                for i, obj in enumerate(object_order):
                    flicker_object(canvas, obj, frequencies[i], duration, object_positions, red if obj=="stop" else white, log_file, queue, cortex, ipc_queue)
                    queue.put({'Unix Timestamp': Decimal(time.time()).quantize(Decimal("0.00"), rounding=ROUND_DOWN),
                               'Human Readable Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               'Label': "Rest"})
                    time.sleep(gap)

            queue.put(None)  # Signal the writer thread to exit
            writer_thread.join()
            #writer_thread.terminate()
        #while not stimuli_complete:
        #    root.update()  # Update the Tkinter window

        cortex.stop()  # Stop the Cortex data recording
        #root.mainloop()
        root.destroy()  # Close the Tkinter window

    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    start_event = multiprocessing.Event()
    
    # Create a Queue for inter-process communication
    ipc_queue = multiprocessing.Queue()
    
    cortex = Cortex(ipc_queue, data_dir)

    process = multiprocessing.Process(target=run_visual_stimuli, args=(start_event, cortex, ipc_queue))
    process.start()

    start_event.wait()  # Wait for the event to be set
    current_label = None  # Set the initial current label

    cortex.start(current_label)  # Start the Cortex data recording

    # Perform any other tasks here while the visual stimuli display is running

    process.join()  # Wait for the visual stimuli display process to finish
