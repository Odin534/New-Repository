import tkinter as tk
import time
import datetime
import multiprocessing
import threading
import csv
import os
from cortex import Cortex
from queue import Queue
from decimal import Decimal, ROUND_DOWN

current_label = None
stimuli_complete = False  # Flag to track completion of flickering

# Screen dimensions and colors
width, height = 1400, 800
white = "#ffffff"
black = "#000000"

# Arrow dimensions
arrow_size = 100


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


def flicker_arrow(canvas, arrow, frequency, duration, arrow_positions, log_file, queue, cortex, ipc_queue):
    global current_label
    current_label = arrow  # Set the global variable to the current arrow direction
    cortex.current_label = current_label  # Set the cortex object's current_label attribute
    period = 1 / frequency
    end_time = time.time() + duration

    label_mapping = {"left": 0, "right": 1, "rest": "Rest"}

    while time.time() < end_time:
        draw_arrow(canvas, arrow, arrow_positions[arrow])
        log_file.write(f"{time.time():.2f}: {arrow}\n")  # Write timestamp and arrow to the log file
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

    if arrow == "right":  # Check if the last arrow in the sequence is shown
        global stimuli_complete
        stimuli_complete = True


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


def run_visual_stimuli(start_event, cortex, ipc_queue):
    try:
        root = tk.Tk()  # Create the root window
        root.title("Visual Stimuli Display Screen")
        canvas = tk.Canvas(root, width=width, height=height, bg=black)
        canvas.pack()

        arrow_positions = {
            "left": ((width // 2) - arrow_size * 2, (height // 2) - arrow_size // 2),
            "right": ((width // 2) + arrow_size, (height // 2) - arrow_size // 2),
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(current_dir, "log.txt")
        csv_file_path = os.path.join(current_dir, "data.csv")

        with open(log_file_path, "w") as log_file, open(csv_file_path, "w", newline='') as csv_file:
            arrows_order = ["left", "right"]
            frequencies = [11, 13]
            duration = 14
            gap = 6

            queue = Queue()
            writer_thread = threading.Thread(target=write_to_csv, args=(queue,))

            writer_thread.start()

            queue.put({'Unix Timestamp': Decimal(time.time()).quantize(Decimal("0.00"), rounding=ROUND_DOWN),
                       'Human Readable Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       'Label': "Rest"})
            time.sleep(10)  # Rest for 10 seconds

            for _ in range(3):
                flicker_arrow(canvas, arrows_order[0], frequencies[0], duration, arrow_positions, log_file, queue, cortex, ipc_queue)
                queue.put({'Unix Timestamp': Decimal(time.time()).quantize(Decimal("0.00"), rounding=ROUND_DOWN),
                           'Human Readable Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                           'Label': "Rest"})
                time.sleep(gap)
                flicker_arrow(canvas, arrows_order[1], frequencies[1], duration, arrow_positions, log_file, queue, cortex, ipc_queue)
                queue.put({'Unix Timestamp': Decimal(time.time()).quantize(Decimal("0.00"), rounding=ROUND_DOWN),
                           'Human Readable Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                           'Label': "Rest"})
                time.sleep(gap)

            queue.put(None)  # Signal the writer thread to exit
            writer_thread.join()

        while not stimuli_complete:
            root.update()  # Update the Tkinter window

        cortex.stop()  # Stop the Cortex data recording
        root.destroy()  # Close the Tkinter window

    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    start_event = multiprocessing.Event()
    
    # Create a Queue for inter-process communication
    ipc_queue = multiprocessing.Queue()
    
    cortex = Cortex(ipc_queue)

    process = multiprocessing.Process(target=run_visual_stimuli, args=(start_event, cortex, ipc_queue))
    process.start()

    start_event.wait()  # Wait for the event to be set
    current_label = None  # Set the initial current label

    cortex.start(current_label)  # Start the Cortex data recording

    # Perform any other tasks here while the visual stimuli display is running

    process.join()  # Wait for the visual stimuli display process to finish
