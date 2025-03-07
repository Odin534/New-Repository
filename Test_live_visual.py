import tkinter as tk
import os
import ctypes
import cv2
from PIL import Image, ImageTk
import subprocess

process = None

# Screen dimensions and colors
white = "#ffffff"
black = "#000000"
red = "#FF0000"

# Arrow and button dimensions
arrow_size = 100
button_size = 100

def start_recording():
    global process
    process = subprocess.Popen(['python', 'cortex_live.py'])

def stop_recording():
    global process
    if process:
        process.terminate()
        

def draw_arrow(canvas, arrow, pos):
    if arrow == "left":
        return canvas.create_polygon(
            pos[0], pos[1] + arrow_size // 2,
            pos[0] + arrow_size, pos[1],
            pos[0] + arrow_size, pos[1] + arrow_size,
            fill=white
        )
    elif arrow == "right":
        return canvas.create_polygon(
            pos[0] + arrow_size, pos[1] + arrow_size // 2,
            pos[0], pos[1],
            pos[0], pos[1] + arrow_size,
            fill=white
        )
    elif arrow == "forward":
        return canvas.create_polygon(
            pos[0] + arrow_size // 2, pos[1],
            pos[0], pos[1] + arrow_size,
            pos[0] + arrow_size, pos[1] + arrow_size,
            fill=white
        )

def draw_button(canvas, pos, color):
    return canvas.create_rectangle(
        pos[0], pos[1],
        pos[0] + button_size, pos[1] + button_size,
        fill=color
    )

def toggle_flicker(canvas, obj_id, frequency, existing_jobs):
    if obj_id in existing_jobs:
        canvas.after_cancel(existing_jobs[obj_id])
        del existing_jobs[obj_id]
        canvas.itemconfig(obj_id, state='normal')
    else:
        period = 1 / frequency
        visible = True

        def flicker():
            nonlocal visible
            if visible:
                canvas.itemconfig(obj_id, state='hidden')
            else:
                canvas.itemconfig(obj_id, state='normal')
            visible = not visible
            existing_jobs[obj_id] = canvas.after(int(period * 1000), flicker)

        flicker()

def setup_key_bindings(root, canvas, object_ids, existing_jobs):
    frequencies = {
        "left": 11,
        "right": 13,
        "forward": 9,
        "stop": 15
    }

    def on_key_press(event):
        if event.keysym == 'Up':
            toggle_flicker(canvas, object_ids["forward"], frequencies["forward"], existing_jobs)
        elif event.keysym == 'Down':
            toggle_flicker(canvas, object_ids["stop"], frequencies["stop"], existing_jobs)
        elif event.keysym == 'Left':
            toggle_flicker(canvas, object_ids["left"], frequencies["left"], existing_jobs)
        elif event.keysym == 'Right':
            toggle_flicker(canvas, object_ids["right"], frequencies["right"], existing_jobs)

    root.bind('<KeyPress>', on_key_press)

def run_visual_stimuli():
    try:
        root = tk.Tk()
        root.title("Visual Stimuli Display Screen")
        width, height = get_screen_dimensions()
        root.geometry(f"{width}x{height}")
    

        canvas = tk.Canvas(root, width=width, height=height, bg=black)
        canvas.pack()

        # Start and Stop Recording buttons
        start_button = tk.Button(root, text="Start Recording", command=start_recording)
        stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
        start_button.place(x=0, y=0)
        stop_button.place(x=width - 100, y=0)
        

        object_positions = {
            "left": (arrow_size // 2, (height // 2) - arrow_size // 2),
            "right": (width - arrow_size * 1.5, (height // 2) - arrow_size // 2),
            "forward": ((width // 2) - arrow_size // 2, arrow_size // 2),
            "stop": ((width // 2) - button_size // 2, height - button_size * 1.5),
        }

        object_ids = {}
        for obj, pos in object_positions.items():
            if obj in ["left", "right", "forward"]:
                object_ids[obj] = draw_arrow(canvas, obj, pos)
            else:
                object_ids[obj] = draw_button(canvas, pos, red)

        existing_jobs = {}  # To store active flicker jobs.

        # Start all arrows and the stop button flickering as soon as the program starts
        frequencies = {
            "left": 11,
            "right": 13,
            "forward": 15,
            "stop": 9
        }
        for obj, freq in frequencies.items():
            toggle_flicker(canvas, object_ids[obj], freq, existing_jobs)

        setup_key_bindings(root, canvas, object_ids, existing_jobs)

        video_window_width = width - 9*arrow_size
        video_window_height = height -6*arrow_size - button_size
        video_window_x = (width - video_window_width) // 2
        video_window_y = (height - video_window_height) // 2

        video_label = tk.Label(canvas, bg=black)
        video_label.place(x=video_window_x, y=video_window_y, width=video_window_width, height=video_window_height)

        # OpenCV video capture setup
        video_source = "http://100.127.12.81:5001/stream.mjpg"  
        cap = cv2.VideoCapture(video_source)

        def update_video_feed():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (video_window_width, video_window_height))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
            else:
                # If there's no video feed
                video_label.configure(text="No Signal", font=("Arial", 24), fg="white", bg="black")

            canvas.after(10, update_video_feed)

        update_video_feed()

        root.mainloop()

    except Exception as e:
        print(f'Error: {e}')

def get_screen_dimensions():
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height

if __name__ == "__main__":
    run_visual_stimuli()
