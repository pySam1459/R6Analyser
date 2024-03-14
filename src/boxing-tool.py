import argparse
import tkinter as tk
from tkinter import Canvas
from PIL import ImageTk
from pyautogui import screenshot
from time import sleep


def on_click(event):
    """Handle the initial click by saving the start coordinates."""
    global start_x, start_y
    start_x, start_y = event.x, event.y

def on_drag(event):
    """Handle the drag operation by updating the overlay with the selection box."""
    global end_x, end_y
    end_x, end_y = event.x, event.y
    canvas.delete("selection_box")  # Remove the old selection box
    # Create a new selection box
    canvas.create_rectangle(start_x, start_y, end_x, end_y, outline='red', width=2, tags="selection_box")

def on_release(event):
    """Handle the release of the mouse button, close the program."""
    global end_x, end_y
    end_x, end_y = event.x, event.y
    x, y = min(start_x, end_x), min(start_y, end_y)
    mx, my = max(start_x, end_x), max(start_y, end_y)
    print(f"Selection: [{x}, {y}, {mx-x}, {my-y}]")
    root.destroy()  # Close the tkinter window to exit the program
    

parser = argparse.ArgumentParser(prog="boxing-tool",
                                 description="Useful tool to select bounding boxes on your screen")
parser.add_argument("-d", "--delay",
                    type=int,
                    help="Time delay between starting the program and the screen capture, in seconds",
                    default=2)

args = parser.parse_args()

if args.delay > 0:
    sleep(args.delay)


start_x = start_y = end_x = end_y = 0
image = screenshot()

# Setup the Tkinter window
root = tk.Tk()
root.attributes('-fullscreen', True)  # Make the window full-screen
root.attributes('-topmost', True)
root.after(100, lambda: root.attributes('-topmost', False))

# Convert the screenshot to a format Tkinter can use
tk_screenshot = ImageTk.PhotoImage(image)

# Create a canvas to display the screenshot and capture mouse events
canvas = Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack(fill=tk.BOTH, expand=True)
canvas.create_image(0, 0, anchor=tk.NW, image=tk_screenshot)  # Display the screenshot on the canvas

# Bind the mouse events
canvas.bind('<Button-1>', on_click)  # Left mouse button click
canvas.bind('<B1-Motion>', on_drag)  # Dragging with left mouse button held down
canvas.bind('<ButtonRelease-1>', on_release)  # Left mouse button release

root.mainloop()
