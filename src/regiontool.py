import argparse
import tkinter as tk
from PIL.ImageTk import PhotoImage
from pyautogui import screenshot
from screeninfo import get_monitors, Monitor


class RegionTool:
    def __init__(self, args: argparse.Namespace):
        monitors = get_monitors()
        self.monitor: Monitor = monitors[args.display]

        self.start_x = self.start_y = self.end_x = self.end_y = 0
        region = (self.monitor.x, self.monitor.y, self.monitor.width, self.monitor.height)
        image = screenshot(region=region, allScreens=True)

        # Setup the Tkinter window
        self.root = tk.Tk()
        self.root.geometry(f"{self.monitor.width}x{self.monitor.height}+{self.monitor.x}+{self.monitor.y}")

        # Convert the screenshot to a format Tkinter can use
        self.image = PhotoImage(image)

        # Create a canvas to display the screenshot and capture mouse events
        self.canvas = tk.Canvas(self.root, width=self.monitor.width, height=self.monitor.height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Bind the mouse events
        self.canvas.bind('<Button-1>', self.__on_click)  # Left mouse button click
        self.canvas.bind('<B1-Motion>', self.__on_drag)  # Dragging with left mouse button held down
        self.canvas.bind('<ButtonRelease-1>', self.__on_release)  # Left mouse button release
        self.root.bind('<Escape>', self.__on_escape)

    def __on_click(self, event):
        """Handle the initial click by saving the start coordinates."""
        self.start_x, self.start_y = event.x, event.y

    def __on_drag(self, event):
        """Handle the drag operation by updating the overlay with the region"""
        self.end_x, self.end_y = event.x, event.y
        self.canvas.delete("region")  # Remove the old region
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y,
                                     outline='red', width=2, tags="region")

    def __on_release(self, event):
        """Handle the release of the mouse button, close the program."""
        self.end_x, self.end_y = event.x, event.y
        x, y = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
        mx, my = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
        print(f"Selection: [{x+self.monitor.x}, {y+self.monitor.y}, {mx-x}, {my-y}]")

    def __on_escape(self, _):
        self.root.destroy()

    def run(self):
        print("PRESS ESCAPE TO QUIT AFTER SELECTING REGIONS")
        self.root.mainloop()
