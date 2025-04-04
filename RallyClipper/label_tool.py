import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import csv

class VideoFrameSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Selector")

        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.segments = []
        self.start_frame = None
        self.tint_color = (0, 255, 0)  # Green tint

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, length=600, command=self.update_frame)
        self.slider.pack()

        self.load_button = tk.Button(self.root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        self.save_button = tk.Button(self.root, text="Save Segments", command=self.save_segments)
        self.save_button.pack()

        self.root.bind("s", self.mark_start)
        self.root.bind("e", self.mark_end)
        self.root.bind("<Left>", self.prev_frame)
        self.root.bind("<Right>", self.next_frame)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.config(to=self.frame_count - 1)
            self.update_frame()

    def update_frame(self, *args):
        if self.cap is not None:
            self.current_frame = int(self.slider.get())
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.is_frame_in_segment(self.current_frame) or \
                   (self.start_frame is not None and self.start_frame <= self.current_frame):
                    overlay = np.full(frame.shape, self.tint_color, dtype=np.uint8)
                    tinted_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                else:
                    tinted_frame = frame

                tinted_frame = cv2.resize(tinted_frame, (640, 480))
                photo = tk.PhotoImage(data=cv2.imencode('.ppm', tinted_frame)[1].tobytes())
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.image = photo

    def is_frame_in_segment(self, frame):
        return any(start <= frame < end for start, end in self.segments)

    def mark_start(self, event):
        if self.cap is not None:
            self.start_frame = self.current_frame
            print(f"Start marked at frame {self.start_frame}")
            self.update_frame()

    def mark_end(self, event):
        if self.cap is not None and self.start_frame is not None:
            end_frame = self.current_frame
            if end_frame > self.start_frame:
                self.segments.append((self.start_frame, end_frame))
                print(f"Segment added: [{self.start_frame}, {end_frame})")
                self.start_frame = None
                self.update_frame()
            else:
                print("End frame must be after start frame")

    def save_segments(self):
        if self.segments:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv")
            if save_path:
                with open(save_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Frame', 'InSegment'])
                    
                    for frame in range(self.frame_count):
                        in_segment = self.is_frame_in_segment(frame)
                        writer.writerow([frame, 1 if in_segment else 0])
                
                print(f"Segments saved to {save_path}")
        else:
            print("No segments marked")

    def prev_frame(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.slider.set(self.current_frame)

    def next_frame(self, event):
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
            self.slider.set(self.current_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFrameSelector(root)
    root.mainloop()