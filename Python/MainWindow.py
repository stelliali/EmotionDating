from pathlib import Path

import tkinter as tk
import threading
import time
from tkinter import Tk, Canvas,Button, PhotoImage, Label

class MainWindow(tk.Tk):

    def __init__(self, events, parameters):
        """
        Returns a new toplevel window and initializes it.

        @param events: Methods, which should be called from user interface
        @param parameters: Parameters, which should be editable from user interface
        """
        self.events = events
        self.parameters = parameters
        self.OUTPUT_PATH = Path(__file__).parent
        self.ASSETS_PATH = self.OUTPUT_PATH / Path("./assets/interface")
        self.window = Tk()
        self.stream = None
        self.tracker = None
        self.window.geometry("720x740")
        self.window.configure(bg = "#FFFFFF")
        self.window.title("FitBrain")
        global __images
        __images = dict()
        self.img_names = ["profile.png","button_start.png","button_exit.png","button_audio_off.png","button_camera_off.png","button_live_off.png","button_meme_off.png",
                            "button_audio_on.png","button_camera_on.png","button_live_on.png","button_meme_on.png"]
        self.load_images()
        self.btn_states = {'audio': False, 'camera': False, 'live': False, 'meme': False}

        self.canvas = self.create_canvas(720, 740)
        self.btn_start = self.create_img_button("button_start.png", 298, 32, 365, 671, lambda: self.window.after(0, self.start_stream))
        self.btn_help = self.create_img_button("button_exit.png", 298, 32, 57, 671, self.close_stream)
        self.btn_audio = self.create_img_button("button_audio_off.png", 198, 32, 107, 593, self.on_audio_click)
        self.btn_camera = self.create_img_button("button_camera_off.png", 198, 32, 107, 529, self.on_camera_click)
        self.btn_live = self.create_img_button("button_live_off.png", 198, 32, 415, 529, self.on_hud_click)
        self.btn_meme = self.create_img_button("button_meme_off.png", 198, 32, 415, 593, self.on_meme_click)

        self.img_profile = self.create_img(self.canvas, "profile.png", 584, 245)

        self.rct_level_right = self.create_rectangle(self.canvas, 641, 370, 665, 394, "#5F27FF","")
        self.rct_level_portrait = self.create_rectangle(self.canvas, 482, 245, 498, 246, "", "#0243EC")
        self.rct_experience_box = self.create_rectangle(self.canvas, 46, 188, 269, 303, "", "#1E2022")
        self.rct_experience_bar_max = self.create_rectangle(self.canvas, 63, 246, 400, 246, "", "#F83E3E")
        self.rct_experience_bar_live = self.create_rectangle(self.canvas, 63, 246, 298, 246, "", "#0017E5")

        self.txt_headline = self.create_text("ATTENTION CONTROL CENTER", 190, 27, "#63728E", ("Helvetica bold", 22 * -1))
        self.txt_profile_sec = self.create_text("PROFILE SECTION", 29, 93, "#63728E", ("Poppins Bold", 18 * -1))
        self.txt_system_sec = self.create_text("SYSTEM SETTINGS", 40, 468, "#63728E", ("Poppins Bold", 18 * -1))
        self.txt_level = self.create_text("3", 647, 372, "#FFFFFF", ("Poppins Bold", 12 * -1), "#5F27FF")
        self.txt_profile = self.create_text("PROFILE", 558, 111, "#6B6974", ("Poppins Medium", 12 * -1))
        self.txt_name = self.create_text("Manuel Totzauer", 506, 371, "#8A8795", ("Poppins Medium", 16 * -1))
        self.txt_level_profile = self.create_text("3", 466, 237, "#0243EC", ("Poppins Bold", 12 * -1))
        self.txt_level_under = self.create_text("Level 3", 563, 403, "#6B6974", ("Poppins Regular", 12 * -1))
        self.txt_session_sec = self.create_text("SESSION SETTINGS", 29, 418, "#63728E", ("Poppins Bold", 18 * -1))
        self.txt_exp = self.create_text("244 EXP missing to lvl 4", 68, 259, "#77838F", ("Poppins Regular", 14 * -1))
        self.txt_Experience = self.create_text("Experience", 63, 208, "#1E2022", ("Poppins Medium", 14 * -1))

        # Experience gaps for each level
        self.score = 0
        self.level_gap = [2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 150, 300, 750, 5000]

        self.load_score()
        self.set_level()

        self.window.resizable(False, False)
        self.window.mainloop()

    def relative_to_assets(self, path: str) -> Path:
        return self.ASSETS_PATH / Path(path)

    def load_images(self):
        """
        Initially loads the "img_names" from the 'assets/interface/' directory and writes it into an array.
        """
        for img_name in self.img_names:
            path = Path('assets/interface/' + img_name)
            image = PhotoImage(file = path)
            __images[img_name] = image

    def create_canvas(self, width, height):
        canvas = Canvas(
            self.window,
            bg = "#FFFFFF",
            height = height,
            width = width,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )
        canvas.place(x = 0, y = 0)
        return canvas

    def create_img(self, canvas, img_name, x, y):
        """
        Creates new image, which is loaded from the __images array.
        """
        canvas.create_image(
            x,
            y,
            image=__images[img_name]
        )

    def create_img_button(self, img_name, width, height, x, y, handler):
        """
        Creates new image button with event handler.
        """
        button = Button(
            image=__images[img_name],
            borderwidth=0,
            highlightthickness=0,
            command=handler,
            relief="flat"
        )
        button.place(
            x=x,
            y=y,
            width=width,
            height=height
        )
        return button

    def on_click(self, sender, button, parameter_id):
        """
        Switches button image and parameter value on each button click.
        """
        # Switches button image on click
        if (self.btn_states[sender] == False):
            image = "button_" + sender + "_on.png"
            button.config(image=__images[image])
            self.btn_states[sender] = True
        else:
            image = "button_" + sender + "_off.png"
            button.config(image=__images[image])
            self.btn_states[sender] = False

        # Negates state of button connected parameter
        if self.parameters[parameter_id][0] == False:
            self.parameters[parameter_id][0] = True
        else: self.parameters[parameter_id][0] = False

    def create_text(self, text, x, y, fill, font, bg = "#FFFFFF"):
        label = Label(
            self.window,
            text=text,
            fg=fill,
            bg=bg,
            font=font
        )
        label.place(
            x=x,
            y=y,
        )
        return label

    def create_rectangle(self, canvas, xStart, yStart, xEnd, yEnd, fill, outline):
        canvas.create_rectangle(
            xStart,
            yStart,
            xEnd,
            yEnd,
            fill=fill,
            outline=outline
        )

    def start_stream(self):
        """
        Starts video stream and score tracker in new threads.
        """
        self.stream = threading.Thread(target=self.events[0])
        self.stream.start()

        self.load_score()
        self.tracker = threading.Thread(target=self.start_score_tracker)
        self.tracker.start()

    def close_stream(self):
        '''
        Sets capture_stop as well as distracted to True and saves the score.
        '''
        self.parameters[0][0] = True
        self.parameters[4][0] = True
        self.save_score()

    def on_audio_click(self):
        self.on_click("audio", self.btn_audio, 1)

    def on_hud_click(self):
        self.on_click("live", self.btn_live, 2)

    def on_camera_click(self):
        self.on_click("camera", self.btn_camera, 3)

    def on_meme_click(self):
        None

    def start_score_tracker(self):
        '''
        Calls the set_level() method in equal intervalls.
        '''
        time.sleep(3)
        # Checks if stream is still running
        while (self.parameters[0][0] == False):
            self.set_level()
            time.sleep(1)

    def set_level(self):
        '''
        Manages the score and level gap system. Score gets increased if user is NOT distracted.\n
        Checks how much exp is needed for the next level and sets all related components to actual value.
        '''
        i = 0
        is_smaller = False

        # Add score if user is NOT distracted
        if self.parameters[4][0] == False: self.score += 1

        # Check which level comes next according to score
        while is_smaller == False:
            if self.score < self.level_gap[i]: is_smaller = True
            else: i += 1

        # Calculate xp needed for level up
        exp_for_lvl_up = self.level_gap[i] - self.score

        # Set text of all level related components
        self.txt_exp.config(text = f"{exp_for_lvl_up} EXP missing to lvl {i+1}")
        self.txt_level.config(text = i)
        self.txt_level_under.config(text = f"Level {i}")
        self.txt_level_profile.config(text = i)

        # Centers label if level gets two digits
        if i>9: self.txt_level.place(x=643,y=372)

        # Calculate percentage of the level progress
        if self.score < self.level_gap[0]: percent = 100 * self.score / self.level_gap[i]
        else: percent = 100 * (self.score - self.level_gap[i-1]) / (self.level_gap[i] - self.level_gap[i-1])

        # Calculate the length of the level bar
        bar_length = (337 / 100 * percent) + 63

        # Redraw level bar with actual progress
        self.rct_experience_bar_max = self.create_rectangle(self.canvas, 63, 246, 400, 246, "", "#F83E3E")
        self.rct_experience_bar_live = self.create_rectangle(self.canvas, 63, 246, bar_length, 246, "", "#0017E5")

    def load_score(self):
        file = open(Path('assets/interface/score.txt'), "r")
        for columns in ( raw.strip().split() for raw in file ):
            self.score = int(columns[0])

    def save_score(self):
        with open(Path('assets/interface/score.txt'), 'w') as f:
            f.write(str(self.score))
