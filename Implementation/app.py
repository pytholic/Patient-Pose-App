from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput


class PatientPoseApp(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1

        # Create text fields grid layout
        self.top_grid = GridLayout()
        self.top_grid.cols = 2

        # Create buttons grid layout
        self.button_grid = GridLayout()
        self.button_grid.cols = 3

        # Set window size
        self.window.size_hint = (0.9, 0.9)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # Set top grid size
        self.top_grid.size_hint = (0.9, 0.9)
        self.top_grid.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        
        # Set button grid size
        self.button_grid.size_hint = (0.9, 0.9)
        self.button_grid.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # Add widgets to window

        # Logo widget
        self.window.add_widget(Image(source="./resources/logo.png"))

        # Data path widgets
        self.train_label = Label(text="Train folder: ",
                                 font_size=20,
                                 color='#00FFCE'
                                 )
        self.test_label = Label(text="Test folder: ",
                                font_size=20,
                                color='#00FFCE'
                                )
        self.top_grid.add_widget(self.train_label)
        self.top_grid.add_widget(self.test_label)

        # Text input widgets
        self.train_path = TextInput(multiline=False,
                                    font_size=18,
                                    padding_y = (15, 10),
                                    size_hint = (1, 0.4),
                                    )
        self.test_path = TextInput(multiline=False,
                                   font_size=18,
                                   padding_y = (15, 10),
                                   size_hint = (1, 0.4),
                                   )
        self.top_grid.add_widget(self.train_path)
        self.top_grid.add_widget(self.test_path)

        # Button widgets
        self.train_button = Button(text="TRAIN MODEL",
                                   font_size=18,
                                   size_hint_y = None,  # important to set this
                                   height=50,
                                   )
        self.test_button = Button(text="TEST MODEL",
                                  font_size=18,
                                  size_hint_y = None,
                                  height=50,
                                  )
        self.export_button = Button(text="EXPORT MODEL",
                                    font_size=18,
                                    size_hint_y = None,
                                    height=50,
                                    )
        self.button_grid.add_widget(self.train_button)
        self.button_grid.add_widget(self.test_button)
        self.button_grid.add_widget(self.export_button)


        # Add secondary grids to the main window grid
        self.window.add_widget(self.top_grid)
        self.window.add_widget(self.button_grid)

        return self.window

if __name__ == "__main__":
    PatientPoseApp().run()