from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

class MyGridLayout(GridLayout):
    # Initialize infinit keywords
    def __init__(self, **kwargs):
        # Call grid layout constructor
        super(MyGridLayout, self).__init__(**kwargs)

        # self.window = GridLayout()
        # self.window.cols = 1

        self.cols = 1

        # self.window = GridLayout()
        # self.window.cols = 1

        # Create text fields grid layout
        self.top_grid = GridLayout(row_force_default=True,
                                   row_default_height=50,
                                   )
        self.top_grid.cols = 2

        # Create buttons grid layout
        self.button_grid = GridLayout(row_force_default=True,
                                      row_default_height=50)
        self.button_grid.cols = 3

        # Set window size
        self.size_hint = (0.9, 0.9)
        self.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # Set top grid size
        self.top_grid.size_hint = (0.9, 0.9)
        self.top_grid.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        
        # Set button grid size
        self.button_grid.size_hint = (0.9, 0.9)
        self.button_grid.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # Add widgets to window

        # Logo widget
        self.add_widget(Image(source="./resources/logo.png",
                                     color = (1,1,1,1)))

        # Data path widgets
        self.train_label = Label(text="Train folder: ",
                                 font_size=20,
                                 color='#00FFCE',
                                 size_hint_x = None,
                                 width = 250
                                 )
        self.test_label = Label(text="Test folder: ",
                                font_size=20,
                                color='#00FFCE',
                                size_hint_x = None,
                                width = 250
                                )

        self.model_label = Label(text="Model folder: ",
                                font_size=20,
                                color='#00FFCE',
                                size_hint_x = None,
                                width = 250
                                )

        self.train_path = TextInput(multiline=False,
                                    font_size=18,
                                    padding_y = (15, 10)
                                    # size_hint = (1, None),
                                    # height=50,
                                    #width=300
                                    )
        self.test_path = TextInput(multiline=False,
                                   font_size=18,
                                   padding_y = (15, 10)
                                   # size_hint = (1, None),
                                   # height=50,
                                   #width=300
                                   )

        self.model_path = TextInput(multiline=False,
                                   font_size=18,
                                   padding_y = (15, 10)
                                   # size_hint = (1, None),
                                   # height=50,
                                   #width=300
                                   )

        self.top_grid.add_widget(self.train_label)
        self.top_grid.add_widget(self.train_path)
        self.top_grid.add_widget(self.test_label)
        self.top_grid.add_widget(self.test_path)
        self.top_grid.add_widget(self.model_label)
        self.top_grid.add_widget(self.model_path)

        # Button widgets
        self.train_button = Button(text="TRAIN MODEL",
                                   font_size=18,
                                   )
        self.test_button = Button(text="TEST MODEL",
                                  font_size=18,
                                  )
        self.export_button = Button(text="EXPORT MODEL",
                                    font_size=18,
                                    )
        self.button_grid.add_widget(self.train_button)
        self.button_grid.add_widget(self.test_button)
        self.button_grid.add_widget(self.export_button)

        # Bind the buttons
        self.train_button.bind(on_press=self.press_train)
        self.test_button.bind(on_press=self.press_test)
        self.export_button.bind(on_press=self.press_export)

        # Add secondary grids to the main window grid
        self.add_widget(self.top_grid)
        self.add_widget(self.button_grid)

    def press_train(self, instance):
        train = self.train_path.text
        print(f"Train path: {train}")

    def press_test(self, instance):
        test = self.test_path.text
        print(f"Test path: {test}")

    def press_export(self, instance):
        model = self.model_path.text
        print(f"Model path: {model}")

class PatientPoseOldApp(App):
    def build(self):
        return MyGridLayout()

if __name__ == "__main__":
    PatientPoseOldApp().run()