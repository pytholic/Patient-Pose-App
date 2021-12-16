from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty

class MyGridLayout(Widget):
    
    train_path = ObjectProperty(None)
    test_path = ObjectProperty(None)
    model_path = ObjectProperty(None)


    def press_train(self):
        train = self.train_path.text
        print(f"Train path: {train}")
        self.train_path.text = ""

    def press_test(self):
        test = self.test_path.text
        print(f"Test path: {test}")
        self.test_path.text = ""

    def press_export(self):
        model = self.model_path.text
        print(f"Model path: {model}")
        self.model_path.text = ""

class PatientPoseApp(App):
    def build(self):
        return MyGridLayout()

if __name__ == "__main__":
    PatientPoseApp().run()