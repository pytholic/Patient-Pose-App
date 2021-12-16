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
    export_path = ObjectProperty(None)


    def press(self):
        train = self.train_path.text
        test = self.test_path.text
        export = self.export_path.text

        print(f"Train path: {train}, Test path: {test}, Export path: {export}")

class PatientPoseApp(App):
    def build(self):
        return MyGridLayout()

if __name__ == "__main__":
    PatientPoseApp().run()