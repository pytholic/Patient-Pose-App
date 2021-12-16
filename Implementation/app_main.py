from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder


Builder.load_file("design.kv")


class MyLayout(Widget):
    
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
        return MyLayout()

if __name__ == "__main__":
    PatientPoseApp().run()