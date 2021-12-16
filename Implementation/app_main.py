import runpy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder


Builder.load_file("design.kv")

train = []
test = []

class MyLayout(Widget):
    
    train_path = ObjectProperty(None)
    test_path = ObjectProperty(None)
    model_path = ObjectProperty(None)


    def press_train(self):
        train_data = self.train_path.text
        train.append(str(train_data))
        self.train_path.text = ""
        print("*****Starting Training*****")
        runpy.run_path('./train.py')
        train.clear()

    def press_test(self):
        test_data = self.test_path.text
        test.append(str(test_data))
        self.test_path.text = ""
        print("*****Starting Training*****")
        runpy.run_path('./test.py')
        test.clear()

    def press_export(self):
        model = self.model_path.text
        print(f"Model path: {model}")
        self.model_path.text = ""

class PatientPoseApp(App):
    def build(self):
        return MyLayout()

if __name__ == "__main__":
    PatientPoseApp().run()