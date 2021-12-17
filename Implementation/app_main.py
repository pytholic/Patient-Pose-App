import runpy
import sys
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.core.window import Window

Builder.load_file("design.kv")

train_dir = []
test_dir = []
model_dir = []

class MyLayout(Widget):
    
    train_path = ObjectProperty(None)
    test_path = ObjectProperty(None)
    model_path = ObjectProperty(None)


    def press_train(self):
        train_data = self.train_path.text
        train_dir.append(str(train_data))
        self.train_path.text = ""
        test_data = self.test_path.text
        test_dir.append(str(test_data))
        self.test_path.text = ""
        if train_data and test_data:
            print("*****Starting Training*****")
            runpy.run_path('./train.py')
            print("*****Finished Training*****")
        else:
            print("Please enter training path of train and test folder!")

        train_dir.clear()
        test_dir.clear()

    def press_test(self):
        test_data = self.test_path.text
        test_dir.append(str(test_data))
        self.test_path.text = ""
        model_data = self.model_path.text
        model_dir.append(str(model_data))
        self.model_path.text = ""
        if test_data:
            print("*****Starting Testing*****")
            runpy.run_path('./test.py')
            print("*****Finished Testing*****")
            test_dir.clear()
        else:
            print(("Please enter path of test folder!"))

    def press_export(self):
        model_data = self.model_path.text
        model_dir.append(str(model_data))
        self.model_path.text = ""
        print("*****Exporting Model*****")
        runpy.run_path('./export.py')
        print("*****Finished Exporting*****")
        model_dir.clear()

class PatientPoseApp(App):
    def build(self):
        Window.clearcolor = (0,0,0,1)
        return MyLayout()

if __name__ == "__main__":
    PatientPoseApp().run()