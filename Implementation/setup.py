from cx_Freeze import setup, Executable
build_exe_options = {'build_exe': 'foo'}

setup(name = "App" ,
      version = "0.1" ,
      description = "" ,
      options = {"build_exe": build_exe_options},
      executables = [Executable("app_main.py")])