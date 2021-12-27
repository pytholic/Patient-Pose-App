from kivy_deps import sdl2, glew
import os

# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

PATH = 'C:\\Users\\rajah\\anaconda3\\Lib\\site-packages'

a = Analysis(['app_main.py'],
             pathex=[],
             binaries=[],
             datas=[("./resources/logo.png", '.'), 
             (os.path.join(PATH, "torch"), "torch"),
             (os.path.join(PATH, "torchvision"), "torchvision"),
             (os.path.join(PATH, "kivy"), "kivy"),
             (os.path.join(PATH, "onnx"), "onnx"),
             (os.path.join(PATH, "numpy"), "numpy"),
             (os.path.join(PATH, "sklearn"), "sklearn")],
             hiddenimports=["sklearn", "onnx", "numpy", "torch", "torchvision", "kivy"],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

a.datas += [('Code\design.kv', 'E:\\skia_projects\\Patient-Pose-App\\app_dev\\app_testing\\design.kv', 'DATA')]

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='app_main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               Tree('E:\\skia_projects\\Patient-Pose-App\\app_dev\\app_testing\\'),
               a.binaries,
               a.zipfiles,
               a.datas, 
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               upx_exclude=[],
               name='app_main')