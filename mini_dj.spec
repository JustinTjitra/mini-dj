
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

hidden = (
    collect_submodules('mediapipe')
    + collect_submodules('cv2')
    + ['rtmidi']
)
binaries = collect_dynamic_libs('mediapipe') + collect_dynamic_libs('cv2')

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=[('images/*', 'images')],
    hiddenimports=hidden,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts,
    name='MiniDJ',
    console=False,  
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='MiniDJ')
