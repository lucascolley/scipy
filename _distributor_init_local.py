import os

# on Windows SciPy loads important DLLs
# and the code below aims to alleviate issues with DLL
# path resolution portability with an absolute path DLL load
if os.name == 'nt':
    from ctypes import WinDLL
    # check for `.pixi\envs\default\Library\bin\openblas.dll`
    libs_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '.pixi', 'envs', 'default', 'Library', 'bin'
    ))
    if os.path.isdir(libs_path):
        try:
            owd = os.getcwd()
            os.chdir(libs_path)
            WinDLL(os.path.abspath(os.path.join(libs_path, 'openblas.dll')))
        finally:
            os.chdir(owd)
