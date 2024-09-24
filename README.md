# Change2sketch
## Describe
To train a model which change normal picture to sketch style. The model is refer to the cycleGAN, [Jun-yan Zhu and his colleagues at UC Berkeley in 2017](<https://arxiv.org/pdf/1703.10593>).

## Designed as user interface by QyPt5
To let user easiler to use this model, I used Qypt5 to design user interface.

## Convert to .exe(for Window)
Using the Kits Pyinstaller, you can change ```.py``` file to ```.exe``` file. For example:
```
pyinstaller -F your_python.py -c --icon=your_picture.ico
```

* -F: package into a single file but run slowly.
* -c: only use the command line, do not open the window.
* --icon=PATH: set icon.