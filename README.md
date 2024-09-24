# Change2sketch
## Describe
To train a model which change normal picture to sketch style. The model is refer to the cycleGAN, [Jun-yan Zhu and his colleagues at UC Berkeley in 2017](<https://arxiv.org/pdf/1703.10593>).
![image](https://github.com/user-attachments/assets/0ce324ef-4f07-484f-85b1-40903a9c2b44)

## Designed as user interface by QyPt5
To let user easiler to use this model, I used Qypt5 to design user interface.
![image](https://github.com/user-attachments/assets/1785082e-cfc1-44d1-b105-4e9337a12c0d)


## Convert to .exe(for Window)
Using the Kits Pyinstaller, you can change ```.py``` file to ```.exe``` file. For example:
```
pyinstaller -F your_python.py -c --icon=your_picture.ico
```

* -F: package into a single file but run slowly.
* -c: only use the command line, do not open the window.
* --icon=PATH: set icon.
