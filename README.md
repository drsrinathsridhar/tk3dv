[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/drsrinathsridhar/tk3dv/blob/master/LICENSE)

# tk3dv
The Toolkit for 3D Vision (tk3dv) is a collection of visualization tools for 3D computer vision/graphics. 

**Note:** To use pyEasel, the visualization component of tk3dv, you must use tk3dv on a machine with a display as it uses OpenGL.

## Requirements

- [Palettable][2]: `pip install palettable`
- [Matplotlib][4]: `conda install -c conda-forge matplotlib`
- Please install GLUT if your OS does not come with it. For Ubunutu, `sudo apt-get install freeglut3-dev`.


## Installation

After the above requirements are installed, you can install tk3dv like so:

`pip install git+https://github.com/drsrinathsridhar/tk3dv.git`

If reinstalling on Ubuntu, make sure to uninstall and repeat the install.


[1]: https://github.com/anderskm/gputil
[2]: https://jiffyclub.github.io/palettable/
[3]: https://pytorch.org/
[4]: https://matplotlib.org/stable/users/installing.html

## Issues

<details><summary>Import error on macOS Big Sur </summary>
<p>

Please find the solution in [this issue](https://github.com/drsrinathsridhar/tk3dv/issues/4).

</p>
</details>
