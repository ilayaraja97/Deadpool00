# Deadpool00
## Setting up environment

### OpenCV
[Windows](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#installing-opencv-from-prebuilt-binaries) | 
[Ubuntu](https://docs.opencv.org/trunk/d2/de6/tutorial_py_setup_in_ubuntu.html)
### Theano
Installing [Theano](http://deeplearning.net/software/theano/install.html#install)
### Keras
Installing [Keras](https://keras.io/#installation)

Change the Keras backend to [theano](https://keras.io/backend/)
###
## Coding style
We will stick to [PEP8](https://www.python.org/dev/peps/pep-0008/) coding style.

## Know who's working how much
````
$ git shortlog -s -n # commits
$ git ls-files | while read f; do git blame --line-porcelain $f | grep '^author '; done | sort -f | uniq -ic | sort -n # loc
````
