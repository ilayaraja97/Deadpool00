# Deadpool00
## Teammates
[Himani](https://github.com/HimaniRathi/Deadpool00) | [Chandna](https://github.com/IamChandna/Deadpool00)
## Setting up environment
Use ubuntu 64-bit Python 3
### OpenCV
`$ sudo apt install python3-opencv`
### Tensorflow
`$ sudo pip3 install tensorflow`
### Keras
`$ sudo pip3 install keras`
And run `testEnvironment.py`
## Coding style
We will stick to [PEP8](https://www.python.org/dev/peps/pep-0008/) coding style.

## Know who's working how much
````
$ git shortlog -s -n # commits
$ git ls-files | while read f; do git blame --line-porcelain $f | grep '^author '; done | sort -f | uniq -ic | sort -n # loc
````
