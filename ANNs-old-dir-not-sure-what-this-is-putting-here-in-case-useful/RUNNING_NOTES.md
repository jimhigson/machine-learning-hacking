
Python versions 
===============

Tensorflow supports python 3.6, not 3.7 which is installed by default by brew

Brew install 3.6/3.7
---------------

https://stackoverflow.com/questions/51125013/how-can-i-install-a-previous-version-of-python-3-in-macos-using-homebrew

```bash
    brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
```

can then change like:

```
    brew switch python 3.6.5_1
    brew switch python 3.7.0
```

virtualenv with 3.6
-------------------

virtualenv ./envs/anns --python=/usr/local/bin/python3

Activate virtual env like:
-------------------------

Yes, virtual envs *do* work in FISH!

```bash
. envs/anns/bin/activate.fish
```

