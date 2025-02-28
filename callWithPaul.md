
In python, only global deps (not local) so environments are kind-of like VMs.
    * Virtual Environment (native python)
    * Conda Environment
        conda package management
        similar to a rpm/apt in redhat/debian
        also includes the python vesion

Setting up conda
================

Download from internet for mac/64 bit etc

* `~/opt/anaconda3/condabin/conda init fish` - to set up my config.fish to show conda base path and have conda in path

`conda config --set changeps1 False` removes the shell prompt stuff (since starship gives this anyway)
`conda config --set auto_activate_base False` don't open base by default

Creating/activating a conda environment
============================

Environments are stored in: `/Users/jimhigson/opt/anaconda3/envs/$env_name` - they're not the project dir
    * jupyter notebooks are the project dir (I think) - conda environments are more like swappable VMs
    * like VMs several projects could use one conda env (ie, several ML projects, several graphic projects etc)

> ` ~/opt/anaconda3/condabin/conda create -n machine_learning python=3.7 ipython pandas numpy`
> `conda install jupyter # too add new libs later could have added in environment create above too`

conda install keras - Paul didn't do this - why not and where did it get keras from then?
>   message to Paul to clarify
>   one question for when you have a moment - I notice looking back at my notes that we didn't run
>   'conda install keras' but the jupyter notebook is importing keras - bit confused how this is
>   possible if we didn't put it into the environment

>   Paul doesn't know either. Just accept it for now.

```sh
jimhigson@budapest ~/dev> ~/opt/anaconda3/condabin/conda info --envs # ask to list the envs
# conda environments:
#
base                  *  /Users/jimhigson/opt/anaconda3
machine_learning         /Users/jimhigson/opt/anaconda3/envs/machine_learning

> conda activate machine_learning # switch envs (like switchable VMs) OS shell will now have been modidifed by conda activate.
```

Jupyter notebook creation and use
================================

`which ipython` will now point to the proper python version in that conda environment (~/opt/anaconda3/envs/machine_learning/bin/ipython) since envionment included ipython when created

ipython = interactive python (normal python still has a REPL) but this is interactive-er with syntax highlighting on the console
    * Jupyter originally called 'ipython notebook'
    * ipython can be connected to jupyter via an ipython kernel, which we must create
    * all of this is possible because we installed ipython into our conda environment, and have that conda env activated
    * ie, jupyter notebooks have extension '.ipynb' - as in ipython notebook. J and iP Very closely related.
    * the kernel is the interactive python session that a notebook runs in. Multiple notebooks have multiple instances of the kernel, ie, if you declare a var in python, it's the kernel that holds that var

need to create a kernel for jupyter notebooks (connector between ipython on jupyter webpage)
`Jupyter kernels live in: /Users/jimhigson/Library/Jupyter/kernels/machine_learning`

> `jupyter-notebook` on cli to start jupyter (go to dir first) - will also open a browser

```sh
ipython kernel install --name machine_learning --user # install it for the user (not globally)
    #name doesn't have to be the same as the environment, but convenient if they are the same
    #lets the jup notebook webserver know which python to connect to
    #
    #can select the kernel from jupyter notebook to match the environment, so that it points to the kernel
    #used in your conda environment. The kernel includes the python version, depedencies etc
```

In shell:
    hit !foo to run shell commands

Are now creating a runnable script that can be run from command line or the web-based editor

Keras: lib built on top of tensorflow (simpler than using tf directly) - can choose different backends (including tf, theano) without changing code
    - higher level abstraction over several NN implementations
    - most people (according to Paul) use Keras rather than TF directly

can pass to keras python lists or numpy arrays. Numpy arrays are typed (and in C), and therefore faster than untyped list. Very common lib
Tensorflow is using numpy under the hood (in python at least, maybe in others, to read up on)

X.shape returns a tuple (uneditable list)

model=Sequential()

model.add(Dense(256, input_dim=2))
model.add(Activation("tanh"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

/* could also add like: */
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

**scikit - learn**
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

Scikit learn provide libraries and is also a tool for learning. Ie, provide a good train/test split library
    from sklearn.model_selection import train_test_split


Scikit learn and Keras both come with some toy datasets to practice on

Read Keras examples CFAR 10 CNN (finding out house numbers)
    - a fairly invoced Conv. ANN

Own the Libs
============

[https://www.opentechguides.com/how-to/article/dataanalytics/179/jupyter-notebook-pandas.html]
*Pandas* provide fast, flexible and easy to use data structures for data analysis
    * pandas is for tabular data (called a dataframe) with rows and cols
    * load from sql db, json, csv etc
    * can get subsets, maniupulate it etc
    * pandas supports matplotlib (support is baked in) for graphs

*Matplotlib* is used visualize data using charts such as bar charts, line plots, scatter plots and many more.

*Scikit*
    * for the science, in a kit
    * http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html - makes data like on a scattergraph

*pydot*
    is an interface to Graphviz
    can parse and dump into the DOT language used by GraphViz,
    is written in pure Python,

*Graphviz*
    * Graphviz is open source graph visualization software

*jupytext*
    * https://github.com/mwouts/jupytext

*nbdev*
    * bridges IDEs and notebooks
    * https://www.fast.ai/2019/12/02/nbdev/

