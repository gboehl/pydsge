
pydsge
======

Contains the functions and classes for solving, filtering and estimating DSGE models @ZLB. Undocumented and very rawwwww.

The code is in alpha state and provided for reasons of replicability and code sharing in the spirit of open science. It does not (and for now, can not) have a toolbox character. The code is operational, but (yet) not ready for public use and I can not provide any support. You are however very welcome to get in touch if you are interested working with the package.

The beta stage will involve considerable restructuring of packages, code, and the API.

The dependencies are listed in the setup.py file. Note that this package depends on the ``econsieve`` and ``grgrlib`` packages which both can be found on my github page (they will thus not be installed automatically via ``pip``\ , at least for now). The code does *not* work with Python 2.x!

There is some `very preliminary documentation <https://pydsge.readthedocs.io/en/latest/index.html>`_ out there.

parser
------

The parser originally was a fork of Ed Herbst's fork from Pablo Winant's (excellent) package dolo. This version seemed slightly easier to adjust in order to obtain the matrices I need than in the up-to-date and more advanced version of dolo.

See https://github.com/EconForge/dolo and https://github.com/eph.

Installation with ``pip`` (simple)
--------------------------------------

First, be sure that you are on Python 3.x. Then, get the ``econsieve`` and ``grgrlib`` packages:

* https://github.com/gboehl/grgrlib

* https://github.com/gboehl/econsieve

The simplest way is to clone the repository and then from within the cloned folder run (Windows user from the Anaconda Prompt):

.. code-block:: bash

   pip3 install .

Installation with ``pip`` (elegant via ``git``\ )
-------------------------------------------------------

The handy way is to first install ``git``. Linux users just use their respective repos. 

Windows users probably use anaconda and can do

.. code-block:: bash

   conda install -c anaconda git

in the conda shell `as they kindly tell us here <https://anaconda.org/anaconda/git>`_. Otherwise you can probably get it `here <https://git-scm.com/download/win>`.

Then you can simply do

.. code-block:: bash

   pip install git+https://github.com/gboehl/grgrlib
   pip install git+https://github.com/gboehl/econsieve
   pip install git+https://github.com/gboehl/pydsge

Maybe you'd have to use `pip3` instead. If you run it and it complains about missing packages, please let me know so that I can update the `setup.py`!
