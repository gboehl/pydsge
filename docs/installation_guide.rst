
Installation Guide
==================

You can either use the repo version from `PyPI <https://pypi.org/>`_, or the bleeding edge version from git.

Installing the repo version is as simple as

.. code-block:: bash

   pip install pydsge

The code is *not* backwards compatible with Python 2.x. 

Installation via ``git``\
--------------------------

As long as this is still work in progress, the version from `PyPI <https://pypi.org/>`_ may not be fully up to date. To get the current git snapshot you have two choices: installing "by hand", or using the ``git`` command. I strongly recommend using ``git`` as it facilitates updates and the like. This is the handy way.

Note that this package depends on the ``emcwrap``, ``grgrlib`` and ``econsieve`` packages which both can be found on my github page, and on `PyPI <https://pypi.org/>`_.

First install ``git``. Linux users just use their respective repos. 

Windows users probably use anaconda and can do

.. code-block:: bash

   conda install -c anaconda git

in the conda shell `as they kindly tell us here <https://anaconda.org/anaconda/git>`_. Otherwise you can probably get it `here <https://git-scm.com/download/win>`_.

Then you can simply do

.. code-block:: bash

   pip install git+https://github.com/gboehl/grgrlib
   pip install git+https://github.com/gboehl/econsieve
   pip install git+https://github.com/gboehl/emcwrap
   pip install git+https://github.com/gboehl/pydsge

Maybe you'd have to use ``pip3`` instead. If you run it and it complains about missing packages, please let me know so that I can update the `setup.py`!


Manual installation of the git version via ``pip``
--------------------------------------------------

First, be sure that you are on Python 3.x. Then, get the ``econsieve`` and ``grgrlib`` packages:

* https://github.com/gboehl/grgrlib

* https://github.com/gboehl/econsieve

The simplest way is to clone the repository and then from within the cloned folder run (Windows user from the Anaconda Prompt):

.. code-block:: bash

   pip3 install .


Updating
--------

The package is updated very frequently (find the history of latest commits `here <https://github.com/gboehl/pydsge/commits/master>`_). I hence recommend pulling and reinstalling whenever something is not working right. Best thing you also upgrade the other packages as you are at it:

.. code-block:: bash

   pip install --upgrade git+https://github.com/gboehl/grgrlib
   pip install --upgrade git+https://github.com/gboehl/econsieve
   pip install --upgrade git+https://github.com/gboehl/pydsge
