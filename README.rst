
pydsge
======

.. image:: https://badge.fury.io/py/pydsge.svg
    :target: https://badge.fury.io/py/pydsge

.. image:: https://github.com/gboehl/pydsge/workflows/Continuous%20Integration%20Workflow/badge.svg?branch=master
    :target: https://github.com/gboehl/pydsge/actions?query=branch%3Aimplementing_CI

----

A package for solving, filtering and estimating **linear** DSGE models with the ZLB (or other occasionally binding constraints).

Check out my `Econpizza <https://github.com/gboehl/econpizza>`_ package if you are interested in simulating **nonlinear** DSGE models with (or without) **heterogeneous agents**.

A collection of models that can be (and were) used with this package can be found in `another repo <https://github.com/gboehl/projectlib/tree/master/yamls>`_.

Installation
-------------

Installing the stable version is as simple as typing

.. code-block:: bash

   pip install pydsge

in your terminal (Linux/MacOS) or Anaconda Prompt (Win). 

Documentation
-------------

Documentation can be found on `ReadTheDocs <https://pydsge.readthedocs.io/en/latest/index.html>`_:

- `Installation Guide <https://pydsge.readthedocs.io/en/latest/installation_guide.html>`_
- `Getting Started <https://pydsge.readthedocs.io/en/latest/getting_started.html>`_
- `Module Documentation <https://pydsge.readthedocs.io/en/latest/modules.html>`_

Citation
--------

**pydsge** is developed by Gregor Boehl to simulate, filter, and estimate DSGE models with the zero lower bound on nominal interest rates in various applications (see `my website <https://gregorboehl.com>`_ for research papers using the package). Please cite it with

.. code-block:: latex

    @TechReport{boehl2022meth,
      title = {{Estimation of DSGE Models with the Effective Lower Bound}},
      author = {Boehl, Gregor and Strobel, Felix},
      year = 2022,
      type = {CRC 224 Discussion Papers},
      institution = {University of Bonn and University of Mannheim, Germany}
    }

.. code-block:: latex

    @techreport{boehl2022obc,
      title = Efficient solution and computation of models with occasionally binding constraints},
      author = Boehl, Gregor},
      journal = Journal of Economic Dynamics and Control},
      volume = 143},
      pages = 104523},
      year = 2022},
      publisher = Elsevier}
    }


We appreciate citations for **pydsge** because it helps us to find out how people have
been using the package and it motivates further work.


Parser
------

The parser originally was a fork of Ed Herbst's fork from Pablo Winant's (excellent) package **dolo**. 

See https://github.com/EconForge/dolo and https://github.com/eph.


References
----------

Boehl, Gregor (2022). `Efficient Solution and Computation of Models with Occasionally Binding Constraints <http://gregorboehl.com/live/obc_boehl.pdf>`_. *Journal of Economic Dynamics and Control*
