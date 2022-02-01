
pydsge
======

.. image:: https://github.com/pcschreiber1/pydsge_OSE_Project_Fork/workflows/Continuous%20Integration%20Workflow/badge.svg?branch=master
    :target: https://github.com/pcschreiber1/pydsge_OSE_Project_Fork/actions?query=branch%3Aimplementing_CI

----

Contains the functions and classes for solving, filtering and estimating DSGE models at the ZLB (or with other occasionally binding constraints).

A collection of models that can (and were) used with this package can be found in `another repo <https://github.com/gboehl/projectlib/tree/master/yamls>`_.

The code is in alpha state and provided for reasons of collaboration, replicability and code sharing in the spirit of open science. It does not (and for now, can not) have a toolbox character. The code is operational, but (yet) not ready for public use and I can not provide any support. You are however very welcome to get in touch if you are interested working with the package.

Installation
-------------

Installing the repo version is as simple as

.. code-block:: bash

   pip install pydsge

Documentation
-------------

There is some `preliminary documentation <https://pydsge.readthedocs.io/en/latest/index.html>`_ out there.

- `Installation Guide <https://pydsge.readthedocs.io/en/latest/installation_guide.html>`_
- `Getting Started <https://pydsge.readthedocs.io/en/latest/getting_started.html>`_
- `Module Documentation <https://pydsge.readthedocs.io/en/latest/modules.html>`_

Citation
--------

**pydsge** is developed by Gregor Boehl to simulate, filter, and estimate DSGE models with the zero lower bound on nominal interest rates in various applications (see `my website <https://gregorboehl.com>`_ for research papers using the package). Please cite it with

.. code-block::

    @techreport{boehl2021method,
      Title = {Efficient Solution and Computation of Models with Occasionally Binding Constraints},
      Author = {Gregor Boehl},
      Year = {2021},
      institution = {Goethe University Frankfurt, Institute for Monetary and Financial Stability (IMFS)},
      type = {IMFS Working Paper Series},
      number = {148},
      url = {https://gregorboehl.com/live/obc_boehl.pdf},
    }

We appreciate citations for **pydsge** because it helps us to find out how people have
been using the package and it motivates further work.


Parser
------

The parser originally was a fork of Ed Herbst's fork from Pablo Winant's (excellent) package **dolo**. 

See https://github.com/EconForge/dolo and https://github.com/eph.


References
----------

Boehl, Gregor (2021). `Efficient Solution and Computation of Models with Occasionally Binding Constraints <http://gregorboehl.com/live/obc_boehl.pdf>`_. *IMFS Working Paper*
