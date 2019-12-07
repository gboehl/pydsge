Getting Started
===============

This section gives an overview over parsing, simulating and filtering models using **pydsge**. It also explains how to load and process data from an estimation. 

Parsing & simulating
--------------------

Let us first import the base class and the example model:

.. code-block:: python

    from pydsge import DSGE, example
    yaml_file, data_file = example

The ``example`` here is nothing than a tuple containing the paths to two files. The first file to the example model file (as a string):

.. code-block:: python

    print(yaml_file)

.. code::

    /home/gboehl/repos/pydsge/pydsge/dfi.yaml

You can use your text editor of choice (which I hope is not Notepad) to open this file and have a look. It contains useful annotations and comments.

Parsing ``*.mod``-files
^^^^^^^^^^^^^^^^^^^^^^^^

So lets parse this thing:

.. code-block:: python

    mod = DSGE.read(yaml_file)

Of course, if you would want to parse your own ``*.yaml`` model, you could easily do that by:

.. code-block:: python

    yaml_path = "/full/path/to/your/model.yaml"
    mod = DSGE.read(yaml_path)

But, lets for now assume you are working with ``dfi.yaml``.

The ``mod`` object is now an instance of the `DSGE` class. 
Lets load the calibrated parameters from the file (they are loaded by default anyways, but for educative purposes...) and instantize the transition function:

.. code-block:: python

    par = mod.set_par('calib')
    mod.get_sys(par, reduce_sys=False, verbose=True)

You should have a look at the functions `set_par` and `get_par` if you want to experiment with the model. 

The ``DSGE``-instance has many attributes that concentrate information and functions of the model. They are all listed in the `module documentation <https://pydsge.readthedocs.io/en/latest/modules.html#>`_.

Simulate IRFs
^^^^^^^^^^^^^

Let us use this to simulate a series of impulse responses:

.. code-block:: python

    shock_list = ('e_u', 5, 1) # (name, size, period)
    X1, (L1, K1) = mod.irfs(shock_list, verbose=True)

Nice. For details see the ``irfs`` function. Lets plot it using the ``pplot`` plot function from the ``grgrlib`` library:

.. code-block:: python

    from grgrlib import pplot
    pplot(X1)

Btw, the ``L1, K1`` arrays contain the series of expected durations to/at the ZLB.


Sample from prior
^^^^^^^^^^^^^^^^^

Now lets assume that you have specified priors and wanted to know how flexible your model is in terms of impulse responses. The ``get_par`` function also allows sampling from the prior:

.. code-block:: python

    par0 = mod.get_par('prior', nsample=50, verbose=True)
    print(par0.shape)

.. code::

    (50, 11)

This is an array with 50 samples of the 10-dimensional parameter vector. 
If you allow for ``verbose=True`` (which is the default) the function will also tell you how much of your prior is not implicitely trunkated by indetermined or explosive regions. 

Lets feed these parameters ``par0`` into our ``irfs()`` and plot it:

.. code-block:: python

    X1, (L1, K1) = mod.irfs(shock_list, par0, verbose=True)
    pplot(X1, labels=mod.vv)

This gives you an idea on how tight your priors are. 


Filtering & smoothing
---------------------

This section treats how to load data, and do Bayesian filtering given a DSGE model.

Load data
^^^^^^^^^

We have just seen how to parse the model. Parsing the data is likewise quite easy. It however assumes that you managed to put your data into pandas' ``DataFrame`` format. pandas knows many ways of loading your data file into a ``DataFrame``, see for example `here <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`_ on how to load a common ``*.csv`` file. 

There is some `preliminary documentation <https://pydsge.readthedocs.io/en/latest/index.html>`_ out there.

Luckily I already prepared an example data file that is already well structured:

.. code-block:: python

    yaml_file, data_file = example
    print(data_file)

Again, this is just the path to a file that you can open and explore. I constructed the file such that I can already load the column ``data`` as a ``DateTimeIndex``, which makes things easier:

.. code-block:: python

    import pandas as pd

    df = pd.read_csv(data_file, parse_dates=['date'], index_col=['date'])
    df.index.freq = 'Q' # let pandas know that this is quartely data
    print(df)

.. code::

    ...

This should give you an idea of how the data looks like. The frame contains the time series of US output growth, inflation, and the FFR from 1995Q1 to 2018Q1.
Let us load this into **pydsge** and combine it with the model we defined above:

.. code-block:: python

    mod.load_data(df)

This automatically selects the obsevables you defined in the ``*.yaml`` and puts them in the ``mod.data`` object. Note that it will complain if it can't find these observables or if they are named differently. So, that's all we want from now.


Run filter
^^^^^^^^^^

We now want to use a Bayesian Filter to smooth out the hidden states of the model. As the example data sample contains the Zero-lower bound period and the solution method is able to deal with that, we should use a nonlinear filter such as the *Transposed Ensemble Kalman Filter (TEnKF)*. This filter is a hybrid between the Kalman Filter and the Particle Filter, we hence have to define the number of particles. For small problems as the one here, a smaller number would be sufficient, but since everything goes so fast, let us chose 500:

.. code-block:: python

    mod.create_filter(N=500, ftype='TEnKF')

The **TEnKF** is the default filter, so specifying ``ftype`` would not even have been necessary. The filter got most of the necessary information (innovation covariance, observation function etc) from the ``*.yaml``. What remains to be specified is the measurement noise. The covariance matrix of the measurement errors are stored as ``mod.filter.R``. Luckily, there is a function that creates a diagonal matrix with its diagonal equal to the fraction `a` of the standard deviation of the respective time series, as it is frequently done:

.. code-block:: python

    mod.filter.R = mod.create_obs_cov(2e-1)

Here, `a=2e-1`. As one last thing before running the filter, we would like to set the ME of the FFR very low as this can be measured directly (note that we can not set it to zero due to numerical reasons, but we can set it sufficiently close).

.. code-block:: python

    # lets get the index of the FFR
    ind = mod.observables.index('FFR')
    # set ME of the FFR to very small value
    mod.filter.R[ind,ind] = 1e-4 


``mod.observables`` contains all the observables. See the `module documentation <https://pydsge.readthedocs.io/en/latest/modules.html#>`_ for more useful class variables. But lets start the filter already!

.. code-block:: python

    FX = mod.run_filter(verbose=True, smoother=True)

``smoother=True`` also directly runs the TEnKF-RTS-Smoother. ``FX`` now contains the states. Lets have a look:

.. code-block:: python

    pplot(FX, mod.data.index, labels=mod.vv)

We can also have a look at the implied observables. The function ``mod.obs()``
is the observation function, implemented to work on particle clouds (such as ``FX``):

.. code-block:: python

    FZ = mod.obs(FX)
    pplot((mod.Z, FZ), mod.data.index, labels=mod.observables)


Note that these particles/ensemble members/"dots" yet do not *fully* obey the nonlinearity of the transition function but contain approximation errors. To get rid of those we need adjustment smoothing.


Adjustment smoothing
--------------------

[TBD]


Simulating counterfactuals
--------------------------

[TBD]


Processing estimation results
-----------------------------

[TBD]

[TODO: set non-standard paths]

[TODO: obtain estimation stats]

[TODO: irfs with postrior draws]

[TODO: filter with postrior draws]

[TODO: counterfactuals with postrior draws]

[TODO: document at all funcs in DSGE module]
