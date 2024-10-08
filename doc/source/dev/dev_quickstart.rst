.. _dev-quickstart:

============================
Contributor quickstart guide
============================

After :ref:`getting the source code from GitHub <git-start>`, there are three
steps to start contributing:

1. **Set up a development environment**

   Using ``mamba``, or some flavor of the many virtual environment management
   tools, you can make sure the development version of SciPy does not interfere
   with any other local installations of SciPy on your machine.

2. **Build SciPy**

   SciPy uses compiled code for speed, which means you might need extra
   dependencies to complete this step depending on your system - see
   :ref:`building-from-source`.

3. **Perform development tasks**

   These can include any changes you want to make to the source code, running
   tests, building the documentation, running benchmarks, etc.

Basic workflow
==============

.. note::

    We **strongly** recommend using a user-activated environment setup, such as
    a conda or virtual environment.

Since SciPy contains parts written in C, C++, and Fortran that need to be
compiled before use, make sure you have the necessary compilers and Python
development headers installed. If you are using ``mamba``, these will be
installed automatically. If you are using ``pip``, check which
:ref:`system-level dependencies <system-level>` you might need.

First, fork a copy of the main SciPy repository in GitHub onto your own
account and then create your local repository via::

    git clone git@github.com:YOURUSERNAME/scipy.git scipy
    cd scipy
    git submodule update --init
    git remote add upstream https://github.com/scipy/scipy.git

Next, set up your development environment. **With**
:ref:`system-level dependencies <system-level>` **installed**, execute the
instructions in :ref:`building-from-source`.

For details on how to test your changes, see the more complete setup
walkthrough in :ref:`development-workflow`.

Other workflows
===============

There are many possible ways to set up your development environment.
For more detailed instructions, see the :ref:`contributor-toc`.

.. note::

    If you are having trouble building SciPy from source or setting up your
    local development environment, you can try to build SciPy with GitHub
    Codespaces. It allows you to create the correct development environment
    right in your browser, reducing the need to install local development
    environments and deal with incompatible dependencies.

    If you have good internet connectivity and want a temporary set-up, it is
    often faster to work on SciPy in a Codespaces environment. For
    documentation on how to get started with Codespaces, see
    `the Codespaces docs <https://docs.github.com/en/codespaces>`__.
    When creating a codespace for the ``scipy/scipy`` repository, the default
    2-core machine type works; 4-core will build and work a bit faster (but of
    course at a cost of halving your number of free usage hours). Once your
    codespace has started, you can run ``conda activate scipy-dev`` and your
    development environment is completely set up - you can then follow the
    relevant parts of the SciPy documentation to build, test, develop, write
    docs, and contribute to SciPy.

    Another alternative is to use `Gitpod <https://www.gitpod.io>`__.
    We do not maintain this solution anymore but some information can be found
    in previous versions of our
    `docs <https://docs.scipy.org/doc/scipy-1.10.1/dev/contributor/quickstart_gitpod.html>`__.
