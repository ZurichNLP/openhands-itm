Installation
============

This section describes how to install 👐OpenHands and ensure it is working.

Getting the toolkit
-------------------

- Ensure you have Python 3.10+ installed.
- 👐OpenHands can either be installed directly using package manager, or built from source.

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^

Run the following command in your terminal:

.. code:: console

	$ pip install --upgrade OpenHands

Building from latest source
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following commands:

.. code:: console

	$ git clone https://github.com/AI4Bharat/OpenHands
	$ cd OpenHands/
	$ pip install .


Checking Installation
---------------------

.. code:: python

	import openhands
	print(openhands.__version__)

This should successfully display the version of the installed 👐OpenHands library version.
