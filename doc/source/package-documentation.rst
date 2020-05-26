Package Documentation
=====================

Magma
-----
.. autoclass:: neural_semigroups.Magma
   :special-members: __init__
   :members:

Cyclic Group
------------
.. autoclass:: neural_semigroups.CyclicGroup
   :special-members: __init__

.. _cayley-database:

Cayley Database
---------------
.. autoclass:: neural_semigroups.CayleyDatabase
   :special-members: __init__
   :members:
      
Denoising Autoencoder for Magmas
--------------------------------
.. autoclass:: neural_semigroups.MagmaDAE
   :special-members: __init__
   :members:

Associator Loss
---------------
.. autoclass:: neural_semigroups.AssociatorLoss
   :special-members: __init__
   :members:

utils
-----
.. currentmodule:: neural_semigroups.utils

A collection of different functions used by other modules.

.. autofunction:: random_semigroup
.. autofunction:: check_filename
.. autofunction:: check_smallsemi_filename
.. autofunction:: get_magma_by_index
.. autofunction:: import_smallsemi_format
.. autofunction:: get_equivalent_magmas
.. autofunction:: download_file_from_url
.. autofunction:: download_smallsemi_data
.. autofunction:: print_report
.. autofunction:: get_newest_file
.. autofunction:: get_two_indices_per_sample
.. autofunction:: make_discrete
.. autofunction:: load_data_and_labels_from_file
.. autofunction:: load_data_and_labels_from_smallsemi
