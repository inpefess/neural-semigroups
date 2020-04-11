More about data
---------------

This package uses data included into a ``smallsemi`` package for GAP system. One can download it from `the GAP page`_.

Data is downloaded and saved automatically. By default it resides in a directory ``$HOME/neural-semigroups-data/smallsemi-data/``. Ona can change this using a ``data_path`` argument of a :ref:`Cayley database<cayley-database>`.

To get some data, just: ::

  from neural_semigroups import CayleyDatabase

  cayley_db = CayleyDatabase(4)


.. _the GAP page: https://www.gap-system.org/pub/gap/gap4/tar.gz/packages/smallsemi-0.6.12.tar.gz
