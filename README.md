# Land subsidence and sea level rise scenario modeling in deltas

## See paper [Tessler et al., 2018, Geomorphology](https://www.sciencedirect.com/science/article/pii/S0169555X1730209X)

This code runs through SCons and generates all the figures in the `figures/`
directory. There are many hardcoded paths in config.py that would need to be
edited. Input datasets referenced there will need to be downloaded and placed in
the correct locations.

Scenario definitions are in the `experiments` dict in config.py. Each
experiment has a parent reference that it inherits any non-specified
configuration from.
