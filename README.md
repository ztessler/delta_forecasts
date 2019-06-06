# Land subsidence and sea level rise scenario modeling in deltas

## See paper [Tessler et al., 2018, Geomorphology](https://www.sciencedirect.com/science/article/pii/S0169555X1730209X)

Estimates of relative sea level rise for a global suite of deltas under a range
of environmental, climatic, and anthropogenic scenarios.

![Global deltas modeled](figures/global_delta_map.png?raw=true)

This code runs through SCons and generates all the figures in the `figures/`
directory. There are many hardcoded paths in config.py that would need to be
updated from those on my machine. Input datasets referenced there will need to
be downloaded and placed in the correct locations.

SCons runs by reading a series of scripts (`SConstruct`, and several
`SConscript` files), using them to create a directed acyclic graph connecting
all output files to their dependencies. When you run `scons <target>` it
calculates which dependencies for that target are out of date, and re-generates
only those before generating the target. Similar to `make`, but a bit smarter
(uses MD5 signatures rather than just timestamps) and the script is in Python.
Additionally, actions to build targets can be defined directly in Python.

`SConstruct` is the entrypoint to the code, which calls `SConscript` files from
various subdirectories.

We also define a suite of environmental scenarios that modify geophysical,
climatic, or anthropogenic conditions, and rerun all figures. Scenario
definitions are in the `experiments` dict in config.py. Each experiment has a
parent reference that it inherits any non-specified configuration from, as well
as specific "partner" scenarios against which it's compared (see figures in
`figures/joint`). See `figures/joint/pristine_contemp` for an example.

