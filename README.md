# spatial-frequency-stimuli

Contains the code necessary to generate spatial and save stimuli for
measuring spatial frequency preferences using fMRI. This is modified
(and a subset of) my [spatial frequency
preferences](https://github.com/billbrod/spatial-frequency-preferences)
project. See the jupyter notebook included in this directory for a bit
more information on these stimuli.

The created stimuli files can be found on the associated [OSF
page](https://osf.io/k2dv5/).

## Usage

- Download and install [miniconda](https://conda.io/miniconda.html)
(for python 3.7).

- Navigate to this directory on your command line.

- Run `conda env create -f environment.yml` to install the
  requirements.
  
- Run `conda activate sf-stim` to activate the virtual environment you
  just installed.
  
- Run `python stimuli.py -h` to view the docstring.
  
- Run `python stimuli.py`. You can probably trust the defaults for
  most things.
  
- These will all be located in the (newly-created) `data/stimuli`
  folder under this one. The `.npy` files are for using in Python, the
  `.mat` file (which contains the log-polar stimuli, constant stimuli,
  and anti-aliasing mask size) is for using in Matlab.
  
- The two csv files created (also in that `data/stimuli` directory)
  summarize the properties of the created stimuli.
