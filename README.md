# spatial-frequency-stimuli

Contains the code necessary to generate spatial and save stimuli for
measuring spatial frequency preferences using fMRI. This is modified
(and a subset of) my [spatial frequency
preferences](https://github.com/billbrod/spatial-frequency-preferences)
project. See the jupyter notebook included in this directory for a bit
more information on these stimuli.

The created stimuli files can be found on the associated [OSF
page](https://osf.io/k2dv5/).

**UPDATE:** In September 2024, updated so that the polar angle increases as you
go counter-clockwise (to match the standard convention), rather than clockwise.
This is equivalent to changing `w_a -> -w_a`. Thus, reverse spirals now have
`w_a=w_r` and forward spirals have `w_a=-w_r`. See release 2.0.0 of the [spatial
frequency
preferences](https://github.com/billbrod/spatial-frequency-preferences) repo for
more details.

## Usage

- Download and install [miniconda](https://conda.io/miniconda.html)
(for python 3.7).

- Navigate to this directory on your command line.

- Run `conda env create -f environment.yml` to install the requirements (note
  that python 3.6 was used in the paper, but as of Feb 2023, 3.7 is necessary
  for conda to be able to solve the environment).
  
- Run `conda activate sf-stim` to activate the virtual environment you
  just installed.
  
- Run `python stimuli.py -h` to view the docstring.
  
- Run `python stimuli.py 714 8.4` to recreate the files as used by the NSD (and
  thus to match the files on the OSF).
  - `714` and `8.4` specify the diameter in pixels and degrees of visual angle,
    respectively.
  - You can probably trust the defaults for most things.
  
- These will all be located in the (newly-created) `data/stimuli`
  folder under this one. The `.npy` files are for using in Python, the
  `.mat` file (which contains the log-polar stimuli, constant stimuli,
  and anti-aliasing mask size) is for using in Matlab.
  
- The two csv files created (also in that `data/stimuli` directory)
  summarize the properties of the created stimuli.
