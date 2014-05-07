BirDetect
=========

BirDetect is simple program for identifying bird sounds. Program was made as a course project for Speech Technology workshop at University of Eastern Finland. Program was tested and made for specific dataset for learning purposes. For example renaming algorithm is taolored for dataset used in course project. Splitting and identifying parts of the software should work also with other datasets, but to get meaningfull results some modifications may be necessary. 

Dependencies
------------

- Python 2.7
- Bob 1.2.2 or later (http://idiap.github.io/bob/)
- NumPy (http://www.numpy.org/)
- PyYAML (http://pyyaml.org/wiki/PyYAML)
- bunch (https://github.com/dsc/bunch)
- matplotlib (http://matplotlib.org/)
- SciPy (http://www.scipy.org/)

All dependencies can be installed using "pip install" procedure (https://pypi.python.org/pypi).


Usage
-----

**Program consists of three separate parts:**
- Renaming sound files in dataset
- Splitting sound files by silent moments
- Training the detection algorithm, detection of bird species from evaluation sounds and analysing the detection efficiency.


