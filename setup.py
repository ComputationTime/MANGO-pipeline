# wheel = binary packaged version of code thats ready for publication (everything the python manager needs to know to install the package [metadata])
# python setup.py bdistwheel: https://www.youtube.com/watch?v=5KEObONUkik
# pip install wheel 
# JUST NEED TO PUT .PATH IN THE INIT FILE!!
# python setup.py bdist_wheel sdist --> pip install . 
# Need to use '--user' when installing on the DSAI cluster
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='mango',
    version='0.1.0',
    description = 'Mulimodal ANtiGen Optimized Transformer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Donovan Vincent Jr.',
    maintainer='Donovan Vincent Jr.',
    maintainer_email='dvincen9@jh.edu',
    package_data={'':['utils/GPT_Vocab.txt']},
    include_package_data=True,
    install_requires=[
        'einops', #0.8.0
        'torch>1.9',
        'numpy>1.9.0',
        'transformers>4.5',
        'datasets>3.5.1',
        'accelerate>=0.26.0',
        'seaborn>=0.13.2',
        'ablang2',
        'fair-esm', # For all ESM2 models
        'scipy', #1.16.0 -> Need for dist matrices
        'Bio', #Should install Bio and Biopython
    ]

)