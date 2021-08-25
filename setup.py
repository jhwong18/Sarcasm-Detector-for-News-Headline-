from setuptools import setup, find_packages
# read the contents of your README file
from os import path
from os.path import basename
from os.path import splitext
from glob import glob

readme_directory = path.abspath(path.dirname(__file__))
with open(path.join(readme_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(name="sarcasm_detector_news_headline",
      version="0.1.0",
      description="Data Visualization and Prediction of Sarcastic Headlines using Deep Learning models",
      long_description=readme,
      long_description_content_type='text/markdown',
      author="Jing Hui Wong",
      entry_points={
          'console_scripts': [
              'sarcasm_detector_news_headline=main:cli',
          ]
      },
      packages=find_packages(where='src', exclude=['test']),
      package_dir={"": "src"},
      author_email="jinghui.me@gmail.com",
      install_requires=["tensorflow==2.5.1",
                        "pandas",
                        "numpy",
                        "scikit-learn==0.21.3",
                        "matplotlib",
                        "nltk==3.4.5",
                        "spacy==2.2.3",
                        "pyLDAvis==2.1.2",
                        "gensim==3.8.1",
                        "pytest==5.2.2",
                        "pytest-mpl==0.10",
                        "pytest-mock==1.11.2"
                        ],
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      include_package_data=True,
      python_requires=">=3.6"
      )


