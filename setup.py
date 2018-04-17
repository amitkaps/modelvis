from setuptools import setup
import re
import os

def get_version():
    """Reads the version from modelvis.py.
    """
    root = os.path.dirname(__file__)
    version_path = os.path.join(root, "modelvis.py")
    text = open(version_path).read()
    rx = re.compile("^__version__ = '(.*)'", re.M)
    m = rx.search(text)
    version = m.group(1)
    return version

version = get_version()

setup(
  name="modelvis",
  version=version,
  description="Visualising Machine Learning Models",
  license="MIT",
  install_requires=[
    'numpy',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'requests',
    'seaborn'
  ],
  py_modules=['modelvis'],
  author="Amit Kapoor",
  author_email="amitkaps@gmail.com",
  platforms="any"
)
