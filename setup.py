from setuptools import setup

setup(
  name="modelvis",
  version="0.1.3",
  description="Visualising Machine Learning Model",
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
