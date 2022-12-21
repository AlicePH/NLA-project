from setuptools import setup

setup(name='NLA project',
      author='JAXteam',
      description="""This is the code for the efficient compression of neural network layers.""",
      install_requires=[
            "numpy >= 1.16.2",
            "scipy >= 1.2.1",
            "torch >= 1.0.1",
            "torchvision >= 0.2.2",
            "scikit-learn >= 0.20.2"
        ],
      long_description=open('README.md').read(),
      )
