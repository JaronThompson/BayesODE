from setuptools import setup, find_packages

setup(
    name="bayes_ode",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "jax",
        "jaxlib"
    ]
)
