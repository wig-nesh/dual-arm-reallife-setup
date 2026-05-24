from setuptools import setup, find_packages

setup(
    name="bimanual_grasping",
    version="0.1",

    packages=find_packages(),

    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "openai",
        "matplotlib",
        "scipy",
        "tqdm"
    ]
)