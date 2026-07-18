from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-plumbing",
    version="0.0.1",
    author="Quantum Plumbing Team",
    author_email="Mariussielcken@gmail.com",
    description="Hypothetical thinking space for AI via quantum-aligned architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Meester-Mus/quantum-plumbing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "quantum": ["qiskit>=0.36.0"],
        "dev": ["pytest>=6.0.0", "black>=21.0", "flake8>=3.9.0"],
    },
    entry_points={
        "console_scripts": [],
    },
)