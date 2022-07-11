import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DQM μDC NMF",
    version="0.0.1",
    author="Martín Alcalde Martínez",
    author_email="uo269426@uniovi.es",
    description="Un paquete para la certificación de los muones producidos en el experimento CMS del LHC aplicando el método NMF.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/magnarex/uo269426-tfg",
    # project_urls={
    #     "Bug Tracker": "https://github.com/magnarex/uo269426-tfg/issues",
    # },
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)