from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="fastann",
    version="1.0",
    rust_extensions=[RustExtension(
        "fastann.fastann", binding=Binding.PyO3, path="fastann/Cargo.toml")],
    packages=["fastann"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
