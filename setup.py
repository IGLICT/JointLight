from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).parent
requirements = [line.strip() for line in (root / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
setup(
    name="jointlight",
    version="0.1.0",
    description="JointLight video relighting training and inference",
    long_description=(root / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["diffsynth", "diffsynth.*", "utils", "utils.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
)
