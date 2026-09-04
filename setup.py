from setuptools import find_packages, setup

setup(
    name="ai_twin_project",
    version="1.0.0",
    description="Smart AI Twin - Behavioral Analysis, Sentiment NLP & Screen Time Forecasting",
    author="Jatin Sharma",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "matplotlib",
        "seaborn",
        "streamlit",
        "fastapi",
        "uvicorn",
        "pydantic",
        "plotly",
        "pyyaml"
    ],
)
