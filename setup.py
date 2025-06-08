from setuptools import setup, find_packages

setup(
    name="ai-clipper",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "yt-dlp>=2024.3.10",
        "faster-whisper>=0.10.0",
        "moviepy>=1.0.3",
        "ffmpeg-python>=0.2.0",
        "flask>=3.0.0",
        "sentence-transformers>=2.5.1",
        "bertopic>=0.16.0",
        "scikit-learn>=1.4.0",
        "nltk>=3.8.1",
        "torch>=2.2.0",
        "numpy>=1.24.0",
    ],
) 