from setuptools import setup

setup(
    name='ARCC_Profiler',
    version='1.0.0',
    packages=[''],
    url='https://github.com/WyoARCC/GPU_benchmarking_toolkit_for_ML/tree/main',
    license='MIT',
    author='Tyson Limato',
    author_email='tlimato@uwyo.edu',
    description='The ARCC benchmarking toolkit is designed to accommodate both bare metal systems (like SLURM workload manager) and Kubernetes-based high-performance computing (HPC) clusters, aligning with the dataset structures of open-source repositories like HuggingFace. It spans an array of GPUs, including A10, A30, A40, A100, RTX A6000, and V100-32gb, which are prevalent in ML/AI workloads, representing cutting-edge CUDA-based computing hardware. Furthermore, the toolkit covers a wide spectrum of ML/AI methods and algorithms, including natural language processing (NLP) algorithms like BERT [5], GPT-2 [12], DNABERT2 [16], image recognition/classification algorithms such as YOLOV8 [1], text-to-speech algorithms like FastPitch [17], Coqui-ai TTS [6], Deep Voice 3 [11], and text-to-image conversion. A multitude of datasets, such as OpenWebtext [2], ThePile [7], Red Pajamma [4], Oscar [10], and Starcoder [9], support various LLMs, facilitating a profound comprehension of CPU/GPU performance across distinct ML/AI workloads.'
)
