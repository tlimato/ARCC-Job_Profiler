ARCC Benchmarking Toolkit
========================

The ARCC Benchmarking Toolkit is a comprehensive project aimed at providing researchers and developers in the field of machine learning (ML) and artificial intelligence (AI) with essential resources and tools to enhance their projects and push the boundaries of their respective domains. In response to the rapid evolution of ML and AI, this toolkit focuses on benchmarking and performance analysis, catering to evolving requirements and ensuring optimal utilization of cutting-edge computing hardware.

Overview
--------

Existing AI benchmarks have their limitations, such as fixed problem sizes and lack of specificity in ML/AI models. The ARCC Benchmarking Toolkit addresses these limitations by offering a purpose-built solution for evaluating ML/AI algorithms. It encompasses a wide range of ML/AI workloads, ensuring scalability across diverse computing platforms, and providing valuable insights into GPU performance under various conditions.

Features
--------

- **Diverse Workloads:** The toolkit covers a diverse array of ML/AI workloads, including natural language processing (NLP) algorithms like BERT, GPT-2, and DNABERT2, image recognition/classification algorithms such as YOLOV8, text-to-speech algorithms like FastPitch, Coqui-ai TTS, Deep Voice 3, and text-to-image conversion.

- **GPU Coverage:** The toolkit supports a variety of GPUs, including A10, A30, A40, A100, RTX A6000, and V100-32gb, representing state-of-the-art CUDA-based computing hardware prevalent in ML/AI workloads.

- **Comprehensive Datasets:** Various datasets, such as OpenWebtext, ThePile, Red Pajamma, Oscar, and Starcoder, cater to different language and machine learning models, providing insights into CPU/GPU performance across a range of ML/AI workloads.

- **Visualization:** The toolkit employs sophisticated statistical analysis to transform raw data into user-friendly visualizations. Crucial computational metrics, including CPU utilization, Disk IOPS, GPU power, and core utilization, are presented comprehensively to offer insights into computing environment performance and ML/AI task efficiency.

Installation and Usage
----------------------

1. **System Requirements:** The toolkit is designed to accommodate both bare metal systems (e.g., SLURM workload manager) and Kubernetes-based high-performance computing (HPC) clusters. Ensure your system meets these requirements.

2. **Clone the Repository:** Clone the ARCC Benchmarking Toolkit repository to your local machine using the following command::

   git clone https://github.com/arcc-benchmarking-toolkit.git

   ==[package not currently published]== Additionally, if you are using an envrionment manager such as miniconda + pip you can simply run ::
   pip install ARCC-Profiler

3. **Setup and Configuration:** Follow the instructions provided in the repository's documentation to set up and configure the toolkit according to your system requirements.
``==[ADD INFO REGARDING FUNCTION CALLS]

4. **Benchmarking Process:** Utilize the toolkit to benchmark your ML/AI algorithms. Refer to the documentation for detailed instructions on how to execute benchmarks and gather performance data.

5. **Visualization and Analysis:** After running benchmarks, explore the generated visualizations and performance metrics. These insights will help you optimize your ML/AI models and identify potential bottlenecks.

Example Insights
----------------

An example observation within the toolkit involved examining Disk IOP performance for GPT-style models. During the Docker Container-based test of GPT2, a 13 percent reduction in IOPS was consistently observed between each epoch. This finding illuminated potential bottlenecks in the Data loading pipeline, offering actionable insights for optimizing future workloads.

Contributing
------------

We welcome contributions from the community to enhance and extend the ARCC Benchmarking Toolkit. Feel free to submit issues, feature requests, or pull requests to help us improve the toolkit's functionality and usability.

Contact
-------

For inquiries and support, please contact our team at benchmarking@arcc.org.

By providing transparent and precise performance data, the ARCC Benchmarking Toolkit aims to empower researchers in making informed GPU selections for specific algorithms, ultimately contributing to impactful research outcomes. This initiative underscores our commitment to advancing ML/AI research and equipping researchers with indispensable tools to drive innovation within the field.
