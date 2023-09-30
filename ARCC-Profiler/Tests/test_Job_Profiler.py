# Author: Tyson Limato
# project: GPU Benchmarking
# Purpose: Testing Library Functions
# Start Date: 6/28/2023

# Modules Used for Testing
import unittest
import time
import os
import csv
from io import StringIO
from contextlib import redirect_stdout
from unittest.mock import patch
# Module That is being Tested
import Plot_nonGPU_functions

# REQUIRES VALIDATION DISCUSS WITH AIJUN
class TestMeasureTime(unittest.TestCase):
    @patch('sys.stdout', new_callable=StringIO)
    def test_measure_time(self, mock_stdout):
        # Define a function to be measured
        def test_function():
            time.sleep(5)  # Simulate some time-consuming operation

        expected_output = "Execution time: 1.00 seconds\n"

        with patch('time.time', side_effect=[0, 1]):
            Plot_nonGPU_functions.measure_time(test_function)

        actual_output = mock_stdout.getvalue()
        self.assertEqual(actual_output, expected_output)

#  REQUIRES VALIDATION DISCUSS WITH AIJUN
class TestRunInThread(unittest.TestCase):
    def test_run_in_thread(self):
        # Define a function to be wrapped by the decorator
        def test_function():
            time.sleep(5)  # Simulate some time-consuming operation

        # Use the decorator to create a threaded version of the function
        threaded_function = Plot_nonGPU_functions.run_in_thread(test_function)
        # Start the threaded function
        threaded_function()
        # Sleep for a while to allow the threaded function to complete
        time.sleep(2)

        # Assert that the threaded function has completed within a reasonable time
        self.assertFalse(threaded_function.is_alive())

#  REQUIRES VALIDATION DISCUSS WITH AIJUN
class TestBASE_DataCollection(unittest.TestCase):
    def setUp(self):
        self.log_file = "test_CPU_RAM_Utilization.csv"

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    @patch('time.sleep', side_effect=lambda _: None)  # Mock time.sleep to avoid actual sleep
    def test_monitor_system_utilization(self, mock_sleep):
        with redirect_stdout(StringIO()) as output:
            data_collection = Plot_nonGPU_functions.BASE_DataCollection(custom_log_file=self.log_file, cpu_time_interval=0.1, autoclean=True)
            data_collection.monitor_system_utilization()

        self.assertTrue(os.path.exists(self.log_file))
        self.assertGreater(os.path.getsize(self.log_file), 0)

        # Verify that the CSV file has the expected headers
        with open(self.log_file, mode='r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            self.assertEqual(headers, ['Core Time', 'CPU Utilization', 'Thread Count', 'RAM Utilization (%)', 'RAM Utilization (MB)'])


"""TO DO: THESE ALL MUST BE DEVELOPED PRIOR TO Ver:1.0.0"""
# Write Unit Test for class ML_DataCollection(BASE_DataCollection):
class TestBASE_ML_DataCollection(unittest.TestCase):
    def setUp(self):
        print("CHANGEME")

    """TEST THIS METHOD OF THE CLASS"""
    @staticmethod  # Make more transparent to the user
    def start_batch():
        start = time.time()
        return start
    """TEST THIS METHOD OF THE CLASS"""
    @staticmethod  # Make more transparent to the user
    def end_batch():
        end = time.time()
        return end
    """TEST __del__(self)"""
        #--------
    """TEST def training_loop_performance(self, epoch: int, batch_num: int,
                                  training_losses: list or int, batch_start_time=None, batch_end_time=None):"""
    
# Write Unit Test for class ML_DataCollection(BASE_DataCollection):
class TestBASE_DiskIOPSMonitor(unittest.TestCase):
    def setUp(self):
        print("CHANGEME")

    """TEST def track_iops(self):"""

    """TEST def start_tracking(self):"""

    """TEST def stop_tracking(self):"""

# Write Unit Test for class CPUMonitor:
class TestBASE_CPUMonitor(unittest.TestCase):
    def setUp(self):
        print("CHANGEME")
    """TEST def monitor_cpu_utilization(self):"""

# Write Unit Test for class MemoryMonitor:
class TestBASE_MemoryMonitor(unittest.TestCase):
    def setUp(self):
        print("CHANGEME")
    """TEST def monitor_memory_utilization(self):"""

# Write Unit Test for class CPUThreadsMonitor:
class TestBASE_CPUThreadsMonitor(unittest.TestCase):
    def setUp(self):
        print("CHANGEME")
    """TEST def def monitor_cpu_threads(self):"""

# The Biggest UNDERTAKING ESPECIALLY GIVE THE GUI AND UX ARE IN FLUX
class TestBASE_DataPlotter(unittest.TestCase):
    def setUp(self):
        print("CHANGEME")

    """TEST @measure_time
    def plot_cpu_utilization(self, Source_file):"""

    """TEST @measure_time
    def plot_thread_count(self, Source_file):"""

    """TEST @measure_time
    def plot_ram_utilization_percent(self, Source_file):"""

    """TEST @measure_time
    def plot_training_loss(self, Source_file):"""

    """TEST @measure_time
    def plot_throughput(self, Source_file):"""

    """TEST  @measure_time
    def plot_disk_iops(self, Source_file):"""

    """TEST @measure_time
    def plot_disk_read_iops(self, Source_file):"""

    """TEST @measure_time
    def plot_disk_write_iops(self, Source_file):"""

    """TEST @measure_time
    def plot_GPU(self, interval=25, height=5, width=10, combine_graphs=False):"""

    """TEST     @measure_time
    def generate_all_graphs(self, CPU_RAM_Data="CPU_RAM_Utilization.csv", Training_Results="training_results.csv"):"""

# The most important class to insure data integrity in an HPC envrionment
class TestBASE_DataPlotter(unittest.TestCase):
    def setUp(self):
        print("CHANGEME")

    