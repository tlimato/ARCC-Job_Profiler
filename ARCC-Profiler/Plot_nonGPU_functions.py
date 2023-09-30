# Author: Tyson Limato
# project: GPU Benchmarking
# Purpose: Seamless visualization of Non-GPU statistics
# Start Date: 6/28/2023
# file Name: Plot_nonGPU_functions.py
import argparse
import multiprocessing
# External Library Error Handling

try:
    # noinspection PyUnresolvedReferences
    import csv
except ImportError:
    print("Unable to import csv")
try:
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
except ImportError:
    print("Unable to import matplotlib")
try:
    # noinspection PyUnresolvedReferences
    import numpy as np
except ImportError:
    print("Unable to import numpy")
try:
    # noinspection PyUnresolvedReferences
    import time
except ImportError:
    print("Unable to import time")
try:
    from datetime import datetime
except ImportError:
    print("Unable to import datetime")
try:
    from tqdm import tqdm
except ImportError:
    print("Unable to import tqdm")
try:
    # noinspection PyUnresolvedReferences
    import warnings
except ImportError:
    print("Unable to import warnings")
try:
    # noinspection PyUnresolvedReferences
    import os
except ImportError:
    print("Unable to import os")
# noinspection PyBroadException
try:
    import psutil
except:
    print("Unable to import psutil")
# noinspection PyBroadException
try:
    import sys
except:
    print("Unable to import sys")
# noinspection PyBroadException
try:
    import threading
except:
    print("Unable to import threading")
# Joshua's GPU Plotter for GPU Stat Log
from utilization_plot import GPU_Plot_Runtime


def measure_time(func):
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


def run_in_thread(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func,
                                  args=args, kwargs=kwargs)
        thread.start()

    return wrapper


class BASE_DataCollection:
    def __init__(self, custom_log_file="custom_log.csv", cpu_ram_output_file='CPU_RAM_Utilization.csv',
                 cpu_time_interval=1, autoclean=False):
        self.CPU_RAM_File = cpu_ram_output_file
        self.CPU_Time_Interval = cpu_time_interval
        self.tracking_results_file = custom_log_file
        self.auto_clean = autoclean
        # Update disk IOPS counters
        self.disk_io_counters = psutil.disk_io_counters()
        self.disk_read_count = 0
        self.disk_write_count = 0

    @run_in_thread
    def monitor_system_utilization(self):
        """
        Monitor System Utilization

        This function monitors the CPU utilization, RAM utilization, and core utilization of the system until a keyboard
        interrupt (Ctrl+C) is triggered.
        """
        file_path = self.CPU_RAM_File
        with open(file_path, mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(
                ['Core Time', 'CPU Utilization', 'Thread Count', 'RAM Utilization (%)', 'RAM Utilization (MB)'])

        try:
            core_time = 0
            while True:
                # Get CPU utilization
                cpu_percent = psutil.cpu_percent(interval=self.CPU_Time_Interval)

                # Get RAM utilization
                ram = psutil.virtual_memory()
                ram_percent = ram.percent
                ram_mb = ram.used / (1024 * 1024)  # Convert bytes to megabytes (MB)

                # Get thread count
                threads = threading.active_count()

                core_time += self.CPU_Time_Interval
                with open(file_path, mode='a', newline='') as results_file:
                    writer = csv.writer(results_file)
                    writer.writerow([core_time, cpu_percent, threads, ram_percent, ram_mb])

                # Sleep for the specified interval
                time.sleep(self.CPU_Time_Interval)

        except KeyboardInterrupt:
            # Handle exception and log error message
            if self.auto_clean is False:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'False' by Default, CPU & RAM Data Collection Aborted.\n"
                    "Make sure you delete old Data files before rerunning code"
                )
                warnings.warn(warning_message, UserWarning)
            else:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'True', CPU & RAM Data Collection Aborted.\n"
                    "Old Data Files have been Deleted"
                )
                warnings.warn(warning_message, UserWarning)
                try:
                    os.remove(file_path)
                    print(f"File '{file_path}' deleted successfully.")
                except FileNotFoundError:
                    print(f"File '{file_path}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            # Close the CSV file properly
            sys.exit(1)


class ML_DataCollection(BASE_DataCollection):
    def __init__(self, training_output_file='training_results.csv', cpu_ram_output_file='CPU_RAM_Utilization.csv',
                 cpu_time_interval=1, batch_size=None, autoclean=False):
        # Call Parent __init__
        # noinspection PyTypeChecker
        super().__init__(cpu_ram_output_file, cpu_time_interval, autoclean)
        self.training_results_file = training_output_file
        if batch_size is None:
            warning_message = (
                "Warning: You should provide a batch size when instantiating the class.\n"
                "Example usage: obj = YourClass(batch_size=64)"
            )
            warnings.warn(warning_message, UserWarning)
            exit()
        else:
            self.dataloader_batch_size = batch_size

        self.batch_time = None
        self.throughput = None
        with open(self.training_results_file, mode='w', newline='') as r_file:
            writer = csv.writer(r_file)
            writer.writerow(['Epoch',
                             'Batch',
                             'Training Loss',
                             'Time',
                             'Throughput (Seq/sec)',
                             'Disk Read IOPS',
                             'Disk Write IOPS'])

    @staticmethod  # Make more transparent to the user
    def start_batch():
        start = time.time()
        return start

    @staticmethod  # Make more transparent to the user
    def end_batch():
        end = time.time()
        return end

    def __del__(self):
        # Handle exception and log error message
        if self.auto_clean is False:
            warning_message = (
                "WARNING: 'auto_clean' set to 'False' by Default, CPU, RAM, and Training Data Collection Aborted.\n"
                "Make sure you delete old Data files before rerunning code"
            )
            warnings.warn(warning_message, UserWarning)
        else:
            warning_message = (
                "WARNING: 'auto_clean' set to 'True', CPU, RAM, and Training Data Collection.\n"
                "Old Data Files have been Deleted"
            )
            warnings.warn(warning_message, UserWarning)
            try:
                os.remove(self.CPU_RAM_File)
                print(f"File '{self.CPU_RAM_File}' deleted successfully.")
            except FileNotFoundError:
                print(f"File '{self.CPU_RAM_File}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
            try:
                os.remove(self.training_results_file)
                print(f"File '{self.training_results_file}' deleted successfully.")
            except FileNotFoundError:
                print(f"File '{self.training_results_file}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
        # Close the CSV file properly
        sys.exit(1)

    @run_in_thread
    def training_loop_performance(self, epoch: int, batch_num: int,
                                  training_losses: list or int, batch_start_time=None, batch_end_time=None):
        """
            Track and record training loop performance metrics.

            This method is used to track and record performance metrics during a training loop batch. It calculates
            various metrics including batch time, throughput, disk read IOPS, and disk write IOPS. The calculated
            metrics are then written to a CSV file specified by 'training_results_file'. must call start_batch at
            beginning of training loop and end_batch() at the end. Must pass both values to this function

            Parameters
            ----------
            epoch : int
                The current epoch number.
            batch_num : int
                The current batch number.
            training_losses : list or int
                List of or singular training losses for the current batch.
            batch_start_time : int
                Timestamp indicating the start time of the batch.
            batch_end_time : int
                Timestamp indicating the end time of the batch.

            Returns
            -------
            None
        """
        # Update disk IOPS counters
        self.disk_read_count = self.disk_io_counters.read_count
        self.disk_write_count = self.disk_io_counters.write_count

        # Calculate Batch time
        self.batch_time = batch_end_time - batch_start_time
        # Calculate throughput of the system for each batch
        self.throughput = self.dataloader_batch_size * (batch_num + 1) / self.batch_time
        # Calculate disk read and write IOPS for each batch
        disk_read_iops = self.disk_read_count / self.batch_time
        disk_write_iops = self.disk_write_count / self.batch_time

        with open(self.training_results_file, mode='a', newline='') as r_file:
            writer = csv.writer(r_file)
            writer.writerow(
                [epoch, batch_num, training_losses[-1], self.batch_time, self.throughput, disk_read_iops,
                 disk_write_iops])


# DEDICATED IOPS TRACKER FOR DEPLOYMENT OF AI MODELS OR GENERAL WORKLOAD
class DiskIOPSMonitor:
    def __init__(self, tracking_results_file):
        self.tracking_thread = None
        self.tracking_results_file = tracking_results_file
        self.stop_tracking = False
        self.lock = threading.Lock()

    def track_iops(self):
        while not self.stop_tracking:
            start_time = time.time()

            # Get disk I/O statistics using psutil
            disk_io_counters = psutil.disk_io_counters()
            start_read_count, start_write_count = disk_io_counters.read_count, disk_io_counters.write_count

            # Simulate some disk I/O operations here (replace this with your actual disk I/O operations)
            time.sleep(1)  # Simulate some disk operations for 1 second

            end_time = time.time()

            # Get disk I/O statistics again
            disk_io_counters = psutil.disk_io_counters()
            end_read_count, end_write_count = disk_io_counters.read_count, disk_io_counters.write_count

            # Calculate disk read and write IOPS for this interval
            disk_read_iops = (end_read_count - start_read_count) / (end_time - start_time)
            disk_write_iops = (end_write_count - start_write_count) / (end_time - start_time)

            # Write the metrics to a CSV file
            with open(self.tracking_results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([end_time, disk_read_iops, disk_write_iops])

    def start_tracking(self):
        self.stop_tracking = False
        self.tracking_thread = threading.Thread(target=self.track_iops)
        self.tracking_thread.start()

    def stop_tracking(self):
        self.stop_tracking = True
        self.tracking_thread.join()


# DEDICATED CPU TRACKER FOR DEPLOYMENT OF AI MODELS OR GENERAL WORKLOAD
class CPUMonitor:
    def __init__(self, output_file_path, cpu_time_interval=1, auto_clean=True):
        self.CPU_Time_Interval = cpu_time_interval
        self.auto_clean = auto_clean
        self.output_file_path = output_file_path

    def monitor_cpu_utilization(self):
        with open(self.output_file_path, mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(['Core Time', 'CPU Utilization'])

        try:
            core_time = 0
            while True:
                # Get CPU utilization
                cpu_percent = psutil.cpu_percent(interval=self.CPU_Time_Interval)

                core_time += self.CPU_Time_Interval
                with open(self.output_file_path, mode='a', newline='') as results_file:
                    writer = csv.writer(results_file)
                    writer.writerow([core_time, cpu_percent])

                # Sleep for the specified interval
                time.sleep(self.CPU_Time_Interval)

        except KeyboardInterrupt:
            # Handle exception and log error message
            if not self.auto_clean:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'False' by Default, CPU Data Collection Aborted.\n"
                    "Make sure you delete old Data files before rerunning code"
                )
                warnings.warn(warning_message, UserWarning)
            else:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'True', CPU Data Collection Aborted.\n"
                    "Old Data Files have been Deleted"
                )
                warnings.warn(warning_message, UserWarning)
                try:
                    os.remove(self.output_file_path)
                    print(f"File '{self.output_file_path}' deleted successfully.")
                except FileNotFoundError:
                    print(f"File '{self.output_file_path}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            # Close the CSV file properly
            sys.exit(1)


# DEDICATED MEMORY TRACKER FOR DEPLOYMENT OF AI MODELS OR GENERAL WORKLOAD
class MemoryMonitor:
    def __init__(self, output_file_path, memory_time_interval=1, auto_clean=True):
        self.Memory_Time_Interval = memory_time_interval
        self.auto_clean = auto_clean
        self.output_file_path = output_file_path

    def monitor_memory_utilization(self):
        with open(self.output_file_path, mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(['Core Time', 'Memory Utilization (%)'])

        try:
            core_time = 0
            while True:
                # Get memory (RAM) utilization
                memory_percent = psutil.virtual_memory().percent

                core_time += self.Memory_Time_Interval
                with open(self.output_file_path, mode='a', newline='') as results_file:
                    writer = csv.writer(results_file)
                    writer.writerow([core_time, memory_percent])

                # Sleep for the specified interval
                time.sleep(self.Memory_Time_Interval)

        except KeyboardInterrupt:
            # Handle exception and log error message
            if not self.auto_clean:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'False' by Default, Memory Data Collection Aborted.\n"
                    "Make sure you delete old Data files before rerunning code"
                )
                warnings.warn(warning_message, UserWarning)
            else:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'True', Memory Data Collection Aborted.\n"
                    "Old Data Files have been Deleted"
                )
                warnings.warn(warning_message, UserWarning)
                try:
                    os.remove(self.output_file_path)
                    print(f"File '{self.output_file_path}' deleted successfully.")
                except FileNotFoundError:
                    print(f"File '{self.output_file_path}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            # Close the CSV file properly
            sys.exit(1)


class CPUThreadsMonitor:
    def __init__(self, output_file_path, threads_time_interval=1, auto_clean=True):
        self.Threads_Time_Interval = threads_time_interval
        self.auto_clean = auto_clean
        self.output_file_path = output_file_path

    def monitor_cpu_threads(self):
        with open(self.output_file_path, mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(['Core Time', 'Active CPU Threads'])

        try:
            core_time = 0
            while True:
                # Get the number of active CPU threads
                active_threads = multiprocessing.active_children()
                core_time += self.Threads_Time_Interval
                with open(self.output_file_path, mode='a', newline='') as results_file:
                    writer = csv.writer(results_file)
                    writer.writerow([core_time, active_threads])

                # Sleep for the specified interval
                time.sleep(self.Threads_Time_Interval)

        except KeyboardInterrupt:
            # Handle exception and log error message
            if not self.auto_clean:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'False' by Default, CPU Threads Data Collection Aborted.\n"
                    "Make sure you delete old Data files before rerunning code"
                )
                warnings.warn(warning_message, UserWarning)
            else:
                warning_message = (
                    "WARNING: 'auto_clean' set to 'True', CPU Threads Data Collection Aborted.\n"
                    "Old Data Files have been Deleted"
                )
                warnings.warn(warning_message, UserWarning)
                try:
                    os.remove(self.output_file_path)
                    print(f"File '{self.output_file_path}' deleted successfully.")
                except FileNotFoundError:
                    print(f"File '{self.output_file_path}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            # Close the CSV file properly
            sys.exit(1)


class DataPlotter:
    def __int__(self, save_folder=None, DPI=300, moving_avg_window=5, line_style="-", marker_size=2, marker_style='o',
                color_main='Red', color_secondary="Blue", average_color="Green"):
        self.graph_count = 0  # To keep track of the number of graphs created
        self.imageDPI = DPI
        self.linestyle = line_style
        self.marker = marker_style
        self.markerSize = marker_size
        self.color_P = color_main
        self.color_S = color_secondary
        self.color_Avg = average_color
        self.moving_Avg = moving_avg_window
        if save_folder is None:
            self.save_directory = ""
        else:
            self.save_directory = save_folder

    @measure_time
    def plot_cpu_utilization(self, Source_file):
        self.graph_count += 1
        core_time = []
        cpu_utilization = []

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            headers = next(reader)  # Read the header row
            # Assuming the first column is 'Core Time' and the second column is 'CPU Utilization'
            time_index = headers.index('Core Time')
            utilization_index = headers.index('CPU Utilization')

            for row in reader:
                core_time_value = row[time_index]
                cpu_utilization_value = row[utilization_index]

                # Skip any non-numeric values in the 'Core Time' column
                if core_time_value.isnumeric() or core_time_value.replace('.', '', 1).isdigit():
                    core_time.append(float(core_time_value))
                    cpu_utilization.append(float(cpu_utilization_value))
        # Close the CSV file
        csv_file.close()

        # Plotting
        plt.plot(core_time, cpu_utilization, linestyle=self.linestyle,
                 marker=self.marker, markersize=self.markerSize, color=self.color_P, label='CPU Utilization')
        plt.xlabel('Core Time')
        plt.ylabel('CPU Utilization')
        plt.title('CPU Utilization')
        plt.grid(True)

        # Calculate and plot the moving average
        window_size = self.moving_Avg
        moving_avg = np.convolve(cpu_utilization, np.ones(window_size), 'valid') / window_size
        plt.plot(core_time[window_size - 1:], moving_avg, linestyle=self.linestyle, color=self.color_S,
                 label='Moving Average (window = 5)')

        # Calculate and display the average value
        average_utilization = np.mean(cpu_utilization)
        plt.axhline(average_utilization, color=self.color_Avg, linestyle='--',
                    label=f'Average Utilization: {average_utilization:.2f}')

        # Save the plot as an image file
        try:
            # Make sure the directory exists before saving
            os.makedirs(os.path.dirname(self.save_directory), exist_ok=True)
            plt.legend()
            plt.savefig(f"{self.save_directory}/CPU_Utilization_Percent.png", dpi=self.imageDPI)
            plt.close()
        except FileNotFoundError:
            print(f"Directory not found: {os.path.dirname(self.save_directory)}")
        except Exception as e:
            print(f"An error occurred while saving the graph: {e}")

    @measure_time
    def plot_thread_count(self, Source_file):
        core_time = []
        thread_count = []
        self.graph_count += 1

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            headers = next(reader)  # Read the header row
            # Assuming the first column is 'Core Time' and the third column is 'Thread Count'
            time_index = headers.index('Core Time')
            thread_count_index = headers.index('Thread Count')

            for row in reader:
                core_time_value = row[time_index]
                thread_count_value = row[thread_count_index]

                # Skip any non-numeric values in the 'Core Time' column
                if core_time_value.isnumeric() or core_time_value.replace('.', '', 1).isdigit():
                    core_time.append(float(core_time_value))
                    thread_count.append(int(thread_count_value))

        # Plotting
        plt.scatter(core_time, thread_count, linestyle=self.linestyle, marker=self.marker,
                    color=self.color_P, label='Thread Count', s=10)
        plt.xlabel('Core Time')
        plt.ylabel('Thread Count')
        plt.title('Thread Count')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_directory}/CPU_Thread_Count.png", dpi=self.imageDPI)
        plt.close()

    @measure_time
    def plot_ram_utilization_percent(self, Source_file):
        core_time = []
        ram_utilization_percent = []
        self.graph_count += 1

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            headers = next(reader)  # Read the header row
            # Assuming the first column is 'Core Time' and the fourth column is 'RAM Utilization (%)'
            time_index = headers.index('Core Time')
            ram_utilization_percent_index = headers.index('RAM Utilization (%)')

            for row in reader:
                core_time_value = row[time_index]
                ram_utilization_percent_value = row[ram_utilization_percent_index]

                # Skip any non-numeric values in the 'Core Time' column
                if core_time_value.isnumeric() or core_time_value.replace('.', '', 1).isdigit():
                    core_time.append(float(core_time_value))
                    ram_utilization_percent.append(float(ram_utilization_percent_value))
        # Close the CSV file
        csv_file.close()

        # Plotting
        plt.plot(core_time, ram_utilization_percent, linestyle=self.linestyle,
                 marker=self.marker, markersize=self.markerSize, color=self.color_P, label='RAM Utilization (%)')
        plt.xlabel('Core Time')
        plt.ylabel('RAM Utilization (%)')
        plt.title('RAM Utilization (%)')
        plt.grid(True)

        # Calculate and plot the moving average
        window_size = self.moving_Avg
        moving_avg = np.convolve(ram_utilization_percent, np.ones(window_size), 'valid') / window_size
        plt.plot(core_time[window_size - 1:], moving_avg, linestyle='-', color=self.color_S,
                 label='Moving Average (window = 5)')

        # Calculate and display the average value
        average_utilization = np.mean(ram_utilization_percent)
        plt.axhline(average_utilization, color=self.color_Avg, linestyle='--',
                    label=f'Average Utilization: {average_utilization:.2f}%')

        # Save the plot as an image file
        plt.legend()
        plt.savefig(f"{self.save_directory}/RAM_Percent_Utilization.png", dpi=self.imageDPI)
        plt.close()

    @measure_time
    def plot_training_loss(self, Source_file):
        """
        Plot Training Loss

        This function plots the Training Loss column against the batch number,
        and incorporates various enhancements to make the graph more informative.

        Parameters
        ----------
        Source_file : str
            The path to the CSV file containing the data.

        Returns
        -------
        None

        """
        batch_number = []
        training_loss = []
        self.graph_count += 1

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header row
            next(reader)  # Skip the header row
            for row in reader:
                batch_number.append(int(row[1]))
                training_loss.append(float(row[2]))
        # Close the CSV file
        csv_file.close()

        # Plotting
        plt.plot(batch_number, training_loss, linestyle=self.linestyle,
                 marker=self.marker, markersize=self.markerSize, color=self.color_P,
                 label='Training Loss')
        plt.xlabel('Batch Number')
        plt.ylabel('Training Loss')
        plt.title('Training Loss')
        plt.grid(True)

        # Calculate and plot the moving average
        window_size = self.moving_Avg
        moving_avg = np.convolve(training_loss, np.ones(window_size), 'valid') / window_size
        plt.plot(batch_number[window_size - 1:], moving_avg, linestyle='-',
                 color=self.color_S, label='Moving Average (window = 5)')

        # Calculate and display the average value
        average_loss = np.mean(training_loss)
        plt.axhline(average_loss, color=self.color_Avg, linestyle='--', label=f'Average Loss: {average_loss:.2f}')

        # Save the plot as an image file
        plt.legend()
        plt.savefig(f"{self.save_directory}/Training_Loss.png", dpi=self.imageDPI)
        plt.close()

    @measure_time
    def plot_throughput(self, Source_file):
        """
        Plot Throughput

        This function plots the Throughput (Seq/sec) column against the batch number,
        and incorporates various enhancements to make the graph more informative.

        Parameters
        ----------
        Source_file : str
            The path to the CSV file containing the data.

        Returns
        -------
        None

        """
        batch_number = []
        throughput = []
        self.graph_count += 1

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header row
            next(reader)  # Skip the header row
            for row in reader:
                batch_number.append(int(row[1]))
                throughput.append(float(row[4]))
        # Close the CSV file
        csv_file.close()

        # Plotting
        plt.scatter(batch_number, throughput, linestyle=self.linestyle, marker=self.marker,
                    color=self.color_P, label='Throughput', s=10)
        plt.xlabel('Batch Number')
        plt.ylabel('Throughput (Seq/sec)')
        plt.title('Throughput (Seq/sec)')
        plt.grid(True)

        # Calculate and plot the moving average
        window_size = self.moving_Avg
        moving_avg = np.convolve(throughput, np.ones(window_size), 'valid') / window_size
        plt.scatter(batch_number[window_size - 1:], moving_avg, linestyle='-', color=self.color_S,
                    label='Moving Average (window = 5)', s=10)

        # Calculate and display the average value
        average_throughput = np.mean(throughput)
        plt.axhline(average_throughput, color=self.color_Avg, linestyle='--',
                    label=f'Average Throughput: {average_throughput:.2f}')

        # Save the plot as an image file
        plt.legend()
        plt.savefig(f"{self.save_directory}/Throughput.png", dpi=self.imageDPI)
        plt.close()

    @measure_time
    def plot_disk_iops(self, Source_file):
        """
        Plot Disk IOPS

        This function plots the Disk Read IOPS and Disk Write IOPS columns against the batch number on the same graph,
        and incorporates various enhancements to make the graph more informative.

        Parameters
        ----------
        Source_file : str
            The path to the CSV file containing the data.

        Returns
        -------
        None

        """
        batch_number = []
        disk_read_iops = []
        disk_write_iops = []
        self.graph_count += 1

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header row
            next(reader)  # Skip the header row
            for row in reader:
                batch_number.append(int(row[1]))
                disk_read_iops.append(float(row[5]))
                disk_write_iops.append(float(row[6]))
        # Close the CSV file
        csv_file.close()

        # Plotting
        plt.scatter(batch_number, disk_read_iops, linestyle='-', marker='o', color='purple',
                    label='Disk Read IOPS', s=10)
        plt.scatter(batch_number, disk_write_iops, linestyle='-', marker='o', color='orange',
                    label='Disk Write IOPS', s=10)
        plt.xlabel('Batch Number')
        plt.ylabel('Disk IOPS')
        plt.title('Disk Read IOPS vs. Disk Write IOPS')
        plt.grid(True)

        # Calculate and plot the moving average for Disk Read IOPS
        window_size = self.moving_Avg
        moving_avg_read = np.convolve(disk_read_iops, np.ones(window_size), 'valid') / window_size
        plt.scatter(batch_number[window_size - 1:], moving_avg_read, linestyle='-', color='blue',
                    label='Moving Average (Read IOPS, window = 5)', s=10)

        # Calculate and plot the moving average for Disk Write IOPS
        moving_avg_write = np.convolve(disk_write_iops, np.ones(window_size), 'valid') / window_size
        plt.scatter(batch_number[window_size - 1:], moving_avg_write, linestyle='-', color='green',
                    label='Moving Average (Write IOPS, window = 5)', s=10)

        # Calculate and display the average values
        average_read_iops = np.mean(disk_read_iops)
        average_write_iops = np.mean(disk_write_iops)
        plt.axhline(average_read_iops, color='red', linestyle='--', label=f'Average Read IOPS: {average_read_iops:.2f}')
        plt.axhline(average_write_iops, color='purple', linestyle='--',
                    label=f'Average Write IOPS: {average_write_iops:.2f}')

        # Save the plot as an image file
        plt.legend()
        plt.savefig(f"{self.save_directory}/Disk_IOPs.png", dpi=self.imageDPI)
        plt.close()

    @measure_time
    def plot_disk_read_iops(self, Source_file):
        """
        Plot Disk Read IOPS

        This function plots the Disk Read IOPS column against the batch number,
        and incorporates various enhancements to make the graph more informative.

        Parameters
        ----------
        Source_file : str
            The path to the CSV file containing the data.

        Returns
        -------
        None

        """
        batch_number = []
        disk_read_iops = []
        self.graph_count += 1

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header row
            next(reader)  # Skip the header row
            for row in reader:
                batch_number.append(int(row[1]))
                disk_read_iops.append(float(row[5].strip()))
        # Close the CSV file
        csv_file.close()

        # Plotting
        plt.scatter(batch_number, disk_read_iops, linestyle='-', marker='o', color='purple',
                    label='Disk Read IOPS', s=10)
        plt.xlabel('Batch Number')
        plt.ylabel('Disk Read IOPS (I/O Operations per Second)')
        plt.title('Disk Read IOPS')
        plt.grid(True)

        # Calculate and plot the moving average
        window_size = self.moving_Avg
        moving_avg = np.convolve(disk_read_iops, np.ones(window_size), 'valid') / window_size
        plt.scatter(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
                    label='Moving Average (window = 5)', s=10)

        # Calculate and display the average value
        average_iops = np.mean(disk_read_iops)
        plt.axhline(average_iops, color='red', linestyle='--', label=f'Average IOPS: {average_iops:.2f}')

        # Save the plot as an image file
        plt.legend()
        plt.savefig(f"{self.save_directory}/Disk_Read_IOPs.png", dpi=self.imageDPI)
        plt.close()

    @measure_time
    def plot_disk_write_iops(self, Source_file):
        batch_number = []
        disk_write_iops = []
        self.graph_count += 1

        with open(Source_file, mode='r', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header row
            next(reader)  # Skip the header row
            for row in reader:
                batch_number.append(int(row[1]))
                disk_write_iops.append(float(row[6].strip()))

        # Close the CSV file
        csv_file.close()

        # Plotting
        plt.scatter(batch_number, disk_write_iops, color='orange', label='Disk Write IOPS', s=10)
        plt.xlabel('Batch Number')
        plt.ylabel('Disk Write IOPS')
        plt.title('Disk Write IOPS')
        plt.grid(True)

        # Calculate and plot the moving average
        window_size = self.moving_Avg
        moving_avg = np.convolve(disk_write_iops, np.ones(window_size), 'valid') / window_size
        plt.scatter(batch_number[window_size - 1:], moving_avg, linestyle='-', color='blue',
                    label='Moving Average (window = 5)', s=10)

        # Calculate and display the average value
        average_iops = np.mean(disk_write_iops)
        plt.axhline(average_iops, color='red', linestyle='--', label=f'Average IOPS: {average_iops:.2f}')

        # Save the plot as an image file
        plt.legend()
        plt.savefig(f"{self.save_directory}/Disk_Write_IOPs.png", dpi=self.imageDPI)
        plt.close()

    @measure_time
    def plot_GPU(self, interval=25, height=5, width=10, combine_graphs=False):
        GPU_Plot_Runtime(interval=interval, height=height, width=width, combine_graphs=combine_graphs)
        self.graph_count += 1

    @measure_time
    def generate_all_graphs(self, CPU_RAM_Data="CPU_RAM_Utilization.csv", Training_Results="training_results.csv"):
        print("Generating Graphs...")
        self.plot_cpu_utilization(Source_file=CPU_RAM_Data)
        self.plot_ram_utilization_percent(Source_file=CPU_RAM_Data)
        self.plot_thread_count(Source_file=CPU_RAM_Data)
        self.plot_disk_iops(Source_file=Training_Results)
        self.plot_disk_write_iops(Source_file=Training_Results)
        self.plot_disk_read_iops(Source_file=Training_Results)
        self.plot_throughput(Source_file=Training_Results)
        self.plot_training_loss(Source_file=Training_Results)
        self.graph_count += 8


# PREPROCESSING FUNCTIONS IN CASE OF OVER THE AIR TRANSFER ERRORS WITH KUBECTL (Or Redundant CSV Writes from
# distributed Workloads)
class DataCleaner:
    def __int__(self, input_file: str, output_file=None):
        self.input_file = input_file
        if output_file is None:
            self.output_file = input_file
        else:
            self.output_file = output_file

    def remove_nul_from_csv(self):

        with open(self.input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
            data = csv_file.read()

        # Remove NUL characters from the data
        clean_data = data.replace('\x00', '')

        # Write the clean data to a new CSV file
        with open(self.output_file, 'w', newline='', encoding='utf-8') as out_file:
            out_file.write(clean_data)

    def remove_non_ascii(self):
        with open(self.input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
            data = csv_file.read()

        # Remove non-ASCII characters from the data
        clean_data = ''.join(char for char in data if ord(char) < 128)

        # Write the clean data to a new CSV file with proper encoding
        with open(self.output_file, 'w', newline='', encoding='utf-8') as out_file:
            out_file.write(clean_data)

    def remove_redundant_headers(self):
        with open(self.input_file, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)

        # Find the index of the first occurrence of the header
        header_index = None
        for i, row in enumerate(data):
            if 'Epoch' in row[0] and 'Core' in row[1]:  # Assuming 'Epoch' and 'Core' are in the first two columns
                header_index = i
                break

        if header_index is not None:
            # Remove the redundant headers
            data = data[:header_index] + data[header_index + 1:]

            # Write the updated data back to the file
            with open(self.output_file, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(data)

    def clean_strings_quotes_from_csv(self):
        with open(self.input_file, mode='r', newline='') as file:
            data = file.read()

        # Split the data by newline characters to get individual rows
        rows = data.strip().split('\n')

        # Remove double-quote characters from both ends of each value in each row
        cleaned_rows = []
        for row in rows:
            cleaned_row = ','.join(value.strip('"') for value in row.split(','))
            cleaned_rows.append(cleaned_row)

        # Join the cleaned rows back into a single string with newline characters
        cleaned_data = '\n'.join(cleaned_rows)

        # Write the cleaned data back to the file
        with open(self.output_file, mode='w', newline='') as file:
            file.write(cleaned_data)

    def remove_quotes_from_csv(self):
        with open(self.input_file, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)

        with open(self.output_file, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            for row in data:
                cleaned_row = [value.strip('"') for value in row]
                csv_writer.writerow(cleaned_row)

    def clean_csv(self):
        # Read the CSV file
        with open(self.input_file, 'r', newline='', encoding='utf-8', errors='replace') as csv_file:
            data = csv_file.read()

        # Remove NUL characters from the data
        clean_data = data.replace('\x00', '')

        # Write the clean data to a new CSV file
        with open(self.output_file, 'w', newline='', encoding='utf-8') as out_file:
            out_file.write(clean_data)

        # Read the cleaned CSV data
        cleaned_data = []
        with open(self.output_file, 'r', newline='', encoding='utf-8', errors='replace') as cleaned_csv_file:
            for line in cleaned_csv_file:
                cleaned_data.append(line.strip().split(','))

        if cleaned_data[-1] == ['']:
            cleaned_data.pop()
        # Find duplicate values in the "Batch" column
        header = cleaned_data[0]
        try:
            batch_index = header.index("Batch")
        except IndexError:
            # Add a user warning
            warning_message = (
                "WARNING: The 'clean_csv' method is intended to clean batch data.\n"
                "Make sure you're using this method with the correct data. Setting header.index(<value>) to "
                "header.index('CoreTime')"
            )
            warnings.warn(warning_message, UserWarning)
            batch_index = header.index("Core Time")
        unique_batches = set()
        final_cleaned_data = [header]
        for row in cleaned_data[1:]:
            if len(row) > batch_index:
                batch_value = row[batch_index]
                if batch_value not in unique_batches:
                    unique_batches.add(batch_value)
                    final_cleaned_data.append(row)

        # Write the final cleaned data to a new CSV file
        with open(self.output_file, 'w', newline='', encoding='utf-8') as final_out_file:
            for row in final_cleaned_data:
                final_out_file.write(','.join(row) + '\n')

    # Function to measure execution time
    @measure_time
    def preprocess_data_all_methods(self):
        print("Preprocessing Data...")
        self.remove_redundant_headers()
        self.remove_nul_from_csv()
        self.remove_non_ascii()
        self.clean_csv()
        self.clean_strings_quotes_from_csv()
        self.remove_quotes_from_csv()
        print("Finished Preprocessing.")