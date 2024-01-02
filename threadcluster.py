import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# NOTE in order to run this accross the different clusters this must be passed "mpiexec -n 4 python threadcluster.py"
# Change "4" to the number of nodes in the cluster to be tested


# Shared resource and lock for synchronization
shared_variable = 0
lock = threading.Lock()

def benchmark_function_with_lock(thread_id, splitnum):
    global shared_variable

    for i in range(splitnum):
        # Using lock to ensure only one thread has access at a time
        with lock:
            shared_variable += 1



    # Print some information
    #print(f"Thread {thread_id}: completed\n")


def benchmark_function_no_lock(thread_id, splitnum):
    global shared_variable

    # Simulate a computation-intensive task
    for i in range(splitnum):
        shared_variable += 1

    # Print some information
    #print(f"Thread {thread_id}: completed\n")


def run_benchmark_with_lock(num_threads, mpi_rank, targetnum):
    # Determine the range of thread IDs for each MPI rank
    start_thread_id = mpi_rank * num_threads
    end_thread_id = start_thread_id + num_threads

    for thread_id in range(start_thread_id, end_thread_id):
        benchmark_function_with_lock(thread_id, targetnum)


def run_benchmark_without_lock(num_threads, mpi_rank, targetnum):
    # Determine the range of thread IDs for each MPI rank
    start_thread_id = mpi_rank * num_threads
    end_thread_id = start_thread_id + num_threads

    for thread_id in range(start_thread_id, end_thread_id):
        benchmark_function_no_lock(thread_id, targetnum)

def results_plotting(xvals, yvalnolock, yvalwithlock):

    x = np.array(xvals)
    yno = np.array(yvalnolock)
    ylock = np.array(yvalwithlock)

    print(len(x))


    plt.title("Execution time of multithreading")
    # Note number of loops may show up as scientific notation by will always be multiple of 10
    plt.xlabel("Number of Loops")
    plt.ylabel("Time to Completion (seconds)")
    plt.plot(x, yno, '--ro', label ="No locks")
    plt.plot(x, ylock, '--bo', label ="With locks")
    plt.legend()
    plt.show()



def main():
    global shared_variable
    # Initialize MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank() # Rank is used to identify a process that belongs to a communicator

    # This is the total number being calculated for the benchmark
    # Add or remove a 0 to increase or decrease difficulty
    targetnummax = 1000000000

    curnum = 1000

    timenolocks = []
    timewithlocks = []
    xvals = []

    num_threads = 5


    # Loop with different benchmark difficulty
    while curnum < targetnummax:
        shared_variable = 0

        # Divide the number into equal parts twice
        splitnum = int(curnum / num_threads / num_threads)

        print("The split number: " + str(splitnum))

        print("Number of threads: " + str(num_threads))
        # num_threads = total_threads

        # Create threads within the MPI rank
        threads = []

        print("Beginning test without locks")

        # Testing time without lock first
        start_time = time.time()
        for _ in range(num_threads):
            thread = threading.Thread(target=run_benchmark_without_lock, args=(num_threads, mpi_rank, splitnum))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        end_time = time.time()


        nolocktime = end_time - start_time

        print("Time without lock testing concluded")
        # Print the final result from the shared variable
        print(f"MPI Rank {mpi_rank}: Shared Variable Value: {shared_variable}")
        print("Cluster Execution Time:", nolocktime)

        print()
        print("Beginning test with locks")

        shared_variable = 0

        start_time = time.time()
        for _ in range(num_threads):
            thread = threading.Thread(target=run_benchmark_with_lock, args=(num_threads, mpi_rank, splitnum))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        end_time = time.time()

        locktime = end_time - start_time

        # Print the final result from the shared variable
        print(f"MPI Rank {mpi_rank}: Shared Variable Value: {shared_variable}")
        print("Cluster Execution Time:", locktime)

        timenolocks.append(nolocktime)
        timewithlocks.append(locktime)

        xvals.append(curnum)
        # Increase by a factor a 10 for next loop
        curnum = curnum * 10
    results_plotting(xvals, timenolocks, timewithlocks)

if __name__ == "__main__":
    main()
