### Advances In Data Mining
### Assignment 2
### Luit Verschuur 1811053, Marcel Kolenbrander 1653415.

import multiprocessing

class Parallels:
    function = ...
    max_number_of_workers = ...

    def __init__(self, target_function_in=None, max_number_of_workers_in=None):
        if target_function_in is not None:
            if not callable(target_function_in):
                raise ValueError("Provided target is not a callable object")
            self.function = target_function_in
        else:
            raise ValueError("No target provided")

        if max_number_of_workers_in is not None:
            self.max_number_of_workers = max_number_of_workers_in
        else:
            self.max_number_of_workers = 1

    def _wrapper_caller(self, fn, args, p_num, return_dict, _kwargs=None):
        """
        Wrapper function which calls the target function and catches their return values in the return dictionary.

        :param fn: Target function
        :param args: Arguments for the function
        :param p_num: Process number (internal PID) used for the return dict
        :param return_dict: Managed process data dictionary used to capture return values
        :param _kwargs: Optional key word arguments
        :return:
        """

        # Check how to unpack the provided arguments
        if isinstance(args, tuple):
            # Store the return value in a managed dictionary
            return_dict[p_num] = fn(*args) if _kwargs is None else fn(*args, **_kwargs)
        elif isinstance(args, dict):
            # Store the return value in a managed dictionary
            return_dict[p_num] = fn(**args)
        else:
            raise ValueError("Incorrect parameter format, must be Tuple or Dict")

    def run(self, args, workers=1, wait=True):
        """
        Run the class init function on workers amount of processes (forks) with the provided arguments. The worker
        scheduler halts and collects all workers if wait is set to True. The results of the functions are returned as
        a dictionary of return values.

        :param args: Arguments to be passed to the class init target function.
        :param workers: Number of processes to be opened.
        :param wait: Collect workers and hang if set to True.
        :return: Process result dictionary.
        """

        process_list = []

        # Collection dictionary containing the return values of the different workers.
        return_dict = multiprocessing.Manager().dict()

        # Run target process with a given number of workers not exceeding the maximum number of workers.
        for p_i in range(0, workers if workers <= self.max_number_of_workers else self.max_number_of_workers):
            process = multiprocessing.Process(target=self._wrapper_caller, args=(self.function, args, p_i, return_dict))
            process.start()
            process_list.append(process)

        # Collect workers and block until they are done.
        if wait is True:
            for process in process_list:
                process.join()

        return return_dict

    def run_map(self, map_values, args=None, wait=True):
        """
        Run the class init function with the map_values mapped to the workers. The worker
        scheduler halts and collects all workers if wait is set to True. The results of the functions are returned as
        a dictionary of return values.

        :param map_values: A list of values and/or objects to pass to the workers.
        :param args: Arguments to be passed to the class init target function.
        :param wait: Collect workers and hang if set to True.
        :return: Process result dictionary.
        """

        if not isinstance(map_values, list):
            raise ValueError("The map values must be entered as a list object")

        process_list = []

        # Collection dictionary containing the return values of the different workers.
        return_dict = multiprocessing.Manager().dict()

        # Divide the provided map over the available processes. If the number of map items exceeds the maximum allowed
        # number of workers, the remaining jobs are further divided once the current workers are done.
        head_of_jobs = 0
        while len(map_values[head_of_jobs:]) > 0:
            if (num_workers := len(map_values[head_of_jobs:])) > self.max_number_of_workers:
                num_workers = self.max_number_of_workers

            # Run target process with a given number of workers not exceeding the maximum number of workers.
            for p_i in range(head_of_jobs, head_of_jobs + num_workers):
                if args is None:
                    process = multiprocessing.Process(target=self._wrapper_caller,
                                                      args=(self.function, (map_values[p_i], ), p_i, return_dict))
                else:
                    process = multiprocessing.Process(target=self._wrapper_caller,
                                                      args=(self.function, (map_values[p_i], *args), p_i, return_dict))

                process.start()
                process_list.append(process)

            # Collect workers and block until they are done.
            if wait is True:
                for process in process_list:
                    process.join()

            head_of_jobs += num_workers

        return return_dict
