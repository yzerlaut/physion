import os
import multiprocessing

def process_datafiles(process_file_func,
                      filenames,
                      output_folder):
    
    """
    general function to parallelizes the processing of a set of files
    """

    cpus = multiprocessing.cpu_count()-1 # leaving 1 cpu for the rest

    nruns = int(len(filenames)/cpus)+1

    for r in range(nruns):
        i0 = r*cpus
        imax = min([i0+cpus, len(filenames)])
        print(' - running set of files %i:%i' % (i0, imax))

        # start the processes
        procs = []
        for i in range(i0,imax):
            proc = multiprocessing.Process(\
                                target=process_file_func, 
                                args=(filenames[i], i, output_folder))
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()

def example_func(filename, i, output_folder):
    print(\
        os.path.join(output_folder,
                        '%i-%s.npy' % (i, filename)))

if __name__=='__main__':

    filenames = ['data-%i.nwb' % i for i in range(100, 150)]
    process_datafiles(example_func,
                      filenames,
                      './temp/')