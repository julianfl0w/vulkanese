import multiprocessing as mp
from synth import runSynth
import os

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    backendProc = ctx.Process(target=runSynth, args=(q,))
    backendProc.start()
    print("PID" + str(backendProc.pid))
    os.sched_setaffinity(backendProc.pid, {7})
    print("CPU affinity mask is modified for process id % s" % backendProc.pid)
    print("Now, process is eligible to run on:", os.sched_getaffinity(backendProc.pid))

    # mp.freeze_support()
    from synthGui import runGui

    frontendProc = ctx.Process(target=runGui, args=(q,))
    frontendProc.start()
    frontendProc.join()
    backendProc.join()
