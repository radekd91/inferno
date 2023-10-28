import sys, os
import subprocess 

def main(): 
    # we are creating a script that has a loop 
    # in the loop it ssh to the cluster, runs a command, and then exits
    # the scripts then sleeps for 10 minutes and then repeats
    username = "rdanecek"
    cmd = "condor_release rdanecek"
    max_loops = 10000

    for i in range(max_loops):
        subprocess.call(["ssh", "%s@login4.cluster.is.localnet" % (username,)] + [cmd])
        # sleep for 10 minutes
        subprocess.call(["sleep", "600"])

    print("Done")


if __name__ == "__main__": 
    main()