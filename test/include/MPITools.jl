module MPITools

import MPI

export silence_stdout

"""
    silence_stdout(comm)

Silence standard output of all but one MPI process.
"""
function silence_stdout(comm, root=0)
    dev_null = @static Sys.iswindows() ? "nul" : "/dev/null"
    MPI.Comm_rank(comm) == root || redirect_stdout(open(dev_null, "w"))
    nothing
end

end
