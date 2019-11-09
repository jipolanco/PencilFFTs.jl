var documenterSearchIndex = {"docs":
[{"location":"Pencils/#Pencils-module-1","page":"Pencils module","title":"Pencils module","text":"","category":"section"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"CurrentModule = PencilFFTs.Pencils","category":"page"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"Pencils","category":"page"},{"location":"Pencils/#PencilFFTs.Pencils","page":"Pencils module","title":"PencilFFTs.Pencils","text":"Module for multidimensional data decomposition using MPI.\n\nHandles different decomposition configurations and data transpositions between them. Also defines relevant data structures for handling distributed data.\n\n\n\n\n\n","category":"module"},{"location":"Pencils/#Types-1","page":"Pencils module","title":"Types","text":"","category":"section"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"MPITopology\nPencil\nPencilArray\nShiftedArrayView","category":"page"},{"location":"Pencils/#PencilFFTs.Pencils.MPITopologies.MPITopology","page":"Pencils module","title":"PencilFFTs.Pencils.MPITopologies.MPITopology","text":"MPITopology{N}\n\nDescribes an N-dimensional Cartesian MPI decomposition topology.\n\n\n\nMPITopology(comm::MPI.Comm, pdims::Dims{N}) where N\n\nCreate N-dimensional MPI topology information.\n\nThe pdims tuple specifies the number of MPI processes to put in every dimension of the topology. The product of its values must be equal to the number of processes in communicator comm.\n\nExample\n\n# Divide 2D topology into 4×2 blocks.\ncomm = MPI.COMM_WORLD\n@assert MPI.Comm_size(comm) == 8\ntopology = MPITopology(comm, (4, 2))\n\n\n\nMPITopology{N}(comm_cart::MPI.Comm) where N\n\nCreate topology information from MPI communicator with Cartesian topology. The topology must have dimension N.\n\n\n\n\n\n","category":"type"},{"location":"Pencils/#PencilFFTs.Pencils.Pencil","page":"Pencils module","title":"PencilFFTs.Pencils.Pencil","text":"Pencil{N,M,T}\n\nDescribes the decomposition of an N-dimensional Cartesian geometry among MPI processes along M directions (with M < N).\n\nThe Pencil describes the decomposition of arrays of element type T.\n\n\n\nPencil(topology::MPITopology{M}, size_global::Dims{N},\n       decomp_dims::Dims{M}, [element_type=Float64];\n       permute=nothing, timer=TimerOutput())\n\nDefine the decomposition of an N-dimensional geometry along M dimensions.\n\nThe dimensions of the geometry are given by size_global = (N1, N2, ...). The Pencil describes the decomposition of an array of dimensions size_global and type T across a group of MPI processes.\n\nData is distributed over the given M-dimensional MPI topology (with M < N). The decomposed dimensions are given by decomp_dims.\n\nThe optional parameter perm should be a tuple defining a permutation of the data indices. This may be useful for performance reasons, since it may be preferable (e.g. for FFTs) that the data is contiguous along the pencil direction.\n\nIt is also possible to pass a TimerOutput to the constructor. See Measuring performance for details.\n\nExamples\n\nDecompose a 3D geometry of global dimensions N_x  N_y  N_z = 4812 along the second (y) and third (z) dimensions.\n\nPencil(topology, (4, 8, 12), (2, 3))             # data is in (x, y, z) order\nPencil(topology, (4, 8, 12), (2, 3), (3, 2, 1))  # data is in (z, y, x) order\n\nIn the second case, the actual data is stored in (z, y, x) order within each MPI process.\n\n\n\nPencil(p::Pencil{N,M}, [element_type=eltype(p)];\n       decomp_dims::Dims{M}=get_decomposition(p),\n       size_global::Dims{N}=size_global(p),\n       permute::P=get_permutation(p),\n       timer::TimerOutput=get_timer(p))\n\nCreate new pencil configuration from an existent one.\n\nThis constructor enables sharing temporary data buffers between the two pencil configurations, leading to reduced global memory usage.\n\n\n\n\n\n","category":"type"},{"location":"Pencils/#PencilFFTs.Pencils.PencilArray","page":"Pencils module","title":"PencilFFTs.Pencils.PencilArray","text":"PencilArray(pencil::P, data::AbstractArray{T,N})\n\nCreate array wrapper with pencil decomposition information.\n\nThe array dimensions and element type must be consistent with those of the given pencil.\n\nThe data array can have one or more extra dimensions to the left (fast indices). For instance, these may correspond to vector or tensor components.\n\nExample\n\nSuppose pencil has local dimensions (10, 20, 30). Then:\n\nPencilArray(pencil, zeros(10, 20, 30))        # works (scalar)\nPencilArray(pencil, zeros(3, 10, 20, 30))     # works (3-component vector)\nPencilArray(pencil, zeros(4, 3, 10, 20, 30))  # works (4×3 tensor)\nPencilArray(pencil, zeros(10, 20, 30, 3))     # fails\n\n\n\nPencilArray(pencil::Pencil, [extra_dims...])\n\nAllocate uninitialised PencilArray that can hold data in the local pencil.\n\nExtra dimensions, for instance representing vector components, can be specified. These dimensions are added to the leftmost (fastest) indices of the resulting array.\n\nExample\n\nSuppose pencil has local dimensions (10, 20, 30). Then:\n\nPencilArray(pencil)        # array dimensions are (10, 20, 30)\nPencilArray(pencil, 4, 3)  # array dimensions are (4, 3, 10, 20, 30)\n\n\n\n\n\n","category":"type"},{"location":"Pencils/#PencilFFTs.Pencils.ShiftedArrays.ShiftedArrayView","page":"Pencils module","title":"PencilFFTs.Pencils.ShiftedArrays.ShiftedArrayView","text":"ShiftedArrayView{T,N}\n\nWraps an array shifting its indices.\n\n\n\n\n\n","category":"type"},{"location":"Pencils/#Functions-1","page":"Pencils module","title":"Functions","text":"","category":"section"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"eltype\ngather\nget_comm\nget_decomposition\nget_permutation\nglobal_view\nhas_indices\nlength\nndims\nndims_extra\nparent\npencil\nrange_local\nsize\nsize_global\nsize_local\nto_local\ntranspose!","category":"page"},{"location":"Pencils/#Base.eltype","page":"Pencils module","title":"Base.eltype","text":"eltype(p::Pencil)\n\nElement type associated to the given pencil.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.gather","page":"Pencils module","title":"PencilFFTs.Pencils.gather","text":"gather(x::PencilArray, [root::Integer=0])\n\nGather data from all MPI processes into one (big) array.\n\nData is received by the root process.\n\nReturns the full array on the root process, and nothing on the other processes.\n\nThis can be useful for testing, but it shouldn't be used with very large datasets!\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.MPITopologies.get_comm","page":"Pencils module","title":"PencilFFTs.Pencils.MPITopologies.get_comm","text":"get_comm(t::MPITopology)\n\nGet MPI communicator associated to an MPI Cartesian topology.\n\n\n\n\n\nget_comm(p::Pencil)\n\nGet MPI communicator associated to an MPI decomposition scheme.\n\n\n\n\n\nget_comm(x::PencilArray)\n\nGet MPI communicator associated to a pencil-distributed array.\n\n\n\n\n\nget_comm(p::PencilFFTPlan)\n\nGet MPI communicator associated to a PencilFFTPlan.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.get_decomposition","page":"Pencils module","title":"PencilFFTs.Pencils.get_decomposition","text":"get_decomposition(p::Pencil)\n\nGet tuple with decomposed dimensions of the given pencil configuration.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.get_permutation","page":"Pencils module","title":"PencilFFTs.Pencils.get_permutation","text":"get_permutation(p::Pencil)\n\nGet index permutation associated to the given pencil configuration.\n\nReturns nothing if there is no associated permutation.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.global_view","page":"Pencils module","title":"PencilFFTs.Pencils.global_view","text":"global_view(x::PencilArray)\n\nCreate a ShiftedArrayView of a PencilArray that takes global indices.\n\nThe order of indices in the returned view is the same as for the original array x. That is, if the indices of x are permuted, so are those of the returned array.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.ShiftedArrays.has_indices","page":"Pencils module","title":"PencilFFTs.Pencils.ShiftedArrays.has_indices","text":"has_indices(x::ShiftedArrayView, indices...)\n\nCheck whether the given set of indices is within the range of a shifted array.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#Base.length","page":"Pencils module","title":"Base.length","text":"length(t::MPITopology)\n\nGet total size of Cartesian topology (i.e. total number of MPI processes).\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#Base.ndims","page":"Pencils module","title":"Base.ndims","text":"ndims(t::MPITopology)\n\nGet dimensionality of Cartesian topology.\n\n\n\n\n\nndims(p::Pencil)\n\nNumber of spatial dimensions associated to pencil data.\n\nThis corresponds to the total number of dimensions of the space, which includes the decomposed and non-decomposed dimensions.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.ndims_extra","page":"Pencils module","title":"PencilFFTs.Pencils.ndims_extra","text":"ndims_extra(x::PencilArray)\n\nNumber of \"extra\" dimensions associated to PencilArray.\n\nThese are the dimensions that are not associated to the domain geometry. For instance, they may correspond to vector or tensor components.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#Base.parent","page":"Pencils module","title":"Base.parent","text":"parent(x::PencilArray)\n\nReturns array wrapped by a PencilArray.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.pencil","page":"Pencils module","title":"PencilFFTs.Pencils.pencil","text":"pencil(x::PencilArray)\n\nReturns decomposition configuration associated to the PencilArray.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.range_local","page":"Pencils module","title":"PencilFFTs.Pencils.range_local","text":"range_local(p::Pencil; permute=true)\n\nLocal data range held by the pencil.\n\nBy default the dimensions are permuted to match those of the associated data arrays.\n\n\n\n\n\nrange_local(x::PencilArray; permute=true)\n\nLocal data range held by the PencilArray.\n\nBy default the dimensions are permuted to match the order of indices in the array.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#Base.size","page":"Pencils module","title":"Base.size","text":"size(t::MPITopology)\n\nGet dimensions of Cartesian topology.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.size_global","page":"Pencils module","title":"PencilFFTs.Pencils.size_global","text":"size_global(p::Pencil)\n\nGlobal dimensions of the Cartesian grid associated to the given domain decomposition.\n\nUnlike size_local, the returned dimensions are not permuted to match the dimensions of the local data.\n\n\n\n\n\nsize_global(x::PencilArray)\n\nGlobal dimensions associated to the given array.\n\nUnlike size, the returned dimensions are not permuted according to the associated pencil configuration.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.size_local","page":"Pencils module","title":"PencilFFTs.Pencils.size_local","text":"size_local(p::Pencil; permute=true)\n\nLocal dimensions of the data held by the pencil.\n\nBy default the dimensions are permuted to match those of the associated data arrays.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#PencilFFTs.Pencils.to_local","page":"Pencils module","title":"PencilFFTs.Pencils.to_local","text":"to_local(p::Pencil, global_inds; permute=true)\n\nConvert non-permuted global indices to local indices.\n\nIndices are permuted by default using the permutation associated to the pencil configuration p.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#LinearAlgebra.transpose!","page":"Pencils module","title":"LinearAlgebra.transpose!","text":"transpose!(dest::PencilArray{T,N}, src::PencilArray{T,N};\n           method=TransposeMethods.IsendIrecv())\n\nTranspose data from one pencil configuration to the other.\n\nThe two pencil configurations must be compatible for transposition:\n\nthey must share the same MPI Cartesian topology,\nthey must have the same global data size,\nwhen written as a sorted tuple, the decomposed dimensions must be almost the same, with at most one difference. For instance, if the input of a 3D dataset is decomposed in (2, 3), then the output may be decomposed in (1, 3), but not in (1, 2). If the decomposed dimensions are the same, then no transposition is performed, and data is just copied if needed.\n\nPerformance tuning\n\nThe method argument allows to choose between transposition implementations. This can be useful to tune performance of MPI data transfers. Two values are currently accepted:\n\nTransposeMethods.IsendIrecv() uses non-blocking point-to-point data transfers (MPI_Isend and MPI_Irecv). This may be more performant since data transfers are interleaved with local data transpositions (index permutation of received data). This is the default.\nTransposeMethods.Alltoallv() uses MPI_Alltoallv for global data transpositions.\n\n\n\n\n\n","category":"function"},{"location":"Pencils/#Pencils.measuring_performance-1","page":"Pencils module","title":"Measuring performance","text":"","category":"section"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"It is possible to measure the time spent in different sections of the MPI data transposition routines using the TimerOutput package. This has a (very small) performance overhead, so it is disabled by default. To enable time measurements, call TimerOutputs.enable_debug_timings(PencilFFTs.Pencils) after loading PencilFFTs. For more details see the TimerOutput docs.","category":"page"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"Minimal example:","category":"page"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"using MPI\nusing PencilFFTs.Pencils\nusing TimerOutput\n\n# Enable timing of `Pencils` functions\nTimerOutputs.enable_debug_timings(PencilFFTs.Pencils)\n\nMPI.Init()\n\npencil = Pencil(#= args... =#)\n\n# [do stuff with `pencil`...]\n\n# Retrieve and print timing data associated to `plan`\nto = get_timer(pencil)\nprint_timer(to)","category":"page"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"By default, each Pencil has its own TimerOutput. If you already have a TimerOutput, you can pass it to the Pencil constructor:","category":"page"},{"location":"Pencils/#","page":"Pencils module","title":"Pencils module","text":"to = TimerOutput()\npencil = Pencil(..., timer=to)\n\n# [do stuff with `pencil`...]\n\nprint_timer(to)","category":"page"},{"location":"Transforms/#Transforms-module-1","page":"Transforms module","title":"Transforms module","text":"","category":"section"},{"location":"Transforms/#","page":"Transforms module","title":"Transforms module","text":"CurrentModule = PencilFFTs.Transforms","category":"page"},{"location":"Transforms/#","page":"Transforms module","title":"Transforms module","text":"Transforms","category":"page"},{"location":"Transforms/#PencilFFTs.Transforms","page":"Transforms module","title":"PencilFFTs.Transforms","text":"Defines different one-dimensional FFT-based transforms.\n\nThe transforms are all subtypes of an AbstractTransform type.\n\nWhen possible, the names of the transforms are kept consistent with the functions exported by AbstractFFTs.jl and FFTW.jl.\n\n\n\n\n\n","category":"module"},{"location":"Transforms/#Transform-types-1","page":"Transforms module","title":"Transform types","text":"","category":"section"},{"location":"Transforms/#","page":"Transforms module","title":"Transforms module","text":"AbstractTransform\n\nNoTransform\n\nFFT\nBFFT\n\nRFFT\nBRFFT\n\nR2R","category":"page"},{"location":"Transforms/#PencilFFTs.Transforms.AbstractTransform","page":"Transforms module","title":"PencilFFTs.Transforms.AbstractTransform","text":"AbstractTransform\n\nSpecifies a one-dimensional FFT-based transform.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#PencilFFTs.Transforms.NoTransform","page":"Transforms module","title":"PencilFFTs.Transforms.NoTransform","text":"NoTransform()\n\nIdentity transform.\n\nSpecifies that no transformation should be applied.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#PencilFFTs.Transforms.FFT","page":"Transforms module","title":"PencilFFTs.Transforms.FFT","text":"FFT()\n\nComplex-to-complex FFT.\n\nSee also AbstractFFTs.fft.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#PencilFFTs.Transforms.BFFT","page":"Transforms module","title":"PencilFFTs.Transforms.BFFT","text":"BFFT()\n\nUnnormalised inverse (backward) complex-to-complex FFT.\n\nLike AbstractFFTs.bfft, this transform is not normalised. To obtain the inverse transform, divide the output by the length of the transformed dimension.\n\nSee also AbstractFFTs.bfft.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#PencilFFTs.Transforms.RFFT","page":"Transforms module","title":"PencilFFTs.Transforms.RFFT","text":"RFFT()\n\nReal-to-complex FFT.\n\nSee also AbstractFFTs.rfft.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#PencilFFTs.Transforms.BRFFT","page":"Transforms module","title":"PencilFFTs.Transforms.BRFFT","text":"BRFFT()\n\nUnnormalised inverse of RFFT.\n\nTo obtain the inverse transform, divide the output by the length of the transformed dimension (of the real output array).\n\nSee also AbstractFFTs.brfft.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#PencilFFTs.Transforms.R2R","page":"Transforms module","title":"PencilFFTs.Transforms.R2R","text":"R2R{kind}()\n\nReal-to-real transform of type kind.\n\nThe possible values of kind are those described in the FFTW.r2r docs and the FFTW manual:\n\ndiscrete cosine transforms: FFTW.REDFT00, FFTW.REDFT01, FFTW.REDFFT10, FFTW.REDFFT11\ndiscrete sine transforms: FFTW.RODFT00, FFTW.RODFT01, FFTW.RODFFT10, FFTW.RODFFT11\ndiscrete Hartley transform: FFTW.DHT\n\nNote: half-complex format DFTs (FFTW.R2HC, FFTW.HC2R) are not currently supported.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#Custom-plans-1","page":"Transforms module","title":"Custom plans","text":"","category":"section"},{"location":"Transforms/#","page":"Transforms module","title":"Transforms module","text":"IdentityPlan","category":"page"},{"location":"Transforms/#PencilFFTs.Transforms.IdentityPlan","page":"Transforms module","title":"PencilFFTs.Transforms.IdentityPlan","text":"IdentityPlan\n\nType of plan associated to NoTransform.\n\n\n\n\n\n","category":"type"},{"location":"Transforms/#Functions-1","page":"Transforms module","title":"Functions","text":"","category":"section"},{"location":"Transforms/#","page":"Transforms module","title":"Transforms module","text":"The following functions are used internally by PencilFFTs.","category":"page"},{"location":"Transforms/#","page":"Transforms module","title":"Transforms module","text":"plan\n\nbinv\nscale_factor\n\neltype_input\neltype_output\nexpand_dims\nkind\nlength_output","category":"page"},{"location":"Transforms/#PencilFFTs.Transforms.plan","page":"Transforms module","title":"PencilFFTs.Transforms.plan","text":"plan(transform::AbstractTransform, A, [dims];\n     flags=FFTW.ESTIMATE, timelimit=Inf)\n\nCreate plan to transform array A along dimensions dims.\n\nIf dims is not specified, all dimensions of A are transformed.\n\nThis function wraps the AbstractFFTs.jl and FFTW.jl plan creation functions. For more details on the function arguments, see AbstractFFTs.plan_fft.\n\n\n\n\n\n","category":"function"},{"location":"Transforms/#PencilFFTs.Transforms.binv","page":"Transforms module","title":"PencilFFTs.Transforms.binv","text":"binv(transform::AbstractTransform)\n\nReturns the backwards transform associated to the given transform.\n\nThe backwards transform returned by this function is not normalised. The normalisation factor for a given array can be obtained by calling scale_factor.\n\nExample\n\njulia> binv(Transforms.FFT())\nBFFT\n\njulia> binv(Transforms.BRFFT())\nRFFT\n\n\n\n\n\n","category":"function"},{"location":"Transforms/#PencilFFTs.Transforms.scale_factor","page":"Transforms module","title":"PencilFFTs.Transforms.scale_factor","text":"scale_factor(transform::AbstractTransform, A, [dims])\n\nGet factor required to normalise the given array after a transformation along dimensions dims (all dimensions by default).\n\nThe array A must have the dimensions of the transform output.\n\nImportant: the dimensions dims must be the same that were passed to plan.\n\nExamples\n\njulia> C = zeros(ComplexF32, 3, 4, 5);\n\njulia> scale_factor(Transforms.FFT(), C)\n60\n\njulia> scale_factor(Transforms.BFFT(), C)\n60\n\njulia> scale_factor(Transforms.BFFT(), C, 2:3)\n20\n\njulia> R = zeros(Float64, 3, 4, 5);\n\njulia> scale_factor(Transforms.BRFFT(), R, 2)\n4\n\njulia> scale_factor(Transforms.BRFFT(), R, 2:3)\n20\n\nThis will fail because the output of RFFT is complex, and R is a real array:\n\njulia> scale_factor(Transforms.RFFT(), R, 2:3)\nERROR: MethodError: no method matching scale_factor(::PencilFFTs.Transforms.RFFT, ::Array{Float64,3}, ::UnitRange{Int64})\n\n\n\n\n\n","category":"function"},{"location":"Transforms/#PencilFFTs.Transforms.eltype_input","page":"Transforms module","title":"PencilFFTs.Transforms.eltype_input","text":"eltype_input(transform::AbstractTransform, real_type<:AbstractFloat)\n\nDetermine input data type for a given transform given the floating point precision of the input data.\n\nFor some transforms such as NoTransform, the input type cannot be identified only from real_type. In this case, Nothing is returned.\n\nExample\n\njulia> eltype_input(Transforms.FFT(), Float32)\nComplex{Float32}\n\njulia> eltype_input(Transforms.RFFT(), Float64)\nFloat64\n\njulia> eltype_input(Transforms.NoTransform(), Float64)\nNothing\n\n\n\n\n\n\n","category":"function"},{"location":"Transforms/#PencilFFTs.Transforms.eltype_output","page":"Transforms module","title":"PencilFFTs.Transforms.eltype_output","text":"eltype_output(transform::AbstractTransform, eltype_input)\n\nReturns the output data type for a given transform given the input type.\n\nThrows ArgumentError if the input data type is incompatible with the transform type.\n\nExample\n\njulia> eltype_output(Transforms.NoTransform(), Float32)\nFloat32\n\njulia> eltype_output(Transforms.RFFT(), Float64)\nComplex{Float64}\n\njulia> eltype_output(Transforms.BRFFT(), ComplexF32)\nFloat32\n\njulia> eltype_output(Transforms.FFT(), Float64)\nERROR: ArgumentError: invalid input data type for PencilFFTs.Transforms.FFT: Float64\n\n\n\n\n\n","category":"function"},{"location":"Transforms/#PencilFFTs.Transforms.expand_dims","page":"Transforms module","title":"PencilFFTs.Transforms.expand_dims","text":"expand_dims(transform::AbstractTransform, Val(N))\n\nExpand a single multidimensional transform into one transform per dimension.\n\nExample\n\n# Expand a real-to-complex transform in 3 dimensions.\njulia> expand_dims(Transforms.RFFT(), Val(3))\n(RFFT, FFT, FFT)\n\njulia> expand_dims(Transforms.BRFFT(), Val(3))\n(BRFFT, BFFT, BFFT)\n\njulia> expand_dims(Transforms.NoTransform(), Val(2))\n(NoTransform, NoTransform)\n\n\n\n\n\n","category":"function"},{"location":"Transforms/#PencilFFTs.Transforms.kind","page":"Transforms module","title":"PencilFFTs.Transforms.kind","text":"kind(transform::R2R)\n\nGet kind of real-to-real transform.\n\n\n\n\n\n","category":"function"},{"location":"Transforms/#PencilFFTs.Transforms.length_output","page":"Transforms module","title":"PencilFFTs.Transforms.length_output","text":"length_output(transform::AbstractTransform, length_in::Integer)\n\nReturns the length of the transform output, given the length of its input.\n\nThe input and output lengths are specified in terms of the respective input and output datatypes. For instance, for real-to-complex transforms, these are respectively the length of input real data and of output complex data.\n\nAlso note that for backward real-to-complex transforms (BRFFT), it is assumed that the real data length is even. See also the AbstractFFTs.irfft docs.\n\n\n\n\n\n","category":"function"},{"location":"#PencilFFTs.jl-1","page":"Home","title":"PencilFFTs.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"CurrentModule = PencilFFTs","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Fast Fourier transforms of MPI-distributed Julia arrays.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This package provides functionality to distribute multidimensional arrays among MPI processes, and to perform multidimensional FFTs (and related transforms) on them.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The name of this package originates from the decomposition of 3D domains along two out of three dimensions, sometimes called pencil decomposition. This is illustrated by the figure below (source), where each coloured block is managed by a different MPI process. Typically, one wants to compute FFTs on a scalar or vector field along the three spatial dimensions. In the case of a pencil decomposition, 3D FFTs are performed one dimension at a time (along the non-decomposed direction, using a serial FFT implementation). Global data transpositions are then needed to switch from one pencil configuration to the other and perform FFTs along the other dimensions.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Image: Pencil decomposition of 3D domains.)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The package is implemented in an efficient generic way that allows to decompose any N-dimensional geometry along M < N dimensions (for the pencil decomposition described above, N = 3 and M = 2). Moreover, the transforms applied along each dimension can be arbitrarily chosen among those supported by FFTW, including complex-to-complex, real-to-complex, and (very soon) real-to-real transforms.","category":"page"},{"location":"#Getting-started-1","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Say you want to perform a 3D FFT of real periodic data defined on N_x  N_y  N_z grid points. The data is to be distributed over 12 MPI processes on a 3  4 grid, as in the figure above.","category":"page"},{"location":"#Creating-plans-1","page":"Home","title":"Creating plans","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The first thing to do is to create a PencilFFTPlan, which requires information on the global dimensions N_x  N_y  N_z of the data, on the transforms that will be applied, and on the way the data is distributed among MPI processes (the MPI Cartesian topology):","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using MPI\nusing PencilFFTs\n\nMPI.Init()\n\n# Input data dimensions (Nx × Ny × Nz)\ndims = (16, 32, 64)\n\n# Apply a real-to-complex (r2c) FFT\ntransform = Transforms.RFFT()\n\n# MPI topology information\ncomm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12\nproc_dims = (3, 4)     # 3×4 Cartesian topology\n\n# Create plan\nplan = PencilFFTPlan(dims, transform, proc_dims, comm)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"See the PencilFFTPlan constructor for details on the accepted options, and the Transforms module for the possible transforms. It is also possible to enable fine-grained performance measurements via the TimerOutputs package, as described in Measuring performance.","category":"page"},{"location":"#Allocating-data-1","page":"Home","title":"Allocating data","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Next, we want to apply the plan on some data. Transforms may only be applied on PencilArrays, which are array wrappers that include MPI decomposition information. The helper function allocate_input may be used to allocate a PencilArray that is compatible with our plan:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"# In our example, this returns a 3D PencilArray of real data (Float64).\nu = allocate_input(plan)\n\n# Fill the array with some (random) data\nusing Random\nrandn!(u)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"PencilArrays are a subtype of AbstractArray, and thus they support all common array operations.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Similarly, to preallocate output data, one can use allocate_output:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"# In our example, this returns a 3D PencilArray of complex data (Complex{Float64}).\nv = allocate_output(plan)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This is only required if one wants to apply the plans using a preallocated output (with mul!, see below).","category":"page"},{"location":"#Applying-plans-1","page":"Home","title":"Applying plans","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The interface to apply plans is consistent with that of AbstractFFTs. Namely, * and mul! are respectively used for forward transforms without and with preallocated output data. Similarly, \\ and ldiv! are used for backward transforms.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using LinearAlgebra  # for mul!, ldiv!\n\n# Apply plan on `u` with `v` as an output\nmul!(v, plan, u)\n\n# Apply backward plan on `v` with `w` as an output\nw = similar(u)\nldiv!(w, plan, v)  # now w ≈ u","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Note that, consistently with AbstractFFTs, backward transforms are normalised, so that the original data is recovered (possibly up to a very small error) when applying a forward followed by a backward transform.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Also note that, at this moment, in-place transforms are not supported.","category":"page"},{"location":"#Implementation-details-1","page":"Home","title":"Implementation details","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The implementation of this package is modular. Distributed FFTs are built on top of the Pencils module that handles data decomposition among MPI processes, including the definition of relevant data structures (e.g. PencilArray) and global data transpositions (see transpose!). As such, the data decomposition functionality may be used independently of the FFTs. See the Pencils module for more details.","category":"page"},{"location":"#Similar-projects-1","page":"Home","title":"Similar projects","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"P3DFFT implements parallel 3D FFTs using pencil decomposition in Fortran and C++.\n2DECOMP&FFT is another parallel 3D FFT library using pencil decomposition written in Fortran.","category":"page"},{"location":"PencilFFTs/#PencilFFTs-module-1","page":"PencilFFTs module","title":"PencilFFTs module","text":"","category":"section"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"CurrentModule = PencilFFTs","category":"page"},{"location":"PencilFFTs/#Types-1","page":"PencilFFTs module","title":"Types","text":"","category":"section"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"PencilFFTPlan","category":"page"},{"location":"PencilFFTs/#PencilFFTs.PencilFFTPlan","page":"PencilFFTs module","title":"PencilFFTs.PencilFFTPlan","text":"PencilFFTPlan{T,N,M}\n\nPlan for N-dimensional FFT-based transform on MPI-distributed data.\n\n\n\nPencilFFTPlan(size_global::Dims{N}, transforms,\n              proc_dims::Dims{M}, comm::MPI.Comm, [real_type=Float64];\n              fftw_flags=FFTW.ESTIMATE, fftw_timelimit=FFTW.NO_TIMELIMIT,\n              permute_dims=Val(true),\n              transpose_method=TransposeMethods.IsendIrecv(),\n              timer=TimerOutput(),\n              )\n\nCreate plan for N-dimensional transform.\n\nsize_global specifies the global dimensions of the input data.\n\ntransforms should be a tuple of length N specifying the transforms to be applied along each dimension. Each element must be a subtype of Transforms.AbstractTransform. For all the possible transforms, see Transform types. Alternatively, transforms may be a single transform that will be automatically expanded into N equivalent transforms. This is illustrated in the example below.\n\nThe transforms are applied one dimension at a time, with the leftmost dimension first for forward transforms. For multidimensional transforms of real data, this means that a real-to-complex transform must be performed along the first dimension, and then complex-to-complex transforms are performed along the other two dimensions (see example below).\n\nThe data is distributed over the MPI processes in the comm communicator. The distribution is performed over M dimensions (with M < N) according to the values in proc_dims, which specifies the number of MPI processes to put along each dimension.\n\nOptional arguments\n\nThe floating point precision can be selected by setting real_type parameter, which is Float64 by default.\nThe keyword arguments fftw_flags and fftw_timelimit are passed to the FFTW plan creation functions (see AbstractFFTs docs).\npermute_dims determines whether the indices of the output data should be reversed. For instance, if the input data has global dimensions (Nx, Ny, Nz), then the output of a complex-to-complex FFT would have dimensions (Nz, Ny, Nx). This enables FFTs to always be performed along the first (i.e. fastest) array dimension, which could lead to performance gains. This option is enabled by default. For type inference reasons, it must be a value type (Val(true) or Val(false)).\ntranspose_method allows to select between implementations of the global data transpositions. See the transpose! docs for details.\ntimer should be a TimerOutput object. See Measuring performance for details.\n\nExample\n\nSuppose we want to perform a 3D transform of real data. The data is to be decomposed along two dimensions, over 8 MPI processes:\n\nsize_global = (64, 32, 128)  # size of real input data\n\n# Perform real-to-complex transform along the first dimension, then\n# complex-to-complex transforms along the other dimensions.\ntransforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())\n# transforms = Transforms.RFFT()  # this is equivalent to the above line\n\nproc_dims = (4, 2)  # 2D decomposition\ncomm = MPI.COMM_WORLD\n\nplan = PencilFFTPlan(size_global, transforms, proc_dims, comm)\n\n\n\n\n\n","category":"type"},{"location":"PencilFFTs/#Functions-1","page":"PencilFFTs module","title":"Functions","text":"","category":"section"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"allocate_input\nallocate_output\nget_timer","category":"page"},{"location":"PencilFFTs/#PencilFFTs.allocate_input","page":"PencilFFTs module","title":"PencilFFTs.allocate_input","text":"allocate_input(p::PencilFFTPlan)\n\nAllocate uninitialised distributed array that can hold input data for the given plan.\n\n\n\n\n\n","category":"function"},{"location":"PencilFFTs/#PencilFFTs.allocate_output","page":"PencilFFTs module","title":"PencilFFTs.allocate_output","text":"allocate_output(p::PencilFFTPlan)\n\nAllocate uninitialised distributed array that can hold output data for the given plan.\n\n\n\n\n\n","category":"function"},{"location":"PencilFFTs/#PencilFFTs.Pencils.get_timer","page":"PencilFFTs module","title":"PencilFFTs.Pencils.get_timer","text":"get_timer(p::Pencil)\n\nGet TimerOutput attached to a Pencil.\n\nSee Measuring performance for details.\n\n\n\n\n\nget_timer(p::PencilFFTPlan)\n\nGet TimerOutput attached to a PencilFFTPlan.\n\nSee Measuring performance for details.\n\n\n\n\n\n","category":"function"},{"location":"PencilFFTs/#PencilFFTs.measuring_performance-1","page":"PencilFFTs module","title":"Measuring performance","text":"","category":"section"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"It is possible to measure the time spent in different sections of the distributed transforms using the TimerOutput package. This has a (very small) performance overhead, so it is disabled by default. To enable time measurements, call TimerOutputs.enable_debug_timings(PencilFFTs) and TimerOutputs.enable_debug_timings(PencilFFTs.Pencils) after loading PencilFFTs. For more details see the TimerOutput docs.","category":"page"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"Minimal example:","category":"page"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"using MPI\nusing PencilFFTs\nusing TimerOutput\n\n# Enable timing of `PencilFFTs` functions\nTimerOutputs.enable_debug_timings(PencilFFTs)\nTimerOutputs.enable_debug_timings(PencilFFTs.Pencils)\n\nMPI.Init()\n\nplan = PencilFFTPlan(#= args... =#)\n\n# [do stuff with `plan`...]\n\n# Retrieve and print timing data associated to `plan`\nto = get_timer(plan)\nprint_timer(to)","category":"page"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"By default, each PencilFFTPlan has its own TimerOutput. If you already have a TimerOutput, you can pass it to the PencilFFTPlan constructor:","category":"page"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"to = TimerOutput()\nplan = PencilFFTPlan(..., timer=to)\n\n# [do stuff with `plan`...]\n\nprint_timer(to)","category":"page"},{"location":"PencilFFTs/#Devdocs-1","page":"PencilFFTs module","title":"Devdocs","text":"","category":"section"},{"location":"PencilFFTs/#","page":"PencilFFTs module","title":"PencilFFTs module","text":"GlobalFFTParams","category":"page"},{"location":"PencilFFTs/#PencilFFTs.GlobalFFTParams","page":"PencilFFTs module","title":"PencilFFTs.GlobalFFTParams","text":"GlobalFFTParams{T, N}\n\nSpecifies the global parameters for an N-dimensional distributed transform. These include the element type T and global data sizes of input and output data, as well as the transform types to be performed along each dimension.\n\n\n\nGlobalFFTParams(size_global, transforms, [real_type=Float64])\n\nDefine parameters for N-dimensional transform.\n\ntransforms must be a tuple of length N specifying the transforms to be applied along each dimension. Each element must be a subtype of Transforms.AbstractTransform. For all the possible transforms, see Transform types.\n\nThe element type must be a real type accepted by FFTW, i.e. either Float32 or Float64.\n\nNote that the transforms are applied one dimension at a time, with the leftmost dimension first for forward transforms.\n\nExample\n\nTo perform a 3D FFT of real data, first a real-to-complex FFT must be applied along the first dimension, followed by two complex-to-complex FFTs along the other dimensions:\n\njulia> size_global = (64, 32, 128);  # size of real input data\n\njulia> transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT());\n\njulia> fft_params = PencilFFTs.GlobalFFTParams(size_global, transforms);\n\n\n\n\n\n","category":"type"}]
}