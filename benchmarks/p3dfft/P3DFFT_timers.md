# P3DFFT timers

## Forward transforms

See `p3dfft_ftran_r2c` in `build/ftran.F90`.

 Timer |    Subroutine     |  What
-------|-------------------|-----------------------------
   1   |   `fcomm1`        | Alltoallv (X -> Y)
   2   |   `fcomm2_trans`  | Alltoallv (Y -> Z)
   5   |   `exec_f_r2c`    | r2c FFT in X
   6   |   `fcomm1`        | pack + unpack data (X -> Y)
   7   |   `exec_f_c1`     | c2c FFT in Y
   8   |   `fcomm2_trans`  | pack data + unpack + c2c FFT in Z

## Backward transforms

See `p3dfft_btran_c2r` in `build/btran.F90`.

Timer  |    Subroutine     |  What
-------|-------------------|-----------------------------
3      |   `bcomm1_trans`  | Alltoallv (Y <- Z)
4      |   `bcomm2`        | Alltoallv (X <- Y)
9      |   `bcomm1_trans`  | c2c FFT in Z + pack data + unpack
10     |   `exec_b_c1`     | c2c FFT in Y
11     |   `bcomm2`        | pack + unpack data (X <- Y)
12     |   `exec_b_c2r`    | c2r FFT in X
