
echo 1
mpirun -np 1 ./a.out 1> r1.out 2> r1.err
echo 2
mpirun -np 2 ./a.out 1> r2.out 2> r2.err
echo 4
mpirun -np 4 ./a.out 1> r4.out 2> r4.err
echo 8
mpirun -np 8 ./a.out 1> r8.out 2> r8.err
