
set terminal pngcairo
set output "p.png"
set logscale y
fname="dat/particles.dat"
L=100.0
U=0.01
plot fname u 2:($1==0 ? $9 : 1/0) w lp, \
     fname u 2:($1==1 ? $9 : 1/0) w lp, \
     U/L

