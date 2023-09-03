
set terminal png
set output "oy.png"
fname="dat/particles.dat"
set logscale y
U = 0.01/32.0
plot fname u 2:(-$13),U