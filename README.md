
# Smoothed Profile Method (SPM) のMPI実装

2023-0830

夏休みなのでゴリゴリに実装してみました．

## Smoothed Profile Method (SPM)

SPMは剛体粒子の位置を，Smoothed Profile (SP)関数$`\phi`$で表す手法です．

剛体粒子の情報は，この$`\phi`$を通して連続体のダイナミクスに還元されます．

粒子$`i`$の中心が座標$`R_i`$にあるとき, 位置$`R`$における関数$`\phi_i(r=|R-R_i|)`$を
```math
  \phi_i(r) = \frac{h((a+\xi/2)-r)}{h((a+\xi/2)-r)+h(r-(a-\xi/2))}
```
とします．ここで，$`a`$は粒子半径, $`\xi`$は粒子ー流体間の界面幅です．
関数$`h(x)`$は格子幅$`\Delta`$で規格化された距離$`r`$に対して．
```math
  h(x) = 
  \begin{cases}
    \exp(-\Delta^2/r^2) & (x>0) \\
    0 & (x\leq 0) \\
  \end{cases}
```
です．

$`\phi_i`$は加成性を持つことを仮定すれば，$`N`$個の粒子に対して，
位置$`R`$におけるSP関数$`\phi_\mathrm{sp}`$は，
```math
  \phi_\mathrm{sp}(R) = \sum_{i=1}^N \phi_i(|R-R_i|)
```
です．

## 支配方程式

剛体粒子の並進と回転運動は次の方程式に従います．
```math
  \frac{d R_i}{dt} = V_i,\quad \frac{d\theta_i}{dt}=\omega_i\\
```
```math
  M\frac{dV_i}{dt}=F_i,\quad I\frac{d\omega_i}{dt}=N_i
```
$`\theta_i`$は回転角, $`V_i`$は速度, $`\omega_i`$は角速度, $`M`$は質量, $`I`$は慣性モーメント, $`F_i`$は力, $`N_i`$はトルクです．


流体の流速$`u`$に対する非圧縮Navie-Stokes 方程式は，SP関数の寄与を考慮して
```math
  \rho(\partial_t+u\cdot\nabla)u = -\nabla p+\eta\nabla^2 u +\rho\phi_\mathrm{sp}f_\mathrm{p}
```
ここで，$`\rho`$: 密度，$`p`$:圧力，$`\eta`$:粘度, $`f_\mathrm{p}`$:粒子の体積力です．

右辺の項を順に，圧力項，粘性項，粒子項と呼ぶことにします．

$`u`$は非圧縮条件$`\nabla\cdot u=0`$を満たします．

簡単のため，$\rho=\eta=1$とします．

## 時間発展の手続き

2次元空間内で，流体をRegular Gridで扱います．格子幅は$\Delta x=\Delta y=\Delta$とします．

時間方向に1次の前進差分を取り，Fractional Step法に沿って手続きを構成します．
(空間方向には中心差分を用いて2次精度で評価するとします)

1. $`n+1`$ステップの粒子位置と角度$`R_i^{n+1}`$, $`\theta_i^{n+1}`$を求める．SP場$`\phi_\mathrm{sp}^{n+1}`$を更新する．

 
2. 粒子項の寄与を除いた仮の速度場$`u^\ast`$を求める．
```math
  \frac{u^\ast-u^n}{\Delta t}=-(u^n\cdot\nabla)u^n -\nabla p^*- \nabla^2 u^n\\
  \mathrm{where}\quad \nabla p^*=\frac{\nabla\cdot u^\ast}{\Delta t}
```

3. $`\Delta t`$の間に粒子$`i`$が流体から受ける力$`F_i^\mathrm{H}`$とトルク$`N_i^\mathrm{H}`$を運動量交換から求め，これにより$`V_i^{n+1}`$と$`\omega_i^{n+1}`$を更新する．
```math
  \int^{t+\Delta t}_t F_i^\mathrm{H} dt = \int\phi_i^{n+1}(u^\ast-u^\ast_{i})dR\\
  \int^{t+\Delta t}_t N_i^\mathrm{H} dt = \int\phi_i^{n+1}(R-R_i^{n+1})\times(u^\ast-u^\ast_{i})dR\\
  \mathrm{where}\quad u^\ast_i=V_i^n+\omega^n_i\times(R-R_i^{n+1})
```

4. 粒子項によって速度場を再度更新し，$`u^{n+1}`$を求める．
```math
  \frac{u^{n+1}-u^\ast}{\Delta t}=\frac{\phi_\mathrm{sp}^{n+1}(u_\mathrm{p}^{n+1}-u^\ast)}{\Delta t}-\nabla p_\mathrm{p}
```
```math
  \mathrm{where}\ \phi_\mathrm{sp}^{n+1}u_\mathrm{p}^{n+1}=\sum_i \phi_i^{n+1}[V_i^{n+1}+\omega^{n+1}_i\times(R-R_i^{n+1})]
```
```math
  \mathrm{and}\ \nabla^2 p_\mathrm{p}=\frac{\nabla\cdot[\phi_\mathrm{sp}^{n+1}(u_\mathrm{p}^{n+1}-u^\ast)]}{\Delta t}
```

## 実装

MPIのため，$`p`$コア並列の場合，$`(N_x/p)\times N_y`$だけ配列を確保します．

通信の際の**のりしろ**となる領域を確保するため$`N_x/p+2`$として両端のところは逐次，両隣のプロセスとの通信に使用します．

粒子情報は全プロセスで共有されます．

1 iterationにつき2回，ポアソン方程式をJacobiの反復法で解きます．


