#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ctime>
#include <chrono>
#include <cmath>
#include <limits>

// parameters
namespace {
  static const int X = 0;
  static const int Y = 1;
  static const int Z = 2;
  static const int NX = 32;
  static const int NY = 32;
  static const int NZ = 32;
  static const double DT = 0.01; 
  static const int MAX_STEP = 10000;
  static const int DIM = 3;
  static const double TOLERANCE = 1e-5;
  static const double SOR_COEFF = 0.9;

  static const int NP = 1;
  static const double XI_SPM = 2.0;
  static const double V_WALL_U[DIM] = {-0.01,0.0,0.0};
  static const double V_WALL_L[DIM] = {+0.01,0.0,0.0};
}

inline double 
relpos(const double from,const double to, const double L){
  double r = to - from;
  if( r > +0.5*L ) r -= L;
  if( r < -0.5*L ) r += L;
  return r;
}

inline double 
pbc(double x, const double l, const double r){
  if( x > r ) x -= (r-l);
  if( x < l ) x += (r-l);
  return x;
}

double 
spmh(const double x){
  return (x>0.0)? std::exp(-1.0/(x*x)):0.0;
}
double 
spmf(const double r, const double rc, const double xi) {
  return spmh((rc+xi*0.5)-r) / (spmh((rc+xi*0.5)-r) + spmh(r-(rc-xi*0.5)));
}

int 
main(int argc, char** argv)
{
  auto time_start = std::chrono::system_clock::now(); //
  FILE* fpp = std::fopen("dat/particles.dat","w");

  // allocation
  auto u_alloc     = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) * (NZ+2) * DIM );
  auto phipup_alloc= std::make_unique<double[]>(     (NX+2) * (NY+2) * (NZ+2) * DIM );
  auto u_tmp_alloc = std::make_unique<double[]>(     (NX+2) * (NY+2) * (NZ+2) * DIM );
  auto p_alloc     = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) * (NZ+2)       );
  auto pp_alloc    = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) * (NZ+2)       );
  auto phi_p_alloc = std::make_unique<double[]>(     (NX+2) * (NY+2) * (NZ+2)       );

  auto u     = reinterpret_cast<double(&)[2][NX+2][NY+2][NZ+2][DIM]>(*     u_alloc.get());
  auto phipup= reinterpret_cast<double(&)   [NX+2][NY+2][NZ+2][DIM]>(*phipup_alloc.get());
  auto u_tmp = reinterpret_cast<double(&)   [NX+2][NY+2][NZ+2][DIM]>(* u_tmp_alloc.get());
  auto p     = reinterpret_cast<double(&)[2][NX+2][NY+2][NZ+2]     >(*     p_alloc.get());
  auto pp    = reinterpret_cast<double(&)[2][NX+2][NY+2][NZ+2]     >(*    pp_alloc.get());
  auto phi_p = reinterpret_cast<double(&)   [NX+2][NY+2][NZ+2]     >(* phi_p_alloc.get());

  double rp[NP];
  double mp[NP];
  double Ip[NP];
  double xp[NP][DIM];
  double vp[NP][DIM];
  double fp[NP][DIM];
  double omegap[NP][DIM*(DIM-1)/2];
  double torp[NP][DIM*(DIM-1)/2];
  double angp[NP][DIM*(DIM-1)/2];

  for(int p = 0; p < NP; p++){
    rp[p] = 2;
    double dp = 2.0*rp[p];
    mp[p] = M_PI*dp*dp*dp/6.0;
    Ip[p] = 0.4*mp[p]*rp[p]*rp[p];
    // xp[p][X] = (p*1.0/NP)*NX;//NX/2;
    xp[p][X] = NX/2;
    xp[p][Y] = NY/2;//NY/2;
    xp[p][Z] = NZ/2;//NZ/2;
    vp[p][X] = 0.0;
    vp[p][Y] = 0.0;
    vp[p][Z] = 0.0;
    fp[p][X] = 0.0;
    fp[p][Y] = 0.0;    
    fp[p][Z] = 0.0;    
    omegap[p][X] = 0.0;
    omegap[p][Y] = 0.0;
    omegap[p][Z] = 0.0;
    torp[p][X] = 0.0;
    torp[p][Y] = 0.0;
    torp[p][Z] = 0.0;
    angp[p][X] = 0.0;
    angp[p][Y] = 0.0;
    angp[p][Z] = 0.0;
  }

  // initialize
  for(int i = 0; i < 2*(NX+2)*(NY+2)*(NZ+2)*DIM; i++){
    u_alloc.get()[i] = 0.0;
  }   
  for(int i = 0; i < 2*(NX+2)*(NY+2)*(NZ+2); i++){
    p_alloc.get()[i] = 0.0;
    pp_alloc.get()[i] = 0.0;
  }
  for(int i = 1; i <= NX; i++) {
    for(int j = 1; j <= NY; j++) {
     for(int k = 1; k <= NZ; k++) {
        double x = 0.5+(i-1);
        double y = 0.5+(j-1);
        double z = 0.5+(k-1);
        u[0][i][j][k][X] = 0.0;
        u[0][i][j][k][Y] = 0.0;
        u[0][i][j][k][Z] = 0.0;
      }
    }
  }

  // boundary condition (x,y,z)
  for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
    for(int d = 0; d < DIM; d++) {
      u[0][0   ][j][k][d] = u[0][NX][j][k][d];
      u[0][NX+1][j][k][d] = u[0][1 ][j][k][d];
  }}}
  for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
    for(int d = 0; d < DIM; d++) {
      u[0][i][0   ][k][d] = u[0][i][NY][k][d];
      u[0][i][NY+1][k][d] = u[0][i][1 ][k][d];
  }}}
  for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
    for(int d = 0; d < DIM; d++) {
      u[0][i][j][0   ][d] = 2.0*V_WALL_L[d] - u[0][i][j][1 ][d]; //u[0][i][j][NZ][d];
      u[0][i][j][NZ+1][d] = 2.0*V_WALL_U[d] - u[0][i][j][NZ][d]; //u[0][i][j][1 ][d];
  }}}


  // main loop
  double time = 0.0;
  for(int step = 0; step < MAX_STEP; step++) {
    // std::cout << step << std::endl;
    const int tn = step%2 == 1;
    const int tp = step%2 == 0;
    auto& un = u[tn];
    auto& up = u[tp];

    // 1. update particle position 
    for (int p = 0; p < NP; p++){
      for (int d = 0; d < DIM; d++){
        xp[p][d] = xp[p][d] + DT * vp[p][d] + (0.5*(DT*DT)/mp[p])*(fp[p][d]);      
      }  
      xp[p][X] = pbc(xp[p][X],0.0,NX);
      xp[p][Y] = pbc(xp[p][Y],0.0,NY);
      xp[p][Z] = pbc(xp[p][Z],0.0,NZ);
    }
    for(int i = 0; i < NX+2; i++){ 
      for(int j = 0; j < NY+2; j++){
        for(int k = 0; k < NZ+2; k++){
          phi_p[i][j][k] = 0.0;
    }}}
    for (int p = 0; p < NP; p++){
      const int il = xp[p][X] - rp[p] - XI_SPM      ; 
      const int ir = xp[p][X] + rp[p] + XI_SPM + 0.5; 
      const int jl = xp[p][Y] - rp[p] - XI_SPM      ; 
      const int jr = xp[p][Y] + rp[p] + XI_SPM + 0.5; 
      const int kl = xp[p][Z] - rp[p] - XI_SPM      ; 
      const int kr = xp[p][Z] + rp[p] + XI_SPM + 0.5; 
      for (int i = il; i <= ir; i++){
        const int ii = (i+NX)%NX+1;
        const double x = ii - 0.5;
        for (int j = jl; j <= jr; j++){
          const int jj = (j+NY)%NY+1;
          const double y = jj - 0.5;
          for (int k = kl; k <= kr; k++){
            const int kk = (k+NZ)%NZ+1;
            const double z = kk - 0.5;
            const double rx = relpos(xp[p][X],x,NX), 
              ry = relpos(xp[p][Y],y,NY),
              rz = relpos(xp[p][Z],z,NZ);
            const double r = std::sqrt( rx*rx + ry*ry + rz*rz);
            phi_p[ii][jj][kk] += spmf(r,rp[p],XI_SPM);
          }
        }
      }
    }
    // boundary condition (x,y,z)
    for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
      phi_p[0   ][j][k] = phi_p[NX][j][k];
      phi_p[NX+1][j][k] = phi_p[1 ][j][k];
    }}
    for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
      phi_p[i][0   ][k] = phi_p[i][NY][k];
      phi_p[i][NY+1][k] = phi_p[i][1 ][k];
    }}
    for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
      phi_p[i][j][0   ] = phi_p[i][j][NZ];
      phi_p[i][j][NZ+1] = phi_p[i][j][1 ];
    }}



    // fractional step w/o pressure term
    for(int i = 1; i <= NX; i++){ for(int j = 1; j <= NY; j++){ for(int k = 1; k <= NZ; k++){
      u_tmp[i][j][k][X] = un[i][j][k][X] 
                + DT * ( 
      // diffusion
      - 6.0 * un[i][j][k][X] + un[i-1][j][k][X] + un[i+1][j][k][X] + un[i][j-1][k][X] + un[i][j+1][k][X] + un[i][j][k-1][X] + un[i][j][k+1][X] 
      // advection (div. form)
      - 0.5 * ( un[i+1][j][k][X]*un[i+1][j][k][X] - un[i-1][j][k][X]*un[i-1][j][k][X]) // d/dx(ux^2) 
      - 0.5 * ( un[i][j+1][k][X]*un[i][j+1][k][Y] - un[i][j-1][k][X]*un[i][j-1][k][Y]) // d/dy(ux*uy)
      - 0.5 * ( un[i][j][k+1][X]*un[i][j][k+1][Z] - un[i][j][k-1][X]*un[i][j][k-1][Z]) // d/dz(ux*uz)
                );
      u_tmp[i][j][k][Y] = un[i][j][k][Y] 
                + DT * ( 
      // diffusion
      - 6.0 * un[i][j][k][Y] + un[i-1][j][k][Y] + un[i+1][j][k][Y] + un[i][j-1][k][Y] + un[i][j+1][k][Y] + un[i][j][k-1][Y] + un[i][j][k+1][Y] 
      // advection (div. form)
      - 0.5 * ( un[i+1][j][k][X]*un[i+1][j][k][Y] - un[i-1][j][k][X]*un[i-1][j][k][Y]) // d/dx(ux*uy) 
      - 0.5 * ( un[i][j+1][k][Y]*un[i][j+1][k][Y] - un[i][j-1][k][Y]*un[i][j-1][k][Y]) // d/dy(uy^2)
      - 0.5 * ( un[i][j][k+1][Y]*un[i][j][k+1][Z] - un[i][j][k-1][Y]*un[i][j][k-1][Z]) // d/dy(uy*uz)
                );
      u_tmp[i][j][k][Z] = un[i][j][k][Z] 
                + DT * ( 
      // diffusion
      - 6.0 * un[i][j][k][Z] + un[i-1][j][k][Z] + un[i+1][j][k][Z] + un[i][j-1][k][Z] + un[i][j+1][k][Z] + un[i][j][k-1][Z] + un[i][j][k+1][Z] 
      // advection (div. form)
      - 0.5 * ( un[i+1][j][k][X]*un[i+1][j][k][Z] - un[i-1][j][k][X]*un[i-1][j][k][Z]) // d/dx(ux*uz) 
      - 0.5 * ( un[i][j+1][k][Y]*un[i][j+1][k][Z] - un[i][j-1][k][Y]*un[i][j-1][k][Z]) // d/dy(uy*yz)
      - 0.5 * ( un[i][j][k+1][Z]*un[i][j][k+1][Z] - un[i][j][k-1][Z]*un[i][j][k-1][Z]) // d/dy(uz^2)
                );
    }}}

    // boundary condition (x,y,z)
    for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++) {
        u_tmp[0   ][j][k][d] = u_tmp[NX][j][k][d];
        u_tmp[NX+1][j][k][d] = u_tmp[1 ][j][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++) {
        u_tmp[i][0   ][k][d] = u_tmp[i][NY][k][d];
        u_tmp[i][NY+1][k][d] = u_tmp[i][1 ][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
      for(int d = 0; d < DIM; d++) {
        u_tmp[i][j][0   ][d] = 2.0*V_WALL_L[d] - u_tmp[i][j][1 ][d]; //u[0][i][j][NZ][d];
        u_tmp[i][j][NZ+1][d] = 2.0*V_WALL_U[d] - u_tmp[i][j][NZ][d]; //u[0][i][j][1 ][d];
    }}}

    // solve poisson equation with jacobi method
    const int max_iter = 10000;
    for (int itr = 0; itr < max_iter; itr++) {    
      const double COEFF = - 0.5 / DT;
      for(int i = 1; i <= NX; i++){for(int j = 1; j <= NY; j++){for(int k = 1; k <= NZ; k++){
        const double D = (k==1||k==NZ)? 1.0/5.0 : 1.0/6.0;
        const double kpv = (k==NZ)? 0.0:1.0;
        const double kmv = (k==1 )? 0.0:1.0;
  p[tp][i][j][k] = 
    + D*COEFF * (u_tmp[i+1][j][k][X] - u_tmp[i-1][j][k][X] + u_tmp[i][j+1][k][Y] - u_tmp[i][j-1][k][Y] + u_tmp[i][j][k+1][Z] - u_tmp[i][j][k-1][Z])  // b/D
    + D       * ( p[tn][i+1][j][k] + p[tn][i-1][j][k] + p[tn][i][j+1][k] + p[tn][i][j-1][k] + kpv*p[tn][i][j][k+1] + kmv*p[tn][i][j][k-1] ); // -A*x/D
      }}}
      double sum = 0.0, residue = 0.0;
      for(int i = 1; i <= NX; i++){ for(int j = 1; j <= NY; j++){ for(int k = 1; k <= NZ; k++){
        double pnew = (1.0-SOR_COEFF)*p[tn][i][j][k] + SOR_COEFF*p[tp][i][j][k];
        double e = pnew - p[tn][i][j][k];
        residue += std::abs(e   );
        sum     += std::abs(pnew);
        p[tn][i][j][k] = pnew;
      }}}
      // boundary condition (x,y,z)
      for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
        p[tn][0   ][j][k] = p[tn][NX][j][k];
        p[tn][NX+1][j][k] = p[tn][1 ][j][k];
      }}
      for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
        p[tn][i][0   ][k] = p[tn][i][NY][k];
        p[tn][i][NY+1][k] = p[tn][i][1 ][k];
      }}
      for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
        p[tn][i][j][0   ] = p[tn][i][j][NZ];
        p[tn][i][j][NZ+1] = p[tn][i][j][1 ];
      }}

      if( residue <= sum*TOLERANCE ) break;
      if( (itr == max_iter-1) || !std::isfinite(residue) ) {
        std::cerr << "ERROR: not converged: poisson solver 1" << std::endl;
        std::cerr << "step: " << step << std::endl;
        std::cerr << "iterator: " << itr << std::endl;
        std::cerr << "residue: " << residue << "/" << sum << std::endl;
        return 1;
      }
    }
    for(int i = 0; i < NX+2; i++){ for(int j = 0; j < NY+2; j++){ for(int k = 0; k < NZ+2; k++){
      p[tp][i][j][k] = p[tn][i][j][k];
    }}}

    // update velocity
    for(int i = 1; i <= NX; i++){ for(int j = 1; j <= NY; j++){ for(int k = 1; k <= NZ; k++){
      u_tmp[i][j][k][X] += - DT * 0.5*(p[tp][i+1][j][k] - p[tp][i-1][j][k]);
      u_tmp[i][j][k][Y] += - DT * 0.5*(p[tp][i][j+1][k] - p[tp][i][j-1][k]);
      u_tmp[i][j][k][Z] += - DT * 0.5*(p[tp][i][j][k+1] - p[tp][i][j][k-1]);
    }}}

    // boundary condition
    for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++) {
        u_tmp[0   ][j][k][d] = u_tmp[NX][j][k][d];
        u_tmp[NX+1][j][k][d] = u_tmp[1 ][j][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++) {
        u_tmp[i][0   ][k][d] = u_tmp[i][NY][k][d];
        u_tmp[i][NY+1][k][d] = u_tmp[i][1 ][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
      for(int d = 0; d < DIM; d++) {
        u_tmp[i][j][0   ][d] = 2.0*V_WALL_L[d] - u_tmp[i][j][1 ][d]; //u[0][i][j][NZ][d];
        u_tmp[i][j][NZ+1][d] = 2.0*V_WALL_U[d] - u_tmp[i][j][NZ][d]; //u[0][i][j][1 ][d];
    }}}


    // 3. exchange momenta of particles and fluids
    for (int p = 0; p < NP; p++){
      for (int d = 0; d < DIM; d++){
        fp[p][d] = 0.0;
        torp[p][d] = 0.0;
      }
    }
    for (int p = 0; p < NP; p++){
      const int il = xp[p][X] - rp[p] - XI_SPM      ; 
      const int ir = xp[p][X] + rp[p] + XI_SPM + 0.5; 
      const int jl = xp[p][Y] - rp[p] - XI_SPM      ; 
      const int jr = xp[p][Y] + rp[p] + XI_SPM + 0.5; 
      const int kl = xp[p][Z] - rp[p] - XI_SPM      ; 
      const int kr = xp[p][Z] + rp[p] + XI_SPM + 0.5; 
      for (int i = il; i <= ir; i++){
        const int ii = (i+NX)%NX+1;
        const double x = ii - 0.5;
        for (int j = jl; j <= jr; j++){
          const int jj = (j+NY)%NY+1;
          const double y = jj - 0.5;
          for (int k = kl; k <= kr; k++){
            const int kk = (k+NZ)%NZ+1;
            const double z = kk - 0.5;
            const double rx = relpos(xp[p][X],x,NX), 
              ry = relpos(xp[p][Y],y,NY),
              rz = relpos(xp[p][Z],z,NZ);
            const double r = std::sqrt( rx*rx + ry*ry + rz*rz);
            double phipp = spmf(r,rp[p],XI_SPM);
            // XXX for 3d
            double phippdux = phipp * (u_tmp[ii][jj][kk][X] - (vp[p][X] + (omegap[p][Y]*rz - omegap[p][Z]*ry))); 
            double phippduy = phipp * (u_tmp[ii][jj][kk][Y] - (vp[p][Y] + (omegap[p][Z]*rx - omegap[p][X]*rz))); 
            double phippduz = phipp * (u_tmp[ii][jj][kk][Z] - (vp[p][Z] + (omegap[p][X]*ry - omegap[p][Y]*rx))); 
            fp[p][X] += phippdux;
            fp[p][Y] += phippduy;
            fp[p][Z] += phippduz;
            torp[p][X] += ry * phippduz - rz * phippduy;
            torp[p][Y] += rz * phippdux - rx * phippduz;
            torp[p][Z] += rx * phippduy - ry * phippdux;
          }
        }
      }
      const double invdt = 1.0/DT;
      fp[p][X] *= invdt;
      fp[p][Y] *= invdt;
      fp[p][Z] *= invdt;
      torp[p][X] *= invdt;
      torp[p][Y] *= invdt;
      torp[p][Z] *= invdt;
    }

    // 4. update particle's velocity and angular vel
    for (int p = 0; p < NP; p++){
      double invm = 1.0/mp[p];
      double invI = 1.0/Ip[p];
      vp[p][X] += invm * DT * fp[p][X];
      vp[p][Y] += invm * DT * fp[p][Y];
      vp[p][Z] += invm * DT * fp[p][Z];
      omegap[p][X] += invI * DT * torp[p][X];
      omegap[p][Y] += invI * DT * torp[p][Y];
      omegap[p][Z] += invI * DT * torp[p][Z];
    }        
    // phipup
    for(int i = 0; i < NX+2; i++){
      for(int j = 0; j < NY+2; j++){
        for(int k = 0; k < NZ+2; k++){
          phipup[i][j][k][X] = 0.0;
          phipup[i][j][k][Y] = 0.0;
          phipup[i][j][k][Z] = 0.0;
    }}}

    for (int p = 0; p < NP; p++){
      const int il = xp[p][X] - rp[p] - XI_SPM      ; 
      const int ir = xp[p][X] + rp[p] + XI_SPM + 0.5; 
      const int jl = xp[p][Y] - rp[p] - XI_SPM      ; 
      const int jr = xp[p][Y] + rp[p] + XI_SPM + 0.5; 
      const int kl = xp[p][Z] - rp[p] - XI_SPM      ; 
      const int kr = xp[p][Z] + rp[p] + XI_SPM + 0.5; 
      for (int i = il; i <= ir; i++){
        const int ii = (i+NX)%NX+1;
        const double x = ii - 0.5;
        for (int j = jl; j <= jr; j++){
          const int jj = (j+NY)%NY+1;
          const double y = jj - 0.5;
          for (int k = kl; k <= kr; k++){
            const int kk = (k+NZ)%NZ+1;
            const double z = kk - 0.5;
            const double rx = relpos(xp[p][X],x,NX), 
              ry = relpos(xp[p][Y],y,NY),
              rz = relpos(xp[p][Z],z,NZ);
            const double r = std::sqrt( rx*rx + ry*ry + rz*rz);
            double phipp = spmf(r,rp[p],XI_SPM);
            phipup[ii][jj][kk][X] += phipp * (vp[p][X] + (omegap[p][Y]*rz - omegap[p][Z]*ry));
            phipup[ii][jj][kk][Y] += phipp * (vp[p][Y] + (omegap[p][Z]*rx - omegap[p][X]*rz));
            phipup[ii][jj][kk][Z] += phipp * (vp[p][Z] + (omegap[p][X]*ry - omegap[p][Y]*rx));
          }
        }
      }
    }
    // boundary condition (x,y,z)
    for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++){
        phipup[0   ][j][k][d] = phipup[NX][j][k][d];
        phipup[NX+1][j][k][d] = phipup[1 ][j][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++){
        phipup[i][0   ][k][d] = phipup[i][NY][k][d];
        phipup[i][NY+1][k][d] = phipup[i][1 ][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
      for(int d = 0; d < DIM; d++){
        phipup[i][j][0   ][d] = phipup[i][j][NZ][d];
        phipup[i][j][NZ+1][d] = phipup[i][j][1 ][d];
    }}}

    // 5. update velocity of next step
    for (int itr = 0; itr < max_iter; itr++) {    
      const double COEFF = - 0.5 / DT;
      for(int i = 1; i <= NX; i++){for(int j = 1; j <= NY; j++){for(int k = 1; k <= NZ; k++){
        const double D = (k==1||k==NZ)? 1.0/5.0 : 1.0/6.0;
        const double kpv = (k==NZ)? 0.0:1.0;
        const double kmv = (k==1 )? 0.0:1.0;
  pp[tp][i][j][k] = 
    + D*COEFF * (phipup[i+1][j][k][X] - phipup[i-1][j][k][X] + phipup[i][j+1][k][Y] - phipup[i][j-1][k][Y] + phipup[i][j][k+1][Z] - phipup[i][j][k-1][Z])  // b/D
    - D*COEFF * (
            + phi_p[i+1][j][k]*u_tmp[i+1][j][k][X] - phi_p[i-1][j][k]*u_tmp[i-1][j][k][X] 
            + phi_p[i][j+1][k]*u_tmp[i][j+1][k][Y] - phi_p[i][j-1][k]*u_tmp[i][j-1][k][Y] 
            + phi_p[i][j][k+1]*u_tmp[i][j][k+1][Z] - phi_p[i][j][k-1]*u_tmp[i][j][k-1][Z])  // b/D
    + D       * ( pp[tn][i+1][j][k] + pp[tn][i-1][j][k] + pp[tn][i][j+1][k] + pp[tn][i][j-1][k] + kpv*pp[tn][i][j][k+1] + kmv*pp[tn][i][j][k-1] ); // -A*x/D
      }}}
      double sum = 0.0, residue = 0.0;
      for(int i = 1; i <= NX; i++){ for(int j = 1; j <= NY; j++){ for(int k = 1; k <= NZ; k++){
        double pnew = (1.0-SOR_COEFF)*pp[tn][i][j][k] + SOR_COEFF*pp[tp][i][j][k];
        double e = pnew - pp[tn][i][j][k];
        residue += std::abs(e   );
        sum     += std::abs(pnew);
        pp[tn][i][j][k] = pnew;
      }}}
      // boundary condition (x,y,z)
      for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
        pp[tn][0   ][j][k] = pp[tn][NX][j][k];
        pp[tn][NX+1][j][k] = pp[tn][1 ][j][k];
      }}
      for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
        pp[tn][i][0   ][k] = pp[tn][i][NY][k];
        pp[tn][i][NY+1][k] = pp[tn][i][1 ][k];
      }}
      for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
        pp[tn][i][j][0   ] = pp[tn][i][j][NZ];
        pp[tn][i][j][NZ+1] = pp[tn][i][j][1 ];
      }}

      if( residue <= sum*TOLERANCE ) break;
      if( (itr == max_iter-1) || !std::isfinite(residue) ) {
        std::cerr << "ERROR: not converged: poisson solver 2" << std::endl;
        std::cerr << "step: " << step << std::endl;
        std::cerr << "iterator: " << itr << std::endl;
        std::cerr << "residue: " << residue << "/" << sum << std::endl;
        return 1;
      }
    }
    for(int i = 0; i < NX+2; i++){ for(int j = 0; j < NY+2; j++){ for(int k = 0; k < NZ+2; k++){
      pp[tp][i][j][k] = pp[tn][i][j][k];
    }}}

    // update velocity
    for(int i = 1; i <= NX; i++) {
      for(int j = 1; j <= NY; j++) {
        for(int k = 1; k <= NZ; k++) {
          up[i][j][k][X] = u_tmp[i][j][k][X] + phipup[i][j][k][X] - phi_p[i][j][k]*u_tmp[i][j][k][X] - DT * 0.5*(pp[tn][i+1][j][k] - pp[tn][i-1][j][k]);
          up[i][j][k][Y] = u_tmp[i][j][k][Y] + phipup[i][j][k][Y] - phi_p[i][j][k]*u_tmp[i][j][k][Y] - DT * 0.5*(pp[tn][i][j+1][k] - pp[tn][i][j-1][k]);
          up[i][j][k][Z] = u_tmp[i][j][k][Z] + phipup[i][j][k][Z] - phi_p[i][j][k]*u_tmp[i][j][k][Z] - DT * 0.5*(pp[tn][i][j][k+1] - pp[tn][i][j][k-1]);
    }}}

    // boundary condition
    for(int j = 1; j <= NY; j++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++) {
        up[0   ][j][k][d] = up[NX][j][k][d];
        up[NX+1][j][k][d] = up[1 ][j][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int k = 1; k <= NZ; k++) {
      for(int d = 0; d < DIM; d++) {
        up[i][0   ][k][d] = up[i][NY][k][d];
        up[i][NY+1][k][d] = up[i][1 ][k][d];
    }}}
    for(int i = 1; i <= NX; i++) { for(int j = 1; j <= NY; j++) {
      for(int d = 0; d < DIM; d++) {
        up[i][j][0   ][d] = 2.0*V_WALL_L[d] - up[i][j][1 ][d]; //u[0][i][j][NZ][d];
        up[i][j][NZ+1][d] = 2.0*V_WALL_U[d] - up[i][j][NZ][d]; //u[0][i][j][1 ][d];
    }}}




    time += DT;
    // if(((step+1) % (MAX_STEP/10))==0){
    //   std::cerr << step+1 << std::endl;
    // }

    // write
    if(true) {
      std::fprintf(stderr,"%lf\n",time);
      for (int p = 0; p < NP; p++){
        std::fprintf(fpp,"%d %lf" 
        " %.8lf %.8lf %.8lf" 
        " %.8lf %.8lf %.8lf" 
        " %.8lf %.8lf %.8lf" 
        " %.8lf %.8lf %.8lf" 
        "\n",
        p,
        time,
        xp[p][X],xp[p][Y],xp[p][Z],
        vp[p][X],vp[p][Y],vp[p][Z],
        fp[p][X],fp[p][Y],fp[p][Z],
        omegap[p][X],omegap[p][Y],omegap[p][Z]);
      }
      std::fflush(fpp);
    }
  } // for step

  std::fclose(fpp);

  auto time_end = std::chrono::system_clock::now();  // 
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
  std::cerr << "time = " << elapsed << " [ms]" << std::endl;
  return 0;
}