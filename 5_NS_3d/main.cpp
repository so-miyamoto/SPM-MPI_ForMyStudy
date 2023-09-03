#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ctime>
#include <cmath>
#include <limits>

// parameters
namespace {
  static const int X = 0;
  static const int Y = 1;
  static const int Z = 2;
  static const int NX = 16;
  static const int NY = 64;
  static const int NZ = 64;
  static const double DT = 0.001; 
  static const int MAX_STEP = 32000;
  static const int DIM = 3;
  static const double TOLERANCE = 1e-8;
  static const double SOR_COEFF = 1.5;
}

int main(int argc, char** argv)
{
  auto time_start = std::chrono::system_clock::now(); //

  // allocation
  auto u_alloc = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) * (NZ+2)* DIM );
  auto p_alloc = std::make_unique<double[]>( 2 * (NX+2) * (NY+2) * (NZ+2)      );
  auto u = reinterpret_cast<double(&)[2][NX+2][NY+2][NZ+2][DIM]>(*u_alloc.get());
  auto p = reinterpret_cast<double(&)[2][NX+2][NY+2][NZ+2]     >(*p_alloc.get());

  // initialize
  for(int i = 0; i < 2*(NX+2)*(NY+2)*DIM; i++){
    u_alloc.get()[i] = 0.0;
  }   
  for(int i = 0; i < 2*(NX+2)*(NY+2); i++){
    p_alloc.get()[i] = 0.0;
  }
  for(int i = 1; i <= NX; i++) {
    for(int j = 1; j <= NY; j++) {
     for(int k = 1; k <= NZ; k++) {
        double x = 0.5+(i-1);
        double y = 0.5+(j-1);
        double z = 0.5+(k-1);
        u[0][i][j][k][X] = 0.0;
        u[0][i][j][k][Y] = - std::sin(y/NY*2*M_PI)*std::cos(z/NZ*2*M_PI);
        u[0][i][j][k][Z] =   std::cos(y/NY*2*M_PI)*std::sin(z/NZ*2*M_PI);
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
      u[0][i][j][0   ][d] = u[0][i][j][NZ][d];
      u[0][i][j][NZ+1][d] = u[0][i][j][1 ][d];
  }}}


  // main loop
  double time = 0.0;
  for(int step = 0; step < MAX_STEP; step++) {
    // std::cout << step << std::endl;
    const int tn = step%2 == 1;
    const int tp = step%2 == 0;
    auto& un = u[tn];
    auto& up = u[tp];

    // fractional step w/o pressure term
    for(int i = 1; i <= NX; i++){ for(int j = 1; j <= NY; j++){ for(int k = 1; k <= NZ; k++){
      up[i][j][k][X] = un[i][j][k][X] 
                + DT * ( 
      // diffusion
      - 6.0 * un[i][j][k][X] + un[i-1][j][k][X] + un[i+1][j][k][X] + un[i][j-1][k][X] + un[i][j+1][k][X] + un[i][j][k-1][X] + un[i][j][k+1][X] 
      // advection (div. form)
      - 0.5 * ( un[i+1][j][k][X]*un[i+1][j][k][X] - un[i-1][j][k][X]*un[i-1][j][k][X]) // d/dx(ux^2) 
      - 0.5 * ( un[i][j+1][k][X]*un[i][j+1][k][Y] - un[i][j-1][k][X]*un[i][j-1][k][Y]) // d/dy(ux*uy)
      - 0.5 * ( un[i][j][k+1][X]*un[i][j][k+1][Z] - un[i][j][k-1][X]*un[i][j][k-1][Z]) // d/dz(ux*uz)
                );
      up[i][j][k][Y] = un[i][j][k][Y] 
                + DT * ( 
      // diffusion
      - 6.0 * un[i][j][k][Y] + un[i-1][j][k][Y] + un[i+1][j][k][Y] + un[i][j-1][k][Y] + un[i][j+1][k][Y] + un[i][j][k-1][Y] + un[i][j][k+1][Y] 
      // advection (div. form)
      - 0.5 * ( un[i+1][j][k][X]*un[i+1][j][k][Y] - un[i-1][j][k][X]*un[i-1][j][k][Y]) // d/dx(ux*uy) 
      - 0.5 * ( un[i][j+1][k][Y]*un[i][j+1][k][Y] - un[i][j-1][k][Y]*un[i][j-1][k][Y]) // d/dy(uy^2)
      - 0.5 * ( un[i][j][k+1][Y]*un[i][j][k+1][Z] - un[i][j][k-1][Y]*un[i][j][k-1][Z]) // d/dy(uy*uz)
                );
      up[i][j][k][Z] = un[i][j][k][Z] 
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
        up[i][j][0   ][d] = up[i][j][NZ][d];
        up[i][j][NZ+1][d] = up[i][j][1 ][d];
    }}}

    // solve poisson equation with jacobi method
    const int max_iter = 10000;
    for (int itr = 0; itr < max_iter; itr++) {    
      const double COEFF = - 0.5 / DT;
      const double D = 1.0 / 6.0;
      for(int i = 1; i <= NX; i++){for(int j = 1; j <= NY; j++){for(int k = 1; k <= NZ; k++){
  p[tp][i][j][k] = 
    + D*COEFF * (up[i+1][j][k][X] - up[i-1][j][k][X] + up[i][j+1][k][Y] - up[i][j-1][k][Y] + up[i][j][k+1][Z] - up[i][j][k-1][Z])  // b/D
    + D       * ( p[tn][i+1][j][k] + p[tn][i-1][j][k] + p[tn][i][j+1][k] + p[tn][i][j-1][k] + p[tn][i][j][k+1] + p[tn][i][j][k-1] ); // -A*x/D
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

      if( residue/sum < TOLERANCE ) break;
      if( (itr == max_iter-1) || !std::isfinite(residue) ) {
        std::cerr << "ERROR: not converged: poisson solver" << std::endl;
        std::cerr << "step: " << step << std::endl;
        std::cerr << "iterator: " << itr << std::endl;
        std::cerr << "residue: " << residue << std::endl;
        return 1;
      }
    }
    for(int i = 0; i < NX+2; i++){ for(int j = 0; j < NY+2; j++){ for(int k = 0; k < NZ+2; k++){
      p[tp][i][j][k] = p[tn][i][j][k];
    }}}

    // update velocity
    for(int i = 1; i <= NX; i++){ for(int j = 1; j <= NY; j++){ for(int k = 1; k <= NZ; k++){
      up[i][j][k][X] += - DT * 0.5*(p[tp][i+1][j][k] - p[tp][i-1][j][k]);
      up[i][j][k][Y] += - DT * 0.5*(p[tp][i][j+1][k] - p[tp][i][j-1][k]);
      up[i][j][k][Z] += - DT * 0.5*(p[tp][i][j][k+1] - p[tp][i][j][k-1]);
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
        up[i][j][0   ][d] = up[i][j][NZ][d];
        up[i][j][NZ+1][d] = up[i][j][1 ][d];
    }}}


    time += DT;
    // if(((step+1) % (MAX_STEP/10))==0){
    //   std::cerr << step+1 << std::endl;
    // }

    // write
    if((step+1)%10==0){
      double err = 0.0;
      for(int i = 1; i <= NX; i++){ for(int j = 1; j <= NY; j++){ for(int k = 1; k <= NZ; k++){
        double x = 0.5+(i-1);
        double y = 0.5+(j-1);
        double z = 0.5+(k-1);
        double aux = 0.0;
        double auy = - std::sin(y/NY*2*M_PI)*std::cos(z/NZ*2*M_PI);
        double auz =   std::cos(y/NY*2*M_PI)*std::sin(z/NZ*2*M_PI);
        aux *= std::exp(-2.0*time/NY);
        auy *= std::exp(-2.0*time/NY);
        auz *= std::exp(-2.0*time/NY);

        err = std::max(err, 
          + (up[i][j][k][X]-aux)*(up[i][j][k][X]-aux)
          + (up[i][j][k][Y]-auy)*(up[i][j][k][Y]-auy)
          + (up[i][j][k][Z]-auz)*(up[i][j][k][Z]-auz))
        ;
      }}}
      std::cout << time << " " << std::sqrt(err) << std::endl;;
    }
  } // for step


  auto time_end = std::chrono::system_clock::now();  // 
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
  std::cerr << "time = " << elapsed << " [ms]" << std::endl;
  return 0;
}