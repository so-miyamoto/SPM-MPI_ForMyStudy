#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ctime>
#include <cmath>
#include <limits>

#include <mpi.h>

// parameters
namespace {
  static const int X = 0;
  static const int Y = 1;
  static const int NX = 1024;
  static const int NY = 1024;
  static const double DT = 0.01;
  static const int MAX_STEP = 10000;
  static const int DIM = 2;
  static const double TOLERANCE = 1e-6;
  static const double SOR_COEFF = 0.9;

  static const int NP = 2;
  static const double XI_SPM = 2.0;
  static const double V_WALL_U[DIM] = {-0.01,0.0};
  static const double V_WALL_L[DIM] = {+0.01,0.0};
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
  MPI_Init(&argc,&argv);
  int num_procs = 1;
  int my_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
  const int rank_r = (my_rank+1)%num_procs;
  const int rank_l = (my_rank+num_procs-1)%num_procs;

  if(NX%num_procs!=0){
    if( my_rank == 0 ) std::fprintf(stderr,"NX/num_procs != 0\n");
    return 1;
  }
  const int NXloc = NX/num_procs;

  FILE* fpp = nullptr; 
  if( my_rank == 0 ) {
    fpp = std::fopen("dat/particles.dat","w");
  }


  auto time_start = std::chrono::system_clock::now(); //

  // allocation
  auto u_alloc     = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2) * DIM );
  auto phipup_alloc= std::make_unique<double[]>(     (NXloc+2) * (NY+2) * DIM );
  auto u_tmp_alloc = std::make_unique<double[]>(     (NXloc+2) * (NY+2) * DIM );
  auto p_alloc     = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2)       );
  auto pp_alloc    = std::make_unique<double[]>( 2 * (NXloc+2) * (NY+2)       );
  auto phi_p_alloc = std::make_unique<double[]>(     (NXloc+2) * (NY+2)       );
  auto u     = reinterpret_cast<double(&)[2][NXloc+2][NY+2][DIM]>(*u_alloc.get());
  auto phipup= reinterpret_cast<double(&)   [NXloc+2][NY+2][DIM]>(*phipup_alloc.get());
  auto u_tmp = reinterpret_cast<double(&)   [NXloc+2][NY+2][DIM]>(*u_tmp_alloc.get());
  auto p     = reinterpret_cast<double(&)[2][NXloc+2][NY+2]     >(*p_alloc.get());
  auto pp    = reinterpret_cast<double(&)[2][NXloc+2][NY+2]     >(*pp_alloc.get());
  auto phi_p = reinterpret_cast<double(&)   [NXloc+2][NY+2]     >(*phi_p_alloc.get());

  
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
    rp[p] = 4;
    mp[p] = M_PI*rp[p]*rp[p];
    Ip[p] = 0.5*mp[p]*rp[p]*rp[p];
    xp[p][X] = (p*1.0/NP)*NX;//NX/2;
    xp[p][Y] = NY/2;//NY/2;
    vp[p][X] = 0.0;
    vp[p][Y] = 0.0;
    fp[p][X] = 0.0;
    fp[p][Y] = 0.0;
    omegap[p][0] = 0.0;
    torp[p][0] = 0.0;
    angp[p][0] = 0.0;
  }
  // const double F_EX[2] = {0.0,-mp[0]*0.1};
  const double F_EX[DIM] = {0.0,0.0};


  // initialize
  for(int i = 0; i < 2*(NXloc+2)*(NY+2)*DIM; i++){
    u_alloc.get()[i] = 0.0;
  }   
  for(int i = 0; i < 2*(NXloc+2)*(NY+2); i++){
    p_alloc.get()[i] = 0.0;
    pp_alloc.get()[i] = 0.0;
  }
  for(int i = 1; i <= NXloc; i++) {
    for(int j = 1; j <= NY; j++) {
      double x = 0.5+(i-1) + my_rank*NXloc;
      double y = 0.5+(j-1);
      // u[0][i][j][X] =   std::cos(x/NX*2*M_PI)*std::sin(y/NY*2*M_PI);
      // u[0][i][j][Y] = - std::sin(x/NX*2*M_PI)*std::cos(y/NY*2*M_PI);
      u[0][i][j][X] = 0.0;
      u[0][i][j][Y] = 0.0;
    }
  }
  // boundary condition
  for(int i = 1; i <= NXloc; i++) {
    for(int d = 0; d < DIM; d++) {
      u[0][i][0   ][d] =  2.0*V_WALL_L[d] - u[0][i][1 ][d]; // u[0][i][NY][d];
      u[0][i][NY+1][d] =  2.0*V_WALL_U[d] - u[0][i][NY][d]; // u[0][i][1 ][d];
    }
  }
  MPI_Sendrecv( &u[0][NXloc  ],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,
                &u[0][0      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
  MPI_Sendrecv( &u[0][1      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,
                &u[0][NXloc+1],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);


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
        xp[p][d] = xp[p][d] + DT * vp[p][d] + (0.5*(DT*DT)/mp[p])*(fp[p][d]+F_EX[d]);      
      }  
      xp[p][X] = pbc(xp[p][X],0.0,NX);
      xp[p][Y] = pbc(xp[p][Y],0.0,NY);
    }
    for(int i = 0; i < NXloc+2; i++){
      for(int j = 0; j < NY+2; j++){
        phi_p[i][j] = 0.0;
    }}
    for (int p = 0; p < NP; p++){
      const int il = xp[p][X] - rp[p] - XI_SPM      ; 
      const int ir = xp[p][X] + rp[p] + XI_SPM + 0.5; 
      const int jl = xp[p][Y] - rp[p] - XI_SPM      ; 
      const int jr = xp[p][Y] + rp[p] + XI_SPM + 0.5; 
      const int xarea_l = my_rank*NXloc;
      const int xarea_r = (my_rank+1)*NXloc;
      for (int i = il; i <= ir; i++){
        const int ix = (i+NX)%NX+1;
        const double x = ix - 0.5;
        if( (x < xarea_l) || (xarea_r < x) ) continue;
        const int ii = ix - xarea_l;
        if( ii <= 0 || NX < ii ) {
          std::cerr << "invalid ii = "<<ii << std::endl;
        }
        for (int j = jl; j <= jr; j++){
          const int jj = (j+NY)%NY+1;
          const double y = jj - 0.5;
          const double rx = relpos(xp[p][X],x,NX), ry = relpos(xp[p][Y],y,NY);
          const double r = std::sqrt( rx*rx + ry*ry );
          phi_p[ii][jj] += spmf(r,rp[p],XI_SPM);
        }
      }
    }
    for(int i = 1; i <= NXloc; i++){
      phi_p[i][0   ] = phi_p[i][NY];
      phi_p[i][NY+1] = phi_p[i][1 ];
    }
    MPI_Sendrecv( &phi_p[NXloc  ],(NY+2),MPI_DOUBLE,rank_r,0,
                  &phi_p[0      ],(NY+2),MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
    MPI_Sendrecv( &phi_p[1      ],(NY+2),MPI_DOUBLE,rank_l,0,
                  &phi_p[NXloc+1],(NY+2),MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);


    // 2. fractional step w/o pressure term
    for(int i = 1; i <= NXloc; i++){
      for(int j = 1; j <= NY; j++){
        u_tmp[i][j][X] = un[i][j][X] 
            + DT * ( 
  // diffusion
  - 4.0 * un[i][j][X] + un[i-1][j][X] + un[i+1][j][X] + un[i][j-1][X] + un[i][j+1][X] 
  // advection (div. form)
  - 0.5 * ( un[i+1][j][X]*un[i+1][j][X] - un[i-1][j][X]*un[i-1][j][X]) // d/dx(ux^2) 
  - 0.5 * ( un[i][j+1][X]*un[i][j+1][Y] - un[i][j-1][X]*un[i][j-1][Y]) // d/dy(ux*uy)
            );
        u_tmp[i][j][Y] = un[i][j][Y] 
            + DT * ( 
  // diffusion
  - 4.0 * un[i][j][Y] + un[i-1][j][Y] + un[i+1][j][Y] + un[i][j-1][Y] + un[i][j+1][Y] 
  // advection (div. form)
  - 0.5 * ( un[i+1][j][X]*un[i+1][j][Y] - un[i-1][j][X]*un[i-1][j][Y]) // d/dx(ux*uy) 
  - 0.5 * ( un[i][j+1][Y]*un[i][j+1][Y] - un[i][j-1][Y]*un[i][j-1][Y]) // d/dy(uy^2)
            );
      }
    }
    // boundary condition
    for(int i = 1; i <= NXloc; i++){
      for(int d = 0; d < DIM; d++){
        u_tmp[i][0   ][d] = 2.0 * V_WALL_L[d] - u_tmp[i][1 ][d]; //u_tmp[i][NY][d];
        u_tmp[i][NY+1][d] = 2.0 * V_WALL_U[d] - u_tmp[i][NY][d]; //u_tmp[i][1 ][d];
      }
    }
    MPI_Sendrecv( &u_tmp[NXloc  ],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,
                  &u_tmp[0      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
    MPI_Sendrecv( &u_tmp[1      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,
                  &u_tmp[NXloc+1],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);


    // solve poisson equation with jacobi method
    const int max_iter = 100000;
    for (int itr = 0; itr < max_iter; itr++) {    
      for(int i = 1; i <= NXloc; i++){
        const double COEFF = - 0.5 / DT ;
        for(int j = 1; j <= NY; j++){
          const double D = (j==1 || j==NY)? 1.0/3.0 : 1.0/4.0;
          const double jpv = (j==NY)? 0.0:1.0;
          const double jmv = (j==1 )? 0.0:1.0;
  p[tp][i][j] = 
    + D*COEFF * (u_tmp[i+1][j][X] - u_tmp[i-1][j][X] + u_tmp[i][j+1][Y] - u_tmp[i][j-1][Y])  // b/D
    + D       * ( p[tn][i+1][j] + p[tn][i-1][j] + jpv*p[tn][i][j+1] + jmv*p[tn][i][j-1] ); // -A*x/D
        }
      }
      double residue = 0.0;
      for(int i = 1; i <= NXloc; i++){
        for(int j = 1; j <= NY; j++){
          double pnew = (1.0-SOR_COEFF)*p[tn][i][j] + SOR_COEFF*p[tp][i][j];
          double e = pnew - p[tn][i][j];
          residue += e*e;
          p[tn][i][j] = pnew;
        }
      }
      double residue_sum = 0.0; MPI_Allreduce(&residue,&residue_sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      residue_sum = std::sqrt(residue_sum);
      // boundary condition
      for(int i = 1; i <= NXloc; i++){
        p[tn][i][0   ] = p[tn][i][1];//p[tn][i][NY];
        p[tn][i][NY+1] = p[tn][i][NY];//p[tn][i][1 ];
      }
      MPI_Sendrecv( &p[tn][NXloc  ],(NY+2),MPI_DOUBLE,rank_r,0,
                    &p[tn][0      ],(NY+2),MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
      MPI_Sendrecv( &p[tn][1      ],(NY+2),MPI_DOUBLE,rank_l,0,
                    &p[tn][NXloc+1],(NY+2),MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);

      if( residue_sum < TOLERANCE ) break;
      if( (itr == max_iter-1) || !std::isfinite(residue_sum) ) {
        std::cerr << "ERROR: not converged: poisson solver for tmp vel" << std::endl;
        std::cerr << "step: " << step << std::endl;
        std::cerr << "iterator: " << itr << std::endl;
        std::cerr << "residue: " << residue_sum << std::endl;
        return 1;
      }
    }
    for(int i = 0; i < NXloc+2; i++){
      for(int j = 0; j < NY+2; j++){
        p[tp][i][j] = p[tn][i][j];
    }}

    // update velocity
    for(int i = 1; i <= NXloc; i++) {
      for(int j = 1; j <= NY; j++) {
        u_tmp[i][j][X] += - DT * 0.5*(p[tn][i+1][j] - p[tn][i-1][j]);
        u_tmp[i][j][Y] += - DT * 0.5*(p[tn][i][j+1] - p[tn][i][j-1]);
    }}

    // boundary condition
    for(int i = 1; i <= NXloc; i++){
      for(int d = 0; d < DIM; d++){
        u_tmp[i][0   ][d] = 2.0 * V_WALL_L[d] - u_tmp[i][1 ][d];//- u_tmp[i][NY][d];
        u_tmp[i][NY+1][d] = 2.0 * V_WALL_U[d] - u_tmp[i][NY][d];//- u_tmp[i][1 ][d];
    }}
    MPI_Sendrecv( &u_tmp[NXloc  ],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,
                  &u_tmp[0      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
    MPI_Sendrecv( &u_tmp[1      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,
                  &u_tmp[NXloc+1],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);


    // 3. exchange momenta of particles and fluids
    for (int p = 0; p < NP; p++){
      for (int d = 0; d < DIM; d++){
        fp[p][d] = 0.0;
      }
      torp[p][0] = 0.0;
    }
    for (int p = 0; p < NP; p++){
      double sendbuf[3] = {0.0,0.0,0.0};
      double recvbuf[3] = {0.0,0.0,0.0};
      const int il = xp[p][X] - rp[p] - XI_SPM      ; 
      const int ir = xp[p][X] + rp[p] + XI_SPM + 0.5; 
      const int jl = xp[p][Y] - rp[p] - XI_SPM      ; 
      const int jr = xp[p][Y] + rp[p] + XI_SPM + 0.5; 
      const int xarea_l = my_rank*NXloc;
      const int xarea_r = (my_rank+1)*NXloc;
      for (int i = il; i <= ir; i++){
        const int ix = (i+NX)%NX+1;
        const double x = ix - 0.5;
        if( (x < xarea_l) || (xarea_r < x) ) continue;
        const int ii = ix - xarea_l;
        for (int j = jl; j <= jr; j++){
          const int jj = (j+NY)%NY+1;
          const double y = jj - 0.5;
          const double rx = relpos(xp[p][X],x,NX), ry = relpos(xp[p][Y],y,NY);
          const double r = std::sqrt( rx*rx + ry*ry );
          double phipp = spmf(r,rp[p],XI_SPM);
          double phippdux = phipp * (u_tmp[ii][jj][X] - (vp[p][X] + (-omegap[p][0])*ry));
          double phippduy = phipp * (u_tmp[ii][jj][Y] - (vp[p][Y] + (+omegap[p][0])*rx));
          sendbuf[0] += phippdux; // fp[p][X]
          sendbuf[1] += phippduy; // fp[p][Y]
          sendbuf[2] += rx * phippduy - ry * phippdux; // torp[p][0]
        }
      }
      const double invdt = 1.0/DT;
      sendbuf[0] *= invdt;
      sendbuf[1] *= invdt;
      sendbuf[2] *= invdt;
      MPI_Allreduce(sendbuf,recvbuf,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      fp[p][X] = recvbuf[0];
      fp[p][Y] = recvbuf[1];
      torp[p][0] = recvbuf[2];
    }

    // 4. update particle's velocity and angular vel
    for (int p = 0; p < NP; p++){
      double invm = 1.0/mp[p];
      double invI = 1.0/Ip[p];
      vp[p][X] += invm * DT * (fp[p][X] + F_EX[X]);
      vp[p][Y] += invm * DT * (fp[p][Y] + F_EX[Y]);
      omegap[p][0] += invI * DT * torp[p][0];
    }        
    // phipup
    for(int i = 0; i < NXloc+2; i++){
      for(int j = 0; j < NY+2; j++){
        phipup[i][j][X] = 0.0;
        phipup[i][j][Y] = 0.0;
    }}
    for (int p = 0; p < NP; p++){
      const int il = xp[p][X] - rp[p] - XI_SPM      ; 
      const int ir = xp[p][X] + rp[p] + XI_SPM + 0.5; 
      const int jl = xp[p][Y] - rp[p] - XI_SPM      ; 
      const int jr = xp[p][Y] + rp[p] + XI_SPM + 0.5; 
      const int xarea_l = my_rank*NXloc;
      const int xarea_r = (my_rank+1)*NXloc;
      for (int i = il; i <= ir; i++){
        const int ix = (i+NX)%NX+1;
        const double x = ix - 0.5;
        if( (x < xarea_l) || (xarea_r < x) ) continue;
        const int ii = ix - xarea_l;
        for (int j = jl; j <= jr; j++){
          const int jj = (j+NY)%NY+1;
          const double y = jj - 0.5;
          const double rx = relpos(xp[p][X],x,NX), ry = relpos(xp[p][Y],y,NY);
          const double r = std::sqrt( rx*rx + ry*ry );
          double phipp = spmf(r,rp[p],XI_SPM);
          phipup[ii][jj][X] +=  phipp * (vp[p][X] + (-omegap[p][0])*ry);
          phipup[ii][jj][Y] +=  phipp * (vp[p][Y] + (+omegap[p][0])*rx);
      }}
    }
    for(int i = 1; i <= NXloc; i++){
      for(int d = 0; d < DIM; d++){
        phipup[i][0   ][d] = phipup[i][NY][d];
        phipup[i][NY+1][d] = phipup[i][1 ][d];
    }}
    MPI_Sendrecv( &phipup[NXloc  ],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,
                  &phipup[0      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
    MPI_Sendrecv( &phipup[1      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,
                  &phipup[NXloc+1],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);


    // 5. update velocity of next step
    for (int itr = 0; itr < max_iter; itr++) {    
      for(int i = 1; i <= NXloc; i++){
        const double COEFF = - 0.5 / DT ;
        for(int j = 1; j <= NY; j++){
          const double D = (j==1 || j==NY)? 1.0/3.0 : 1.0/4.0;
          const double jpv = (j==NY)? 0.0:1.0;
          const double jmv = (j==1 )? 0.0:1.0;

  pp[tp][i][j] = 
    + D*COEFF * (phipup[i+1][j][X] - phipup[i-1][j][X] + phipup[i][j+1][Y] - phipup[i][j-1][Y])  // b/D
    - D*COEFF * (phi_p[i+1][j]*u_tmp[i+1][j][X] - phi_p[i-1][j]*u_tmp[i-1][j][X] 
             + phi_p[i][j+1]*u_tmp[i][j+1][Y] - phi_p[i][j-1]*u_tmp[i][j-1][Y])  // b/D
    + D       * ( pp[tn][i+1][j] + pp[tn][i-1][j] + jpv*pp[tn][i][j+1] + jmv*pp[tn][i][j-1] ); // -A*x/D
        }
      }
      double residue = 0.0;
      for(int i = 1; i <= NXloc; i++){
        for(int j = 1; j <= NY; j++){
          double pnew = (1.0-SOR_COEFF)*pp[tn][i][j] + SOR_COEFF*pp[tp][i][j];
          double e = pnew - pp[tn][i][j];
          residue += e*e;
          pp[tn][i][j] = pnew;
        }
      }
      double residue_sum = 0.0; MPI_Allreduce(&residue,&residue_sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      residue_sum = std::sqrt(residue_sum);
      // boundary condition
      for(int i = 1; i <= NXloc; i++){
        pp[tn][i][0   ] = pp[tn][i][1];//p[tn][i][NY];
        pp[tn][i][NY+1] = pp[tn][i][NY];//p[tn][i][1 ];
      }
      MPI_Sendrecv( &pp[tn][NXloc  ],(NY+2),MPI_DOUBLE,rank_r,0,
                    &pp[tn][0      ],(NY+2),MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
      MPI_Sendrecv( &pp[tn][1      ],(NY+2),MPI_DOUBLE,rank_l,0,
                    &pp[tn][NXloc+1],(NY+2),MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);

      if( residue_sum < TOLERANCE ) break;
      if( (itr == max_iter-1) || !std::isfinite(residue_sum) ) {
        std::cerr << "ERROR: not converged: poisson solver for particle" << std::endl;
        std::cerr << "step: " << step << std::endl;
        std::cerr << "iterator: " << itr << std::endl;
        std::cerr << "residue: " << residue_sum << std::endl;
        return 1;
      }
    }
    for(int i = 0; i < NXloc+2; i++){
      for(int j = 0; j < NY+2; j++){
        pp[tp][i][j] = pp[tn][i][j];
    }}

    // update velocity
    for(int i = 1; i <= NXloc; i++) {
      for(int j = 1; j <= NY; j++) {
        up[i][j][X] = u_tmp[i][j][X] + phipup[i][j][X] - phi_p[i][j]*u_tmp[i][j][X] - DT * 0.5*(pp[tn][i+1][j] - pp[tn][i-1][j]);
        up[i][j][Y] = u_tmp[i][j][Y] + phipup[i][j][Y] - phi_p[i][j]*u_tmp[i][j][Y] - DT * 0.5*(pp[tn][i][j+1] - pp[tn][i][j-1]);
    }}
    // boundary condition
    for(int i = 1; i <= NXloc; i++){
      for(int d = 0; d < DIM; d++){
        up[i][0   ][d] = 2.0 * V_WALL_L[d]  - up[i][1 ][d]; //- up[i][NY][d];
        up[i][NY+1][d] = 2.0 * V_WALL_U[d]  - up[i][NY][d]; //- up[i][1 ][d];
    }}
    MPI_Sendrecv( &up[NXloc  ],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,
                  &up[0      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
    MPI_Sendrecv( &up[1      ],(NY+2)*DIM,MPI_DOUBLE,rank_l,0,
                  &up[NXloc+1],(NY+2)*DIM,MPI_DOUBLE,rank_r,0,MPI_COMM_WORLD,MPI_STATUSES_IGNORE);

    time += DT;

    if(((step+1) % (MAX_STEP/100))==0){
      // // write
      if( my_rank == 0 ) 
        std::fprintf(stdout,"step = %d\n",step+1);
    //   FILE* fpf = std::fopen(std::string("dat/fluid_"+std::to_string(step+1)+"_"+std::to_string(my_rank)+".dat").c_str(),"w");
    //   // double err = 0.0;
    //   for(int i = 1; i <= NXloc; i++) {
    //     for(int j = 1; j <= NY; j++) {
    //       double x = 0.5+(i-1) + my_rank*NXloc;
    //       double y = 0.5+(j-1);
    //       std::fprintf(fpf,"%lf %lf %lf %lf %lf\n",x,y,up[i][j][X],up[i][j][Y],phi_p[i][j]);
    //     }
    //   }
    //   std::fclose(fpf);
    //   if( my_rank == 0 ){
    //     for (int p = 0; p < NP; p++){
    //       std::fprintf(fpp,"%d %lf %.8lf %.8lf %.8lf %.8lf %.8lf %.8lf %.8lf\n",p,time,xp[p][X],xp[p][Y],vp[p][X],vp[p][Y],fp[p][X],fp[p][Y],omegap[p][0]);
    //     }
    //     std::fflush(fpp);
    //   }
    }
  } // for step


  if( my_rank == 0 ){
    std::fclose(fpp);
  }
  MPI_Finalize();
  auto time_end = std::chrono::system_clock::now();  // 
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
  std::cerr << "time = " << elapsed << " [ms]" << std::endl;
  return 0;
}