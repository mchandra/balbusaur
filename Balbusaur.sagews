︠005a29ff-901d-4bb7-9aab-71dbce8edd7e︠
%auto
typeset_mode(True)
︡a00edfa0-e65f-4196-bd04-c039fd1071c5︡{"auto":true}︡
︠cfc00840-33c2-4987-9665-f39a6125d569i︠
%html
<p style="text-align: center;"><img style="vertical-align: middle;" src="http://art.ngfiles.com/images/284000/284257_emutoons_derpemon-bulbasaur.png" alt="" width="320" height="180" /></p>
<h2 style="text-align: center;">Balbusaur</h2>

︡20b74f6e-5774-4e74-ae37-cae6916814d1︡{"html":"<p style=\"text-align: center;\"><img style=\"vertical-align: middle;\" src=\"http://art.ngfiles.com/images/284000/284257_emutoons_derpemon-bulbasaur.png\" alt=\"\" width=\"320\" height=\"180\" /></p>\n<h2 style=\"text-align: center;\">Balbusaur</h2>\n\n"}︡
︠82740d77-8853-4cb1-a194-df8f1673c123i︠
%md

#### Variables involved:
1. Density $\rho$
2. Internal energy $u$
3. Velocity in $x^1$ direction $u^1$
4. Velocity in $x^2$ direction $u^2$
5. Magnetic field in $x^1$ direction $B^1$
6. Magnetic field in $x^2$ direction $B^2$
7. Heat flux magnitude $\phi$
8. Shear stress magnitude $\psi$

#### Problem descriptions:

1. CONDUCTION_1D:
   * Mean variables      : $\rho_0$, $u_0$, $B^1_0$
   * Perturbed variables : $\rho$, $u$, $u^1$, $\phi$
   * Wavenumbers         : $k_1$

2. VISCOSITY_1D:
   * Mean variables      : $\rho_0$, $u_0$, $B^1_0$
   * Perturbed variables : $\rho$, $u$, $u^1$, $\psi$
   * Wavenumbers         : $k_1$

3. CONDUCTION_2D:
   * Mean variables      : $\rho_0$, $u_0$, $B^1_0$, $B^2_0$
   * Perturbed variables : $\rho$, $u$, $u^1$, $u^2$, $\phi$
   * Wavenumbers         : $k_1$, $k_2$

4. VISCOSITY_2D:
   * Mean variables      : $\rho_0$, $u_0$, $B^1_0$, $B^2_0$
   * Perturbed variables : $\rho$, $u$, $u^1$, $u^2$, $B^1$, $B^2$, $\psi$
   * Wavenumbers         : $k_1$, $k_2$

5. CONDUCTION_AND_VISCOSITY_1D:
   * Mean variables      : $\rho_0$, $u_0$, $B^1_0$
   * Perturbed variables : $\rho$, $u$, $u^1$, $\phi$, $\psi$
   * Wavenumbers         : $k_1$

3. CONDUCTION_AND_VISCOSITY_2D:
   * Mean variables      : $\rho_0$, $u_0$, $B^1_0$, $B^2_0$
   * Perturbed variables : $\rho$, $u$, $u^1$, $u^2$, $\phi$, $\psi$
   * Wavenumbers         : $k_1$, $k_2$
︡7d01e076-a4fd-40cf-acbb-a223b84f62b9︡{"md":"\n#### Variables involved:\n1. Density $\\rho$\n2. Internal energy $u$\n3. Velocity in $x^1$ direction $u^1$\n4. Velocity in $x^2$ direction $u^2$\n5. Magnetic field in $x^1$ direction $B^1$\n6. Magnetic field in $x^2$ direction $B^2$\n7. Heat flux magnitude $\\phi$\n8. Shear stress magnitude $\\psi$\n\n#### Problem descriptions:\n\n1. CONDUCTION_1D: \n   * Mean variables      : $\\rho_0$, $u_0$, $B^1_0$\n   * Perturbed variables : $\\rho$, $u$, $u^1$, $\\phi$\n   * Wavenumbers         : $k_1$\n   \n2. VISCOSITY_1D: \n   * Mean variables      : $\\rho_0$, $u_0$, $B^1_0$\n   * Perturbed variables : $\\rho$, $u$, $u^1$, $\\psi$\n   * Wavenumbers         : $k_1$\n\n3. CONDUCTION_2D: \n   * Mean variables      : $\\rho_0$, $u_0$, $B^1_0$, $B^2_0$\n   * Perturbed variables : $\\rho$, $u$, $u^1$, $u^2$, $\\phi$\n   * Wavenumbers         : $k_1$, $k_2$\n   \n4. VISCOSITY_2D: \n   * Mean variables      : $\\rho_0$, $u_0$, $B^1_0$, $B^2_0$\n   * Perturbed variables : $\\rho$, $u$, $u^1$, $u^2$, $B^1$, $B^2$, $\\psi$\n   * Wavenumbers         : $k_1$, $k_2$\n\n5. CONDUCTION_AND_VISCOSITY_1D: \n   * Mean variables      : $\\rho_0$, $u_0$, $B^1_0$\n   * Perturbed variables : $\\rho$, $u$, $u^1$, $\\phi$, $\\psi$\n   * Wavenumbers         : $k_1$\n\n3. CONDUCTION_AND_VISCOSITY_2D: \n   * Mean variables      : $\\rho_0$, $u_0$, $B^1_0$, $B^2_0$\n   * Perturbed variables : $\\rho$, $u$, $u^1$, $u^2$, $\\phi$, $\\psi$\n   * Wavenumbers         : $k_1$, $k_2$\n"}︡
︠c953ed04-b6ab-4a98-a57f-2711f7d0d62bs︠
# Inputs:

# Choose problem here:
problem = "FIREHOSE"

# Inputs for numerical diagonalization for finite k modes
rho0_num  = 1.
u0_num    = 2.0
B10_num   = 0.01
B20_num   = 0

Gamma_num = 4./3
P0_num    = (Gamma_num - 1.)*u0_num
T0_num    = P0_num/rho0_num
psi0_num  = -10.
k1_num    = 2.*pi
k2_num    = 0
kappa_num = 0.1
eta_num   = 1.
tau_num   = 0.0001
︡44d5b998-780e-4a8f-b19f-ce06bf2b28f1︡
︠a7637a36-f643-4769-aed3-98f0e4e300b9︠
# Spatiotemporal variables
t, omega, k1, k2 = var('t, omega, k1, k2')

# Constants:
# Gamma : Adiabatic index
# kappa : Heat conductivity
# eta   : shear viscosity
# tau   : relaxation time scale
Gamma, kappa, eta, tau = var('Gamma, kappa, eta, tau')

# Background mean values
rho0, u0, B10, B20, psi0 = var('rho0, u0, B10, B20, psi0')

# Perturbations in space
delta_rho, delta_u, delta_u1, delta_u2, delta_u3, delta_B1, delta_B2, delta_B3, delta_phi, delta_psi = \
    var('delta_rho, delta_u, delta_u1, delta_u2, delta_u3, delta_B1, delta_B2, delta_B3, delta_phi, delta_psi')

# Perturbations in time
delta_rho_dt, delta_u_dt, delta_u1_dt, delta_u2_dt, delta_u3_dt, delta_B1_dt, delta_B2_dt, delta_B3_dt, delta_phi_dt, delta_psi_dt = \
    var('delta_rho_dt, delta_u_dt, delta_u1_dt, delta_u2_dt, delta_u3_dt, delta_B1_dt, delta_B2_dt, delta_B3_dt, delta_phi_dt, delta_psi_dt')

#eta = 0

if (problem=="CONDUCTION_1D"):
    rho = rho0 + delta_rho
    u   = u0 + delta_u
    u1  = delta_u1
    u2  = 0
    u3  = 0
    B1  = B10
    B2  = 0
    B3  = 0
    phi = delta_phi
    psi = 0

elif (problem=="CONDUCTION_2D"):
    rho = rho0 + delta_rho
    u   = u0 + delta_u
    u1  = delta_u1
    u2  = delta_u2
    u3  = 0
    B1  = B10
    B2  = B20
    B3  = 0
    phi = delta_phi
    psi = 0

elif (problem=="VISCOSITY_1D"):
    rho = rho0 + delta_rho
    u   = u0 + delta_u
    u1  = delta_u1
    u2  = 0
    u3  = 0
    B1  = B10
    B2  = 0
    B3  = 0
    phi = 0
    psi = delta_psi

elif (problem=="VISCOSITY_2D"):
    rho = rho0 + delta_rho
    u   = u0 + delta_u
    u1  = delta_u1
    u2  = delta_u2
    u3  = 0
    B1  = B10  + delta_B1
    B2  = B20 + delta_B2
    B3  = 0
    phi = 0
    psi = delta_psi

elif (problem=="CONDUCTION_AND_VISCOSITY_1D"):
    rho = rho0 + delta_rho
    u   = u0 + delta_u
    u1  = delta_u1
    u2  = 0
    u3  = 0
    B1  = B10
    B2  = 0
    B3  = 0
    phi = delta_phi
    psi = delta_psi

elif (problem=="CONDUCTION_AND_VISCOSITY_2D"):
    rho = rho0 + delta_rho
    u   = u0 + delta_u
    u1  = delta_u1
    u2  = delta_u2
    u3  = 0
    B1  = B10
    B2  = B20
    B3  = 0
    phi = delta_phi
    psi = delta_psi

elif (problem=="FIREHOSE"):
    rho = rho0 + delta_rho
    u   = u0 + delta_u
    u1  = delta_u1
    u2  = 0
    u3  = 0
    B1  = B10 + delta_B1
    B2  = 0
    B3  = 0
    phi = 0
    psi = psi0 + delta_psi


gcon = Matrix([ [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
              ]
             )

gcov = gcon.inverse()

gamma = sqrt(1 +    gcov[1][1]*u1*u1 + gcov[2][2]*u2*u2 + gcov[3][3]*u3*u3
               + 2*(gcov[1][2]*u1*u2 + gcov[1][3]*u1*u3 + gcov[2][3]*u2*u3)
            )

ucon = [gamma, u1, u2, u3]
ucov = [-gamma, u1, u2, u3]

bcon0 = B1*ucov[1] + B2*ucov[2] + B3*ucov[3]
bcon1 = (B1 + bcon0*ucon[1])/ucon[0]
bcon2 = (B2 + bcon0*ucon[2])/ucon[0]
bcon3 = (B3 + bcon0*ucon[3])/ucon[0]

bcon = [bcon0, bcon1, bcon2, bcon3]
bcov = [-bcon0, bcon1, bcon2, bcon3]

bsqr = bcon[0]*bcov[0] + bcon[1]*bcov[1] + bcon[2]*bcov[2] + bcon[3]*bcov[3]

P = (Gamma - 1)*u
T = P/rho


def linearize(term):
    return taylor(term, (delta_rho, 0), \
                        (delta_u, 0),   \
                        (delta_u1, 0),  \
                        (delta_u2, 0),  \
                        (delta_u3, 0),  \
                        (delta_B1, 0),  \
                        (delta_B2, 0),  \
                        (delta_B3, 0),  \
                        (delta_phi, 0), \
                        (delta_psi, 0), \
                        (delta_rho_dt, 0), \
                        (delta_u_dt, 0),   \
                        (delta_u1_dt, 0),  \
                        (delta_u2_dt, 0),  \
                        (delta_u3_dt, 0),  \
                        (delta_B1_dt, 0),  \
                        (delta_B2_dt, 0),  \
                        (delta_B3_dt, 0),  \
                        (delta_phi_dt, 0), \
                        (delta_psi_dt, 0), 1 \
                 ).simplify_full()

def d_dX1(term):
    term  = Expression(SR, linearize(term))

    expr =   term.coefficient(delta_rho) * I * k1 * delta_rho \
           + term.coefficient(delta_u)   * I * k1 * delta_u   \
           + term.coefficient(delta_u1)  * I * k1 * delta_u1  \
           + term.coefficient(delta_u2)  * I * k1 * delta_u2  \
           + term.coefficient(delta_u3)  * I * k1 * delta_u3  \
           + term.coefficient(delta_B1)  * I * k1 * delta_B1  \
           + term.coefficient(delta_B2)  * I * k1 * delta_B2  \
           + term.coefficient(delta_B3)  * I * k1 * delta_B3  \
           + term.coefficient(delta_phi) * I * k1 * delta_phi \
           + term.coefficient(delta_psi) * I * k1 * delta_psi

    return expr

def d_dX2(term):
    term  = Expression(SR, linearize(term))

    expr =   term.coefficient(delta_rho) * I * k2 * delta_rho \
           + term.coefficient(delta_u)   * I * k2 * delta_u   \
           + term.coefficient(delta_u1)  * I * k2 * delta_u1  \
           + term.coefficient(delta_u2)  * I * k2 * delta_u2  \
           + term.coefficient(delta_u3)  * I * k2 * delta_u3  \
           + term.coefficient(delta_B1)  * I * k2 * delta_B1  \
           + term.coefficient(delta_B2)  * I * k2 * delta_B2  \
           + term.coefficient(delta_B3)  * I * k2 * delta_B3  \
           + term.coefficient(delta_phi) * I * k2 * delta_phi \
           + term.coefficient(delta_psi) * I * k2 * delta_psi

    return expr


def d_dt(term):
    term  = Expression(SR, linearize(term))

    expr =   term.coefficient(delta_rho) * delta_rho_dt \
           + term.coefficient(delta_u)   * delta_u_dt   \
           + term.coefficient(delta_u1)  * delta_u1_dt  \
           + term.coefficient(delta_u2)  * delta_u2_dt  \
           + term.coefficient(delta_u3)  * delta_u3_dt  \
           + term.coefficient(delta_B1)  * delta_B1_dt  \
           + term.coefficient(delta_B2)  * delta_B2_dt  \
           + term.coefficient(delta_B3)  * delta_B3_dt  \
           + term.coefficient(delta_phi) * delta_phi_dt \
           + term.coefficient(delta_psi) * delta_psi_dt

    return expr


def delta(mu, nu):
    if (mu==nu):
        return 1
    else:
        return 0

def TUpDown(mu, nu):

    return  (rho + u + P + bsqr)*ucon[mu]*ucov[nu] + (P + bsqr/2)*delta(mu, nu) - bcon[mu]*bcov[nu] \
          + phi/sqrt(bsqr)*(bcon[mu]*ucov[nu] + ucon[mu]*bcov[nu]) + psi/bsqr*(bcon[mu]*bcov[nu]) - psi/3*(ucon[mu]*ucov[nu] + delta(mu, nu))

def acon(mu):
    return linearize(ucon[0]*d_dt(ucon[mu]) + ucon[1]*d_dX1(ucon[mu]) + ucon[2]*d_dX2(ucon[mu]))

def qconEckart(mu):
    acov = [-acon(0), acon(1), acon(2), acon(3)]

    ans = -kappa*(ucon[mu]*ucon[0] + gcon[mu, 0])*(d_dt(T) +  T*acov[0]) \
          -kappa*(ucon[mu]*ucon[1] + gcon[mu, 1])*(d_dX1(T) + T*acov[1]) \
          -kappa*(ucon[mu]*ucon[2] + gcon[mu, 2])*(d_dX2(T) + T*acov[2])

    return linearize(ans)

Eqn_rho = linearize(d_dt(rho*ucon[0])   + d_dX1(rho*ucon[1])   + d_dX2(rho*ucon[2]))
Eqn_u   = linearize(d_dt(TUpDown(0, 0)) + d_dX1(TUpDown(1, 0)) + d_dX2(TUpDown(2, 0)))
Eqn_u1  = linearize(d_dt(TUpDown(0, 1)) + d_dX1(TUpDown(1, 1)) + d_dX2(TUpDown(2, 1)))
Eqn_u2  = linearize(d_dt(TUpDown(0, 2)) + d_dX1(TUpDown(1, 2)) + d_dX2(TUpDown(2, 2)))

Eqn_B1  = linearize(d_dt(B1) + d_dX2(bcon[1]*ucon[2] - bcon[2]*ucon[1]) )
Eqn_B2  = linearize(d_dt(B2) + d_dX1(bcon[2]*ucon[1] - bcon[1]*ucon[2]) )

beta1 = tau/(kappa*T)
beta2 = tau/(2*eta)
phi_relaxed = (bcov[0]*qconEckart(0) + bcov[1]*qconEckart(1) + bcov[2]*qconEckart(2) + bcov[3]*qconEckart(3) )/sqrt(bsqr)
psi_relaxed = 0
for nu in xrange(4):
    psi_relaxed = psi_relaxed - 3*eta/bsqr * (bcon[nu]* (bcon[0]*d_dt(ucov[nu]) + bcon[1]*d_dX1(ucov[nu]) + bcon[2]*d_dX2(ucov[nu])) )

psi_relaxed = psi_relaxed + eta*(d_dt(ucon[0]) + d_dX1(ucon[1]) + d_dX2(ucon[2]) )

Eqn_phi = linearize(   ucon[0]*d_dt(phi) + ucon[1]*d_dX1(phi) + ucon[2]*d_dX2(phi) + (phi - phi_relaxed)/tau \
                    + (phi*T/(2*beta1))*(d_dt(beta1*ucon[0]/T) + d_dX1(beta1*ucon[1]/T) + d_dX2(beta1*ucon[2]/T))
                   )
Eqn_psi = linearize(   ucon[0]*d_dt(psi) + ucon[1]*d_dX1(psi) + ucon[2]*d_dX2(psi) + (psi - psi_relaxed)/tau \
                    + (psi*T/(2*beta2))*(d_dt(beta2*ucon[0]/T) + d_dX1(beta2*ucon[1]/T) + d_dX2(beta2*ucon[2]/T))
                   )

if (problem=='CONDUCTION_1D'):

    Eqns          = [Eqn_rho==0, Eqn_u==0, Eqn_u1==0, Eqn_phi==0]
    delta_vars    = [delta_rho, delta_u, delta_u1, delta_phi]
    delta_vars_dt = [delta_rho_dt, delta_u_dt, delta_u1_dt, delta_phi_dt]

elif (problem=='CONDUCTION_2D'):

    Eqns          = [Eqn_rho==0, Eqn_u==0, Eqn_u1==0, Eqn_u2==0, Eqn_phi==0]
    delta_vars    = [delta_rho, delta_u, delta_u1, delta_u2, delta_phi]
    delta_vars_dt = [delta_rho_dt, delta_u_dt, delta_u1_dt, delta_u2_dt, delta_phi_dt]

elif (problem=='VISCOSITY_1D'):

    Eqns          = [Eqn_rho==0, Eqn_u==0, Eqn_u1==0, Eqn_psi==0]
    delta_vars    = [delta_rho, delta_u, delta_u1, delta_psi]
    delta_vars_dt = [delta_rho_dt, delta_u_dt, delta_u1_dt, delta_psi_dt]

elif (problem=='VISCOSITY_2D'):

    Eqns          = [Eqn_rho==0, Eqn_u==0, Eqn_u1==0, Eqn_u2==0, Eqn_B1==0, Eqn_B2==0, Eqn_psi==0]
    delta_vars    = [delta_rho, delta_u, delta_u1, delta_u2, delta_B1, delta_B2, delta_psi]
    delta_vars_dt = [delta_rho_dt, delta_u_dt, delta_u1_dt, delta_u2_dt, delta_B1_dt, delta_B2_dt, delta_psi_dt]

elif (problem=='CONDUCTION_AND_VISCOSITY_1D'):

    Eqns          = [Eqn_rho==0, Eqn_u==0, Eqn_u1==0, Eqn_phi==0, Eqn_psi==0]
    delta_vars    = [delta_rho, delta_u, delta_u1, delta_phi, delta_psi]
    delta_vars_dt = [delta_rho_dt, delta_u_dt, delta_u1_dt, delta_phi_dt, delta_psi_dt]

elif (problem=='CONDUCTION_AND_VISCOSITY_2D'):

    Eqns          = [Eqn_rho==0, Eqn_u==0, Eqn_u1==0, Eqn_u2==0, Eqn_phi==0, Eqn_psi==0]
    delta_vars    = [delta_rho, delta_u, delta_u1, delta_u2, delta_phi, delta_psi]
    delta_vars_dt = [delta_rho_dt, delta_u_dt, delta_u1_dt, delta_u2_dt, delta_phi_dt, delta_psi_dt]

elif (problem=='FIREHOSE'):

    Eqns          = [Eqn_rho==0, Eqn_u==0, Eqn_u1==0, Eqn_B1==0, Eqn_psi==0]
    delta_vars    = [delta_rho, delta_u, delta_u1, delta_B1, delta_psi]
    delta_vars_dt = [delta_rho_dt, delta_u_dt, delta_u1_dt, delta_B1_dt, delta_psi_dt]

solutions = solve(Eqns, delta_vars_dt, solution_dict=True)

solns_delta_vars_dt = []
for dvar_dt in delta_vars_dt:
    solns_delta_vars_dt.append(solutions[0][dvar_dt])

M = jacobian(solns_delta_vars_dt, delta_vars)
M = M.apply_map(lambda x : x.simplify_full())

pretty_print("Linearized system : ", )
print("\n")
pretty_print(Matrix(delta_vars_dt).transpose(), " = ", M, Matrix(delta_vars).transpose())
print("\n\n")
pretty_print("Eigenvalues and eigenvectors in the $k \\rightarrow 0$ limit : ", )
M.subs(k1=0, k2=0).eigenvectors_right()


# Numerical diagonalization:

M_numerical = M.subs(rho0=rho0_num, u0=u0_num, B10=B10_num, B20=B20_num, psi0=psi0_num, Gamma=Gamma_num, kappa=kappa_num, eta=eta_num, tau=tau_num, k1=k1_num, k2=k2_num)
M_numerical = M_numerical.change_ring(CDF)
eigenvecs   = M_numerical.eigenvectors_right()

print "Numerical eigenvalues and eigenvectors for k > 0:\n"
print "--------------------------\n"

if (problem=='CONDUCTION_1D'):

    print "kappa = ", kappa_num
    print "tau   = ", tau_num
    print "--------------------------"
    for i in xrange(len(eigenvecs)):
        print "Eigenvalue = ",     eigenvecs[i][0]
        print "delta_rho  = ",     eigenvecs[i][1][0][0]
        print "delta_u    = ",     eigenvecs[i][1][0][1]
        print "delta_u1   = ",     eigenvecs[i][1][0][2]
        print "delta_phi  = ",     eigenvecs[i][1][0][3]
        print "--------------------------"

if (problem=='CONDUCTION_2D'):

    print "kappa = ", kappa_num
    print "tau   = ", tau_num
    print "--------------------------"
    for i in xrange(len(eigenvecs)):
        print "Eigenvalue = ",     eigenvecs[i][0]
        print "delta_rho  = ",     eigenvecs[i][1][0][0]
        print "delta_u    = ",     eigenvecs[i][1][0][1]
        print "delta_u1   = ",     eigenvecs[i][1][0][2]
        print "delta_u2   = ",     eigenvecs[i][1][0][3]
        print "delta_phi  = ",     eigenvecs[i][1][0][4]
        print "--------------------------"

if (problem=='VISCOSITY_1D'):

    print "eta   = ", eta_num
    print "tau   = ", tau_num
    print "--------------------------"
    for i in xrange(len(eigenvecs)):
        print "Eigenvalue = ",     eigenvecs[i][0]
        print "delta_rho  = ",     eigenvecs[i][1][0][0]
        print "delta_u    = ",     eigenvecs[i][1][0][1]
        print "delta_u1   = ",     eigenvecs[i][1][0][2]
        print "delta_psi  = ",     eigenvecs[i][1][0][3]
        print "--------------------------"

if (problem=='VISCOSITY_2D'):

    print "eta   = ", eta_num
    print "tau   = ", tau_num
    print "--------------------------"
    for i in xrange(len(eigenvecs)):
        print "Eigenvalue = ",     eigenvecs[i][0]
        print "delta_rho  = ",     eigenvecs[i][1][0][0]
        print "delta_u    = ",     eigenvecs[i][1][0][1]
        print "delta_u1   = ",     eigenvecs[i][1][0][2]
        print "delta_u2   = ",     eigenvecs[i][1][0][3]
        print "delta_B1   = ",     eigenvecs[i][1][0][4]
        print "delta_B2   = ",     eigenvecs[i][1][0][5]
        print "delta_psi  = ",     eigenvecs[i][1][0][6]
        print "--------------------------"

if (problem=='CONDUCTION_AND_VISCOSITY_1D'):

    print "kappa = ", kappa_num
    print "eta   = ", eta_num
    print "tau   = ", tau_num
    print "--------------------------"
    for i in xrange(len(eigenvecs)):
        print "Eigenvalue = ",     eigenvecs[i][0]
        print "delta_rho  = ",     eigenvecs[i][1][0][0]
        print "delta_u    = ",     eigenvecs[i][1][0][1]
        print "delta_u1   = ",     eigenvecs[i][1][0][2]
        print "delta_phi  = ",     eigenvecs[i][1][0][3]
        print "delta_psi  = ",     eigenvecs[i][1][0][4]
        print "--------------------------"

if (problem=='CONDUCTION_AND_VISCOSITY_2D'):

    print "kappa = ", kappa_num
    print "eta   = ", eta_num
    print "tau   = ", tau_num
    print "--------------------------"
    for i in xrange(len(eigenvecs)):
        print "Eigenvalue = ",     eigenvecs[i][0]
        print "delta_rho  = ",     eigenvecs[i][1][0][0]
        print "delta_u    = ",     eigenvecs[i][1][0][1]
        print "delta_u1   = ",     eigenvecs[i][1][0][2]
        print "delta_u2   = ",     eigenvecs[i][1][0][3]
        print "delta_phi  = ",     eigenvecs[i][1][0][4]
        print "delta_psi  = ",     eigenvecs[i][1][0][5]
        print "--------------------------"

if (problem=='FIREHOSE'):

    print "eta   = ", eta_num
    print "tau   = ", tau_num
    print "--------------------------"
    for i in xrange(len(eigenvecs)):
        print "Eigenvalue = ",     eigenvecs[i][0]
        print "delta_rho  = ",     eigenvecs[i][1][0][0]
        print "delta_u    = ",     eigenvecs[i][1][0][1]
        print "delta_u1   = ",     eigenvecs[i][1][0][2]
        print "delta_B1   = ",     eigenvecs[i][1][0][3]
        print "delta_psi  = ",     eigenvecs[i][1][0][4]
        print "--------------------------"


︡fe0031d6-9f98-4dff-bcd6-2c71d4038fa6︡{"html":"<div align='center'>Linearized system : </div>"}︡{"stdout":"\n\n"}︡{"html":"<div align='center'>$\\displaystyle \\left(\\begin{array}{r}\n\\delta_{\\rho_{\\mathit{dt}}} \\\\\n\\delta_{u_{\\mathit{dt}}} \\\\\n\\delta_{\\mathit{u1}_{\\mathit{dt}}} \\\\\n\\delta_{\\mathit{B1}_{\\mathit{dt}}} \\\\\n\\delta_{\\psi_{\\mathit{dt}}}\n\\end{array}\\right)$  =  $\\displaystyle \\left(\\begin{array}{rrrrr}\n0 &amp; 0 &amp; -i \\, k_{1} \\rho_{0} &amp; 0 &amp; 0 \\\\\n0 &amp; 0 &amp; -i \\, \\Gamma k_{1} u_{0} - \\frac{2}{3} i \\, k_{1} \\psi_{0} &amp; 0 &amp; 0 \\\\\n0 &amp; \\frac{{\\left(-3 i \\, \\Gamma + 3 i\\right)} k_{1}}{3 \\, \\Gamma u_{0} + 2 \\, \\psi_{0} + 3 \\, \\rho_{0}} &amp; 0 &amp; \\frac{3 i \\, B_{10} k_{1}}{3 \\, \\Gamma u_{0} + 2 \\, \\psi_{0} + 3 \\, \\rho_{0}} &amp; -\\frac{2 i \\, k_{1}}{3 \\, \\Gamma u_{0} + 2 \\, \\psi_{0} + 3 \\, \\rho_{0}} \\\\\n0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\\\\n0 &amp; 0 &amp; \\frac{-2 i \\, k_{1} \\psi_{0}^{2} \\tau - {\\left(3 i \\, \\Gamma k_{1} \\psi_{0} \\tau + 12 i \\, \\eta k_{1}\\right)} u_{0}}{6 \\, \\tau u_{0}} &amp; 0 &amp; -\\frac{1}{\\tau}\n\\end{array}\\right)$ $\\displaystyle \\left(\\begin{array}{r}\n\\delta_{\\rho} \\\\\n\\delta_{u} \\\\\n\\delta_{u_{1}} \\\\\n\\delta_{B_{1}} \\\\\n\\delta_{\\psi}\n\\end{array}\\right)$</div>"}︡{"stdout":"\n\n\n"}︡{"html":"<div align='center'>Eigenvalues and eigenvectors in the $k \\rightarrow 0$ limit : </div>"}︡{"tex":{"tex":"\\left[\\left(-\\frac{1}{\\tau}, \\left[\\left(0,\\,0,\\,0,\\,0,\\,1\\right)\\right], 1\\right), \\left(0, \\left[\\left(1,\\,0,\\,0,\\,0,\\,0\\right), \\left(0,\\,1,\\,0,\\,0,\\,0\\right), \\left(0,\\,0,\\,1,\\,0,\\,0\\right), \\left(0,\\,0,\\,0,\\,1,\\,0\\right)\\right], 4\\right)\\right]","display":true}}︡{"stdout":"Numerical eigenvalues and eigenvectors for k > 0:\n\n"}︡{"stdout":"--------------------------\n\n"}︡{"stdout":"eta   =  1.00000000000000\ntau   =  0.000100000000000000\n--------------------------\nEigenvalue =  0.0\ndelta_rho  =  1.0\ndelta_u    =  0.0\ndelta_u1   =  0.0\ndelta_B1   =  0.0\ndelta_psi  =  0.0\n--------------------------\nEigenvalue =  1.06411716644\ndelta_rho  =  -0.215357106645\ndelta_u    =  0.861428426578\ndelta_u1   =  -0.0364727734249*I\ndelta_B1   =  0.0\ndelta_psi  =  -0.458510762468\n--------------------------\nEigenvalue =  16.4599069833\ndelta_rho  =  0.0300815011839\ndelta_u    =  -0.120326004736\ndelta_u1   =  0.0788037734365*I\ndelta_B1   =  0.0\ndelta_psi  =  0.989144438979\n--------------------------\nEigenvalue =  -10017.5240241\ndelta_rho  =  8.74231312256e-08\ndelta_u    =  -3.49692524903e-07\ndelta_u1   =  -0.00013938206093*I\ndelta_B1   =  0.0\ndelta_psi  =  0.999999990286\n--------------------------\nEigenvalue =  0.0\ndelta_rho  =  0.0\ndelta_u    =  0.0299865091057\ndelta_u1   =  -6.58486930972e-20*I\ndelta_B1   =  0.999550303522\ndelta_psi  =  -1.25304501849e-19\n--------------------------\n"}︡
︠6347b530-12dc-4d27-8f3e-6b58bc364164i︠






e205da3b-ddd4-49a9-b495-e456c9c47147





b31eb8b8-9b1-4db7-b04-357e710e2c







e205d3b-ddd4-499-b495-e456c9c47147

















