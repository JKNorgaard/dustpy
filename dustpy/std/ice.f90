subroutine jacobian_coagulation_generator(A, cStick, eps, iLF, iRM, iStick, m, phi, Rf, Rs, Sigma, SigmaFloor, alphaIce, dat, row, col, Nr, Nm)
  ! Ice coagulation Jacobian at interior radii (2..Nr-1). Fortran indexing.

  use math, only: kdelta
  implicit none

  ! ----------- Arguments ----------------
  integer,          intent(in)  :: Nr, Nm
  double precision, intent(in)  :: A(Nm, Nm)
  double precision, intent(in)  :: cStick(4, Nm, Nm)
  double precision, intent(in)  :: eps(Nm, Nm)
  integer,          intent(in)  :: iLF(Nm, Nm)   
  integer,          intent(in)  :: iRM(Nm, Nm)  
  integer,          intent(in)  :: iStick(4, Nm, Nm)
  double precision, intent(in)  :: m(Nm)
  double precision, intent(in)  :: phi(Nm, Nm) 
  double precision, intent(in)  :: Rf(Nr, Nm, Nm)
  double precision, intent(in)  :: Rs(Nr, Nm, Nm)
  double precision, intent(in)  :: Sigma(Nr, Nm)
  double precision, intent(in)  :: SigmaFloor(Nr, Nm)
  double precision, intent(in)  :: alphaIce(Nm, Nm)

  double precision, intent(out) :: dat((Nr-2)*Nm*Nm)
  integer,          intent(out) :: row((Nr-2)*Nm*Nm)
  integer,          intent(out) :: col((Nr-2)*Nm*Nm)

  ! ------------- Locals ----------------
  double precision :: agrid, ratef
  double precision :: jac(Nr, Nm, Nm)
  double precision :: N(Nr, Nm)
  double precision :: b, ce
  double precision :: D, F
  double precision :: D_m(Nm,Nm)
  integer :: ir, i, j, k, l, imax, q, start, p, s, jmax

  ! Determination of the mass bin range that contributes to
  ! full fragmentation
  p = Nm - minloc( m(:), 1, iRM(:, Nm) .EQ. -1 )

  ! Bandwidth
  agrid = LOG10( m(1)/m(Nm) ) / (1.d0 - Nm)
  q     = INT( CEILING( LOG10(2.d0)/agrid ) )

  ! b is defined as
  ! m(i+1) = m(i) * 10**a
  b = log10(m(1)/m(Nm) ) / (1.d0 - Nm)

  ! ce from Brauer et al. (2008) equation (A.6):
  ! m(k-1) + m(i) < m(k) for any i with i <= k - ce
  ce = int( -1.d0/b * log10(1.d0 - 10.d0**(-b) ) ) + 1

  ! init
  jac(:, :, :) = 0.d0
  dat(:) = 0.d0; row(:) = 0; col(:) = 0

  ! Number densities
  do ir=1, Nr
    N(ir, :) = Sigma(ir, :) / m(:)
  end do

  ! Normalization factor for sticking
  do i=1, Nm
    do j=1, Nm
      D_m(j, i) = m(j)/m(i)
    end do
  end do

  ! interior radii only
  do ir=2, Nr-1

    ! Largest mass bin that is above floor value
    imax = min( maxloc(m(:), 1, Sigma(ir, :) .GT. SigmaFloor(ir, :)), Nm )

    do i=1, imax

      jmax = MIN(Nm-1, INT(LOG10(10.d0**(b*Nm)-10.d0**(b*i))/b))
      do j=1, i

        ! ==========================================================
        !                       STICKING
        ! ==========================================================
        if (Rs(ir, j, i) .GT. 0.d0) then

          ! Calculate D and F terms
          if (j <= i+1 - ce) then
            D = - m(i+1) * m(j) / ((m(i+1)-m(i))*(m(i)+m(j)))
          else
            D = -1.d0
          end if
          if (i <= j+1 - ce) then
            F = m(j) * (m(j+1)-(m(j)+m(i))) / ((m(j+1)-m(j))*(m(j)+m(i)))
          else
            F = 0.d0
          end if

          ! loop through collision products
          do l=1,4
            k = iStick(l, j, i) + 1
            if (k .EQ. 0) cycle
            jac(ir, k, i) = jac(ir, k, i) + cStick(l, j, i) * Rs(ir, j, i) * m(k) * N(ir, j) / (m(i) + m(j))
          end do

          if (j <= jmax) then
            ! Add D and F terms
            jac(ir, i, i) = jac(ir, i, i) + Rs(ir, i, j) * D * N(ir, j)
            jac(ir, j, i) = jac(ir, j, i) + Rs(ir, j, i) * F * N(ir, j)
          end if
        end if ! sticking

        ! ==========================================================
        !                FRAGMENTATION / EROSION 
        ! ==========================================================
        if (Rf(ir, j, i) .GT. 0.d0) then
          ratef = Rf(ir, j, i) * N(ir, j)   ! multiplies I_i terms
        
          ! target contribution to fragments
          jac(ir, :, i) = jac(ir, :, i) + ratef * phi(ilf(j, i)+1, :) * alphaIce(j,i)

          ! Erosion
          if (j .LE. i-p-1) then
            k = irm(j, i) + 1
            if (k .EQ. i-1) then
              jac(ir, i-1, i) = jac(ir, i-1, i) + D_m(i-1,i) * eps(j, i)           * ratef
              jac(ir, i,   i) = jac(ir, i,   i) -             eps(j, i)            * ratef
            else
              jac(ir, k,   i) = jac(ir, k,   i) + D_m(k,   i) * eps(j, i)          * ratef
              jac(ir, k+1, i) = jac(ir, k+1, i) + D_m(k+1, i) * (1.d0 - eps(j, i)) * ratef
              jac(ir, i,   i) = jac(ir, i,   i) - ratef
            end if
          else
            ! full fragmentation
            jac(ir, i, i) = jac(ir, i, i) - ratef            
          end if
        end if ! fragmentation/erosion
      end do ! j
    end do ! i
  end do ! ir

  ! Filling the data array
  k = 1
  do ir=2, Nr-1
    start = (ir-1)*Nm - 1
    do i=1, Nm
      s = MAX(i-q, 1)
      do j=s, Nm
        dat(k) = jac(ir, i, j)
        row(k) = start + i
        col(k) = start + j
        k = k + 1
      end do
    end do
  end do

end subroutine jacobian_coagulation_generator

subroutine m_generator(A, cStick, eps, iLF, iRM, iStick, m, phi, Rf, Rs, Sigma, Sigma_Ice, SigmaFloor, alphaIce, MF, Nr, Nm)
  ! Subroutine generates the coupling matrix M for ice coagulation.

  use math, only: kdelta
  implicit none

  ! ---------- Arguments ----------
  integer,          intent(in)  :: Nr, Nm
  double precision, intent(in)  :: A(Nm, Nm)
  double precision, intent(in)  :: cStick(4, Nm, Nm)
  double precision, intent(in)  :: eps(Nm, Nm)
  integer,          intent(in)  :: iLF(Nm, Nm)        ! python indexing
  integer,          intent(in)  :: iRM(Nm, Nm)        ! python indexing
  integer,          intent(in)  :: iStick(4, Nm, Nm)
  double precision, intent(in)  :: m(Nm)
  double precision, intent(in)  :: phi(Nm, Nm)
  double precision, intent(in)  :: Rf(Nr, Nm, Nm)
  double precision, intent(in)  :: Rs(Nr, Nm, Nm)
  double precision, intent(in)  :: Sigma(Nr, Nm)
  double precision, intent(in)  :: Sigma_Ice(Nr, Nm)
  double precision, intent(in)  :: SigmaFloor(Nr, Nm)
  double precision, intent(in)  :: alphaIce(Nm, Nm)

  double precision, intent(out) :: MF(Nr, Nm, Nm)

  ! ---------- Locals ----------
  double precision :: agrid, ratef
  integer          :: q, p
  integer          :: ir, i, j, k, l, imax, jmax
  double precision :: b, ce
  double precision :: D, F

  ! Determination of the mass bin range that contributes to
  ! full fragmentation
  p = Nm - minloc( m(:), 1, iRM(:, Nm) .EQ. -1 )

  ! grid bandwidth
  agrid = LOG10( m(1)/m(Nm) ) / (1.d0 - Nm)
  q     = INT( CEILING( LOG10(2.d0)/agrid ) )

  ! b is defined as
  ! m(i+1) = m(i) * 10**b
  b = log10(m(1)/m(Nm) ) / (1.d0 - Nm)

  ! ce from Brauer et al. (2008) equation (A.6):
  ! m(k-1) + m(i) < m(k) for any i with i <= k - ce
  ce = int( -1.d0/b * log10(1.d0 - 10.d0**(-b) ) ) + 1

  ! init outputs
  MF(:,:,:) = 0.d0

  ! Loop over interior radii only
  do ir = 2, Nr-1
    ! Largest mass bin above floor
    imax = min( maxloc(m(:), 1, Sigma(ir, :) > SigmaFloor(ir, :)), Nm )

    do i = 1, imax
      jmax = MIN(Nm-1, INT(LOG10(10.d0**(b*Nm)-10.d0**(b*i))/b))
      do j = 1, i

        ! ==========================================================
        !                       STICKING
        ! ==========================================================
        if (Rs(ir, j, i) > 0.d0) then

          ! Calculate D and F terms
          if (i <= j+1 - ce) then
            D = - m(j+1) * m(i) / ((m(j+1)-m(j))*(m(j)+m(i)))
          else
            D = -1.d0
          end if
          if (j <= i+1 - ce) then
            F = m(i) * (m(i+1)-(m(i)+m(j))) / ((m(i+1)-m(i))*(m(i)+m(j)))
          else
            F = 0.d0
          end if

          ! Loop through collision products
          do l=1,4
            k = iStick(l, j, i) + 1
            if (k .eq. 0) cycle
            MF(ir, k, i) = MF(ir, k, i) + cStick(l, j, i) * m(k) * Rs(ir, j, i) * Sigma_ice(ir, j) / (m(i) + m(j))
          end do

          if (j<=jmax) then
            ! Add D and F terms
            MF(ir, j, i) = MF(ir, j, i) + Rs(ir, j, i) * Sigma_ice(ir, j) * D
            MF(ir, i, i) = MF(ir, i, i) + Rs(ir, i, j) * Sigma_ice(ir, j) * F
          end if
        end if ! sticking

        ! ==========================================================
        !                FRAGMENTATION / EROSION
        ! ==========================================================

        if (Rf(ir, j, i) .GT. 0.d0) then
          ratef = Rf(ir, j, i) * Sigma_ice(ir, j)
          ! Fragments distribution
          MF(ir, :, i) = MF(ir, :, i) + ratef * phi(ilf(j, i)+1, :)
          MF(ir, j, i) = MF(ir, j, i) - ratef
        end if ! fragmentation/erosion
      end do ! j
    end do   ! i
  end do     ! ir

end subroutine m_generator

subroutine coagulation_parameters_ice(m, cratRatio, cstick, cstick_ind, Xi, Nm)
  ! Subroutine calculates the sticking parameters needed to calculate the
  ! sticking sources for ice. The sticking matrix is calculated with the method
  ! described in appendices A.1. and A.2. of Brauer et al. (2008).
  !
  ! Parameters
  ! ----------
  ! m(Nm) : Mass grid
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! cstick(4, Nm, Nm) : Non-zero elements of sticking matrix
  ! cstick_ind(4, Nm, Nm) : location of non-zero elements in third dimension
  !                         in Python indexing starting from 0

  use math, only: kdelta, theta

  implicit none

  double precision, intent(in)  :: cratRatio
  double precision, intent(in)  :: m(Nm)
  double precision, intent(out) :: cstick(4, Nm, Nm)
  integer,          intent(out) :: cstick_ind(4, Nm, Nm)
  integer,          intent(in)  :: Nm
  double precision, intent(out) :: Xi(Nm, Nm)
  

  double precision :: a
  double precision :: cpod(Nm, Nm, Nm)
  double precision :: cpodmod(Nm, Nm, Nm)
  double precision :: dum(Nm, Nm, Nm)
  double precision :: D(Nm, Nm)
  double precision :: E(Nm, Nm)
  double precision :: eps
  double precision :: mi_m_mim1(Nm)
  double precision :: mip1_m_mi(Nm)
  double precision :: mip1_m_mim1(Nm)
  double precision :: mrm
  double precision :: mtot(Nm, Nm)
  integer :: ce
  integer :: i
  integer :: j
  integer :: jmax
  integer :: jmin
  integer :: k
  integer :: inc
  integer :: p

  ! Initialization
  cpod(:, :, :) = 0.d0
  cpodmod(:, :, :) = 0.d0
  cstick(:, :, :) = 0.d0
  Xi(:, :) = 0.d0
  cstick_ind(:, :, :) = -1
  D(:, :) = -1.d0
  mtot(:, :) = 0.d0

  ! Grid constants
  ! These grid constants are only valid for a regular logarithmic grid.
  ! For other grids this coagulation method will produce wrong results.

  ! a is defined as
  ! m(i+1) = m(i) * 10**a
  a = log10(m(1)/m(Nm) ) / (1.d0 - Nm)
  
  ! ce from Brauer et al. (2008) equation (A.6):
  ! m(k-1) + m(i) < m(k) for any i with i <= k - ce
  ce = int( -1.d0/a * log10(1.d0 - 10.d0**(-a) ) ) + 1

  ! Since the grid is regular logarithmic particles with masses m(i) and m(j)
  ! fully fragment if j >= i-p.
  p = floor(log10(cratRatio)/a)

  ! COAGULATION

  ! Helper arrays to avoid out-of-bounds errors
  ! mi_m_mim1(i) = m(i) - m(i-1)
  mi_m_mim1(:)   = m(:) * ( 1.d0    - 10.d0**(-a))
  ! mip1_m_mi(i) = m(i+1) - m(i)
  mip1_m_mi(:)   = m(:) * (10.d0**a -  1.d0      )
  ! mip1_m_mim1(i) = m(i+1) - m(i-1)
  mip1_m_mim1(:) = m(:) * (10.d0**a - 10.d0**(-a))

  do i=1, Nm
    do j=1, i-p-1
      Xi(j, i) = (m(j)) / m(i)
    end do
    jmin = max(1, i-p)
    do j=jmin, i 
      Xi(j, i) = 1.d0
    end do
    do j=1, Nm


      ! Total mass of collision partners
      mtot(j, i) = m(j) + m(i)

      ! Podolak coefficients according Brauer et al. (2008), A.1.
      ! Meaning of cpod(k, j, i):
      ! if mmass (i) collides with mass m(j), the fraction of eps(j, i)
      ! will be filled into mass m(k) and the fraction of 1 - eps(j, i)
      ! into mass m(k+1), since the total mass of both collision partners
      ! will lie between the two masses m(k) and m(k+1).
      k = minloc(m, 1, mtot(j, i) < m) - 1
      if(k .gt. 0) then
        eps             = ( m(k+1) - mtot(j, i) ) / mip1_m_mi(k)
        cpod(k, j, i)   = eps
        cpod(k+1, j, i) = 1.d0 - eps
      end if

      ! Modified Podolak coefficients according Brauer et al. (2008), A.2.
      ! If the two colliding masses differ by more than fifteen orders of 
      ! magnitude double precision is not precise enough, i.e.,
      ! m(i) + m(j) = m(i) from a numerical point of view. This violates
      ! mass conservation. To prevent this, the following procedure is applied.
      ! Please read Brauer et al. (2008), appendix A.2. for details.

      ! D matrix
      if(j .le. i+1-ce) then
        D(j, i) = - m(j) /  mip1_m_mi(i)
      end if

      ! E matrix
      if(j .le. i-ce) then
        E(j, i) = m(j) / mi_m_mim1(i)
      else
        E(j, i) = ( 1.d0 - ( m(j) - mi_m_mim1(i) ) / mip1_m_mi(i) )  *  theta( mip1_m_mim1(i) - m(j)  )
      end if
    end do
  end do

  ! Building modified Podolak coefficients
  ! The outer sum does not need to go all the way up to Nm,
  ! because sticking collisions involving m(Nm) will be outside the mass grid.
  ! The inner sum will only go to a maximum integer number jmax, such that
  ! the resulting mass is within the mass grid.
  do i=1, Nm-1
    jmax = MIN(Nm-1, INT(LOG10(10.d0**(a*Nm)-10.d0**(a*i))/a))
    do j=1, jmax
      ! Only if resulting mass inside of grid
      do k=1, Nm
        cpodmod(k, j, i) = 0.5d0 * kdelta(j, i) * cpod(k, j, i) &
                           & + cpod(k, j, i) * theta( (k-j)*1.d0 - 1.5d0 ) * theta( (j-i)*1.d0 - 0.5d0 ) &
                           & + kdelta(j, k-1) * e(i, k) * theta( (k-i)*1.d0 - 1.5d0 )
      end do
    end do
  end do

  ! cpodmod so far is not symmetric:
  ! cpodmod(k, j, i) + cpodmod(k, i, j) is the fraction that gets added to
  ! mass m(k), if masses m(i) and m(j) collide.
  ! Here we make it symmetric and the discard one half.
  ! So cpodmod(k, j, i) contains the full value if j <= i and is zero,
  ! if j > i.
  dum(:, :, :) = cpodmod(:, :, :)
  do i=1, Nm
    do j=1, i
      cpodmod(:, i, j) = 0.d0
      cpodmod(:, j, i) = dum(:, j, i) + dum(:, i, j)
    end do
  end do

  ! copodmod(:, j, i) has at most four non-zero elements for any combination of (j, i).
  ! We only store the non-zero elemtens.
  do i=1, Nm
    do j=1, i
      inc = 1
      do k=1, Nm
        if (cpodmod(k, j, i) .NE. 0.d0) then
          cstick_ind(inc, j, i) = k - 1 ! Minus one to convert to Python indexing
          cstick(inc, j, i) = cpodmod(k, j, i)
          inc = inc + 1
        end if
      end do
    end do
  end do

end subroutine coagulation_parameters_ice