! ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
! 6.4 Effective lengths
! ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

write(*,*) ' '
write(*,*) '=== Effective Lengths ==='
write(2,*) ' '
write(2,*) '=== Effective Lengths ==='

! Allocate Corr and corrlen
ALLOCATE (corrlen(nparamorig))
! Go through each parameter
DO j=1,nparamorig
IF( Rflag(j)==2 .OR. Rflag(j)==3 ) THEN
! Assign A and B for a given lag
corrOK = 0
lag = -1
DO WHILE ( corrOK == 0 )
lag = lag + 1
ALLOCATE (A(nlength-nburn-lag))
ALLOCATE (B(nlength-nburn-lag))
m = 0
DO i=nburn+1,nlength-lag
m = m + 1
A(m) = R(j,i)
B(m) = R(j,i+lag)
END DO
! Call correl
call correl(nlength-nburn-lag,1,nlength-nburn-lag,A,B,autocorr)
!!write(*,*) 'Lag ',lag,' has autocorr = ',autocorr
! SUBROUTINE correl(n,istart,iend,A,B,corr)
IF( DABS(autocorr) .LT. 0.50D0 ) THEN
corrOK = 1
corrlen(j) = lag
ELSE IF( lag .GE. (nlength-nburn-1) ) THEN
corrOK = 1
corrlen(j) = lag
END IF
DEALLOCATE (A)
DEALLOCATE (B)
END DO
!write(*,*) Rpar(j) ,'; Corr. Length = ',corrlen(j),'; ',&
!           'Eff Length = ',INT( (nlength-nburn)/corrlen(j) )
write(2,*) Rpar(j) ,'; Corr. Length = ',corrlen(j),'; ',&
'Eff Length = ',INT( (nlength-nburn)/corrlen(j) )
END IF
END DO
! Find lowest effective length
lowesteff = 99.9D99
DO j=1,nparamorig
IF( ((nlength-nburn)/corrlen(j)) .LT. lowesteff ) THEN
IF( Rflag(j) .EQ. 2 .OR. Rflag(j) .EQ. 3 ) THEN
lowesteff = ((nlength-nburn)/corrlen(j))
lowesteffj = j
END IF
END IF
END DO

write(2,*) 'Lowest Eff. Length for ',Rpar(lowesteffj),' = ',lowesteff
write(2,*) 'Highest Av. Corr. Length = ',(nlength-nburn)/lowesteff
IF( lowesteff .GT. 1000.0D0 ) THEN
write(2,*) 'Suggest running ',&
INT(((1000.0D0/lowesteff)*(nlength-nburn))+nburn),' trials'
ELSE
write(2,*) '*** CAUTION! JAMMI RECOMMENDS INCREASING MCMC LENGTH TO ',&
INT(((1000.0D0/lowesteff)*(nlength-nburn))+nburn),' ***'
END IF
IF( lowesteff .LT. 1000.0D0 ) THEN
write(*,*) 'Lowest Eff. Length for ',Rpar(lowesteffj),' = ',lowesteff
write(*,*) 'Highest Av. Corr. Length = ',(nlength-nburn)/lowesteff
write(*,*) '*** CAUTION! JAMMI RECOMMENDS INCREASING MCMC LENGTH TO ',&
INT(((1000.0D0/lowesteff)*(nlength-nburn))+nburn),' ***'
END IF