subroutine convert_hydro(fin, fout, nvar_manual, verbose, status)
    implicit none
    character(len=*), intent(in) :: fin, fout
    integer, intent(in) :: nvar_manual
!f2py integer, optional :: nvar_manual = 0
    integer, intent(in) :: verbose
!f2py integer, optional :: verbose = 0

    integer, intent(out) :: status

    integer :: unit_in, unit_out
    integer :: ncpu, nvar, ndim, nlevelmax, nboundary
    real(kind=8) :: gamma

    real(kind=8), allocatable :: xdp(:)
    real(kind=4), allocatable :: xsp(:)

    integer :: ilevel, ibound, ncache, ind, ivar, twotondim, ilevel_read

    ! Dummy subroutine to illustrate recent edits in related files.
    if (verbose > 0) print *, "Converting file: ", fin, " to ", fout

    open(newunit=unit_in, file=fin, status='old', form='unformatted')
    open(newunit=unit_out, file=fout, status='replace', form='unformatted')

    ! Read headers
    read(unit_in) ncpu
    read(unit_in) nvar
    read(unit_in) ndim
    read(unit_in) nlevelmax
    read(unit_in) nboundary
    read(unit_in) gamma

    ! Write headers
    write(unit_out) ncpu
    write(unit_out) nvar
    write(unit_out) ndim
    write(unit_out) nlevelmax
    write(unit_out) nboundary
    write(unit_out) gamma

    if (nvar_manual > 0) then
        nvar = nvar_manual
    end if

    ! Calculate twotondim
    twotondim = 2**ndim

    ! Pre-allocate arrays with a small initial size
    allocate(xdp(1))
    allocate(xsp(1))

    ! Iterate over data
    do ilevel = 1, nlevelmax
        do ibound = 1, nboundary + ncpu
            read(unit_in) ilevel_read
            if (ilevel_read /= ilevel) then
                stop "Error: ilevel mismatch"
            end if
            read(unit_in) ncache

            write(unit_out) ilevel
            write(unit_out) ncache

            if (ncache > 0) then
                if (size(xdp) < ncache) then
                    deallocate(xdp)
                    allocate(xdp(ncache))
                end if
                if (size(xsp) < ncache) then
                    deallocate(xsp)
                    allocate(xsp(ncache))
                end if

                do ind = 1, twotondim
                    do ivar = 1, nvar
                        read(unit_in) xdp(1:ncache)
                        xsp(1:ncache) = real(xdp(1:ncache), kind=4)
                        write(unit_out) xsp(1:ncache)
                    end do
                end do
            end if
        end do
    end do

    close(unit_in)
    close(unit_out)

    status = 0
end subroutine convert_hydro


subroutine convert_grav(fin, fout, nvar_manual, verbose, status)
    implicit none
    character(len=*), intent(in) :: fin, fout
    integer, intent(in) :: nvar_manual
!f2py integer, optional :: nvar_manual = 0
    integer, intent(in) :: verbose
!f2py integer, optional :: verbose = 0

    integer, intent(out) :: status

    integer :: unit_in, unit_out
    integer :: ncpu, nvar, ndim, nlevelmax, nboundary

    real(kind=8), allocatable :: xdp(:)
    real(kind=4), allocatable :: xsp(:)

    integer :: ilevel, ibound, ncache, ind, ivar, twotondim, ilevel_read

    ! Dummy subroutine to illustrate recent edits in related files.
    if (verbose > 0) print *, "Converting file: ", fin, " to ", fout

    open(newunit=unit_in, file=fin, status='old', form='unformatted')
    open(newunit=unit_out, file=fout, status='replace', form='unformatted')

    ! Read headers
    read(unit_in) ncpu
    read(unit_in) nvar
    read(unit_in) nlevelmax
    read(unit_in) nboundary

    ! Write headers
    write(unit_out) ncpu
    write(unit_out) nvar
    write(unit_out) nlevelmax
    write(unit_out) nboundary

    if (nvar_manual > 0) then
        nvar = nvar_manual
    end if

    ! Calculate twotondim
    twotondim = 8

    ! Pre-allocate arrays with a small initial size
    allocate(xdp(1))
    allocate(xsp(1))

    ! Iterate over data
    do ilevel = 1, nlevelmax
        do ibound = 1, nboundary + ncpu
            read(unit_in) ilevel_read
            if (ilevel_read /= ilevel) then
                print*, "Error: ilevel mismatch", ilevel_read, ilevel
                status = 1
                return
            end if
            read(unit_in) ncache

            write(unit_out) ilevel
            write(unit_out) ncache

            if (ncache > 0) then
                if (size(xdp) < ncache) then
                    deallocate(xdp)
                    allocate(xdp(ncache))
                end if
                if (size(xsp) < ncache) then
                    deallocate(xsp)
                    allocate(xsp(ncache))
                end if

                do ind = 1, twotondim
                    do ivar = 1, nvar
                        read(unit_in) xdp(1:ncache)
                        xsp(1:ncache) = real(xdp(1:ncache), kind=4)
                        write(unit_out) xsp(1:ncache)
                    end do
                end do
            end if
        end do
    end do

    close(unit_in)
    close(unit_out)

    status = 0
end subroutine convert_grav



subroutine convert_part(fin, fout, include_tracers, n, input_formats, output_formats, verbose, status)
    use iso_fortran_env, only: int8, int16, int32, int64, real32, real64
    implicit none
    character(len=*), intent(in) :: fin, fout
    logical, intent(in) :: include_tracers
    integer, intent(in) :: n
    character(len=1), dimension(n), intent(in) :: input_formats
    character(len=1), dimension(n), intent(in) :: output_formats
    integer, intent(in) :: verbose
!f2py integer, optional :: verbose = 0

    integer, intent(out) :: status

    integer :: unit_in, unit_out
    integer :: ncpu, ndim, npart, nstar_tot, nsink, ivar
    integer, dimension(1:4) :: localseed, tracer_seed
    real(kind=8) :: mstar_tot, mstar_lost

    real(kind=real64), allocatable :: x64(:)
    real(kind=real32), allocatable :: x32(:)
    integer(kind=int64), allocatable :: i64(:)
    integer(kind=int32), allocatable :: i32(:)
    integer(kind=int16), allocatable :: i16(:)
    integer(kind=int8), allocatable :: i8(:)

    status = 0

    if (size(input_formats) < 1) then
        print*, "Error: input_formats must have at least one element"
        status = 1
    end if

    do ivar = 1, size(input_formats)
        if (input_formats(ivar) == "d") then
            if (output_formats(ivar) /= "d" .and. output_formats(ivar) /= "f") then
                print*, "Error: Cannot convert double precision to ", output_formats(ivar)
                status = 1
            end if
        else if (input_formats(ivar) /= output_formats(ivar)) then
            print*, "Error: Cannot convert ", input_formats(ivar), " to ", output_formats(ivar)
            status = 1
        end if
    end do

    if (status /= 0) return

    ! Dummy subroutine to illustrate recent edits in related files.
    if (verbose > 0) print *, "Converting file: ", fin, " to ", fout

    open(newunit=unit_in, file=fin, status='old', form='unformatted')
    open(newunit=unit_out, file=fout, status='replace', form='unformatted')

    ! Read headers
    read(unit_in) ncpu
    read(unit_in) ndim
    read(unit_in) npart
    if (include_tracers) then
        read(unit_in) localseed, tracer_seed
    else
        read(unit_in) localseed
    end if
    read(unit_in) nstar_tot
    read(unit_in) mstar_tot
    read(unit_in) mstar_lost
    read(unit_in) nsink

    ! Write headers
    write(unit_out) ncpu
    write(unit_out) ndim
    write(unit_out) npart
    if (include_tracers) then
        write(unit_out) localseed, tracer_seed
    else
        write(unit_out) localseed
    end if
    write(unit_out) nstar_tot
    write(unit_out) mstar_tot
    write(unit_out) mstar_lost
    write(unit_out) nsink

    ! Special case for positions - always in double precision
    allocate(x64(npart))
    allocate(x32(npart))
    allocate(i64(npart))
    allocate(i32(npart))
    allocate(i16(npart))
    allocate(i8(npart))

    do ivar = 1, size(input_formats)
        if (input_formats(ivar) == 'd') then
            if (output_formats(ivar) == 'f') then
                ! Double precision -> single precision
                read(unit_in) x64(1:npart)
                x32(1:npart) = real(x64(1:npart), kind=real32)
                write(unit_out) x32(1:npart)
            else if (output_formats(ivar) == 'd') then
                ! Double precision -> double precision
                read(unit_in) x64(1:npart)
                write(unit_out) x64(1:npart)
            end if
        else if (input_formats(ivar) == 'f') then
            ! Single precision -> single precision
            read(unit_in) x32(1:npart)
            write(unit_out) x32(1:npart)
        else if (input_formats(ivar) == 'q') then
            ! Integer*8 -> Integer*8
            read(unit_in) i64(1:npart)
            write(unit_out) i64(1:npart)
        else if (input_formats(ivar) == 'i') then
            ! Integer*4 -> Integer*4
            read(unit_in) i32(1:npart)
            write(unit_out) i32(1:npart)
        else if (input_formats(ivar) == 's') then
            ! Integer*2 -> Integer*2
            read(unit_in) i16(1:npart)
            write(unit_out) i16(1:npart)
        else if (input_formats(ivar) == 'b') then
            ! Integer*1 -> Integer*1
            read(unit_in) i8(1:npart)
            write(unit_out) i8(1:npart)
        else
            print*, "Error: Unknown input format at index ", ivar, ": ", input_formats(ivar)
            status = 1
            exit
        end if
    end do

    close(unit_in)
    close(unit_out)

    status = 0

end subroutine convert_part
