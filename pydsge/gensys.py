#!/bin/python
# -*- coding: utf-8 -*-

"""contains functions related to (re)compiling the model with different parameters
"""

import time
import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
from grgrlib import fast0, eig, re_bk, shredder
from .engine import preprocess
from .stats import post_mean


def preprocess_evs(self):

    A, N = self.sys[0:2]

    wa, vla = sl.eig(A)
    wn, vln = sl.eig(N)
    vra = nl.inv(vla)
    vrn = nl.inv(vln)

    self.evs = vra, wa, vla, vrn, wn, vln

    return 


def gen_sys(self, par=None, l_max=None, k_max=None, tol=1e-8, verbose=True):
    """Generate system matrices expressed in the one-sided, first-order compressed dimensionality reduction given a set of parameters. 

    Details can be found in "Efficient Solution of Models with Occasionally Binding Constraints" (Gregor Boehl).
    If no parameters are given this will default to the calibration in the `yaml` file.

    Parameters
    ----------
    par : array or list, optional
        The parameters to parse into the transition function. (defaults to calibration in `yaml`)
    l_max : int, optional
        The expected number of periods *until* the constraint binds (defaults to 3).
    k_max : int, optional
        The expected number of periods for which the constraint binds (defaults to 17).
    tol : float, optional
    verbose : bool or int, optional
        Level of verbosity
    """

    st = time.time()

    # set default values of l_max & k_max
    if l_max is not None:
        if l_max < 2:
            print('[get_sys:]'.ljust(15, ' ') +
                  ' `l_max` must be at least 2 (is %s). Correcting...' % l_max)
            l_max = 2
        # effective l_max is one lower because algorithm exists on l_max
        l_max += 1
        # TODO: test if true

    elif hasattr(self, 'lks'):
        l_max = self.lks[0]
    else:
        l_max = 3

    if k_max is not None:
        pass
    elif hasattr(self, 'lks'):
        k_max = self.lks[1]
    else:
        k_max = 17

    self.lks = np.array([l_max, k_max])

    self.par = self.p0() if par is None else list(par)
    try:
        self.ppar = self.pcompile(self.par)  # parsed par
    except TypeError:
        raise SyntaxError(
            "At least one parameter is a function of other parameters, and should be declared in `parafunc`.")

    if not self.const_var:
        raise NotImplementedError('Package is only meant to work with OBCs')

    # fix value of x_bar
    if 'x_bar' in [p.name for p in self.parameters]:
        x_bar = self.par[[p.name for p in self.parameters].index('x_bar')]
    elif 'x_bar' in self.parafunc[0]:
        pf = self.parafunc
        x_bar = pf[1](self.par)[pf[0].index('x_bar')]
    else:
        print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        x_bar = -1

    # start
    vv0 = np.array([v.name for v in self.variables])

    # z-space is the space of the original variables
    dimz = len(vv0)
    dimeps = len(self.shocks)
    # y-space = space of original variables augmented by the shocks
    dimy = dimz+dimeps

    AA0 = self.AA(self.ppar)              # forward
    BB0 = self.BB(self.ppar)              # contemp
    CC0 = self.CC(self.ppar)              # backward
    DD0 = -self.PSI(self.ppar).astype(float)
    fbc = self.bb(self.ppar).flatten().astype(float)  # constraint
    fd0 = -self.bb_PSI(self.ppar).flatten().astype(float)  # constraint
    fb0 = fbc[:dimz]
    fc0 = fbc[dimz:]

    # observables from z
    ZZ0 = self.ZZ0(self.ppar)
    ZZ1 = self.ZZ1(self.ppar).squeeze()
    c_arg = list(vv0).index(str(self.const_var))

    # convention: constraint var is first var in z-representation
    AA0[:, [0, c_arg]] = AA0[:, [c_arg, 0]]
    BB0[:, [0, c_arg]] = BB0[:, [c_arg, 0]]
    CC0[:, [0, c_arg]] = CC0[:, [c_arg, 0]]
    fc0[[0, c_arg]] = -fc0[[c_arg, 0]]/fb0[c_arg]
    fb0[[0, c_arg]] = -fb0[[c_arg, 0]]/fb0[c_arg]
    ZZ0[:, [0, c_arg]] = ZZ0[:, [c_arg, 0]]
    vv0[[0, c_arg]] = vv0[[c_arg, 0]]

    # create representation in y-space
    AA0 = np.pad(AA0, ((0, dimeps), (0, dimeps)))
    BB0 = sl.block_diag(BB0, np.eye(dimeps))
    CC0 = np.block([[CC0, DD0], [np.zeros((dimeps, dimy))]])
    fb0 = np.pad(fb0, (0, dimeps))
    fc0 = np.hstack((fc0, fd0))
    vv0 = np.hstack((vv0, self.shocks))

    # create dummy for r(+1)
    AA1 = np.pad(AA0, ((0,1),(1,0)))
    BB1 = np.pad(BB0, ((0,1),(1,0)))
    CC1 = np.pad(CC0, ((0,1),(1,0)))
    fb1 = np.pad(fb0, (1,0))
    fc1 = np.pad(fc0, (1,0))
    vv1 = np.hstack(('rp1_dummy', vv0))

    # ...and populate it
    AA1[-1] = fb1
    BB1[-1] = fc1
    AA1[-1,1] = 0
    BB1[-1,0] = -1

    # determine which variables shall be states
    inq = ~fast0(CC1,0) | ~fast0(fc1)
    dimq = sum(inq)
    dimx = sum(~inq)

    # find auxiliary variables for y(+1)
    eia = ~fast0(AA1, 1)

    u, s, v = nl.svd((AA1[eia]), full_matrices=False)
    S = np.diag(s) @ v
    dimp = len(s)

    AA2 = np.pad(np.eye(dimp), ((0,dimy+1),(0,0)))
    BU1 = np.vstack((u.T.conj() @ BB1[eia], BB1[~eia], fb1, -S))
    BR1 = np.vstack((u.T.conj() @ BB1[eia], BB1[~eia], np.zeros_like(fb1), -S))
    BR1[-dimp-1,1] = -1
    BP1 = np.pad(np.eye(dimp), ((dimy+1,0),(0,0)))
    CU1 = np.vstack((u.T.conj() @ CC1[eia], CC1[~eia], fc1, np.zeros_like(S)))[:,inq]
    CR1 = np.vstack((u.T.conj() @ CC1[eia], CC1[~eia], np.zeros_like(fc1), np.zeros_like(S)))[:,inq]
    # keep track of position of constraint value
    gg1 = np.pad([1],(dimy,dimp))

    q,r = sl.qr(BU1[:,~inq])

    AUP0 = q.T.conj() @ AA2
    BUQ0 = q.T.conj() @ BU1[:,inq]
    BUP0 = q.T.conj() @ BP1
    CUQ0 = q.T.conj() @ CU1

    # r(+1) should not depend on static vars
    assert np.allclose(r[0,1:],0)
    # r(+1) should not depend on (t-1)-vars
    assert np.allclose(CUQ0[0],0)

    # with fap & fbq we can check whether constraint holds
    fap = -AUP0[0]/r[0,0]
    fbq = -BUQ0[0]/r[0,0]
    ff = np.hstack((fap, fbq))

    # obtain x (unconstraint sys)
    TUI = nl.inv(r[1:dimx,1:])
    TUAP = -TUI @ AUP0[1:dimx]
    TUBQ = -TUI @ BUQ0[1:dimx]
    TUBP = -TUI @ BUP0[1:dimx]
    TUCQ = -TUI @ CUQ0[1:dimx]
    # TU = np.hstack((TUAP, TUBQ, TUBP, TUCQ))

    TU = np.zeros((dimy,2*(dimp+dimq)))
    TU[inq[1:]] = np.pad(np.eye(dimq), ((0,0),(dimp, dimp+dimq)))
    TU[~inq[1:]] = np.hstack((TUAP, TUBQ, TUBP, TUCQ))

    if verbose:
        print('det & cond T', nl.det(r[:dimx]), nl.cond(r[:dimx]))

    # do same for constraint sys
    q,r = sl.qr(BR1[:,~inq])

    ARP0 = q.T.conj() @ AA2
    BRQ0 = q.T.conj() @ BR1[:,inq]
    BRP0 = q.T.conj() @ BP1
    CRQ0 = q.T.conj() @ CR1
    gg2 = q.T.conj() @ gg1

    # the results should be identical
    assert np.allclose(fap, -ARP0[0]/r[0,0])
    assert np.allclose(fbq, -BRQ0[0]/r[0,0])

    # obtain x (constraint sys)
    TRI = nl.inv(r[1:dimx,1:])
    TRAP = -TRI @ ARP0[1:dimx]
    TRBQ = -TRI @ BRQ0[1:dimx]
    TRBP = -TRI @ BRP0[1:dimx]
    TRCQ = -TRI @ CRQ0[1:dimx]

    TR0 = np.zeros((dimy,2*(dimp+dimq)))
    TR0[inq[1:]] = np.pad(np.eye(dimq), ((0,0),(dimp, dimp+dimq)))
    TR0[~inq[1:]] = np.hstack((TRAP, TRBQ, TRBP, TRCQ))

    TR1 = np.zeros(dimy)
    TR1[~inq[1:]] = -TRI @ gg2[1:dimx] * x_bar

    if verbose:
        print('det & cond T', nl.det(r[:dimx]), nl.cond(r[:dimx]))

    # prepare observation function
    HU0 = ZZ0 @ TU[:-dimeps]
    HU1 = ZZ1
    HR0 = ZZ0 @ TR0[:-dimeps]
    HR1 = ZZ0 @ TR1[:-dimeps] + ZZ1

    # create system matrices
    PU0 = -np.hstack((AUP0[dimx:], BUQ0[dimx:]))
    NU0 = np.hstack((BUP0[dimx:], CUQ0[dimx:]))
    PR0 = -np.hstack((ARP0[dimx:], BRQ0[dimx:]))
    NR0 = np.hstack((BRP0[dimx:], CRQ0[dimx:]))
    gg3 = gg2[dimx:]

    # find linear RE solution for when constraint does not bind
    omg = re_bk(NU0, PU0, d_endo=dimp)
    J = np.hstack((np.eye(omg.shape[0]), -omg))

    # desingularization of P
    U, s, V = nl.svd(PR0)
    s0 = s < tol

    PR1 = U.T.conj() @ PR0
    NR1 = U.T.conj() @ NR0
    gg4 = U.T.conj() @ gg3
    PU1 = U.T.conj() @ PU0
    NU1 = U.T.conj() @ NU0

    # actual desingularization by iterating equations in M forward
    PR1[s0] = NR1[s0]
    PU1[s0] = NU1[s0]

    # future values must take future (l,k) into account. Not (yet) implemented
    if not fast0(gg4[s0], 2):
        raise NotImplementedError('The system depends directly or indirectly on whether the constraint holds in a future period.\n')

    if verbose:
        print('PU1 (det, cond, max error): ', nl.det(PU1), nl.cond(
            PU1), abs(nl.inv(PU1) @ PU1 - np.eye(PU1.shape[0])).max())
        print('PR1 (det, cond, max error): ', nl.det(PR1), nl.cond(
            PR1), abs(nl.inv(PR1) @ PR1 - np.eye(PR1.shape[0])).max())

    A = nl.inv(PU1) @ NU1
    N = nl.inv(PR1) @ NR1
    g = nl.inv(PR1) @ gg4

    # finally add relevant stuff to the class
    self.vv = vv0[:-len(self.shocks)]
    self.inq = inq[1:-dimeps]
    self.dimq = dimq
    self.dimy = dimy
    self.dimz = dimz
    self.dimeps = dimeps

    self.hx = ZZ0, ZZ1
    self.sys = A, N, J, g, x_bar, ff, TU, TR0, TR1, HU0, HU1, HR0, HR1

    # preprocess all system matrices until (l_max, k_max)
    preprocess(self, verbose)

    # precalculate eigenvalues and eigenvectors
    preprocess_evs(self)

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+' Creation of system matrices finished in %ss.' %
              np.round(time.time() - st, 3))

    return self


def gen_sys_from_ext(AA, BB, CC, DD, fb, fc, x_bar, vv, shocks, const_var, l_max=None, k_max=None, tol=1e-8, verbose=True):
    """Generate system matrices expressed in the one-sided, first-order compressed dimensionality reduction given a set of parameters. 

    Details can be found in "Efficient Solution of Models with Occasionally Binding Constraints" (Gregor Boehl).
    If no parameters are given this will default to the calibration in the `yaml` file.

    Parameters
    ----------
    l_max : int, optional
        The expected number of periods *until* the constraint binds (defaults to 3).
    k_max : int, optional
        The expected number of periods for which the constraint binds (defaults to 17).
    tol : float, optional
    verbose : bool or int, optional
        Level of verbosity
    """

    st = time.time()

    # create dummy DSGE class and self-assign
    class DSGE_DUMMY:
        pass

    self = DSGE_DUMMY()

    from .parser import DSGE
    from .__init__ import example

    self = DSGE.read(example[0])

    # set default values of l_max & k_max
    if l_max is not None:
        if l_max < 2:
            print('[get_sys:]'.ljust(15, ' ') +
                  ' `l_max` must be at least 2 (is %s). Correcting...' % l_max)
            l_max = 2
        # effective l_max is one lower because algorithm exists on l_max
        l_max += 1
        # TODO: test if true
    else:
        l_max = 3

    if k_max is not None:
        pass
    else:
        k_max = 17

    self.lks = np.array([l_max, k_max])

    # start
    vv0 = vv

    # z-space is the space of the original variables
    dimz = len(vv0)
    dimeps = len(shocks)
    # y-space = space of original variables augmented by the shocks
    dimy = dimz+dimeps

    AA0 = AA
    BB0 = BB
    CC0 = CC
    DD0 = DD

    fb0 = fb
    fc0 = fc
    fd0 = np.zeros(dimeps)

    # observables from z
    # ZZ0 = self.ZZ0(self.ppar)
    # ZZ1 = self.ZZ1(self.ppar).squeeze()
    c_arg = list(vv0).index(const_var)

    # convention: constraint var is first var in z-representation
    AA0[:, [0, c_arg]] = AA0[:, [c_arg, 0]]
    BB0[:, [0, c_arg]] = BB0[:, [c_arg, 0]]
    CC0[:, [0, c_arg]] = CC0[:, [c_arg, 0]]
    fc0[[0, c_arg]] = -fc0[[c_arg, 0]]/fb0[c_arg]
    fb0[[0, c_arg]] = -fb0[[c_arg, 0]]/fb0[c_arg]
    # ZZ0[:, [0, c_arg]] = ZZ0[:, [c_arg, 0]]
    vv0[[0, c_arg]] = vv0[[c_arg, 0]]

    # create representation in y-space
    AA0 = np.pad(AA0, ((0, dimeps), (0, dimeps)))
    BB0 = sl.block_diag(BB0, np.eye(dimeps))
    CC0 = np.block([[CC0, DD0], [np.zeros((dimeps, dimy))]])
    fb0 = np.pad(fb0, (0, dimeps))
    fc0 = np.hstack((fc0, fd0))
    vv0 = np.hstack((vv0, shocks))

    # create dummy for r(+1)
    AA1 = np.pad(AA0, ((0,1),(1,0)))
    BB1 = np.pad(BB0, ((0,1),(1,0)))
    CC1 = np.pad(CC0, ((0,1),(1,0)))
    fb1 = np.pad(fb0, (1,0))
    fc1 = np.pad(fc0, (1,0))
    vv1 = np.hstack(('rp1_dummy', vv0))

    # ...and populate it
    AA1[-1] = fb1
    BB1[-1] = fc1
    AA1[-1,1] = 0
    BB1[-1,0] = -1

    # determine which variables shall be states
    inq = ~fast0(CC1,0) | ~fast0(fc1)
    dimq = sum(inq)
    dimx = sum(~inq)

    # find auxiliary variables for y(+1)
    eia = ~fast0(AA1, 1)
    ina = ~fast0(AA1, 0)

    if sum(ina) >= sum(eia):

        u, s, v = nl.svd((AA1[eia]), full_matrices=False)
        S = np.diag(s) @ v
        dimp = len(s)

        AA2 = np.pad(np.eye(dimp), ((0,dimy+1),(0,0)))
        BU1 = np.vstack((u.T.conj() @ BB1[eia], BB1[~eia], fb1, -S))
        BR1 = np.vstack((u.T.conj() @ BB1[eia], BB1[~eia], np.zeros_like(fb1), -S))
        BR1[-dimp-1,1] = -1
        BP1 = np.pad(np.eye(dimp), ((dimy+1,0),(0,0)))
        CU1 = np.vstack((u.T.conj() @ CC1[eia], CC1[~eia], fc1, np.zeros_like(S)))[:,inq]
        CR1 = np.vstack((u.T.conj() @ CC1[eia], CC1[~eia], np.zeros_like(fc1), np.zeros_like(S)))[:,inq]

    else:

        dimp = sum(ina)
        S = np.zeros((dimp,dimy+1))
        S[:,ina] = np.eye(dimp)

        AA2 = np.pad(AA1[:,ina], ((0,dimp+1),(0,0)))
        BU1 = np.vstack((BB1, fb1, -S))
        BR1 = np.vstack((BB1, np.zeros_like(fb1), -S))
        BR1[-dimp-1,1] = -1
        BP1 = np.pad(np.eye(dimp), ((dimy+1,0),(0,0)))
        CU1 = np.vstack((CC1, fc1, np.zeros_like(S)))[:,inq]
        CR1 = np.vstack((CC1, np.zeros_like(fc1), np.zeros_like(S)))[:,inq]

    # keep track of position of constraint value
    gg1 = np.pad([1],(dimy,dimp))

    q,r = sl.qr(BU1[:,~inq])

    AUP0 = q.T.conj() @ AA2
    BUQ0 = q.T.conj() @ BU1[:,inq]
    BUP0 = q.T.conj() @ BP1
    CUQ0 = q.T.conj() @ CU1

    # r(+1) should not depend on static vars
    assert np.allclose(r[0,1:],0)
    # r(+1) should not depend on (t-1)-vars
    assert np.allclose(CUQ0[0],0)

    # with fap & fbq we can check whether constraint holds
    fap = -AUP0[0]/r[0,0]
    fbq = -BUQ0[0]/r[0,0]
    ff = np.hstack((fap, fbq))

    # obtain x (unconstraint sys)
    TUI = nl.inv(r[1:dimx,1:])
    TUAP = -TUI @ AUP0[1:dimx]
    TUBQ = -TUI @ BUQ0[1:dimx]
    TUBP = -TUI @ BUP0[1:dimx]
    TUCQ = -TUI @ CUQ0[1:dimx]

    TU = np.zeros((dimy,2*(dimp+dimq)))
    TU[inq[1:]] = np.pad(np.eye(dimq), ((0,0),(dimp, dimp+dimq)))
    TU[~inq[1:]] = np.hstack((TUAP, TUBQ, TUBP, TUCQ))

    if verbose:
        print('det & cond T', nl.det(r[:dimx]), nl.cond(r[:dimx]))

    # do same for constraint sys
    q,r = sl.qr(BR1[:,~inq])

    ARP0 = q.T.conj() @ AA2
    BRQ0 = q.T.conj() @ BR1[:,inq]
    BRP0 = q.T.conj() @ BP1
    CRQ0 = q.T.conj() @ CR1
    gg2 = q.T.conj() @ gg1

    # the results should be identical
    assert np.allclose(fap, -ARP0[0]/r[0,0])
    assert np.allclose(fbq, -BRQ0[0]/r[0,0])

    # obtain x (constraint sys)
    TRI = nl.inv(r[1:dimx,1:])
    TRAP = -TRI @ ARP0[1:dimx]
    TRBQ = -TRI @ BRQ0[1:dimx]
    TRBP = -TRI @ BRP0[1:dimx]
    TRCQ = -TRI @ CRQ0[1:dimx]

    TR0 = np.zeros((dimy,2*(dimp+dimq)))
    TR0[inq[1:]] = np.pad(np.eye(dimq), ((0,0),(dimp, dimp+dimq)))
    TR0[~inq[1:]] = np.hstack((TRAP, TRBQ, TRBP, TRCQ))

    TR1 = np.zeros(dimy)
    TR1[~inq[1:]] = -TRI @ gg2[1:dimx] * x_bar

    if verbose:
        print('det & cond T', nl.det(r[:dimx]), nl.cond(r[:dimx]))

    # prepare observation function
    # HU0 = ZZ0 @ TU[:-dimeps]
    # HU1 = ZZ1
    # HR0 = ZZ0 @ TR0[:-dimeps]
    # HR1 = ZZ0 @ TR1[:-dimeps] + ZZ1

    # create system matrices
    PU0 = -np.hstack((AUP0[dimx:], BUQ0[dimx:]))
    NU0 = np.hstack((BUP0[dimx:], CUQ0[dimx:]))
    PR0 = -np.hstack((ARP0[dimx:], BRQ0[dimx:]))
    NR0 = np.hstack((BRP0[dimx:], CRQ0[dimx:]))
    gg3 = gg2[dimx:]

    print(PU0.shape)

    inpu = ~fast0(PU0, 0)
    innu = ~fast0(NU0, 0)

    # find linear RE solution for when constraint does not bind
    omg = re_bk(NU0, PU0, d_endo=dimp)
    J = np.hstack((np.eye(omg.shape[0]), -omg))

    # desingularization of P
    U, s, V = nl.svd(PR0)
    s0 = s < tol

    PR1 = U.T.conj() @ PR0
    NR1 = U.T.conj() @ NR0
    gg4 = U.T.conj() @ gg3
    PU1 = U.T.conj() @ PU0
    NU1 = U.T.conj() @ NU0

    # actual desingularization by iterating equations in M forward
    PR1[s0] = NR1[s0]
    PU1[s0] = NU1[s0]

    # future values must take future (l,k) into account. Not (yet) implemented
    # if not fast0(gg4[s0], 2):
        # raise NotImplementedError('The system depends directly or indirectly on whether the constraint holds in a future period.\n')

    if verbose:
        print('PU1 (det, cond, max error): ', nl.det(PU1), nl.cond(
            PU1), abs(nl.inv(PU1) @ PU1 - np.eye(PU1.shape[0])).max())
        print('PR1 (det, cond, max error): ', nl.det(PR1), nl.cond(
            PR1), abs(nl.inv(PR1) @ PR1 - np.eye(PR1.shape[0])).max())

    A = nl.inv(PU1) @ NU1
    N = nl.inv(PR1) @ NR1
    g = nl.inv(PR1) @ gg4

    # finally add relevant stuff to the class

    self['shk_ordering'] = shocks
    self.vv = vv0[:-len(shocks)]
    self.inq = inq[1:-dimeps]
    self.dimq = dimq
    self.dimy = dimy
    self.dimz = dimz
    self.dimeps = dimeps

    # self.hx = ZZ0, ZZ1
    # self.sys = A, N, J, g, x_bar, ff, TU, TR0, TR1, HU0, HU1, HR0, HR1
    self.sys = A, N, J, g, x_bar, ff, TU, TR0, TR1, np.zeros_like(TR0), np.zeros_like(TR1), np.zeros_like(TR0), np.zeros_like(TR1)

    # preprocess all system matrices until (l_max, k_max)
    preprocess(self, verbose)

    # precalculate eigenvalues and eigenvectors
    # preprocess_evs(self)

    # some sane defaults
    self.debug = False

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+' Creation of system matrices finished in %ss.' %
              np.round(time.time() - st, 3))

    return self
