"""
A module for Linear Gaussian State Space models.


Classes
-------
StateSpaceModel
LinearDSGEModel
"""
import numpy as np
import pandas as p

from scipy.linalg import solve_discrete_lyapunov

# from .gensys import gensys
# from .filters import chand_recursion, kalman_filter, filter_and_smooth

# filt_choices = {'chand_recursion': chand_recursion,
                # 'kalman_filter': kalman_filter}

class StateSpaceModel(object):
    r"""
    Object for holding state space model

    .. math::

    s_t &=& T(\theta) s_{t-1} + R(\theta) \epsilon_t,
    \quad \epsilon_t \sim N(0,Q(\theta)) \\

    y_t &=& D(\theta) + Z(\theta) s_{t} + \eta_t,
    \quad \eta_t  \sim N(0, H(\theta))

    Attributes
    ----------
    yy : array_like or pandas.DataFrame
        Dataset containing the observables of the model.

    t0 : int, optional
        Number of initial observations to condition on for
        likelihood evaluation. The default is 0.

    shock_names : list or None, optional
        Names of the shocks.

    state_names : list or None, optional
        Names of the states.

    obs_names : list or None, optional
        Names of observables.

    Methods
    -------
    TT(para), RR(para), QQ(para)
        Define the state transition matrices as function of a parameter vector.
    DD(para), ZZ(para), HH(para)
        Define the observable transition matrices as function of a parameter vector.
    log_lik(para)
        Computes the likelihood of the model at parameter value para.
    impulse_response(para, h=20)
        Computes the impulse response function at parameter value para.
    pred(para, h=20, shocks=True, append=False)
        Simulates a draw from the predictive distribution at parameter
        value para.
    kf_everything(para)
        Generates the filtered and smoothed posterior means of the state vector.
    """

    fast_filter = 'chand_recursion'

    def __init__(self, yy, TT, RR, QQ, DD, ZZ, HH, t0=0,
                 shock_names=None, state_names=None, obs_names=None):

        if len(yy.shape) < 2:
            yy = np.swapaxes(np.atleast_2d(yy), 0, 1)

        self.yy = yy

        self.TT = TT
        self.RR = RR
        self.QQ = QQ
        self.DD = DD
        self.ZZ = ZZ
        self.HH = HH


        self.t0 = t0

        self.shock_names = None
        self.state_names = None
        self.obs_names = None

    def log_lik(self, para, *args, **kwargs):
        """
        Computes the log likelihood of the model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        t0 : int, optional
            Number of initial observations to condition on. 
        y : 2d array-like, optional
            Dataset of observables (T x nobs). The default is the observable set pass during
            class instantiation.
        P0 : 2d array-like or string, optional
            [ns x ns] initial covariance matrix of states, or `unconditional` to use the one
            associated with the invariant distribution.  The default is `unconditional.`


        Returns
        -------
        lik : float
            The log likelihood.


        See Also
        --------
        StateSpaceModel.kf_everything

        """

        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        if np.isnan(yy).any().any():
            default_filter = 'kalman_filter'
        else:
            default_filter = 'chand_recursion'

        filt = kwargs.pop('filter', default_filter)
        filt_func = filt_choices[filt]

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para)

        if (np.isnan(TT)).any():
            lik = -1000000000000.0
            return lik

        if P0=='unconditional':
            P0 = solve_discrete_lyapunov(TT, RR.dot(QQ).dot(RR.T))

        lik = filt_func(np.asarray(yy), TT, RR, QQ,
                        np.asarray(DD,dtype=float),
                        np.asarray(ZZ,dtype=float),
                        np.asarray(HH,dtype=float),
                        np.asarray(P0,dtype=float),t0=t0)
        return lik


    def kf_everything(self, para, *args, **kwargs):
        """
        Runs the kalman filter and returns objects of interest.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        t0 : int, optional
            Number of initial observations to condition on. 
        y : 2d array-like, optional
            Dataset of observables (T x nobs). The default is the observable set pass during
            class instantiation.
        P0 : 2d arry-like or string, optional
            [ns x ns] initial covariance matrix of states, or `unconditional` to use the one
            associated with the invariant distribution.  The default is `unconditional.`


        Returns
        -------
        results : dict of p.DataFrames with
             `log_lik` -- the sequence of log likelihoods
             `filtered_means' -- the filtered means of the states 
             `filtered_std' -- the filtered stds of the states
             `forecast_means' -- the forecasted means of the states 
             `forecast_std' -- the forecasted stds of the states
             `smoothed_means' -- the smoothed means of the model
             `smoothed_stds' -- the smoothed stds of the model 

        Notes
        -----
        Can be used with missing (NaN) observations.
        """

        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        yy = p.DataFrame(yy)

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0 = solve_discrete_lyapunov(TT, RR.dot(QQ).dot(RR.T))

        
        res = filter_and_smooth(np.asarray(yy), TT, RR, QQ,
                                np.asarray(DD,dtype=float),
                                np.asarray(ZZ,dtype=float),
                                np.asarray(HH,dtype=float),
                                np.asarray(P0,dtype=float),t0=t0)

        (loglh, filtered_means, filtered_stds, filtered_cov,
         forecast_means, forecast_stds, forecast_cov,
         smoothed_means, smoothed_stds, smoothed_cov) = res

        results = {}
        results['log_lik'] = p.DataFrame(loglh, columns=['log_lik'],
                                         index=yy.index)

        for resname, res in [('filtered_means', filtered_means),
                             ('filtered_stds',  filtered_stds),
                             ('forecast_means', forecast_means),
                             ('forecast_stds',  forecast_stds),
                             ('smoothed_means', smoothed_means),
                             ('smoothed_stds',  smoothed_stds)]:

            resdf = p.DataFrame(res, columns=self.state_names, index=yy.index)
            results[resname] = resdf

        return results

    def pred(self, para, h=20, shocks=True, append=False, *args, **kwargs):
        """
        Draws from the predictive distribution $p(Y_{t+1:t+h}|Y_{1:T}, \theta)$.


        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        h : int, optional
            The horizon of the distribution.
        y : 2d array-like, optional
            Dataset of observables (T x nobs). The default is the observable set pass during
            class instantiation.
        append : bool, optional
            Return the draw appended to yy (default = FALSE).

        Returns
        -------
        ysim : pandas.DataFrame
            A dataframe containing the draw from the predictive distribution.

        """
        yy = kwargs.pop('y', self.yy)
        res = self.kf_everything(para, y=yy, *args, **kwargs)

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        At = res['smoothed_states'].iloc[-1].values
        ysim  = np.zeros((h, DD.size))

        index0 = res['smoothed_states'].index[-1]+1
        index = []
        for i in range(h):
            e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
            At = TT.dot(At) + shocks*RR.dot(e)
            h = np.random.multivariate_normal(np.zeros((HH.shape[0])), HH)
            At = np.asarray(At).squeeze()
            ysim[i, :] = DD.T + ZZ.dot(At) + shocks*np.atleast_2d(h)
            index.append(index0+i)

        ysim = p.DataFrame(ysim, columns=self.obs_names, index=index)

        if append:
            ysim = self.yy.append(ysim)
        return ysim

    def system_matrices(self, para, *args, **kwargs):
        """
        Returns the system matrices of the state space model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.


        Returns
        -------
        TT : np.array (ns x ns)
        RR : np.array (ns x neps)
        QQ : np.array (neps x neps)
        DD : np.array (nobs)
        ZZ : np.array (ny x ns)
        HH : np.array (ny x ny)


        Notes
        -----

        """
        TT = np.atleast_2d(self.TT(para, *args, **kwargs))
        RR = np.atleast_2d(self.RR(para, *args, **kwargs))
        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))

        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        return TT, RR, QQ, DD, ZZ, HH


    def abcd_representation(self, para, *args, **kwargs):
        """
        Returns ABCD representation of state space system.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.
        

        Returns 
        -------
        A, B, C, D


        Notes
        -----
        

        """

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if not(np.all(HH==0)): fdkjsl
        
        A = TT
        B = RR

        C = ZZ.dot(TT) #ZZ @ TT
        D = ZZ.dot(RR)

        return A, B, C, D
        
        
        
    def impulse_response(self, para, h=20, *args, **kwargs):
        """
        Computes impulse response functions of model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.

        h : int, optional
           The maximum horizon length for the impulse responses.


        Returns
        -------
        irf : pandas.Panel (nshocks x h x nvariables)


        Notes
        -----
        These are of the model state variables impulse responses to
        1 standard deviation shocks.  The method does not return IRFs in terms
        of the model observables.
        """
        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if self.shock_names==None:
            self.shock_names = ['shock_' + str(i) for i in range(QQ.shape[0])]
            self.state_names = ['state_' + str(i) for i in range(TT.shape[0])]
            self.obs_names = ['obs_' + str(i) for i in range(ZZ.shape[0])]


        neps = QQ.shape[0]

        irfs = {}
        for i in range(neps):

            At = np.zeros((TT.shape[0], h+1))
            QQz = np.zeros_like(QQ)
            QQz[i, i] = QQ[i, i]
            cQQz = np.sqrt(QQz)

            #cQQz = np.linalg.cholesky(QQz)

            At[:, 0] = (RR.dot(cQQz)[:, i]).squeeze()

            for j in range(h):
                At[:, j+1] = TT.dot(At[:, j])

            irfs[self.shock_names[i]] = p.DataFrame(At.T, columns=self.state_names)

        return p.Panel(irfs)

    def simulate(self, para, nsim=200, *args, **kwargs):
        """
        Simulates the observables of the model.

        Parameters
        ----------
        para : array-like
            An npara length vector of parameter values that defines the system matrices.

        nsim : int, optional
            The length of the simulation. The default value is 200.

        Returns
        -------
        ysim : np.array (nsim x nobs)


        Notes
        -----
        The simulation is initialized from the steady-state mean and subsequently
        a simulation of length 2*nsim is created, with the final nsim observations
        return.
        """

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)
        ysim  = np.zeros((nsim*2, DD.size))
        At = np.zeros((TT.shape[0],))

        for i in range(nsim*2):
            e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
            At = TT.dot(At) + RR.dot(e)

            h = np.random.multivariate_normal(np.zeros((HH.shape[0])), HH)
            At = np.asarray(At).squeeze()
            ysim[i, :] = DD.T + ZZ.dot(At) + np.atleast_2d(h)

        return ysim[nsim:, :]


    def forecast(self, para, h=20, shocks=True, *args, **kwargs):
        t0 = kwargs.pop('t0', self.t0)
        yy = kwargs.pop('y', self.yy)
        P0 = kwargs.pop('P0', 'unconditional')

        TT, RR, QQ, DD, ZZ, HH = self.system_matrices(para, *args, **kwargs)

        if P0=='unconditional':
            P0 = solve_discrete_lyapunov(TT, RR.dot(QQ).dot(RR.T))

        data = np.asarray(yy)
        nobs, ny = yy.shape
        ns = TT.shape[0]

        At = np.zeros(shape=(nobs, ns))

        RQR = np.dot(np.dot(RR, QQ), RR.T)

        Pt = P0

        ZZ = np.atleast_2d(ZZ)
        HH = np.atleast_2d(HH)
        loglh = np.zeros((nobs))
        AA = At[0, :]

        for i in np.arange(0, nobs):

            At[i, :] = AA


            yhat = np.dot(ZZ, AA) + DD.flatten()
            nut = data[i, :] - yhat
            nut = np.atleast_2d(nut)
            ind = (np.isnan(data[i, :])==False).nonzero()[0]
            nyact = ind.size

            if nyact > 0:
                Ft = np.dot(np.dot(ZZ[ind, :], Pt), ZZ[ind, :].T) + HH[ind, :][:, ind]
                Ft = 0.5*(Ft + Ft.T)

                dFt = np.log(np.linalg.det(Ft))

                iFtnut = sp.linalg.solve(Ft, nut[:, ind].T, sym_pos=True)

                if i+1 > t0:
                    loglh[i]= - 0.5*nyact*np.log(2*np.pi) - 0.5*dFt \
                              - 0.5*np.dot(nut[:, ind], iFtnut)

                TTPt = np.dot(TT, Pt)
                Kt = np.dot(TTPt, ZZ[ind, :].T)
            else:
                TTPt = np.dot(TT, Pt)

                Kt = np.zeros((ns, ny))
                Ft = np.eye(ny)
                iFtnut = np.eye(ny)



            AA = np.dot(TT, AA) + np.dot(Kt, iFtnut).squeeze()
            AA = np.asarray(AA).squeeze()

            Pt = np.dot(TTPt, TT.T) - np.dot(Kt, sp.linalg.solve(Ft, Kt.T, sym_pos=True)) + RQR


        if isinstance(yy,p.DataFrame):
            loglh = p.DataFrame(loglh,columns=['Log Lik.'])
            loglh.index = yy.index


        results = {}
        results['filtered_states']   = p.DataFrame(At, columns=self.state_names,index=yy.index)
        results['one_step_forecast'] = []
        results['log_lik'] = loglh

        return results



    def historical_decomposition(self, para, *args, **kwargs):
        pass


class LinearDSGEModel(StateSpaceModel):

    def __init__(self, yy, GAM0, GAM1, PSI, PPI,
                 QQ, DD, ZZ, HH, t0=0,
                 shock_names=None, state_names=None, obs_names=None,
                 prior=None):

        if len(yy.shape) < 2:
            yy = np.swapaxes(np.atleast_2d(yy), 0, 1)

        self.yy = yy

        self.GAM0 = GAM0
        self.GAM1 = GAM1
        self.PSI = PSI
        self.PPI = PPI
        self.QQ = QQ
        self.DD = DD
        self.ZZ = ZZ
        self.HH = HH

        self.t0 = t0

        self.shock_names = shock_names
        self.state_names = state_names
        self.obs_names = obs_names

        self.prior = prior

    def solve_LRE(self, para, *args, **kwargs):

        G0 = self.GAM0(para, *args, **kwargs)
        G1 = self.GAM1(para, *args, **kwargs)
        PSI = self.PSI(para, *args, **kwargs)
        PPI = self.PPI(para, *args, **kwargs)

        G0 = np.atleast_2d(G0)
        G1 = np.atleast_2d(G1)
        PSI = np.atleast_2d(PSI)
        PPI = np.atleast_2d(PPI)

        C0 = np.zeros(G0.shape[0])

        nf = PPI.shape[1]

        if nf > 0:
            TT, RR, RC = gensys(G0, G1, PSI,PPI, C0)
            RC = RC[0]*RC[1]
            #TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensysw.gensys.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)
        else:
            TT = np.linalg.inv(G0).dot(G1)
            RR = np.linalg.inv(G0).dot(PSI)
            RC = 1

        return TT, RR, RC

    def system_matrices(self, para, *args, **kwargs):

        TT, RR, RC = self.solve_LRE(para, *args, **kwargs)

        QQ = np.atleast_2d(self.QQ(para, *args, **kwargs))
        DD = np.atleast_1d(self.DD(para, *args, **kwargs))
        ZZ = np.atleast_2d(self.ZZ(para, *args, **kwargs))
        HH = np.atleast_1d(self.HH(para, *args, **kwargs))

        if RC!=1:
            TT = np.nan*TT

        return TT, RR, QQ, DD, ZZ, HH

    def log_pr(self, para, *args, **kwargs):
        try:
            return self.prior.logpdf(para)
        except:
            pass
            #raise("no prior specified")

    def log_post(self, para, *args, **kwargs):
        x = self.log_lik(para) + self.log_pr(para)
        if np.isnan(x):
            x = -1000000000.
        if x < -1000000000.:
            x = -1000000000.
        return x

