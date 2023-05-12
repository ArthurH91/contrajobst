import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi


# ### CASADI+IPOPT SOLVER
class CasadiSolver:
    def __init__(self, QP):
        """
        Trivial initialization from a QP object as defined by Arthur.
        """
        self.QP = QP

    def solve(self, Q0=None):
        """
        Solve the OCP problem defined in QP using Casadi (for derivatives)
        and IpOpt (for NLP algorithm).
        All hyperparms are taken from QP. The functions are reimplemented,
        so beware of possible differences (no automatic enforcement).
        Returns the optimal variable and the residuals at optimum.
        """

        QP = self.QP

        # HYPERPARAMS
        T = QP._T
        w_q0 = QP._weight_q0
        w_dq = QP._weight_dq
        w_term = QP._weight_term_pos
        q0 = QP._q0
        target = QP._target

        ### CASADI HELPERS
        cmodel = cpin.Model(QP._rmodel)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", QP._rmodel.nq)

        cpin.framesForwardKinematics(cmodel, cdata, cq)
        endeff = casadi.Function("p", [cq], [cdata.oMf[QP._EndeffID].translation])

        ### CASADI PROBLEM
        self.opti = opti = casadi.Opti()
        # Decision variables
        self.var_qs = qs = [opti.variable(QP._rmodel.nq) for model in range(T + 1)]

        residuals = (
            [w_q0 * (qs[0] - q0)]
            + [w_dq * (qa - qb) for (qa, qb) in zip(qs[1:], qs[:-1])]
            + [w_term * (endeff(qs[-1]) - target)]
        )
        self.residuals = residuals = casadi.vertcat(*residuals)

        ### Optim
        opti.minimize(casadi.sumsqr(residuals) / 2)
        if Q0 is not None:
            print(" With specific warm start")
            for qws, vq in zip(np.split(Q0, T + 1), qs):
                opti.set_initial(vq, qws)
        else:
            print(" With default (q0) warm start")
            for vq in qs:
                opti.set_initial(vq, q0)

        opti.solver("ipopt")  # set numerical backend
        # Caution: in case the solver does not converge, we are picking the candidate values
        # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
        try:
            sol = opti.solve_limited()
            qs_sol = np.concatenate([opti.value(q) for q in qs])
            residuals_sol = opti.value(residuals)
        except:
            print("ERROR in convergence, plotting debug info.")
            qs_sol = np.concatenate([opti.debug.value(q) for q in qs])
            residuals_sol = opti.debug.value(residuals)

        return qs_sol, residuals_sol

    def evalResiduals(self, Q):
        """
        Evaluate (numerical value) the residual of the problem for a numerical
        trajectory Q = np.array((T+1)*NQ).
        Returns the np.array((T+1)*NQ+3)
        """
        Q = np.split(Q, self.QP._T + 1)
        r = self.residuals
        for var, val in zip(self.var_qs, Q):
            r = casadi.substitute(r, var, val)
        return np.array(casadi.evalf(r)).squeeze()

    def evalJacobian(self, Q):
        Q = np.split(Q, self.QP._T + 1)
        r = self.residuals
        J = []
        for varq, valq in zip(self.var_qs, Q):
            Jk = casadi.jacobian(r, varq)
            # In general, the Jacobian would need a substituttion with respect
            # to all variables. For this particular problem, the substitution
            # below is sufficient, but that might not work if you modify the
            # problem (for example, if some cost q_0*q_1 is added)
            Jk = np.array(casadi.evalf(casadi.substitute(Jk, varq, valq)))
            J.append(Jk)
        return np.hstack(J)
