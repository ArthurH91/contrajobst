# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

from solver_base import SolverBase


class SolverNewtonMt(SolverBase):
    def __init__(
        self,
        f,
        grad,
        hess,
        max_iter=1e3,
        callback=None,
        alpha_increase=1.2,
        alpha_decrease=0.5,
        regularization_increase=10,
        regularization_decrease=0.5,
        armijo_const=1e-2,
        beta=1e-2,
        init_regu=1e-6,
        min_regu=1e-9,
        alpha=1.0,
        verbose=True,
        lin_solver=np.linalg.solve,
        eps=1e-8,
        bool_plot_results=False,
    ):
        """Initialize solver object with the cost function and its gradient, along with numerical and categorical parameters.


        Parameters
        ----------
        f : function handle
            cost function
        grad : function handle
            gradient function of the cost function
        hess : function handle
            hessian function of the cost function
        max_iter : float, optional
            number max of iterations, by default 1e3
        callback : function handle, optional
            callback at each iteration, can be a display of meshcat for instance, by default None
        alpha_increase : float, optional
            increase of the alpha, by default 1.2
        alpha_decrease : float, optional
            decrease of the alpha, by default 0.5
        regularization_increase : float, optional
            increase of the damping, by default 10
        regularization_decrease : float, optional
            decrease of the damping, by default 0.5
        armijo_const : _type_, optional
            _description_, by default 1e-2
        beta : float, optional
            constant c in the backtracking linesearch in Nocedal, by default 1e-2
        init_regu : float, optional
            intial damping, by default 1e-6
        min_regu : float, optional
            minimal damping, by default 1e-9
        alpha : float, optional
            initial alpha, by default 1.0
        verbose : bool, optional
            boolean describing whether the user wants the verbose mode, by default True
        lin_solver : function, optional
            solver of the equation ax = b, by default np.linalg.solve
        eps : float, optional
            esperance used in the stopping criteria, by default 1e-8
        bool_plot_results : bool, optional
            _description_, by default False
        """

        self._f = f
        self._grad = grad
        self._hess = hess
        self._max_iter = max_iter
        self._callback = callback
        self._alpha_increase = alpha_increase
        self._alpha_decrease = alpha_decrease
        self._regularization_decrease = regularization_decrease
        self._regularization_increase = regularization_increase
        self._armijo_const = armijo_const
        self._beta = beta
        self._init_regu = init_regu
        self._min_regu = min_regu
        self._alpha = alpha
        self._verbose = verbose
        self._lin_solver = lin_solver
        self._eps = eps
        self._bool_plot_results = bool_plot_results

    def __call__(self, x0: np.ndarray):
        """Performs a trust-region optimization algorithm on the function f.


        Parameters
        ----------
        x0 : np.ndarray
            initial guess

        Returns
        -------
        xval_k : np.ndarray
            final configuration
        fval_k : float
            final cost value
        gradfval_k : float
            final gradient value
        """

        # Initialization of the step size
        self._alpha_k = self._alpha

        # Initialization of the damping
        self._regu_k = self._init_regu

        # Initial guess
        self._xval_k = x0

        # Initialize iteration counter
        self._iter_cnter = 0

        # Create a list for the values of cost function
        self._fval_history = []

        # Create a list for the values of the gradient function
        self._gradfval_history = []

        # Create a list for the values of step size
        self._alphak_history = []

        # Create a list for the values of the regularization
        self._reguk_history = []

        # Printing a small explanation of the algorithm
        self._print_start()

        # Printing the header if the user wants a verbose solver
        if self._verbose:
            self._print_header()

        # Start
        while True:
            # Cost of the step
            self._fval_k = self._f(self._xval_k)
            # Gradient of the cost function
            self._gradfval_k = self._grad(self._xval_k)
            # Norm of the gradient function
            self._norm_gradfval_k = np.linalg.norm(self._gradfval_k)
            # Hessian of the cost function
            self._hessval_k = self._hess(self._xval_k)

            if self._verbose:
                # Print current iterate
                self._print_iteration()
                # Every 30 iterations print header
                if self._iter_cnter % 30 == 29:
                    self._print_header()

            # Check stopping conditions
            if self._convergence_condition() or self._exceeded_maximum_iterations():
                break

            # Linesearch
            self._Ls_bool = False
            while not self._Ls_bool:
                # Computing search direction
                self._search_dir_k = self._compute_search_direction()

                # Computing directionnal derivative
                self._dir_deriv_k = self._compute_current_directional_derivative()

                # Linesearch, if the step is accepted Ls_bool = True and the alpha_k is kept. If it's not, the search direction is
                # computed once again with an increase in the damping.
                self._alpha_k = self._backtracking()

                # If the step is not accepted, increase the damping
                if not self._Ls_bool:
                    self._regu_k *= self._regularization_increase

            # Computing next step
            self._xval_k = self._compute_next_step()

            # Increasing step size (according to Marc Toussaint)
            self._alpha_k = min(self._alpha_increase * self._alpha_k, 1)

            # Updating the trust-region
            if self._alpha_k == 1 and self._regu_k > self._min_regu:
                self._regu_k /= 10

            # Iterate the loop
            self._iter_cnter += 1

            # Adding the cost function value to the list
            self._fval_history.append(self._fval_k)

            # Adding the step size to the list
            self._alphak_history.append(self._alpha_k)

            # Adding the value of the norm of the gradient to the list
            self._gradfval_history.append(self._norm_gradfval_k)

            # Adding the value of the regularization to the list
            self._reguk_history.append(self._regu_k)

        # Printing outputs
        self._print_output()

        # Plotting outputs
        if self._bool_plot_results:
            self._plot_variables()

        return self._xval_k, self._fval_k, self._gradfval_k

    def _backtracking(self):
        """Calculates a step using backtracking.

        Returns:
            float: Step value computed by the backtracking.
        """
        # Initialize the step iterate
        alpha = self._alpha
        # Repeat
        while True:
            # Caclulate current function value
            fval_curr = self._f(self._xval_k + alpha * self._search_dir_k)
            # Check stopping conditions
            if self._armijo_condition_is_true(alpha=alpha, fval_alpha=fval_curr):
                break
            # Otherwise diminish alpha
            alpha = self._beta * alpha

            # Trust region
            if alpha <= 2e-10:
                self._Ls_bool = False
                return alpha
        # Return
        self._Ls_bool = True
        return alpha

    def _compute_search_direction(self):
        """Computes the search direction for trust-region.

        Returns
        -------
        search_dir_k : np.ndarray
            search direction for trust region
        """
        return self._lin_solver(
            self._hessval_k + self._regu_k * np.eye(len(self._gradfval_k)),
            -self._gradfval_k,
        )

    def _print_start(self):
        print("Start of the Newton method of Marc Toussaint")

    def _plot_variables(self):
        """Plotting the outputs, which are the values of the cost function, the gradient, the alphas and the regularization term through the iterations"""
        try:
            import matplotlib.pyplot as plt

            plt.subplot(411)
            plt.plot(self._fval_history, "-ob", label="Marc Toussaint's method")
            plt.yscale("log")
            plt.ylabel("Cost")
            plt.legend()

            plt.subplot(412)
            plt.plot(self._gradfval_history, "-ob", label="Marc Toussaint's method")
            plt.yscale("log")
            plt.ylabel("Gradient")
            plt.legend()

            plt.subplot(413)
            plt.plot(self._alphak_history, "-ob", label="Marc Toussaint's method")
            plt.yscale("log")
            plt.ylabel("Alpha")
            plt.legend()

            plt.subplot(414)
            plt.plot(self._reguk_history, "-ob", label="Marc Toussaint's method")
            plt.yscale("log")
            plt.ylabel("Regularization")
            plt.xlabel("Iterations")
            plt.legend()

            plt.suptitle("Trust-region method from Marc Toussaint ")
            plt.show()
        except:
            print("No module named matplotlib.pyplot")

    def _print_output(self):
        """Prints the final message of the search.

        Properties used:
            self._iter_cnter (int): Iteration counter.
            self._fval_k (float): Function value at current iterate.
            self._norm_gradfval_k (float): Gradient norm value at current iterate.
        """
        # If the algorithm converged
        if self._convergence_condition():
            print()
            print(
                "Marc Toussaint's descent successfully converged in %d iterations."
                % self._iter_cnter
            )
            print("Optimal function value: %.4e." % self._fval_k)
            print("Optimality conditions : %.4e." % self._norm_gradfval_k)

        # If the algorithm exceeded the iterations
        if self._exceeded_maximum_iterations():
            print()
            print(
                "Marc Toussaint's descent exceeded the maximum number (%d) of iterations."
                % self._max_iter
            )
            print("Current function value: %.4e." % self._fval_k)
            print("Current optimality conditions : %.4e." % self._norm_gradfval_k)


if __name__ == "__main__":
    pass
