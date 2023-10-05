from csdl import Model, ImplicitOperation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np
# from csdl_om import Simulator
import csdl

from lsdo_uvlm.uvlm_system.solve_circulations.rhs_group import RHS
from lsdo_uvlm.uvlm_system.solve_circulations.utils.einsum_kij_kj_ki import EinsumKijKjKi


class SolveMatrix(Model):
    """
    Solve the AIC linear system to obtain the vortex ring circulations.
    A \gamma_b + b + M \gamma_w = 0

    A        size: (A_row, A_col)
        A_row = sum((nx[i] - 1) * (ny[i] - 1))
        A_col = sum((nx[i] - 1) * (ny[i] - 1))
    \gamma_b size: sum((nx[i] - 1) * (ny[i] - 1))
    b        size: sum((nx[i] - 1) * (ny[i] - 1))
    M        size: 
        M_row = sum((nx[i] - 1) * (ny[i] - 1))
        M_col = sum((n_wake_pts_chord - 1) * (ny[i] - 1))
    \gamma_w size: sum((n_wake_pts_chord - 1) * (ny[i] - 1))
    Parameters
    ----------
    mtx[system_size, system_size] : numpy array
        Final fully assembled AIC matrix that is used to solve for the
        circulations.
    rhs[system_size] : numpy array
        Right-hand side of the AIC linear system, constructed from the
        freestream velocities and panel normals.
    Returns
    -------
    circulations[system_size] : numpy array
        The vortex ring circulations obtained by solving the AIC linear system.
    """
    def initialize(self):
        self.parameters.declare('n_wake_pts_chord', types=int)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('bd_vortex_shapes', types=list)
        self.parameters.declare('delta_t')
        # pass

    def define(self):
        surface_names = self.parameters['surface_names']
        bd_vortex_shapes = self.parameters['bd_vortex_shapes']
        num_nodes = bd_vortex_shapes[0][0]

        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        delta_t = self.parameters['delta_t']

        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in bd_vortex_shapes
        ]

        # print('bd_coll_pts_shapes', bd_coll_pts_shapes)

        # aic_bd_proj_names = [x + '_aic_bd_proj' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((item[0],n_wake_pts_chord, item[2], item[3]))
            for item in bd_vortex_shapes
        ]

        model = Model()
        '''1. add the rhs'''
        model.add(
            RHS(
                n_wake_pts_chord=n_wake_pts_chord,
                surface_names=surface_names,
                bd_vortex_shapes=bd_vortex_shapes,
                delta_t=delta_t,
            ), 'RHS_group')
        self.add(model, 'prepossing_before_Solve')
        '''3. solve'''
        model = Model()
        M_shape_row = M_shape_col = 0

        for i in range(len(bd_coll_pts_shapes)):
            M_shape_row += (bd_coll_pts_shapes[i][1] *
                            bd_coll_pts_shapes[i][2])
            M_shape_col += ((wake_vortex_pts_shapes[i][1] ) *
                            (wake_vortex_pts_shapes[i][2] - 1))

        M_shape = (num_nodes,M_shape_row, M_shape_col)
        M = model.declare_variable('M_mat', shape=M_shape)
        # M_reshaped = model.declare_variable('M_reshaped',
        #                                     shape=(num_nodes, M_shape_row,
        #                                            M_shape_row))

        gamma_b_shape = sum((i[1] - 1) * (i[2] - 1) for i in bd_vortex_shapes)
        # this is gamma_b_shape per time step
        # print('solve_group gamma_b_shape', gamma_b_shape)

        aic_bd_proj_shape = \
            (num_nodes, ) + (gamma_b_shape, ) + (gamma_b_shape, )
        '''declare terms in the equations'''
        # MTX = model.declare_variable('MTX', shape=(aic_bd_proj_shape))
        # print('solve_group before implicit MTX shape', MTX.shape)
        # print('solve_group before implicit M_reshaped shape', M_reshaped.shape)
        aic_bd_proj_name = 'aic_bd_proj'
        aic_shape_row = aic_shape_col = 0
        for i in range(len(bd_coll_pts_shapes)):
            aic_shape_row += (bd_coll_pts_shapes[i][1] *
                              bd_coll_pts_shapes[i][2])
            aic_shape_col += ((bd_coll_pts_shapes[i][1]) *
                              (bd_coll_pts_shapes[i][2]))
        aic_bd_proj = model.declare_variable(aic_bd_proj_name,
                                            shape=(num_nodes, aic_shape_row,
                                                   aic_shape_col))
        gamma_b = model.declare_variable('gamma_b',
                                         shape=(num_nodes, gamma_b_shape))
        b = model.declare_variable('b', shape=(num_nodes, gamma_b_shape))
        '''hardcode here to test function'''
        gamma_w_shape = (num_nodes, n_wake_pts_chord,)+ (sum((i[2] - 1) for i in bd_vortex_shapes),)
        gamma_w = model.declare_variable('gamma_w',shape=gamma_w_shape)

        product = csdl.custom(aic_bd_proj,
                                gamma_b,
                                op=EinsumKijKjKi(in_name_1=aic_bd_proj_name,
                                                    in_name_2='gamma_b',
                                                    in_shape=(num_nodes, aic_shape_row, aic_shape_col),
                                                    out_name='product'))    
        
        gamma_w_reshaped = csdl.reshape(gamma_w,(num_nodes,gamma_w.shape[1]*gamma_w.shape[2]))
        model.register_output('gamma_w_reshaped',gamma_w_reshaped)
        product_w = csdl.custom(M,
                                gamma_w_reshaped,
                                op=EinsumKijKjKi(in_name_1='M_mat',
                                                    in_name_2='gamma_w_reshaped',
                                                    in_shape=M_shape,
                                                    out_name='product_w'))    


        # gamma_w_reshape = model.declare_variable(gamma_w,shape=(num_nodes,gamma_w.shape[1]*gamma_w.shape[2]))
        y = product + product_w+ b
        # y = product + csdl.einsum(M, csdl.reshape(gamma_w,(num_nodes,gamma_w.shape[1]*gamma_w.shape[2])), subscripts='kij,kj->ki')+ b
        # csdl.einsum(MTX, gamma_b, subscripts='kij,kj->ki') + b

        model.register_output('y', y)

        solve = self.create_implicit_operation(model)
        solve.declare_state('gamma_b', residual='y')
        solve.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=5,
            iprint=True,
        )
        solve.linear_solver = ScipyKrylov()

        aic_bd_proj = self.declare_variable('aic_bd_proj',
                                    shape=(num_nodes, aic_shape_row,
                                                   aic_shape_col))
        '''hardcode the normal vec and gamma_w '''
        gamma_w = self.declare_variable('gamma_w',shape=gamma_w_shape)
        # gamma_w_reshape = self.declare_variable('gamma_w_reshape',shape=(num_nodes,gamma_w.shape[1]*gamma_w.shape[2]))
        b = self.declare_variable('b', shape=(num_nodes, gamma_b_shape))
        # print('solve_group after implicit M_shape_row', M_shape_row)
        # print('solve_group after implicit MTX shape', MTX.shape)
        M = self.declare_variable('M_mat', shape=M_shape)

        gamma_b = solve(aic_bd_proj,M, gamma_w, b)
        '''hardcode'''

        # self.register_output('wing_gamma_w',gamma_w+0.)
        # model.visualize_sparsity(recursive=True)
        # self.print_var(aic_bd_proj)
        # self.print_var(M)
        # self.print_var(gamma_w)
        # self.print_var(gamma_b)
        # self.print_var(b)
        # self.print_var(y)



if __name__ == "__main__":

    sim = Simulator(
        SolveMatrix(n_wake_pts_chord=3,
                    surface_names=['wing'],
                    bd_vortex_shapes=[(2, 2, 3)]))
    sim.run()
    sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])