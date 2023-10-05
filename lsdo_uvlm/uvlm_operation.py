import m3l
import csdl

import numpy as np

from lsdo_uvlm.uvlm_system.wake_rollup.combine_gamma_w import CombineGammaW
from lsdo_uvlm.uvlm_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
from lsdo_uvlm.uvlm_preprocessing.adapter_comp import AdapterComp
from lsdo_uvlm.uvlm_system.solve_circulations.solve_group import SolveMatrix
from lsdo_uvlm.uvlm_system.wake_rollup.seperate_gamma_b import SeperateGammab
from lsdo_uvlm.uvlm_system.wake_rollup.compute_wake_total_vel import ComputeWakeTotalVel
from lsdo_uvlm.uvlm_outputs.compute_force.compute_lift_drag import LiftDrag
from lsdo_uvlm.uvlm_outputs.compute_force.compute_net_thrust import ThrustDrag
from lsdo_uvlm.examples.profile_outputs.profile_op_model import ProfileOpModel


class UVLMCore(m3l.ImplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('name', default='uvlm')
    def assign_atributes(self):
        self.surface_names = self.parameters['surface_names']
        self.surface_shapes = self.parameters['surface_shapes']
        self.delta_t = self.parameters['delta_t']
        self.nt = self.parameters['nt']
        self.name = self.parameters['name']
    def evaluate(self):
        self.assign_atributes()
        num_nodes = self.nt - 1
        self.residual_names = []
        self.ode_parameters = ['u',
                               'v',
                               'w',
                               'p',   # these are declared but not used
                               'q',
                               'r',
                               'theta',
                               'psi',
                            #    'x',
                            #    'y',
                            #    'z',
                            #    'phiw',
                               'gamma',
                               'psiw']     # TODO: add rho - need to modify adapter_comp to not be hard coded
        surface_names = self.surface_names
        surface_shapes = self.surface_shapes
        for i in range(len(surface_names)):
            # print('surface_names in run.py',surface_names)
            # print('surface_shapes in run.py',surface_shapes)
            ####################################
            # ode parameter names
            ####################################
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            self.ode_parameters.append(surface_name)
            surface_name = surface_name
            ####################################
            # ode states names
            ####################################
            gamma_w_name = surface_name + '_gamma_w'
            wing_wake_coords_name = surface_name + '_wake_coords'
            # gamma_w_name_list.append(gamma_w_name)
            # wing_wake_coords_name_list.append(wing_wake_coords_name)
            # Inputs names correspond to respective upstream CSDL variables
            ####################################
            # ode outputs names
            ####################################
            dgammaw_dt_name = surface_name + '_dgammaw_dt'
            dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            ####################################
            # IC names
            ####################################
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            ####################################
            # states and outputs names
            ####################################
            gamma_w_int_name = surface_name + '_gamma_w_int'
            wake_coords_int_name = surface_name + '_wake_coords_int'
            self.residual_names.append((gamma_w_name,dgammaw_dt_name,(num_nodes, ny-1)))
            self.residual_names.append((wing_wake_coords_name,dwake_coords_dt_name,(num_nodes,ny,3)))

        self.inputs = {}
        self.arguments = {}
        
        residual = m3l.Variable(self.residual_names[0][1], shape=(), operation=self)
        return residual #, frame_vel, bd_vec, horseshoe_circulation
    def compute_residual(self, num_nodes):
        model = ODESystemModel(num_nodes=num_nodes, 
                               surface_names=self.surface_names,
                               surface_shapes=self.surface_shapes,
                               delta_t=self.delta_t,
                               nt=self.nt)
        return model


# class UVLMForces(m3l.ExplicitOperation):
#     def evaluate(self, frame_vel, alpha, beta, bd_vec, horseshoe_circulation, ):

#         return




class ODESystemModel(csdl.Model):
    '''
    contains
    1. MeshPreprocessing_comp
    2. SolveMatrix
    3. solve_gamma_b_group
    3. seperate_gamma_b_comp
    4. extract_gamma_w_comp
    '''
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')

    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']

        # set conventional names
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        # set shapes
        bd_vortex_shapes = surface_shapes
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)
        ode_surface_shapes = [(n, ) + item for item in surface_shapes]
        # wake_vortex_pts_shapes = [tuple((item[0],nt, item[2], 3)) for item in ode_surface_shapes]
        # wake_vel_shapes = [(n,x[1] * x[2], 3) for x in wake_vortex_pts_shapes]
        ode_bd_vortex_shapes = ode_surface_shapes
        gamma_w_shapes = [tuple((n,nt-1, item[2]-1)) for item in ode_surface_shapes]

        '''1. add a module here to compute surface_gamma_b, given mesh and ACstates'''
        # 1.1.1 declare the ode parameter surface for the current time step
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            

            surface = self.declare_variable(surface_name, shape=(n, nx, ny, 3))
        # 1.1.2 from the declared surface mesh, compute 6 preprocessing outputs
        # surface_bd_vtx_coords,coll_pts,l_span,l_chord,s_panel,bd_vec_all
        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes),
                 name='MeshPreprocessing_comp')
        # 1.2.1 declare the ode parameter AcStates for the current time step
        u = self.declare_variable('u',  shape=(n,1))
        v = self.declare_variable('v',  shape=(n,1))
        w = self.declare_variable('w',  shape=(n,1))
        p = self.declare_variable('p',  shape=(n,1))
        q = self.declare_variable('q',  shape=(n,1))
        r = self.declare_variable('r',  shape=(n,1))
        theta = self.declare_variable('theta',  shape=(n,1))
        psi = self.declare_variable('psi',  shape=(n,1))
        x = self.declare_variable('x',  shape=(n,1))
        y = self.declare_variable('y',  shape=(n,1))
        z = self.declare_variable('z',  shape=(n,1))
        phiw = self.declare_variable('phiw',  shape=(n,1))
        gamma = self.declare_variable('gamma',  shape=(n,1))
        psiw = self.declare_variable('psiw',  shape=(n,1))


        #  1.2.2 from the AcStates, compute 5 preprocessing outputs
        # frame_vel, alpha, v_inf_sq, beta, rho
        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes
        )
        self.add(m, name='adapter_comp')

        self.add(CombineGammaW(surface_names=surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=nt-1),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=nt-1,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=delta_t),
                    name='solve_gamma_b_group') # TODO: check this for potential speedup

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes),
                 name='seperate_gamma_b')

        # ODE system with surface gamma's
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            val = np.zeros((n, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_w_name = surface_name + '_gamma_w'
            surface_dgammaw_dt_name = surface_name + '_dgammaw_dt'
            surface_gamma_b_name = surface_name +'_gamma_b'
            #######################################
            #states 1
            #######################################

            surface_gamma_w = self.declare_variable(surface_gamma_w_name,
                                                    shape=val.shape)
            #para for state 1

            surface_gamma_b = self.declare_variable(surface_gamma_b_name,
                                                    shape=(n, (nx - 1) *
                                                           (ny - 1), ))
            #outputs for state 1
            surface_dgammaw_dt = self.create_output(surface_dgammaw_dt_name,
                                                    shape=(n, nt - 1, ny - 1))

            gamma_b_last = csdl.reshape(surface_gamma_b[:,(nx - 2) * (ny - 1):],
                                        new_shape=(n, 1, ny - 1))

            surface_dgammaw_dt[:, 0, :] = (gamma_b_last -
                                           surface_gamma_w[:, 0, :]) / delta_t
            surface_dgammaw_dt[:, 1:, :] = (
                surface_gamma_w[:, :(surface_gamma_w.shape[1] - 1), :] -
                surface_gamma_w[:, 1:, :]) / delta_t

            # self.print_var(surface_gamma_w)
            # self.print_var(surface_gamma_b)


        #######################################
        #states 2
        #######################################
        # TODO: fix this comments to eliminate first row
        # t=0       [TE,              TE,                 TE,                TE]
        # t = 1,    [TE,              TE+v_ind(TE,w+bd),  TE,                TE] -> bracket 0-1
        # c11 = TE+v_ind(TE,w+bd)

        # t = 2,    [TE,              TE+v_ind(t=1, bracket 0),  c11+v_ind(t=1, bracket 1),   TE] ->  bracket 0-1-2
        # c21 =  TE+v_ind(t=1, bracket 0)
        # c22 =  c11+v_ind(t=1, bracket 1)

        # t = 3,    [TE,              TE+v_ind(t=2, bracket 0),  c21+vind(t=2, bracket 1), c22+vind(t=2, bracket 2)] -> bracket 0-1-2-3
        # Then, the shedding is
        '''2. add a module here to compute wake_total_vel, given mesh and ACstates'''
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface_wake_coords_name = surface_name + '_wake_coords'
            surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            #states 2
            surface_wake_coords = self.declare_variable(surface_wake_coords_name,
                                                    shape=(n, nt - 1, ny, 3))
            '''2. add a module here to compute wake rollup'''

        self.add(ComputeWakeTotalVel(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes,
                                n_wake_pts_chord=nt-1),
                 name='ComputeWakeTotalVel')            
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface_wake_coords_name = surface_name + '_wake_coords'
            surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            surface_bd_vtx = self.declare_variable(surface_names[i]+'_bd_vtx_coords', shape=(n, nx, ny, 3))
            wake_total_vel = self.declare_variable(v_total_wake_names[i],val=np.zeros((n, nt - 1, ny, 3)))
            surface_wake_coords = self.declare_variable(surface_wake_coords_name, shape=(n, nt - 1, ny, 3))


            surface_dwake_coords_dt = self.create_output(
                surface_dwake_coords_dt_name, shape=((n, nt - 1, ny, 3)))
            # print(surface_dwake_coords_dt.name,surface_dwake_coords_dt.shape)

            TE = surface_bd_vtx[:, nx - 1, :, :]

            surface_dwake_coords_dt[:, 0, :, :] = (
                TE  + wake_total_vel[:, 0, :, :]*delta_t - surface_wake_coords[:, 0, :, :]) / delta_t


            surface_dwake_coords_dt[:, 1:, :, :] = (
                surface_wake_coords[:, :
                                    (surface_wake_coords.shape[1] - 1), :, :] -
                surface_wake_coords[:, 1:, :, :] +
                wake_total_vel[:, 1:, :, :] * delta_t) / delta_t


class LiftDragM3L(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_shapes')
        self.parameters.declare('sprs')

        # self.parameters.declare('rho', default=0.9652)
        self.parameters.declare('eval_pts_names', types=None)

        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']

        # fix surface names
        for i in range(len(surface_names)):
            surface_names[i] = surface_names[i]

        # set conventional names
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        # set shapes
        bd_vortex_shapes = surface_shapes
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)
        ode_surface_shapes = [(n, ) + item for item in surface_shapes]
        # wake_vortex_pts_shapes = [tuple((item[0],nt, item[2], 3)) for item in ode_surface_shapes]
        # wake_vel_shapes = [(n,x[1] * x[2], 3) for x in wake_vortex_pts_shapes]
        ode_bd_vortex_shapes = ode_surface_shapes
        gamma_w_shapes = [tuple((n,nt-1, item[2]-1)) for item in ode_surface_shapes]

        '''1. add a module here to compute surface_gamma_b, given mesh and ACstates'''
        # 1.1.1 declare the ode parameter surface for the current time step
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            

            surface = self.declare_variable(surface_name, shape=(n, nx, ny, 3))
        # 1.1.2 from the declared surface mesh, compute 6 preprocessing outputs
        # surface_bd_vtx_coords,coll_pts,l_span,l_chord,s_panel,bd_vec_all
        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes),
                 name='MeshPreprocessing_comp')
        # 1.2.1 declare the ode parameter AcStates for the current time step
        u = self.declare_variable('u',  shape=(n,1))
        v = self.declare_variable('v',  shape=(n,1))
        w = self.declare_variable('w',  shape=(n,1))
        p = self.declare_variable('p',  shape=(n,1))
        q = self.declare_variable('q',  shape=(n,1))
        r = self.declare_variable('r',  shape=(n,1))
        theta = self.declare_variable('theta',  shape=(n,1))
        psi = self.declare_variable('psi',  shape=(n,1))
        x = self.declare_variable('x',  shape=(n,1))
        y = self.declare_variable('y',  shape=(n,1))
        z = self.declare_variable('z',  shape=(n,1))
        phiw = self.declare_variable('phiw',  shape=(n,1))
        gamma = self.declare_variable('gamma',  shape=(n,1))
        psiw = self.declare_variable('psiw',  shape=(n,1))


        #  1.2.2 from the AcStates, compute 5 preprocessing outputs
        # frame_vel, alpha, v_inf_sq, beta, rho
        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes
        )
        self.add(m, name='adapter_comp')
        
        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        ode_surface_shapes = [(n, ) + item for item in surface_shapes]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]

        ld_model = LiftDrag(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_option='auto',
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
        )
        self.add(ld_model, name='LiftDrag')




if __name__ == '__main__':
    pass
    import python_csdl_backend
    from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
    from lsdo_uvlm.uvlm_outputs.compute_force.compute_lift_drag import LiftDrag

    ########################################
    # define mesh here
    ########################################
    nx = 19
    ny = 5 # actually 14 in the book


    chord = 1
    span = 12
    # num_nodes = 9*16
    # num_nodes = 16 *2
    num_nodes = 30
    # num_nodes = 3
    nt = num_nodes+1

    alpha = np.deg2rad(5)

    # define the direction of the flapping motion (hardcoding for now)

    # u_val = np.concatenate((np.array([0.01, 0.5,1.]),np.ones(num_nodes-3))).reshape(num_nodes,1)
    # u_val = np.ones(num_nodes).reshape(num_nodes,1)
    u_val = np.ones(num_nodes).reshape(num_nodes,1)*10
    # theta_val = np.linspace(0,alpha,num=num_nodes)
    theta_val = np.ones((num_nodes, 1))*alpha

    # name0 = 'wang'
    # name = name0 + '_'
    # name0 = ''
    # name = name0
    uvlm_parameters = [('u',True,u_val),
                       ('v',True,np.zeros((num_nodes, 1))),
                       ('w',True,np.ones((num_nodes, 1))),
                       ('p',True,np.zeros((num_nodes, 1))),
                       ('q',True,np.zeros((num_nodes, 1))),
                       ('r',True,np.zeros((num_nodes, 1))),
                       ('theta',True,theta_val),
                       ('psi',True,np.zeros((num_nodes, 1))),
                    #    ('x',True,np.zeros((num_nodes, 1))),
                    #    ('y',True,np.zeros((num_nodes, 1))),
                    #    ('z',True,np.zeros((num_nodes, 1))),
                    #    ('phiw',True,np.zeros((num_nodes, 1))),
                       ('gamma',True,np.zeros((num_nodes, 1))),
                       ('psiw',True,np.zeros((num_nodes, 1)))]
    
    mesh_dict = {
        "num_y": ny,
        "num_x": nx,
        "wing_type": "rect",
        "symmetry": False,
        "span": span,
        "root_chord": chord,
        "span_cos_spacing": False,
        "chord_cos_spacing": False,
    }

    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict)
    mesh2 = generate_mesh(mesh_dict)

    # mesh_val = generate_simple_mesh(nx, ny, num_nodes)
    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1]

    mesh_val_2 = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val_2[i, :, :, 2] = mesh2.copy()[:, :, 2] + .25
        mesh_val_2[i, :, :, 0] = mesh2.copy()[:, :, 0] + 5
        mesh_val_2[i, :, :, 1] = mesh2.copy()[:, :, 1]

    uvlm_parameters.append(('wing', True, mesh_val))
    # uvlm_parameters.append(('uvlm_wing2',True,mesh_val_2))


    # surface_names=['wing','wing2']
    # surface_shapes=[(nx, ny, 3),(nx, ny, 3)]
    surface_names=['wing']
    surface_shapes=[(nx, ny, 3)]
    h_stepsize = delta_t = 1/16


    initial_conditions = []
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        gamma_w_0_name = surface_name + '_gamma_w_0'
        wake_coords_0_name = surface_name + '_wake_coords_0'
        surface_shape = surface_shapes[i]
        nx = surface_shape[0]
        ny = surface_shape[1]
        initial_conditions.append((gamma_w_0_name, np.zeros((num_nodes, ny - 1))))

        initial_conditions.append((wake_coords_0_name, np.zeros((num_nodes, ny, 3))))

    profile_outputs = []
    # profile outputs are outputs from the ode integrator that are not states. 
    # instead they are outputs of a function of the solved states and parameters
    profile_outputs.append(('wing_gamma_b', ((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),)))
    profile_outputs.append(('wing_gamma_w', (num_nodes,4)))
    profile_outputs.append(('wing_eval_pts_coords', ((surface_shapes[0][0]-1),(surface_shapes[0][1]-1),3)))
    profile_outputs.append(('wing_s_panel', ((surface_shapes[0][0]-1),(surface_shapes[0][1]-1))))
    profile_outputs.append(('wing_eval_total_vel', ((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),3)))
    profile_outputs.append(('rho',(1,)))
    profile_outputs.append(('alpha',(1,)))
    profile_outputs.append(('beta',(1,)))
    profile_outputs.append(('frame_vel',(3,)))
    # profile_outputs.append(('evaluation_pt'))
    profile_outputs.append(('bd_vec', ((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),3)))

    profile_outputs.append(('horseshoe_circulation', ((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),)))
    # profile_outputs.append(('F', (num_nodes, 3)) )
    # profile_outputs.append(('uvlm_wing_L', (num_nodes,1)))
    # # profile_outputs.append(('uvlm_wing2_C_L', (num_nodes,1)))
    # profile_outputs.append(('uvlm_wing_D', (num_nodes,1)))
    # profile_outputs.append(('uvlm_wing2_C_D_i', (num_nodes,1)))

    profile_system = ProfileOpModel
    profile_params_dict = {
            # 'num_nodes': nt-1,
            'surface_names': ['wing'],
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'nt': nt
        }

    model = m3l.DynamicModel()
    uvlm = UVLMCore(surface_names=surface_names, surface_shapes=surface_shapes, delta_t=delta_t, nt=num_nodes+1)
    uvlm_residual = uvlm.evaluate()
    model.register_output(uvlm_residual)
    model.set_dynamic_options(initial_conditions=initial_conditions,
                              num_times=num_nodes,
                              h_stepsize=delta_t,
                              parameters=uvlm_parameters,
                              integrator='ForwardEuler',
                              profile_outputs=profile_outputs,
                              profile_system=profile_system,
                              profile_parameters=profile_params_dict)
    model_csdl = model.assemble()

    surface_names = ['wing']
    eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
    ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]
    eval_pts_shapes =        [
        tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
        for item in ode_surface_shapes
    ]
    
    # submodel = LiftDrag(
    #         surface_names=surface_names,
    #         surface_shapes=ode_surface_shapes,
    #         eval_pts_option='auto',
    #         eval_pts_shapes=eval_pts_shapes,
    #         eval_pts_names=eval_pts_names,
    #         sprs=None,
    #         coeffs_aoa=None,
    #         coeffs_cd=None
    #     )
    submodel = ThrustDrag(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_option='auto',
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
        )
    
    model_csdl.add(submodel, name='LiftDrag')



    sim = python_csdl_backend.Simulator(model_csdl, analytics=True)
    # Before code
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    sim.run()
    # After code
    profiler.disable()
    profiler.dump_stats('output')

    print(sim['LiftDrag.wing_D'])
    print(sim['LiftDrag.wing_L'])


    # print(sim['prob.' + name + 'horseshoe_circulation'])
    exit()

    # print('Lift 1: ' + str(sim['LiftDrag.' + name + 'wing_L']))
    # print('Drag 1: ' + str(sim['LiftDrag.' + name + 'wing_D']))


    if True:
        from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
        axs = Axes(
            xrange=(0, 35),
            yrange=(-10, 10),
            zrange=(-3, 4),
        )
        video = Video("uvlm_m3l_test.gif", duration=10, backend='ffmpeg')
        for i in range(nt - 1):
            vp = Plotter(
                bg='beige',
                bg2='lb',
                # axes=0,
                #  pos=(0, 0),
                offscreen=False,
                interactive=1)
            # Any rendering loop goes here, e.g.:
            for surface_name in surface_names:
                surface_name = 'prob.' + surface_name
                vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                            r=8,
                            c='red')
                vp += vps
                vp += __doc__
                vps = Points(np.reshape(sim[surface_name+'_wake_coords_integrated'][i, 0:i, :, :],
                                        (-1, 3)),
                            r=8,
                            c='blue')
                vp += vps
                vp += __doc__
            # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
            # video.action(cameras=[cam1, cam1])
            vp.show(axs, elevation=-60, azimuth=-0,
                    axes=False, interactive=False)  # render the scene
            video.add_frame()  # add individual frame
            # time.sleep(0.1)
            # vp.interactive().close()
            vp.close_window()
        vp.close_window()
        video.close()  # merge all the recorded frames