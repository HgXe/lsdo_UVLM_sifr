import m3l
import csdl

import numpy as np

from lsdo_uvlm.uvlm_system.wake_rollup.combine_gamma_w import CombineGammaW
from lsdo_uvlm.uvlm_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
from lsdo_uvlm.uvlm_preprocessing.adapter_comp import AdapterComp
from lsdo_uvlm.uvlm_system.solve_circulations.solve_group import SolveMatrix
from lsdo_uvlm.uvlm_system.wake_rollup.seperate_gamma_b import SeperateGammab
from lsdo_uvlm.uvlm_system.wake_rollup.compute_wake_total_vel import ComputeWakeTotalVel

class UVLMCore(m3l.ImplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='uvlm')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
    def assign_atributes(self):
        self.name = self.parameters['name']
        self.surface_names = self.parameters['surface_names']
        self.surface_shapes = self.parameters['surface_shapes']
        self.delta_t = self.parameters['delta_t']
        self.nt = self.parameters['nt']
    def evaluate(self):
        self.assign_atributes()
        self.residual_names = []
        name = self.name + '_'
        self.ode_parameters = ['u',
                               'v',
                               'w',
                            #    'p',
                            #    'q',
                            #    'r',
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

        for i in range(len(self.ode_parameters)-1):
            self.ode_parameters[i] = name + self.ode_parameters[i]
        self.inputs = {}
        self.arguments = {}
        
        residual = m3l.Variable(self.residual_names[0][1], shape=(), operation=self)
        return residual
    def compute_residual(self, num_nodes):
        model = ODESystemModel(num_nodes=num_nodes, 
                               surface_names=self.surface_names,
                               surface_shapes=self.surface_shapes,
                               delta_t=self.delta_t,
                               nt=self.nt,
                               name=self.name)
        return model


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
        self.parameters.declare('name')

    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']
        name = self.parameters['name'] + '_'

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
        u = self.declare_variable(name+'u',  shape=(n,1))
        v = self.declare_variable(name+'v',  shape=(n,1))
        w = self.declare_variable(name+'w',  shape=(n,1))
        p = self.declare_variable(name+'p',  shape=(n,1))
        q = self.declare_variable(name+'q',  shape=(n,1))
        r = self.declare_variable(name+'r',  shape=(n,1))
        theta = self.declare_variable(name+'theta',  shape=(n,1))
        psi = self.declare_variable(name+'psi',  shape=(n,1))
        x = self.declare_variable(name+'x',  shape=(n,1))
        y = self.declare_variable(name+'y',  shape=(n,1))
        z = self.declare_variable(name+'z',  shape=(n,1))
        phiw = self.declare_variable(name+'phiw',  shape=(n,1))
        gamma = self.declare_variable(name+'gamma',  shape=(n,1))
        psiw = self.declare_variable(name+'psiw',  shape=(n,1))


        #  1.2.2 from the AcStates, compute 5 preprocessing outputs
        # frame_vel, alpha, v_inf_sq, beta, rho
        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            name=name
        )
        self.add(m, name='adapter_comp')

        self.add(CombineGammaW(surface_names=surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=nt-1),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=nt-1,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=delta_t),
                    name='solve_gamma_b_group')

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
            

if __name__ == '__main__':
    import python_csdl_backend
    from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh

    ########################################
    # define mesh here
    ########################################
    nx = 29
    ny = 5 # actually 14 in the book


    chord = 1
    span = 12
    # num_nodes = 9*16
    # num_nodes = 16 *2
    num_nodes = 20
    # num_nodes = 3
    nt = num_nodes+1

    alpha = np.deg2rad(5)

    # define the direction of the flapping motion (hardcoding for now)

    # u_val = np.concatenate((np.array([0.01, 0.5,1.]),np.ones(num_nodes-3))).reshape(num_nodes,1)
    # u_val = np.ones(num_nodes).reshape(num_nodes,1)
    u_val = np.concatenate((np.array([0.001]), np.ones(num_nodes-1))).reshape(num_nodes,1)*10
    # theta_val = np.linspace(0,alpha,num=num_nodes)
    theta_val = np.ones((num_nodes, 1))*alpha

    name = 'uvlm_'
    uvlm_parameters = [(name+'u',True,u_val),
                       (name+'v',True,np.zeros((num_nodes, 1))),
                       (name+'w',True,np.ones((num_nodes, 1))),
                    #    (name+'p',True,np.zeros((num_nodes, 1))),
                    #    (name+'q',True,np.zeros((num_nodes, 1))),
                    #    (name+'r',True,np.zeros((num_nodes, 1))),
                       (name+'theta',True,theta_val),
                       (name+'psi',True,np.zeros((num_nodes, 1))),
                    #    (name+'x',True,np.zeros((num_nodes, 1))),
                    #    (name+'y',True,np.zeros((num_nodes, 1))),
                    #    (name+'z',True,np.zeros((num_nodes, 1))),
                    #    (name+'phiw',True,np.zeros((num_nodes, 1))),
                       (name+'gamma',True,np.zeros((num_nodes, 1))),
                       (name+'psiw',True,np.zeros((num_nodes, 1)))]
    
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

    # mesh_val = generate_simple_mesh(nx, ny, num_nodes)
    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1]

    uvlm_parameters.append(('wing',True,mesh_val))

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

    model = m3l.DynamicModel()
    uvlm = UVLMCore(surface_names=surface_names,surface_shapes=surface_shapes,delta_t=delta_t,nt=num_nodes+1)
    uvlm_residual = uvlm.evaluate()
    model.register_output(uvlm_residual)
    model.set_dynamic_options(initial_conditions=initial_conditions,
                              num_times=num_nodes,
                              h_stepsize=delta_t,
                              parameters=uvlm_parameters,
                              integrator='ForwardEuler')
    model_csdl = model.assemble()
    sim = python_csdl_backend.Simulator(model_csdl, analytics=True)
    sim.run()














    exit()
    def generate_simple_mesh(nx, ny, n_wake_pts_chord=None,offset=0):
        if n_wake_pts_chord == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((n_wake_pts_chord, nx, ny, 3))
            for i in range(n_wake_pts_chord):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0. + offset
        return mesh

    nx=3
    ny=4
    surface_names=['wing','wing_1']

    surface_shapes = [(nx,ny,3),(nx,ny,3)]

    wing_val = generate_simple_mesh(nx, ny, 1)
    wing_val_1 = generate_simple_mesh(nx,ny, 1, 1)
    
    num_nodes = 10
    delta_t = 1/16
    model = m3l.DynamicModel()
    uvlm = UVLMCore(surface_names=surface_names,surface_shapes=surface_shapes,delta_t=delta_t,nt=num_nodes+1)
    uvlm_residual = uvlm.evaluate()
    model.register_output(uvlm_residual)

    
    alpha_deg = 10
    alpha = alpha_deg / 180 * np.pi
    name = 'uvlm_'
    uvlm_parameters = [(name+'u',True,np.ones((num_nodes, 1))* np.cos(alpha)),
                       (name+'v',True,np.zeros((num_nodes, 1))),
                       (name+'w',True,np.ones((num_nodes, 1))* np.sin(alpha)),
                       (name+'p',True,np.zeros((num_nodes, 1))),
                       (name+'q',True,np.zeros((num_nodes, 1))),
                       (name+'r',True,np.zeros((num_nodes, 1))),
                       (name+'theta',True,np.ones((num_nodes, 1))*alpha),
                       (name+'psi',True,np.zeros((num_nodes, 1))),
                       (name+'x',True,np.zeros((num_nodes, 1))),
                       (name+'y',True,np.zeros((num_nodes, 1))),
                       (name+'z',True,np.zeros((num_nodes, 1))),
                       (name+'phiw',True,np.zeros((num_nodes, 1))),
                       (name+'gamma',True,np.zeros((num_nodes, 1))),
                       (name+'psiw',True,np.zeros((num_nodes, 1))),
                       ('wing',False,wing_val),
                       ('wing_1',False,wing_val_1)]
    
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
    
    model.set_dynamic_options(initial_conditions=initial_conditions,
                              num_times=num_nodes,
                              h_stepsize=delta_t,
                              parameters=uvlm_parameters)
    model_csdl = model.assemble()
    sim = python_csdl_backend.Simulator(model_csdl, analytics=True)
    sim.run()
    sim.check_partials()

