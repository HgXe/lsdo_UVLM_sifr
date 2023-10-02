import time
from turtle import shape

import matplotlib.pyplot as plt
import openmdao.api as om
from lsdo_uvlm.examples.profile_outputs.plunging_system_free import ODESystemModel
from lsdo_uvlm.examples.profile_outputs.plunging_profile_outputs import ProfileSystemModel
from lsdo_uvlm.examples.profile_outputs.profile_op_model import ProfileOpModel
from lsdo_uvlm.uvlm_outputs.compute_force.compute_net_thrust import ThrustDrag

from ozone.api import ODEProblem
import csdl

import numpy as np
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show


from lsdo_uvlm.uvlm_preprocessing.actuation_model_temp import ActuationModel
from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh


class ODEProblemTest(ODEProblem):
    def setup(self):

        # profile outputs are outputs from the ode integrator that are not states. 
        # instead they are outputs of a function of the solved states and parameters
        self.add_profile_output('wing_gamma_b', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),))
        self.add_profile_output('wing_eval_pts_coords', shape=((surface_shapes[0][0]-1),(surface_shapes[0][1]-1),3))
        self.add_profile_output('wing_s_panel', shape=((surface_shapes[0][0]-1),(surface_shapes[0][1]-1)))
        self.add_profile_output('wing_eval_total_vel', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),3))
        self.add_profile_output('rho')
        self.add_profile_output('alpha')
        self.add_profile_output('beta')
        self.add_profile_output('frame_vel',shape=(3,))
        # self.add_profile_output('evaluation_pt')
        self.add_profile_output('bd_vec', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),3))

        self.add_profile_output('horseshoe_circulation', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),))


        ####################################
        # ode parameter names
        ####################################        
        self.add_parameter('u', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('v', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('w', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('p', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('q', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('r', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('theta',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('x',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('y',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('z',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psi',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('gamma',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psiw',dynamic=True, shape=(self.num_times, 1))

        gamma_w_name_list = []
        wing_wake_coords_name_list = []

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
            self.add_parameter(surface_name,
                               dynamic=True,
                               shape=(self.num_times, nx, ny, 3))
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
            self.add_state(gamma_w_name,
                           dgammaw_dt_name,
                           shape=(nt - 1, ny - 1),
                           initial_condition_name=gamma_w_0_name,
                           output=gamma_w_int_name)
            self.add_state(wing_wake_coords_name,
                           dwake_coords_dt_name,
                           shape=(nt - 1, ny, 3),
                           initial_condition_name=wake_coords_0_name,
                           output=wake_coords_int_name)

            ####################################
            # profile outputs
            ####################################
            # F_name = surface_name + '_F'
            # self.add_profile_output(F_name)
        

        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.set_ode_system(ODESystemModel)
        self.set_profile_system(ProfileOpModel)


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')
        self.parameters.declare('h_stepsize')

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = self.parameters['h_stepsize']

        ####################################
        # Create parameters
        ####################################
        for data in AcStates_val_dict:
            string_name = data
            val = AcStates_val_dict[data]            
            # print('{:15} = {},shape{}'.format(string_name, val, val.shape))

            variable = self.create_input(string_name,
                                         val=val)

        initial_mesh_names = [
            x + '_initial_mesh' for x in surface_names
        ]

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            ####################################
            # Create parameters
            ####################################
            '''1. wing'''
            wing_val = mesh_val
            wing = self.create_input(surface_name, wing_val)

            ########################################
            # Initial condition for states
            ########################################
            '''1. wing_gamma_w_0'''
            wing_gamma_w_0 = self.create_input(gamma_w_0_name, np.zeros((nt - 1, ny - 1)))

            '''2. wing_wake_coords_0'''
            wing_wake_coords_0_val = np.zeros((nt - 1, ny, 3))
            wing_wake_coords_0 = self.create_input(wake_coords_0_name, wing_wake_coords_0_val)

        ########################################
        # Timestep vector
        ########################################
        h_vec = np.ones(num_times - 1) * h_stepsize
        h = self.create_input('h', h_vec)
        ########################################
        # params_dict to the init of ODESystem
        ########################################
        params_dict = {
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'nt': nt,
        }

        profile_params_dict = {
            # 'num_nodes': nt-1,
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'nt': nt,
        }

        # add an actuation model on the upstream
        # self.add(ActuationModel(surface_names=surface_names, surface_shapes=surface_shapes, num_nodes=nt-1),'actuation_temp')

        # Create Model containing integrator
        # ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching checkpointing', num_times, display='default', visualization='None')
        ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict,
                                                profile_parameters=profile_params_dict), 'subgroup')
        # self.add(ProfileSystemModel(**profile_params_dict),'profile_outputs')
        # self.add_design_variable('u',lower=1e-3, upper=10)
        # self.add_objective('res')
        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        ode_surface_shapes = [(num_times, ) + item for item in surface_shapes]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]
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
        self.add(submodel, name='ThrustDrag')

if __name__ == "__main__":
    # Script to create optimization problem
    be = 'python_csdl_backend'
    # be = 'csdl_om'
    make_video = 1

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
    theta_val = np.linspace(0,alpha,num=num_nodes)

    AcStates_val_dict = {
        'u': u_val,
        'v': np.zeros((num_nodes, 1)),
        'w': np.zeros((num_nodes, 1)),
        'p': np.zeros((num_nodes, 1)),
        'q': np.zeros((num_nodes, 1)),
        'r': np.zeros((num_nodes, 1)),
        'theta': theta_val,
        'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)),
        'y': np.zeros((num_nodes, 1)),
        'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)),
        'gamma': np.zeros((num_nodes, 1)),
        'psiw': np.zeros((num_nodes, 1)),
    }

    
    # https://github.com/LSDOlab/nasa_uli_tc1/blob/222d877228b609076dd352945f4cfe2d158d4973/execution_scripts/c172_climb.py#L33

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


    surface_names=['wing']
    surface_shapes=[(nx, ny, 3)]
    h_stepsize = delta_t = 1/16 

    
    if be == 'csdl_om':
        import csdl_om
        sim = csdl_om.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
    if be == 'python_csdl_backend':
        import python_csdl_backend
        sim = python_csdl_backend.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
    if be == 'csdl_lite':
        import csdl_lite
        sim = csdl_lite.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
        
    t_start = time.time()
    sim.run()
    print('simulation time is', time.time() - t_start)
    np.savetxt('cl12full',sim['wing_C_L'])
    # exit()

    # print('#' * 50, 'print states', '#' * 50)
    # # print('wing_gamma_w_int\n', sim['wing_gamma_w_int'])
    # # print('wing_wake_coords_int\n', sim['wing_wake_coords_int'])

    # print('#' * 50, 'print wings', '#' * 50)
    # print('wing', sim['wing'])
    # print('wing_gamma_w_int[-1]\n', sim['wing_gamma_w_int'][-1,:,:])

    # print('#' * 50, 'print wing_wake_total_vel', '#' * 50)
    # print('wing_wake_total_vel', sim['wing_wake_total_vel'])
    ######################################################
    # make video
    ######################################################

    if make_video == 1:
        axs = Axes(
            xrange=(0, 35),
            yrange=(-10, 10),
            zrange=(-3, 4),
        )
        video = Video("uvlm_test.gif", duration=10, backend='ffmpeg')
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
                vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                            r=8,
                            c='red')
                vp += vps
                vp += __doc__
                vps = Points(np.reshape(sim[surface_name+'_wake_coords_int'][i, 0:i, :, :],
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

    # sim.compute_totals(of='',wrt='*')
    ######################################################
    # end make video
    ######################################################

    # sim.visualize_implementation()
    # partials = sim.check_partials(compact_print=True)
    # sim.prob.check_totals(compact_print=True)