import time

import matplotlib.pyplot as plt
import openmdao.api as om
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_system_free import ODESystemModel
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_profile_outputs import ProfileSystemModel

from ozone.api import ODEProblem
import csdl

import numpy as np
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show


from lsdo_uvlm.uvlm_preprocessing.actuation_model_temp import ActuationModel
from modopt.scipy_library import SLSQP
from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem


def generate_simple_mesh(nx, ny, nt=None, offset=0):
    if nt == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0. + offset
    else:
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            if i == 0:
                mesh[i, :, :, 2] = offset
            else:
                # mesh[i, :, :, 2] = mesh[i - 1, :, :, 2] + offset
                mesh[i, :, :, 2] = mesh[0, :, :, 2] 
    return mesh


class ODEProblemTest(ODEProblem):
    def setup(self):

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
            print('{:15} = {},shape{}'.format(string_name, val, val.shape))

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
            wing_val = generate_simple_mesh(nx, ny, offset=0)
            wing = self.create_input(initial_mesh_names[i], wing_val)

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
            'num_nodes': nt-1,
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'nt': nt,
        }

        # add an actuation model on the upstream
        self.add(ActuationModel(surface_names=surface_names, surface_shapes=surface_shapes, num_nodes=nt-1),'actuation_temp')

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict), 'subgroup')
        self.add(ProfileSystemModel(**profile_params_dict),'profile_outputs')
        self.add_design_variable('u',lower=1e-3, upper=10)
        self.add_objective('res')



if __name__ == "__main__":
    # Script to create optimization problem
    be = 'python_csdl_backend'
    make_video = 1


    num_nodes = 32
    nt = num_nodes+1

    alpha_deg =0
    alpha = alpha_deg / 180 * np.pi
    deg = -np.deg2rad(10)

    dir = ([1, ] * 8 + [-1, ] * 8 + [1, ] * 8 + [-1, ] * 8 )[:num_nodes] 
    # define the direction of the flapping motion (hardcoding for now)

    AcStates_val_dict = {
        'u': np.ones((num_nodes, 1))* np.cos(alpha)*2,
        'v': np.zeros((num_nodes, 1)),
        'w': np.ones((num_nodes, 1))* np.sin(alpha),
        'p': np.zeros((num_nodes, 1)),
        'q': np.array(dir)*deg,
        'r': np.zeros((num_nodes, 1)),
        'theta': np.ones((num_nodes, 1))*alpha,
        'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)),
        'y': np.zeros((num_nodes, 1)),
        'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)),
        'gamma': np.zeros((num_nodes, 1)),
        'psiw': np.zeros((num_nodes, 1)),
    }

    nx = 3
    ny = 4

    surface_names=['wing']
    surface_shapes=[(nx, ny, 3)]
    h_stepsize = delta_t = 0.5

    
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
    # exit()


    # prob = CSDLProblem(problem_name='TrimAnalysis', simulator=sim)
    # # Setup your preferred optimizer (SLSQP) with the Problem object
    # # optimizer = SLSQP(prob, maxiter=100, ftol=1e-10)
    # optimizer = SNOPT(prob, Major_iterations=100, Optimality_tolerance=1e-8)
    # # Solve your optimization problem
    # optimizer.solve()
    # # Print results of optimization
    # optimizer.print_results()

    prob = CSDLProblem(problem_name='TrimAnalysis', simulator=sim)
    optimizer = SNOPT(
        prob, 
        Major_iterations = 3000, 
        Major_optimality=1e-7, 
        Major_feasibility=2e-5,
        append2file=True,
    )
    # optimizer = SLSQP(prob, maxiter=100, ftol=1e-10)
    optimizer.solve()
    optimizer.print_results()

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
        video = Video("spider.gif", duration=10, backend='ffmpeg')
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
                    axes=False)  # render the scene
            video.addFrame()  # add individual frame
            # time.sleep(0.1)
            # vp.interactive().close()
            vp.closeWindow()
        vp.closeWindow()
        video.close()  # merge all the recorded frames
    ######################################################
    # end make video
    ######################################################

    # sim.visualize_implementation()
    # partials = sim.check_partials(compact_print=True)
    # sim.prob.check_totals(compact_print=True)