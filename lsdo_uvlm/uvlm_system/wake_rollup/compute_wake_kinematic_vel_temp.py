# 1. compute wake kinematic vel
# -frame_vel + rot_vel
# TODO: figure out this rot_vel for the plunging wing case


# 2. compute wake induced vel
# BS (surface_wake_coords, (induced by) all bound and wake coords) 
# gamma_b and gamma_W (depending on the ordering in the last line)


# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix

from lsdo_uvlm.uvlm_system.wake_rollup.eval_pts_velocities_mls import EvalPtsVel


class ComputeWakeKinematicVel(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.
    A \gamma_b = b - M \gamma_w
    parameters
    ----------

    collocation_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the bd vertices collocation_pts     
    wake_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the wake panel collcation pts 
    wake_circulations[num_wake_panel] : csdl array
        a concatenate vector of the wake circulation strength
    Returns
    -------
    vel_col_w[num_evel_pts_x*num_vortex_panel_x* num_evel_pts_y*num_vortex_panel_y,3]
    csdl array
        the velocities computed using the aic_col_w from biot svart's law
        on bound vertices collcation pts induces by the wakes
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)


        self.parameters.declare('n_wake_pts_chord', default=2)

        self.parameters.declare('name', default='')



    def define(self):
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        name = self.parameters['name']

        num_nodes = surface_shapes[0][0]
        
        wake_kinematic_vel_shapes = [
            tuple((item[0],n_wake_pts_chord, item[2], item[3]))
            for item in surface_shapes
        ]
        wake_kinematic_vel_names = [x + '_wake_kinematic_vel' for x in surface_names]
        # wake_kinematic_vel_names = [x + '_wake_total_vel' for x in surface_names]

        frame_vel = self.declare_variable(name + 'frame_vel',shape=(num_nodes,3))

        for i in range(len(surface_names)):
            surface_wake_kinematic_vel = csdl.expand(-frame_vel,(wake_kinematic_vel_shapes[i]),'il->ijkl')
            self.register_output(wake_kinematic_vel_names[i],surface_wake_kinematic_vel)
