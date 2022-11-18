import openmm as mm
from openmm.unit import *
import openmm_qml_lightning as ot
import numpy as np
import torch as pt
from simtk.openmm import *
from openmm.app import *
from time import time
from sys import stdout
import sys
from mendeleev.fetch import fetch_table

def load_xyz(fname):

    fp = open(fname, 'r')
    
    lines = fp.readlines()
    
    fp.close()
    
    natoms = int(lines[0])
        
    cell = np.zeros((3, 3))
    
    if (',' in lines[1]):
        cell_info = lines[1].split(',')[1:]
        for i, vec_str in enumerate(cell_info):
            cell[i] = np.array([str(v) for v in vec_str.split()])
         
    lines = lines[2:]
    
    data = np.loadtxt(fname, skiprows=2, dtype=np.dtype([('element', 'U2'), ('coordinates', 'f4', (3))]))
    
    df = fetch_table('elements')
    
    charges = [df.loc[df['symbol'] == str(v)]['atomic_number'].iloc[0] for v in data['element']]
    
    return  data['element'], np.array(charges), data['coordinates'], cell

def unit_test_model():
    
    print(Platform.getPluginLoadFailures())
    
    from openmm.app.element import  Element
    
    system = mm.System()
    
    elements, charges, coordinates, cell = load_xyz('aspirin.xyz')
        
    natoms = coordinates.shape[0]

    force = ot.TorchForce('model_sorf.pt')
    
    force.setCharges(charges.tolist())
    
    system.addForce(force)
    
    if (not np.all(cell == 0.0)):
        system.setDefaultPeriodicBoxVectors(cell[0], cell[1], cell[2])
    
    top = mm.app.topology.Topology()
    chain = top.addChain(0)
    
    res = top.addResidue("mol", chain)
    
    for _ in range(natoms):
        element = Element.getBySymbol(elements[_])
        top.addAtom(element.symbol, element, res)
        system.addParticle(element.mass)
        
    integ = LangevinMiddleIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds)
    
    platform = mm.Platform.getPlatformByName('CUDA')
    
    simulation = Simulation(top, system, integ, platform)
    
    simulation.context.setPositions(coordinates * angstrom)
    
    state = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
    
    energy = state.getPotentialEnergy()
    ref_energy = -122.51275
    
    assert np.abs(energy._value  - ref_energy) < 0.01, f"potential energy difference is larger than 0.01: {energy._value  - ref_energy}"
    
    print ("aspirin model test passed")
    
unit_test_model()

