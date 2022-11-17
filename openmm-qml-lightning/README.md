OpenMM-QML-Lightning
======================================

This plugin allows [QML-Lightning](https://github.com/nickjbrowning/qml-lightning) models to be used with [OpenMM](http://openmm.org). This works via defining a `TorchForce` object which can load a compiled TorchScript save file. This plugin and tutorial is adapted from the [OpenMM-Torch](https://github.com/openmm/openmm-torch) plugin. I recommended reading through and faimliarising yourself with the installation of OpenMM-Torch before attempting to install this plugin. 

Note that the TorchForce object expects initial input positions to be in angstrom, and for the model to be trained on energies in kcal/mol and forces in kcal/mol/angstrom. The TorchForce object converts to OpenMM units (kJ/mol and kJ/mol/nm) internally.

Building from Source
======================================
This plugin uses [CMake](https://cmake.org/) as its build system.  
Before compiling you must install the Pytorch C++ API [LibTorch](https://pytorch.org/cppdocs/installing.html), by following the instructions at https://pytorch.org.
You can then follow these steps:

1. Build QML-Lightning and OpenMM following their respective instructions.

2. Create a directory in which to build the plugin.

3. Run the CMake GUI or `ccmake`, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

4. Press "Configure".  (Do not worry if it produces an error message about not being able to find PyTorch.)

5. Set `OPENMM_DIR` to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.  If you are unsure of what directory this is, the script at the bottom of this section will print this out.

6. Set `QML_LIGHTNING_DIR` to point to the directory where the compiled TorchScript QML-Lightning libraries (`torchscript_fchl19.so`, `torchscript_sorf.so`) are located (for example, `/home/software/qml_lightning/build/lib.X/qml_lightning`).

7. Set `PYTORCH_DIR` to point to the directory where you installed LibTorch.

8. Set `CMAKE_INSTALL_PREFIX` to the directory where the plugin should be installed.  Usually,
this will be the same as `OPENMM_DIR`, so the plugin will be added to your OpenMM installation.

9. If you plan to build the CUDA platform, make sure that `CUDA_TOOLKIT_ROOT_DIR` is set correctly
and that `NN_BUILD_CUDA_LIB` is selected.

10. Press "Configure" again if necessary, then press "Generate".

11. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install` to install the plugin, and `make PythonInstall` to
install the Python wrapper.

```python
from simtk import openmm
import os
print(os.path.dirname(openmm.version.openmm_library_path))
```


Using the OpenMM-QML-Lightning Plugin
======================================

The following script provides an example of how to run MD with QML-Lightning.

```python
import openmm as mm
from openmm.unit import *
import openmmtorch as ot
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
    
    coordinates = np.zeros((natoms, 3))
    charges = np.zeros(natoms)
    
    data = np.loadtxt(fname, skiprows=2, dtype=np.dtype([('element', 'U2'), ('coordinates', 'f4', (3))]))
    
    df = fetch_table('elements')
    
    charges = [df.loc[df['symbol'] == str(v)]['atomic_number'].iloc[0] for v in data['element']]
    
    return  data['element'], np.array(charges), data['coordinates'], cell

def run_md_openmm_qml_lightning():
    
    print(Platform.getPluginLoadFailures())
    
    from openmm.app.element import  Element
    
    system = mm.System()
    
    elements, charges, coordinates, cell = load_xyz(sys.argv[1])
        
    natoms = coordinates.shape[0]

    force = ot.TorchForce(sys.argv[2])
    
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
   
    simulation.minimizeEnergy()
   
    start = time()
    simulation.step(1000)
    end = time()
    
    print (end - start)
    

run_md_openmm_qml_lightning()

```

This script can be run as follows:

```
python3 md_script.py water.xyz model.pt
```
where `model.pt` is the jitted model obtained through training with QML-Lightning. Note that the method:

```
force.setCharges(charges.tolist())
```

must be called after instantiation of the TorchForce object, as this is required for the FCHL representation and element-based projections used within QML-Lightning.
