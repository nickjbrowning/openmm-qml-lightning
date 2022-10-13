/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2022 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaTorchKernels.h"
#include "CudaTorchKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <cuda_runtime_api.h>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
#define CHECK_RESULT(result, prefix) \
if (result != CUDA_SUCCESS) { \
    std::stringstream m; \
    m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
    throw OpenMMException(m.str());\
}

CudaCalcTorchForceKernel::CudaCalcTorchForceKernel(string name, const Platform &platform, CudaContext &cu) :
		CalcTorchForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
	// Explicitly activate the primary context
	CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

CudaCalcTorchForceKernel::~CudaCalcTorchForceKernel() {
	cuDevicePrimaryCtxRelease(cu.getDevice());
}

void CudaCalcTorchForceKernel::initialize(const System &system, const TorchForce &force, torch::jit::script::Module &module) {

	this->module = module;

	// Initialize CUDA objects for PyTorch
	const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initialize PyTorch

	usePeriodic = force.usesPeriodicBoundaryConditions();

	for (int i = 0; i < force.getNumGlobalParameters(); i++)
		globalNames.push_back(force.getGlobalParameterName(i));

	int numParticles = system.getNumParticles();

	torch::TensorOptions options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

	posTensor = torch::empty( { 1, numParticles, 3 }, options.requires_grad(true));
	charges = torch::empty( { 1, numParticles }, torch::TensorOptions().device(device).dtype(torch::kFloat32));

	molIDs = torch::zeros(numParticles, torch::TensorOptions().device(device).dtype(torch::kInt32));
	atomIDs = torch::empty(numParticles, torch::TensorOptions().device(device).dtype(torch::kInt32));
	atom_counts = torch::zeros(1, torch::TensorOptions().device(device).dtype(torch::kInt32));

	atom_counts[0] = numParticles;

	for (int i = 0; i < numParticles; i++) {
		charges[0][i] = force.charges[i];
		atomIDs[i] = i;
	}

	// Push the PyTorch context
	// NOTE: Pytorch is always using the primary context.
	//       It makes the primary context current, if it is not a case.
	CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

	module.to(device);

	if (usePeriodic) {
		boxTensor = torch::empty( { 1, 3, 3 }, options);
		invboxTensor = torch::empty( { 1, 3, 3 }, options);
	} else {
		boxTensor = torch::empty( { 0, 3, 3 }, options);
		invboxTensor = torch::empty( { 0, 3, 3 }, options);
	}

	// Pop the PyToch context
	CUcontext ctx;
	CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
	assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack

	// Initialize CUDA objects for OpenMM-Torch
	ContextSelector selector(cu); // Switch to the OpenMM context
	map < string, string > defines;
	CUmodule program = cu.createModule(CudaTorchKernelSources::torchForce, defines);
	copyInputsKernel = cu.getKernel(program, "copyInputs");
	addForcesKernel = cu.getKernel(program, "addForces");
}

double CudaCalcTorchForceKernel::execute(ContextImpl &context, bool includeForces, bool includeEnergy) {
	int numParticles = cu.getNumAtoms();

	// Push to the PyTorch context
	CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

	// Get pointers to the atomic positions and simulation box
	void *posData;
	void *boxData;
	void *invboxData;

	posData = posTensor.data_ptr<float>();
	boxData = boxTensor.data_ptr<float>();
	invboxData = invboxTensor.data_ptr<float>();

	// Copy the atomic positions and simulation box to PyTorch tensors
	{
		ContextSelector selector(cu); // Switch to the OpenMM context
		void *inputArgs[] = { &posData, &boxData, &invboxData, &cu.getPosq().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles,
				cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(), &usePeriodic };
		cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
		CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
	}

	// Prepare the input of the PyTorch model
	vector < torch::jit::IValue > inputs = { posTensor * 10.0, charges, atomIDs, molIDs, atom_counts, boxTensor, invboxTensor };

	//    def forward(self, X, Z, atomIDs, molIDs, atom_counts, cell=torch.empty(0, 3, 3, device=torch.device('cuda')), inv_cell=torch.empty(0, 3, 3, device=torch.device('cuda'))):

	// Execute the PyTorch model

	torch::Tensor energyTensor, forceTensor;

	energyTensor = module.forward(inputs).toTensor();

	if (includeForces) {

		// Compute force by backprogating the PyTorch model

		energyTensor.backward();
		//std::cout << posTensor.grad() << std::endl;
		forceTensor = 4.184 * posTensor.grad();
		//std::cout << forceTensor << std::endl;

		// Get a pointer to the computed forces
		void *forceData;

		if (!(forceTensor.dtype() == torch::kFloat32)) // TODO: simplify the logic when support for PyTorch 1.7 is dropped
			forceTensor = forceTensor.to(torch::kFloat32);

		forceData = forceTensor.data_ptr<float>();

		CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the OpenMM context

		// Add the computed forces to the total atomic forces
		{
			ContextSelector selector(cu); // Switch to the OpenMM context
			int paddedNumAtoms = cu.getPaddedNumAtoms();
			int forceSign = -1;
			void *forceArgs[] = { &forceData, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms,
					&forceSign };
			cu.executeKernel(addForcesKernel, forceArgs, numParticles);
			CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
		}

		// Reset the forces
		posTensor.grad().zero_();
	}

	// Get energy
	const double energy = 4.184 * energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context

			// Pop to the PyTorch context
	CUcontext ctx;
	CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
	assert(primaryContext == ctx); // Check that the correct context was popped

	return energy;
}
