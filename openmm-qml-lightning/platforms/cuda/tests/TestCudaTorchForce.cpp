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

/**
 * This tests the CUDA implementation of TorchForce.
 */

#include "TorchForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerTorchCudaKernelFactories();


/*
21

C          2.05639       -0.16543        0.39727
C          0.68890        2.02854       -0.64816
C          2.74071        1.02327        0.12533
C          2.05734        2.11782       -0.40016
C         -3.59396        1.03209        0.04589
C          0.67738       -0.27507        0.15097
C          0.00274        0.84234       -0.35887
O          0.63293       -2.54526        0.93360
O         -1.76452        1.43252        1.56571
O         -1.24894       -1.68411        0.12975
C          0.05278       -1.58881        0.44958
C         -2.15199        1.12841        0.44513
O         -1.36167        0.82832       -0.65825
H         -1.47652       -2.60111        0.39232
H          2.60948       -1.01173        0.80284
H          0.15708        2.88266       -1.05823
H          3.80777        1.09126        0.32364
H          2.59044        3.04081       -0.61405
H         -3.81640        0.02049       -0.30287
H         -4.22480        1.24591        0.91440
H         -3.81296        1.76576       -0.73332
*/
void testAspirinForce() {

	cout << "Running test on hardcoded aspirin configuration..." << endl;

	vector < Vec3 > positions(21);

	positions[0]  = Vec3(2.05639,       -0.16543,        0.39727) /10.0;
	positions[1]  = Vec3(0.68890,        2.02854,       -0.64816)/10.0;
	positions[2]  = Vec3(2.74071,        1.02327,        0.12533)/10.0;
	positions[3]  = Vec3(2.05734,        2.11782,       -0.40016)/10.0;
	positions[4]  = Vec3(-3.59396,        1.03209,        0.04589)/10.0;
	positions[5]  = Vec3(0.67738,       -0.27507,        0.15097)/10.0;
	positions[6]  = Vec3(0.00274,        0.84234,       -0.35887)/10.0;
	positions[7]  = Vec3(0.63293,       -2.54526,        0.93360)/10.0;
	positions[8]  = Vec3(-1.76452,        1.43252,        1.56571)/10.0;
	positions[9]  = Vec3(-1.24894 ,      -1.68411,        0.12975)/ 10.0;
	positions[10]  = Vec3(0.05278 ,      -1.58881,        0.44958)/ 10.0;
	positions[11]  = Vec3(-2.15199,        1.12841,        0.44513)/ 10.0;
	positions[12]  = Vec3(-1.36167,        0.82832,       -0.65825)/ 10.0;
	positions[13]  = Vec3(-1.47652,       -2.60111,        0.39232)/ 10.0;
	positions[14]  = Vec3(2.60948 ,      -1.01173 ,       0.80284)/10.0;
	positions[15]  = Vec3( 0.15708,        2.88266,       -1.05823)/ 10.0;
	positions[16]  = Vec3(3.80777 ,       1.09126 ,       0.32364)/ 10.0;
	positions[17]  = Vec3(2.59044 ,       3.04081 ,      -0.61405)/ 10.0;
	positions[18]  = Vec3(-3.81640,        0.02049,       -0.30287)/ 10.0;
	positions[19]  = Vec3(-4.22480,        1.24591,        0.91440)/ 10.0;
	positions[20]  = Vec3(-3.81296,        1.76576,       -0.73332)/ 10.0;

	vector <float> charges(21);

	charges[0] = 6.0;
	charges[1] = 6.0;
	charges[2] = 6.0;
	charges[3] = 6.0;
	charges[4] = 6.0;
	charges[5] = 6.0;
	charges[6] = 6.0;
	charges[7] = 8.0;
	charges[8] = 8.0;
	charges[9] = 8.0;
	charges[10] = 6.0;
	charges[11] = 6.0;
	charges[12] = 8.0;
	charges[13] = 1.0;
	charges[14] = 1.0;
	charges[15] = 1.0;
	charges[16] = 1.0;
	charges[17] = 1.0;
	charges[18] = 1.0;
	charges[19] = 1.0;
	charges[20] = 1.0;

	System system;


	for (int i = 0; i < positions.size(); i++) {
		system.addParticle(charges[i]); // not correct but only matters for MD...
	}

	TorchForce *force = new TorchForce("tests/model_sorf.pt");

	force->setCharges(charges);

	system.addForce(force);

	// Compute the forces and energy.

	VerletIntegrator integ(1.0);
	Platform &platform = Platform::getPlatformByName("CUDA");
	Context context(system, integ, platform);
	context.setPositions(positions);
	State state = context.getState(State::Energy | State::Forces);

	float expectedEnergy = -122.511;

	cout << "-- Potential Energy --" << endl;

	cout << state.getPotentialEnergy() << endl;

	vector<Vec3> forces = state.getForces();

	cout << "-- Forces --" << endl;
	for (int i = 0; i < forces.size(); i++) {
		cout  << forces[i] << endl;
	}

	ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-3);

	cout << "Assertion passed - result looks OK." << endl;

}


int main(int argc, char *argv[]) {
	try {
		registerTorchCudaKernelFactories();
		if (argc > 1)
			Platform::getPlatformByName("CUDA").setPropertyDefaultValue("Precision", string(argv[1]));
		testAspirinForce();
	} catch (const std::exception &e) {
		std::cout << "exception: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Done with all tests." << std::endl;
	return 0;
}
