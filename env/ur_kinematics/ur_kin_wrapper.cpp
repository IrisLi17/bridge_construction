#define IKFAST_HAS_LIBRARY // Build IKFast with API functions
#define IKFAST_NO_MAIN // Don't include main() from IKFast

#include "ur_kin.cpp"

#include <vector>

namespace robots {
    class Kinematics {
        public: int num_of_joints, num_free_parameters;
        Kinematics();
        ~Kinematics();
        std::vector<float> forward(std::vector<float> joint_config);
        std::vector<float> inverse(std::vector<float> ee_pose);
    };

    Kinematics::Kinematics() {
        num_of_joints = GetNumJoints();
        num_free_parameters = GetNumFreeParameters();
    }

    Kinematics::~Kinematics() {}

    std::vector<float> Kinematics::forward(std::vector<float> joint_config) {
        IkReal eerot[9], eetrans[3];
        std::vector<float> ee_pose;
        if (joint_config.size() != num_of_joints) {
            printf("\nError: (forward kinematics) expects vector of %d values describing joint angles (in radians).\n\n", num_of_joints);
            return ee_pose;
        }
        // Parse joint_config
        IkReal joints[num_of_joints];
        for (unsigned int i=0; i<num_of_joints; i++) {
            joints[i] = joint_config[i];
        }

        ComputeFk(joints, eetrans, eerot);
        for (unsigned int i=0; i<3; i++) {
            ee_pose.push_back(eerot[3 * i + 0]);
            ee_pose.push_back(eerot[3 * i + 1]);
            ee_pose.push_back(eerot[3 * i + 2]);
            ee_pose.push_back(eetrans[i]);
        }
        return ee_pose;
    }

    std::vector<float> Kinematics::inverse(std::vector<float> ee_pose) {
        IkReal eerot[9], eetrans[3];
        std::vector<float> joint_configs;

        if (ee_pose.size() == 7) {
            // position (3) + quaternion (w, x, y, z)
            IkSolutionList<IkReal> solutions;
            std::vector<IkReal> vfree(num_free_parameters);

            // parse input
            eetrans[0] = ee_pose[0];
            eetrans[1] = ee_pose[1];
            eetrans[2] = ee_pose[2];
            double qw = ee_pose[3];
            double qx = ee_pose[4];
            double qy = ee_pose[5];
            double qz = ee_pose[6];
            double n = 1.0f / (qx * qx + qy * qy + qz * qz + qw * qw);
            qw *= n;
            qx *= n;
            qy *= n;
            qz *= n;
            eerot[0] = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;  eerot[1] = 2.0f*qx*qy - 2.0f*qz*qw;         eerot[2] = 2.0f*qx*qz + 2.0f*qy*qw;
            eerot[3] = 2.0f*qx*qy + 2.0f*qz*qw;         eerot[4] = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;  eerot[5] = 2.0f*qy*qz - 2.0f*qx*qw;
            eerot[6] = 2.0f*qx*qz - 2.0f*qy*qw;         eerot[7] = 2.0f*qy*qz + 2.0f*qx*qw;         eerot[8] = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;

            bool bSuccess = ComputeIk(eetrans, eerot, vfree.size()>0?&vfree[0]:NULL, solutions);

            if (!bSuccess) {
                return joint_configs;
            }

            unsigned int num_of_solutions = (int)solutions.GetNumSolutions();
            std::vector<IkReal> solvalues(num_of_joints);
            for(std::size_t i = 0; i < num_of_solutions; ++i) {
                const IkSolutionBase<IkReal>& sol = solutions.GetSolution(i);
                int this_sol_free_params = (int)sol.GetFree().size();

                // printf("sol%d (free=%d): ", (int)i, this_sol_free_params );
                std::vector<IkReal> vsolfree(this_sol_free_params);

                sol.GetSolution(&solvalues[0],vsolfree.size()>0?&vsolfree[0]:NULL);

                for( std::size_t j = 0; j < solvalues.size(); ++j)
                    joint_configs.push_back(solvalues[j]);
            }
        }
        else if (ee_pose.size() == 12) {
            // rotation matrix
            IkSolutionList<IkReal> solutions;
            std::vector<IkReal> vfree(num_free_parameters);
            eerot[0] = ee_pose[0]; eerot[1] = ee_pose[1]; eerot[2] = ee_pose[2];  eetrans[0] = ee_pose[3];
            eerot[3] = ee_pose[4]; eerot[4] = ee_pose[5]; eerot[5] = ee_pose[6];  eetrans[1] = ee_pose[7];
            eerot[6] = ee_pose[8]; eerot[7] = ee_pose[9]; eerot[8] = ee_pose[10]; eetrans[2] = ee_pose[11];

            bool bSuccess = ComputeIk(eetrans, eerot, vfree.size() > 0 ? &vfree[0] : NULL, solutions);
            if( !bSuccess ) {
                return joint_configs;
            }

            unsigned int num_of_solutions = (int)solutions.GetNumSolutions();
            std::vector<IkReal> solvalues(num_of_joints);
            for(std::size_t i = 0; i < num_of_solutions; ++i) {
                const IkSolutionBase<IkReal>& sol = solutions.GetSolution(i);
                int this_sol_free_params = (int)sol.GetFree().size();

                // printf("sol%d (free=%d): ", (int)i, this_sol_free_params );
                std::vector<IkReal> vsolfree(this_sol_free_params);

                sol.GetSolution(&solvalues[0],vsolfree.size()>0?&vsolfree[0]:NULL);

                for( std::size_t j = 0; j < solvalues.size(); ++j)
                    joint_configs.push_back(solvalues[j]);
            }
        }
        else {
            printf("\nError: (inverse kinematics) please specify transformation of end effector with one of the following formats:\n"
                   "    1) A vector of 7 values: a 3x1 translation (tX), and a 1x4 quaternion (w + i + j + k)\n"
                   "    2) A (row-major) vector of 12 values: a 3x4 rigid transformation matrix with a 3x3 rotation R (rXX), and a 3x1 translation (tX)\n\n");
            return joint_configs;

        }
        return joint_configs;
    }
}