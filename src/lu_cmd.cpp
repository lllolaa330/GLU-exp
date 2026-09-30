#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include "symbolic.h"
#include "numeric.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"
#include <chrono>

using namespace std;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void help_message()
{
    cout << endl;
    cout << "GLU program V3.0" << endl;
    cout << "Usage: ./lu_cmd -i inputfile" << endl;
    cout << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cout << "-p to enable perturbation" << endl;
}

int main(int argc, char** argv)
{
    SNicsLU *nicslu = nullptr;

    const char *matrixName = nullptr;
    bool PERTURB = false;

    double *ax = NULL, *ax_backup = NULL;
    unsigned int *ai = NULL, *ai_backup = NULL;
    unsigned int *ap = NULL, *ap_backup = NULL;
    unsigned int n, nnz_backup;

    if (argc < 3) {
        help_message();
        return -1;
    }

    for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-i") == 0) {
            if(i + 1 >= argc) {
                help_message();
                return -1;
            }
            matrixName = argv[i+1];
            i += 2;
        }
        else if (strcmp(argv[i], "-p") == 0) {
            PERTURB = true;
            i += 1;
        }        
        else {
            help_message();
            return -1;
        }
    }

    if (matrixName == nullptr) {
        help_message();
        return -1;
    }

    const auto total_start = std::chrono::steady_clock::now();
    nicslu = static_cast<SNicsLU *>(malloc(sizeof(SNicsLU)));
    if (nicslu == nullptr) {
        cerr << "Failed to allocate NicsLU context." << endl;
        return -1;
    }

    // Build a permutation/scaling-adjusted matrix and keep the numeric arrays in CSC.
    int err = preprocess(matrixName, nicslu, &ax, &ai, &ap, &ax_backup, &ai_backup, &ap_backup, &nnz_backup);
    if (err)
    {
        cerr << "Matrix preprocessing failed." << endl;
        free(nicslu);
        return -1;
    }

    n = nicslu->n;

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;

    const auto after_prepare = Clock::now();

    Symbolic_Matrix A_sym(n, cout, cerr);
    A_sym.fill_in(ai, ap);
    const auto after_symbolic = Clock::now();

    A_sym.csr();
    const auto after_csr = Clock::now();

    A_sym.predictLU(ai, ap, ax);
    const auto after_values = Clock::now();

    A_sym.leveling();
    const auto after_levels = Clock::now();

#if GLU_DEBUG
    A_sym.ABFTCalculateCCA();
//    A_sym.PrintLevel();
#endif

    // Numeric factorization on GPU updates A_sym.val in place to LU factors.
    LUonDevice(A_sym, cout, cerr, PERTURB);
    const auto after_device = Clock::now();

#if GLU_DEBUG
    A_sym.ABFTCheckResult();
#endif

    // Solve Ax=b with the computed factors and NICSLU permutations/scales.
    vector<REAL> b(n, 1.);
    vector<REAL> x = A_sym.solve_CSR(nicslu, b);
    
    const auto total_stop = std::chrono::steady_clock::now();
    const double total_ms = 
         std::chrono::duration<double, std::milli>(total_stop - total_start).count();
    cout << std::setprecision(12);
    cout << "Total solve wall time: " << total_ms << " ms" << endl;
    cout << "GLU input_preprocess_mixed: " << elapsed_ms(total_start, after_prepare) << " ms" << endl;
    cout << "GLU symbolic: " << elapsed_ms(after_prepare, after_symbolic) << " ms" << endl;
    cout << "GLU csr_structure: " << elapsed_ms(after_symbolic, after_csr) << " ms" << endl;
    cout << "GLU values_prepare: " << elapsed_ms(after_csr, after_values) << " ms" << endl;
    cout << "GLU leveling: " << elapsed_ms(after_values, after_levels) << " ms" << endl;
    cout << "GLU device_pipeline: " << elapsed_ms(after_levels, after_device) << " ms" << endl;
    cout << "GLU rhs_solve: " << elapsed_ms(after_device, total_stop) << " ms" << endl;
         
    {
        ofstream x_f("x.dat");
        x_f << std::scientific << std::setprecision(17); // fixed改为scientific 10改为17 (只改变了保留解的方式)
        for (REAL xx: x)
            x_f << xx << '\n';
    }

    // Residual verification: compute ||Ax - b|| using the original matrix
    {
        REAL n1 = 0., n2 = 0., ni = 0.;
        for (unsigned i = 0; i < n; ++i) {
            REAL sum = 0.;
            unsigned end = ap_backup[i + 1];
            for (unsigned j = ap_backup[i]; j < end; ++j) {
                sum += (REAL)ax_backup[j] * x[ai_backup[j]];
            }
            REAL tmp = sum - b[i];
            if (tmp < 0.) tmp = -tmp;
            n1 += tmp;
            n2 += tmp * tmp;
            if (tmp > ni) ni = tmp;
        }
        cout << "Ax-b (1-norm): " << n1 << endl;
        cout << "Ax-b (2-norm): " << sqrt(n2) << endl;
        cout << "Ax-b (infinite-norm): " << ni << endl;
    }

    NicsLU_Destroy(nicslu);
    free(nicslu);
    free(ax);
    free(ai);
    free(ap);
    free(ax_backup);
    free(ai_backup);
    free(ap_backup);

    return 0;
}
