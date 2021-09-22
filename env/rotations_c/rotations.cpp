#include <stdio.h>
#include <math.h>
#include "Python.h"


// static PyObject* SpamError;

static double internalGetFloatFromSequence(PyObject* seq, int index)
{
	double v = 0.0;
	PyObject* item;

	if (PyList_Check(seq))
	{
		item = PyList_GET_ITEM(seq, index);
		v = PyFloat_AsDouble(item);
	}
	else
	{
		item = PyTuple_GET_ITEM(seq, index);
		v = PyFloat_AsDouble(item);
	}
	return v;
}

static PyObject* mat2quat(PyObject* self, PyObject* args, PyObject* keywds) {
    double mat[9];
    PyObject* matObj;
    static char* kwlist[] = {"mat", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &matObj))
	{
		return NULL;
	}
    if (matObj)
	{
		PyObject* seq;
		int len, i;
		seq = PySequence_Fast(matObj, "expected a sequence");
		len = PySequence_Size(matObj);
		if (len == 9)
		{
			for (i = 0; i < 9; i++)
			{
				mat[i] = internalGetFloatFromSequence(seq, i);
			}
		}
		else
		{
			// PyErr_SetString(SpamError, "Rotation matrix need 9 elements.");
            printf("Rotation matrix need 9 elements");
			Py_DECREF(seq);
			return NULL;
		}
		Py_DECREF(seq);
	}
	else
	{
		// PyErr_SetString(SpamError, "Rotation matrix need 9 elements.");
        printf("Rotation matrix need 9 elements");
		return NULL;
	}
    
    double qr = 0.5 * sqrt(1.0 + mat[0] + mat[4] + mat[8]);
    double qi, qj, qk;

    if (qr > 1e-6) {
        qi = (mat[7] - mat[5]) / (4 * qr);
        qj = (mat[2] - mat[6]) / (4 * qr);
        qk = (mat[3] - mat[1]) / (4 * qr);
    } else {
        double qi_square = (mat[0] + 1) / 2.0;
        double qj_square = (mat[4] + 1) / 2.0;
        double qk_square = (mat[8] + 1) / 2.0;
        qi = sqrt(qi_square);
        if (mat[1] > 0) {
            qj = sqrt(qj_square);
        } else {
            qj = -sqrt(qj_square);
        }
        if (mat[2] > 0) {
            qk = sqrt(qk_square);
        } else {
            qk = -sqrt(qk_square);
        }        
    }

    double quat[4] = {qi, qj, qk, qr};

    PyObject* pylist;
    pylist = PyTuple_New(4);
    for (int i = 0; i < 4; i++)
        PyTuple_SetItem(pylist, i, PyFloat_FromDouble(quat[i]));
    return pylist;
}

static PyObject* quat_mul(PyObject* self, PyObject* args, PyObject* keywds) {
    double q1[4], q2[4];
    PyObject* q1Obj, *q2Obj;
    static char* kwlist[] = {"q1", "q2", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist, &q1Obj, &q2Obj)) {
		return NULL;
	}
    if (q1Obj) {
		PyObject* seq;
		int len, i;
		seq = PySequence_Fast(q1Obj, "expected a sequence");
		len = PySequence_Size(q1Obj);
		if (len == 4) {
			for (i = 0; i < 4; i++) {
				q1[i] = internalGetFloatFromSequence(seq, i);
			}
		} else {
			printf("A quaternion needs 4 elements");
			Py_DECREF(seq);
			return NULL;
		}
		Py_DECREF(seq);
	} else {
		printf("A quaternion needs 4 elements");
		return NULL;
	}
    if (q2Obj) {
		PyObject* seq;
		int len, i;
		seq = PySequence_Fast(q2Obj, "expected a sequence");
		len = PySequence_Size(q2Obj);
		if (len == 4) {
			for (i = 0; i < 4; i++) {
				q2[i] = internalGetFloatFromSequence(seq, i);
			}
		} else {
			printf("A quaternion needs 4 elements");
			Py_DECREF(seq);
			return NULL;
		}
		Py_DECREF(seq);
	} else {
		printf("A quaternion needs 4 elements");
		return NULL;
	}

    if (abs(q1[0]*q1[0] + q1[1]*q1[1] + q1[2]*q1[2] + q1[3]*q1[3] - 1) > 1e-5) {
        printf("Norm of q1 is not correct.");
        return NULL;
    }
    if (abs(q2[0]*q2[0] + q2[1]*q2[1] + q2[2]*q2[2] + q2[3]*q2[3] - 1) > 1e-5) {
        printf("Norm of q2 is not correct.");
        return NULL;
    }
    double w1 = q1[3], x1 = q1[0], y1 = q1[1], z1 = q1[2];
    double w2 = q2[3], x2 = q2[0], y2 = q2[1], z2 = q2[2];

    double w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    double x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    double y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2;
    double z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2;
    double q[4] = {x, y, z, w};

    PyObject* pylist;
    pylist = PyTuple_New(4);
    for (int i = 0; i < 4; i++)
        PyTuple_SetItem(pylist, i, PyFloat_FromDouble(q[i]));
    return pylist;
}

static PyObject* quat_conjugate(PyObject* self, PyObject* args, PyObject* keywds) {
    double q0[4];
    static char* kwlist[] = {"q", NULL};

    PyObject* qObj;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &qObj)) {
		return NULL;
	}
    if (qObj) {
		PyObject* seq;
		int len, i;
		seq = PySequence_Fast(qObj, "expected a sequence");
		len = PySequence_Size(qObj);
		if (len == 4) {
			for (i = 0; i < 4; i++) {
				q0[i] = internalGetFloatFromSequence(seq, i);
			}
		} else {
			printf("A quaternion needs 4 elements");
			Py_DECREF(seq);
			return NULL;
		}
		Py_DECREF(seq);
	} else {
		printf("A quaternion needs 4 elements");
		return NULL;
	}

    for (int i=0; i<3; i++) {
        q0[i] *= -1;
    }
    PyObject* pylist;
    pylist = PyTuple_New(4);
    for (int i = 0; i < 4; i++)
        PyTuple_SetItem(pylist, i, PyFloat_FromDouble(q0[i]));
    return pylist;
}

/*
static PyObject* quat_rot_vec(PyObject* self, PyObject* args, PyObject* keywds) {
    double q[4], v0[3];
    PyObject* qObj, * vecObj;
    static char* kwlist[] = {"q", "v0", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist, &qObj, &vecObj)) {
		return NULL;
	}
//    if (qObj) {
//		PyObject* seq;
//		int len, i;
//		seq = PySequence_Fast(qObj, "expected a sequence");
//		len = PySequence_Size(qObj);
//		if (len == 4) {
//			for (i = 0; i < 4; i++) {
//				q[i] = internalGetFloatFromSequence(seq, i);
//			}
//		} else {
//			printf("A quaternion needs 4 elements");
//			Py_DECREF(seq);
//			return NULL;
//		}
//		Py_DECREF(seq);
//	} else {
//		printf("A quaternion needs 4 elements");
//		return NULL;
//	}
    if (vecObj) {
        PyObject* seq;
        int len, i;
        seq = PySequence_Fast(vecObj, "expected a sequence");
        len = PySequence_Size(vecObj);
        if (len == 3) {
            for (i = 0; i < 3; i++) {
                v0[i] = internalGetFloatFromSequence(seq, i);
            }
        } else {
            printf("A vector needs 3 elements");
            Py_DECREF(seq);
            return NULL;
        }
        Py_DECREF(seq);
    } else {
        printf("A vector needs 3 elements");
        return NULL;
    }
    double v0_norm = sqrt(v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]);
    double q0_vec[4] = {v0[0] / v0_norm, v0[1] / v0_norm, v0[2] / v0_norm, 0.};
    PyObject* q0_vecObj = PyTuple_New(4);
    for (int i = 0; i < 4; i++) {
        PyTuple_SetItem(q0_vecObj, i, PyFloat_FromDouble(q0_vec[i]));
    }
    PyObject* q_vObj;
    {
        PyObject* q_conjObj = quat_conjugate(self, qObj, NULL);
        PyObject* arg1 = PyTuple_New(2);
        PyTuple_SetItem(arg1, 0, q0_vecObj);
        PyTuple_SetItem(arg1, 1, q_conjObj);
        PyObject* temp = quat_mul(self, arg1, NULL);
        PyObject* arg2 = PyTuple_New(2);
        PyTuple_SetItem(arg2, 0, qObj);
        PyTuple_SetItem(arg2, 1, temp);
        q_vObj = quat_mul(self, arg2, NULL);
        Py_DECREF(q_conjObj);
        Py_DECREF(q0_vecObj);
        Py_DECREF(arg1);
        Py_DECREF(temp);
        Py_DECREF(arg2);
    }

    double result_v[3];
    for (int i = 0; i < 3; i++) {
        result_v[i] = internalGetFloatFromSequence(q_vObj, i);
    }
    Py_DECREF(q_vObj);
    PyObject* result_vObj = PyTuple_New(3);
    for (int i = 0; i < 3; i++) {
        PyTuple_SetItem(result_vObj, i, PyFloat_FromDouble(result_v[i]));
    }
    return result_vObj;
}
*/

static PyObject* quat2mat(PyObject* self, PyObject* args, PyObject* keywds) {
    double quat[4];
    PyObject* quatObj;
    static char* kwlist[] = {"quat", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &quatObj))
	{
		return NULL;
	}
    if (quatObj) {
		PyObject* seq;
		int len, i;
		seq = PySequence_Fast(quatObj, "expected a sequence");
		len = PySequence_Size(quatObj);
		if (len == 4) {
			for (i = 0; i < 4; i++) {
				quat[i] = internalGetFloatFromSequence(seq, i);
			}
		} else {
			printf("Quaternion needs 4 elements");
			Py_DECREF(seq);
			return NULL;
		}
		Py_DECREF(seq);
	} else {
		printf("Quaternion needs 4 elements");
		return NULL;
	}
    if (abs(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3] - 1) > 1e-5) {
        return NULL;
    }
    double mat[9];
    double qi = quat[0], qj = quat[1], qk = quat[2], qr = quat[3];
    mat[0] = 1 - 2 * (qj * qj + qk * qk);
    mat[1] = 2 * (qi * qj - qk * qr);
    mat[2] = 2 * (qi * qk + qj * qr);
    mat[3] = 2 * (qi * qj + qk * qr);
    mat[4] = 1 - 2 * (qi * qi + qk * qk);
    mat[5] = 2 * (qj * qk - qi * qr);
    mat[6] = 2 * (qi * qk - qj * qr);
    mat[7] = 2 * (qj * qk + qi * qr);
    mat[8] = 1 - 2 * (qi * qi + qj * qj);
    PyObject* pylist = PyTuple_New(9);
    for (int i = 0; i < 9; i++) {
        PyTuple_SetItem(pylist, i, PyFloat_FromDouble(mat[i]));
    }
    return pylist;
}

static PyMethodDef SpamMethods[] = {
    {"mat2quat", (PyCFunction)mat2quat, METH_VARARGS | METH_KEYWORDS,
	 "Convert a rotation matrix with 9 elements to quaternion [x, y, z, w] as in URDF/SDF convention"},
    {"quat_mul", (PyCFunction)quat_mul, METH_VARARGS | METH_KEYWORDS,
     "Quaternion multiplication q1 * q2"},
    {"quat_conjugate", (PyCFunction)quat_conjugate, METH_VARARGS | METH_KEYWORDS,
     "Quaternion conjugation"},
    // {"quat_rot_vec", (PyCFunction)quat_rot_vec, METH_VARARGS | METH_KEYWORDS,
    //  "Rotate a vector by quaternion"},
    {"quat2mat", (PyCFunction)quat2mat, METH_VARARGS | METH_KEYWORDS,
     "Convert quaternion to rotation matrix of 9 elements"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT, "rotations", /* m_name */
	"Python bindings for Pybullet rotations ", /* m_doc */
	-1,            /* m_size */
	SpamMethods,   /* m_methods */
	NULL,          /* m_reload */
	NULL,          /* m_traverse */
	NULL,          /* m_clear */
	NULL,          /* m_free */
};

PyMODINIT_FUNC PyInit_rotations(void) {
    PyObject* m;
    m = PyModule_Create(&moduledef);
    return m;
}