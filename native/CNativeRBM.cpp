// CNativeRBM.cpp : Definiert die exportierten Funktionen für die DLL-Anwendung.
//

#include "stdafx.h"
#include "RBM.h"
#include "berlin_iconn_rbm_NativeRBM.h"
#include <iostream>
#include <map>

int id = 0;
std::map<int, RBM *> rbmMap;


/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    createNativeRBM
* Signature: ([FI[FII[FFI)V
*/
JNIEXPORT jint JNICALL Java_berlin_iconn_rbm_NativeRBM_createNativeRBM
(JNIEnv * env, jobject obj, jfloatArray weightsData, jint rows, jint cols, jint threads)
{
	float* weightsBody = env->GetFloatArrayElements(weightsData, false);
	RBM * rbm = new RBM(
		weightsBody, rows, cols, threads);
	env->ReleaseFloatArrayElements(weightsData, weightsBody, 0);
	rbmMap[id++] = rbm;
	//for (int i = 0; i < weightsCols; i++)
	//{
	//	for (int j = 0; j < dataCols; j++)
	//		std::cout << rbm->getWeights()[i * dataCols + j] << ", ";

	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;
	//for (int i = 0; i < dataCols; i++)
	//{
	//	for (int j = 0; j < dataRows; j++) 
	//	{
	//		std::cout << rbm->getData()[i * dataRows + j] << ", ";
	//	}

	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;
	return id - 1;
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    deleteNativeRBM
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_deleteNativeRBM
(JNIEnv * env, jobject obj, jint id)
{
	delete rbmMap[id];
	rbmMap.erase(id);
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeWithoutError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeWithoutError
(JNIEnv * env, jobject obj, jint id)
{
	rbmMap[id]->trainWithoutError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeWithError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeWithError
(JNIEnv * env, jobject obj, jint id)
{
	rbmMap[id]->trainWithError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeBinarizedWithoutError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeBinarizedWithoutError
(JNIEnv * env, jobject obj, jint id)
{
	rbmMap[id]->trainBinarizedWithoutError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeBinarizedWithError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeBinarizedWithError
(JNIEnv * env, jobject obj, jint id)
{
	rbmMap[id]->trainBinarizedWithError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    getNativeWeights
* Signature: ()[F
*/
JNIEXPORT jfloatArray JNICALL Java_berlin_iconn_rbm_NativeRBM_getNativeWeights
(JNIEnv * env, jobject obj, jint id)
{
	jfloatArray result = env->NewFloatArray(rbmMap[id]->getWeightsLength());
	env->SetFloatArrayRegion(result, 0, rbmMap[id]->getWeightsLength(), rbmMap[id]->getWeights());

	return result;
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    getNativeError
* Signature: ()F
*/
JNIEXPORT jfloat JNICALL Java_berlin_iconn_rbm_NativeRBM_getNativeError
(JNIEnv * env, jobject obj, jint id) 
{
	return rbmMap[id]->getError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    setNativeWeights
* Signature: ([FI)V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_setNativeWeights
(JNIEnv * env, jobject obj, jint id, jfloatArray weights, jint weightsRows, jint weightsCols)
{
	float* weightsBody = env->GetFloatArrayElements(weights, false);
	rbmMap[id]->setWeights(weightsBody, weightsRows, weightsCols);
	env->ReleaseFloatArrayElements(weights, weightsBody, 0);

}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    setNativeData
* Signature: ([FII[F)V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_setNativeData
(JNIEnv * env, jobject obj, jint id, jfloatArray data, jint dataRows)
{
	float* dataBody = env->GetFloatArrayElements(data, false);
	rbmMap[id]->setData(dataBody, dataRows);
	env->ReleaseFloatArrayElements(data, dataBody, 0);

}


JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_setNativeLearningRate
(JNIEnv * env, jobject obj, jint id, jfloat learningRate) 
{
	rbmMap[id]->setLearningRate(learningRate);
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    runHidden
* Signature: (I[FI)[F
*/
JNIEXPORT jfloatArray JNICALL Java_berlin_iconn_rbm_NativeRBM_runHidden
(JNIEnv * env, jobject obj, jint id, jfloatArray data, jint rows) {

	float * dataBody = env->GetFloatArrayElements(data, false);
	float * hiddenBody = rbmMap[id]->runHidden(dataBody, rows);
	env->ReleaseFloatArrayElements(data, dataBody, 0);

	int hiddenLength = rows * rbmMap[id]->getWeightsCols();

	jfloatArray hidden = env->NewFloatArray(hiddenLength);
	env->SetFloatArrayRegion(hidden, 0, hiddenLength, hiddenBody);

	delete[] hiddenBody;
	return hidden;
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    runVisible
* Signature: (I[FI)[F
*/
JNIEXPORT jfloatArray JNICALL Java_berlin_iconn_rbm_NativeRBM_runVisible
(JNIEnv * env, jobject obj, jint id, jfloatArray data, jint rows) {

	float * dataBody = env->GetFloatArrayElements(data, false);
	float * visibleBody = rbmMap[id]->runVisible(dataBody, rows);
	env->ReleaseFloatArrayElements(data, dataBody, 0);

	int visibleLength = rows * rbmMap[id]->getWeightsRows();

	jfloatArray visible = env->NewFloatArray(visibleLength);
	env->SetFloatArrayRegion(visible, 0, visibleLength, visibleBody);
	delete[] visibleBody;
	return visible;

}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    error
* Signature: (I[FI)F
*/
JNIEXPORT jfloat JNICALL Java_berlin_iconn_rbm_NativeRBM_error
(JNIEnv * env, jobject obj, jint id, jfloatArray data, jint rows) {

	float * dataBody = env->GetFloatArrayElements(data, false);
	float error = rbmMap[id]->getError(dataBody, rows);
	env->ReleaseFloatArrayElements(data, dataBody, 0);

	return error;
}
