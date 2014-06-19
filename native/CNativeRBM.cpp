// CNativeRBM.cpp : Definiert die exportierten Funktionen für die DLL-Anwendung.
//

#include "stdafx.h"
#include "RBM.h"
#include "berlin_iconn_rbm_NativeRBM.h"
#include <iostream>


RBM *rbm;


/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    createNativeRBM
* Signature: ([FI[FII[FFI)V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_createNativeRBM
(JNIEnv * env, jobject obj, 
jfloatArray weights, jint weightsCols, 
jfloatArray data, jint dataRows, jint dataCols,
jfloatArray mean, 
jfloat learningRate, 
jint threads)
{

	float* dataBody = env->GetFloatArrayElements(data, false);
	float* weightsBody = env->GetFloatArrayElements(weights, false);
	rbm = new RBM(
		weightsBody, weightsCols,
		dataBody, dataRows, dataCols, 
		learningRate, threads);
	env->ReleaseFloatArrayElements(weights, weightsBody, 0);
	env->ReleaseFloatArrayElements(data, dataBody, 0);

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

}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    deleteNativeRBM
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_deleteNativeRBM
(JNIEnv * env, jobject obj)
{
	delete rbm;
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeWithoutError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeWithoutError
(JNIEnv * env, jobject obj)
{
	rbm->trainWithoutError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeWithError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeWithError
(JNIEnv * env, jobject obj)
{
	rbm->trainWithError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeBinarizedWithoutError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeBinarizedWithoutError
(JNIEnv * env, jobject obj)
{
	rbm->trainBinarizedWithoutError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    trainNativeBinarizedWithError
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_trainNativeBinarizedWithError
(JNIEnv * env, jobject obj)
{
	rbm->trainBinarizedWithError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    getNativeWeights
* Signature: ()[F
*/
JNIEXPORT jfloatArray JNICALL Java_berlin_iconn_rbm_NativeRBM_getNativeWeights
(JNIEnv * env, jobject obj)
{
	jfloatArray result = env->NewFloatArray(rbm->getWeightsLength());
	env->SetFloatArrayRegion(result, 0, rbm->getWeightsLength(), rbm->getWeights());

	return result;
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    getNativeError
* Signature: ()F
*/
JNIEXPORT jfloat JNICALL Java_berlin_iconn_rbm_NativeRBM_getNativeError
(JNIEnv * env, jobject obj) 
{
	return rbm->getError();
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    setNativeWeights
* Signature: ([FI)V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_setNativeWeights
(JNIEnv * env, jobject obj, jfloatArray weights, jint weightsCols)
{

	float* weightsBody = env->GetFloatArrayElements(weights, false);
	rbm->setWeights(weightsBody, weightsCols);
	env->ReleaseFloatArrayElements(weights, weightsBody, 0);
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    setNativeData
* Signature: ([FII[F)V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_setNativeData
(JNIEnv * env, jobject obj, jfloatArray data, jint dataRows, jint dataCols, jfloatArray mean)
{

	float* dataBody = env->GetFloatArrayElements(data, false);
	rbm->setData(dataBody, dataRows, dataCols);
	env->ReleaseFloatArrayElements(data, dataBody, 0);
}

/*
* Class:     berlin_iconn_rbm_NativeRBM
* Method:    setNativeLearningRate
* Signature: (F)V
*/
JNIEXPORT void JNICALL Java_berlin_iconn_rbm_NativeRBM_setNativeLearningRate
(JNIEnv * env, jobject obj, jfloat learningRate) 
{
	rbm->setLearningRate(learningRate);
}