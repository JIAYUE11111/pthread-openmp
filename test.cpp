#include<pthread.h>
#include<iostream>
#include<cmath>
#include <arm_neon.h>
#include <semaphore.h>
#include <stdio.h>
#include <sys/time.h>
#define ROW 1024
#define TASK 8
#define INTERVAL 10000
using namespace std;
float matrix[ROW][ROW];
float revmat[ROW][ROW];
typedef long long ll;
typedef struct {
	int k;
	int t_id;
}threadParam_t;

//信号量定义，用于静态线程
sem_t sem_leader;
sem_t sem_Divsion[32];
sem_t sem_Elimination[32];
//新改进版
pthread_barrier_t division;
pthread_barrier_t elemation;
//静态线程数量
int NUM_THREADS = 4;
//动态线程分配：待分配行数
int remain = ROW;
pthread_mutex_t remainLock;

void reverse()
{
	for (int i = 0;i < ROW;i++)
		for (int j = 0;j < ROW;j++)
			revmat[j][i] = matrix[i][j];
}

void init()
{
	for (int i = 0;i < ROW;i++)
	{
		for (int j = 0;j < i;j++)
			matrix[i][j] = 0;
		for(int j = i;j<ROW;j++)
			matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0;k < 1000;k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0;j < ROW;j++)
			matrix[row1][j] += mult * matrix[row2][j];
	}
	reverse();
}

void plain() {
	for (int i = 0; i < ROW - 1; i++) {
		for (int j = i + 1; j < ROW; j++) {
			matrix[i][j] = matrix[i][j] / matrix[i][i];
		}
		matrix[i][i] = 1;
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				matrix[k][j] = matrix[k][j] - matrix[i][j] * matrix[k][i];
			}
			matrix[k][i] = 0;
		}
	}
}

void SIMD()
{
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)
		{//消去
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
	}
}

void ColSIMD()
{
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)//i为列
		{//逐列消去
			int j;//j为目标行
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_lane_f32(&matrix[j][i], subbee, 0);
				mult1 = vld1q_lane_f32(&matrix[j][k], mult1, 0);
				subbee = vld1q_lane_f32(&matrix[j + 1][i], subbee, 1);
				mult1 = vld1q_lane_f32(&matrix[j + 1][k], mult1, 1);
				subbee = vld1q_lane_f32(&matrix[j + 2][i], subbee, 2);
				mult1 = vld1q_lane_f32(&matrix[j + 2][k], mult1, 2);
				subbee = vld1q_lane_f32(&matrix[j + 3][i], subbee, 3);
				mult1 = vld1q_lane_f32(&matrix[j + 3][k], mult1, 3);
				mult2 = vld1q_dup_f32(&matrix[k][i]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_lane_f32(&matrix[j][i], subbee, 0);
				vst1q_lane_f32(&matrix[j + 1][i], subbee, 1);
				vst1q_lane_f32(&matrix[j + 2][i], subbee, 2);
				vst1q_lane_f32(&matrix[j + 3][i], subbee, 3);
			}
		}
		for (int i = k + 1; i < ROW; i++)
			matrix[i][k] = 0;
	}
}

void ColSIMDcached()
{
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&revmat[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			revmat[j][k] = revmat[j][k] / revmat[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_lane_f32(&revmat[j][k], divee, 0);
			divee = vld1q_lane_f32(&revmat[j+1][k], divee, 1);
			divee = vld1q_lane_f32(&revmat[j+2][k], divee, 2);
			divee = vld1q_lane_f32(&revmat[j+3][k], divee, 3);
			divee = vdivq_f32(divee, diver);
			vst1q_lane_f32(&revmat[j][k], divee, 0);
			vst1q_lane_f32(&revmat[j + 1][k], divee, 1);
			vst1q_lane_f32(&revmat[j + 2][k], divee, 2);
			vst1q_lane_f32(&revmat[j + 3][k], divee, 3);
		}
		revmat[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)//i为行
		{//轉置前逐列消去，轉置后逐行消去
			int j;//j为目标列
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_f32(&revmat[i][j]);
				mult1 = vld1q_f32(&revmat[k][j]);
				mult2 = vld1q_dup_f32(&revmat[i][k]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_f32(&revmat[i][j], subbee);
			}
		}
		for (int i = k + 1; i < ROW; i++)
			revmat[k][i] = 0;
	}
}

void GaussElimi() {
	float32x4_t vt, va, vaji, vaik, vajk, vx;
	for (int i = 0; i <= ROW-1; ++i) {
		float xx = 1.0 / matrix[i][i];
		vt = vld1q_dup_f32(&xx);
		for (int j = i + 1; j + 4 <= ROW - 1; j += 4) {
			va = vld1q_f32(matrix[i] + j);
			va = vmulq_f32(va, vt);
			vst1q_f32(matrix[i] + j, va);
		}
		for (int j = ROW - ROW % 4; j < ROW; ++j) {
			matrix[i][j] /= matrix[i][i];

		}
		matrix[i][i] = 1.0;
		for (int j = i + 1; j <= ROW - 1; ++j) {
			vaji = vld1q_dup_f32(matrix[j] + i);
			for (int k = i + 1; k + 4 <= ROW - 1; k += 4) {
				vaik = vld1q_f32(matrix[i] + k);
				vajk = vld1q_f32(matrix[j] + k);
				vx = vmulq_f32(vaik, vaji);
				vajk = vsubq_f32(vajk, vx);
				vst1q_f32(matrix[j] + k, vajk);
			}
			for (int k = ROW - ROW % 4; k < ROW; ++k) {
				matrix[j][k] = matrix[j][k] - matrix[i][k] * matrix[j][i];
			}
			matrix[j][i] = 0.0;
		}
	}
}//不对齐


void* dynamicFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int j = k + 1;j < ROW;++j)
		matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
	matrix[i][k] = 0;
	pthread_exit(NULL);
}

void* dynamicFuncSIMD(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
	int j;
	for (j = k + 1;j < ROW&& ((ROW - j) & 3);++j)//串行处理对齐
		matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
	for (;j < ROW;j += 4) 
	{
		float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
		float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
		mult2 = vmulq_f32(mult1, mult2);
		sub1 = vsubq_f32(sub1, mult2);
		vst1q_f32(&matrix[i][j], sub1);
	}
	matrix[i][k] = 0;
	pthread_exit(NULL);
}

void dynamicMain(void* (*threadFunc)(void*))
{
	for(int k = 0; k < ROW; ++k)
	{//主线程做除法操作
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//创建工作线程，进行消去操作
		int worker_count = ROW-1-k; //工作线程数量
		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for(int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* newDynamicFuncSIMD(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int i = k + t_id + 1;i < ROW;i += NUM_THREADS)
	{
		float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
		for (;j < ROW;j += 4)
		{
			float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
			float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
			mult2 = vmulq_f32(mult1, mult2);
			sub1 = vsubq_f32(sub1, mult2);
			vst1q_f32(&matrix[i][j], sub1);
		}
		matrix[i][k] = 0;
	}
	pthread_exit(NULL);
}

void newDynamicMain(void* (*threadFunc)(void*))
{
	for (int k = 0; k < ROW; ++k)
	{//主线程做除法操作
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//创建工作线程，进行消去操作
		int worker_count = NUM_THREADS; //工作线程数量
		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* staticFunc(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		pthread_barrier_wait(&division);// 阻塞，等待主线完成除法操作
		//循环划分任务
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//消去
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void staticMain()
{
	//初始化信号量
	pthread_barrier_init(&division, NULL, NUM_THREADS + 1);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS + 1);
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, staticFunc, (void*)t_id);

	for (int k = 0; k < ROW; ++k)
	{
		//主线程做除法操作
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		pthread_barrier_wait(&division);
		// 主线程再次唤醒工作线程进入下一轮次的消去任务
		pthread_barrier_wait(&elemation);
	}
	for(int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void* staticFuncOpt(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		// t_id 为 0 的线程做除法操作，其它工作线程先等待
		// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
		// 比信号量更简洁的同步方式是使用 barrier
		if (t_id == 0)
		{
			float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[k][j] = matrix[k][j] / matrix[k][k];
			for (;j < ROW;j += 4)
			{
				float32x4_t divee = vld1q_f32(&matrix[k][j]);
				divee = vdivq_f32(divee, diver);
				vst1q_f32(&matrix[k][j], divee);
			}
			matrix[k][k] = 1.0;
		}
		else sem_wait(&sem_Divsion[t_id-1]); // 阻塞，等待完成除法操作
		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_post(&sem_Divsion[t_id]);

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//消去
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// 所有线程一起进入下一轮
		if (t_id == 0)
		{
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_wait(&sem_leader);
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_post(&sem_Elimination[t_id]);
		}
		else
		{
			sem_post(&sem_leader);
			sem_wait(&sem_Elimination[t_id-1]);
		}
	}
	pthread_exit(NULL);
}

void staticOptMain(void* (*threadFunc)(void*))
{
	//初始化信号量
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < NUM_THREADS-1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	for(int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	for(int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);
	sem_destroy(&sem_leader);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{ 
		sem_destroy(&sem_Divsion[t_id]);
		sem_destroy(&sem_Elimination[t_id]);
	}
}

void* staticFuncOptNew(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//除法
		int count = (ROW - k - 1) / NUM_THREADS;
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		//子线程处理k+1+count*t_id~k+count*(t_id+1)
		int endIt = k + 1 + count * (t_id + 1);//向量末端
		for (j = k + 1 + count*t_id;j < endIt && ((endIt - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < endIt;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		pthread_barrier_wait(&division);
		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//消去
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void staticNewOptMain(void* (*threadFunc)(void*))
{
	//初始化barrier
	pthread_barrier_init(&division, NULL, NUM_THREADS);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// 创建对应的 Handle
	long* param = new long[NUM_THREADS - 1];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//主函数看作第NUM_THREADS-1号线程
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
		//主线程处理ROW-count~ROW-1
		for (j = ROW - count;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		pthread_barrier_wait(&division);
		for (int i = k + NUM_THREADS; i < ROW; i += NUM_THREADS)
		{//消去
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void staticNewColMain(void* (*threadFunc)(void*))
{
	//初始化barrier
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS-1];// 创建对应的 Handle
	long* param = new long[NUM_THREADS-1];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS-1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//主函数看作第NUM_THREADS-1号线程
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
		//主线程处理ROW-count~ROW-1
		for (j = ROW-count;j < ROW&& ((ROW-j)& 3);++j)//串行处理对齐
		matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = ROW - count; i < ROW ; i++)//i为列
		{//逐列消去
			int j;//j为目标行
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_lane_f32(&matrix[j][i], subbee, 0);
				mult1 = vld1q_lane_f32(&matrix[j][k], mult1, 0);
				subbee = vld1q_lane_f32(&matrix[j + 1][i], subbee, 1);
				mult1 = vld1q_lane_f32(&matrix[j + 1][k], mult1, 1);
				subbee = vld1q_lane_f32(&matrix[j + 2][i], subbee, 2);
				mult1 = vld1q_lane_f32(&matrix[j + 2][k], mult1, 2);
				subbee = vld1q_lane_f32(&matrix[j + 3][i], subbee, 3);
				mult1 = vld1q_lane_f32(&matrix[j + 3][k], mult1, 3);
				mult2 = vld1q_dup_f32(&matrix[k][i]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_lane_f32(&matrix[j][i], subbee, 0);
				vst1q_lane_f32(&matrix[j + 1][i], subbee, 1);
				vst1q_lane_f32(&matrix[j + 2][i], subbee, 2);
				vst1q_lane_f32(&matrix[j + 3][i], subbee, 3);
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
		for (int j = k + NUM_THREADS;j < ROW;j+=NUM_THREADS)
			matrix[j][k] = 0;
	}
	//处理已被消去的列
	for (int t_id = 0; t_id < NUM_THREADS-1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&elemation);
}

void* staticFuncCol(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//子线程处理k+1+count*t_id~k+count*(t_id+1)列
		int count = (ROW - k - 1) / NUM_THREADS;
		//除法
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int endIt = k + 1 + count * (t_id + 1);//向量末端
		for (j = k + 1 + count*t_id;j < endIt && ((endIt - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < endIt;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		//循环划分任务：纵向划分
		for (int i = k + 1 + t_id*count; i < k + count * (t_id + 1); i++)//i为列
		{//逐列消去
			int j;//j为目标行
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_lane_f32(&matrix[j][i], subbee, 0);
				mult1 = vld1q_lane_f32(&matrix[j][k], mult1, 0);
				subbee = vld1q_lane_f32(&matrix[j+1][i], subbee, 1);
				mult1 = vld1q_lane_f32(&matrix[j+1][k], mult1, 1);
				subbee = vld1q_lane_f32(&matrix[j+2][i], subbee, 2);
				mult1 = vld1q_lane_f32(&matrix[j+2][k], mult1, 2);
				subbee = vld1q_lane_f32(&matrix[j+3][i], subbee, 3);
				mult1 = vld1q_lane_f32(&matrix[j+3][k], mult1, 3);
				mult2= vld1q_dup_f32(&matrix[k][i]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_lane_f32(&matrix[j][i], subbee, 0);
				vst1q_lane_f32(&matrix[j+1][i], subbee, 1);
				vst1q_lane_f32(&matrix[j+2][i], subbee, 2);
				vst1q_lane_f32(&matrix[j+3][i], subbee, 3);
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
		//处理已被消去的列
		for (int j = k + 1 + t_id;j < ROW;j += NUM_THREADS)
			matrix[j][k] = 0;
	}
	pthread_exit(NULL);
}

void ColCachedMain(void* (*threadFunc)(void*))
{
	//初始化barrier
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// 创建对应的 Handle
	long* param = new long[NUM_THREADS - 1];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//主函数看作第NUM_THREADS-1号线程
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&revmat[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
		//主线程处理ROW-count~ROW-1
		for (j = ROW - count;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			revmat[j][k] = revmat[j][k] / revmat[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_lane_f32(&revmat[j][k], divee, 0);
			divee = vld1q_lane_f32(&revmat[j + 1][k], divee, 1);
			divee = vld1q_lane_f32(&revmat[j + 2][k], divee, 2);
			divee = vld1q_lane_f32(&revmat[j + 3][k], divee, 3);
			divee = vdivq_f32(divee, diver);
			vst1q_lane_f32(&revmat[j][k], divee, 0);
			vst1q_lane_f32(&revmat[j + 1][k], divee, 1);
			vst1q_lane_f32(&revmat[j + 2][k], divee, 2);
			vst1q_lane_f32(&revmat[j + 3][k], divee, 3);
		}
		revmat[k][k] = 1.0;
		for (int i = ROW - count; i < ROW; i++)//i为列
		{//逐列消去
			int j;//j为目标行
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_f32(&revmat[i][j]);
				mult1 = vld1q_f32(&revmat[k][j]);
				mult2 = vld1q_dup_f32(&revmat[i][k]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_f32(&revmat[i][j], subbee);
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
		for (int j = k + NUM_THREADS;j < ROW;j += NUM_THREADS)
			matrix[k][j] = 0;
		//处理已被消去的列
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&elemation);
}

void* cachedFuncCol(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//子线程处理k+1+count*t_id~k+count*(t_id+1)列
		int count = (ROW - k - 1) / NUM_THREADS;
		//除法
		float32x4_t diver = vld1q_dup_f32(&revmat[k][k]);
		int j;
		int endIt = k + 1 + count * (t_id + 1);//向量末端
		for (j = k + 1 + count * t_id;j < endIt && ((endIt - j) & 3);++j)//串行处理对齐
			revmat[j][k] = revmat[j][k] / revmat[k][k];
		for (;j < endIt;j += 4)
		{
			float32x4_t divee = vld1q_lane_f32(&revmat[j][k], divee, 0);
			divee = vld1q_lane_f32(&revmat[j + 1][k], divee, 1);
			divee = vld1q_lane_f32(&revmat[j + 2][k], divee, 2);
			divee = vld1q_lane_f32(&revmat[j + 3][k], divee, 3);
			divee = vdivq_f32(divee, diver);
			vst1q_lane_f32(&revmat[j][k], divee, 0);
			vst1q_lane_f32(&revmat[j + 1][k], divee, 1);
			vst1q_lane_f32(&revmat[j + 2][k], divee, 2);
			vst1q_lane_f32(&revmat[j + 3][k], divee, 3);
		}
		//循环划分任务：纵向划分
		for (int i = k + 1 + t_id * count; i < k + count * (t_id + 1); i++)//i为列
		{//逐列消去
			int j;//j为目标行
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_f32(&revmat[i][j]);
				mult1 = vld1q_f32(&revmat[k][j]);
				mult2 = vld1q_dup_f32(&revmat[i][k]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_f32(&revmat[i][j], subbee);
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
		//处理已被消去的列
		for (int j = k + 1 + t_id;j < ROW;j += NUM_THREADS)
			revmat[k][j] = 0;
	}
	pthread_exit(NULL);
}

void DynamicDivMain(void* (*threadFunc)(void*))
{
	//初始化锁
	pthread_mutex_init(&remainLock, NULL);
	//初始化barrier
	pthread_barrier_init(&division, NULL, NUM_THREADS);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// 创建对应的 Handle
	long* param = new long[NUM_THREADS - 1];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//主函数看作第NUM_THREADS-1号线程
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
		//主线程处理ROW-count~ROW-1
		for (j = ROW - count;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//发出任务
		remain = k + 1;
		//等待子线程就位
		pthread_barrier_wait(&division);
		while(true)
		{
			int i;//行
			//处理i`i+TASK - 1
			//领取任务
			pthread_mutex_lock(&remainLock);
			if (remain >= ROW) 
			{
				pthread_mutex_unlock(&remainLock);
				break;
			}
			i = remain;
			remain += TASK;
			pthread_mutex_unlock(&remainLock);
			int end = min(ROW, i + TASK);
			for (;i < end;i++)
			{
				//消去
				float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
				int j;//列
				for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
					matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
				for (;j < ROW;j += 4)
				{
					float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
					float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
					mult2 = vmulq_f32(mult1, mult2);
					sub1 = vsubq_f32(sub1, mult2);
					vst1q_f32(&matrix[i][j], sub1);
				}
				matrix[i][k] = 0.0;
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
	pthread_mutex_destroy(&remainLock);
}

void* DynamicDivFunc(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//除法
		int count = (ROW - k - 1) / NUM_THREADS;
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		//子线程处理k+1+count*t_id~k+count*(t_id+1)
		int endIt = k + 1 + count * (t_id + 1);//向量末端
		for (j = k + 1 + count * t_id;j < endIt && ((endIt - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < endIt;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		pthread_barrier_wait(&division);
		//循环划分任务（同学们可以尝试多种任务划分方式）
		while (true)
		{
			int i;
			pthread_mutex_lock(&remainLock);
			if (remain >= ROW)
			{
				pthread_mutex_unlock(&remainLock);
				break;
			}
			i = remain;
			remain += TASK;
			pthread_mutex_unlock(&remainLock);
			int end = min(ROW, i + TASK);
			for (; i < end; i++)
			{//消去
				float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
				int j;
				for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
					matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
				for (;j < ROW;j += 4)
				{
					float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
					float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
					mult2 = vmulq_f32(mult1, mult2);
					sub1 = vsubq_f32(sub1, mult2);
					vst1q_f32(&matrix[i][j], sub1);
				}
				matrix[i][k] = 0.0;
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void timing(void (*func)())
{
	timeval tv_begin, tv_end;
	int counter(0);
	double time = 0;
	while (INTERVAL > time)
	{
		init();
		gettimeofday(&tv_begin, 0);
		func();
		gettimeofday(&tv_end, 0);
		counter++;
		time += ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec) * 1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec) / 1000.0;
	}
	cout << time / counter << "," << counter << '\n';
}

void timing(void (*func)(void* (*threadFunc)(void*)), void* (*threadFunc)(void*))
{
	timeval tv_begin, tv_end;
	int counter(0);
	double time = 0;
	while (INTERVAL > time)
	{
		init();
		gettimeofday(&tv_begin, 0);
		func(threadFunc);
		gettimeofday(&tv_end, 0);
		counter++;
		time += ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec) * 1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec) / 1000.0;
	}
	cout << time / counter << "," << counter << '\n';
}

int main()
{
	
	for (NUM_THREADS = 8;NUM_THREADS <= 8;NUM_THREADS++)
	{
		
		cout << "dynamicDiv: ";
		timing(DynamicDivMain, DynamicDivFunc);
	}
	cout << "dynamic: ";
	timing(dynamicMain, dynamicFunc);
	cout << "dynamic+SIMD: ";
	timing(dynamicMain, dynamicFuncSIMD);
}
