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

//�ź������壬���ھ�̬�߳�
sem_t sem_leader;
sem_t sem_Divsion[32];
sem_t sem_Elimination[32];
//�¸Ľ���
pthread_barrier_t division;
pthread_barrier_t elemation;
//��̬�߳�����
int NUM_THREADS = 4;
//��̬�̷߳��䣺����������
int remain = ROW;
pthread_mutex_t remainLock;

void reverse()
{
	for (int i = 0; i < ROW; i++)
		for (int j = 0; j < ROW; j++)
			revmat[j][i] = matrix[i][j];
}

void init()
{
	for (int i = 0; i < ROW; i++)
	{
		for (int j = 0; j < i; j++)
			matrix[i][j] = 0;
		for (int j = i; j < ROW; j++)
			matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0; k < 1000; k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0; j < ROW; j++)
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
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)
		{//��ȥ
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (; j < ROW; j += 4)
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
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)//iΪ��
		{//������ȥ
			int j;//jΪĿ����
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//�����в���
				matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
			for (; j < ROW; j += 4)
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
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&revmat[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
			revmat[j][k] = revmat[j][k] / revmat[k][k];
		for (; j < ROW; j += 4)
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
		for (int i = k + 1; i < ROW; i++)//iΪ��
		{//�D��ǰ������ȥ���D�ú�������ȥ
			int j;//jΪĿ����
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//�����в���
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (; j < ROW; j += 4)
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
	for (int i = 0; i <= ROW - 1; ++i) {
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
}//������


void* dynamicFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int j = k + 1; j < ROW; ++j)
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
	for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
		matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
	for (; j < ROW; j += 4)
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
	for (int k = 0; k < ROW; ++k)
	{//���߳�����������
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//���������̣߳�������ȥ����
		int worker_count = ROW - 1 - k; //�����߳�����
		pthread_t* handles = new pthread_t[worker_count];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[worker_count];// ������Ӧ���߳����ݽṹ
		//��������
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* newDynamicFuncSIMD(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int i = k + t_id + 1; i < ROW; i += NUM_THREADS)
	{
		float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
		for (; j < ROW; j += 4)
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
	{//���߳�����������
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//���������̣߳�������ȥ����
		int worker_count = NUM_THREADS; //�����߳�����
		pthread_t* handles = new pthread_t[worker_count];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[worker_count];// ������Ӧ���߳����ݽṹ
		//��������
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* staticFunc(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		pthread_barrier_wait(&division);// �������ȴ�������ɳ�������
		//ѭ����������
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//��ȥ
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (; j < ROW; j += 4)
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
	//��ʼ���ź���
	pthread_barrier_init(&division, NULL, NUM_THREADS + 1);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS + 1);
	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, staticFunc, (void*)t_id);

	for (int k = 0; k < ROW; ++k)
	{
		//���߳�����������
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		pthread_barrier_wait(&division);
		// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void* staticFuncOpt(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		// ���ź���������ͬ����ʽ��ʹ�� barrier
		if (t_id == 0)
		{
			float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
				matrix[k][j] = matrix[k][j] / matrix[k][k];
			for (; j < ROW; j += 4)
			{
				float32x4_t divee = vld1q_f32(&matrix[k][j]);
				divee = vdivq_f32(divee, diver);
				vst1q_f32(&matrix[k][j], divee);
			}
			matrix[k][k] = 1.0;
		}
		else sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_post(&sem_Divsion[t_id]);

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//��ȥ
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// �����߳�һ�������һ��
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
			sem_wait(&sem_Elimination[t_id - 1]);
		}
	}
	pthread_exit(NULL);
}

void staticOptMain(void* (*threadFunc)(void*))
{
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < NUM_THREADS - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
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
		//����
		int count = (ROW - k - 1) / NUM_THREADS;
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		//���̴߳���k+1+count*t_id~k+count*(t_id+1)
		int endIt = k + 1 + count * (t_id + 1);//����ĩ��
		for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < endIt; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		pthread_barrier_wait(&division);
		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//��ȥ
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// �����߳�һ�������һ��
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void staticNewOptMain(void* (*threadFunc)(void*))
{
	//��ʼ��barrier
	pthread_barrier_init(&division, NULL, NUM_THREADS);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// ������Ӧ�� Handle
	long* param = new long[NUM_THREADS - 1];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//������������NUM_THREADS-1���߳�
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//���߳�Ҫ���������
		//���̴߳���ROW-count~ROW-1
		for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		pthread_barrier_wait(&division);
		for (int i = k + NUM_THREADS; i < ROW; i += NUM_THREADS)
		{//��ȥ
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// �����߳�һ�������һ��
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void staticNewColMain(void* (*threadFunc)(void*))
{
	//��ʼ��barrier
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// ������Ӧ�� Handle
	long* param = new long[NUM_THREADS - 1];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//������������NUM_THREADS-1���߳�
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//���߳�Ҫ���������
		//���̴߳���ROW-count~ROW-1
		for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = ROW - count; i < ROW; i++)//iΪ��
		{//������ȥ
			int j;//jΪĿ����
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//�����в���
				matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
			for (; j < ROW; j += 4)
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
		// �����߳�һ�������һ��
		pthread_barrier_wait(&elemation);
		for (int j = k + NUM_THREADS; j < ROW; j += NUM_THREADS)
			matrix[j][k] = 0;
	}
	//�����ѱ���ȥ����
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&elemation);
}

void* staticFuncCol(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//���̴߳���k+1+count*t_id~k+count*(t_id+1)��
		int count = (ROW - k - 1) / NUM_THREADS;
		//����
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int endIt = k + 1 + count * (t_id + 1);//����ĩ��
		for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < endIt; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		//ѭ�������������򻮷�
		for (int i = k + 1 + t_id * count; i < k + count * (t_id + 1); i++)//iΪ��
		{//������ȥ
			int j;//jΪĿ����
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//�����в���
				matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
			for (; j < ROW; j += 4)
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
		// �����߳�һ�������һ��
		pthread_barrier_wait(&elemation);
		//�����ѱ���ȥ����
		for (int j = k + 1 + t_id; j < ROW; j += NUM_THREADS)
			matrix[j][k] = 0;
	}
	pthread_exit(NULL);
}

void ColCachedMain(void* (*threadFunc)(void*))
{
	//��ʼ��barrier
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// ������Ӧ�� Handle
	long* param = new long[NUM_THREADS - 1];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//������������NUM_THREADS-1���߳�
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&revmat[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//���߳�Ҫ���������
		//���̴߳���ROW-count~ROW-1
		for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)//���д������
			revmat[j][k] = revmat[j][k] / revmat[k][k];
		for (; j < ROW; j += 4)
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
		for (int i = ROW - count; i < ROW; i++)//iΪ��
		{//������ȥ
			int j;//jΪĿ����
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//�����в���
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (; j < ROW; j += 4)
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
		// �����߳�һ�������һ��
		pthread_barrier_wait(&elemation);
		for (int j = k + NUM_THREADS; j < ROW; j += NUM_THREADS)
			matrix[k][j] = 0;
		//�����ѱ���ȥ����
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&elemation);
}

void* cachedFuncCol(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//���̴߳���k+1+count*t_id~k+count*(t_id+1)��
		int count = (ROW - k - 1) / NUM_THREADS;
		//����
		float32x4_t diver = vld1q_dup_f32(&revmat[k][k]);
		int j;
		int endIt = k + 1 + count * (t_id + 1);//����ĩ��
		for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)//���д������
			revmat[j][k] = revmat[j][k] / revmat[k][k];
		for (; j < endIt; j += 4)
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
		//ѭ�������������򻮷�
		for (int i = k + 1 + t_id * count; i < k + count * (t_id + 1); i++)//iΪ��
		{//������ȥ
			int j;//jΪĿ����
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//�����в���
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (; j < ROW; j += 4)
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
		// �����߳�һ�������һ��
		pthread_barrier_wait(&elemation);
		//�����ѱ���ȥ����
		for (int j = k + 1 + t_id; j < ROW; j += NUM_THREADS)
			revmat[k][j] = 0;
	}
	pthread_exit(NULL);
}

void DynamicDivMain(void* (*threadFunc)(void*))
{
	//��ʼ����
	pthread_mutex_init(&remainLock, NULL);
	//��ʼ��barrier
	pthread_barrier_init(&division, NULL, NUM_THREADS);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// ������Ӧ�� Handle
	long* param = new long[NUM_THREADS - 1];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//������������NUM_THREADS-1���߳�
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//���߳�Ҫ���������
		//���̴߳���ROW-count~ROW-1
		for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//��������
		remain = k + 1;
		//�ȴ����߳̾�λ
		pthread_barrier_wait(&division);
		while (true)
		{
			int i;//��
			//����i`i+TASK - 1
			//��ȡ����
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
			{
				//��ȥ
				float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
				int j;//��
				for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
					matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
				for (; j < ROW; j += 4)
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
		// �����߳�һ�������һ��
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
		//����
		int count = (ROW - k - 1) / NUM_THREADS;
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		//���̴߳���k+1+count*t_id~k+count*(t_id+1)
		int endIt = k + 1 + count * (t_id + 1);//����ĩ��
		for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)//���д������
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (; j < endIt; j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		pthread_barrier_wait(&division);
		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
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
			{//��ȥ
				float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
				int j;
				for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//���д������
					matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
				for (; j < ROW; j += 4)
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
		// �����߳�һ�������һ��
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void timing(void(*func)())
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

void timing(void(*func)(void* (*threadFunc)(void*)), void* (*threadFunc)(void*))
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
	cout << "colSIMDcached: ";
	timing(ColSIMDcached);
	cout << "plain: ";
	timing(plain);
	cout << "SIMD: ";
	timing(SIMD);
	cout << "ColSIMD: ";
	timing(ColSIMD);
	for (NUM_THREADS = 2; NUM_THREADS <= 32; NUM_THREADS++)
	{
		cout << "using " << NUM_THREADS << " threads" << endl;
		cout << "dynamic(reasonable): ";
		timing(newDynamicMain, newDynamicFuncSIMD);
		cout << "static: ";
		timing(staticMain);
		cout << "static(add division): ";
		timing(staticOptMain, staticFuncOpt);
		cout << "static(myOpt row div): ";
		timing(staticNewOptMain, staticFuncOptNew);
		cout << "static(myOpt col div): ";
		timing(staticNewColMain, staticFuncCol);
		cout << "static(myOpt col div cached): ";
		timing(ColCachedMain, cachedFuncCol);
		cout << "dynamicDiv: ";
		timing(DynamicDivMain, DynamicDivFunc);
	}
	cout << "dynamic: ";
	timing(dynamicMain, dynamicFunc);
	cout << "dynamic+SIMD: ";
	timing(dynamicMain, dynamicFuncSIMD);
}
